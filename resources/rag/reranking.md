# Cross-Encoder Reranking
> Reranking is the difference between "retrieved the relevant document somewhere in the top 50" and "put it in the top 3." It costs ~$0.002/query and adds ~100ms, but improves precision by 10-25%.

## What It Is

Reranking is the second stage of a two-stage retrieval pipeline. Stage 1 (retrieval) casts a wide net using fast but approximate methods (vector search, BM25). Stage 2 (reranking) uses a slower but more precise model to re-score and re-order the candidates.

The key difference is architectural:
- **Bi-encoder (Stage 1)**: Embeds query and documents separately, compares vectors. Fast (can search millions in milliseconds) but approximate.
- **Cross-encoder (Stage 2)**: Processes query AND document together in one forward pass. Slow (one inference per document) but much more precise.

This is why reranking only works on a small candidate set (20-100 documents). The cross-encoder sees the full interaction between query and document text, catching nuances that bi-encoders miss.

## How It Works

### Two-Stage Architecture

```
Query: "How do I configure rate limiting in the API gateway?"

Stage 1: Fast Retrieval (bi-encoder)
  -> Query embedding: [0.12, -0.34, 0.56, ...]
  -> Compare against 100,000 document embeddings
  -> Top 50 candidates in ~20ms
  -> Candidates include:
     [1] "API gateway rate limiting configuration" (score: 0.89)
     [2] "Rate limiting best practices" (score: 0.87)
     [3] "API gateway authentication setup" (score: 0.85)
     ...
     [47] "Configuring per-client rate limits in nginx" (score: 0.71)

Stage 2: Precise Reranking (cross-encoder)
  -> Process 50 (query, document) pairs through cross-encoder
  -> Each pair scored independently: ~2ms per pair, ~100ms total
  -> Reranked top 5:
     [1] "Configuring per-client rate limits in nginx" (was #47!)
     [2] "API gateway rate limiting configuration" (was #1)
     [3] "Rate limiting with token buckets and leaky buckets" (was #12)
     [4] "Rate limiting best practices" (was #2)
     [5] "Nginx API gateway throttling tutorial" (was #23)
```

Notice: document #47 in the retrieval stage moved to #1 after reranking. The cross-encoder understood that "configuring per-client rate limits in nginx" was the most specific match for the query, even though the bi-encoder scored it lower because it had different vocabulary.

### Why Cross-Encoders Are More Precise

```
Bi-encoder (Stage 1):
  query_vector = embed("How do I configure rate limiting?")
  doc_vector = embed("Configuring per-client rate limits in nginx")
  score = cosine(query_vector, doc_vector)
  -> Encodes each text INDEPENDENTLY -> misses cross-text interactions

Cross-encoder (Stage 2):
  score = model("[CLS] How do I configure rate limiting? [SEP] 
                  Configuring per-client rate limits in nginx [SEP]")
  -> Processes BOTH texts TOGETHER -> captures word-level interactions
  -> Knows "configure" in query matches "configuring" in doc
  -> Knows "rate limiting" maps to "rate limits"
  -> Knows "API gateway" is related to "nginx"
```

## Production Implementation

### Using Cohere Rerank API

```python
"""Reranking with Cohere Rerank API -- production-ready."""
import cohere
import time
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    text: str
    score: float
    original_index: int
    metadata: dict


class CohereReranker:
    """Reranker using Cohere's Rerank API.
    
    Cost: ~$0.002 per query (1000 queries = $2)
    Latency: ~50-150ms for 50 documents
    """

    def __init__(self, api_key: str, model: str = "rerank-v3.5"):
        self.client = cohere.Client(api_key)
        self.model = model

    async def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
        text_key: str = "text",
        min_score: float = 0.0,
    ) -> list[RerankResult]:
        """Rerank documents using Cohere cross-encoder.
        
        Args:
            query: The search query
            documents: List of document dicts (must have text_key field)
            top_k: Number of top results to return
            text_key: Key in document dict containing text
            min_score: Minimum relevance score threshold (0.0-1.0)
        
        Returns:
            Top-k documents reranked by relevance
        """
        if not documents:
            return []

        start = time.time()

        doc_texts = [doc[text_key] for doc in documents]

        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=doc_texts,
            top_n=top_k,
        )

        elapsed_ms = int((time.time() - start) * 1000)
        logger.info(
            f"Reranked {len(documents)} docs in {elapsed_ms}ms, "
            f"returning top {top_k}"
        )

        results = []
        for hit in response.results:
            if hit.relevance_score >= min_score:
                original_doc = documents[hit.index]
                results.append(RerankResult(
                    text=original_doc[text_key],
                    score=hit.relevance_score,
                    original_index=hit.index,
                    metadata=original_doc.get("metadata", {}),
                ))

        return results


# Usage in RAG pipeline:
async def rag_with_reranking(query: str, vector_db, reranker: CohereReranker):
    """Complete retrieval + reranking pipeline."""
    
    # Stage 1: Fast retrieval (50 candidates)
    candidates = await vector_db.search(query, top_k=50)
    
    # Stage 2: Precise reranking (top 5)
    reranked = await reranker.rerank(
        query=query,
        documents=candidates,
        top_k=5,
        min_score=0.3,  # Filter out irrelevant results
    )
    
    return reranked
```

### Self-Hosted Cross-Encoder (BGE Reranker)

```python
"""Self-hosted reranker using BGE cross-encoder model."""
from sentence_transformers import CrossEncoder
import numpy as np
from typing import Optional


class BGEReranker:
    """Self-hosted reranker using BAAI/bge-reranker-v2-m3.
    
    Cost: Free (your GPU/CPU)
    Latency: ~100-300ms for 50 documents on GPU, ~500ms-2s on CPU
    Model size: ~560MB
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name, max_length=512)

    async def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
        text_key: str = "text",
    ) -> list[dict]:
        """Rerank documents using local cross-encoder."""
        if not documents:
            return []

        # Create (query, document) pairs
        pairs = [(query, doc[text_key]) for doc in documents]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Sort by score descending
        scored_docs = list(zip(scores, range(len(documents)), documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, orig_idx, doc in scored_docs[:top_k]:
            results.append({
                **doc,
                "rerank_score": float(score),
                "original_index": orig_idx,
            })

        return results
```

## Reranker Comparison

| Reranker | Latency (50 docs) | Cost per Query | Quality (NDCG@10) | Self-Hosted | Best For |
|----------|-------------------|----------------|-------------------|------------|----------|
| Cohere Rerank v3.5 | ~80-120ms | $0.002 | 0.72 | No (API) | Production SaaS |
| BGE-reranker-v2-m3 | ~150-300ms (GPU) | Free | 0.70 | Yes | Data-sensitive workloads |
| BGE-reranker-large | ~200-400ms (GPU) | Free | 0.68 | Yes | Budget-conscious |
| ColBERT v2 | ~20-50ms | Free | 0.71 | Yes | Low-latency requirement |
| Jina Reranker v2 | ~100-200ms | $0.002 | 0.69 | API + local | Alternative to Cohere |
| FlashRank | ~30-80ms | Free | 0.65 | Yes | CPU-only environments |

## Decision Tree: Should You Add Reranking?

```
Is your current retrieval accuracy sufficient?
  |
  YES --> Don't add reranking (unnecessary cost and latency)
  |
  NO --> Do relevant docs appear in your top-50 but not top-5?
          |
          YES --> Reranking will help significantly
          |       Choose:
          |       |-- Need data privacy? -> BGE self-hosted
          |       |-- Need lowest latency? -> ColBERT or FlashRank
          |       |-- Need best quality? -> Cohere Rerank
          |
          NO --> Relevant docs not even in top-50?
                  |
                  YES --> Fix retrieval first (better embeddings, hybrid search)
                  |       Reranking can't fix retrieval failure
                  NO --> Check chunking strategy (docs may be split poorly)
```

## When NOT to Use Reranking

- **Retrieval is already precise**: If top-5 accuracy is >90%, reranking adds cost without benefit
- **Extreme latency requirements**: <50ms end-to-end budgets don't have room for reranking
- **Very small collections**: <1000 documents -- just retrieve all and rank by score
- **Retrieval failure**: If relevant documents aren't in the top-50 candidates, reranking can't help. Fix the retrieval stage first.

## Tradeoffs

| Aspect | With Reranking | Without Reranking |
|--------|---------------|-------------------|
| Precision@5 | +10-25% improvement | Baseline |
| Latency | +80-200ms | No added latency |
| Cost per query | +$0.002 (Cohere) | $0.00 |
| Infrastructure | Reranker model or API | None |
| Complexity | Two-stage pipeline | Single-stage |
| Failure modes | One more thing that can fail | Simpler |

## Real-World Examples

### Before and After Reranking
```
Query: "How to rotate API keys without downtime"

Without reranking (top 3 from vector search):
  1. "API key management best practices" (score: 0.87)     -- too general
  2. "Authentication and authorization overview" (score: 0.85) -- wrong topic
  3. "API key rotation guide" (score: 0.83)                 -- correct!

With reranking (cross-encoder re-scored):
  1. "API key rotation guide" (rerank: 0.94)                -- correct, now #1
  2. "Zero-downtime credential rotation strategies" (rerank: 0.89) -- relevant
  3. "API key management best practices" (rerank: 0.71)     -- less relevant
```

## Failure Modes

| Failure | Cause | Impact | Fix |
|---------|-------|--------|-----|
| Reranker hallucination | Cross-encoder scores irrelevant doc high | Wrong context sent to LLM | Set min_score threshold (0.3-0.5) |
| Latency spike | Too many documents to rerank | User-visible delay | Limit candidates to 50 max |
| API rate limit | Cohere rate limit exceeded | Reranking fails | Fallback to retrieval-only ordering |
| Model size mismatch | Self-hosted model too large for CPU | OOM or very slow inference | Use smaller model (FlashRank) |
| Retrieval failure masked | Reranking "fixes" bad retrieval by luck | Inconsistent quality | Monitor retrieval recall independently |

## Source(s) and Further Reading

- Cohere, "Rerank Documentation" (2024) -- API reference and pricing
- BAAI, "BGE Reranker" (GitHub) -- self-hosted cross-encoder models
- ColBERT v2 Paper (Santhanam et al., 2022) -- efficient late-interaction reranking
- RAG_Techniques repository (GitHub, 26.2k stars) -- reranking integration patterns
- MS MARCO Leaderboard -- reranking benchmark results
