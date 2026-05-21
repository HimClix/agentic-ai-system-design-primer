# Hybrid Search: BM25 + Vector Search
> Neither keyword search nor vector search alone is sufficient. Hybrid search with Reciprocal Rank Fusion combines the strengths of both and consistently outperforms either approach by 5-15% on retrieval benchmarks.

## What It Is

Hybrid search combines two fundamentally different retrieval methods:

1. **BM25 (keyword/sparse search)**: Matches on exact terms. Excels at finding specific identifiers, codes, acronyms, and exact phrases. Based on term frequency and inverse document frequency.

2. **Vector search (semantic/dense search)**: Matches on meaning. Excels at finding conceptually related content even when wording differs. Based on embedding similarity.

These two methods have complementary strengths and weaknesses. Hybrid search runs both in parallel and merges the results using a fusion algorithm -- most commonly Reciprocal Rank Fusion (RRF).

## How It Works

### Why Neither Alone Is Sufficient

```
Query: "ERR_CONNECTION_REFUSED on pod rzp-api-7b4f9d"

BM25 finds:
  [1] Log entry: "ERR_CONNECTION_REFUSED from pod rzp-api-7b4f9d at 14:32 UTC"
  [2] KB article: "Common error codes: ERR_CONNECTION_REFUSED, ERR_TIMEOUT..."
  --> Excellent at exact error codes and pod names

Vector search finds:
  [1] KB article: "Troubleshooting network connectivity issues between pods"
  [2] Runbook: "When services cannot reach each other, check network policies"
  --> Excellent at understanding the concept (connectivity failure)

Hybrid combines both:
  [1] Log entry with exact error (from BM25)
  [2] Troubleshooting guide for concept (from vector)
  [3] KB article with error codes (from BM25)
  --> Best of both worlds
```

### When BM25 Beats Vector Search

| Scenario | Example Query | Why BM25 Wins |
|----------|--------------|---------------|
| Exact identifiers | "order_id OD123456789" | Vector embeds meaning, not exact strings |
| Error codes | "ERR_AUTH_FAILED" | Specific codes need exact matching |
| Technical acronyms | "CORS policy violation" | Acronyms may not embed well |
| API endpoints | "GET /api/v2/users" | Structured paths need exact match |
| Numeric values | "invoice #INV-2024-0042" | Numbers are poorly embedded |
| Boolean search | "kubernetes AND pod AND crashloopbackoff" | BM25 supports boolean operators |

### When Vector Search Beats BM25

| Scenario | Example Query | Why Vector Wins |
|----------|--------------|----------------|
| Semantic similarity | "how to make service faster" | BM25 misses synonyms like "optimize performance" |
| Paraphrased questions | "what happens if payment fails" | Original doc says "payment decline handling" |
| Concept search | "best practices for security" | Documents may use different terminology |
| Multilingual | "How to fix this?" (English query, Hindi docs) | Cross-lingual embeddings bridge languages |
| Typos | "kubernets deployment" | BM25 fails on "kubernets", vector maps to "kubernetes" |

### Reciprocal Rank Fusion (RRF)

RRF is the standard algorithm for merging ranked lists from different retrieval methods.

**Formula:**
```
RRF_score(d) = sum over all rankers:  1 / (k + rank_i(d))

Where:
  d     = document
  k     = constant (typically 60)
  rank_i(d) = rank of document d in result list i (1-indexed)
```

**Example:**
```
BM25 results:     [doc_A (rank 1), doc_B (rank 2), doc_C (rank 3)]
Vector results:   [doc_C (rank 1), doc_A (rank 2), doc_D (rank 3)]

RRF scores (k=60):
  doc_A: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
  doc_B: 1/(60+2) + 0         = 0.01613
  doc_C: 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226
  doc_D: 0        + 1/(60+3) = 0.01587

Final ranking: [doc_A, doc_C, doc_B, doc_D]
  doc_A ranks high in both -> #1
  doc_C ranks high in both -> #2
  doc_B only in BM25 -> #3
  doc_D only in vector -> #4
```

## Production Implementation

### Hybrid Search with Qdrant

```python
"""Hybrid search implementation using Qdrant vector database."""
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    PointStruct, Distance, VectorParams,
    SparseVector, SparseVectorParams, SparseIndexParams,
    SearchRequest, NamedVector, NamedSparseVector,
    Prefetch, Query, FusionQuery,
)
import numpy as np
from typing import Optional
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """Hybrid search combining dense vectors and sparse BM25 in Qdrant."""

    def __init__(
        self,
        collection_name: str = "documents",
        qdrant_url: str = "http://localhost:6333",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        dense_dim: int = 384,
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(url=qdrant_url)
        self.embedding_model_name = embedding_model
        self.dense_dim = dense_dim
        
        # Initialize the dense embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)

    def create_collection(self):
        """Create collection with both dense and sparse vectors."""
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.dense_dim,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        logger.info(f"Created hybrid collection: {self.collection_name}")

    def _embed_dense(self, text: str) -> list[float]:
        """Generate dense embedding for text."""
        inputs = self.tokenizer(
            text, return_tensors="pt", 
            truncation=True, max_length=512, padding=True,
        )
        outputs = self.model(**inputs)
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings[0].detach().numpy().tolist()

    def _embed_sparse(self, text: str) -> tuple[list[int], list[float]]:
        """Generate sparse BM25-style embedding for text.
        
        In production, use a trained sparse encoder like SPLADE or 
        Qdrant's built-in BM25. This is a simplified version.
        """
        # Tokenize and compute term frequencies
        tokens = text.lower().split()
        token_freq = {}
        for token in tokens:
            token_freq[token] = token_freq.get(token, 0) + 1
        
        # Convert to sparse vector format
        # In production: use vocabulary indices, not hash
        indices = [hash(t) % 30000 for t in token_freq.keys()]
        values = list(token_freq.values())
        
        return indices, values

    async def index_documents(self, documents: list[dict]):
        """Index documents with both dense and sparse vectors."""
        points = []
        
        for i, doc in enumerate(documents):
            dense_vector = self._embed_dense(doc["text"])
            sparse_indices, sparse_values = self._embed_sparse(doc["text"])
            
            points.append(PointStruct(
                id=i,
                vector={
                    "dense": dense_vector,
                },
                payload={
                    "text": doc["text"],
                    "source": doc.get("source", ""),
                    "metadata": doc.get("metadata", {}),
                },
            ))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info(f"Indexed {len(points)} documents")

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[dict]:
        """Execute hybrid search with RRF fusion.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            dense_weight: Weight for vector search (0.0 to 1.0)
            sparse_weight: Weight for keyword search (0.0 to 1.0)
        
        Returns:
            List of documents with scores, ranked by RRF fusion
        """
        # Generate query embeddings
        dense_query = self._embed_dense(query)
        sparse_indices, sparse_values = self._embed_sparse(query)
        
        # Qdrant native hybrid search with RRF fusion
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_query,
                    using="dense",
                    limit=top_k * 3,  # Over-fetch for fusion
                ),
                # For BM25, in production use Qdrant's built-in sparse vectors
            ],
            query=FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
        )
        
        return [
            {
                "id": hit.id,
                "text": hit.payload.get("text", ""),
                "source": hit.payload.get("source", ""),
                "score": hit.score,
                "metadata": hit.payload.get("metadata", {}),
            }
            for hit in results.points
        ]


# Standalone RRF implementation (framework-agnostic)

def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
    top_n: int = 10,
    id_key: str = "id",
) -> list[dict]:
    """Merge multiple ranked result lists using RRF.
    
    Args:
        result_lists: List of result lists, each ordered by relevance
        k: RRF constant (default 60, higher = more uniform weighting)
        top_n: Number of results to return
        id_key: Key to use for document identity
    
    Returns:
        Merged and re-ranked results
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}
    
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            doc_id = doc[id_key]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
            doc_map[doc_id] = doc  # Keep last seen version
    
    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    results = []
    for doc_id in sorted_ids[:top_n]:
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = rrf_scores[doc_id]
        results.append(doc)
    
    return results
```

## Decision Tree: Dense, Sparse, or Hybrid?

```
What types of queries will your users ask?
  |
  |-- Exact identifiers, codes, error messages?
  |     --> BM25 must be included (hybrid or pure BM25)
  |
  |-- Conceptual, natural language questions?
  |     --> Vector search must be included (hybrid or pure vector)
  |
  |-- Mix of both?
  |     --> Hybrid search (recommended for most production systems)
  |
  v
What is your latency budget?
  |
  |-- < 50ms: Pure vector OR pure BM25 (not both)
  |-- 50-200ms: Hybrid search (run in parallel, merge)
  |-- > 200ms: Hybrid search + reranking
```

## When NOT to Use Hybrid Search

- **Purely semantic queries**: If users only ask conceptual questions, vector search alone may suffice
- **Purely keyword queries**: Log search, exact match systems -- BM25 alone is faster and sufficient
- **Extreme latency constraints**: Hybrid adds ~20-50ms over single-method search
- **Very small datasets**: Under 1000 documents, simple keyword search may be adequate

## Tradeoffs

| Aspect | BM25 Only | Vector Only | Hybrid (BM25 + Vector) |
|--------|----------|-------------|----------------------|
| Exact match | Excellent | Poor | Excellent |
| Semantic match | Poor | Excellent | Excellent |
| Latency | ~5-20ms | ~10-50ms | ~30-100ms |
| Infrastructure | Elasticsearch/Lucene | Vector DB | Both (or Qdrant with sparse) |
| Index size | Smaller | Larger (embeddings) | Largest |
| Tuning complexity | Low (few params) | Medium | Higher (fusion weights) |
| Retrieval quality | Good for keyword queries | Good for semantic queries | Best overall |

## Real-World Examples

### Support Ticket Search
```
Query: "customer can't login SSO SAML error"

BM25 top 3:
  1. "SAML SSO configuration error troubleshooting guide" (exact terms)
  2. "Error: SSO authentication failed - SAML response invalid" (exact match)
  3. "SSO setup documentation for enterprise customers" (keyword match)

Vector top 3:
  1. "Troubleshooting authentication and access issues" (semantic)
  2. "Single sign-on problems and solutions" (conceptual match)
  3. "How to fix login failures for enterprise users" (paraphrase)

Hybrid (RRF) top 3:
  1. "SAML SSO configuration error troubleshooting guide" (both)
  2. "Troubleshooting authentication and access issues" (vector + related)
  3. "Error: SSO authentication failed - SAML response invalid" (BM25 exact)
```

## Failure Modes

| Failure | Cause | Impact | Fix |
|---------|-------|--------|-----|
| BM25 dominates | Sparse weight too high | Misses semantic matches | Tune weights: 0.6-0.7 dense, 0.3-0.4 sparse |
| Vector dominates | Dense weight too high | Misses exact terms | Include BM25 with adequate weight |
| RRF constant wrong | k too low or too high | Skewed fusion | k=60 is standard; tune per dataset |
| Embedding model mismatch | Model not trained on domain | Poor semantic retrieval | Use domain-adapted embeddings or fine-tune |
| Sparse index stale | BM25 index not updated | Misses new documents | Real-time indexing or scheduled refresh |

## Source(s) and Further Reading

- Qdrant, "Hybrid Search with Reciprocal Rank Fusion" (2024) -- native hybrid search
- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" -- BM25 foundations
- Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion" (2009) -- RRF original paper
- RAG_Techniques repository (GitHub, 26.2k stars) -- hybrid search patterns
- Vespa, "Hybrid Search" documentation -- alternative hybrid implementation
