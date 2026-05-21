# RAG Pipeline Architecture
> RAG is not "embed, retrieve, generate." Production RAG is a 5-stage pipeline where each stage has its own failure modes, tuning parameters, and cost tradeoffs.

## What It Is

Retrieval-Augmented Generation (RAG) is an architecture pattern where an LLM's response is grounded in retrieved documents rather than relying solely on parametric knowledge. Instead of fine-tuning a model on your data, you retrieve relevant context at query time and inject it into the prompt.

A production RAG pipeline has 5 stages:
1. **Query Processing** -- rewrite, decompose, or expand the user query
2. **Retrieval** -- fetch candidate documents (vector search, keyword search, or hybrid)
3. **Reranking** -- re-score candidates for precision using a cross-encoder
4. **Context Assembly** -- select, order, and format retrieved context for the LLM
5. **Generation** -- LLM generates a response grounded in the assembled context

## How It Works

### End-to-End Architecture

```
User Query: "What is our refund policy for enterprise customers?"
  |
  v
[Stage 1: Query Processing]
  |-- Query rewriting: "enterprise customer refund policy terms conditions"
  |-- HyDE: Generate hypothetical answer, embed that instead
  |-- Multi-query: Split into sub-queries if complex
  |
  v
[Stage 2: Retrieval] (fast, high recall)
  |-- Vector search (semantic similarity) -> top 50 candidates
  |-- BM25 keyword search -> top 50 candidates
  |-- Hybrid: merge with Reciprocal Rank Fusion -> top 30
  |
  v
[Stage 3: Reranking] (slow, high precision)
  |-- Cross-encoder reranker scores each (query, document) pair
  |-- Re-sort by relevance score -> top 5-10
  |
  v
[Stage 4: Context Assembly]
  |-- Order: most relevant first (primacy bias)
  |-- Format: add source citations, section headers
  |-- Trim: fit within context window budget (20-40% of total)
  |
  v
[Stage 5: Generation]
  |-- System prompt: "Answer based on the provided context"
  |-- Context: assembled documents
  |-- User query: original question
  |-- Guardrails: "Say 'I don't know' if not in context"
  |
  v
Response with citations: "According to the Enterprise Terms (Section 4.2),
refunds are processed within 30 days..."
```

### The 5-Stage Pipeline in Detail

| Stage | Latency | Purpose | Key Technique |
|-------|---------|---------|---------------|
| Query Processing | 50-500ms | Improve retrieval quality | HyDE, multi-query |
| Retrieval | 20-100ms | High recall (find candidates) | Hybrid search (vector + BM25) |
| Reranking | 50-200ms | High precision (rank candidates) | Cross-encoder (Cohere, BGE) |
| Context Assembly | <10ms | Optimize context for LLM | Truncation, ordering, formatting |
| Generation | 500-3000ms | Generate grounded response | LLM with context |

## Production Implementation

```python
"""Production RAG pipeline with all 5 stages."""
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Document:
    id: str
    text: str
    metadata: dict
    score: float = 0.0
    source: str = ""


@dataclass  
class RAGResult:
    answer: str
    sources: list[Document]
    confidence: float
    latency_ms: dict  # Per-stage latency


class RAGPipeline:
    """5-stage production RAG pipeline."""

    def __init__(
        self,
        query_processor,      # Stage 1
        retriever,            # Stage 2
        reranker,             # Stage 3
        context_assembler,    # Stage 4
        generator,            # Stage 5
    ):
        self.query_processor = query_processor
        self.retriever = retriever
        self.reranker = reranker
        self.context_assembler = context_assembler
        self.generator = generator

    async def run(self, query: str, top_k: int = 5) -> RAGResult:
        """Execute the full RAG pipeline."""
        import time
        latency = {}

        # Stage 1: Query Processing
        t0 = time.time()
        processed_queries = await self.query_processor.process(query)
        latency["query_processing_ms"] = int((time.time() - t0) * 1000)
        logger.info(f"Query processed: {len(processed_queries)} variants")

        # Stage 2: Retrieval
        t0 = time.time()
        candidates = []
        for q in processed_queries:
            docs = await self.retriever.retrieve(q, top_k=50)
            candidates.extend(docs)
        # Deduplicate by document ID
        seen = set()
        unique_candidates = []
        for doc in candidates:
            if doc.id not in seen:
                seen.add(doc.id)
                unique_candidates.append(doc)
        latency["retrieval_ms"] = int((time.time() - t0) * 1000)
        logger.info(f"Retrieved: {len(unique_candidates)} unique candidates")

        # Stage 3: Reranking
        t0 = time.time()
        reranked = await self.reranker.rerank(query, unique_candidates, top_k=top_k)
        latency["reranking_ms"] = int((time.time() - t0) * 1000)
        logger.info(f"Reranked: top {len(reranked)} documents")

        # Stage 4: Context Assembly
        t0 = time.time()
        context = self.context_assembler.assemble(reranked, max_tokens=3000)
        latency["assembly_ms"] = int((time.time() - t0) * 1000)

        # Stage 5: Generation
        t0 = time.time()
        answer, confidence = await self.generator.generate(
            query=query,
            context=context,
        )
        latency["generation_ms"] = int((time.time() - t0) * 1000)

        total = sum(latency.values())
        latency["total_ms"] = total
        logger.info(f"RAG pipeline complete: {total}ms total")

        return RAGResult(
            answer=answer,
            sources=reranked,
            confidence=confidence,
            latency_ms=latency,
        )
```

## Decision Tree: RAG vs Fine-Tuning vs Long Context

```
Do you need the model to use YOUR specific data?
  |
  NO --> Use base model as-is
  |
  YES --> Is the data changing frequently (daily/weekly)?
           |
           YES --> RAG (retrieval keeps data fresh without retraining)
           |
           NO --> Does the data fit in context window (< 128K tokens)?
                   |
                   YES --> Long context (simpler, no retrieval infra)
                   |       BUT: U-shaped attention curve degrades at 60-80% fill
                   |
                   NO --> Is it a knowledge base (facts, docs, policies)?
                           |
                           YES --> RAG (designed for knowledge retrieval)
                           |
                           NO --> Is it a behavioral change (tone, format, style)?
                                   |
                                   YES --> Fine-tuning (teaches behavior, not facts)
                                   NO  --> RAG + fine-tuning (knowledge + behavior)
```

## When NOT to Use RAG

- **The model already knows the answer**: General knowledge questions don't need retrieval
- **Behavioral adaptation**: Teaching a model to respond in a certain format is fine-tuning, not RAG
- **Small, static datasets**: If your entire knowledge base fits in context (<50K tokens), just include it directly
- **Real-time data**: RAG adds latency. If you need sub-100ms responses, pre-compute answers
- **Structured data queries**: SQL/database queries are better than RAG for structured data

## Tradeoffs

| Aspect | RAG | Fine-Tuning | Long Context |
|--------|-----|-------------|-------------|
| Data freshness | Real-time | Stale (re-train needed) | Current (load at query time) |
| Setup cost | Medium (vector DB, embeddings) | High (training infra) | Low |
| Per-query cost | Medium (retrieval + generation) | Low (generation only) | High (long prompts) |
| Accuracy on specific facts | High (if retrieved) | Medium (can hallucinate) | Degrades with length |
| Scalability | Millions of documents | Limited by training data | Limited by context window |
| Latency | +100-300ms for retrieval | No added latency | Slower inference (longer context) |
| Hallucination control | Good (grounded in sources) | Poor (parametric) | Medium |
| Infrastructure | Vector DB + embedding model | GPU training | None |

## Real-World Examples

### Customer Support RAG
```
Knowledge base: 10,000 support articles
Embedding model: text-embedding-3-small ($0.02/1M tokens)
Vector DB: Qdrant (self-hosted)
Reranker: Cohere Rerank ($0.002/query)
Generator: Claude 3.5 Sonnet

Pipeline:
  Query -> HyDE rewrite -> Hybrid search (BM25 + vector)
  -> Cohere rerank top 5 -> Assemble context -> Generate

Performance:
  - Retrieval accuracy (top-5): 92%
  - Answer accuracy: 87%
  - Latency: 1.2s average
  - Cost: $0.015 per query
```

### Legal Document RAG
```
Knowledge base: 50,000 legal documents, 500M tokens
Chunking: 512-token semantic chunks with 50-token overlap
Embedding: text-embedding-3-large (higher accuracy for legal terms)
Search: Hybrid (legal terms need exact match via BM25)
Reranker: BGE-reranker-large (self-hosted for data privacy)

Pipeline:
  Query -> Multi-query decomposition -> Hybrid search
  -> Cross-encoder rerank top 10 -> Assemble with citations -> Generate

Performance:
  - Retrieval accuracy (top-10): 89%
  - Citation accuracy: 95%
  - Latency: 2.1s average
```

## Failure Modes

| Failure | Stage | Symptom | Detection | Mitigation |
|---------|-------|---------|-----------|------------|
| Retrieval failure | 2 | Relevant docs not in candidates | Low retrieval recall metric | Better chunking, hybrid search, query rewriting |
| Ranking failure | 3 | Relevant docs ranked low | Relevant doc in candidates but not top-K | Improve reranker, tune threshold |
| Synthesis failure | 5 | LLM ignores or misinterprets context | Answer contradicts retrieved docs | Explicit grounding instructions |
| Stale data | 1-2 | Answer based on outdated information | Timestamp check on retrieved docs | Incremental index updates |
| Context overflow | 4 | Too much context, LLM loses focus | Answer quality drops with more docs | Limit to 5-7 docs, use summarization |
| Hallucination | 5 | LLM generates info not in context | Factual verification against sources | "Only answer from context" instruction |

## Source(s) and Further Reading

- RAG_Techniques repository (GitHub, 26.2k stars) -- comprehensive RAG technique catalog
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020) -- original RAG paper
- Qdrant, "Hybrid Search Documentation" (2024) -- vector + keyword search
- LlamaIndex Documentation -- production RAG framework
- LangChain RAG Documentation -- alternative RAG framework
- Anthropic, "Contextual Retrieval" (2024) -- chunking and retrieval improvements
