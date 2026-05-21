# RAG Failure Modes
> RAG doesn't fail silently -- it confidently generates wrong answers from the wrong documents. Knowing the 6 failure modes and their detection signals is the difference between a demo and production.

## What It Is

RAG failure modes are the specific, categorizable ways a RAG pipeline can produce incorrect, incomplete, or unhelpful answers. Unlike a simple "error," RAG failures often return plausible-sounding answers that are subtly wrong -- the system found documents, assembled context, and generated a response, but the response is incorrect because of a failure in one of the pipeline stages.

Understanding these failure modes is critical because:
1. They are not caught by standard error handling (no exceptions are thrown)
2. The LLM sounds confident even when the context is wrong
3. Users trust RAG answers because they come "from the documents"
4. Each failure mode requires a different mitigation strategy

## The 6 Failure Modes

| # | Failure Mode | Stage | What Goes Wrong | Detection Signal |
|---|-------------|-------|-----------------|------------------|
| 1 | Retrieval Failure | Retrieval | Relevant documents not in candidates | Low semantic similarity scores in top results |
| 2 | Ranking Failure | Reranking | Relevant docs in candidates but ranked low | Relevant doc found in top-50 but not top-5 |
| 3 | Context Overflow | Assembly | Too much context, LLM loses focus | Answer quality degrades with more documents |
| 4 | Synthesis Failure | Generation | LLM ignores or misinterprets context | Answer contradicts information in retrieved docs |
| 5 | Stale Data | Indexing | Index contains outdated information | Document timestamps older than data freshness SLA |
| 6 | Missing Knowledge | Corpus | Information doesn't exist in the corpus | No documents above relevance threshold |

## How Each Failure Mode Works

### 1. Retrieval Failure

```
Query: "How to configure CORS in our API gateway?"
Documents in corpus: "Cross-Origin Resource Sharing setup guide for nginx proxy"

What happens:
  -> Query embedding ≠ Document embedding (vocabulary mismatch)
  -> Relevant document not in top-50 candidates
  -> All top-50 candidates are about unrelated API topics
  -> LLM generates answer from irrelevant context
  -> User gets wrong answer with confident tone

Detection:
  -> Average similarity score of top-5 < 0.3
  -> Large gap between top-1 score and remaining scores
  -> Documents retrieved are thematically unrelated to query

Mitigation:
  -> Hybrid search (BM25 + vector) catches vocabulary mismatches
  -> Query rewriting (HyDE) bridges question-document gap
  -> Better chunking (semantic, not fixed-size)
  -> Domain-adapted embedding model
```

### 2. Ranking Failure

```
Query: "What are the SLA terms for our enterprise plan?"
Top 50 candidates include the correct doc at position #37.

What happens:
  -> Bi-encoder ranks it low because embedding similarity is approximate
  -> Top 5 results are about "SLA monitoring" and "enterprise features" (related but wrong)
  -> LLM generates answer from adjacent-but-wrong documents
  -> Answer sounds right but contains wrong SLA terms

Detection:
  -> Recall@50 is high but Recall@5 is low
  -> Manual spot-checking reveals relevant docs ranked 20-50
  -> Reranker would have caught this

Mitigation:
  -> Add cross-encoder reranking (Cohere Rerank, BGE)
  -> Increase retrieval top-k (50-100) to catch more candidates
  -> Improve chunking to make relevant chunks more focused
```

### 3. Context Overflow

```
Query: "Summarize our security compliance requirements"
Retrieved: 15 documents, 12,000 tokens of context

What happens:
  -> LLM's attention degrades with long contexts (U-shaped curve)
  -> Information at positions 4-12 (middle) is effectively ignored
  -> LLM focuses on first 2-3 docs and last 2-3 docs
  -> Answer misses critical compliance requirements from middle docs

Detection:
  -> Answer quality inversely correlated with context length
  -> Key information from middle documents missing from answer
  -> Context fills >60% of the context window

Mitigation:
  -> Limit to 5-7 most relevant documents
  -> Use document summarization for long documents
  -> Place most important context first (primacy bias)
  -> Compress less relevant context
```

### 4. Synthesis Failure

```
Query: "Can enterprise customers get monthly billing?"
Context: Document clearly states "Enterprise plans are billed annually only."

What happens:
  -> LLM has parametric knowledge: "Many SaaS companies offer monthly billing"
  -> LLM's parametric knowledge conflicts with retrieved context
  -> LLM generates: "Yes, enterprise customers can typically choose monthly billing"
  -> Answer directly contradicts the retrieved document

Detection:
  -> Answer contradicts explicit statements in context
  -> LLM uses hedging language ("typically," "usually," "in most cases")
  -> Factual claims not traceable to any retrieved document

Mitigation:
  -> Strong grounding instructions: "ONLY use information from the provided context"
  -> Citation requirement: "Cite the specific document for every claim"
  -> Confidence scoring: ask LLM to rate its confidence
  -> Faithfulness evaluation in automated testing
```

### 5. Stale Data

```
Query: "What are our current pricing plans?"
Index contains: Pricing page from 6 months ago (plans have changed)

What happens:
  -> Retrieval finds the old pricing page (high relevance score)
  -> LLM generates answer with outdated prices
  -> User makes decisions based on wrong pricing
  -> No error signal -- everything "worked"

Detection:
  -> Document last_modified timestamp older than freshness SLA
  -> Crawl/sync timestamps show lag
  -> User reports contradict system answers

Mitigation:
  -> Metadata filtering: exclude docs older than X days
  -> Real-time sync for critical data (pricing, policies)
  -> Timestamp injection: "This document was last updated on [DATE]"
  -> Freshness indicator in the response
```

### 6. Missing Knowledge

```
Query: "What is our policy on cryptocurrency payments?"
Corpus: No document about cryptocurrency exists

What happens:
  -> Retrieval returns documents about "payments" (somewhat related)
  -> LLM uses these tangentially related docs to fabricate an answer
  -> "Based on our payment policies, cryptocurrency payments are..."
  -> Hallucinated policy that doesn't exist

Detection:
  -> All retrieved documents have low relevance scores (<0.3)
  -> Retrieved documents are thematically unrelated to the specific question
  -> LLM generates speculative language

Mitigation:
  -> Minimum relevance threshold: reject results below 0.3
  -> Explicit "I don't have information about this" instruction
  -> Track query topics with no good matches (knowledge gap report)
  -> Confidence-based abstention
```

## Production Implementation: Failure Detection System

```python
"""RAG failure detection and mitigation system."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import logging
import time

logger = logging.getLogger(__name__)


class FailureMode(str, Enum):
    RETRIEVAL_FAILURE = "retrieval_failure"
    RANKING_FAILURE = "ranking_failure"
    CONTEXT_OVERFLOW = "context_overflow"
    SYNTHESIS_FAILURE = "synthesis_failure"
    STALE_DATA = "stale_data"
    MISSING_KNOWLEDGE = "missing_knowledge"
    NONE = "none"


@dataclass
class RAGDiagnostics:
    """Diagnostics for a single RAG query."""
    query: str
    detected_failures: list[FailureMode]
    retrieval_scores: list[float]
    context_token_count: int
    document_ages_days: list[float]
    confidence: float
    warnings: list[str]


class RAGFailureDetector:
    """Detect failure modes in RAG pipeline outputs."""
    
    def __init__(
        self,
        min_relevance_threshold: float = 0.3,
        max_context_tokens: int = 4000,
        max_document_age_days: int = 90,
        max_context_fill_ratio: float = 0.6,
        context_window_size: int = 8000,
    ):
        self.min_relevance = min_relevance_threshold
        self.max_context_tokens = max_context_tokens
        self.max_doc_age = max_document_age_days
        self.max_fill_ratio = max_context_fill_ratio
        self.context_window = context_window_size

    def diagnose(
        self,
        query: str,
        retrieved_docs: list[dict],
        context_tokens: int,
        answer: str = "",
    ) -> RAGDiagnostics:
        """Run all failure detection checks."""
        failures = []
        warnings = []
        scores = [doc.get("score", 0) for doc in retrieved_docs]
        ages = [doc.get("age_days", 0) for doc in retrieved_docs]

        # 1. Retrieval Failure: all scores below threshold
        if not scores or max(scores) < self.min_relevance:
            failures.append(FailureMode.RETRIEVAL_FAILURE)
            warnings.append(
                f"Retrieval failure: Best relevance score ({max(scores) if scores else 0:.2f}) "
                f"is below threshold ({self.min_relevance}). "
                f"Results may not be relevant to the query."
            )

        # 2. Missing Knowledge: no docs above threshold
        if scores and all(s < self.min_relevance for s in scores[:5]):
            failures.append(FailureMode.MISSING_KNOWLEDGE)
            warnings.append(
                "No documents with sufficient relevance found. "
                "The knowledge base may not contain information about this topic."
            )

        # 3. Context Overflow: too many tokens
        fill_ratio = context_tokens / self.context_window
        if fill_ratio > self.max_fill_ratio:
            failures.append(FailureMode.CONTEXT_OVERFLOW)
            warnings.append(
                f"Context overflow: Using {fill_ratio:.0%} of context window. "
                f"Information in the middle may be lost due to attention degradation. "
                f"Consider reducing to {int(self.context_window * 0.4)} tokens."
            )

        # 4. Stale Data: old documents
        stale_docs = [age for age in ages if age > self.max_doc_age]
        if stale_docs:
            failures.append(FailureMode.STALE_DATA)
            warnings.append(
                f"{len(stale_docs)} document(s) are older than {self.max_doc_age} days. "
                f"Information may be outdated."
            )

        # 5. Ranking Failure: score drop-off pattern
        if len(scores) >= 5:
            top1 = scores[0]
            top5_avg = sum(scores[:5]) / 5
            if top1 > 0.5 and top5_avg < 0.35:
                # Large gap suggests ranking may have missed relevant docs
                failures.append(FailureMode.RANKING_FAILURE)
                warnings.append(
                    "Possible ranking failure: large relevance score drop-off. "
                    "Consider adding a reranker or increasing candidate pool."
                )

        # Confidence estimation
        confidence = self._estimate_confidence(scores, failures)

        if not failures:
            failures.append(FailureMode.NONE)

        return RAGDiagnostics(
            query=query,
            detected_failures=failures,
            retrieval_scores=scores[:10],
            context_token_count=context_tokens,
            document_ages_days=ages[:10],
            confidence=confidence,
            warnings=warnings,
        )

    def _estimate_confidence(
        self, scores: list[float], failures: list[FailureMode]
    ) -> float:
        """Estimate answer confidence based on retrieval quality."""
        if not scores:
            return 0.0

        base_confidence = min(scores[0], 1.0)  # Top score as baseline

        # Penalize for each failure mode
        penalty = len([f for f in failures if f != FailureMode.NONE]) * 0.2
        
        return max(0.0, base_confidence - penalty)


    def should_abstain(self, diagnostics: RAGDiagnostics) -> bool:
        """Should the system refuse to answer?"""
        critical_failures = {
            FailureMode.RETRIEVAL_FAILURE,
            FailureMode.MISSING_KNOWLEDGE,
        }
        return bool(critical_failures.intersection(diagnostics.detected_failures))


# Usage in pipeline:
detector = RAGFailureDetector()

async def safe_rag_query(query: str, pipeline) -> dict:
    """RAG query with failure detection and graceful degradation."""
    result = await pipeline.run(query)
    
    diagnostics = detector.diagnose(
        query=query,
        retrieved_docs=result.sources,
        context_tokens=sum(len(d.text.split()) for d in result.sources),  # approximate
        answer=result.answer,
    )
    
    if detector.should_abstain(diagnostics):
        return {
            "answer": "I don't have enough relevant information to answer this question "
                      "accurately. Please check the documentation directly or contact support.",
            "confidence": diagnostics.confidence,
            "warnings": diagnostics.warnings,
            "sources": [],
        }
    
    return {
        "answer": result.answer,
        "confidence": diagnostics.confidence,
        "warnings": diagnostics.warnings,
        "sources": [{"text": d.text[:200], "score": d.score} for d in result.sources],
    }
```

## Decision Tree: Diagnosing RAG Issues

```
Is the answer wrong?
  |
  YES --> Are relevant documents in the corpus?
  |         |
  |         NO  --> MISSING KNOWLEDGE
  |         |       Fix: Add missing documents to corpus
  |         |
  |         YES --> Were they retrieved (in top-50)?
  |                   |
  |                   NO  --> RETRIEVAL FAILURE
  |                   |       Fix: Hybrid search, query rewriting, better embeddings
  |                   |
  |                   YES --> Were they in the top-5 after reranking?
  |                             |
  |                             NO  --> RANKING FAILURE
  |                             |       Fix: Add/improve reranker
  |                             |
  |                             YES --> Did the LLM use them correctly?
  |                                       |
  |                                       NO  --> SYNTHESIS FAILURE
  |                                       |       Fix: Better grounding prompt, citations
  |                                       |
  |                                       YES --> Is the information current?
  |                                                 |
  |                                                 NO  --> STALE DATA
  |                                                 |       Fix: Index refresh, timestamps
  |                                                 |
  |                                                 YES --> CONTEXT OVERFLOW
  |                                                         Fix: Reduce context, better ordering
```

## When NOT to Use Failure Detection

- **Prototyping**: Over-engineering detection before the pipeline works at all
- **Simple Q&A**: Straightforward lookups with small, curated knowledge bases
- **Low-stakes applications**: When wrong answers have minimal consequences

## Tradeoffs

| Aspect | With Detection | Without Detection |
|--------|---------------|-------------------|
| Answer reliability | Known confidence level | Unknown |
| User trust | Abstains when unsure | Sounds confident even when wrong |
| Complexity | Diagnostic pipeline + thresholds | Simpler |
| Latency | +10-50ms for checks | No overhead |
| False abstentions | May refuse valid queries (threshold too high) | Never refuses |
| Debugging | Clear failure categorization | "It gave the wrong answer" (opaque) |

## Real-World Examples

### Production Dashboard Metrics
```
Weekly RAG Health Report (10,000 queries):
  - No failure detected:     72% (7,200 queries)
  - Retrieval failure:        8% (800) -- need better hybrid search
  - Ranking failure:           5% (500) -- adding Cohere reranker
  - Context overflow:          3% (300) -- reducing to 5 docs max
  - Synthesis failure:         4% (400) -- improving grounding prompt
  - Stale data:               3% (300) -- increasing sync frequency
  - Missing knowledge:         5% (500) -- knowledge gap report generated

Action items:
  1. Deploy hybrid search (estimated -60% retrieval failures)
  2. Add Cohere reranker (estimated -80% ranking failures)
  3. Cap context at 5 documents (estimated -90% overflow failures)
```

## Failure Modes of the Failure Detector Itself

| Meta-Failure | Cause | Impact | Fix |
|-------------|-------|--------|-----|
| False positive (over-abstention) | Threshold too high | System refuses valid queries | Lower min_relevance threshold |
| False negative (missed failure) | Threshold too low | Wrong answers pass through | Raise threshold, add more checks |
| Synthesis failure undetected | No faithfulness check | LLM contradicts context unnoticed | Add faithfulness eval (LLM-as-judge) |
| Stale threshold wrong | max_doc_age doesn't match data velocity | Stale data serves or fresh data rejected | Tune per data source |

## Source(s) and Further Reading

- Barnett et al., "Seven Failure Points When Engineering a RAG" (2024) -- taxonomy of RAG failures
- RAG_Techniques repository (GitHub, 26.2k stars) -- failure detection patterns
- Anthropic, "Contextual Retrieval" (2024) -- reducing retrieval failures
- RAGAS Framework -- automated RAG evaluation metrics
- Liu et al., "Lost in the Middle" (2023) -- context overflow and attention curves
