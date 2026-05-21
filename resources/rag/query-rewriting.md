# Query Rewriting for RAG
> The user's query is almost never the best retrieval query. Query rewriting bridges the gap between how humans ask questions and how documents are written.

## What It Is

Query rewriting transforms a user's original query into one or more optimized queries before retrieval. This addresses the fundamental vocabulary mismatch problem: users ask questions in natural, conversational language, but documents are written in formal, technical language. Without query rewriting, relevant documents are missed because the query terms don't match the document terms.

Three major techniques:
1. **HyDE (Hypothetical Document Embeddings)**: Generate a hypothetical answer, embed that instead of the query
2. **Multi-Query Decomposition**: Break complex queries into simpler sub-queries
3. **Step-Back Prompting**: Generalize the query to retrieve broader context first

## How It Works

### HyDE (Hypothetical Document Embeddings)

```
Problem: User asks "why is my API slow?"
         But the relevant document says "API latency optimization techniques"
         Embedding "why is my API slow?" doesn't match well because
         the document never says "slow"

HyDE approach:
  Step 1: Ask LLM to generate a hypothetical answer
          "API slowness can be caused by several factors including
           database query latency, network overhead, inefficient
           serialization, and lack of caching..."

  Step 2: Embed the hypothetical answer (not the question)
          This embedding is much closer to how actual documents
          discuss the topic

  Step 3: Use hypothetical-answer embedding for retrieval
          Now finds: "API latency optimization techniques"
                     "Caching strategies to reduce response time"
                     "Database query optimization guide"
```

**Why it works**: The hypothetical answer uses the same vocabulary and style as real documents. The embedding of a document-like text matches other documents better than the embedding of a question.

### Multi-Query Decomposition

```
Complex query: "Compare our refund policy for enterprise vs SMB customers
                and what changed in Q4 2024"

Decomposed into:
  Query 1: "enterprise customer refund policy"
  Query 2: "SMB small business refund policy"
  Query 3: "refund policy changes Q4 2024"
  Query 4: "refund policy update October November December 2024"

Each sub-query retrieves its own candidates.
Results are merged via RRF, giving comprehensive coverage.
```

### Step-Back Prompting

```
Specific query: "Why does my Kubernetes pod keep getting OOMKilled 
                 when running the data pipeline?"

Step-back query: "Kubernetes pod memory management and OOMKilled errors"

The step-back query retrieves broader context:
  - How Kubernetes memory limits work
  - Common causes of OOMKilled
  - Memory profiling tools

This context helps the LLM give a more comprehensive answer than
if we only retrieved exact matches for the specific scenario.
```

## Production Implementation

### HyDE

```python
"""HyDE: Hypothetical Document Embeddings for query rewriting."""
from typing import Optional
import openai


class HyDEQueryRewriter:
    """Generate hypothetical answers for embedding-based retrieval.
    
    Cost: One LLM call per query (~$0.001 with GPT-4o-mini)
    Latency: ~200-500ms added
    Quality improvement: 10-30% on retrieval recall
    """

    def __init__(self, llm_client, embedding_model):
        self.llm = llm_client
        self.embedder = embedding_model

    async def rewrite(self, query: str, domain: str = "technical documentation") -> dict:
        """Generate a hypothetical document and return its embedding.
        
        Args:
            query: Original user query
            domain: Domain hint for better hypothetical generation
            
        Returns:
            Dict with hypothetical text and embedding
        """
        # Generate hypothetical answer
        response = await self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a {domain} writer. Given a question, write a "
                        f"short paragraph that would be found in documentation "
                        f"answering this question. Write factually and technically. "
                        f"Do not start with 'The answer is' -- write as if this "
                        f"is an excerpt from a real document."
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=200,
            temperature=0.0,  # Deterministic for caching
        )

        hypothetical_doc = response.choices[0].message.content

        # Embed the hypothetical document (not the original query)
        hyde_embedding = self.embedder.encode(hypothetical_doc)

        return {
            "original_query": query,
            "hypothetical_document": hypothetical_doc,
            "embedding": hyde_embedding,
            "strategy": "hyde",
        }
```

### Multi-Query Decomposition

```python
"""Multi-query decomposition for complex queries."""
import json


class MultiQueryRewriter:
    """Decompose complex queries into simpler sub-queries.
    
    Cost: One LLM call per query (~$0.001)
    Latency: ~200-400ms added
    Quality improvement: 15-25% on complex multi-part queries
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    async def decompose(self, query: str, max_queries: int = 4) -> list[str]:
        """Break a complex query into simpler sub-queries.
        
        Args:
            query: Original complex query
            max_queries: Maximum number of sub-queries to generate
            
        Returns:
            List of sub-queries (includes original query)
        """
        response = await self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a search query optimizer. Given a complex question, "
                        "break it down into 2-4 simpler, self-contained search queries "
                        "that together would retrieve all information needed to answer "
                        "the original question.\n\n"
                        "Rules:\n"
                        "- Each sub-query should be a standalone search query\n"
                        "- Include different phrasings and synonyms\n"
                        "- Cover all aspects of the original question\n"
                        "- Return as a JSON array of strings\n\n"
                        "Example:\n"
                        'Input: "How does Stripe handle refunds for disputed charges?"\n'
                        'Output: ["Stripe refund process", "Stripe disputed charge handling", '
                        '"Stripe chargeback refund policy"]'
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=300,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        try:
            result = json.loads(response.choices[0].message.content)
            sub_queries = result.get("queries", result.get("sub_queries", []))
            if isinstance(sub_queries, list):
                # Always include the original query
                all_queries = [query] + sub_queries[:max_queries - 1]
                return list(dict.fromkeys(all_queries))  # Deduplicate, preserve order
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: return original query
        return [query]
```

### Step-Back Prompting

```python
"""Step-back prompting for query generalization."""


class StepBackRewriter:
    """Generate a more general query to retrieve broader context.
    
    Cost: One LLM call per query (~$0.001)
    Latency: ~200-400ms added
    Quality improvement: 10-20% on specific/narrow queries
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    async def step_back(self, query: str) -> list[str]:
        """Generate both original and step-back queries.
        
        Returns both the original (for specific matches) and 
        a generalized query (for broader context).
        """
        response = await self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a search query generalizer. Given a specific question, "
                        "generate a broader, more general version that would retrieve "
                        "background context useful for answering the original question.\n\n"
                        "Examples:\n"
                        "Specific: 'Why does my Python 3.11 asyncio code deadlock on gather?'\n"
                        "General: 'Python asyncio concurrency patterns and deadlock prevention'\n\n"
                        "Specific: 'How to fix OOMKilled on pod rzp-api in namespace payments?'\n"
                        "General: 'Kubernetes pod memory management and OOMKilled troubleshooting'\n\n"
                        "Return ONLY the general query, nothing else."
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=100,
            temperature=0.0,
        )

        general_query = response.choices[0].message.content.strip()
        return [query, general_query]  # Both specific and general


# Combined query processor

class QueryProcessor:
    """Combines multiple query rewriting strategies."""

    def __init__(self, llm_client, embedding_model):
        self.hyde = HyDEQueryRewriter(llm_client, embedding_model)
        self.multi_query = MultiQueryRewriter(llm_client)
        self.step_back = StepBackRewriter(llm_client)

    async def process(
        self, query: str, strategy: str = "auto"
    ) -> list[str]:
        """Process a query using the appropriate strategy.
        
        Args:
            query: Original user query
            strategy: "hyde", "multi_query", "step_back", or "auto"
        """
        if strategy == "auto":
            strategy = self._detect_strategy(query)

        if strategy == "hyde":
            result = await self.hyde.rewrite(query)
            return [result["hypothetical_document"]]
        elif strategy == "multi_query":
            return await self.multi_query.decompose(query)
        elif strategy == "step_back":
            return await self.step_back.step_back(query)
        else:
            return [query]

    def _detect_strategy(self, query: str) -> str:
        """Auto-detect the best strategy based on query characteristics."""
        words = query.split()

        # Complex multi-part queries -> decompose
        if len(words) > 15 or " and " in query.lower() or " vs " in query.lower():
            return "multi_query"

        # Very specific queries -> step back
        if any(kw in query.lower() for kw in ["my", "specific", "exact", "pod", "error"]):
            return "step_back"

        # Short conceptual queries -> HyDE
        if len(words) < 10:
            return "hyde"

        return "multi_query"
```

## Decision Tree: Which Technique to Use

```
What kind of query is it?
  |
  |-- Short, conceptual ("what is X", "how does Y work")?
  |     --> HyDE (hypothetical answer matches document style)
  |
  |-- Complex, multi-part ("compare X and Y", "X and also Y")?
  |     --> Multi-query decomposition (cover all sub-topics)
  |
  |-- Very specific ("error X on pod Y in namespace Z")?
  |     --> Step-back prompting (get general context + specific)
  |
  |-- Already well-formed search query?
  |     --> No rewriting needed (pass through directly)
  |
  |-- Not sure?
        --> Multi-query is the safest default
```

## When NOT to Rewrite Queries

- **Already optimized queries**: If the user provides a clear search query, rewriting may add noise
- **Exact-match lookups**: "Get order #12345" should not be rewritten
- **Latency-critical paths**: Each rewrite adds 200-500ms of LLM inference
- **High-volume, low-variance queries**: Repeated similar queries should use cached rewrites
- **Very short queries**: "pricing" doesn't benefit from decomposition

## Tradeoffs

| Technique | Latency Added | Cost Added | Best Improvement | Worst Case |
|-----------|--------------|-----------|------------------|------------|
| HyDE | 200-500ms | ~$0.001 | +30% recall on concept queries | Hallucinated hypothesis misleads retrieval |
| Multi-Query | 200-400ms | ~$0.001 | +25% recall on complex queries | Redundant sub-queries waste retrieval time |
| Step-Back | 200-400ms | ~$0.001 | +20% on specific queries | Over-generalized query retrieves noise |
| None | 0ms | $0 | Baseline | Vocabulary mismatch misses documents |

## Real-World Examples

### HyDE for Customer Support
```
Query: "why can't I accept payments?"
HyDE generates: "Payment acceptance failures can occur due to several reasons 
  including expired API keys, incorrect merchant category codes, missing KYC 
  verification, or disabled payment methods in the merchant dashboard..."
Retrieval now finds: "Troubleshooting payment acceptance issues" (exact topic match)
```

## Failure Modes

| Failure | Cause | Impact | Fix |
|---------|-------|--------|-----|
| HyDE hallucination | LLM generates wrong hypothesis | Retrieves wrong documents | Use low temperature (0.0), domain-specific prompts |
| Over-decomposition | Simple query split into too many parts | Diluted retrieval, wasted compute | Limit to 4 sub-queries max |
| Step-back too general | "How to fix X" becomes "general debugging" | Retrieves noise | Include original specific query in results |
| Rewrite changes intent | Rewriting alters the original question's meaning | Wrong documents retrieved | Always include original query |
| Latency budget exceeded | Multiple LLM calls for rewriting | Slow response | Cache rewrites for common queries |

## Source(s) and Further Reading

- Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2023) -- HyDE paper
- Zheng et al., "Take a Step Back: Evoking Reasoning via Abstraction" (2023) -- step-back prompting
- RAG_Techniques repository (GitHub, 26.2k stars) -- query rewriting patterns
- LlamaIndex, "Query Transformations" -- multi-query implementation
- Anthropic, "Contextual Retrieval" (2024) -- query-context alignment
