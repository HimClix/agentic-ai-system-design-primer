# Semantic Caching
> Cache semantically similar queries to avoid redundant LLM calls -- 20-40% cache hit rate for typical support workloads.

## What It Is

Semantic caching stores LLM responses and serves them for future queries that are semantically similar (not just string-identical). Unlike exact caching, which requires the same input text, semantic caching uses embedding similarity to match queries that mean the same thing but are phrased differently.

Examples that should hit the same cache entry:
- "How do I reset my password?"
- "I forgot my password, how to change it?"
- "password reset process"

## How It Works

### Cache Flow

```
User Query
    │
    ▼
┌──────────────────┐
│ Generate Embedding│ (embedding model: ~$0.02/1M tokens)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     similarity > threshold?
│ Vector Similarity │────── YES ──► Return cached response
│ Search (Redis/PG) │                (cache HIT)
└────────┬─────────┘
         │ NO (cache MISS)
         ▼
┌──────────────────┐
│   Call LLM        │ ($3-15/M tokens)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Store in Cache    │ (embedding + response + metadata)
└──────────────────┘
```

### Similarity Threshold Tuning

| Threshold | Cache Hit Rate | False Match Rate | Use Case |
|-----------|---------------|-----------------|----------|
| 0.98+ | 5-10% | ~0% | Safety-critical, must be nearly identical |
| 0.95 | 15-25% | 1-2% | General production (recommended start) |
| 0.92 | 25-35% | 3-5% | FAQ/support workloads |
| 0.88 | 35-50% | 8-12% | Aggressive caching, lower quality bar |
| 0.85 | 40-55% | 15-20% | Too aggressive, quality degrades |

**Start at 0.95 and lower based on measured false match rate.**

### Cache Hit Rates by Workload

| Workload Type | Expected Hit Rate | Why |
|--------------|-------------------|-----|
| Customer support (FAQ) | 30-40% | Many similar questions about same topics |
| Internal knowledge base Q&A | 25-35% | Employees ask similar questions |
| Code assistance | 10-20% | More unique context per query |
| Creative writing | 5-10% | Highly unique requests |
| Data analysis | 15-25% | Similar analytical patterns |

## Production Implementation

```python
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np
import redis
import anthropic


@dataclass
class CacheEntry:
    query: str
    response: str
    embedding: list[float]
    model: str
    created_at: float
    hit_count: int = 0
    metadata: dict = None


class SemanticCache:
    """
    Semantic cache for LLM responses using Redis + vector similarity.
    
    Uses a two-tier approach:
    1. Exact hash match (fastest, zero false positives)
    2. Semantic similarity search (handles paraphrases)
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        embedding_client: anthropic.Anthropic,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 86400,  # 24 hours
        max_cache_entries: int = 10_000,
        namespace: str = "default",
    ):
        self.redis = redis_client
        self.embedding_client = embedding_client
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
        self.max_entries = max_cache_entries
        self.namespace = namespace
    
    def _hash_key(self, query: str) -> str:
        """Generate exact-match hash key."""
        normalized = query.strip().lower()
        h = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        return f"cache:exact:{self.namespace}:{h}"
    
    def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding using Voyage or another embedding model."""
        # Using Anthropic's recommended embedding approach
        # In production, use voyage-3 or text-embedding-3-small
        response = self.embedding_client.messages.create(
            model="claude-haiku-3.5",
            max_tokens=1,
            system="Output only a single token.",
            messages=[{"role": "user", "content": text}],
        )
        # In practice, use a dedicated embedding API:
        # import voyageai
        # client = voyageai.Client()
        # result = client.embed([text], model="voyage-3")
        # return result.embeddings[0]
        raise NotImplementedError("Use a dedicated embedding model in production")
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_np = np.array(a)
        b_np = np.array(b)
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))
    
    def lookup(self, query: str, context_hash: Optional[str] = None) -> Optional[str]:
        """
        Look up a cached response for the given query.
        
        Args:
            query: The user's query
            context_hash: Hash of system prompt + tools (cache is invalid if context changes)
        
        Returns:
            Cached response string or None
        """
        # Tier 1: Exact match (< 1ms)
        exact_key = self._hash_key(query)
        cached = self.redis.get(exact_key)
        if cached:
            entry = json.loads(cached)
            if context_hash and entry.get("context_hash") != context_hash:
                return None  # Context changed, invalidate
            self.redis.hincrby(f"cache:stats:{self.namespace}", "exact_hits", 1)
            return entry["response"]
        
        # Tier 2: Semantic match (10-50ms)
        query_embedding = self._get_embedding(query)
        
        # Scan semantic cache entries (use Redis vector search in production)
        # In production, use RediSearch FT.SEARCH with vector similarity
        semantic_keys = self.redis.keys(f"cache:semantic:{self.namespace}:*")
        
        best_match = None
        best_similarity = 0.0
        
        for key in semantic_keys[:self.max_entries]:
            entry_data = self.redis.get(key)
            if not entry_data:
                continue
            entry = json.loads(entry_data)
            
            if context_hash and entry.get("context_hash") != context_hash:
                continue
            
            similarity = self._cosine_similarity(query_embedding, entry["embedding"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry
        
        if best_match and best_similarity >= self.threshold:
            self.redis.hincrby(f"cache:stats:{self.namespace}", "semantic_hits", 1)
            return best_match["response"]
        
        self.redis.hincrby(f"cache:stats:{self.namespace}", "misses", 1)
        return None
    
    def store(
        self,
        query: str,
        response: str,
        model: str,
        context_hash: Optional[str] = None,
    ) -> None:
        """Store a query-response pair in both exact and semantic caches."""
        embedding = self._get_embedding(query)
        
        entry = {
            "query": query,
            "response": response,
            "embedding": embedding,
            "model": model,
            "context_hash": context_hash,
            "created_at": time.time(),
        }
        
        # Store exact match
        exact_key = self._hash_key(query)
        self.redis.setex(exact_key, self.ttl, json.dumps(entry))
        
        # Store semantic entry
        semantic_key = f"cache:semantic:{self.namespace}:{hashlib.sha256(query.encode()).hexdigest()[:16]}"
        self.redis.setex(semantic_key, self.ttl, json.dumps(entry))
    
    def get_stats(self) -> dict:
        """Return cache performance statistics."""
        stats = self.redis.hgetall(f"cache:stats:{self.namespace}")
        exact_hits = int(stats.get(b"exact_hits", 0))
        semantic_hits = int(stats.get(b"semantic_hits", 0))
        misses = int(stats.get(b"misses", 0))
        total = exact_hits + semantic_hits + misses
        
        return {
            "exact_hits": exact_hits,
            "semantic_hits": semantic_hits,
            "total_hits": exact_hits + semantic_hits,
            "misses": misses,
            "hit_rate": round((exact_hits + semantic_hits) / max(total, 1) * 100, 1),
            "exact_hit_rate": round(exact_hits / max(total, 1) * 100, 1),
            "semantic_hit_rate": round(semantic_hits / max(total, 1) * 100, 1),
        }
    
    def invalidate(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries. Use when underlying data changes."""
        if pattern:
            keys = self.redis.keys(f"cache:*:{self.namespace}:{pattern}*")
        else:
            keys = self.redis.keys(f"cache:*:{self.namespace}:*")
        
        if keys:
            return self.redis.delete(*keys)
        return 0


# --- Production Redis Vector Search (RediSearch) ---

REDIS_VECTOR_INDEX_SCHEMA = """
FT.CREATE idx:semantic_cache 
  ON HASH PREFIX 1 "cache:semantic:" 
  SCHEMA 
    query TEXT 
    response TEXT 
    embedding VECTOR FLAT 6 
      TYPE FLOAT32 
      DIM 1024 
      DISTANCE_METRIC COSINE 
    context_hash TAG 
    created_at NUMERIC SORTABLE
"""

# Production query with RediSearch:
# FT.SEARCH idx:semantic_cache 
#   "(@context_hash:{ctx_hash})=>[KNN 5 @embedding $query_vec AS score]" 
#   PARAMS 2 query_vec <binary_embedding>
#   RETURN 2 response score
#   SORTBY score ASC
#   LIMIT 0 1
```

## Decision Tree: Exact vs Semantic Caching

```
Should I cache this query?
│
├── Is the response deterministic? (e.g., lookup, FAQ)
│   └── YES → Cache with long TTL (hours/days)
│
├── Does the response depend on user context?
│   ├── YES → Include context hash in cache key
│   └── NO → Cache globally
│
├── Is freshness critical? (e.g., stock prices, live data)
│   ├── YES → Short TTL (minutes) or skip caching
│   └── NO → Standard TTL (hours)
│
└── Which cache type?
    ├── High volume, exact repeats → Exact hash cache (faster, simpler)
    ├── Paraphrased queries common → Semantic cache (handles variations)
    └── Both → Two-tier (exact first, semantic fallback)
```

## When NOT to Use Semantic Caching

- **Personalized responses**: If every response must be tailored to user context, caching provides little benefit.
- **Rapidly changing data**: If the underlying knowledge changes frequently (live stock data), cached answers become stale.
- **Creative/generative tasks**: Users expect unique outputs for creative writing.
- **Low query volume**: < 100 queries/day, embedding costs may exceed LLM savings.

## Tradeoffs

| Aspect | Exact Caching | Semantic Caching |
|--------|--------------|-----------------|
| Lookup speed | < 1ms | 10-50ms |
| False positives | 0% | 1-15% (threshold dependent) |
| Hit rate | 5-15% | 20-40% |
| Storage cost | Low (key-value) | Higher (embeddings) |
| Implementation | Simple (hash map) | Complex (vector DB) |
| Cache invalidation | Easy (exact key) | Hard (which embeddings to remove?) |
| Embedding cost | None | ~$0.02/1M tokens |

## Real-World Examples

- **Customer support chatbot**: 30% cache hit rate on FAQ-heavy workload. Saves $400/month on $1,200 LLM spend.
- **Internal knowledge base**: 25% hit rate. Common questions about company policies, HR procedures.
- **Code completion**: 12% hit rate. Common import patterns and boilerplate code.

## Failure Modes

1. **False semantic matches**: "How to delete my account" matches "How to create an account" at threshold 0.88. Mitigation: start conservative (0.95), lower only with monitoring.
2. **Stale cache serving outdated info**: Policy changed but cached answer reflects old policy. Mitigation: cache invalidation on knowledge base updates.
3. **Cache poisoning**: A hallucinated response gets cached and served repeatedly. Mitigation: validate responses before caching, quality scoring.
4. **Embedding model drift**: Different embedding model versions produce incompatible vectors. Mitigation: version-tag cache entries, flush on model change.
5. **Memory bloat**: Storing thousands of embedding vectors in Redis. Mitigation: LRU eviction, TTL-based expiry, max entry limits.

## Source(s) and Further Reading

- GPTCache: https://github.com/zilliztech/GPTCache
- Redis Vector Search: https://redis.io/docs/interact/search-and-query/search/vectors/
- "Semantic Caching for LLM Applications" - Zilliz Blog
- LangChain Caching: https://python.langchain.com/docs/integrations/llm_caching/
- Voyage AI Embeddings: https://www.voyageai.com/
