# Memory Sharding for Agent Systems

> Partitioning agent memory across nodes: sharding by user_id, namespace, or consistent hashing for vector store distribution. When to shard, how to rebalance, and platform-specific implementations.

## What It Is

Memory sharding distributes agent memory (conversation history, user preferences, vector embeddings) across multiple database nodes or partitions. Without sharding, a single memory store becomes a bottleneck at scale.

Sharding triggers:
- **>10M vectors** in a single vector store collection
- **>100K concurrent users** accessing memory simultaneously
- **>1TB** total memory storage
- **Query latency exceeding SLA** (p99 > 200ms for vector search)

### Sharding Strategies

| Strategy | Partition Key | Best For | Complexity |
|----------|---------------|----------|------------|
| **User-based** | `user_id` or `tenant_id` | Multi-tenant SaaS | Low |
| **Namespace-based** | `namespace` (e.g., "support", "sales") | Domain separation | Low |
| **Consistent hashing** | Hash of document ID | Even distribution | Medium |
| **Geographic** | User region | Latency optimization | High |
| **Time-based** | `created_at` month/quarter | Archival, cold storage | Medium |

## How It Works

### User-Based Sharding Architecture

```
              ┌─────────────────────────────────┐
              │           SHARD ROUTER            │
              │  route(user_id) → shard_number    │
              │  shard = hash(user_id) % N_SHARDS │
              └──────────┬──────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
    ┌─────▼─────┐  ┌────▼─────┐  ┌────▼─────┐
    │ Shard 0    │  │ Shard 1   │  │ Shard 2   │
    │            │  │           │  │           │
    │ Users:     │  │ Users:    │  │ Users:    │
    │ A, D, G    │  │ B, E, H   │  │ C, F, I   │
    │            │  │           │  │           │
    │ Vectors:   │  │ Vectors:  │  │ Vectors:  │
    │ 3.2M       │  │ 3.5M      │  │ 3.1M      │
    │            │  │           │  │           │
    │ Postgres   │  │ Postgres  │  │ Postgres  │
    │ + pgvector │  │ + pgvector│  │ + pgvector│
    └────────────┘  └───────────┘  └───────────┘
```

## Production Implementation

```python
import hashlib
from typing import Optional, Any
from dataclasses import dataclass


@dataclass
class ShardConfig:
    """Configuration for a single shard."""
    shard_id: int
    host: str
    port: int
    database: str
    max_vectors: int = 5_000_000


class MemoryShardRouter:
    """
    Routes memory operations to the correct shard based on partition key.
    
    Sharding strategies:
    - Modulo hashing: simple, but rebalancing is painful
    - Consistent hashing: handles node addition/removal gracefully
    - Range-based: good for time-series data
    """

    def __init__(self, shards: list[ShardConfig], strategy: str = "consistent"):
        self.shards = {s.shard_id: s for s in shards}
        self.num_shards = len(shards)
        self.strategy = strategy

        if strategy == "consistent":
            self._ring = self._build_hash_ring(shards)

    def _build_hash_ring(
        self, shards: list[ShardConfig], virtual_nodes: int = 150
    ) -> list[tuple[int, int]]:
        """Build a consistent hash ring with virtual nodes."""
        ring = []
        for shard in shards:
            for i in range(virtual_nodes):
                key = f"{shard.shard_id}:{i}"
                hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
                ring.append((hash_val, shard.shard_id))
        ring.sort(key=lambda x: x[0])
        return ring

    def get_shard(self, partition_key: str) -> ShardConfig:
        """Route a partition key to its shard."""
        if self.strategy == "modulo":
            shard_id = int(hashlib.md5(partition_key.encode()).hexdigest(), 16) % self.num_shards
            return self.shards[shard_id]

        elif self.strategy == "consistent":
            key_hash = int(hashlib.md5(partition_key.encode()).hexdigest(), 16)
            # Find the first node on the ring >= key_hash
            for ring_hash, shard_id in self._ring:
                if ring_hash >= key_hash:
                    return self.shards[shard_id]
            # Wrap around to first node
            return self.shards[self._ring[0][1]]

        raise ValueError(f"Unknown strategy: {self.strategy}")


# --- Sharded Vector Store ---

class ShardedVectorStore:
    """
    Vector store that distributes data across multiple shards.
    Each shard is an independent vector store instance.
    """

    def __init__(self, router: MemoryShardRouter):
        self.router = router
        self._connections: dict[int, Any] = {}

    def _get_connection(self, shard: ShardConfig):
        """Get or create a connection to a shard."""
        if shard.shard_id not in self._connections:
            # Example: Qdrant client per shard
            from qdrant_client import QdrantClient
            self._connections[shard.shard_id] = QdrantClient(
                host=shard.host,
                port=shard.port,
            )
        return self._connections[shard.shard_id]

    def upsert(
        self,
        user_id: str,
        vectors: list[list[float]],
        payloads: list[dict],
        ids: list[str],
    ) -> None:
        """Insert vectors into the correct shard based on user_id."""
        shard = self.router.get_shard(user_id)
        client = self._get_connection(shard)

        from qdrant_client.models import PointStruct
        points = [
            PointStruct(id=vid, vector=vec, payload={**payload, "user_id": user_id})
            for vid, vec, payload in zip(ids, vectors, payloads)
        ]
        client.upsert(collection_name="agent_memory", points=points)

    def search(
        self,
        user_id: str,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """Search within a specific user's shard."""
        shard = self.router.get_shard(user_id)
        client = self._get_connection(shard)

        from qdrant_client.models import Filter, FieldCondition, MatchValue
        results = client.search(
            collection_name="agent_memory",
            query_vector=query_vector,
            limit=top_k,
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
        )
        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]

    def cross_shard_search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search across ALL shards. Expensive but sometimes necessary
        (e.g., admin searching all users).
        
        Use sparingly -- this fans out to every shard.
        """
        import asyncio

        all_results = []
        for shard_id, shard in self.router.shards.items():
            client = self._get_connection(shard)
            results = client.search(
                collection_name="agent_memory",
                query_vector=query_vector,
                limit=top_k,
            )
            all_results.extend(
                {"id": r.id, "score": r.score, "payload": r.payload, "shard": shard_id}
                for r in results
            )

        # Re-rank across shards
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]


# --- Platform-Specific Implementations ---

# Pinecone Namespaces (built-in sharding)
PINECONE_SHARDING = """
import pinecone

# Pinecone uses "namespaces" as built-in sharding
pc = pinecone.Pinecone(api_key="...")
index = pc.Index("agent-memory")

# Each tenant gets their own namespace (automatic sharding)
index.upsert(
    vectors=[{"id": "v1", "values": [0.1, 0.2, ...]}],
    namespace=f"tenant_{tenant_id}"  # Built-in shard key
)

# Search within a namespace (only searches that shard)
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    namespace=f"tenant_{tenant_id}"
)
"""

# pgvector with Postgres Partitioning
PGVECTOR_PARTITIONING = """
-- Range partitioning by tenant_id hash
CREATE TABLE agent_memory (
    id UUID NOT NULL,
    tenant_id VARCHAR(64) NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    content TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    partition_key INTEGER GENERATED ALWAYS AS (
        abs(hashtext(tenant_id)) % 8  -- 8 partitions
    ) STORED
) PARTITION BY LIST (partition_key);

-- Create partitions
CREATE TABLE agent_memory_p0 PARTITION OF agent_memory FOR VALUES IN (0);
CREATE TABLE agent_memory_p1 PARTITION OF agent_memory FOR VALUES IN (1);
CREATE TABLE agent_memory_p2 PARTITION OF agent_memory FOR VALUES IN (2);
CREATE TABLE agent_memory_p3 PARTITION OF agent_memory FOR VALUES IN (3);
CREATE TABLE agent_memory_p4 PARTITION OF agent_memory FOR VALUES IN (4);
CREATE TABLE agent_memory_p5 PARTITION OF agent_memory FOR VALUES IN (5);
CREATE TABLE agent_memory_p6 PARTITION OF agent_memory FOR VALUES IN (6);
CREATE TABLE agent_memory_p7 PARTITION OF agent_memory FOR VALUES IN (7);

-- IVFFlat index per partition (much faster than global index)
CREATE INDEX ON agent_memory_p0 USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON agent_memory_p1 USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- ... repeat for all partitions

-- Query automatically routes to correct partition
SELECT content, 1 - (embedding <=> $1) AS similarity
FROM agent_memory
WHERE tenant_id = $2
ORDER BY embedding <=> $1
LIMIT 5;
"""

# Qdrant Distributed Mode
QDRANT_DISTRIBUTED = """
# qdrant-config.yaml for distributed mode
storage:
  optimizers:
    default_segment_number: 4

cluster:
  enabled: true
  p2p:
    port: 6335

# Collection with sharding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, ShardingMethod

client = QdrantClient("qdrant-node-1:6333")

client.create_collection(
    collection_name="agent_memory",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    shard_number=6,                              # Number of shards
    sharding_method=ShardingMethod.CUSTOM,        # Control shard placement
    replication_factor=2,                          # 2 replicas per shard
)
"""
```

## Decision Tree: When to Shard

```
    Should I shard agent memory?
                    │
         ┌──────────▼──────────┐
         │ How many vectors?    │
         └──┬──────────────┬──┘
          <1M           >10M
            │              │
     Don't shard      ┌───▼───┐
     (single node     │ SHARD │
      is fine)        └───────┘
            │
         1M-10M
            │
     ┌──────▼──────────┐
     │ Query latency    │
     │ p99 > 200ms?     │
     └──┬──────────┬───┘
       Yes         No
        │           │
    SHARD      Monitor and
               shard when needed
```

## When NOT to Use

1. **< 1M vectors**: A single node handles 1M vectors with < 50ms p99 latency. Sharding adds complexity without benefit.
2. **Single-tenant applications**: No need to isolate by user if there is only one user/tenant.
3. **Read-heavy, write-light**: If writes are rare, a single node with read replicas is simpler than sharding.
4. **Managed vector DBs**: Pinecone, Weaviate Cloud, and Qdrant Cloud handle sharding internally. Do not add application-level sharding on top.
5. **Prototype/MVP stage**: Premature sharding slows development. Start with one node, shard when metrics demand it.

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Linear scalability (add nodes = add capacity) | Cross-shard queries are slow and expensive |
| Fault isolation (one shard fails, others work) | Rebalancing when adding/removing shards |
| Per-tenant resource isolation | Increased operational complexity |
| Reduced index size per shard (faster searches) | Connection management for multiple shards |
| Enables geographic distribution | Data consistency across shards |

## Failure Modes

### 1. Hot Shard
One tenant has 10x more data than others, overloading their shard.
**Mitigation**: Monitor shard sizes. Split hot shards. Use consistent hashing for even distribution.

### 2. Cross-Shard Query Performance
Admin needs to search across all users, causing fan-out to every shard.
**Mitigation**: Maintain a global search index (e.g., Elasticsearch) for cross-tenant queries. Cache common cross-shard queries.

### 3. Rebalancing Data Loss
Adding a new shard requires moving data. If the migration fails, data is split between old and new locations.
**Mitigation**: Use consistent hashing (only ~1/N of data moves). Copy-then-delete migration. Verify after migration.

### 4. Connection Exhaustion
Each shard requires its own database connection pool. With 20 shards * 10 connections = 200 connections per application instance.
**Mitigation**: Connection pooling (PgBouncer). Limit concurrent connections per shard. Use connection-less protocols where possible.

### 5. Shard Key Choice Lock-In
You chose to shard by `user_id` but now need to query by `session_id` across users. The shard key cannot easily be changed.
**Mitigation**: Choose the most common query pattern as the shard key. Build secondary indexes for cross-shard queries.

## Sources and Further Reading

- [Qdrant Distributed Deployment](https://qdrant.tech/documentation/guides/distributed_deployment/)
- [Pinecone Namespaces](https://docs.pinecone.io/docs/namespaces)
- [pgvector Partitioning](https://github.com/pgvector/pgvector)
- [Consistent Hashing](https://en.wikipedia.org/wiki/Consistent_hashing)
- [Weaviate Multi-Tenancy](https://weaviate.io/developers/weaviate/concepts/data#multi-tenancy)
