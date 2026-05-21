# Multi-Tenancy for Agentic Systems
> Every dimension of an agent must be isolated per tenant: memory, credentials, budgets, RAG indexes, model routing, and audit trails.

## What It Is

Multi-tenancy is the ability to serve multiple customers (tenants) from a single deployment of the agent system while maintaining complete isolation between them. This is a hard requirement for any B2B SaaS product built on agentic AI.

The challenge is that agents interact with many systems -- LLMs, tools, databases, vector stores -- and **every single one** needs tenant-scoped isolation.

## How It Works

### The 7 Dimensions of Tenant Isolation

```
1. Memory Isolation      → Each tenant's conversation history is invisible to others
2. Tool Credentials      → Each tenant has their own API keys for tools
3. Token Budgets         → Each tenant has independent spending limits
4. RAG Indexes           → Each tenant searches only their own documents
5. Model Routing         → Each tenant can have different model preferences
6. Audit Trails          → Each tenant's logs are accessible only to them
7. Configuration         → Each tenant can customize agent behavior
```

### Isolation Architecture

```
Tenant A Request                    Tenant B Request
     │                                    │
     ▼                                    ▼
┌─────────────────────────────────────────────────┐
│              API Gateway                         │
│  Extract tenant_id from JWT / API key            │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│  Tenant Context  │    │  Tenant Context  │
│  tenant_id: A    │    │  tenant_id: B    │
│  model: sonnet   │    │  model: opus     │
│  budget: $500/mo │    │  budget: $2K/mo  │
│  tools: [a,b,c]  │    │  tools: [d,e,f]  │
│  rag_ns: ns_a    │    │  rag_ns: ns_b    │
└────────┬────────┘    └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│ Namespaced       │    │ Namespaced       │
│ Resources:       │    │ Resources:       │
│ - Redis ns:A:*   │    │ - Redis ns:B:*   │
│ - PG schema: A   │    │ - PG schema: B   │
│ - Pinecone ns: A │    │ - Pinecone ns: B │
│ - Vault path: A/ │    │ - Vault path: B/ │
└─────────────────┘    └─────────────────┘
```

## Production Implementation

```python
from dataclasses import dataclass, field
from typing import Optional
import redis.asyncio as redis


@dataclass
class TenantConfig:
    tenant_id: str
    display_name: str
    
    # Model routing
    default_model: str = "claude-sonnet-4"
    allowed_models: list[str] = field(default_factory=lambda: [
        "claude-haiku-3.5", "claude-sonnet-4"
    ])
    
    # Budgets
    monthly_token_budget: int = 10_000_000
    daily_token_budget: int = 500_000
    max_session_tokens: int = 50_000
    
    # RAG
    rag_namespace: str = ""  # Set to tenant_id by default
    rag_collection: str = ""
    
    # Tools
    enabled_tools: list[str] = field(default_factory=list)
    tool_credentials_path: str = ""  # Vault path
    
    # Agent behavior
    system_prompt_override: Optional[str] = None
    max_steps_per_session: int = 25
    confidence_threshold: float = 0.85
    
    def __post_init__(self):
        if not self.rag_namespace:
            self.rag_namespace = f"tenant_{self.tenant_id}"
        if not self.tool_credentials_path:
            self.tool_credentials_path = f"secret/agents/{self.tenant_id}"


class TenantContextManager:
    """Manages tenant-scoped resources for agent execution."""
    
    def __init__(self, redis_client: redis.Redis, config_store):
        self.redis = redis_client
        self.config_store = config_store
        self._tenant_configs: dict[str, TenantConfig] = {}
    
    async def get_config(self, tenant_id: str) -> TenantConfig:
        """Load tenant configuration (cached)."""
        if tenant_id not in self._tenant_configs:
            config = await self.config_store.get_tenant_config(tenant_id)
            self._tenant_configs[tenant_id] = config
        return self._tenant_configs[tenant_id]
    
    def namespace_key(self, tenant_id: str, key: str) -> str:
        """Namespace a Redis key to a tenant."""
        return f"tenant:{tenant_id}:{key}"
    
    async def get_conversation_history(
        self, tenant_id: str, session_id: str
    ) -> list[dict]:
        """Get conversation history scoped to tenant."""
        key = self.namespace_key(tenant_id, f"session:{session_id}:messages")
        messages = await self.redis.lrange(key, 0, -1)
        return [json.loads(m) for m in messages]
    
    async def get_rag_results(
        self, tenant_id: str, query: str, top_k: int = 5
    ) -> list[dict]:
        """Search RAG index scoped to tenant namespace."""
        config = await self.get_config(tenant_id)
        
        # Pinecone namespace isolation
        results = self.pinecone_index.query(
            vector=embed(query),
            top_k=top_k,
            namespace=config.rag_namespace,  # Tenant-scoped namespace
            include_metadata=True,
        )
        return results.matches
    
    async def get_tool_credentials(self, tenant_id: str, tool_name: str) -> dict:
        """Fetch tool credentials from Vault, scoped to tenant."""
        config = await self.get_config(tenant_id)
        vault_path = f"{config.tool_credentials_path}/{tool_name}"
        return await self.vault_client.read(vault_path)
    
    async def check_budget(self, tenant_id: str, estimated_tokens: int) -> bool:
        """Check if tenant has remaining budget."""
        config = await self.get_config(tenant_id)
        month = time.strftime("%Y-%m")
        key = self.namespace_key(tenant_id, f"budget:monthly:{month}")
        used = int(await self.redis.get(key) or 0)
        return (used + estimated_tokens) <= config.monthly_token_budget
    
    async def record_usage(self, tenant_id: str, tokens: int):
        """Record token usage for tenant."""
        month = time.strftime("%Y-%m")
        key = self.namespace_key(tenant_id, f"budget:monthly:{month}")
        await self.redis.incrby(key, tokens)
        await self.redis.expire(key, 32 * 86400)


# --- Database isolation ---

# Option 1: Schema-per-tenant (recommended for < 1000 tenants)
CREATE_TENANT_SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS tenant_{tenant_id};

CREATE TABLE tenant_{tenant_id}.agent_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    status VARCHAR(20) NOT NULL,
    total_tokens INTEGER DEFAULT 0,
    total_cost DECIMAL(10,6) DEFAULT 0
);

CREATE TABLE tenant_{tenant_id}.tool_calls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID REFERENCES tenant_{tenant_id}.agent_runs(id),
    tool_name VARCHAR(100) NOT NULL,
    input JSONB,
    output JSONB,
    status VARCHAR(20),
    latency_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE tenant_{tenant_id}.agent_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);
"""

# Option 2: Row-level isolation (recommended for > 1000 tenants)
CREATE_SHARED_TABLES_SQL = """
CREATE TABLE agent_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(50) NOT NULL,
    session_id UUID NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    status VARCHAR(20) NOT NULL,
    total_tokens INTEGER DEFAULT 0
);

-- Row Level Security
ALTER TABLE agent_runs ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON agent_runs
    USING (tenant_id = current_setting('app.current_tenant'));
    
-- Index for performance
CREATE INDEX idx_agent_runs_tenant ON agent_runs(tenant_id, created_at DESC);
"""
```

## Decision Tree: Isolation Strategy

```
How many tenants?
│
├── < 50 tenants
│   └── Schema-per-tenant
│       ├── Complete query isolation
│       ├── Easy backup/restore per tenant
│       └── Hard to manage 50+ schemas
│
├── 50-1000 tenants
│   └── Row-level security (RLS) in shared tables
│       ├── Single schema, tenant_id column
│       ├── PostgreSQL RLS policies
│       └── Indexes on tenant_id
│
└── > 1000 tenants
    └── Database-per-tenant (if data is large) or RLS with partitioning
        ├── Partition by tenant_id
        ├── Separate connection pools
        └── Consider managed multi-tenant DBs
```

## When NOT to Use Full Multi-Tenancy

- **Single-tenant deployment**: Self-hosted by customers on their own infrastructure.
- **Internal tools**: Used only by your company. One tenant.
- **B2C products**: Users don't have "tenants" -- use user-level isolation instead (lighter weight).

## Tradeoffs

| Isolation Level | Security | Operational Cost | Performance | Customization |
|----------------|----------|-----------------|------------|--------------|
| Shared everything | Low | Lowest | Best | Limited |
| Namespace isolation | Medium | Low | Good | Per-namespace |
| Schema-per-tenant | High | Medium | Good | Per-schema |
| Database-per-tenant | Highest | Highest | Best (no contention) | Full |

## Failure Modes

1. **Cross-tenant data leak**: Forgetting to apply tenant filter in a query. Mitigation: RLS policies at database level, middleware that always injects tenant_id.
2. **Noisy neighbor**: One tenant's heavy workload degrades others. Mitigation: per-tenant rate limiting, resource quotas, queue priorities.
3. **Credential mix-up**: Tenant A's API key used for tenant B's tool call. Mitigation: credential fetch always includes tenant_id, never cache credentials across tenants.
4. **Budget enforcement race condition**: Two concurrent requests both pass budget check, both execute, total exceeds budget. Mitigation: atomic Redis operations (INCRBY with check).

## Source(s) and Further Reading

- PostgreSQL Row Level Security: https://www.postgresql.org/docs/current/ddl-rowsecurity.html
- Pinecone Namespaces: https://docs.pinecone.io/guides/data/understanding-namespaces
- HashiCorp Vault Multi-Tenancy: https://developer.hashicorp.com/vault/tutorials/enterprise/namespaces
- "Multi-Tenant SaaS Architecture" - AWS Well-Architected
