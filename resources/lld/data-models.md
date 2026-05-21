# Data Models for Agentic Systems
> Three core tables: agent_runs (execution history), tool_calls (action audit trail), agent_memory (persistent knowledge with pgvector embeddings).

## What It Is

Data models for agents define how execution history, tool call audit trails, conversation state, and agent memory are stored persistently. These schemas support three critical needs: debugging (what happened in this run?), compliance (what actions were taken?), and intelligence (what has the agent learned?).

## How It Works

### Core Tables

```
agent_runs           ← One row per agent invocation
├── tool_calls       ← One row per tool call within a run
├── agent_messages   ← One row per message in conversation
└── agent_checkpoints← LangGraph state snapshots

agent_memory         ← Persistent knowledge with embeddings
agent_sessions       ← User session tracking
```

## Production Implementation

### SQL Schema

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- Table: agent_runs
-- Purpose: Track every agent invocation with metadata
-- ============================================================
CREATE TABLE agent_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(50) NOT NULL,
    session_id UUID NOT NULL,
    user_id VARCHAR(100),
    
    -- Execution status
    status VARCHAR(20) NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled', 'escalated')),
    
    -- Input/Output
    input_message TEXT NOT NULL,
    output_message TEXT,
    
    -- Execution metrics
    steps_completed INTEGER DEFAULT 0,
    total_input_tokens INTEGER DEFAULT 0,
    total_output_tokens INTEGER DEFAULT 0,
    total_cost_usd DECIMAL(10, 6) DEFAULT 0,
    model_used VARCHAR(50),
    confidence DECIMAL(3, 2),
    
    -- Error tracking
    error_message TEXT,
    error_code VARCHAR(50),
    
    -- Timing
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}'
);

-- Indexes for common queries
CREATE INDEX idx_runs_tenant_created ON agent_runs(tenant_id, created_at DESC);
CREATE INDEX idx_runs_session ON agent_runs(session_id, created_at DESC);
CREATE INDEX idx_runs_status ON agent_runs(status) WHERE status IN ('queued', 'running');
CREATE INDEX idx_runs_user ON agent_runs(user_id, created_at DESC);
CREATE INDEX idx_runs_tags ON agent_runs USING GIN(tags);

-- Partition by month for high-volume deployments
-- CREATE TABLE agent_runs (
--     ...
-- ) PARTITION BY RANGE (created_at);
-- CREATE TABLE agent_runs_2025_01 PARTITION OF agent_runs
--     FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');


-- ============================================================
-- Table: tool_calls
-- Purpose: Audit trail of every tool invocation
-- ============================================================
CREATE TABLE tool_calls (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    tenant_id VARCHAR(50) NOT NULL,
    
    -- Tool identification
    tool_name VARCHAR(100) NOT NULL,
    tool_version VARCHAR(20),
    
    -- Input/Output (JSONB for flexible schemas)
    input_args JSONB NOT NULL,
    output_result JSONB,
    
    -- Execution
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'success', 'error', 'timeout', 'rejected')),
    error_message TEXT,
    
    -- Performance
    latency_ms INTEGER,
    retry_count INTEGER DEFAULT 0,
    
    -- Agent reasoning (why was this tool called?)
    reasoning TEXT,
    
    -- Timing
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    
    -- Step tracking
    step_number INTEGER NOT NULL
);

CREATE INDEX idx_tool_calls_run ON tool_calls(run_id, step_number);
CREATE INDEX idx_tool_calls_tool ON tool_calls(tool_name, created_at DESC);
CREATE INDEX idx_tool_calls_tenant ON tool_calls(tenant_id, created_at DESC);
CREATE INDEX idx_tool_calls_status ON tool_calls(status) WHERE status NOT IN ('success');


-- ============================================================
-- Table: agent_memory
-- Purpose: Persistent knowledge with vector embeddings
-- ============================================================
CREATE TABLE agent_memory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(100),
    
    -- Memory content
    memory_type VARCHAR(50) NOT NULL
        CHECK (memory_type IN ('fact', 'preference', 'procedure', 'episode', 'summary')),
    content TEXT NOT NULL,
    
    -- Vector embedding for semantic search
    embedding vector(1024),  -- Voyage-3 dimensions
    
    -- Metadata
    source VARCHAR(100),     -- Where this memory came from
    confidence DECIMAL(3, 2) DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    
    -- Lifecycle
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    expires_at TIMESTAMPTZ,  -- NULL = never expires
    is_active BOOLEAN DEFAULT true,
    
    -- Structured metadata
    metadata JSONB DEFAULT '{}'
);

-- Vector similarity search index (IVFFlat for large datasets)
CREATE INDEX idx_memory_embedding ON agent_memory 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Standard indexes
CREATE INDEX idx_memory_tenant_user ON agent_memory(tenant_id, user_id, memory_type);
CREATE INDEX idx_memory_type ON agent_memory(memory_type, created_at DESC);
CREATE INDEX idx_memory_active ON agent_memory(is_active) WHERE is_active = true;

-- For HNSW index (better for smaller datasets, faster queries):
-- CREATE INDEX idx_memory_embedding_hnsw ON agent_memory 
--     USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);


-- ============================================================
-- Table: agent_sessions
-- Purpose: Track user sessions for conversation continuity
-- ============================================================
CREATE TABLE agent_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(100),
    
    -- Session state
    status VARCHAR(20) DEFAULT 'active'
        CHECK (status IN ('active', 'idle', 'closed', 'expired')),
    
    -- Conversation summary (updated periodically)
    summary TEXT,
    turn_count INTEGER DEFAULT 0,
    
    -- Token tracking
    total_tokens_used INTEGER DEFAULT 0,
    total_cost_usd DECIMAL(10, 6) DEFAULT 0,
    
    -- Timing
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_activity_at TIMESTAMPTZ DEFAULT now(),
    expires_at TIMESTAMPTZ,
    
    -- Configuration
    model_preference VARCHAR(50),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_sessions_tenant_user ON agent_sessions(tenant_id, user_id, status);
CREATE INDEX idx_sessions_active ON agent_sessions(status, last_activity_at) WHERE status = 'active';
```

### Python ORM Models (SQLAlchemy)

```python
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector
import uuid
from datetime import datetime

Base = declarative_base()


class AgentRun(Base):
    __tablename__ = "agent_runs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String(50), nullable=False, index=True)
    session_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(String(100))
    status = Column(String(20), nullable=False, default="queued")
    input_message = Column(Text, nullable=False)
    output_message = Column(Text)
    steps_completed = Column(Integer, default=0)
    total_input_tokens = Column(Integer, default=0)
    total_output_tokens = Column(Integer, default=0)
    total_cost_usd = Column(Float, default=0.0)
    model_used = Column(String(50))
    confidence = Column(Float)
    error_message = Column(Text)
    error_code = Column(String(50))
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    metadata = Column(JSONB, default={})
    
    tool_calls = relationship("ToolCall", back_populates="run", cascade="all, delete-orphan")


class ToolCall(Base):
    __tablename__ = "tool_calls"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False)
    tenant_id = Column(String(50), nullable=False)
    tool_name = Column(String(100), nullable=False)
    input_args = Column(JSONB, nullable=False)
    output_result = Column(JSONB)
    status = Column(String(20), nullable=False, default="pending")
    error_message = Column(Text)
    latency_ms = Column(Integer)
    reasoning = Column(Text)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    step_number = Column(Integer, nullable=False)
    
    run = relationship("AgentRun", back_populates="tool_calls")


class AgentMemory(Base):
    __tablename__ = "agent_memory"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String(50), nullable=False)
    user_id = Column(String(100))
    memory_type = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1024))  # pgvector
    source = Column(String(100))
    confidence = Column(Float, default=1.0)
    access_count = Column(Integer, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    metadata = Column(JSONB, default={})


# --- Query Examples ---

async def get_recent_runs(session, tenant_id: str, limit: int = 20):
    """Get recent runs for a tenant."""
    return await session.execute(
        select(AgentRun)
        .where(AgentRun.tenant_id == tenant_id)
        .order_by(AgentRun.created_at.desc())
        .limit(limit)
    )

async def search_memory(session, tenant_id: str, query_embedding: list, top_k: int = 5):
    """Semantic search over agent memory using pgvector."""
    return await session.execute(
        select(AgentMemory)
        .where(AgentMemory.tenant_id == tenant_id)
        .where(AgentMemory.is_active == True)
        .order_by(AgentMemory.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )
```

### Indexing Strategy

| Table | Index | Purpose | Type |
|-------|-------|---------|------|
| agent_runs | (tenant_id, created_at DESC) | List runs by tenant | B-tree |
| agent_runs | (status) WHERE status IN (...) | Find active runs | Partial B-tree |
| tool_calls | (run_id, step_number) | Get tool calls for a run | B-tree |
| tool_calls | (tool_name, created_at DESC) | Tool usage analytics | B-tree |
| agent_memory | (embedding) | Semantic search | IVFFlat/HNSW |
| agent_memory | (tenant_id, user_id, memory_type) | Filter by scope | B-tree |

### Partitioning for High-Volume

```sql
-- Partition agent_runs by month (for > 1M runs/month)
ALTER TABLE agent_runs RENAME TO agent_runs_old;

CREATE TABLE agent_runs (
    LIKE agent_runs_old INCLUDING ALL
) PARTITION BY RANGE (created_at);

-- Create monthly partitions
CREATE TABLE agent_runs_y2025m01 PARTITION OF agent_runs
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE agent_runs_y2025m02 PARTITION OF agent_runs
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
-- ... auto-create with pg_partman
```

## When NOT to Use This Schema

- **Prototype**: Use SQLite or just in-memory dicts. Don't over-engineer storage early.
- **< 1000 runs/day**: A single table without partitioning is fine.
- **Fully managed platform**: LangGraph Cloud handles persistence for you.

## Tradeoffs

| Storage | Best For | Drawback |
|---------|----------|----------|
| PostgreSQL + pgvector | All-in-one (relational + vector) | Vector search slower than dedicated vector DB at scale |
| PostgreSQL + Pinecone | Relational data + fast vector search | Two databases to manage |
| Redis + PostgreSQL | Fast session state + durable history | Redis data loss risk on restart |

## Failure Modes

1. **pgvector index rebuild**: IVFFlat index needs periodic rebuilding as data grows. Mitigation: schedule weekly `REINDEX` during low traffic.
2. **JSONB bloat**: Large tool results in JSONB columns cause table bloat. Mitigation: store large results in S3, keep reference in JSONB.
3. **Partition management**: Forgetting to create future partitions. Mitigation: use pg_partman for automatic partition creation.

## Source(s) and Further Reading

- pgvector: https://github.com/pgvector/pgvector
- pgvector SQLAlchemy: https://github.com/pgvector/pgvector-python
- PostgreSQL Partitioning: https://www.postgresql.org/docs/current/ddl-partitioning.html
- pg_partman: https://github.com/pgpartman/pg_partman
- LangGraph Persistence: https://langchain-ai.github.io/langgraph/concepts/persistence/
