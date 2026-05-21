# Episodic Memory: Execution Traces

> Execution traces let an agent say "I've seen this before -- last time, step 3 failed because of X, so this time I'll try Y first."

## What It Is

Execution traces are structured records of past agent runs that can be retrieved and fed into future context to improve decision-making. Unlike checkpointing (which stores exact state for resumption), execution traces store *learnings* from past experiences -- what worked, what failed, what decisions were made and why.

This is true episodic memory: the agent's autobiographical record of past events that informs future behavior. A human debugging a production issue does not start from scratch every time -- they recall "last time we saw this error pattern, it was a connection pool leak." Execution traces give agents the same capability.

### Traces vs Checkpoints vs Memory

| Aspect | Execution Traces | Checkpoints | Long-Term Memory |
|---|---|---|---|
| **What** | Summary of past runs | Exact state at each step | Extracted facts |
| **Why** | Learn from experience | Resume interrupted work | Personalize responses |
| **Query by** | Similarity to current task | Thread ID (exact) | Semantic similarity |
| **Scope** | Task/workflow level | Step level | Fact level |
| **Example** | "Debugging OOM: checked heap, found leak in handler" | `{state: {step: 3, heap_size: "4GB"}}` | "Service X uses 4GB heap" |

## How It Works

### Trace Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION TRACE SYSTEM                       │
│                                                                 │
│  WRITE PATH (during agent execution):                           │
│                                                                 │
│  Agent Run                                                      │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                        │
│  │Step 1│→ │Step 2│→ │Step 3│→ │Result│                         │
│  │plan  │  │search│  │fix   │  │done  │                         │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘                        │
│     │         │         │         │                             │
│     └─────────┴─────────┴─────────┘                             │
│                    │                                            │
│              ┌─────▼──────┐                                     │
│              │   TRACE    │                                     │
│              │ FORMATTER  │                                     │
│              │            │                                     │
│              │ Structured │                                     │
│              │ summary of │                                     │
│              │ the run    │                                     │
│              └─────┬──────┘                                     │
│                    │                                            │
│              ┌─────▼──────────┐                                 │
│              │  TRACE STORE   │                                 │
│              │  (Vector DB +  │                                 │
│              │   Metadata)    │                                 │
│              └────────────────┘                                 │
│                                                                 │
│  READ PATH (during future agent execution):                     │
│                                                                 │
│  New Task: "Debug OOM in payment service"                       │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────┐                                            │
│  │ Search traces by │                                           │
│  │ task similarity  │                                           │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────┐                │
│  │ Top 3 similar past traces:                  │                │
│  │ 1. "Debug OOM in checkout service" (0.92)   │                │
│  │ 2. "Fix memory leak in API gateway" (0.85)  │                │
│  │ 3. "Investigate high heap usage" (0.78)     │                │
│  └────────┬────────────────────────────────────┘                │
│           │                                                     │
│           ▼                                                     │
│  Inject into agent's context as                                 │
│  "relevant past experiences"                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Trace Format

A good trace captures *what happened* and *what was learned*, not raw logs:

```json
{
  "trace_id": "trace_20250519_001",
  "task": "Debug OOM errors in payment-service pod",
  "task_embedding": [0.12, -0.34, ...],
  "started_at": "2025-05-19T10:00:00Z",
  "completed_at": "2025-05-19T10:15:00Z",
  "duration_seconds": 900,
  "outcome": "resolved",
  "user_id": "user_123",
  "agent_id": "devops-agent",
  
  "steps": [
    {
      "step": 1,
      "action": "kubectl top pods -n payments",
      "observation": "payment-service-7d8f consuming 3.8GB/4GB memory",
      "reasoning": "Check current memory usage to confirm OOM"
    },
    {
      "step": 2, 
      "action": "kubectl logs payment-service-7d8f --tail=500",
      "observation": "Repeated pattern: 'Failed to allocate 256MB for request buffer'",
      "reasoning": "Look for memory allocation patterns in logs"
    },
    {
      "step": 3,
      "action": "Check heap dump analysis",
      "observation": "Large number of unclosed HTTP connections holding request buffers",
      "reasoning": "Request buffers suggest connection pool or handler leak"
    }
  ],
  
  "root_cause": "HTTP handler not closing response bodies, causing connection pool exhaustion and memory leak",
  "resolution": "Added defer resp.Body.Close() to all HTTP client calls in payment-service",
  "learnings": [
    "In Go services, always check for unclosed response bodies when diagnosing OOM",
    "Connection pool metrics (active_connections) are the first thing to check",
    "The payment-service uses a custom HTTP client without timeout -- this should be fixed"
  ],
  
  "tags": ["oom", "memory-leak", "go", "http-client", "payment-service"],
  "metadata": {
    "cluster": "prod-green-eks",
    "namespace": "payments",
    "service": "payment-service",
    "severity": "P1"
  }
}
```

## Production Implementation

### Trace Storage and Retrieval

```python
import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import asyncpg
from openai import OpenAI

client = OpenAI()


@dataclass
class TraceStep:
    step: int
    action: str
    observation: str
    reasoning: str
    tool_used: Optional[str] = None
    duration_ms: Optional[int] = None


@dataclass 
class ExecutionTrace:
    trace_id: str
    task: str
    user_id: str
    agent_id: str
    started_at: str
    outcome: str  # "resolved", "failed", "escalated", "partial"
    steps: List[TraceStep] = field(default_factory=list)
    root_cause: Optional[str] = None
    resolution: Optional[str] = None
    learnings: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    completed_at: Optional[str] = None
    duration_seconds: Optional[int] = None


class TraceStore:
    """Production trace storage with PostgreSQL + pgvector."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        self.embedder = client

    async def initialize(self):
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_traces (
                    trace_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    task_embedding vector(1536),
                    user_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    started_at TIMESTAMPTZ NOT NULL,
                    completed_at TIMESTAMPTZ,
                    duration_seconds INT,
                    steps JSONB NOT NULL DEFAULT '[]',
                    root_cause TEXT,
                    resolution TEXT,
                    learnings JSONB NOT NULL DEFAULT '[]',
                    tags TEXT[] DEFAULT '{}',
                    metadata JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_traces_embedding 
                ON execution_traces USING ivfflat (task_embedding vector_cosine_ops)
                WITH (lists = 50);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_traces_user 
                ON execution_traces (user_id, outcome);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_traces_tags 
                ON execution_traces USING gin (tags);
            """)

    def _embed(self, text: str) -> List[float]:
        response = self.embedder.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    async def store_trace(self, trace: ExecutionTrace):
        """Store a completed execution trace."""
        # Create rich embedding from task + learnings + root cause
        embed_text = f"{trace.task}"
        if trace.root_cause:
            embed_text += f" Root cause: {trace.root_cause}"
        if trace.learnings:
            embed_text += f" Learnings: {'. '.join(trace.learnings)}"

        embedding = self._embed(embed_text)

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO execution_traces 
                    (trace_id, task, task_embedding, user_id, agent_id, outcome,
                     started_at, completed_at, duration_seconds, steps,
                     root_cause, resolution, learnings, tags, metadata)
                VALUES ($1, $2, $3::vector, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                ON CONFLICT (trace_id) DO UPDATE SET
                    outcome = EXCLUDED.outcome,
                    completed_at = EXCLUDED.completed_at,
                    steps = EXCLUDED.steps,
                    root_cause = EXCLUDED.root_cause,
                    resolution = EXCLUDED.resolution,
                    learnings = EXCLUDED.learnings
            """,
                trace.trace_id, trace.task, str(embedding),
                trace.user_id, trace.agent_id, trace.outcome,
                trace.started_at, trace.completed_at, trace.duration_seconds,
                json.dumps([asdict(s) for s in trace.steps]),
                trace.root_cause, trace.resolution,
                json.dumps(trace.learnings), trace.tags,
                json.dumps(trace.metadata),
            )

    async def find_similar_traces(
        self,
        task_description: str,
        top_k: int = 3,
        outcome_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Find similar past traces by task similarity."""
        embedding = self._embed(task_description)

        filters = ["1=1"]
        params = [str(embedding), top_k]
        param_idx = 3

        if outcome_filter:
            filters.append(f"outcome = ${param_idx}")
            params.append(outcome_filter)
            param_idx += 1

        if tag_filter:
            filters.append(f"tags && ${param_idx}::text[]")
            params.append(tag_filter)
            param_idx += 1

        filter_clause = " AND ".join(filters)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT trace_id, task, outcome, root_cause, resolution,
                       learnings, steps, tags, duration_seconds,
                       1 - (task_embedding <=> $1::vector) as similarity
                FROM execution_traces
                WHERE {filter_clause}
                ORDER BY task_embedding <=> $1::vector
                LIMIT $2
            """, *params)

            return [
                {
                    "trace_id": row["trace_id"],
                    "task": row["task"],
                    "outcome": row["outcome"],
                    "root_cause": row["root_cause"],
                    "resolution": row["resolution"],
                    "learnings": json.loads(row["learnings"]),
                    "steps_count": len(json.loads(row["steps"])),
                    "tags": row["tags"],
                    "duration_seconds": row["duration_seconds"],
                    "similarity": float(row["similarity"]),
                }
                for row in rows
            ]

    def format_traces_for_context(self, traces: List[Dict]) -> str:
        """Format retrieved traces for injection into agent context."""
        if not traces:
            return ""

        parts = ["[Relevant past experiences]"]
        for i, trace in enumerate(traces, 1):
            parts.append(f"\n--- Experience {i} (similarity: {trace['similarity']:.0%}) ---")
            parts.append(f"Task: {trace['task']}")
            parts.append(f"Outcome: {trace['outcome']}")
            if trace.get("root_cause"):
                parts.append(f"Root cause: {trace['root_cause']}")
            if trace.get("resolution"):
                parts.append(f"Resolution: {trace['resolution']}")
            if trace.get("learnings"):
                parts.append("Learnings:")
                for learning in trace["learnings"]:
                    parts.append(f"  - {learning}")

        return "\n".join(parts)
```

### Trace Collector (Wraps Agent Execution)

```python
import uuid
from contextlib import asynccontextmanager

class TraceCollector:
    """Wraps agent execution to automatically collect traces."""

    def __init__(self, store: TraceStore, agent_id: str):
        self.store = store
        self.agent_id = agent_id

    @asynccontextmanager
    async def trace_run(self, task: str, user_id: str):
        """Context manager that automatically records execution traces."""
        trace = ExecutionTrace(
            trace_id=f"trace_{uuid.uuid4().hex[:12]}",
            task=task,
            user_id=user_id,
            agent_id=self.agent_id,
            started_at=datetime.utcnow().isoformat() + "Z",
            outcome="running",
        )
        start_time = time.time()

        try:
            yield trace  # Agent code runs here, appending steps to trace

            trace.outcome = "resolved"
        except Exception as e:
            trace.outcome = "failed"
            trace.learnings.append(f"Failed with error: {str(e)}")
            raise
        finally:
            trace.completed_at = datetime.utcnow().isoformat() + "Z"
            trace.duration_seconds = int(time.time() - start_time)
            await self.store.store_trace(trace)

    def add_step(self, trace: ExecutionTrace, action: str, observation: str, reasoning: str):
        """Add a step to the current trace."""
        trace.steps.append(TraceStep(
            step=len(trace.steps) + 1,
            action=action,
            observation=observation,
            reasoning=reasoning,
        ))


# ─── Usage ───

collector = TraceCollector(store=trace_store, agent_id="devops-agent")

async def debug_oom(service_name: str, user_id: str):
    async with collector.trace_run(
        task=f"Debug OOM in {service_name}",
        user_id=user_id,
    ) as trace:
        # Step 1
        result = await kubectl_top(service_name)
        collector.add_step(trace, 
            action=f"kubectl top pods -n {service_name}",
            observation=result,
            reasoning="Check current memory usage",
        )

        # Step 2
        logs = await kubectl_logs(service_name, tail=500)
        collector.add_step(trace,
            action=f"kubectl logs {service_name} --tail=500",
            observation=summarize(logs),
            reasoning="Look for memory allocation patterns",
        )

        # ... more steps ...

        trace.root_cause = "Unclosed HTTP response bodies"
        trace.resolution = "Added defer resp.Body.Close()"
        trace.learnings = [
            "Always check for unclosed response bodies in Go HTTP clients",
            "Connection pool metrics are the fastest diagnostic",
        ]
        trace.tags = ["oom", "go", "http-client", service_name]
```

### Feeding Traces into Future Context

```python
async def agent_with_episodic_memory(
    task: str,
    user_id: str,
    trace_store: TraceStore,
):
    """Agent that recalls relevant past experiences before acting."""

    # 1. Search for similar past experiences
    similar_traces = await trace_store.find_similar_traces(
        task_description=task,
        top_k=3,
        outcome_filter="resolved",  # Only learn from successes
    )

    # 2. Format traces for context injection
    experience_context = trace_store.format_traces_for_context(similar_traces)

    # 3. Build prompt with episodic memory
    system_prompt = f"""You are a DevOps debugging assistant.

{experience_context}

Use the above past experiences to guide your approach. If a past experience 
is highly relevant (similarity > 80%), prioritize the approach that worked before.
If no past experience is relevant, proceed with standard debugging methodology.

Always explain your reasoning, especially when drawing from past experience."""

    # 4. Run agent with experience-informed context
    response = await llm.ainvoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ])

    return response
```

## Decision Tree

```
Should I implement execution traces?
  │
  ├─ Does my agent perform repeatable tasks?
  │   ├─ YES → Traces will improve performance over time
  │   └─ NO → Each task is unique, limited value
  │
  ├─ Does the agent make multi-step decisions?
  │   ├─ YES → Traces capture decision chains that inform future runs
  │   └─ NO → Single-step agents have limited trace value
  │
  ├─ How often do similar tasks recur?
  │   ├─ Frequently → High value (debugging, support, DevOps)
  │   ├─ Occasionally → Medium value
  │   └─ Rarely → Low value, probably not worth the overhead
  │
  └─ What's your storage budget?
      ├─ Each trace: ~1-5KB
      ├─ 1000 traces/day → ~5MB/day → ~150MB/month
      └─ Manageable for any production database
```

## When NOT to Use

- **Privacy-sensitive tasks** -- traces may contain PII, credentials, or sensitive business data. If you cannot sanitize traces reliably, do not store them.
- **Unique, non-repeating tasks** -- if every task is fundamentally different, past traces have no relevance to future tasks.
- **Simple query-response agents** -- if the agent just answers questions from a knowledge base, there is no multi-step execution to trace.
- **When latency is critical** -- retrieving and processing past traces adds 100-300ms to each request. For sub-50ms requirements, skip trace retrieval.

## Tradeoffs

| Aspect | With Traces | Without Traces |
|---|---|---|
| First-run quality | Same as without | Baseline |
| Repeat-task quality | Improves over time (learns from past) | Same every time |
| Storage cost | ~5KB/trace, grows linearly | None |
| Retrieval latency | +100-300ms per request | None |
| Implementation effort | Medium (trace collector + store + retrieval) | None |
| Debugging | Excellent (full execution history) | Limited |

## Real-World Examples

1. **Incident response agent** -- Stores traces of every incident resolution. When a new P1 fires with similar symptoms, the agent recalls "last time this alert pattern appeared, the root cause was a misconfigured load balancer rule, and the fix was to update the health check path." Resolution time drops from 45 minutes to 10 minutes.

2. **Code review agent** -- Stores traces of past reviews including what issues were found and how they were fixed. When reviewing new code with similar patterns, the agent flags issues proactively: "In a similar PR last month, this pattern caused a race condition."

3. **Customer support agent** -- Stores traces of complex support cases. When a new case matches a past one, the agent suggests the resolution that worked before, skipping the diagnostic steps that were already proven effective.

## Failure Modes

| Failure | Symptom | Fix |
|---|---|---|
| Trace pollution | Agent recalls irrelevant past experiences | Increase similarity threshold (> 0.8), filter by tags |
| Stale traces | Past resolutions no longer work (system changed) | TTL on traces, weight recent traces higher |
| Over-reliance on past | Agent blindly follows past trace instead of reasoning | Instruct: "Use past experience as a guide, not a script" |
| Storage bloat | Millions of traces consuming storage | Retention policies, archive old traces, compact similar ones |
| PII in traces | Sensitive data stored in trace observations | Sanitize observations before storage, PII detection |

## Source(s) and Further Reading

- [State of AI Agent Memory 2026](https://mem0.ai) -- Mem0 research report
- [LangGraph Persistence Documentation](https://langchain-ai.github.io/langgraph/concepts/persistence/) -- LangChain
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) -- Shinn et al., 2023
- [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291) -- Wang et al., 2023
