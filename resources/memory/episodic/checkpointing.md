# Episodic Memory: Checkpointing

> Checkpointing is a save-game file for agent workflows -- it lets you resume from exactly where you left off, but it is not memory by itself.

## What It Is

Checkpointing is the mechanism of persisting the complete state of an agent's execution at each step so that the agent can resume from any point after interruption. In the LangGraph ecosystem, this is implemented through **checkpointers** -- storage backends that automatically save graph state after every node execution.

Checkpointing serves two purposes:
1. **Fault tolerance** -- resume a workflow after a crash, timeout, or deployment
2. **Human-in-the-loop** -- pause execution, wait for human input, then resume

### Checkpointing vs Memory

This distinction is critical and commonly confused:

| Aspect | Checkpointing | Memory |
|---|---|---|
| **What it stores** | Complete graph state at each step | Extracted facts, preferences, knowledge |
| **Scope** | Single execution thread | Cross-session, cross-user |
| **Retrieval** | Exact state by thread_id | Semantic search by relevance |
| **Purpose** | Resume interrupted execution | Personalize future interactions |
| **Analogy** | Save game file | Life experience |
| **Size** | Full state (~KBs-MBs per checkpoint) | Compact facts (~bytes-KBs per memory) |
| **Retention** | Usually temporary (days-weeks) | Usually permanent |

**Checkpointing is a building block for episodic memory.** You can build episodic memory on top of checkpoints by extracting learnings from past execution traces. But raw checkpoints alone are not memory -- they are state snapshots.

## How It Works

### LangGraph Checkpoint Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    LANGGRAPH EXECUTION                       │
│                                                              │
│  Thread: "thread_abc123"                                     │
│                                                              │
│  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐   ┌──────┐   │
│  │Node 1│───▶│Save  │───▶│Node 2│───▶│Save  │──▶│Node 3│   │
│  │      │    │CP #1 │    │      │    │CP #2 │   │      │   │
│  └──────┘    └──┬───┘    └──────┘    └──┬───┘   └──────┘   │
│                 │                       │                    │
│                 ▼                       ▼                    │
│           ┌──────────┐           ┌──────────┐               │
│           │Checkpoint│           │Checkpoint│               │
│           │Store     │           │Store     │               │
│           │          │           │          │               │
│           │{thread:  │           │{thread:  │               │
│           │ "abc123",│           │ "abc123",│               │
│           │ step: 1, │           │ step: 2, │               │
│           │ state:   │           │ state:   │               │
│           │ {...}    │           │ {...}    │               │
│           │}         │           │}         │               │
│           └──────────┘           └──────────┘               │
│                                                              │
│  RESUME (after crash at Node 3):                             │
│  Load CP #2 → Re-execute Node 3 → Continue                  │
└──────────────────────────────────────────────────────────────┘
```

### Thread-Scoped vs Cross-Session Checkpoints

```
THREAD-SCOPED (default):
  Thread "abc123" → [CP#1, CP#2, CP#3, ...]
  Thread "def456" → [CP#1, CP#2, ...]
  
  Each thread is an isolated execution.
  Cannot access checkpoints from other threads.
  Use case: Resuming an interrupted workflow.

CROSS-SESSION (with namespace):
  User "user_123":
    Session 1 (Thread "abc") → [CP#1, CP#2]  
    Session 2 (Thread "def") → [CP#1, CP#2]
    Session 3 (Thread "ghi") → [CP#1]
  
  Checkpoints scoped to user, queryable across sessions.
  Use case: Reviewing past execution history.
  
  Implementation: Store user_id in checkpoint metadata,
  query by metadata to find past sessions.
```

## Production Implementation

### LangGraph Checkpointers: Three Options

| Checkpointer | Best For | Persistence | Latency | Setup |
|---|---|---|---|---|
| **MemorySaver** | Development, testing | In-memory (lost on restart) | ~0ms | 1 line |
| **SqliteSaver** | Single-server, low traffic | File-based | ~1-5ms | 2 lines |
| **PostgresSaver** | Production | Durable, scalable | ~5-20ms | Connection pool |
| **RedisSaver** | High-throughput, ephemeral | In-memory with persistence | ~1-5ms | Redis cluster |

### PostgresSaver Production Setup

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from typing import TypedDict, List, Annotated
import operator

# ─── 1. State Definition ───

class WorkflowState(TypedDict):
    user_id: str
    task: str
    messages: Annotated[List[dict], operator.add]
    intermediate_results: dict
    final_result: str
    status: str  # "running", "waiting_human", "completed", "failed"

# ─── 2. Graph Nodes ───

async def analyze_task(state: WorkflowState) -> dict:
    """Step 1: Analyze the incoming task."""
    # ... LLM analysis ...
    return {
        "intermediate_results": {"analysis": "Task requires database migration"},
        "status": "running",
    }

async def plan_execution(state: WorkflowState) -> dict:
    """Step 2: Create execution plan."""
    return {
        "intermediate_results": {
            **state["intermediate_results"],
            "plan": ["backup", "migrate schema", "migrate data", "verify"],
        },
        "status": "waiting_human",  # Pause for human approval
    }

async def execute_plan(state: WorkflowState) -> dict:
    """Step 3: Execute the approved plan."""
    return {
        "final_result": "Migration completed successfully",
        "status": "completed",
    }

def should_wait_for_human(state: WorkflowState) -> str:
    """Route based on whether human approval is needed."""
    if state["status"] == "waiting_human":
        return "wait"
    return "continue"

# ─── 3. Build Graph ───

graph = StateGraph(WorkflowState)
graph.add_node("analyze", analyze_task)
graph.add_node("plan", plan_execution)
graph.add_node("execute", execute_plan)

graph.add_edge(START, "analyze")
graph.add_edge("analyze", "plan")
graph.add_conditional_edges(
    "plan",
    should_wait_for_human,
    {"wait": END, "continue": "execute"},  # END = pause, save checkpoint
)
graph.add_edge("execute", END)

# ─── 4. Production Checkpointing with PostgreSQL ───

DB_URI = "postgresql://user:pass@localhost:5432/agent_checkpoints"

async def main():
    async with AsyncConnectionPool(
        conninfo=DB_URI,
        min_size=5,
        max_size=20,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        
        # Create tables (run once)
        await checkpointer.setup()
        
        # Compile graph with checkpointer
        agent = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["execute"],  # Always pause before execution
        )
        
        # ─── Start a new workflow ───
        thread_id = "migration_workflow_001"
        config = {"configurable": {"thread_id": thread_id}}
        
        result = await agent.ainvoke(
            {
                "user_id": "user_123",
                "task": "Migrate payments DB from MySQL to PostgreSQL",
                "messages": [],
                "intermediate_results": {},
                "final_result": "",
                "status": "running",
            },
            config=config,
        )
        
        print(f"Workflow paused at status: {result['status']}")
        print(f"Plan: {result['intermediate_results']['plan']}")
        
        # ─── Resume after human approval ───
        # (Could be hours or days later, even after server restart)
        
        # Update state with human input
        await agent.aupdate_state(
            config,
            {"status": "running", "messages": [{"role": "human", "content": "Approved"}]},
        )
        
        # Resume execution
        final = await agent.ainvoke(None, config=config)
        print(f"Final result: {final['final_result']}")


asyncio.run(main())
```

### Resume from Failure Pattern

```python
async def run_with_retry(
    agent,
    initial_state: dict,
    thread_id: str,
    max_retries: int = 3,
):
    """Run agent workflow with automatic resume on failure."""
    config = {"configurable": {"thread_id": thread_id}}
    
    for attempt in range(max_retries):
        try:
            # Check if there's an existing checkpoint to resume from
            state = await agent.aget_state(config)
            
            if state.values:
                # Resume from last checkpoint
                print(f"Resuming from checkpoint (attempt {attempt + 1})")
                result = await agent.ainvoke(None, config=config)
            else:
                # Start fresh
                print(f"Starting new workflow (attempt {attempt + 1})")
                result = await agent.ainvoke(initial_state, config=config)
            
            if result.get("status") == "completed":
                return result
            elif result.get("status") == "waiting_human":
                return result  # Paused, not failed
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                # Mark as failed in checkpoint
                await agent.aupdate_state(
                    config,
                    {"status": "failed", "messages": [{"role": "system", "content": f"Failed: {e}"}]},
                )
                raise
            
            # Wait before retry (the checkpoint persists, so we can resume)
            await asyncio.sleep(2 ** attempt)
    
    raise RuntimeError("Max retries exceeded")
```

### Checkpoint Cleanup (Production)

```python
async def cleanup_old_checkpoints(
    pool: AsyncConnectionPool,
    retention_days: int = 30,
):
    """Delete checkpoints older than retention period.
    
    In production, run this as a daily cron job.
    """
    async with pool.connection() as conn:
        # LangGraph checkpoint tables: checkpoints, checkpoint_writes, checkpoint_blobs
        deleted = await conn.execute("""
            DELETE FROM checkpoints 
            WHERE created_at < NOW() - INTERVAL '%s days'
            AND thread_id NOT IN (
                -- Keep checkpoints for active/waiting threads
                SELECT DISTINCT thread_id 
                FROM checkpoints 
                WHERE metadata->>'status' IN ('running', 'waiting_human')
            )
        """, (retention_days,))
        
        print(f"Cleaned up {deleted.rowcount} old checkpoints")
        
        # Also clean up orphaned blobs and writes
        await conn.execute("""
            DELETE FROM checkpoint_blobs 
            WHERE thread_id NOT IN (SELECT DISTINCT thread_id FROM checkpoints)
        """)
        await conn.execute("""
            DELETE FROM checkpoint_writes 
            WHERE thread_id NOT IN (SELECT DISTINCT thread_id FROM checkpoints)
        """)
```

### Monitoring Checkpoints

```python
async def checkpoint_health(pool: AsyncConnectionPool) -> dict:
    """Get checkpoint system health metrics."""
    async with pool.connection() as conn:
        stats = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_checkpoints,
                COUNT(DISTINCT thread_id) as total_threads,
                pg_size_pretty(pg_total_relation_size('checkpoints')) as storage_size,
                MAX(created_at) as latest_checkpoint,
                COUNT(*) FILTER (
                    WHERE metadata->>'status' = 'waiting_human'
                ) as threads_waiting_human,
                COUNT(*) FILTER (
                    WHERE metadata->>'status' = 'failed'
                ) as threads_failed
            FROM checkpoints
        """)
        
        return {
            "total_checkpoints": stats["total_checkpoints"],
            "total_threads": stats["total_threads"],
            "storage_size": stats["storage_size"],
            "latest_checkpoint": stats["latest_checkpoint"],
            "threads_waiting_human": stats["threads_waiting_human"],
            "threads_failed": stats["threads_failed"],
        }
```

## Decision Tree

```
Do I need checkpointing?
  │
  ├─ Does my workflow have multiple steps?
  │   ├─ YES → Likely need checkpointing
  │   └─ NO → Single LLM call, no checkpointing needed
  │
  ├─ Can my workflow fail mid-execution?
  │   ├─ YES → Checkpointing enables resume-from-failure
  │   └─ NO → Nice-to-have, not required
  │
  ├─ Do I need human-in-the-loop?
  │   ├─ YES → Checkpointing is essential (pause/resume)
  │   └─ NO → Optional
  │
  ├─ Which checkpointer?
  │   ├─ Development → MemorySaver (in-memory)
  │   ├─ Single server, low traffic → SqliteSaver
  │   ├─ Production, multi-server → PostgresSaver
  │   └─ High-throughput, short-lived → RedisSaver
  │
  └─ Do I also need memory (not just checkpointing)?
      ├─ YES → Checkpointing + long-term memory (Mem0/Zep/LangMem)
      └─ NO → Checkpointing alone is fine
```

## When NOT to Use

- **Simple single-turn agents** -- if the agent takes one input and produces one output, there is no state to checkpoint.
- **Stateless API calls** -- if you are wrapping an LLM call in a REST API, checkpointing adds storage cost with no benefit.
- **When you need semantic recall** -- checkpoints store exact state, not queryable knowledge. If you need "what did we discuss last week?", you need memory, not checkpoints.
- **High-frequency, low-value workflows** -- if you are running 10K short workflows per minute, checkpoint storage costs may exceed the value of resumability.

## Tradeoffs

| Checkpointer | Durability | Latency | Multi-Server | Cost | Ops Burden |
|---|---|---|---|---|---|
| MemorySaver | None (lost on restart) | ~0ms | No | $0 | None |
| SqliteSaver | Good (file-based) | ~1-5ms | No | $0 | Very low |
| PostgresSaver | Excellent | ~5-20ms | Yes | Postgres costs | Medium |
| RedisSaver | Good (with AOF/RDB) | ~1-5ms | Yes | Redis costs | Medium |

## Real-World Examples

1. **Multi-step data pipeline agent** -- Ingests CSV, validates schema, transforms data, loads into warehouse. Each step checkpointed. When the transform step fails on row 50K of 100K, resume from row 50K instead of restarting. PostgresSaver with 7-day retention.

2. **Contract review agent** -- Analyzes legal contract, extracts clauses, flags risks, waits for lawyer approval, generates summary. Human-in-the-loop at the approval step -- lawyer may take 2 days. Checkpoint persists across server deployments. PostgresSaver with 90-day retention.

3. **Deployment automation agent** -- Plans deployment, runs pre-checks, deploys to staging, waits for QA approval, deploys to production. Each gate is a checkpoint. If production deploy fails, resume from the staging checkpoint and retry. RedisSaver for fast access during active deployments, PostgresSaver for audit trail.

## Failure Modes

| Failure | Symptom | Fix |
|---|---|---|
| Checkpoint storage full | Writes fail, workflows cannot be checkpointed | Retention policy, regular cleanup cron |
| Stale thread resumed | Agent resumes workflow from weeks ago, context is outdated | TTL on threads, notify user before resuming old threads |
| Checkpoint corruption | State cannot be deserialized after code change | Version your state schema, migration scripts |
| Checkpoint too large | Slow writes, high storage costs | Minimize state size, store large data externally (S3) |
| Concurrent writes | Two processes resume the same thread | Use database locks or single-writer per thread |

## Source(s) and Further Reading

- [LangGraph Persistence Documentation](https://langchain-ai.github.io/langgraph/concepts/persistence/) -- Official docs
- [LangGraph Checkpointing How-To](https://langchain-ai.github.io/langgraph/how-tos/persistence/) -- LangChain
- [PostgresSaver Reference](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.postgres) -- LangGraph
- [State of AI Agent Memory 2026](https://mem0.ai) -- Mem0 research report
