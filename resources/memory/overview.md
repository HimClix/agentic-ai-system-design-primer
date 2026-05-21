# Memory Systems for Agentic AI

> Memory is what separates a stateless chatbot from a useful agent -- the ability to accumulate knowledge, learn from past interactions, and adapt behavior across sessions.

## What It Is

Memory in agentic systems is the set of mechanisms that let an agent retain, retrieve, and act on information beyond the current request. Without memory, every conversation starts from zero. With memory, an agent knows that "the user prefers dark mode," that "last week's deployment failed because of a schema migration," and that "step 3 always needs a confirmation before proceeding."

There are four distinct memory types, each solving a different problem:

| Memory Type | Analogy | Scope | Persistence | Example |
|---|---|---|---|---|
| **Short-Term** | Working memory / scratchpad | Single request or session | Ephemeral (context window) | Current conversation turns, retrieved docs |
| **Long-Term** | Semantic memory / facts | Cross-session, cross-user | Durable (vector store, graph DB) | "User prefers concise answers," product facts |
| **Episodic** | Autobiographical memory | Past events and experiences | Durable (checkpoints, traces) | "Last time this error appeared, the fix was X" |
| **Procedural** | Muscle memory / how-to | Self-improving behavior | Durable (prompt store) | Rewritten system prompts, learned workflows |

## How They Work Together

```
┌─────────────────────────────────────────────────────────────┐
│                     AGENT REQUEST                           │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              SHORT-TERM MEMORY                       │   │
│  │  ┌─────────┐  ┌──────────┐  ┌───────────────────┐   │   │
│  │  │ System   │  │ Current  │  │  Retrieved RAG    │   │   │
│  │  │ Prompt   │  │ Convo    │  │  Context          │   │   │
│  │  └─────────┘  └──────────┘  └───────────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
│         ▲                ▲                  ▲                │
│         │                │                  │                │
│  ┌──────┴──┐    ┌───────┴───────┐   ┌──────┴──────┐        │
│  │PROCEDURAL│   │   EPISODIC    │   │  LONG-TERM  │        │
│  │  Memory  │   │   Memory      │   │   Memory    │        │
│  │          │   │               │   │             │        │
│  │ Self-    │   │ Past run      │   │ User prefs  │        │
│  │ improved │   │ checkpoints,  │   │ facts,      │        │
│  │ prompts  │   │ traces        │   │ knowledge   │        │
│  └──────────┘   └───────────────┘   └─────────────┘        │
│                                                             │
│  ────────────────── HOT PATH ──────────────────────         │
│    Redis / in-memory     < 10ms     session data            │
│                                                             │
│  ────────────────── COLD PATH ─────────────────────         │
│    Postgres + Vector DB  < 200ms    cross-session           │
└─────────────────────────────────────────────────────────────┘
```

### The Interaction Pattern

1. **Request arrives** -- the agent loads short-term context (conversation history, system prompt).
2. **Long-term retrieval** -- the agent queries vector stores or knowledge graphs for relevant user facts, domain knowledge.
3. **Episodic recall** -- if the task resembles a past task, the agent retrieves relevant execution traces or checkpoints.
4. **Procedural injection** -- if the agent has self-improved its prompts, those refined instructions replace the default system prompt.
5. **Everything merges into the context window** -- the short-term memory is the final assembly point. All other memory types feed into it.
6. **After completion** -- the agent writes back: new facts to long-term memory, execution traces to episodic memory, prompt improvements to procedural memory.

## Decision Tree: Which Memory Type Do You Need?

```
START
  │
  ├─ Does the agent need to remember within a single conversation?
  │   └─ YES → SHORT-TERM MEMORY
  │       ├─ Is the conversation long (>20 turns)?
  │       │   └─ YES → Add COMPRESSION (summarization, sliding window)
  │       └─ Is RAG context needed?
  │           └─ YES → Add TOKEN BUDGETING
  │
  ├─ Does the agent need to remember across sessions?
  │   └─ YES → LONG-TERM MEMORY
  │       ├─ Mostly user preferences and facts?
  │       │   └─ YES → Vector store (Mem0, Zep)
  │       ├─ Need temporal relationships ("when did X change?")?
  │       │   └─ YES → Knowledge graph (Zep Graphiti, Mem0 Graph)
  │       └─ Both?
  │           └─ Hybrid store (Mem0 Pro, Zep Cloud)
  │
  ├─ Does the agent need to resume interrupted workflows?
  │   └─ YES → EPISODIC MEMORY (checkpointing)
  │       ├─ LangGraph? → PostgresSaver / RedisSaver
  │       └─ Custom? → Your own checkpoint table
  │
  ├─ Does the agent need to learn from past runs?
  │   └─ YES → EPISODIC MEMORY (execution traces)
  │       └─ Store structured traces, retrieve similar past runs
  │
  └─ Should the agent improve its own behavior over time?
      └─ YES → PROCEDURAL MEMORY
          ├─ Low risk? → LangMem procedural memory
          └─ High risk? → Human-in-the-loop prompt evolution
```

## The Dual-Layer Production Pattern

Every production memory system should have two layers:

### Hot Path (< 10ms)
- **Storage**: Redis, in-memory cache
- **Contains**: Current session data, recently accessed user facts, active checkpoints
- **TTL**: 1-24 hours
- **Use case**: Everything the agent needs for the current interaction

### Cold Path (< 200ms)
- **Storage**: PostgreSQL + pgvector, dedicated vector DB, Neo4j
- **Contains**: All historical user data, full execution traces, knowledge graph
- **TTL**: Indefinite (with retention policies)
- **Use case**: Cross-session recall, analytics, compliance

```python
class DualLayerMemory:
    def __init__(self, redis_client, pg_pool, vector_store):
        self.hot = redis_client      # Hot path: session cache
        self.cold_relational = pg_pool  # Cold path: structured data
        self.cold_vector = vector_store # Cold path: semantic search

    async def recall(self, user_id: str, query: str) -> dict:
        # 1. Check hot cache first
        cached = await self.hot.get(f"mem:{user_id}:latest")
        if cached:
            return json.loads(cached)

        # 2. Fall back to cold path
        facts = await self.cold_vector.search(query, filter={"user_id": user_id}, top_k=5)
        
        # 3. Warm the cache for next request
        result = {"facts": facts, "retrieved_at": time.time()}
        await self.hot.setex(f"mem:{user_id}:latest", 3600, json.dumps(result))
        return result

    async def remember(self, user_id: str, fact: str, metadata: dict):
        # Write-through: update both layers
        await self.cold_vector.upsert(fact, metadata={"user_id": user_id, **metadata})
        await self.hot.delete(f"mem:{user_id}:latest")  # Invalidate cache
```

## How Memory Differs from LangGraph Checkpointing

This is a common confusion. They are related but serve different purposes:

| Aspect | LangGraph Checkpointing | Agent Memory |
|---|---|---|
| **Purpose** | Resume interrupted graph execution | Remember information across interactions |
| **Scope** | Single graph run (thread) | Cross-session, cross-user |
| **What's stored** | Full graph state at each node | Extracted facts, preferences, knowledge |
| **Retrieval** | By thread_id, exact state restoration | By semantic similarity, user_id, filters |
| **Analogy** | Save game file | Life experience |
| **You need it when** | Long-running workflows, human-in-the-loop | Personalization, learning, adaptation |

**Checkpointing is a building block for episodic memory**, but it is not memory by itself. A checkpoint says "here is exactly where we were." Memory says "here is what I learned from that experience."

## Tradeoffs

| Approach | Latency | Cost | Complexity | Best For |
|---|---|---|---|---|
| Context window only | 0ms (no retrieval) | High (re-sending everything) | Low | Short conversations, prototypes |
| Compression + sliding window | 5-20ms | Medium | Medium | Long single-session conversations |
| Vector store long-term | 50-200ms | Low per query | Medium | User personalization at scale |
| Knowledge graph | 100-500ms | Higher | High | Temporal reasoning, relationship queries |
| Full dual-layer | 10-200ms | Medium | High | Production systems with SLAs |

## Real-World Examples

1. **Customer support agent** -- Short-term (current ticket), long-term (customer history, past tickets), episodic (similar resolved tickets). No procedural.
2. **Coding assistant** -- Short-term (current file context), long-term (codebase knowledge, user preferences), episodic (past debugging sessions), procedural (learned coding patterns).
3. **Sales outreach agent** -- Short-term (current conversation), long-term (CRM facts, relationship graph), episodic (past interactions with this prospect). Procedural (improved pitch templates).

## Failure Modes

| Failure | Symptom | Mitigation |
|---|---|---|
| Memory pollution | Agent acts on wrong user's data | Strict user_id isolation, tenant-scoped storage |
| Stale facts | Agent uses outdated information | TTL on cached facts, temporal validity windows |
| Context overflow | Token limit exceeded, truncation | Token budgeting, compression, prioritized retrieval |
| Privacy leakage | PII exposed across sessions | Encryption at rest, PII detection before storage |
| Memory hallucination | Agent "remembers" things that never happened | Always attribute memories to sources, confidence scoring |

## Source(s) and Further Reading

- [State of AI Agent Memory 2026](https://mem0.ai) -- Mem0 research report
- [Mem0 Paper](https://arxiv.org/abs/2504.19413) -- arXiv:2504.19413
- [8 Agent Memory Frameworks Compared](https://vectorize.io) -- vectorize.io
- [Mem0 vs Zep vs LangMem Comparison](https://dev.to) -- dev.to
- [LangGraph Persistence Documentation](https://langchain-ai.github.io/langgraph/concepts/persistence/) -- LangChain
