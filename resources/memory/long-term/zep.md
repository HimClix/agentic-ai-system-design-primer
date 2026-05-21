# Zep: Temporal Knowledge Graph Memory

> Zep's Graphiti engine gives you what Mem0 paywalls at $249/mo -- graph memory with temporal reasoning -- starting at $25/mo, and it scores 30% higher on LongMemEval.

## What It Is

Zep is an AI memory infrastructure that combines a temporal knowledge graph engine (Graphiti) with traditional vector and session memory. Its key differentiator is **temporal reasoning**: every fact in Zep's graph has `valid_at` and `invalid_at` timestamps, so the system knows not just what is true, but when it became true and when it stopped being true.

Zep Cloud scored **63.8% on LongMemEval** (vs Mem0's 49%) and **94.8% on Deep Memory Retrieval (DMR)**, making it the strongest commercially available memory framework for factual recall with temporal awareness.

### Key Differentiators

| Feature | Zep | Mem0 | Why It Matters |
|---|---|---|---|
| Graph memory pricing | $25/mo (all tiers) | $249/mo (Pro only) | 10x cheaper for relationship queries |
| LongMemEval | 63.8% | 49.0% | 30% better factual recall |
| DMR score | 94.8% | Not reported | Near-perfect deep retrieval |
| Temporal reasoning | First-class (`valid_at`/`invalid_at`) | Weak | "What was true on March 1?" |
| Knowledge graph engine | Graphiti (purpose-built) | Neo4j/FalkorDB (generic) | Optimized for agent memory |

## How It Works

### Graphiti: Temporal Knowledge Graph Engine

Graphiti is Zep's open-source temporal knowledge graph engine. It differs from traditional knowledge graphs in three ways:

1. **Bi-temporal edges** -- every relationship has `valid_at` (when the fact became true in the real world) and `invalid_at` (when it stopped being true). An edge without `invalid_at` is currently true.

2. **Episodic nodes** -- raw conversation episodes are stored as nodes, linked to the entities they mention. This preserves provenance -- you can always trace back to the original conversation.

3. **Community detection** -- Graphiti automatically clusters related entities and summarizes communities, enabling higher-level reasoning about groups of facts.

```
┌─────────────────────────────────────────────────────────────────┐
│                    GRAPHITI ARCHITECTURE                        │
│                                                                 │
│  Episode (conversation turn):                                   │
│  "I just moved to Bangalore from Mumbai last week"              │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  ENTITY EXTRACTION                                       │   │
│  │                                                          │   │
│  │  [User] ──lives_in──▶ [Bangalore]                        │   │
│  │            valid_at: 2025-05-12                           │   │
│  │            invalid_at: null (current)                     │   │
│  │            source_episode: ep_001                         │   │
│  │                                                          │   │
│  │  [User] ──lived_in──▶ [Mumbai]                           │   │
│  │            valid_at: unknown (past)                       │   │
│  │            invalid_at: 2025-05-12                         │   │
│  │            source_episode: ep_001                         │   │
│  │                                                          │   │
│  │  [Episode ep_001] ──mentions──▶ [User]                   │   │
│  │  [Episode ep_001] ──mentions──▶ [Bangalore]              │   │
│  │  [Episode ep_001] ──mentions──▶ [Mumbai]                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  TEMPORAL QUERY: "Where did the user live before May 12?"       │
│  → Traverse edges where invalid_at <= 2025-05-12                │
│  → Answer: "Mumbai"                                             │
│                                                                 │
│  CURRENT QUERY: "Where does the user live?"                     │
│  → Traverse edges where invalid_at IS NULL                      │
│  → Answer: "Bangalore"                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Zep Cloud Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     ZEP CLOUD                            │
│                                                          │
│  ┌────────────┐   ┌────────────────┐   ┌──────────────┐ │
│  │  Session    │   │   Knowledge    │   │    Fact       │ │
│  │  Memory     │   │   Graph        │   │    Memory     │ │
│  │             │   │  (Graphiti)    │   │              │ │
│  │ Conversation│   │  Entities +    │   │ Extracted    │ │
│  │ history per │   │  temporal      │   │ facts as     │ │
│  │ session     │   │  edges         │   │ embeddings   │ │
│  │             │   │                │   │              │ │
│  │ Auto-       │   │  Neo4j under   │   │ Hybrid       │ │
│  │ summarized  │   │  the hood      │   │ search       │ │
│  └──────┬──────┘   └───────┬────────┘   └──────┬───────┘ │
│         │                  │                    │         │
│         └──────────┬───────┴────────────────────┘         │
│                    │                                      │
│           ┌────────▼────────┐                             │
│           │  UNIFIED API    │                             │
│           │                 │                             │
│           │ memory.add()    │                             │
│           │ memory.search() │                             │
│           │ memory.get()    │                             │
│           └────────┬────────┘                             │
│                    │                                      │
│  ──────────────────┼──────── SDK ───────────────────────  │
└────────────────────┼──────────────────────────────────────┘
                     │
              Your Agent Code
```

## Production Implementation

### Zep Cloud Setup

```python
from zep_cloud.client import Zep
from zep_cloud.types import Message
import os

# Initialize client
zep = Zep(api_key=os.environ["ZEP_API_KEY"])

# Create a user
user_id = "user_123"
try:
    zep.user.add(user_id=user_id, metadata={"name": "Alex", "role": "backend_engineer"})
except Exception:
    pass  # User already exists

# Create a session for this conversation
session_id = "session_20250519_001"
zep.memory.add_session(
    session_id=session_id,
    user_id=user_id,
    metadata={"channel": "web", "topic": "database_migration"},
)
```

### Adding Memories (Conversation-Based)

```python
# Add conversation turns -- Zep auto-extracts facts and builds the graph
messages = [
    Message(
        role_type="user",
        role="Alex",
        content="I'm migrating our payment service from MySQL to PostgreSQL. "
                "We process about 10K transactions per second.",
    ),
    Message(
        role_type="assistant",
        role="Agent",
        content="PostgreSQL is a great choice for high-throughput payment systems. "
                "With 10K TPS, you'll want to look at connection pooling with PgBouncer.",
    ),
    Message(
        role_type="user",
        role="Alex",
        content="We already use PgBouncer. The main challenge is the schema migration -- "
                "we have 200+ tables and need zero-downtime migration.",
    ),
]

# Zep processes these asynchronously:
# 1. Stores raw messages in session memory
# 2. Extracts facts: "payment service migrating MySQL→PostgreSQL", "10K TPS", "uses PgBouncer"
# 3. Creates/updates knowledge graph entities and edges
# 4. Embeds facts for semantic search
result = zep.memory.add(session_id=session_id, messages=messages)
```

### Searching Memory

```python
# Semantic search across all memory types
results = zep.memory.search_sessions(
    user_id=user_id,
    text="database migration status",
    search_scope="facts",  # or "messages" or "summary"
    limit=5,
)

for result in results:
    print(f"Fact: {result.fact}")
    print(f"Score: {result.score}")
    print(f"Created: {result.created_at}")
    print("---")


# Graph-based search (relationship traversal)
graph_results = zep.graph.search(
    user_id=user_id,
    query="What databases is the user working with?",
    limit=10,
)

for edge in graph_results.edges:
    print(f"{edge.source_node.name} --{edge.relation}--> {edge.target_node.name}")
    print(f"  Valid from: {edge.valid_at}")
    print(f"  Valid until: {edge.invalid_at or 'current'}")
```

### Temporal Queries

```python
# "What was true on a specific date?"
# Zep's graph edges have valid_at and invalid_at timestamps

from datetime import datetime

# Get user's graph at a specific point in time
# This is the killer feature -- temporal knowledge traversal
graph_state = zep.graph.search(
    user_id=user_id,
    query="user's database preferences",
    limit=10,
)

# Filter edges by temporal validity
as_of_date = datetime(2025, 3, 1)
current_facts = []
historical_facts = []

for edge in graph_state.edges:
    if edge.valid_at and edge.valid_at <= as_of_date:
        if edge.invalid_at is None or edge.invalid_at > as_of_date:
            current_facts.append(edge)
        else:
            historical_facts.append(edge)

print(f"Facts true as of {as_of_date.date()}:")
for e in current_facts:
    print(f"  {e.source_node.name} --{e.relation}--> {e.target_node.name}")

print(f"\nFacts no longer true by {as_of_date.date()}:")
for e in historical_facts:
    print(f"  {e.source_node.name} --{e.relation}--> {e.target_node.name} (until {e.invalid_at})")
```

### Integration with LangGraph

```python
from langgraph.graph import StateGraph, START, END
from zep_cloud.client import Zep
from typing import TypedDict, List, Annotated
import operator

zep = Zep(api_key=os.environ["ZEP_API_KEY"])

class AgentState(TypedDict):
    user_id: str
    session_id: str
    messages: Annotated[List[dict], operator.add]
    memory_context: str
    response: str

def recall_memory(state: AgentState) -> dict:
    """Recall relevant memories before responding."""
    last_message = state["messages"][-1]["content"]
    
    # Search facts
    results = zep.memory.search_sessions(
        user_id=state["user_id"],
        text=last_message,
        search_scope="facts",
        limit=5,
    )
    
    # Search graph
    graph_results = zep.graph.search(
        user_id=state["user_id"],
        query=last_message,
        limit=5,
    )
    
    context_parts = []
    if results:
        facts = "\n".join(f"- {r.fact}" for r in results)
        context_parts.append(f"Known facts:\n{facts}")
    
    if graph_results and graph_results.edges:
        relations = "\n".join(
            f"- {e.source_node.name} → {e.relation} → {e.target_node.name}"
            for e in graph_results.edges
        )
        context_parts.append(f"Relationships:\n{relations}")
    
    return {"memory_context": "\n\n".join(context_parts) or "No relevant memories."}

def generate_response(state: AgentState) -> dict:
    """Generate response with memory context."""
    # Use memory_context in your LLM call
    system = f"""You are a helpful assistant.

{state['memory_context']}

Use the above context to personalize your response."""
    
    # ... LLM call here ...
    return {"response": "..."}

def store_memory(state: AgentState) -> dict:
    """Store new information from this interaction."""
    from zep_cloud.types import Message
    messages = [
        Message(role_type="user", content=state["messages"][-1]["content"]),
        Message(role_type="assistant", content=state["response"]),
    ]
    zep.memory.add(session_id=state["session_id"], messages=messages)
    return {}

# Build graph
graph = StateGraph(AgentState)
graph.add_node("recall", recall_memory)
graph.add_node("generate", generate_response)
graph.add_node("store", store_memory)
graph.add_edge(START, "recall")
graph.add_edge("recall", "generate")
graph.add_edge("generate", "store")
graph.add_edge("store", END)
agent = graph.compile()
```

## Decision Tree

```
Should I use Zep?
  │
  ├─ Do you need temporal reasoning?
  │   ├─ YES → Zep is the strongest choice (valid_at/invalid_at)
  │   └─ NO → Zep still competitive, but Mem0 or LangMem may suffice
  │
  ├─ Do you need graph memory?
  │   ├─ YES, budget < $100/mo → Zep ($25/mo -- graph at all tiers)
  │   ├─ YES, budget $249+/mo → Zep or Mem0 Pro
  │   └─ NO → Vector-only solutions work
  │
  ├─ Is benchmark accuracy a priority?
  │   ├─ YES → Zep (63.8% LongMemEval, 94.8% DMR) > Mem0 (49%)
  │   │         But Hindsight (91.4%) is highest if available
  │   └─ NO → Any framework works
  │
  ├─ Self-hosted requirement?
  │   ├─ YES → Zep OSS + Neo4j (Graphiti is open source)
  │   └─ NO → Zep Cloud is easiest
  │
  └─ Already using LangGraph?
      ├─ YES → LangMem is more native, but Zep integrates well too
      └─ NO → Zep integrates with most frameworks
```

## When NOT to Use

- **When memory footprint matters** -- Zep's graph memory can consume **600K+ tokens** of memory state for complex user histories. If you are on a tight context budget, this is problematic. You must be selective about what you retrieve.
- **Neo4j operational burden** -- self-hosted Zep requires Neo4j, which is a non-trivial database to operate (Java-based, memory-hungry, requires tuning). If you lack ops capacity, use Zep Cloud.
- **Simple preference recall** -- if all you need is "user likes dark mode," a vector store is simpler and cheaper than a temporal knowledge graph.
- **When you need the highest possible accuracy** -- Hindsight scores 91.4% on LongMemEval vs Zep's 63.8%. If accuracy is paramount and you can use Hindsight, it beats Zep.
- **Real-time applications** -- graph queries can take 200-500ms. For sub-100ms requirements, use a cached vector store.

## Tradeoffs

| Aspect | Zep Cloud | Zep Self-Hosted | Mem0 (comparison) |
|---|---|---|---|
| Graph memory | Yes ($25/mo) | Yes (Neo4j required) | $249/mo |
| LongMemEval | 63.8% | 63.8% | 49.0% |
| DMR | 94.8% | 94.8% | Not reported |
| Temporal reasoning | Excellent | Excellent | Weak |
| Memory footprint | Large (600K+ tokens) | Large | Smaller |
| Operational burden | None (managed) | High (Neo4j + Zep) | Low (managed) |
| Pricing | $25/mo | Free (+ Neo4j costs) | $99-$249/mo |
| Community size | Growing | Growing | Larger (25K+ stars) |

## Real-World Examples

1. **Legal research agent** -- Zep tracks case law relationships with temporal validity. "What was the precedent for X before the 2024 ruling?" uses `valid_at`/`invalid_at` to show the precedent chain at a specific point in time. Graph traversal connects cases → judges → courts → jurisdictions.

2. **Healthcare coordination agent** -- Zep stores patient medication history with temporal bounds. "What medications was the patient on in January?" traverses edges where `valid_at <= Jan 31` and `invalid_at IS NULL OR invalid_at > Jan 1`. Critical for drug interaction checks.

3. **CRM assistant** -- Zep's graph connects contacts → companies → deals → activities with timestamps. "Show me all interactions with Acme Corp since Q3" is a natural graph query. Community detection groups related contacts automatically.

## Failure Modes

| Failure | Symptom | Fix |
|---|---|---|
| Memory footprint explosion | Context window overflow when retrieving graph state | Limit retrieval depth, use `limit` parameter, summarize graph context |
| Neo4j instability (self-hosted) | Slow queries, OOM errors | Tune Neo4j memory settings, add indexes, use Zep Cloud instead |
| Temporal edge conflicts | Two edges claim contradictory facts for the same time period | Conflict resolution rules, latest-write-wins or human review |
| Over-extraction | Graphiti creates entities for trivial mentions | Tune extraction prompts, set entity importance thresholds |
| Cold start | New users have empty graph, no personalization | Fallback to non-personalized behavior, explicit onboarding questions |

## Source(s) and Further Reading

- [Zep Documentation](https://docs.getzep.com) -- Official docs
- [Graphiti GitHub Repository](https://github.com/getzep/graphiti) -- Open-source temporal knowledge graph
- [State of AI Agent Memory 2026](https://mem0.ai) -- Mem0 research report (includes Zep benchmarks)
- [Mem0 vs Zep vs LangMem Comparison](https://dev.to) -- dev.to
- [8 Agent Memory Frameworks Compared](https://vectorize.io) -- vectorize.io
