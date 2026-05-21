# LangMem: LangGraph-Native Memory SDK

> LangMem is the only memory framework that lets agents rewrite their own system prompts -- procedural memory that no other framework offers -- and it is completely free (MIT).

## What It Is

LangMem is LangChain's native memory SDK, purpose-built for the LangGraph ecosystem. It distinguishes itself from Mem0 and Zep in two ways:

1. **Three memory types including procedural** -- episodic (what happened), semantic (what is true), and procedural (how to behave). Procedural memory means the agent can learn from feedback and rewrite its own instructions. No other framework offers this.

2. **LangGraph-native** -- LangMem is designed as LangGraph nodes and tools, not as a standalone service. Memory operations are part of the graph execution, not external API calls.

### Key Numbers

| Metric | Value | Context |
|---|---|---|
| LOCOMO score | 58.1% | Mid-tier accuracy. Above Mem0 on some tasks, below on others |
| p95 search latency | 59.82s | Critically slow. This is a real production concern |
| Price | Free (MIT) | Zero cost for the SDK. You pay for LLM calls and storage |
| Self-hosted | Yes (default) | No cloud service -- you own everything |
| Procedural memory | Yes (unique) | Only framework with self-improving prompts |

## How It Works

### Three Memory Types

```
┌─────────────────────────────────────────────────────────────────┐
│                    LANGMEM MEMORY TYPES                         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  SEMANTIC MEMORY (facts)                                │    │
│  │                                                         │    │
│  │  "User prefers PostgreSQL"                              │    │
│  │  "User works at a fintech company"                       │    │
│  │  "User's timezone is IST"                               │    │
│  │                                                         │    │
│  │  Storage: Vector store (your choice)                    │    │
│  │  Retrieval: Semantic similarity search                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  EPISODIC MEMORY (experiences)                          │    │
│  │                                                         │    │
│  │  "On May 15, user debugged a connection pool leak.      │    │
│  │   Root cause: PgBouncer max_client_conn too low.        │    │
│  │   Fix: Increased from 100 to 400."                      │    │
│  │                                                         │    │
│  │  Storage: Thread-scoped in LangGraph                    │    │
│  │  Retrieval: By similarity to current task               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  PROCEDURAL MEMORY (self-improving instructions)  ★     │    │
│  │                                                         │    │
│  │  Before: "When debugging database issues, check logs"   │    │
│  │  After:  "When debugging database issues at Acme Payments:   │    │
│  │           1. Check PgBouncer stats first                │    │
│  │           2. Then check slow query log                  │    │
│  │           3. Check connection pool metrics              │    │
│  │           This order resolves 80% of issues faster."    │    │
│  │                                                         │    │
│  │  Storage: Prompt/instruction store                      │    │
│  │  Trigger: User feedback (thumbs down, corrections)      │    │
│  │  UNIQUE TO LANGMEM -- no other framework has this       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LANGGRAPH AGENT                       │
│                                                         │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Graph Node: recall_memories                      │  │
│  │  → Search semantic + episodic memory              │  │
│  │  → Load procedural instructions                   │  │
│  │  → Inject into state                              │  │
│  └─────────────────────┬──────────────────────────────┘  │
│                        │                                 │
│  ┌─────────────────────▼──────────────────────────────┐  │
│  │  Graph Node: agent (LLM)                          │  │
│  │  → System prompt includes procedural instructions │  │
│  │  → Context includes recalled memories             │  │
│  │  → Responds to user                               │  │
│  └─────────────────────┬──────────────────────────────┘  │
│                        │                                 │
│  ┌─────────────────────▼──────────────────────────────┐  │
│  │  Graph Node: store_memories (background)          │  │
│  │  → Extract new semantic facts                     │  │
│  │  → Store episodic experience                      │  │
│  │  → If feedback received: update procedural memory │  │
│  └────────────────────────────────────────────────────┘  │
│                                                         │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Storage Layer (your infrastructure)              │  │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────────────┐ │  │
│  │  │ Vector  │  │ LangGraph│  │ Prompt/Instruction│ │  │
│  │  │ Store   │  │ Checkpts │  │ Store (any DB)    │ │  │
│  │  │         │  │          │  │                    │ │  │
│  │  │Semantic │  │ Episodic │  │ Procedural         │ │  │
│  │  └─────────┘  └──────────┘  └──────────────────┘ │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Production Implementation

### Basic LangMem Setup

```python
from langmem import create_memory_store_manager, create_memory_tools
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Annotated
import operator

llm = ChatOpenAI(model="gpt-4o")

# Create memory tools that the agent can call
memory_tools = create_memory_tools(
    # Namespace for organizing memories
    namespace=("memories", "{user_id}"),
    # Types of memory to enable
    memory_types=["semantic", "episodic"],
    # Embedding model for search
    embedding_model="text-embedding-3-small",
)

# Create the agent with memory tools
agent = create_react_agent(
    llm,
    tools=[*your_other_tools, *memory_tools],
    prompt="You are a helpful assistant. Use memory tools to remember and recall user information.",
)
```

### Semantic Memory (Facts and Preferences)

```python
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.store.memory import InMemoryStore
# For production, use a persistent store:
# from langgraph.store.postgres import PostgresStore

# Initialize store
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

# Create tools
manage_memory = create_manage_memory_tool(
    namespace=("user_facts", "{user_id}"),
)
search_memory = create_search_memory_tool(
    namespace=("user_facts", "{user_id}"),
)

# The agent uses these as regular tools:
# - manage_memory.invoke({"action": "create", "content": "User prefers PostgreSQL"})
# - search_memory.invoke({"query": "database preferences"})
```

### Procedural Memory (Self-Improving Instructions)

This is LangMem's unique feature. The agent rewrites its own system prompt based on feedback.

```python
from langmem import create_prompt_optimizer

# Define the initial system prompt
initial_prompt = """You are a database expert assistant.

When helping with database issues:
1. Ask about the database type
2. Check error messages
3. Suggest solutions

Be concise and direct."""

# Create the prompt optimizer
optimizer = create_prompt_optimizer(
    llm=ChatOpenAI(model="gpt-4o"),
    kind="prompt_memory",  # Stores prompt improvements
    namespace=("prompts", "db_assistant"),
)

# After receiving negative feedback:
async def handle_feedback(
    conversation: list,
    feedback: str,  # "You should have checked PgBouncer first"
    current_prompt: str,
):
    """Update system prompt based on user feedback."""
    # The optimizer analyzes the conversation and feedback,
    # then rewrites the system prompt
    improved_prompt = await optimizer.optimize(
        current_prompt=current_prompt,
        conversation=conversation,
        feedback=feedback,
    )
    
    # Store the improved prompt
    await optimizer.store.put(
        namespace=("prompts", "db_assistant"),
        key="system_prompt",
        value={"prompt": improved_prompt},
    )
    
    return improved_prompt

# Example:
# Before feedback: "When helping with database issues: 1. Ask about type..."
# After feedback:  "When helping with database issues at Acme Payments:
#                   1. Check PgBouncer stats and connection pool first
#                   2. Review slow query log  
#                   3. Ask about the specific database (PostgreSQL/MySQL)
#                   This order catches the most common issues faster."
```

### Full LangGraph Agent with All Three Memory Types

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_openai import ChatOpenAI
from langmem import (
    create_manage_memory_tool,
    create_search_memory_tool,
    create_prompt_optimizer,
)
from typing import TypedDict, List, Optional
import json

class AgentState(TypedDict):
    user_id: str
    messages: List[dict]
    memory_context: str
    system_prompt: str
    response: str
    feedback: Optional[str]

# Components
llm = ChatOpenAI(model="gpt-4o")
search_tool = create_search_memory_tool(namespace=("facts", "{user_id}"))
manage_tool = create_manage_memory_tool(namespace=("facts", "{user_id}"))
prompt_optimizer = create_prompt_optimizer(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    kind="prompt_memory",
    namespace=("prompts", "assistant"),
)

DEFAULT_PROMPT = "You are a helpful assistant. Be concise and direct."

async def load_procedural_memory(state: AgentState) -> dict:
    """Load the latest self-improved system prompt."""
    try:
        stored = await prompt_optimizer.store.get(
            namespace=("prompts", "assistant"),
            key="system_prompt",
        )
        prompt = stored.value["prompt"] if stored else DEFAULT_PROMPT
    except Exception:
        prompt = DEFAULT_PROMPT
    return {"system_prompt": prompt}

async def recall_semantic_memory(state: AgentState) -> dict:
    """Search for relevant facts about this user."""
    last_msg = state["messages"][-1]["content"]
    results = search_tool.invoke({
        "query": last_msg,
        "user_id": state["user_id"],
    })
    
    if results:
        context = "Known facts about this user:\n" + "\n".join(
            f"- {r['content']}" for r in results[:5]
        )
    else:
        context = ""
    return {"memory_context": context}

async def generate_response(state: AgentState) -> dict:
    """Generate response with memory-augmented prompt."""
    system = state["system_prompt"]
    if state["memory_context"]:
        system += f"\n\n{state['memory_context']}"
    
    messages = [{"role": "system", "content": system}] + state["messages"]
    response = await llm.ainvoke(messages)
    return {"response": response.content}

async def store_new_memories(state: AgentState) -> dict:
    """Extract and store new facts from this interaction."""
    last_user_msg = state["messages"][-1]["content"]
    
    # Simple extraction (production: use LLM-based extraction)
    manage_tool.invoke({
        "action": "create",
        "content": f"User interaction: {last_user_msg[:200]}",
        "user_id": state["user_id"],
    })
    return {}

async def handle_feedback_node(state: AgentState) -> dict:
    """If feedback was provided, update procedural memory."""
    if not state.get("feedback"):
        return {}
    
    improved = await prompt_optimizer.optimize(
        current_prompt=state["system_prompt"],
        conversation=state["messages"],
        feedback=state["feedback"],
    )
    
    await prompt_optimizer.store.put(
        namespace=("prompts", "assistant"),
        key="system_prompt",
        value={"prompt": improved},
    )
    return {"system_prompt": improved}

def should_process_feedback(state: AgentState) -> str:
    return "feedback" if state.get("feedback") else "end"

# Build graph
graph = StateGraph(AgentState)
graph.add_node("load_prompt", load_procedural_memory)
graph.add_node("recall", recall_semantic_memory)
graph.add_node("generate", generate_response)
graph.add_node("store", store_new_memories)
graph.add_node("feedback", handle_feedback_node)

graph.add_edge(START, "load_prompt")
graph.add_edge("load_prompt", "recall")
graph.add_edge("recall", "generate")
graph.add_edge("generate", "store")
graph.add_conditional_edges("store", should_process_feedback, {"feedback": "feedback", "end": END})
graph.add_edge("feedback", END)

# Compile with persistent checkpointing (episodic memory)
checkpointer = PostgresSaver.from_conn_string("postgresql://...")
agent = graph.compile(checkpointer=checkpointer)
```

## Decision Tree

```
Should I use LangMem?
  │
  ├─ Are you already using LangGraph?
  │   ├─ YES → LangMem is the most natural choice (native integration)
  │   └─ NO → Consider Mem0 or Zep (framework-agnostic)
  │
  ├─ Do you need self-improving prompts (procedural memory)?
  │   ├─ YES → LangMem is the ONLY option
  │   └─ NO → Any framework works
  │
  ├─ Is budget $0?
  │   ├─ YES → LangMem (MIT license, free forever)
  │   └─ NO → Consider managed services (Mem0 Cloud, Zep Cloud)
  │
  ├─ Is p95 latency < 1s a hard requirement?
  │   ├─ YES → LangMem's 59.82s p95 is a deal-breaker. Use Mem0 or Zep.
  │   └─ NO → LangMem is usable
  │
  └─ Do you need temporal reasoning?
      ├─ YES → LangMem is weak here. Use Zep.
      └─ NO → LangMem works
```

## When NOT to Use

- **When latency matters** -- LangMem's **59.82s p95 search latency** is an order of magnitude worse than Mem0 or Zep. This makes it unsuitable for real-time applications. The cause is that LangMem performs LLM-based processing during search, not just embedding comparison.
- **When you are not on LangGraph** -- LangMem is deeply coupled to the LangGraph ecosystem. Using it outside LangGraph is possible but loses most of its value.
- **When you need a managed service** -- LangMem is self-hosted only. You manage the vector store, the storage, the infrastructure. If you want hands-off memory, use Mem0 Cloud or Zep Cloud.
- **When you need graph/temporal memory** -- LangMem has no knowledge graph support. For relationship queries or temporal reasoning, use Zep.
- **When benchmark accuracy is critical** -- 58.1% LOCOMO is mid-tier. Not bad, but not great. Hindsight (91.4% LongMemEval) and Zep (63.8%) score higher.

## Tradeoffs

| Aspect | LangMem | Mem0 | Zep |
|---|---|---|---|
| Price | Free (MIT) | $0-$249/mo | $0-$25/mo |
| LangGraph integration | Native (best) | Plugin | Plugin |
| Procedural memory | Yes (unique) | No | No |
| Search latency (p95) | 59.82s (worst) | < 1s | < 1s |
| LOCOMO score | 58.1% | N/A | N/A |
| Graph memory | No | Pro only ($249) | All tiers ($25) |
| Temporal reasoning | Basic | Weak | Excellent |
| Self-hosted | Yes (only option) | Yes + Cloud | Yes + Cloud |
| Operational burden | Medium (you manage everything) | Low (managed) | Low-Medium |

## Real-World Examples

1. **Internal coding assistant** -- LangMem stores semantic memories about the team's codebase conventions. Procedural memory evolves the system prompt based on code review feedback: "You suggested camelCase but our Go codebase uses snake_case" -> prompt updated to enforce snake_case. Free, LangGraph-native, and the 59s latency is acceptable for a non-real-time tool.

2. **Learning management agent** -- LangMem's episodic memory stores each tutoring session. Semantic memory stores what the student knows. Procedural memory adapts teaching style based on feedback: "Too many examples, I learn better from analogies" -> prompt updated to prefer analogies over examples.

3. **DevOps runbook agent** -- Procedural memory is the killer feature here. Each incident resolution teaches the agent better debugging steps. After 50 incidents, the system prompt contains battle-tested runbook procedures that the agent learned from real failures.

## Failure Modes

| Failure | Symptom | Fix |
|---|---|---|
| Search timeout | 60s+ waits during memory recall | Add timeout, fallback to no-memory response |
| Procedural drift | System prompt evolves in unhelpful direction | Version prompts, A/B test changes, human review gate |
| Memory store corruption | Lost memories after restart | Use PostgresStore instead of InMemoryStore |
| Over-memorization | Agent stores every trivial fact | Add importance filtering before storage |
| LangGraph coupling | Cannot use with non-LangGraph frameworks | Accept the coupling or switch to Mem0/Zep |

## Source(s) and Further Reading

- [LangMem Documentation](https://langchain-ai.github.io/langmem/) -- Official docs
- [LangMem GitHub Repository](https://github.com/langchain-ai/langmem) -- MIT license
- [State of AI Agent Memory 2026](https://mem0.ai) -- Mem0 research report
- [Mem0 vs Zep vs LangMem Comparison](https://dev.to) -- dev.to
- [8 Agent Memory Frameworks Compared](https://vectorize.io) -- vectorize.io
- [LangGraph Memory Documentation](https://langchain-ai.github.io/langgraph/concepts/memory/) -- LangChain
