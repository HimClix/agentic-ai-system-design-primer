# LangGraph

> LangGraph is the enterprise default for agentic AI -- a state machine framework that gives you explicit control over agent flow, with built-in persistence, streaming, and human-in-the-loop. Found in 239 of 1,056 AI job listings (22.6%).

## What It Is

LangGraph is a framework for building stateful, multi-step AI applications as directed graphs. Built by LangChain Inc., it represents agents as state machines where:
- **Nodes** = Functions that transform state (LLM calls, tool execution, logic)
- **Edges** = Transitions between nodes (conditional or fixed)
- **State** = A shared, typed dictionary that persists across steps

Unlike LangChain's chains (which are linear), LangGraph supports cycles, branching, parallel execution, and human-in-the-loop -- everything needed for production agents.

## How It Works

### Core Concepts

```
                    +----------------+
                    |   StateGraph   |
                    +-------+--------+
                            |
              +-------------+-------------+
              |             |             |
        +-----v----+  +----v-----+  +----v-----+
        |  State   |  |  Nodes   |  |  Edges   |
        | TypedDict|  | Functions|  | Routing  |
        | shared   |  | that     |  | logic    |
        | across   |  | transform|  | (cond/   |
        | all nodes|  | state    |  |  fixed)  |
        +----------+  +----------+  +----------+

State: The single source of truth, passed to every node
Nodes: Pure functions that receive state and return updates
Edges: Define the control flow (can be conditional)
```

### The State Machine Model

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END, START
import operator

# 1. Define State
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # Reducer: append
    step_count: int                           # Overwrite

# 2. Define Nodes
def agent(state: AgentState) -> dict:
    """LLM decision node."""
    response = llm.invoke(state["messages"])
    return {"messages": [response], "step_count": state["step_count"] + 1}

def tools(state: AgentState) -> dict:
    """Tool execution node."""
    result = execute_tools(state["messages"][-1].tool_calls)
    return {"messages": [result]}

# 3. Build Graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools", tools)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")

# 4. Compile and Run
app = graph.compile()
result = app.invoke({"messages": [HumanMessage("...")], "step_count": 0})
```

## Production Implementation

### Checkpointing (Persistence)

Checkpointers save state after every node execution, enabling:
- Resume after crashes
- Human-in-the-loop (pause, approve, resume)
- Time-travel debugging

```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import MemorySaver

# Development: In-memory
memory_checkpointer = MemorySaver()

# Production: PostgreSQL
DB_URI = "postgresql://user:pass@localhost:5432/langgraph"
postgres_checkpointer = PostgresSaver.from_conn_string(DB_URI)

# Redis checkpointer for high-throughput
from langgraph.checkpoint.redis import RedisSaver
redis_checkpointer = RedisSaver.from_conn_string("redis://localhost:6379")

# Compile with checkpointer
app = graph.compile(checkpointer=postgres_checkpointer)

# Every invocation needs a thread_id for state isolation
config = {"configurable": {"thread_id": "user-123-session-456"}}
result = app.invoke({"messages": [HumanMessage("...")]}, config=config)

# Resume from checkpoint (e.g., after crash)
# Just invoke with the same thread_id -- it picks up where it left off
result = app.invoke({"messages": [HumanMessage("continue")]}, config=config)
```

### Human-in-the-Loop (HITL)

```python
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command

def sensitive_action(state):
    """Node that requires human approval."""
    action = state["proposed_action"]

    # This pauses execution and waits for human input
    human_response = interrupt({
        "question": f"Approve this action? {action}",
        "options": ["approve", "reject", "modify"]
    })

    if human_response == "approve":
        return {"action_approved": True}
    elif human_response == "reject":
        return {"action_approved": False}
    else:
        return {"proposed_action": human_response}  # Modified action

# Build graph with interrupt
graph = StateGraph(AgentState)
graph.add_node("plan", plan_node)
graph.add_node("approve", sensitive_action)
graph.add_node("execute", execute_node)

graph.set_entry_point("plan")
graph.add_edge("plan", "approve")
graph.add_conditional_edges("approve", check_approval, {
    "execute": "execute", "plan": "plan"
})
graph.add_edge("execute", END)

app = graph.compile(checkpointer=postgres_checkpointer)

# First invocation -- will pause at "approve" node
config = {"configurable": {"thread_id": "thread-1"}}
result = app.invoke({"messages": [HumanMessage("Delete user account")]}, config=config)
# result indicates the interrupt question

# Human provides answer -- resume execution
result = app.invoke(Command(resume="approve"), config=config)
```

### Streaming

```python
# Stream events as they happen
async for event in app.astream_events(
    {"messages": [HumanMessage("Research AI trends")]},
    config=config,
    version="v2"
):
    if event["event"] == "on_chat_model_stream":
        # Token-by-token streaming from LLM
        print(event["data"]["chunk"].content, end="")
    elif event["event"] == "on_tool_start":
        print(f"\nCalling tool: {event['name']}")
    elif event["event"] == "on_tool_end":
        print(f"\nTool result: {event['data'].content[:100]}")

# Stream full node outputs
async for chunk in app.astream(
    {"messages": [HumanMessage("...")]},
    config=config,
    stream_mode="updates"  # or "values" for full state
):
    print(f"Node: {list(chunk.keys())[0]}")
    print(f"Output: {list(chunk.values())[0]}")
```

### Subgraphs (Composable Agents)

```python
# Build specialist agents as subgraphs
research_graph = build_research_agent()  # Returns compiled StateGraph
code_graph = build_code_agent()

# Compose into larger system
parent_graph = StateGraph(ParentState)
parent_graph.add_node("research", research_graph)  # Subgraph as node
parent_graph.add_node("code", code_graph)           # Subgraph as node
parent_graph.add_node("supervisor", supervisor_node)

parent_graph.set_entry_point("supervisor")
# ... add edges ...

app = parent_graph.compile()
```

## Why It Is the Enterprise Default

### Job Market Data (2026)

From the LangChain Job Market Report:

```
Framework mentions in 1,056 AI engineering job listings:

LangChain:     362 listings (34.3%)
LangGraph:     239 listings (22.6%)
LlamaIndex:    112 listings (10.6%)
CrewAI:         67 listings  (6.3%)
AutoGen:        45 listings  (4.3%)
Pydantic AI:    28 listings  (2.7%)
DSPy:           19 listings  (1.8%)

LangGraph appears in 66% of listings that mention LangChain,
indicating it's the execution layer for LangChain-based systems.
```

### Enterprise Adoption Reasons

| Reason | Detail |
|--------|--------|
| **Explicit control flow** | Graphs are auditable, testable, debuggable |
| **Persistence** | PostgreSQL/Redis checkpointing for crash recovery |
| **HITL** | Built-in interrupt/resume for approval workflows |
| **Streaming** | Token-by-token and node-by-node streaming |
| **Subgraphs** | Compose complex systems from tested components |
| **LangSmith integration** | Production tracing, evaluation, monitoring |
| **Type safety** | TypedDict state prevents runtime errors |
| **Model agnostic** | Works with any LLM provider |

## Decision Tree: When to Use

```
     Should I use LangGraph?
                |
     +----------v----------+
     | Do you need explicit |
     | control over agent   |
     | execution flow?      |
     +---+------------+----+
         |            |
        Yes          No --> Consider CrewAI (simpler) or raw SDK
         |
     +---v-----------+----+
     | Do you need any of: |
     | - Persistence        |
     | - HITL              |
     | - Streaming         |
     | - Multi-agent       |
     +---+------------+---+
         |            |
        Yes          No --> Simple ReAct with raw SDK may suffice
         |
     +---v-----------+----+
     | Enterprise         |
     | requirements?      |
     | (audit, compliance,|
     |  observability)    |
     +---+------------+---+
         |            |
        Yes          No --> LangGraph still good, but evaluate simpler options
         |
     +---v-------------------+
     | USE LangGraph         |
     +------------------------+
```

## When NOT to Use

1. **Simple single-turn tool use**: Raw Claude/OpenAI API with tool calling is simpler
2. **Prototype with no persistence needs**: CrewAI or Pydantic AI are faster to start
3. **Non-Python environments**: LangGraph is Python-only (JS version exists but is less mature)
4. **Maximum performance**: The abstraction adds overhead vs raw API calls
5. **You want magic**: LangGraph is explicit -- if you want "just works" agents, try CrewAI

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Full control over execution flow | Verbose compared to higher-level frameworks |
| Production-ready persistence | Learning curve (state machines, reducers) |
| Built-in HITL with interrupt/resume | Python-primary (JS is catching up) |
| Model agnostic | Tied to LangChain ecosystem for best experience |
| LangSmith for observability | Abstraction overhead vs raw API |
| Growing enterprise adoption (22.6% of listings) | Can be over-engineered for simple tasks |

## Failure Modes

### 1. State Explosion
Too many fields in the state dictionary, each growing with every iteration.
**Mitigation**: Use reducers carefully. Summarize or trim state periodically.

### 2. Checkpointer Bottleneck
PostgreSQL checkpointer becomes a bottleneck at high throughput.
**Mitigation**: Use Redis for high-throughput, Postgres for durability. Consider write-behind caching.

### 3. Graph Complexity
Graph becomes so complex (20+ nodes, 50+ edges) that it is hard to debug.
**Mitigation**: Use subgraphs to encapsulate complexity. Each subgraph should be independently testable.

### 4. Reducer Conflicts
Two nodes try to update the same state field with incompatible reducers.
**Mitigation**: Design state schema carefully. Use Annotated reducers (operator.add for lists, custom for dicts).

## Sources and Further Reading

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [LangChain AI Job Market Report 2026](https://blog.langchain.dev/ai-job-market-report/)
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
