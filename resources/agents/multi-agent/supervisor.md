# Supervisor Pattern

> A central supervisor agent routes tasks to specialist agents and synthesizes their results -- the most common multi-agent pattern in production.

## What It Is

The Supervisor pattern uses a single "manager" agent that:
1. Receives the user's request
2. Decides which specialist agent(s) to invoke
3. Passes relevant context to each specialist
4. Collects results from specialists
5. Synthesizes a final response or delegates the next step

This maps directly to the Orchestrator-Workers workflow pattern from Anthropic's taxonomy, but with LLM-driven routing instead of code-driven routing.

## How It Works

### Architecture

```
                    +------------------+
                    |   User Query     |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   SUPERVISOR      |
                    |   (Router LLM)    |
                    |                   |
                    |   Decides:        |
                    |   - Which agent?  |
                    |   - What context? |
                    |   - When to stop? |
                    +---+----+----+----+
                        |    |    |
              +---------+    |    +---------+
              |              |              |
     +--------v---+  +------v-----+  +-----v------+
     | Research   |  | Code       |  | Writing    |
     | Agent      |  | Agent      |  | Agent      |
     | (search,   |  | (code gen, |  | (draft,    |
     |  RAG)      |  |  test,     |  |  edit,     |
     +--------+---+  |  debug)    |  |  format)   |
              |      +------+-----+  +-----+------+
              |             |              |
              +------+------+------+-------+
                     |             |
                     v             v
              +------+-------------+------+
              |     SUPERVISOR              |
              |     Synthesizes results     |
              |     or routes next step     |
              +-----------------------------+
```

### State Flow

```
1. User: "Analyze our Q4 sales data and write a report with charts"

2. Supervisor: Route to Research Agent
   -> "Query the sales database for Q4 metrics"

3. Research Agent returns: {revenue: $2.3M, growth: 12%, ...}

4. Supervisor: Route to Code Agent
   -> "Generate Python charts for this data: {revenue: $2.3M, ...}"

5. Code Agent returns: chart_code.py

6. Supervisor: Route to Writing Agent
   -> "Write executive summary using data: {revenue...} and charts: chart_code.py"

7. Writing Agent returns: report.md

8. Supervisor: All tasks complete. Return final report to user.
```

## Production Implementation

### Using LangGraph

```python
from typing import TypedDict, Annotated, Literal, Sequence
from langgraph.graph import StateGraph, END, START
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

# -- Shared State --
class SupervisorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    iteration_count: int

# -- Specialist Agents --
def research_agent(state: SupervisorState) -> dict:
    """Specialist: web search, RAG, database queries."""
    research_llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    # Agent has access to search and RAG tools
    response = research_llm.invoke([
        {"role": "system", "content": "You are a research specialist. Search for data and return structured findings."},
        *state["messages"]
    ])
    return {"messages": [AIMessage(content=f"[Research Agent]: {response.content}")]}

def code_agent(state: SupervisorState) -> dict:
    """Specialist: code generation, testing, debugging."""
    code_llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    response = code_llm.invoke([
        {"role": "system", "content": "You are a code specialist. Write, test, and debug code."},
        *state["messages"]
    ])
    return {"messages": [AIMessage(content=f"[Code Agent]: {response.content}")]}

def writing_agent(state: SupervisorState) -> dict:
    """Specialist: writing, editing, formatting."""
    writing_llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    response = writing_llm.invoke([
        {"role": "system", "content": "You are a writing specialist. Draft, edit, and format text."},
        *state["messages"]
    ])
    return {"messages": [AIMessage(content=f"[Writing Agent]: {response.content}")]}

# -- Supervisor Node --
SUPERVISOR_SYSTEM = """You are a supervisor managing specialist agents:
- research_agent: For web search, data retrieval, RAG queries
- code_agent: For code generation, testing, debugging
- writing_agent: For drafting, editing, formatting text

Given the conversation so far, decide which agent should act next.
If the task is complete, respond with FINISH.

Respond with ONLY one of: research_agent, code_agent, writing_agent, FINISH"""

supervisor_llm = ChatAnthropic(model="claude-opus-4-20250514", temperature=0)

def supervisor_node(state: SupervisorState) -> dict:
    """Route to the appropriate specialist."""
    response = supervisor_llm.invoke([
        {"role": "system", "content": SUPERVISOR_SYSTEM},
        *state["messages"]
    ])

    next_agent = response.content.strip()
    return {
        "next_agent": next_agent,
        "iteration_count": state["iteration_count"] + 1
    }

# -- Routing --
MAX_ITERATIONS = 10

def route_supervisor(state: SupervisorState) -> str:
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "end"
    if state["next_agent"] == "FINISH":
        return "end"
    return state["next_agent"]

# -- Build Graph --
graph = StateGraph(SupervisorState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("research_agent", research_agent)
graph.add_node("code_agent", code_agent)
graph.add_node("writing_agent", writing_agent)

graph.set_entry_point("supervisor")

graph.add_conditional_edges("supervisor", route_supervisor, {
    "research_agent": "research_agent",
    "code_agent": "code_agent",
    "writing_agent": "writing_agent",
    "end": END
})

# All agents route back to supervisor after completing
graph.add_edge("research_agent", "supervisor")
graph.add_edge("code_agent", "supervisor")
graph.add_edge("writing_agent", "supervisor")

app = graph.compile()

# -- Run --
result = app.invoke({
    "messages": [HumanMessage(content="Analyze our Q4 sales and write a report")],
    "next_agent": "",
    "iteration_count": 0
})
```

### State Management Patterns

#### Pattern 1: Shared State (Simple)
All agents read/write to the same state dictionary. Simple but can cause conflicts.

```python
class SharedState(TypedDict):
    messages: list         # Shared conversation
    research_data: dict    # Research agent output
    code_artifacts: list   # Code agent output
    draft_text: str        # Writing agent output
```

#### Pattern 2: Scoped State (Production)
Each agent has its own state namespace. Supervisor copies relevant data between them.

```python
class ScopedState(TypedDict):
    supervisor_messages: list
    research_context: dict    # Only research agent writes
    code_context: dict        # Only code agent writes
    writing_context: dict     # Only writing agent writes
    shared_artifacts: list    # Supervisor manages
```

#### Pattern 3: Message-Passing (Cleanest)
Agents communicate only through messages appended to a shared history.

```python
# Each agent appends a prefixed message:
"[Research Agent]: Found Q4 revenue is $2.3M with 12% growth..."
"[Code Agent]: Generated bar chart showing monthly breakdown..."
"[Writing Agent]: Draft report attached: executive_summary.md"

# Supervisor reads the full history to decide next steps
```

## When Supervisor Becomes a Bottleneck

The supervisor is a single point of failure and a potential bottleneck:

### Problem Signs
1. **Latency**: Every agent call requires a supervisor LLM call (2x the calls)
2. **Cost**: Supervisor context grows with every agent result
3. **Misrouting**: Supervisor sends tasks to wrong agents (>10% misroute rate)
4. **Bottleneck at scale**: All concurrent requests queue through one supervisor

### Solutions

| Problem | Solution |
|---------|----------|
| High latency | Cache supervisor routing decisions for common patterns |
| Growing context | Summarize agent results before adding to supervisor context |
| Misrouting | Add a classification layer before supervisor (code-based routing) |
| Single bottleneck | Use hierarchical pattern (supervisor of supervisors) |
| Too many agents | Split into groups of 3-5 per supervisor |

### Rule of thumb: A supervisor handles **3-7 agents** well. At 8+, switch to hierarchical.

## Decision Tree: When to Use

```
     Should I use the Supervisor pattern?
                |
     +----------v----------+
     | Do you have multiple |
     | specialist agents    |
     | that need to         |
     | coordinate?          |
     +---+------------+----+
         |            |
        Yes          No --> Single agent (pick a pattern from patterns/)
         |
     +---v-----------+----+
     | Are tasks routed    |
     | sequentially        |
     | (one agent at       |
     |  a time)?           |
     +---+------------+---+
         |            |
        Yes          No --> Consider parallel fan-out
         |
     +---v-----------+----+
     | Do you have         |
     | 3-7 specialist      |
     | agents?             |
     +---+------------+---+
         |            |
        Yes          No (8+) --> Use hierarchical pattern
         |
     +---v-------------------+
     | USE Supervisor        |
     +------------------------+
```

## When NOT to Use

1. **All tasks are independent**: Use parallel fan-out (map-reduce) instead
2. **Simple routing**: If routing is deterministic, use a code-based router (no LLM needed)
3. **>8 specialist agents**: Supervisor quality degrades; use hierarchical
4. **Latency-critical**: Double LLM calls (supervisor + agent) add latency
5. **Single-domain tasks**: If one agent can handle everything, don't add a supervisor

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Clear separation of concerns | Supervisor is a bottleneck/SPOF |
| Easy to add/remove specialists | Every task requires 2+ LLM calls |
| Centralized decision-making | Supervisor context grows with each agent result |
| Easy to debug (supervisor trace shows routing) | Misrouting wastes entire agent execution |
| Works with any specialist agent pattern | Added complexity vs single agent |

## Real-World Examples

1. **Enterprise customer support**: Supervisor routes to billing agent, technical support agent, or escalation agent based on the customer's issue.
2. **Code review system**: Supervisor routes to security reviewer, style checker, and performance analyzer, then synthesizes findings.
3. **Content pipeline**: Supervisor coordinates research agent, writing agent, and fact-checking agent to produce articles.

## Failure Modes

### 1. Supervisor Routing Loops
Supervisor keeps passing tasks back and forth between two agents without progress.
**Mitigation**: Track agent visit count. If agent called >2 times, force supervisor to synthesize or escalate.

### 2. Context Explosion
Each agent result adds ~500-1000 tokens. After 5 agent calls, supervisor has 3000+ tokens of context just from results.
**Mitigation**: Summarize agent results before adding to supervisor context. Keep only actionable information.

### 3. Agent Capability Confusion
Supervisor routes a research task to the code agent because the query mentions "data."
**Mitigation**: Include explicit examples in the supervisor system prompt. Use structured output for routing decisions.

### 4. Responsibility Gaps
No agent claims a task that falls between specialties (e.g., "create a chart" -- code or writing?).
**Mitigation**: Define a default agent for ambiguous tasks. Add catch-all capabilities.

## Sources and Further Reading

- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents)
- [LangGraph Multi-Agent Supervisor Tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)
- [Multi-Agent Architectures - LangGraph Docs](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
