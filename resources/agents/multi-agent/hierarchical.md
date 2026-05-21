# Hierarchical Multi-Agent Pattern

> When a single supervisor has too many agents (8+), add levels of supervision -- but beware of latency and cost multiplication at each level.

## What It Is

The hierarchical pattern extends the supervisor pattern by creating a tree of supervisors, where each supervisor manages a small group (3-5) of agents or sub-supervisors. This mirrors how human organizations work: a VP manages directors, who manage managers, who manage individual contributors.

This pattern becomes necessary when:
- You have 8+ specialist agents
- Tasks span multiple domains that need sub-coordination
- The top-level supervisor's context becomes too large to route effectively

## How It Works

### Architecture

```
                    +--------------------+
                    |   TOP SUPERVISOR   |
                    |   (Strategic)      |
                    +---+----+----+------+
                        |    |    |
           +------------+    |    +------------+
           |                 |                 |
    +------v------+   +-----v------+   +------v------+
    | RESEARCH    |   | ENGINEERING|   | CONTENT     |
    | SUPERVISOR  |   | SUPERVISOR |   | SUPERVISOR  |
    +--+-----+----+   +--+----+---+   +--+-----+----+
       |     |           |    |          |     |
    +--v-+ +-v--+     +--v-+ +v--+   +--v-+ +-v--+
    |Web | |RAG |     |Code| |Test|  |Write| |Edit|
    |Srch| |    |     |Gen | |    |  |    | |    |
    +----+ +----+     +----+ +----+  +----+ +----+
```

### Execution Flow

```
User: "Research competitors, build a comparison dashboard, and write a report"

1. TOP SUPERVISOR:
   -> Route to RESEARCH SUPERVISOR: "Research competitors"
   -> (waits for results)

2. RESEARCH SUPERVISOR:
   -> Route to Web Search Agent: "Find top 5 competitors"
   -> Route to RAG Agent: "Pull internal competitive intel"
   -> Synthesize research findings
   -> Return to TOP SUPERVISOR

3. TOP SUPERVISOR:
   -> Route to ENGINEERING SUPERVISOR: "Build comparison dashboard with this data: {...}"

4. ENGINEERING SUPERVISOR:
   -> Route to Code Agent: "Generate dashboard component"
   -> Route to Test Agent: "Validate dashboard renders correctly"
   -> Return dashboard artifact to TOP SUPERVISOR

5. TOP SUPERVISOR:
   -> Route to CONTENT SUPERVISOR: "Write report using data:{...} and dashboard:{...}"

6. CONTENT SUPERVISOR:
   -> Route to Writing Agent: "Draft executive report"
   -> Route to Editing Agent: "Review and polish the draft"
   -> Return final report to TOP SUPERVISOR

7. TOP SUPERVISOR: Compile final deliverable and return to user.
```

## Production Implementation

```python
from typing import TypedDict, Annotated, Sequence, List
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

# -- State --
class HierarchicalState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_node: str
    results: dict
    depth: int
    iteration_count: int

# -- Sub-supervisors as compiled subgraphs --

def build_research_team() -> StateGraph:
    """Build a sub-supervisor with its own specialist agents."""

    def research_supervisor(state):
        llm = ChatAnthropic(model="claude-sonnet-4-20250514")
        response = llm.invoke([
            {"role": "system", "content": """You supervise research agents:
            - web_search: Search the internet
            - rag_search: Search internal knowledge base
            Decide which to use. Reply with agent name or DONE."""},
            *state["messages"]
        ])
        return {"next_node": response.content.strip()}

    def web_search_agent(state):
        llm = ChatAnthropic(model="claude-haiku-4-20250514")
        response = llm.invoke([
            {"role": "system", "content": "Search the web and return findings."},
            *state["messages"]
        ])
        return {"messages": [AIMessage(content=f"[Web Search]: {response.content}")]}

    def rag_search_agent(state):
        llm = ChatAnthropic(model="claude-haiku-4-20250514")
        response = llm.invoke([
            {"role": "system", "content": "Search internal knowledge base."},
            *state["messages"]
        ])
        return {"messages": [AIMessage(content=f"[RAG Search]: {response.content}")]}

    graph = StateGraph(HierarchicalState)
    graph.add_node("supervisor", research_supervisor)
    graph.add_node("web_search", web_search_agent)
    graph.add_node("rag_search", rag_search_agent)

    graph.set_entry_point("supervisor")

    def route(state):
        if state["next_node"] == "DONE":
            return "end"
        return state["next_node"]

    graph.add_conditional_edges("supervisor", route, {
        "web_search": "web_search",
        "rag_search": "rag_search",
        "end": END
    })
    graph.add_edge("web_search", "supervisor")
    graph.add_edge("rag_search", "supervisor")

    return graph.compile()

# -- Top-Level Supervisor --
def build_top_supervisor():
    """Top-level supervisor managing sub-teams."""

    research_team = build_research_team()
    # engineering_team = build_engineering_team()  # Similar pattern
    # content_team = build_content_team()          # Similar pattern

    def top_supervisor(state):
        llm = ChatAnthropic(model="claude-opus-4-20250514")
        response = llm.invoke([
            {"role": "system", "content": """You are the top-level supervisor.
            Route tasks to teams:
            - research_team: For data gathering and analysis
            - engineering_team: For code and dashboard creation
            - content_team: For writing and editing
            Reply with team name or FINISH."""},
            *state["messages"]
        ])
        return {"next_node": response.content.strip()}

    def invoke_research_team(state):
        result = research_team.invoke(state)
        return {"messages": result["messages"]}

    graph = StateGraph(HierarchicalState)
    graph.add_node("supervisor", top_supervisor)
    graph.add_node("research_team", invoke_research_team)
    # graph.add_node("engineering_team", invoke_engineering_team)
    # graph.add_node("content_team", invoke_content_team)

    graph.set_entry_point("supervisor")

    def route(state):
        if state["next_node"] == "FINISH":
            return "end"
        return state["next_node"]

    graph.add_conditional_edges("supervisor", route, {
        "research_team": "research_team",
        # "engineering_team": "engineering_team",
        # "content_team": "content_team",
        "end": END
    })
    graph.add_edge("research_team", "supervisor")

    return graph.compile()
```

## Latency and Cost Multiplication

### The Math of Hierarchy Depth

Each level of hierarchy multiplies both latency and cost:

```
Level 0 (Top Supervisor):    1 LLM call per routing decision
Level 1 (Sub-Supervisors):   1 LLM call per routing + 1 LLM call per agent
Level 2 (Sub-Sub-Supervisors): Same multiplication again

Total LLM calls for a 3-level, 3-teams-of-3 system:
  Top supervisor routing:      3 calls (one per sub-team invocation)
  Sub-supervisor routing:      3 calls per team * 3 teams = 9 calls
  Agent execution:             1 call per agent * 9 agents = 9 calls
  Total:                       21 LLM calls

Compare to flat supervisor:
  Supervisor routing:          9 calls (one per agent)
  Agent execution:             9 calls
  Total:                       18 LLM calls

Overhead: ~17% more calls for 2 levels vs flat.
For 3 levels: ~35% more calls.
```

### Latency Calculation

```
Flat supervisor (9 agents, sequential):
  9 * (supervisor_call + agent_call) = 9 * (0.5s + 1.5s) = 18 seconds

Hierarchical (3 teams of 3, sequential):
  top_routing: 0.5s
  + sub_routing: 0.5s + agent: 1.5s (per agent in team, 3 sequential)
  = 0.5 + 3 * (0.5 + 1.5)
  = 0.5 + 6.0 = 6.5 seconds per team
  Total: 0.5s + 3 * 6.5s = 20 seconds (worse!)

BUT with parallel team execution:
  Total: 0.5s + 6.5s = 7 seconds (much better!)
```

**Key insight**: Hierarchical is only faster than flat if teams execute in parallel.

## When to Use (vs Flat Supervisor)

```
     Flat supervisor struggling?
                |
     +----------v----------+
     | More than 8 agents  |
     | in one supervisor?  |
     +---+------------+----+
         |            |
        Yes          No --> Keep flat supervisor
         |
     +---v-----------+----+
     | Can agents be       |
     | grouped into 3-5    |
     | coherent teams?     |
     +---+------------+---+
         |            |
        Yes          No --> Consider reducing agent count
         |
     +---v-----------+----+
     | Can teams execute   |
     | in parallel?        |
     +---+------------+---+
         |            |
        Yes          No --> Latency will increase, weigh carefully
         |
     +---v-------------------+
     | USE Hierarchical      |
     +------------------------+
```

## When NOT to Use

1. **<8 agents**: Flat supervisor handles this well; hierarchy adds unnecessary complexity
2. **Tightly coupled agents**: If every agent needs every other agent's output, hierarchy adds latency without benefit
3. **Latency-critical with sequential teams**: Hierarchy is slower than flat when teams must execute serially
4. **Simple routing**: If a rule-based router can decide, don't add LLM supervisors
5. **Cost-constrained**: Each supervisor level adds 10-35% more LLM calls

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Handles 10-50+ agents | Added latency per hierarchy level |
| Mirrors organizational structure | 10-35% more LLM calls |
| Each supervisor has smaller scope (better routing) | More complex to debug |
| Sub-teams can be developed/tested independently | State management across levels is hard |
| Parallel team execution can reduce latency | Failure in one level can cascade |

## Failure Modes

### 1. Over-Hierarchization
Creating 3 levels when 2 would suffice. Each level costs latency and tokens.
**Mitigation**: Start flat, add levels only when routing accuracy drops below 90%.

### 2. Information Loss Between Levels
Top supervisor summarizes for sub-supervisor, losing critical details.
**Mitigation**: Pass full context, not summaries. Or pass structured data objects.

### 3. Sub-Supervisor Confusion
Sub-supervisor doesn't understand the broader context of why its team was called.
**Mitigation**: Top supervisor includes task context and expected output format in the delegation message.

### 4. Cascading Failures
If the research team fails, the engineering team gets bad input, producing bad output.
**Mitigation**: Validate intermediate results at each supervisor level before forwarding.

## Sources and Further Reading

- [LangGraph Hierarchical Agent Teams](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/)
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents)
- [Multi-Agent Architectures - LangGraph Concepts](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
