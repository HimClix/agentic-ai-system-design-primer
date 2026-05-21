# State Machine Design with LangGraph
> The agent loop is a state machine where nodes are processing steps and edges are deterministic routing decisions -- even though the LLM inside each node is nondeterministic.

## What It Is

State machine design for agents defines the explicit states an agent can be in, the transitions between states, and the conditions that determine which transition to take. LangGraph implements this as a `StateGraph` with nodes (functions), edges (transitions), and conditional edges (routing logic).

The key insight: by making the macro-level flow deterministic (state machine), you contain the nondeterminism of the LLM to individual nodes, making the overall system predictable and debuggable.

## How It Works

### LangGraph Concepts

```
StateGraph
├── State: TypedDict defining all data flowing through the graph
├── Nodes: Functions that process state and return updates
├── Edges: Fixed transitions (A always goes to B)
├── Conditional Edges: Dynamic routing based on state
├── Entry Point: Where execution starts
├── END: Terminal node
└── Checkpointer: Persists state between steps for recovery
```

### Common Agent State Machine Pattern

```
                    ┌──────────┐
                    │  START   │
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │  parse   │ Extract intent, entities
                    │  input   │ from user message
                    └────┬─────┘
                         │
                    ┌────▼─────┐    ┌──────────┐
                    │  plan    │───►│ escalate  │──► END
                    │          │    │ (if too   │
                    └────┬─────┘    │ complex)  │
                         │         └──────────┘
                         │
                    ┌────▼─────┐
              ┌────►│ execute  │
              │     │ step     │
              │     └────┬─────┘
              │          │
              │     ┌────▼─────┐
              │     │ evaluate │
              │     │ result   │
              │     └────┬─────┘
              │          │
              │    ┌─────┴──────┐
              │    │            │
              │  done?     more steps?
              │    │            │
              │    ▼            │
              │  ┌──────┐      │
              │  │verify│      │
              │  └──┬───┘      │
              │     │          │
              │  passed?  ─────┘ (retry)
              │     │
              │     ▼
              │  ┌──────────┐
              └──│ respond   │──► END
                 └──────────┘
```

## Production Implementation

```python
from typing import TypedDict, Annotated, Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
import operator
import json


# --- State Definition ---

class AgentState(TypedDict):
    """Complete state for a production agent."""
    # Messages accumulate (append-only via operator.add)
    messages: Annotated[list[BaseMessage], operator.add]
    
    # Execution tracking
    step_count: int
    current_phase: str
    
    # Planning
    plan: Optional[list[str]]
    current_plan_step: int
    
    # Tool execution
    pending_tool_calls: list[dict]
    tool_results: list[dict]
    
    # Quality control
    confidence: float
    errors: list[str]
    
    # Human-in-the-loop
    requires_approval: bool
    approval_reason: str
    
    # Output
    final_response: Optional[str]


# --- Node Functions ---

async def parse_input(state: AgentState) -> dict:
    """Parse user input and extract intent."""
    last_message = state["messages"][-1]
    
    model = ChatAnthropic(model="claude-haiku-3.5")
    result = await model.ainvoke([
        {"role": "system", "content": "Extract the user's intent. Output JSON: {intent, entities, complexity}"},
        {"role": "user", "content": last_message.content},
    ])
    
    parsed = json.loads(result.content)
    return {
        "current_phase": "parsed",
        "step_count": state.get("step_count", 0) + 1,
    }


async def create_plan(state: AgentState) -> dict:
    """Create an execution plan based on the parsed intent."""
    model = ChatAnthropic(model="claude-sonnet-4-20250514")
    
    result = await model.ainvoke([
        {"role": "system", "content": """Create a step-by-step plan. Output JSON:
        {"steps": ["step1", "step2", ...], "estimated_complexity": "low|medium|high"}"""},
        *state["messages"],
    ])
    
    plan_data = json.loads(result.content)
    return {
        "plan": plan_data["steps"],
        "current_plan_step": 0,
        "current_phase": "planned",
        "step_count": state["step_count"] + 1,
    }


async def execute_step(state: AgentState) -> dict:
    """Execute the current plan step using tools."""
    model = ChatAnthropic(model="claude-sonnet-4-20250514")
    model_with_tools = model.bind_tools(TOOLS)
    
    current_step = state["plan"][state["current_plan_step"]]
    
    result = await model_with_tools.ainvoke([
        {"role": "system", "content": f"Execute this step: {current_step}. Use tools as needed."},
        *state["messages"],
    ])
    
    tool_calls = []
    if hasattr(result, "tool_calls") and result.tool_calls:
        for tc in result.tool_calls:
            tool_result = await execute_tool(tc["name"], tc["args"])
            tool_calls.append({
                "tool": tc["name"],
                "args": tc["args"],
                "result": tool_result,
            })
    
    return {
        "messages": [result],
        "tool_results": tool_calls,
        "current_plan_step": state["current_plan_step"] + 1,
        "step_count": state["step_count"] + 1,
        "current_phase": "executing",
    }


async def verify_output(state: AgentState) -> dict:
    """Verify the quality of the agent's output."""
    # Deterministic checks first
    checks = {
        "has_response": state.get("final_response") is not None,
        "within_step_limit": state["step_count"] <= 25,
        "no_errors": len(state.get("errors", [])) == 0,
    }
    
    all_passed = all(checks.values())
    confidence = sum(checks.values()) / len(checks)
    
    return {
        "confidence": confidence,
        "current_phase": "verified" if all_passed else "needs_retry",
    }


async def generate_response(state: AgentState) -> dict:
    """Generate the final user-facing response."""
    model = ChatAnthropic(model="claude-sonnet-4-20250514")
    
    result = await model.ainvoke([
        {"role": "system", "content": "Synthesize a clear, helpful response based on the tool results."},
        *state["messages"],
    ])
    
    return {
        "messages": [result],
        "final_response": result.content,
        "current_phase": "responded",
    }


async def escalate_to_human(state: AgentState) -> dict:
    """Escalate to human when agent can't handle the request."""
    return {
        "final_response": "I'm connecting you with a human agent who can better assist you.",
        "current_phase": "escalated",
        "requires_approval": True,
        "approval_reason": f"Agent exceeded limits or low confidence ({state.get('confidence', 0):.2f})",
    }


# --- Routing Functions ---

def route_after_plan(state: AgentState) -> Literal["execute_step", "escalate"]:
    """Route based on plan complexity and step count."""
    if state["step_count"] > 20:
        return "escalate"
    if state.get("plan") and len(state["plan"]) > 10:
        return "escalate"  # Too complex, get a human
    return "execute_step"


def route_after_execute(state: AgentState) -> Literal["execute_step", "verify", "escalate"]:
    """Route based on execution status."""
    if state["step_count"] > 25:
        return "escalate"  # Bounded autonomy
    
    if state.get("errors") and len(state["errors"]) > 3:
        return "escalate"  # Too many errors
    
    # More steps in the plan?
    if state["current_plan_step"] < len(state.get("plan", [])):
        return "execute_step"  # Continue executing plan
    
    return "verify"  # Plan complete, verify output


def route_after_verify(state: AgentState) -> Literal["generate_response", "execute_step", "escalate"]:
    """Route based on verification results."""
    if state["confidence"] >= 0.85:
        return "generate_response"
    if state["step_count"] < 20:
        return "execute_step"  # Retry
    return "escalate"


# --- Graph Construction ---

def build_production_agent():
    """Build the complete production agent graph."""
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("parse_input", parse_input)
    graph.add_node("create_plan", create_plan)
    graph.add_node("execute_step", execute_step)
    graph.add_node("verify", verify_output)
    graph.add_node("generate_response", generate_response)
    graph.add_node("escalate", escalate_to_human)
    
    # Add edges
    graph.set_entry_point("parse_input")
    graph.add_edge("parse_input", "create_plan")
    graph.add_conditional_edges("create_plan", route_after_plan)
    graph.add_conditional_edges("execute_step", route_after_execute)
    graph.add_conditional_edges("verify", route_after_verify)
    graph.add_edge("generate_response", END)
    graph.add_edge("escalate", END)
    
    # Compile with PostgreSQL checkpointing
    checkpointer = AsyncPostgresSaver.from_conn_string(
        "postgresql://user:pass@localhost:5432/agents"
    )
    
    return graph.compile(checkpointer=checkpointer)


# --- Usage ---

async def handle_user_message(session_id: str, message: str):
    agent = build_production_agent()
    
    config = {
        "configurable": {
            "thread_id": session_id,  # Enables conversation persistence
        }
    }
    
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "step_count": 0,
        "current_phase": "started",
        "plan": None,
        "current_plan_step": 0,
        "pending_tool_calls": [],
        "tool_results": [],
        "confidence": 0.0,
        "errors": [],
        "requires_approval": False,
        "approval_reason": "",
        "final_response": None,
    }
    
    result = await agent.ainvoke(initial_state, config=config)
    return result["final_response"]
```

## Decision Tree: State Machine Complexity

```
How complex is the agent's task?
│
├── Single-step (classify, extract, answer)
│   └── No state machine needed. Single LLM call.
│
├── Multi-step, fixed sequence
│   └── Linear graph: A → B → C → D → END
│
├── Multi-step, conditional flow
│   └── State machine with conditional edges (this page's pattern)
│
└── Multi-agent with delegation
    └── Hierarchical state machines (supervisor pattern)
```

## When NOT to Use State Machines

- **Single-turn Q&A**: No need for state management. Direct LLM call.
- **Simple ReAct loop**: LangGraph's `create_react_agent` handles this without explicit state machine design.
- **Prototype**: Start with `create_react_agent`, add explicit state machine when you need more control.

## Tradeoffs

| Approach | Flexibility | Debuggability | Complexity |
|----------|------------|---------------|-----------|
| ReAct (automatic loop) | High (LLM decides) | Low (opaque loop) | Low |
| Linear graph | Low (fixed path) | High (predictable) | Low |
| Conditional state machine | High (controlled) | High (explicit transitions) | Medium |
| Hierarchical (multi-agent) | Highest | Medium (distributed state) | High |

## Failure Modes

1. **State explosion**: Too many conditional edges make the graph unmaintainable. Mitigation: max 10-15 nodes, decompose complex flows into sub-graphs.
2. **Checkpoint deserialization failure**: State schema changes between deployments. Mitigation: version state schemas, backward-compatible changes.
3. **Deadlock**: Two conditional edges create a cycle with no exit condition. Mitigation: step counter as universal escape hatch.

## Source(s) and Further Reading

- LangGraph Low-Level API: https://langchain-ai.github.io/langgraph/concepts/low_level/
- LangGraph Checkpointing: https://langchain-ai.github.io/langgraph/concepts/persistence/
- State Machines in UI: https://stately.ai/docs (similar concepts, different domain)
