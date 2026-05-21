# Workflow Graph Execution (DAG-Based)

> Pre-defined directed acyclic graphs where nodes are processing steps and edges define data flow. The graph is deterministic -- code decides the execution path, not the LLM.

## What It Is

Workflow Graph Execution is the pattern where agent logic is encoded as a **directed acyclic graph (DAG)** of processing steps. Unlike ReAct (where the LLM decides the next action) or Plan-and-Execute (where the LLM creates a plan), the workflow graph is **defined in code** and executes deterministically.

This is what Anthropic calls a **"workflow"** in their taxonomy (as opposed to an "agent"):
- **Workflow**: Predefined control flow, LLMs used at specific nodes
- **Agent**: LLM dynamically decides control flow

### The Spectrum

```
MORE DETERMINISTIC ◄─────────────────────────► MORE AUTONOMOUS

DAG Workflow    Plan-and-Execute    ReAct          Full Agent
(code decides)  (LLM plans,        (LLM decides   (LLM decides
                 code executes)     each step)     everything)

    ▲                                                  ▲
    │                                                  │
 This file                                     react.md, reflexion.md
```

## How It Works

### Core Concepts

```
DAG Definition:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  [Input] → [Classify] → [Route] ──► [Summarize] → [Output] │
│                            │                                │
│                            ├──► [Extract Entities] → [Output]│
│                            │                                │
│                            └──► [Translate] → [Output]      │
│                                                             │
│  Nodes: Processing steps (may or may not involve LLMs)      │
│  Edges: Data flow (deterministic routing)                   │
│  State: Shared data structure passed through the graph      │
└─────────────────────────────────────────────────────────────┘
```

Key properties:
1. **No cycles** (DAG, not a general graph) -- prevents infinite loops by construction
2. **Deterministic routing** -- conditional edges are based on state values, not LLM decisions
3. **Parallel execution** -- independent branches run concurrently
4. **Statically analyzable** -- you can verify the graph before execution

### Workflow vs Agent Decision-Making

```
ReAct (agent decides):
  User: "Translate this document and summarize it"
  LLM thinks: "I should translate first..."
  LLM acts: translate()
  LLM thinks: "Now I should summarize..."
  LLM acts: summarize()
  → LLM decides the order at runtime

Workflow Graph (code decides):
  graph.add_edge("input", "translate")
  graph.add_edge("translate", "summarize")
  graph.add_edge("summarize", "output")
  → Code defines the order at compile time
```

## Production Implementation

### LangGraph DAG Implementation

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# --- State Definition ---
class WorkflowState(TypedDict):
    input_text: str
    classification: str        # "support" | "sales" | "technical"
    sentiment: str             # "positive" | "negative" | "neutral"
    response: str
    metadata: dict

# --- Node Functions ---
llm = ChatAnthropic(model="claude-haiku-3.5", temperature=0)

def classify_intent(state: WorkflowState) -> dict:
    """Node 1: Classify the user's intent. Uses fast model."""
    result = llm.invoke([
        HumanMessage(content=(
            f"Classify this message into exactly one category: "
            f"support, sales, or technical.\n\n"
            f"Message: {state['input_text']}\n\n"
            f"Respond with ONLY the category name."
        ))
    ])
    return {"classification": result.content.strip().lower()}

def analyze_sentiment(state: WorkflowState) -> dict:
    """Node 2: Analyze sentiment. Runs in PARALLEL with classification."""
    result = llm.invoke([
        HumanMessage(content=(
            f"What is the sentiment of this message? "
            f"Respond with ONLY: positive, negative, or neutral.\n\n"
            f"Message: {state['input_text']}"
        ))
    ])
    return {"sentiment": result.content.strip().lower()}

def handle_support(state: WorkflowState) -> dict:
    """Node 3a: Handle support queries."""
    tone = "empathetic" if state["sentiment"] == "negative" else "friendly"
    result = llm.invoke([
        HumanMessage(content=(
            f"You are a {tone} customer support agent. "
            f"Respond to: {state['input_text']}"
        ))
    ])
    return {"response": result.content}

def handle_sales(state: WorkflowState) -> dict:
    """Node 3b: Handle sales queries."""
    result = llm.invoke([
        HumanMessage(content=(
            f"You are a helpful sales assistant. "
            f"Respond to: {state['input_text']}"
        ))
    ])
    return {"response": result.content}

def handle_technical(state: WorkflowState) -> dict:
    """Node 3c: Handle technical queries with a stronger model."""
    strong_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
    result = strong_llm.invoke([
        HumanMessage(content=(
            f"You are a technical support engineer. "
            f"Provide a detailed technical response to: {state['input_text']}"
        ))
    ])
    return {"response": result.content}

def format_output(state: WorkflowState) -> dict:
    """Node 4: Format the final response with metadata."""
    return {
        "metadata": {
            "classification": state["classification"],
            "sentiment": state["sentiment"],
            "model_used": (
                "claude-sonnet" if state["classification"] == "technical"
                else "claude-haiku"
            ),
        }
    }

# --- Routing Function (deterministic, based on state) ---
def route_by_classification(state: WorkflowState) -> Literal["support", "sales", "technical"]:
    """Deterministic routing -- code decides, not LLM."""
    return state["classification"]

# --- Build the DAG ---
workflow = StateGraph(WorkflowState)

# Add nodes
workflow.add_node("classify", classify_intent)
workflow.add_node("sentiment", analyze_sentiment)
workflow.add_node("support", handle_support)
workflow.add_node("sales", handle_sales)
workflow.add_node("technical", handle_technical)
workflow.add_node("format", format_output)

# Define edges (the DAG structure)
workflow.set_entry_point("classify")

# Parallel execution: sentiment runs alongside classification
# (In LangGraph, you'd use a fan-out pattern or separate subgraph)

# Conditional routing after classification
workflow.add_conditional_edges(
    "classify",
    route_by_classification,
    {
        "support": "support",
        "sales": "sales",
        "technical": "technical",
    }
)

# All handlers converge to format
workflow.add_edge("support", "format")
workflow.add_edge("sales", "format")
workflow.add_edge("technical", "format")
workflow.add_edge("format", END)

# Compile
app = workflow.compile()

# Execute
result = app.invoke({
    "input_text": "My API integration is throwing 429 errors",
    "classification": "",
    "sentiment": "",
    "response": "",
    "metadata": {},
})
```

### Pure Python DAG Engine (No Frameworks)

```python
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DAGNode:
    name: str
    func: Callable[[dict], Awaitable[dict]]
    dependencies: list[str] = field(default_factory=list)
    condition: Callable[[dict], bool] | None = None  # Skip if returns False
    timeout_seconds: float = 30.0
    retry_count: int = 0
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    duration_ms: float = 0.0


class WorkflowDAG:
    """
    Lightweight DAG executor for agent workflows.
    - Automatic parallel execution of independent nodes
    - Conditional node execution
    - Timeout and retry support
    - Execution tracing
    """

    def __init__(self):
        self.nodes: dict[str, DAGNode] = {}
        self.state: dict = {}

    def add_node(
        self,
        name: str,
        func: Callable,
        dependencies: list[str] = None,
        condition: Callable = None,
        timeout: float = 30.0,
        retries: int = 0,
    ) -> "WorkflowDAG":
        self.nodes[name] = DAGNode(
            name=name,
            func=func,
            dependencies=dependencies or [],
            condition=condition,
            timeout_seconds=timeout,
            retry_count=retries,
        )
        return self

    def _validate_dag(self) -> None:
        """Ensure no cycles exist (topological sort check)."""
        visited = set()
        in_stack = set()

        def dfs(node_name: str):
            if node_name in in_stack:
                raise ValueError(f"Cycle detected involving node: {node_name}")
            if node_name in visited:
                return
            in_stack.add(node_name)
            for dep in self.nodes[node_name].dependencies:
                if dep not in self.nodes:
                    raise ValueError(f"Node '{node_name}' depends on unknown node '{dep}'")
                dfs(dep)
            in_stack.remove(node_name)
            visited.add(node_name)

        for name in self.nodes:
            dfs(name)

    def _get_ready_nodes(self) -> list[DAGNode]:
        """Find nodes whose dependencies are all completed."""
        ready = []
        for node in self.nodes.values():
            if node.status != NodeStatus.PENDING:
                continue
            deps_met = all(
                self.nodes[dep].status in (NodeStatus.COMPLETED, NodeStatus.SKIPPED)
                for dep in node.dependencies
            )
            if deps_met:
                ready.append(node)
        return ready

    async def _execute_node(self, node: DAGNode) -> None:
        """Execute a single node with timeout and retry."""
        # Check condition
        if node.condition and not node.condition(self.state):
            node.status = NodeStatus.SKIPPED
            logger.info(f"Node '{node.name}' skipped (condition not met)")
            return

        node.status = NodeStatus.RUNNING
        start = time.monotonic()

        for attempt in range(node.retry_count + 1):
            try:
                result = await asyncio.wait_for(
                    node.func(self.state),
                    timeout=node.timeout_seconds,
                )
                # Merge result into shared state
                if isinstance(result, dict):
                    self.state.update(result)
                node.result = result
                node.status = NodeStatus.COMPLETED
                node.duration_ms = (time.monotonic() - start) * 1000
                logger.info(
                    f"Node '{node.name}' completed in {node.duration_ms:.0f}ms"
                )
                return

            except asyncio.TimeoutError:
                logger.warning(
                    f"Node '{node.name}' timed out (attempt {attempt + 1})"
                )
            except Exception as e:
                logger.error(
                    f"Node '{node.name}' failed (attempt {attempt + 1}): {e}"
                )

        node.status = NodeStatus.FAILED
        node.duration_ms = (time.monotonic() - start) * 1000
        raise RuntimeError(f"Node '{node.name}' failed after {node.retry_count + 1} attempts")

    async def execute(self, initial_state: dict) -> dict:
        """Execute the DAG with maximum parallelism."""
        self._validate_dag()
        self.state = initial_state.copy()

        while True:
            ready = self._get_ready_nodes()
            if not ready:
                # Check if all nodes are done
                pending = [n for n in self.nodes.values() if n.status == NodeStatus.PENDING]
                if pending:
                    raise RuntimeError(
                        f"Deadlock: nodes {[n.name for n in pending]} cannot execute"
                    )
                break

            # Execute all ready nodes in parallel
            await asyncio.gather(
                *(self._execute_node(node) for node in ready)
            )

        return self.state

    def get_trace(self) -> list[dict]:
        """Return execution trace for observability."""
        return [
            {
                "node": node.name,
                "status": node.status.value,
                "duration_ms": node.duration_ms,
                "dependencies": node.dependencies,
            }
            for node in self.nodes.values()
        ]


# --- Usage Example ---

async def main():
    import anthropic

    client = anthropic.AsyncAnthropic()

    async def classify(state: dict) -> dict:
        resp = await client.messages.create(
            model="claude-haiku-3.5",
            max_tokens=20,
            messages=[{"role": "user", "content": f"Classify as support/sales/technical: {state['input']}"}],
        )
        return {"category": resp.content[0].text.strip().lower()}

    async def enrich(state: dict) -> dict:
        # Lookup customer info from DB (parallel with classify)
        return {"customer_tier": "enterprise", "account_age_days": 450}

    async def generate_response(state: dict) -> dict:
        model = "claude-sonnet-4-20250514" if state["customer_tier"] == "enterprise" else "claude-haiku-3.5"
        resp = await client.messages.create(
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": f"As a {state['category']} agent, respond to: {state['input']}"}],
        )
        return {"response": resp.content[0].text}

    # Build DAG
    dag = WorkflowDAG()
    dag.add_node("classify", classify)
    dag.add_node("enrich", enrich)  # No dependencies → runs in parallel with classify
    dag.add_node("respond", generate_response, dependencies=["classify", "enrich"])

    result = await dag.execute({"input": "My API is returning 500 errors"})
    print(result["response"])
```

## Decision Tree: Workflow Graph vs Agent

```
         Should I use a DAG workflow or a dynamic agent?
                              │
                  ┌───────────▼────────────┐
                  │ Do you know all the     │
                  │ processing steps at     │
                  │ design time?            │
                  └───┬────────────────┬───┘
                     Yes              No ──► Use ReAct or
                      │                      Plan-and-Execute
                  ┌───▼────────────────┐
                  │ Are the steps       │
                  │ always the same     │
                  │ (just different     │
                  │ data)?              │
                  └───┬────────────┬───┘
                     Yes          Sometimes
                      │            │
                      │       ┌────▼───────────────┐
                      │       │ Can routing be      │
                      │       │ based on simple     │
                      │       │ conditions (not     │
                      │       │ LLM judgment)?      │
                      │       └───┬────────────┬───┘
                      │          Yes           No ──► Use agent
                      │           │                    with tools
                  ┌───▼───────────▼──┐
                  │  USE DAG WORKFLOW  │
                  └───────────────────┘
```

## When NOT to Use

1. **Exploratory tasks**: If you cannot enumerate the steps upfront (research, debugging), use ReAct.
2. **Highly dynamic routing**: If which step to take depends on nuanced LLM understanding, code-based routing is too rigid.
3. **Simple single-step tasks**: If the task is just "call LLM once," a DAG is overkill.
4. **Rapidly evolving requirements**: If the workflow changes weekly, maintaining a DAG becomes expensive. Use a flexible agent instead.
5. **User-driven conversation**: Chatbots cannot be pre-defined as DAGs (the user controls the flow).

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Deterministic execution (testable, predictable) | Cannot handle unexpected tasks |
| Maximum parallelism (independent nodes run concurrently) | Rigid structure requires code changes |
| Easy to debug (trace which node failed) | Cannot adapt to novel situations |
| Cost-predictable (known number of LLM calls) | Over-engineering for simple tasks |
| Static analysis (detect cycles, dead nodes at compile time) | Routing logic limited to what code can express |
| Works with existing orchestration tools (Airflow, Prefect) | LLM nodes add non-determinism to an otherwise deterministic graph |

## Comparison with Orchestration Frameworks

| Framework | Type | Best For | Agent Support |
|-----------|------|----------|---------------|
| **LangGraph** | Graph + state machine | LLM workflows with conditional logic | Native (nodes can be LLM calls) |
| **Prefect** | DAG orchestrator | Data pipelines, ML workflows | Via task decorators |
| **Airflow** | DAG orchestrator | Batch processing, ETL | DAG-as-code, not real-time |
| **Temporal** | Durable execution | Long-running workflows with retries | Activity-based |
| **Step Functions** | State machine (AWS) | Cloud-native orchestration | Via Lambda integration |

## Failure Modes

### 1. Routing Mismatch
Classification node outputs "billing" but routing only handles "support/sales/technical." The request falls through.
**Mitigation**: Always include a default/fallback route. Validate classification output against known categories.

### 2. Cascading Failures
Node 3 depends on Node 2 which depends on Node 1. Node 1 fails, causing the entire pipeline to fail.
**Mitigation**: Implement fallback values, circuit breakers, and graceful degradation per node.

### 3. State Corruption
Two parallel nodes both write to the same state key, creating a race condition.
**Mitigation**: Use node-scoped output keys. Merge state immutably. Use a state schema that prevents conflicts.

### 4. Timeout Cascade
A slow LLM call in one node blocks downstream nodes, causing the entire workflow to exceed SLA.
**Mitigation**: Per-node timeouts. Abort early if the total workflow budget is exceeded.

### 5. Over-Rigid Routing
The DAG handles 95% of cases but 5% require a step that does not exist in the graph.
**Mitigation**: Add a "human escalation" or "agent fallback" node for cases that do not match any route.

## Sources and Further Reading

- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents) -- Defines "workflows" vs "agents"
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) -- Primary framework for LLM-aware DAGs
- [Prefect](https://www.prefect.io/) -- General-purpose workflow orchestration
- [Apache Airflow](https://airflow.apache.org/) -- DAG-based batch workflow engine
- [Temporal.io](https://temporal.io/) -- Durable execution for long-running workflows
- [AWS Step Functions](https://aws.amazon.com/step-functions/) -- Serverless state machine orchestration
