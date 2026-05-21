# Plan-and-Execute

> Separate planning from execution -- use a strong model to plan once, then a cheaper model to execute each step. Saves ~70% on tokens versus ReAct.

## What It Is

Plan-and-Execute (also called Plan-then-Execute) separates the agent into two distinct phases:

1. **Planner**: A strong LLM creates a full plan of steps before any execution begins
2. **Executor**: A (potentially cheaper) LLM executes each step independently, using tools as needed

This separation was formalized in the "Plan-and-Solve Prompting" paper (Wang et al., 2023) and extended to agentic systems by LangChain and others. The key insight: planning and execution require different capabilities, so they can use different models at different costs.

## How It Works

### Architecture

```
     +------------------+
     |   User Query     |
     +--------+---------+
              |
     +--------v---------+
     |   PLANNER         |     (Strong model: Claude Opus / GPT-4o)
     |   "Break this     |
     |    into steps"    |
     +--------+---------+
              |
              v
     +------------------+
     | Plan:            |
     | 1. Search for X  |
     | 2. Extract Y     |
     | 3. Calculate Z   |
     | 4. Format answer |
     +--------+---------+
              |
     +--------v---------+     +-------------------+
     | EXECUTOR Step 1  |---->| Tool: search(X)   |
     | (Cheap model)    |<----| Result: ...       |
     +--------+---------+     +-------------------+
              |
     +--------v---------+     +-------------------+
     | EXECUTOR Step 2  |---->| Tool: extract(Y)  |
     | (Cheap model)    |<----| Result: ...       |
     +--------+---------+     +-------------------+
              |
     +--------v---------+
     | EXECUTOR Step 3  |---->| Tool: calculate(Z)|
     | (Cheap model)    |<----| Result: ...       |
     +--------+---------+     +-------------------+
              |
     +--------v---------+
     | EXECUTOR Step 4  |     Final formatting
     | (Cheap model)    |
     +--------+---------+
              |
     +--------v---------+
     | RE-PLANNER       |     (Optional: check if plan needs updating)
     | "Is the plan     |
     |  still valid?"   |
     +--------+---------+
              |
     +--------v---------+
     |   Final Answer   |
     +------------------+
```

### Step-by-Step Flow

```
User: "Compare the market caps of Apple and Microsoft and calculate which is larger by what percentage."

PLANNER (Claude Opus):
  Step 1: Search for Apple's current market cap
  Step 2: Search for Microsoft's current market cap
  Step 3: Calculate which is larger and by what percentage
  Step 4: Format a comparison summary

EXECUTOR (Claude Haiku - Step 1):
  Tool: search("Apple market cap 2024")
  Result: "Apple market cap: $3.45 trillion"
  Output: "Apple's market cap is $3.45 trillion"

EXECUTOR (Claude Haiku - Step 2):
  Tool: search("Microsoft market cap 2024")
  Result: "Microsoft market cap: $3.12 trillion"
  Output: "Microsoft's market cap is $3.12 trillion"

EXECUTOR (Claude Haiku - Step 3):
  Tool: calculator("(3.45 - 3.12) / 3.12 * 100")
  Result: "10.58%"
  Output: "Apple is larger by 10.58%"

EXECUTOR (Claude Haiku - Step 4):
  Output: "Apple ($3.45T) has a larger market cap than Microsoft ($3.12T) by 10.58%."

RE-PLANNER: Plan complete, all steps executed. Return final answer.
```

## Production Implementation

### Using LangGraph

```python
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# -- State --
class PlanStep(BaseModel):
    step_number: int
    description: str
    status: str = "pending"  # pending | completed | failed
    result: Optional[str] = None

class PlanExecuteState(TypedDict):
    query: str
    plan: List[PlanStep]
    current_step: int
    results: List[str]
    final_answer: Optional[str]
    replan_count: int

# -- Models --
planner_llm = ChatAnthropic(model="claude-opus-4-20250514", temperature=0)
executor_llm = ChatAnthropic(model="claude-haiku-4-20250514", temperature=0)
replanner_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

# -- Planner Node --
class Plan(BaseModel):
    steps: List[str] = Field(description="List of steps to complete the task")

def plan_node(state: PlanExecuteState) -> dict:
    """Generate an execution plan."""
    response = planner_llm.with_structured_output(Plan).invoke([
        SystemMessage(content="""You are a planning expert. Break the user's query
        into concrete, executable steps. Each step should be independently
        actionable. Keep the plan to 3-7 steps."""),
        HumanMessage(content=state["query"])
    ])

    plan = [
        PlanStep(step_number=i+1, description=step)
        for i, step in enumerate(response.steps)
    ]
    return {"plan": plan, "current_step": 0, "results": []}

# -- Executor Node --
def execute_step(state: PlanExecuteState) -> dict:
    """Execute the current step using cheaper model + tools."""
    step = state["plan"][state["current_step"]]
    prior_results = "\n".join([
        f"Step {i+1} result: {r}" for i, r in enumerate(state["results"])
    ])

    response = executor_llm.invoke([
        SystemMessage(content=f"""Execute this step precisely.
        Prior results for context:
        {prior_results}"""),
        HumanMessage(content=step.description)
    ])

    step.status = "completed"
    step.result = response.content

    new_results = state["results"] + [response.content]
    return {
        "results": new_results,
        "current_step": state["current_step"] + 1
    }

# -- Re-planner Node --
def replan_node(state: PlanExecuteState) -> dict:
    """Check if plan needs adjustment based on results so far."""
    completed_info = "\n".join([
        f"Step {i+1}: {state['plan'][i].description} -> {r}"
        for i, r in enumerate(state["results"])
    ])

    response = replanner_llm.invoke([
        SystemMessage(content="""Review the original query and completed steps.
        If the plan is complete and the query is answered, set is_complete=true.
        If the plan needs new steps, provide them."""),
        HumanMessage(content=f"""
        Original query: {state['query']}
        Completed steps: {completed_info}
        Remaining plan: {[s.description for s in state['plan'][state['current_step']:]]}
        """)
    ])

    # Simplified: in production, parse structured output for replanning
    return {"replan_count": state["replan_count"] + 1}

# -- Routing --
def should_continue(state: PlanExecuteState) -> str:
    if state["current_step"] >= len(state["plan"]):
        return "replan"
    return "execute"

def after_replan(state: PlanExecuteState) -> str:
    if state["replan_count"] > 3:
        return "finish"
    # Check if complete (simplified)
    return "finish"

def finish_node(state: PlanExecuteState) -> dict:
    all_results = "\n".join(state["results"])
    return {"final_answer": all_results}

# -- Build Graph --
graph = StateGraph(PlanExecuteState)
graph.add_node("plan", plan_node)
graph.add_node("execute", execute_step)
graph.add_node("replan", replan_node)
graph.add_node("finish", finish_node)

graph.set_entry_point("plan")
graph.add_edge("plan", "execute")
graph.add_conditional_edges("execute", should_continue, {
    "execute": "execute",
    "replan": "replan"
})
graph.add_conditional_edges("replan", after_replan, {
    "execute": "execute",
    "finish": "finish"
})
graph.add_edge("finish", END)

app = graph.compile()
```

## Cost Comparison: Plan-and-Execute vs ReAct

### Same task: "Compare market caps of Apple and Microsoft"

```
=== ReAct (Claude Sonnet for everything) ===

  Iter 1 (think+search):  Input  600  | Output 350   | $3/M, $15/M
  Iter 2 (think+search):  Input 1450  | Output 350
  Iter 3 (think+calc):    Input 2300  | Output 350
  Iter 4 (think+format):  Input 3150  | Output 400
  Iter 5 (final answer):  Input 4000  | Output 300
  -------------------------------------------------------
  Total Input:  11,500 tokens * $3/M  = $0.0345
  Total Output:  1,750 tokens * $15/M = $0.0263
  TOTAL: $0.0608

=== Plan-and-Execute (Opus plan, Haiku execute) ===

  Plan (Opus):     Input  600 | Output 200  | $15/M, $75/M
  Exec 1 (Haiku):  Input  400 | Output 150  | $0.25/M, $1.25/M
  Exec 2 (Haiku):  Input  500 | Output 150
  Exec 3 (Haiku):  Input  600 | Output 150
  Exec 4 (Haiku):  Input  700 | Output 200
  -------------------------------------------------------
  Planner:  600 * $15/M + 200 * $75/M = $0.024
  Executor: 2200 * $0.25/M + 650 * $1.25/M = $0.0014
  TOTAL: $0.0254

  SAVINGS: 58% cheaper
  (With prompt caching on executor context: ~70% cheaper)
```

### At Scale (10,000 queries/day)

| Metric | ReAct | Plan-and-Execute | Savings |
|--------|-------|------------------|---------|
| Cost/query | $0.061 | $0.025 | 59% |
| Daily cost | $610 | $250 | $360/day |
| Monthly cost | $18,300 | $7,500 | $10,800/mo |
| Latency (avg) | 8.2s | 5.1s | 38% faster |

## Re-planning Strategies

### 1. Always Re-plan (Safest, Most Expensive)
After every step, check if the plan needs updating. Use when results are unpredictable.

### 2. Re-plan on Failure Only
Only invoke the re-planner when a step fails or returns unexpected results.

### 3. Re-plan at Checkpoints
Re-plan after every N steps (e.g., every 3 steps). Good balance of safety and cost.

### 4. Never Re-plan (Cheapest, Riskiest)
Execute the original plan without modification. Use only when tasks are highly predictable.

```python
# Re-plan on failure pattern
def execute_with_replan(state):
    try:
        result = execute_step(state)
        if is_result_valid(result):
            return result
        else:
            # Result was unexpected -- trigger replan
            return replan_and_continue(state, reason="unexpected_result")
    except ToolExecutionError as e:
        return replan_and_continue(state, reason=f"tool_error: {e}")
```

## Decision Tree: When to Use

```
     Should I use Plan-and-Execute?
                |
     +----------v----------+
     | Can you break the    |
     | task into discrete   |
     | steps BEFORE seeing  |
     | any results?         |
     +---+------------+----+
         |            |
        Yes          No --> Use ReAct
         |
     +---v-----------+----+
     | Is cost a          |
     | concern at scale?  |
     +---+------------+---+
         |            |
        Yes          No --> ReAct is fine
         |
     +---v-----------+----+
     | Do you need >5     |
     | steps typically?   |
     +---+------------+---+
         |            |
        Yes          No --> ReAct is fine for short tasks
         |
     +---v-------------------+
     | USE Plan-and-Execute  |
     +------------------------+
```

## When NOT to Use

1. **Purely exploratory tasks**: If you genuinely cannot predict the steps, ReAct is better
2. **Tasks with fewer than 3 steps**: Planning overhead is not worth it
3. **Highly interactive tasks**: User asks follow-ups that invalidate the plan
4. **Real-time streaming**: Users expect token-by-token responses, not "planning..." silence

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| ~70% cheaper than ReAct at scale | Initial planning step adds latency |
| Executor steps are independent (parallelizable) | Plan can be wrong, requiring expensive replanning |
| Can use different models for different steps | Less flexible for truly exploratory tasks |
| Better for long tasks (no context accumulation) | Two-model setup adds operational complexity |
| Explicit plan is auditable and debuggable | Planner may under/over-decompose |
| Steps can be checkpointed and resumed | Rigid plan may miss opportunities found mid-execution |

## Real-World Examples

1. **Report generation**: Plan the sections, research each section independently, compile.
2. **Multi-step data analysis**: Plan analysis steps, execute each query, synthesize findings.
3. **Compliance checking**: Plan which regulations to check, verify each independently.
4. **Travel planning**: Plan the components (flights, hotels, activities), research each, compile itinerary.

## Failure Modes

### 1. Over-Decomposition
Planner creates 15 steps for a 3-step task, wasting executor calls.
**Mitigation**: Constrain planner to 3-7 steps. Add "combine trivial steps."

### 2. Under-Decomposition
Planner creates 2 vague steps that the executor cannot execute.
**Mitigation**: Include examples of good decomposition in the planner prompt.

### 3. Plan Invalidation
Step 2's result shows that Step 3 is impossible as planned.
**Mitigation**: Re-plan after each step or at checkpoints.

### 4. Executor Scope Creep
Executor tries to do more than its assigned step (e.g., answering the full question in Step 1).
**Mitigation**: Strong system prompt: "Execute ONLY this step. Do NOT answer the original question."

### 5. Lost Context Between Steps
Each executor step is independent, so Step 3 may not know what Step 1 found.
**Mitigation**: Pass accumulated results as context to each executor step.

## Sources and Further Reading

- [Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning](https://arxiv.org/abs/2305.04091) (Wang et al., 2023)
- [LangGraph Plan-and-Execute Tutorial](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/)
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents)
- [Understanding AI Agent Architecture](https://blog.langchain.dev/planning-agents/) (LangChain Blog)
