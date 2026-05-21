# ReWOO: Reasoning Without Observation

> ReWOO plans ALL tool calls upfront and executes them in parallel -- the lowest cost and lowest latency agent pattern when lookups are independent.

## What It Is

ReWOO (Reasoning Without Observation) was introduced by Xu et al. (2023) as an alternative to ReAct. The key insight: if the agent can plan all its tool calls before seeing any results, it can execute them in parallel and avoid the growing-context problem of ReAct.

While ReAct interleaves reasoning and acting (think-act-observe-think-act-observe...), ReWOO separates them completely:
1. **Plan**: Generate ALL thoughts and tool calls in one pass
2. **Execute**: Run all tool calls (potentially in parallel)
3. **Solve**: Combine all results into a final answer

## How It Works

### Architecture Comparison

```
=== ReAct (Sequential) ===              === ReWOO (Parallel) ===

Think -> Act -> Observe                 Plan ALL actions at once
Think -> Act -> Observe                 Execute ALL in parallel
Think -> Act -> Observe                 Solve with all results
Think -> Answer
                                        
Time: ████████████████ (serial)         Time: ████████ (parallel)

LLM calls: 4+ (growing context)        LLM calls: 2 (fixed context)
```

### Step-by-Step Flow

```
User: "What is the combined population of France and Germany?"

=== PLAN PHASE (Single LLM call) ===

Thought 1: I need to find the population of France.
Action 1: search("France population 2024") -> #E1

Thought 2: I need to find the population of Germany.
Action 2: search("Germany population 2024") -> #E2

Thought 3: I need to add both populations.
Action 3: calculate("#E1 + #E2") -> #E3
  (Note: #E1 and #E2 are placeholders, resolved after execution)

=== EXECUTE PHASE (Parallel where possible) ===

#E1 = search("France population 2024")  -->  "67.75 million"  |
#E2 = search("Germany population 2024") -->  "84.48 million"  | (parallel)
#E3 = calculate("67.75 + 84.48")        -->  "152.23 million" (waits for #E1, #E2)

=== SOLVE PHASE (Single LLM call) ===

Given results:
  #E1 = 67.75 million
  #E2 = 84.48 million
  #E3 = 152.23 million

Answer: "The combined population of France (67.75M) and Germany (84.48M) is
         approximately 152.23 million."
```

### Execution DAG

```
        Plan
       / | \
      /  |  \
   #E1  #E2  (parallel -- independent)
      \  |
       \ |
       #E3   (depends on #E1 and #E2)
        |
      Solve
```

## Production Implementation

```python
import asyncio
import re
from typing import TypedDict, Dict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# -- State --
class ReWOOState(TypedDict):
    query: str
    plan: str                          # Raw plan text
    tool_calls: List[Dict]             # Parsed tool calls with placeholders
    results: Dict[str, str]            # #E1 -> result mapping
    dependency_graph: Dict[str, List]  # #E3 -> [#E1, #E2]
    final_answer: Optional[str]

# -- Models --
planner_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
solver_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

# -- Planner Node --
PLAN_PROMPT = """For the given task, create a plan with tool calls.
Use #E1, #E2, etc. as evidence placeholders for tool results.
You can reference earlier evidence in later steps.

Available tools:
- search(query): Search the web
- calculate(expression): Evaluate a math expression

Format each step as:
Thought N: <reasoning>
Action N: tool_name(args) -> #EN

Plan all steps before any execution."""

def plan_node(state: ReWOOState) -> dict:
    response = planner_llm.invoke([
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state["query"])
    ])

    plan_text = response.content
    tool_calls, dep_graph = parse_plan(plan_text)

    return {
        "plan": plan_text,
        "tool_calls": tool_calls,
        "dependency_graph": dep_graph
    }

def parse_plan(plan_text: str) -> tuple:
    """Parse plan into structured tool calls and dependency graph."""
    tool_calls = []
    dep_graph = {}

    # Regex to find Action N: tool(args) -> #EN
    pattern = r'Action \d+:\s*(\w+)\((.+?)\)\s*->\s*(#E\d+)'
    for match in re.finditer(pattern, plan_text):
        tool_name = match.group(1)
        args = match.group(2)
        evidence_id = match.group(3)

        # Find dependencies (references to #E1, #E2, etc. in args)
        deps = re.findall(r'#E\d+', args)
        dep_graph[evidence_id] = deps

        tool_calls.append({
            "tool": tool_name,
            "args": args,
            "evidence_id": evidence_id,
            "deps": deps
        })

    return tool_calls, dep_graph

# -- Executor Node (Parallel) --
async def execute_tool(tool_name: str, args: str, results: dict) -> str:
    """Execute a single tool, resolving any placeholders."""
    # Replace evidence placeholders with actual results
    resolved_args = args
    for eid, result in results.items():
        resolved_args = resolved_args.replace(eid, result)

    if tool_name == "search":
        return await async_search(resolved_args)
    elif tool_name == "calculate":
        return str(eval(resolved_args))  # Sandbox in production!
    else:
        raise ValueError(f"Unknown tool: {tool_name}")

async def execute_node(state: ReWOOState) -> dict:
    """Execute all tool calls, respecting dependencies, maximizing parallelism."""
    results = {}
    tool_calls = state["tool_calls"]
    dep_graph = state["dependency_graph"]

    # Topological sort for execution order
    executed = set()
    while len(executed) < len(tool_calls):
        # Find calls whose dependencies are all satisfied
        ready = [
            tc for tc in tool_calls
            if tc["evidence_id"] not in executed
            and all(d in executed for d in tc["deps"])
        ]

        if not ready:
            raise RuntimeError("Circular dependency in plan")

        # Execute all ready calls in parallel
        tasks = [
            execute_tool(tc["tool"], tc["args"], results)
            for tc in ready
        ]
        task_results = await asyncio.gather(*tasks)

        for tc, result in zip(ready, task_results):
            results[tc["evidence_id"]] = result
            executed.add(tc["evidence_id"])

    return {"results": results}

# -- Solver Node --
def solve_node(state: ReWOOState) -> dict:
    """Combine all results into a final answer."""
    evidence_text = "\n".join([
        f"{eid}: {result}" for eid, result in state["results"].items()
    ])

    response = solver_llm.invoke([
        SystemMessage(content="Use the evidence to answer the question concisely."),
        HumanMessage(content=f"""
        Question: {state['query']}
        Plan: {state['plan']}
        Evidence:
        {evidence_text}
        """)
    ])

    return {"final_answer": response.content}

# -- Build Graph --
graph = StateGraph(ReWOOState)
graph.add_node("plan", plan_node)
graph.add_node("execute", execute_node)
graph.add_node("solve", solve_node)

graph.set_entry_point("plan")
graph.add_edge("plan", "execute")
graph.add_edge("execute", "solve")
graph.add_edge("solve", END)

app = graph.compile()
```

## Token Cost Analysis

ReWOO is the cheapest agent pattern because:
1. Only 2 LLM calls regardless of how many tools are used
2. No growing context window -- each call has fixed context size

```
=== ReWOO Cost (Same task: combined population of France and Germany) ===

Plan call:   Input 600 tokens  | Output 300 tokens
Solve call:  Input 800 tokens  | Output 200 tokens
Total:       Input 1,400       | Output 500

Cost (Claude Sonnet): 1400 * $3/M + 500 * $15/M = $0.0117

=== Comparison ===

| Pattern          | Input Tokens | Output Tokens | Cost    | LLM Calls |
|------------------|-------------|---------------|---------|-----------|
| ReAct            | 11,500      | 1,750         | $0.061  | 5         |
| Plan-and-Execute | 3,800       | 850           | $0.025  | 5         |
| ReWOO            | 1,400       | 500           | $0.012  | 2         |

ReWOO is 5x cheaper than ReAct and 2x cheaper than Plan-and-Execute.
```

## Decision Tree: When to Use

```
     Should I use ReWOO?
                |
     +----------v----------+
     | Are ALL lookups      |
     | independent?         |
     | (no result depends   |
     |  on another result)  |
     +---+------------+----+
         |            |
        Yes          No
         |            |
         |     +------v---------+
         |     | Can you group  |
         |     | into mostly    |
         |     | independent    |
         |     | batches?       |
         |     +--+--------+---+
         |        |        |
         |       Yes      No --> Use ReAct or Plan-and-Execute
         |        |
     +---v--------v-------+
     | Is cost/latency    |
     | critical?          |
     +---+------------+---+
         |            |
        Yes          No --> ReAct is fine (more flexible)
         |
     +---v-------------------+
     | USE ReWOO             |
     +------------------------+
```

## When NOT to Use

1. **Sequential dependencies**: If step 2 depends on step 1's result for its query (not just its result), ReWOO cannot work
2. **Exploratory tasks**: When you don't know what to search until you see initial results
3. **Dynamic tool selection**: When which tools to call depends on intermediate results
4. **Tasks requiring iteration**: When you might need to refine a search based on initial results
5. **Conversation-style interactions**: Where each step is informed by human feedback

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Lowest token cost (2 LLM calls only) | Cannot handle sequential dependencies |
| Lowest latency (parallel execution) | Plan must be correct on first attempt |
| No growing context problem | No mid-execution course correction |
| Simple, predictable execution | Limited to tasks with independent lookups |
| Easy to parallelize and scale | Planner must anticipate all needed information |
| Fixed cost regardless of tool count | Cannot adapt if a tool returns unexpected results |

## Real-World Examples

1. **Multi-entity fact lookup**: "What are the capitals and populations of France, Germany, and Spain?" -- three independent searches.
2. **Parallel API calls**: Fetching user profile, order history, and payment status simultaneously.
3. **Document comparison**: Retrieving multiple documents to compare -- all retrievals are independent.
4. **Multi-source news aggregation**: Searching for news on a topic across multiple sources in parallel.

## Failure Modes

### 1. Hidden Dependencies
The planner creates steps that look independent but actually depend on each other:
```
#E1: search("Who is the CEO of Apple?")
#E2: search("#E1 net worth")  <-- Depends on #E1!
```
**Mitigation**: Validate dependency graph before execution. If a placeholder appears in args, it must be listed as a dependency.

### 2. Planner Underestimates Steps
The planner misses a needed lookup because it cannot see intermediate results:
```
Query: "What is the tallest building in the country that won the most Olympic gold medals in 2024?"
Plan: search("most Olympic gold medals 2024") -> #E1
      search("tallest building in #E1") -> #E2
      
Problem: #E2 depends on #E1 -- this is NOT parallelizable!
ReWOO either handles this as sequential or fails.
```
**Mitigation**: Detect sequential dependencies and fall back to Plan-and-Execute for those chains.

### 3. Tool Failure Without Recovery
If a parallel tool call fails, there's no opportunity to retry with a different query.
**Mitigation**: Implement retry logic at the tool execution level (not the agent level).

### 4. Over-Planning
The planner creates more tool calls than necessary, wasting execution time.
**Mitigation**: Constrain the planner prompt to minimize tool calls.

## Sources and Further Reading

- [ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models](https://arxiv.org/abs/2305.18323) (Xu et al., 2023)
- [LangGraph ReWOO Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rewoo/rewoo/)
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents)
