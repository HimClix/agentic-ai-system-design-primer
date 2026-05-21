# Parallel Fan-Out (Map-Reduce for Agents)

> When sub-tasks are independent, fan out to parallel agents and merge results -- like map-reduce but for LLM agents.

## What It Is

The parallel fan-out pattern (also called map-reduce for agents) splits a task into independent sub-tasks, executes them in parallel across multiple agents, and merges the results. This is the agent-level equivalent of map-reduce: the "map" phase distributes work, the "reduce" phase combines results.

In LangGraph, this is implemented using the **Send API**, which dynamically creates parallel branches at runtime.

## How It Works

### Architecture

```
                    +------------------+
                    |   User Query     |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   DECOMPOSER     |
                    |   (Split task    |
                    |    into N parts) |
                    +--+-+--+--+--+---+
                       | |  |  |  |
          +------------+ |  |  |  +------------+
          |              |  |  |               |
     +----v----+  +------v+ | +v------+ +-----v----+
     | Agent 1 |  |Agent 2| | |Agent 4| | Agent 5  |
     | (Part A)|  |(Part B| | |(Part D)| | (Part E) |
     +---------+  +-------+ | +-------+ +----------+
                        +----v----+
                        | Agent 3 |
                        | (Part C)|
                        +---------+
          |              |  |  |               |
          +------+-------+--+--+-------+-------+
                 |                     |
                 v                     v
          +------+-----------+---------+------+
          |         MERGER / REDUCER          |
          |   Combine all results into        |
          |   final coherent output           |
          +-----------------------------------+
```

### Step-by-Step Example

```
User: "Summarize the top 5 news stories from today"

DECOMPOSER:
  Part 1: "Summarize story about [topic A]"
  Part 2: "Summarize story about [topic B]"
  Part 3: "Summarize story about [topic C]"
  Part 4: "Summarize story about [topic D]"
  Part 5: "Summarize story about [topic E]"

PARALLEL EXECUTION (all run simultaneously):
  Agent 1 -> Summary A (2.1 seconds)
  Agent 2 -> Summary B (1.8 seconds)
  Agent 3 -> Summary C (2.4 seconds)  <-- determines total time
  Agent 4 -> Summary D (1.5 seconds)
  Agent 5 -> Summary E (1.9 seconds)

Total time: max(2.1, 1.8, 2.4, 1.5, 1.9) = 2.4 seconds
(vs sequential: 2.1+1.8+2.4+1.5+1.9 = 9.7 seconds)

MERGER:
  Combines all 5 summaries into a cohesive news digest
  with consistent formatting and cross-references.
```

## Production Implementation

### Using LangGraph Send API

```python
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
import operator

# -- State --
class FanOutState(TypedDict):
    query: str
    sub_tasks: List[str]
    results: Annotated[List[str], operator.add]
    final_answer: str

class WorkerState(TypedDict):
    """State for individual parallel workers."""
    task: str
    result: str

# -- Decomposer --
decomposer_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

def decompose(state: FanOutState) -> dict:
    """Split the task into independent sub-tasks."""
    response = decomposer_llm.invoke([
        {"role": "system", "content": """Break this task into independent sub-tasks.
        Return each sub-task on a new line, prefixed with 'TASK:'.
        Each sub-task should be independently executable."""},
        {"role": "user", "content": state["query"]}
    ])

    tasks = [
        line.replace("TASK:", "").strip()
        for line in response.content.split("\n")
        if line.strip().startswith("TASK:")
    ]
    return {"sub_tasks": tasks}

# -- Fan-out using Send API --
def route_to_workers(state: FanOutState):
    """Create parallel Send() calls for each sub-task."""
    return [
        Send("worker", {"task": task, "result": ""})
        for task in state["sub_tasks"]
    ]

# -- Worker (runs in parallel for each sub-task) --
worker_llm = ChatAnthropic(model="claude-haiku-4-20250514", temperature=0)

def worker(state: WorkerState) -> dict:
    """Execute a single sub-task. Runs in parallel with other workers."""
    response = worker_llm.invoke([
        {"role": "system", "content": "Complete this task concisely and accurately."},
        {"role": "user", "content": state["task"]}
    ])
    return {"result": response.content}

# -- Collect results (runs after all workers complete) --
def collect_results(state: FanOutState) -> dict:
    """Collect results from all parallel workers."""
    # Results are automatically aggregated via the operator.add reducer
    return {}

# -- Merger --
merger_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

def merge(state: FanOutState) -> dict:
    """Combine all parallel results into a coherent output."""
    results_text = "\n\n".join([
        f"--- Result {i+1} ---\n{result}"
        for i, result in enumerate(state["results"])
    ])

    response = merger_llm.invoke([
        {"role": "system", "content": """Combine these parallel results into a single,
        coherent response. Resolve any contradictions. Ensure consistent formatting."""},
        {"role": "user", "content": f"""
        Original query: {state['query']}
        Results to merge:
        {results_text}
        """}
    ])

    return {"final_answer": response.content}

# -- Build Graph --
graph = StateGraph(FanOutState)

graph.add_node("decompose", decompose)
graph.add_node("worker", worker)
graph.add_node("merge", merge)

graph.set_entry_point("decompose")
graph.add_conditional_edges("decompose", route_to_workers)
graph.add_edge("worker", "merge")
graph.add_edge("merge", END)

app = graph.compile()

# -- Run --
result = app.invoke({
    "query": "Compare the economies of USA, China, and India on GDP, growth rate, and inflation",
    "sub_tasks": [],
    "results": [],
    "final_answer": ""
})
```

### Merge Strategies

#### 1. Simple Concatenation
Just join results. Fastest but may have inconsistencies.
```python
final = "\n\n".join(results)
```

#### 2. LLM Synthesis (Default)
An LLM reads all results and produces a unified output. Best quality.
```python
final = merger_llm.invoke(f"Synthesize these results: {results}")
```

#### 3. Structured Aggregation
Each worker returns structured data; merge via code.
```python
# Workers return JSON
merged_data = {}
for result in results:
    data = json.loads(result)
    merged_data.update(data)
final = json.dumps(merged_data)
```

#### 4. Voting/Consensus
Multiple workers answer the same question; take majority vote.
```python
from collections import Counter
answers = [worker(same_task) for _ in range(5)]
final = Counter(answers).most_common(1)[0][0]
```

## Cost and Latency Analysis

```
=== Sequential (5 sub-tasks, one at a time) ===
Latency: 5 * 2.0s = 10.0 seconds
LLM calls: 5 (worker) + 1 (merge) = 6
Cost: 5 * $0.01 + $0.015 = $0.065

=== Parallel Fan-Out (5 sub-tasks, all at once) ===
Latency: max(2.0s) + 0.5s (merge) = 2.5 seconds  (4x faster!)
LLM calls: 5 (worker) + 1 (decompose) + 1 (merge) = 7
Cost: 5 * $0.01 + $0.01 + $0.015 = $0.075

Tradeoff: 15% more cost for 4x less latency
```

## Decision Tree: When to Use

```
     Should I use Parallel Fan-Out?
                |
     +----------v----------+
     | Can the task be      |
     | split into truly     |
     | INDEPENDENT parts?   |
     +---+------------+----+
         |            |
        Yes          No --> Use supervisor or sequential pattern
         |
     +---v-----------+----+
     | Is latency more     |
     | important than      |
     | cost?               |
     +---+------------+---+
         |            |
        Yes          No --> Sequential may be fine
         |
     +---v-----------+----+
     | Do all parts use    |
     | the same agent      |
     | type / tools?       |
     +---+------------+---+
         |            |
        Yes          No --> Mixed: supervisor with parallel sub-groups
         |
     +---v-------------------+
     | USE Parallel Fan-Out  |
     +------------------------+
```

## When NOT to Use

1. **Dependent sub-tasks**: If Part B needs Part A's result, you cannot parallelize
2. **Small tasks**: Decomposition + merge overhead exceeds the speedup for <3 sub-tasks
3. **Rate-limited APIs**: Parallel calls may hit API rate limits
4. **Non-decomposable tasks**: Some tasks are inherently sequential (step-by-step reasoning)
5. **Cost-sensitive with many parts**: 10+ parallel agents = 10+ LLM calls simultaneously

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Dramatic latency reduction | Slightly higher total cost (decompose + merge overhead) |
| Scales linearly with sub-tasks | Cannot handle dependent sub-tasks |
| Each worker has small, focused context | Merge step can lose nuance |
| Easy to retry individual failed workers | Decomposition quality affects everything |
| Workers can use different models/tools | Rate limits may throttle parallelism |

## Real-World Examples

1. **Multi-entity analysis**: "Compare 10 companies on revenue, employees, and growth" -- 10 independent lookups.
2. **Document review**: "Review these 5 contracts for risk clauses" -- each contract reviewed independently.
3. **Multi-language translation**: "Translate this document into French, German, and Japanese" -- independent tasks.
4. **Parallel testing**: "Run these test suites: unit, integration, e2e" -- independent execution.
5. **Content generation**: "Write 5 blog post drafts on different aspects of AI" -- independent writing.

## Failure Modes

### 1. Non-Independent Tasks Treated as Independent
Decomposer creates tasks that actually depend on each other, causing inconsistent results.
**Mitigation**: Validate independence in the decomposition step. If tasks reference each other, flag them.

### 2. Inconsistent Results Across Workers
Workers produce contradictory outputs that are hard to merge.
**Mitigation**: Provide consistent context to all workers. Use the merge step to resolve contradictions.

### 3. Slowest Worker Dominates Latency
One worker takes 10x longer than others, negating parallelism benefits.
**Mitigation**: Set per-worker timeouts. Proceed without the slow worker's result if acceptable.

### 4. Merge Quality Degradation
With many results (10+), the merge step struggles to synthesize coherently.
**Mitigation**: Hierarchical merge -- merge in groups of 3-4, then merge the merges.

## Sources and Further Reading

- [LangGraph Map-Reduce Tutorial](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)
- [LangGraph Send API Reference](https://langchain-ai.github.io/langgraph/concepts/low_level/#send)
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents)
