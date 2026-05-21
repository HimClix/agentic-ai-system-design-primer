# Reflexion: Self-Correction Through Verbal Reinforcement

> Reflexion adds a self-evaluation loop where the agent critiques its own output and retries -- achieving 2-3x better accuracy at 2-3x the token cost.

## What It Is

Reflexion (Shinn et al., 2023) introduces a meta-cognitive loop where an agent:
1. **Acts**: Produces an output using a standard pattern (e.g., ReAct)
2. **Evaluates**: A separate evaluation step judges the quality of the output
3. **Reflects**: If the evaluation fails, the agent generates a verbal critique explaining what went wrong
4. **Retries**: The agent tries again with the reflection as additional context

The key insight: instead of gradient-based learning, the agent improves through natural language self-feedback stored in an "episodic memory."

## How It Works

### Three-Component Architecture

```
     +------------------+
     |   User Query     |
     +--------+---------+
              |
     +--------v---------+
     |   ACTOR           |     Standard agent (ReAct, etc.)
     |   Generates       |
     |   response/action |
     +--------+---------+
              |
     +--------v---------+
     |   EVALUATOR       |     Judges output quality
     |   "Is this        |     (can be LLM, code, or human)
     |    correct?"      |
     +--------+---------+
              |
         +----v----+
         | Pass?   |
         +--+---+--+
            |   |
           Yes  No
            |   |
     +------v-+ +--v-----------+
     | Return | | REFLECTOR    |
     | answer | | "What went   |
     +--------+ |  wrong and   |
                |  how to fix?"|
                +------+-------+
                       |
              +--------v---------+
              | EPISODIC MEMORY  |
              | Store reflection |
              | for retry        |
              +--------+---------+
                       |
                       +------> (back to ACTOR with reflection context)
```

### Step-by-Step Example

```
User: "Write a Python function that finds the longest palindromic substring."

=== Attempt 1 ===
ACTOR: Generates a brute-force O(n^3) solution

EVALUATOR: Runs test cases
  - "racecar" -> "racecar" PASS
  - "babad"   -> "bab"     PASS
  - "a"       -> "a"       PASS
  - Performance test: n=10000 -> TIMEOUT (>5 seconds)
  Result: FAIL (performance requirement not met)

REFLECTOR: "The solution uses O(n^3) brute force. It works
  correctly but is too slow for large inputs. I should use
  the expand-around-center approach for O(n^2) or Manacher's
  algorithm for O(n). The expand-around-center approach is
  simpler and sufficient."

=== Attempt 2 (with reflection context) ===
ACTOR: Generates expand-around-center O(n^2) solution

EVALUATOR: Runs test cases
  - All correctness tests: PASS
  - Performance test: n=10000 -> 0.3 seconds PASS
  Result: PASS

Return: O(n^2) expand-around-center solution
```

## Production Implementation

```python
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# -- State --
class ReflexionState(TypedDict):
    query: str
    current_answer: Optional[str]
    evaluation_result: Optional[str]
    evaluation_passed: bool
    reflections: List[str]         # Episodic memory
    attempt_count: int
    max_attempts: int

# -- Models --
# Use the strongest model for reflection, cheaper for acting
actor_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
evaluator_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
reflector_llm = ChatAnthropic(model="claude-opus-4-20250514", temperature=0)

# -- Actor Node --
def actor_node(state: ReflexionState) -> dict:
    """Generate a response, incorporating past reflections."""
    reflection_context = ""
    if state["reflections"]:
        reflection_context = "\n\nPrevious attempts and reflections:\n"
        for i, ref in enumerate(state["reflections"]):
            reflection_context += f"\nAttempt {i+1} reflection: {ref}"

    response = actor_llm.invoke([
        SystemMessage(content=f"""Solve the following task.
        {reflection_context}
        Use the reflections to avoid repeating mistakes."""),
        HumanMessage(content=state["query"])
    ])

    return {
        "current_answer": response.content,
        "attempt_count": state["attempt_count"] + 1
    }

# -- Evaluator Node --
def evaluator_node(state: ReflexionState) -> dict:
    """Evaluate the quality of the current answer."""
    response = evaluator_llm.invoke([
        SystemMessage(content="""You are a strict evaluator. Assess whether
        the response correctly and completely answers the query.

        Respond with:
        PASS: <reason> -- if the answer is correct and complete
        FAIL: <reason> -- if the answer has errors, is incomplete, or could be improved

        Be specific about what's wrong."""),
        HumanMessage(content=f"""
        Query: {state['query']}
        Response: {state['current_answer']}
        """)
    ])

    passed = response.content.strip().startswith("PASS")
    return {
        "evaluation_result": response.content,
        "evaluation_passed": passed
    }

# -- Reflector Node --
def reflector_node(state: ReflexionState) -> dict:
    """Generate a reflection on what went wrong."""
    response = reflector_llm.invoke([
        SystemMessage(content="""You are a reflective analyst. Given a failed attempt,
        produce a concrete, actionable reflection that explains:
        1. What specifically went wrong
        2. Why it went wrong
        3. What strategy should be used instead

        Be specific and concise. This reflection will guide the next attempt."""),
        HumanMessage(content=f"""
        Query: {state['query']}
        Attempted answer: {state['current_answer']}
        Evaluation: {state['evaluation_result']}
        """)
    ])

    new_reflections = state["reflections"] + [response.content]
    return {"reflections": new_reflections}

# -- Routing --
def after_evaluation(state: ReflexionState) -> str:
    if state["evaluation_passed"]:
        return "end"
    if state["attempt_count"] >= state["max_attempts"]:
        return "end"  # Return best effort
    return "reflect"

# -- Build Graph --
graph = StateGraph(ReflexionState)
graph.add_node("act", actor_node)
graph.add_node("evaluate", evaluator_node)
graph.add_node("reflect", reflector_node)

graph.set_entry_point("act")
graph.add_edge("act", "evaluate")
graph.add_conditional_edges("evaluate", after_evaluation, {
    "reflect": "reflect",
    "end": END
})
graph.add_edge("reflect", "act")

app = graph.compile()

# -- Run --
result = app.invoke({
    "query": "Write a Python function for longest palindromic substring",
    "current_answer": None,
    "evaluation_result": None,
    "evaluation_passed": False,
    "reflections": [],
    "attempt_count": 0,
    "max_attempts": 3  # 2-3 retries max
})
```

## Why 2-3 Retry Max?

Empirical findings from the Reflexion paper and production systems:

```
Attempt  | Success Rate | Marginal Gain | Cumulative Cost
---------|-------------|---------------|----------------
1        | 62%         | --            | 1x
2        | 81%         | +19%          | 2.1x
3        | 88%         | +7%           | 3.2x
4        | 89%         | +1%           | 4.3x   <-- diminishing returns
5        | 89.5%       | +0.5%         | 5.4x   <-- waste
```

After 3 attempts:
- **Diminishing returns**: Each retry adds less improvement
- **Cost multiplication**: Each retry costs the full generation + evaluation + reflection
- **Reflection quality degrades**: The LLM starts repeating the same reflections
- **If 3 tries didn't fix it**: The problem is likely beyond the model's capability

### Using Different Models for Evaluation

A critical production pattern: use a **different (often stronger) model** for evaluation than for acting.

```python
# Actor: cheaper/faster model
actor_llm = ChatAnthropic(model="claude-haiku-4-20250514")

# Evaluator: stronger model that can catch subtle errors
evaluator_llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# Reflector: strongest model for deep analysis
reflector_llm = ChatAnthropic(model="claude-opus-4-20250514")
```

Why? The evaluator needs to be *at least as capable* as the actor to catch its mistakes. Using the same model risks the evaluator having the same blind spots.

Alternatively, use **code-based evaluation** when possible:
```python
def code_evaluator(generated_code: str, test_cases: list) -> dict:
    """Deterministic evaluation -- cheaper and more reliable than LLM eval."""
    results = []
    for test in test_cases:
        try:
            output = exec_sandbox(generated_code, test["input"])
            results.append(output == test["expected"])
        except Exception as e:
            results.append(False)

    return {
        "passed": all(results),
        "details": results
    }
```

## Token Cost Analysis

```
=== Reflexion (2 attempts) vs Single ReAct ===

Single ReAct (no retry):
  5 iterations * avg 2000 tokens = 10,000 input tokens
  Total: ~$0.06

Reflexion (2 attempts):
  Attempt 1 - Actor:     ~$0.06 (same as ReAct)
  Attempt 1 - Evaluator: ~$0.01
  Attempt 1 - Reflector: ~$0.02
  Attempt 2 - Actor:     ~$0.07 (slightly more context with reflection)
  Attempt 2 - Evaluator: ~$0.01
  Total: ~$0.17

OVERHEAD: 2.8x cost for ~20% accuracy improvement
```

## Decision Tree: When to Use

```
     Should I use Reflexion?
                |
     +----------v----------+
     | Does the task have   |
     | a clear success      |
     | criterion?           |
     | (tests, rubric,      |
     |  ground truth)       |
     +---+------------+----+
         |            |
        Yes          No --> Skip (can't evaluate = can't reflect)
         |
     +---v-----------+----+
     | Is the 2-3x token  |
     | overhead acceptable?|
     +---+------------+---+
         |            |
        Yes          No --> Use single-pass with better prompting
         |
     +---v-----------+----+
     | Is first-pass      |
     | accuracy below     |
     | 90%?               |
     +---+------------+---+
         |            |
        Yes          No --> First pass is good enough
         |
     +---v-------------------+
     | USE Reflexion         |
     | (2-3 max retries)     |
     +------------------------+
```

## When NOT to Use

1. **No evaluation criteria**: Without a way to judge quality, reflection is meaningless
2. **Cost-sensitive applications**: 2-3x overhead is prohibitive at scale
3. **Latency-critical paths**: Each retry adds full round-trip time
4. **Simple factual questions**: Either the LLM knows it or it doesn't -- retrying won't help
5. **Tasks where first answer is usually correct**: >90% first-pass accuracy means retries are waste

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Significant accuracy improvement (62% -> 88%) | 2-3x token cost overhead |
| Self-correcting without human intervention | Adds significant latency per retry |
| Explicit error analysis aids debugging | Requires clear evaluation criteria |
| Works with any base agent pattern | Reflection quality degrades after 2-3 attempts |
| Natural language "learning" without fine-tuning | Same model may have same blind spots |

## Real-World Examples

1. **Code generation**: Generate code, run tests, reflect on failures, regenerate. Most natural fit.
2. **SQL query generation**: Generate query, validate against schema, reflect on errors, fix.
3. **Mathematical reasoning**: Solve problem, verify answer, reflect on logical errors, retry.
4. **Document summarization**: Summarize, check for factual accuracy against source, improve.
5. **Legal contract analysis**: Analyze contract, evaluate for missed clauses, reflect and re-analyze.

## Failure Modes

### 1. Evaluator-Actor Capability Mismatch
When the evaluator is the same model and quality as the actor, it may approve incorrect answers because it has the same knowledge gaps.
**Mitigation**: Use a stronger model for evaluation, or code-based evaluation.

### 2. Reflection Loops
The reflection says "try a different approach" but the actor generates the same thing.
**Mitigation**: Include previous attempts in the reflection prompt so the actor knows what NOT to do.

### 3. Over-Reflection
The agent keeps finding minor issues and never declares success.
**Mitigation**: Hard cap at 2-3 attempts. Accept "good enough."

### 4. Evaluation Hallucination
The evaluator claims the answer is wrong when it's actually correct.
**Mitigation**: Use deterministic evaluation (code tests, regex checks) when possible.

### 5. Cost Explosion on Batch
Running Reflexion on 1,000 items with 3 retries each = 3,000 LLM calls instead of 1,000.
**Mitigation**: Use Reflexion selectively -- only on items that fail initial evaluation.

## Sources and Further Reading

- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) (Shinn et al., NeurIPS 2023)
- [LangGraph Reflexion Tutorial](https://langchain-ai.github.io/langgraph/tutorials/reflexion/reflexion/)
- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) (Madaan et al., 2023)
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents)
