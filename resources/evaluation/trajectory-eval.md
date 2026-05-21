# Trajectory Evaluation

> A correct final answer does not mean the agent did the right thing -- trajectory eval scores every step, catching wrong tool calls at step 2 that happen to produce a correct-looking final answer.

## What It Is

Trajectory evaluation scores **every step** in an agent's execution path, not just the final output. It catches a critical class of bugs that final-answer eval misses: an agent that arrives at the right answer through the wrong reasoning. These agents will fail on harder queries because their reasoning process is flawed even when the output appears correct.

## How It Works

### What Final-Answer Eval Misses

```
Agent Task: "What's the shipping status for order #789?"

TRAJECTORY:
  Step 1: Agent calls get_customer_profile(customer_id="C-123")     ← WRONG TOOL
  Step 2: Profile returns no shipping info                          ← EXPECTED
  Step 3: Agent calls get_order(order_id="789")                     ← CORRECT (recovery)
  Step 4: Order shows "shipped, arriving Thursday"                  ← CORRECT
  Step 5: Agent responds "Your order is shipped, arriving Thursday" ← CORRECT OUTPUT

FINAL-ANSWER EVAL:  PASS ✓ (answer matches expected)
TRAJECTORY EVAL:    FAIL ✗ 
  - Step 1 called wrong tool (get_customer_profile instead of get_order)
  - Step 2 was wasted computation (unnecessary API call)
  - Agent required error recovery (fragile reasoning)
```

### What Trajectory Eval Catches

| Issue | Final-Answer Catches | Trajectory Catches |
|-------|---------------------|-------------------|
| Wrong final answer | Yes | Yes |
| Wrong tool at step N | No | Yes |
| Unnecessary steps | No | Yes |
| Wrong reasoning, right answer | No | Yes |
| Correct but slow path | No | Yes |
| Tool called with wrong args | No (if result still usable) | Yes |
| Missing safety checks | No (if answer is safe) | Yes |

### Trajectory Scoring Rubric

Score each step on:

1. **Tool Selection** (0-1): Did the agent call the right tool?
2. **Argument Correctness** (0-1): Were the arguments correct?
3. **Necessity** (0-1): Was this step necessary? (Penalize redundant steps)
4. **Reasoning Quality** (0-1): Was the reasoning for this step sound?
5. **Error Handling** (0-1): Did the agent handle errors appropriately?

**Trajectory Score** = Average of all step scores

## Production Implementation

```python
"""
Trajectory evaluation for multi-step agent execution.
"""
from dataclasses import dataclass, field
from typing import Any, Optional
import json


@dataclass
class TrajectoryStep:
    """One step in the agent's execution trajectory."""
    step_number: int
    action_type: str         # "llm_call", "tool_call", "decision"
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_result: Optional[Any] = None
    reasoning: str = ""       # Agent's stated reasoning for this step
    success: bool = True


@dataclass
class StepScore:
    """Score for a single trajectory step."""
    step_number: int
    tool_selection: float     # 0-1: right tool?
    argument_correctness: float  # 0-1: right args?
    necessity: float          # 0-1: was this step needed?
    reasoning_quality: float  # 0-1: sound reasoning?
    error_handling: float     # 0-1: errors handled well?
    
    @property
    def average(self) -> float:
        scores = [
            self.tool_selection,
            self.argument_correctness,
            self.necessity,
            self.reasoning_quality,
            self.error_handling,
        ]
        return sum(scores) / len(scores)
    
    @property
    def issues(self) -> list[str]:
        issues = []
        if self.tool_selection < 0.5:
            issues.append("wrong_tool")
        if self.argument_correctness < 0.5:
            issues.append("wrong_args")
        if self.necessity < 0.5:
            issues.append("unnecessary_step")
        if self.reasoning_quality < 0.5:
            issues.append("poor_reasoning")
        if self.error_handling < 0.5:
            issues.append("poor_error_handling")
        return issues


@dataclass 
class TrajectoryScore:
    """Complete trajectory evaluation result."""
    step_scores: list[StepScore]
    final_answer_correct: bool
    
    @property
    def trajectory_score(self) -> float:
        if not self.step_scores:
            return 0.0
        return sum(s.average for s in self.step_scores) / len(self.step_scores)
    
    @property
    def all_issues(self) -> list[str]:
        issues = []
        for s in self.step_scores:
            for issue in s.issues:
                issues.append(f"step_{s.step_number}:{issue}")
        return issues
    
    @property
    def efficiency(self) -> float:
        """What fraction of steps were necessary?"""
        if not self.step_scores:
            return 1.0
        necessary = sum(1 for s in self.step_scores if s.necessity >= 0.5)
        return necessary / len(self.step_scores)


class TrajectoryEvaluator:
    """
    Evaluates agent trajectories using LLM-as-judge.
    """
    
    EVAL_PROMPT = """You are evaluating an AI agent's execution trajectory step by step.

## Task Given to Agent:
{task}

## Expected Ideal Trajectory:
{ideal_trajectory}

## Actual Agent Trajectory:
{actual_trajectory}

## Evaluation Instructions:
For EACH step in the actual trajectory, score the following on a 0-1 scale:

1. **tool_selection** (0-1): Did the agent call the right tool for this step?
   - 1.0: Perfect tool choice
   - 0.5: Acceptable but suboptimal tool
   - 0.0: Completely wrong tool

2. **argument_correctness** (0-1): Were the arguments to the tool correct?
   - 1.0: All arguments correct
   - 0.5: Some arguments correct
   - 0.0: Arguments wrong or missing

3. **necessity** (0-1): Was this step necessary?
   - 1.0: Essential step toward the goal
   - 0.5: Helpful but not required
   - 0.0: Completely unnecessary (wasted computation)

4. **reasoning_quality** (0-1): Was the agent's reasoning for this step sound?
   - 1.0: Clear, logical reasoning
   - 0.5: Reasoning is acceptable but could be better
   - 0.0: Reasoning is flawed or missing

5. **error_handling** (0-1): If an error occurred, was it handled appropriately?
   - 1.0: Error handled gracefully (or no error occurred)
   - 0.5: Error somewhat handled
   - 0.0: Error ignored or handled poorly

Return a JSON array of step evaluations:
[
  {{"step": 1, "tool_selection": 0.9, "argument_correctness": 1.0, "necessity": 1.0, "reasoning_quality": 0.8, "error_handling": 1.0}},
  ...
]

Return ONLY the JSON array, no other text."""
    
    def __init__(self, judge_model: str = "gpt-4o"):
        self.judge_model = judge_model
    
    async def evaluate(
        self,
        task: str,
        actual_trajectory: list[TrajectoryStep],
        ideal_trajectory: Optional[list[str]] = None,
        final_answer_correct: bool = True,
    ) -> TrajectoryScore:
        """Evaluate a complete agent trajectory."""
        import openai
        
        # Format trajectories for the judge
        actual_str = self._format_trajectory(actual_trajectory)
        ideal_str = "\n".join(ideal_trajectory) if ideal_trajectory else "Not provided -- evaluate based on task requirements."
        
        prompt = self.EVAL_PROMPT.format(
            task=task,
            ideal_trajectory=ideal_str,
            actual_trajectory=actual_str,
        )
        
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        
        # Parse judge response
        judge_output = json.loads(response.choices[0].message.content)
        
        # Handle both direct array and wrapped object
        if isinstance(judge_output, list):
            step_evals = judge_output
        else:
            step_evals = judge_output.get("steps", judge_output.get("evaluations", []))
        
        step_scores = []
        for eval_data in step_evals:
            step_scores.append(StepScore(
                step_number=eval_data.get("step", len(step_scores) + 1),
                tool_selection=eval_data.get("tool_selection", 0.5),
                argument_correctness=eval_data.get("argument_correctness", 0.5),
                necessity=eval_data.get("necessity", 0.5),
                reasoning_quality=eval_data.get("reasoning_quality", 0.5),
                error_handling=eval_data.get("error_handling", 1.0),
            ))
        
        return TrajectoryScore(
            step_scores=step_scores,
            final_answer_correct=final_answer_correct,
        )
    
    def _format_trajectory(self, trajectory: list[TrajectoryStep]) -> str:
        lines = []
        for step in trajectory:
            line = f"Step {step.step_number}: [{step.action_type}]"
            if step.tool_name:
                line += f" Tool: {step.tool_name}"
            if step.tool_args:
                line += f" Args: {json.dumps(step.tool_args)}"
            if step.reasoning:
                line += f"\n  Reasoning: {step.reasoning}"
            if step.tool_result:
                result_str = str(step.tool_result)[:200]
                line += f"\n  Result: {result_str}"
            if not step.success:
                line += f"\n  ERROR: Step failed"
            lines.append(line)
        return "\n\n".join(lines)


# ============================================================
# Usage with test dataset
# ============================================================

async def run_trajectory_eval():
    evaluator = TrajectoryEvaluator()
    
    # Define test case
    task = "Find the shipping status for order #789"
    
    actual_steps = [
        TrajectoryStep(
            step_number=1,
            action_type="tool_call",
            tool_name="get_customer_profile",
            tool_args={"customer_id": "C-123"},
            reasoning="Let me look up the customer first",
        ),
        TrajectoryStep(
            step_number=2,
            action_type="tool_call",
            tool_name="get_order",
            tool_args={"order_id": "789"},
            reasoning="Profile didn't have shipping info, let me check the order directly",
        ),
    ]
    
    ideal_trajectory = [
        "Step 1: Call get_order(order_id='789') to get order status directly",
    ]
    
    result = await evaluator.evaluate(
        task=task,
        actual_trajectory=actual_steps,
        ideal_trajectory=ideal_trajectory,
    )
    
    print(f"Trajectory Score: {result.trajectory_score:.2f}")
    print(f"Efficiency: {result.efficiency:.2f}")
    print(f"Issues: {result.all_issues}")
```

## Decision Tree / When to Use

```
Does your agent make MULTI-STEP decisions?
  YES --> Trajectory eval is essential
  
Does your agent call MULTIPLE TOOLS?
  YES --> Trajectory eval catches wrong tool selection
  
Is final-answer eval showing high scores but users are unhappy?
  YES --> Trajectory eval will reveal hidden reasoning problems
  
Is agent behavior non-deterministic (different paths for same query)?
  YES --> Trajectory eval ensures all paths are valid
```

## When NOT to Use

- **Single-step agents** (one LLM call, no tools) -- final-answer eval is sufficient
- **Simple QA without tools** -- RAGAS or LLM-as-judge is enough
- **Early prototyping** -- Invest in trajectory eval when the agent architecture stabilizes

## Tradeoffs

| Approach | Cost | What It Catches | Complexity |
|----------|------|-----------------|------------|
| Final-answer only | ~$0.01 per eval | Wrong answers | Low |
| Trajectory eval | ~$0.05 per eval | Wrong reasoning, wrong tools, inefficiency | High |
| Trajectory + final-answer | ~$0.06 per eval | Everything | High |

## Failure Modes

| Failure | Cause | Impact | Prevention |
|---------|-------|--------|------------|
| **Ideal trajectory too rigid** | Only one "correct" path defined | Valid alternative paths penalized | Define multiple acceptable trajectories |
| **Step count variance** | Different valid approaches have different step counts | Efficiency metric misleading | Normalize by task complexity |
| **Judge inconsistency** | LLM judge scores differently across runs | Flaky tests | Low temperature + multiple runs averaged |

## Source(s) and Further Reading

- EDDOps (arXiv:2411.13768): Evaluation-driven development advocating trajectory-level evaluation
- LangSmith Trajectory Evaluation: https://docs.smith.langchain.com/evaluation
- "Agent Trajectory Evaluation" patterns in LangGraph documentation
- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2023) -- foundational work on agent reasoning traces
