# Evaluation Overview

> Eval is the hardest skill to fake in AI engineering -- you can copy prompts and architectures, but building a reliable eval pipeline that catches regressions before users do requires deep understanding of what "correct" means for your system.

## What It Is

Evaluation (eval) is the systematic measurement of AI agent quality across accuracy, safety, cost, and latency dimensions. For agentic systems, eval is uniquely challenging because agents make **multi-step decisions** with **non-deterministic outputs** -- the same input can produce different but equally valid outputs, and correctness depends on the entire reasoning trajectory, not just the final answer.

The EDDOps framework (arXiv:2411.13768) formalizes evaluation-driven development as the core practice for production LLM systems.

## How It Works

### Offline vs Online Evaluation

| Dimension | Offline Eval | Online Eval |
|-----------|-------------|-------------|
| **When** | Before deployment (CI/CD) | After deployment (production) |
| **Data** | Curated test datasets | Real user traffic |
| **Purpose** | Catch regressions before release | Monitor quality in production |
| **Cost** | Fixed (test dataset size) | Variable (sampling rate) |
| **Latency impact** | None (runs in CI) | None (async, sampled) |
| **Coverage** | Known scenarios | Unknown/novel scenarios |
| **Examples** | pytest eval suite, RAGAS benchmark | LLM-as-judge on 5% of traces |

### Trajectory Eval vs Final-Answer Eval

| Approach | What It Evaluates | Catches | Misses |
|----------|------------------|---------|--------|
| **Final-answer eval** | Only the output | Wrong answers, format errors | Wrong reasoning that leads to correct answer |
| **Trajectory eval** | Every step in the reasoning chain | Wrong tool calls, unnecessary steps, reasoning errors | Nothing (but more expensive) |

**Why trajectory eval matters:**
```
Scenario: Agent arrives at correct answer via wrong reasoning

Step 1: User asks "What's the refund policy for order #789?"
Step 2: Agent calls get_user_profile() instead of get_order()     ← WRONG TOOL
Step 3: Agent calls get_order() after profile returns nothing     ← WASTED STEP
Step 4: Agent reads refund policy from response                   ← CORRECT
Step 5: Agent gives correct answer                                ← CORRECT OUTPUT

Final-answer eval: PASS ✓ (answer is correct)
Trajectory eval:   FAIL ✗ (step 2 was wrong tool, step 3 was recovery from error)
                         This agent will fail on harder queries.
```

### Decision Tree: Which Eval Approach

```
What are you evaluating?

RAG pipeline (retrieval + generation)?
  └── RAGAS (faithfulness, context precision, context recall, answer relevancy)

Agent tool-calling sequences?
  └── Trajectory eval (score every step, not just final answer)

LLM output quality (helpfulness, safety, tone)?
  └── LLM-as-judge (stronger model scores weaker model)

Prompt changes?
  └── CI eval gate (run eval suite, block merge if scores drop)

Safety/security?
  └── Red-teaming (adversarial testing before launch)

All of the above?
  └── Yes. You need all of them. Start with LLM-as-judge + CI gate.
```

### The Eval Pyramid

```
                    ┌──────────┐
                    │ Red-team │  ← Pre-launch, quarterly
                    │  testing │     adversarial probing
                    ├──────────┤
                    │ Online   │  ← Production, 5% sampling
                    │ eval     │     LLM-as-judge scoring
                    ├──────────┤
                    │ CI eval  │  ← Every PR that changes prompts/agents
                    │ gates    │     blocks merge on regression
                    ├──────────┤
                    │ Offline  │  ← During development
                    │ eval     │     RAGAS, trajectory eval
                    │ suite    │     curated test datasets
                    └──────────┘
```

## Production Implementation

```python
"""
Evaluation pipeline architecture -- composable eval framework.
"""
from dataclasses import dataclass
from typing import Any, Optional, Callable
from enum import Enum


class EvalType(Enum):
    FINAL_ANSWER = "final_answer"
    TRAJECTORY = "trajectory"
    RAG = "rag"
    SAFETY = "safety"


@dataclass
class EvalResult:
    eval_type: EvalType
    score: float        # 0.0 to 1.0
    passed: bool        # score >= threshold
    threshold: float
    details: dict
    evaluator: str      # Which evaluator produced this


class EvalPipeline:
    """
    Composable evaluation pipeline.
    Add evaluators and run them against agent outputs.
    """
    
    def __init__(self, threshold: float = 0.7):
        self.evaluators: list[tuple[str, Callable, float]] = []
        self.default_threshold = threshold
    
    def add_evaluator(
        self,
        name: str,
        eval_fn: Callable,
        threshold: Optional[float] = None,
    ):
        self.evaluators.append((name, eval_fn, threshold or self.default_threshold))
    
    async def evaluate(
        self,
        input_data: dict,
        output_data: dict,
        context: Optional[dict] = None,
    ) -> list[EvalResult]:
        results = []
        
        for name, eval_fn, threshold in self.evaluators:
            score = await eval_fn(input_data, output_data, context)
            results.append(EvalResult(
                eval_type=EvalType.FINAL_ANSWER,
                score=score,
                passed=score >= threshold,
                threshold=threshold,
                details={"input": str(input_data)[:200]},
                evaluator=name,
            ))
        
        return results
    
    def all_passed(self, results: list[EvalResult]) -> bool:
        return all(r.passed for r in results)


# Usage
pipeline = EvalPipeline(threshold=0.7)

async def helpfulness_evaluator(inp, out, ctx):
    # LLM-as-judge implementation
    return 0.85

async def safety_evaluator(inp, out, ctx):
    # Safety check implementation
    return 0.95

pipeline.add_evaluator("helpfulness", helpfulness_evaluator, threshold=0.7)
pipeline.add_evaluator("safety", safety_evaluator, threshold=0.9)
```

## Decision Tree / When to Use

- **Every production agent needs eval** -- the question is which types and when
- **Start with**: LLM-as-judge on 5% of production traces (cheapest, most value)
- **Add next**: CI eval gate (prevent regressions on prompt changes)
- **Add next**: RAGAS (if you have a RAG pipeline)
- **Add next**: Trajectory eval (if agent has complex multi-step workflows)
- **Quarterly**: Red-teaming (adversarial security testing)

## When NOT to Use

- **Never skip evaluation** for production agents
- Simple chatbots can use lighter eval (LLM-as-judge only)
- Trajectory eval is overkill for single-step agents

## Tradeoffs

| Eval Type | Cost Per Eval | Coverage | False Positive Rate | Implementation Effort |
|-----------|--------------|----------|--------------------|-----------------------|
| LLM-as-judge | ~$0.01 | Broad | Medium (~15%) | Low |
| RAGAS | ~$0.02 | RAG-specific | Low (~5%) | Medium |
| Trajectory eval | ~$0.05 | Deep (multi-step) | Low (~5%) | High |
| CI eval gate | ~$0.50 per PR | Regression-focused | Low | Medium |
| Red-teaming | $500-5000 per session | Security/safety | Very low | High |

## Real-World Examples

1. **EDDOps (arXiv:2411.13768)**: Formalizes evaluation as the central practice in LLM development -- eval datasets drive development, not the other way around.
2. **Anthropic**: Uses constitutional AI as a form of automated evaluation, scoring model outputs against principles.
3. **OpenAI**: Extensive eval suite (openai/evals) with thousands of test cases, including adversarial examples.

## Failure Modes

| Failure | Cause | Impact | Prevention |
|---------|-------|--------|------------|
| **Eval gaming** | Agent optimized to pass eval, not serve users | High eval scores, low user satisfaction | Diverse eval datasets, user feedback loop |
| **Stale eval data** | Test cases don't match current user patterns | Regressions in new scenarios not caught | Regularly add production examples to eval set |
| **LLM judge bias** | Judge model has systematic preferences | Skewed scores | Use multiple judges, calibrate regularly |
| **Eval too slow** | Large eval suite blocks CI | Developer friction, eval bypassed | Prioritize eval cases, run in parallel |

## Source(s) and Further Reading

- EDDOps: Eval-Driven Development for LLM Systems (arXiv:2411.13768)
- RAGAS Documentation: https://docs.ragas.io/
- LangSmith Evaluation: https://docs.smith.langchain.com/evaluation
- OpenAI Evals: https://github.com/openai/evals
- Anthropic, "Evaluating Language Model Performance" (2024)
- Breck et al., "The ML Test Score: A Rubric for ML Production Readiness" (Google, 2017)
