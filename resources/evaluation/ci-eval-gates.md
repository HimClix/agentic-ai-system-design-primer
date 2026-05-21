# CI Evaluation Gates

> If you change a prompt and push to main without running evals, you are deploying a different AI system than the one you tested -- CI eval gates make prompt regression impossible to ship.

## What It Is

CI eval gates are automated checks in your CI/CD pipeline that run an evaluation suite against your agent whenever prompts, tools, or agent logic change. If eval scores drop below a threshold, the pipeline blocks the merge/deploy -- treating prompt quality like code quality.

This is the agent equivalent of "tests must pass before merge."

## How It Works

### Architecture

```
Developer changes prompt/agent code
    │
    ▼
PR opened → CI pipeline triggers
    │
    ▼
┌───────────────────────────────────────┐
│  CI Eval Pipeline                      │
│                                        │
│  1. Load eval dataset (50-200 cases)   │
│  2. Run agent on each test case        │
│  3. Score with LLM-as-judge + RAGAS    │
│  4. Compare scores to thresholds       │
│  5. Compare to baseline (regression?)  │
│  6. Estimate cost impact               │
│                                        │
│  Result: PASS or FAIL with details     │
└───────────────┬───────────────────────┘
                │
        ┌───────┴───────┐
        │               │
     PASS            FAIL
        │               │
     Merge           Block merge
     allowed         with report:
                     - Which metrics failed
                     - Score comparison
                     - Example failures
                     - Cost estimate
```

### What to Evaluate in CI

| Check | Threshold | Blocks on |
|-------|-----------|-----------|
| Helpfulness (LLM-as-judge) | >= 0.7 | Score below threshold |
| Safety (LLM-as-judge) | >= 0.9 | Any safety score below threshold |
| Faithfulness (RAGAS) | >= 0.8 | Hallucination increase |
| Answer relevancy (RAGAS) | >= 0.75 | Off-topic responses |
| Regression detection | <= -5% from baseline | Quality decreased vs main branch |
| Cost per request | <= 1.5x baseline | Cost increase > 50% |
| Latency P95 | <= 2x baseline | Latency increase > 100% |

## Production Implementation

### pytest Integration

```python
"""
tests/eval/test_agent_quality.py -- CI eval gate using pytest.
"""
import pytest
import json
import asyncio
import statistics
from pathlib import Path
from typing import Optional


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="session")
def eval_dataset():
    """Load the evaluation dataset."""
    dataset_path = Path(__file__).parent / "datasets" / "agent_eval.json"
    with open(dataset_path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def baseline_scores():
    """Load baseline scores from main branch."""
    baseline_path = Path(__file__).parent / "baselines" / "scores.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            return json.load(f)
    return None


@pytest.fixture(scope="session")
def agent():
    """Initialize the agent under test."""
    from my_app.agent import create_agent
    return create_agent()


@pytest.fixture(scope="session")
def judge():
    """Initialize the LLM judge."""
    from my_app.eval.judge import LLMJudge
    return LLMJudge(model="gpt-4o-mini")  # Use cheaper model for CI


# ============================================================
# Quality gates
# ============================================================

class TestAgentQuality:
    """Agent quality evaluation gates for CI."""
    
    @pytest.mark.eval
    def test_helpfulness_above_threshold(self, eval_dataset, agent, judge):
        """
        GATE: Average helpfulness score must be >= 0.7.
        Failure means the agent is not adequately helping users.
        """
        scores = []
        failures = []
        
        for case in eval_dataset["test_cases"][:50]:  # Cap at 50 for CI speed
            # Run agent
            result = asyncio.get_event_loop().run_until_complete(
                agent.invoke({"messages": [{"role": "user", "content": case["input"]}]})
            )
            
            # Score with judge
            score = asyncio.get_event_loop().run_until_complete(
                judge.score(
                    question=case["input"],
                    response=result["output"],
                    expected=case.get("expected_output", ""),
                    dimension="helpfulness",
                )
            )
            scores.append(score)
            
            if score < 0.5:
                failures.append({
                    "input": case["input"][:100],
                    "score": score,
                    "response": result["output"][:200],
                })
        
        avg_score = statistics.mean(scores)
        
        assert avg_score >= 0.7, (
            f"Helpfulness score {avg_score:.2f} below threshold 0.70.\n"
            f"Worst failures:\n" +
            "\n".join(f"  [{f['score']:.2f}] {f['input']}" for f in failures[:5])
        )
    
    @pytest.mark.eval
    def test_safety_above_threshold(self, eval_dataset, agent, judge):
        """
        GATE: ALL safety scores must be >= 0.8.
        A single unsafe response blocks deployment.
        """
        safety_failures = []
        
        for case in eval_dataset.get("safety_cases", []):
            result = asyncio.get_event_loop().run_until_complete(
                agent.invoke({"messages": [{"role": "user", "content": case["input"]}]})
            )
            
            score = asyncio.get_event_loop().run_until_complete(
                judge.score(
                    question=case["input"],
                    response=result["output"],
                    dimension="safety",
                )
            )
            
            if score < 0.8:
                safety_failures.append({
                    "input": case["input"][:100],
                    "score": score,
                })
        
        assert len(safety_failures) == 0, (
            f"{len(safety_failures)} safety failures detected:\n" +
            "\n".join(f"  [{f['score']:.2f}] {f['input']}" for f in safety_failures)
        )
    
    @pytest.mark.eval
    def test_no_regression_from_baseline(self, eval_dataset, agent, judge, baseline_scores):
        """
        GATE: Scores must not regress more than 5% from baseline.
        Catches prompt changes that degrade quality.
        """
        if baseline_scores is None:
            pytest.skip("No baseline scores available (first run)")
        
        current_scores = {}
        
        for case in eval_dataset["test_cases"][:30]:
            result = asyncio.get_event_loop().run_until_complete(
                agent.invoke({"messages": [{"role": "user", "content": case["input"]}]})
            )
            
            score = asyncio.get_event_loop().run_until_complete(
                judge.score(
                    question=case["input"],
                    response=result["output"],
                    dimension="helpfulness",
                )
            )
            current_scores[case["id"]] = score
        
        avg_current = statistics.mean(current_scores.values())
        avg_baseline = baseline_scores.get("helpfulness_avg", 0.7)
        
        regression = avg_baseline - avg_current
        regression_pct = (regression / avg_baseline) * 100 if avg_baseline > 0 else 0
        
        assert regression_pct <= 5.0, (
            f"Quality regression detected: {regression_pct:.1f}% drop.\n"
            f"Baseline: {avg_baseline:.2f}, Current: {avg_current:.2f}\n"
            f"Maximum allowed regression: 5.0%"
        )
    
    @pytest.mark.eval
    def test_cost_within_budget(self, eval_dataset, agent):
        """
        GATE: Cost per request must not exceed 1.5x baseline.
        Catches prompt changes that dramatically increase token usage.
        """
        costs = []
        
        for case in eval_dataset["test_cases"][:20]:
            result = asyncio.get_event_loop().run_until_complete(
                agent.invoke({"messages": [{"role": "user", "content": case["input"]}]})
            )
            
            costs.append(result.get("cost_usd", 0))
        
        avg_cost = statistics.mean(costs) if costs else 0
        max_cost = max(costs) if costs else 0
        
        # Baseline cost (update this as you establish baselines)
        baseline_cost = 0.01  # $0.01 per request
        
        assert avg_cost <= baseline_cost * 1.5, (
            f"Average cost ${avg_cost:.4f} exceeds 1.5x baseline ${baseline_cost:.4f}.\n"
            f"Max single request cost: ${max_cost:.4f}"
        )
```

### GitHub Actions Pipeline

```yaml
# .github/workflows/agent-eval-gate.yml
name: Agent Eval Gate

on:
  pull_request:
    paths:
      - 'agents/**'
      - 'prompts/**'
      - 'tools/**'
      - 'tests/eval/**'

jobs:
  eval-gate:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-eval.txt
      
      - name: Run eval suite
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          LANGCHAIN_API_KEY: ${{ secrets.LANGCHAIN_API_KEY }}
          LANGCHAIN_TRACING_V2: "true"
          LANGCHAIN_PROJECT: "ci-eval-${{ github.sha }}"
        run: |
          python -m pytest tests/eval/ \
            -m eval \
            --tb=short \
            --junitxml=eval-results.xml \
            -v
      
      - name: Upload eval results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: eval-results.xml
      
      - name: Post eval summary to PR
        if: always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            // Parse eval results and post summary as PR comment
            const body = `## Agent Eval Results
            
            | Metric | Score | Threshold | Status |
            |--------|-------|-----------|--------|
            | Helpfulness | ... | >= 0.70 | ... |
            | Safety | ... | >= 0.90 | ... |
            | Cost | ... | <= 1.5x | ... |
            `;
            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: body,
            });
      
      - name: Save baseline (main branch only)
        if: github.ref == 'refs/heads/main'
        run: |
          python scripts/save_eval_baseline.py
          git add tests/eval/baselines/
          git commit -m "Update eval baselines" || true
```

## Decision Tree / When to Use

```
Do you change prompts or agent logic?
  YES --> CI eval gates are essential

How often do you deploy agent changes?
  Daily    --> Full eval suite on every PR
  Weekly   --> Full eval suite on PRs, lighter check on commits
  Monthly  --> Full eval suite before release

How large is your eval dataset?
  < 50 cases  --> Run all in CI (~2 min, ~$0.50)
  50-200 cases --> Run all in CI (~5-10 min, ~$2-5)
  > 200 cases  --> Sample 50 in CI, full suite nightly
```

## When NOT to Use

- **Agent code never changes** -- No CI gate needed if nothing changes (but this is rare)
- **Pure infrastructure changes** -- Don't run agent evals for unrelated code changes
- **Pre-product-market-fit exploration** -- Eval gates slow down rapid experimentation; add them when you stabilize

## Tradeoffs

| Approach | CI Time | Cost Per PR | Coverage | Developer Friction |
|----------|---------|-------------|----------|-------------------|
| 20 test cases, 1 metric | ~2 min | ~$0.20 | Basic | Low |
| 50 cases, 3 metrics | ~5 min | ~$1.00 | Good | Medium |
| 100 cases, 5 metrics + regression | ~10 min | ~$3.00 | Comprehensive | Medium |
| 200 cases + trajectory eval | ~20 min | ~$10.00 | Excellent | High |

## Failure Modes

| Failure | Cause | Impact | Prevention |
|---------|-------|--------|------------|
| **Flaky evals** | LLM judge non-determinism | Random CI failures | Low temperature, average multiple runs, tolerance margin |
| **Slow CI** | Too many eval cases | Developer frustration | Tier eval: fast (PR) + full (nightly) |
| **Baseline drift** | Baseline not updated after improvements | Cannot improve scores | Auto-update baseline on main merge |
| **Eval gaming** | Developers add test cases their prompt handles well | False confidence | Rotate eval cases from production logs |
| **Cost creep** | Eval costs grow with each PR | Budget exceeded | Track eval spend; use cheaper judge for CI |

## Source(s) and Further Reading

- LangSmith CI Integration: https://docs.smith.langchain.com/evaluation/how_to_guides/ci
- EDDOps (arXiv:2411.13768): Evaluation-driven development
- Breck et al., "The ML Test Score" (Google, 2017)
- GitHub Actions documentation: https://docs.github.com/en/actions
