# CI/CD for Agent Applications
> Agent CI/CD adds three stages traditional pipelines lack: eval suites with trajectory testing, prompt regression detection, and cost estimation before deploy.

## What It Is

CI/CD for agent applications extends traditional continuous integration and deployment with agent-specific stages: LLM evaluation suites, prompt regression testing, cost estimation, and staged rollouts for prompt changes. Because agents are nondeterministic, you cannot rely solely on unit tests -- you need statistical evaluation.

## How It Works

### Pipeline Stages

```
Code Push
    │
    ▼
┌────────────────┐
│ 1. Lint + Type  │ Standard: ruff, mypy
└────────┬───────┘
         │
┌────────┴───────┐
│ 2. Unit Tests   │ Mock LLM responses, test tool logic
└────────┬───────┘
         │
┌────────┴───────┐
│ 3. Integration  │ Real LLM calls, test agent flows
│    Tests        │ (uses test API keys, rate-limited)
└────────┬───────┘
         │
┌────────┴───────┐
│ 4. Eval Suite   │ RAGAS metrics, trajectory evaluation
│                 │ on golden datasets (50-200 cases)
└────────┬───────┘
         │
┌────────┴───────┐
│ 5. Prompt       │ Compare outputs against baseline
│    Regression   │ Flag if quality drops > 5%
└────────┬───────┘
         │
┌────────┴───────┐
│ 6. Cost         │ Estimate monthly cost impact
│    Estimation   │ Alert if > 20% increase
└────────┬───────┘
         │
┌────────┴───────┐
│ 7. Deploy       │ Canary → staging → production
│    Staging      │ with smoke tests at each stage
└────────┬───────┘
         │
┌────────┴───────┐
│ 8. Smoke Test   │ 5 critical paths in staging
└────────┬───────┘
         │
┌────────┴───────┐
│ 9. Production   │ Staged rollout: 1% → 10% → 50% → 100%
│    Deploy       │ with automatic rollback on error spike
└────────────────┘
```

## Production Implementation

### GitHub Actions CI/CD Pipeline

```yaml
name: Agent CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY_TEST }}
  LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}

jobs:
  lint-and-type:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff mypy
      - run: ruff check src/
      - run: mypy src/ --ignore-missing-imports

  unit-tests:
    runs-on: ubuntu-latest
    needs: lint-and-type
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt -r requirements-test.txt
      - run: pytest tests/unit/ -v --tb=short
        env:
          # No real LLM calls in unit tests
          MOCK_LLM: "true"

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    # Only run on main branch (costs money due to real LLM calls)
    if: github.ref == 'refs/heads/main'
    services:
      redis:
        image: redis:7-alpine
        ports: ["6379:6379"]
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_DB: test_agents
          POSTGRES_PASSWORD: test
        ports: ["5432:5432"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt -r requirements-test.txt
      - run: pytest tests/integration/ -v --tb=short -x
        env:
          REDIS_URL: redis://localhost:6379
          DATABASE_URL: postgresql://postgres:test@localhost:5432/test_agents
          # Real LLM calls with test key (rate-limited)
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY_TEST }}

  eval-suite:
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt ragas langsmith
      - name: Run evaluation suite
        run: |
          python -m pytest tests/eval/ \
            --eval-dataset=tests/eval/golden_dataset.jsonl \
            --min-accuracy=0.85 \
            --min-faithfulness=0.80 \
            --min-answer-relevancy=0.75 \
            --max-hallucination-rate=0.10 \
            -v
      - name: Upload eval results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: tests/eval/results/

  prompt-regression:
    runs-on: ubuntu-latest
    needs: eval-suite
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2  # Need previous commit for diff
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - name: Check for prompt changes
        id: prompt-check
        run: |
          # Detect if any prompt files changed
          CHANGED=$(git diff --name-only HEAD~1 -- 'src/prompts/' '*.prompt' '*.txt')
          if [ -n "$CHANGED" ]; then
            echo "prompts_changed=true" >> $GITHUB_OUTPUT
            echo "Changed prompt files: $CHANGED"
          fi
      - name: Run prompt regression test
        if: steps.prompt-check.outputs.prompts_changed == 'true'
        run: python scripts/prompt_regression_test.py --baseline=main~1 --current=main

  cost-estimation:
    runs-on: ubuntu-latest
    needs: eval-suite
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - name: Estimate cost impact
        run: |
          python scripts/cost_estimator.py \
            --volume=100000 \
            --output=cost_report.json
      - name: Check cost threshold
        run: |
          python -c "
          import json
          report = json.load(open('cost_report.json'))
          monthly = report['estimated_monthly_cost']
          baseline = report.get('baseline_monthly_cost', monthly)
          increase_pct = ((monthly - baseline) / max(baseline, 1)) * 100
          print(f'Estimated: \${monthly:.2f}/month (baseline: \${baseline:.2f})')
          if increase_pct > 20:
              print(f'WARNING: Cost increase of {increase_pct:.1f}% exceeds 20% threshold')
              exit(1)
          "

  deploy-staging:
    runs-on: ubuntu-latest
    needs: [prompt-regression, cost-estimation]
    if: github.ref == 'refs/heads/main'
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - name: Build and push Docker image
        run: |
          docker build -t ${{ secrets.REGISTRY }}/agent:${{ github.sha }} .
          docker push ${{ secrets.REGISTRY }}/agent:${{ github.sha }}
      - name: Deploy to staging
        run: |
          kubectl set image deployment/agent-service \
            agent=${{ secrets.REGISTRY }}/agent:${{ github.sha }} \
            --namespace=staging
          kubectl rollout status deployment/agent-service --namespace=staging --timeout=300s

  smoke-test:
    runs-on: ubuntu-latest
    needs: deploy-staging
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Run smoke tests against staging
        run: |
          python tests/smoke/run_smoke_tests.py \
            --base-url=${{ secrets.STAGING_URL }} \
            --timeout=120

  deploy-production:
    runs-on: ubuntu-latest
    needs: smoke-test
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
      - name: Canary deploy (1%)
        run: |
          kubectl set image deployment/agent-service-canary \
            agent=${{ secrets.REGISTRY }}/agent:${{ github.sha }} \
            --namespace=production
      - name: Monitor canary (5 minutes)
        run: sleep 300
      - name: Check canary health
        run: |
          python scripts/check_canary_health.py \
            --error-threshold=0.05 \
            --latency-threshold=5000
      - name: Full production rollout
        run: |
          kubectl set image deployment/agent-service \
            agent=${{ secrets.REGISTRY }}/agent:${{ github.sha }} \
            --namespace=production
          kubectl rollout status deployment/agent-service --namespace=production --timeout=600s
```

### Eval Suite Script

```python
# tests/eval/test_agent_quality.py
import json
import pytest
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy


def load_golden_dataset(path: str) -> list[dict]:
    """Load golden test cases."""
    with open(path) as f:
        return [json.loads(line) for line in f]


@pytest.fixture
def golden_dataset():
    return load_golden_dataset("tests/eval/golden_dataset.jsonl")


def test_faithfulness(golden_dataset):
    """Test that agent responses are faithful to retrieved context."""
    results = evaluate(
        dataset=golden_dataset,
        metrics=[faithfulness],
    )
    assert results["faithfulness"] >= 0.80, \
        f"Faithfulness {results['faithfulness']:.2f} < 0.80 threshold"


def test_answer_relevancy(golden_dataset):
    """Test that agent responses are relevant to the question."""
    results = evaluate(
        dataset=golden_dataset,
        metrics=[answer_relevancy],
    )
    assert results["answer_relevancy"] >= 0.75, \
        f"Answer relevancy {results['answer_relevancy']:.2f} < 0.75 threshold"


def test_trajectory_correctness(golden_dataset):
    """Test that agent takes correct tool-call sequences."""
    correct = 0
    total = 0
    
    for case in golden_dataset:
        if "expected_tools" not in case:
            continue
        
        total += 1
        actual_tools = run_agent_and_get_tools(case["question"])
        expected_tools = case["expected_tools"]
        
        if actual_tools == expected_tools:
            correct += 1
    
    accuracy = correct / max(total, 1)
    assert accuracy >= 0.85, \
        f"Trajectory accuracy {accuracy:.2f} < 0.85 threshold"
```

## Decision Tree: What Tests to Run When

```
What changed in this PR?
│
├── Tool code changed
│   └── Run: unit tests + integration tests + eval suite
│
├── Prompt changed
│   └── Run: ALL stages including prompt regression
│
├── Infrastructure changed (Docker, K8s)
│   └── Run: unit tests + smoke tests (skip eval)
│
├── Dependencies updated
│   └── Run: unit tests + integration tests
│
└── Agent graph structure changed
    └── Run: ALL stages (most risky change type)
```

## When NOT to Use Full CI/CD Pipeline

- **Prototype stage**: Skip eval suite and prompt regression. Ship fast.
- **< 10 users**: Manual testing is faster than setting up the pipeline.
- **Prompt-only changes**: Consider a fast-path that skips build/deploy and only runs eval.

## Tradeoffs

| Pipeline Stage | Time | Cost per Run | Value |
|---------------|------|-------------|-------|
| Lint + type check | 30s | $0 | Catches obvious errors |
| Unit tests (mocked LLM) | 1-2 min | $0 | Tests deterministic logic |
| Integration tests (real LLM) | 5-10 min | $0.50-2.00 | Tests actual agent behavior |
| Eval suite (50 cases) | 10-20 min | $2-10 | Statistical quality assurance |
| Prompt regression | 5-10 min | $1-5 | Catches quality regressions |
| Cost estimation | 1 min | $0 | Prevents cost surprises |

## Failure Modes

1. **Eval suite flakiness**: Nondeterministic LLM outputs cause eval to randomly fail. Mitigation: run each test case 3 times, use majority vote, set temperature=0.
2. **Cost estimation inaccuracy**: Estimated cost doesn't match production due to different query distributions. Mitigation: use production query sample for estimation.
3. **Prompt regression false positives**: Minor wording changes flag as regression. Mitigation: use semantic similarity for output comparison, not exact match.

## Source(s) and Further Reading

- RAGAS Evaluation: https://docs.ragas.io/
- LangSmith Testing: https://docs.smith.langchain.com/
- GitHub Actions: https://docs.github.com/en/actions
- "Testing LLM Applications" - Eugene Yan (2024)
- Kubernetes Rollout Strategies: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
