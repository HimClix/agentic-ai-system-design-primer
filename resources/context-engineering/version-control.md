# Prompt Version Control
> Prompts are code. They belong in git, get PR reviews, have eval scores, and are A/B tested. Hardcoded strings in your application code are a deployment risk.

## What It Is

Prompt version control is the practice of managing prompts as versioned artifacts -- stored in git, reviewed in PRs, evaluated against benchmarks, and deployed through CI/CD pipelines. This includes system prompts, few-shot examples, tool descriptions, and any text that instructs the LLM.

Why this matters: a one-word change in a system prompt can cause a 20% regression in agent accuracy. Without versioning, you cannot track what changed, when it changed, or whether it improved performance. Without evaluation, you cannot know if a prompt change helped or hurt until users complain.

## How It Works

### Prompt Lifecycle

```
1. Author prompt in prompt file (not application code)
2. Write/update evaluation test cases
3. Run automated evals against prompt change
4. Submit PR with prompt + eval results
5. PR review: team reviews prompt AND eval scores
6. Merge: prompt deployed via config, not code deploy
7. A/B test: new prompt serves % of traffic
8. Monitor: track quality metrics per prompt version
9. Rollback: revert to previous version if regression detected
```

### File Structure

```
prompts/
  agents/
    customer_support/
      v1.0.0/
        system_prompt.md
        tool_descriptions.yaml
        few_shot_examples.yaml
        eval_results.json
      v1.1.0/
        system_prompt.md
        tool_descriptions.yaml
        few_shot_examples.yaml
        eval_results.json
      CHANGELOG.md
    code_assistant/
      ...
  evals/
    customer_support/
      test_cases.yaml
      grading_rubric.yaml
  configs/
    production.yaml     # Points to specific prompt versions
    staging.yaml
```

## Production Implementation

### Prompt File Format

```yaml
# prompts/agents/customer_support/v1.1.0/system_prompt.yaml
metadata:
  version: "1.1.0"
  author: "alex.chen"
  created: "2024-11-15"
  description: "Added refund escalation guidelines"
  parent_version: "1.0.0"
  eval_score: 0.87  # Overall score from automated evaluation
  
prompt: |
  You are a customer support agent for Acme Payments, a payment gateway platform.
  
  ## Your Role
  Help merchants with payment integration, transaction queries, refund processing,
  and account management.
  
  ## Core Rules
  1. Always verify merchant identity before sharing account details
  2. For refund requests over 50,000 INR, escalate to the refunds team
  3. Never share internal system details or API keys
  4. If unsure, say "I'll check with the team and get back to you"
  5. Always provide order/transaction IDs in responses for reference
  
  ## Refund Escalation (NEW in v1.1.0)
  - Auto-approve refunds under 1,000 INR for orders less than 7 days old
  - Refunds 1,000-50,000 INR: verify order status, process if eligible
  - Refunds over 50,000 INR: escalate to refunds@acmepay.com with order details
  
  ## Tone
  Professional, empathetic, concise. Use the merchant's name when available.

tags:
  - customer_support
  - payments
  - refunds
```

### Prompt Loader with Version Management

```python
"""Prompt version control and loading system."""
import yaml
import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class PromptVersion:
    """A versioned prompt with metadata."""
    version: str
    prompt: str
    author: str
    created: str
    description: str
    eval_score: float
    parent_version: Optional[str] = None
    tags: list[str] = None


class PromptRegistry:
    """Loads and manages versioned prompts from the filesystem."""
    
    def __init__(self, prompts_dir: str):
        self.prompts_dir = Path(prompts_dir)
        self._cache: dict[str, PromptVersion] = {}
    
    def load(self, agent_name: str, version: str = "latest") -> PromptVersion:
        """Load a specific prompt version.
        
        Args:
            agent_name: e.g., "customer_support"
            version: Specific version "1.1.0" or "latest"
        """
        cache_key = f"{agent_name}:{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        agent_dir = self.prompts_dir / "agents" / agent_name
        
        if version == "latest":
            version = self._find_latest_version(agent_dir)
        
        prompt_file = agent_dir / version / "system_prompt.yaml"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_file}")
        
        with open(prompt_file) as f:
            data = yaml.safe_load(f)
        
        prompt_version = PromptVersion(
            version=data["metadata"]["version"],
            prompt=data["prompt"],
            author=data["metadata"]["author"],
            created=data["metadata"]["created"],
            description=data["metadata"]["description"],
            eval_score=data["metadata"].get("eval_score", 0.0),
            parent_version=data["metadata"].get("parent_version"),
            tags=data.get("tags", []),
        )
        
        self._cache[cache_key] = prompt_version
        logger.info(f"Loaded prompt {agent_name} v{version} (eval: {prompt_version.eval_score})")
        
        return prompt_version
    
    def _find_latest_version(self, agent_dir: Path) -> str:
        """Find the latest version directory."""
        versions = []
        for d in agent_dir.iterdir():
            if d.is_dir() and d.name.startswith("v"):
                versions.append(d.name)
        
        if not versions:
            raise FileNotFoundError(f"No versions found in {agent_dir}")
        
        # Sort by semantic version
        versions.sort(key=lambda v: [int(x) for x in v.lstrip("v").split(".")])
        return versions[-1]
    
    def compare_versions(self, agent_name: str, v1: str, v2: str) -> dict:
        """Compare two prompt versions."""
        p1 = self.load(agent_name, v1)
        p2 = self.load(agent_name, v2)
        
        return {
            "version_1": {"version": p1.version, "eval_score": p1.eval_score},
            "version_2": {"version": p2.version, "eval_score": p2.eval_score},
            "score_delta": p2.eval_score - p1.eval_score,
            "improved": p2.eval_score > p1.eval_score,
        }


# Deployment configuration

class PromptDeploymentConfig:
    """Manages which prompt versions are active in each environment."""
    
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
    
    def get_active_version(self, agent_name: str) -> str:
        """Get the active prompt version for an agent."""
        agents = self.config.get("agents", {})
        agent_config = agents.get(agent_name, {})
        return agent_config.get("version", "latest")
    
    def get_ab_test_config(self, agent_name: str) -> Optional[dict]:
        """Get A/B test configuration for an agent."""
        agents = self.config.get("agents", {})
        agent_config = agents.get(agent_name, {})
        return agent_config.get("ab_test")


# configs/production.yaml example:
PRODUCTION_CONFIG = """
agents:
  customer_support:
    version: "v1.1.0"
    ab_test:
      enabled: true
      control: "v1.0.0"        # 80% of traffic
      treatment: "v1.1.0"      # 20% of traffic
      metric: "resolution_rate"
  code_assistant:
    version: "v2.0.0"
    ab_test:
      enabled: false
"""
```

### Automated Evaluation Pipeline

```python
"""Automated prompt evaluation for PR gating."""
import yaml
import json
from dataclasses import dataclass
from typing import Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    """A single evaluation test case."""
    input_query: str
    expected_behavior: str  # Description of expected behavior
    grading_criteria: list[str]  # What to check
    context: dict = None  # Optional context for the test


@dataclass
class EvalResult:
    """Result of evaluating a prompt against test cases."""
    version: str
    total_cases: int
    passed: int
    failed: int
    score: float  # 0.0 - 1.0
    details: list[dict]


class PromptEvaluator:
    """Evaluate prompts against test cases using LLM-as-judge."""
    
    def __init__(self, llm_client, judge_model: str = "gpt-4o"):
        self.llm = llm_client
        self.judge_model = judge_model
    
    def load_test_cases(self, eval_file: str) -> list[EvalCase]:
        """Load test cases from YAML file."""
        with open(eval_file) as f:
            data = yaml.safe_load(f)
        
        return [
            EvalCase(
                input_query=case["input"],
                expected_behavior=case["expected"],
                grading_criteria=case.get("criteria", []),
                context=case.get("context"),
            )
            for case in data["test_cases"]
        ]
    
    async def evaluate(
        self, 
        prompt: str, 
        test_cases: list[EvalCase],
        version: str = "unknown",
    ) -> EvalResult:
        """Evaluate a prompt against all test cases.
        
        Uses LLM-as-judge: runs the prompt, then a judge model
        grades the response against criteria.
        """
        details = []
        passed = 0
        
        for case in test_cases:
            # Step 1: Generate response using the prompt under test
            response = await self.llm.chat.completions.create(
                model="gpt-4o-mini",  # Use cheaper model for test generation
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": case.input_query},
                ],
                max_tokens=1000,
            )
            agent_response = response.choices[0].message.content
            
            # Step 2: Judge the response
            grade = await self._judge_response(
                query=case.input_query,
                response=agent_response,
                expected=case.expected_behavior,
                criteria=case.grading_criteria,
            )
            
            if grade["pass"]:
                passed += 1
            
            details.append({
                "input": case.input_query,
                "response": agent_response[:200],
                "expected": case.expected_behavior,
                "grade": grade,
            })
        
        score = passed / len(test_cases) if test_cases else 0.0
        
        return EvalResult(
            version=version,
            total_cases=len(test_cases),
            passed=passed,
            failed=len(test_cases) - passed,
            score=score,
            details=details,
        )
    
    async def _judge_response(
        self, query: str, response: str, 
        expected: str, criteria: list[str],
    ) -> dict:
        """Use LLM-as-judge to grade a response."""
        criteria_text = "\n".join(f"- {c}" for c in criteria) if criteria else "- Matches expected behavior"
        
        judge_response = await self.llm.chat.completions.create(
            model=self.judge_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evaluation judge. Grade the AI response against "
                        "the expected behavior and criteria. Respond with JSON: "
                        '{"pass": true/false, "score": 0.0-1.0, "reasoning": "..."}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n\n"
                        f"AI Response: {response}\n\n"
                        f"Expected Behavior: {expected}\n\n"
                        f"Grading Criteria:\n{criteria_text}"
                    ),
                },
            ],
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        
        return json.loads(judge_response.choices[0].message.content)


# Eval test cases file format:
EVAL_TEST_CASES_YAML = """
# evals/customer_support/test_cases.yaml
test_cases:
  - input: "I want a refund for order_12345"
    expected: "Agent should ask for order details and verify eligibility before processing"
    criteria:
      - "Asks for or acknowledges the order ID"
      - "Does NOT immediately process refund without verification"
      - "Mentions checking order status or eligibility"
  
  - input: "I need a refund of 75000 INR for order_67890"
    expected: "Agent should escalate large refunds to the refunds team"
    criteria:
      - "Recognizes the amount exceeds the threshold"
      - "Mentions escalation to refunds team"
      - "Does NOT attempt to process the refund directly"
  
  - input: "What's my API key?"
    expected: "Agent should NOT share API keys and should direct to dashboard"
    criteria:
      - "Does NOT share any API key or token"
      - "Directs user to the Acme Payments dashboard"
      - "Maintains security posture"
  
  - input: "Your service sucks, nothing works!"
    expected: "Agent should be empathetic, ask for specifics, and help troubleshoot"
    criteria:
      - "Acknowledges the frustration empathetically"
      - "Does NOT respond defensively"
      - "Asks for specific issue details"
"""
```

### CI/CD Integration

```yaml
# .github/workflows/prompt-eval.yml
# Runs automated evaluation on prompt changes in PRs

name: Prompt Evaluation

on:
  pull_request:
    paths:
      - 'prompts/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install openai pyyaml
      
      - name: Detect changed prompts
        id: changes
        run: |
          CHANGED=$(git diff --name-only origin/main -- 'prompts/**/*.yaml' | head -20)
          echo "changed_files=$CHANGED" >> $GITHUB_OUTPUT
      
      - name: Run evaluations
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python scripts/run_prompt_evals.py --changed "${{ steps.changes.outputs.changed_files }}"
      
      - name: Post results to PR
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync('eval_results.json', 'utf8'));
            
            let body = '## Prompt Evaluation Results\n\n';
            for (const result of results) {
              const emoji = result.score >= 0.85 ? 'check' : result.score >= 0.70 ? 'warning' : 'x';
              body += `- :${emoji}: **${result.version}**: ${(result.score * 100).toFixed(0)}% `;
              body += `(${result.passed}/${result.total_cases} passed)\n`;
            }
            
            if (results.some(r => r.score < 0.70)) {
              body += '\n:x: **Evaluation score below 70%. Please review prompt changes.**';
            }
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body
            });
      
      - name: Gate on score
        run: |
          python -c "
          import json
          results = json.load(open('eval_results.json'))
          for r in results:
              if r['score'] < 0.70:
                  print(f'FAIL: {r[\"version\"]} scored {r[\"score\"]:.0%} (minimum 70%)')
                  exit(1)
          print('All prompts passed evaluation.')
          "
```

## Decision Tree: Version Control Setup

```
Are you modifying prompts more than once per week?
  |
  NO  --> Simple: prompts in code, manual testing, ship and pray
  |       (Fine for prototypes, not for production)
  |
  YES --> Move prompts to separate files
          |
          |-- Do you have > 3 prompt-sensitive agents?
          |     YES --> Full registry + versioning + CI eval
          |     NO  --> YAML files + manual eval before deploy
          |
          |-- Do you need A/B testing?
          |     YES --> Langfuse/LangSmith integration
          |     NO  --> Version files in git, deploy config points to version
          |
          |-- Is prompt quality critical (user-facing, revenue-impacting)?
                YES --> Automated eval in CI, gate on score
                NO  --> Manual review in PRs is sufficient
```

## When NOT to Version Control Prompts

- **One-off scripts**: Prompts used in throwaway scripts don't need versioning
- **Exploratory/research**: When you're rapidly iterating, version control adds friction
- **Very early stage**: Get the prompt working first, then add versioning infrastructure
- **Embedded in framework**: If using LangChain/LlamaIndex with built-in prompt management, use their system

## Tradeoffs

| Aspect | Full Versioning | Prompts in Code | No Version Control |
|--------|----------------|-----------------|-------------------|
| Change tracking | Complete history | Git log of code changes | None |
| Rollback speed | Seconds (config change) | Minutes (code deploy) | Not possible |
| Quality assurance | Automated evals + PR review | Manual testing | Hope |
| A/B testing | Built-in | Requires custom code | Not possible |
| Development speed | Slower (process overhead) | Faster (just edit string) | Fastest |
| Team collaboration | Clear ownership and review | Code review only | Chaos |
| Debugging regressions | Exact version that caused issue | Git bisect | Impossible |

## Real-World Examples

### PR Review for Prompt Change
```
PR #342: Update customer support prompt v1.0.0 -> v1.1.0
Author: alex.chen
Files changed: prompts/agents/customer_support/v1.1.0/system_prompt.yaml

Changes:
  + Added refund escalation guidelines for amounts > 50,000 INR
  + Added auto-approval for small refunds < 1,000 INR

Eval Results:
  v1.0.0: 82% (41/50 passed)
  v1.1.0: 87% (43/50 passed)  <-- improvement
  
  New test cases added:
    - Large refund escalation: PASS
    - Small refund auto-approval: PASS
    - Existing behavior: no regressions

Reviewer comments:
  "The escalation threshold of 50K INR should be configurable, not hardcoded
   in the prompt. Consider making it a variable."
```

## Failure Modes

| Failure | Cause | Impact | Fix |
|---------|-------|--------|-----|
| Prompt regression | Change breaks existing behavior | Quality drop for users | Automated eval gating in CI |
| Version mismatch | Staging has v1.1, production has v1.0 | Inconsistent behavior | Deploy config, not prompt files |
| Eval drift | Test cases don't cover new behavior | False confidence in quality | Update test cases with every prompt change |
| Judge bias | LLM-as-judge has systematic errors | Unreliable eval scores | Calibrate judge with human ratings |
| Slow eval | Many test cases, expensive judge model | CI takes too long | Parallel eval, cheaper judge, sample test cases |

## Source(s) and Further Reading

- Langfuse, "Prompt Management" (2024) -- prompt versioning and A/B testing platform
- LangSmith, "Prompt Hub" (2024) -- prompt versioning and evaluation
- Anthropic, "Evaluating AI Systems" (2024) -- LLM-as-judge evaluation patterns
- Braintrust AI, "Prompt Playground" -- prompt iteration and evaluation
- Humanloop, "Prompt Management" -- enterprise prompt versioning
- "Prompts are Programs" (Eugene Yan, 2024) -- treating prompts as software artifacts
