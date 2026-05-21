# LangSmith: LangChain-Native Observability

> LangSmith has zero overhead tracing because it was designed from day one as the observability layer for LangChain/LangGraph -- traces export asynchronously without adding latency to agent execution.

## What It Is

LangSmith is LangChain's observability and evaluation platform, providing the deepest integration with LangChain and LangGraph. It offers tracing, dataset management, online evaluation, CI eval gates, and prompt management with zero-overhead async export.

**Key facts:**
- **Integration**: Deepest LangGraph integration available
- **Overhead**: Zero overhead (async background export)
- **Features**: Tracing + datasets + online eval + CI gates + prompt hub
- **OTel support**: OpenTelemetry export available
- **Hosting**: Cloud-hosted (managed by LangChain)

## How It Works

### Tracing Architecture

```
Agent Code (LangGraph)
    │
    ├── [Automatic instrumentation]
    │   Every LLM call, tool call, chain step automatically captured
    │
    ├── [Async export] ──→ LangSmith Cloud
    │   Background thread, zero latency impact
    │
    └── Agent response (unblocked)

LangSmith Cloud:
    ├── Trace storage & visualization
    ├── Dataset management
    ├── Online evaluation (LLM-as-judge)
    ├── CI evaluation gates
    └── Prompt versioning (Prompt Hub)
```

### Key Differentiators

1. **Zero overhead**: Traces export asynchronously -- agent latency is unaffected
2. **Automatic instrumentation**: LangChain/LangGraph calls traced with zero code changes
3. **Deep LangGraph support**: Visualizes graph state, node transitions, and interrupts
4. **CI eval gates**: Block deployments when eval scores drop below thresholds
5. **Online evaluation**: Score production traces automatically with LLM-as-judge

## Production Implementation

```python
"""
LangSmith integration for LangGraph agents.
"""
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls_..."
os.environ["LANGCHAIN_PROJECT"] = "customer-support-prod"

# That's it. All LangChain/LangGraph calls are now automatically traced.

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langsmith import Client
from typing import TypedDict


# ============================================================
# Pattern 1: Automatic tracing (zero code)
# ============================================================

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

# Every call to this LLM is automatically traced in LangSmith
response = llm.invoke("What is the refund policy?")
# Trace appears in LangSmith with: model, tokens, latency, input, output


# ============================================================
# Pattern 2: LangGraph with full tracing
# ============================================================

class SupportState(TypedDict):
    messages: list
    order_id: str
    resolution: str

def research_node(state: SupportState) -> SupportState:
    """Each node execution creates a span in the trace."""
    response = llm.invoke(state["messages"])
    return {**state, "messages": state["messages"] + [response]}

def resolve_node(state: SupportState) -> SupportState:
    return {**state, "resolution": "resolved"}

graph = StateGraph(SupportState)
graph.add_node("research", research_node)
graph.add_node("resolve", resolve_node)
graph.set_entry_point("research")
graph.add_edge("research", "resolve")
graph.add_edge("resolve", END)
agent = graph.compile()

# All graph execution is automatically traced:
# - Each node as a span
# - Each LLM call as a child span
# - State transitions visible in UI
result = agent.invoke({
    "messages": [{"role": "user", "content": "Where is my order?"}],
    "order_id": "ORD-123",
    "resolution": "",
})


# ============================================================
# Pattern 3: Datasets for evaluation
# ============================================================

client = Client()

# Create evaluation dataset
dataset = client.create_dataset(
    "customer_support_eval",
    description="Test cases for customer support agent",
)

# Add examples
client.create_example(
    dataset_id=dataset.id,
    inputs={"messages": [{"role": "user", "content": "How do I get a refund?"}]},
    outputs={"expected": "Explain the 30-day refund policy and provide steps"},
)

client.create_example(
    dataset_id=dataset.id,
    inputs={"messages": [{"role": "user", "content": "My order hasn't arrived"}]},
    outputs={"expected": "Ask for order ID, check tracking status, provide update"},
)


# ============================================================
# Pattern 4: Run evaluation
# ============================================================

from langsmith.evaluation import evaluate


def helpfulness_evaluator(run, example):
    """LLM-as-judge evaluator for helpfulness."""
    judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    score_response = judge_llm.invoke(
        f"""Rate this customer support response 1-5 for helpfulness.
        
        Customer question: {example.inputs['messages'][-1]['content']}
        Agent response: {run.outputs.get('resolution', 'N/A')}
        Expected behavior: {example.outputs['expected']}
        
        Return only a number 1-5."""
    )
    
    score = int(score_response.content.strip())
    return {"score": score / 5.0, "key": "helpfulness"}


results = evaluate(
    agent.invoke,
    data="customer_support_eval",
    evaluators=[helpfulness_evaluator],
    experiment_prefix="cs-agent-v2.3",
)


# ============================================================
# Pattern 5: Online evaluation (production scoring)
# ============================================================

def setup_online_eval():
    """Configure automatic scoring of production traces."""
    # LangSmith supports automated evaluation rules that
    # score production traces using LLM-as-judge
    # This runs asynchronously, does not affect agent latency
    
    # Configure in LangSmith UI:
    # 1. Select project
    # 2. Add evaluation rule
    # 3. Choose evaluator (LLM-as-judge, regex, etc.)
    # 4. Set sampling rate (e.g., 5% of traces)
    # 5. View scores on dashboard
    pass
```

### CI Evaluation Gates

```yaml
# .github/workflows/eval-gate.yml
name: Agent Eval Gate
on:
  pull_request:
    paths:
      - 'agents/**'
      - 'prompts/**'

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run agent evaluation
        env:
          LANGCHAIN_API_KEY: ${{ secrets.LANGCHAIN_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          pip install langsmith langchain-openai
          python -m pytest tests/eval/ --tb=short
      
      - name: Check eval scores
        run: |
          # Fetch latest eval results from LangSmith
          python scripts/check_eval_gate.py \
            --dataset customer_support_eval \
            --min-helpfulness 0.7 \
            --min-accuracy 0.8 \
            --fail-on-regression
```

## Decision Tree / When to Use

```
Using LangChain/LangGraph?
  YES --> LangSmith is the natural choice (deepest integration)

Need CI eval gates to block bad deployments?
  YES --> LangSmith (built-in CI integration)

Need self-hosted / data residency?
  NO  --> LangSmith cloud works
  YES --> Consider Langfuse self-hosted instead

Need framework-agnostic support?
  YES --> Langfuse or OpenTelemetry may be better
  NO  --> LangSmith
```

## When NOT to Use

- **Not using LangChain/LangGraph** -- Integration advantage disappears
- **Need self-hosted deployment** -- LangSmith is cloud-only
- **Need data residency in specific regions** -- Limited region options
- **Budget-sensitive** -- Paid service (free tier available but limited)

## Tradeoffs

| Aspect | LangSmith | Langfuse | OpenTelemetry |
|--------|----------|---------|---------------|
| **LangGraph depth** | Excellent (native) | Good (callback) | Basic (manual) |
| **Setup effort** | 2 env vars | Docker compose | SDK + collector |
| **Overhead** | ~0ms (async) | ~5ms | ~2ms |
| **CI eval gates** | Built-in | Manual | Manual |
| **Self-hosted** | No | Yes | Yes |
| **Datasets** | Built-in | Built-in | No |
| **Cost** | Paid (free tier) | Free (self-hosted) | Free (you host) |

## Real-World Examples

1. **CI gate preventing prompt regression**: PR changes system prompt, eval gate catches 15% helpfulness drop, blocks merge until fixed.
2. **Online eval catching quality drift**: LLM-as-judge scores 5% of production traces, alerts when average helpfulness drops below 0.7.
3. **A/B testing prompts**: Two prompt versions traced to same project, performance compared via LangSmith experiment view.

## Failure Modes

| Failure | Cause | Impact | Prevention |
|---------|-------|--------|------------|
| **Traces not appearing** | Missing env vars or API key | No observability | Health check on startup |
| **Eval score variance** | LLM-as-judge non-determinism | Flaky CI gates | Run multiple eval rounds, average |
| **Cost at scale** | High trace volume on paid plan | Unexpected bills | Sampling; monitor trace volume |

## Source(s) and Further Reading

- LangSmith Documentation: https://docs.smith.langchain.com/
- LangSmith Evaluation Guide: https://docs.smith.langchain.com/evaluation
- LangSmith CI/CD Integration: https://docs.smith.langchain.com/evaluation/how_to_guides/ci
- LangSmith + LangGraph: https://docs.smith.langchain.com/concepts/tracing#langgraph
