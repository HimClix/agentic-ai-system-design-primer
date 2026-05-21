# Datadog LLM Observability

> Native LLM monitoring in the Datadog platform: auto-instrumentation for OpenAI/Anthropic/LangChain, token usage dashboards, cost tracking, and integration with existing APM infrastructure.

## What It Is

Datadog LLM Observability is Datadog's native solution for monitoring LLM applications. It extends the Datadog APM platform with LLM-specific features:

1. **Auto-instrumentation**: Automatic tracing of OpenAI, Anthropic, LangChain, and other LLM SDK calls
2. **Token tracking**: Input/output token counts, costs per model
3. **Prompt/response logging**: Full input/output capture (with PII redaction)
4. **Cost dashboards**: Per-model, per-team, per-feature cost breakdown
5. **Error monitoring**: LLM-specific errors (rate limits, context overflow, content filtering)
6. **Integration with existing Datadog**: Correlate LLM performance with infrastructure metrics

### When to Use Datadog LLM Observability

```
Use Datadog LLM Observability if:
  ✓ You already use Datadog for infrastructure/APM monitoring
  ✓ You want LLM observability in the SAME platform as your other monitoring
  ✓ You need to correlate LLM performance with infrastructure metrics
  ✓ Your organization already pays for Datadog
  ✓ You need enterprise-grade alerting on LLM metrics

Use Langfuse/LangSmith instead if:
  ✓ You don't use Datadog already (adding Datadog just for LLM is expensive)
  ✓ You need deeper LLM-specific features (prompt versioning, eval frameworks)
  ✓ You want open-source / self-hosted
  ✓ You are cost-sensitive (Datadog is the most expensive option)
```

### Datadog vs LLM-Specific Platforms

| Feature | Datadog LLM | Langfuse | LangSmith | W&B Weave |
|---------|-------------|----------|-----------|-----------|
| LLM tracing | Yes | Yes | Yes | Yes |
| Infrastructure correlation | Yes (native) | No | No | No |
| Existing APM integration | Yes | No | No | No |
| Custom dashboards | Excellent | Good | Good | Good |
| Alerting | Enterprise-grade | Basic | Basic | Basic |
| Pricing | Expensive ($$$) | Affordable | Moderate | Moderate |
| Setup (if using Datadog) | Minutes | Hours | Hours | Hours |
| Setup (from scratch) | Days (full Datadog setup) | Minutes | Minutes | Minutes |
| Open source | No | Yes | No | Weave only |

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                  YOUR APPLICATION                    │
│                                                     │
│  Agent Code                                         │
│     │                                               │
│  ┌──▼──────────────────────────────────────────┐   │
│  │  ddtrace auto-instrumentation               │   │
│  │                                              │   │
│  │  Patches:                                    │   │
│  │  - anthropic.messages.create()               │   │
│  │  - openai.chat.completions.create()          │   │
│  │  - langchain.invoke()                        │   │
│  │                                              │   │
│  │  Captures:                                   │   │
│  │  - Input messages (system + user)            │   │
│  │  - Output response                           │   │
│  │  - Token counts (input/output)               │   │
│  │  - Model name, latency, status               │   │
│  │  - Cost (calculated from tokens + model)     │   │
│  └──────────────────┬──────────────────────────┘   │
│                     │                               │
│  ┌──────────────────▼──────────────────────────┐   │
│  │  Datadog Agent (dd-agent)                    │   │
│  │  Collects traces, metrics, logs              │   │
│  └──────────────────┬──────────────────────────┘   │
└─────────────────────┼───────────────────────────────┘
                      │
               ┌──────▼──────┐
               │  Datadog     │
               │  Backend     │
               │              │
               │  - APM Traces│
               │  - LLM Spans │
               │  - Metrics   │
               │  - Dashboards│
               └──────────────┘
```

## Production Implementation

### Setup with ddtrace (Python)

```python
# requirements.txt
# ddtrace>=2.8.0
# anthropic>=0.25.0
# openai>=1.0.0

# --- Option 1: Auto-instrumentation (recommended) ---
# Run your application with:
# DD_SERVICE=my-agent DD_ENV=production DD_LLMOBS_ENABLED=1 \
#   DD_LLMOBS_AGENTLESS_ENABLED=1 DD_API_KEY=<your-key> \
#   DD_LLMOBS_ML_APP=my-agent-app \
#   ddtrace-run python my_agent.py

# That's it -- all Anthropic/OpenAI calls are automatically traced.


# --- Option 2: Manual instrumentation (more control) ---

from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import llm, workflow, tool, task, agent

# Initialize LLM Observability
LLMObs.enable(
    ml_app="support-agent",
    api_key="<datadog-api-key>",
    site="datadoghq.com",
    agentless_enabled=True,  # No dd-agent needed
)


@agent(name="support_agent")
def handle_support_query(user_message: str) -> str:
    """Top-level agent span -- groups all nested operations."""
    intent = classify_intent(user_message)
    context = search_knowledge_base(user_message)
    response = generate_response(user_message, context, intent)
    return response


@task(name="classify_intent")
def classify_intent(message: str) -> str:
    """Task span for intent classification."""
    import anthropic
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-haiku-3.5",
        max_tokens=20,
        system="Classify as support/sales/technical. Output only the category.",
        messages=[{"role": "user", "content": message}],
    )
    return response.content[0].text.strip()


@tool(name="search_kb")
def search_knowledge_base(query: str) -> list[str]:
    """Tool span for knowledge base search."""
    # Your vector search implementation
    results = vector_search(query, top_k=5)
    return [r["content"] for r in results]


@llm(model_name="claude-sonnet", model_provider="anthropic")
def generate_response(message: str, context: list[str], intent: str) -> str:
    """LLM span with explicit model metadata."""
    import anthropic
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=f"You are a {intent} agent. Context:\n{chr(10).join(context)}",
        messages=[{"role": "user", "content": message}],
    )
    return response.content[0].text


# --- Custom Metrics ---

from ddtrace import tracer
from datadog import statsd

def track_agent_metrics(response: str, intent: str, model: str):
    """Send custom metrics to Datadog for LLM monitoring."""

    # Token cost metric
    statsd.increment("llm.requests.total", tags=[
        f"model:{model}",
        f"intent:{intent}",
    ])

    # Response length distribution
    statsd.histogram("llm.response.length", len(response), tags=[
        f"model:{model}",
    ])


# --- Custom Dashboards (Terraform) ---

DATADOG_DASHBOARD_CONFIG = """
# Terraform configuration for LLM monitoring dashboard
resource "datadog_dashboard" "llm_monitoring" {
  title       = "LLM Agent Monitoring"
  description = "Real-time monitoring of agent LLM calls"
  layout_type = "ordered"

  widget {
    timeseries_definition {
      title = "LLM Request Rate by Model"
      request {
        q = "sum:llm.requests.total{*} by {model}.as_rate()"
        display_type = "bars"
      }
    }
  }

  widget {
    timeseries_definition {
      title = "Token Usage Over Time"
      request {
        q = "sum:llm.tokens.input{*} by {model}"
        display_type = "line"
      }
      request {
        q = "sum:llm.tokens.output{*} by {model}"
        display_type = "line"
      }
    }
  }

  widget {
    query_value_definition {
      title = "Estimated LLM Cost (24h)"
      request {
        q = "sum:llm.cost.usd{*}.rollup(sum, 86400)"
      }
    }
  }

  widget {
    timeseries_definition {
      title = "LLM Latency (p50, p95, p99)"
      request {
        q = "avg:llm.latency{*} by {model}"
        display_type = "line"
      }
    }
  }

  widget {
    timeseries_definition {
      title = "Error Rate by Model"
      request {
        q = "sum:llm.errors{*} by {model,error_type}.as_rate()"
        display_type = "bars"
      }
    }
  }
}
"""

# --- Alerting Configuration ---

DATADOG_ALERTS = """
# Alert: LLM cost exceeding budget
resource "datadog_monitor" "llm_cost_alert" {
  name    = "LLM Daily Cost Exceeding $500"
  type    = "query alert"
  query   = "sum(last_24h):sum:llm.cost.usd{*} > 500"
  message = <<-EOT
    LLM spending exceeded $500 in the last 24 hours.
    Current spend: {{value}}
    @slack-engineering-alerts
  EOT
  
  monitor_thresholds {
    critical = 500
    warning  = 400
  }
}

# Alert: LLM error rate spike
resource "datadog_monitor" "llm_error_rate" {
  name    = "LLM Error Rate > 5%"
  type    = "query alert"
  query   = "sum(last_5m):sum:llm.errors{*}.as_count() / sum:llm.requests.total{*}.as_count() > 0.05"
  message = <<-EOT
    LLM error rate exceeded 5% in the last 5 minutes.
    Error rate: {{value}}
    @pagerduty-on-call
  EOT
}

# Alert: LLM latency degradation
resource "datadog_monitor" "llm_latency" {
  name    = "LLM p95 Latency > 5s"
  type    = "query alert"
  query   = "percentile(last_10m):p95:llm.latency{*} > 5000"
  message = <<-EOT
    LLM p95 latency exceeded 5 seconds.
    Current p95: {{value}}ms
    @slack-engineering-alerts
  EOT
}
"""
```

### Integration with Existing APM

```python
"""
The key advantage of Datadog LLM Observability: 
correlate LLM calls with your existing APM traces.

Example: An API endpoint that uses an LLM agent.
The full trace shows:
  1. HTTP request received (Datadog APM)
  2. Database query for user context (Datadog APM)
  3. LLM call for intent classification (Datadog LLM Obs)
  4. Vector search for context (Datadog APM)
  5. LLM call for response generation (Datadog LLM Obs)
  6. HTTP response sent (Datadog APM)

All in a single trace view.
"""

from ddtrace import tracer
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/api/chat", methods=["POST"])
def chat():
    # This endpoint is automatically traced by Datadog APM
    user_message = request.json["message"]
    user_id = request.json["user_id"]

    # Database call (APM traces this)
    user_context = get_user_context(user_id)

    # LLM call (LLM Observability traces this)
    response = handle_support_query(user_message)

    return jsonify({"response": response})
```

## Decision Tree: When to Use Datadog LLM Observability

```
    Should I use Datadog for LLM monitoring?
                        │
             ┌──────────▼──────────┐
             │ Do you already use   │
             │ Datadog for infra    │
             │ monitoring?          │
             └──┬──────────────┬──┘
               Yes             No
                │               │
         ┌──────▼──────┐    Don't start
         │ Do you want  │    Datadog just
         │ LLM traces   │    for LLM.
         │ in the SAME  │    Use Langfuse
         │ platform?    │    or LangSmith.
         └──┬────────┬─┘
           Yes      No
            │        │
    Datadog LLM    Use Langfuse
    Observability  alongside Datadog
    (enable it,    (LLM-specific tool
     it's easy)     + Datadog for infra)
```

## When NOT to Use

1. **You do not use Datadog already**: Adding Datadog solely for LLM monitoring is expensive ($23/host/month for APM + LLM Obs). Use Langfuse or LangSmith instead.
2. **You need deep prompt management**: Datadog does not have prompt versioning, A/B testing, or prompt registries. Use LangSmith or W&B for this.
3. **You need evaluation frameworks**: Datadog does not provide built-in eval pipelines like RAGAS or LLM-as-Judge. Use separate eval tools.
4. **Open-source requirement**: Datadog is fully proprietary. Use Langfuse for open-source LLM observability.
5. **Cost-sensitive startups**: Datadog's pricing model (per host + per feature) can be expensive. Langfuse's free tier is more generous.

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Single pane of glass (infra + LLM) | Most expensive option |
| Auto-instrumentation (minutes to set up) | Less LLM-specific depth than Langfuse |
| Enterprise-grade alerting (PagerDuty, Slack) | No prompt versioning or management |
| Correlate LLM latency with infrastructure | No built-in evaluation framework |
| Existing Datadog expertise transfers | Vendor lock-in (Datadog-specific format) |
| Scales to millions of traces/day | Overkill for simple agent monitoring |

## Pricing (2025)

```
Datadog APM + LLM Observability:
  APM Host:      $31/host/month (annual) or $36/month (on-demand)
  LLM Obs:       Included with APM (no additional cost for traces)
  Log Management: $1.70/M log events/month (if logging prompts/responses)
  
  Typical 3-host setup with LLM tracing:
    3 × $31 (APM) + ~$50 (logs for 30M events) = $143/month

Langfuse comparison:
  Self-hosted: $0 (infrastructure costs only)
  Cloud: $59/month (10K traces/month included)

LangSmith comparison:
  Free: 5K traces/month
  Plus: $39/seat/month
```

## Failure Modes

### 1. Trace Volume Costs
Auto-instrumentation captures every LLM call, generating massive trace volume at scale.
**Mitigation**: Use Datadog's ingestion controls to sample traces (e.g., keep 10% in production). Set trace retention policies.

### 2. PII in Traces
LLM inputs/outputs often contain user PII that gets stored in Datadog.
**Mitigation**: Enable Datadog's Sensitive Data Scanner. Configure ddtrace to redact prompt/response content for sensitive workloads.

### 3. Latency Overhead
The ddtrace library adds instrumentation overhead.
**Mitigation**: Datadog's overhead is typically < 5ms per trace. Use async trace submission. Negligible for LLM calls (which take 500ms+).

### 4. Missing LLM-Specific Context
Datadog traces show latency and tokens but not semantic evaluation quality.
**Mitigation**: Supplement with Langfuse or custom eval pipelines for quality metrics. Use Datadog for operational metrics, specialized tools for quality.

### 5. Dashboard Sprawl
Too many dashboards and monitors are created, making it hard to find relevant information.
**Mitigation**: Use a single "LLM Overview" dashboard. Create monitors only for actionable alerts. Use Datadog's team-based dashboard organization.

## Sources and Further Reading

- [Datadog LLM Observability Docs](https://docs.datadoghq.com/llm_observability/)
- [ddtrace LLM Integration](https://docs.datadoghq.com/tracing/llm_observability/sdk/python/)
- [Datadog + Anthropic Integration](https://docs.datadoghq.com/integrations/anthropic/)
- [Datadog + LangChain Integration](https://docs.datadoghq.com/integrations/langchain/)
- [Datadog Pricing](https://www.datadoghq.com/pricing/)
- [Datadog Sensitive Data Scanner](https://docs.datadoghq.com/sensitive_data_scanner/)
