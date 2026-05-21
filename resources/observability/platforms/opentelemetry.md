# OpenTelemetry for LLM Observability

> OpenTelemetry is the convergence point -- every LLM observability platform either supports OTel export or will. Instrument once with OTel, export to any backend, switch vendors without code changes.

## What It Is

OpenTelemetry (OTel) is the CNCF vendor-neutral observability standard for traces, metrics, and logs. For LLM applications, **OpenLLMetry** (by Traceloop) provides auto-instrumentation that captures LLM calls, token usage, and tool invocations -- exporting to any OTel-compatible backend (Grafana, Datadog, New Relic, Jaeger, Langfuse, etc.).

**Key facts:**
- **Standard**: CNCF graduated project (vendor-neutral)
- **OpenLLMetry**: Auto-instrumentation for 20+ LLM/framework libraries
- **Export targets**: Any OTel-compatible backend
- **Overhead**: ~2ms per span
- **Best for**: Teams with existing OTel infrastructure or multi-vendor strategy

## How It Works

### Architecture

```
Your Agent Code
    │
    ├── OpenLLMetry auto-instruments:
    │   ├── OpenAI SDK calls
    │   ├── Anthropic SDK calls
    │   ├── LangChain calls
    │   ├── LlamaIndex calls
    │   ├── ChromaDB/Pinecone queries
    │   └── ... (20+ libraries)
    │
    ▼
OTel SDK (collects spans + metrics)
    │
    ▼
OTel Collector (optional, for processing/routing)
    │
    ├── → Grafana Tempo (traces)
    ├── → Datadog APM
    ├── → New Relic
    ├── → Jaeger
    ├── → Langfuse
    ├── → Honeycomb
    └── → Any OTLP endpoint
```

### LLM-Specific Semantic Conventions

OpenTelemetry is developing semantic conventions for GenAI/LLM operations:

| Attribute | Description | Example |
|-----------|------------|---------|
| `gen_ai.system` | AI provider | `openai`, `anthropic` |
| `gen_ai.request.model` | Model name | `gpt-4o` |
| `gen_ai.request.temperature` | Temperature | `0.1` |
| `gen_ai.request.max_tokens` | Max tokens | `1024` |
| `gen_ai.usage.input_tokens` | Input tokens used | `1500` |
| `gen_ai.usage.output_tokens` | Output tokens used | `300` |
| `gen_ai.response.finish_reasons` | Why model stopped | `["stop"]` |
| `gen_ai.response.id` | Response ID | `chatcmpl-abc123` |

## Production Implementation

### Installation

```bash
pip install traceloop-sdk
# or for specific instrumentations:
pip install opentelemetry-instrumentation-openai
pip install opentelemetry-instrumentation-langchain
```

### Auto-Instrumentation with OpenLLMetry

```python
"""
OpenLLMetry auto-instrumentation -- 3 lines to instrument everything.
"""
from traceloop.sdk import Traceloop

# Initialize once at startup -- instruments all supported libraries
Traceloop.init(
    app_name="customer-support-agent",
    api_endpoint="https://your-otel-collector:4318",  # OTLP endpoint
    # Or export to specific backends:
    # api_endpoint="https://otel.langfuse.com",  # Langfuse OTel
    # api_endpoint="https://api.honeycomb.io",   # Honeycomb
)

# Now ALL OpenAI, Anthropic, LangChain calls are automatically traced
import openai

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
# Span automatically created with:
# - gen_ai.system: openai
# - gen_ai.request.model: gpt-4o
# - gen_ai.usage.input_tokens: X
# - gen_ai.usage.output_tokens: Y
# - Duration, status, etc.
```

### Manual Instrumentation (Custom Agents)

```python
"""
Manual OTel instrumentation for custom agent frameworks.
"""
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
import openai

# Setup
resource = Resource.create({
    "service.name": "customer-support-agent",
    "service.version": "v2.3.1",
    "deployment.environment": "production",
})

provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(
    OTLPSpanExporter(endpoint="http://otel-collector:4317")
)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("agent.tracer")


async def agent_with_otel(user_input: str) -> str:
    """Agent with manual OTel instrumentation."""
    
    with tracer.start_as_current_span(
        "agent.run",
        attributes={
            "agent.name": "customer_support",
            "agent.version": "v2.3.1",
        },
    ) as agent_span:
        
        # LLM call span
        with tracer.start_as_current_span(
            "gen_ai.chat",
            attributes={
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.request.temperature": 0.1,
            },
        ) as llm_span:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": user_input}],
                temperature=0.1,
            )
            
            llm_span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
            llm_span.set_attribute("gen_ai.usage.output_tokens", response.usage.completion_tokens)
            llm_span.set_attribute("gen_ai.response.finish_reasons", [response.choices[0].finish_reason])
        
        # Tool call span
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                with tracer.start_as_current_span(
                    f"tool.{tool_call.function.name}",
                    attributes={
                        "tool.name": tool_call.function.name,
                        "tool.args": tool_call.function.arguments,
                    },
                ) as tool_span:
                    # Execute tool
                    tool_result = await execute_tool(
                        tool_call.function.name,
                        tool_call.function.arguments,
                    )
                    tool_span.set_attribute("tool.success", True)
                    tool_span.set_attribute("tool.result_size", len(str(tool_result)))
        
        output = response.choices[0].message.content
        agent_span.set_attribute("agent.output_length", len(output))
        
        return output


# Export to Grafana/Prometheus for dashboards
# OTel Collector config:
# receivers:
#   otlp:
#     protocols:
#       grpc: { endpoint: "0.0.0.0:4317" }
# exporters:
#   otlp/tempo:
#     endpoint: "tempo:4317"
#   prometheus:
#     endpoint: "0.0.0.0:8889"
# service:
#   pipelines:
#     traces:
#       receivers: [otlp]
#       exporters: [otlp/tempo]
#     metrics:
#       receivers: [otlp]
#       exporters: [prometheus]
```

## Decision Tree / When to Use

```
Already have OTel infrastructure (Datadog, Grafana, New Relic)?
  YES --> Use OTel + OpenLLMetry (add LLM observability to existing stack)

Need vendor-neutral / multi-vendor strategy?
  YES --> OTel (instrument once, export anywhere)

Want LLM-specific features (prompt mgmt, eval)?
  YES --> Use OTel for tracing + Langfuse/LangSmith for LLM features

Starting from scratch with no existing infrastructure?
  Consider Langfuse or LangSmith first (simpler setup)
```

## When NOT to Use

- **Need prompt management and eval in one platform** -- OTel is transport, not an LLM platform
- **Small team wanting quick setup** -- Langfuse or LangSmith are faster to get started
- **No existing OTel infrastructure** -- Building OTel infra just for LLMs is overkill

## Tradeoffs

| Aspect | OpenTelemetry | Langfuse | LangSmith |
|--------|-------------|---------|-----------|
| **Vendor lock-in** | None | Low (open-source) | Medium (LangChain) |
| **Setup complexity** | High (collector, backend) | Medium (Docker) | Low (env vars) |
| **LLM-specific features** | Basic (via conventions) | Rich | Rich |
| **Existing APM integration** | Excellent | Limited | Limited |
| **Auto-instrumentation** | OpenLLMetry (20+ libs) | Callbacks | LangChain auto |
| **Cost** | Depends on backend | Free (self-hosted) | Paid |

## Real-World Examples

1. **Enterprise with Datadog**: Team adds OpenLLMetry to existing OTel pipeline. LLM traces appear alongside microservice traces in Datadog APM.
2. **Multi-vendor strategy**: Company uses OTel to export to both Grafana (ops) and Langfuse (ML team), same instrumentation.
3. **Custom framework**: Team using neither LangChain nor LlamaIndex instruments their custom agent with OTel semantic conventions.

## Failure Modes

| Failure | Cause | Impact | Prevention |
|---------|-------|--------|------------|
| **Collector bottleneck** | OTel Collector overloaded | Traces dropped | Scale collector, configure sampling |
| **Missing LLM attributes** | OpenLLMetry not covering new SDK version | Incomplete traces | Pin versions, monitor coverage |
| **Backend mismatch** | Backend doesn't support GenAI semantic conventions | Attributes not indexed | Verify backend support before choosing |

## Source(s) and Further Reading

- OpenTelemetry: https://opentelemetry.io/
- OpenLLMetry (Traceloop): https://www.traceloop.com/openllmetry
- OTel GenAI Semantic Conventions: https://opentelemetry.io/docs/specs/semconv/gen-ai/
- OpenLLMetry GitHub: https://github.com/traceloop/openllmetry
