# Platform Comparison: Langfuse vs LangSmith vs Phoenix vs Helicone

> Pick Langfuse for open-source self-hosted, LangSmith for LangGraph-native CI gates, OpenTelemetry for vendor-neutral, and Helicone for zero-code proxy monitoring.

## What It Is

A decision guide comparing the major LLM observability platforms across 6 dimensions: features, integration depth, data residency, cost, setup complexity, and best-fit scenarios.

## How It Works

### Side-by-Side Comparison

| Dimension | **Langfuse** | **LangSmith** | **Phoenix (Arize)** | **Helicone** | **OpenTelemetry** | **Datadog LLM Obs** |
|-----------|------------|-------------|-------------------|------------|-----------------|-------------------|
| **License** | MIT (open-source) | Proprietary | Apache 2.0 | Proprietary | Apache 2.0 | Proprietary |
| **Self-hosted** | Yes (Postgres+ClickHouse) | No | Yes (local only) | No | Yes (you choose backend) | No |
| **Framework** | Agnostic | LangChain-native | Agnostic | Agnostic (proxy) | Agnostic | Agnostic |
| **Setup effort** | Docker compose (~30min) | 2 env vars (~2min) | pip install (~5min) | 1 line change (~1min) | Collector + backend (~1hr) | Agent install (~15min) |
| **Tracing** | Full hierarchical | Full + LangGraph-native | Full hierarchical | Basic (per-request) | Full (manual or auto) | Full hierarchical |
| **Prompt mgmt** | Yes (versioned) | Yes (Prompt Hub) | No | No | No | No |
| **Eval/Scoring** | Scores, datasets | Datasets, CI gates, online eval | Evals, experiments | No | No | No |
| **CI eval gates** | Manual (via API) | Built-in | Manual | No | No | No |
| **Cost tracking** | Yes | Yes | Yes | Yes (primary feature) | Manual | Yes |
| **Data residency** | Full control (self-hosted) | US cloud | Local | US/EU cloud | Your choice | Multi-region |
| **Adoption** | 2,300+ companies | Broad (LangChain users) | Growing | Growing | Universal | Enterprise |
| **Overhead** | ~5ms | ~0ms (async) | ~5ms | ~10ms (proxy) | ~2ms | ~5ms |
| **Pricing** | Free (self-hosted) or cloud | Free tier + paid | Free (OSS) | Freemium | Free (OSS) + backend | Paid (enterprise) |

### Decision Guide by Use Case

```
┌─────────────────────────────────────────────┐
│           What is your PRIMARY need?         │
└───────────────────┬─────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
 Open-source    LangChain      Existing APM
 + self-host    native         infrastructure
    │               │               │
    ▼               ▼               ▼
 Langfuse       LangSmith      OpenTelemetry
                                + OpenLLMetry
    
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
 Quick proxy    Local dev      Enterprise
 monitoring     + debugging    + multi-team
    │               │               │
    ▼               ▼               ▼
 Helicone       Phoenix        Datadog LLM Obs
```

### Decision by Stack

| Your Stack | Best Choice | Why |
|-----------|------------|-----|
| LangChain + LangGraph | **LangSmith** | Zero-overhead native integration, CI eval gates |
| Custom framework | **Langfuse** | Framework-agnostic, full-featured |
| LlamaIndex | **Langfuse** or **Phoenix** | Both have good LlamaIndex support |
| OpenAI SDK directly | **Helicone** (simple) or **Langfuse** (full) | Depends on needs |
| Existing Datadog/Grafana | **OpenTelemetry** | Adds LLM obs to existing infrastructure |
| Regulated industry (self-hosted required) | **Langfuse** | Only full-featured self-hosted option |
| Budget = $0 | **Langfuse** (self-hosted) or **Phoenix** (local) | Both free and open-source |

### Decision by Data Residency

| Requirement | Options |
|------------|---------|
| Data stays in my cloud | Langfuse (self-hosted), OpenTelemetry (your backend) |
| EU data residency | Langfuse (self-hosted in EU), Helicone (EU region) |
| US cloud acceptable | Any platform |
| Air-gapped / on-prem | Langfuse (self-hosted), Phoenix (local) |

### Decision by Cost

| Budget | Recommendation |
|--------|---------------|
| **Free** | Langfuse (self-hosted) or Phoenix (local) |
| **< $100/mo** | Langfuse Cloud (free tier) or Helicone (free tier) |
| **$100-500/mo** | LangSmith (Plus plan) or Langfuse Cloud |
| **$500+/mo** | LangSmith (Enterprise) or Datadog LLM Obs |

## Production Implementation

```python
"""
Multi-platform export pattern -- instrument once, export to multiple backends.
"""
from typing import Protocol, Any


class ObservabilityBackend(Protocol):
    """Interface for pluggable observability backends."""
    
    def trace(self, name: str, **kwargs) -> Any: ...
    def span(self, name: str, **kwargs) -> Any: ...
    def score(self, trace_id: str, name: str, value: float) -> None: ...


class LangfuseBackend:
    def __init__(self):
        from langfuse import Langfuse
        self.client = Langfuse()
    
    def trace(self, name, **kwargs):
        return self.client.trace(name=name, **kwargs)
    
    def span(self, name, **kwargs):
        pass  # Via trace.span()
    
    def score(self, trace_id, name, value):
        self.client.score(trace_id=trace_id, name=name, value=value)


class OTelBackend:
    def __init__(self):
        from opentelemetry import trace
        self.tracer = trace.get_tracer("agent")
    
    def trace(self, name, **kwargs):
        return self.tracer.start_span(name, attributes=kwargs)
    
    def span(self, name, **kwargs):
        return self.tracer.start_span(name, attributes=kwargs)
    
    def score(self, trace_id, name, value):
        # OTel doesn't have native scoring -- emit as metric
        pass


class MultiBackend:
    """Export to multiple backends simultaneously."""
    
    def __init__(self, backends: list[ObservabilityBackend]):
        self.backends = backends
    
    def trace(self, name, **kwargs):
        return [b.trace(name, **kwargs) for b in self.backends]
    
    def score(self, trace_id, name, value):
        for b in self.backends:
            b.score(trace_id, name, value)


# Usage: Export to both Langfuse AND OTel
multi = MultiBackend([LangfuseBackend(), OTelBackend()])
```

## When NOT to Use

- **Don't use multiple platforms for the same purpose** -- Pick one primary, export to others only if needed
- **Don't choose based on hype** -- Choose based on your stack, data residency needs, and team size

## Tradeoffs

| Choice | You Get | You Give Up |
|--------|---------|------------|
| **Langfuse** | Full control, self-hosted, framework-agnostic | Must manage infrastructure |
| **LangSmith** | Deepest LangGraph integration, zero setup | Vendor lock-in, cloud-only |
| **Phoenix** | Free local development, good for debugging | Not designed for production scale |
| **Helicone** | Simplest setup (proxy), great cost tracking | Limited tracing depth |
| **OpenTelemetry** | Vendor-neutral, future-proof | No LLM-specific features built-in |
| **Datadog LLM Obs** | Integrates with existing Datadog | Expensive, enterprise-focused |

## Source(s) and Further Reading

- Langfuse: https://langfuse.com/docs
- LangSmith: https://docs.smith.langchain.com/
- Arize Phoenix: https://docs.arize.com/phoenix
- Helicone: https://helicone.ai/docs
- OpenLLMetry: https://www.traceloop.com/openllmetry
- Datadog LLM Observability: https://docs.datadoghq.com/llm_observability/
