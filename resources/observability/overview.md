# Observability Overview: Why Agent Observability Is Different

> Traditional APM answers "is the service up?" -- agent observability answers "is the agent making good decisions and why?"

## What It Is

Agent observability is the practice of understanding what agents decide, why they decide it, and whether those decisions are correct. It goes beyond traditional application performance monitoring (APM) because agents are **non-deterministic, multi-step reasoning systems** where the same input can produce different outputs, and correctness is subjective.

### How Agent Observability Differs from APM

| Dimension | Traditional APM | Agent Observability |
|-----------|----------------|---------------------|
| **Primary question** | Is the service healthy? | Is the agent making good decisions? |
| **Key metric** | Latency, error rate, throughput | Decision quality, reasoning correctness |
| **Determinism** | Same input = same output | Same input = different outputs |
| **Trace structure** | Request → microservices | Request → LLM calls → tool calls → reasoning steps |
| **Cost** | Compute cost | Compute + token cost per request |
| **Failure mode** | Crash, timeout, error | Plausible but wrong answer, hallucination, wrong tool |
| **Debugging** | Stack traces, logs | Reasoning traces, prompt versions, context analysis |

## How It Works

### The 3 Pillars for Agents

**Pillar 1: Traces** -- Structured records of every step the agent takes (LLM calls, tool calls, reasoning, decisions). Unlike APM traces that capture service-to-service calls, agent traces capture the reasoning chain.

**Pillar 2: Metrics** -- Quantitative measures of agent behavior: P95 latency, token usage, tool success rate, hallucination rate, cost per request, finish reason distribution.

**Pillar 3: Alerts** -- Automated detection of behavioral anomalies: runaway loops, token explosions, repeated failures, quality degradation, safety violations.

### Behavioral Observability: What Agents Decide and Why

```
Traditional: "The API returned 200 in 45ms"
Agent:       "The agent decided to call get_order_status with order_id=123
              because the user asked 'where is my order?'
              The tool returned 'shipped, arriving Thursday'
              The agent then composed a response with tracking link
              Total: 3 LLM calls, 1 tool call, 2,400 tokens, $0.007"
```

### Decision Tree: Which Platform for Which Stack

```
Using LangChain/LangGraph?
  │
  ├── Need deepest LangGraph integration + CI eval gates?
  │   └── LangSmith (native integration, zero overhead)
  │
  ├── Need open-source + self-hosted + data residency?
  │   └── Langfuse (open-source, Postgres + ClickHouse)
  │
  └── Need vendor-neutral standard?
      └── OpenTelemetry + OpenLLMetry
  
Using custom framework?
  │
  ├── Need LLM-specific features (prompt mgmt, eval)?
  │   └── Langfuse (framework-agnostic, most flexible)
  │
  └── Need to integrate with existing APM (Datadog, New Relic)?
      └── OpenTelemetry (exports to any backend)

Enterprise with compliance requirements?
  └── Langfuse self-hosted (full data control)
      OR OpenTelemetry → your own backend
```

## Production Implementation

```python
"""
Agent observability architecture -- composable tracing + metrics + alerts.
"""
from dataclasses import dataclass, field
from typing import Any, Optional
import time
import uuid


@dataclass
class AgentSpan:
    """A single span in an agent trace."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str                    # "llm_call", "tool_call", "reasoning"
    agent_name: str
    start_time: float
    end_time: float = 0
    
    # LLM-specific
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    # Tool-specific
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    tool_result: str = ""
    tool_success: bool = True
    
    # Quality
    finish_reason: str = ""      # "stop", "length", "tool_calls", "content_filter"
    
    # Metadata
    status: str = "ok"           # "ok", "error"
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000
    
    @property
    def cost_usd(self) -> float:
        """Estimate cost based on model pricing."""
        # GPT-4o pricing as example
        pricing = {
            "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
            "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
            "claude-sonnet-4-20250514": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
        }
        rates = pricing.get(self.model, {"input": 0, "output": 0})
        return (self.prompt_tokens * rates["input"] +
                self.completion_tokens * rates["output"])


class AgentTracer:
    """Lightweight agent tracer that can export to any backend."""
    
    def __init__(self, agent_name: str, export_fn=None):
        self.agent_name = agent_name
        self.export_fn = export_fn or self._default_export
        self._active_traces: dict[str, list[AgentSpan]] = {}
    
    def start_trace(self) -> str:
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        self._active_traces[trace_id] = []
        return trace_id
    
    def start_span(
        self,
        trace_id: str,
        name: str,
        parent_span_id: Optional[str] = None,
        **kwargs,
    ) -> AgentSpan:
        span = AgentSpan(
            trace_id=trace_id,
            span_id=f"span_{uuid.uuid4().hex[:8]}",
            parent_span_id=parent_span_id,
            name=name,
            agent_name=self.agent_name,
            start_time=time.time(),
            **kwargs,
        )
        self._active_traces.setdefault(trace_id, []).append(span)
        return span
    
    def end_span(self, span: AgentSpan, **kwargs):
        span.end_time = time.time()
        for k, v in kwargs.items():
            setattr(span, k, v)
    
    def end_trace(self, trace_id: str):
        spans = self._active_traces.pop(trace_id, [])
        self.export_fn(trace_id, spans)
    
    def _default_export(self, trace_id: str, spans: list[AgentSpan]):
        """Default: print trace summary."""
        total_tokens = sum(s.prompt_tokens + s.completion_tokens for s in spans)
        total_cost = sum(s.cost_usd for s in spans)
        total_duration = sum(s.duration_ms for s in spans)
        
        print(f"\n=== Trace {trace_id} ===")
        print(f"Spans: {len(spans)} | Tokens: {total_tokens} | "
              f"Cost: ${total_cost:.4f} | Duration: {total_duration:.0f}ms")
        for span in spans:
            indent = "  " if span.parent_span_id else ""
            print(f"{indent}[{span.name}] {span.tool_name or span.model} "
                  f"({span.duration_ms:.0f}ms)")
```

## Decision Tree / When to Use

- **Every production agent needs observability** -- the question is which level
- **Level 1 (Minimum)**: Trace ID + latency + token count + error rate
- **Level 2 (Standard)**: Full traces + per-tool metrics + cost tracking
- **Level 3 (Production)**: Full traces + eval scores + alerting + prompt versioning

## When NOT to Use

- Never skip observability for production agents
- Development can use minimal (Level 1) tracing

## Tradeoffs

| Platform | Best For | Data Residency | Cost | Overhead |
|----------|---------|----------------|------|----------|
| **Langfuse** | Open-source, self-hosted | Full control | Free (self-hosted) | ~5ms |
| **LangSmith** | LangChain/LangGraph native | US cloud | Paid | ~0ms (async) |
| **OpenTelemetry** | Vendor-neutral | Your choice | Varies by backend | ~2ms |
| **Phoenix (Arize)** | Local development + debugging | Local | Free | ~5ms |
| **Helicone** | Simple proxy-based, minimal code | US cloud | Freemium | ~10ms |

## Real-World Examples

1. **Langfuse**: 2,300+ companies use it for agent tracing, prompt management, and evaluation scoring.
2. **LangSmith**: Zero-overhead tracing for LangGraph agents with integrated dataset management and CI eval gates.
3. **OpenLLMetry**: Auto-instruments LLM calls and exports to any OTel-compatible backend.

## Failure Modes

| Failure | Cause | Impact | Prevention |
|---------|-------|--------|------------|
| **Trace loss** | Async export drops spans | Cannot debug production issues | Buffered export with retry |
| **Cost blindness** | No token tracking | Surprise bills | Per-request cost tracking + alerts |
| **Alert fatigue** | Too many alerts, wrong thresholds | Real issues missed | Tune thresholds based on baselines |
| **Observability overhead** | Synchronous tracing slows agent | Latency increase | Async export, sampling for high-volume |

## Source(s) and Further Reading

- Langfuse Documentation: https://langfuse.com/docs
- LangSmith Documentation: https://docs.smith.langchain.com/
- OpenTelemetry for LLMs: https://openllmetry.com/
- Arize Phoenix: https://docs.arize.com/phoenix
- EDDOps paper (arXiv:2411.13768): Eval-driven development for LLM systems
