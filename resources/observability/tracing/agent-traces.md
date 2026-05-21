# Agent Traces: Structured Execution Tracing

> A trace is a complete, replayable record of every reasoning step, LLM call, tool invocation, and decision an agent made -- without it, debugging non-deterministic systems is guesswork.

## What It Is

Agent traces are structured, hierarchical records of agent execution that capture every step from user input to final output. Unlike traditional request traces (which capture service-to-service calls), agent traces capture the **reasoning chain** -- what the agent observed, what it considered, what tools it called, and what decisions it made.

## How It Works

### What to Capture Per Step

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `trace_id` | string | Unique ID for the entire agent run | `trace_abc123def456` |
| `span_id` | string | Unique ID for this step | `span_001` |
| `parent_span_id` | string | Parent step (for nesting) | `span_000` |
| `agent_name` | string | Which agent is executing | `customer_support_v2` |
| `agent_version` | string | Agent deployment version | `v2.3.1-abc123` |
| `step_type` | enum | Category of step | `llm_call`, `tool_call`, `retrieval`, `decision`, `human_review` |
| `timestamp_start` | datetime | When step began | `2025-05-19T14:30:22.543Z` |
| `timestamp_end` | datetime | When step completed | `2025-05-19T14:30:24.891Z` |
| `input` | object | Input to this step | `{"messages": [...], "context": {...}}` |
| `output` | object | Output from this step | `{"response": "...", "tool_calls": [...]}` |
| `model_name` | string | LLM model used | `gpt-4o-2024-08-06` |
| `model_parameters` | object | Temperature, max_tokens, etc. | `{"temperature": 0.1, "max_tokens": 1024}` |
| `prompt_template_id` | string | Which prompt version was used | `prompt_cs_v12` |
| `tokens_input` | int | Input/prompt tokens | `1,847` |
| `tokens_output` | int | Output/completion tokens | `342` |
| `tokens_total` | int | Total tokens consumed | `2,189` |
| `cost_usd` | float | Estimated cost of this step | `0.0043` |
| `finish_reason` | string | Why the model stopped | `stop`, `length`, `tool_calls`, `content_filter` |
| `tool_name` | string | Tool called (if tool_call) | `search_orders` |
| `tool_args` | object | Arguments passed to tool | `{"order_id": "ORD-123"}` |
| `tool_result` | object | Tool return value | `{"status": "shipped"}` |
| `tool_latency_ms` | float | Tool execution time | `234.5` |
| `tool_success` | bool | Whether tool call succeeded | `true` |
| `error` | object | Error details if failed | `{"type": "ToolError", "message": "..."}` |
| `metadata` | object | Custom key-value pairs | `{"user_tier": "premium", "region": "us-east"}` |

### Hierarchical Span Model for Multi-Agent Systems

```
Trace: trace_abc123 (Customer Support Session)
│
├── Span: orchestrator_call (Supervisor Agent)
│   ├── input: "User asks about refund for order ORD-789"
│   ├── model: gpt-4o
│   └── decision: "Route to refund_agent"
│
├── Span: refund_agent_call (Refund Agent)
│   │
│   ├── Span: llm_reasoning (Planning)
│   │   ├── model: gpt-4o
│   │   ├── tokens: 1,200
│   │   └── reasoning: "Need to check order status first"
│   │
│   ├── Span: tool_call (get_order)
│   │   ├── tool: get_order
│   │   ├── args: {"order_id": "ORD-789"}
│   │   ├── result: {"status": "delivered", "amount": 49.99}
│   │   └── latency: 120ms
│   │
│   ├── Span: llm_reasoning (Decision)
│   │   ├── model: gpt-4o
│   │   ├── tokens: 800
│   │   └── reasoning: "Order delivered, eligible for refund within 30 days"
│   │
│   ├── Span: tool_call (check_refund_policy)
│   │   ├── tool: check_refund_policy
│   │   ├── args: {"order_id": "ORD-789"}
│   │   ├── result: {"eligible": true, "reason": "within_window"}
│   │   └── latency: 45ms
│   │
│   └── Span: llm_response (Final Response)
│       ├── model: gpt-4o
│       ├── tokens: 600
│       └── output: "I can process your refund for $49.99..."
│
└── Trace Summary:
    ├── total_spans: 6
    ├── total_tokens: 2,600
    ├── total_cost: $0.0052
    ├── total_latency: 3,240ms
    └── tools_called: [get_order, check_refund_policy]
```

## Production Implementation

```python
"""
Production agent tracing with Langfuse integration.
"""
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional, Generator
from langfuse import Langfuse


class ProductionTracer:
    """
    Agent tracer with hierarchical spans and Langfuse export.
    """
    
    def __init__(self, agent_name: str, agent_version: str):
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.langfuse = Langfuse()  # Reads LANGFUSE_* env vars
    
    def start_trace(
        self,
        user_id: str,
        session_id: str,
        metadata: Optional[dict] = None,
    ):
        """Start a new trace for an agent session."""
        trace = self.langfuse.trace(
            name=self.agent_name,
            user_id=user_id,
            session_id=session_id,
            metadata={
                "agent_version": self.agent_version,
                **(metadata or {}),
            },
        )
        return trace
    
    @contextmanager
    def llm_span(
        self,
        trace,
        model: str,
        input_messages: list[dict],
        parent_span=None,
        prompt_template_id: str = "",
    ) -> Generator:
        """Context manager for LLM call spans."""
        generation = trace.generation(
            name="llm_call",
            model=model,
            input=input_messages,
            metadata={
                "prompt_template_id": prompt_template_id,
                "agent_version": self.agent_version,
            },
        )
        
        result = {"output": None, "tokens": {}, "finish_reason": ""}
        
        try:
            yield result
        finally:
            generation.end(
                output=result.get("output"),
                usage={
                    "input": result.get("tokens", {}).get("input", 0),
                    "output": result.get("tokens", {}).get("output", 0),
                },
                metadata={
                    "finish_reason": result.get("finish_reason", ""),
                },
            )
    
    @contextmanager
    def tool_span(
        self,
        trace,
        tool_name: str,
        tool_args: dict,
        parent_span=None,
    ) -> Generator:
        """Context manager for tool call spans."""
        span = trace.span(
            name=f"tool:{tool_name}",
            input=tool_args,
            metadata={"tool_name": tool_name},
        )
        
        result = {"output": None, "success": True, "error": None}
        
        try:
            yield result
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            span.update(
                level="ERROR",
                status_message=str(e),
            )
            raise
        finally:
            span.end(
                output=result.get("output"),
                metadata={
                    "success": result["success"],
                    "error": result.get("error"),
                },
            )


# Usage with a LangGraph agent
async def traced_agent_step(state: dict, tracer: ProductionTracer):
    """Example of a fully traced agent step."""
    trace = tracer.start_trace(
        user_id=state["user_id"],
        session_id=state["session_id"],
    )
    
    # Trace LLM call
    with tracer.llm_span(trace, model="gpt-4o", input_messages=state["messages"]) as result:
        # Make LLM call
        response = await llm.ainvoke(state["messages"])
        result["output"] = response.content
        result["tokens"] = {"input": response.usage.input_tokens, "output": response.usage.output_tokens}
        result["finish_reason"] = response.finish_reason
    
    # Trace tool call
    if response.tool_calls:
        for tool_call in response.tool_calls:
            with tracer.tool_span(trace, tool_call["name"], tool_call["args"]) as tool_result:
                output = await execute_tool(tool_call["name"], tool_call["args"])
                tool_result["output"] = output
    
    return state
```

## Decision Tree / When to Use

- **Every production agent** -- Minimum Level 1 (trace_id, latency, tokens, error)
- **Agents with tools** -- Level 2 (full tool call traces with args and results)
- **Multi-agent systems** -- Level 3 (hierarchical spans showing agent delegation)
- **Compliance-sensitive** -- Level 3 + full reasoning capture

## When NOT to Use

- Never skip tracing in production
- Local development can use simplified `print`-based tracing

## Tradeoffs

| Tracing Level | Debug Value | Storage Cost | Performance Impact | Compliance |
|--------------|------------|-------------|-------------------|-----------|
| Trace ID only | Minimal | ~1 KB/request | None | Insufficient |
| + Latency + tokens | Basic | ~5 KB/request | None | Minimal |
| + Full spans | Good | ~50 KB/request | ~2ms | Good |
| + Reasoning + I/O | Excellent | ~200 KB/request | ~5ms | Full |

## Failure Modes

| Failure | Cause | Impact | Prevention |
|---------|-------|--------|------------|
| **Missing spans** | Async export drops data | Incomplete traces | Buffered export with retry; sync for critical paths |
| **PII in traces** | User data in tool args/results | Privacy violation | PII scrubbing before export |
| **Trace explosion** | High-volume agent creates massive traces | Storage cost, query slowness | Sampling for high-volume; summarize long conversations |
| **Clock skew** | Distributed system time differences | Incorrect span ordering | Use monotonic clocks; NTP sync |

## Source(s) and Further Reading

- Langfuse Tracing Docs: https://langfuse.com/docs/tracing
- LangSmith Tracing Concepts: https://docs.smith.langchain.com/concepts/tracing
- OpenTelemetry Trace Specification: https://opentelemetry.io/docs/specs/otel/trace/
- Arize Phoenix Tracing: https://docs.arize.com/phoenix/tracing
