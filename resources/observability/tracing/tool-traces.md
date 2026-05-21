# Tool Traces: Per-Tool Observability

> When an agent fails, it is usually because a tool returned unexpected data -- but without per-tool traces, you will blame the LLM when the real culprit is a flaky API.

## What It Is

Tool traces capture the full lifecycle of every tool invocation an agent makes: the tool name, arguments passed, result returned, latency, success/failure status, and the credential used. This enables correlating tool failures with reasoning failures -- when the agent makes a bad decision, tool traces reveal whether the decision was based on bad data from a tool.

## How It Works

### What to Capture Per Tool Call

| Field | Description | Why It Matters |
|-------|------------|----------------|
| `tool_name` | Tool identifier | Which tool was called |
| `tool_version` | Tool/API version | Detect version-related failures |
| `args` | Arguments passed | What the agent asked for |
| `args_hash` | Hash of args (for PII) | Audit without storing sensitive data |
| `result` | Return value | What data the agent received |
| `result_size_bytes` | Response payload size | Detect oversized responses |
| `latency_ms` | Execution duration | Detect slow tools |
| `success` | Boolean | Did the call succeed |
| `error_type` | Error classification | Timeout, auth, rate_limit, not_found, server_error |
| `http_status` | HTTP status (if API) | Correlate with API health |
| `retry_count` | Number of retries | Detect flaky tools |
| `credential_token_id` | Token used (not value) | Audit credential usage |
| `trace_id` | Parent agent trace | Correlate with agent reasoning |
| `span_id` | This span's ID | For hierarchical traces |

### Correlating Tool Failures with Reasoning Failures

```
Scenario: Agent gives wrong answer about order status

Without tool traces:
  "The agent said the order was delivered, but it's actually still shipping."
  → Team blames the LLM. Prompt engineering begins. No improvement.

With tool traces:
  Span: tool:get_order_status
    args: {"order_id": "ORD-789"}
    result: {"status": "delivered", "tracking": "1Z999..."} ← STALE CACHE
    latency: 2ms ← suspiciously fast (cache hit)
    
  → Root cause: Order status API returned cached data (2ms vs normal 200ms)
  → Fix: Bypass cache for status queries, not prompt changes
```

## Production Implementation

```python
"""
Per-tool tracing with failure correlation.
"""
import time
import functools
import logging
from typing import Any, Callable, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ToolTrace:
    tool_name: str
    args: dict
    result: Any = None
    success: bool = True
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    latency_ms: float = 0
    retry_count: int = 0
    http_status: Optional[int] = None
    result_size_bytes: int = 0
    trace_id: str = ""
    span_id: str = ""


class ToolTracer:
    """Wraps tool functions with automatic tracing."""
    
    def __init__(self, export_fn: Optional[Callable] = None):
        self.export_fn = export_fn or self._default_export
        self.tool_stats: dict[str, dict] = {}  # Per-tool aggregates
    
    def traced_tool(self, func: Callable) -> Callable:
        """Decorator to add tracing to a tool function."""
        tool_name = func.__name__
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start = time.time()
            trace = ToolTrace(
                tool_name=tool_name,
                args=kwargs if kwargs else {"positional": list(args)},
            )
            
            try:
                result = await func(*args, **kwargs)
                trace.result = result
                trace.success = True
                trace.result_size_bytes = len(str(result).encode())
            except TimeoutError as e:
                trace.success = False
                trace.error_type = "timeout"
                trace.error_message = str(e)
                raise
            except PermissionError as e:
                trace.success = False
                trace.error_type = "auth"
                trace.error_message = str(e)
                raise
            except Exception as e:
                trace.success = False
                trace.error_type = type(e).__name__
                trace.error_message = str(e)
                raise
            finally:
                trace.latency_ms = (time.time() - start) * 1000
                self._update_stats(trace)
                self.export_fn(trace)
            
            return result
        
        return wrapper
    
    def _update_stats(self, trace: ToolTrace):
        """Update per-tool aggregate statistics."""
        name = trace.tool_name
        if name not in self.tool_stats:
            self.tool_stats[name] = {
                "total_calls": 0, "failures": 0,
                "total_latency_ms": 0, "max_latency_ms": 0,
                "error_types": {},
            }
        
        stats = self.tool_stats[name]
        stats["total_calls"] += 1
        stats["total_latency_ms"] += trace.latency_ms
        stats["max_latency_ms"] = max(stats["max_latency_ms"], trace.latency_ms)
        
        if not trace.success:
            stats["failures"] += 1
            err = trace.error_type or "unknown"
            stats["error_types"][err] = stats["error_types"].get(err, 0) + 1
    
    def get_tool_health(self) -> dict:
        """Get health metrics for all traced tools."""
        health = {}
        for name, stats in self.tool_stats.items():
            total = stats["total_calls"]
            health[name] = {
                "success_rate": (total - stats["failures"]) / total if total > 0 else 1.0,
                "avg_latency_ms": stats["total_latency_ms"] / total if total > 0 else 0,
                "p_max_latency_ms": stats["max_latency_ms"],
                "total_calls": total,
                "error_breakdown": stats["error_types"],
            }
        return health
    
    def _default_export(self, trace: ToolTrace):
        status = "OK" if trace.success else f"FAIL({trace.error_type})"
        logger.info(
            f"TOOL [{trace.tool_name}] {status} "
            f"latency={trace.latency_ms:.0f}ms "
            f"size={trace.result_size_bytes}B"
        )


# Usage
tracer = ToolTracer()

@tracer.traced_tool
async def search_orders(customer_id: str, status: str = "all") -> list[dict]:
    """Search orders for a customer."""
    # Actual implementation
    pass

@tracer.traced_tool
async def get_order_details(order_id: str) -> dict:
    """Get details for a specific order."""
    pass

# After some calls:
health = tracer.get_tool_health()
# {"search_orders": {"success_rate": 0.95, "avg_latency_ms": 234, ...}}
```

## Decision Tree / When to Use

- **Every agent with tool calling** -- Tool traces are mandatory for debugging
- **Especially important** when tools call external APIs (failure correlation)
- **Less critical** for purely deterministic local tools (math, formatting)

## When NOT to Use

- Agents without tool calling do not need tool traces (but still need LLM traces)

## Tradeoffs

| Detail Level | Storage | Debug Value | Performance |
|-------------|---------|-------------|------------|
| Name + success + latency | ~100B | Basic | None |
| + args + result summary | ~1KB | Good | ~1ms |
| + full result + retries | ~10KB | Excellent | ~2ms |

## Failure Modes

| Failure | Cause | Impact | Prevention |
|---------|-------|--------|------------|
| **Tool flakiness masked** | No per-tool success tracking | Agent blamed for tool failures | Per-tool success rate dashboard |
| **Stale cache blamed on LLM** | Cache returns old data, no latency flag | Wrong debugging direction | Track latency anomalies (too fast = cache) |
| **PII in tool traces** | User data in args/results | Privacy violation | Hash/redact sensitive fields |
| **Missing correlation** | Tool trace not linked to agent trace | Cannot find root cause | Always propagate trace_id |

## Source(s) and Further Reading

- OpenTelemetry Semantic Conventions for LLM Tool Calls
- Langfuse Tool Call Tracing: https://langfuse.com/docs/tracing
- LangSmith Run Types: https://docs.smith.langchain.com/concepts/tracing#run-types
