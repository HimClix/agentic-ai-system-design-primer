# Per-Tool Circuit Breakers
> A global circuit breaker kills your entire agent when one API goes down. Per-tool circuit breakers isolate failures so the agent can route around broken tools.

## What It Is

A circuit breaker is a reliability pattern that stops calling a failing service after N consecutive failures, waits for a recovery period, then probes with a single test call before resuming full traffic. In agentic systems, circuit breakers must be per-tool, not global -- because an agent typically has 15-25 tools backed by different services, and one service going down should not disable all tools.

The circuit breaker has three states:
- **Closed** (normal): All calls pass through. Failures are counted.
- **Open** (tripped): All calls are rejected immediately. Returns a structured error telling the LLM to use an alternative approach.
- **Half-Open** (probing): One test call is allowed through. If it succeeds, the circuit closes. If it fails, the circuit reopens.

## How It Works

### Why Global Circuit Breakers Fail for Agents

```
Agent has 20 tools:
  - 5 tools call Stripe API
  - 5 tools call GitHub API
  - 5 tools call internal services
  - 5 tools do local computation

Scenario: Stripe API goes down

Global circuit breaker:
  -> 5 Stripe tools fail
  -> Global breaker trips
  -> ALL 20 tools disabled  <-- Agent is completely broken
  -> GitHub tools, internal tools, local tools all blocked

Per-tool circuit breaker:
  -> 5 Stripe tools fail
  -> 5 Stripe circuit breakers trip
  -> 15 other tools still work  <-- Agent routes around failure
  -> LLM told "payment tools unavailable, try X instead"
```

### State Machine

```
        success
    ┌──────────────┐
    │              │
    v              │
 [CLOSED] ──failures >= threshold──> [OPEN]
    ^                                  │
    │                                  │
    │                         recovery_timeout
    │                         expires
    │                                  │
    │                                  v
    └────── probe succeeds ──── [HALF_OPEN]
                                       │
                                  probe fails
                                       │
                                       v
                                    [OPEN]
```

## Production Implementation

```python
"""Per-tool circuit breaker with sliding window failure detection."""
import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for a per-tool circuit breaker."""
    failure_threshold: int = 5          # Failures to trip the breaker
    failure_window_seconds: int = 60    # Sliding window for counting failures
    recovery_timeout_seconds: int = 30  # How long to stay open before probing
    half_open_max_calls: int = 1        # Probe calls in half-open state
    success_threshold: int = 2          # Successes in half-open to close


@dataclass
class CircuitBreaker:
    """Per-tool circuit breaker with sliding window failure tracking."""
    
    tool_name: str
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    
    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_timestamps: deque = field(default_factory=deque, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _half_open_successes: int = field(default=0, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    
    # Metrics
    _total_calls: int = field(default=0, init=False)
    _total_blocked: int = field(default=0, init=False)
    _total_failures: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.recovery_timeout_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    self._half_open_successes = 0
                    logger.info(
                        f"Circuit breaker [{self.tool_name}]: OPEN -> HALF_OPEN "
                        f"(after {elapsed:.0f}s recovery timeout)"
                    )
            return self._state

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        current_state = self.state
        with self._lock:
            self._total_calls += 1
            
            if current_state == CircuitState.CLOSED:
                return True
            
            if current_state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            
            # OPEN
            self._total_blocked += 1
            return False

    def record_success(self):
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_timestamps.clear()
                    logger.info(f"Circuit breaker [{self.tool_name}]: HALF_OPEN -> CLOSED")
            # In CLOSED state, success just resets things naturally

    def record_failure(self):
        """Record a failed call."""
        now = time.time()
        with self._lock:
            self._total_failures += 1
            
            if self._state == CircuitState.HALF_OPEN:
                # Probe failed, reopen
                self._state = CircuitState.OPEN
                self._last_failure_time = now
                logger.warning(
                    f"Circuit breaker [{self.tool_name}]: HALF_OPEN -> OPEN "
                    f"(probe call failed)"
                )
                return

            # Sliding window: remove old failures
            window_start = now - self.config.failure_window_seconds
            while self._failure_timestamps and self._failure_timestamps[0] < window_start:
                self._failure_timestamps.popleft()
            
            self._failure_timestamps.append(now)
            
            if len(self._failure_timestamps) >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
                self._last_failure_time = now
                logger.warning(
                    f"Circuit breaker [{self.tool_name}]: CLOSED -> OPEN "
                    f"({len(self._failure_timestamps)} failures in "
                    f"{self.config.failure_window_seconds}s window)"
                )

    def get_error_response(self) -> dict:
        """Return a structured error for the LLM when circuit is open."""
        return {
            "success": False,
            "error_type": "circuit_open",
            "message": (
                f"Tool '{self.tool_name}' is temporarily unavailable due to "
                f"repeated failures. The underlying service may be experiencing "
                f"issues. Try an alternative approach or wait "
                f"{self.config.recovery_timeout_seconds} seconds."
            ),
            "retry_after_seconds": self.config.recovery_timeout_seconds,
            "suggestions": [
                f"Wait {self.config.recovery_timeout_seconds} seconds and try again.",
                "Use an alternative tool if available.",
                "Inform the user that this service is temporarily unavailable.",
            ],
        }

    @property
    def metrics(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "state": self.state.value,
            "total_calls": self._total_calls,
            "total_blocked": self._total_blocked,
            "total_failures": self._total_failures,
            "recent_failures": len(self._failure_timestamps),
        }


class CircuitBreakerRegistry:
    """Manages per-tool circuit breakers."""

    def __init__(self, default_config: CircuitBreakerConfig = None):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._default_config = default_config or CircuitBreakerConfig()
        self._lock = threading.Lock()

    def get(self, tool_name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create a circuit breaker for a tool."""
        with self._lock:
            if tool_name not in self._breakers:
                self._breakers[tool_name] = CircuitBreaker(
                    tool_name=tool_name,
                    config=config or self._default_config,
                )
            return self._breakers[tool_name]

    def get_all_metrics(self) -> list[dict]:
        """Get metrics for all circuit breakers."""
        return [cb.metrics for cb in self._breakers.values()]


# Usage in tool execution:
registry = CircuitBreakerRegistry(
    default_config=CircuitBreakerConfig(
        failure_threshold=5,
        failure_window_seconds=60,
        recovery_timeout_seconds=30,
    )
)


async def execute_tool(tool_name: str, func: Callable, **kwargs) -> dict:
    """Execute a tool with circuit breaker protection."""
    cb = registry.get(tool_name)

    if not cb.allow_request():
        return cb.get_error_response()

    try:
        result = await func(**kwargs)
        cb.record_success()
        return result
    except Exception as e:
        cb.record_failure()
        raise
```

## Decision Tree: Configuring Circuit Breakers

```
What kind of external service does the tool call?

High-SLA API (Stripe, AWS):
  -> failure_threshold: 10
  -> failure_window: 120s
  -> recovery_timeout: 60s
  (These rarely fail; when they do, it's serious)

Standard SaaS API (GitHub, Slack):
  -> failure_threshold: 5
  -> failure_window: 60s
  -> recovery_timeout: 30s
  (Standard configuration)

Internal microservice:
  -> failure_threshold: 3
  -> failure_window: 30s
  -> recovery_timeout: 15s
  (Faster failure detection, faster recovery)

Flaky/unstable service:
  -> failure_threshold: 3
  -> failure_window: 30s
  -> recovery_timeout: 60s
  (Trip fast, recover slow to avoid flapping)
```

## When NOT to Use Circuit Breakers

- **Local computation tools**: Tools that don't call external services (text processing, math, formatting)
- **Stateless retry scenarios**: When every call is independent and retries are cheap
- **Single-use tools**: Tools called once per session with no retry expectation
- **Development/testing**: Circuit breakers make debugging harder when you want to see every failure

## Tradeoffs

| Aspect | Per-Tool Circuit Breakers | Global Circuit Breaker | No Circuit Breaker |
|--------|--------------------------|----------------------|-------------------|
| Failure isolation | One tool down, others work | All tools down together | No isolation |
| Complexity | Higher (N breakers for N tools) | Simple (one breaker) | None |
| Memory | O(N) for N tools | O(1) | None |
| Recovery speed | Per-tool, fast | All-or-nothing | Immediate (no protection) |
| Cascading failures | Prevented | Partially prevented | Likely |
| Agent availability | High (routes around failures) | Low (all-or-nothing) | Varies |

## Real-World Examples

### E-commerce Agent With 3 Service Failures
```
Stripe API: DOWN (circuit OPEN)
  -> create_payment: "Payment service unavailable, inform user"
  -> get_payment_status: "Payment service unavailable, inform user"

Shipping API: SLOW (circuit CLOSED, some timeouts)
  -> track_shipment: Working but slow
  -> create_shipment: Working but slow

Inventory API: HEALTHY
  -> check_inventory: Working normally
  -> update_inventory: Working normally

Agent response to user: "I can check inventory and track existing
shipments, but the payment system is temporarily unavailable. 
I'll let you know when it's back up."
```

## Failure Modes

| Failure | Cause | Impact | Fix |
|---------|-------|--------|-----|
| Premature tripping | Threshold too low | Tool disabled unnecessarily | Increase failure_threshold |
| Slow recovery | Recovery timeout too long | Tool disabled too long after service recovers | Decrease recovery_timeout |
| Flapping | Service is intermittently failing | Circuit rapidly opens/closes | Increase failure_window, add jitter |
| Memory leak | Failure timestamps accumulate | Memory grows unbounded | Use bounded deque, prune old entries |
| State lost on restart | In-memory circuit state | All breakers reset on deploy | Accept this (conservative default is CLOSED) |

## Source(s) and Further Reading

- Michael Nygard, "Release It!" (2018) -- original circuit breaker pattern
- Martin Fowler, "CircuitBreaker" -- pattern definition and state machine
- "9 Practices for Production Agentic Workflows" (arXiv, 2025) -- per-tool isolation
- Netflix Hystrix (archived) -- production circuit breaker at scale
- Python `circuitbreaker` package -- lightweight implementation reference
