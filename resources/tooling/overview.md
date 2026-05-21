# Tool Design for Agentic Systems
> Tool design is the #1 cause of production agent failures -- not prompts, not models, not orchestration.

## What It Is

Tools are the functions, APIs, and executables that an LLM agent can invoke to interact with the outside world. In an agentic system, the model generates structured function calls; the runtime executes them and returns results. Every external action -- reading a database, sending an email, running code -- flows through a tool.

Tool design encompasses the schema definition (what the LLM sees), the implementation (what actually runs), and the reliability layer (what catches failures). When any of these three are poorly designed, the agent breaks in production -- not during demos.

## Why Tool Design Is the #1 Cause of Agent Failures

Research from Anthropic's production deployments and the arXiv paper "9 Practices for Production Agentic Workflows" identifies tool failures as the dominant failure mode:

1. **Ambiguous tool descriptions** cause the LLM to pick the wrong tool 30-40% of the time
2. **Multi-action tools** (do-everything functions) confuse the LLM about required parameters
3. **Raw exception returns** halt the agent -- the LLM cannot self-correct from a Python traceback
4. **Missing idempotency** causes duplicate side effects when the LLM retries (which it will)
5. **No circuit breakers** means one broken API takes down the entire agent

## The 4 Non-Negotiables

Every tool in a production agentic system must satisfy these four properties:

### 1. Single Responsibility
One tool = one action. Never build `manage_database(action="insert"|"delete"|"query", ...)`. Split into `query_database()`, `insert_record()`, `delete_record()`. LLMs are dramatically better at selecting from specific tools than at correctly parameterizing multi-mode tools.

### 2. Idempotency
LLMs retry. Network failures cause retries. Orchestration loops cause retries. Every tool with side effects must produce the same result when called multiple times with the same input. Use idempotency keys, upsert semantics, and conditional writes.

### 3. Error Contracts
Never return raw exceptions. Return structured errors with:
- Error type (validation, timeout, rate_limit, auth, internal)
- Human-readable message the LLM can use to self-correct
- Field-level errors for validation failures
- Example of correct format when input was wrong

### 4. Schemas as Documentation
The JSON Schema IS the documentation. The LLM reads `description` fields, not your README. Write tool and parameter descriptions like you're explaining to a careful but literal junior developer who has never seen your API.

## The Per-Tool Reliability Stack (10-Item Checklist)

Every tool should implement this stack. Score your tools against it:

| # | Check | Why |
|---|-------|-----|
| 1 | Single responsibility | LLM tool selection accuracy |
| 2 | JSON Schema with descriptions | LLM understands parameters |
| 3 | Input validation before execution | Fail fast, clear errors |
| 4 | Idempotency key or upsert pattern | Safe retries |
| 5 | Structured error responses | LLM self-correction |
| 6 | Timeout enforcement (30s default) | Prevent agent hangs |
| 7 | Circuit breaker (per-tool) | Isolate failures |
| 8 | Retry with exponential backoff | Transient failure recovery |
| 9 | Rate limit awareness (429 handling) | API sustainability |
| 10 | Logging with trace_id + tool_name | Production debugging |

## How It Works: Tool Execution Flow

```
Agent Loop
  |
  v
LLM generates tool_call(name, arguments)
  |
  v
[Schema Validation] -- fail --> Return structured validation error
  |
  v
[Circuit Breaker Check] -- open --> Return "service unavailable, try alternative"
  |
  v
[Rate Limit Check] -- exceeded --> Return "rate limited, retry after X seconds"
  |
  v
[Execute Tool with Timeout]
  |           |
  success     failure
  |           |
  v           v
Return       [Classify Error]
result         |
               |-- retryable? --> [Exponential Backoff + Retry]
               |-- validation? --> Return field-level error + example
               |-- auth? --> Return "auth failed, cannot retry"
               |-- timeout? --> Return "timed out after Xs"
               |-- internal? --> Return generic safe message
```

## Production Implementation

```python
"""Base tool infrastructure for agentic systems."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time
import logging
import json

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    INTERNAL = "internal"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class ToolError:
    error_type: ErrorType
    message: str
    field_errors: dict[str, str] = field(default_factory=dict)
    correct_format_example: Optional[dict] = None
    retry_after_seconds: Optional[int] = None

    def to_dict(self) -> dict:
        result = {
            "success": False,
            "error_type": self.error_type.value,
            "message": self.message,
        }
        if self.field_errors:
            result["field_errors"] = self.field_errors
        if self.correct_format_example:
            result["correct_format_example"] = self.correct_format_example
        if self.retry_after_seconds:
            result["retry_after_seconds"] = self.retry_after_seconds
        return result


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: Optional[ToolError] = None

    def to_dict(self) -> dict:
        if self.success:
            return {"success": True, "data": self.data}
        return self.error.to_dict()


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    half_open_max_calls: int = 1

    _failure_count: int = 0
    _last_failure_time: float = 0
    _state: CircuitState = CircuitState.CLOSED
    _half_open_calls: int = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time > self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    def record_success(self):
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN

    def allow_request(self) -> bool:
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            return self._half_open_calls <= self.half_open_max_calls
        return False


class BaseTool(ABC):
    """Base class for all agent tools with built-in reliability."""

    name: str
    description: str
    timeout_seconds: int = 30

    def __init__(self):
        self.circuit_breaker = CircuitBreaker()
        self._call_count = 0

    @abstractmethod
    def get_schema(self) -> dict:
        """Return JSON Schema for this tool."""
        pass

    @abstractmethod
    def validate_input(self, **kwargs) -> Optional[ToolError]:
        """Validate input before execution. Return None if valid."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool action."""
        pass

    def __call__(self, trace_id: str = "", **kwargs) -> dict:
        """Full execution pipeline with all reliability layers."""
        self._call_count += 1

        # Layer 1: Circuit breaker
        if not self.circuit_breaker.allow_request():
            logger.warning(f"[{trace_id}] Circuit open for tool={self.name}")
            return ToolError(
                error_type=ErrorType.CIRCUIT_OPEN,
                message=f"Tool '{self.name}' is temporarily unavailable. "
                        f"Too many recent failures. Try an alternative approach.",
                retry_after_seconds=self.circuit_breaker.recovery_timeout,
            ).to_dict()

        # Layer 2: Input validation
        validation_error = self.validate_input(**kwargs)
        if validation_error:
            logger.info(f"[{trace_id}] Validation failed for tool={self.name}: {validation_error.message}")
            return validation_error.to_dict()

        # Layer 3: Execute with timeout
        try:
            start = time.time()
            result = self.execute(**kwargs)
            elapsed = time.time() - start

            if elapsed > self.timeout_seconds:
                self.circuit_breaker.record_failure()
                return ToolError(
                    error_type=ErrorType.TIMEOUT,
                    message=f"Tool '{self.name}' timed out after {self.timeout_seconds}s.",
                ).to_dict()

            if result.success:
                self.circuit_breaker.record_success()
            else:
                self.circuit_breaker.record_failure()

            logger.info(f"[{trace_id}] tool={self.name} success={result.success} elapsed={elapsed:.2f}s")
            return result.to_dict()

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"[{trace_id}] tool={self.name} exception={str(e)}")
            return ToolError(
                error_type=ErrorType.INTERNAL,
                message=f"Tool '{self.name}' encountered an internal error. "
                        f"The operation may not have completed. "
                        f"Try rephrasing your request or use a different approach.",
            ).to_dict()
```

## Decision Tree: Designing a New Tool

```
Should I build this tool?
  |
  |-- Does the agent need to interact with an external system?
  |     YES --> Build a tool
  |     NO  --> Can the LLM do it with just text? --> Don't build a tool
  |
  v
How should I scope it?
  |
  |-- Does it do more than one thing?
  |     YES --> Split into multiple tools
  |     NO  --> Good, single responsibility
  |
  v
Does it have side effects?
  |
  YES --> Implement idempotency key pattern
  |       Add confirmation step for destructive actions
  NO  --> Read-only tools are simpler, still need timeout + circuit breaker
  |
  v
What errors can occur?
  |
  |-- List every error type
  |-- Map each to ErrorType enum
  |-- Write specific messages that help the LLM self-correct
  |-- Include correct-format examples for validation errors
  |
  v
Write the schema
  |
  |-- description: What does this tool do? (1-2 sentences)
  |-- Each parameter: What is it? What format? What are valid values?
  |-- required vs optional: Mark clearly
  |-- Add examples in descriptions
```

## When NOT to Build a Tool

- **The LLM can do it natively**: Text transformation, summarization, translation, math
- **It requires multi-step orchestration**: That's a workflow, not a tool. Build the individual steps as tools.
- **It's a compound action**: `create_user_and_send_welcome_email()` should be two tools
- **The API is unstable**: Wait until the API is stable. Changing tool schemas mid-deployment breaks agent behavior.

## Tradeoffs

| Decision | More Tools | Fewer Tools |
|----------|-----------|-------------|
| LLM selection accuracy | Higher (specific names) | Lower (ambiguous) |
| Context window usage | More tokens for schemas | Fewer tokens |
| Maintenance burden | More code to maintain | Less code |
| Error isolation | Per-tool circuit breakers | Blast radius larger |
| Testing surface | More unit tests | Fewer tests |

**Rule of thumb**: 15-25 tools is the sweet spot for most agents. Beyond 40, the LLM's tool selection accuracy degrades measurably.

## Real-World Examples

### Example 1: Stripe Payment Agent
```
BAD:  manage_payments(action, amount, currency, customer_id, ...)
GOOD: create_payment_intent(amount, currency, customer_id)
      capture_payment(payment_intent_id)
      refund_payment(payment_intent_id, amount)
      get_payment_status(payment_intent_id)
```

### Example 2: GitHub PR Agent
```
BAD:  github_api(method, endpoint, body)
GOOD: create_pull_request(repo, title, head, base, body)
      list_pull_requests(repo, state, author)
      merge_pull_request(repo, pr_number, merge_method)
      add_pr_review(repo, pr_number, body, event)
```

## Failure Modes

| Failure | Symptom | Root Cause | Fix |
|---------|---------|------------|-----|
| Wrong tool selected | Agent calls `delete` when it should `update` | Ambiguous descriptions | Rewrite descriptions, add "Use this when..." |
| Infinite retry loop | Agent retries the same failing call 20+ times | No max retry limit | Cap at 3 retries + circuit breaker |
| Cascading failure | All tools fail when one API is down | Shared circuit breaker | Per-tool circuit breakers |
| Silent data corruption | Duplicate records created | Missing idempotency | Idempotency keys on all write tools |
| Agent hangs | No response for minutes | No timeout | 30s default timeout on all tools |
| Context overflow | Too many tool schemas | 50+ tools registered | Reduce to 15-25, use dynamic tool loading |

## Source(s) and Further Reading

- "9 Practices for Production Agentic Workflows" (arXiv, 2025) -- production failure taxonomy
- Anthropic, "Building Effective Agents" (2024) -- tool design principles
- OpenAI, "Function Calling Guide" (2024) -- JSON Schema best practices for LLMs
- LangChain Tool documentation -- BaseTool patterns
- CrewAI Tool documentation -- tool reliability patterns
- Stripe Agent Toolkit -- production tool design reference implementation
