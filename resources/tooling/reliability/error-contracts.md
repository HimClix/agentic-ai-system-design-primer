# Error Taxonomy for the Reliability Layer
> Classify every error into one of 8 categories. Each category maps to a specific agent behavior: retry, self-correct, escalate, or give up.

## What It Is

The reliability-layer error contract is a classification system that sits between the raw error (HTTP status, exception type, timeout) and the structured error response sent to the LLM. Its job is to take any failure -- from any tool, any service, any error format -- and normalize it into a consistent taxonomy that the agent orchestrator and the LLM can both reason about.

This is distinct from the design-principles error contract (which focuses on making errors useful to the LLM for self-correction). The reliability layer focuses on **error classification** and **automated response** -- what the infrastructure does before the error ever reaches the LLM.

## How It Works

### The Error Classification Pipeline

```
Raw Error (HTTP 503, ConnectionTimeout, ValueError, etc.)
  |
  v
[Error Classifier] -- Maps to ErrorCategory
  |
  v
[Response Router]
  |
  |-- TRANSIENT -> Retry (handled by retry layer, LLM never sees it)
  |-- RATE_LIMIT -> Wait + Retry (use Retry-After)
  |-- VALIDATION -> Return to LLM with field errors (self-correction)
  |-- NOT_FOUND -> Return to LLM with search suggestions
  |-- AUTH -> Return to LLM (cannot retry, inform user)
  |-- CONFLICT -> Return to LLM with current state
  |-- TIMEOUT -> Retry once, then return to LLM
  |-- INTERNAL -> Retry once, then return safe message to LLM
```

### The 8 Error Categories

| Category | HTTP Codes | Auto-Retry? | LLM Action | Agent Behavior |
|----------|-----------|-------------|------------|----------------|
| TRANSIENT | 500, 502, 503, 504 | Yes (3x) | Never sees it | Silent retry |
| RATE_LIMIT | 429 | Yes (with Retry-After) | Waits if told | Respect server timing |
| VALIDATION | 400, 422 | No | Self-correct input | Fix and retry |
| NOT_FOUND | 404 | No | Search differently | Try alternative identifier |
| AUTH | 401, 403 | No | Inform user | Stop, escalate |
| CONFLICT | 409 | No | Check current state | Read-then-decide |
| TIMEOUT | request timeout | Once | Simplify request | Check if operation completed |
| INTERNAL | unclassified | Once | Report to user | Log, alert, fail gracefully |

## Production Implementation

```python
"""Error taxonomy and classification for the reliability layer."""
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Any
import httpx
import logging

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """The 8 error categories for agent tool failures."""
    TRANSIENT = "transient"        # Temporary server issues, auto-retry
    RATE_LIMIT = "rate_limit"      # Too many requests, wait and retry
    VALIDATION = "validation"      # Bad input from LLM, needs self-correction
    NOT_FOUND = "not_found"        # Resource doesn't exist
    AUTH = "auth"                  # Authentication or authorization failure
    CONFLICT = "conflict"          # Resource state conflict
    TIMEOUT = "timeout"            # Operation timed out
    INTERNAL = "internal"          # Unclassified or unexpected error


@dataclass
class ClassifiedError:
    """A classified error with routing metadata."""
    category: ErrorCategory
    message: str
    retryable: bool
    max_retries: int = 0
    retry_after_seconds: Optional[float] = None
    field_errors: Optional[dict[str, str]] = None
    correct_format: Optional[dict] = None
    suggestions: Optional[list[str]] = None
    original_error: Optional[str] = None  # For logging only, never sent to LLM
    
    @property
    def should_auto_retry(self) -> bool:
        """Whether the retry layer should handle this silently."""
        return self.retryable and self.category in {
            ErrorCategory.TRANSIENT,
            ErrorCategory.RATE_LIMIT,
        }
    
    @property
    def should_return_to_llm(self) -> bool:
        """Whether this error should be sent to the LLM."""
        return not self.should_auto_retry or self.max_retries == 0
    
    def to_llm_response(self) -> dict:
        """Format this error for LLM consumption. Strips internal details."""
        response = {
            "success": False,
            "error_type": self.category.value,
            "message": self.message,
        }
        if self.field_errors:
            response["field_errors"] = self.field_errors
        if self.correct_format:
            response["correct_format_example"] = self.correct_format
        if self.retry_after_seconds:
            response["retry_after_seconds"] = int(self.retry_after_seconds)
        if self.suggestions:
            response["suggestions"] = self.suggestions
        return response


class ErrorClassifier:
    """Classifies raw errors into the 8-category taxonomy."""
    
    def classify_http(self, response: httpx.Response, tool_name: str) -> ClassifiedError:
        """Classify an HTTP response error."""
        status = response.status_code
        body = self._safe_body(response)
        
        if status == 429:
            retry_after = self._parse_retry_after(response)
            return ClassifiedError(
                category=ErrorCategory.RATE_LIMIT,
                message=f"Rate limit exceeded for {tool_name}. "
                        f"Wait {retry_after}s before retrying.",
                retryable=True,
                max_retries=3,
                retry_after_seconds=retry_after,
                original_error=f"HTTP 429: {body}",
            )
        
        if status in {500, 502, 503, 504}:
            return ClassifiedError(
                category=ErrorCategory.TRANSIENT,
                message=f"Temporary server error from {tool_name}.",
                retryable=True,
                max_retries=3,
                original_error=f"HTTP {status}: {body}",
            )
        
        if status == 400:
            field_errors = self._extract_field_errors(body)
            return ClassifiedError(
                category=ErrorCategory.VALIDATION,
                message=f"Invalid input for {tool_name}. Check field_errors for details.",
                retryable=False,
                field_errors=field_errors,
                original_error=f"HTTP 400: {body}",
            )
        
        if status == 422:
            field_errors = self._extract_field_errors(body)
            return ClassifiedError(
                category=ErrorCategory.VALIDATION,
                message=f"Validation failed for {tool_name}. See field_errors.",
                retryable=False,
                field_errors=field_errors,
                original_error=f"HTTP 422: {body}",
            )
        
        if status == 401:
            return ClassifiedError(
                category=ErrorCategory.AUTH,
                message=f"Authentication failed for {tool_name}. "
                        f"Cannot retry -- inform the user.",
                retryable=False,
                original_error=f"HTTP 401: {body}",
            )
        
        if status == 403:
            return ClassifiedError(
                category=ErrorCategory.AUTH,
                message=f"Permission denied for {tool_name}. "
                        f"The agent does not have access to this resource.",
                retryable=False,
                original_error=f"HTTP 403: {body}",
            )
        
        if status == 404:
            return ClassifiedError(
                category=ErrorCategory.NOT_FOUND,
                message=f"Resource not found in {tool_name}.",
                retryable=False,
                suggestions=[
                    "Verify the identifier is correct.",
                    "Try searching with a different identifier.",
                ],
                original_error=f"HTTP 404: {body}",
            )
        
        if status == 409:
            return ClassifiedError(
                category=ErrorCategory.CONFLICT,
                message=f"Conflict in {tool_name}. The resource may have been "
                        f"modified by another process.",
                retryable=False,
                suggestions=[
                    "Read the current state of the resource first.",
                    "Then decide whether to proceed with the operation.",
                ],
                original_error=f"HTTP 409: {body}",
            )
        
        # Unknown status codes
        return ClassifiedError(
            category=ErrorCategory.INTERNAL,
            message=f"Unexpected error from {tool_name}.",
            retryable=status >= 500,
            max_retries=1,
            original_error=f"HTTP {status}: {body}",
        )
    
    def classify_exception(self, error: Exception, tool_name: str) -> ClassifiedError:
        """Classify a Python exception."""
        if isinstance(error, TimeoutError) or isinstance(error, asyncio.TimeoutError):
            return ClassifiedError(
                category=ErrorCategory.TIMEOUT,
                message=f"Tool '{tool_name}' timed out. The operation may still "
                        f"be processing. Check the status before retrying.",
                retryable=True,
                max_retries=1,
                suggestions=[
                    "Wait a moment and check if the operation completed.",
                    "Retry with a simpler request.",
                ],
                original_error=f"{type(error).__name__}: {str(error)}",
            )
        
        if isinstance(error, ConnectionError):
            return ClassifiedError(
                category=ErrorCategory.TRANSIENT,
                message=f"Connection failed for {tool_name}. Will retry.",
                retryable=True,
                max_retries=3,
                original_error=f"{type(error).__name__}: {str(error)}",
            )
        
        if isinstance(error, ValueError):
            return ClassifiedError(
                category=ErrorCategory.VALIDATION,
                message=f"Invalid value for {tool_name}: {str(error)}",
                retryable=False,
                original_error=f"ValueError: {str(error)}",
            )
        
        # Unknown exceptions -- never expose internals
        return ClassifiedError(
            category=ErrorCategory.INTERNAL,
            message=f"Tool '{tool_name}' encountered an internal error. "
                    f"You may retry once.",
            retryable=True,
            max_retries=1,
            original_error=f"{type(error).__name__}: {str(error)}",
        )
    
    def _safe_body(self, response: httpx.Response) -> str:
        try:
            return response.text[:500]
        except Exception:
            return "(unable to read response body)"
    
    def _parse_retry_after(self, response: httpx.Response) -> float:
        header = response.headers.get("Retry-After", "")
        try:
            return float(header)
        except (ValueError, TypeError):
            return 5.0  # Default 5 seconds
    
    def _extract_field_errors(self, body: str) -> dict[str, str]:
        """Best-effort extraction of field-level errors from response body."""
        import json
        try:
            data = json.loads(body)
            # Common patterns from different APIs
            if "errors" in data and isinstance(data["errors"], dict):
                return data["errors"]
            if "detail" in data and isinstance(data["detail"], list):
                return {
                    item.get("loc", ["unknown"])[-1]: item.get("msg", "invalid")
                    for item in data["detail"]
                    if isinstance(item, dict)
                }
            if "message" in data:
                return {"_general": data["message"]}
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return {"_general": body[:200]}
```

## Decision Tree: Classifying an Unknown Error

```
Is it an HTTP response?
  |
  YES --> Use status code mapping (see table above)
  |
  NO --> Is it a Python exception?
          |
          |-- TimeoutError -> TIMEOUT
          |-- ConnectionError -> TRANSIENT
          |-- ValueError -> VALIDATION
          |-- TypeError -> VALIDATION
          |-- PermissionError -> AUTH
          |-- FileNotFoundError -> NOT_FOUND
          |-- Everything else -> INTERNAL
```

## When NOT to Use This Taxonomy

- **LLM API errors**: These should be handled by the orchestration layer, not the tool layer. Model rate limits, context overflow, and content filter blocks are not tool errors.
- **Orchestration errors**: Tool selection failures, max-turn limits, and agent loop errors are framework-level concerns.
- **User input errors**: If the user provided bad input (not the LLM), this should be handled in the prompt layer, not the tool error layer.

## Tradeoffs

| Aspect | Strict Classification | Loose Classification |
|--------|----------------------|---------------------|
| Error handling precision | Each category has specific behavior | Fewer categories, simpler |
| Development effort | Must classify every error source | Just retry or fail |
| LLM self-correction | High (specific guidance per type) | Low (generic messages) |
| False classification | Risk of misclassifying edge cases | N/A with fewer categories |
| Debugging | Clear error trails per category | Harder to trace |
| Alert quality | Category-specific alerts | Generic error alerts |

## Real-World Examples

### Error Flow: Stripe API 429
```
1. Tool calls Stripe API
2. Stripe returns HTTP 429 with Retry-After: 2
3. ErrorClassifier -> RATE_LIMIT, retryable=True, retry_after=2.0
4. Retry layer waits 2s, retries
5. Stripe returns HTTP 200
6. LLM never sees the error
```

### Error Flow: Database Record Not Found
```
1. Tool queries database for user_id="usr_xyz"
2. Database returns empty result
3. Tool returns NOT_FOUND classification
4. LLM receives: {"error_type": "not_found", "message": "User 'usr_xyz' not found",
                   "suggestions": ["Search by email using search_users"]}
5. LLM calls search_users(email="...") instead
```

## Failure Modes

| Failure | Cause | Impact | Fix |
|---------|-------|--------|-----|
| Misclassified retryable as non-retryable | Missing status code in map | Transient errors surface to user | Audit classification coverage |
| Misclassified non-retryable as retryable | 400 error retried 3x | Wasted attempts, same error | Strict 4xx = non-retryable rule |
| Information leak in error message | Exception string includes secrets | Security vulnerability | Never pass raw exception to LLM |
| Over-aggressive retry | Everything marked retryable | Overwhelms failing service | Only TRANSIENT and RATE_LIMIT auto-retry |
| Missing error category | New error type not in taxonomy | Falls to INTERNAL (safe default) | Periodic audit of INTERNAL errors |

## Source(s) and Further Reading

- Google Cloud, "Error Model" -- standard error classification for APIs
- gRPC status codes -- canonical error taxonomy for distributed systems
- "9 Practices for Production Agentic Workflows" (arXiv, 2025) -- error handling in agents
- HTTP Status Code Registry (IANA) -- official status code semantics
- Stripe API, "Error Types" -- production error categorization example
