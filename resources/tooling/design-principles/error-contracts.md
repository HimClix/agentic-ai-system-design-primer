# Error Contracts: Structured Errors for LLM Self-Correction
> An LLM cannot self-correct from "TypeError: 'NoneType' object is not subscriptable". It CAN self-correct from {"error_type": "validation", "field_errors": {"date": "Must be YYYY-MM-DD format"}, "example": {"date": "2024-01-15"}}.

## What It Is

An error contract is a standardized format for returning errors from tools to the LLM. Instead of raising exceptions, crashing with tracebacks, or returning generic "something went wrong" messages, every tool returns structured JSON that the LLM can parse and use to self-correct.

The key insight: **LLMs are excellent at self-correction when given the right information.** Field-level errors with correct-format examples enable the LLM to fix its own mistakes in the next turn. Raw Python tracebacks provide zero actionable information to the model.

## How It Works

### The Self-Correction Loop

```
LLM generates tool call with bad input
  |
  v
Tool validates input
  |
  v
Tool returns structured error:
  - error_type: "validation"
  - message: "Invalid date format"
  - field_errors: {"date": "Expected YYYY-MM-DD, got '15/01/2024'"}
  - correct_format_example: {"date": "2024-01-15"}
  |
  v
LLM reads error, understands:
  1. What went wrong (date format)
  2. What it sent ("15/01/2024")
  3. What to send ("2024-01-15")
  |
  v
LLM retries with corrected input
  -> Tool succeeds
```

### Why Tracebacks Break Agents

```python
# ANTI-PATTERN: Raw exception propagated to LLM
def search_orders(start_date: str, end_date: str):
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        raise  # LLM sees:
        # Traceback (most recent call last):
        #   File "tools.py", line 4, in search_orders
        #     start = datetime.strptime(start_date, "%Y-%m-%d")
        # ValueError: time data '01-15-2024' does not match format '%Y-%m-%d'
```

The LLM receives the traceback and:
1. Does not know which parameter was wrong (it has to parse Python code)
2. Does not know what format to use (it has to decode `%Y-%m-%d` strftime syntax)
3. May hallucinate a fix or give up entirely
4. Wastes a turn generating a response about the error instead of fixing it

## Production Implementation

### Error Schema

```python
"""Structured error contract for agent tools."""
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class ErrorType(str, Enum):
    """Error categories the LLM can reason about."""
    VALIDATION = "validation"      # Bad input, LLM can fix
    NOT_FOUND = "not_found"        # Resource doesn't exist
    TIMEOUT = "timeout"            # Operation took too long
    RATE_LIMIT = "rate_limit"      # Too many requests
    AUTH = "auth"                  # Authentication/authorization failed
    CONFLICT = "conflict"          # Resource state conflict
    INTERNAL = "internal"          # Server error, not LLM's fault
    CIRCUIT_OPEN = "circuit_open"  # Service temporarily unavailable


class ToolErrorResponse(BaseModel):
    """Standard error response that LLMs can parse and act on."""
    
    success: bool = Field(
        default=False, 
        description="Always False for error responses",
    )
    error_type: ErrorType = Field(
        ..., 
        description="Category of error to guide LLM behavior",
    )
    message: str = Field(
        ..., 
        description="Human-readable error message the LLM can use to explain the issue",
    )
    field_errors: Optional[dict[str, str]] = Field(
        default=None,
        description="Per-field error messages for validation errors. "
                    "Keys are parameter names, values describe the issue.",
    )
    correct_format_example: Optional[dict[str, Any]] = Field(
        default=None,
        description="Example of correct input format for the failed parameters. "
                    "LLM should use these exact formats in the retry.",
    )
    retry_after_seconds: Optional[int] = Field(
        default=None,
        description="Seconds to wait before retrying. Only set for rate_limit and circuit_open.",
    )
    suggestions: Optional[list[str]] = Field(
        default=None,
        description="Alternative actions the LLM can take instead of retrying.",
    )


# LLM self-correction behavior by error type:
ERROR_BEHAVIOR_GUIDE = {
    ErrorType.VALIDATION: "Fix the input using field_errors and correct_format_example, then retry.",
    ErrorType.NOT_FOUND: "The resource doesn't exist. Ask the user for correct identifier or search.",
    ErrorType.TIMEOUT: "The operation timed out. Retry once. If it fails again, tell the user.",
    ErrorType.RATE_LIMIT: "Wait retry_after_seconds, then retry. Do not retry immediately.",
    ErrorType.AUTH: "Cannot retry. Tell the user there's an authentication issue.",
    ErrorType.CONFLICT: "Resource is in wrong state. Read current state first, then decide.",
    ErrorType.INTERNAL: "Server error. Retry once. If it fails again, tell the user to try later.",
    ErrorType.CIRCUIT_OPEN: "Service is down. Use suggestions for alternative approaches.",
}
```

### Building Error Responses

```python
"""Error response builders for common tool scenarios."""


def validation_error(
    message: str,
    field_errors: dict[str, str],
    correct_example: dict[str, Any] = None,
) -> dict:
    """Build a validation error that enables LLM self-correction.
    
    Example:
        return validation_error(
            message="Invalid date format in search parameters",
            field_errors={
                "start_date": "Expected YYYY-MM-DD format, got '01/15/2024'",
                "end_date": "Expected YYYY-MM-DD format, got '01/20/2024'",
            },
            correct_example={
                "start_date": "2024-01-15",
                "end_date": "2024-01-20",
            },
        )
    """
    return ToolErrorResponse(
        error_type=ErrorType.VALIDATION,
        message=message,
        field_errors=field_errors,
        correct_format_example=correct_example,
    ).model_dump(exclude_none=True)


def not_found_error(resource_type: str, identifier: str, suggestions: list[str] = None) -> dict:
    """Build a not-found error with search suggestions.
    
    Example:
        return not_found_error(
            resource_type="customer",
            identifier="cust_abc123",
            suggestions=[
                "Search for the customer by email using search_customers(email=...)",
                "Ask the user to verify the customer ID",
            ],
        )
    """
    return ToolErrorResponse(
        error_type=ErrorType.NOT_FOUND,
        message=f"{resource_type} with ID '{identifier}' was not found.",
        suggestions=suggestions or [
            f"Search for the {resource_type} using a different identifier.",
            f"Ask the user to verify the {resource_type} ID.",
        ],
    ).model_dump(exclude_none=True)


def rate_limit_error(retry_after: int, service_name: str = "the service") -> dict:
    """Build a rate-limit error with retry timing."""
    return ToolErrorResponse(
        error_type=ErrorType.RATE_LIMIT,
        message=f"Rate limit exceeded for {service_name}. Wait {retry_after} seconds before retrying.",
        retry_after_seconds=retry_after,
    ).model_dump(exclude_none=True)


def timeout_error(tool_name: str, timeout_seconds: int) -> dict:
    """Build a timeout error."""
    return ToolErrorResponse(
        error_type=ErrorType.TIMEOUT,
        message=f"Tool '{tool_name}' did not complete within {timeout_seconds} seconds. "
                f"The operation may still be processing. Check the status before retrying.",
        suggestions=[
            "Wait a moment and check the status of the operation.",
            "Retry with a simpler request if possible.",
        ],
    ).model_dump(exclude_none=True)


def internal_error(tool_name: str) -> dict:
    """Build a safe internal error that hides implementation details."""
    return ToolErrorResponse(
        error_type=ErrorType.INTERNAL,
        message=f"Tool '{tool_name}' encountered an internal error. "
                f"This is not caused by your input. You may retry once.",
        suggestions=[
            "Retry the same request once.",
            "If it fails again, inform the user that the service is experiencing issues.",
        ],
    ).model_dump(exclude_none=True)
```

### Complete Tool With Error Contract

```python
"""Example tool with full error contract implementation."""
from datetime import datetime, date


class SearchOrdersTool:
    name = "search_orders"
    description = (
        "Search for orders within a date range. Returns order ID, status, "
        "amount, and customer name. Maximum 100 results per query."
    )

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format (e.g., '2024-01-15')",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format (e.g., '2024-01-31')",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "paid", "shipped", "delivered", "refunded"],
                            "description": "Filter by order status. Omit to include all statuses.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results to return. Default 20, maximum 100.",
                            "default": 20,
                        },
                    },
                    "required": ["start_date", "end_date"],
                },
            },
        }

    def execute(self, **kwargs) -> dict:
        start_date_str = kwargs.get("start_date", "")
        end_date_str = kwargs.get("end_date", "")
        status = kwargs.get("status")
        limit = kwargs.get("limit", 20)

        # Validate dates
        field_errors = {}
        correct_example = {}

        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            field_errors["start_date"] = (
                f"Invalid date format: '{start_date_str}'. "
                f"Must be YYYY-MM-DD (e.g., '2024-01-15')."
            )
            correct_example["start_date"] = "2024-01-15"

        try:
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            field_errors["end_date"] = (
                f"Invalid date format: '{end_date_str}'. "
                f"Must be YYYY-MM-DD (e.g., '2024-01-31')."
            )
            correct_example["end_date"] = "2024-01-31"

        # Validate limit
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            field_errors["limit"] = "Must be an integer between 1 and 100."
            correct_example["limit"] = 20

        # Validate status
        valid_statuses = {"pending", "paid", "shipped", "delivered", "refunded"}
        if status and status not in valid_statuses:
            field_errors["status"] = (
                f"Invalid status: '{status}'. "
                f"Must be one of: {', '.join(sorted(valid_statuses))}."
            )
            correct_example["status"] = "paid"

        # Return validation errors if any
        if field_errors:
            return validation_error(
                message="Invalid search parameters. See field_errors for details.",
                field_errors=field_errors,
                correct_example=correct_example,
            )

        # Validate date range
        if start_date > end_date:
            return validation_error(
                message="start_date must be before end_date.",
                field_errors={
                    "start_date": f"'{start_date_str}' is after end_date '{end_date_str}'",
                },
                correct_example={
                    "start_date": end_date_str,
                    "end_date": start_date_str,
                },
            )

        # Date range too large
        if (end_date - start_date).days > 90:
            return validation_error(
                message="Date range cannot exceed 90 days.",
                field_errors={
                    "date_range": f"Range is {(end_date - start_date).days} days (max 90).",
                },
                correct_example={
                    "start_date": str(end_date - timedelta(days=90)),
                    "end_date": str(end_date),
                },
            )

        # Execute query (simplified)
        try:
            orders = db.orders.find(
                start_date=start_date,
                end_date=end_date,
                status=status,
                limit=limit,
            )
            return {
                "success": True,
                "data": {
                    "orders": [o.to_dict() for o in orders],
                    "total_count": len(orders),
                    "has_more": len(orders) == limit,
                },
            }
        except DatabaseTimeoutError:
            return timeout_error("search_orders", 30)
        except Exception:
            return internal_error("search_orders")
```

## Decision Tree: Choosing Error Type

```
What went wrong?
  |
  |-- LLM sent bad input?
  |     --> VALIDATION
  |     Include: field_errors + correct_format_example
  |     LLM behavior: Fix input, retry
  |
  |-- Resource doesn't exist?
  |     --> NOT_FOUND
  |     Include: suggestions (search alternatives)
  |     LLM behavior: Ask user or search differently
  |
  |-- Operation took too long?
  |     --> TIMEOUT
  |     Include: suggestions (check status, simplify request)
  |     LLM behavior: Retry once, then inform user
  |
  |-- Too many requests?
  |     --> RATE_LIMIT
  |     Include: retry_after_seconds
  |     LLM behavior: Wait, then retry
  |
  |-- Auth/permission problem?
  |     --> AUTH
  |     Include: nothing extra (cannot self-fix)
  |     LLM behavior: Inform user, do not retry
  |
  |-- Resource state conflict?
  |     --> CONFLICT
  |     Include: current_state, valid_transitions
  |     LLM behavior: Read current state, adjust approach
  |
  |-- Server/unknown error?
  |     --> INTERNAL
  |     Include: suggestions (retry, try later)
  |     LLM behavior: Retry once, then inform user
```

## When NOT to Use Error Contracts

- **Internal error details**: Never expose stack traces, database queries, internal IPs, or secret names
- **Overly technical messages**: "Foreign key constraint violation on orders.customer_id" should become "Customer with this ID does not exist"
- **Every edge case**: For truly unexpected errors, a generic INTERNAL error is fine. Don't over-engineer error paths.

## Tradeoffs

| Aspect | Structured Errors | Raw Exceptions |
|--------|-------------------|----------------|
| LLM self-correction rate | 80-90% on validation errors | 10-20% |
| Development effort | Higher (error builders) | Lower (just raise) |
| Security | Safe (no internals exposed) | Dangerous (tracebacks leak info) |
| Debugging | Harder (need logging separately) | Easier (full traceback) |
| Token usage | More tokens per error | Fewer tokens but wasted on traceback |
| Agent reliability | High (recovers from errors) | Low (errors halt the agent) |

## Real-World Examples

### Good Error: Date Validation
```json
{
  "success": false,
  "error_type": "validation",
  "message": "Invalid date format in search parameters.",
  "field_errors": {
    "start_date": "Expected YYYY-MM-DD format, got '15 January 2024'"
  },
  "correct_format_example": {
    "start_date": "2024-01-15"
  }
}
```

### Good Error: Resource Not Found
```json
{
  "success": false,
  "error_type": "not_found",
  "message": "Customer with ID 'cust_xyz' was not found.",
  "suggestions": [
    "Search for the customer by email using search_customers(email='...')",
    "Ask the user to verify the customer ID"
  ]
}
```

### Bad Error: Raw Traceback (Anti-Pattern)
```
Traceback (most recent call last):
  File "/app/tools/orders.py", line 47, in search_orders
    cursor = db.execute("SELECT * FROM orders WHERE created_at >= %s", (start_date,))
  File "/usr/lib/python3.11/sqlite3/connection.py", line 142, in execute
    return self._cursor.execute(sql, parameters)
sqlite3.OperationalError: no such table: orders
```

## Failure Modes

| Failure | Cause | Impact | Fix |
|---------|-------|--------|-----|
| LLM cannot self-correct | Raw traceback instead of structured error | Agent gives up or hallucinates | Use ToolErrorResponse schema |
| LLM retries non-retryable error | No error_type classification | Wasted turns, same error repeated | Include error_type field |
| Information leakage | Traceback exposes internal paths/queries | Security vulnerability | Always use INTERNAL type for server errors |
| LLM ignores error | Error message too long or unclear | Agent acts on partial/wrong data | Keep messages under 100 words |
| Wrong correction | Example shows wrong format | LLM corrects to still-wrong input | Validate correct_format_example values |

## Source(s) and Further Reading

- Anthropic, "Building Effective Agents" (2024) -- error handling for tool use
- "9 Practices for Production Agentic Workflows" (arXiv, 2025) -- error contract patterns
- OpenAI, "Function Calling Best Practices" (2024) -- error response format
- Stripe API Error Handling -- industry-standard error codes and messages
- Google API Design Guide, "Errors" -- structured error response patterns
