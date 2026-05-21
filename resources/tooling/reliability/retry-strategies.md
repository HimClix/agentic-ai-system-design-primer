# Retry Strategies for Agent Tools
> Retry only what's safe to retry. Exponential backoff + jitter. Maximum 3 attempts. Never retry non-idempotent operations without an idempotency key.

## What It Is

A retry strategy defines how and when a failed tool call should be automatically retried before returning an error to the LLM. In agentic systems, retries are particularly important because:

1. The LLM's own retry (re-calling the tool) costs an entire inference turn ($0.01-0.10)
2. Infrastructure retries at the HTTP level are 1000x cheaper
3. Many failures are transient (network blips, momentary rate limits, cold starts)

The goal: resolve transient failures silently at the tool level, escalate persistent failures as structured errors to the LLM.

## How It Works

### Exponential Backoff + Jitter

```
Attempt 1: Execute immediately
  -> Failure
Attempt 2: Wait base_delay * 2^1 + jitter (e.g., 1.3s)
  -> Failure
Attempt 3: Wait base_delay * 2^2 + jitter (e.g., 4.7s)
  -> Failure
Return structured error to LLM
```

**Why jitter?** Without jitter, when a service recovers from an outage, all agents retry at the exact same intervals, creating a thundering herd that crashes the service again. Jitter randomizes the retry timing to spread the load.

### Retryable vs Non-Retryable Classification

```
HTTP 429 (Rate Limited)  -> Retryable (wait for Retry-After header)
HTTP 500 (Server Error)  -> Retryable (transient server issue)
HTTP 502 (Bad Gateway)   -> Retryable (upstream temporary failure)
HTTP 503 (Unavailable)   -> Retryable (service restarting)
HTTP 504 (Gateway Timeout) -> Retryable (upstream slow)
Connection timeout       -> Retryable (network blip)
DNS resolution failure   -> Retryable (DNS propagation)

HTTP 400 (Bad Request)   -> NOT retryable (bad input, will fail again)
HTTP 401 (Unauthorized)  -> NOT retryable (auth is wrong)
HTTP 403 (Forbidden)     -> NOT retryable (no permission)
HTTP 404 (Not Found)     -> NOT retryable (resource doesn't exist)
HTTP 409 (Conflict)      -> NOT retryable (state conflict, needs resolution)
HTTP 422 (Unprocessable) -> NOT retryable (validation failure)
```

## Production Implementation

```python
"""Retry strategy for agent tools with exponential backoff and jitter."""
import asyncio
import random
import time
import logging
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Set
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)


class RetryableError(Exception):
    """Marks an error as retryable."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class NonRetryableError(Exception):
    """Marks an error as non-retryable. Do not retry."""
    pass


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3              # Total attempts including first try
    base_delay_seconds: float = 1.0    # Base delay before first retry
    max_delay_seconds: float = 30.0    # Cap on delay growth
    jitter_factor: float = 0.5         # Random jitter as fraction of delay
    retryable_status_codes: Set[int] = field(
        default_factory=lambda: {429, 500, 502, 503, 504}
    )
    retryable_exceptions: tuple = field(
        default_factory=lambda: (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            RetryableError,
        )
    )


def calculate_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    jitter_factor: float,
    retry_after: Optional[float] = None,
) -> float:
    """Calculate delay with exponential backoff + jitter.
    
    Formula: min(base_delay * 2^attempt + random_jitter, max_delay)
    If Retry-After header is present, use that instead.
    """
    if retry_after is not None:
        return retry_after

    exponential_delay = base_delay * (2 ** attempt)
    jitter = random.uniform(0, jitter_factor * exponential_delay)
    return min(exponential_delay + jitter, max_delay)


def with_retry(config: RetryConfig = None):
    """Decorator that adds retry logic to async tool functions.
    
    Only retries on retryable errors. Non-retryable errors
    are raised immediately.
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    result = await func(*args, **kwargs)
                    
                    if attempt > 0:
                        logger.info(
                            f"Retry succeeded for {func.__name__} "
                            f"on attempt {attempt + 1}/{config.max_attempts}"
                        )
                    return result

                except NonRetryableError:
                    # Never retry these
                    raise

                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        # Last attempt, don't sleep, just fail
                        logger.error(
                            f"All {config.max_attempts} attempts failed for "
                            f"{func.__name__}: {str(e)}"
                        )
                        break

                    retry_after = getattr(e, "retry_after", None)
                    delay = calculate_delay(
                        attempt=attempt,
                        base_delay=config.base_delay_seconds,
                        max_delay=config.max_delay_seconds,
                        jitter_factor=config.jitter_factor,
                        retry_after=retry_after,
                    )
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed for "
                        f"{func.__name__}: {str(e)}. Retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

                except Exception:
                    # Unknown exceptions are not retried
                    raise

            raise last_exception

        return wrapper
    return decorator


# HTTP response classifier for API tools
import httpx


def classify_http_error(response: httpx.Response) -> Exception:
    """Classify an HTTP error as retryable or non-retryable."""
    status = response.status_code
    
    if status == 429:
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after else 5.0
        return RetryableError(
            f"Rate limited (429). Retry after {delay}s.",
            retry_after=delay,
        )
    
    if status in {500, 502, 503, 504}:
        return RetryableError(f"Server error ({status}): {response.text[:200]}")
    
    if status == 401:
        return NonRetryableError(f"Authentication failed (401)")
    
    if status == 403:
        return NonRetryableError(f"Permission denied (403)")
    
    if status == 404:
        return NonRetryableError(f"Resource not found (404)")
    
    if status in {400, 422}:
        return NonRetryableError(f"Invalid request ({status}): {response.text[:200]}")
    
    return NonRetryableError(f"Unexpected HTTP {status}: {response.text[:200]}")


# Usage example: API tool with retry
@with_retry(RetryConfig(
    max_attempts=3,
    base_delay_seconds=1.0,
    max_delay_seconds=15.0,
))
async def fetch_user_profile(user_id: str) -> dict:
    """Fetch user profile from external API with automatic retry."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"https://api.example.com/users/{user_id}",
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        
        raise classify_http_error(response)
```

### Rate Limit Handling (429s)

```python
"""Specialized rate limit handler with Retry-After respect."""
import asyncio
from datetime import datetime


class RateLimitTracker:
    """Track rate limits per tool and respect Retry-After headers."""
    
    def __init__(self):
        self._blocked_until: dict[str, float] = {}
    
    def is_rate_limited(self, tool_name: str) -> Optional[float]:
        """Check if a tool is currently rate-limited.
        
        Returns remaining wait time in seconds, or None if not limited.
        """
        if tool_name in self._blocked_until:
            remaining = self._blocked_until[tool_name] - time.time()
            if remaining > 0:
                return remaining
            else:
                del self._blocked_until[tool_name]
        return None
    
    def record_rate_limit(self, tool_name: str, retry_after_seconds: float):
        """Record that a tool was rate-limited."""
        self._blocked_until[tool_name] = time.time() + retry_after_seconds
    
    async def wait_if_limited(self, tool_name: str) -> bool:
        """Wait if rate-limited. Returns True if we had to wait."""
        remaining = self.is_rate_limited(tool_name)
        if remaining:
            logger.info(f"Rate limited: {tool_name}, waiting {remaining:.1f}s")
            await asyncio.sleep(remaining)
            return True
        return False


rate_limiter = RateLimitTracker()


@with_retry(RetryConfig(max_attempts=3))
async def call_stripe_api(endpoint: str, method: str = "GET", **kwargs) -> dict:
    """Call Stripe API with rate limit awareness."""
    # Pre-check: are we currently rate-limited?
    await rate_limiter.wait_if_limited("stripe")
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.request(
            method=method,
            url=f"https://api.stripe.com/v1/{endpoint}",
            headers={"Authorization": f"Bearer {STRIPE_KEY}"},
            **kwargs,
        )
        
        if response.status_code == 429:
            retry_after = float(response.headers.get("Retry-After", "5"))
            rate_limiter.record_rate_limit("stripe", retry_after)
            raise RetryableError(
                f"Stripe rate limited. Retry after {retry_after}s.",
                retry_after=retry_after,
            )
        
        if response.status_code >= 400:
            raise classify_http_error(response)
        
        return {"success": True, "data": response.json()}
```

## Decision Tree: Should This Call Be Retried?

```
Did the tool call fail?
  |
  YES --> Is the error retryable?
  |         |
  |         |-- HTTP 429: YES, wait for Retry-After
  |         |-- HTTP 5xx: YES, exponential backoff
  |         |-- Timeout: YES, exponential backoff
  |         |-- Connection error: YES, exponential backoff
  |         |-- HTTP 4xx: NO, return error to LLM
  |         |-- Auth error: NO, return error to LLM
  |         |-- Validation error: NO, return error to LLM
  |
  |     Is the operation idempotent?
  |         |
  |         YES --> Safe to retry
  |         NO  --> Only retry if we're sure it didn't execute
  |                 (timeout before response != timeout after execution)
  |
  |     Have we exceeded max attempts (3)?
  |         |
  |         YES --> Return structured error to LLM
  |         NO  --> Calculate delay, wait, retry
```

## When NOT to Retry

- **Non-idempotent operations without idempotency keys**: Retrying a payment charge without an idempotency key risks double-charging
- **Validation errors (400, 422)**: The same input will fail the same way
- **Auth errors (401, 403)**: Credentials won't change between retries
- **Business logic errors (409)**: State conflicts need resolution, not retries
- **User-facing latency-sensitive paths**: 3 retries with backoff adds 7+ seconds

## Tradeoffs

| Aspect | With Retries | Without Retries |
|--------|-------------|-----------------|
| Transient failure recovery | Automatic | Fails on first error |
| Latency (worst case) | +7s for 3 attempts | No added latency |
| API load | Higher (retried calls) | Lower |
| Complexity | Retry logic + classification | Simpler |
| Cost | Extra API calls | Extra LLM turns (more expensive) |
| User experience | Smoother (fewer visible errors) | More error messages |

## Real-World Examples

### API Call Timeline With Retries
```
t=0.0s   Attempt 1: POST /api/orders -> 503 Service Unavailable
t=1.3s   Attempt 2: POST /api/orders -> 503 Service Unavailable  (delay: 1.0 * 2^1 + 0.3 jitter)
t=5.7s   Attempt 3: POST /api/orders -> 200 OK                  (delay: 1.0 * 2^2 + 1.4 jitter)
Total: 5.7s, user sees success. Without retry: immediate failure at t=0.
```

### Rate Limit Handling Timeline
```
t=0.0s   Attempt 1: GET /api/users -> 429 (Retry-After: 10)
t=10.0s  Attempt 2: GET /api/users -> 200 OK (waited Retry-After)
Total: 10s. Respected the server's rate limit window.
```

## Failure Modes

| Failure | Cause | Impact | Fix |
|---------|-------|--------|-----|
| Retry storm | Many agents retry simultaneously | Service overwhelmed | Add jitter, respect Retry-After |
| Duplicate side effects | Retrying non-idempotent operation | Double payments, double emails | Only retry idempotent operations |
| Retry amplification | Retries at multiple layers compound | 3 * 3 * 3 = 27 total attempts | Retry at one layer only |
| Timeout during retry | Max delay exceeds tool timeout | Tool times out waiting to retry | Cap max_delay < tool_timeout / max_attempts |
| Incorrect classification | Retrying a 400 error | Wasting attempts on permanent failures | Strict status code classification |

## Source(s) and Further Reading

- AWS, "Exponential Backoff and Jitter" (2024) -- jitter strategies
- Google Cloud, "Retry Strategy Best Practices" -- retry classification
- "9 Practices for Production Agentic Workflows" (arXiv, 2025) -- retry in agent context
- Stripe, "Rate Limiting" -- Retry-After header usage
- Marc Brooker (AWS), "Timeouts, retries, and backoff with jitter" -- mathematical analysis
