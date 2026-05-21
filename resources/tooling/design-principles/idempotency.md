# Idempotency in Agent Tools
> LLMs retry. Orchestrators retry. Networks fail and retry. If your tool creates a payment, it will create duplicate payments without idempotency.

## What It Is

Idempotency means calling a tool multiple times with the same input produces the same result as calling it once. For read operations this is trivial -- reading data twice returns the same data. For write operations (create, update, delete), idempotency requires explicit engineering.

In agentic systems, idempotency is not optional. Agents operate in retry loops by design. When a tool call times out, the orchestrator retries it. When the LLM's response is malformed, the entire turn (including tool calls) may be replayed. Without idempotency, every retry is a potential duplicate side effect.

## How It Works

### Why Agents Retry More Than Traditional Systems

```
Traditional API:  Client -> Server (1 retry on timeout)
Agent System:     LLM -> Orchestrator -> Tool -> External API
                        ^       ^          ^
                        |       |          |
                      Retry   Retry      Retry
                      (LLM    (tool      (HTTP
                      loop)   framework) client)
```

An agent tool call passes through at least 3 retry layers:
1. **LLM loop retry**: The agent framework re-invokes the LLM if the response is malformed, which may re-emit the same tool call
2. **Orchestrator retry**: LangChain/CrewAI/custom frameworks retry tool execution on timeout or transient errors
3. **HTTP client retry**: The tool's internal HTTP client retries on network failures

The result: a single user request can trigger 3-9 executions of the same tool call.

### Idempotency Key Pattern

The standard solution is an idempotency key -- a unique identifier for each intended operation. The server stores the result of the first execution and returns it for all subsequent calls with the same key.

```
First call:   idempotency_key="abc123" -> Execute -> Store result -> Return result
Second call:  idempotency_key="abc123" -> Find stored result -> Return same result  
Third call:   idempotency_key="abc123" -> Find stored result -> Return same result
```

### Key Generation Strategies

| Strategy | How | Best For |
|----------|-----|----------|
| Content hash | `hash(tool_name + sorted(params))` | When same input = same intent |
| Caller-provided | Agent passes explicit key | When caller controls dedup |
| Turn-based | `hash(conversation_id + turn_number + tool_index)` | Chat agents |
| Request-scoped | `hash(request_id + tool_name + params)` | API-backed agents |

## Production Implementation

### Core Idempotency Layer

```python
"""Idempotency layer for agent tools."""
import hashlib
import json
import time
from typing import Any, Optional, Callable
from dataclasses import dataclass
from functools import wraps
import redis
import logging

logger = logging.getLogger(__name__)


@dataclass
class IdempotencyRecord:
    key: str
    result: dict
    created_at: float
    tool_name: str


class IdempotencyStore:
    """Redis-backed idempotency store with TTL."""

    def __init__(self, redis_client: redis.Redis, ttl_seconds: int = 3600):
        self.redis = redis_client
        self.ttl = ttl_seconds

    def _make_redis_key(self, idempotency_key: str) -> str:
        return f"idempotent:{idempotency_key}"

    def get(self, idempotency_key: str) -> Optional[dict]:
        """Return cached result if this key was already processed."""
        data = self.redis.get(self._make_redis_key(idempotency_key))
        if data:
            return json.loads(data)
        return None

    def set(self, idempotency_key: str, result: dict, tool_name: str):
        """Store result for idempotency key with TTL."""
        record = {
            "result": result,
            "tool_name": tool_name,
            "created_at": time.time(),
        }
        self.redis.setex(
            self._make_redis_key(idempotency_key),
            self.ttl,
            json.dumps(record),
        )

    def acquire_lock(self, idempotency_key: str, timeout: int = 30) -> bool:
        """Acquire a lock to prevent concurrent execution of the same operation."""
        lock_key = f"lock:{idempotency_key}"
        return self.redis.set(lock_key, "1", nx=True, ex=timeout)

    def release_lock(self, idempotency_key: str):
        """Release the execution lock."""
        lock_key = f"lock:{idempotency_key}"
        self.redis.delete(lock_key)


def generate_idempotency_key(tool_name: str, **params) -> str:
    """Generate a deterministic idempotency key from tool name and parameters.
    
    Sorts parameters to ensure consistent key regardless of argument order.
    """
    # Sort params for deterministic hashing
    sorted_params = json.dumps(params, sort_keys=True, default=str)
    content = f"{tool_name}:{sorted_params}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def idempotent(ttl_seconds: int = 3600):
    """Decorator to make a tool idempotent using content-based key generation."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, **kwargs):
            # Generate idempotency key from content
            key = kwargs.pop("idempotency_key", None)
            if key is None:
                key = generate_idempotency_key(self.name, **kwargs)

            store: IdempotencyStore = self.idempotency_store

            # Check for existing result
            existing = store.get(key)
            if existing:
                logger.info(f"Idempotent hit for tool={self.name} key={key}")
                result = existing["result"]
                result["_idempotent_replay"] = True
                return result

            # Acquire lock to prevent concurrent execution
            if not store.acquire_lock(key):
                logger.warning(f"Concurrent execution blocked for tool={self.name} key={key}")
                return {
                    "success": False,
                    "error_type": "concurrent_execution",
                    "message": "This operation is already being processed. Please wait.",
                    "retry_after_seconds": 5,
                }

            try:
                # Execute the actual tool
                result = await func(self, **kwargs)

                # Store result for future replay
                store.set(key, result, self.name)

                return result
            finally:
                store.release_lock(key)

        return wrapper
    return decorator
```

### FastAPI Integration

```python
"""FastAPI tool endpoint with idempotency."""
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional
import redis

app = FastAPI()
redis_client = redis.Redis(host="localhost", port=6379, db=0)
idempotency_store = IdempotencyStore(redis_client)


class CreatePaymentRequest(BaseModel):
    amount: int = Field(..., description="Amount in smallest currency unit (cents)")
    currency: str = Field(default="usd", description="Three-letter ISO currency code")
    customer_id: str = Field(..., description="Customer identifier")
    description: str = Field(default="", description="Payment description")


class PaymentResponse(BaseModel):
    success: bool
    payment_id: Optional[str] = None
    status: Optional[str] = None
    error_type: Optional[str] = None
    message: Optional[str] = None


@app.post("/tools/create_payment", response_model=PaymentResponse)
async def create_payment(
    request: CreatePaymentRequest,
    x_idempotency_key: Optional[str] = Header(None),
    x_trace_id: Optional[str] = Header(None),
):
    """Create a payment intent. Idempotent -- safe to retry.
    
    Pass X-Idempotency-Key header to ensure the payment is created 
    exactly once, even if this endpoint is called multiple times.
    """
    # Generate key from content if not provided
    key = x_idempotency_key or generate_idempotency_key(
        "create_payment",
        amount=request.amount,
        currency=request.currency,
        customer_id=request.customer_id,
    )

    # Check for existing result
    existing = idempotency_store.get(key)
    if existing:
        return PaymentResponse(**existing["result"])

    # Acquire lock
    if not idempotency_store.acquire_lock(key):
        raise HTTPException(
            status_code=409,
            detail="Payment is already being processed. Retry in 5 seconds.",
        )

    try:
        # Actually create the payment
        payment = await stripe.PaymentIntent.create(
            amount=request.amount,
            currency=request.currency,
            customer=request.customer_id,
            description=request.description,
            idempotency_key=key,  # Stripe also supports idempotency
        )

        result = {
            "success": True,
            "payment_id": payment.id,
            "status": payment.status,
        }

        idempotency_store.set(key, result, "create_payment")
        return PaymentResponse(**result)

    except stripe.error.CardError as e:
        result = {
            "success": False,
            "error_type": "card_declined",
            "message": f"Card was declined: {e.user_message}",
        }
        # Still cache failed results to prevent re-charging on retry
        idempotency_store.set(key, result, "create_payment")
        return PaymentResponse(**result)

    finally:
        idempotency_store.release_lock(key)
```

## Decision Tree: Does This Tool Need Idempotency?

```
Does the tool modify external state?
  |
  NO (read-only) --> Idempotency is automatic. No extra work needed.
  |
  YES --> Does it CREATE something?
           |
           YES --> Content-hash idempotency key
           |       + Store result for replay
           |       + Dedup on (tool_name + params) hash
           |
           NO --> Does it UPDATE something?
                   |
                   YES --> Use upsert/conditional write
                   |       + "UPDATE ... WHERE version = expected_version"
                   |       + Last-write-wins is often acceptable
                   |
                   NO --> Does it DELETE something?
                           |
                           YES --> DELETE is naturally idempotent
                                   (deleting twice = same result)
                                   Just handle "not found" gracefully
```

## When NOT to Use Idempotency Keys

- **Read-only tools**: `get_user()`, `search_records()` -- naturally idempotent
- **Truly unique operations**: When every call SHOULD create a new entity (rare in agent contexts)
- **Time-sensitive queries**: `get_current_price()` should return fresh data, not cached results
- **Delete operations**: Deleting an already-deleted resource should return success, not an error

## Tradeoffs

| Aspect | With Idempotency | Without Idempotency |
|--------|------------------|---------------------|
| Data integrity | Safe from duplicate side effects | Duplicates on every retry |
| Latency | +1-5ms for Redis lookup | No overhead |
| Complexity | Idempotency store + key generation | Simpler code |
| Storage | Redis/DB for idempotency records | None |
| Failure modes | Key collision (extremely rare with SHA-256) | Duplicate records/payments |
| Debugging | Can trace replayed results | Hard to distinguish retries from new calls |

## Real-World Failure: Duplicate Payment Processing

### The Scenario

A customer support agent processes a refund. The tool call to the payment gateway times out after 28 seconds. The orchestrator retries. The payment gateway had actually processed the refund -- it just took 29 seconds to respond. Result: double refund.

```
Turn 1: User says "Refund order #12345"
  -> Agent calls refund_payment(order_id="12345", amount=99.99)
  -> HTTP timeout at 28s (gateway processed at 29s)
  -> Orchestrator retries
  -> Agent calls refund_payment(order_id="12345", amount=99.99) AGAIN
  -> Gateway processes second refund
  -> Customer refunded $199.98 instead of $99.99
```

### The Fix

```python
@idempotent(ttl_seconds=86400)  # 24-hour window
async def refund_payment(self, order_id: str, amount: float, reason: str) -> dict:
    """Process a refund. Idempotent -- calling twice with same order_id
    and amount will only process one refund."""
    
    # The idempotency decorator handles:
    # 1. Generate key from (tool_name + order_id + amount + reason)
    # 2. Check Redis for existing result
    # 3. If found, return cached result (no second refund)
    # 4. If not found, execute and cache
    
    result = await payment_gateway.refund(
        order_id=order_id,
        amount=amount,
        reason=reason,
    )
    return {"refund_id": result.id, "status": result.status}
```

## Failure Modes

| Failure | Cause | Impact | Mitigation |
|---------|-------|--------|------------|
| Double payment | No idempotency on payment creation | Financial loss | Idempotency key on all payment tools |
| Key collision | Poor hash function | Wrong cached result returned | Use SHA-256, include tool name in key |
| Stale replay | TTL too long | Outdated result returned | Set TTL appropriate to tool (1h for payments, 5m for status) |
| Lock timeout | Execution takes longer than lock TTL | Concurrent duplicate execution | Set lock TTL > max tool timeout |
| Redis failure | Idempotency store unavailable | Falls back to non-idempotent | Circuit breaker on Redis, fall through with warning |

## Source(s) and Further Reading

- Stripe API, "Idempotent Requests" -- industry-standard idempotency key pattern
- "9 Practices for Production Agentic Workflows" (arXiv, 2025) -- idempotency as critical practice
- Amazon, "Idempotency in Event-Driven Architectures" -- distributed systems perspective
- Anthropic, "Building Effective Agents" (2024) -- retry behavior in agent loops
