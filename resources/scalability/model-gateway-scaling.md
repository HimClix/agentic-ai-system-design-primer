# Model Gateway Scaling

> Scaling the LLM gateway/proxy layer: LiteLLM, load balancing across providers, failover routing, request queuing, connection pooling, and horizontal scaling patterns.

## What It Is

A model gateway (or LLM proxy) is a stateless service that sits between your agent backend and LLM providers. It handles:

1. **Unified API**: Single interface to multiple providers (Anthropic, OpenAI, Google, local models)
2. **Load balancing**: Distribute requests across provider endpoints and API keys
3. **Failover**: Automatically switch providers when one is down (Claude -> GPT -> local)
4. **Rate limit management**: Queue requests when hitting provider rate limits
5. **Caching**: Cache identical requests to avoid redundant LLM calls
6. **Observability**: Centralized logging, metrics, and cost tracking

### Why a Gateway?

```
WITHOUT gateway:
  Agent A ──► Anthropic API  (manages own keys, retries, fallback)
  Agent B ──► Anthropic API  (duplicate logic, separate rate limits)
  Agent C ──► OpenAI API     (different client, different error handling)

  Problems:
  - Duplicated retry/fallback logic in every agent
  - No centralized rate limit management (agents compete)
  - No unified cost tracking
  - Provider changes require updating every agent

WITH gateway:
  Agent A ─┐
  Agent B ─┼──► Model Gateway ──► Anthropic / OpenAI / Google / Local
  Agent C ─┘    (LiteLLM)
  
  Benefits:
  - Single point for failover, retries, caching
  - Centralized rate limiting across all agents
  - One place to change providers
  - Unified observability
```

## How It Works

### Gateway Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL GATEWAY                               │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐    │
│  │ Request Queue │   │ Rate Limiter  │   │ Response Cache    │    │
│  │ (Redis/SQS)  │   │ (per provider │   │ (Redis, semantic  │    │
│  │              │   │  per API key) │   │  or exact match)  │    │
│  └──────┬───────┘   └──────┬───────┘   └────────┬─────────┘    │
│         │                  │                     │               │
│  ┌──────▼──────────────────▼─────────────────────▼────────────┐ │
│  │                    ROUTER / LOAD BALANCER                   │ │
│  │                                                             │ │
│  │  Strategy: weighted round-robin / least-latency / cost-opt │ │
│  └──┬──────────────┬────────────────┬────────────────┬───────┘ │
│     │              │                │                │          │
│  ┌──▼───────┐  ┌──▼──────────┐  ┌──▼──────────┐  ┌─▼────────┐│
│  │Anthropic  │  │ OpenAI      │  │ Google      │  │ Local     ││
│  │Claude     │  │ GPT-4o      │  │ Gemini      │  │ Llama     ││
│  │           │  │             │  │             │  │ (vLLM)    ││
│  │Key Pool:  │  │Key Pool:    │  │Key Pool:    │  │           ││
│  │ key1(50%) │  │ key1(100%)  │  │ key1(100%)  │  │ No keys   ││
│  │ key2(50%) │  │             │  │             │  │           ││
│  └───────────┘  └─────────────┘  └─────────────┘  └──────────┘│
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Health Checker: ping each provider every 30s              │   │
│  │ Metrics: latency, errors, tokens, cost per provider       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Failover Chain

```
Primary: Claude Sonnet (Anthropic direct)
  │ 
  │ If 429 (rate limit) or 500+ (server error) or timeout (>10s):
  │
Fallback 1: Claude Sonnet (AWS Bedrock)
  │
  │ If also failing:
  │
Fallback 2: GPT-4o (OpenAI)
  │
  │ If also failing:
  │
Fallback 3: Llama 3.1 70B (local vLLM)
  │
  │ If all fail:
  │
Circuit Breaker: Return error with retry-after header
```

## Production Implementation

### LiteLLM Configuration

```python
# litellm_config.yaml
# Production LiteLLM proxy configuration

model_list:
  # Primary: Anthropic Direct API
  - model_name: "agent-default"
    litellm_params:
      model: "claude-sonnet-4-20250514"
      api_key: "sk-ant-XXXX"
      max_retries: 2
      timeout: 30
    model_info:
      max_tokens: 8192
      max_input_tokens: 200000

  # Fallback 1: Same model via Bedrock
  - model_name: "agent-default"
    litellm_params:
      model: "bedrock/anthropic.claude-sonnet-4-20250514-v1:0"
      aws_region_name: "us-east-1"
      max_retries: 1
      timeout: 30

  # Fallback 2: OpenAI GPT-4o
  - model_name: "agent-default"
    litellm_params:
      model: "gpt-4o"
      api_key: "sk-XXXX"
      max_retries: 1
      timeout: 30

  # Fast model for classification/routing
  - model_name: "agent-fast"
    litellm_params:
      model: "claude-haiku-3.5"
      api_key: "sk-ant-XXXX"
      max_retries: 2
      timeout: 10

  - model_name: "agent-fast"
    litellm_params:
      model: "gpt-4o-mini"
      api_key: "sk-XXXX"
      max_retries: 1
      timeout: 10

router_settings:
  routing_strategy: "latency-based-routing"  # or "cost-based-routing"
  num_retries: 3
  timeout: 30
  retry_after: 5           # Wait 5s between retries
  allowed_fails: 3         # Mark unhealthy after 3 consecutive failures
  cooldown_time: 60        # Seconds before retrying unhealthy endpoint

  # Rate limiting
  redis_host: "redis://localhost:6379"
  
general_settings:
  master_key: "sk-gateway-XXXX"
  database_url: "postgresql://litellm:pass@localhost/litellm"  # For cost tracking
```

### Custom Gateway Implementation

```python
import asyncio
import time
import random
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
import httpx
import logging

logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ProviderEndpoint:
    name: str                  # "anthropic", "openai", "bedrock"
    model: str                 # "claude-sonnet-4-20250514"
    api_base: str
    api_key: str
    weight: float = 1.0        # For weighted routing
    max_rpm: int = 1000        # Requests per minute
    priority: int = 0          # Lower = higher priority (for failover)
    status: ProviderStatus = ProviderStatus.HEALTHY
    consecutive_failures: int = 0
    avg_latency_ms: float = 0.0
    total_requests: int = 0
    total_errors: int = 0
    last_error_at: float = 0.0
    current_rpm: int = 0


class ModelGateway:
    """
    Production LLM gateway with load balancing, failover, and rate limiting.
    
    Features:
    - Weighted round-robin load balancing
    - Automatic failover with circuit breaker
    - Per-provider rate limiting
    - Request queuing during rate limits
    - Health monitoring
    - Cost tracking
    """

    def __init__(
        self,
        endpoints: list[ProviderEndpoint],
        max_queue_size: int = 1000,
        health_check_interval: int = 30,
    ):
        self.endpoints = sorted(endpoints, key=lambda e: e.priority)
        self.max_queue_size = max_queue_size
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._request_count = 0

    def _select_endpoint(self) -> Optional[ProviderEndpoint]:
        """
        Select the best available endpoint.
        
        Strategy:
        1. Filter healthy endpoints
        2. Filter by rate limit availability
        3. Select lowest-priority (primary first)
        """
        available = [
            ep for ep in self.endpoints
            if ep.status != ProviderStatus.UNHEALTHY
            and ep.current_rpm < ep.max_rpm
        ]

        if not available:
            # All endpoints busy or unhealthy
            return None

        # Weighted selection among available endpoints at the same priority level
        best_priority = min(ep.priority for ep in available)
        candidates = [ep for ep in available if ep.priority == best_priority]

        if len(candidates) == 1:
            return candidates[0]

        # Weighted random selection
        total_weight = sum(c.weight for c in candidates)
        r = random.uniform(0, total_weight)
        cumulative = 0
        for candidate in candidates:
            cumulative += candidate.weight
            if r <= cumulative:
                return candidate

        return candidates[0]

    async def complete(
        self,
        messages: list[dict],
        model: str = "agent-default",
        max_tokens: int = 1024,
        system: str = "",
        timeout: float = 30.0,
    ) -> dict:
        """
        Send a completion request through the gateway.
        Handles routing, retries, and failover.
        """
        last_error = None

        for attempt in range(len(self.endpoints)):
            endpoint = self._select_endpoint()

            if endpoint is None:
                # All endpoints exhausted, queue the request
                logger.warning("All endpoints busy, request queued")
                await asyncio.sleep(2)  # Back off
                continue

            try:
                start = time.monotonic()
                result = await self._call_provider(
                    endpoint, messages, max_tokens, system, timeout
                )
                latency = (time.monotonic() - start) * 1000

                # Update metrics
                endpoint.total_requests += 1
                endpoint.current_rpm += 1
                endpoint.avg_latency_ms = (
                    endpoint.avg_latency_ms * 0.9 + latency * 0.1
                )
                endpoint.consecutive_failures = 0
                endpoint.status = ProviderStatus.HEALTHY

                return {
                    "content": result,
                    "provider": endpoint.name,
                    "model": endpoint.model,
                    "latency_ms": latency,
                    "attempt": attempt + 1,
                }

            except Exception as e:
                last_error = e
                endpoint.total_errors += 1
                endpoint.consecutive_failures += 1
                endpoint.last_error_at = time.time()

                # Circuit breaker: mark unhealthy after 3 failures
                if endpoint.consecutive_failures >= 3:
                    endpoint.status = ProviderStatus.UNHEALTHY
                    logger.error(
                        f"Endpoint {endpoint.name} marked UNHEALTHY "
                        f"after {endpoint.consecutive_failures} failures"
                    )

                logger.warning(
                    f"Endpoint {endpoint.name} failed (attempt {attempt + 1}): {e}"
                )

        raise RuntimeError(
            f"All {len(self.endpoints)} endpoints failed. Last error: {last_error}"
        )

    async def _call_provider(
        self,
        endpoint: ProviderEndpoint,
        messages: list[dict],
        max_tokens: int,
        system: str,
        timeout: float,
    ) -> str:
        """Make the actual API call to a provider."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            if "anthropic" in endpoint.name:
                response = await client.post(
                    f"{endpoint.api_base}/v1/messages",
                    headers={
                        "x-api-key": endpoint.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": endpoint.model,
                        "max_tokens": max_tokens,
                        "system": system,
                        "messages": messages,
                    },
                )
                response.raise_for_status()
                return response.json()["content"][0]["text"]

            elif "openai" in endpoint.name:
                msgs = messages.copy()
                if system:
                    msgs.insert(0, {"role": "system", "content": system})
                response = await client.post(
                    f"{endpoint.api_base}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {endpoint.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": endpoint.model,
                        "max_tokens": max_tokens,
                        "messages": msgs,
                    },
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]

    def get_health(self) -> dict:
        """Return health status of all endpoints."""
        return {
            "endpoints": [
                {
                    "name": ep.name,
                    "model": ep.model,
                    "status": ep.status.value,
                    "avg_latency_ms": round(ep.avg_latency_ms, 0),
                    "total_requests": ep.total_requests,
                    "error_rate": (
                        round(ep.total_errors / max(ep.total_requests, 1) * 100, 1)
                    ),
                    "current_rpm": ep.current_rpm,
                }
                for ep in self.endpoints
            ],
        }


# --- Horizontal Scaling ---

SCALING_NOTES = """
The gateway is STATELESS (rate limit state in Redis, no local state).
This means horizontal scaling is straightforward:

Kubernetes deployment:
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: model-gateway
  spec:
    replicas: 3  # Start with 3, HPA scales to 10
    template:
      spec:
        containers:
          - name: litellm
            image: ghcr.io/berriai/litellm:main-latest
            resources:
              requests:
                cpu: "500m"
                memory: "512Mi"
              limits:
                cpu: "2000m"
                memory: "2Gi"
            env:
              - name: REDIS_HOST
                value: "redis-cluster:6379"
        
  ---
  # HPA: scale on CPU
  apiVersion: autoscaling/v2
  kind: HorizontalPodAutoscaler
  metadata:
    name: model-gateway-hpa
  spec:
    scaleTargetRef:
      apiVersion: apps/v1
      kind: Deployment
      name: model-gateway
    minReplicas: 3
    maxReplicas: 10
    metrics:
      - type: Resource
        resource:
          name: cpu
          target:
            type: Utilization
            averageUtilization: 70
"""
```

## Decision Tree: Gateway Architecture

```
    Do I need a model gateway?
                    │
         ┌──────────▼──────────┐
         │ Do you use multiple  │
         │ LLM providers?      │
         └──┬──────────────┬──┘
           Yes             No
            │               │
        GATEWAY         ┌───▼───────────┐
        RECOMMENDED     │ Do you need   │
                        │ failover?     │
                        └──┬────────┬──┘
                          Yes      No
                           │        │
                       GATEWAY   Direct API
                       RECOMMENDED  calls are fine
```

## When NOT to Use

1. **Single provider, single model**: If you only use Claude Sonnet via Anthropic's API, a gateway is overhead.
2. **< 100 requests/day**: The operational cost of running a gateway exceeds the benefit at low volume.
3. **Serverless/edge**: Lambda or Cloudflare Workers cannot maintain persistent gateway connections. Use direct API calls with retries.
4. **Prototyping**: During rapid prototyping, direct SDK calls are faster to iterate on.
5. **When the SDK handles retries**: Anthropic and OpenAI SDKs have built-in retries. If that is sufficient, a gateway is unnecessary.

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Unified interface to all providers | Additional infrastructure to maintain |
| Automatic failover (high availability) | Adds 5-20ms latency per request |
| Centralized rate limiting and cost tracking | Single point of failure (if not HA) |
| Easy provider switching (change config, not code) | Configuration complexity |
| Caching at the gateway level | Debugging through an extra layer |
| Connection pooling for better throughput | Resource overhead (CPU/memory for proxy) |

## Failure Modes

### 1. Gateway as Single Point of Failure
The gateway itself goes down, taking all LLM access with it.
**Mitigation**: Run 3+ replicas behind a load balancer. Kubernetes with PodDisruptionBudget. Health checks on /health endpoint.

### 2. Cascading Failover Storm
Primary provider goes down. All traffic shifts to fallback, which also gets overwhelmed.
**Mitigation**: Rate limit the fallback provider. Implement backpressure (return 503 instead of overwhelming fallback).

### 3. Stale Health Status
Provider recovered but gateway still marks it as unhealthy (cooldown too long).
**Mitigation**: Active health probes every 30s. Reduce cooldown to 60s. Allow manual health override.

### 4. Key Pool Exhaustion
All API keys for a provider are rate-limited simultaneously.
**Mitigation**: Multiple API keys with round-robin. Monitor per-key usage. Alert when keys are at 80% rate limit.

### 5. Cache Poisoning
An incorrect response gets cached and served for subsequent identical requests.
**Mitigation**: Cache only for idempotent requests. Set short TTLs. Allow cache bypass header.

## Sources and Further Reading

- [LiteLLM Documentation](https://docs.litellm.ai/) -- Most popular open-source LLM gateway
- [LiteLLM Proxy Server](https://docs.litellm.ai/docs/simple_proxy) -- Production proxy setup
- [Portkey AI Gateway](https://portkey.ai/) -- Managed LLM gateway
- [Martian Model Router](https://withmartian.com/) -- Intelligent model routing
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents)
- [OpenAI Rate Limits](https://platform.openai.com/docs/guides/rate-limits)
