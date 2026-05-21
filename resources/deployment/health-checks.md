# Health Checks for Agent Services
> Agents need three-state health checks because a process can be alive but not ready to serve -- and you must handle SIGTERM gracefully for long-running sessions.

## What It Is

Health checks for agent services go beyond traditional "is the server up" checks. Agents have dependencies (LLM providers, vector DBs, Redis) that can be independently healthy or unhealthy, and they run long-lived sessions that must be gracefully drained on shutdown.

The three-state model: `/livez` (process alive), `/readyz` (dependencies healthy, can accept new work), `/healthz` (full diagnostic detail).

## How It Works

### Three-State Health Model

```
/livez  → Is the process running?
          ├── 200: Process is alive
          └── 503: Process is dead (K8s restarts the pod)
          Used by: Kubernetes liveness probe

/readyz → Can the service accept new requests?
          ├── 200: All dependencies healthy, accepting requests
          └── 503: Dependencies unhealthy or draining
          Used by: Kubernetes readiness probe, load balancer

/healthz → Full diagnostic information
           ├── 200: Everything healthy (with details)
           └── 503: Something unhealthy (with details)
           Used by: Monitoring, debugging, dashboards
```

### Why Three Endpoints Matter for Agents

```
Scenario: Redis is down but LLM provider is up
  /livez  → 200 (process is fine, don't restart)
  /readyz → 503 (can't accept requests, remove from LB)
  /healthz → 503 (details: redis_down, llm_ok, pg_ok)

Scenario: Graceful shutdown in progress
  /livez  → 200 (process alive, finishing current sessions)
  /readyz → 503 (stop sending new requests)
  /healthz → 200 (details: draining, 3 active sessions)

Scenario: LLM provider rate limited
  /livez  → 200
  /readyz → 200 (still accepting, will queue/retry)
  /healthz → 200 (details: llm_degraded, using fallback model)
```

## Production Implementation

```python
import time
import asyncio
import signal
from dataclasses import dataclass, field
from enum import Enum
from fastapi import FastAPI, Response
import redis.asyncio as redis
import asyncpg
import httpx


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class DependencyCheck:
    name: str
    status: HealthStatus
    latency_ms: float
    message: str = ""


@dataclass
class HealthReport:
    status: HealthStatus
    uptime_seconds: float
    active_sessions: int
    is_draining: bool
    dependencies: list[DependencyCheck] = field(default_factory=list)


class HealthChecker:
    """Three-state health checker for agent services."""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        pg_pool: asyncpg.Pool,
        llm_api_url: str = "https://api.anthropic.com",
    ):
        self.redis = redis_client
        self.pg_pool = pg_pool
        self.llm_api_url = llm_api_url
        self.start_time = time.time()
        self.is_draining = False
        self.active_sessions = 0
    
    async def livez(self) -> bool:
        """Liveness: is the process alive? Minimal check."""
        return True  # If this code runs, the process is alive
    
    async def readyz(self) -> bool:
        """Readiness: can we accept new requests?"""
        if self.is_draining:
            return False
        
        # Check critical dependencies only
        redis_ok = await self._check_redis()
        pg_ok = await self._check_postgres()
        
        return redis_ok and pg_ok
    
    async def healthz(self) -> HealthReport:
        """Full health report with all dependency details."""
        checks = await asyncio.gather(
            self._check_redis_detailed(),
            self._check_postgres_detailed(),
            self._check_llm_detailed(),
            return_exceptions=True,
        )
        
        dependencies = []
        for check in checks:
            if isinstance(check, Exception):
                dependencies.append(DependencyCheck(
                    name="unknown",
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=0,
                    message=str(check),
                ))
            else:
                dependencies.append(check)
        
        # Overall status
        if any(d.status == HealthStatus.UNHEALTHY for d in dependencies):
            overall = HealthStatus.UNHEALTHY
        elif any(d.status == HealthStatus.DEGRADED for d in dependencies):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY
        
        return HealthReport(
            status=overall,
            uptime_seconds=time.time() - self.start_time,
            active_sessions=self.active_sessions,
            is_draining=self.is_draining,
            dependencies=dependencies,
        )
    
    async def _check_redis(self) -> bool:
        try:
            await asyncio.wait_for(self.redis.ping(), timeout=2.0)
            return True
        except Exception:
            return False
    
    async def _check_postgres(self) -> bool:
        try:
            async with self.pg_pool.acquire() as conn:
                await asyncio.wait_for(conn.fetchval("SELECT 1"), timeout=2.0)
            return True
        except Exception:
            return False
    
    async def _check_redis_detailed(self) -> DependencyCheck:
        start = time.time()
        try:
            await asyncio.wait_for(self.redis.ping(), timeout=2.0)
            return DependencyCheck(
                name="redis",
                status=HealthStatus.HEALTHY,
                latency_ms=round((time.time() - start) * 1000, 1),
            )
        except asyncio.TimeoutError:
            return DependencyCheck(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                latency_ms=2000,
                message="Connection timed out",
            )
        except Exception as e:
            return DependencyCheck(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                latency_ms=round((time.time() - start) * 1000, 1),
                message=str(e),
            )
    
    async def _check_postgres_detailed(self) -> DependencyCheck:
        start = time.time()
        try:
            async with self.pg_pool.acquire() as conn:
                await asyncio.wait_for(conn.fetchval("SELECT 1"), timeout=2.0)
            return DependencyCheck(
                name="postgres",
                status=HealthStatus.HEALTHY,
                latency_ms=round((time.time() - start) * 1000, 1),
            )
        except Exception as e:
            return DependencyCheck(
                name="postgres",
                status=HealthStatus.UNHEALTHY,
                latency_ms=round((time.time() - start) * 1000, 1),
                message=str(e),
            )
    
    async def _check_llm_detailed(self) -> DependencyCheck:
        start = time.time()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.llm_api_url}/v1/models",
                    timeout=5.0,
                    headers={"x-api-key": "dummy"},  # Just checking connectivity
                )
            status = HealthStatus.HEALTHY if resp.status_code in (200, 401) else HealthStatus.DEGRADED
            return DependencyCheck(
                name="llm_provider",
                status=status,
                latency_ms=round((time.time() - start) * 1000, 1),
            )
        except Exception as e:
            return DependencyCheck(
                name="llm_provider",
                status=HealthStatus.DEGRADED,  # Degraded, not unhealthy (can use cache/fallback)
                latency_ms=round((time.time() - start) * 1000, 1),
                message=str(e),
            )


# --- Graceful Shutdown ---

class GracefulShutdown:
    """Handle SIGTERM for long-running agent sessions."""
    
    def __init__(self, health_checker: HealthChecker, drain_timeout: int = 30):
        self.health = health_checker
        self.drain_timeout = drain_timeout
        self._shutdown_event = asyncio.Event()
    
    def setup(self):
        """Register signal handlers."""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_signal)
    
    def _handle_signal(self):
        """Start graceful drain."""
        self.health.is_draining = True  # readyz returns 503
        self._shutdown_event.set()
    
    async def wait_for_drain(self):
        """Wait for active sessions to complete or timeout."""
        await self._shutdown_event.wait()
        
        # Wait for active sessions to finish
        start = time.time()
        while self.health.active_sessions > 0:
            if time.time() - start > self.drain_timeout:
                break  # Force shutdown after timeout
            await asyncio.sleep(1)


# --- FastAPI Integration ---

app = FastAPI()
health_checker: HealthChecker = None  # Initialized at startup


@app.get("/livez")
async def livez():
    """Liveness probe: is the process alive?"""
    return Response(status_code=200, content="OK")


@app.get("/readyz")
async def readyz():
    """Readiness probe: can we accept new requests?"""
    ready = await health_checker.readyz()
    status_code = 200 if ready else 503
    return Response(status_code=status_code, content="ready" if ready else "not ready")


@app.get("/healthz")
async def healthz():
    """Full health check with dependency details."""
    report = await health_checker.healthz()
    status_code = 200 if report.status != HealthStatus.UNHEALTHY else 503
    
    return {
        "status": report.status.value,
        "uptime_seconds": round(report.uptime_seconds),
        "active_sessions": report.active_sessions,
        "is_draining": report.is_draining,
        "dependencies": [
            {
                "name": d.name,
                "status": d.status.value,
                "latency_ms": d.latency_ms,
                "message": d.message,
            }
            for d in report.dependencies
        ],
    }
```

### Kubernetes Probe Configuration

```yaml
spec:
  containers:
  - name: agent
    livenessProbe:
      httpGet:
        path: /livez
        port: 8000
      initialDelaySeconds: 10
      periodSeconds: 30
      timeoutSeconds: 5
      failureThreshold: 3     # Restart after 3 consecutive failures
    
    readinessProbe:
      httpGet:
        path: /readyz
        port: 8000
      initialDelaySeconds: 5
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 2     # Remove from LB after 2 failures
      successThreshold: 1
    
  terminationGracePeriodSeconds: 60  # Allow 60s for active sessions to drain
```

## When NOT to Use Three-State Health Checks

- **Development environment**: A simple `/health` endpoint returning 200 is sufficient.
- **Serverless (Lambda)**: Health checks are managed by the platform.

## Tradeoffs

| Approach | Complexity | Accuracy | Latency Impact |
|----------|-----------|----------|---------------|
| Single `/health` endpoint | Low | Low (all-or-nothing) | None |
| Three-state model | Medium | High (granular) | None |
| Active health checks (with dep calls) | High | Highest | +10-50ms |

## Failure Modes

1. **Health check timeout**: Slow database makes `/readyz` timeout, pod removed from LB during transient slowness. Mitigation: appropriate timeout values, distinguish transient from persistent failures.
2. **Draining takes too long**: Long agent session prevents pod from shutting down within `terminationGracePeriodSeconds`. Mitigation: set session timeout < grace period, checkpoint state for resumption.
3. **False liveness failures**: `/livez` check fails due to GC pause or CPU spike, pod unnecessarily restarted. Mitigation: generous `failureThreshold` (3+).

## Source(s) and Further Reading

- Kubernetes Probes: https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/
- Kubernetes Graceful Shutdown: https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-termination
- Google SRE Health Checking: https://sre.google/sre-book/monitoring-distributed-systems/
