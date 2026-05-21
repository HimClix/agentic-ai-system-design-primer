# Horizontal Scaling Patterns
> Scale agent workloads by adding more instances of stateless components and using queue-based decoupling for stateful ones.

## What It Is

Horizontal scaling means handling more load by running more instances of a component rather than upgrading to a bigger machine. For agentic systems, this requires separating concerns so that each component can scale independently based on its own bottleneck.

The key challenge: agent orchestrators are stateful (they track conversation state, step progress, pending tool calls), so they cannot be naively replicated behind a round-robin load balancer.

## How It Works

### Architecture for Horizontal Scaling

```
                        ┌─────────────────┐
                        │   Load Balancer  │
                        │  (sticky session │
                        │   by session_id) │
                        └────────┬────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
            ┌──────────┐ ┌──────────┐ ┌──────────┐
            │ Orch Pod 1│ │ Orch Pod 2│ │ Orch Pod N│
            │ (sessions │ │ (sessions │ │ (sessions │
            │  A, D, G) │ │  B, E, H) │ │  C, F, I) │
            └─────┬────┘ └─────┬────┘ └─────┬────┘
                  │             │             │
                  └─────────────┼─────────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
            ┌────────────┐ ┌─────────┐ ┌──────────┐
            │ Task Queue  │ │ Redis   │ │PostgreSQL│
            │ (SQS/Redis) │ │ (state) │ │ (persist)│
            └──────┬─────┘ └─────────┘ └──────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ Worker 1 │ │ Worker 2 │ │ Worker N │
  │ (tools)  │ │ (tools)  │ │ (tools)  │
  └──────────┘ └──────────┘ └──────────┘
```

### Stateless Tool Workers

Tool workers are the easiest to scale. They receive a task (e.g., "search the web for X"), execute it, and return the result. No state is retained between calls.

```python
# Kubernetes HPA for tool workers
"""
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: tool-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tool-workers
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: External
    external:
      metric:
        name: sqs_queue_depth
        selector:
          matchLabels:
            queue: tool-tasks
      target:
        type: AverageValue
        averageValue: "10"  # Scale up when >10 tasks per worker
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 5
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
"""
```

### Sticky Sessions for Orchestrators

```python
# NGINX sticky session configuration
"""
upstream orchestrator {
    ip_hash;  # Simple: hash by client IP
    # Or use cookie-based sticky sessions:
    # sticky cookie srv_id expires=1h;
    
    server orchestrator-1:8000;
    server orchestrator-2:8000;
    server orchestrator-3:8000;
}

server {
    location /api/v1/agent/ {
        proxy_pass http://orchestrator;
        proxy_set_header X-Session-ID $http_x_session_id;
        
        # For SSE/streaming responses
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_read_timeout 300s;
    }
}
"""

# ALB sticky sessions (AWS)
"""
resource "aws_lb_target_group" "orchestrator" {
  name     = "agent-orchestrator"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = var.vpc_id

  stickiness {
    type            = "app_cookie"
    cookie_name     = "agent_session_id"
    cookie_duration = 3600  # 1 hour
  }

  health_check {
    path                = "/healthz"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 10
  }
}
"""
```

### Externalized State Pattern

The best approach: externalize all session state to Redis/PostgreSQL so orchestrators become effectively stateless.

```python
import json
import redis.asyncio as redis
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


class ExternalizedStateOrchestrator:
    """
    Orchestrator that stores all session state externally.
    Any pod can handle any session.
    """
    
    def __init__(self, redis_client: redis.Redis, pg_connection_string: str):
        self.redis = redis_client
        self.checkpointer = AsyncPostgresSaver.from_conn_string(pg_connection_string)
    
    async def handle_message(self, session_id: str, message: str) -> dict:
        """Handle an incoming message for any session."""
        
        # 1. Load session state from Redis (fast path)
        state = await self._load_state(session_id)
        
        # 2. Process the message using LangGraph with checkpoint
        config = {"configurable": {"thread_id": session_id}}
        result = await self.graph.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
        )
        
        # 3. State is automatically checkpointed by LangGraph
        # 4. Update Redis cache for fast access
        await self._save_state(session_id, result)
        
        return result
    
    async def _load_state(self, session_id: str) -> dict:
        cached = await self.redis.get(f"session:{session_id}")
        if cached:
            return json.loads(cached)
        
        # Fall back to PostgreSQL checkpoint
        config = {"configurable": {"thread_id": session_id}}
        checkpoint = await self.checkpointer.aget(config)
        if checkpoint:
            state = checkpoint.get("channel_values", {})
            # Cache in Redis for next access
            await self.redis.setex(
                f"session:{session_id}", 3600, json.dumps(state, default=str)
            )
            return state
        return {}
    
    async def _save_state(self, session_id: str, state: dict):
        await self.redis.setex(
            f"session:{session_id}", 3600, json.dumps(state, default=str)
        )
```

## Decision Tree: Scaling Approach

```
Current bottleneck?
│
├── API requests > capacity
│   └── Add more orchestrator pods (with sticky sessions or externalized state)
│
├── Tool execution queue growing
│   └── Add more tool worker pods (auto-scale on queue depth)
│
├── LLM rate limits hit
│   ├── Single provider? → Add second provider as fallback
│   └── Multiple providers? → Implement load-balanced model routing
│
├── Database slow
│   ├── Read-heavy? → Add read replicas
│   ├── Write-heavy? → Vertical scale first, then partition
│   └── Both? → CQRS: separate read/write databases
│
└── Vector DB slow
    ├── Indexing? → Separate index-write from query-read instances
    └── Query latency? → Add replicas, optimize index parameters
```

## When NOT to Use Horizontal Scaling

- **< 50 concurrent sessions**: Single instance handles this. Complexity not worth it.
- **GPU-bound workloads**: Self-hosted LLMs need vertical scaling (bigger GPUs), not more instances.
- **Strong consistency requirements**: Horizontally scaled stateful services are eventually consistent.

## Tradeoffs

| Pattern | Scalability | Complexity | Consistency | Cost |
|---------|------------|-----------|-------------|------|
| Single instance | Limited | Lowest | Strong | Low |
| Sticky sessions | Good (10-100 pods) | Medium | Session-local | Medium |
| Externalized state | Excellent (100+ pods) | High | Eventual (Redis) | Higher |
| Serverless (Lambda) | Excellent (1000+ concurrent) | Low | Stateless | Variable |

## Failure Modes

1. **Session affinity storm**: One orchestrator pod gets disproportionate load because of sticky session hash collision. Mitigation: use cookie-based (not IP-based) stickiness.
2. **State deserialization mismatch**: Orchestrator v2 can't read state written by v1 during rolling deployment. Mitigation: versioned state schemas, backward-compatible changes.
3. **Queue backlog during scale-up**: Workers take 60s to start, queue grows during this time. Mitigation: keep minimum worker count above baseline load.

## Source(s) and Further Reading

- LangGraph Cloud Architecture: https://langchain-ai.github.io/langgraph/concepts/langgraph_cloud/
- Kubernetes HPA: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
- AWS ALB Sticky Sessions: https://docs.aws.amazon.com/elasticloadbalancing/latest/application/sticky-sessions.html
- Redis Cluster: https://redis.io/docs/management/scaling/
