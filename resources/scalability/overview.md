# Scalability for Agentic Systems
> Not all components scale the same way -- stateless tool workers scale freely, but orchestrators with conversation state need careful architectural decisions.

## What It Is

Scalability in agentic systems is about handling increasing load (more users, more concurrent sessions, more complex tasks) without degrading performance or reliability. The challenge is unique because agents mix stateless computation (LLM calls, tool execution) with stateful orchestration (conversation history, checkpoints, in-progress task state).

The fundamental rule: **separate stateless from stateful components and scale them independently.**

## How It Works

### Component Classification

```
STATELESS (scales freely with replicas)
├── API Gateway / Load Balancer
├── LLM Proxy (LiteLLM, routing layer)
├── Tool Workers (search, calculate, API calls)
├── Embedding Service
├── Guardrail Validators
└── Output Formatters

STATEFUL (needs careful scaling)
├── Orchestrator (conversation state, step tracking)
├── Memory Store (short-term conversation context)
├── Checkpoint Store (LangGraph persistence)
├── Session Manager (user session tracking)
└── Approval Queue (pending human reviews)

SHARED INFRASTRUCTURE (scale independently)
├── Vector Database (RAG index)
├── Relational Database (agent_runs, audit logs)
├── Cache Layer (semantic cache, prompt cache)
├── Message Queue (async task queue)
└── Observability Pipeline (logs, metrics, traces)
```

### Scaling Characteristics

| Component | Scaling Method | Bottleneck | State Required |
|-----------|---------------|-----------|---------------|
| API Gateway | Horizontal | Network I/O | None (stateless) |
| LLM Proxy | Horizontal | Rate limits from providers | None |
| Tool Workers | Horizontal | External API rate limits | None |
| Orchestrator | Horizontal w/ sticky sessions | Memory per session | In-memory or externalized |
| Vector DB | Vertical + sharding | Read throughput, memory | On-disk indexes |
| PostgreSQL | Vertical + read replicas | Write throughput | Transaction log |
| Redis | Cluster mode | Memory capacity | In-memory data |
| Message Queue | Horizontal (partitioned) | Consumer throughput | Message persistence |

## Production Implementation

```python
from dataclasses import dataclass
from enum import Enum


class ScalingStrategy(Enum):
    HORIZONTAL = "horizontal"      # Add more instances
    VERTICAL = "vertical"          # Bigger instances
    SHARDED = "sharded"            # Partition by key
    REPLICATED = "replicated"      # Read replicas


@dataclass
class ScalingConfig:
    component: str
    strategy: ScalingStrategy
    min_replicas: int
    max_replicas: int
    scale_metric: str              # CPU, memory, queue_depth, etc.
    scale_threshold: float         # When to scale up
    cooldown_seconds: int = 300


# Production scaling configurations
SCALING_CONFIGS = {
    "api_gateway": ScalingConfig(
        component="api-gateway",
        strategy=ScalingStrategy.HORIZONTAL,
        min_replicas=2,
        max_replicas=20,
        scale_metric="requests_per_second",
        scale_threshold=500,
    ),
    "tool_workers": ScalingConfig(
        component="tool-workers",
        strategy=ScalingStrategy.HORIZONTAL,
        min_replicas=3,
        max_replicas=50,
        scale_metric="queue_depth",
        scale_threshold=100,  # Scale up when >100 pending tasks
    ),
    "orchestrator": ScalingConfig(
        component="orchestrator",
        strategy=ScalingStrategy.HORIZONTAL,  # With sticky sessions
        min_replicas=2,
        max_replicas=10,
        scale_metric="active_sessions",
        scale_threshold=200,  # Per pod
        cooldown_seconds=600,  # Longer cooldown: sessions are long-lived
    ),
    "vector_db": ScalingConfig(
        component="vector-db",
        strategy=ScalingStrategy.VERTICAL,  # Scale up before sharding
        min_replicas=1,
        max_replicas=3,  # Read replicas
        scale_metric="p99_latency_ms",
        scale_threshold=100,  # Scale when p99 > 100ms
    ),
}
```

## Decision Tree for Scaling Strategy

```
Is the component stateless?
│
├── YES → Scale horizontally behind a load balancer
│   ├── CPU-bound? → Scale on CPU utilization (>70%)
│   ├── I/O-bound? → Scale on concurrent connections
│   └── Queue-based? → Scale on queue depth
│
└── NO → What kind of state?
    │
    ├── Session state (conversation in progress)
    │   ├── Can externalize to Redis? → Do it, then scale horizontally
    │   └── Must be in-memory? → Sticky sessions + graceful drain
    │
    ├── Persistent state (checkpoints, memory)
    │   ├── Read-heavy? → Read replicas
    │   ├── Write-heavy? → Vertical scaling first, then shard
    │   └── Both? → CQRS pattern (separate read/write paths)
    │
    └── Shared state (vector index, cache)
        ├── Can tolerate eventual consistency? → Replicate async
        └── Needs strong consistency? → Single-writer + read replicas
```

## When NOT to Focus on Scalability

- **< 100 concurrent users**: A single well-provisioned server handles this. Optimize for development speed.
- **Prototype/MVP**: Scale concerns are premature. Ship first.
- **Batch-only workloads**: Queue-based processing already scales naturally.

## Tradeoffs

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| Monolith | Simple deployment, low latency | Hard to scale individual parts | < 1K users |
| Microservices | Independent scaling, fault isolation | Complexity, network overhead | > 10K users |
| Serverless (Lambda) | Auto-scaling, pay-per-use | Cold starts, 15min timeout | Sporadic workloads |
| Kubernetes | Flexible, cloud-agnostic | Operational complexity | Teams with K8s experience |

## Real-World Examples

- **LangGraph Cloud**: Managed deployment with auto-scaling orchestrators. Handles checkpointing via PostgreSQL, scales tool workers independently.
- **OpenAI Assistants API**: Stateful sessions managed server-side. Users get a thread_id, all state is managed by the API.
- **Enterprise RAG agent (100K users)**: API gateway (ALB) -> 10 orchestrator pods (sticky sessions) -> 30 tool worker pods (auto-scaling) -> PostgreSQL RDS Multi-AZ.

## Failure Modes

1. **Sticky session affinity breaking**: Load balancer reassigns session to new pod, losing in-memory state. Mitigation: externalize state to Redis.
2. **Thundering herd on vector DB**: All agents search simultaneously after a knowledge base update. Mitigation: stagger refresh, read replicas.
3. **LLM rate limits during scale-up**: More pods = more concurrent LLM calls = rate limited. Mitigation: centralized rate limiter before LLM proxy.
4. **Memory leak in long sessions**: Orchestrator accumulates conversation state, OOM after hours. Mitigation: conversation summarization, session time limits.

## Source(s) and Further Reading

- LangGraph Deployment: https://langchain-ai.github.io/langgraph/concepts/deployment_options/
- Kubernetes HPA: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
- "Designing Data-Intensive Applications" - Martin Kleppmann
- AWS Well-Architected Framework: https://aws.amazon.com/architecture/well-architected/
