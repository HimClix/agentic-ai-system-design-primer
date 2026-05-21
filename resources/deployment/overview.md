# Deployment Options for Agentic Systems
> Choose deployment based on team size and operational maturity -- managed platforms for speed, self-hosted for control.

## What It Is

Deployment for agentic systems is more complex than traditional web apps because agents have unique requirements: long-running sessions, streaming responses, stateful orchestration with checkpoints, and integration with LLM providers, vector databases, and tool services.

The deployment choice depends on: team size, operational expertise, regulatory requirements, and cost constraints.

## How It Works

### Deployment Options Spectrum

```
Simplest ──────────────────────────────────────────────── Most Control

LangGraph     Standalone      ECS Fargate    Kubernetes    Self-Hosted
Platform      Docker          (AWS)          Helm Chart    Everything
(Managed)                                                  (K8s + vLLM)

Team: 1-3     Team: 2-5      Team: 3-8      Team: 5-15   Team: 10+
$50-500/mo    $100-500/mo    $200-2K/mo     $500-5K/mo   $2K-20K/mo
```

### Option Comparison

| Dimension | LangGraph Platform | Standalone Docker | ECS Fargate | K8s Helm | Self-Hosted Full |
|-----------|-------------------|-------------------|-------------|----------|-----------------|
| Setup time | Minutes | Hours | Days | Days | Weeks |
| Scaling | Auto | Manual | Auto (task-based) | HPA/KEDA | Full control |
| Checkpointing | Built-in | You manage | You manage | You manage | You manage |
| Streaming | Built-in SSE | You implement | You implement | You implement | You implement |
| LLM hosting | API only | API only | API only | API or self-host | Self-host option |
| GPU support | No | Docker GPU | No | Yes (GPU nodes) | Full control |
| Cost at scale | Higher (managed markup) | Low (infra only) | Medium | Medium | Lowest (at scale) |
| Vendor lock-in | High (LangGraph specific) | Low | Medium (AWS) | Low | None |
| Compliance | Shared responsibility | You own it | AWS compliant | You own it | You own it |

## Decision Tree

```
START: What is your team's Kubernetes experience?
│
├── None / Minimal
│   ├── Need production ASAP (< 1 week)?
│   │   └── LangGraph Platform (managed)
│   └── Have more time?
│       ├── Already on AWS?
│       │   └── ECS Fargate (serverless containers)
│       └── Not on AWS?
│           └── Standalone Docker on a VM (simplest self-hosted)
│
├── Moderate (can deploy Helm charts)
│   ├── Need GPU for self-hosted LLMs?
│   │   └── K8s with GPU node pools + vLLM
│   └── Using API-based LLMs only?
│       └── K8s Helm chart (standard deployment)
│
└── Expert (platform engineering team)
    └── Full self-hosted stack
        ├── K8s + vLLM for LLM serving
        ├── Custom orchestrator deployment
        └── Full observability stack
```

## Production Implementation

### Docker Compose (Standalone)

```yaml
version: '3.8'
services:
  agent-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:pass@postgres:5432/agents
    depends_on:
      - redis
      - postgres
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: agents
      POSTGRES_PASSWORD: pass
    ports:
      - "5432:5432"
    volumes:
      - pg-data:/var/lib/postgresql/data

volumes:
  redis-data:
  pg-data:
```

### Kubernetes Helm Values (Production)

```yaml
# values.yaml for agent deployment
replicaCount: 3

image:
  repository: your-registry/agent-service
  tag: "latest"
  pullPolicy: Always

resources:
  requests:
    cpu: 500m
    memory: 512Mi
  limits:
    cpu: 2000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70

env:
  - name: ANTHROPIC_API_KEY
    valueFrom:
      secretKeyRef:
        name: agent-secrets
        key: anthropic-api-key
  - name: REDIS_URL
    value: "redis://redis-master:6379"
  - name: DATABASE_URL
    valueFrom:
      secretKeyRef:
        name: agent-secrets
        key: database-url

livenessProbe:
  httpGet:
    path: /livez
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /readyz
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10

service:
  type: ClusterIP
  port: 8000

ingress:
  enabled: true
  annotations:
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
```

## When NOT to Use Each Option

- **LangGraph Platform**: When you need GPU hosting, custom infra, or can't accept vendor lock-in.
- **Standalone Docker**: When you need auto-scaling or high availability.
- **ECS Fargate**: When you need GPUs (Fargate doesn't support GPU tasks well).
- **Kubernetes**: When your team doesn't have K8s experience and doesn't have time to learn.

## Tradeoffs

| Factor | Managed Platform | Self-Hosted K8s |
|--------|-----------------|----------------|
| Time to deploy | Hours | Days-Weeks |
| Operational burden | Low | High |
| Cost at < 100K requests/month | Lower | Higher (infra overhead) |
| Cost at > 1M requests/month | Higher (managed markup) | Lower (amortized infra) |
| Customization | Limited | Unlimited |
| Debugging | Limited visibility | Full visibility |
| Compliance | Depends on provider | Full control |

## Source(s) and Further Reading

- LangGraph Platform: https://langchain-ai.github.io/langgraph/concepts/langgraph_platform/
- LangGraph Helm Chart: https://github.com/langchain-ai/helm
- AWS ECS Best Practices: https://docs.aws.amazon.com/AmazonECS/latest/bestpracticesguide/
- Kubernetes Production Best Practices: https://learnk8s.io/production-best-practices
