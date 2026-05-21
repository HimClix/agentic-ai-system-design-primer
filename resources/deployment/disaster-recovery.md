# Disaster Recovery for Agent Systems
> Agent DR is harder than traditional DR: stateful agent sessions cannot simply be restarted, memory stores contain irreplaceable context, and LLM providers introduce third-party dependency chains you don't control.

## What It Is

Disaster Recovery (DR) for agentic AI systems is the set of strategies, processes, and infrastructure that ensure an agent platform can recover from failures -- ranging from LLM provider outages and memory store corruption to full datacenter loss -- while preserving agent state, conversation context, and user trust. Unlike traditional DR where stateless services restart cleanly, agent systems carry multi-turn state, learned memory, and in-flight tool calls that must be recovered or gracefully terminated.

### What Makes Agent DR Different from Traditional DR

| Dimension | Traditional Service DR | Agent System DR |
|-----------|----------------------|-----------------|
| State | Stateless (request-response) | Stateful (multi-turn sessions, in-flight tool calls) |
| Data loss impact | Retry request | Lost conversation context, broken workflows |
| Provider dependency | Your infrastructure only | LLM provider + your infrastructure |
| Recovery granularity | Restart pod | Resume from specific checkpoint in agent graph |
| Memory | Application DB only | Short-term + long-term + episodic memory stores |
| Nondeterminism | Same input → same output | Same input → different output (LLM variance) |
| Cost of re-execution | Negligible | $0.01-1.00+ per agent turn (token costs) |

## How It Works

### Recovery Architecture

```
                        ┌──────────────────────────────────┐
                        │       DR Control Plane            │
                        │  ┌─────────┐  ┌───────────────┐  │
                        │  │ Health   │  │ Failover      │  │
                        │  │ Monitor  │  │ Orchestrator  │  │
                        │  └────┬────┘  └───────┬───────┘  │
                        │       │               │          │
                        └───────┼───────────────┼──────────┘
                                │               │
             ┌──────────────────┼───────────────┼──────────────────┐
             │                  │               │                  │
    ┌────────▼────────┐ ┌──────▼───────┐ ┌─────▼──────┐ ┌────────▼────────┐
    │  LLM Provider   │ │  Agent State │ │  Memory    │ │  Tool Endpoint  │
    │  Failover       │ │  Recovery    │ │  Store DR  │ │  Failover       │
    │                 │ │              │ │            │ │                 │
    │ Primary: Claude │ │ Checkpoint   │ │ Primary:   │ │ Primary → Fallback│
    │ Secondary: GPT  │ │ → Resume     │ │ PostgreSQL │ │ with circuit     │
    │ Tertiary: Local │ │ → Replay     │ │ Standby:   │ │ breakers         │
    │ (Ollama/vLLM)   │ │ → Terminate  │ │ Multi-AZ   │ │                 │
    └─────────────────┘ └──────────────┘ └────────────┘ └─────────────────┘
```

### Failure Categories and Recovery Strategies

```
Failure Type
│
├── LLM Provider Outage
│   ├── Partial (rate limit / 429) → Backoff + retry
│   ├── Regional (one endpoint down) → Failover to secondary region
│   └── Full provider outage → Switch to secondary provider
│
├── Agent State Loss
│   ├── Pod crash (in-flight request) → Resume from last checkpoint
│   ├── Node failure → Reschedule pod, load checkpoint from store
│   └── Cluster failure → Failover to standby cluster
│
├── Memory Store Failure
│   ├── Primary DB down → Promote standby
│   ├── Data corruption → Point-in-time recovery from backup
│   └── Full data loss → Restore from cross-region backup
│
├── Tool Endpoint Failure
│   ├── Single tool down → Circuit breaker, skip or fallback
│   ├── Multiple tools down → Graceful degradation mode
│   └── All tools down → Read-only mode with cached data
│
└── Full Datacenter / Region Loss
    ├── Active-passive → Promote passive region
    └── Active-active → Route traffic away from failed region
```

## Production Implementation

### 1. Checkpoint-Based Recovery with LangGraph

```python
"""
Checkpoint-based agent recovery using LangGraph PostgresSaver.
Every state transition is persisted -- on failure, resume from last checkpoint.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import interrupt

import psycopg


# --- Checkpoint Store Configuration ---

CHECKPOINT_DB_CONFIG = {
    "primary": {
        "host": "agent-checkpoints-primary.us-east-1.rds.amazonaws.com",
        "port": 5432,
        "dbname": "agent_checkpoints",
        "user": "agent_svc",
        "password_secret": "agent-db-password",       # from AWS Secrets Manager
        "sslmode": "verify-full",
    },
    "standby": {
        "host": "agent-checkpoints-standby.us-west-2.rds.amazonaws.com",
        "port": 5432,
        "dbname": "agent_checkpoints",
        "user": "agent_svc",
        "password_secret": "agent-db-password-west",
        "sslmode": "verify-full",
    },
}


async def get_checkpoint_saver(
    region: str = "primary",
) -> AsyncPostgresSaver:
    """
    Create a PostgresSaver connected to the appropriate region.
    Falls back to standby if primary is unreachable.
    """
    config = CHECKPOINT_DB_CONFIG[region]
    password = await get_secret(config["password_secret"])

    conn_string = (
        f"postgresql://{config['user']}:{password}"
        f"@{config['host']}:{config['port']}/{config['dbname']}"
        f"?sslmode={config['sslmode']}"
    )

    try:
        conn = await psycopg.AsyncConnection.connect(conn_string)
        saver = AsyncPostgresSaver(conn)
        await saver.setup()  # Create tables if not exist
        return saver
    except Exception as e:
        if region == "primary":
            print(f"Primary checkpoint DB unreachable: {e}. Failing over to standby.")
            return await get_checkpoint_saver("standby")
        raise


# --- Agent with Checkpoint Recovery ---

async def resume_or_start_agent(
    thread_id: str,
    user_input: str,
    checkpointer: AsyncPostgresSaver,
):
    """
    Resume an agent from its last checkpoint if one exists,
    otherwise start a new session.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # Build the agent graph
    graph = build_agent_graph()
    app = graph.compile(checkpointer=checkpointer)

    # Check for existing checkpoint
    state = await app.aget_state(config)

    if state and state.values:
        # Resume from checkpoint -- the agent picks up
        # exactly where it left off in the graph
        print(f"Resuming thread {thread_id} from checkpoint")
        print(f"  Last node: {state.next}")
        print(f"  Messages so far: {len(state.values.get('messages', []))}")

        # If the agent was waiting for human input (interrupt),
        # supply the new input and continue
        if state.next:
            result = await app.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
        else:
            # Agent completed -- start new turn
            result = await app.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
    else:
        # No checkpoint -- fresh session
        print(f"Starting new session for thread {thread_id}")
        result = await app.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )

    return result
```

### 2. LLM Provider Failover Chain

```python
"""
Multi-provider failover for LLM calls.
Primary → Secondary → Tertiary (local) with circuit breakers.
"""
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import httpx

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


class ProviderStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"         # High latency, partial errors
    CIRCUIT_OPEN = "circuit_open" # Too many failures, stop trying


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5            # Failures before opening circuit
    recovery_timeout_sec: int = 60        # How long circuit stays open
    half_open_max_calls: int = 2          # Test calls in half-open state

    failure_count: int = 0
    last_failure_time: float = 0.0
    status: ProviderStatus = ProviderStatus.HEALTHY
    half_open_calls: int = 0

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.status = ProviderStatus.CIRCUIT_OPEN
            print(f"Circuit OPENED after {self.failure_count} failures")

    def record_success(self):
        self.failure_count = 0
        self.status = ProviderStatus.HEALTHY
        self.half_open_calls = 0

    def can_attempt(self) -> bool:
        if self.status == ProviderStatus.HEALTHY:
            return True
        if self.status == ProviderStatus.CIRCUIT_OPEN:
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.recovery_timeout_sec:
                self.status = ProviderStatus.DEGRADED   # half-open
                self.half_open_calls = 0
                return True
            return False
        if self.status == ProviderStatus.DEGRADED:
            return self.half_open_calls < self.half_open_max_calls


@dataclass
class LLMProvider:
    name: str
    model: str
    client: object               # LangChain chat model
    circuit: CircuitBreaker = field(default_factory=CircuitBreaker)
    priority: int = 0            # Lower = higher priority


class LLMFailoverChain:
    """
    Failover chain: try providers in priority order.
    Each provider has its own circuit breaker.
    """

    def __init__(self):
        self.providers = [
            LLMProvider(
                name="anthropic",
                model="claude-sonnet-4-20250514",
                client=ChatAnthropic(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    timeout=30,
                ),
                priority=0,
            ),
            LLMProvider(
                name="openai",
                model="gpt-4o",
                client=ChatOpenAI(
                    model="gpt-4o",
                    max_tokens=4096,
                    timeout=30,
                ),
                priority=1,
            ),
            LLMProvider(
                name="local-vllm",
                model="meta-llama/Llama-3.1-70B-Instruct",
                client=ChatOpenAI(
                    base_url="http://vllm-local:8000/v1",
                    model="meta-llama/Llama-3.1-70B-Instruct",
                    max_tokens=2048,
                    timeout=60,
                ),
                priority=2,
            ),
        ]
        self.providers.sort(key=lambda p: p.priority)

    async def invoke(self, messages: list, **kwargs) -> str:
        last_error = None

        for provider in self.providers:
            if not provider.circuit.can_attempt():
                print(f"Skipping {provider.name}: circuit open")
                continue

            try:
                start = time.time()
                response = await provider.client.ainvoke(messages, **kwargs)
                latency = time.time() - start

                provider.circuit.record_success()
                print(
                    f"LLM call succeeded via {provider.name} "
                    f"({provider.model}) in {latency:.2f}s"
                )
                return response

            except Exception as e:
                provider.circuit.record_failure()
                last_error = e
                print(
                    f"LLM call failed via {provider.name}: {e}. "
                    f"Trying next provider..."
                )

        raise RuntimeError(
            f"All LLM providers failed. Last error: {last_error}"
        )
```

### 3. Memory Store Backup and Recovery

```python
"""
Backup strategies for agent memory stores.
Covers PostgresSaver checkpoints, vector stores, and long-term memory.
"""
import subprocess
import boto3
from datetime import datetime


class AgentMemoryBackup:
    """
    Backup agent memory stores to S3 with configurable schedules.

    Backup targets:
    1. PostgresSaver checkpoint DB  (RTO <5 min)
    2. Vector store (Pgvector)      (RTO <15 min)
    3. Long-term memory store       (RTO <30 min)
    """

    def __init__(self, s3_bucket: str, region: str = "us-east-1"):
        self.s3 = boto3.client("s3", region_name=region)
        self.bucket = s3_bucket

    def backup_postgres_checkpoints(
        self,
        db_host: str,
        db_name: str,
        db_user: str,
    ) -> str:
        """
        Continuous WAL archiving + periodic base backups.

        Schedule:
        - WAL archiving: continuous (RPO ~0)
        - Base backup: every 6 hours
        - Retention: 7 days of PITR capability
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = f"/tmp/checkpoint_backup_{timestamp}.sql.gz"
        s3_key = f"backups/checkpoints/{timestamp}/checkpoint_backup.sql.gz"

        # pg_dump with custom format for parallel restore
        cmd = [
            "pg_dump",
            f"--host={db_host}",
            f"--dbname={db_name}",
            f"--username={db_user}",
            "--format=custom",
            "--compress=9",
            "--no-owner",
            "--no-privileges",
            # Only back up checkpoint-related tables
            "--table=checkpoints",
            "--table=checkpoint_blobs",
            "--table=checkpoint_writes",
            f"--file={backup_file}",
        ]
        subprocess.run(cmd, check=True)

        self.s3.upload_file(backup_file, self.bucket, s3_key)
        return s3_key

    def backup_vector_store(
        self,
        db_host: str,
        db_name: str,
        db_user: str,
    ) -> str:
        """
        Backup pgvector embeddings.

        Schedule:
        - Full backup: daily at 02:00 UTC
        - Incremental: every 4 hours
        - Retention: 30 days
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = f"/tmp/vectorstore_backup_{timestamp}.sql.gz"
        s3_key = f"backups/vectorstore/{timestamp}/vectorstore_backup.sql.gz"

        cmd = [
            "pg_dump",
            f"--host={db_host}",
            f"--dbname={db_name}",
            f"--username={db_user}",
            "--format=custom",
            "--compress=9",
            "--table=langchain_pg_embedding",
            "--table=langchain_pg_collection",
            f"--file={backup_file}",
        ]
        subprocess.run(cmd, check=True)

        self.s3.upload_file(backup_file, self.bucket, s3_key)
        return s3_key


# --- Kubernetes CronJob for Automated Backups ---
BACKUP_CRONJOB_YAML = """
apiVersion: batch/v1
kind: CronJob
metadata:
  name: agent-checkpoint-backup
  namespace: agent-platform
spec:
  schedule: "0 */6 * * *"          # Every 6 hours
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      backoffLimit: 3
      template:
        spec:
          serviceAccountName: agent-backup-sa
          containers:
          - name: backup
            image: agent-platform/backup-tool:1.4.0
            env:
            - name: DB_HOST
              valueFrom:
                secretKeyRef:
                  name: checkpoint-db-secret
                  key: host
            - name: DB_NAME
              value: agent_checkpoints
            - name: S3_BUCKET
              value: agent-dr-backups-prod
            - name: BACKUP_TYPE
              value: checkpoint
            resources:
              requests:
                cpu: 500m
                memory: 1Gi
              limits:
                cpu: 1000m
                memory: 2Gi
          restartPolicy: OnFailure
---
apiVersion: batch/v1
kind: CronJob
metadata:
  name: agent-vectorstore-backup
  namespace: agent-platform
spec:
  schedule: "0 2 * * *"            # Daily at 02:00 UTC
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      backoffLimit: 3
      template:
        spec:
          serviceAccountName: agent-backup-sa
          containers:
          - name: backup
            image: agent-platform/backup-tool:1.4.0
            env:
            - name: BACKUP_TYPE
              value: vectorstore
            resources:
              requests:
                cpu: 1000m
                memory: 2Gi
          restartPolicy: OnFailure
"""
```

### 4. Multi-Region Active-Passive Setup

```yaml
# Terraform module for multi-region agent DR
# Primary: us-east-1, Standby: us-west-2

# --- Primary Region RDS (PostgreSQL for checkpoints + memory) ---
resource "aws_db_instance" "checkpoint_primary" {
  identifier     = "agent-checkpoints-primary"
  engine         = "postgres"
  engine_version = "16.4"
  instance_class = "db.r6g.xlarge"

  allocated_storage     = 100
  max_allocated_storage = 500
  storage_encrypted     = true

  multi_az               = true       # AZ-level HA within primary region
  backup_retention_period = 7         # 7 days of automated backups
  backup_window          = "03:00-04:00"

  # Enable cross-region read replica for DR
  replicate_source_db = null          # This is the primary

  tags = {
    Service     = "agent-platform"
    Environment = "production"
    DR_Role     = "primary"
  }
}

# --- Standby Region Read Replica ---
resource "aws_db_instance" "checkpoint_standby" {
  provider = aws.us-west-2

  identifier     = "agent-checkpoints-standby"
  instance_class = "db.r6g.xlarge"

  # Cross-region read replica of primary
  replicate_source_db = aws_db_instance.checkpoint_primary.arn

  storage_encrypted = true
  multi_az          = true

  tags = {
    Service     = "agent-platform"
    Environment = "production"
    DR_Role     = "standby"
  }
}

# --- Route53 Health Checks for Automatic DNS Failover ---
resource "aws_route53_health_check" "agent_primary" {
  fqdn              = "agent-api.us-east-1.internal"
  port               = 443
  type               = "HTTPS"
  resource_path      = "/health"
  failure_threshold  = 3
  request_interval   = 10

  tags = {
    Name = "agent-primary-health"
  }
}

resource "aws_route53_record" "agent_api" {
  zone_id = var.route53_zone_id
  name    = "agent-api.example.com"
  type    = "A"

  # Primary record
  set_identifier = "primary"
  failover_routing_policy {
    type = "PRIMARY"
  }
  health_check_id = aws_route53_health_check.agent_primary.id

  alias {
    name                   = aws_lb.agent_primary.dns_name
    zone_id                = aws_lb.agent_primary.zone_id
    evaluate_target_health = true
  }
}

resource "aws_route53_record" "agent_api_secondary" {
  zone_id = var.route53_zone_id
  name    = "agent-api.example.com"
  type    = "A"

  # Secondary (failover) record
  set_identifier = "secondary"
  failover_routing_policy {
    type = "SECONDARY"
  }

  alias {
    name                   = aws_lb.agent_secondary.dns_name
    zone_id                = aws_lb.agent_secondary.zone_id
    evaluate_target_health = true
  }
}
```

### 5. DR Runbook Template

```markdown
## Agent DR Runbook: LLM Provider Outage

### Trigger
- Automated: >50% of LLM calls failing for >2 minutes
- Manual: LLM provider status page shows incident

### Steps

1. **Verify** (0-1 min)
   - Check provider status page (status.anthropic.com / status.openai.com)
   - Confirm via internal metrics: `llm_call_error_rate > 0.5`
   - Determine scope: full outage vs. rate limiting vs. regional

2. **Activate failover** (1-3 min)
   - Circuit breaker should auto-trigger. If not:
     ```bash
     kubectl set env deployment/agent-service \
       LLM_PROVIDER_OVERRIDE=openai \
       -n agent-platform
     ```
   - Verify secondary provider is handling traffic:
     ```bash
     kubectl logs -l app=agent-service -n agent-platform | \
       grep "LLM call succeeded via"
     ```

3. **Communicate** (3-5 min)
   - Post to #agent-platform-incidents Slack channel
   - Update status page if customer-facing impact

4. **Monitor** (ongoing)
   - Watch error rates on secondary provider
   - Monitor latency (secondary may be slower)
   - Track cost impact (different provider = different pricing)

5. **Recovery** (when primary returns)
   - Wait for 5 minutes of stable primary provider health
   - Remove override:
     ```bash
     kubectl set env deployment/agent-service \
       LLM_PROVIDER_OVERRIDE- \
       -n agent-platform
     ```
   - Verify primary is handling traffic
   - Write incident report

### Escalation
- If all providers fail: Page on-call SRE (PagerDuty: agent-platform-critical)
- If data loss suspected: Page database DBA + engineering lead
```

## Decision Tree: Choosing a Recovery Strategy

```
What type of failure?
│
├── LLM provider down
│   ├── Rate limiting only? → Backoff + retry (no failover needed)
│   ├── Single region? → Use provider's secondary region endpoint
│   └── Full outage? → Switch to secondary provider via circuit breaker
│
├── Agent state lost (pod crash)
│   ├── Checkpoint exists? → Resume from checkpoint (seconds)
│   ├── No checkpoint? → Restart session, apologize to user
│   └── Corrupted checkpoint? → Delete checkpoint, restart session
│
├── Memory store failure
│   ├── Primary DB down, standby available? → Promote standby (5-15 min)
│   ├── Data corruption? → PITR from last good backup
│   └── Both regions down? → Restore from S3 backup (30-60 min)
│
├── Tool endpoint failure
│   ├── Single tool? → Circuit breaker, agent works without it
│   ├── Critical tool? → Queue requests, retry when available
│   └── All tools? → Read-only mode with cached responses
│
└── Full region loss
    ├── Active-passive configured? → DNS failover to standby (5-10 min)
    └── No standby? → Provision new region from IaC + restore backups (1-4 hr)
```

## When NOT to Use Full DR

- **Prototype / MVP stage**: Focus on shipping. DR adds infra complexity.
- **< 100 users**: Manual recovery from backups is acceptable.
- **Non-critical agents** (internal tools, demos): Simple restart with apology is fine.
- **Stateless agents** (no memory, no multi-turn): Standard K8s pod restart is sufficient DR.
- **Cost-constrained**: Multi-region standby doubles your infrastructure cost.

## Tradeoffs

| Strategy | RTO | RPO | Monthly Cost Overhead | Complexity |
|----------|-----|-----|----------------------|------------|
| No DR (restart from scratch) | 5-30 min | Total loss | $0 | None |
| Checkpoint-only (single region) | < 1 min | Last checkpoint (~seconds) | ~$50 (DB storage) | Low |
| Multi-AZ within region | < 5 min | ~0 (sync replication) | +30% infra cost | Medium |
| Cross-region active-passive | 5-15 min | < 1 min (async replication) | +80-100% infra cost | High |
| Cross-region active-active | < 1 min | ~0 | +150-200% infra cost | Very high |
| Multi-provider LLM failover | < 30 sec | N/A (stateless calls) | ~$20/mo (health checks) | Medium |

### RTO/RPO Targets by Agent Type

| Agent Type | Target RTO | Target RPO | Justification |
|-----------|-----------|-----------|---------------|
| Stateless (single-turn Q&A) | < 30 sec | N/A | Pod restart is sufficient |
| Stateful chat (multi-turn) | < 5 min | < 30 sec | Checkpoints cover state |
| Long-running workflow agent | < 15 min | < 1 min | Complex state to recover |
| Mission-critical (financial) | < 2 min | ~0 | Business impact too high |
| Internal tooling agent | < 30 min | < 1 hr | Lower SLA acceptable |

## Failure Modes

1. **Split-brain during failover**: Both primary and standby accept writes after network partition heals. Mitigation: use fencing (STONITH) or leader election; PostgreSQL streaming replication prevents this if standby is read-only until promoted.

2. **Stale checkpoint resume**: Agent resumes from a checkpoint but the external world has changed (e.g., tool call already executed). Mitigation: make tool calls idempotent; include external state validation in checkpoint recovery logic.

3. **Provider failover quality degradation**: Secondary LLM provider produces lower-quality outputs, causing downstream tool call failures. Mitigation: test failover providers in CI with the same eval suite; accept quality tradeoff as documented.

4. **Backup corruption undetected**: Backups run on schedule but produce corrupted files. Mitigation: automated backup verification (restore to test DB and run integrity checks weekly).

5. **Memory store replication lag**: Cross-region replication lag means standby is missing recent memories. Mitigation: monitor replication lag metric; during failover, warn users that recent context may be missing.

6. **Cascading failover**: Primary LLM fails → secondary overloaded → secondary fails → local model overloaded. Mitigation: rate-limit failover traffic; shed load (queue excess requests) rather than cascading.

## Source(s) and Further Reading

- LangGraph Persistence: https://langchain-ai.github.io/langgraph/concepts/persistence/
- PostgresSaver Reference: https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.postgres
- AWS Multi-Region DR: https://docs.aws.amazon.com/whitepapers/latest/disaster-recovery-workloads-on-aws/disaster-recovery-options-in-the-cloud.html
- Circuit Breaker Pattern: https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker
- PostgreSQL PITR: https://www.postgresql.org/docs/16/continuous-archiving.html
- "Designing Data-Intensive Applications" - Martin Kleppmann (Chapter 9: Consistency and Consensus)
- AWS RTO/RPO Whitepaper: https://docs.aws.amazon.com/whitepapers/latest/disaster-recovery-workloads-on-aws/business-continuity-plan-bcp.html
