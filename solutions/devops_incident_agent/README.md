# Design a DevOps Incident Response Agent

> Note: This document links directly to relevant areas of the
> [Agentic AI System Design Primer](../../README.md)

## Step 1: Outline use cases and constraints

### Use cases

- **Detect anomalies** — monitor alerts from Prometheus/Grafana and correlate related alerts into a single incident
- **Diagnose root cause** — query metrics, logs, and traces to narrow down the failing component, service, and root cause
- **Execute runbook steps** — follow pre-defined runbook procedures (read-only diagnostics first, remediation with human approval)
- **Page on-call** — notify the right person via PagerDuty based on service ownership and escalation policies
- **Communicate status** — post incident updates to Slack war room with current findings and status
- **Write postmortem** — after resolution, generate a structured postmortem document with timeline, root cause, impact, and action items
- **Learn from past incidents** — reference similar past incidents and their resolutions to accelerate diagnosis

### Constraints and assumptions

#### State assumptions

- **50 services** in production (microservices architecture)
- **200 incidents/month** (~7/day on average, peak ~15/day during deploy windows)
- **Severity distribution**: P1 (5%), P2 (15%), P3 (40%), P4 (40%)
- **Latency target**: alert → first diagnostic action < 30 seconds; preliminary diagnosis < 5 minutes
- **MTTR target**: P1 < 30 min (currently 45 min), P2 < 2 hours (currently 3 hours)
- **Accuracy bar**: 80% correct root cause identification (matching human engineer's diagnosis)
- **Safety**: ABSOLUTELY NO WRITE operations to production infrastructure without human approval. Agent has READ-ONLY access by default.
- **Monitoring stack**: Prometheus + Grafana (metrics), Coralogix/ELK (logs), Jaeger (traces), PagerDuty (alerting), Slack (communication)
- **Runbooks**: 30 documented runbooks covering top incident types, stored in Confluence/Notion

#### Calculate usage

```
Incidents/month:              200
Incidents/day:                ~7 (peak: 15)
Agent runs per incident:      ~15 (diagnostic steps, queries, updates)
Total agent runs/day:         ~105
Agent runs/second:            ~0.001 (bursty — multiple runs in rapid succession during incident)

Tokens per incident (avg):
  System prompt:              2,500 tokens (includes runbook context, service map)
  Alert context:              500 tokens (alert payload, labels, metadata)
  Metric query results:       3,000 tokens (~5 Prometheus queries x 600 tokens)
  Log search results:         5,000 tokens (~3 log searches x 1,667 tokens)
  Trace analysis:             2,000 tokens
  Past incident context:      2,000 tokens (similar incidents from episodic memory)
  Reasoning/diagnosis:        3,000 tokens
  Slack updates + postmortem: 2,000 tokens
  Total input:                20,000 tokens/incident
  Total output:               5,000 tokens/incident

Tokens/day:
  Input:  7 incidents x 20,000  = 140K input tokens
  Output: 7 incidents x 5,000   = 35K output tokens

Cost/day (Claude 3.5 Sonnet):
  Input:  0.14M x $3/M    = $0.42
  Output: 0.035M x $15/M  = $0.53
  Total:                   = $0.95/day -> ~$28.50/month

Cost per incident: $0.95 / 7 = $0.14/incident

Additional infrastructure:
  Prometheus queries:     Free (self-hosted)
  Log searches:           Included in Coralogix license
  PagerDuty:              Included in existing plan
  Slack:                  Free (bot posts)
  Total infra:            ~$0/month incremental

Total monthly cost: ~$28.50 (LLM only) + ~$200 (compute for agent service) = ~$230/month
```

**ROI calculation:**
- Average P1 incident: 45 min MTTR x 3 engineers x $100/hour = $225/incident
- With AI agent: 30 min MTTR x 2 engineers (agent handles initial diagnosis) = $100/incident
- **Savings per P1 incident: $125**
- 10 P1 incidents/month x $125 = $1,250/month in engineer time saved
- Plus: faster diagnosis prevents revenue loss during outages

### Out of scope

- Infrastructure provisioning or capacity planning
- CI/CD pipeline management
- Security incident response (SOC/SIEM — separate domain)
- Multi-cloud incident correlation (single cloud for v1)
- Automated code rollback (too high risk for v1)

---

## Step 2: Create a high-level design

```
 Alert Sources (Prometheus/Grafana/PagerDuty)
        │
        ▼
 ┌────────────────────────────────────────────────────────────┐
 │                   ALERT INGESTION                          │
 │  (Deduplicate, correlate, enrich with service metadata)    │
 └──────────────────────────┬─────────────────────────────────┘
                            │
 ┌──────────────────────────▼─────────────────────────────────┐
 │                   TRIAGE CLASSIFIER                        │
 │  (Determine severity, owning team, relevant runbook)       │
 │  (Fast model: Haiku for <1s classification)                │
 └──────────────────────────┬─────────────────────────────────┘
                            │
     ┌──────────────────────▼──────────────────────┐
     │          DIAGNOSTIC AGENT (ReAct)            │
     │                                              │
     │  Observe → Hypothesize → Query → Refine      │
     │                                              │
     │  Accesses:                                    │
     │  - Prometheus (metrics)                       │
     │  - Coralogix (logs)                          │
     │  - Jaeger (traces)                           │
     │  - kubectl (read-only)                       │
     │  - Service dependency map                    │
     │  - Runbook library                           │
     │  - Past incident database                    │
     └──────────────┬──────────────────────────────┘
                    │
                    │ diagnosis + recommended actions
                    │
     ┌──────────────▼──────────────────────────────┐
     │        REMEDIATION AGENT (with HITL)         │
     │                                              │
     │  1. Present diagnosis to on-call             │
     │  2. Propose remediation steps                │
     │  3. WAIT for human approval                  │
     │  4. Execute approved steps (if any)           │
     │  5. Verify fix                                │
     └──────────────┬──────────────────────────────┘
                    │
     ┌──────────────▼──────────────────────────────┐
     │         COMMUNICATION LAYER                  │
     │                                              │
     │  ┌────────────┐  ┌────────────┐  ┌────────┐ │
     │  │  Slack      │  │ PagerDuty  │  │ Status │ │
     │  │  War Room   │  │ (Paging)   │  │ Page   │ │
     │  └────────────┘  └────────────┘  └────────┘ │
     └──────────────┬──────────────────────────────┘
                    │
     ┌──────────────▼──────────────────────────────┐
     │         POSTMORTEM GENERATOR                 │
     │                                              │
     │  After resolution:                           │
     │  - Timeline reconstruction                   │
     │  - Root cause analysis                       │
     │  - Impact assessment                         │
     │  - Action items                              │
     └──────────────┬──────────────────────────────┘
                    │
     ┌──────────────▼──────────────────────────────┐
     │         MEMORY LAYER                         │
     │                                              │
     │  ┌──────────────────────┐  ┌──────────────┐ │
     │  │ Episodic Memory       │  │ Service Map  │ │
     │  │ (Past incidents,      │  │ (Dependencies│ │
     │  │  resolutions,         │  │  ownership,  │ │
     │  │  runbook outcomes)    │  │  SLOs)       │ │
     │  │                       │  │              │ │
     │  │ PostgreSQL + Qdrant   │  │ PostgreSQL   │ │
     │  └──────────────────────┘  └──────────────┘ │
     └──────────────────────────────────────────────┘
                    │
     ┌──────────────▼──────────────────────────────┐
     │      Observability (Langfuse)                │
     │   Diagnosis accuracy, MTTR, cost/incident    │
     └──────────────────────────────────────────────┘
```

---

## Step 3: Design core components

### Agent Architecture Decision

**Multi-Agent: Diagnostic Agent + Remediation Agent with HITL gate between them.**

> See: [Multi-Agent Patterns — Supervisor](../../README.md#supervisor-pattern)

**Justification for two agents (not one):**

1. **Security boundary**: The Diagnostic Agent has READ-ONLY access to infrastructure. The Remediation Agent has WRITE access but ONLY operates after human approval. This is a hard security boundary that must be architecturally enforced, not just prompt-enforced.
2. **Different risk profiles**: Diagnosis is safe to fully automate (worst case: wasted queries). Remediation is dangerous to automate (worst case: cascading failure from wrong action).
3. **Different tool sets**: Diagnostic tools (Prometheus, logs, traces) vs. Remediation tools (kubectl apply, service restart, config change). Mixing them in one agent increases the attack surface.

**Why NOT single-agent with tool restrictions:**
- Prompt-based tool restrictions are not sufficient for production safety. A prompt injection or reasoning error could lead the agent to call a remediation tool during the diagnostic phase.
- Separate agents with separate tool definitions make it architecturally impossible for the diagnostic phase to trigger a remediation action.

### Pattern Selection

**ReAct for both agents — diagnosis is inherently exploratory.**

> See: [ReAct Pattern](../../README.md#react-reasoning--acting)

**Why ReAct (not Plan-and-Execute):**
- Incident diagnosis is fundamentally reactive: each observation changes the hypothesis
- You can't pre-plan a diagnosis because you don't know what's wrong yet
- Each metric query, log search, or trace reveals new information that redirects the investigation
- This mirrors how human engineers debug: look, think, investigate, refine

**Diagnostic Agent ReAct loop example:**

```
Alert: "High error rate on payment-service (5xx > 10% for 5 minutes)"

Thought: High 5xx rate on payment-service. Let me check the error rate trend and
         which specific endpoints are failing.
Action: query_prometheus(
  query='sum(rate(http_requests_total{service="payment-service", status=~"5.."}[5m])) by (endpoint)',
  range="30m"
)
Observation: /api/v1/charge endpoint has 35% error rate. Other endpoints are normal.

Thought: The /charge endpoint is failing. Let me check what errors are being returned.
Action: search_logs(
  query='service:payment-service AND endpoint:/api/v1/charge AND level:error',
  time_range="15m",
  limit=20
)
Observation: Repeated error: "connection refused: downstream-risk-service:8080"

Thought: payment-service can't reach risk-service. Let me check if risk-service is running.
Action: kubectl_read(
  command="get pods -n payments -l app=risk-service"
)
Observation: risk-service-7d4f8b6c9-abc12  0/1  CrashLoopBackOff  5  3m

Thought: risk-service is in CrashLoopBackOff. Let me check the pod logs for the crash reason.
Action: search_logs(
  query='service:risk-service AND level:fatal',
  time_range="15m",
  limit=5
)
Observation: "FATAL: failed to connect to Redis: connection refused. Host: redis-risk.payments.svc"

Thought: risk-service is crashing because it can't connect to its Redis instance.
         Let me check Redis status.
Action: kubectl_read(
  command="get pods -n payments -l app=redis-risk"
)
Observation: redis-risk-0  0/1  OOMKilled  1  2m

Thought: ROOT CAUSE FOUND. Redis for risk-service was OOMKilled, causing risk-service
         to crash, which caused payment-service /charge endpoint to fail.

DIAGNOSIS:
- Root cause: redis-risk OOMKilled (memory limit exceeded)
- Impact: payment-service /charge endpoint 35% error rate for ~5 minutes
- Affected services: payment-service → risk-service → redis-risk
- Recommended action: Increase redis-risk memory limit from 512Mi to 1Gi
```

### Tool Design

#### Tool 1: Prometheus Query

```json
{
  "name": "query_prometheus",
  "description": "Execute a PromQL query against Prometheus to retrieve metric data. Returns time series data for the specified range. Use for checking error rates, latency, resource utilization, and custom metrics.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "PromQL expression (e.g., 'rate(http_requests_total{status=~\"5..\"}[5m])')"
      },
      "range": {
        "type": "string",
        "default": "30m",
        "description": "Time range to query (e.g., '15m', '1h', '6h'). Max: 24h"
      },
      "step": {
        "type": "string",
        "default": "1m",
        "description": "Query resolution step (e.g., '30s', '1m', '5m')"
      }
    },
    "required": ["query"]
  },
  "returns": {
    "type": "object",
    "properties": {
      "result_type": { "type": "string", "enum": ["matrix", "vector", "scalar"] },
      "results": [{
        "metric": { "type": "object", "description": "Label set" },
        "values": [["timestamp", "value"]]
      }],
      "summary": { "type": "string", "description": "Human-readable summary of the data" }
    }
  },
  "error_contract": {
    "INVALID_QUERY": "PromQL syntax error",
    "QUERY_TIMEOUT": "Query exceeded 30s timeout (reduce range or simplify query)",
    "NO_DATA": "No data points found for this metric/label combination",
    "PROMETHEUS_UNAVAILABLE": "Cannot reach Prometheus instance"
  },
  "safety": "READ-ONLY. Cannot modify recording rules, alerts, or configuration."
}
```

#### Tool 2: Log Search

```json
{
  "name": "search_logs",
  "description": "Search application logs in Coralogix/ELK. Supports structured queries with service, level, and time filters. Returns log entries with timestamps and metadata.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query (Lucene syntax). Example: 'service:payment-service AND level:error AND \"connection refused\"'"
      },
      "time_range": {
        "type": "string",
        "default": "15m",
        "description": "Time range (e.g., '5m', '1h'). Max: 24h"
      },
      "limit": {
        "type": "integer",
        "default": 20,
        "maximum": 100
      },
      "sort": {
        "type": "string",
        "enum": ["timestamp_desc", "timestamp_asc"],
        "default": "timestamp_desc"
      }
    },
    "required": ["query"]
  },
  "returns": {
    "logs": [{
      "timestamp": "string (ISO 8601)",
      "service": "string",
      "level": "string",
      "message": "string",
      "trace_id": "string | null",
      "metadata": "object"
    }],
    "total_hits": "integer",
    "truncated": "boolean"
  },
  "error_contract": {
    "INVALID_QUERY": "Query syntax error",
    "LOG_STORE_UNAVAILABLE": "Cannot reach log store",
    "QUERY_TIMEOUT": "Log search exceeded 30s timeout",
    "NO_RESULTS": "No logs match the query"
  },
  "safety": "READ-ONLY. Cannot modify log configuration or retention."
}
```

#### Tool 3: Kubernetes Read Operations

```json
{
  "name": "kubectl_read",
  "description": "Execute READ-ONLY kubectl commands against production clusters. Only get/describe/logs/top operations are permitted. No apply, delete, patch, edit, or scale operations.",
  "parameters": {
    "type": "object",
    "properties": {
      "command": {
        "type": "string",
        "description": "kubectl command (read-only). Examples: 'get pods -n payments', 'describe pod risk-service-abc12 -n payments', 'logs risk-service-abc12 -n payments --tail=50', 'top pods -n payments'"
      },
      "cluster": {
        "type": "string",
        "default": "production",
        "description": "Cluster context to query"
      }
    },
    "required": ["command"]
  },
  "returns": {
    "output": "string (kubectl output)",
    "exit_code": "integer"
  },
  "error_contract": {
    "COMMAND_BLOCKED": "Command is not read-only (blocked: apply, delete, patch, edit, scale, exec, port-forward)",
    "NAMESPACE_DENIED": "Access denied to this namespace",
    "CLUSTER_UNAVAILABLE": "Cannot reach cluster API server",
    "POD_NOT_FOUND": "Specified pod/resource not found"
  },
  "safety": {
    "allowed_verbs": ["get", "describe", "logs", "top", "explain"],
    "blocked_verbs": ["apply", "delete", "patch", "edit", "scale", "exec", "port-forward", "run", "create"],
    "enforcement": "Command is parsed and validated BEFORE execution. Blocked commands are rejected without execution."
  }
}
```

#### Tool 4: PagerDuty Operations

```json
{
  "name": "pagerduty_operation",
  "description": "Interact with PagerDuty for incident management and on-call paging.",
  "parameters": {
    "type": "object",
    "properties": {
      "operation": {
        "type": "string",
        "enum": ["get_on_call", "create_incident", "update_incident", "add_note", "get_incident"]
      },
      "service_id": { "type": "string" },
      "incident_id": { "type": "string" },
      "severity": {
        "type": "string",
        "enum": ["critical", "error", "warning", "info"]
      },
      "title": { "type": "string" },
      "description": { "type": "string" },
      "note": { "type": "string" }
    },
    "required": ["operation"]
  },
  "error_contract": {
    "SERVICE_NOT_FOUND": "PagerDuty service ID not found",
    "ALREADY_PAGED": "On-call engineer already paged for this incident",
    "PD_API_ERROR": "PagerDuty API error"
  }
}
```

#### Tool 5: Slack Notifications

```json
{
  "name": "slack_notify",
  "description": "Post incident updates to Slack channels. Creates war room channels for P1/P2 incidents.",
  "parameters": {
    "type": "object",
    "properties": {
      "operation": {
        "type": "string",
        "enum": ["post_update", "create_war_room", "post_to_war_room"]
      },
      "channel": { "type": "string", "description": "Channel name or ID" },
      "message": { "type": "string" },
      "severity": { "type": "string", "enum": ["P1", "P2", "P3", "P4"] },
      "incident_id": { "type": "string" },
      "blocks": {
        "type": "array",
        "description": "Slack Block Kit formatted message (optional)"
      }
    },
    "required": ["operation", "message"]
  }
}
```

#### Tool 6: Runbook Lookup

```json
{
  "name": "lookup_runbook",
  "description": "Search and retrieve relevant runbook procedures for the current incident type. Runbooks are stored in Confluence/Notion with structured steps.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query (e.g., 'redis OOMKilled', 'high error rate payment-service')"
      },
      "service": { "type": "string" },
      "incident_type": {
        "type": "string",
        "enum": ["crash_loop", "high_error_rate", "high_latency", "oom_killed", "pod_eviction",
                 "disk_pressure", "certificate_expiry", "connection_pool_exhaustion",
                 "database_replication_lag", "deployment_failure"]
      }
    },
    "required": ["query"]
  },
  "returns": {
    "runbook": {
      "title": "string",
      "last_updated": "string",
      "steps": [{
        "step_number": "integer",
        "description": "string",
        "command": "string | null",
        "requires_approval": "boolean",
        "category": "diagnostic | remediation"
      }],
      "estimated_resolution_time": "string",
      "escalation_path": "string"
    }
  }
}
```

### Memory Architecture

> See: [Memory Systems](../../README.md#memory-systems)

| Memory Type | Store | Scope | Purpose |
|---|---|---|---|
| **Episodic (Past Incidents)** | PostgreSQL + Qdrant | Global, permanent | Past incident → diagnosis → resolution mappings for pattern matching |
| **Service Map** | PostgreSQL | Global, updated by deploys | Service dependencies, ownership, SLOs, runbook references |
| **Working Memory** | Redis + LangGraph state | Per-incident, TTL 7 days | Current diagnostic state, hypotheses, queries run, results |
| **Runbook Cache** | Redis | Global, 1-hour TTL | Cached runbook content from Confluence/Notion |

**Episodic Memory (Past Incidents):**

```json
{
  "incident_id": "inc_20260510_001",
  "severity": "P1",
  "title": "Payment service 5xx spike caused by Redis OOM",
  "services_affected": ["payment-service", "risk-service", "redis-risk"],
  "root_cause": "redis-risk OOMKilled due to memory limit 512Mi with growing cache",
  "symptoms": [
    "payment-service 5xx rate > 10%",
    "risk-service CrashLoopBackOff",
    "redis-risk OOMKilled"
  ],
  "diagnostic_path": [
    "Checked payment-service error rate → found /charge endpoint failing",
    "Checked payment-service logs → found 'connection refused: risk-service'",
    "Checked risk-service pods → found CrashLoopBackOff",
    "Checked risk-service logs → found 'failed to connect to Redis'",
    "Checked redis-risk pods → found OOMKilled"
  ],
  "resolution": "Increased redis-risk memory limit to 1Gi. Added memory alerts at 80% threshold.",
  "time_to_detect": "2 min",
  "time_to_diagnose": "8 min",
  "time_to_resolve": "22 min",
  "action_items": [
    "Add memory monitoring alerts for all Redis instances",
    "Set up cache eviction policies for risk-service Redis"
  ],
  "embedding_vector": "[0.023, -0.145, ...]"
}
```

**Similar Incident Retrieval:**
When a new incident starts, the agent:
1. Embeds the alert payload + initial symptoms
2. Searches episodic memory for similar past incidents (vector similarity)
3. If a match is found (similarity > 0.85), uses the past diagnostic path as a starting hypothesis
4. This can reduce diagnosis time from 8 minutes to 2-3 minutes for recurring patterns

### API Design

#### Ingest Alert (webhook from Prometheus/Grafana)

```
POST /api/v1/incidents/ingest
```

**Request (Alertmanager webhook format):**
```json
{
  "alerts": [
    {
      "status": "firing",
      "labels": {
        "alertname": "HighErrorRate",
        "service": "payment-service",
        "severity": "critical",
        "namespace": "payments"
      },
      "annotations": {
        "summary": "payment-service has > 10% 5xx error rate",
        "description": "Error rate is 15.3% over the last 5 minutes",
        "runbook_url": "https://wiki.internal/runbooks/high-error-rate"
      },
      "startsAt": "2026-05-19T10:30:00Z"
    }
  ]
}
```

**Response:**
```json
{
  "incident_id": "inc_20260519_003",
  "severity": "P1",
  "status": "investigating",
  "assigned_to": "oncall: alice@company.com",
  "war_room": "#inc-20260519-003-payment-service",
  "diagnostic_agent": "started",
  "similar_incidents": [
    {
      "incident_id": "inc_20260510_001",
      "similarity": 0.91,
      "title": "Payment service 5xx spike caused by Redis OOM",
      "resolution": "Increased redis-risk memory limit"
    }
  ],
  "estimated_diagnosis_time": "3-5 minutes"
}
```

#### Get Incident Status

```
GET /api/v1/incidents/{incident_id}
```

**Response:**
```json
{
  "incident_id": "inc_20260519_003",
  "severity": "P1",
  "status": "diagnosed",
  "timeline": [
    {
      "timestamp": "2026-05-19T10:30:00Z",
      "event": "Alert received: HighErrorRate on payment-service",
      "actor": "alertmanager"
    },
    {
      "timestamp": "2026-05-19T10:30:05Z",
      "event": "Diagnostic agent started. Paged alice@company.com via PagerDuty.",
      "actor": "agent"
    },
    {
      "timestamp": "2026-05-19T10:30:15Z",
      "event": "Found similar past incident inc_20260510_001 (91% match). Following similar diagnostic path.",
      "actor": "agent"
    },
    {
      "timestamp": "2026-05-19T10:31:00Z",
      "event": "Prometheus: /api/v1/charge endpoint 35% error rate. Other endpoints normal.",
      "actor": "agent"
    },
    {
      "timestamp": "2026-05-19T10:31:30Z",
      "event": "Logs: 'connection refused: risk-service:8080' in payment-service",
      "actor": "agent"
    },
    {
      "timestamp": "2026-05-19T10:32:00Z",
      "event": "kubectl: risk-service in CrashLoopBackOff. redis-risk OOMKilled.",
      "actor": "agent"
    },
    {
      "timestamp": "2026-05-19T10:32:15Z",
      "event": "ROOT CAUSE: redis-risk OOMKilled (512Mi limit). Same root cause as inc_20260510_001.",
      "actor": "agent"
    }
  ],
  "diagnosis": {
    "root_cause": "redis-risk OOMKilled due to memory limit 512Mi",
    "impact": "payment-service /charge endpoint 35% error rate affecting ~500 transactions/min",
    "affected_services": ["payment-service", "risk-service", "redis-risk"],
    "dependency_chain": "payment-service → risk-service → redis-risk (OOMKilled)",
    "confidence": 0.92
  },
  "recommended_actions": [
    {
      "action": "Increase redis-risk memory limit to 1Gi",
      "command": "kubectl -n payments patch statefulset redis-risk -p '{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"redis\",\"resources\":{\"limits\":{\"memory\":\"1Gi\"}}}]}}}}'",
      "risk": "low",
      "requires_approval": true,
      "status": "pending_approval"
    },
    {
      "action": "Restart risk-service pods after Redis is stable",
      "command": "kubectl -n payments rollout restart deployment risk-service",
      "risk": "low",
      "requires_approval": true,
      "status": "pending_approval"
    }
  ],
  "metadata": {
    "diagnostic_duration_seconds": 135,
    "llm_calls": 8,
    "tokens_used": { "input": 18000, "output": 4200 },
    "cost_usd": 0.12,
    "tools_invoked": ["query_prometheus", "search_logs", "kubectl_read", "lookup_runbook", "pagerduty_operation", "slack_notify"]
  }
}
```

---

## Step 4: Scale the design

### Bottleneck Analysis

| Bottleneck | Symptom | At Scale (200 incidents/month) | Mitigation |
|---|---|---|---|
| Alert storm (correlated alerts) | 50+ alerts fire simultaneously for one outage | Agent spawns 50 independent diagnoses | Alert correlation: group alerts by service dependency graph. Create one incident per root service. Suppress downstream alerts. |
| Prometheus query latency | Slow during high-cardinality queries | Long-range queries (6h+) time out | Pre-compute key diagnostic queries as recording rules. Limit query range to 1h for initial diagnosis. Fan out to longer ranges only if needed. |
| Log search volume | Coralogix/ELK slow under load | Concurrent log searches during incidents | Targeted queries (specific service + time range). Use sampling for high-volume services. Cache repeated queries within an incident. |
| Concurrent P1 incidents | Multiple P1s simultaneously | Agent resources spread thin | Priority queue: P1 > P2. Dedicate separate agent instances per active P1. Queue P3/P4 for async diagnosis. |
| Context window with long incidents | Multi-hour incidents overflow context | Diagnostic history grows beyond window | Summarize older diagnostic steps. Keep last 10 steps in full detail, compress earlier steps. Max 15 ReAct iterations before escalating. |

### Cost Estimation

| Component | Per Incident | Monthly (200) | Notes |
|---|---|---|---|
| Claude Sonnet (diagnosis + postmortem) | $0.12 | $24 | ~8 LLM calls/incident |
| Claude Haiku (triage + classification) | $0.005 | $1 | 1 call/alert for classification |
| Prometheus queries | $0 | $0 | Self-hosted |
| Log searches (Coralogix) | $0 | $0 | Included in license |
| PagerDuty API | $0 | $0 | Included in license |
| Slack API | $0 | $0 | Free |
| Qdrant (episodic memory) | — | $50 | Self-hosted, minimal resources |
| Agent compute (ECS/K8s) | — | $200 | 2 vCPU, 4GB RAM, always-on |
| Langfuse (observability) | $0.01 | $2 | |
| **Total** | **~$0.14** | **~$277** | |

This is an exceptionally cost-effective system. At $0.14/incident and $277/month total, the ROI is enormous:
- **P1 MTTR reduction** (45min → 30min): saves $1,250/month in engineer time
- **Faster diagnosis for P2**: saves ~$2,000/month
- **Postmortem automation**: saves ~$500/month (2 hours engineer time x $100/hr x 5 postmortems)
- **Total monthly savings: ~$3,750 vs $277 cost = 13.5x ROI**

### Failure Modes

> See: [Failure Modes & Mitigation](../../README.md#failure-modes--mitigation)

| Failure Mode | Impact | Likelihood | Mitigation |
|---|---|---|---|
| **Wrong root cause diagnosis** | Engineers chase wrong lead, extending MTTR | Medium | Present diagnosis with confidence score. If confidence < 0.7, explicitly say "uncertain — multiple hypotheses." Show all evidence for and against. Always page the human engineer — the agent assists, never replaces. |
| **Executing wrong runbook** | Remediation worsens the incident | Low (with HITL) | Remediation Agent NEVER executes without human approval. Agent proposes actions with full commands shown. Human reviews and clicks "Approve" or "Reject" in Slack. No timeout auto-approve for P1/P2. |
| **Alert fatigue from false positives** | Engineers ignore the agent | Medium | Track agent accuracy over time (was diagnosis correct?). If accuracy drops below 70% for a specific alert type, disable auto-diagnosis for that alert and flag for tuning. Aggregate and deduplicate alerts before creating incidents. |
| **Agent unavailable during incident** | No automated assistance | Low | Agent runs as a highly-available service (3 replicas, multi-AZ). Fallback: if agent is down, alerts still go to PagerDuty directly (existing path). Agent is additive, not a replacement for existing alerting. |
| **Circular dependency in diagnosis** | Agent keeps querying the same metrics in a loop | Medium | Track queries within an incident session. If the same query is executed 3 times, break the loop and present current findings. Max 15 ReAct iterations per diagnosis. |
| **Stale runbook** | Agent follows outdated procedures | Medium | Runbooks have a "last_updated" timestamp. If runbook is > 90 days old, warn the engineer: "This runbook hasn't been updated in 90 days — verify steps before proceeding." Track which runbooks lead to successful resolutions and flag stale ones. |
| **Confidential data in Slack updates** | Sensitive data posted to broad channels | Low | PII/secret detection on all Slack messages before posting. Never include credentials, tokens, or customer data in updates. Mask any detected secrets. |

### Observability Setup

> See: [Observability & Monitoring](../../README.md#observability--monitoring)

**Key Metrics:**

| Metric | Target | Alert Threshold |
|---|---|---|
| Time to first diagnostic action | < 30 seconds | > 2 minutes |
| Diagnosis accuracy (validated post-resolution) | > 80% | < 65% |
| P1 MTTR | < 30 minutes | > 45 minutes |
| P2 MTTR | < 2 hours | > 3 hours |
| Alert-to-page latency | < 60 seconds | > 3 minutes |
| False positive diagnosis rate | < 15% | > 25% |
| Runbook match rate | > 70% | < 50% |
| Cost per incident | < $0.20 | > $0.50 |
| Agent availability | 99.9% | < 99.5% |
| Postmortem generated within 24h | > 90% | < 70% |

**Tracing (Langfuse):**
```
Trace: inc_20260519_003 (Payment service 5xx spike)
  ├── Alert Ingestion (0.1s)
  │   └── Correlated 3 related alerts into 1 incident
  ├── Triage (Haiku, 0.5s)
  │   └── Severity: P1, Service: payment-service, Runbook: high-error-rate
  ├── Similar Incident Search (Qdrant, 0.2s)
  │   └── Match: inc_20260510_001 (91% similarity)
  ├── Diagnostic Agent (ReAct, 8 iterations, 125s total)
  │   ├── Iter 1: query_prometheus (error rate by endpoint, 2.1s)
  │   ├── Iter 2: search_logs (payment-service errors, 3.2s)
  │   ├── Iter 3: kubectl_read (risk-service pods, 1.5s)
  │   ├── Iter 4: search_logs (risk-service fatal, 2.8s)
  │   ├── Iter 5: kubectl_read (redis-risk pods, 1.3s)
  │   ├── Iter 6: LLM reasoning (Sonnet, 2.5s, diagnosis formed)
  │   ├── Iter 7: lookup_runbook (redis OOMKilled, 0.8s)
  │   └── Iter 8: LLM reasoning (Sonnet, 1.8s, remediation plan)
  ├── Communication
  │   ├── pagerduty: page alice@company.com (0.5s)
  │   ├── slack: create war room (0.8s)
  │   └── slack: post diagnosis + recommended actions (0.3s)
  ├── HITL Gate: Awaiting approval for remediation
  │   └── alice@company.com approved 2 actions (4m 22s wait)
  ├── Remediation Agent
  │   ├── kubectl apply: increase redis memory (3.2s, approved)
  │   ├── kubectl rollout: restart risk-service (2.1s, approved)
  │   └── Verification: error rate back to <0.1% (checked at 2m intervals)
  └── Postmortem Generation (Sonnet, 5.2s)
  Total: 135s diagnostic + 262s wait + 12s remediation = 6m 49s
  Cost: $0.14 | Diagnosis correct: YES (validated)
```

---

## Additional Talking Points

### Safety Architecture (Critical Topic)
> See: [Safety & Guardrails](../../README.md#safety--guardrails)

This is the most important aspect of the design. The safety model is **defense in depth**:

1. **Architectural separation**: Diagnostic Agent and Remediation Agent are separate processes with separate tool configurations. The Diagnostic Agent literally cannot call kubectl apply — the tool is not in its tool list.

2. **Command validation layer**: Even the Remediation Agent's kubectl commands are parsed and validated before execution:
   ```python
   ALLOWED_REMEDIATION_COMMANDS = [
       r"kubectl .* rollout restart deployment .*",
       r"kubectl .* scale deployment .* --replicas=\d+",
       r"kubectl .* patch (deployment|statefulset) .* -p .*",
   ]
   BLOCKED_PATTERNS = [
       r"kubectl .* delete .*",
       r"kubectl .* exec .*",
       r"kubectl .* apply -f https?://.*",  # No remote manifests
   ]
   ```

3. **Human approval gate**: No remediation action executes without explicit human approval via Slack button or CLI confirmation.

4. **Blast radius control**: Agent can only operate on namespaces it owns. It cannot touch `kube-system`, `monitoring`, or cross-tenant namespaces.

5. **Audit trail**: Every command (proposed AND executed) is logged with full context: who approved, what was executed, what was the outcome.

### Postmortem Generation
- After incident resolution, the agent generates a structured postmortem:
  - **Timeline**: Reconstructed from agent trace + alert timestamps + Slack messages
  - **Root Cause**: From diagnosis (validated by engineer)
  - **Impact**: Calculated from metrics (error rate x traffic = affected requests)
  - **Detection**: How the alert fired and how long it took
  - **Response**: What the agent did and what the human did
  - **Action Items**: Generated from: (a) runbook gaps found, (b) monitoring gaps, (c) similar incident prevention
- Postmortem is posted to Confluence and linked in the incident Slack channel
- Engineers review and edit before finalizing

### Episodic Memory and Learning
> See: [Memory Systems](../../README.md#memory-systems)
- Every resolved incident is stored in episodic memory with its full diagnostic path
- Over time, the agent builds a "diagnostic playbook" that goes beyond static runbooks
- For recurring incidents (>3 times same root cause), the agent flags this as a systemic issue and creates an action item to fix the underlying problem
- The similarity search enables "transfer learning" between incidents: if `redis-risk` OOMKilled is similar to `redis-cache` OOMKilled, the diagnostic path transfers

### Integration with Existing Incident Workflow
- The agent does NOT replace the existing incident workflow — it augments it
- PagerDuty still pages the on-call engineer
- Slack war room is still created
- The difference: when the engineer arrives, they find a preliminary diagnosis, relevant metrics, and suggested next steps already in the war room
- Engineer's job shifts from "figure out what's wrong" to "validate the agent's diagnosis and approve remediation"
- This typically saves 10-15 minutes on P1 incidents
