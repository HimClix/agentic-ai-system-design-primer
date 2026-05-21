# Non-Functional Requirements Checklist for Agentic AI Systems
> Traditional NFRs focus on latency and throughput. Agent NFRs add three new dimensions: cost (every request has a dollar cost), correctness (nondeterministic outputs need statistical SLOs), and explainability (you must prove why the agent did what it did).

## What It Is

A Non-Functional Requirements (NFR) checklist specifically designed for agentic AI systems. These requirements define how the system should perform, not what it should do. In agent systems, NFRs are more complex than traditional services because of nondeterminism (same input, different outputs), per-request cost (LLM tokens are not free), stateful sessions (multi-turn context), and the need for auditability (every agent decision needs a trail).

This checklist covers 14 NFR categories with specific, measurable targets. Teams should fill out the template table at the end for their specific system.

## How It Works

Use this checklist during system design to ensure you have addressed every non-functional requirement before implementation begins. Each category includes:
- **What to define**: The specific metrics and targets
- **Why it matters for agents**: How it differs from traditional systems
- **Reference targets**: Industry-standard starting points you can adjust

---

## 1. Latency SLOs

### Why Agent Latency Is Different
Traditional services have a single request-response cycle. Agents have multiple: time to first token (TTFT), per-LLM-call latency, tool call latency, and end-to-end turn latency. Users tolerate different latencies for different interaction types.

### Reference Targets

| Metric | Chat / Interactive | Complex Reasoning | Background / Async |
|--------|-------------------|-------------------|-------------------|
| Time to First Token (TTFT) | < 500ms | < 2s | N/A |
| Single LLM call (P95) | < 3s | < 15s | < 60s |
| End-to-end agent turn (P95) | < 5s | < 30s | < 5 min |
| Streaming chunk interval | < 100ms | < 200ms | N/A |
| Tool call latency budget | < 1s per tool | < 3s per tool | < 10s per tool |

### Checklist

- [ ] Define TTFT target for primary use case
- [ ] Define end-to-end agent turn latency target (P50, P95, P99)
- [ ] Define per-tool latency budget for each integrated tool
- [ ] Define timeout per LLM call (recommended: 30s for chat, 120s for reasoning)
- [ ] Define timeout for entire agent turn (recommended: 60s for chat, 300s for workflows)
- [ ] Define streaming behavior (required for turns > 3s)
- [ ] Identify which operations can be async (batch processing, long analysis)

---

## 2. Throughput

### Reference Targets

| Metric | Small (< 1K users) | Medium (1K-100K) | Large (100K+) |
|--------|-------------------|-------------------|---------------|
| Concurrent agent sessions | 10-50 | 50-500 | 500-10,000 |
| Requests per second (agent turns) | 1-5 RPS | 5-50 RPS | 50-500 RPS |
| Token throughput per user | 500 tok/s | 200 tok/s | 100 tok/s |
| Peak-to-average ratio | 3x | 5x | 10x |

### Checklist

- [ ] Define concurrent session target (steady state and peak)
- [ ] Define requests per second target
- [ ] Define per-user token throughput requirement
- [ ] Define burst handling strategy (queue, reject, scale)
- [ ] Define rate limits per user / tenant / API key
- [ ] Define backpressure mechanism (queue depth before rejecting)

---

## 3. Reliability

### Reference Targets

| Metric | Target | How to Measure |
|--------|--------|---------------|
| Agent success rate | > 85% (simple tasks), > 70% (complex) | `agent_task_total{outcome="success"} / agent_task_total` |
| Tool availability SLA per tool | > 99.5% per tool | `tool_call_total{status="success"} / tool_call_total` |
| LLM provider uptime | > 99.9% (with multi-provider) | Health check frequency |
| System availability | > 99.9% (3 nines = 8.7 hr/yr downtime) | Standard uptime monitoring |
| Data durability (memory/checkpoints) | > 99.999% (5 nines) | Storage replication factor |

### Checklist

- [ ] Define agent task success rate target (by task type)
- [ ] Define tool availability SLA for each tool
- [ ] Configure multi-provider LLM failover (primary + secondary)
- [ ] Define acceptable failure modes (graceful degradation vs. hard fail)
- [ ] Define retry policy per operation type (LLM call, tool call, checkpoint)
- [ ] Define idempotency requirements for tool calls

---

## 4. Fault Tolerance

### Checklist

- [ ] **Checkpoint-based recovery**: Define checkpoint frequency (every N steps, every state transition)
- [ ] **Graceful degradation**: Define fallback behavior per failure type:

| Failure | Fallback Behavior |
|---------|-------------------|
| Primary LLM down | Switch to secondary provider |
| Secondary LLM down | Switch to local/smaller model |
| All LLMs down | Return cached response or error message |
| Tool X unavailable | Skip tool, inform user, continue |
| All tools unavailable | Read-only mode, answer from knowledge only |
| Memory store down | Operate without memory, warn user |
| Checkpoint store down | Continue without checkpointing, warn ops |

- [ ] **Circuit breaker policies**: Define per-dependency:
  - Failure threshold before opening (recommended: 5 failures in 60s)
  - Recovery timeout (recommended: 60s)
  - Half-open test call count (recommended: 2)
- [ ] **Timeout cascading**: Ensure total timeout > sum of individual timeouts
- [ ] **Dead letter queue**: Define what happens to failed agent requests

---

## 5. Cost Constraints

### Why Cost Is a First-Class NFR for Agents

Every agent request has a measurable dollar cost (tokens are priced). A runaway agent loop can cost hundreds of dollars in minutes. Cost NFRs prevent bill shock.

### Reference Targets

| Metric | Target Range | Alert Threshold |
|--------|-------------|-----------------|
| Per-request token budget (input + output) | 1K-50K tokens | 100K tokens |
| Per-request cost ceiling | $0.01-0.50 | 2x average |
| Per-user daily budget | $1-10 | $20 |
| Per-user monthly budget | $20-200 | $300 |
| Total platform daily cost | $100-2,000 | +30% over 7-day average |
| Cost per successful task | $0.02-1.00 | Track, no hard limit |

### Checklist

- [ ] Define per-request token budget (hard limit, kill agent if exceeded)
- [ ] Define per-request cost ceiling
- [ ] Define per-user daily/monthly budget
- [ ] Define platform-wide daily/monthly budget
- [ ] Define cost alert thresholds (warning at 80%, critical at 100%)
- [ ] Define automatic model downgrade policy:

| Cost Threshold | Action |
|---------------|--------|
| Normal | Use primary model (e.g., Claude Sonnet) |
| > 80% daily budget | Downgrade to cheaper model (e.g., Haiku) |
| > 100% daily budget | Reject new requests, complete in-flight only |

- [ ] Define cost attribution (per tenant, per agent type, per use case)
- [ ] Define cost reporting frequency (daily reports, weekly summaries)

---

## 6. Observability

### Checklist

- [ ] **Trace coverage**: 100% of agent runs must have end-to-end traces
- [ ] **Trace storage**: Define trace retention period (recommended: 30 days hot, 90 days cold)
- [ ] **Metric collection**: Define scrape interval (recommended: 15s for Prometheus)
- [ ] **Log retention**: Define log retention period (recommended: 30 days)
- [ ] **Required metrics** (minimum set):

| Metric | Type | Labels |
|--------|------|--------|
| `llm_tokens_total` | Counter | model, direction, tenant_id |
| `llm_request_duration_seconds` | Histogram | model, provider, status |
| `llm_time_to_first_token_seconds` | Histogram | model, provider |
| `agent_task_total` | Counter | agent_type, tenant_id, outcome |
| `agent_turn_duration_seconds` | Histogram | agent_type |
| `agent_step_count` | Histogram | agent_type |
| `tool_call_duration_seconds` | Histogram | tool_name, status |
| `tool_call_total` | Counter | tool_name, status |
| `llm_tokens_cost_dollars_total` | Counter | model, tenant_id |

- [ ] **Dashboards**: Define required dashboards (Agent Ops, LLM Infra, Cost, per-Tenant)
- [ ] **Alerting**: Define alert rules for each SLO (see Prometheus-Grafana guide)
- [ ] **Correlation**: Traces, metrics, and logs must be correlatable (shared trace_id)
- [ ] **Observability platform**: Choose Langfuse/LangSmith (agent traces) + Prometheus/Datadog (infra)

---

## 7. Security

### Checklist

- [ ] **Authentication**:
  - [ ] User → Agent: OAuth 2.0 / JWT / API key
  - [ ] Agent → LLM Provider: API key rotation schedule (recommended: 90 days)
  - [ ] Agent → Tools: Per-tool auth method (OAuth, API key, mTLS)

- [ ] **Authorization**:
  - [ ] RBAC per tool (which users can trigger which tools)
  - [ ] RBAC per agent (which users can use which agent types)
  - [ ] Per-tenant tool access control
  - [ ] Human-in-the-loop gates for destructive tools (delete, send, pay)

- [ ] **Data encryption**:
  - [ ] At rest: AES-256 for memory stores, checkpoints, vector stores
  - [ ] In transit: TLS 1.3 for all connections
  - [ ] LLM API calls: verify provider TLS certificates

- [ ] **PII handling**:
  - [ ] Define PII detection method (regex, NER model, manual classification)
  - [ ] Define PII masking strategy (before sending to LLM, in stored traces)
  - [ ] Define PII retention policy (max retention period by data type)
  - [ ] Define PII in prompts policy (never send SSN/credit card to LLM)

- [ ] **Input validation**:
  - [ ] Max input length (recommended: 10K characters for chat, 100K for document analysis)
  - [ ] Prompt injection detection (input guardrails)
  - [ ] Output validation (output guardrails)

- [ ] **Secret management**:
  - [ ] All API keys in secret manager (AWS Secrets Manager, HashiCorp Vault)
  - [ ] No secrets in environment variables or config files
  - [ ] Secret rotation automation

---

## 8. Multi-Tenancy

### Checklist

- [ ] **Tenant isolation level**:

| Level | Description | Use Case | Overhead |
|-------|-------------|----------|----------|
| Namespace | Shared DB, tenant_id column filter | SaaS, < 100 tenants | Low |
| Schema | Separate DB schema per tenant | Regulated industries | Medium |
| Database | Separate DB instance per tenant | Enterprise, high compliance | High |
| Cluster | Separate K8s cluster per tenant | Government, military | Very high |

- [ ] **Per-tenant configuration**:
  - [ ] Model selection (tenant A uses Sonnet, tenant B uses Haiku)
  - [ ] Tool access (tenant A has DB access, tenant B does not)
  - [ ] Token budgets (per-tenant cost limits)
  - [ ] Custom system prompts (per-tenant persona)
  - [ ] Memory isolation (tenant A cannot see tenant B's memories)

- [ ] **Cross-tenant data leakage prevention**:
  - [ ] Tenant ID in every database query (enforced at ORM level)
  - [ ] Tenant ID in every LLM call metadata
  - [ ] Separate vector store collections per tenant
  - [ ] Separate memory namespaces per tenant
  - [ ] Audit log for any cross-tenant access

---

## 9. Privacy

### Checklist

- [ ] **Data retention policies**:

| Data Type | Retention Period | Justification |
|-----------|-----------------|---------------|
| Conversation history | 90 days | User experience (context) |
| Agent traces | 30 days hot, 90 days cold | Debugging and compliance |
| Long-term memory | Until user deletion request | Core feature |
| Checkpoint data | 7 days after session end | Recovery window |
| Evaluation data | 1 year | Model improvement |
| Audit logs | 7 years | Compliance (SOX, GDPR) |

- [ ] **Right to deletion** (GDPR Article 17):
  - [ ] API endpoint to delete all user data
  - [ ] Deletion includes: conversations, memories, checkpoints, traces, embeddings
  - [ ] Deletion propagates to vector stores (delete embedded chunks)
  - [ ] Deletion confirmation within 30 days
  - [ ] Deletion does not affect aggregated/anonymized analytics

- [ ] **Consent for memory storage**:
  - [ ] Explicit opt-in for long-term memory
  - [ ] Clear explanation of what is stored and why
  - [ ] Ability to view stored memories
  - [ ] Ability to delete individual memories

- [ ] **Data residency**:
  - [ ] Define allowed regions for data storage
  - [ ] Ensure LLM provider processes data in allowed regions
  - [ ] Document cross-border data transfer mechanisms (SCCs, adequacy decisions)

---

## 10. Auditability

### Checklist

- [ ] **Decision trail**: Every agent action logged with:
  - Timestamp
  - Agent type and version
  - User ID and tenant ID
  - Input (or hash if PII)
  - LLM model used
  - Tool calls made (name, input, output)
  - Final output
  - Reasoning chain / intermediate steps
  - Token count and estimated cost

- [ ] **Compliance reporting**:
  - [ ] Generate compliance reports on demand (which data was accessed, by whom)
  - [ ] SOC 2 Type II evidence collection automated
  - [ ] GDPR data processing records maintained

- [ ] **Audit log retention**: Minimum 7 years for financial services, 3 years general

- [ ] **Tamper-proof logs**: Append-only log storage (S3 with object lock, or blockchain-backed)

---

## 11. Explainability

### Checklist

- [ ] **Reasoning traces**: Available to end users on request
  - [ ] Show which tools were called and why
  - [ ] Show retrieved documents (RAG sources)
  - [ ] Show planning steps (for multi-step agents)

- [ ] **Tool selection justification**: Log why agent chose tool A over tool B

- [ ] **Confidence scores**: Define how to communicate uncertainty:
  - [ ] High confidence: direct answer
  - [ ] Medium confidence: answer with caveat
  - [ ] Low confidence: escalate to human or ask clarifying question

- [ ] **Source attribution**: For RAG-based responses, cite source documents

- [ ] **User-facing explanation API**: Endpoint that returns reasoning chain for a given response

---

## 12. Scalability

### Checklist

- [ ] **Horizontal scaling targets**:

| Load Level | Expected Behavior | Infrastructure |
|-----------|-------------------|----------------|
| 1x (baseline) | Normal operation | 2 pods, 1 DB instance |
| 10x (growth) | Auto-scale, no degradation | 10-20 pods, read replicas |
| 100x (viral) | Graceful degradation, queue excess | 50+ pods, sharded DB |

- [ ] **Auto-scaling triggers**:
  - [ ] CPU > 70% for 2 minutes → scale up
  - [ ] Memory > 80% → scale up
  - [ ] Queue depth > 100 → scale up
  - [ ] Active sessions per pod > 50 → scale up
  - [ ] Custom: `agent_active_sessions > threshold` → scale up

- [ ] **Performance under load testing**: Define load test plan:
  - [ ] Baseline: 1x load for 1 hour
  - [ ] Stress: 10x load for 30 minutes
  - [ ] Spike: 100x load for 5 minutes
  - [ ] Soak: 2x load for 24 hours

- [ ] **Bottleneck identification**: Document known bottlenecks:
  - LLM API rate limits (usually the primary bottleneck)
  - Vector store query throughput
  - Checkpoint write throughput
  - Memory store read throughput

---

## 13. Disaster Recovery

### Checklist

- [ ] **RTO/RPO targets**:

| Component | RTO Target | RPO Target |
|-----------|-----------|-----------|
| Stateless agent (single-turn) | < 30 seconds | N/A |
| Stateful agent (multi-turn) | < 5 minutes | < 30 seconds |
| Memory store | < 15 minutes | < 1 minute |
| Vector store | < 30 minutes | < 1 hour |
| Checkpoint store | < 5 minutes | < 10 seconds |

- [ ] **Backup frequency**:
  - Checkpoints: continuous (WAL archiving) + base backup every 6 hours
  - Vector store: daily full backup
  - Memory store: continuous replication + daily backup
  - Configuration: version-controlled (GitOps)

- [ ] **Failover strategy**:
  - [ ] LLM provider failover (primary → secondary → local)
  - [ ] Database failover (multi-AZ standby promotion)
  - [ ] Cross-region failover (DNS-based, < 15 min RTO)
  - [ ] Tool endpoint failover (circuit breaker per tool)

- [ ] **DR testing**: Quarterly DR drills (test actual failover, not just backups)

---

## 14. Compliance & Regulatory

### Checklist

- [ ] **Applicable regulations**:

| Regulation | Applies If | Key Requirements |
|-----------|-----------|-----------------|
| GDPR | EU users or EU data | Consent, right to deletion, DPO |
| CCPA | California users | Opt-out of data sale, disclosure |
| HIPAA | Health data (US) | BAA with LLM provider, encryption, access control |
| SOC 2 | Enterprise customers | Security controls, audit trails |
| SOX | Financial reporting | Data integrity, access controls |
| PCI DSS | Payment data | Tokenize card data, never send to LLM |
| AI Act (EU) | High-risk AI systems | Transparency, human oversight |

- [ ] **LLM provider compliance**:
  - [ ] Verify provider's data processing agreement (DPA)
  - [ ] Verify provider does not train on your data (opt-out confirmed)
  - [ ] Verify provider's SOC 2 / ISO 27001 certification
  - [ ] Verify data residency compliance

---

## NFR Template Table

Copy this table and fill it out for your specific system. Every row should have a concrete, measurable target.

```markdown
| # | Category | Requirement | Target | Measurement Method | Priority | Owner |
|---|----------|-------------|--------|-------------------|----------|-------|
| 1 | Latency | Time to first token | < ___ ms | `llm_time_to_first_token_seconds` P95 | P0 | |
| 2 | Latency | End-to-end agent turn | < ___ s (P95) | `agent_turn_duration_seconds` P95 | P0 | |
| 3 | Latency | Tool call budget per tool | < ___ s per tool | `tool_call_duration_seconds` P95 | P1 | |
| 4 | Throughput | Concurrent sessions | ___ sessions | `agent_active_sessions` gauge | P1 | |
| 5 | Throughput | Requests per second | ___ RPS | `rate(agent_task_total[5m])` | P1 | |
| 6 | Throughput | Rate limit per user | ___ req/min | Application-level rate limiter | P1 | |
| 7 | Reliability | Agent success rate | > ___% | `agent_task_total{success} / total` | P0 | |
| 8 | Reliability | System availability | ___% (nines) | Uptime monitoring | P0 | |
| 9 | Reliability | LLM provider uptime | > ___% | Health check + failover | P1 | |
| 10 | Fault Tolerance | Checkpoint recovery | Resume within ___ s | DR drill results | P1 | |
| 11 | Fault Tolerance | Graceful degradation | Defined per failure type | Chaos testing | P1 | |
| 12 | Cost | Per-request budget | < ___ tokens / $___ | `llm_tokens_total` per request | P0 | |
| 13 | Cost | Per-user daily budget | < $___ | `llm_tokens_cost_dollars_total` by user | P1 | |
| 14 | Cost | Platform daily budget | < $___ | `sum(increase(cost[24h]))` | P0 | |
| 15 | Observability | Trace coverage | ___% of requests | Trace sampling rate config | P1 | |
| 16 | Observability | Metric scrape interval | ___ seconds | Prometheus config | P2 | |
| 17 | Observability | Log retention | ___ days | Log storage policy | P2 | |
| 18 | Security | Authentication method | OAuth / JWT / API key | Security review | P0 | |
| 19 | Security | PII handling | Mask before LLM / in traces | PII scanner audit | P0 | |
| 20 | Security | Encryption at rest | AES-256 | Infrastructure audit | P0 | |
| 21 | Multi-tenancy | Isolation level | Namespace / Schema / DB | Architecture decision | P0 | |
| 22 | Multi-tenancy | Cross-tenant leakage | Zero tolerance | Penetration testing | P0 | |
| 23 | Privacy | Data retention | ___ days by data type | Automated deletion jobs | P1 | |
| 24 | Privacy | Right to deletion | < ___ days to complete | GDPR compliance audit | P1 | |
| 25 | Auditability | Decision trail | 100% of agent actions | Trace + audit log coverage | P1 | |
| 26 | Auditability | Audit log retention | ___ years | Storage policy | P2 | |
| 27 | Explainability | Reasoning traces | Available to ___ (users/admins/both) | Feature flag | P2 | |
| 28 | Scalability | Auto-scale trigger | CPU > ___% / sessions > ___ | HPA config | P1 | |
| 29 | Scalability | Load test target | ___ x baseline | Load test results | P1 | |
| 30 | DR | RTO | < ___ minutes | DR drill results | P1 | |
| 31 | DR | RPO | < ___ minutes | Backup verification | P1 | |
| 32 | DR | Backup frequency | Every ___ hours | CronJob schedule | P2 | |
| 33 | Compliance | Applicable regulations | GDPR / HIPAA / SOC2 / ___ | Legal review | P0 | |
| 34 | Compliance | LLM provider DPA | Signed and verified | Legal review | P0 | |
```

## Decision Tree: Setting NFR Priorities

```
What stage is the project?
│
├── Prototype / MVP
│   └── Focus on: Latency (P0), Cost ceiling (P0), Basic auth (P0)
│       Skip: Multi-tenancy, DR, Compliance, Explainability
│
├── Early production (< 1K users)
│   └── Add: Reliability SLOs, Observability, Security hardening
│       Defer: Multi-region DR, Load testing at 100x
│
├── Growth (1K-100K users)
│   └── Add: Multi-tenancy, Auto-scaling, Cost attribution, Privacy (GDPR)
│       Defer: Cross-region active-active
│
└── Scale (100K+ users)
    └── Everything is P0. All 14 categories fully addressed.
```

## When NOT to Require All NFRs

- **Internal tools**: Skip multi-tenancy, privacy (GDPR), and explainability if single-org use.
- **Hackathon / demo**: Only define latency and cost ceiling. Everything else is noise.
- **Research / experimentation**: Skip all. Use Langfuse for ad-hoc observability.

## Tradeoffs

| NFR Investment | Implementation Cost | Operational Benefit | When to Invest |
|---------------|--------------------|--------------------|----------------|
| Latency optimization | Medium (caching, streaming) | Direct UX improvement | Always |
| Cost controls | Low (counters + limits) | Prevents bill shock | Always |
| Full observability | High (Prometheus + Langfuse + dashboards) | Debugging speed 10x | > 100 users |
| Multi-tenancy | Very high (schema changes throughout) | Required for SaaS | Multi-customer |
| DR / multi-region | Very high (2x infrastructure cost) | Business continuity | Revenue-critical |
| Compliance (GDPR etc.) | High (legal + engineering) | Market access | EU/regulated markets |
| Explainability | Medium (logging + UI) | User trust + debugging | Customer-facing agents |

## Failure Modes

1. **NFR targets too aggressive**: Setting TTFT < 200ms when the LLM provider can't deliver it. Mitigation: benchmark actual provider latency before committing to SLOs.

2. **Cost ceiling too low**: Agent gets killed mid-task because it hit the token limit. Mitigation: set per-request limits with 2x headroom; alert at 80%, kill at 100%.

3. **Observability blind spots**: Traces cover LLM calls but miss tool calls. Mitigation: instrument every external call; verify trace coverage in integration tests.

4. **Multi-tenancy leakage**: Shared vector store returns documents from other tenants. Mitigation: tenant_id filter enforced at retrieval layer, not just application layer.

5. **Compliance theater**: Checkboxes checked but controls not actually enforced. Mitigation: automated compliance tests in CI/CD; quarterly audits.

## Source(s) and Further Reading

- Google SRE Book (SLOs): https://sre.google/sre-book/service-level-objectives/
- GDPR Article 17 (Right to Erasure): https://gdpr-info.eu/art-17-gdpr/
- SOC 2 Compliance: https://www.aicpa.org/soc2
- EU AI Act: https://artificialintelligenceact.eu/
- NIST AI Risk Management Framework: https://www.nist.gov/artificial-intelligence/ai-risk-management-framework
- "Designing Data-Intensive Applications" - Martin Kleppmann (Chapter 1: Reliability, Scalability, Maintainability)
- Prometheus Alerting Best Practices: https://prometheus.io/docs/practices/alerting/
- AWS Well-Architected Framework (Reliability Pillar): https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/
