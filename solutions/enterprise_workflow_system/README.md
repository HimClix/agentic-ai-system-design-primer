# Design a Multi-Agent Enterprise Workflow System

> Note: This document links directly to relevant areas of the
> [Agentic AI System Design Primer](../../README.md)
>
> **Difficulty: Very Hard** -- This is the most complex case study. True multi-agent coordination with hierarchical supervision, complex state machines, and HITL approval gates.

## Step 1: Outline use cases and constraints

### Use cases

- **Invoice processing** -- vendor submits invoice (PDF/image) -> agent extracts line items, validates against PO, routes for approval based on amount thresholds, posts to ERP, triggers payment
- **Contract review** -- legal uploads contract -> agent extracts key terms (payment terms, liability clauses, renewal dates), flags non-standard clauses, routes to appropriate reviewer
- **Expense report automation** -- employee submits receipts -> agent extracts amounts, categorizes expenses, checks against policy limits, routes for manager approval, posts to accounting
- **Report generation** -- "Generate Q2 revenue report by product line" -> agent queries ERP, pulls financial data, generates charts, formats report, distributes to stakeholders
- **Cross-system data entry** -- new customer onboarding -> agent enters data in CRM, creates account in billing system, sets up support portal, sends welcome email
- **Approval routing** -- any document above threshold values gets routed through a multi-level approval chain (manager -> director -> VP based on amount)

### Constraints and assumptions

#### State assumptions

- **Users:** 500 enterprise users across 5 departments (Finance, Legal, HR, Operations, Sales)
- **Workflows/day:** 5,000 workflow executions (mix of automated and HITL)
- **Latency:** 
  - Document processing: < 30 seconds for extraction, < 2 minutes for full workflow
  - Report generation: < 5 minutes
  - Approval routing: async (hours/days), but notification within 10 seconds
- **Accuracy:** 
  - Data extraction: 98%+ (financial documents are high-stakes)
  - Routing: 99%+ (wrong approval routing is a compliance failure)
- **Compliance:** SOC2 Type II, audit trail on every action, PII handling per GDPR
- **Multi-tenancy:** Department-level isolation for data and permissions

#### Calculate usage

```
Workflows/day: 5,000
Breakdown by type:
  Invoice processing:    2,000/day (40%)
  Expense reports:       1,500/day (30%)
  Data entry:             800/day (16%)
  Report generation:      500/day (10%)
  Contract review:        200/day  (4%)

Agent calls per workflow type (avg):
  Invoice:     Supervisor(1) + DocAgent(2) + DataAgent(1) + ApprovalAgent(1) = 5 LLM calls
  Expense:     Supervisor(1) + DocAgent(1) + DataAgent(1) + ApprovalAgent(1) = 4 LLM calls
  Data entry:  Supervisor(1) + DataAgent(3)                                 = 4 LLM calls
  Reports:     Supervisor(1) + DataAgent(2) + ReportAgent(3)                = 6 LLM calls
  Contracts:   Supervisor(1) + DocAgent(3) + DataAgent(1) + ApprovalAgent(1) = 6 LLM calls

Total LLM calls/day:
  2000*5 + 1500*4 + 800*4 + 500*6 + 200*6 = 10000+6000+3200+3000+1200 = 23,400

Token usage per LLM call (avg by agent):
  Supervisor:   2,000 input + 500 output  (routing decisions, low complexity)
  DocAgent:     5,000 input + 1,500 output (document content + extraction prompts)
  DataAgent:    3,000 input + 800 output   (API calls, data validation)
  ApprovalAgent:1,500 input + 300 output   (simple routing logic)
  ReportAgent:  6,000 input + 3,000 output (data aggregation, chart descriptions, formatting)

Weighted avg: ~3,500 input + 1,000 output per call

Daily tokens:
  Input:  23,400 * 3,500 = 81.9M tokens
  Output: 23,400 * 1,000 = 23.4M tokens

Model routing:
  Supervisor + Approval (low complexity, 40% of calls): Haiku
    Input:  32.8M * $0.25/M = $8.20/day
    Output: 9.4M * $1.25/M  = $11.70/day

  DocAgent + DataAgent (medium complexity, 45% of calls): Sonnet
    Input:  36.9M * $3/M    = $110.55/day
    Output: 10.5M * $15/M   = $157.95/day

  ReportAgent (high complexity, 15% of calls): Sonnet
    Input:  12.3M * $3/M    = $36.86/day
    Output: 3.5M * $15/M    = $52.50/day

  Total LLM cost: ~$378/day = ~$11.3K/month

OCR/Document processing:
  Documents/day: ~3,700 (invoices + expenses + contracts)
  Azure Document Intelligence: $1.50 per 1K pages
  Avg pages: 2 per document = 7,400 pages/day
  Cost: $11.10/day = $333/month
```

### Out of scope

- Building the ERP system itself (integrate with existing SAP/Oracle via APIs)
- Email/Slack notification service (use existing infrastructure)
- User management and SSO (integrate with existing IdP)
- Building OCR/document AI models (use Azure Document Intelligence or Google Document AI)
- Mobile interface

---

## Step 2: Create a high-level design

```
                          ┌──────────────────────┐
                          │   Users (Web UI)      │
                          │   Finance · Legal ·   │
                          │   HR · Ops · Sales    │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │    API Gateway        │
                          │  Auth · RBAC · Audit  │
                          │  Dept Routing         │
                          └──────────┬───────────┘
                                     │
              ┌──────────────────────▼──────────────────────┐
              │          WORKFLOW ENGINE (LangGraph)          │
              │                                              │
              │  ┌────────────────────────────────────────┐ │
              │  │      TOP-LEVEL SUPERVISOR AGENT         │ │
              │  │  Classifies workflow type, delegates     │ │
              │  │  to department supervisor                │ │
              │  └───────────────┬────────────────────────┘ │
              │                  │                           │
              │    ┌─────────────┼─────────────┐            │
              │    ▼             ▼             ▼            │
              │  ┌─────┐    ┌─────┐    ┌─────────┐        │
              │  │Fin.  │    │Legal│    │Ops      │        │
              │  │Super.│    │Super│    │Super.   │        │
              │  └──┬──┘    └──┬──┘    └────┬────┘        │
              │     │          │            │              │
              └─────┼──────────┼────────────┼──────────────┘
                    │          │            │
         ┌──────────▼──────────▼────────────▼──────────┐
         │            SPECIALIST AGENTS                  │
         │                                               │
         │  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
         │  │ Document  │ │ Data     │ │ Approval     │ │
         │  │ Agent     │ │ Agent    │ │ Agent        │ │
         │  │           │ │          │ │              │ │
         │  │ - OCR     │ │ - ERP    │ │ - Route by   │ │
         │  │ - Extract │ │   CRUD   │ │   threshold  │ │
         │  │ - Validate│ │ - CRM    │ │ - Notify     │ │
         │  │ - Classify│ │ - DB     │ │ - Escalate   │ │
         │  └──────────┘ └──────────┘ └──────────────┘ │
         │                                               │
         │  ┌──────────┐ ┌──────────────────────────┐  │
         │  │ Report   │ │ Notification Agent        │  │
         │  │ Agent    │ │ (Email/Slack dispatch)     │  │
         │  │          │ │                            │  │
         │  │ - Query  │ │                            │  │
         │  │ - Chart  │ │                            │  │
         │  │ - Format │ │                            │  │
         │  └──────────┘ └──────────────────────────┘  │
         └─────────────────────┬───────────────────────┘
                               │
         ┌─────────────────────▼───────────────────────┐
         │              STATE MANAGEMENT                 │
         │                                               │
         │  ┌──────────┐  ┌──────────┐  ┌────────────┐│
         │  │ LangGraph │  │ HITL     │  │ Workflow   ││
         │  │ Postgres  │  │ Approval │  │ History    ││
         │  │ Checkptr  │  │ Queue    │  │ (Audit)    ││
         │  └──────────┘  └──────────┘  └────────────┘│
         └─────────────────────┬───────────────────────┘
                               │
         ┌─────────────────────▼───────────────────────┐
         │              TOOL LAYER                       │
         │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐  │
         │  │ OCR  │ │ ERP  │ │ CRM  │ │ Email/   │  │
         │  │ Doc  │ │ API  │ │ API  │ │ Slack    │  │
         │  │ AI   │ │(SAP/ │ │(SFDC)│ │ Notify   │  │
         │  │      │ │Oracle│ │      │ │          │  │
         │  └──────┘ └──────┘ └──────┘ └──────────┘  │
         │  ┌──────┐ ┌──────┐ ┌──────┐               │
         │  │ DB   │ │ Chart│ │ PDF  │               │
         │  │ Query│ │ Gen  │ │ Gen  │               │
         │  └──────┘ └──────┘ └──────┘               │
         └─────────────────────┬───────────────────────┘
                               │
         ┌─────────────────────▼───────────────────────┐
         │           OBSERVABILITY + COMPLIANCE          │
         │  Langfuse traces · Audit logs (immutable)    │
         │  Cost per dept · SLA tracking · Compliance    │
         └─────────────────────────────────────────────┘
```

**Why this is multi-agent (not single agent):**

This system earns multi-agent complexity because:
1. **Distinct expertise domains** -- document parsing, ERP interaction, approval routing, and report generation are genuinely different skill sets requiring different tools and system prompts
2. **Measurable benefit** -- a single agent with 15+ tools and a 10K-token system prompt would have severe instruction dilution. Specialists with 3-4 tools each maintain higher accuracy.
3. **Independent scaling** -- document processing spikes at month-end (invoices). Report generation spikes at quarter-end. Different scaling profiles.
4. **Coordination infrastructure exists** -- LangGraph provides the state management, checkpointing, and handoff mechanisms needed

Cross-reference: [When to Go Multi-Agent](../../README.md#when-to-go-multi-agent) -- all three conditions are met.

---

## Step 3: Design core components

### Agent Architecture Decision

**Hierarchical Supervisor pattern** with department-level sub-supervisors.

Cross-reference: [Hierarchical Agents](../../README.md#hierarchical-agents)

```
Level 0: Top-Level Supervisor
  - Classifies workflow type (invoice, contract, expense, report, data entry)
  - Routes to appropriate department supervisor
  - Handles cross-department coordination

Level 1: Department Supervisors (Finance, Legal, Ops)
  - Understand department-specific rules and policies
  - Delegate to specialist agents
  - Handle department-specific approval thresholds

Level 2: Specialist Agents
  - Document Agent: OCR, extraction, validation
  - Data Agent: ERP CRUD, CRM updates, database queries
  - Approval Agent: Routing logic, notification, escalation
  - Report Agent: Data aggregation, visualization, formatting
```

**Why hierarchical (not flat supervisor):** With 500 users across 5 departments, each with different approval policies, different ERP configurations, and different compliance requirements, a single supervisor would need an enormous system prompt. Department supervisors encapsulate department-specific knowledge.

### Pattern Selection

**Supervisor pattern** at the orchestration level, with **Plan-and-Execute** within each specialist agent.

Cross-reference: [Supervisor Pattern](../../README.md#supervisor-pattern), [Plan-and-Execute](../../README.md#plan-and-execute)

**Why Supervisor for orchestration:** The top-level supervisor needs to make dynamic routing decisions. An invoice from the legal department should go to the Legal Supervisor, not the Finance Supervisor. This routing decision requires reasoning.

**Why Plan-and-Execute for specialists:** Once a workflow type is identified, the steps are predictable. An invoice processing workflow ALWAYS follows: extract -> validate -> route for approval -> post to ERP. The planner creates this plan once, and the executor runs each step cheaply.

```python
# LangGraph state definition
from langgraph.graph import StateGraph
from langgraph.checkpoint.postgres import PostgresSaver
from typing import Literal

class WorkflowState(TypedDict):
    workflow_id: str
    workflow_type: Literal["invoice", "contract", "expense", "report", "data_entry"]
    department: str
    user_id: str
    
    # Document processing state
    document_url: str | None
    extracted_data: dict | None
    validation_result: dict | None
    
    # Approval state
    approval_status: Literal["pending", "approved", "rejected", "escalated"] | None
    approval_chain: list[dict]     # [{approver, threshold, status, timestamp}]
    current_approver: str | None
    
    # Cross-system state
    erp_record_id: str | None
    crm_record_id: str | None
    
    # Agent coordination
    messages: list                 # Shared message bus
    current_agent: str
    completed_steps: list[str]
    error_log: list[dict]
    
    # HITL
    requires_human_review: bool
    human_review_reason: str | None
    
    # Cost tracking
    total_tokens_used: int
    total_cost_usd: float
```

### Tool Design

**Document Agent tools:**

```json
{
  "tools": [
    {
      "name": "extract_document",
      "description": "Extract structured data from a document (invoice, receipt, contract) using OCR + document AI. Returns extracted fields with confidence scores.",
      "input_schema": {
        "type": "object",
        "properties": {
          "document_url": {"type": "string", "description": "S3 URL of the uploaded document"},
          "document_type": {"type": "string", "enum": ["invoice", "receipt", "contract", "purchase_order", "general"]},
          "extract_fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific fields to extract. e.g., ['vendor_name', 'total_amount', 'line_items', 'due_date']"
          }
        },
        "required": ["document_url", "document_type"]
      }
    },
    {
      "name": "validate_extraction",
      "description": "Validate extracted data against business rules. Checks: amounts match line items, dates are valid, vendor exists in system, PO number matches.",
      "input_schema": {
        "type": "object",
        "properties": {
          "extracted_data": {"type": "object"},
          "validation_rules": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Rules to check: ['line_items_sum', 'vendor_exists', 'po_match', 'date_valid', 'amount_reasonable']"
          }
        },
        "required": ["extracted_data"]
      }
    },
    {
      "name": "classify_document",
      "description": "Classify an unknown document into a type (invoice, contract, receipt, purchase order) with confidence score.",
      "input_schema": {
        "type": "object",
        "properties": {
          "document_url": {"type": "string"}
        },
        "required": ["document_url"]
      }
    }
  ]
}
```

**Data Agent tools:**

```json
{
  "tools": [
    {
      "name": "erp_operation",
      "description": "Perform operations on the ERP system (SAP/Oracle). Supports reading, creating, and updating records. All writes require idempotency_key.",
      "input_schema": {
        "type": "object",
        "properties": {
          "system": {"type": "string", "enum": ["sap", "oracle"]},
          "operation": {"type": "string", "enum": ["read", "create", "update"]},
          "entity": {"type": "string", "enum": ["invoice", "purchase_order", "vendor", "payment", "journal_entry"]},
          "data": {"type": "object", "description": "Entity data for create/update"},
          "query": {"type": "object", "description": "Query params for read"},
          "idempotency_key": {"type": "string", "description": "Required for create/update operations"}
        },
        "required": ["system", "operation", "entity"]
      }
    },
    {
      "name": "database_query",
      "description": "Execute a read-only SQL query against the reporting database. Returns max 1000 rows.",
      "input_schema": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "SQL query (SELECT only)"},
          "database": {"type": "string", "enum": ["finance_db", "hr_db", "ops_db"]},
          "max_rows": {"type": "integer", "default": 100, "maximum": 1000}
        },
        "required": ["query", "database"]
      }
    }
  ]
}
```

**Approval Agent tools:**

```json
{
  "tools": [
    {
      "name": "route_for_approval",
      "description": "Route a document/workflow for human approval based on amount thresholds and department policies.",
      "input_schema": {
        "type": "object",
        "properties": {
          "workflow_id": {"type": "string"},
          "amount_usd": {"type": "number"},
          "department": {"type": "string"},
          "document_type": {"type": "string"},
          "urgency": {"type": "string", "enum": ["normal", "urgent", "critical"]},
          "context_summary": {"type": "string", "description": "One-paragraph summary for the approver"}
        },
        "required": ["workflow_id", "amount_usd", "department", "document_type"]
      }
    },
    {
      "name": "check_approval_status",
      "description": "Check the current approval status of a workflow. Returns approver chain with status of each level.",
      "input_schema": {
        "type": "object",
        "properties": {
          "workflow_id": {"type": "string"}
        },
        "required": ["workflow_id"]
      }
    },
    {
      "name": "send_notification",
      "description": "Send a notification to a user or group via email or Slack.",
      "input_schema": {
        "type": "object",
        "properties": {
          "recipient": {"type": "string", "description": "Email address or Slack channel"},
          "channel": {"type": "string", "enum": ["email", "slack"]},
          "template": {"type": "string", "enum": ["approval_request", "approval_granted", "approval_rejected", "workflow_complete", "workflow_error"]},
          "data": {"type": "object", "description": "Template variables"}
        },
        "required": ["recipient", "channel", "template", "data"]
      }
    }
  ]
}
```

### HITL Approval Gates

Cross-reference: [Human-in-the-Loop](../../README.md#human-in-the-loop-hitl)

```
Approval Thresholds (configurable per department):

Finance Department:
  $0 - $5,000:      Auto-approved (agent processes autonomously)
  $5,001 - $50,000:  Manager approval (1 approver)
  $50,001 - $500,000: Director approval (2 approvers: manager + director)
  $500,001+:          VP approval (3 approvers: manager + director + VP)

Legal Department:
  Standard contracts: Legal counsel review (always human)
  Non-standard clauses detected: Senior counsel + department head

HR Department:
  Expense < $500:    Auto-approved
  Expense $500+:     Manager approval
  Policy exception:  HR Director approval
```

**Implementation in LangGraph:**
```python
def approval_gate(state: WorkflowState):
    amount = state["extracted_data"]["total_amount"]
    dept = state["department"]
    thresholds = get_dept_thresholds(dept)
    
    if amount <= thresholds["auto_approve"]:
        return Command(goto="execute_action")  # No human needed
    elif amount <= thresholds["manager"]:
        state["approval_chain"] = [{"role": "manager", "status": "pending"}]
        state["requires_human_review"] = True
        return Command(goto="wait_for_approval", update=state)
    elif amount <= thresholds["director"]:
        state["approval_chain"] = [
            {"role": "manager", "status": "pending"},
            {"role": "director", "status": "pending"}
        ]
        state["requires_human_review"] = True
        return Command(goto="wait_for_approval", update=state)
    # ... VP level
```

### Memory Architecture

| Memory Type | Store | What It Holds | TTL |
|---|---|---|---|
| **Workflow state** | PostgreSQL (LangGraph checkpointer) | Current state of each active workflow, all agent messages, tool results | Until workflow completes + 90 days archive |
| **Long-term** (document templates) | PostgreSQL + pgvector | Learned extraction patterns per vendor: "ACME Corp invoices always have PO# in top-right corner" | Updated on successful extraction, 1 year expiry |
| **Episodic** (past workflows) | PostgreSQL | Complete history of past workflow runs per user/department. "Last 10 invoices from Vendor X" | 2 years (compliance) |
| **Department policies** | PostgreSQL | Approval thresholds, routing rules, compliance requirements per department | Admin-managed, versioned |

### API Design

```
POST /api/v1/workflow/submit
Content-Type: multipart/form-data
Authorization: Bearer <jwt_token>

Request:
{
  "workflow_type": "invoice_processing",
  "department": "finance",
  "documents": [<file_upload>],
  "metadata": {
    "vendor_name": "ACME Corp",        // optional: pre-filled metadata
    "po_number": "PO-2026-1234"        // optional: for validation
  },
  "priority": "normal",
  "callback_url": "https://erp.internal/webhook"
}

Response:
{
  "workflow_id": "wf_abc123",
  "status": "processing",
  "estimated_completion": "2026-05-19T10:02:00Z",
  "tracking_url": "/api/v1/workflow/wf_abc123"
}

GET /api/v1/workflow/wf_abc123
Response:
{
  "workflow_id": "wf_abc123",
  "status": "awaiting_approval",
  "workflow_type": "invoice_processing",
  "steps_completed": [
    {"step": "document_extraction", "status": "completed", "duration_ms": 8500},
    {"step": "validation", "status": "completed", "duration_ms": 3200},
    {"step": "approval_routing", "status": "completed", "duration_ms": 1500}
  ],
  "current_step": {
    "step": "manager_approval",
    "approver": "jane.smith@company.com",
    "submitted_at": "2026-05-19T10:01:30Z"
  },
  "extracted_data": {
    "vendor": "ACME Corp",
    "total": 12500.00,
    "currency": "USD",
    "line_items": [...],
    "confidence": 0.97
  },
  "cost_so_far": {"tokens": 15200, "usd": 0.068}
}

POST /api/v1/workflow/wf_abc123/approve
{
  "approver_id": "user_jane",
  "decision": "approved",
  "notes": "Verified against PO-2026-1234"
}
```

---

## Step 4: Scale the design

### Bottleneck Analysis

| Bottleneck | At 5K workflows/day | At 50K workflows/day | Mitigation |
|---|---|---|---|
| **Document OCR** | 3,700 docs/day, manageable | 37K docs/day, provider rate limits | Batch OCR calls. Use multiple providers (Azure + Google). Queue with priority. |
| **ERP API rate limits** | SAP: 100 req/min shared | Will hit limits at 10x | Request batching. Priority queue (approval-triggered writes first). Retry with backoff. |
| **LangGraph checkpointer** | 23K checkpoint writes/day | 230K writes/day | PostgreSQL with connection pooling (pgbouncer). Shard by department. |
| **Approval queue** | Manageable | Approver fatigue at 10x | Auto-approve more (raise thresholds for trusted vendors). Batch similar approvals. |
| **Multi-agent coordination** | ~5 agent handoffs per workflow | Same but 10x volume | Worker pool per agent type. Decouple via Redis queues between agents. |

### Cost Estimation

| Component | Monthly Cost | Notes |
|---|---|---|
| **LLM - Haiku (supervisor + approval)** | $597 | 40% of 23.4K calls/day, $0.00085/call avg |
| **LLM - Sonnet (doc + data + report)** | $10,764 | 60% of 23.4K calls/day, $0.0254/call avg |
| **Document AI (Azure)** | $333 | 7.4K pages/day at $1.50/K pages |
| **PostgreSQL (RDS)** | $1,200 | db.r6g.xlarge, heavy write workload |
| **Redis (queues + cache)** | $500 | r6g.large |
| **S3 (documents + audit)** | $150 | ~50GB/month documents + audit logs |
| **Infrastructure (ECS)** | $2,500 | 6 worker nodes (2 per agent type) |
| **Langfuse** | $59 | Observability |
| **Total** | **~$16,103/month** | **$0.107 per workflow** |

**Cost per workflow type:**

| Workflow Type | Avg Cost | Breakdown |
|---|---|---|
| Invoice processing | $0.14 | OCR ($0.003) + 5 LLM calls ($0.09) + ERP writes ($0.02) + infra ($0.03) |
| Expense report | $0.08 | OCR ($0.002) + 4 LLM calls ($0.05) + validation ($0.01) + infra ($0.02) |
| Data entry | $0.07 | 4 LLM calls ($0.04) + API calls ($0.01) + infra ($0.02) |
| Report generation | $0.22 | 6 LLM calls ($0.15) + DB queries ($0.02) + chart gen ($0.02) + infra ($0.03) |
| Contract review | $0.19 | OCR ($0.005) + 6 LLM calls ($0.13) + analysis ($0.03) + infra ($0.03) |

### Failure Modes

Cross-reference: [Failure Modes & Mitigation](../../README.md#failure-modes--mitigation), [MAST Taxonomy](../../README.md#mast-taxonomy--three-failure-categories)

This system is vulnerable to most of the 14 MAST failure modes. Here are the critical ones:

| Failure Mode | MAST Category | Severity | Detection | Mitigation |
|---|---|---|---|---|
| **Coordination deadlock** | Coordination | **Critical** | Two agents waiting on each other (DocAgent waits for DataAgent validation, DataAgent waits for DocAgent extraction) | Directed acyclic workflow graph. No circular dependencies. Timeout per agent (60s). |
| **Document parsing error cascade** | Verification | High | OCR extracts wrong amount ($1,250 read as $12,500) -> wrong approval tier -> wrong approver -> delayed processing | Confidence thresholds: if extraction confidence < 0.90, flag for human review. Double-extraction: run OCR twice, compare results. |
| **Approval timeout** | Coordination | Medium | Approver doesn't respond for 48 hours | Auto-escalation: after 24h, notify approver's manager. After 48h, escalate to department head. After 72h, auto-reject with notification. |
| **ERP write failure after approval** | Coordination | **Critical** | Invoice approved but ERP posting fails (system down, validation error) | Idempotent writes with retry. Store approved state in checkpoint. Retry ERP write on system recovery. Never re-request approval. |
| **Step repetition** (MAST: 15.7%) | Specification | Medium | Agent re-extracts already-extracted document | `completed_steps` list in state. Each agent checks "did I already do this?" before executing. |
| **Wrong department routing** | Specification | Medium | Invoice from Legal routed to Finance supervisor | Department extracted from user's profile, not inferred from document. Validation step confirms routing. |
| **Multi-tenancy data leak** | Security | **Critical** | Finance department agent accesses HR database | Database credentials scoped per department. Agent tools parameterized with department context. Query-level RBAC. |
| **Report with wrong aggregation** | Verification | High | Revenue report sums wrong columns or double-counts | Reflexion pattern: generate report, then verify aggregations with a separate validation query. |

### Observability Setup

**Tracing (Langfuse):**
- Workflow-level trace: one trace per workflow_id, spanning all agent calls
- Agent-level spans: each agent call is a child span with input state, output state, tools called
- Cross-agent visibility: can see the full flow from document upload to ERP posting

**Metrics:**

| Metric | Alert Threshold | Business Impact |
|---|---|---|
| `workflow_completion_rate` | < 95% over 1 hour | Workflows stuck or failing |
| `workflow_duration_p95{type="invoice"}` | > 5 min | Processing too slow |
| `document_extraction_confidence_p50` | < 0.92 | OCR quality degradation |
| `approval_pending_count` | > 100 | Approver bottleneck |
| `approval_avg_wait_hours` | > 24h | SLA breach risk |
| `erp_write_error_rate` | > 2% | ERP integration issues |
| `agent_handoff_failure_rate` | > 1% | Coordination problems |
| `cost_per_workflow_p95` | > $0.50 (3.5x avg) | Cost anomaly |
| `cross_department_access_attempt` | > 0 | Security breach attempt |

**Audit trail (compliance):**
Every workflow action is logged to an append-only audit table:
```sql
CREATE TABLE workflow_audit_log (
    id BIGSERIAL PRIMARY KEY,
    workflow_id UUID NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    actor_type VARCHAR(20) NOT NULL,     -- 'agent', 'human', 'system'
    actor_id VARCHAR(100) NOT NULL,       -- agent name or user_id
    action VARCHAR(100) NOT NULL,         -- 'document_extracted', 'approval_requested', 'erp_posted'
    details JSONB NOT NULL,               -- full action details
    department VARCHAR(50) NOT NULL,
    -- Immutable: no UPDATE or DELETE allowed (enforced by DB trigger)
    CONSTRAINT no_updates CHECK (true)    -- table-level policy
);
```

---

## Additional Talking Points

### Why This Is the Hardest Case Study

1. **True multi-agent coordination** -- not just routing to specialists, but managing state across agents that depend on each other's outputs
2. **Mixed sync/async** -- document extraction is sync (wait 30s), approval is async (wait hours/days). The workflow engine must handle both.
3. **Financial accuracy** -- a $12,500 vs $1,250 OCR error has real financial consequences. The system needs verification at every step.
4. **Compliance** -- SOC2 requires immutable audit trails. Every agent action must be logged. Every approval must be traceable.
5. **Multi-tenancy** -- department isolation is not optional. Finance cannot see HR data. This complicates the shared agent pool.

### LangGraph Checkpointing for Long-Running Workflows

Approval workflows can span days. The LangGraph PostgresSaver checkpointer handles this:
```python
# Workflow pauses at approval gate
# State is persisted to PostgreSQL
# When approver responds (hours later):
config = {"configurable": {"thread_id": workflow_id}}
state = graph.get_state(config)  # Resume from checkpoint
graph.update_state(config, {"approval_status": "approved"})
# Workflow continues from where it left off
```

This is the key architectural advantage of LangGraph for enterprise workflows: the agent can "sleep" for days and resume exactly where it stopped.

### Error Recovery Strategy

```
Error hierarchy:
Level 1 (Retryable): Network timeout, rate limit, transient API error
  → Retry 3x with exponential backoff (1s, 2s, 4s)

Level 2 (Recoverable): OCR low confidence, validation warning
  → Flag for human review, continue other steps in parallel

Level 3 (Blocking): ERP validation error, approval rejection
  → Pause workflow, notify submitter, provide error details

Level 4 (Fatal): Auth failure, data corruption, security violation
  → Abort workflow, alert ops team, preserve state for investigation
```

### Interview Cross-References
- Why multi-agent is justified here: [When to Go Multi-Agent](../../README.md#when-to-go-multi-agent)
- Hierarchical supervisor pattern: [Hierarchical Agents](../../README.md#hierarchical-agents)
- MAST failure modes: [Failure Modes & Mitigation](../../README.md#failure-modes--mitigation)
- Checkpointing for long-running workflows: [Scalability](../../README.md#scalability)
- HITL approval gates: [Human-in-the-Loop](../../README.md#human-in-the-loop-hitl)
- Multi-tenancy isolation: [Scalability -- Multi-Tenancy](../../README.md#multi-tenancy)
- Audit trail requirements: [Security](../../README.md#security)
- Cost per workflow: [Cost Engineering](../../README.md#cost-engineering)
- Tool idempotency: [Tool Architecture](../../README.md#tool-architecture)
