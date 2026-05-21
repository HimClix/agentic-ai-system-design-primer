# Design an AI Customer Support Agent for E-Commerce

> Note: This document links directly to relevant areas of the
> [Agentic AI System Design Primer](../../README.md)

## Step 1: Outline use cases and constraints

### Use cases

- **Answer product questions** — query a knowledge base for product specs, sizing, compatibility, shipping info
- **Order status lookup** — fetch real-time order status from the order database
- **Process refunds** — validate refund eligibility and initiate refunds with human approval for amounts > $50
- **Handle returns** — generate return labels, update order status
- **Escalate complex issues** — route to human agents when confidence is low or issue is outside agent's scope
- **Multi-tenant support** — serve multiple merchant brands with different catalogs, policies, and tone-of-voice
- **Multilingual support** — handle English, Spanish, and Portuguese queries (top 3 languages by volume)

### Constraints and assumptions

#### State assumptions

- **10,000 DAU** across all merchant tenants
- **50,000 agent runs/day** (~5 interactions per user per day)
- **80% of tickets** are answerable without human escalation (industry benchmark)
- **Average conversation**: 4-6 turns
- **Latency target**: first response < 3 seconds, subsequent < 2 seconds
- **Accuracy bar**: 95% correct answers for product questions, 99.9% accuracy for financial operations (refunds)
- **Multi-tenant**: 50 merchant brands, each with their own knowledge base (avg 500 articles/merchant)
- **Compliance**: PCI-DSS for payment data, GDPR for EU customers, PII masking in logs

#### Calculate usage

```
Agent runs/day:           50,000
Agent runs/second:        ~0.58 (peak: ~2.0 at 10am-2pm)
Avg turns/conversation:   5
LLM calls/run:            ~8 (5 user turns + planning + tool calls + guardrails)
Total LLM calls/day:      400,000

Tokens per run (avg):
  System prompt:          1,500 tokens (includes policy, persona, tool defs)
  Conversation context:   2,000 tokens
  Tool results:           1,000 tokens
  Output:                 500 tokens
  Total:                  5,000 tokens/run

Tokens/day:
  Input:  50,000 runs x 4,500 input tokens   = 225M input tokens
  Output: 50,000 runs x 500 output tokens     = 25M output tokens

Cost/day (Claude 3.5 Sonnet: $3/M input, $15/M output):
  Input:  225M x $3/M   = $675
  Output: 25M x $15/M   = $375
  Total:                 = $1,050/day -> ~$31,500/month

Cost per ticket: $1,050 / 50,000 = $0.021/ticket

Knowledge base:
  50 merchants x 500 articles x ~2,000 tokens = 50M tokens
  Embedded + indexed in vector DB
  Storage: ~2GB (embeddings + metadata)
```

**Cost comparison (Telefonica reference):**
- AI agent: $0.021/ticket (~EUR 0.02)
- Human agent: $3.50-$5.00/ticket
- Telefonica reported EUR 0.35 vs EUR 3.50 — our estimate is even lower due to simpler use case
- **Blended cost** (80% AI, 20% human): $0.021 x 0.8 + $4.00 x 0.2 = $0.82/ticket

### Out of scope

- Voice/phone support (text-only for v1)
- Proactive outbound messaging (marketing campaigns)
- Agent-to-agent transfer within the same conversation
- Custom ML models per merchant (use shared model with per-tenant prompts)
- Real-time inventory management (read-only access)

---

## Step 2: Create a high-level design

```
                            ┌─────────────────────────────────────────┐
                            │         Multi-Tenant API Gateway        │
                            │   (Auth, Rate Limit, Tenant Routing)    │
                            └──────────────┬──────────────────────────┘
                                           │
                            ┌──────────────▼──────────────────────────┐
                            │        Tenant Context Loader            │
                            │  (Load brand persona, policies, KB ref) │
                            └──────────────┬──────────────────────────┘
                                           │
                ┌──────────────────────────▼──────────────────────────┐
                │              ORCHESTRATOR AGENT (ReAct)              │
                │                                                      │
                │  ┌───────────┐  ┌──────────────┐  ┌──────────────┐  │
                │  │  Planner  │  │  Tool Router  │  │  Guardrails  │  │
                │  └─────┬─────┘  └──────┬───────┘  └──────┬───────┘  │
                │        │               │                  │          │
                └────────┼───────────────┼──────────────────┼──────────┘
                         │               │                  │
           ┌─────────────┼───────────────┼──────────────────┼─────────┐
           │             │               │                  │         │
    ┌──────▼─────┐ ┌─────▼──────┐ ┌──────▼─────┐ ┌────────▼───┐ ┌───▼──────┐
    │ Knowledge  │ │  Order DB  │ │   Refund   │ │ Escalation │ │  PII     │
    │ Base Search│ │  Lookup    │ │ Processor  │ │  Handler   │ │ Detector │
    │ (RAG)      │ │            │ │ (HITL Gate)│ │            │ │          │
    └──────┬─────┘ └─────┬──────┘ └──────┬─────┘ └────────┬───┘ └───┬──────┘
           │             │               │                │         │
    ┌──────▼─────┐ ┌─────▼──────┐ ┌──────▼─────┐ ┌───────▼────┐    │
    │  Pinecone  │ │ PostgreSQL │ │  Payment   │ │   Slack/   │    │
    │  / Qdrant  │ │  (Orders)  │ │  Gateway   │ │  Zendesk   │    │
    └────────────┘ └────────────┘ └────────────┘ └────────────┘    │
                                                                    │
                    ┌───────────────────────────────────────────────┘
                    │
             ┌──────▼──────────────────────────────┐
             │        Memory Layer                  │
             │  ┌──────────┐  ┌──────────────────┐  │
             │  │  Redis    │  │  Mem0 (Long-term │  │
             │  │ (Session) │  │  User Prefs)     │  │
             │  └──────────┘  └──────────────────┘  │
             └─────────────────────────────────────┘
                    │
             ┌──────▼──────────────────────────────┐
             │     Observability (Langfuse)         │
             │  Traces, Cost, Latency, Evals        │
             └─────────────────────────────────────┘
```

**Component Responsibilities:**

| Component | Role | What breaks without it |
|---|---|---|
| API Gateway | Auth, rate limiting, tenant routing | Unauthorized access, no multi-tenancy |
| Tenant Context Loader | Loads brand-specific prompt, KB reference, policies | Wrong tone, wrong policies applied |
| Orchestrator Agent | ReAct loop: reason, select tool, act, verify | No dynamic decision-making |
| Knowledge Base Search | RAG over merchant product catalogs | Can't answer product questions |
| Order DB Lookup | Real-time order status, history | Can't provide order info |
| Refund Processor | Validate eligibility + HITL gate + process | Can't handle refunds or does so unsafely |
| Escalation Handler | Route to human when agent can't help | Stuck conversations, frustrated users |
| PII Detector | Mask SSN, credit cards, emails in logs | Compliance violation |
| Memory Layer | Session context + long-term user preferences | Amnesia between turns, no personalization |
| Observability | Trace every LLM call, tool call, cost | Blind to failures, can't optimize cost |

---

## Step 3: Design core components

### Agent Architecture Decision

**Single Agent** is the right choice here.

> See: [Single Agent vs Multi-Agent](../../README.md#single-agent-vs-multi-agent)

**Justification:**
- All tasks (product Q&A, orders, refunds) share the same customer context
- A single agent with 4-5 tools fits within one context window (~8K tokens per turn)
- Multi-agent would add coordination overhead for simple conversations
- Telefonica and Intercom both use single-agent architectures for customer support at much larger scale

**When to revisit:** If we add specialized capabilities (e.g., fraud detection, inventory management) that require separate domain expertise AND separate data access patterns.

### Pattern Selection

**ReAct (Reasoning + Acting)** for the main conversation loop.

> See: [ReAct Pattern](../../README.md#react-reasoning--acting)

**Why ReAct:**
- Customer queries are **exploratory** — the agent doesn't know in advance which tools it needs
- "What's my order status?" needs `order_lookup`; "Can I get a refund?" needs `order_lookup` then `check_refund_policy` then `process_refund`
- Each observation informs the next action (e.g., order lookup reveals it's past refund window, changing the response)

**Why NOT Plan-and-Execute:**
- Customer support conversations are reactive, not plan-first
- Users frequently change topics mid-conversation
- Plans would be invalidated by each user reply

**HITL overlay for financial actions:**

> See: [Human-in-the-Loop](../../README.md#human-in-the-loop-hitl)

```python
# HITL gate is applied within the ReAct loop
HITL_REQUIRED_ACTIONS = {
    "process_refund": lambda ctx: ctx.amount > 50.00,
    "issue_credit": lambda ctx: True,  # Always require approval
    "cancel_order": lambda ctx: ctx.order_status == "shipped",
}
```

### Tool Design

#### Tool 1: Knowledge Base Search

```json
{
  "name": "search_knowledge_base",
  "description": "Search the merchant's product knowledge base for answers to customer questions about products, policies, shipping, sizing, and compatibility. Returns top-k relevant passages with confidence scores.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language search query derived from the customer's question"
      },
      "merchant_id": {
        "type": "string",
        "description": "Tenant merchant ID to scope the search"
      },
      "top_k": {
        "type": "integer",
        "default": 3,
        "description": "Number of results to return"
      },
      "category_filter": {
        "type": "string",
        "enum": ["products", "shipping", "returns", "sizing", "general"],
        "description": "Optional category to narrow search"
      }
    },
    "required": ["query", "merchant_id"]
  },
  "returns": {
    "type": "object",
    "properties": {
      "results": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "content": { "type": "string" },
            "source": { "type": "string" },
            "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
            "article_id": { "type": "string" }
          }
        }
      }
    }
  },
  "error_contract": {
    "KB_NOT_FOUND": "Knowledge base for merchant_id not found",
    "SEARCH_TIMEOUT": "Vector search exceeded 2s timeout",
    "NO_RESULTS": "No relevant articles found (confidence < 0.5)"
  }
}
```

#### Tool 2: Order Database Lookup

```json
{
  "name": "lookup_order",
  "description": "Look up order details including status, items, shipping, and payment info. Masks sensitive payment data automatically.",
  "parameters": {
    "type": "object",
    "properties": {
      "order_id": {
        "type": "string",
        "pattern": "^ORD-[A-Z0-9]{8}$",
        "description": "Order ID in format ORD-XXXXXXXX"
      },
      "customer_email": {
        "type": "string",
        "format": "email",
        "description": "Customer email for verification (must match order)"
      }
    },
    "required": ["order_id", "customer_email"]
  },
  "returns": {
    "type": "object",
    "properties": {
      "order_id": { "type": "string" },
      "status": { "type": "string", "enum": ["pending", "confirmed", "shipped", "delivered", "cancelled"] },
      "items": { "type": "array" },
      "total": { "type": "number" },
      "shipping_eta": { "type": "string", "format": "date" },
      "refund_eligible": { "type": "boolean" },
      "refund_window_days_remaining": { "type": "integer" }
    }
  },
  "error_contract": {
    "ORDER_NOT_FOUND": "No order found with this ID",
    "EMAIL_MISMATCH": "Customer email does not match order records",
    "DB_TIMEOUT": "Database query exceeded 3s timeout"
  }
}
```

#### Tool 3: Refund Processor

```json
{
  "name": "process_refund",
  "description": "Initiate a refund for an eligible order. Requires HITL approval for amounts > $50. Idempotent — duplicate calls with same refund_request_id are no-ops.",
  "parameters": {
    "type": "object",
    "properties": {
      "order_id": { "type": "string" },
      "refund_amount": { "type": "number", "minimum": 0.01 },
      "reason": {
        "type": "string",
        "enum": ["defective", "wrong_item", "not_as_described", "changed_mind", "late_delivery"]
      },
      "refund_request_id": {
        "type": "string",
        "description": "Idempotency key — UUID generated by the agent"
      }
    },
    "required": ["order_id", "refund_amount", "reason", "refund_request_id"]
  },
  "hitl_gate": {
    "condition": "refund_amount > 50.00",
    "approval_channel": "slack:#refund-approvals",
    "timeout": "15 minutes",
    "fallback": "escalate_to_human"
  },
  "error_contract": {
    "NOT_ELIGIBLE": "Order is not eligible for refund (past window or already refunded)",
    "AMOUNT_EXCEEDS_ORDER": "Refund amount exceeds original order total",
    "APPROVAL_TIMEOUT": "Human approval not received within timeout",
    "PAYMENT_GATEWAY_ERROR": "Payment gateway returned error during refund"
  }
}
```

#### Tool 4: Escalation to Human Agent

```json
{
  "name": "escalate_to_human",
  "description": "Transfer the conversation to a human support agent. Includes full conversation context and reason for escalation.",
  "parameters": {
    "type": "object",
    "properties": {
      "reason": {
        "type": "string",
        "enum": ["low_confidence", "customer_request", "complex_issue", "safety_concern", "repeated_failure"]
      },
      "priority": {
        "type": "string",
        "enum": ["low", "medium", "high", "urgent"]
      },
      "summary": {
        "type": "string",
        "description": "Brief summary of the issue and what has been tried"
      }
    },
    "required": ["reason", "priority", "summary"]
  },
  "error_contract": {
    "NO_AGENTS_AVAILABLE": "No human agents online — queued with ETA",
    "QUEUE_FULL": "Escalation queue at capacity — offer callback"
  }
}
```

### Memory Architecture

> See: [Memory Systems](../../README.md#memory-systems)

| Memory Type | Store | TTL | Purpose |
|---|---|---|---|
| **Short-term (Session)** | Redis | 30 min inactivity | Current conversation context, tool results |
| **Long-term (User Prefs)** | Mem0 + PostgreSQL | Indefinite | User preferences, past issues, communication style |
| **Semantic (Knowledge)** | Pinecone/Qdrant | Until re-indexed | Product catalogs, policies per merchant |
| **Episodic (Past Tickets)** | PostgreSQL | 1 year | Past resolution patterns for similar issues |

**Session Memory (Redis):**
```json
{
  "session_id": "sess_abc123",
  "tenant_id": "merchant_nike",
  "customer_id": "cust_xyz",
  "conversation_history": [
    {"role": "user", "content": "Where is my order ORD-1234ABCD?"},
    {"role": "assistant", "content": "Let me look that up for you..."},
    {"role": "tool", "name": "lookup_order", "result": {"status": "shipped", "eta": "2026-05-22"}}
  ],
  "tools_used": ["lookup_order"],
  "started_at": "2026-05-19T10:00:00Z",
  "turn_count": 3
}
```

**Long-term Memory (Mem0):**
```json
{
  "customer_id": "cust_xyz",
  "preferences": {
    "communication_style": "concise",
    "preferred_language": "en",
    "past_issues": ["late_delivery_2026_03", "sizing_question_2026_01"],
    "satisfaction_trend": "improving",
    "refund_history_count": 1
  }
}
```

### API Design

#### Start Conversation

```
POST /api/v1/conversations
```

**Request:**
```json
{
  "tenant_id": "merchant_nike",
  "customer_id": "cust_xyz",
  "channel": "web_chat",
  "initial_message": "I want to return my shoes, order ORD-1234ABCD",
  "metadata": {
    "page_url": "https://nike.com/orders",
    "locale": "en-US"
  }
}
```

**Response:**
```json
{
  "conversation_id": "conv_abc123",
  "session_id": "sess_def456",
  "response": {
    "message": "I'd be happy to help you with your return! Let me pull up your order details first.",
    "suggested_actions": ["View return policy", "Talk to a human"],
    "confidence": 0.92
  },
  "metadata": {
    "tools_invoked": ["lookup_order"],
    "latency_ms": 1840,
    "tokens_used": { "input": 3200, "output": 420 },
    "cost_usd": 0.016
  }
}
```

#### Send Message

```
POST /api/v1/conversations/{conversation_id}/messages
```

**Request:**
```json
{
  "message": "Yes, the shoes don't fit. Can I get a refund?",
  "attachments": []
}
```

**Response:**
```json
{
  "message_id": "msg_ghi789",
  "response": {
    "message": "Your order ORD-1234ABCD is eligible for a refund of $89.99. The refund will be processed to your original payment method within 5-7 business days. Shall I proceed?",
    "requires_confirmation": true,
    "pending_action": {
      "tool": "process_refund",
      "params": { "order_id": "ORD-1234ABCD", "refund_amount": 89.99, "reason": "wrong_item" }
    },
    "confidence": 0.95
  },
  "metadata": {
    "tools_invoked": ["lookup_order", "check_refund_policy"],
    "hitl_triggered": false,
    "latency_ms": 2100,
    "tokens_used": { "input": 5400, "output": 380 },
    "cost_usd": 0.022
  }
}
```

---

## Step 4: Scale the design

### Bottleneck Analysis

| Bottleneck | Symptom | At Scale | Mitigation |
|---|---|---|---|
| LLM API rate limits | 429 errors at peak | 2 req/s peak, fine for Claude | Use batch API for async, implement retry with exponential backoff |
| Vector DB latency | Slow knowledge base search | 50 merchants x concurrent queries | Per-merchant index sharding, pre-filter by tenant |
| Redis session storage | Memory pressure | 10K concurrent sessions x ~10KB each = 100MB | Acceptable. Set 30-min TTL, compress conversation history |
| Order DB queries | Slow lookups during peak | Connection pool exhaustion | Read replicas, connection pooling (PgBouncer), query caching for repeat lookups |
| Context window limits | Long conversations overflow | >20 turns = >15K tokens | Sliding window: keep first 2 + last 8 turns, summarize middle |

### Cost Estimation

| Component | Tokens/Request | Cost/Request | Daily Volume | Monthly Cost |
|---|---|---|---|---|
| Claude 3.5 Sonnet (main reasoning) | 4,500 in + 500 out | $0.021 | 50,000 | $31,500 |
| Claude 3.5 Haiku (guardrails/PII) | 500 in + 50 out | $0.0004 | 50,000 | $600 |
| Pinecone (vector search) | N/A | $0.0002/query | 30,000 | $180 |
| Redis (session memory) | N/A | $0.000001/op | 400,000 | $12 |
| Mem0 (long-term memory) | N/A | $0.001/read | 50,000 | $1,500 |
| Langfuse (observability) | N/A | $0.002/trace | 50,000 | $3,000 |
| **Total** | | **~$0.024/ticket** | | **~$36,792/month** |

**Cost optimization levers:**
1. **Prompt caching** (Anthropic): Cache system prompt + tenant context. Saves ~40% on input tokens for repeat turns. Monthly savings: ~$9,000
2. **Semantic caching**: Cache common Q&A pairs (Redis). Hit rate ~30% for product FAQs. Monthly savings: ~$5,000
3. **Model routing**: Use Haiku for simple FAQs (50% of queries), Sonnet for complex. Monthly savings: ~$8,000
4. **Optimized cost estimate**: ~$14,800/month after all optimizations ($0.010/ticket)

### Failure Modes

> See: [Failure Modes & Mitigation](../../README.md#failure-modes--mitigation)

| Failure Mode | Impact | Likelihood | Mitigation |
|---|---|---|---|
| **Hallucinated order IDs** | Agent references non-existent orders | Medium | Validate all order IDs against DB before referencing. Tool returns error for invalid IDs. Never let agent fabricate an order ID — it must come from tool output. |
| **Refund without verification** | Financial loss | Low (with HITL) | Two-step: (1) agent confirms eligibility via tool, (2) HITL gate for amounts > $50, (3) idempotency key prevents duplicates |
| **PII leakage in logs** | GDPR/compliance violation | Medium | Run PII detector (Presidio) on all messages before logging. Mask credit card numbers, SSNs, full addresses in Langfuse traces |
| **Wrong tenant context** | Nike agent answers with Adidas policies | Low | Tenant ID validated at API gateway, injected into every tool call, verified in system prompt |
| **Infinite reasoning loop** | Agent stuck reasoning without acting | Medium | Max 10 ReAct iterations per turn. Circuit breaker: if 3 consecutive tool calls fail, escalate to human |
| **Stale knowledge base** | Outdated product info | Medium | Nightly re-indexing of KB. Version stamp on search results. Fallback: "Let me connect you with a specialist for the latest info" |
| **Toxic/adversarial input** | Agent produces harmful content | Low | Input guardrail (Anthropic content filter). Output guardrail checks for policy violations before sending |

### Observability Setup

> See: [Observability & Monitoring](../../README.md#observability--monitoring)

**Key Metrics:**

| Metric | Target | Alert Threshold |
|---|---|---|
| First response latency (p50/p95) | p50 < 2s, p95 < 4s | p95 > 6s |
| Resolution rate (no escalation) | > 80% | < 70% |
| Customer satisfaction (CSAT) | > 4.2/5 | < 3.8/5 |
| Cost per ticket | < $0.03 | > $0.05 |
| Escalation rate | < 20% | > 30% |
| Tool call failure rate | < 2% | > 5% |
| Hallucination rate (eval) | < 3% | > 5% |
| PII detection triggers/day | Monitoring | > 50/day (investigate) |

**Tracing (Langfuse):**
```
Trace: conv_abc123 / msg_ghi789
  ├── LLM Call: plan (Sonnet, 1.2s, 3200 in / 180 out)
  ├── Tool: lookup_order (0.3s, success)
  ├── Tool: check_refund_policy (0.1s, success)
  ├── LLM Call: synthesize_response (Sonnet, 0.8s, 5400 in / 380 out)
  ├── Guardrail: pii_check (0.05s, pass)
  └── Guardrail: content_safety (0.05s, pass)
  Total: 2.1s | Cost: $0.022 | Tokens: 5880
```

**Alerts:**
- PagerDuty: Escalation rate > 40% for 15 min (potential outage or bad deployment)
- Slack: Cost per ticket > $0.05 (model routing issue or infinite loops)
- Langfuse: Hallucination eval score drops below 90% (prompt regression)

---

## Additional Talking Points

### Multi-Tenancy Deep Dive
- Each merchant gets a **tenant config** stored in PostgreSQL: brand voice, escalation policy, refund limits, KB index reference
- System prompt is composed at runtime: `base_prompt + tenant_persona + tenant_policies + tool_definitions`
- **Isolation**: tenant_id is injected into every tool call as a mandatory parameter; vector search is scoped by tenant namespace

### Guardrails Architecture
> See: [Safety & Guardrails](../../README.md#safety--guardrails)
- **Input guardrail**: Prompt injection detection, topic deflection (competitor mentions), profanity filter
- **Output guardrail**: PII masking, factuality check against KB sources, tone consistency with brand voice
- **Financial guardrail**: All monetary actions require explicit tool-call confirmation, never inferred from conversation

### HITL Integration
> See: [Human-in-the-Loop](../../README.md#human-in-the-loop-hitl)
- **Confidence-based**: If agent confidence < 0.7 on any response, auto-escalate
- **Policy-based**: Refunds > $50, account changes, legal/safety complaints
- **Customer-initiated**: "Talk to a human" intent detection with immediate routing
- **Seamless handoff**: Full conversation context + agent's analysis passed to human agent in Zendesk/Intercom

### Security Considerations
> See: [Security](../../README.md#security)
- **No direct DB access**: Agent only accesses data through validated tool interfaces
- **PCI compliance**: Credit card data never enters the LLM context. Tools return masked values (****1234)
- **Audit trail**: Every tool invocation logged with inputs, outputs, and user context for compliance review
- **Rate limiting**: Per-customer rate limits (20 messages/minute) to prevent abuse
