# Design an AI SDR (Sales Development Representative) Agent

> Note: This document links directly to relevant areas of the
> [Agentic AI System Design Primer](../../README.md)

## Step 1: Outline use cases and constraints

### Use cases

- **Prospect research** — gather information about target companies and decision-makers from LinkedIn, company websites, news, and CRM data
- **Personalized outreach generation** — write hyper-personalized cold emails and LinkedIn messages based on prospect context
- **Follow-up scheduling** — automatically schedule follow-up sequences (day 3, day 7, day 14) with varied messaging
- **Lead qualification** — score inbound and outbound leads using BANT/MEDDIC frameworks based on available data
- **CRM updates** — log all activities, update contact records, track pipeline stage changes in Salesforce/HubSpot
- **Reply handling** — classify prospect replies (interested, not interested, objection, question, out-of-office) and draft appropriate responses
- **Meeting booking** — when a prospect expresses interest, find available slots and send calendar invites
- **Performance analytics** — track open rates, reply rates, and conversion rates per outreach template and personalization strategy

### Constraints and assumptions

#### State assumptions

- **500 sales reps** using the platform
- **10,000 outreach actions/day** (emails, LinkedIn messages, follow-ups)
- **Average prospect research**: 3-5 minutes of agent time per prospect
- **Latency target**: prospect research < 30s, email draft < 10s, reply classification < 3s
- **Accuracy bar**: 95% correct personalization (no wrong company/role), 99% no duplicate emails to same prospect, 90% lead scoring accuracy
- **CRM**: Salesforce (primary), HubSpot (secondary)
- **Email**: SendGrid for delivery, tracked opens/clicks
- **HITL**: Human approval required before sending emails to VP+ level prospects or accounts worth >$100K ARR
- **Compliance**: CAN-SPAM, GDPR (opt-out honored within 24 hours), no scraping of personal email addresses

#### Calculate usage

```
Outreach actions/day:        10,000
  New prospect research:     2,000 (20%)
  Initial cold emails:       3,000 (30%)
  Follow-up emails:          3,500 (35%)
  Reply handling:            1,000 (10%)
  LinkedIn messages:         500 (5%)

Agent runs/second:           ~0.12 (peak: ~0.5 during 9-11am send windows)

Tokens per action type (avg):
  Prospect research:
    System prompt:           1,500 tokens
    Web search results:      3,000 tokens (3 searches x 1,000)
    LinkedIn/CRM data:       1,500 tokens
    Analysis output:         800 tokens
    Total:                   6,800 tokens

  Email generation:
    System prompt:           1,500 tokens
    Prospect context:        2,000 tokens (from research)
    Template + brand voice:  500 tokens
    Output:                  400 tokens
    Total:                   4,400 tokens

  Follow-up email:
    System prompt:           1,500 tokens
    Previous emails:         1,000 tokens
    Context:                 500 tokens
    Output:                  350 tokens
    Total:                   3,350 tokens

  Reply handling:
    System prompt:           1,500 tokens
    Reply + thread:          800 tokens
    Output:                  500 tokens
    Total:                   2,800 tokens

Weighted average:            ~4,200 tokens/action

Tokens/day:
  Input:  10,000 x 3,500 input    = 35M input tokens
  Output: 10,000 x 700 output     = 7M output tokens

Cost/day (Claude 3.5 Sonnet):
  Input:  35M x $3/M     = $105
  Output: 7M x $15/M     = $105
  Total:                  = $210/day -> ~$6,300/month

Cost per outreach action: $210 / 10,000 = $0.021/action
Cost per sales rep/month: $6,300 / 500 = $12.60/month
```

**ROI calculation:**
- Average SDR salary: $65,000/year + $30,000 commission = $95,000 total
- SDR cost per outreach action (manual): ~$2.50 (accounting for research time, writing, sending)
- AI SDR cost per action: $0.021
- **119x cost reduction per action**
- But AI SDR handles ~70% of actions (routine outreach); humans handle 30% (strategic accounts, complex negotiations)
- **Net savings per SDR**: ~$35,000/year in time freed for high-value activities

**Reference: Warmly.ai benchmarks:**
- AI SDR achieves 2.5x more outreach volume per rep
- 15-25% improvement in reply rates due to better personalization
- 40% reduction in time-to-first-response for inbound leads

### Out of scope

- Phone/voice calls (text-based outreach only)
- Contract negotiation or pricing discussions
- Account management post-deal (separate system)
- Marketing campaign management (handles individual sales outreach only)
- Video creation (e.g., personalized Loom videos)

---

## Step 2: Create a high-level design

```
 Sales Rep (Dashboard / Slack)
        │
        ▼
 ┌────────────────────────────────────────────────────────────┐
 │                      API Gateway                           │
 │              (Auth, Rate Limit, Rep Routing)               │
 └──────────────────────────┬─────────────────────────────────┘
                            │
 ┌──────────────────────────▼─────────────────────────────────┐
 │               ORCHESTRATOR AGENT                            │
 │          (Plan-and-Execute for outreach strategy)           │
 │                                                             │
 │  1. Research prospect                                       │
 │  2. Score/qualify lead                                      │
 │  3. Select outreach strategy                                │
 │  4. Generate personalized message                           │
 │  5. HITL check (if high-value)                             │
 │  6. Send via appropriate channel                            │
 │  7. Schedule follow-ups                                     │
 │  8. Update CRM                                              │
 └──────┬──────────┬──────────┬───────────┬───────────────────┘
        │          │          │           │
 ┌──────▼───┐ ┌───▼────┐ ┌───▼─────┐ ┌──▼──────────────────┐
 │ Research │ │ Email  │ │ Reply   │ │ Follow-Up           │
 │ Module   │ │ Writer │ │ Handler │ │ Scheduler           │
 │          │ │        │ │         │ │                     │
 │ - Web    │ │- Draft │ │- Classify│ │- Sequence engine   │
 │   search │ │- Person│ │- Draft  │ │- Timing optimizer  │
 │ - LinkedIn│ │  alize │ │  reply  │ │- A/B variants      │
 │ - CRM    │ │- A/B   │ │- Route  │ │                     │
 │   lookup │ │  test  │ │         │ │                     │
 └──────┬───┘ └───┬────┘ └───┬─────┘ └──┬───────────────────┘
        │         │          │          │
 ┌──────▼─────────▼──────────▼──────────▼───────────────────┐
 │                       TOOL LAYER                          │
 │                                                           │
 │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
 │  │ CRM      │ │ Email    │ │ LinkedIn │ │ Web Search   │ │
 │  │(Salesforce│ │(SendGrid)│ │ API      │ │ (Tavily)     │ │
 │  │ /HubSpot)│ │          │ │          │ │              │ │
 │  ├──────────┤ ├──────────┤ ├──────────┤ ├──────────────┤ │
 │  │ Calendar │ │ Template │ │ Company  │ │ Lead Scoring │ │
 │  │(Google/  │ │ Store    │ │ DB       │ │ Engine       │ │
 │  │ Outlook) │ │          │ │(Clearbit)│ │              │ │
 │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘ │
 └───────────────────────────┬───────────────────────────────┘
                             │
 ┌───────────────────────────▼───────────────────────────────┐
 │                    MEMORY LAYER                            │
 │                                                            │
 │  ┌──────────────────┐  ┌──────────────────┐               │
 │  │ Long-Term         │  │ Procedural        │              │
 │  │ (Prospect history │  │ (Winning outreach │              │
 │  │  interaction log, │  │  templates,       │              │
 │  │  preferences)     │  │  best practices)  │              │
 │  │                   │  │                   │              │
 │  │ PostgreSQL +      │  │ Vector DB +       │              │
 │  │ Mem0              │  │ PostgreSQL        │              │
 │  └──────────────────┘  └──────────────────┘               │
 └───────────────────────────┬───────────────────────────────┘
                             │
 ┌───────────────────────────▼───────────────────────────────┐
 │              HITL GATEWAY                                  │
 │  (Slack approval for high-value outreach)                  │
 │                                                            │
 │  Rules:                                                    │
 │  - VP+ title: require approval                             │
 │  - Account >$100K ARR: require approval                    │
 │  - First-ever outreach to Fortune 500: require approval    │
 │  - Reply to "not interested": require approval             │
 └───────────────────────────┬───────────────────────────────┘
                             │
 ┌───────────────────────────▼───────────────────────────────┐
 │           Observability (Langfuse + CRM Analytics)        │
 │     Open rates, Reply rates, Conversion, Cost/action      │
 └───────────────────────────────────────────────────────────┘
```

---

## Step 3: Design core components

### Agent Architecture Decision

**Single agent with module-based tool routing.**

> See: [Single Agent vs Multi-Agent](../../README.md#single-agent-vs-multi-agent)

**Justification:**
- All SDR tasks (research, write, send, follow-up) form a linear pipeline — no parallelism needed
- Each outreach action is relatively simple (research + write + send)
- A single agent with 6-8 tools comfortably fits within one context window
- Separating into multi-agent (Researcher + Writer + Sender) would add handoff overhead for minimal benefit
- Warmly, Apollo, and other AI SDR tools use single-agent architectures

**When to revisit:** If we add real-time conversation handling (live chat with prospects) requiring a dedicated conversational agent alongside the outreach agent.

### Pattern Selection

**Plan-and-Execute for outreach strategy + ReAct for dynamic conversations**

> See: [Plan-and-Execute](../../README.md#plan-and-execute)

**Plan-and-Execute for outreach campaigns:**

```
Sales Rep: "Research and reach out to CTOs at Series B fintech companies in the Bay Area"

Plan:
  Step 1: Search for Series B fintech companies in Bay Area (web search + Crunchbase)
  Step 2: For each company, find CTO (LinkedIn + company website)
  Step 3: Cross-reference with CRM — exclude existing contacts
  Step 4: Score/qualify each prospect (BANT framework)
  Step 5: For top 10 prospects, generate personalized email
  Step 6: Submit emails for human approval (batch)
  Step 7: Schedule approved emails for optimal send times
  Step 8: Set up follow-up sequences (day 3, day 7, day 14)
  Step 9: Update CRM with all new contacts and activities
```

**ReAct for reply handling (dynamic):**

> See: [ReAct](../../README.md#react-reasoning--acting)

```
Incoming reply: "Thanks for reaching out. We're actually evaluating solutions in this space.
                 Can you send me a case study for fintech companies?"

Thought: This is a positive reply expressing interest. They want a case study specific to fintech.
Action: search_knowledge_base(query="fintech case study", type="case_study")
Observation: Found 3 case studies: [Stripe integration, PayPal migration, Square analytics]
Thought: The Stripe case study is most relevant for a fintech CTO. I should draft a reply with
         the case study attached and suggest a meeting.
Action: draft_reply(tone="warm", include_attachment="stripe_case_study.pdf",
                    include_calendar_link=true)
Observation: Draft ready for review
Thought: This is a high-interest prospect. Check if they qualify for HITL.
Action: check_hitl_rules(prospect_id="p_123")
Observation: No HITL required (not VP+, account <$100K)
Action: send_email(draft_id="d_456")
Action: update_crm(contact_id="c_789", status="engaged", next_action="meeting_requested")
```

### Tool Design

#### Tool 1: CRM Operations (Salesforce/HubSpot)

```json
{
  "name": "crm_operation",
  "description": "Perform read/write operations on the CRM (Salesforce or HubSpot). Supports contact lookup, creation, update, and activity logging.",
  "parameters": {
    "type": "object",
    "properties": {
      "operation": {
        "type": "string",
        "enum": ["search_contact", "get_contact", "create_contact", "update_contact", "log_activity", "get_deal", "update_deal_stage"]
      },
      "contact": {
        "type": "object",
        "properties": {
          "email": { "type": "string" },
          "first_name": { "type": "string" },
          "last_name": { "type": "string" },
          "company": { "type": "string" },
          "title": { "type": "string" },
          "phone": { "type": "string" },
          "linkedin_url": { "type": "string" },
          "lead_score": { "type": "integer", "minimum": 0, "maximum": 100 },
          "lead_status": {
            "type": "string",
            "enum": ["new", "contacted", "engaged", "qualified", "disqualified"]
          }
        }
      },
      "activity": {
        "type": "object",
        "properties": {
          "type": { "type": "string", "enum": ["email_sent", "email_replied", "linkedin_message", "meeting_booked", "call_made"] },
          "subject": { "type": "string" },
          "body": { "type": "string" },
          "timestamp": { "type": "string", "format": "date-time" }
        }
      },
      "search_query": {
        "type": "object",
        "properties": {
          "company": { "type": "string" },
          "title_contains": { "type": "string" },
          "lead_status": { "type": "string" },
          "last_contacted_before": { "type": "string", "format": "date" }
        }
      }
    },
    "required": ["operation"]
  },
  "error_contract": {
    "CONTACT_NOT_FOUND": "No contact found matching search criteria",
    "DUPLICATE_CONTACT": "Contact with this email already exists in CRM",
    "CRM_API_ERROR": "Salesforce/HubSpot API returned an error",
    "RATE_LIMITED": "CRM API rate limit exceeded — retry after 30s",
    "FIELD_VALIDATION": "Required CRM fields missing or invalid"
  }
}
```

#### Tool 2: Email Operations

```json
{
  "name": "email_operation",
  "description": "Send tracked emails via SendGrid. Supports drafting, scheduling, and tracking opens/clicks. Enforces CAN-SPAM compliance automatically (unsubscribe link, physical address).",
  "parameters": {
    "type": "object",
    "properties": {
      "operation": {
        "type": "string",
        "enum": ["draft", "send", "schedule", "check_status"]
      },
      "to": { "type": "string", "format": "email" },
      "from_name": { "type": "string", "description": "Sales rep's name" },
      "from_email": { "type": "string", "format": "email" },
      "subject": { "type": "string", "maxLength": 100 },
      "body_html": { "type": "string" },
      "body_text": { "type": "string" },
      "schedule_at": {
        "type": "string",
        "format": "date-time",
        "description": "UTC timestamp for scheduled send"
      },
      "tracking": {
        "type": "object",
        "properties": {
          "track_opens": { "type": "boolean", "default": true },
          "track_clicks": { "type": "boolean", "default": true }
        }
      },
      "idempotency_key": {
        "type": "string",
        "description": "Prevents duplicate sends. Format: {prospect_id}_{sequence_step}_{date}"
      }
    },
    "required": ["operation", "idempotency_key"]
  },
  "error_contract": {
    "DUPLICATE_SEND": "Email with this idempotency_key was already sent",
    "INVALID_EMAIL": "Recipient email address is invalid or bounced previously",
    "UNSUBSCRIBED": "Recipient has opted out — cannot send",
    "RATE_LIMITED": "SendGrid rate limit exceeded",
    "CONTENT_BLOCKED": "Email content flagged by spam filter"
  }
}
```

#### Tool 3: Prospect Research (Web + LinkedIn)

```json
{
  "name": "research_prospect",
  "description": "Research a prospect by gathering data from web search, LinkedIn (via Proxycurl or similar API), and company databases. Returns structured prospect intelligence.",
  "parameters": {
    "type": "object",
    "properties": {
      "prospect_name": { "type": "string" },
      "company": { "type": "string" },
      "linkedin_url": { "type": "string", "format": "uri" },
      "research_depth": {
        "type": "string",
        "enum": ["basic", "standard", "deep"],
        "default": "standard",
        "description": "basic: name+title only. standard: +recent activity, company news. deep: +funding, tech stack, competitors"
      }
    },
    "required": ["company"]
  },
  "returns": {
    "prospect": {
      "name": "string",
      "title": "string",
      "company": "string",
      "company_size": "string",
      "industry": "string",
      "funding_stage": "string",
      "recent_news": ["string"],
      "tech_stack": ["string"],
      "linkedin_summary": "string",
      "recent_posts": ["string"],
      "mutual_connections": "integer",
      "personalization_hooks": ["string (3-5 specific talking points)"]
    }
  },
  "error_contract": {
    "PROSPECT_NOT_FOUND": "Could not find this person/company",
    "LINKEDIN_RATE_LIMITED": "LinkedIn API rate limit — retry in 60s",
    "INSUFFICIENT_DATA": "Not enough public data for meaningful research"
  }
}
```

#### Tool 4: Lead Scoring

```json
{
  "name": "score_lead",
  "description": "Score a lead using BANT (Budget, Authority, Need, Timeline) framework based on available data. Returns a 0-100 score with breakdown.",
  "parameters": {
    "type": "object",
    "properties": {
      "prospect_data": {
        "type": "object",
        "description": "Output from research_prospect tool"
      },
      "scoring_model": {
        "type": "string",
        "enum": ["bant", "meddic"],
        "default": "bant"
      },
      "icp_criteria": {
        "type": "object",
        "description": "Ideal Customer Profile criteria from the sales team",
        "properties": {
          "min_company_size": { "type": "integer" },
          "target_industries": { "type": "array", "items": { "type": "string" } },
          "target_titles": { "type": "array", "items": { "type": "string" } },
          "target_funding_stages": { "type": "array", "items": { "type": "string" } }
        }
      }
    },
    "required": ["prospect_data"]
  },
  "returns": {
    "score": "integer (0-100)",
    "grade": "string (A/B/C/D/F)",
    "breakdown": {
      "budget_signal": { "score": "integer", "evidence": "string" },
      "authority_signal": { "score": "integer", "evidence": "string" },
      "need_signal": { "score": "integer", "evidence": "string" },
      "timeline_signal": { "score": "integer", "evidence": "string" }
    },
    "recommendation": "string (pursue/nurture/disqualify)"
  }
}
```

### Memory Architecture

> See: [Memory Systems](../../README.md#memory-systems)

| Memory Type | Store | Scope | Purpose |
|---|---|---|---|
| **Long-term (Prospect History)** | PostgreSQL + Mem0 | Per-prospect, permanent | All interactions, email threads, reply sentiments, meeting notes |
| **Procedural (Winning Templates)** | PostgreSQL + Vector DB | Per-team | Best-performing email templates, subject lines, personalization strategies |
| **Working Memory** | Redis | Per-action, ephemeral | Current outreach action state, research results being processed |
| **Analytics Memory** | PostgreSQL (analytics) | Global | Open rates, reply rates, conversion by template/persona/industry |

**Prospect Interaction History (Mem0 + PostgreSQL):**

```json
{
  "prospect_id": "p_cto_fintech_123",
  "company": "PayStack Inc.",
  "contact": {
    "name": "Sarah Chen",
    "title": "CTO",
    "email": "sarah@paystack.io"
  },
  "interaction_timeline": [
    {
      "date": "2026-05-01",
      "type": "email_sent",
      "subject": "Quick question about PayStack's infrastructure",
      "personalization": "Referenced their Series B announcement",
      "status": "opened",
      "opened_count": 3
    },
    {
      "date": "2026-05-04",
      "type": "follow_up_sent",
      "subject": "Re: Quick question about PayStack's infrastructure",
      "status": "replied"
    },
    {
      "date": "2026-05-04",
      "type": "reply_received",
      "sentiment": "interested",
      "content_summary": "Asked for fintech case study",
      "response_time_hours": 72
    }
  ],
  "qualification": {
    "score": 78,
    "grade": "B+",
    "status": "engaged",
    "next_action": "send_case_study",
    "next_action_date": "2026-05-05"
  },
  "preferences": {
    "preferred_channel": "email",
    "best_contact_time": "morning",
    "communication_style": "direct_and_technical"
  }
}
```

**Procedural Memory (Winning Templates):**

```json
{
  "template_id": "tmpl_fintech_cto_cold",
  "name": "Fintech CTO Cold Outreach",
  "performance": {
    "times_used": 245,
    "open_rate": 0.62,
    "reply_rate": 0.18,
    "positive_reply_rate": 0.12,
    "meeting_rate": 0.05
  },
  "template": {
    "subject": "{{prospect.recent_news_hook}} — quick thought",
    "body": "Hi {{prospect.first_name}},\n\nSaw {{prospect.company}} just {{personalization_hook}}. Congrats!\n\nWe help fintech CTOs like you {{value_prop}}. {{social_proof}}.\n\nWorth a 15-min chat this week?\n\n{{sender.first_name}}"
  },
  "best_for": {
    "industry": "fintech",
    "title_level": "C-level",
    "company_stage": "Series A-C"
  },
  "embedding_vector": "[0.023, -0.145, ...]"
}
```

### API Design

#### Create Outreach Campaign

```
POST /api/v1/campaigns
```

**Request:**
```json
{
  "rep_id": "rep_456",
  "campaign_name": "Series B Fintech CTOs - Bay Area",
  "target_criteria": {
    "titles": ["CTO", "VP Engineering", "Head of Engineering"],
    "industries": ["fintech", "financial services"],
    "company_size": { "min": 50, "max": 500 },
    "location": "San Francisco Bay Area",
    "funding_stage": ["Series B", "Series C"],
    "exclude_existing_contacts": true
  },
  "outreach_config": {
    "max_prospects": 25,
    "channel": "email",
    "sequence": {
      "steps": [
        { "day": 0, "type": "cold_email", "template_hint": "reference recent news" },
        { "day": 3, "type": "follow_up", "template_hint": "add value, share resource" },
        { "day": 7, "type": "follow_up", "template_hint": "social proof, case study" },
        { "day": 14, "type": "breakup_email", "template_hint": "last attempt, leave door open" }
      ]
    },
    "require_approval": true,
    "send_window": { "start": "09:00", "end": "11:00", "timezone": "America/Los_Angeles" }
  }
}
```

**Response:**
```json
{
  "campaign_id": "camp_789",
  "status": "researching",
  "estimated_prospects": 25,
  "estimated_completion": "15 minutes",
  "estimated_cost": "$3.50",
  "approval_required": true,
  "next_step": "Prospect research in progress. You'll be notified in Slack when drafts are ready for review."
}
```

#### Review and Approve Outreach

```
GET /api/v1/campaigns/{campaign_id}/drafts
```

**Response:**
```json
{
  "campaign_id": "camp_789",
  "drafts": [
    {
      "draft_id": "draft_001",
      "prospect": {
        "name": "Sarah Chen",
        "title": "CTO",
        "company": "PayStack Inc.",
        "lead_score": 78,
        "grade": "B+"
      },
      "email": {
        "subject": "PayStack's Series B — quick thought on scaling",
        "body": "Hi Sarah,\n\nSaw PayStack just closed your Series B — congrats! Scaling transaction processing from 1K to 100K TPS is one of the hardest problems in fintech infra.\n\nWe help fintech CTOs reduce infrastructure costs by 40% during exactly this growth phase. Stripe's platform team used us to cut their p99 latency by 60%.\n\nWorth a 15-min chat this week?\n\nBest,\nMike",
        "personalization_hooks_used": [
          "Series B announcement (2 weeks ago)",
          "Company is in payment processing (relevant to our product)",
          "CTO title — decision maker"
        ]
      },
      "qualification": {
        "score": 78,
        "recommendation": "pursue",
        "rationale": "Strong ICP fit: fintech, Series B, CTO-level. Company likely scaling infra."
      },
      "approval_status": "pending"
    }
  ],
  "total_drafts": 22,
  "pending_approval": 8,
  "auto_approved": 14
}
```

---

## Step 4: Scale the design

### Bottleneck Analysis

| Bottleneck | Symptom | At Scale (500 reps) | Mitigation |
|---|---|---|---|
| LinkedIn API rate limits | Research fails for some prospects | 2,000 research requests/day | Cache prospect profiles (7-day TTL). Stagger research across the day. Use multiple API providers (Proxycurl, RocketReach, Clearbit). |
| SendGrid sending limits | Emails queued but not sent | 10K emails/day | Well within SendGrid limits (100K/day). Warm up new sending domains gradually. Distribute across multiple sending domains. |
| CRM API rate limits | CRM updates fail or lag | 15K+ CRM operations/day | Batch CRM updates (bulk API). Queue non-critical updates. Salesforce limit: 100K API calls/day per org. |
| Email deliverability | Emails going to spam | High volume from single domain | Multiple sending domains. Warm-up sequences. SPF/DKIM/DMARC configured. Monitor sender reputation. |
| HITL approval bottleneck | Approved emails not sent within send window | 800 approvals pending at 9am | Batch approval UI in Slack. Auto-approve if no response within 4 hours (for non-critical). Manager can approve entire batch at once. |

### Cost Estimation

| Component | Per Action | Daily (10K) | Monthly | Notes |
|---|---|---|---|---|
| Claude Sonnet (email gen + research) | $0.018 | $180 | $5,400 | Main reasoning model |
| Claude Haiku (reply classification) | $0.001 | $10 | $300 | Fast classification |
| Tavily Search | $0.01 | $20 | $600 | 2 searches/research action |
| Proxycurl (LinkedIn data) | $0.03 | $60 | $1,800 | Per-profile lookup |
| SendGrid (email delivery) | $0.001 | $10 | $300 | Volume pricing |
| Salesforce API | $0.00 | $0 | $0 | Included in Salesforce license |
| Mem0 (prospect memory) | $0.001 | $10 | $300 | Long-term storage |
| Langfuse (observability) | $0.001 | $10 | $300 | |
| **Total** | **$0.062** | **$300** | **$9,000** | |
| **Per rep/month** | | | **$18.00** | |

**Cost optimization levers:**
1. **Prompt caching**: Cache system prompt + ICP criteria + template library. Saves ~35% input tokens. Savings: ~$1,500/month
2. **Template reuse**: For follow-up emails with minor variations, use Haiku instead of Sonnet. Savings: ~$800/month
3. **Research caching**: Cache prospect research for 7 days. If another rep targets the same company, reuse research. Hit rate ~15%. Savings: ~$250/month
4. **Batch processing**: Research all prospects in a campaign at once (parallel), reducing overhead. No direct cost savings but 3x faster campaign setup.

### Failure Modes

> See: [Failure Modes & Mitigation](../../README.md#failure-modes--mitigation)

| Failure Mode | Impact | Likelihood | Mitigation |
|---|---|---|---|
| **Sending duplicate emails** | Prospect annoyed, brand damage | HIGH | Idempotency key per email: `{prospect_id}_{sequence_step}_{date}`. SendGrid checks idempotency before sending. CRM activity log checked before any send. |
| **Wrong personalization** | "Congrats on your Series B" to a public company | Medium | Validate all facts against source data. Cross-reference company stage with Crunchbase data. Flag uncertain facts as "[verify]" for human review. |
| **CRM data corruption** | Wrong contact linked to wrong company | Medium | All CRM writes are idempotent (upsert by email). No delete operations from the agent. Audit log for all CRM changes with rollback capability. |
| **Outreach to opted-out prospects** | CAN-SPAM violation, legal risk | Low (with guardrails) | Check global suppression list before every send. Honor unsubscribe within 1 hour (not the legal 10-day max). Never re-add unsubscribed contacts. |
| **Tone mismatch** | Casual email to formal enterprise buyer | Medium | Per-account persona setting (formal/conversational/technical). Tone classifier on generated emails (Haiku). Sales rep can set default tone per campaign. |
| **Meeting scheduling conflicts** | Double-booking the sales rep | Low | Calendar API integration with real-time availability check. Use Calendly/Cal.com links instead of direct booking where possible. |
| **Hallucinated company facts** | Agent invents news, funding rounds, or metrics | Medium | Every fact in a personalized email must be traceable to a search result or CRM record. No unsourced claims. Fact-check step before sending. |

### Observability Setup

> See: [Observability & Monitoring](../../README.md#observability--monitoring)

**Key Metrics:**

| Metric | Target | Alert Threshold |
|---|---|---|
| Email open rate (7-day avg) | > 55% | < 40% (deliverability issue) |
| Reply rate (7-day avg) | > 12% | < 6% (personalization quality) |
| Positive reply rate | > 8% | < 4% |
| Meeting booked rate | > 3% | < 1.5% |
| Duplicate email rate | 0% | > 0 (immediate alert) |
| CRM sync failure rate | < 1% | > 5% |
| HITL approval time (median) | < 2 hours | > 6 hours |
| Cost per outreach action | < $0.07 | > $0.15 |
| Prospect research success rate | > 90% | < 80% |

**Tracing (Langfuse):**
```
Trace: camp_789 / prospect_sarah_chen
  ├── Research (Sonnet, 4.2s)
  │   ├── Web Search: "PayStack fintech Series B" (1.2s, 8 results)
  │   ├── LinkedIn Lookup (0.8s, profile found)
  │   └── CRM Check: not in system (0.3s)
  ├── Lead Scoring (Haiku, 0.5s, score: 78/100, grade: B+)
  ├── Email Generation (Sonnet, 2.1s)
  │   ├── Template Selected: fintech_cto_cold (best performer)
  │   ├── Personalization: Series B hook, infra scaling pain point
  │   └── Subject: "PayStack's Series B — quick thought on scaling"
  ├── HITL Check: VP+ title → approval required
  ├── CRM Update: contact created, activity logged (0.4s)
  └── Queued for approval in Slack
  Total: 8.3s | Cost: $0.055 | Status: pending_approval
```

---

## Additional Talking Points

### A/B Testing Framework
- Every campaign generates 2 variants (A/B) per email: different subject lines, different personalization hooks
- Variants are randomly assigned to prospects within a campaign
- After 48 hours, winning variant metrics are fed back into procedural memory
- Over time, the template library evolves based on real performance data
- This is a key differentiator: the agent gets better with every campaign

### HITL Workflow (Slack Integration)
> See: [Human-in-the-Loop](../../README.md#human-in-the-loop-hitl)
- Sales rep gets a Slack message with the email draft, prospect research summary, and lead score
- Three buttons: **Approve**, **Edit & Send** (opens editor), **Reject** (with reason)
- Batch mode: "Approve all 8 drafts" button for efficient workflows
- If no response in 4 hours: auto-send for prospects scored < 70; re-notify for prospects scored > 70
- All approval decisions logged for training data

### Compliance and Ethics
> See: [Security](../../README.md#security)
- **CAN-SPAM**: Every email includes unsubscribe link and physical address
- **GDPR**: Prospects in EU can request data deletion; agent checks prospect location before outreach
- **Rate limiting**: Max 50 emails/day per prospect (across all reps). No more than 200 emails from a single sending domain per day (deliverability best practice).
- **Transparency**: Agent never pretends to be human. Email signature identifies it as "AI-assisted outreach by [rep_name]" (configurable by company policy)
