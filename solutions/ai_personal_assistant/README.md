# Design an AI Personal Assistant

> Note: This document links directly to relevant areas of the
> [Agentic AI System Design Primer](../../README.md)

## Step 1: Outline use cases and constraints

### Use cases

- **Natural language task management** -- user says "remind me to buy groceries tomorrow at 5pm," agent creates a task with the correct datetime and sends a push notification at the right time
- **Calendar scheduling** -- "schedule a 30-min meeting with Priya next Tuesday afternoon" -- agent checks both users' availability, proposes a slot, and creates the event after confirmation
- **Email drafting and sending** -- "draft a follow-up email to the vendor about the delayed shipment" -- agent pulls context from previous email threads, drafts, and waits for approval before sending
- **Web search and summarization** -- "what's the latest on the EU AI Act?" -- agent searches the web, synthesizes top sources, returns a concise answer with citations
- **Smart home control** -- "turn off all the lights and set the thermostat to 22C" -- agent calls smart home APIs in parallel
- **Proactive reminders** -- agent learns user routines (e.g., "always check email at 9am") and proactively surfaces relevant information without being asked
- **Multi-step composite tasks** -- "plan my trip to Tokyo next month" -- agent chains flight search, hotel booking, calendar blocking, and packing list generation

### Constraints and assumptions

#### State assumptions

- **Users:** 100K monthly active users
- **Interactions:** 500K interactions/day (5 per user on average)
- **Latency:** < 3 seconds for simple tasks (smart home, reminders), < 10 seconds for complex tasks (scheduling with others, trip planning)
- **Accuracy:** 99% for deterministic actions (calendar, smart home), 90%+ for open-ended tasks (email drafting, web search)
- **Availability:** 99.9% uptime (assistant must feel always-on)
- **Privacy:** All user data encrypted at rest and in transit. No cross-user data leakage. GDPR-compliant memory management with right-to-delete.

#### Calculate usage

```
Users: 100K MAU, ~30K DAU
Interactions/day: 500K
Interactions/second: 500K / 86,400 = ~5.8 req/s (avg), ~25 req/s (peak, 4x)

Token breakdown per interaction (avg):
  System prompt:         1,500 tokens
  User preferences:        500 tokens (injected from memory)
  Conversation history:  1,000 tokens (last 5 turns)
  Tool definitions:      1,200 tokens (7 tools)
  User message:            100 tokens
  Agent reasoning:         800 tokens
  Tool calls + results:    600 tokens
  Response:                300 tokens
  Total input:          ~4,800 tokens
  Total output:         ~1,200 tokens

Daily token usage:
  Input:  500K * 4,800 = 2.4B tokens
  Output: 500K * 1,200 = 600M tokens

Model routing assumption (70% simple / 30% complex):
  Haiku (simple: smart home, reminders, simple queries):
    Input:  1.68B tokens * $0.25/M = $420/day
    Output: 420M tokens * $1.25/M  = $525/day
  Sonnet (complex: scheduling, email, trip planning):
    Input:  720M tokens * $3/M     = $2,160/day
    Output: 180M tokens * $15/M    = $2,700/day

  Total LLM cost: ~$5,805/day = ~$174K/month

Storage:
  User preferences: 100K users * 10KB avg = 1GB
  Episodic memory: 100K users * 50 interactions/month * 2KB = 10GB/month
  Conversation logs: 500K/day * 5KB avg = 2.5GB/day = 75GB/month
```

### Out of scope

- Building the voice-to-text / text-to-voice pipeline (assume existing STT/TTS services like Whisper / ElevenLabs)
- Building the mobile app frontend
- Payment processing for bookings
- Training or fine-tuning custom models
- Multi-language support (English only for v1)

---

## Step 2: Create a high-level design

```
                          ┌─────────────────────┐
                          │   User (Voice/Text)  │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   STT / Text Input   │
                          │   (Whisper / API)     │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │    API Gateway       │
                          │  Auth · Rate Limit   │
                          │  Session Routing     │
                          └──────────┬──────────┘
                                     │
              ┌──────────────────────▼──────────────────────┐
              │              ORCHESTRATOR AGENT              │
              │          (ReAct loop, LangGraph)             │
              │                                              │
              │  ┌────────┐  ┌──────────┐  ┌────────────┐  │
              │  │ Intent  │  │  LLM     │  │ Guardrails │  │
              │  │ Router  │  │  Router  │  │ Engine     │  │
              │  └────┬───┘  └────┬─────┘  └──────┬─────┘  │
              │       │           │               │         │
              └───────┼───────────┼───────────────┼─────────┘
                      │           │               │
         ┌────────────▼───────────▼───────────────▼─────────┐
         │                  TOOL LAYER                       │
         │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
         │  │ Calendar  │ │ Email    │ │ Web Search       │ │
         │  │ (Google)  │ │ (Gmail)  │ │ (Tavily/Brave)   │ │
         │  ├──────────┤ ├──────────┤ ├──────────────────┤ │
         │  │ Task Mgr  │ │ Smart   │ │ Proactive        │ │
         │  │ (Todoist) │ │ Home    │ │ Reminder Engine  │ │
         │  └──────────┘ │ (Hue/   │ └──────────────────┘ │
         │               │  Nest)  │                       │
         │               └──────────┘                      │
         └──────────────────────┬───────────────────────────┘
                                │
         ┌──────────────────────▼───────────────────────────┐
         │               MEMORY LAYER                        │
         │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
         │  │ Short-   │ │ Long-    │ │ Episodic         │ │
         │  │ Term     │ │ Term     │ │ Memory           │ │
         │  │ (Redis)  │ │ (Mem0)   │ │ (Postgres)       │ │
         │  ├──────────┤ ├──────────┤ ├──────────────────┤ │
         │  │ Session  │ │ User     │ │ Procedural       │ │
         │  │ Context  │ │ Prefs    │ │ (learned         │ │
         │  │          │ │          │ │  routines)        │ │
         │  └──────────┘ └──────────┘ └──────────────────┘ │
         └──────────────────────┬───────────────────────────┘
                                │
         ┌──────────────────────▼───────────────────────────┐
         │            OBSERVABILITY LAYER                    │
         │  Langfuse traces · Prometheus metrics             │
         │  PagerDuty alerts · Cost dashboards               │
         └──────────────────────┬───────────────────────────┘
                                │
                     ┌──────────▼──────────┐
                     │   TTS / Text Output  │
                     │  (ElevenLabs / API)   │
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │   User (Voice/Text)  │
                     └─────────────────────┘
```

**Why each component exists:**

| Component | Purpose | What breaks without it |
|---|---|---|
| **API Gateway** | Auth, rate limiting, session routing | No auth, DDoS exposure, no session continuity |
| **Intent Router** | Classify user intent to select model tier and tools | Expensive model used for "turn off lights," wasted cost |
| **LLM Router** | Route to Haiku (simple) or Sonnet (complex) | 3x cost overrun |
| **Guardrails Engine** | Block sending emails to wrong people, prevent privacy violations | Sends email to wrong recipient, reputation damage |
| **Tool Layer** | Execute real-world actions | Agent can only talk, never act |
| **Memory Layer** | Remember user preferences and past interactions | Every conversation starts from zero |
| **Observability** | Trace failures, monitor cost, alert on anomalies | Silent failures, cost explosions, no debugging |

---

## Step 3: Design core components

### Agent Architecture Decision

**Single agent** with tool routing.

**Justification:** A personal assistant handles diverse tasks, but they all flow through one user's context. The agent needs to see the full conversation to handle follow-ups like "also add that to my calendar." A multi-agent system would fragment the conversation state and add unnecessary coordination overhead.

Cross-reference: [Single Agent vs Multi-Agent](../../README.md#single-agent-vs-multi-agent) -- "Default rule: single agent first."

We earn multi-agent complexity only if a single agent demonstrably fails. At 7 tools and a well-scoped domain, a single ReAct agent with model routing handles this well.

### Pattern Selection

**ReAct (Reasoning + Acting)** -- Cross-reference: [ReAct](../../README.md#react-reasoning--acting)

**Why ReAct:**

1. **Dynamic task execution** -- "Plan my trip to Tokyo" requires multiple tool calls where each step depends on what the previous step returned (search flights, then check calendar for those dates, then book)
2. **Observation-driven adaptation** -- if the calendar shows a conflict, the agent needs to reason about alternatives
3. **Conversational context** -- the agent must interpret follow-ups like "make it an hour longer" in context

**Why NOT Plan-and-Execute:** Most personal assistant interactions are 1-3 tool calls. The overhead of generating a formal plan for "turn off the lights" is wasteful. ReAct naturally handles both simple (1-step) and complex (5-step) tasks.

**Max iterations:** 8 (prevents infinite loops on ambiguous tasks)
**Temperature:** 0.1 for tool-calling steps, 0.4 for conversational responses

### Tool Design

```json
{
  "tools": [
    {
      "name": "manage_calendar",
      "description": "Create, read, update, or delete calendar events. Use this for scheduling meetings, checking availability, or blocking time. Always check availability before creating events with other attendees.",
      "input_schema": {
        "type": "object",
        "properties": {
          "action": {
            "type": "string",
            "enum": ["create_event", "list_events", "check_availability", "update_event", "delete_event"],
            "description": "The calendar operation to perform"
          },
          "title": {"type": "string", "description": "Event title (required for create)"},
          "start_time": {"type": "string", "format": "date-time", "description": "ISO 8601 start time"},
          "end_time": {"type": "string", "format": "date-time", "description": "ISO 8601 end time"},
          "attendees": {
            "type": "array",
            "items": {"type": "string", "format": "email"},
            "description": "Email addresses of attendees"
          },
          "date_range_start": {"type": "string", "format": "date", "description": "For list/availability queries"},
          "date_range_end": {"type": "string", "format": "date"},
          "event_id": {"type": "string", "description": "Required for update/delete"}
        },
        "required": ["action"]
      }
    },
    {
      "name": "manage_email",
      "description": "Draft, search, or send emails via Gmail. IMPORTANT: Always draft first and request user approval before sending. Never send without explicit confirmation.",
      "input_schema": {
        "type": "object",
        "properties": {
          "action": {
            "type": "string",
            "enum": ["draft", "send_draft", "search", "read"],
            "description": "draft = create draft only, send_draft = send a previously drafted email (requires draft_id and user approval)"
          },
          "to": {"type": "array", "items": {"type": "string", "format": "email"}},
          "subject": {"type": "string"},
          "body": {"type": "string"},
          "draft_id": {"type": "string", "description": "Required for send_draft action"},
          "search_query": {"type": "string", "description": "Gmail search query for search action"},
          "thread_id": {"type": "string", "description": "Reply to existing thread"}
        },
        "required": ["action"]
      }
    },
    {
      "name": "web_search",
      "description": "Search the web for current information. Returns top 5 results with snippets. Use for questions about current events, facts, or anything not in the user's personal data.",
      "input_schema": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "Search query"},
          "num_results": {"type": "integer", "default": 5, "maximum": 10},
          "time_range": {
            "type": "string",
            "enum": ["day", "week", "month", "year", "any"],
            "default": "any"
          }
        },
        "required": ["query"]
      }
    },
    {
      "name": "manage_tasks",
      "description": "Create, list, update, or complete tasks in the user's task manager. Supports due dates and priorities.",
      "input_schema": {
        "type": "object",
        "properties": {
          "action": {
            "type": "string",
            "enum": ["create", "list", "complete", "update", "delete"]
          },
          "title": {"type": "string"},
          "due_date": {"type": "string", "format": "date-time"},
          "priority": {"type": "string", "enum": ["low", "medium", "high"]},
          "task_id": {"type": "string"},
          "list_filter": {"type": "string", "enum": ["today", "upcoming", "overdue", "all"]}
        },
        "required": ["action"]
      }
    },
    {
      "name": "smart_home_control",
      "description": "Control smart home devices. Supports lights, thermostat, locks, and media. Actions are idempotent -- turning off an already-off light returns success.",
      "input_schema": {
        "type": "object",
        "properties": {
          "device_type": {
            "type": "string",
            "enum": ["light", "thermostat", "lock", "media", "all_lights"]
          },
          "action": {"type": "string", "enum": ["on", "off", "set", "status"]},
          "value": {
            "type": "object",
            "description": "e.g., {\"temperature\": 22, \"unit\": \"celsius\"} or {\"brightness\": 80}"
          },
          "room": {"type": "string", "description": "Room name, e.g., 'living room', 'bedroom'. Omit for all rooms."}
        },
        "required": ["device_type", "action"]
      }
    },
    {
      "name": "get_user_context",
      "description": "Retrieve user preferences, routines, and past interaction context from long-term memory. Use this to personalize responses.",
      "input_schema": {
        "type": "object",
        "properties": {
          "context_type": {
            "type": "string",
            "enum": ["preferences", "routines", "recent_topics", "contacts"]
          },
          "query": {"type": "string", "description": "Semantic search query for relevant context"}
        },
        "required": ["context_type"]
      }
    },
    {
      "name": "set_reminder",
      "description": "Set a time-based or location-based reminder. Triggers a push notification at the specified time.",
      "input_schema": {
        "type": "object",
        "properties": {
          "message": {"type": "string", "description": "Reminder text"},
          "trigger_time": {"type": "string", "format": "date-time"},
          "trigger_type": {"type": "string", "enum": ["time", "location"]},
          "location": {"type": "string", "description": "Required if trigger_type is location"},
          "recurrence": {"type": "string", "enum": ["none", "daily", "weekly", "monthly"], "default": "none"}
        },
        "required": ["message", "trigger_type"]
      }
    }
  ]
}
```

**Error contract (all tools):**

```json
{
  "success": false,
  "error_type": "calendar_conflict",
  "message": "Time slot 2:00-2:30 PM is already booked with 'Team Standup'",
  "suggestion": "Available slots: 3:00-3:30 PM, 4:00-4:30 PM",
  "retry_with": {
    "start_time": "2026-05-20T15:00:00",
    "end_time": "2026-05-20T15:30:00"
  }
}
```

### Memory Architecture

Cross-reference: [Memory Systems](../../README.md#memory-systems)

| Memory Type | Store | What It Holds | TTL |
|---|---|---|---|
| **Short-term** (session) | Redis | Current conversation (last 10 turns), active tool results | Session duration (30 min idle timeout) |
| **Long-term** (preferences) | Mem0 | "Prefers morning meetings," "vegetarian," contact nicknames | Indefinite (user can delete) |
| **Episodic** | PostgreSQL + pgvector | Past interactions indexed by topic. "Last time you searched for flights to Tokyo..." | 6 months rolling window |
| **Procedural** (routines) | Mem0 | Learned patterns: "Every Monday at 9am, user asks for weekly calendar summary" | Updated weekly |

**Why Mem0 for long-term:** Broadest integration support (works with any framework), graph memory on Pro tier for relationship tracking (e.g., "Priya is in the Marketing team"), 80% token reduction through compressed memory retrieval.

**Memory injection flow:**
```
User message arrives
  → Retrieve from Redis: last 5 conversation turns
  → Retrieve from Mem0: top 3 relevant preferences (semantic search on user message)
  → Retrieve from Episodic: top 2 relevant past interactions
  → Inject all into context window (~2,000 tokens of memory)
  → Send to LLM with tool definitions
```

### API Design

```
POST /api/v1/assistant/message
Content-Type: application/json
Authorization: Bearer <jwt_token>

Request:
{
  "session_id": "sess_abc123",          // optional: resume session
  "message": "Schedule a meeting with Priya tomorrow at 2pm for 30 minutes",
  "stream": true,
  "input_modality": "text",             // or "voice_transcript"
  "timezone": "Asia/Kolkata"
}

Response (SSE stream):
data: {"type": "status", "content": "Checking your calendar for tomorrow..."}
data: {"type": "tool_call", "tool": "manage_calendar", "action": "check_availability", "args": {"date_range_start": "2026-05-20"}}
data: {"type": "status", "content": "You're free at 2pm. Checking Priya's availability..."}
data: {"type": "tool_call", "tool": "manage_calendar", "action": "check_availability", "args": {"attendees": ["priya@company.com"]}}
data: {"type": "approval_request", "content": "Create '30-min meeting with Priya' on May 20 at 2:00-2:30 PM IST?", "action_id": "act_789"}
data: {"type": "done", "usage": {"input_tokens": 4200, "output_tokens": 850, "cost_usd": 0.018, "latency_ms": 3400}}

POST /api/v1/assistant/approve
{
  "action_id": "act_789",
  "approved": true
}
```

---

## Step 4: Scale the design

### Bottleneck Analysis

| Bottleneck | At Current Scale (5.8 rps) | At 10x Scale (58 rps) | Mitigation |
|---|---|---|---|
| **LLM inference** | Manageable | Rate limits hit on provider APIs | LiteLLM proxy with multi-provider failover (Anthropic primary, OpenAI fallback). Queue with backpressure. |
| **Calendar API rate limits** | Google Calendar: 100 req/s shared | Will hit per-user rate limits | Batch availability checks. Cache calendar data for 60s. |
| **Memory retrieval latency** | Mem0 ~50ms per query | Acceptable | Warm cache for active users. Pre-fetch preferences on session start. |
| **Smart home API latency** | 200-500ms per device call | Parallel execution handles this | Fan-out parallel calls for multi-device commands. |
| **Session state** | Redis single node | Redis cluster with 3 shards | Shard by user_id hash. |

### Cost Estimation

| Component | Monthly Cost | Notes |
|---|---|---|
| **LLM (Haiku)** | $28,350 | 70% of 500K/day interactions, $0.0027 avg/interaction |
| **LLM (Sonnet)** | $145,800 | 30% of 500K/day interactions, $0.0324 avg/interaction |
| **Mem0 (Pro)** | $249 | Long-term memory, graph features |
| **Redis (ElastiCache)** | $500 | r6g.large, session state |
| **PostgreSQL (RDS)** | $800 | db.r6g.large, episodic memory + pgvector |
| **Langfuse (Pro)** | $59 | Observability |
| **Google Workspace APIs** | $0 | Free tier covers 100K users at 5 interactions/day |
| **Infrastructure (ECS/K8s)** | $3,000 | 4 application nodes, load balancer |
| **Total** | **~$178,758/month** | **$0.012 per interaction average** |

**Cost optimization levers:**
1. **Semantic caching** -- cache responses for common queries ("what's the weather"). Expected 15-20% cache hit rate. Saves ~$26K/month.
2. **Aggressive model routing** -- move "turn off lights" to Haiku (currently some route to Sonnet). Target 80/20 split. Saves ~$15K/month.
3. **Prompt caching** -- Anthropic cache_control on system prompt + tool definitions (5,200 tokens). At 500K req/day, saves ~$4K/month.
4. **Session summarization** -- after 10 turns, summarize older turns to reduce context. Saves ~10% on token cost.

### Failure Modes

Cross-reference: [Failure Modes & Mitigation](../../README.md#failure-modes--mitigation)

| Failure Mode | Severity | Detection | Mitigation |
|---|---|---|---|
| **Scheduling conflict** | Medium | Calendar API returns 409 | Agent reads error, proposes alternative slots from availability check |
| **Wrong email recipient** | **Critical** | Pre-send validation against user's known contacts | HITL: always require approval before sending. Flag unknown recipients with warning. |
| **Privacy violation** | **Critical** | Guardrail classifier on output | Block responses that contain other users' private data. PII filter on all LLM outputs. |
| **Infinite loop on ambiguous task** | Medium | Step count > 8 | Hard max_iterations=8 in LangGraph. Escalate to "I need more information" after hitting limit. |
| **Smart home device offline** | Low | Device API returns timeout | Inform user: "Living room light appears offline. Check your Hue bridge." |
| **LLM provider outage** | High | Health check on /readyz | Failover: Anthropic -> OpenAI -> cached response -> "I'm temporarily unavailable" |
| **Stale calendar data** | Medium | User reports incorrect availability | Cache TTL of 60s on calendar data. Force-refresh on scheduling actions. |
| **Proactive reminder at wrong time** | Low | User dismisses repeatedly | Feedback loop: if user dismisses 3x, disable that routine and ask for correction. |

### Observability Setup

Cross-reference: [Observability & Monitoring](../../README.md#observability--monitoring)

**Tracing (Langfuse):**
- Every agent run gets a trace with: user_id, session_id, intent classification, tools called, tool results, final response, token count, cost, latency
- Tool calls are child spans of the agent trace

**Metrics (Prometheus + Grafana):**

| Metric | Alert Threshold |
|---|---|
| `assistant_request_latency_p95` | > 8s sustained for 5 min |
| `assistant_tool_error_rate{tool="manage_calendar"}` | > 5% in 10 min window |
| `assistant_tool_error_rate{tool="manage_email"}` | > 2% (higher severity) |
| `assistant_tokens_per_request_p95` | > 15,000 (3x expected) |
| `assistant_cost_per_hour` | > $350 (2x expected) |
| `assistant_hitl_approval_rate` | < 60% (users rejecting too many actions -- agent quality issue) |
| `assistant_session_length` | > 20 turns (possible loop or confusion) |
| `assistant_model_fallback_rate` | > 10% (primary LLM provider issues) |

**Alerts:**
- **P1 (PagerDuty):** Email sent without approval (guardrail bypass), cost/hour > 3x budget, all tool types failing simultaneously
- **P2 (Slack):** Single tool error rate > 10%, P95 latency > 15s, model fallback rate > 20%
- **P3 (Dashboard):** Cache hit rate drops below 10%, proactive reminder dismissal rate > 50%

---

## Additional Talking Points

### Voice Pipeline Integration
The multi-modal flow (voice -> text -> agent -> voice) adds ~1.5s of latency:
- STT (Whisper): ~500ms
- Agent processing: ~2-5s
- TTS (ElevenLabs): ~300ms

Optimization: Stream the agent's text response to TTS as it generates, rather than waiting for the full response. This reduces perceived latency by ~1s.

### Proactive Reminder Engine
This is a background service (cron + event-driven), not part of the ReAct loop:
1. Every morning at a configured time, query user's calendar + tasks for the day
2. Generate a daily briefing using Haiku (cheap, simple summarization)
3. Deliver via push notification
4. Learn from user engagement: if they always dismiss the weather portion, stop including it

Cross-reference: [Memory Systems -- Procedural Memory](../../README.md#procedural-memory)

### Privacy Architecture
- User data is namespaced by `user_id` in all stores (Redis, Mem0, Postgres)
- No cross-user queries are possible at the data layer
- Memory deletion: `DELETE /api/v1/assistant/memory?user_id=X` purges all stores (GDPR Article 17)
- LLM calls use ephemeral sessions -- no training on user data
- Tool credentials are per-user OAuth tokens, stored in Vault, rotated every 30 days

### Interview Cross-References
- Agent pattern selection: [Pattern Selection Guide](../../README.md#pattern-selection-guide)
- Why single agent: [Single Agent vs Multi-Agent](../../README.md#single-agent-vs-multi-agent)
- HITL for email sending: [Human-in-the-Loop](../../README.md#human-in-the-loop-hitl)
- Token cost management: [Cost Engineering](../../README.md#cost-engineering)
- Tool design principles: [Tool Architecture](../../README.md#tool-architecture)
- Prompt caching savings: [Context Engineering -- Prompt Caching](../../README.md#prompt-caching)
