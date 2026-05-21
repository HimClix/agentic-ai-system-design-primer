# Functional Requirements Template for Agentic AI Systems
> Functional requirements define what the agent does. For agent systems, this means: what tasks it can perform, which tools it can call, how it plans, what it remembers, and when it asks a human for help.

## What It Is

A functional requirements template specifically designed for agentic AI systems. Unlike traditional software where functional requirements map to CRUD operations, agent functional requirements describe capabilities across 13 dimensions: task execution, tool calling, planning, memory, collaboration, human oversight, retrieval, event handling, integrations, async execution, streaming, multi-modal inputs, and output formatting.

This template includes a checklist format, a filled-out example (customer support agent), and a blank template for teams to use.

## How It Works

Walk through each of the 13 functional requirement categories below. For each one, decide:
1. **Is this relevant?** Not every agent needs every capability.
2. **What specifically?** Define the exact behavior.
3. **What are the constraints?** Limits, guardrails, edge cases.

---

## Functional Requirements Categories

### 1. Natural Language Task Execution

What tasks can the agent perform based on natural language input?

**Checklist:**
- [ ] Define the set of tasks the agent can handle (task taxonomy)
- [ ] Define how tasks are identified from user input (intent classification vs. LLM reasoning)
- [ ] Define out-of-scope handling (what the agent should NOT do)
- [ ] Define ambiguity resolution (ask clarifying questions vs. assume)
- [ ] Define multi-language support requirements
- [ ] Define input modality (text only, text + images, voice)

### 2. Tool Calling

Which external tools can the agent invoke?

**Checklist:**
- [ ] List all tools with name, description, input schema, output schema
- [ ] Define tool call authorization (which tools require user confirmation)
- [ ] Define tool call ordering constraints (must call A before B)
- [ ] Define tool call retry policy per tool
- [ ] Define tool call timeout per tool
- [ ] Define parallel vs. sequential tool execution strategy
- [ ] Define error handling per tool (retry, skip, escalate)

**Tool Registry Template:**

| Tool Name | Description | Input Schema | Output Schema | Auth Required | Side Effects | Timeout | Retry |
|-----------|-------------|-------------|---------------|--------------|-------------|---------|-------|
| | | | | Yes/No | Read/Write/Delete | ___ s | 0-3x |

### 3. Planning & Reasoning

How does the agent decompose complex tasks?

**Checklist:**
- [ ] Define planning pattern:

| Pattern | When to Use | Complexity |
|---------|------------|-----------|
| ReAct (reason + act) | Simple tool-use tasks | Low |
| Plan-then-execute | Multi-step workflows with known steps | Medium |
| Reflection | Tasks requiring self-correction | Medium |
| Multi-agent delegation | Tasks spanning multiple domains | High |
| Human-in-the-loop planning | High-stakes multi-step workflows | High |

- [ ] Define maximum planning depth (recommended: 3-5 steps for chat, 10-20 for workflows)
- [ ] Define replanning trigger (when to abandon plan and replan)
- [ ] Define plan visibility (show plan to user? editable by user?)

### 4. Multi-Step Workflow Execution

Can the agent execute sequences of actions?

**Checklist:**
- [ ] Define workflow types the agent supports (linear, branching, looping)
- [ ] Define state management between steps (how context flows)
- [ ] Define partial completion handling (what if step 3 of 5 fails?)
- [ ] Define workflow timeout (max total execution time)
- [ ] Define workflow cancellation (user can cancel mid-workflow?)
- [ ] Define workflow resume (can user resume after interruption?)
- [ ] Define conditional branching rules (if X then do Y, else do Z)

### 5. Memory Persistence

What does the agent remember, and for how long?

**Checklist:**
- [ ] Define memory types needed:

| Memory Type | What It Stores | Lifetime | Storage |
|------------|----------------|----------|---------|
| Short-term (working) | Current conversation context | Session | In-memory / checkpoint |
| Long-term (semantic) | User preferences, facts, patterns | Permanent | Vector store / DB |
| Episodic | Past interaction summaries | Permanent | DB |
| Procedural | Learned task execution patterns | Permanent | Prompt / DB |

- [ ] Define memory read strategy (when to query memory, relevance threshold)
- [ ] Define memory write strategy (what triggers a memory save)
- [ ] Define memory update strategy (how to handle contradictions)
- [ ] Define memory deletion (user can request deletion)
- [ ] Define cross-session memory (does the agent remember previous sessions?)

### 6. Agent Collaboration

Does the system use single or multiple agents?

**Checklist:**
- [ ] Define agent architecture:

| Architecture | When to Use |
|-------------|------------|
| Single agent | Simple tasks, < 5 tools |
| Supervisor + workers | Multiple domains, need routing |
| Peer-to-peer | Collaborative tasks, debate/consensus |
| Hierarchical | Complex org structure, approval chains |

- [ ] Define agent communication protocol (direct message, shared state, event bus)
- [ ] Define agent handoff rules (when does agent A hand off to agent B?)
- [ ] Define shared context strategy (what context is passed between agents?)
- [ ] Define conflict resolution (what if two agents disagree?)

### 7. Human Approvals (HITL)

Where does a human need to approve or intervene?

**Checklist:**
- [ ] Define approval gates:

| Action Type | Approval Required? | Threshold |
|------------|-------------------|-----------|
| Read data | No | -- |
| Write data | Yes, if production | Always |
| Delete data | Yes | Always |
| Send email/message | Yes, if external | Always |
| Financial transaction | Yes | > $0 |
| API call with side effects | Configurable | Per-tool |

- [ ] Define approval workflow (sync wait vs. async notification)
- [ ] Define approval timeout (how long to wait for human, then what?)
- [ ] Define escalation path (if primary approver unavailable)
- [ ] Define override capability (can admin bypass approval gates?)

### 8. Retrieval Augmentation (RAG)

Does the agent need to retrieve external knowledge?

**Checklist:**
- [ ] Define knowledge sources (documents, databases, APIs, web search)
- [ ] Define indexing strategy (chunk size, overlap, embedding model)
- [ ] Define retrieval strategy (semantic search, hybrid, reranking)
- [ ] Define context window management (how many chunks to include)
- [ ] Define source citation format (inline, footnotes, links)
- [ ] Define knowledge freshness requirements (real-time vs. batch indexed)
- [ ] Define knowledge update pipeline (how are new documents indexed?)

### 9. Real-Time Event Handling

Does the agent react to external events?

**Checklist:**
- [ ] Define event sources (webhooks, message queues, SSE, WebSocket)
- [ ] Define event-to-action mapping (which events trigger which agent actions)
- [ ] Define event filtering (which events are relevant)
- [ ] Define event ordering requirements (FIFO, best-effort)
- [ ] Define event deduplication strategy
- [ ] Define event backpressure handling (queue overflow)

### 10. External API Integrations

Which external services does the agent connect to?

**Checklist:**
- [ ] List all external APIs with:
  - Service name
  - API version
  - Authentication method
  - Rate limits
  - SLA / expected availability
  - Fallback if unavailable
- [ ] Define API client configuration (timeout, retry, circuit breaker)
- [ ] Define data transformation requirements (API format → agent format)
- [ ] Define API versioning strategy (how to handle API changes)

### 11. Long-Running Task Execution

Can the agent handle tasks that take minutes or hours?

**Checklist:**
- [ ] Define async execution pattern (background jobs, polling, webhooks)
- [ ] Define progress reporting (how user tracks progress)
- [ ] Define cancellation capability
- [ ] Define timeout for long-running tasks
- [ ] Define result delivery (push notification, email, in-app)
- [ ] Define checkpoint frequency for long-running tasks

### 12. Streaming Responses

How does the agent deliver responses in real-time?

**Checklist:**
- [ ] Define streaming protocol (SSE, WebSocket, HTTP/2 streaming)
- [ ] Define what gets streamed (LLM tokens, tool call status, progress updates)
- [ ] Define streaming granularity (token-by-token, sentence-by-sentence, chunk)
- [ ] Define streaming error handling (mid-stream failure recovery)
- [ ] Define streaming for multi-step (show each step as it completes)

### 13. Multi-Modal Inputs

What input types does the agent accept?

**Checklist:**
- [ ] Define supported input modalities:

| Modality | Supported? | Max Size | Processing |
|----------|-----------|----------|-----------|
| Text | Yes/No | ___ chars | Direct to LLM |
| Images | Yes/No | ___ MB | Vision model / OCR |
| Audio | Yes/No | ___ min | Whisper / STT |
| Video | Yes/No | ___ min | Frame extraction + vision |
| Files (PDF, DOCX) | Yes/No | ___ MB | Parser + chunking |
| Structured data (CSV, JSON) | Yes/No | ___ MB | Direct parsing |

- [ ] Define multi-modal reasoning (can agent reason across text + image?)
- [ ] Define file upload handling (storage, processing pipeline, cleanup)

---

## Filled-Out Example: Customer Support Agent

### System: AI Customer Support Agent for E-Commerce Platform

**1. Natural Language Task Execution**
- [x] Tasks: Order status lookup, return/refund initiation, product questions, shipping inquiries, account issues, complaint escalation
- [x] Out-of-scope: Price negotiations, custom product requests, legal disputes
- [x] Ambiguity resolution: Ask clarifying question if intent confidence < 0.8
- [x] Multi-language: English, Spanish, French, German
- [x] Input: Text only (v1), text + images (v2 for product defect photos)

**2. Tool Calling**

| Tool Name | Description | Auth | Side Effects | Timeout | Retry |
|-----------|-------------|------|-------------|---------|-------|
| `lookup_order` | Get order details by order ID or email | No | Read | 3s | 2x |
| `search_products` | Search product catalog | No | Read | 2s | 2x |
| `initiate_return` | Start return process | Yes (HITL) | Write | 5s | 1x |
| `process_refund` | Issue refund to customer | Yes (HITL) | Write | 10s | 0 |
| `send_email` | Send email to customer | Yes (HITL) | Write | 5s | 1x |
| `search_kb` | Search knowledge base articles | No | Read | 2s | 2x |
| `escalate_to_human` | Transfer to human agent | No | Write | 3s | 1x |
| `update_shipping` | Update shipping address | Yes (HITL) | Write | 5s | 1x |

**3. Planning & Reasoning**
- [x] Pattern: ReAct (simple enough for single-domain tasks)
- [x] Max depth: 5 tool calls per turn
- [x] Replanning: If tool call fails, try alternative approach (max 1 replan)
- [x] Plan visibility: Not shown to user (too technical)

**4. Multi-Step Workflow Execution**
- [x] Supported workflows:
  - Return flow: verify order → check return policy → initiate return → confirm with customer
  - Refund flow: verify order → calculate refund amount → get HITL approval → process refund → send confirmation email
- [x] Partial failure: If refund processing fails, inform customer and create internal ticket
- [x] Cancellation: Customer can say "never mind" at any point
- [x] Resume: Not supported (stateless within session)

**5. Memory Persistence**
- [x] Short-term: Full conversation context within session
- [x] Long-term: Customer preferences (preferred contact method, past issues)
- [x] Episodic: Summary of past interactions (last 10 conversations)
- [x] Cross-session: Yes, remembers customer name and past issues

**6. Agent Collaboration**
- [x] Architecture: Single agent (v1), Supervisor + specialists (v2)
- [x] v2 specialists: Order Agent, Product Agent, Shipping Agent, Escalation Agent

**7. Human Approvals**
- [x] Return initiation: Auto-approve if within policy, HITL if exception
- [x] Refund > $50: Always requires human approval
- [x] Send external email: Always requires human approval
- [x] Approval timeout: 5 minutes, then inform customer of delay

**8. Retrieval Augmentation (RAG)**
- [x] Knowledge sources: Product FAQ (500 articles), Return policy (1 doc), Shipping policy (1 doc)
- [x] Indexing: 512-token chunks, 50-token overlap, `text-embedding-3-small`
- [x] Retrieval: Hybrid search (semantic + BM25), top-5 chunks, reranked
- [x] Citation: "Based on our return policy: [link]"
- [x] Update pipeline: Automated weekly re-index from CMS

**9. Real-Time Event Handling**
- [x] Not required for v1
- [x] v2: Webhook from shipping provider to proactively notify customer of delays

**10. External API Integrations**

| Service | API Version | Auth | Rate Limit | Fallback |
|---------|-----------|------|-----------|---------|
| Order Management System | v3 | API key | 100 RPS | Cached order data |
| Payment Gateway | v2 | OAuth 2.0 | 50 RPS | Queue refund for manual |
| Shipping Provider (FedEx) | v1 | API key | 30 RPS | Show last known status |
| Email Service (SendGrid) | v3 | API key | 100 RPS | Queue for retry |

**11. Long-Running Task Execution**
- [x] Not required (all operations complete within 30 seconds)

**12. Streaming Responses**
- [x] Protocol: SSE
- [x] Streaming: Token-by-token for LLM responses
- [x] Tool status: "Looking up your order..." shown as tool executes
- [x] Error: Mid-stream failure shows "I encountered an issue, let me try again"

**13. Multi-Modal Inputs**
- [x] v1: Text only
- [x] v2: Text + images (product defect photos, max 5MB, JPEG/PNG)
- [x] File upload: Not supported

---

## Blank Template

Copy this template for your system.

```markdown
# Functional Requirements: [System Name]

## 1. Natural Language Task Execution
- [ ] Task taxonomy: ___
- [ ] Out-of-scope tasks: ___
- [ ] Ambiguity resolution strategy: ___
- [ ] Multi-language: ___
- [ ] Input modality: ___

## 2. Tool Calling

| Tool Name | Description | Auth | Side Effects | Timeout | Retry |
|-----------|-------------|------|-------------|---------|-------|
| | | | | | |

## 3. Planning & Reasoning
- [ ] Pattern: ___
- [ ] Max depth: ___ tool calls per turn
- [ ] Replanning trigger: ___
- [ ] Plan visibility to user: Yes / No

## 4. Multi-Step Workflow Execution
- [ ] Supported workflows: ___
- [ ] Partial failure handling: ___
- [ ] Cancellation: ___
- [ ] Resume after interruption: ___

## 5. Memory Persistence
- [ ] Short-term: ___
- [ ] Long-term: ___
- [ ] Episodic: ___
- [ ] Cross-session memory: Yes / No

## 6. Agent Collaboration
- [ ] Architecture: Single / Supervisor+Workers / Peer-to-Peer / Hierarchical
- [ ] Handoff rules: ___
- [ ] Shared context: ___

## 7. Human Approvals (HITL)

| Action | Approval Required? | Threshold | Timeout |
|--------|-------------------|-----------|---------|
| | | | |

## 8. Retrieval Augmentation (RAG)
- [ ] Knowledge sources: ___
- [ ] Indexing strategy: ___
- [ ] Retrieval strategy: ___
- [ ] Citation format: ___
- [ ] Update frequency: ___

## 9. Real-Time Event Handling
- [ ] Event sources: ___
- [ ] Event-to-action mapping: ___
- [ ] Ordering: ___

## 10. External API Integrations

| Service | Version | Auth | Rate Limit | Fallback |
|---------|---------|------|-----------|---------|
| | | | | |

## 11. Long-Running Task Execution
- [ ] Async pattern: ___
- [ ] Progress reporting: ___
- [ ] Max execution time: ___

## 12. Streaming Responses
- [ ] Protocol: SSE / WebSocket / HTTP/2
- [ ] Granularity: Token / Sentence / Chunk
- [ ] Mid-stream error handling: ___

## 13. Multi-Modal Inputs

| Modality | Supported? | Max Size | Processing |
|----------|-----------|----------|-----------|
| Text | | | |
| Images | | | |
| Audio | | | |
| Files | | | |
```

## Decision Tree: Which Requirements Apply?

```
What type of agent are you building?
│
├── Simple Q&A / chatbot
│   └── Need: #1 (tasks), #2 (tools, maybe), #8 (RAG), #12 (streaming)
│       Skip: #4 (workflows), #6 (multi-agent), #9 (events), #11 (async)
│
├── Task automation agent (e.g., code agent, data agent)
│   └── Need: #1-5 (all core), #7 (HITL for writes), #10 (APIs), #11 (async)
│       Skip: #9 (events), #13 (multi-modal usually)
│
├── Customer-facing support agent
│   └── Need: #1-5, #7 (HITL), #8 (RAG), #10 (APIs), #12 (streaming)
│       Skip: #6 (single agent OK), #9 (events v2), #11 (not needed)
│
├── Multi-domain platform agent
│   └── Need: ALL 13 categories
│
└── Internal copilot (IDE, Slack bot)
    └── Need: #1 (tasks), #2 (tools), #3 (planning), #12 (streaming)
        Skip: #7 (less HITL), #8 (maybe), #9 (events via Slack)
```

## When NOT to Write Detailed Functional Requirements

- **Hackathon / prototype**: Write a one-paragraph description of what the agent should do. Ship it.
- **Research / experimentation**: Define the task and evaluation criteria. Skip the template.
- **Solo developer project**: The template is for team communication. If you are the team, just build it.

## Tradeoffs

| Requirement Area | Adding It Costs | Skipping It Risks |
|-----------------|----------------|-------------------|
| Tool calling | Schema definition + error handling per tool | Agent can't take actions |
| Memory persistence | Storage cost + privacy compliance | Agent forgets everything |
| HITL approvals | Latency (waiting for human) | Uncontrolled agent actions |
| RAG pipeline | Indexing infra + retrieval latency | Agent hallucinates answers |
| Multi-agent | Coordination complexity + debugging difficulty | Single agent bottleneck |
| Streaming | WebSocket/SSE infrastructure | Poor UX for long responses |
| Multi-modal | Vision/audio model costs + processing pipeline | Limited input types |

## Failure Modes

1. **Underspecified tool schemas**: Agent hallucinates tool parameters because the schema is too vague. Mitigation: strict JSON Schema with examples for every tool.

2. **Missing out-of-scope definition**: Agent attempts tasks it should not (e.g., giving medical advice). Mitigation: explicit out-of-scope list in system prompt + output guardrails.

3. **HITL bottleneck**: Every action requires approval, creating a queue. Mitigation: define approval thresholds (only approve high-risk actions).

4. **Memory over-reliance**: Agent trusts stale memories over current context. Mitigation: timestamp memories; prefer recent context over old memory.

5. **Tool ordering not enforced**: Agent calls `process_refund` before `verify_order`. Mitigation: enforce ordering in graph structure (LangGraph edges), not just in prompt.

## Source(s) and Further Reading

- LangChain Tool Calling: https://python.langchain.com/docs/concepts/tool_calling/
- LangGraph Human-in-the-Loop: https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
- Anthropic Tool Use: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- "Building LLM Applications" - Chip Huyen (2024)
- Agent Design Patterns: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/
