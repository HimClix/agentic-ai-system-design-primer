# Interview Template: Agentic System Design (45 Minutes)
> A structured template for whiteboard interviews -- what to draw first, what to explain, and how to budget your time.

## What It Is

This is a blank template and structured approach for tackling agentic AI system design questions in 45-minute interviews. It provides a 12-step checklist with time budgets, suggested talking points, and common tradeoff discussions.

## Time Budget

```
Total: 45 minutes

Phase 1: Requirements (5 min)
├── Step 1: Clarify the problem (2 min)
└── Step 2: Define scope and constraints (3 min)

Phase 2: High-Level Design (15 min)
├── Step 3: Draw the architecture (5 min)
├── Step 4: Define the agent loop (3 min)
├── Step 5: Identify tools and data sources (3 min)
└── Step 6: Design the data flow (4 min)

Phase 3: Deep Dive (20 min)
├── Step 7: Detail 2-3 critical components (8 min)
├── Step 8: Discuss failure modes and reliability (4 min)
├── Step 9: Address scalability (4 min)
└── Step 10: Cover cost engineering (4 min)

Phase 4: Wrap-Up (5 min)
├── Step 11: Discuss tradeoffs and alternatives (3 min)
└── Step 12: Summarize and Q&A (2 min)
```

## The 12-Step Checklist

### Step 1: Clarify the Problem (2 min)

**Ask these questions:**
- What is the agent's primary task? (support, code review, data analysis)
- Who are the users? (customers, internal team, API consumers)
- What is the expected volume? (requests/day, concurrent sessions)
- What are the latency requirements? (real-time chat, async batch)
- Are there compliance requirements? (GDPR, SOC2, PII handling)

**Template answer:**
> "So we're building a [type] agent that [primary action] for [users]. It needs to handle [volume] with [latency] requirements. Let me confirm: [repeat back key constraints]."

### Step 2: Define Scope (3 min)

**What's in scope:**
- [ ] Core agent loop (always)
- [ ] Tool integrations (which ones?)
- [ ] Memory (short-term? long-term?)
- [ ] RAG (knowledge base needed?)
- [ ] Multi-tenancy (B2B SaaS?)

**What's out of scope (explicitly state):**
- "I'll mention but not deep-dive into: deployment infrastructure, CI/CD pipeline, monitoring dashboards."

### Step 3: Draw the Architecture (5 min)

**Draw in this order:**
1. User on the left
2. API Gateway
3. Orchestrator (the box the agent lives in)
4. LLM Provider (cloud/arrow)
5. Tool boxes below the orchestrator
6. Memory/Database to the right
7. RAG pipeline if needed
8. Observability across the top

**Minimal viable diagram:**
```
User → Gateway → Orchestrator → LLM
                     │
              ┌──────┼──────┐
              ▼      ▼      ▼
           Tool A  Tool B  Memory
              │
              ▼
          RAG / VectorDB
```

### Step 4: Define the Agent Loop (3 min)

**Draw the state machine:**
```
receive_input → plan → execute → verify → respond
                  │                  │
                  └── loop ──────────┘
```

**Key talking points:**
- "The orchestrator runs a state machine. Each transition is deterministic, even though the LLM's reasoning is not."
- "I'd use LangGraph for the state machine because it gives me checkpointing and human-in-the-loop for free."
- "Maximum N steps to prevent infinite loops (bounded autonomy)."

### Step 5: Identify Tools (3 min)

**For each tool, state:**
1. What it does (one sentence)
2. Read or write operation
3. Latency expectation
4. Failure handling

**Example:**
> "The agent needs three tools: (1) search_knowledge_base -- read-only, returns in ~200ms, falls back to keyword search if vector search fails. (2) create_ticket -- write operation, requires confirmation for important tickets. (3) send_email -- write operation, always requires human approval."

### Step 6: Design Data Flow (4 min)

**Trace a single request end-to-end:**
> "User sends 'I want a refund.' The gateway authenticates and routes to the orchestrator. The orchestrator sends the message to the LLM with the system prompt and conversation history. The LLM decides to call search_knowledge_base('refund policy'). The tool returns the policy. The LLM reads it and generates a response. The orchestrator runs guardrails, then streams the response back via SSE."

### Step 7: Deep Dive on Critical Components (8 min)

**Pick 2-3 based on the interviewer's interest or the problem's core challenge.**

Common deep dives:
- **RAG pipeline**: Chunking strategy, embedding model, reranking, hybrid search
- **Agent loop**: State machine design, conditional edges, error handling
- **Cost engineering**: Model routing, token budgets, caching
- **Memory**: Short-term (conversation), long-term (user preferences), episodic (past interactions)

### Step 8: Failure Modes (4 min)

**Reference the MAST taxonomy:**
> "The three most common failure modes I'd watch for are: (1) step repetition -- the agent calls the same tool repeatedly. I'd add a deduplication check. (2) Not recognizing completion -- the agent keeps going after answering. I'd add explicit completion criteria. (3) Tool parameter misuse -- I'd use Pydantic validation on all tool inputs."

**Reliability stack:**
> "For production reliability, I'd implement: bounded autonomy (max 25 steps), per-tool circuit breakers, deterministic output validators, and a fallback ladder (primary tool -> backup tool -> admit unknown)."

### Step 9: Scalability (4 min)

**State the scaling strategy:**
> "The stateless components -- API gateway, tool workers, guardrails -- scale horizontally with auto-scaling on CPU/queue depth. The orchestrator is stateful but I'd externalize state to Redis, making it effectively stateless for scaling. The database scales with read replicas. The vector DB is the one component I'd watch -- at 10M+ vectors, I'd consider sharding by tenant."

### Step 10: Cost Engineering (4 min)

**Key numbers to mention:**
- "Average support ticket costs 3,550 tokens, about $0.01 on Sonnet."
- "I'd implement model routing: classification and extraction on Haiku ($0.80/M), reasoning on Sonnet ($3/M), complex cases on Opus ($15/M). This saves 60-70% vs using Opus for everything."
- "For the projected volume of X, monthly LLM cost would be approximately Y."

### Step 11: Tradeoffs (3 min)

**Always discuss at least two tradeoffs:**
1. "I chose X over Y because [reason], but Y would be better if [condition]."
2. "The main risk of this design is [risk]. I'd mitigate it by [mitigation]."

**Common tradeoffs:**
- Single agent vs multi-agent
- API LLM vs self-hosted
- Sync vs async execution
- Strong consistency vs eventual consistency for memory

### Step 12: Summary (2 min)

**Template:**
> "To summarize: we built a [type] agent using [orchestrator] with [N] tools. The key architectural decisions were: (1) [decision], (2) [decision], (3) [decision]. The main risks are [risk] and [risk], mitigated by [mitigation]. At [volume], monthly cost would be approximately [cost]."

## Common Interviewer Questions and Answers

| Question | Good Answer Template |
|----------|---------------------|
| "How would you handle 10x the traffic?" | "The stateless components auto-scale. For the orchestrator, I'd externalize state to Redis. The bottleneck would be [X], which I'd address by [Y]." |
| "What if the LLM hallucinates?" | "Three layers: RAG for grounding, deterministic output validation, and confidence scoring. Below 0.85 confidence, I escalate to a human." |
| "How do you test this?" | "Unit tests with mocked LLM, integration tests with real LLM calls, and an eval suite on 50-200 golden test cases measuring faithfulness and answer relevancy." |
| "What about security?" | "Zero-trust: session-scoped credentials, per-tool authorization, network egress allowlist, and full audit logging of every tool call with reasoning." |
| "How do you keep costs under control?" | "Model routing (60-70% savings), token budgets per tenant, conversation summarization for long sessions, and semantic caching for repeated queries." |

## Failure Modes (Of the Interview Itself)

1. **Spending too long on requirements**: Cap at 5 minutes. Make reasonable assumptions and state them.
2. **Drawing too detailed a diagram**: Start high-level, add detail only where asked.
3. **Not discussing tradeoffs**: Every design decision should acknowledge alternatives.
4. **Ignoring cost**: Always mention cost implications. It shows production awareness.
5. **Over-engineering**: For an MVP or early-stage product, simpler is better. Say so.

## Source(s) and Further Reading

- "System Design Interview" - Alex Xu
- "Building Effective Agents" - Anthropic (2024)
- "Designing Machine Learning Systems" - Chip Huyen
- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
