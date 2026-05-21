# High-Level Design Methodology for Agentic Systems
> HLD defines WHAT components exist and WHY -- LLD defines HOW each component works internally.

## What It Is

High-Level Design (HLD) for agentic systems describes the system architecture at the component level: what services exist, how they communicate, where data flows, and what responsibilities each component owns. HLD answers "what are we building and why" without prescribing implementation details.

In interviews, HLD is what you draw on the whiteboard in the first 15-20 minutes. It establishes the architectural boundaries before diving into any component's internals.

## How It Works

### What Goes in HLD vs LLD

| Aspect | HLD (Architecture) | LLD (Implementation) |
|--------|-------------------|---------------------|
| Components | Which services exist | How each service works internally |
| Communication | Sync (HTTP) vs async (queue) | API contracts, request/response schemas |
| Data flow | Which data goes where | Data models, schemas, indexes |
| State management | Where state lives (Redis, PG) | State machine design, transitions |
| Scaling | Which components scale independently | Auto-scaling rules, pod specs |
| Reliability | Failure boundaries | Circuit breaker config, retry policies |
| Security | Trust boundaries | Auth implementation, credential flow |

### The Canonical 12-Component Agentic HLD

Every agentic system, regardless of use case, maps to these 12 components:

```
1.  User Interface       → How users interact (chat, API, Slack)
2.  API Gateway          → Rate limiting, auth, routing
3.  Orchestrator         → Agent loop, state management
4.  Agent(s)             → LLM-powered decision-makers
5.  LLM Provider         → Model inference (API or self-hosted)
6.  Tool Layer           → External integrations (MCP servers)
7.  Memory Store         → Short-term (Redis) + long-term (PG)
8.  RAG Pipeline         → Document retrieval + reranking
9.  Persistence          → Checkpoints, audit logs, run history
10. Observability        → Logging, metrics, tracing
11. Guardrails           → Input/output validation, safety
12. Output Delivery      → Response formatting, streaming
```

Not every system needs all 12 at full complexity, but every system has a decision about each one (even if the decision is "we don't need this yet").

## Decision Tree: HLD Depth by Component

```
For each of the 12 components, decide:
│
├── CRITICAL for this use case?
│   └── Design in detail: scaling, failure modes, data model
│
├── IMPORTANT but standard?
│   └── Mention the component, state the technology choice, move on
│
└── NOT NEEDED for this use case?
    └── Explicitly state "we're not including X because Y"
        (shows interviewer you considered it)
```

## When NOT to Start with HLD

- **Coding interview**: They want working code, not architecture diagrams.
- **Simple enhancement to existing system**: Start with the existing architecture and show the delta.
- **LLD-focused interview**: Some interviews explicitly ask for low-level design (state machines, APIs). Start there.

## Tradeoffs

| HLD Style | Interview Suitability | Completeness | Risk |
|-----------|----------------------|-------------|------|
| Full 12-component | 60-min design interview | Complete | May run out of time |
| Focused 6-component | 45-min interview | Core components only | May miss important aspects |
| Delta on existing system | Enhancement interviews | Shows practical judgment | Assumes shared context |

## Real-World Examples

- **Customer support agent HLD**: User (chat widget) -> API Gateway (rate-limited) -> Orchestrator (LangGraph) -> Agent (Claude) -> Tools (KB search, ticketing) -> Memory (Redis sessions) -> RAG (Pinecone) -> Output (SSE streaming)
- **Code review agent HLD**: GitHub webhook -> API Gateway -> Orchestrator -> Agent (Claude) -> Tools (git diff, linter, test runner) -> Persistence (PostgreSQL) -> Output (PR comment)

## Source(s) and Further Reading

- "Building Effective Agents" - Anthropic (2024)
- System Design Interview - Alex Xu
- LangGraph Architecture: https://langchain-ai.github.io/langgraph/concepts/high_level/
