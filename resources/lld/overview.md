# Low-Level Design Methodology for Agentic Systems
> LLD answers HOW each component works: state machines, API contracts, data models, streaming protocols, and prompt contracts.

## What It Is

Low-Level Design (LLD) for agentic systems specifies the internal implementation of each component identified in the HLD. While HLD defines boxes and arrows, LLD defines what happens inside each box: the state transitions, data schemas, API endpoints, error handling, and performance characteristics.

In interviews, LLD is what you dive into after establishing the architecture -- typically for the 2-3 most critical or interesting components.

## How It Works

### LLD Components for Agentic Systems

| LLD Area | What You Specify | Agentic-Specific Aspect |
|----------|-----------------|------------------------|
| **State Machine** | States, transitions, conditions | Agent loop with nondeterministic transitions |
| **API Contracts** | Endpoints, request/response schemas | Streaming (SSE), long-running operations |
| **Data Models** | SQL schemas, indexes, partitioning | Embeddings (pgvector), JSONB for tool calls |
| **Streaming Protocol** | Message format, event types | Intermediate steps, tool call results |
| **Prompt Contracts** | System prompts, tool descriptions | The "code" that controls agent behavior |
| **Error Handling** | Error types, retry policies | LLM errors vs tool errors vs validation errors |

### What Level of Detail

```
Interview LLD depth (pick 2-3):
├── State machine: Full state graph with conditional edges
├── API contract: One primary endpoint with full schema
├── Data model: 2-3 key tables with indexes
├── Streaming: Event format with one example flow
└── Prompt contract: System prompt structure (not full text)

Production LLD depth (all):
├── State machine: Complete graph + edge conditions + error states
├── API contracts: All endpoints + auth + pagination + error codes
├── Data models: All tables + indexes + migrations + partitioning
├── Streaming: Full protocol spec + reconnection + backpressure
├── Prompt contracts: Versioned prompts + A/B test framework
└── Error handling: Complete error taxonomy + retry policies
```

## When LLD Matters in Interviews

- **"How does the agent decide what to do next?"** --> State machine design
- **"Show me the API"** --> API contracts
- **"How do you store conversation history?"** --> Data models
- **"How does the user see real-time progress?"** --> Streaming protocol
- **"Walk me through the prompt"** --> Prompt contracts

## Decision Tree: Which LLD to Deep Dive

```
What's the interview emphasis?
│
├── Backend/Infrastructure focus
│   └── API contracts + data models + state machine
│
├── AI/ML focus
│   └── State machine + prompt contracts + RAG pipeline
│
├── Product/UX focus
│   └── Streaming protocol + API contracts + error handling
│
└── Systems/Reliability focus
    └── State machine + error handling + data models
```

## Tradeoffs

| LLD Area | Time to Design | Implementation Risk | Interview Value |
|----------|---------------|-------------------|----------------|
| State machine | 1-2 hours | Medium (complex transitions) | Very high |
| API contracts | 1 hour | Low (well-understood) | High |
| Data models | 1-2 hours | Medium (performance tuning) | High |
| Streaming | 30 min | Low | Medium |
| Prompt contracts | 30 min | Low | High for AI roles |

## Source(s) and Further Reading

- LangGraph State Machine: https://langchain-ai.github.io/langgraph/concepts/low_level/
- API Design Guidelines: https://google.aip.dev/
- PostgreSQL Performance: https://use-the-index-luke.com/
