# Reference Architecture for Agentic Systems
> The expanded 12-component architecture with scaling boundaries, ownership boundaries, and what breaks if you remove each component.

## What It Is

This is the canonical reference architecture for production agentic systems. Every component is explained with its purpose, failure impact, scaling characteristics, and ownership boundary. Use this as the starting point for any agentic system design.

## The Architecture

```
                                    ┌─────────────────────────────────┐
                                    │         OBSERVABILITY           │
                                    │  (LangSmith / Datadog / OTEL)  │
                                    │  Logs, Metrics, Traces, Evals  │
                                    └────────────────┬────────────────┘
                                                     │ collects from all
                                                     │
┌──────────┐    ┌──────────────┐    ┌────────────────┴────────────────┐
│          │    │              │    │                                  │
│  User    ├───►│  API Gateway ├───►│         ORCHESTRATOR             │
│ Interface│    │              │    │                                  │
│          │◄───│ - Auth       │◄───│  ┌──────────┐  ┌─────────────┐ │
│ - Chat   │    │ - Rate limit │    │  │  Agent   │  │ Guardrails  │ │
│ - API    │    │ - Routing    │    │  │  Loop    │  │ - Input     │ │
│ - Slack  │    │ - TLS       │    │  │  (State  │  │ - Output    │ │
│ - Email  │    │              │    │  │  Machine)│  │ - PII       │ │
└──────────┘    └──────────────┘    │  └────┬─────┘  └─────────────┘ │
                                    │       │                         │
                                    │  ┌────┴──────────────────────┐  │
                                    │  │     LLM PROVIDER          │  │
                                    │  │                           │  │
                                    │  │  ┌────────┐ ┌──────────┐ │  │
                                    │  │  │ Router │ │ Provider │ │  │
                                    │  │  │(model  │ │(Anthropic│ │  │
                                    │  │  │ select)│ │ OpenAI,  │ │  │
                                    │  │  └────────┘ │ self-host│ │  │
                                    │  │             └──────────┘ │  │
                                    │  └──────────────────────────┘  │
                                    └────────┬───────────┬───────────┘
                                             │           │
                              ┌──────────────┘           └──────────────┐
                              ▼                                          ▼
                 ┌────────────────────┐                     ┌────────────────────┐
                 │    TOOL LAYER      │                     │   MEMORY STORE     │
                 │                    │                     │                    │
                 │ ┌──────┐ ┌──────┐ │                     │ ┌───────┐ ┌──────┐│
                 │ │MCP   │ │MCP   │ │                     │ │Redis  │ │PG    ││
                 │ │Server│ │Server│ │                     │ │(short)│ │(long)││
                 │ │(KB)  │ │(CRM) │ │                     │ └───────┘ └──────┘│
                 │ └──────┘ └──────┘ │                     │                    │
                 │ ┌──────┐ ┌──────┐ │                     │ ┌────────────────┐│
                 │ │MCP   │ │Code  │ │                     │ │ Checkpoint     ││
                 │ │Server│ │Exec  │ │                     │ │ Store (PG)     ││
                 │ │(API) │ │(Sbox)│ │                     │ └────────────────┘│
                 │ └──────┘ └──────┘ │                     └────────────────────┘
                 └────────────────────┘
                              │
                              ▼
                 ┌────────────────────┐
                 │   RAG PIPELINE     │
                 │                    │
                 │ Query → Embed →    │
                 │ Search → Rerank → │
                 │ Format             │
                 │                    │
                 │ ┌────────────────┐ │
                 │ │ Vector DB      │ │
                 │ │ (Pinecone/PG)  │ │
                 │ └────────────────┘ │
                 └────────────────────┘
```

## Component Details

### 1. User Interface

| Attribute | Details |
|-----------|---------|
| **Purpose** | Accept user input, display agent responses |
| **If removed** | No way for users to interact with the agent |
| **Scales by** | Frontend scaling (CDN, static hosting) |
| **Owns** | UI state, user session, display formatting |
| **Technology** | React/Next.js (chat), REST API (programmatic), Slack Bot SDK |

### 2. API Gateway

| Attribute | Details |
|-----------|---------|
| **Purpose** | Authentication, rate limiting, request routing, TLS termination |
| **If removed** | No auth, no rate limiting, direct exposure of internal services |
| **Scales by** | Horizontal (stateless) -- Kong, NGINX, AWS ALB |
| **Owns** | Auth tokens, rate limit counters, routing rules |
| **Technology** | Kong, NGINX, AWS ALB, Cloudflare |

### 3. Orchestrator

| Attribute | Details |
|-----------|---------|
| **Purpose** | Manage the agent execution loop, state transitions, step tracking |
| **If removed** | No coordination between agent steps; single-shot only |
| **Scales by** | Horizontal with sticky sessions or externalized state |
| **Owns** | Execution state, step history, current phase |
| **Technology** | LangGraph, custom state machine, Temporal |

### 4. Agent(s) / LLM Provider

| Attribute | Details |
|-----------|---------|
| **Purpose** | Make decisions: which tool to call, what to say, when to stop |
| **If removed** | No intelligence -- just a pipeline of hardcoded steps |
| **Scales by** | Provider rate limits (API) or GPU instances (self-hosted) |
| **Owns** | Reasoning, tool selection, response generation |
| **Technology** | Claude Sonnet/Opus, GPT-4o, Llama (self-hosted via vLLM) |

### 5. Tool Layer

| Attribute | Details |
|-----------|---------|
| **Purpose** | Execute actions in the real world (search, create, update) |
| **If removed** | Agent can only answer from training data (no real-time info) |
| **Scales by** | Horizontal (stateless workers), rate limited by external APIs |
| **Owns** | Tool schemas, API integrations, credential management |
| **Technology** | MCP servers, LangChain tools, direct API wrappers |

### 6. Memory Store

| Attribute | Details |
|-----------|---------|
| **Purpose** | Maintain context within and across sessions |
| **If removed** | Agent forgets everything between turns (stateless) |
| **Scales by** | Redis cluster (short-term), PG read replicas (long-term) |
| **Owns** | Conversation history, user preferences, learned facts |
| **Technology** | Redis (sessions), PostgreSQL (persistent), Mem0/Zep (managed) |

### 7. RAG Pipeline

| Attribute | Details |
|-----------|---------|
| **Purpose** | Retrieve relevant knowledge to ground agent responses |
| **If removed** | Agent relies only on training data -- will hallucinate on domain-specific questions |
| **Scales by** | Vector DB read replicas, embedding service horizontal scaling |
| **Owns** | Document index, chunking strategy, relevance scoring |
| **Technology** | Pinecone/pgvector (vector DB), Cohere/Voyage (reranking) |

### 8. Persistence

| Attribute | Details |
|-----------|---------|
| **Purpose** | Durable storage for runs, audit logs, checkpoints |
| **If removed** | No history, no audit trail, no checkpoint recovery |
| **Scales by** | PG vertical + read replicas, partitioning by time |
| **Owns** | Run history, tool call logs, compliance data |
| **Technology** | PostgreSQL, S3 (large artifacts) |

### 9. Guardrails

| Attribute | Details |
|-----------|---------|
| **Purpose** | Validate inputs and outputs for safety, format, policy |
| **If removed** | Agent can produce harmful, off-topic, or malformatted responses |
| **Scales by** | Horizontal (stateless validators) |
| **Owns** | Content policies, format rules, PII detection |
| **Technology** | Custom validators, Guardrails AI, NeMo Guardrails |

### 10. Observability

| Attribute | Details |
|-----------|---------|
| **Purpose** | Monitor agent health, performance, quality, and cost |
| **If removed** | Blind to failures, no debugging capability, no quality metrics |
| **Scales by** | Collector horizontal scaling, storage backend |
| **Owns** | Dashboards, alerts, trace storage |
| **Technology** | LangSmith, Datadog, OpenTelemetry, Prometheus + Grafana |

### 11. Output Delivery

| Attribute | Details |
|-----------|---------|
| **Purpose** | Format and stream responses to users |
| **If removed** | Raw LLM output without formatting; no streaming UX |
| **Scales by** | Connection-based (SSE connections per pod) |
| **Owns** | Response formatting, streaming protocol, delivery guarantees |
| **Technology** | SSE, WebSocket, webhook callbacks |

## Scaling Boundaries

```
Independent Scaling Groups:

Group A: User-Facing (scale by request volume)
├── API Gateway
├── Output Delivery
└── User Interface

Group B: Orchestration (scale by active sessions)
├── Orchestrator
└── Memory Store (Redis)

Group C: Execution (scale by tool call volume)
├── Tool Workers
├── RAG Pipeline
└── Embedding Service

Group D: Persistence (scale by data volume)
├── PostgreSQL
├── Vector Database
└── Checkpoint Store

Group E: Cross-Cutting (scale by event volume)
├── Observability Pipeline
└── Guardrail Validators
```

## When NOT to Build the Full Architecture

- **MVP**: Start with components 3 (orchestrator), 4 (agent), 5 (1-2 tools), and 12 (output). Add others as needed.
- **Internal tool**: Skip 2 (gateway), 9 (guardrails). Lighter security needs.
- **Stateless agent**: Skip 7 (memory store), 8 (persistence). Each request is independent.

## Source(s) and Further Reading

- "Building Effective Agents" - Anthropic (2024)
- LangGraph Architecture: https://langchain-ai.github.io/langgraph/concepts/high_level/
- "Designing Data-Intensive Applications" - Martin Kleppmann
- AWS Well-Architected Framework for AI/ML
