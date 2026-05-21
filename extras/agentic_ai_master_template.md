# Agentic AI & Multi-Agent Orchestration — Master Template HLD

> Equivalent to the Backend System Design HLD Master Template, but for AI agent systems.
> Every production agentic system is a composition of these layers.

---

## Full Architecture

```mermaid
flowchart TB
    User(["👤 User / External System"])
    User --> GW["🔐 API Gateway\nAuth · Rate Limit · Routing"]
    GW --> OC["🧠 Orchestrator Agent\nPlanner · Router · Loop Controller"]

    %% side components hang off orchestrator
    OC <--> SM["📋 State Manager\nsession + scratchpad"]
    OC <--> TQ["📨 Task Queue\nCelery / BullMQ / SQS"]
    OC -->|escalate| HQ["🛑 HITL Approval Queue"]
    HQ --> HUI["👤 Human Review UI"] --> OC

    OC --> AGENTS

    subgraph AGENTS["🤖 Specialist Agent Layer"]
        direction LR
        SA1["🔍 Research\nAgent"] --- SA2["💻 Code\nAgent"] --- SA3["⚡ Action\nAgent"] --- SA4["✅ Critic /\nVerifier"]
    end

    AGENTS --> LR["🔀 LLM Router\ncost · latency · capability"]
    LR --> LLMS

    subgraph LLMS["🧩 LLM Inference Layer"]
        direction LR
        LM1["Claude Opus\n/ GPT-4o"] --- LM2["Haiku\n/ GPT-4o-mini"] --- LM3["Specialist\nModel"]
    end

    SA3 --> TC["🛠 Tool Caller  (MCP / OpenAPI)"]
    TC --> TOOLS

    subgraph TOOLS["⚙️ Tool / Action Layer"]
        direction LR
        T1["🌐 Web Search"] --- T2["🖥 Code Exec\nsandbox"] --- T3["🔌 External\nAPIs"] --- T4["🖱 Browser\nPlaywright"] --- T5["📧 Comms\nSlack / Email"]
    end

    SA1 --> RAG

    subgraph RAG["📚 Knowledge / RAG Layer"]
        direction LR
        KB["Knowledge\nBase"] --> EM["Embedding\nModel"] --> VDB["Vector Store\npgvector/Pinecone"] --> RR["Re-ranker"]
    end

    RAG --> SA1

    OC --> MEM

    subgraph MEM["🧠 Memory Layer"]
        direction LR
        M1["⚡ Short-term\nRedis / in-context"] --- M2["🗃 Long-term\nVector DB"] --- M3["📖 Episodic\nrun logs"] --- M4["📚 Procedural\nskill registry"]
    end

    AGENTS --> STORE
    TOOLS  --> STORE

    subgraph STORE["💾 Persistence Layer"]
        direction LR
        PS["🐘 PostgreSQL\nusers · tasks · runs"] --- DS["🍃 MongoDB\noutputs"] --- CS["⚡ Redis\ncache"] --- FS["☁️ S3\nartifacts"]
    end

    LLMS  --> OBS
    STORE --> OBS

    subgraph OBS["📡 Observability & Tracing"]
        direction LR
        OT["🔭 LLM Observability\nLangfuse / LangSmith"] --- TT["📡 Tracing\nOpenTelemetry"] --- ME["📊 Metrics\ntoken · cost · latency"] --- AL["🚨 Alerting\nPagerDuty / Grafana"]
    end

    OBS --> OR["📤 Response Assembler\nstream / batch / webhook"]
    OR -->|final answer| User

    %% ── Styling ──────────────────────────────────────────
    classDef entry   fill:#1e3a5f,color:#fff,stroke:#4a90d9,stroke-width:2px
    classDef orch    fill:#0d3d2e,color:#fff,stroke:#27ae60,stroke-width:2px
    classDef llm     fill:#2d1a4a,color:#fff,stroke:#9b59b6,stroke-width:2px
    classDef tool    fill:#3d1f00,color:#fff,stroke:#e67e22,stroke-width:2px
    classDef memory  fill:#003d3d,color:#fff,stroke:#1abc9c,stroke-width:2px
    classDef persist fill:#1a1a3d,color:#fff,stroke:#3498db,stroke-width:2px
    classDef obs     fill:#3d0000,color:#fff,stroke:#e74c3c,stroke-width:2px
    classDef hitl    fill:#3d3d00,color:#fff,stroke:#f1c40f,stroke-width:2px
    classDef output  fill:#2a2a2a,color:#fff,stroke:#95a5a6,stroke-width:2px

    class GW entry
    class OC,SM,TQ orch
    class LR,LM1,LM2,LM3 llm
    class TC,T1,T2,T3,T4,T5 tool
    class M1,M2,M3,M4,KB,EM,VDB,RR memory
    class PS,DS,CS,FS persist
    class OT,TT,ME,AL obs
    class HQ,HUI hitl
    class OR output
```

---

## Layer Reference

| Layer | What it does | Key technologies |
|---|---|---|
| **Entry** | Auth, routing, rate limiting | API Gateway, OAuth, JWT |
| **Orchestration** | Plans tasks, routes to agents, manages state | LangGraph, CrewAI, custom planner |
| **Specialist Agents** | Domain-specific reasoning and action | ReAct loop, tool-use, reflection |
| **LLM Inference** | Language model calls, cost/latency routing | Claude, GPT-4o, Gemini, vLLM |
| **Tools / Actions** | Real-world side effects | MCP, OpenAPI, Playwright, sandboxed exec |
| **Memory** | Short-term (context), long-term (vector), episodic | Redis, Pinecone, Weaviate, pgvector |
| **RAG / Knowledge** | Retrieval-augmented generation pipeline | Embeddings, re-ranker, vector store |
| **Persistence** | Durable storage for runs, users, artifacts | PostgreSQL, MongoDB, S3, Redis |
| **Observability** | Traces, token usage, cost, errors | Langfuse, LangSmith, OpenTelemetry |
| **HITL** | Human approval for high-risk / low-confidence actions | Approval queues, Slack bots |
| **Output** | Assembles and delivers final response or artifact | Streaming, batch, webhooks |

---

## Key Differences from Backend HLD

| Backend HLD | Agentic AI HLD |
|---|---|
| Load Balancer | Orchestrator Agent |
| Microservices | Specialist Agents |
| Database reads | Memory (short/long-term) + RAG |
| Message Queue | Task Queue + Agent loops |
| API calls | Tool calls (MCP / OpenAPI) |
| Logging | LLM Observability + Tracing |
| Human ops | Human-in-the-Loop (HITL) layer |
```
