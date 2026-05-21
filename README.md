# Agentic AI System Design Primer

<p align="center">
  <strong>Learn how to design production-grade AI agent systems</strong>
</p>

<p align="center">
  <em>The "System Design Primer" for Agentic AI — covering architecture, patterns, tradeoffs, and real-world production engineering</em>
</p>

> Everything is a tradeoff. In agentic AI, the tradeoffs are: **autonomy vs control**, **cost vs quality**, **latency vs accuracy**, and **simplicity vs capability**.

---

## Deep-Dive Production Guide

> This README is the primer — overview + interview prep. For **production-depth content** with code, decision trees, and real-world patterns, explore the `/resources/` folder:

| Category | Overview | What's Inside |
|---|---|---|
| [Agents](resources/agents/overview.md) | Patterns, multi-agent, frameworks, SDKs | [ReAct](resources/agents/patterns/react.md) · [Plan-and-Execute](resources/agents/patterns/plan-and-execute.md) · [Reflexion](resources/agents/patterns/reflexion.md) · [Pattern Decision Tree](resources/agents/patterns/decision-tree.md) · [Supervisor](resources/agents/multi-agent/supervisor.md) · [When to Go Multi-Agent](resources/agents/multi-agent/when-to-go-multi-agent.md) · [LangChain](resources/agents/frameworks/langchain.md) · [LangGraph](resources/agents/frameworks/langgraph.md) · [CrewAI](resources/agents/frameworks/crewai.md) · [AutoGen](resources/agents/frameworks/autogen.md) · [LangFlow](resources/agents/frameworks/langflow.md) · [Framework Comparison](resources/agents/frameworks/comparison.md) · [Claude SDK](resources/agents/sdks/claude-agent-sdk.md) · [OpenAI SDK](resources/agents/sdks/openai-agents-sdk.md) |
| [Memory](resources/memory/overview.md) | Short-term, long-term, episodic, procedural | [Context Window](resources/memory/short-term/context-window.md) · [Token Budgeting](resources/memory/short-term/token-budgeting.md) · [Compression](resources/memory/short-term/compression.md) · [Mem0](resources/memory/long-term/mem0.md) · [Zep](resources/memory/long-term/zep.md) · [LangMem](resources/memory/long-term/langmem.md) · [Memory Comparison](resources/memory/long-term/comparison.md) · [Checkpointing](resources/memory/episodic/checkpointing.md) · [Self-Improving Agents](resources/memory/procedural/self-improving-agents.md) |
| [Tooling](resources/tooling/overview.md) | Design principles, reliability, categories | [Single Responsibility](resources/tooling/design-principles/single-responsibility.md) · [Idempotency](resources/tooling/design-principles/idempotency.md) · [Error Contracts](resources/tooling/design-principles/error-contracts.md) · [Circuit Breakers](resources/tooling/reliability/circuit-breakers.md) · [Code Execution](resources/tooling/categories/code-execution.md) · [Browser Tools](resources/tooling/categories/browser-tools.md) |
| [RAG](resources/rag/overview.md) | Pipeline, chunking, search, reranking | [Chunking](resources/rag/chunking.md) · [Hybrid Search](resources/rag/hybrid-search.md) · [Reranking](resources/rag/reranking.md) · [Query Rewriting](resources/rag/query-rewriting.md) · [Failure Modes](resources/rag/failure-modes.md) |
| [Context Engineering](resources/context-engineering/overview.md) | Window mgmt, caching, injection, versioning | [Prompt Caching](resources/context-engineering/prompt-caching.md) · [Dynamic Injection](resources/context-engineering/dynamic-injection.md) · [Version Control](resources/context-engineering/version-control.md) |
| [Safety](resources/safety/overview.md) | Guardrails, injection defense, sandbox, governance | [Direct Injection](resources/safety/prompt-injection/direct.md) · [Indirect Injection](resources/safety/prompt-injection/indirect.md) · [Defense Architecture](resources/safety/prompt-injection/defense.md) · [NeMo Guardrails](resources/safety/guardrails/nemo-guardrails.md) · [E2B Sandbox](resources/safety/sandbox/e2b.md) · [EU AI Act](resources/safety/governance/eu-ai-act.md) |
| [Observability](resources/observability/overview.md) | Tracing, platforms, metrics, alerting | [Agent Traces](resources/observability/tracing/agent-traces.md) · [Langfuse](resources/observability/platforms/langfuse.md) · [LangSmith](resources/observability/platforms/langsmith.md) · [OpenTelemetry](resources/observability/platforms/opentelemetry.md) · [Platform Comparison](resources/observability/platforms/comparison.md) · [What to Measure](resources/observability/metrics/what-to-measure.md) |
| [Evaluation](resources/evaluation/overview.md) | RAGAS, LLM-as-Judge, trajectory, CI gates | [RAGAS](resources/evaluation/ragas.md) · [LLM-as-Judge](resources/evaluation/llm-as-judge.md) · [Trajectory Eval](resources/evaluation/trajectory-eval.md) · [CI Eval Gates](resources/evaluation/ci-eval-gates.md) · [Red Teaming](resources/evaluation/red-teaming.md) |
| [Cost Engineering](resources/cost-engineering/overview.md) | Token budgets, routing, caching, real numbers | [Token Budgets](resources/cost-engineering/token-budgets.md) · [Model Routing](resources/cost-engineering/model-routing.md) · [Semantic Caching](resources/cost-engineering/semantic-caching.md) · [Real-World Numbers](resources/cost-engineering/real-world-numbers.md) |
| [Failure Modes](resources/failure-modes/overview.md) | MAST taxonomy, mitigation playbook | [MAST Taxonomy](resources/failure-modes/mast-taxonomy.md) · [Mitigation Playbook](resources/failure-modes/mitigation-playbook.md) |
| [Scalability](resources/scalability/overview.md) | Horizontal scaling, multi-tenancy, async | [Horizontal Scaling](resources/scalability/horizontal-scaling.md) · [Multi-Tenancy](resources/scalability/multi-tenancy.md) · [Async Orchestration](resources/scalability/async-orchestration.md) |
| [Security](resources/security/overview.md) | Zero-trust, credentials, isolation | [Zero-Trust Agents](resources/security/zero-trust-agents.md) · [Credential Management](resources/security/credential-management.md) |
| [Deployment](resources/deployment/overview.md) | Containers, K8s, model serving, CI/CD | [Containerization](resources/deployment/containerization.md) · [Model Serving](resources/deployment/model-serving.md) · [Health Checks](resources/deployment/health-checks.md) · [CI/CD](resources/deployment/ci-cd.md) |
| [Protocols](resources/protocols/mcp.md) | MCP, A2A | [MCP](resources/protocols/mcp.md) · [A2A](resources/protocols/a2a.md) · [MCP vs A2A](resources/protocols/mcp-vs-a2a.md) |
| [HLD](resources/hld/overview.md) | Reference architecture, interview template | [Reference Architecture](resources/hld/reference-architecture.md) · [Interview Template](resources/hld/interview-template.md) |
| [LLD](resources/lld/overview.md) | State machines, APIs, data models | [State Machine Design](resources/lld/state-machine-design.md) · [API Contracts](resources/lld/api-contracts.md) · [Data Models](resources/lld/data-models.md) |
| [HITL](resources/hitl/overview.md) | Approval gates, escalation, confidence | [Approval Gates](resources/hitl/approval-gates.md) · [Escalation Policies](resources/hitl/escalation-policies.md) · [Confidence Thresholds](resources/hitl/confidence-thresholds.md) |

---

## Motivation

**Why learn agentic AI system design?**

Agentic AI job postings grew **280% year-over-year** in 2026, reaching ~90,000 US listings ([Stanford AI Index 2026](https://aiindex.stanford.edu/report/)). 40% of enterprise applications are projected to integrate task-specific AI agents by end of 2026 ([Gartner](https://www.gartner.com/)). The salary band for Agentic AI Engineers is **$185K–$320K** base in the US.

Yet **over 40% of multi-agent AI projects fail** ([MAST taxonomy, NeurIPS 2025](https://arxiv.org/abs/2503.13657)). The gap isn't in building demos — it's in building systems that work reliably at scale.

**How is this different from traditional system design?**

| Traditional System Design | Agentic AI System Design |
|---|---|
| Deterministic request-response | Non-deterministic reasoning loops |
| Fixed API contracts | Dynamic tool selection by LLM |
| Stateless microservices | Stateful agent sessions with memory |
| Horizontal scaling | Token budget + cost management |
| Error codes | Hallucination detection + guardrails |
| Load balancers | Agent orchestrators |
| Caching (Redis) | Semantic caching + prompt caching |
| Health checks | Agent behavior monitoring + eval |

**Who is this for?**

- Backend/software engineers transitioning to AI engineering
- AI engineers preparing for system design interviews
- Engineers building production agentic systems
- Architects evaluating agent architectures for their organization

---

## Study Guide

How to use this resource based on how much time you have:

| Timeline | What to Focus On | Sections |
|---|---|---|
| **1 week** (interview prep) | Patterns + tradeoffs + 2 case studies | [Start Here](#agentic-ai-topics-start-here), [Agent Patterns](#agent-architecture-patterns), [Memory](#memory-systems), [Interview Approach](#how-to-approach-an-agentic-system-design-interview), any 2 [Solutions](#agentic-system-design-interview-questions-with-solutions) |
| **2 weeks** (solid foundation) | Above + production concerns | Add [Tool Architecture](#tool-architecture), [RAG](#rag-retrieval-augmented-generation), [Cost Engineering](#cost-engineering), [Failure Modes](#failure-modes--mitigation), [Observability](#observability--monitoring) |
| **1 month** (comprehensive) | Everything + all case studies | All sections + all 10 [Solutions](#agentic-system-design-interview-questions-with-solutions) + [Appendix](#appendix) |
| **Ongoing reference** | Use as needed for architecture decisions | Bookmark specific sections, reference during design reviews |

---

## How to Approach an Agentic System Design Interview

The interview format at Anthropic, Cohere, Mistral, and FAANG for agentic roles follows a **4-step framework** (45-60 minutes):

### Step 1: Clarify requirements and constraints

> Don't jump to architecture. Spend 5-10 minutes here.

- **What is the agent supposed to do?** (Functional requirements)
- **What can it NOT do?** (Scope boundaries)
- **Who is the user?** (End-user, internal tool, API consumer)
- **What are the latency expectations?** (Seconds? Minutes? Async?)
- **What is the accuracy bar?** (Can it be wrong 5% of the time? 0.1%?)
- **Does it need human approval for any actions?** (Financial, production, customer-facing)
- **What's the scale?** (Users/day, requests/second, tokens/month)
- **What's the cost budget?** (Per-request, per-user, monthly)
- **What compliance constraints exist?** (PII, HIPAA, SOC2, GDPR)

**Calculate usage** (back-of-the-envelope):
```
Users: 10,000 DAU
Interactions/user/day: 5
Agent runs/day: 50,000
Agent runs/second: ~0.6
Tokens/run (avg): 4,000 (input) + 1,000 (output) = 5,000
Tokens/day: 250M
Cost/day at $3/M input + $15/M output:
  Input: 200M × $3/M = $600
  Output: 50M × $15/M = $750
  Total: ~$1,350/day → ~$40K/month
```

### Step 2: Create a high-level design

Draw the canonical agentic architecture:

```
User → API Gateway → Orchestrator Agent
                         ├── Planner
                         ├── Tool Executor → [Tools: Search, Code, APIs, DB]
                         ├── Memory Manager → [Short-term | Long-term | Episodic]
                         ├── LLM Router → [Primary LLM | Fast LLM | Specialist]
                         ├── Guardrails Engine
                         └── Response Assembler → User
                              ↕
                    Observability (Langfuse/LangSmith/OTel)
```

Explain **why each component exists** and what happens if you remove it.

### Step 3: Design core agent components

This is where you go deep. For each core component:

- **Agent pattern selection** — ReAct? Plan-and-Execute? Why?
- **Tool design** — What tools? What schemas? Idempotency?
- **Memory architecture** — What types? What store? Eviction strategy?
- **LLM routing** — Which model for which step? Cost implications?
- **Safety** — Where are the guardrails? What gets blocked?

Show pseudocode or API contracts where it helps.

### Step 4: Scale and harden the design

- **Identify bottlenecks** — What breaks at 10x scale?
- **Cost optimization** — Model routing, caching, token budgets
- **Failure modes** — What can go wrong? How do you detect it?
- **Observability** — What do you trace, measure, alert on?
- **HITL** — Where do humans need to be in the loop?

> **Pro tip from Cohere interviews:** They also run a **presentation round** — you present a past project and get cross-examined on design choices, failed approaches, and alternatives. Prepare a 15-min talk on your strongest project.

---

## Agentic System Design Interview Questions with Solutions

| Question | Difficulty |
|---|---|
| [Design an AI Customer Support Agent](solutions/customer_support_agent) | Medium |
| [Design an AI Coding Agent](solutions/ai_coding_agent) | Hard |
| [Design an Autonomous Research Assistant](solutions/research_assistant) | Medium |
| [Design an AI SDR/Sales Agent](solutions/ai_sdr_sales_agent) | Medium |
| [Design a DevOps Incident Agent](solutions/devops_incident_agent) | Hard |
| [Design an AI Personal Assistant](solutions/ai_personal_assistant) | Medium |
| [Design a Browser Automation Agent](solutions/browser_automation_agent) | Hard |
| [Design a Multi-Agent Enterprise Workflow System](solutions/enterprise_workflow_system) | Very Hard |
| [Design an AI Data Analyst](solutions/data_analyst_agent) | Medium |
| [Design an AI Knowledge Management Platform](solutions/knowledge_management_platform) | Hard |

---

## Agentic AI Topics — Quick Reference

> Each topic below is a brief overview. Click the deep-dive links for production-grade content with code, decision trees, and real-world patterns.

### Agents — Patterns, Multi-Agent, Frameworks, SDKs

**Core question:** Which agent pattern do I use? → [Pattern Decision Tree](resources/agents/patterns/decision-tree.md)

**Single vs Multi-Agent:** Default to single agent. Go multi-agent only when you've measured a single-agent ceiling AND tasks decompose into distinct domains AND you have coordination infrastructure. Real case: $8/query multi-agent → $0.40/query single-agent, <1% accuracy difference.

| Factor | Single Agent | Multi-Agent |
|---|---|---|
| **Complexity** | Low — one loop, one state | High — coordination, handoffs, state sync |
| **Cost** | Lower — fewer LLM calls | Higher — N agents × M calls each |
| **Latency** | Predictable | Variable — depends on coordination |
| **Debugging** | Straightforward | Hard — failures cascade across agents |
| **When to use** | Task fits one context window + toolset | Tasks span distinct expertise domains |
Deep dives: [Overview](resources/agents/overview.md) · [When to Go Multi-Agent](resources/agents/multi-agent/when-to-go-multi-agent.md)

**Agent patterns at a glance:**

| Pattern | Token Cost | Best For | Deep Dive |
|---|---|---|---|
| ReAct | Medium | Exploratory, dynamic tasks | [→ ReAct](resources/agents/patterns/react.md) |
| Plan-and-Execute | Low (70% savings) | Predictable multi-step | [→ P&E](resources/agents/patterns/plan-and-execute.md) |
| Reflexion | 2-3× overhead | Tasks with clear pass/fail | [→ Reflexion](resources/agents/patterns/reflexion.md) |
| ReWOO | Lowest | Independent parallel lookups | [→ ReWOO](resources/agents/patterns/rewoo.md) |

**Frameworks at a glance:**

| Framework | Best For | Deep Dive |
|---|---|---|
| LangGraph | Stateful workflows, enterprise default | [→ LangGraph](resources/agents/frameworks/langgraph.md) |
| CrewAI | Role-based multi-agent prototyping | [→ CrewAI](resources/agents/frameworks/crewai.md) |
| AutoGen | Conversation-based multi-agent | [→ AutoGen](resources/agents/frameworks/autogen.md) |
| LlamaIndex | Data-centric RAG + agents | [→ LlamaIndex](resources/agents/frameworks/llamaindex.md) |
| Pydantic AI | Type-safe, dependency injection | [→ Pydantic AI](resources/agents/frameworks/pydantic-ai.md) |
| DSPy | Programmatic prompt optimization | [→ DSPy](resources/agents/frameworks/dspy.md) |

Full comparison: [→ Framework Comparison](resources/agents/frameworks/comparison.md)

---

### Memory

**Core question:** Which memory type + store do I use? → [Memory Overview](resources/memory/overview.md)

| Memory Type | What It Stores | Deep Dive |
|---|---|---|
| Short-term | Current context window | [Context Window](resources/memory/short-term/context-window.md) · [Token Budgeting](resources/memory/short-term/token-budgeting.md) · [Compression](resources/memory/short-term/compression.md) |
| Long-term | Facts, preferences across sessions | [Mem0](resources/memory/long-term/mem0.md) · [Zep](resources/memory/long-term/zep.md) · [LangMem](resources/memory/long-term/langmem.md) · [Comparison](resources/memory/long-term/comparison.md) |
| Episodic | Past interactions, execution history | [Checkpointing](resources/memory/episodic/checkpointing.md) · [Execution Traces](resources/memory/episodic/execution-traces.md) |
| Procedural | Learned strategies, self-improving prompts | [Self-Improving Agents](resources/memory/procedural/self-improving-agents.md) |

**Quick benchmark:** Mem0 49% vs Zep 63.8% vs Hindsight 91.4% on LongMemEval.

---

### Tooling

**Core question:** How do I design reliable tools for agents? → [Tooling Overview](resources/tooling/overview.md)

Four non-negotiables: [Single Responsibility](resources/tooling/design-principles/single-responsibility.md) · [Idempotency](resources/tooling/design-principles/idempotency.md) · [Error Contracts](resources/tooling/design-principles/error-contracts.md) · [Schema Design](resources/tooling/design-principles/schema-design.md)

Reliability: [Circuit Breakers](resources/tooling/reliability/circuit-breakers.md) · [Retry Strategies](resources/tooling/reliability/retry-strategies.md)

Categories: [Code Execution](resources/tooling/categories/code-execution.md) · [Browser Tools](resources/tooling/categories/browser-tools.md) · [API Integrations](resources/tooling/categories/api-integrations.md)

---

### RAG (Retrieval-Augmented Generation)

**Core question:** How do I build a production RAG pipeline? → [RAG Overview](resources/rag/overview.md)

Pipeline: Query → [Chunking](resources/rag/chunking.md) → [Hybrid Search](resources/rag/hybrid-search.md) → [Reranking](resources/rag/reranking.md) → Generation → Citations

Also: [Query Rewriting](resources/rag/query-rewriting.md) · [RAG Failure Modes](resources/rag/failure-modes.md)

---

### Context Engineering

**Core question:** How do I manage what goes into the context window? → [Context Engineering Overview](resources/context-engineering/overview.md)

Key techniques: [Prompt Caching](resources/context-engineering/prompt-caching.md) (90% savings Anthropic, 50% OpenAI) · [Dynamic Injection](resources/context-engineering/dynamic-injection.md) · [Version Control](resources/context-engineering/version-control.md)

---

### Protocols

**MCP** connects agents to tools. **A2A** connects agents to other agents. They're complementary.

Deep dives: [MCP](resources/protocols/mcp.md) · [A2A](resources/protocols/a2a.md) · [MCP vs A2A](resources/protocols/mcp-vs-a2a.md)

---

### Safety & Guardrails

**Core question:** How do I prevent agents from doing harmful things? → [Safety Overview](resources/safety/overview.md)

4-layer taxonomy: Content → Evaluation → Sandbox → Action

Deep dives: [Direct Injection](resources/safety/prompt-injection/direct.md) · [Indirect Injection](resources/safety/prompt-injection/indirect.md) · [Defense Architecture](resources/safety/prompt-injection/defense.md) · [NeMo Guardrails](resources/safety/guardrails/nemo-guardrails.md) · [E2B Sandbox](resources/safety/sandbox/e2b.md) · [EU AI Act](resources/safety/governance/eu-ai-act.md)

---

### Observability & Monitoring

**Core question:** How do I understand what my agents are doing in production? → [Observability Overview](resources/observability/overview.md)

Platforms: [Langfuse](resources/observability/platforms/langfuse.md) · [LangSmith](resources/observability/platforms/langsmith.md) · [OpenTelemetry](resources/observability/platforms/opentelemetry.md) · [Comparison](resources/observability/platforms/comparison.md)

What to monitor: [Metrics](resources/observability/metrics/what-to-measure.md) · [Alert Design](resources/observability/alerting/alert-design.md) · [Agent Traces](resources/observability/tracing/agent-traces.md)

---

### Evaluation

**Core question:** How do I know if my agent is working? → [Evaluation Overview](resources/evaluation/overview.md)

Deep dives: [RAGAS](resources/evaluation/ragas.md) · [LLM-as-Judge](resources/evaluation/llm-as-judge.md) · [Trajectory Eval](resources/evaluation/trajectory-eval.md) · [CI Eval Gates](resources/evaluation/ci-eval-gates.md) · [Red Teaming](resources/evaluation/red-teaming.md)

---

### Cost Engineering

**Core question:** How do I control agent costs at scale? → [Cost Overview](resources/cost-engineering/overview.md)

Key insight: Token costs = 70-90% of total agent cost. Teams underestimate by 5-10×.

Deep dives: [Token Budgets](resources/cost-engineering/token-budgets.md) · [Model Routing](resources/cost-engineering/model-routing.md) · [Semantic Caching](resources/cost-engineering/semantic-caching.md) · [Real-World Numbers](resources/cost-engineering/real-world-numbers.md)

---

### Human-in-the-Loop (HITL)

**Core question:** Where do humans need to approve agent actions? → [HITL Overview](resources/hitl/overview.md)

Deep dives: [Approval Gates](resources/hitl/approval-gates.md) · [Escalation Policies](resources/hitl/escalation-policies.md) · [Confidence Thresholds](resources/hitl/confidence-thresholds.md)

---

### Failure Modes

**Core question:** What breaks in production agent systems? → [Failure Modes Overview](resources/failure-modes/overview.md)

Key stat: 41-86.7% failure rate across 7 frameworks, 1,642 traces (MAST, NeurIPS 2025).

Deep dives: [MAST Taxonomy (14 modes)](resources/failure-modes/mast-taxonomy.md) · [Mitigation Playbook](resources/failure-modes/mitigation-playbook.md)

---

### Scalability · Security · Deployment

| Topic | Core Question | Deep Dive |
|---|---|---|
| Scalability | How do I scale agents horizontally? | [Overview](resources/scalability/overview.md) · [Horizontal Scaling](resources/scalability/horizontal-scaling.md) · [Multi-Tenancy](resources/scalability/multi-tenancy.md) · [Async Orchestration](resources/scalability/async-orchestration.md) |
| Security | How do I secure agent systems? | [Overview](resources/security/overview.md) · [Zero-Trust Agents](resources/security/zero-trust-agents.md) · [Credential Management](resources/security/credential-management.md) |
| Deployment | How do I deploy agents to production? | [Overview](resources/deployment/overview.md) · [Containerization](resources/deployment/containerization.md) · [Model Serving](resources/deployment/model-serving.md) · [Health Checks](resources/deployment/health-checks.md) · [CI/CD](resources/deployment/ci-cd.md) |

---

### HLD · LLD

| Topic | What's Inside | Deep Dive |
|---|---|---|
| HLD | Reference architecture, service decomposition, interview template | [Overview](resources/hld/overview.md) · [Reference Architecture](resources/hld/reference-architecture.md) · [Interview Template](resources/hld/interview-template.md) |
| LLD | State machines, API contracts, data models, streaming | [Overview](resources/lld/overview.md) · [State Machine Design](resources/lld/state-machine-design.md) · [API Contracts](resources/lld/api-contracts.md) · [Data Models](resources/lld/data-models.md) |

---

## Appendix

### Token Reference Numbers

| Content | Approximate Tokens |
|---|---|
| 1 word (English) | ~1.3 tokens |
| 1 page of text | ~500 tokens |
| 1 code file (200 lines) | ~800 tokens |
| System prompt (typical agent) | 2,000-4,000 tokens |
| RAG context (5 chunks) | 2,500-5,000 tokens |
| Full conversation (10 turns) | 5,000-15,000 tokens |

### Latency Numbers for AI Systems

| Operation | Latency |
|---|---|
| LLM inference (first token, Claude Sonnet) | 500-800ms |
| LLM inference (first token, Haiku) | 200-400ms |
| Vector search (Qdrant, 1M vectors) | 5-20ms |
| Re-ranking (Cohere, 50 docs) | 80-120ms |
| Redis read | < 1ms |
| Postgres query | 1-10ms |
| External API call | 100-500ms |
| Full agent turn (tool use + reasoning) | 2-8 seconds |
| Multi-agent coordination (3 agents) | 5-20 seconds |

### Additional Interview Questions

**Architecture & Patterns:**
1. When would you NOT use an agent?
2. Compare LangGraph vs CrewAI vs AutoGen — when would you pick each?
3. How do you prevent infinite loops in an agent?
4. What's the difference between a workflow and an agent?
5. Design a fallback strategy for when your primary LLM provider goes down.

**Memory & Context:**
6. Design the memory architecture for a customer support agent that remembers user preferences across sessions.
7. How do you handle context window overflow in a long-running agent?
8. When would you use Mem0 vs Zep vs just pgvector?

**Safety & Reliability:**
9. How would you handle prompt injection in an agent that processes user-submitted documents?
10. What can go wrong in a multi-agent system that wouldn't go wrong in a single agent?
11. Walk me through how you'd build guardrails for an agent that executes code in production.
12. Your agent looks correct in testing but fails silently in production. How do you diagnose that?

**Cost & Scale:**
13. How do you control cost when your agent runs 50K sessions/day?
14. How do you decide between RAG, fine-tuning, and prompt engineering for a given problem?
15. Estimate the monthly cost of running a customer support agent for 10K daily users.

**Evaluation:**
16. How do you evaluate if your agent is working?
17. What's the difference between offline and online evaluation?
18. How would you build a CI eval gate that blocks bad prompt changes?

### Real-World Agent Architectures

| Company | System | What It Does | Source |
|---|---|---|---|
| **Telefónica** | Voice AI agents | Customer service at €0.35/interaction vs €3.50 human baseline | [10 Agentic AI Examples](https://www.warmly.ai/p/blog/agentic-ai-examples) |
| **Siemens + PepsiCo** | Digital Twin Composer | AI agents simulate supply chain changes with physics-level accuracy | [CES 2026](https://aimultiple.com/agentic-ai-trends) |
| **PwC** | Structured orchestration | 7x accuracy gains through structured multi-agent orchestration | [MAST analysis](https://arxiv.org/abs/2503.13657) |
| **Cursor** | AI coding agent | Autonomous multi-file agents (Cursor 3), $2B ARR with ~50 employees | [Cursor](https://cursor.com) |
| **Anthropic** | Claude Code | Agent SDK powering Claude Code — subagents, MCP, permissions | [Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview) |

### Company AI Engineering Blogs

| Company | Blog | Focus |
|---|---|---|
| Anthropic | [anthropic.com/research](https://www.anthropic.com/research) | Agent patterns, safety, alignment |
| LangChain | [blog.langchain.dev](https://blog.langchain.dev/) | LangGraph, LangSmith, production agents |
| Cohere | [cohere.com/blog](https://cohere.com/blog) | Enterprise RAG, agentic workflows |
| OpenAI | [openai.com/research](https://openai.com/research) | Model capabilities, agents, evals |
| Latent.Space | [latent.space](https://www.latent.space/) | Highest-signal AI engineering newsletter |

### Papers Worth Reading

| Paper | URL | Why It Matters | Read Time |
|---|---|---|---|
| Anthropic: Building Effective Agents | [anthropic.com](https://www.anthropic.com/research/building-effective-agents) | The 5 production patterns. Start here. | 30 min |
| ReAct: Synergizing Reasoning and Acting | [arXiv](https://huggingface.co/papers/2210.03629) | The foundational tool-use paper | 45 min |
| MAST: Why Do Multi-Agent LLM Systems Fail? | [arXiv](https://arxiv.org/abs/2503.13657) | 14 failure modes from 1,642 traces | 45 min |
| Plan-then-Execute: Resilient LLM Agents | [arXiv](https://arxiv.org/pdf/2509.08646) | P-t-E architecture guide | 1h |
| Mem0: Production-Ready Long-Term Memory | [arXiv](https://arxiv.org/pdf/2504.19413) | 80% token reduction benchmark | 1h |
| EDDOps: Eval-Driven Development | [arXiv](https://arxiv.org/abs/2411.13768) | Formal framework for continuous eval | 1h |
| Securing AI Agents Against Prompt Injection | [arXiv](https://arxiv.org/pdf/2511.15759) | Defense architectures | 45 min |
| Agentic Design Patterns | [arXiv](https://arxiv.org/html/2601.19752) | GoF-format patterns for agents | 1h |
| Production-Grade Agentic AI Workflows | [arXiv](https://arxiv.org/pdf/2512.08769) | 9 production best practices | 1h |

---

## Credits

- [Anthropic](https://www.anthropic.com/) — Building Effective Agents, Claude Agent SDK
- [LangChain](https://www.langchain.com/) — LangGraph, LangSmith, LangChain
- [donnemartin/system-design-primer](https://github.com/donnemartin/system-design-primer) — the structure and format this resource is modeled after
- [MAST authors](https://arxiv.org/abs/2503.13657) — Mert Cemri, Melissa Z. Pan, Shuyi Yang et al.
- [Mem0](https://mem0.ai/) — memory benchmarks and research
- All sources linked throughout this document

---

## License

*Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)*

*Last updated: May 2026*
