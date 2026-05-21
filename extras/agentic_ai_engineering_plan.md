# Agentic AI Engineering — Complete Preparation Plan
> Backend engineer → AI Engineer / Agentic AI Engineer
> Research: aiengg.dev, Naukri/Glassdoor job scans, Anthropic/OpenAI/LangChain official docs, AI Career Lab 2026
> Last updated: May 21, 2026 (Phase 2.5 added: PyTorch, fine-tuning, vLLM, AutoGen, LlamaIndex — based on Material/Srijan Senior AI Engineer JD gap analysis)

---

## 1. What You Need vs. What You Can Skip

### Skip Entirely
| Topic | Why |
|---|---|
| Statistics / probability | Model-builder skill. You build on top of models. |
| Traditional ML (sklearn, TensorFlow) | Not in agentic AI JDs; PyTorch basics covered in Phase 2.5 |
| NLP fundamentals (POS, NER, tokenizers from scratch) | Conceptual only — 1 hour read max |
| Data science (pandas, EDA, matplotlib) | Data engineer skill, not AI engineer skill |
| Deep fine-tuning (full-parameter, RLHF, DPO) | Hands-on QLoRA/PEFT covered in Phase 2.5; skip full fine-tuning and RLHF |
| Jupyter notebooks as primary development environment | Exploration only; production = Python modules |

### Conceptual Only (1-2 hours each, no coding)
- **Transformer architecture + attention mechanisms**: how self-attention works, encoder vs decoder, why context windows have limits — [Andrej Karpathy: "Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) (required viewing, interview filter at AI labs)
- **Embeddings**: what they are, why cosine similarity works, trade-offs between embedding models
- **Tokenization**: affects context window and cost reasoning, BPE vs SentencePiece
- **Fine-tuning vs RAG vs prompting**: which to use when (architectural decision; hands-on fine-tuning in Phase 2.5)
- **Context engineering**: the discipline of curating, structuring, and version-controlling what goes into the context window — [Anthropic's context engineering practices](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)

---

## 2. Market Reality (India, May 2026)

- **22,000+** AI Engineer jobs in Bengaluru (Naukri)
- **3,500+** LangChain/LangGraph job listings (Naukri, April 2026)
- **6,000+** "agentic AI engineer" listings on Glassdoor India
- Active hirers: TCS, Infosys, Persistent, Deloitte, EY, Siemens, Flexday AI, CRED, PhonePe, Groww

### Role Titles to Target
| Title | What it means | Your fit |
|---|---|---|
| AI Engineer | LLM APIs + RAG + agent frameworks | Primary target |
| Agentic AI Engineer | Multi-agent systems, orchestration | Primary target |
| AI Platform Engineer | Infra for AI workloads, k8s, serving | Strong (backend bg) |
| LLM Engineer | Deep LLM integration, evals | Secondary target |
| AI Backend Engineer | FastAPI + agent backends + state mgmt | Strong fit |

### What JDs Actually Ask For (Ranked by Frequency)
**Non-negotiable (80%+ of JDs):**
1. LangGraph — most asked, go very deep
2. LangChain — ecosystem knowledge, RAG chains, tool use
3. LLM API integration — OpenAI, Anthropic, Gemini; tool calling lifecycle
4. RAG pipeline — chunking, embedding, vector DB, retrieval, re-ranking
5. FastAPI + Python async
6. Vector databases — Chroma, Qdrant, Pinecone, pgvector

**Senior differentiators (what separates you):**
7. Evals + LLM-as-a-Judge — "hardest skill to fake," most skipped
8. Observability — Langfuse or LangSmith tracing
9. Multi-agent patterns — supervisor/worker, parallel, sequential
10. MCP (Model Context Protocol) — becoming the USB-C of AI tooling
11. Agent state + memory — Redis checkpoints, entity stores
12. Streaming + SSE — real-time token-by-token output
13. Cost control — per-user token budgets, circuit breakers, LiteLLM routing
14. Fine-tuning (QLoRA/PEFT) — required by pharma/healthcare/enterprise JDs
15. Model serving (vLLM) — preferred qualification in senior roles
16. PyTorch basics — must-have in JDs that list "deep learning frameworks"

**Your existing backend skills that transfer directly:**
| Go/Backend Skill | AI Engineering Equivalent |
|---|---|
| Goroutines + channels | Python asyncio, concurrent tool calls |
| gRPC / HTTP service design | MCP server design, FastAPI agent endpoints |
| Kafka / event-driven | Agent task queues, async execution |
| Redis | Agent state checkpointing, short-term memory |
| OpenSearch / ES | Hybrid search (BM25 + vector), re-ranking |
| Postgres | pgvector, conversation history |
| k8s + Docker | Containerised agent services |
| Rate limiting, circuit breakers | LLM cost control, per-user token budgets |
| Multi-tenant payment systems | Agent memory isolation, RBAC |

---

## 3. Phase-by-Phase Learning Plan

### Phase 0 — Python Fast Track (2 weeks)
Only learn the delta from Go. Skip Python 101.

| Topic | Resources | Time |
|---|---|---|
| async/await + asyncio | [Real Python asyncio guide](https://realpython.com/async-io-python/) | 3 days |
| Type hints + Pydantic v2 | [Pydantic v2 docs](https://docs.pydantic.dev/latest/) | 2 days |
| FastAPI | [FastAPI official tutorial](https://fastapi.tiangolo.com/tutorial/) | 3 days |
| uv / Poetry | [uv docs](https://docs.astral.sh/uv/) | 1 day |
| pytest + httpx | [pytest-asyncio docs](https://pytest-asyncio.readthedocs.io/) | 2 days |

**GitHub repos:**
- [tiangolo/fastapi](https://github.com/tiangolo/fastapi) — official repo, read the examples

**Build:** a FastAPI "hello world" with async routes, Pydantic models, background tasks, SSE endpoint.

---

### Phase 1 — LLM APIs + Agent SDKs Deep (4 weeks)
Most important phase. Understand internals, not just API calls. Cover all three major provider SDKs.

| Topic | Resources | Time |
|---|---|---|
| OpenAI SDK (chat, tool calling, Responses API) | [OpenAI docs](https://platform.openai.com/docs) | 3 days |
| Anthropic SDK (messages, tool use, caching) | [Anthropic docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) | 3 days |
| Tool / function calling lifecycle (both providers) | [Anthropic: Implement Tool Use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use) | 2 days |
| **Claude Agent SDK** (subagents, permissions, tool loop) | [Claude Agent SDK docs](https://code.claude.com/docs/en/agent-sdk/overview) · [Anthropic Agent SDK GitHub](https://github.com/anthropics/claude-code/tree/main/packages/agent) | 2 days |
| **OpenAI Agents SDK** (handoffs, tool registration, guardrails) | [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) | 2 days |
| **Google ADK** (Agent Development Kit, multi-agent on GCP) | [Google ADK docs](https://google.github.io/adk-docs/) | 1 day |
| Structured outputs + instructor | [instructor docs](https://python.useinstructor.com/) | 2 days |
| Token economics | [OpenAI pricing page](https://openai.com/api/pricing/) + [Anthropic pricing](https://www.anthropic.com/pricing) | 1 day |
| LiteLLM unified routing | [LiteLLM docs](https://docs.litellm.ai/docs/) | 2 days |
| Prompt engineering | [Anthropic prompt engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) | 3 days |
| Streaming in FastAPI (SSE) | [FastAPI streaming](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse) | 2 days |
| TypeScript AI SDK awareness | [Vercel AI SDK docs](https://ai-sdk.dev/docs) | 1 day |

**GitHub repos:**
- [anthropics/anthropic-cookbook](https://github.com/anthropics/anthropic-cookbook) — tool use, MCP, streaming, caching examples
- [anthropics/claude-code](https://github.com/anthropics/claude-code) — Agent SDK source, reference for building agents
- [openai/openai-agents-python](https://github.com/openai/openai-agents-python) — OpenAI Agents SDK, handoff patterns
- [google/adk-python](https://github.com/google/adk-python) — Google Agent Development Kit
- [BerriAI/litellm](https://github.com/BerriAI/litellm) — unified LLM interface, fallback routing

**Build:** FastAPI service that calls Claude/OpenAI with tool use, streams the response via SSE, handles tool errors and retries. Then rebuild the same agent using Claude Agent SDK to understand the difference between raw API and SDK abstractions.

---

### Phase 2 — LangChain Deep + RAG + Retrieval Systems (3.5 weeks)
Your OpenSearch background = 1 week head start. LangChain is the ecosystem glue — learn it properly here before LangGraph.

| Topic | Resources | Time |
|---|---|---|
| **LangChain core** (chains, LCEL, runnables, callbacks) | [LangChain docs](https://python.langchain.com/docs/) · [LangChain Academy](https://academy.langchain.com/) | 3 days |
| **LangChain RAG chains** (document loaders, splitters, retrievers) | [LangChain RAG tutorial](https://python.langchain.com/docs/tutorials/rag/) | 2 days |
| Embeddings fundamentals | [OpenAI embeddings guide](https://platform.openai.com/docs/guides/embeddings) | 2 days |
| Vector DBs (Qdrant + pgvector + Chroma + Pinecone) | [Qdrant docs](https://qdrant.tech/documentation/) · [pgvector](https://github.com/pgvector/pgvector) | 3 days |
| Chunking strategies | [RAG_Techniques chunking notebooks](https://github.com/NirDiamant/rag_techniques) | 2 days |
| Hybrid search (BM25 + vector) | [Qdrant hybrid search](https://qdrant.tech/documentation/concepts/hybrid-queries/) | 2 days |
| Re-ranking | [Cohere Rerank docs](https://docs.cohere.com/docs/rerank-2) | 2 days |
| Query rewriting (HyDE, multi-query) | [RAG_Techniques notebooks](https://github.com/NirDiamant/rag_techniques) | 2 days |
| Multimodal RAG | [ColPali paper + code](https://github.com/illuin-tech/colpali) | 2 days |
| Memory: mem0 + Zep | [mem0 docs](https://docs.mem0.ai/) · [Zep docs](https://docs.getzep.com/) | 2 days |

**GitHub repos:**
- [langchain-ai/langchain](https://github.com/langchain-ai/langchain) — 105k+ stars, the ecosystem core
- [langchain-ai/learning-langchain](https://github.com/langchain-ai/learning-langchain) — O'Reilly book companion
- [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/rag_techniques) — 26.2k stars, one notebook per RAG technique
- [KazKozDev/production-rag](https://github.com/KazKozDev/production-rag) — production-quality, pluggable components
- [infiniflow/ragflow](https://github.com/infiniflow/ragflow) — 46k stars, full open-source RAG engine
- [HKUDS/RAG-Anything](https://github.com/HKUDS/RAG-Anything) — multimodal RAG
- [Danielskry/Awesome-RAG](https://github.com/Danielskry/Awesome-RAG) — curated index of all RAG patterns

**Build:** RAG pipeline over PDFs using LangChain — hybrid search (Qdrant + BM25) + cross-encoder re-ranking + FastAPI SSE + Langfuse tracing.

---

### Phase 2.5 — PyTorch, Fine-tuning & Model Serving (1.5 weeks)
Many senior AI engineer JDs (especially healthcare/pharma) require hands-on fine-tuning and PyTorch. This phase gives you enough depth to execute, not just discuss.

| Topic | Resources | Time |
|---|---|---|
| PyTorch fundamentals (tensors, model loading, inference) | [PyTorch 60-minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) | 2 days |
| Hugging Face Transformers (load, infer, tokenize) | [HF Transformers quickstart](https://huggingface.co/docs/transformers/quicktour) | 1 day |
| Fine-tuning with QLoRA/PEFT | [HF PEFT docs](https://huggingface.co/docs/peft) · [Unsloth](https://github.com/unslothai/unsloth) | 3 days |
| Quantization basics (GPTQ, AWQ, bitsandbytes) | [HF Quantization guide](https://huggingface.co/docs/transformers/quantization) | 1 day |
| vLLM local model serving | [vLLM quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) | 1 day |
| Ollama (local model running, dev workflow) | [Ollama docs](https://ollama.com/) · [GitHub](https://github.com/ollama/ollama) | 0.5 day |

**GitHub repos:**
- [unslothai/unsloth](https://github.com/unslothai/unsloth) — 2x faster fine-tuning, 60% less memory
- [huggingface/peft](https://github.com/huggingface/peft) — official PEFT/LoRA library
- [vllm-project/vllm](https://github.com/vllm-project/vllm) — high-throughput inference engine
- [ollama/ollama](https://github.com/ollama/ollama) — local model running, dev-friendly

**Why this matters for JDs:**
- "Execute fine-tuning and optimization tasks (Quantization, PEFT/LoRA)" — direct JD requirement
- "Proficiency with deep learning frameworks (PyTorch)" — must-have in many senior roles
- "vLLM, DeepSpeed, or Triton Inference Server" — preferred qualification
- LlamaIndex + AutoGen appear in JD framework lists alongside LangChain/LangGraph

**Build:** Fine-tune a small model (e.g., Mistral-7B or Llama-3.1-8B) on a domain-specific dataset using QLoRA via Unsloth. Serve it locally with vLLM. Compare inference quality vs. base model with a simple eval.

---

### Phase 3 — Agent Frameworks Deep Dive (5 weeks)
Goal: be hands-on strong in EVERY major agent framework — not framework-agnostic, but framework-fluent. This is your core differentiator.

#### 3A — LangGraph (deep, 2 weeks)
The enterprise default for stateful agent workflows. Go deepest here.

| Topic | Resources | Time |
|---|---|---|
| LangGraph fundamentals (state, nodes, edges) | [LangChain Academy: LangGraph Essentials](https://academy.langchain.com/collections) (free) | 4 days |
| State machines + conditional edges | [langchain-ai/langgraph-101](https://github.com/langchain-ai/langgraph-101) | 2 days |
| Checkpointers (Redis + Postgres) | [LangChain persistence docs](https://docs.langchain.com/oss/javascript/langgraph/persistence) | 2 days |
| Human-in-the-loop | [LangGraph HITL docs](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) | 2 days |
| Multi-agent: supervisor + hierarchical patterns | [LangGraph hierarchical teams](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/) | 3 days |
| Parallel fan-out via Send API + map-reduce | [LangGraph parallelism](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/) | 1 day |

#### 3B — CrewAI (solid, 1 week)
Leader for role-based multi-agent prototyping. Used in 85% of production agent systems alongside LangChain.

| Topic | Resources | Time |
|---|---|---|
| CrewAI fundamentals (Agents, Tasks, Crews) | [CrewAI docs](https://docs.crewai.com/) | 2 days |
| Crews + Flows (sequential, hierarchical) | [CrewAI Flows guide](https://docs.crewai.com/concepts/flows) | 2 days |
| CrewAI tool integration + custom tools | [CrewAI tools docs](https://docs.crewai.com/concepts/tools) | 1 day |

#### 3C — AutoGen (solid, 1 week)
Microsoft's agent conversation framework. Strong in enterprise JDs (BlackRock, Deloitte).

| Topic | Resources | Time |
|---|---|---|
| AutoGen core (ConversableAgent, GroupChat) | [AutoGen docs](https://microsoft.github.io/autogen/) | 2 days |
| Multi-agent conversation patterns | [AutoGen notebooks](https://microsoft.github.io/autogen/docs/notebooks/) | 2 days |
| AutoGen + tool use + code execution | [AutoGen code execution](https://microsoft.github.io/autogen/docs/topics/code-execution/) | 1 day |

#### 3D — LlamaIndex Agents (solid, 3 days)
The data framework alternative to LangChain. Shows up in Cohere, Mistral, enterprise JDs.

| Topic | Resources | Time |
|---|---|---|
| LlamaIndex agent framework (QueryEngine, ReAct agent) | [LlamaIndex agent docs](https://docs.llamaindex.ai/en/stable/understanding/agent/) | 2 days |
| LlamaIndex vs LangChain for RAG (compare approaches) | Build same RAG pipeline in both | 1 day |

#### 3E — Pydantic AI (solid, 3 days)
Type-safe agents with dependency injection. Growing fast (16.5k stars).

| Topic | Resources | Time |
|---|---|---|
| Pydantic AI agents (type-safe, DI model) | [Pydantic AI docs](https://ai.pydantic.dev/) | 2 days |
| Pydantic AI + MCP integration | [Pydantic AI MCP docs](https://ai.pydantic.dev/mcp/) | 1 day |

#### 3F — DSPy + LangFlow (awareness, 2 days)

| Topic | Resources | Time |
|---|---|---|
| **DSPy** — programmatic prompt optimization (compile vs. prompt) | [DSPy docs](https://dspy.ai/) · [GitHub](https://github.com/stanfordnlp/dspy) | 1 day |
| **LangFlow** — visual LangChain/LangGraph builder (no-code/low-code) | [LangFlow docs](https://docs.langflow.org/) · [GitHub](https://github.com/langflow-ai/langflow) | 1 day |

**GitHub repos (all frameworks):**
- [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) — 12k+ stars, official
- [langchain-ai/langgraph-101](https://github.com/langchain-ai/langgraph-101) — 101 + 201 notebooks
- [crewAIInc/crewAI](https://github.com/crewaiinc/crewai) — 30k stars, role-based agents
- [microsoft/autogen](https://github.com/microsoft/autogen) — 57.8k stars, conversation patterns
- [run-llama/llama_index](https://github.com/run-llama/llama_index) — 40k+ stars, data framework
- [pydantic/pydantic-ai](https://github.com/pydantic/pydantic-ai) — 16.5k stars, type-safe
- [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) — 22k+ stars, prompt programming
- [langflow-ai/langflow](https://github.com/langflow-ai/langflow) — 50k+ stars, visual builder
- [von-development/awesome-LangGraph](https://github.com/von-development/awesome-LangGraph) — curated ecosystem
- [dipanjanS/mastering-intelligent-agents-langgraph-workshop-dhs2025](https://github.com/dipanjanS/mastering-intelligent-agents-langgraph-workshop-dhs2025) — full workshop
- [martimfasantos/ai-agents-frameworks](https://github.com/martimfasantos/ai-agents-frameworks) — side-by-side comparison of 8 frameworks

**DeepLearning.AI courses (free):**
- [AI Agents in LangGraph](https://www.deeplearning.ai/courses/ai-agents-in-langgraph) — Harrison Chase (LangChain CEO)
- [Agentic AI](https://www.deeplearning.ai/courses/agentic-ai/) — Andrew Ng, 4 patterns in raw Python

**Build:** Supervisor agent + 3 specialist agents in LangGraph. Then rebuild the SAME system in CrewAI and AutoGen to deeply understand the tradeoffs. Redis checkpointing. Human approval gate. Streaming SSE. Eval harness.

---

### Phase 4 — Evals + Observability + Safety (3 weeks)
Most skipped. Most valued by interviewers. Gets you senior roles. Cover BOTH observability platforms and the full safety stack.

#### 4A — Evaluation Frameworks (1 week)

| Topic | Resources | Time |
|---|---|---|
| Eval fundamentals + EDDOps | [Anthropic: Agentic AI course](https://www.deeplearning.ai/courses/agentic-ai/) (free) | 2 days |
| RAGAS (faithfulness, precision, answer relevancy) | [RAGAS docs](https://docs.ragas.io/) + [GitHub](https://github.com/explodinggradients/ragas) | 2 days |
| LLM-as-a-Judge patterns | [LangSmith eval docs](https://docs.langchain.com/langsmith/evaluation) | 2 days |
| Trajectory eval (not just final output) | Build eval that scores agent step-by-step reasoning | 1 day |

#### 4B — Observability Platforms (1 week)
Know BOTH deeply. Langfuse = self-hostable OSS. LangSmith = deep LangChain integration.

| Topic | Resources | Time |
|---|---|---|
| **Langfuse** setup + LangGraph tracing | [Langfuse LangGraph cookbook](https://langfuse.com/guides/cookbook/integration_langgraph) | 2 days |
| Langfuse prompt management + versioning | [Langfuse prompt management](https://langfuse.com/docs/prompt-management/overview) | 1 day |
| **LangSmith** tracing + datasets + online eval | [LangSmith docs](https://docs.smith.langchain.com/) · [LangSmith cookbook](https://github.com/langchain-ai/langsmith-cookbook) | 2 days |
| LangSmith CI eval gates (eval in CI pipeline) | [LangSmith testing guide](https://docs.smith.langchain.com/evaluation) | 1 day |
| Arize Phoenix (OTel-native, bonus) | [Phoenix docs](https://docs.arize.com/phoenix) | 0.5 day |

#### 4C — Guardrails + Safety + Governance (1 week)
Now a first-class JD requirement at every senior level. "Embed security, guardrails, sandbox isolation, auditability into agent runtimes."

| Topic | Resources | Time |
|---|---|---|
| Guardrails AI (input/output validation) | [Guardrails AI docs](https://www.guardrailsai.com/docs) | 1.5 days |
| **NVIDIA NeMo Guardrails** (runtime safety) | [NeMo Guardrails docs](https://docs.nvidia.com/nemo/guardrails/) · [GitHub](https://github.com/NVIDIA/NeMo-Guardrails) | 1.5 days |
| Prompt injection defense (direct + indirect) | [Lakera blog](https://www.lakera.ai/blog/indirect-prompt-injection) · [arXiv: Securing AI Agents](https://arxiv.org/pdf/2511.15759) | 1 day |
| Sandbox isolation for code execution agents | [E2B docs](https://e2b.dev/docs) · [Docker sandbox patterns](https://docs.docker.com/engine/security/) | 1 day |
| Audit trails + governance awareness | NIST AI RMF overview, ISO/IEC 42001 awareness, agent decision logging | 1 day |

**GitHub repos:**
- [explodinggradients/ragas](https://github.com/explodinggradients/ragas) — 7k stars, RAG evaluation
- [langfuse/langfuse](https://github.com/langfuse/langfuse) — 11k stars, self-hostable observability
- [langchain-ai/langsmith-cookbook](https://github.com/langchain-ai/langsmith-cookbook) — tracing, eval, prompt bootstrapping
- [NVIDIA/NeMo-Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) — runtime safety rails
- [arize-ai/phoenix](https://github.com/arize-ai/phoenix) — OTel-native AI observability
- [e2b-dev/e2b](https://github.com/e2b-dev/e2b) — sandboxed code execution for AI agents
- [hparreao/Awesome-AI-Evaluation-Guide](https://github.com/hparreao/Awesome-AI-Evaluation-Guide) — all eval tools compared

**Key metrics to monitor per trace:**
- LLM duration, time-to-first-token
- Prompt tokens, cached tokens, completion tokens
- Estimated cost per request
- Tool call count, success/failure rate per tool name
- Faithfulness / groundedness score (for RAG)
- Finish reason (stop / length / content_filter / tool_calls)

**Build:** Add full eval harness + Langfuse tracing to your Phase 2 RAG pipeline. Capture trajectories, not just final outputs.

---

### Phase 5 — MCP (Model Context Protocol) (1 week)
Go background = unique differentiator. Very few candidates have this.

| Topic | Resources | Time |
|---|---|---|
| MCP protocol internals | [Anthropic MCP docs](https://docs.anthropic.com/en/docs/agents-and-tools/mcp) | 2 days |
| Building MCP servers in Python | [fastmcp docs](https://github.com/jlowin/fastmcp) | 2 days |
| Building MCP servers in Go | [mark3labs/mcp-go](https://github.com/mark3labs/mcp-go) | 2 days |
| MCP client integration | [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk) | 1 day |

**GitHub repos:**
- [mark3labs/mcp-go](https://github.com/mark3labs/mcp-go) — Go MCP SDK, stdio + SSE + HTTP transports
- [jlowin/fastmcp](https://github.com/jlowin/fastmcp) — Python MCP framework, 1M+ daily downloads
- [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk) — official Anthropic Python SDK for MCP

**Build:** Go MCP server exposing a real API (DB queries, internal tool) connected to a LangGraph agent.

---

### Phase 6 — Production Infrastructure (2 weeks)
You'll move fast here. Apply existing backend knowledge to AI.

| Topic | Resources | Time |
|---|---|---|
| Celery + Redis for async tasks | [MarkAICode: long-running AI jobs](https://markaicode.com/redis-celery-long-running-ai-jobs/) | 2 days |
| Streaming + SSE production | [FastAPI LLM deployment patterns](https://activewizards.com/blog/fastapi-for-llm-systems-production-langchain-template) | 1 day |
| Agent state persistence | [SitePoint: Redis vs Postgres for AI memory](https://www.sitepoint.com/state-management-for-long-running-agents-redis-vs-postgres/) | 2 days |
| LiteLLM production setup | [LiteLLM prod best practices](https://docs.litellm.ai/docs/proxy/prod) + [fallbacks](https://docs.litellm.ai/docs/proxy/reliability) | 1 day |
| Rate limiting (token-based) | [Portkey: rate limiting for LLM apps](https://portkey.ai/blog/rate-limiting-for-llm-applications/) | 1 day |
| Cost control + token budgets | [Token budget management guide](https://tianpan.co/blog/2025-11-11-managing-token-budgets-production-llm-systems) | 1 day |
| Multi-tenancy | [Multi-tenant AI infra that scales](https://medium.com/@vamshidhar.pandrapagada/how-to-deploy-multi-tenant-ai-agent-infrastructure-that-actually-scales-433f44515837) | 2 days |
| Health checks + graceful shutdown | [FastAPI health checks guide](https://medium.com/@bhagyarana80/fastapi-health-checks-and-timeouts-avoiding-zombie-containers-in-production-411a27c2a019) | 2 days |

**GitHub repos:**
- [wassim249/fastapi-langgraph-agent-production-ready-template](https://github.com/wassim249/fastapi-langgraph-agent-production-ready-template) — most complete production template: SSE streaming, mem0+pgvector, Langfuse, Prometheus+Grafana, JWT, rate limiting
- [JoshuaC215/agent-service-toolkit](https://github.com/JoshuaC215/agent-service-toolkit) — LangGraph + FastAPI + Docker end-to-end
- [vstorm-co/full-stack-fastapi-nextjs-llm-template](https://github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template) — full-stack with 5 agent frameworks, Celery, k8s, 75+ config options

---

### Phase 7 — Deployment (LangGraph + Docker + k8s) (1 week)
Your DevOps background handles this fast.

| Topic | Resources | Time |
|---|---|---|
| LangGraph standalone container | [Standalone container docs](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/cloud/deployment/standalone_container.md) | 1 day |
| LangGraph on k8s (Helm) | [Official Helm chart](https://github.com/langchain-ai/helm/blob/main/charts/langgraph-cloud/README.md) | 1 day |
| LangGraph Platform (managed) | [Platform GA blog](https://blog.langchain.com/langgraph-platform-ga/) | 1 day |
| FastAPI production deployment | [FastAPI production guide](https://craftyourstartup.com/cys-docs/fastapi-production-deployment/) | 1 day |
| CI/CD for AI | Prompt versioning in git, eval gates in CI, model rollbacks | 1 day |

---

### Phase 8 — Cloud AI Platforms (1 week, do last)
Learn enough to use, not to master. Learn on the job once hired.

| Topic | Resources | Time |
|---|---|---|
| AWS Bedrock Agents | [AWS Bedrock docs](https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html) | 2 days |
| GCP Vertex AI | [Vertex AI Agent Builder](https://cloud.google.com/vertex-ai/docs/generative-ai/agents/overview) | 2 days |
| Azure AI Foundry | [Azure AI Foundry docs](https://learn.microsoft.com/en-us/azure/ai-foundry/) | 2 days |

---

## 4. Timeline Summary

| Phase | Duration | Cumulative | Milestone |
|---|---|---|---|
| Phase 0 — Python fast track | 2 weeks | Week 2 | — |
| Phase 1 — LLM APIs + Agent SDKs | 4 weeks | Week 6 | Claude/OpenAI/Google SDKs done |
| Phase 2 — LangChain + RAG + retrieval | 3.5 weeks | Week 9.5 | Project 1 done |
| Phase 2.5 — PyTorch, fine-tuning & model serving | 1.5 weeks | Week 11 | Fine-tuned model + vLLM + Ollama |
| Phase 3 — Agent Frameworks Deep Dive | 5 weeks | Week 16 | LangGraph+CrewAI+AutoGen+LlamaIndex+PydanticAI+DSPy. **Start applying here** |
| Phase 4 — Evals + Observability + Safety | 3 weeks | Week 19 | Langfuse+LangSmith+RAGAS+Guardrails. Project 2 done |
| Phase 5 — MCP | 1 week | Week 20 | Project 3 done |
| Phase 6 — Production infra | 2 weeks | Week 22 | — |
| Phase 7 — Deployment | 1 week | Week 23 | — |
| Phase 8 — Cloud platforms | 1 week | Week 24 | Fully hireable |

**Start applying at Week 16. Don't wait for Week 24.**
**Total: ~24 weeks (6 months) — up from 17 weeks because you're going deep on every framework instead of picking one.**

---

## 5. GitHub Repos — Master Reference

### By Category

#### LangChain Core + LangGraph + LangFlow
| Repo | Stars | What It Teaches |
|---|---|---|
| [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | 105k+ | The ecosystem core — chains, LCEL, runnables, integrations |
| [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | 12k+ | Stateful agent workflows — state machines, persistence, deployment |
| [langchain-ai/langgraph-101](https://github.com/langchain-ai/langgraph-101) | — | 101 + 201 tracks, notebook-per-concept |
| [langflow-ai/langflow](https://github.com/langflow-ai/langflow) | 50k+ | Visual LangChain/LangGraph builder — no-code/low-code |
| [von-development/awesome-LangGraph](https://github.com/von-development/awesome-LangGraph) | — | Curated ecosystem index |
| [dipanjanS/mastering-intelligent-agents-langgraph-workshop-dhs2025](https://github.com/dipanjanS/mastering-intelligent-agents-langgraph-workshop-dhs2025) | — | Full workshop: fundamentals → deployment |
| [jkmaina/LangGraphProjects](https://github.com/jkmaina/LangGraphProjects) | — | 50+ real-world business agent implementations |

#### Agent SDKs (Provider-Native)
| Repo | Stars | What It Teaches |
|---|---|---|
| [anthropics/claude-code](https://github.com/anthropics/claude-code) | — | Claude Agent SDK — subagents, tools, permissions |
| [openai/openai-agents-python](https://github.com/openai/openai-agents-python) | — | OpenAI Agents SDK — handoffs, tool registration, guardrails |
| [google/adk-python](https://github.com/google/adk-python) | — | Google Agent Development Kit — multi-agent on GCP |

#### RAG
| Repo | Stars | What It Teaches |
|---|---|---|
| [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/rag_techniques) | 26.2k | One notebook per RAG technique, beginner to advanced |
| [KazKozDev/production-rag](https://github.com/KazKozDev/production-rag) | — | Production RAG: hybrid search, re-ranking, eval |
| [infiniflow/ragflow](https://github.com/infiniflow/ragflow) | 46k+ | Full open-source RAG engine |
| [HKUDS/RAG-Anything](https://github.com/HKUDS/RAG-Anything) | — | Multimodal RAG (text, images, tables, charts) |
| [Danielskry/Awesome-RAG](https://github.com/Danielskry/Awesome-RAG) | — | Curated index of all RAG patterns |

#### Fine-tuning & Model Serving
| Repo | Stars | What It Teaches |
|---|---|---|
| [unslothai/unsloth](https://github.com/unslothai/unsloth) | Notable | 2x faster QLoRA fine-tuning, 60% less memory |
| [huggingface/peft](https://github.com/huggingface/peft) | Notable | Official PEFT/LoRA library |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | 50k+ | High-throughput inference, PagedAttention, OpenAI-compatible API |
| [run-llama/llama_index](https://github.com/run-llama/llama_index) | 40k+ | Data framework for LLMs, RAG alternative to LangChain |

#### MCP
| Repo | Stars | What It Teaches |
|---|---|---|
| [mark3labs/mcp-go](https://github.com/mark3labs/mcp-go) | Notable | Go MCP SDK — tools, resources, auth, middleware |
| [jlowin/fastmcp](https://github.com/jlowin/fastmcp) | Notable | Python MCP — decorator-based, 1M+ daily downloads |
| [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk) | — | Official Anthropic Python SDK for MCP |

#### Multi-Agent
| Repo | Stars | What It Teaches |
|---|---|---|
| [pydantic/pydantic-ai](https://github.com/pydantic/pydantic-ai) | 16.5k | Type-safe agents, MCP + A2A, DI model |
| [crewAIInc/crewAI](https://github.com/crewaiinc/crewai) | 30k | Role-based agents, Crews + Flows |
| [microsoft/autogen](https://github.com/microsoft/autogen) | 57.8k | Agent conversation patterns (reference) |
| [martimfasantos/ai-agents-frameworks](https://github.com/martimfasantos/ai-agents-frameworks) | — | Side-by-side comparison of 8 frameworks |

#### DSPy + Prompt Optimization
| Repo | Stars | What It Teaches |
|---|---|---|
| [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) | 22k+ | Programmatic prompt optimization — compile vs. prompt |

#### Evals + Observability
| Repo | Stars | What It Teaches |
|---|---|---|
| [explodinggradients/ragas](https://github.com/explodinggradients/ragas) | 7k+ | RAG evaluation: faithfulness, precision, recall |
| [langfuse/langfuse](https://github.com/langfuse/langfuse) | 11k+ | Self-hostable LLM observability platform |
| [langchain-ai/langsmith-cookbook](https://github.com/langchain-ai/langsmith-cookbook) | — | LangSmith tracing, eval, prompt bootstrapping |
| [arize-ai/phoenix](https://github.com/arize-ai/phoenix) | Notable | OTel-native AI observability, self-hostable |
| [hparreao/Awesome-AI-Evaluation-Guide](https://github.com/hparreao/Awesome-AI-Evaluation-Guide) | — | All eval tools compared |

#### Safety + Guardrails
| Repo | Stars | What It Teaches |
|---|---|---|
| [NVIDIA/NeMo-Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) | Notable | Runtime safety rails for LLM apps |
| [e2b-dev/e2b](https://github.com/e2b-dev/e2b) | Notable | Sandboxed code execution for AI agents |
| [guardrails-ai/guardrails](https://github.com/guardrails-ai/guardrails) | Notable | Input/output validation for LLMs |

#### Production Templates
| Repo | Stars | What It Teaches |
|---|---|---|
| [wassim249/fastapi-langgraph-agent-production-ready-template](https://github.com/wassim249/fastapi-langgraph-agent-production-ready-template) | — | Most complete production LangGraph backend |
| [JoshuaC215/agent-service-toolkit](https://github.com/JoshuaC215/agent-service-toolkit) | — | LangGraph + FastAPI + Docker, complete stack |
| [vstorm-co/full-stack-fastapi-nextjs-llm-template](https://github.com/vstorm-co/full-stack-fastapi-nextjs-llm-template) | 614 | Full-stack, 5 agent frameworks, Celery, k8s |
| [NirDiamant/agents-towards-production](https://github.com/NirDiamant/agents-towards-production) | — | End-to-end production agent tutorials |

#### Official Cookbooks
| Repo | What It Teaches |
|---|---|
| [anthropics/anthropic-cookbook](https://github.com/anthropics/anthropic-cookbook) | Tool use, MCP, streaming, caching, agents |
| [langchain-ai/cookbooks](https://github.com/langchain-ai/cookbooks) | LangGraph + LangSmith patterns |
| [langchain-ai/learning-langchain](https://github.com/langchain-ai/learning-langchain) | O'Reilly book companion |
| [langchain-ai/langsmith-cookbook](https://github.com/langchain-ai/langsmith-cookbook) | Tracing, eval, prompt bootstrapping |

---

## 6. Courses (Free → Paid)

### Free — Start Here
| Course | URL | Duration | What It Covers |
|---|---|---|---|
| LangChain Academy | https://academy.langchain.com/collections | 4 courses, 2-6h each | LangGraph, LangSmith, agents — official |
| DeepLearning.AI: AI Agents in LangGraph | https://www.deeplearning.ai/courses/ai-agents-in-langgraph | ~4h | Build agent → rebuild in LangGraph, HITL, persistence |
| DeepLearning.AI: Agentic AI (Andrew Ng, Oct 2025) | https://www.deeplearning.ai/courses/agentic-ai/ | Self-paced | 4 patterns in raw Python: Reflection, Tool Use, Planning, Multi-Agent |
| Hugging Face: AI Agents Course | https://huggingface.co/learn/agents-course/en/unit0/introduction | ~6 weeks, 3-4h/week | smolagents, LangGraph, LlamaIndex, evals, observability — with certificate |
| DeepLearning.AI: Building Systems with ChatGPT API | https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/ | ~1h | Multi-step LLM pipelines, safety eval |

### Paid (High Signal)
| Course | URL | Cost | What It Covers |
|---|---|---|---|
| LLM Engineering, RAG & AI Agents Masterclass | https://www.udemy.com/course/become-an-llm-agentic-ai-engineer-14-day-bootcamp-2025/ | ~$15-20 | Full bootcamp — LLMs, RAG, agents, MCP, Docker, AWS |
| Agentic AI Engineering with LangChain & LangGraph | https://www.udemy.com/course/langchain/ | ~$15-20 | LangChain v1.2+, LangGraph, production agents, RAG |

---

## 7. Structured Roadmaps

| Roadmap | URL | Best For |
|---|---|---|
| roadmap.sh — AI Engineer | https://roadmap.sh/ai-engineer | Visual, step-by-step, 355k+ GitHub stars |
| roadmap.sh — AI Agents | https://roadmap.sh/ai-agents | Agent-specific path |
| DataQuest: How to Become AI Engineer in 2026 | https://www.dataquest.io/blog/ai-engineer-roadmap/ | Phase-by-phase, 8-12 month plan |
| Practical AI Agents Roadmap (Neural Maze) | https://theneuralmaze.substack.com/p/the-practical-ai-agents-roadmap | Practitioner-oriented with specific tools |
| The Ultimate AI Agents Roadmap 2025 (AI Corner) | https://www.the-ai-corner.com/p/the-ultimate-ai-agents-roadmap-2025 | Curated videos, repos, papers, courses |

---

## 8. Newsletters + YouTube to Follow

### Newsletters
| Newsletter | URL | Why Follow |
|---|---|---|
| Latent.Space | https://www.latent.space/ | Highest signal AI engineering newsletter. Andrej Karpathy: "best AI newsletter atm." Deep technical interviews |
| Chip Huyen's Blog | https://huyenchip.com/blog/ | AI systems in production, RAG, evaluation. Her "AI Engineering" book is #1 on O'Reilly |
| The Batch (DeepLearning.AI) | https://www.deeplearning.ai/the-batch | Weekly — Andrew Ng's commentary on what actually matters |
| Neural Maze | https://theneuralmaze.substack.com/ | Practical RAG + agents with real code |
| swyx.io | https://www.swyx.io/ | Originated "Rise of the AI Engineer" concept |

### YouTube Channels
| Channel | URL | What It Covers |
|---|---|---|
| Andrej Karpathy | https://www.youtube.com/@AndrejKarpathy | LLM internals — essential for understanding what you build on top of |
| LangChain Official | https://www.youtube.com/@LangChain | LangGraph + LangSmith tutorials |
| AssemblyAI | https://www.youtube.com/@AssemblyAI | RAG, function calling, eval methodology |
| Cole Medin | https://www.youtube.com/@ColeMedin | Agents, LangGraph, practical build-alongs |
| Latent Space | https://www.youtube.com/channel/UCxBcwypKK-W3GHd_RZ9FZrQ | Deep technical AI engineering interviews |

---

## 9. Senior-Level Architecture Decisions

### 9.1 Single Agent vs. Multi-Agent

**Default rule: single agent first. Earn multi-agent complexity with measurement.**

Move to multi-agent only when ALL three conditions are met:
1. Your single-agent baseline hits a measurable ceiling despite prompt + tool iteration
2. Tasks divide into demonstrably distinct domains where specialists outperform a generalist
3. You have coordination infra defined: state contracts, failure handling, audit trails, human escalation

**The cost warning:** A single-agent → two-agent refactor at one team: $8-12/query P95 18s → $0.40/query 3s. Less than 1% accuracy difference. Multi-agent complexity ate the budget.

**Sources:**
- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — required reading
- [OpenAI: A Practical Guide to Building AI Agents](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/)
- [Single-Agent vs Multi-Agent CTO Framework](https://www.codebridge.tech/articles/single-agent-vs-multi-agent-architecture-what-changes-in-reliability-cost-and-debuggability)

### 9.2 Agent Patterns — When to Use Which

| Pattern | Use When | Token Cost | Flexibility |
|---|---|---|---|
| **ReAct** (default) | Exploratory, dynamic, each step depends on previous | Medium | High |
| **Plan-and-Execute** | Predictable multi-step, lower cost matters | Lower | Low |
| **Reflexion** | Repeating same mistake with clear pass/fail criteria | 2-3x overhead | Medium |
| **ReWOO** | All lookups are independent (parallel) | Lowest | Zero |

**Decision rule:**
```
Steps independent? → ReWOO
Steps depend on prior results AND cost matters? → Plan-and-Execute
Steps depend on prior results AND self-correction needed? → Reflexion (max 2-3 retries)
Default → ReAct
```

**Reflexion caveat:** Single-agent Reflexion repeats earlier mistakes because the same model generates both output and critique. Hard limit retries to 2-3.

**Source:** [The 4 Single-Agent Patterns — The AI Engineer](https://theaiengineer.substack.com/p/the-4-single-agent-patterns?action=share)

### 9.3 Memory Architecture

| Memory Type | What It Stores | Tool |
|---|---|---|
| Working / Short-term | Current context window | In-context; context engineering |
| Episodic | Ordered event sequences, conversation history | Zep, raw DB |
| Semantic | Distilled facts, preferences, summaries | Mem0, vector DB |
| Procedural | Learned rules, action strategies | System prompts, fine-tuned models |

**Mem0 vs Zep decision:**
- **Mem0**: token efficiency matters (80% reduction), production personalization at scale, AWS validates it
- **Zep**: multi-hop relational reasoning needed, long-term conversation analysis — 63.8% vs 49% on LongMemEval

**Sources:** [Mem0 paper (arXiv:2504.19413)](https://arxiv.org/pdf/2504.19413) · [State of AI Agent Memory 2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026)

### 9.4 Tool Design Principles

**Four non-negotiables:**
1. **Single responsibility** — one tool does one thing. `manage_database(action, table, data)` is an anti-pattern
2. **Idempotency on all write operations** — LLMs retry; side effects must not multiply
3. **Structured error contracts for LLM self-correction** — return field-level errors + correct-format example, never raw exceptions
4. **Schemas as documentation** — write tool descriptions like docs for a junior dev: examples, edge cases, natural language boundaries

**Reliability stack per tool:**
```
- Pydantic v2 strict mode input validation
- Input sanitization / PII filter
- Idempotency key enforcement on writes
- Per-tool timeout (not global)
- Retry with exponential backoff + jitter (idempotent errors only, max 3 attempts)
- Per-tool circuit breaker (open after N failures in time window)
- Structured error response for LLM self-correction
- Audit log: request_id, tool name, args hash, result, latency
- Output schema validation before returning to agent
- Temperature: 0.0-0.2 for all tool-calling inference
```

### 9.5 Context Window Management

**The core problem:** Instruction fidelity degrades before capacity is hit. At 60-80% context fill, instruction fidelity drops. Place critical instructions at start or end (not middle) — LLMs exhibit a U-shaped performance curve.

**Four techniques (apply in order):**
1. **Context prioritization + recency windowing** — keep last 10-15 exchanges verbatim, summarize earlier
2. **RAG + semantic retrieval** — inject only relevant chunks, not everything
3. **Prompt compression** — 5-20x compression = 70-94% cost reduction
4. **External memory** — for workflows exceeding ~40 tool calls, use Mem0/Zep

**Sources:** [Context Engineering Guide — Mem0](https://mem0.ai/blog/context-engineering-ai-agents-guide) · [Weaviate context engineering](https://weaviate.io/blog/context-engineering)

### 9.6 MCP vs A2A — When to Use Which

| Protocol | Purpose | Use When |
|---|---|---|
| **MCP** | Connect agents to tools and data | Target is a database, API, traditional system |
| **A2A** | Connect agents to other agents | Target is an AI agent on a different framework |

These are complementary. Typical production stack: MCP provides the data layer (DB, APIs) + A2A provides agent coordination layer. Start with MCP. Add A2A when you genuinely need distributed agent intelligence.

**A2A reference:** [a2a-protocol.org](https://a2a-protocol.org/latest/specification/) · [GitHub: a2aproject/A2A](https://github.com/a2aproject/A2A)

### 9.7 Context Engineering (Emerging Discipline, 2026)

**"Context Engineer" is now an actual job title at frontier companies.** The lever that moves agent reliability most isn't a better prompt — it's curated, version-controlled context.

**Core techniques:**
1. **Context curation** — select what goes in; what stays out matters more than what's included
2. **Context structuring** — system instructions at start, user data in middle, critical constraints at end (U-shaped attention)
3. **Prompt caching** — Anthropic's cache_control, OpenAI's automatic caching; reduces cost 90% on repeated prefixes
4. **Version-controlled context** — prompts in git, not hardcoded; A/B test prompt versions via Langfuse/LangSmith
5. **Context compression** — summarize old turns, use RAG to inject only relevant chunks
6. **Dynamic context assembly** — per-request context built from memory + RAG + tool results + system prompt

**Sources:** [Context Engineering Guide — Mem0](https://mem0.ai/blog/context-engineering-ai-agents-guide) · [Weaviate context engineering](https://weaviate.io/blog/context-engineering) · [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)

---

## 10. Observability + Monitoring Stack

### Platform Recommendations
| Need | Best Tool |
|---|---|
| LangGraph tracing + prompt versioning | **Langfuse** (self-hostable, open-source) |
| Deep LangChain eval + CI datasets | **LangSmith** |
| OTel-native, self-hostable | **Arize Phoenix** |
| Infra metrics + LLM on one dashboard | **Grafana Cloud AI Observability** |
| RAG-specific evaluation | **RAGAS** |
| Multi-provider cost + latency proxy | **Helicone**, LiteLLM |

### Grafana + Prometheus Stack for AI Agents
```
Prometheus → scrape vLLM/TGI inference metrics + app instrumentation
Tempo → distributed trace storage (receives OTel spans from OpenLLMetry)
Loki → structured log aggregation
Grafana → correlate all three; cost panels using llm_agent_tokens_total × per_token_price
```

### What to Log Per Agent Step (Minimum)
```
- Trace ID + correlation ID (propagated across services)
- Agent name, step number, parent agent
- Input state (messages + memory snapshot)
- Tool calls: name, args, result, latency
- LLM call: model, prompt tokens, completion tokens, finish reason
- Output state changes
- Memory reads and writes (namespace + key)
- Errors with type: rate_limit / timeout / tool_failure / reasoning_failure
```

### Key Metrics to Alert On
```
Cost spikes:    P95 token counts (not median); per-feature cost; budget burn rate
Latency P99:    Sustained degradation (not short-lived spikes); use error budgets
Tool failures:  Error rate per tool name; finish_reason breakdowns
Quality:        Eval-driven alerts when faithfulness/relevance drops below threshold
Silent failures: Trace-level eval — spans return 200 OK while chain is wrong
```

### Key Resources
- [Langfuse LangGraph cookbook](https://langfuse.com/guides/cookbook/integration_langgraph)
- [OpenLLMetry — auto-instrumentation for all LLM providers](https://github.com/traceloop/openllmetry)
- [Arize Phoenix GitHub](https://github.com/arize-ai/phoenix)
- [Grafana Cloud AI Observability](https://grafana.com/docs/grafana-cloud/monitor-applications/ai-observability/)
- [AI Engineer's Guide to LLM OTel](https://agenta.ai/blog/the-ai-engineer-s-guide-to-llm-observability-with-opentelemetry)

---

## 11. Production Deployment Reference

### Self-hosted Model Serving (when using open-source models)
| Tool | Use Case | Key Benefit |
|---|---|---|
| **vLLM** | High-throughput serving of open models (Llama, Mistral, etc.) | PagedAttention, continuous batching, OpenAI-compatible API |
| **Triton Inference Server** | Multi-framework serving (PyTorch, TensorFlow, ONNX) | NVIDIA ecosystem, dynamic batching |
| **TGI (Text Generation Inference)** | HuggingFace model serving | Simplest setup, good for prototyping |

**vLLM production config:**
```
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 2 \     # multi-GPU
  --max-model-len 8192 \         # context window cap
  --gpu-memory-utilization 0.9 \ # leave 10% for overhead
  --enable-prefix-caching         # KV cache reuse for shared prefixes
```

### LangGraph Deployment Options (in order of complexity)
1. **LangGraph Platform (Cloud SaaS)** — fastest, pay $0.001/node execution, built-in Postgres + Redis, no infra to manage. [Platform GA blog](https://blog.langchain.com/langgraph-platform-ga/)
2. **Standalone Docker container** — `langgraph build` → Docker image, bring your own Postgres + Redis. [Standalone container docs](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/cloud/deployment/standalone_container.md)
3. **Kubernetes (Helm)** — [Official Helm chart](https://github.com/langchain-ai/helm/blob/main/charts/langgraph-cloud/README.md), 3 replicas, HPA, external Postgres + Redis
4. **AWS ECS Fargate** — [Production ECS template](https://ajay-arunachalam08.medium.com/creating-a-production-ready-aws-deployment-template-that-gets-your-langgraph-agentic-ai-workflow-c78fe42efa50)

### Health Checks + Graceful Shutdown Pattern
```python
# Three-state health model
GET /livez   → is process alive? (never check external deps here)
GET /readyz  → is process ready to serve? (check DB, Redis, LLM provider)
GET /healthz → full health with latency details

# SIGTERM handling: flip readFlag → return 503 from /readyz → 
# Kubernetes drains traffic → receive SIGTERM → shutdown
```

### FastAPI + Gunicorn + Uvicorn Production Config
```
gunicorn -k uvicorn.workers.UvicornWorker \
  --workers $(nproc) \
  --max-requests 1000 \           # restart worker after N requests (memory leak mitigation)
  --graceful-timeout 120 \        # for long-running agent requests
  --timeout 300                   # agent tool call chains can take minutes
```

**Source:** [FastAPI graceful shutdown — SIGTERM handling](https://logicandlegacy.blogspot.com/2026/05/fastapi-graceful-shutdown-handling.html)

### LiteLLM Production Setup
```
Key configs before going live:
- LITELLM_SALT_KEY for encrypted key storage
- --max_requests_before_restart for memory leak mitigation
- Redis for distributed rate limit coordination across pods
- readOnlyRootFilesystem in Kubernetes
- Fallback order: primary → secondary → cached snapshot → human escalation
```

---

## 12. Production Failure Modes (Senior-Level Awareness)

Based on Berkeley's MAST taxonomy — 14 system-level failure modes in multi-agent pipelines, none detectable at the individual agent level:

| Failure Mode | What Happens | Mitigation |
|---|---|---|
| Context degradation | Instruction fidelity loss at 60-80% fill | Context engineering, external memory |
| Temperature schema drift | Tool call schema inconsistency above 0.2 temp | Set temp 0.0-0.2 for tool-calling |
| Authentication rot | OAuth tokens expire, API keys rotate silently | Automated token refresh monitoring |
| Schema drift | Tool schema version upgrades break arg generation | Pin schema versions, validate on deploy |
| Coordination failures | Handoff errors between agents amplify 17.2x | Typed state contracts, strict schema handoffs |
| Memory drift | Semantic hallucination accumulates across sessions | Ground-truth episodic store in external memory |
| Tool failure cascade | One tool failing halts multi-step workflow | Per-tool circuit breakers, fallback ladders |
| Overconfidence | Model declares success despite failure | Output validation + success criteria checks |
| Prompt injection via tools | Malicious content in tool output hijacks agent | Trust boundary enforcement, least privilege tools |

**Key interview question:** *"What can go wrong in a multi-agent system that wouldn't go wrong in a single agent?"*
Answer areas: hallucination propagation, circular sub-agent loops, state drift, prompt injection via tool results, cost explosion from recursive calls, coordination failures amplify individual errors.

**Sources:**
- [Why AI Agents Keep Failing in Production](https://medium.com/data-science-collective/why-ai-agents-keep-failing-in-production-cdd335b22219)
- [Berkeley MAST failure taxonomy (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6572478)

---

## 13. Production Readiness Checklist

Before shipping any agent to production:
```
Foundation
  □ Eval harness with trajectory capture (not just final output scoring)
  □ Success criteria defined as measurable metrics before build starts

Tool Layer
  □ Idempotency keys on all write operations
  □ Per-tool circuit breakers, timeouts, fallback ladders
  □ Structured error contracts (not raw exceptions) for LLM self-correction
  □ Schema validation on tool inputs AND outputs
  □ Temperature 0.0-0.2 for tool-calling inference

Agent Infrastructure
  □ Postgres-backed checkpointer (or equivalent) for resumability
  □ Context window management strategy defined
  □ External memory for sessions exceeding ~40 tool calls

Observability
  □ LangSmith / Langfuse / OTel tracing on every node transition
  □ Per-tool metrics: success rate, p50/p95 latency, error taxonomy
  □ Cost attribution per tenant/feature/user

Security
  □ Trust boundary enforcement (system instructions vs. external data separated)
  □ Least privilege per tool (short-lived credentials per session)
  □ Prompt injection testing before launch

Deployment
  □ Health checks: /livez + /readyz separated
  □ SIGTERM graceful shutdown (drain active agent sessions)
  □ Rate limiting: token-based (not just request-based)

Continuous
  □ Online eval signals feeding back to offline regression suite
  □ Failure taxonomy tracked (not just "it failed")
  □ Prompt versioning in git, eval gates in CI pipeline
```

---

## 14. Papers Worth Reading (Engineering Focus)

| Paper | URL | Why It Matters | Read Time |
|---|---|---|---|
| Anthropic: Building Effective Agents | https://www.anthropic.com/research/building-effective-agents | The 5 production patterns. Highest ROI document in this list. | 30 min |
| ReAct: Synergizing Reasoning and Acting | https://huggingface.co/papers/2210.03629 | The foundational tool-use paper. Every agent framework implements this. | 45 min |
| Mem0: Production-Ready AI Agents with Long-Term Memory | https://arxiv.org/pdf/2504.19413 | 80% token reduction, 91% latency improvement benchmark. | 1h |
| EDDOps: Eval-Driven Development and Ops | https://arxiv.org/abs/2411.13768 | Formal framework for continuous eval in production. | 1h |
| Agentic Design Patterns: System-Theoretic Framework | https://arxiv.org/html/2601.19752 | GoF-format patterns: Reflection, Tool Use, Planning, Multi-Agent | 1h |
| MAST: Failure Taxonomy for Multi-Agent Systems | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6572478 | 14 system-level failure modes. None detectable at single agent level. | 45 min |
| Securing AI Agents Against Prompt Injection | https://arxiv.org/pdf/2511.15759 | Defense architectures for production injection attacks | 45 min |
| 2025 AI Engineering Reading List (Latent.Space) | https://www.latent.space/p/2025-papers | One curated paper/week for all of 2025. Use as ongoing reading. | Ongoing |

---

## 15. Three Portfolio Projects

### Project 1 — Production RAG API (after Phase 2)
**Stack:** FastAPI + Qdrant (hybrid BM25 + vector) + cross-encoder re-ranking + Langfuse tracing + RAGAS eval suite + Docker

**What it must have:**
- PDF/doc ingestion pipeline with semantic chunking
- Hybrid search (BM25 + dense vector) + RRF fusion
- Cross-encoder re-ranking (Cohere or BGE)
- SSE streaming responses
- Langfuse tracing on every request
- RAGAS eval suite: faithfulness, context precision, answer relevancy

**What it signals:** You understand production RAG, not just toy demos. You know evals.

---

### Project 2 — Multi-Agent LangGraph System (after Phase 3 + 4)
**Stack:** LangGraph + FastAPI + Redis checkpointing + Langfuse + RAGAS + SSE

**What it must have:**
- Supervisor agent + 3 specialist agents (e.g., researcher, writer, reviewer)
- Redis checkpointing — resumable across process restarts
- Human-in-the-loop approval gate before final output
- Streaming output via SSE
- Eval harness: LLM-as-a-Judge scoring on trajectory (not just final output)
- Per-tool circuit breakers + structured error contracts
- Langfuse tracing with cost attribution

**What it signals:** You can architect complex agentic workflows. You think about reliability, not just happy paths.

---

### Project 3 — Go MCP Server (after Phase 5)
**Stack:** Go (mark3labs/mcp-go) + real API (DB queries or internal tool) + connected to LangGraph agent or Claude Desktop

**What it must have:**
- Auth, error handling, retry logic
- Tool + resource primitives
- Connected to and tested with a real LangGraph agent

**What it signals:** Go + AI = unique profile. Very few candidates have this. Direct differentiator.

---

## 16. Interview Prep

### What Gets You Screened In
- GitHub with these 3 projects (not forked tutorials)
- Can describe a **real failure** in an agent system you built
- Know LangGraph state machine model, not just surface API
- Can answer: "How do you evaluate if your agent is working?"

### What Gets You Screened Out
- "I know LangChain" with no project proof
- Cannot explain token economics or cost control
- No understanding of evals

### Senior-Level Questions Interviewers Ask
1. *"What can go wrong in a multi-agent system that wouldn't go wrong in a single agent?"*
2. *"How do you decide between RAG, fine-tuning, and prompt engineering for a given problem?"*
3. *"Your agent looks correct in testing but fails silently in production. How do you diagnose that?"*
4. *"Design the memory architecture for a customer support agent that needs to remember user preferences across sessions."*
5. *"How would you handle prompt injection in an agent that processes user-submitted documents?"*
6. *"When would you NOT use an agent? Give me a real example."*
7. *"Compare LangGraph vs CrewAI vs AutoGen — when would you pick each?"*
8. *"Walk me through how you'd build guardrails for an agent that executes code in production."*

### Agentic System Design Interview Prep (3 days dedicated practice)
Anthropic, Cohere, Mistral, and FAANG now run agent-specific system design rounds — completely different from traditional backend system design.

**Format:** 45-60 min whiteboard. You design an agent system, interviewer pushes on tradeoffs.

**Practice problems (design each end-to-end):**
1. Design a multi-agent customer support system handling refunds, FAQs, and escalations
2. Design an agentic code review system that reads PRs, runs tests, and suggests fixes
3. Design a RAG-powered legal document analyst for a compliance team
4. Design an autonomous data pipeline agent that monitors, diagnoses, and fixes data quality issues
5. Design a multi-agent research assistant that searches, summarizes, and fact-checks

**For each design, cover:**
- Single agent vs multi-agent decision (and why)
- Agent pattern choice (ReAct / Plan-and-Execute / Reflexion)
- Tool design (what tools, idempotency, error contracts)
- Memory architecture (short-term, long-term, episodic)
- Context engineering strategy
- Guardrails and safety boundaries
- Cost estimation (tokens per request, budget per user)
- Observability setup (what traces, what metrics, what alerts)
- Human-in-the-loop: where and why
- Failure modes and mitigations

**Cohere-specific:** They also have a **presentation round** — you present a past project and get cross-examined on design choices, failed approaches, and alternatives you considered. Prepare a 15-min talk on your best portfolio project.

**Resources:**
- [Complete Agentic AI System Design Interview Guide 2026](https://atul4u.medium.com/the-complete-agentic-ai-system-design-interview-guide-2026-f95d0cfeb7cf)
- [DataCamp: Top 30 Agentic AI Interview Questions](https://www.datacamp.com/blog/agentic-ai-interview-questions)
- [Every AI Engineer Interview Question from 100+ Real Interviews](https://adilshamim8.medium.com/every-ai-engineer-interview-question-you-need-to-know-in-2026-from-100-real-interviews-b5b7ae4b961a)
- [Anthropic Technical Interview Guide 2026](https://jobright.ai/blog/anthropic-technical-interview-questions-complete-guide-2026/)

---

*Last updated: May 21, 2026 | Based on: aiengg.dev (Gaurav Sen), Naukri/Glassdoor job scan, Anthropic/OpenAI/Cohere/Mistral/LangChain official docs and JDs, AI Career Lab 2026, Berkeley MAST taxonomy, Material/Srijan JD gap analysis, HN "Who is Hiring" May 2026*
