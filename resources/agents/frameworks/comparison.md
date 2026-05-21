# Framework Comparison

> This is the "which framework should I use?" reference. LangGraph dominates enterprise (239/1,056 listings), but the right choice depends on your task, team, and constraints.

## What It Is

A comprehensive side-by-side comparison of the six major agent frameworks plus three vendor SDKs, with data-driven recommendations for different use cases.

## Master Comparison Table

| Dimension | LangGraph | CrewAI | AutoGen | LlamaIndex | Pydantic AI | DSPy |
|-----------|-----------|--------|---------|------------|-------------|------|
| **Core model** | State machine (graph) | Role-based teams | Conversable agents | Data/RAG-first | Type-safe agents | Compiled prompts |
| **Primary strength** | Flow control + persistence | Rapid prototyping | Multi-agent chat | Data retrieval | Type safety + DI | Prompt optimization |
| **Multi-agent** | Full (supervisor, hierarchical, fan-out) | Yes (crews, flows) | Yes (GroupChat) | Limited | Limited | No (single-call) |
| **Persistence** | PostgreSQL, Redis, SQLite | Limited | Limited | Limited | No built-in | No |
| **HITL** | Built-in (interrupt/resume) | Basic | Human proxy agent | No | No | No |
| **Streaming** | Full (token + node level) | Basic | Basic | Basic | Basic | No |
| **MCP support** | Via LangChain tools | No | No | No | First-class | No |
| **Type safety** | TypedDict state | Pydantic models | Dict-based | Pydantic models | Strongest | Signature-based |
| **Learning curve** | Medium-High | Low | Medium | Medium | Low | High |
| **Maturity** | Production-ready | Growing | Production (v0.4) | Production | Growing | Research-mature |
| **Language** | Python (JS beta) | Python | Python, C# | Python, TS | Python | Python |
| **Backing** | LangChain Inc. | CrewAI Inc. | Microsoft | LlamaIndex Inc. | Pydantic team | Stanford NLP |

## Job Market Data (2026)

From the LangChain AI Job Market Report, analyzing 1,056 AI engineering job listings:

```
Framework/Tool          Listings   % of Total   YoY Growth
----------------------------------------------------------
LangChain               362       34.3%         +12%
LangGraph                239       22.6%         +89%  <-- fastest growing
LlamaIndex               112       10.6%         +23%
CrewAI                    67        6.3%         +156% <-- highest % growth
AutoGen                   45        4.3%         +34%
Pydantic AI               28        2.7%         +340% <-- from very low base
DSPy                      19        1.8%         +58%

Vendor SDKs:
OpenAI SDK               287       27.2%         -5%
Anthropic SDK             89        8.4%         +67%
Google AI SDK             54        5.1%         +28%
```

### Key Insight

LangGraph appears in **66% of listings that mention LangChain**, indicating it has become the de facto execution layer for LangChain-based systems. CrewAI shows the highest percentage growth, suggesting rapid adoption for prototyping and mid-complexity use cases.

## Decision Guide

### By Use Case

| Use Case | Recommended | Runner-Up | Why |
|----------|-------------|-----------|-----|
| Enterprise agent with audit trail | LangGraph | AutoGen | Persistence, HITL, LangSmith |
| Rapid prototype / hackathon | CrewAI | Pydantic AI | Fastest to working demo |
| Multi-agent conversation | AutoGen | LangGraph | Native conversation model |
| RAG-heavy application | LlamaIndex | LangGraph | 160+ data connectors, sub-questions |
| Type-safe production agent | Pydantic AI | LangGraph | DI, validation, MCP |
| Prompt optimization pipeline | DSPy | None (unique) | Compiled prompts, model-portable |
| Complex workflow with cycles | LangGraph | None (unique) | Only framework with true cycles |
| Code generation + execution | AutoGen | LangGraph | Built-in Docker sandboxing |
| Microsoft/Azure ecosystem | AutoGen | LangGraph | Native Azure OpenAI integration |

### By Team Experience

| Team Background | Recommended | Why |
|----------------|-------------|-----|
| Backend engineers (Python) | LangGraph or Pydantic AI | Explicit, typed, testable |
| Data scientists | DSPy or LlamaIndex | Familiar paradigms (compile, index) |
| Product/startup team | CrewAI | Fastest time-to-value |
| Enterprise/compliance | LangGraph | Audit trail, HITL, persistence |
| ML engineers | DSPy + LangGraph | Optimize + orchestrate |

### By Scale

| Scale | Recommended | Why |
|-------|-------------|-----|
| Personal project | CrewAI or Pydantic AI | Simplest setup |
| Startup MVP | CrewAI or LangGraph | Fast but upgradeable |
| Production service | LangGraph | Persistence, monitoring, scaling |
| Enterprise platform | LangGraph + LlamaIndex | Full stack agent + RAG |

## Framework Combination Patterns

In production, frameworks are often combined:

### Pattern 1: LangGraph + LlamaIndex (Most Common Enterprise Stack)

```
LangGraph: Agent orchestration, state management, HITL
LlamaIndex: Data ingestion, indexing, retrieval
Together: Agent graph where nodes use LlamaIndex query engines as tools
```

### Pattern 2: LangGraph + DSPy

```
LangGraph: Agent flow control, persistence
DSPy: Optimized LLM calls within each node
Together: Each graph node is a compiled DSPy module
```

### Pattern 3: CrewAI for Prototype, LangGraph for Production

```
CrewAI: Build and validate agent logic quickly
LangGraph: Reimplement with persistence, HITL, streaming for production
Together: Sequential development phases
```

## The "Start Simple" Ladder

From Anthropic's recommendation, progress through complexity only as needed:

```
Level 0: Single LLM call with structured output
         (No framework needed)
              |
              v
Level 1: LLM + tool use (single turn)
         (Raw Claude/OpenAI API)
              |
              v
Level 2: Agent loop (multi-turn tool use)
         (Pydantic AI or raw SDK)
              |
              v
Level 3: Agent with persistence and HITL
         (LangGraph)
              |
              v
Level 4: Multi-agent with supervisor
         (LangGraph or CrewAI)
              |
              v
Level 5: Hierarchical multi-agent with optimized prompts
         (LangGraph + DSPy + LlamaIndex)
```

## Anti-Patterns

| Anti-Pattern | Consequence | Fix |
|-------------|-------------|-----|
| Starting with multi-agent framework for simple tasks | Over-engineering, slow development | Start at Level 0-1, add complexity as needed |
| Using LangChain chains when you need cycles | Cannot implement retry or self-correction loops | Upgrade to LangGraph |
| Using DSPy without training data | Compiler cannot optimize, wasted setup | Use manual prompts until you have data |
| Using CrewAI for complex conditional flows | Fighting the abstraction | Switch to LangGraph |
| Using LangGraph for simple RAG | Over-engineering a query pipeline | Use LlamaIndex standalone |

## How to Answer "Which Framework?" in Interviews

> "I start with the simplest approach -- often a raw API call with tool use. I reach for a framework only when I need something the raw API doesn't provide. For production agents that need persistence, human-in-the-loop, or complex flow control, LangGraph is my default -- it's in 22.6% of job listings and gives me explicit control. For rapid prototyping, I use CrewAI or Pydantic AI. For data-heavy RAG agents, LlamaIndex. And if I have training data and want to optimize prompts systematically, I add DSPy. The key principle is: don't add abstraction until the complexity demands it."

## Sources and Further Reading

- [LangChain AI Job Market Report 2026](https://blog.langchain.dev/ai-job-market-report/)
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [CrewAI Documentation](https://docs.crewai.com/)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [DSPy Documentation](https://dspy.ai/)
