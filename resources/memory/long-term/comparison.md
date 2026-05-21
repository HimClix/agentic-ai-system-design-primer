# Memory Framework Comparison: Mem0 vs Zep vs LangMem vs Hindsight vs Letta

> No single memory framework wins on every axis -- the right choice depends on whether you prioritize accuracy, latency, cost, or self-improvement.

## What It Is

This is the definitive comparison of the five major agent memory frameworks as of 2026. Each framework takes a different approach to the same problem: giving AI agents durable, queryable memory that persists across conversations.

## The Comparison Table

### Benchmark Scores

| Framework | LongMemEval | LOCOMO | DMR | Best Benchmark | Interpretation |
|---|---|---|---|---|---|
| **Hindsight** | **91.4%** | N/A | N/A | LongMemEval | Highest factual recall by far |
| **Letta** | **83.2%** | N/A | N/A | LongMemEval | Strong + self-improving |
| **Zep** | **63.8%** | N/A | **94.8%** | DMR | Best temporal reasoning |
| **LangMem** | N/A | **58.1%** | N/A | LOCOMO | Mid-tier, procedural memory unique |
| **Mem0** | **49.0%** | N/A | N/A | LongMemEval | Most adopted, lowest accuracy |

**Benchmark definitions:**
- **LongMemEval** -- Tests long-term memory across 500+ multi-session conversations. Measures fact retention, temporal reasoning, and relationship understanding.
- **LOCOMO** -- LOng COnversation Memory benchmark. Tests memory over 600+ turn conversations.
- **DMR** (Deep Memory Retrieval) -- Tests ability to retrieve deeply embedded facts across sessions.

### Feature Comparison

| Feature | Mem0 | Zep | LangMem | Hindsight | Letta |
|---|---|---|---|---|---|
| **Vector memory** | Yes | Yes | Yes | Yes | Yes |
| **Graph memory** | Pro only ($249) | All tiers ($25) | No | Limited | Yes |
| **Temporal reasoning** | Weak | Excellent | Basic | Good | Basic |
| **Procedural memory** | No | No | **Yes (unique)** | No | **Yes** |
| **Self-improving** | No | No | Yes | No | **Yes** |
| **Framework integrations** | 21 | 8 | LangGraph only | 3 | 5 |
| **Cloud offering** | Yes | Yes | No | No | Yes (beta) |
| **Self-hosted** | Yes | Yes | Yes (only option) | Yes | Yes |
| **Open source** | Yes (Apache 2.0) | Yes (Apache 2.0) | Yes (MIT) | Yes | Yes (Apache 2.0) |

### Pricing

| Tier | Mem0 | Zep | LangMem | Hindsight | Letta |
|---|---|---|---|---|---|
| **Free / OSS** | Self-hosted, no graph | Self-hosted + Neo4j | Full features (MIT) | Self-hosted | Self-hosted |
| **Starter** | $99/mo (vector only) | $25/mo (graph included) | N/A (free) | N/A | Free beta |
| **Pro** | $249/mo (graph added) | $25/mo (same) | N/A | N/A | Custom |
| **Enterprise** | Custom | Custom | N/A | Custom | Custom |
| **Graph included at** | $249/mo | $25/mo | Never | N/A | Free |

### Performance

| Metric | Mem0 | Zep | LangMem | Hindsight | Letta |
|---|---|---|---|---|---|
| **Search latency (p50)** | ~100ms | ~150ms | ~5s | ~200ms | ~300ms |
| **Search latency (p95)** | ~500ms | ~800ms | **59.82s** | ~1s | ~2s |
| **Memory footprint** | Small-Medium | **Large (600K+ tokens)** | Small | Medium | Medium |
| **Token reduction claim** | 80% | 70% | N/A | N/A | 75% |
| **Write latency** | ~200ms | ~300ms | ~500ms | ~400ms | ~500ms |

### Architecture

| Aspect | Mem0 | Zep | LangMem | Hindsight | Letta |
|---|---|---|---|---|---|
| **Storage backend** | Qdrant/pgvector/Pinecone + Neo4j | Neo4j (Graphiti) + vector | Any LangGraph store | Custom | Postgres + vector |
| **Extraction method** | LLM-based (GPT-4o-mini) | LLM-based (Graphiti) | LLM-based | LLM-based | LLM-based |
| **Deduplication** | Cosine similarity | Graph-based | Similarity | Temporal | Similarity |
| **Contradiction handling** | v1.1+ (basic) | Temporal (valid_at/invalid_at) | Manual | Automatic | Agent-driven |
| **Memory hierarchy** | User/Session/Agent | User/Session/Graph | Semantic/Episodic/Procedural | Flat | Agent-managed |

## How They Compare: Deep Dive

### Accuracy vs Adoption

```
LongMemEval Score
    │
95% │                                    ★ Hindsight (91.4%)
    │
85% │                          ★ Letta (83.2%)
    │
75% │
    │
65% │            ★ Zep (63.8%)
    │
55% │
    │
50% │                                              ★ Mem0 (49.0%)
    │
    └──────────────────────────────────────────────────────────
        Low adoption              →              High adoption
        (Hindsight)                               (Mem0: 25K★)
```

The inverse correlation is striking: the most adopted framework (Mem0, 25K GitHub stars) has the lowest benchmark score. This suggests the market optimizes for developer experience and integrations over raw accuracy.

### Cost vs Capability

```
Monthly Cost (managed)
    │
$250│  ★ Mem0 Pro (graph)
    │
$200│
    │
$150│
    │
$100│  ★ Mem0 Cloud (no graph)
    │
 $50│
    │
 $25│  ★ Zep Cloud (graph included!)
    │
  $0│  ★ LangMem (free, self-hosted)  ★ Hindsight (self-hosted)  ★ Letta (free beta)
    │
    └──────────────────────────────────────────────────────────
       Vector only        Vector+Graph        Full featured
```

### Unique Strengths

| Framework | Unique Strength | No Other Framework Has This |
|---|---|---|
| **Mem0** | 21 framework integrations | Broadest ecosystem compatibility |
| **Zep** | Bi-temporal knowledge graph (valid_at/invalid_at) | True temporal reasoning at $25/mo |
| **LangMem** | Procedural memory (self-improving prompts) | Agents that rewrite their own instructions |
| **Hindsight** | 91.4% LongMemEval accuracy | Highest factual recall |
| **Letta** | Self-evolving agent architecture | Agent manages its own memory autonomously |

## Decision Guide

### Quick Decision Matrix

| Your Situation | Best Choice | Runner-Up |
|---|---|---|
| "I need the highest accuracy" | Hindsight (91.4%) | Letta (83.2%) |
| "I need graph memory on a budget" | Zep ($25/mo) | Mem0 OSS + Neo4j ($0) |
| "I need the most integrations" | Mem0 (21 frameworks) | Zep (8 frameworks) |
| "I need self-improving agents" | LangMem (procedural) or Letta | -- |
| "I need temporal reasoning" | Zep (Graphiti) | Hindsight |
| "I have zero budget" | LangMem (MIT, free) | Mem0 OSS |
| "I am on LangGraph" | LangMem (native) | Zep (good integration) |
| "I need a managed service" | Zep Cloud ($25) or Mem0 Cloud ($99) | -- |
| "I need lowest latency" | Mem0 (~100ms p50) | Zep (~150ms p50) |
| "I need an autonomous agent" | Letta (83.2%, self-managing) | LangMem (procedural) |

### Detailed Decision Tree

```
What is your primary requirement?
  │
  ├─ ACCURACY (factual recall)
  │   ├─ Can you self-host? → Hindsight (91.4%)
  │   ├─ Need managed? → Zep Cloud (63.8%)
  │   └─ Budget $0? → LangMem (58.1% LOCOMO)
  │
  ├─ TEMPORAL REASONING ("what was true on date X?")
  │   └─ Zep with Graphiti (only real option)
  │
  ├─ SELF-IMPROVEMENT (agents learn from feedback)
  │   ├─ On LangGraph? → LangMem (procedural memory)
  │   └─ Not on LangGraph? → Letta (autonomous memory management)
  │
  ├─ ECOSYSTEM INTEGRATION
  │   ├─ Need LangChain + CrewAI + AutoGen + ...? → Mem0 (21 integrations)
  │   ├─ LangGraph only? → LangMem (native)
  │   └─ Flexible? → Zep (REST API works everywhere)
  │
  ├─ COST OPTIMIZATION
  │   ├─ $0 total → LangMem (MIT) or Mem0 OSS (Apache 2.0)
  │   ├─ < $50/mo → Zep Cloud ($25, includes graph)
  │   ├─ < $100/mo → Mem0 Cloud ($99, no graph)
  │   └─ < $250/mo → Mem0 Pro ($249, with graph)
  │
  └─ LATENCY REQUIREMENT
      ├─ < 200ms p95 → Mem0 Cloud or custom pgvector
      ├─ < 1s p95 → Mem0 or Zep
      └─ Relaxed (> 1s OK) → Any framework, including LangMem
```

## When NOT to Use Any of These

- **Simple chatbots** -- if your conversation never exceeds 10 turns and never spans sessions, the context window is your memory. No framework needed.
- **Batch processing** -- if your agent processes documents one at a time with no user state, memory adds complexity without value.
- **Compliance-heavy domains** -- before adopting any memory framework, verify it meets your data residency, encryption, and audit requirements. Many do not out of the box.
- **When context window keeps growing** -- with 1M+ token context windows (Gemini 2.5 Pro), some use cases that previously needed memory can now fit in context. Evaluate before adding infrastructure.

## Tradeoffs Summary

| What You Optimize For | What You Give Up | Recommendation |
|---|---|---|
| Highest accuracy | Cost (self-host), ecosystem (fewer integrations) | Hindsight |
| Lowest cost | Accuracy, managed convenience | LangMem (free) |
| Best developer experience | Accuracy (49% LongMemEval) | Mem0 |
| Temporal reasoning | Memory footprint (600K+ tokens), complexity | Zep |
| Self-improvement | Latency (59.82s p95), framework lock-in | LangMem |
| Autonomous agents | Maturity, predictability | Letta |

## Real-World Selection Examples

1. **Startup building a customer support agent** -- Needs 10+ framework support (LangChain backend, various frontends), managed service, reasonable accuracy. **Choice: Mem0 Cloud ($99/mo)**. The 49% accuracy is acceptable because support agents can re-ask customers.

2. **Enterprise building a compliance-aware financial agent** -- Needs temporal reasoning ("what was the risk rating on March 1?"), high accuracy, graph relationships between entities. **Choice: Zep Cloud ($25/mo)** or Zep self-hosted if data residency requires it.

3. **Solo developer building a coding assistant** -- Zero budget, already using LangGraph, wants the agent to learn from code review feedback. **Choice: LangMem (free, MIT)**. The 59s latency is fine for non-real-time code assistance.

4. **Research team building a high-accuracy knowledge agent** -- Accuracy is the top priority, willing to self-host. **Choice: Hindsight (91.4% LongMemEval)**. Best factual recall available.

5. **AI lab building an autonomous agent** -- Wants the agent to manage its own memory, evolve over time, minimal human intervention. **Choice: Letta (83.2%, self-improving)**. High accuracy with autonomous memory management.

## Failure Modes (Common Across All)

| Failure | Affected Frameworks | Mitigation |
|---|---|---|
| Memory pollution (wrong facts stored) | All | Importance filtering, human review for critical facts |
| Cross-user data leakage | All | Strict tenant isolation, user_id on every query |
| Benchmark scores do not match production | All | Benchmarks test specific patterns; your data may differ. Run your own eval. |
| Vendor lock-in | Mem0 Cloud, Zep Cloud | Use open-source versions; abstract memory behind an interface |
| LLM-based extraction non-determinism | All | temperature=0, deterministic extraction prompts |

## Source(s) and Further Reading

- [State of AI Agent Memory 2026](https://mem0.ai) -- Mem0 research report (all benchmark numbers)
- [Mem0 Paper](https://arxiv.org/abs/2504.19413) -- arXiv:2504.19413
- [Mem0 vs Zep vs LangMem Comparison](https://dev.to) -- dev.to
- [8 Agent Memory Frameworks Compared](https://vectorize.io) -- vectorize.io
- [Mem0 GitHub](https://github.com/mem0ai/mem0) -- 25K+ stars
- [Zep / Graphiti GitHub](https://github.com/getzep/graphiti) -- Temporal knowledge graph
- [LangMem GitHub](https://github.com/langchain-ai/langmem) -- MIT license
- [Letta GitHub](https://github.com/letta-ai/letta) -- Formerly MemGPT
