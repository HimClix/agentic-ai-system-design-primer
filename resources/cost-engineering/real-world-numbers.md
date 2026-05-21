# Real-World Cost Numbers Reference
> The single page you keep open during system design interviews -- actual costs, not estimates.

## What It Is

This is the definitive reference table for agentic AI system costs in production. Every number is sourced from public pricing pages, published case studies, or industry benchmarks. Use these numbers when designing systems, estimating budgets, and answering interview questions about cost.

## Per-Action Costs (2025 Pricing)

### Support Ticket Resolution

| Component | Tokens | Model | Cost |
|-----------|--------|-------|------|
| System prompt | 500 | Input | -- |
| Customer message | 150 | Input | -- |
| Knowledge base context (RAG) | 1,200 | Input | -- |
| Conversation history (avg) | 800 | Input | -- |
| Tool call overhead | 500 | Input | -- |
| **Total input** | **3,150** | Sonnet ($3/M) | **$0.0095** |
| Agent response | 400 | Output ($15/M) | **$0.0060** |
| Embedding (for RAG) | 1,200 | Voyage ($0.06/M) | $0.0001 |
| **Total per ticket** | **3,550** | | **$0.016** |

With model routing (60% Haiku):
- Simple tickets on Haiku: $0.005/ticket
- Complex tickets on Sonnet: $0.016/ticket
- **Blended average: $0.009/ticket**

**Comparison**: Intercom charges $0.99/resolution. Human agent costs $5-15/ticket.

### Document Analysis

| Approach | Cost per Document | Notes |
|----------|------------------|-------|
| All-frontier (Opus) | $1.40 | 10K input + 2K output tokens |
| Model-routed | $0.34 | Extract (Haiku) → Analyze (Sonnet) |
| With caching (30% hit) | $0.24 | Similar documents skip LLM call |
| Batch API (50% discount) | $0.17 | Non-real-time processing |

### Other Common Actions

| Action | Typical Tokens | Cost (Sonnet) | Cost (Haiku) |
|--------|---------------|---------------|-------------|
| Intent classification | 300 in / 20 out | $0.001 | $0.0003 |
| Entity extraction | 500 in / 200 out | $0.005 | $0.001 |
| Code review (single file) | 3,000 in / 1,000 out | $0.024 | $0.006 |
| Email draft | 800 in / 500 out | $0.010 | $0.003 |
| SQL query generation | 1,500 in / 200 out | $0.008 | $0.002 |
| Meeting summary | 5,000 in / 800 out | $0.027 | $0.007 |

## Monthly Operating Costs

### Small Scale (Startup/MVP)

```
Volume: 10K agent sessions/month
Average session: 5 turns, 3,500 tokens/turn

Token costs:
  Input:  10K x 5 x (500 + 800 + 1200 + 500) = 150M tokens
  Output: 10K x 5 x 400 = 20M tokens
  
  Sonnet: (150 x $3 + 20 x $15) / 1M = $750/month
  Haiku:  (150 x $0.80 + 20 x $4) / 1M = $200/month
  Routed: ~$400/month

Infrastructure:
  LangGraph server (2 replicas): $200/month
  Redis (cache + state): $50/month
  PostgreSQL (RDS): $100/month
  Vector DB (Pinecone starter): $70/month
  Observability (LangSmith): $0 (free tier)

Total: $400 (tokens) + $420 (infra) = ~$820/month
```

### Mid Scale (Growth Stage)

```
Volume: 100K agent sessions/month
Average session: 8 turns, 4,000 tokens/turn

Token costs:
  Input:  100K x 8 x 3,000 = 2.4B tokens
  Output: 100K x 8 x 400 = 320M tokens

  All-Sonnet: (2,400 x $3 + 320 x $15) / 1M = $12,000/month
  Routed (60% Haiku, 30% Sonnet, 10% Opus):
    Haiku input:  1.44B x $0.80/M = $1,152
    Sonnet input: 720M x $3.00/M = $2,160
    Opus input:   240M x $15.00/M = $3,600
    Haiku output: 192M x $4.00/M = $768
    Sonnet output: 96M x $15.00/M = $1,440
    Opus output:   32M x $75.00/M = $2,400
    Total routed: $11,520/month  

  With caching (25% hit rate): $8,640/month

Infrastructure:
  K8s cluster (3 nodes): $600/month
  Redis cluster: $200/month
  PostgreSQL (RDS Multi-AZ): $400/month
  Vector DB (Pinecone Standard): $350/month
  LangSmith (Pro): $400/month
  Monitoring (Datadog): $200/month

Total: $8,640 (tokens) + $2,150 (infra) = ~$10,800/month
```

### Enterprise Scale

```
Volume: 1M+ agent sessions/month
Average session: 10 turns, 5,000 tokens/turn

Token costs (routed + cached + optimized): $50,000-80,000/month
Infrastructure: $8,000-15,000/month
Total: $58,000-95,000/month
```

## 3-Year Total Cost of Ownership

### Enterprise TCO Comparison

| Component | Naive (All-Frontier) | Optimized | Savings |
|-----------|---------------------|-----------|---------|
| LLM API costs (3 years) | EUR 280,000 | EUR 98,000 | 65% |
| Infrastructure | EUR 36,000 | EUR 28,000 | 22% |
| Engineering (build + maintain) | EUR 25,000 | EUR 45,000 | -80% |
| Monitoring & observability | EUR 12,000 | EUR 15,000 | -25% |
| Incident response | EUR 15,000 | EUR 8,000 | 47% |
| **Total 3-Year TCO** | **EUR 368,000** | **EUR 194,000** | **47%** |

Key insight: Engineering costs are higher for the optimized version (routing, caching, budgets), but LLM API savings more than compensate.

## Cost Per API Provider

### LLM Pricing (as of May 2025)

| Model | Input ($/1M) | Output ($/1M) | Cached Input | Batch Input |
|-------|-------------|---------------|-------------|------------|
| Claude Opus 4 | $15.00 | $75.00 | $1.50 | $7.50 |
| Claude Sonnet 4 | $3.00 | $15.00 | $0.30 | $1.50 |
| Claude Haiku 3.5 | $0.80 | $4.00 | $0.08 | $0.40 |
| GPT-4o | $2.50 | $10.00 | $1.25 | $1.25 |
| GPT-4o-mini | $0.15 | $0.60 | $0.075 | $0.075 |
| Gemini 2.5 Pro | $1.25 | $10.00 | -- | -- |
| Gemini 2.5 Flash | $0.15 | $0.60 | $0.0375 | -- |
| Llama 3.3 70B (self-hosted) | ~$0.50* | ~$0.50* | N/A | N/A |

*Self-hosted costs include GPU amortization.

### Embedding Pricing

| Model | Cost ($/1M tokens) | Dimensions | Notes |
|-------|-------------------|------------|-------|
| Voyage 3 | $0.06 | 1024 | Best quality |
| text-embedding-3-small | $0.02 | 512-1536 | Cheapest OpenAI |
| text-embedding-3-large | $0.13 | 256-3072 | Best OpenAI |
| Cohere Embed v3 | $0.10 | 1024 | Multilingual |

### Vector Database Pricing

| Service | Free Tier | Production | Notes |
|---------|-----------|-----------|-------|
| Pinecone | 100K vectors | $70/month (1M vectors) | Serverless |
| Weaviate Cloud | 50K vectors | $25/month (500K vectors) | Managed |
| Qdrant Cloud | 1M vectors | $30/month (1M vectors) | Managed |
| pgvector (self-hosted) | N/A | $100-400/month (RDS) | On existing PG |
| ChromaDB (self-hosted) | N/A | $0 (compute costs only) | Open source |

## Cost Optimization Quick Reference

| Technique | Expected Savings | Implementation Time | Risk Level |
|-----------|-----------------|-------------------|------------|
| Prompt caching (Anthropic) | 90% on cached portion | 1 day | Very Low |
| Batch API (non-real-time) | 50% | 1-2 days | Low |
| Model routing | 60-70% | 1-2 weeks | Medium |
| Semantic caching | 20-40% | 1-2 weeks | Medium |
| Context summarization | 40-60% | 1 week | Medium |
| Output length control | 15-25% | 1 day | Low |
| Prompt optimization | 10-30% | 2-3 days | Low |
| Self-hosting (at scale) | 40-60% | 1-2 months | High |

## Failure Modes

1. **Sticker shock at scale**: 100 users in beta cost $200/month, projected linearly to $20K for 10K users. Actual cost with quadratic accumulation: $50K+.
2. **Forgetting output token costs**: Input-focused optimization ignoring that output tokens are 3-5x more expensive.
3. **Embedding costs at scale**: 1M documents x 10 chunks x $0.06/M = $0.60 per full re-index. Re-indexing daily = $18/month. Not huge, but not zero.
4. **Free tier budget planning**: Building on Pinecone free tier, then needing to pay $70/month at 150K vectors.

## Source(s) and Further Reading

- Anthropic Pricing: https://www.anthropic.com/pricing (accessed May 2025)
- OpenAI Pricing: https://openai.com/api/pricing (accessed May 2025)
- Google AI Pricing: https://ai.google.dev/pricing (accessed May 2025)
- Pinecone Pricing: https://www.pinecone.io/pricing/
- LangSmith Pricing: https://www.langchain.com/langsmith
- "The Economics of AI Applications" - a16z (2024)
- Enterprise TCO study: Deloitte AI Cost Benchmarks (2024)
