# Cost Engineering for Agentic Systems
> Token costs are 70-90% of total agent cost, and naive estimates are 5-10x wrong due to quadratic accumulation in multi-turn conversations.

## What It Is

Cost engineering is the discipline of understanding, predicting, and controlling the financial cost of running LLM-powered agent systems in production. Unlike traditional software where compute is the dominant cost, agentic systems are dominated by **token costs** -- the per-token charges from LLM API providers.

The fundamental challenge: costs grow **quadratically** in multi-turn conversations because the full conversation history is re-sent with each turn. A 10-turn conversation doesn't cost 10x a single turn -- it costs roughly 55x (sum of 1+2+3+...+10).

## How It Works

### The Cost Stack

```
Total Agent Cost = Token Costs (70-90%)
                 + Embedding Costs (5-15%)
                 + Infrastructure (5-10%)
                 + Tool Execution (<5%)
```

### Why Simple Estimates Are 5-10x Wrong

**The quadratic accumulation problem:**

```
Turn 1: Send 500 tokens    → 500 input tokens
Turn 2: Send 500 + 800     → 1,300 input tokens  
Turn 3: Send 1,300 + 800   → 2,100 input tokens
Turn 4: Send 2,100 + 800   → 2,900 input tokens
...
Turn 10: Send ~7,700 tokens → 7,700 input tokens

Naive estimate: 10 turns x 500 = 5,000 tokens
Actual input:   500 + 1,300 + 2,100 + ... + 7,700 = ~41,000 tokens
```

The naive estimate is **8.2x wrong** just for input tokens.

### Output Tokens Are 3-8x More Expensive

| Provider | Input ($/1M) | Output ($/1M) | Ratio |
|----------|-------------|---------------|-------|
| Claude Sonnet 4 | $3.00 | $15.00 | 5.0x |
| Claude Opus 4 | $15.00 | $75.00 | 5.0x |
| GPT-4o | $2.50 | $10.00 | 4.0x |
| Claude Haiku 3.5 | $0.80 | $4.00 | 5.0x |
| Gemini 2.5 Flash | $0.15 | $0.60 | 4.0x |

This means **controlling output verbosity** is often more impactful than reducing input tokens.

## Production Implementation

```python
from dataclasses import dataclass
from enum import Enum

class CostTier(Enum):
    CHEAP = "cheap"       # Haiku-class: $0.25-0.80/M input
    STANDARD = "standard" # Sonnet-class: $3.00/M input
    PREMIUM = "premium"   # Opus-class: $15.00/M input

@dataclass
class ModelPricing:
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float = 0.0

PRICING = {
    "claude-haiku-3.5": ModelPricing(0.80, 4.00, 0.08),
    "claude-sonnet-4": ModelPricing(3.00, 15.00, 0.30),
    "claude-opus-4": ModelPricing(15.00, 75.00, 1.50),
}

def estimate_session_cost(
    turns: int,
    avg_user_tokens: int = 150,
    avg_system_tokens: int = 500,
    avg_response_tokens: int = 400,
    model: str = "claude-sonnet-4"
) -> dict:
    """Estimate total session cost accounting for quadratic accumulation."""
    pricing = PRICING[model]
    total_input = 0
    total_output = 0
    
    for turn in range(1, turns + 1):
        # Each turn re-sends full history
        turn_input = avg_system_tokens + (turn * avg_user_tokens) + ((turn - 1) * avg_response_tokens)
        total_input += turn_input
        total_output += avg_response_tokens
    
    input_cost = (total_input / 1_000_000) * pricing.input_per_million
    output_cost = (total_output / 1_000_000) * pricing.output_per_million
    
    naive_estimate = turns * (avg_system_tokens + avg_user_tokens + avg_response_tokens)
    
    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(input_cost + output_cost, 6),
        "naive_estimate_tokens": naive_estimate,
        "actual_vs_naive_ratio": round(total_input / naive_estimate, 1),
    }

# Example: 8-turn support conversation
result = estimate_session_cost(turns=8, model="claude-sonnet-4")
# actual_vs_naive_ratio: ~4.5x
```

## Decision Tree for Cost Optimization Strategy

```
START: What is your dominant cost?
│
├── Token costs > 80% of total?
│   ├── Multi-turn conversations averaging > 5 turns?
│   │   ├── YES → Implement conversation summarization (saves 40-60%)
│   │   └── NO → Focus on prompt optimization and output control
│   │
│   ├── Same queries repeated frequently?
│   │   ├── YES → Implement semantic caching (saves 20-40% on cache hits)
│   │   └── NO → Focus on model routing
│   │
│   └── Using frontier model for everything?
│       ├── YES → Implement model routing (saves 60-70%)
│       └── NO → Already routing, optimize thresholds
│
├── Embedding costs > 15%?
│   ├── Re-embedding unchanged documents?
│   │   └── YES → Cache embeddings, only re-embed on change
│   └── High-dimensional embeddings for simple tasks?
│       └── YES → Use smaller embedding models for classification
│
└── Infrastructure costs > 10%?
    └── Over-provisioned GPU instances?
        └── YES → Right-size instances, use spot for batch workloads
```

## When NOT to Use Cost Engineering

- **Prototyping phase**: Optimize for iteration speed, not cost. Use the best model.
- **< $100/month spend**: Engineering time exceeds savings. Just use one model.
- **Latency-critical applications**: Cost optimization (caching, routing) adds latency. Measure the tradeoff.

## Tradeoffs

| Strategy | Cost Savings | Latency Impact | Quality Risk | Implementation Effort |
|----------|-------------|----------------|-------------|----------------------|
| Model routing | 60-70% | +50-100ms (router) | Medium (wrong routing) | Medium |
| Semantic caching | 20-40% | -200ms (cache hit) / +50ms (miss) | Low (stale cache) | High |
| Conversation summarization | 40-60% | +200-500ms per summary | Medium (info loss) | Medium |
| Prompt optimization | 10-30% | None | Low | Low |
| Output length control | 15-25% | None | Low-Medium | Low |
| Batch processing | 40-50% | +hours (async) | None | Low |

## Real-World Examples

- **Customer support agent**: Average ticket = 3,550 tokens. At $3/M (Sonnet), that's $0.01/ticket. At 100K tickets/month = $1,000/month. With routing, drops to $350/month.
- **Document analysis pipeline**: Per document = $1.40 all-frontier vs $0.34 with routing (76% savings).
- **Enterprise 3-year TCO**: Naive all-frontier = EUR 368K vs optimized = EUR 158K (57% savings).

## Failure Modes

1. **Optimizing too early**: Spending $5K engineering time to save $50/month.
2. **Quality degradation from aggressive routing**: Sending complex queries to cheap models.
3. **Cache poisoning**: Serving wrong cached answers due to overly loose semantic matching.
4. **Budget enforcement killing user sessions**: Users mid-conversation get cut off abruptly.
5. **Not accounting for retries**: Failed tool calls and retries multiply token costs 1.5-2x.

## Source(s) and Further Reading

- Anthropic Pricing: https://www.anthropic.com/pricing
- OpenAI Pricing: https://openai.com/api/pricing
- "Building Effective Agents" - Anthropic (2024)
- LiteLLM Documentation: https://docs.litellm.ai/
- "The Hidden Costs of LLM Applications" - a]6z (2024)
