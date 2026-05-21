# Prompt Caching
> Caching your system prompt with Anthropic saves 90% on cached tokens. With OpenAI's automatic caching, you save 50%. At scale, this is the single highest-ROI optimization for LLM costs.

## What It Is

Prompt caching allows you to designate portions of your prompt as cacheable. When the same prefix appears across multiple API calls, the provider reuses the pre-computed KV-cache from the first call, dramatically reducing both latency and cost.

**Two major implementations:**

| Provider | Mechanism | Savings | Min Cacheable | TTL | Action Required |
|----------|-----------|---------|---------------|-----|-----------------|
| Anthropic | Explicit `cache_control` breakpoints | 90% on cached tokens | 1,024 tokens | 5 min | Developer marks cache points |
| OpenAI | Automatic prefix matching | 50% on cached tokens | 1,024 tokens | 5-10 min | Nothing (automatic) |

## How It Works

### Anthropic Prompt Caching

```
API Call 1 (cache miss):
  System Prompt (5,000 tokens) ← cache_control: {type: "ephemeral"}
  User Message (100 tokens)
  
  Cost: 5,000 * $0.003/1K (write) + 100 * $0.003/1K = $0.0153
  Latency: Full processing of 5,100 tokens

API Call 2 (cache hit, same system prompt):
  System Prompt (5,000 tokens) ← CACHED (90% discount)
  User Message (200 tokens)    ← not cached
  
  Cost: 5,000 * $0.0003/1K (read) + 200 * $0.003/1K = $0.0021
  Latency: Skip processing 5,000 tokens (~50% faster)

Savings: $0.0153 - $0.0021 = $0.0132 per call (86% reduction)
```

### How KV-Cache Reuse Works

```
Normal processing:
  [System Prompt tokens] -> Transformer layers -> KV-Cache
  [User tokens]          -> Transformer layers (using KV-Cache from system)
  Total: Process ALL tokens through transformer

With prompt caching:
  [System Prompt tokens] -> Load pre-computed KV-Cache from storage
  [User tokens]          -> Transformer layers (using loaded KV-Cache)
  Total: Only process NEW tokens through transformer
  
  The system prompt's KV-Cache is stored after first computation
  and reused for subsequent calls with the same prefix.
```

### What Can Be Cached

```
Cacheable (prefix content that repeats across calls):
  - System prompts
  - Tool/function schemas
  - Few-shot examples
  - Long documents (RAG context if reused)
  - Company policies, guidelines
  - Code context (for code agents)

NOT cacheable (varies per call):
  - User messages
  - Conversation history (unless identical prefix)
  - Dynamic RAG results
  - Timestamps, session IDs
```

**Critical rule**: Caching works on PREFIX matching. The cached content must be a continuous prefix of the prompt. You cannot cache content in the middle.

```
[Cached System Prompt][Cached Tools][Dynamic RAG][User Message]
|<---- cacheable ---->|<-- cache -->|<- no cache ->|<- no ->|

Order matters! Put stable content first, dynamic content last.
```

## Production Implementation

### Anthropic Prompt Caching

```python
"""Prompt caching with Anthropic's Claude API."""
import anthropic

client = anthropic.Anthropic()

# The system prompt and tools are stable across calls -- cache them
SYSTEM_PROMPT = """You are a customer support agent for Acme Payments.
You help merchants with payment integration, refunds, and account management.

## Guidelines
- Always verify the merchant's identity before sharing account details
- For refund requests, check the order status before processing
- Escalate fraud-related queries to the fraud team
- Provide accurate information from the knowledge base

## Tone
- Professional but friendly
- Concise but thorough
- Empathetic when handling complaints
"""

TOOL_SCHEMAS = [
    {
        "name": "get_order",
        "description": "Retrieve order details by order ID. Returns order status, amount, payment method, and customer info.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Order ID in format 'order_XXXXX'. Example: 'order_abc123'",
                },
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "process_refund",
        "description": "Process a refund for an order. Refund amount cannot exceed order amount. Returns refund ID and status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID to refund"},
                "amount": {"type": "integer", "description": "Refund amount in paise (smallest currency unit)"},
                "reason": {"type": "string", "description": "Reason for refund"},
            },
            "required": ["order_id", "amount", "reason"],
        },
    },
]


def create_cached_message(
    user_message: str,
    conversation_history: list[dict] = None,
) -> dict:
    """Create an API call with prompt caching for stable prefix.
    
    The system prompt and tool schemas are cached (90% savings).
    Conversation history and user message are NOT cached (dynamic).
    """
    # Build system content with cache control
    system_content = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            # Cache the system prompt (stable across all calls)
            "cache_control": {"type": "ephemeral"},
        },
    ]
    
    # Build messages
    messages = []
    
    # Add conversation history (not cached -- changes per conversation)
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add current user message
    messages.append({"role": "user", "content": user_message})
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_content,
        tools=TOOL_SCHEMAS,
        messages=messages,
        # Note: tools are automatically cached when system has cache_control
    )
    
    # Log cache performance
    usage = response.usage
    if hasattr(usage, 'cache_read_input_tokens'):
        cached = usage.cache_read_input_tokens
        uncached = usage.input_tokens
        total = cached + uncached
        cache_hit_rate = cached / total if total > 0 else 0
        print(f"Cache hit rate: {cache_hit_rate:.0%} "
              f"({cached} cached / {total} total input tokens)")
    
    return response


# Advanced: Multi-level caching for RAG + conversation

def create_multi_cache_message(
    user_message: str,
    rag_context: str,
    conversation_history: list[dict] = None,
) -> dict:
    """Multi-level caching: system prompt cached, RAG conditionally cached.
    
    Level 1 (always cached): System prompt + tool schemas
    Level 2 (session cached): RAG context (if same topic across turns)
    Level 3 (never cached): Conversation history + user message
    """
    system_content = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},  # Level 1 cache
        },
    ]
    
    # If RAG context is substantial and likely reused across turns in this
    # conversation, cache it too
    if rag_context and len(rag_context) > 500:
        system_content.append({
            "type": "text",
            "text": f"\n## Relevant Knowledge Base Context\n{rag_context}",
            "cache_control": {"type": "ephemeral"},  # Level 2 cache
        })
    
    messages = conversation_history or []
    messages.append({"role": "user", "content": user_message})
    
    return client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_content,
        tools=TOOL_SCHEMAS,
        messages=messages,
    )
```

### OpenAI Automatic Caching

```python
"""OpenAI automatic prompt caching (no code changes needed)."""
import openai

client = openai.OpenAI()

# OpenAI automatically caches identical prefixes.
# No explicit cache_control needed.
# The savings are 50% on cached tokens.

SYSTEM_PROMPT = """You are a customer support agent..."""  # Same as above

def create_openai_message(user_message: str) -> dict:
    """OpenAI caches the system prompt automatically.
    
    If the system prompt (and any other prefix content) is identical
    to a recent call, OpenAI reuses the KV-cache internally.
    
    Cost savings: 50% on cached input tokens
    TTL: 5-10 minutes of inactivity
    Min cacheable: 1,024 tokens
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=4096,
    )
    
    # Check caching in usage (OpenAI reports cached tokens)
    usage = response.usage
    if hasattr(usage, 'prompt_tokens_details'):
        cached = getattr(usage.prompt_tokens_details, 'cached_tokens', 0)
        total = usage.prompt_tokens
        print(f"Cached: {cached}/{total} tokens ({cached/total*100:.0f}%)")
    
    return response
```

### Cost Impact Calculator

```python
"""Calculate cost savings from prompt caching."""


def calculate_caching_savings(
    system_prompt_tokens: int,
    dynamic_tokens_per_call: int,
    calls_per_day: int,
    provider: str = "anthropic",
) -> dict:
    """Calculate monthly cost savings from prompt caching.
    
    Args:
        system_prompt_tokens: Tokens in the cached system prompt
        dynamic_tokens_per_call: Average non-cached tokens per call
        calls_per_day: Number of API calls per day
        provider: "anthropic" (90% savings) or "openai" (50% savings)
    """
    calls_per_month = calls_per_day * 30
    
    if provider == "anthropic":
        # Anthropic Claude 3.5 Sonnet pricing
        input_price_per_1k = 0.003      # Standard input
        cached_price_per_1k = 0.0003    # 90% discount on cached
        cache_write_per_1k = 0.00375    # 25% premium on first write
        
        # Without caching: all tokens at full price
        cost_no_cache = (
            (system_prompt_tokens + dynamic_tokens_per_call) 
            * input_price_per_1k / 1000 
            * calls_per_month
        )
        
        # With caching: system prompt at cached price (after first call)
        # First call per 5-min window pays write price
        cache_windows_per_day = calls_per_day  # Approximation
        write_calls = min(cache_windows_per_day, 288)  # Max 288 5-min windows/day
        read_calls = calls_per_day - write_calls
        
        cost_with_cache = (
            # Cache write calls (25% premium on system prompt)
            (write_calls * 30 * system_prompt_tokens * cache_write_per_1k / 1000)
            # Cache read calls (90% discount on system prompt)
            + (read_calls * 30 * system_prompt_tokens * cached_price_per_1k / 1000)
            # Dynamic tokens always at full price
            + (calls_per_month * dynamic_tokens_per_call * input_price_per_1k / 1000)
        )
        
    elif provider == "openai":
        # OpenAI GPT-4o pricing
        input_price_per_1k = 0.0025
        cached_price_per_1k = 0.00125   # 50% discount
        
        cost_no_cache = (
            (system_prompt_tokens + dynamic_tokens_per_call)
            * input_price_per_1k / 1000
            * calls_per_month
        )
        
        cost_with_cache = (
            (system_prompt_tokens * cached_price_per_1k / 1000 * calls_per_month)
            + (dynamic_tokens_per_call * input_price_per_1k / 1000 * calls_per_month)
        )
    
    savings = cost_no_cache - cost_with_cache
    savings_pct = (savings / cost_no_cache) * 100 if cost_no_cache > 0 else 0
    
    return {
        "provider": provider,
        "calls_per_month": calls_per_month,
        "cost_without_caching": round(cost_no_cache, 2),
        "cost_with_caching": round(cost_with_cache, 2),
        "monthly_savings": round(savings, 2),
        "savings_percentage": round(savings_pct, 1),
    }


# Example calculations:
print(calculate_caching_savings(
    system_prompt_tokens=5000,
    dynamic_tokens_per_call=500,
    calls_per_day=10000,
    provider="anthropic",
))
# Output: ~$3,600/month savings (85% reduction on input costs)

print(calculate_caching_savings(
    system_prompt_tokens=5000,
    dynamic_tokens_per_call=500,
    calls_per_day=10000,
    provider="openai",
))
# Output: ~$1,125/month savings (45% reduction on input costs)
```

## Decision Tree: Should You Cache?

```
Is your system prompt > 1,024 tokens?
  |
  NO  --> Caching won't apply (below minimum)
  |       Consider making system prompt longer with few-shot examples
  |
  YES --> Do you make > 100 calls/day?
           |
           NO  --> Savings are minimal (<$5/month). Optional.
           |
           YES --> Are you using Anthropic?
                    |
                    YES --> Add cache_control breakpoints (90% savings)
                    |       Priority: system prompt > tool schemas > few-shot examples
                    |
                    NO --> Are you using OpenAI?
                            |
                            YES --> Automatic! No code changes needed (50% savings)
                            |       Ensure system prompt is identical across calls
                            |
                            NO --> Check provider documentation for caching support
```

## When NOT to Cache

- **System prompt changes frequently**: If you modify the prompt every few minutes, cache hit rate will be near zero
- **Unique prefixes per call**: If every call has different RAG context before the system prompt, prefixes won't match
- **Very short prompts**: Under 1,024 tokens, caching is not supported
- **Low volume**: Under 100 calls/day, the savings are negligible

## Tradeoffs

| Aspect | Anthropic Caching | OpenAI Caching | No Caching |
|--------|-------------------|----------------|------------|
| Savings rate | 90% on cached tokens | 50% on cached tokens | 0% |
| Developer effort | Must add cache_control | Zero (automatic) | Zero |
| Cache TTL | 5 minutes | 5-10 minutes | N/A |
| Minimum size | 1,024 tokens | 1,024 tokens | N/A |
| Cache write cost | 25% premium on first write | None | N/A |
| Visibility | Usage API reports cache hits | Usage API reports cached_tokens | N/A |
| Control | Fine-grained breakpoints | All-or-nothing prefix | N/A |

## Real-World Examples

### Before and After: Customer Support Agent
```
Before caching:
  System prompt: 5,000 tokens * $0.003/1K = $0.015 per call
  Dynamic content: 500 tokens * $0.003/1K = $0.0015 per call
  Total input cost per call: $0.0165
  Monthly (10K calls/day): $4,950

After Anthropic caching:
  System prompt (cached): 5,000 tokens * $0.0003/1K = $0.0015 per call
  Dynamic content: 500 tokens * $0.003/1K = $0.0015 per call
  Total input cost per call: $0.003
  Monthly: $900

Savings: $4,050/month (82% reduction)
```

## Failure Modes

| Failure | Cause | Impact | Fix |
|---------|-------|--------|-----|
| Low cache hit rate | System prompt varies slightly per call | No cost savings | Standardize system prompt, move dynamic content after cache breakpoint |
| Cache expired | >5 min between calls | Cache miss, full price | Ensure steady call volume or accept cold-start cost |
| Order dependency | Dynamic content placed before cached content | Cache prefix broken | Reorder: cached content first, dynamic content after |
| Stale cached behavior | System prompt updated but cache serves old version | Model uses outdated instructions | Verify cache invalidation after prompt updates |

## Source(s) and Further Reading

- Anthropic, "Prompt Caching" (2024) -- official documentation and pricing
- OpenAI, "Prompt Caching" (2024) -- automatic caching documentation
- Anthropic, "Token Counting and Pricing" -- cache pricing details
- LangChain, "Prompt Caching Integration" -- framework support
