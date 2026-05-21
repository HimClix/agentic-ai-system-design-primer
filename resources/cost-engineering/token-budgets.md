# Token Budget Enforcement
> Without budgets, a single runaway agent session can consume your entire monthly LLM allocation in minutes.

## What It Is

Token budgets are hard and soft limits on LLM token consumption at multiple granularities: per-request, per-session, per-user-daily, and per-tenant-monthly. They prevent cost overruns and ensure fair resource allocation across users and workloads.

The key insight: budgets must be **enforced before the LLM call**, not after. Once tokens are consumed, the cost is incurred. Budget enforcement is a pre-flight check.

## How It Works

### Budget Hierarchy

```
Organization Monthly Budget ($5,000)
├── Tenant A Monthly Budget ($2,000)
│   ├── User 1 Daily Budget ($10)
│   │   ├── Session Budget (5,000 tokens)
│   │   │   └── Request Budget (2,000 tokens)
│   │   └── Session Budget (5,000 tokens)
│   └── User 2 Daily Budget ($10)
│       └── ...
└── Tenant B Monthly Budget ($3,000)
    └── ...
```

### Budget Tiers by Action Type

| Action Type | Typical Token Cost | Suggested Budget |
|-------------|-------------------|-----------------|
| Simple classification | 200-500 tokens | 1,000 per request |
| Support ticket response | 3,550 tokens | 5,000 per request |
| Document analysis | 8,000-15,000 tokens | 20,000 per request |
| Multi-step research | 20,000-50,000 tokens | 75,000 per session |
| Code generation session | 30,000-100,000 tokens | 150,000 per session |

### Real Numbers: Support Ticket Breakdown

```
Average support ticket = 3,550 tokens total:
  System prompt:           500 tokens
  Customer message:        150 tokens  
  Knowledge base context: 1,200 tokens
  Conversation history:    800 tokens
  Agent response:          400 tokens (output)
  Tool call overhead:      500 tokens
```

## Production Implementation

```python
import time
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import redis.asyncio as redis


class BudgetAction(Enum):
    ALLOW = "allow"
    DOWNGRADE_MODEL = "downgrade_model"
    SUMMARIZE_CONTEXT = "summarize_context"
    REJECT = "reject"


@dataclass
class BudgetConfig:
    request_limit: int = 4_000          # Max tokens per single request
    session_limit: int = 50_000         # Max tokens per session
    user_daily_limit: int = 500_000     # Max tokens per user per day
    tenant_monthly_limit: int = 10_000_000  # Max tokens per tenant per month
    
    # Soft limit thresholds (percentage of hard limit)
    soft_limit_pct: float = 0.80  # Trigger downgrade at 80%
    warn_limit_pct: float = 0.60  # Log warning at 60%


@dataclass
class UsageTracker:
    request_tokens: int = 0
    session_tokens: int = 0
    user_daily_tokens: int = 0
    tenant_monthly_tokens: int = 0


class TokenBudgetEnforcer:
    """Enforces token budgets at multiple granularities."""
    
    def __init__(self, redis_client: redis.Redis, config: BudgetConfig = None):
        self.redis = redis_client
        self.config = config or BudgetConfig()
    
    async def check_budget(
        self,
        tenant_id: str,
        user_id: str,
        session_id: str,
        estimated_tokens: int,
    ) -> BudgetAction:
        """
        Pre-flight budget check before making an LLM call.
        Returns the action to take based on current usage.
        """
        usage = await self._get_usage(tenant_id, user_id, session_id)
        
        # Hard limits - reject if any would be exceeded
        checks = [
            (usage.session_tokens + estimated_tokens, self.config.session_limit, "session"),
            (usage.user_daily_tokens + estimated_tokens, self.config.user_daily_limit, "user_daily"),
            (usage.tenant_monthly_tokens + estimated_tokens, self.config.tenant_monthly_limit, "tenant_monthly"),
        ]
        
        for projected, limit, scope in checks:
            if projected > limit:
                return BudgetAction.REJECT
        
        # Soft limits - downgrade model if approaching limits
        for projected, limit, scope in checks:
            if projected > limit * self.config.soft_limit_pct:
                return BudgetAction.DOWNGRADE_MODEL
        
        # Request-level budget - summarize if single request is too large
        if estimated_tokens > self.config.request_limit:
            return BudgetAction.SUMMARIZE_CONTEXT
        
        return BudgetAction.ALLOW
    
    async def record_usage(
        self,
        tenant_id: str,
        user_id: str,
        session_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record token usage after an LLM call completes."""
        total = input_tokens + output_tokens
        pipe = self.redis.pipeline()
        
        # Session counter (expire after 24h)
        session_key = f"budget:session:{session_id}"
        pipe.incrby(session_key, total)
        pipe.expire(session_key, 86400)
        
        # Daily user counter (expire at midnight)
        today = time.strftime("%Y-%m-%d")
        daily_key = f"budget:daily:{user_id}:{today}"
        pipe.incrby(daily_key, total)
        pipe.expire(daily_key, 86400)
        
        # Monthly tenant counter (expire after 32 days)
        month = time.strftime("%Y-%m")
        monthly_key = f"budget:monthly:{tenant_id}:{month}"
        pipe.incrby(monthly_key, total)
        pipe.expire(monthly_key, 32 * 86400)
        
        await pipe.execute()
    
    async def _get_usage(
        self, tenant_id: str, user_id: str, session_id: str
    ) -> UsageTracker:
        """Fetch current usage from Redis."""
        today = time.strftime("%Y-%m-%d")
        month = time.strftime("%Y-%m")
        
        pipe = self.redis.pipeline()
        pipe.get(f"budget:session:{session_id}")
        pipe.get(f"budget:daily:{user_id}:{today}")
        pipe.get(f"budget:monthly:{tenant_id}:{month}")
        results = await pipe.execute()
        
        return UsageTracker(
            session_tokens=int(results[0] or 0),
            user_daily_tokens=int(results[1] or 0),
            tenant_monthly_tokens=int(results[2] or 0),
        )
    
    async def get_remaining_budget(
        self, tenant_id: str, user_id: str, session_id: str
    ) -> dict:
        """Return remaining budget at each level for observability."""
        usage = await self._get_usage(tenant_id, user_id, session_id)
        return {
            "session_remaining": self.config.session_limit - usage.session_tokens,
            "user_daily_remaining": self.config.user_daily_limit - usage.user_daily_tokens,
            "tenant_monthly_remaining": self.config.tenant_monthly_limit - usage.tenant_monthly_tokens,
            "session_pct_used": round(usage.session_tokens / self.config.session_limit * 100, 1),
            "user_daily_pct_used": round(usage.user_daily_tokens / self.config.user_daily_limit * 100, 1),
        }


# --- Usage in an agent loop ---

async def agent_step(enforcer: TokenBudgetEnforcer, context: dict):
    """Example: budget-aware agent execution step."""
    estimated = estimate_tokens(context["messages"])  # Your token estimator
    
    action = await enforcer.check_budget(
        tenant_id=context["tenant_id"],
        user_id=context["user_id"],
        session_id=context["session_id"],
        estimated_tokens=estimated,
    )
    
    if action == BudgetAction.ALLOW:
        model = context["preferred_model"]  # e.g., "claude-sonnet-4"
    
    elif action == BudgetAction.DOWNGRADE_MODEL:
        model = "claude-haiku-3.5"  # Switch to cheaper model
        # Log the downgrade for observability
    
    elif action == BudgetAction.SUMMARIZE_CONTEXT:
        context["messages"] = await summarize_history(context["messages"])
        model = context["preferred_model"]
    
    elif action == BudgetAction.REJECT:
        return {
            "error": "budget_exceeded",
            "message": "Token budget exceeded. Please try again later or contact support.",
            "remaining": await enforcer.get_remaining_budget(
                context["tenant_id"], context["user_id"], context["session_id"]
            ),
        }
    
    # Make the LLM call
    response = await call_llm(model=model, messages=context["messages"])
    
    # Record actual usage
    await enforcer.record_usage(
        tenant_id=context["tenant_id"],
        user_id=context["user_id"],
        session_id=context["session_id"],
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
    
    return response
```

## Decision Tree: What to Do When Budget Exceeded

```
Budget check returns non-ALLOW:
│
├── DOWNGRADE_MODEL (soft limit hit at 80%)
│   ├── Current model is Opus? → Switch to Sonnet
│   ├── Current model is Sonnet? → Switch to Haiku
│   └── Already on Haiku? → Proceed but log warning
│
├── SUMMARIZE_CONTEXT (request too large)
│   ├── Conversation > 10 turns? → Summarize older turns
│   ├── System prompt > 2,000 tokens? → Use compressed prompt
│   └── RAG context > 3,000 tokens? → Reduce retrieved chunks
│
└── REJECT (hard limit hit at 100%)
    ├── Session limit? → "Session token limit reached. Start new session."
    ├── Daily user limit? → "Daily limit reached. Resets at midnight UTC."
    └── Monthly tenant limit? → "Monthly allocation exhausted. Contact admin."
```

## When NOT to Use Token Budgets

- **Internal development tools**: Engineering productivity tools where cost is acceptable.
- **Low-volume applications**: < 100 requests/day, manual cost monitoring suffices.
- **Single-turn APIs**: No session accumulation risk, per-request limits are enough.

## Tradeoffs

| Approach | Pros | Cons |
|----------|------|------|
| Hard reject at limit | Predictable costs | Poor UX, mid-conversation cutoff |
| Model downgrade at soft limit | Graceful degradation | Quality drop may frustrate users |
| Context summarization | Maintains model quality | Summary may lose critical details |
| User notification at 60% | User can self-manage | Users ignore warnings |
| Per-request estimation | Prevents overruns | Token estimation is ~10-15% inaccurate |
| Post-hoc tracking only | No latency overhead | No prevention, only monitoring |

## Real-World Examples

- **Intercom's Fin agent**: Per-resolution pricing ($0.99/resolution) implicitly includes a token budget per interaction.
- **GitHub Copilot**: Rate-limits completions per user per hour, effectively a token budget.
- **OpenAI usage tiers**: Organization-level spending caps with email alerts at thresholds.

## Failure Modes

1. **Token estimation drift**: Estimator underestimates by 15%, budget enforcer allows calls that exceed limit. Mitigation: add 20% safety margin to estimates.
2. **Redis downtime**: Budget checks fail, default to ALLOW, costs spike. Mitigation: fail-closed (reject if budget check unavailable).
3. **Timezone misalignment**: "Daily" resets at different times for different users. Mitigation: use UTC consistently.
4. **Budget inheritance conflicts**: Tenant limit hit but user has plenty of daily budget left. Mitigation: communicate which level was hit.
5. **Session ID reuse**: User creates new sessions to bypass session limits. Mitigation: tie session budgets to user daily budgets.

## Source(s) and Further Reading

- Anthropic Token Counting: https://docs.anthropic.com/en/docs/build-with-claude/token-counting
- OpenAI Rate Limits: https://platform.openai.com/docs/guides/rate-limits
- LiteLLM Budget Management: https://docs.litellm.ai/docs/proxy/budget_manager
- Redis INCRBY for atomic counters: https://redis.io/commands/incrby
