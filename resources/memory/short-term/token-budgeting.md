# Token Budgeting

> Every token is money, latency, and attention -- budgeting tokens is budgeting all three simultaneously.

## What It Is

Token budgeting is the practice of pre-allocating fixed token limits to each component of an LLM request: system prompt, tool definitions, RAG context, conversation history, and response output. Without budgeting, agents silently degrade as conversations grow -- context overflows, costs spike, and the LLM starts ignoring critical information buried in the middle.

A token budget operates at three levels:
1. **Request budget** -- how tokens are allocated within a single API call
2. **Session budget** -- total tokens consumed across an entire conversation session
3. **User daily budget** -- cost ceiling per user per day

## How It Works

### The Five Compartments

Every LLM API call has five compartments competing for space:

```
┌─────────────────────────────────────────────────────────────┐
│                    MODEL CONTEXT WINDOW                     │
│                    (e.g., 200K tokens)                      │
│                                                             │
│  ┌─────────────┐  Compartment 1: SYSTEM PROMPT              │
│  │  ~2K tokens  │  Instructions, persona, constraints       │
│  └─────────────┘  Fixed -- same every request               │
│                                                             │
│  ┌─────────────┐  Compartment 2: TOOL DEFINITIONS           │
│  │  ~3K tokens  │  Function schemas, descriptions           │
│  └─────────────┘  Fixed -- changes only when tools change   │
│                                                             │
│  ┌─────────────┐  Compartment 3: RAG CONTEXT                │
│  │  ~5K tokens  │  Retrieved documents, knowledge           │
│  └─────────────┘  Variable -- depends on query              │
│                                                             │
│  ┌─────────────┐  Compartment 4: CONVERSATION HISTORY       │
│  │  ~8K tokens  │  Past turns in current session            │
│  └─────────────┘  Grows with conversation -- must be trimmed │
│                                                             │
│  ┌─────────────┐  Compartment 5: RESPONSE OUTPUT            │
│  │  ~4K tokens  │  Reserved for model response              │
│  └─────────────┘  Set via max_tokens parameter              │
│                                                             │
│  Total used: ~22K / 200K = 11% fill ratio                   │
│  Headroom for growth: plenty                                │
└─────────────────────────────────────────────────────────────┘
```

### Real Numbers for a Production Agent

| Compartment | Tokens | % of 128K | % of 200K | Notes |
|---|---|---|---|---|
| System prompt | 2,000 | 1.6% | 1.0% | Persona + instructions + output format |
| Tool definitions | 3,000 | 2.3% | 1.5% | 10-15 tools with descriptions |
| RAG context | 5,000 | 3.9% | 2.5% | Top 3-5 retrieved chunks |
| Conversation history | 8,000 | 6.3% | 4.0% | ~15-20 turns of dialogue |
| Response output | 4,000 | 3.1% | 2.0% | Typical structured response |
| **Total** | **22,000** | **17.2%** | **11.0%** | Well within optimal range |

### Budget Enforcement Flow

```
User message arrives
       │
       ▼
┌─────────────────────┐
│ Count incoming tokens│
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐     ┌───────────────────┐
│ Check session budget │────▶│ Session exhausted? │
└─────────┬───────────┘     │ → Return error     │
          │ OK              │   "Session limit    │
          ▼                 │    reached"         │
┌─────────────────────┐     └───────────────────┘
│ Check user daily    │
│ budget              │────▶ Daily limit? → Queue or reject
└─────────┬───────────┘
          │ OK
          ▼
┌─────────────────────┐
│ Allocate per-request │
│ budget compartments  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Trim conversation   │
│ if over budget       │
│ (oldest turns first) │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Truncate RAG context │
│ if still over budget │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Assemble & send      │
│ to LLM API           │
└─────────────────────┘
```

## Production Implementation

### Complete Budget Enforcement System

```python
import tiktoken
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import time
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class BudgetConfig:
    """Token budget configuration for a single agent."""
    # Per-request limits
    system_prompt_max: int = 2_000
    tool_definitions_max: int = 3_000
    rag_context_max: int = 5_000
    conversation_max: int = 8_000
    output_max: int = 4_000

    # Session limits
    session_input_max: int = 500_000    # Total input tokens per session
    session_output_max: int = 100_000   # Total output tokens per session

    # User daily limits
    user_daily_input_max: int = 2_000_000   # ~$5/day on Claude Sonnet
    user_daily_output_max: int = 400_000

    # Model constraints
    model_context_window: int = 200_000
    max_fill_ratio: float = 0.40  # Never fill past 40%

    @property
    def effective_input_limit(self) -> int:
        return int(self.model_context_window * self.max_fill_ratio) - self.output_max

    @property
    def per_request_total(self) -> int:
        return (
            self.system_prompt_max
            + self.tool_definitions_max
            + self.rag_context_max
            + self.conversation_max
        )


class TokenCounter:
    """Count tokens for different models."""

    def __init__(self, model: str = "cl100k_base"):
        self.encoder = tiktoken.get_encoding(model)

    def count(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        total = 0
        for msg in messages:
            total += 4  # message overhead (role, content markers)
            total += self.count(msg.get("content", ""))
            if msg.get("name"):
                total += self.count(msg["name"])
        total += 2  # assistant reply priming
        return total

    def count_tools(self, tools: List[dict]) -> int:
        return self.count(json.dumps(tools))


class SessionTracker:
    """Track token usage within a session."""

    def __init__(self, session_id: str, config: BudgetConfig):
        self.session_id = session_id
        self.config = config
        self.input_used = 0
        self.output_used = 0
        self.request_count = 0
        self.started_at = time.time()

    def record(self, input_tokens: int, output_tokens: int):
        self.input_used += input_tokens
        self.output_used += output_tokens
        self.request_count += 1

    @property
    def input_remaining(self) -> int:
        return max(0, self.config.session_input_max - self.input_used)

    @property
    def output_remaining(self) -> int:
        return max(0, self.config.session_output_max - self.output_used)

    @property
    def is_exhausted(self) -> bool:
        return self.input_remaining == 0 or self.output_remaining == 0


class UserDailyTracker:
    """Track token usage per user per day. 
    
    In production, back this with Redis:
        key: f"tokens:{user_id}:{date}" 
        TTL: 86400
    """

    def __init__(self):
        self._usage: Dict[str, Dict[str, int]] = {}

    def _key(self, user_id: str) -> str:
        from datetime import date
        return f"{user_id}:{date.today().isoformat()}"

    def record(self, user_id: str, input_tokens: int, output_tokens: int):
        key = self._key(user_id)
        if key not in self._usage:
            self._usage[key] = {"input": 0, "output": 0}
        self._usage[key]["input"] += input_tokens
        self._usage[key]["output"] += output_tokens

    def get_remaining(self, user_id: str, config: BudgetConfig) -> Tuple[int, int]:
        key = self._key(user_id)
        used = self._usage.get(key, {"input": 0, "output": 0})
        return (
            max(0, config.user_daily_input_max - used["input"]),
            max(0, config.user_daily_output_max - used["output"]),
        )


class TokenBudgetEnforcer:
    """Main budget enforcement class."""

    def __init__(self, config: BudgetConfig):
        self.config = config
        self.counter = TokenCounter()
        self.sessions: Dict[str, SessionTracker] = {}
        self.daily_tracker = UserDailyTracker()

    def get_session(self, session_id: str) -> SessionTracker:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionTracker(session_id, self.config)
        return self.sessions[session_id]

    def enforce(
        self,
        user_id: str,
        session_id: str,
        system_prompt: str,
        tools: List[dict],
        rag_docs: List[str],
        conversation: List[Dict[str, str]],
        current_query: str,
    ) -> Dict:
        """Enforce budget and return trimmed components."""

        # 1. Check session budget
        session = self.get_session(session_id)
        if session.is_exhausted:
            raise BudgetExhaustedError(
                f"Session {session_id} exhausted: "
                f"{session.input_used} input, {session.output_used} output"
            )

        # 2. Check user daily budget
        daily_input_rem, daily_output_rem = self.daily_tracker.get_remaining(
            user_id, self.config
        )
        if daily_input_rem == 0:
            raise BudgetExhaustedError(f"User {user_id} daily input budget exhausted")

        # 3. Count fixed components
        system_tokens = self.counter.count(system_prompt)
        tool_tokens = self.counter.count_tools(tools) if tools else 0
        query_tokens = self.counter.count(current_query)

        if system_tokens > self.config.system_prompt_max:
            logger.warning(
                f"System prompt ({system_tokens}) exceeds budget ({self.config.system_prompt_max})"
            )

        if tool_tokens > self.config.tool_definitions_max:
            logger.warning(
                f"Tools ({tool_tokens}) exceed budget ({self.config.tool_definitions_max})"
            )

        fixed_tokens = system_tokens + tool_tokens + query_tokens

        # 4. Budget remaining for RAG + conversation
        available = self.config.effective_input_limit - fixed_tokens

        # 5. Allocate RAG (priority over old conversation)
        trimmed_rag = self._trim_rag(rag_docs, min(available, self.config.rag_context_max))
        rag_tokens = sum(self.counter.count(doc) for doc in trimmed_rag)

        # 6. Allocate conversation (trim oldest first)
        convo_budget = min(available - rag_tokens, self.config.conversation_max)
        trimmed_convo = self._trim_conversation(conversation, convo_budget)
        convo_tokens = self.counter.count_messages(trimmed_convo)

        total_input = fixed_tokens + rag_tokens + convo_tokens
        fill_ratio = total_input / self.config.model_context_window

        logger.info(
            f"Budget: {total_input} tokens, {fill_ratio:.1%} fill, "
            f"system={system_tokens}, tools={tool_tokens}, "
            f"rag={rag_tokens}, convo={convo_tokens}, query={query_tokens}"
        )

        return {
            "system_prompt": system_prompt,
            "tools": tools,
            "rag_docs": trimmed_rag,
            "conversation": trimmed_convo,
            "current_query": current_query,
            "budget_report": {
                "total_input_tokens": total_input,
                "fill_ratio": fill_ratio,
                "output_reserved": self.config.output_max,
                "session_input_remaining": session.input_remaining,
                "daily_input_remaining": daily_input_rem,
                "turns_dropped": len(conversation) - len(trimmed_convo),
                "rag_docs_dropped": len(rag_docs) - len(trimmed_rag),
            },
        }

    def _trim_rag(self, docs: List[str], max_tokens: int) -> List[str]:
        """Keep top-ranked docs within budget."""
        trimmed = []
        used = 0
        for doc in docs:  # Assumes docs are ranked by relevance
            doc_tokens = self.counter.count(doc)
            if used + doc_tokens > max_tokens:
                break
            trimmed.append(doc)
            used += doc_tokens
        return trimmed

    def _trim_conversation(
        self, conversation: List[Dict[str, str]], max_tokens: int
    ) -> List[Dict[str, str]]:
        """Keep most recent turns within budget, drop oldest first."""
        if not conversation:
            return []
        trimmed = []
        used = 0
        for msg in reversed(conversation):
            msg_tokens = self.counter.count(msg.get("content", "")) + 4
            if used + msg_tokens > max_tokens:
                break
            trimmed.insert(0, msg)
            used += msg_tokens
        return trimmed


class BudgetExhaustedError(Exception):
    pass


# ─── Usage Example ───

config = BudgetConfig(
    system_prompt_max=2_000,
    tool_definitions_max=3_000,
    rag_context_max=5_000,
    conversation_max=8_000,
    output_max=4_000,
    model_context_window=200_000,
    max_fill_ratio=0.40,
    session_input_max=500_000,
    user_daily_input_max=2_000_000,
)

enforcer = TokenBudgetEnforcer(config)

result = enforcer.enforce(
    user_id="user_123",
    session_id="session_abc",
    system_prompt="You are a helpful assistant...",
    tools=[{"type": "function", "function": {"name": "search", "parameters": {}}}],
    rag_docs=["Document 1 content...", "Document 2 content..."],
    conversation=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": "Tell me about X"},
        {"role": "assistant", "content": "X is..."},
    ],
    current_query="What about Y?",
)

print(result["budget_report"])
# {
#     "total_input_tokens": 312,
#     "fill_ratio": 0.00156,
#     "output_reserved": 4000,
#     "session_input_remaining": 499688,
#     "daily_input_remaining": 1999688,
#     "turns_dropped": 0,
#     "rag_docs_dropped": 0,
# }
```

### Redis-Backed Daily Tracker (Production)

```python
import redis
from datetime import date

class RedisDailyTracker:
    def __init__(self, redis_client: redis.Redis):
        self.r = redis_client

    def record(self, user_id: str, input_tokens: int, output_tokens: int):
        key = f"tokens:{user_id}:{date.today().isoformat()}"
        pipe = self.r.pipeline()
        pipe.hincrby(key, "input", input_tokens)
        pipe.hincrby(key, "output", output_tokens)
        pipe.expire(key, 86400)  # Auto-expire at end of day
        pipe.execute()

    def get_remaining(self, user_id: str, config: BudgetConfig) -> Tuple[int, int]:
        key = f"tokens:{user_id}:{date.today().isoformat()}"
        used = self.r.hgetall(key)
        input_used = int(used.get(b"input", 0))
        output_used = int(used.get(b"output", 0))
        return (
            max(0, config.user_daily_input_max - input_used),
            max(0, config.user_daily_output_max - output_used),
        )
```

## Decision Tree

```
Setting up token budgets?
  │
  ├─ What model are you using?
  │   ├─ 128K context → max_fill_ratio = 0.35 (conservative)
  │   ├─ 200K context → max_fill_ratio = 0.40 (standard)
  │   └─ 1M context  → max_fill_ratio = 0.30 (Gemini -- even more middle-loss)
  │
  ├─ How many tools?
  │   ├─ 1-5 tools → 1K token budget
  │   ├─ 5-15 tools → 3K token budget
  │   └─ 15+ tools → Consider dynamic tool loading
  │
  ├─ Do you use RAG?
  │   ├─ No RAG → Allocate extra to conversation
  │   ├─ Light RAG (1-3 docs) → 3K budget
  │   └─ Heavy RAG (5+ docs) → 5-10K budget, summarize before injecting
  │
  └─ What's your cost target?
      ├─ < $0.01/request → Keep under 10K input tokens
      ├─ < $0.05/request → Keep under 50K input tokens
      └─ No limit → Still budget for quality, not cost
```

## When NOT to Use

- **Prototyping** -- during rapid iteration, budget enforcement adds friction. Just monitor costs.
- **Single-turn applications** -- if there is no conversation history or RAG, budget is trivially simple. No framework needed.
- **Unlimited-budget internal tools** -- some internal tools have no cost pressure. Still budget for quality (fill ratio), but skip cost tracking.

## Tradeoffs

| Strategy | Quality Impact | Cost Impact | Implementation Effort |
|---|---|---|---|
| No budgeting | Degrades over time | Unpredictable | None |
| Request-only budgeting | Stable per request | Predictable per request | Low |
| Request + session | Stable, session-capped | Capped per session | Medium |
| Full 3-tier (request + session + daily) | Stable, fully controlled | Fully predictable | Medium-High |
| Dynamic budgeting (adjust per-request) | Optimal | Optimal | High |

## Real-World Examples

1. **SaaS chatbot (free tier)** -- 50K tokens/day per user, 10K per session, 2K conversation per request. Summarize after 10 turns. Cost: ~$0.15/user/day on Claude Sonnet.

2. **Enterprise support agent** -- 2M tokens/day per user (generous), 500K per session, 15K conversation per request. RAG from knowledge base (5K). Cost: ~$6/user/day on Claude Sonnet.

3. **Coding assistant** -- No strict daily limit. 100K per session. Large RAG context (10K for code files). Smaller conversation (5K). Cost varies wildly with code size.

## Failure Modes

| Failure | Symptom | Fix |
|---|---|---|
| No output budget reserved | Response truncates mid-sentence | Always set `max_tokens` and subtract from available input |
| Conversation grows unchecked | Costs spike, quality degrades | Trim oldest turns, summarize when over budget |
| RAG docs too large | Crowd out conversation history | Summarize RAG docs before injection, limit to top-K |
| Tool definitions bloated | 8K+ tokens just for tools | Prune tool descriptions, use dynamic tool loading |
| Budget too tight | Agent loses important context | Monitor "turns_dropped" metric, increase if > 30% |

## Source(s) and Further Reading

- [Anthropic Token Counting Guide](https://docs.anthropic.com/en/docs/build-with-claude/token-counting)
- [OpenAI Cookbook: Managing Tokens](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
- [State of AI Agent Memory 2026](https://mem0.ai) -- Mem0 research report
