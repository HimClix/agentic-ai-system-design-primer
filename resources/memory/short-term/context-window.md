# Context Window as Working Memory

> The context window is the only memory an LLM actually "sees" during inference -- everything else is just a retrieval pipeline feeding into it.

## What It Is

The context window is the fixed-size token buffer that holds everything the LLM processes in a single request: system prompt, tools, conversation history, retrieved documents, and the response it generates. It is the agent's working memory -- the scratchpad where all reasoning happens.

Every token in the context window costs money and latency. More importantly, not all positions in the window are created equal. LLMs exhibit a **U-shaped attention curve**: they attend strongly to the beginning and end of the context but lose fidelity in the middle. This has profound implications for how you structure agent memory.

### Current Context Window Sizes (2025-2026)

| Model | Context Window | Effective Sweet Spot |
|---|---|---|
| GPT-4o | 128K tokens | 12K-50K tokens |
| Claude Sonnet 4 | 200K tokens | 20K-80K tokens |
| Claude Opus 4 | 200K tokens | 20K-80K tokens |
| Gemini 2.5 Pro | 1M tokens | 100K-400K tokens |
| Llama 3.3 70B | 128K tokens | 12K-50K tokens |

"Effective sweet spot" is where you get reliable performance without significant quality degradation or cost explosion.

## How It Works

### The U-Shaped Attention Curve

Research consistently shows that LLM attention follows a U-shape across the context window:

```
Attention
Fidelity
    │
100%│██                                              ██
    │████                                          ████
 80%│██████                                      ██████
    │████████                                  ████████
 60%│          ████████████████████████████████
    │              ▼ "Lost in the Middle" ▼
 40%│
    │
 20%│
    │
  0%├───────────────────────────────────────────────────
    0%    20%    40%    60%    80%    100%
                   Position in Context
```

**Key findings:**
- **Primacy effect**: The first ~10% of context gets disproportionate attention. System prompts and critical instructions go here.
- **Recency effect**: The last ~10-15% (most recent conversation turns, the query) gets strong attention.
- **Middle sag**: Content in the 30-70% range can be partially ignored, especially in longer contexts. This is where retrieved documents often land -- and get missed.
- **Critical threshold**: When the context window fills to **60-80%**, overall fidelity drops noticeably. At 90%+, quality degrades significantly.

### Optimal Operating Range: 10-40% Fill

The safest operating range is **10-40% of the context window**:

| Fill Ratio | Quality | Cost | Recommendation |
|---|---|---|---|
| < 10% | Excellent, but under-utilizing | Low | Fine for simple tasks |
| **10-40%** | **Excellent** | **Moderate** | **Optimal for production** |
| 40-60% | Good, minor degradation | Higher | Acceptable with careful positioning |
| 60-80% | Noticeable degradation | High | Compress or trim |
| 80-100% | Significant degradation | Very high | Avoid -- you are losing quality |

For a 200K-token model, optimal is 20K-80K tokens in the context. For a 128K model, aim for 12K-50K.

### Position Effects: What Goes Where

```
┌────────────────────────────────────────────────────┐
│  POSITION 1 (Start)         ← HIGHEST ATTENTION    │
│  ┌──────────────────────────────────────────────┐  │
│  │ System prompt                                │  │
│  │ Critical instructions                        │  │
│  │ Output format requirements                   │  │
│  │ Safety constraints                           │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  POSITION 2 (Early middle)  ← HIGH ATTENTION       │
│  ┌──────────────────────────────────────────────┐  │
│  │ Tool definitions                             │  │
│  │ Available actions                            │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  POSITION 3 (Middle)        ← LOWER ATTENTION      │
│  ┌──────────────────────────────────────────────┐  │
│  │ RAG retrieved documents                      │  │
│  │ Background context                           │  │
│  │ Historical conversation (older turns)        │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  POSITION 4 (Late middle)   ← MODERATE ATTENTION   │
│  ┌──────────────────────────────────────────────┐  │
│  │ Recent conversation history                  │  │
│  │ Summarized earlier context                   │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  POSITION 5 (End)           ← HIGHEST ATTENTION    │
│  ┌──────────────────────────────────────────────┐  │
│  │ Current user message                         │  │
│  │ Specific question/task                       │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
```

## Production Implementation

### Token Budget Template

```python
from dataclasses import dataclass
from enum import Enum

class ModelConfig(Enum):
    GPT_4O = {"max_tokens": 128_000, "output_limit": 16_384, "cost_per_1k_input": 0.0025}
    CLAUDE_SONNET = {"max_tokens": 200_000, "output_limit": 8_192, "cost_per_1k_input": 0.003}
    CLAUDE_OPUS = {"max_tokens": 200_000, "output_limit": 8_192, "cost_per_1k_input": 0.015}

@dataclass
class TokenBudget:
    """Allocate tokens across context window components."""
    model_max: int
    output_reserve: int

    # Fixed allocations
    system_prompt: int = 2_000
    tool_definitions: int = 3_000

    # Variable allocations
    rag_context: int = 5_000
    conversation_history: int = 8_000

    # Safety margin (never fill past this)
    max_fill_ratio: float = 0.4

    @property
    def effective_input_limit(self) -> int:
        """Max input tokens respecting fill ratio."""
        return int(self.model_max * self.max_fill_ratio) - self.output_reserve

    @property
    def available_for_content(self) -> int:
        """Tokens available after fixed allocations."""
        fixed = self.system_prompt + self.tool_definitions
        return self.effective_input_limit - fixed

    def allocate(self) -> dict:
        available = self.available_for_content
        # If conversation + RAG exceed available, trim conversation first
        rag = min(self.rag_context, available)
        convo = min(self.conversation_history, available - rag)

        return {
            "system_prompt": self.system_prompt,
            "tool_definitions": self.tool_definitions,
            "rag_context": rag,
            "conversation_history": convo,
            "output_reserve": self.output_reserve,
            "total_input": self.system_prompt + self.tool_definitions + rag + convo,
            "fill_ratio": (self.system_prompt + self.tool_definitions + rag + convo) / self.model_max,
            "headroom": self.effective_input_limit - (self.system_prompt + self.tool_definitions + rag + convo),
        }


# Example: Claude Sonnet 4 at 40% fill
budget = TokenBudget(
    model_max=200_000,
    output_reserve=8_192,
    system_prompt=2_000,
    tool_definitions=3_000,
    rag_context=5_000,
    conversation_history=8_000,
)

alloc = budget.allocate()
# total_input: 18,000 tokens
# fill_ratio: 9% -- well within optimal range
# headroom: 53,808 tokens for growth
```

### Context Assembly with Position Optimization

```python
from typing import List, Dict

def assemble_context(
    system_prompt: str,
    tools: List[dict],
    rag_docs: List[str],
    conversation: List[Dict[str, str]],
    current_query: str,
    budget: TokenBudget,
) -> List[Dict[str, str]]:
    """Assemble context with optimal positioning."""
    alloc = budget.allocate()
    messages = []

    # POSITION 1: System prompt (highest attention)
    messages.append({
        "role": "system",
        "content": system_prompt  # Keep under alloc["system_prompt"] tokens
    })

    # POSITION 2: Tools are defined separately in API calls
    # (tool definitions go in the tools parameter, not messages)

    # POSITION 3: RAG context (middle -- lower attention)
    # Place in an early assistant/user exchange so it lands in the middle
    if rag_docs:
        rag_block = "\n---\n".join(rag_docs[:3])  # Limit to top 3
        rag_tokens = count_tokens(rag_block)
        if rag_tokens <= alloc["rag_context"]:
            messages.append({
                "role": "user",
                "content": f"Reference information:\n{rag_block}"
            })
            messages.append({
                "role": "assistant",
                "content": "I've noted the reference information. How can I help?"
            })

    # POSITION 4: Conversation history (trim oldest first)
    convo_tokens = 0
    trimmed_convo = []
    for msg in reversed(conversation):  # Start from most recent
        msg_tokens = count_tokens(msg["content"])
        if convo_tokens + msg_tokens > alloc["conversation_history"]:
            break
        trimmed_convo.insert(0, msg)
        convo_tokens += msg_tokens
    messages.extend(trimmed_convo)

    # POSITION 5: Current query (highest attention -- end position)
    messages.append({"role": "user", "content": current_query})

    return messages
```

## Decision Tree

```
Need to decide context window strategy?
  │
  ├─ How many turns does a typical conversation last?
  │   ├─ < 5 turns → Raw conversation, no compression needed
  │   ├─ 5-20 turns → Sliding window with oldest-first trimming
  │   └─ > 20 turns → Summarization + sliding window (see compression.md)
  │
  ├─ Does the agent use RAG?
  │   ├─ YES → Budget 5K-10K tokens for retrieved docs
  │   └─ NO → Allocate that budget to conversation history
  │
  ├─ How many tools does the agent have?
  │   ├─ < 5 tools → ~1K tokens for tool definitions
  │   ├─ 5-20 tools → ~3K tokens
  │   └─ > 20 tools → 5K+ tokens -- consider dynamic tool loading
  │
  └─ What's the response complexity?
      ├─ Short answers → Reserve 1K output tokens
      ├─ Code generation → Reserve 4K-8K output tokens
      └─ Long documents → Reserve 8K-16K output tokens
```

## When NOT to Use (Raw Context Window)

- **Conversations exceeding 30+ turns** -- you will hit quality degradation. Use compression.
- **Multi-session agents** -- the context window resets between API calls. You need long-term memory.
- **Cost-sensitive applications** -- large contexts are expensive. A 100K-token request on GPT-4o costs $0.25. Compression can cut this 80%+.
- **When you need exact recall from the middle** -- position effects mean middle content may be missed. Use explicit retrieval instead.

## Tradeoffs

| Strategy | Quality | Cost | Latency | Complexity |
|---|---|---|---|---|
| Small context (< 20%) | Excellent recall | Low | Fast | Low |
| Moderate context (20-40%) | Very good recall | Moderate | Moderate | Low |
| Large context (40-70%) | Good, some middle-loss | High | Slower | Medium (need positioning) |
| Max context (70-100%) | Degraded, unreliable middle | Very high | Slow | High (need compression) |
| Compression + small context | Good (depends on summary quality) | Low | Fast + compression overhead | Medium |

## Real-World Examples

1. **Customer support bot** -- System prompt (1.5K) + tools (2K) + customer history from CRM (3K in RAG) + last 5 conversation turns (2K) + current question. Total: ~8.5K on a 128K model = 6.6% fill. Excellent quality.

2. **Code review agent** -- System prompt (2K) + tools (4K) + diff content (varies, 1K-50K) + repo context from RAG (5K) + PR comments (2K). Total: 14K-63K. For large diffs, this pushes into the 50% range -- consider chunking the diff.

3. **Research agent** -- System prompt (2K) + tools (3K) + 10 retrieved papers (20K) + conversation (5K). Total: 30K on a 200K model = 15%. Comfortable, but those 10 papers in the middle may suffer from position effects. Better to summarize them first.

## Failure Modes

| Failure | Symptom | Fix |
|---|---|---|
| Context overflow | API returns error or silently truncates | Enforce token budget before calling API |
| Lost in the middle | Agent ignores RAG docs placed in middle | Move critical info to start or end; summarize RAG docs |
| Stale conversation | Agent references early turns inaccurately | Sliding window with summarization of dropped turns |
| Output truncation | Response cuts off mid-sentence | Reserve sufficient output tokens in budget |
| Cost explosion | Bills spike for long conversations | Session token budgets, compression after N turns |

## Source(s) and Further Reading

- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) -- Liu et al., 2023
- [Anthropic Claude Context Window Documentation](https://docs.anthropic.com/en/docs/build-with-claude/context-windows)
- [State of AI Agent Memory 2026](https://mem0.ai) -- Mem0 research report
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) -- for counting tokens
