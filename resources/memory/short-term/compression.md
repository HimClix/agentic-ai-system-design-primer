# Context Compression

> Compression lets you keep 20 turns of conversation in 4 turns of space -- trading perfect recall for 70-94% cost reduction.

## What It Is

Context compression is the set of techniques that reduce the token count of conversation history, retrieved documents, and other context without losing critical information. When a 50-turn customer support conversation would consume 40K tokens raw, compression can bring it down to 4K-8K tokens -- a **5-20x reduction** translating to **70-94% cost savings** and significantly better LLM attention.

The core tension: compression is lossy. You always lose some detail. The art is losing the right details -- keeping facts, decisions, and commitments while discarding chitchat, repetition, and resolved sub-topics.

## How It Works

### Four Core Techniques

```
┌─────────────────────────────────────────────────────────────┐
│              COMPRESSION TECHNIQUES                         │
│                                                             │
│  1. SLIDING WINDOW                                          │
│     Keep last N turns, drop oldest                          │
│     ┌──┬──┬──┬──┬──┬──┬──┬──┐                               │
│     │T1│T2│T3│T4│T5│T6│T7│T8│  → Keep T5-T8, drop T1-T4    │
│     └──┴──┴──┴──┴──┴──┴──┴──┘                               │
│     Compression: ~50% (keep half the turns)                 │
│     Loss: Everything before the window is gone              │
│                                                             │
│  2. CONVERSATION SUMMARIZATION                              │
│     Summarize older turns into a paragraph                  │
│     ┌──┬──┬──┬──┬──┬──┬──┬──┐                               │
│     │T1│T2│T3│T4│T5│T6│T7│T8│                               │
│     └──┴──┴──┴──┴──┴──┴──┴──┘                               │
│          ↓                                                  │
│     ┌─────────────┬──┬──┬──┐                                │
│     │Summary(T1-5)│T6│T7│T8│                                │
│     └─────────────┴──┴──┴──┘                                │
│     Compression: 5-10x on summarized portion                │
│     Loss: Details of summarized turns                       │
│                                                             │
│  3. SELECTIVE RETENTION                                     │
│     Keep only messages matching criteria                    │
│     - Messages with tool calls/results                      │
│     - Messages containing decisions or commitments          │
│     - Messages with entities (names, IDs, dates)            │
│     Compression: 3-8x                                       │
│     Loss: Conversational flow, nuance                       │
│                                                             │
│  4. MAP-REDUCE SUMMARIZATION                                │
│     For very long contexts (30K+ tokens)                    │
│     Split → Summarize each chunk → Merge summaries          │
│     Compression: 10-20x                                     │
│     Loss: Cross-chunk relationships                         │
└─────────────────────────────────────────────────────────────┘
```

### Compression Ratios and Cost Impact

| Technique | Compression Ratio | Cost Reduction | Fidelity | Best For |
|---|---|---|---|---|
| Sliding window (keep last 10) | 2-5x | 50-80% | Low (loses old context) | Simple chatbots |
| Summarization (running) | 5-10x | 80-90% | Medium | General agents |
| Summarization (hierarchical) | 10-15x | 90-94% | Medium | Very long sessions |
| Selective retention | 3-8x | 70-88% | High (for retained items) | Tool-heavy agents |
| Map-reduce | 10-20x | 90-95% | Low-Medium | Document processing |
| Combined (summary + window) | 8-15x | 88-94% | Medium-High | Production agents |

### When to Compress vs When to Use RAG

```
Question: How should I handle growing context?
  │
  ├─ Is the information from the current conversation?
  │   └─ YES → COMPRESS (summarize older turns)
  │
  ├─ Is the information from external sources?
  │   └─ YES → RAG (retrieve on demand, don't store in context)
  │
  ├─ Is the information from past conversations?
  │   └─ YES → LONG-TERM MEMORY (store in vector DB, retrieve relevant bits)
  │
  └─ Is it a mix?
      └─ COMPRESS current conversation + RAG for external + LONG-TERM for past
```

**Key rule**: Compression handles the current session. RAG and long-term memory handle everything else. Don't compress what should be in a database.

## Production Implementation

### 1. Sliding Window with Summary Prefix

The most common production pattern: keep the last N turns verbatim, summarize everything before them.

```python
from typing import List, Dict, Optional
from openai import OpenAI  # or anthropic

client = OpenAI()

SUMMARIZE_PROMPT = """Summarize the following conversation into a concise paragraph.
Preserve:
- All decisions made
- All commitments or action items
- Key facts and numbers mentioned
- User preferences expressed
- Current state of any ongoing task

Do NOT preserve:
- Greetings and small talk
- Repeated information
- Exploratory questions that were resolved
- Verbose explanations that led to a conclusion

Conversation:
{conversation}

Concise summary:"""


class SlidingWindowWithSummary:
    def __init__(
        self,
        window_size: int = 10,       # Keep last 10 turns verbatim
        summary_model: str = "gpt-4o-mini",  # Cheap model for summarization
        max_summary_tokens: int = 500,
    ):
        self.window_size = window_size
        self.summary_model = summary_model
        self.max_summary_tokens = max_summary_tokens
        self.summary: Optional[str] = None
        self.full_history: List[Dict[str, str]] = []

    def add_turn(self, role: str, content: str):
        self.full_history.append({"role": role, "content": content})

    def get_context(self) -> List[Dict[str, str]]:
        """Return compressed conversation context."""
        messages = []

        # Add summary of older turns if available
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"Summary of earlier conversation:\n{self.summary}"
            })

        # Add recent turns verbatim
        recent = self.full_history[-self.window_size:]
        messages.extend(recent)

        return messages

    async def compress_if_needed(self):
        """Compress when history exceeds window size."""
        if len(self.full_history) <= self.window_size:
            return  # Nothing to compress

        # Turns that will be dropped from verbatim window
        to_summarize = self.full_history[:-self.window_size]

        # Build conversation text for summarization
        convo_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in to_summarize
        )

        # If there's an existing summary, include it for continuity
        if self.summary:
            convo_text = f"Previous summary: {self.summary}\n\nNew turns:\n{convo_text}"

        # Generate summary using cheap model
        response = client.chat.completions.create(
            model=self.summary_model,
            messages=[{"role": "user", "content": SUMMARIZE_PROMPT.format(conversation=convo_text)}],
            max_tokens=self.max_summary_tokens,
            temperature=0.0,
        )
        self.summary = response.choices[0].message.content

        # Stats
        original_tokens = sum(len(msg["content"].split()) for msg in to_summarize) * 1.3
        summary_tokens = len(self.summary.split()) * 1.3
        print(f"Compressed {len(to_summarize)} turns: {int(original_tokens)} → {int(summary_tokens)} tokens "
              f"({(1 - summary_tokens/original_tokens)*100:.0f}% reduction)")


# Usage
memory = SlidingWindowWithSummary(window_size=10)

# Simulate 25-turn conversation
for i in range(25):
    memory.add_turn("user", f"Turn {i}: User message about topic {i % 5}")
    memory.add_turn("assistant", f"Turn {i}: Assistant response about topic {i % 5}")
    if len(memory.full_history) > memory.window_size:
        await memory.compress_if_needed()

context = memory.get_context()
# Returns: summary of turns 0-14 + verbatim turns 15-24
```

### 2. Selective Retention (Keep Important Messages)

```python
import re
from dataclasses import dataclass

@dataclass
class RetentionRule:
    name: str
    check: callable
    priority: int  # Higher = more important to keep

RETENTION_RULES = [
    RetentionRule(
        name="tool_calls",
        check=lambda msg: msg.get("tool_calls") is not None or msg.get("role") == "tool",
        priority=10,
    ),
    RetentionRule(
        name="contains_decision",
        check=lambda msg: any(
            kw in msg.get("content", "").lower()
            for kw in ["decided", "confirmed", "agreed", "approved", "let's go with", "i'll do"]
        ),
        priority=9,
    ),
    RetentionRule(
        name="contains_entity",
        check=lambda msg: bool(
            re.search(r"(?:order|ticket|id|account)[\s#:_-]*\w{4,}", msg.get("content", ""), re.I)
        ),
        priority=8,
    ),
    RetentionRule(
        name="contains_number",
        check=lambda msg: bool(
            re.search(r"\$[\d,]+|\d+%|\d{4,}", msg.get("content", ""))
        ),
        priority=7,
    ),
    RetentionRule(
        name="error_or_problem",
        check=lambda msg: any(
            kw in msg.get("content", "").lower()
            for kw in ["error", "failed", "issue", "problem", "broken", "bug"]
        ),
        priority=8,
    ),
    RetentionRule(
        name="recent_message",
        check=lambda msg: msg.get("_is_recent", False),
        priority=6,
    ),
]


class SelectiveRetention:
    def __init__(self, max_messages: int = 20, recent_window: int = 6):
        self.max_messages = max_messages
        self.recent_window = recent_window

    def compress(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Keep only important messages within budget."""
        if len(messages) <= self.max_messages:
            return messages

        # Mark recent messages
        for i, msg in enumerate(messages):
            msg["_is_recent"] = i >= len(messages) - self.recent_window

        # Score each message
        scored = []
        for i, msg in enumerate(messages):
            score = 0
            matched_rules = []
            for rule in RETENTION_RULES:
                if rule.check(msg):
                    score += rule.priority
                    matched_rules.append(rule.name)
            scored.append((i, score, matched_rules, msg))

        # Sort by score (descending), keep top N
        scored.sort(key=lambda x: x[1], reverse=True)
        kept_indices = sorted([s[0] for s in scored[:self.max_messages]])

        result = [messages[i] for i in kept_indices]

        # Clean up temp fields
        for msg in result:
            msg.pop("_is_recent", None)
        for msg in messages:
            msg.pop("_is_recent", None)

        dropped = len(messages) - len(result)
        print(f"Selective retention: kept {len(result)}/{len(messages)} messages (dropped {dropped})")
        return result
```

### 3. Map-Reduce for Large Documents

```python
async def map_reduce_summarize(
    text: str,
    chunk_size: int = 4000,
    model: str = "gpt-4o-mini",
) -> str:
    """Summarize very large text using map-reduce pattern."""

    # MAP: Split into chunks and summarize each
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": f"Summarize this section concisely, preserving key facts and decisions:\n\n{chunk}"
            }],
            max_tokens=500,
            temperature=0.0,
        )
        chunk_summaries.append(response.choices[0].message.content)
        print(f"Map step {i+1}/{len(chunks)}: {len(chunk.split())} words → {len(chunk_summaries[-1].split())} words")

    # REDUCE: Merge chunk summaries into final summary
    merged = "\n\n---\n\n".join(chunk_summaries)
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": (
                "Merge these section summaries into a single coherent summary. "
                "Preserve all key facts, decisions, and action items. "
                "Remove redundancies.\n\n"
                f"{merged}"
            )
        }],
        max_tokens=1000,
        temperature=0.0,
    )

    final = response.choices[0].message.content
    original_words = len(words)
    final_words = len(final.split())
    print(f"Map-reduce complete: {original_words} words → {final_words} words "
          f"({original_words/final_words:.0f}x compression)")
    return final
```

## Decision Tree

```
How should I compress my context?
  │
  ├─ How many conversation turns?
  │   ├─ < 10 → No compression needed
  │   ├─ 10-30 → Sliding window + summary of old turns
  │   └─ 30+ → Hierarchical summarization or selective retention
  │
  ├─ Is tool call history important?
  │   ├─ YES → Use selective retention (always keep tool turns)
  │   └─ NO → Simple summarization is fine
  │
  ├─ How much does fidelity matter?
  │   ├─ High (legal, medical) → Selective retention + wide window
  │   ├─ Medium (support, coding) → Summary + 10-turn window
  │   └─ Low (casual chat) → Aggressive summarization + 5-turn window
  │
  └─ Budget for compression itself?
      ├─ Minimal → Sliding window (no LLM call, just truncate)
      ├─ Standard → Summarization with gpt-4o-mini ($0.0001 per summary)
      └─ High → Selective retention + hierarchical summarization
```

## When NOT to Use

- **Short conversations** (< 10 turns) -- overhead of compression exceeds savings.
- **When exact quotes matter** -- compression loses verbatim text. Use full history or retrieval.
- **Audit/compliance contexts** -- store full history in database; compress only the LLM context, not the record of truth.
- **When latency is critical** -- summarization adds 200-500ms per compression step. Use sliding window (no LLM call) instead.

## Tradeoffs

| Technique | Compression | Fidelity | Latency Cost | Dollar Cost | Complexity |
|---|---|---|---|---|---|
| Sliding window | 2-5x | Low (loses old) | 0ms | $0 | Very low |
| Running summary | 5-10x | Medium | 200-500ms | ~$0.0001/summary | Low |
| Hierarchical summary | 10-15x | Medium | 500-1500ms | ~$0.001/compression | Medium |
| Selective retention | 3-8x | High (for kept items) | 5-20ms | $0 | Medium |
| Map-reduce | 10-20x | Low-Medium | 2-10s | ~$0.005/document | Medium |
| Combined best-of | 8-15x | Medium-High | 200-600ms | ~$0.0002/step | High |

## Real-World Examples

1. **Customer support agent** -- 40-turn conversation about a billing issue. Sliding window (last 10 turns) + summary of turns 1-30. Summary captures: customer name, account number, billing dispute amount, steps already tried. 40 turns (~30K tokens) compressed to ~6K tokens. **5x compression, 80% cost reduction**.

2. **Coding assistant** -- 60-turn pair programming session. Selective retention keeps all code snippets and error messages, drops explanatory dialogue. 60 turns (~50K tokens) compressed to ~10K tokens. **5x compression**, but keeps all code.

3. **Research agent** -- Processes 10 papers (200K tokens total). Map-reduce summarizes each paper to 500 tokens, then merges. 200K tokens → 5K tokens. **40x compression, 97.5% reduction**. Loses individual paper details but retains cross-paper themes.

## Failure Modes

| Failure | Symptom | Fix |
|---|---|---|
| Over-compression | Agent forgets critical facts from early conversation | Wider window, selective retention for key facts |
| Summary hallucination | Summary includes facts not in original conversation | Use temperature=0, add "only include facts explicitly stated" |
| Compression latency | Visible delay when conversation gets long | Compress asynchronously, pre-compute summaries |
| Lost tool context | Agent re-calls tools it already called | Always retain tool call/result pairs in selective retention |
| Cascading summaries | Summary of summary of summary -- quality degrades | Cap hierarchy depth at 2-3 levels, expand window |

## Source(s) and Further Reading

- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) -- Liu et al., 2023
- [LLMLingua: Compressing Prompts for Accelerated Inference](https://arxiv.org/abs/2310.05736) -- Jiang et al., 2023
- [State of AI Agent Memory 2026](https://mem0.ai) -- Mem0 research report
- [LangChain Conversation Summary Memory](https://python.langchain.com/docs/modules/memory/)
