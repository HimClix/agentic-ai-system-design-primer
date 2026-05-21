# Conversation Summarization Strategies

> The best summary is invisible -- the agent remembers everything important without the user knowing half the conversation was compressed away.

## What It Is

Conversation summarization is the specific compression technique of distilling multi-turn dialogue into a shorter representation that preserves the essential information. Unlike generic text summarization, conversation summarization must handle turn-taking dynamics, preserve speaker attribution, maintain temporal order of decisions, and track evolving state (a user might change their mind).

Three main strategies exist, each with distinct characteristics:

| Strategy | How It Works | When to Use |
|---|---|---|
| **Running Summary** | Incrementally update a single summary after each turn | Most conversations, low latency requirement |
| **Hierarchical Summary** | Summarize chunks, then summarize summaries | Very long conversations (50+ turns) |
| **Entity Extraction** | Extract structured facts instead of prose summary | When facts matter more than narrative |

## How It Works

### Strategy 1: Running Summary

Maintain a single evolving summary that gets updated every N turns.

```
Turn 1-5:   [Summary v1] ← Initial summary
Turn 6-10:  [Summary v2] ← Summary v1 + new turns summarized together
Turn 11-15: [Summary v3] ← Summary v2 + new turns summarized together
...

Context at turn 20:
┌──────────────────────────────┐
│ Summary v3 (turns 1-15)     │  ~300 tokens
│ Turn 16 (verbatim)          │  ~200 tokens
│ Turn 17 (verbatim)          │  ~200 tokens
│ Turn 18 (verbatim)          │  ~200 tokens
│ Turn 19 (verbatim)          │  ~200 tokens
│ Turn 20 (current query)     │  ~200 tokens
└──────────────────────────────┘
Total: ~1,300 tokens instead of ~4,000 (3x compression)
```

**Pros**: Simple, low latency (one LLM call every N turns), preserves recent detail.
**Cons**: Summary quality degrades over many iterations (information gets "washed out").

### Strategy 2: Hierarchical Summary

Create summaries at multiple levels of granularity.

```
Level 0 (raw turns):     T1  T2  T3  T4  T5  T6  T7  T8  T9  T10 T11 T12
                          └──────┘   └──────┘   └──────┘   └──────────────┘
Level 1 (chunk summary): Chunk1-3   Chunk4-6   Chunk7-9     (verbatim)
                          └─────────────────┘
Level 2 (mega summary):  MegaSummary(1-9)                   T10 T11 T12

Context at turn 12:
┌──────────────────────────────┐
│ Mega summary (turns 1-9)    │  ~200 tokens
│ Turn 10 (verbatim)          │  ~200 tokens
│ Turn 11 (verbatim)          │  ~200 tokens
│ Turn 12 (current)           │  ~200 tokens
└──────────────────────────────┘
Total: ~800 tokens instead of ~4,800 (6x compression)
```

**Pros**: Better fidelity at scale, less "washing out" than running summary.
**Cons**: More complex, multiple LLM calls, higher compression latency.

### Strategy 3: Entity Extraction

Instead of a prose summary, extract structured data.

```
Turn 1:  User: "I'm John, I need help with order #12345"
Turn 2:  Agent: "I see order #12345 for $299. What's the issue?"
Turn 3:  User: "I was charged twice"
Turn 4:  Agent: "I can see two charges. I'll refund the duplicate."
Turn 5:  User: "Thanks. Also, my email should be john@example.com"

Extracted entities:
{
  "customer_name": "John",
  "order_id": "#12345",
  "order_amount": "$299",
  "issue": "duplicate charge",
  "resolution": "refund duplicate charge",
  "resolution_status": "in progress",
  "email_update": "john@example.com",
  "key_decisions": ["Agent will refund duplicate charge"]
}

Context injection (instead of 5 turns):
"Known context: Customer John, order #12345 ($299), 
issue: duplicate charge, resolution: refunding duplicate, 
updated email: john@example.com"

~50 tokens instead of ~500 (10x compression)
```

**Pros**: Highest compression, structured and queryable, no narrative loss for fact-based conversations.
**Cons**: Loses conversational nuance, tone, and reasoning chains.

## Production Implementation

### Running Summary (Most Common Pattern)

```python
from typing import List, Dict, Optional
import json

RUNNING_SUMMARY_PROMPT = """You are a conversation summarizer. Update the existing summary with the new conversation turns.

Rules:
1. Preserve ALL of the following from both the existing summary and new turns:
   - Names, IDs, account numbers, order numbers
   - Decisions made and their reasoning
   - Action items and commitments  
   - User preferences and corrections
   - Current task state (what's done, what's pending)
   - Errors encountered and their resolutions
2. Drop:
   - Greetings, thanks, acknowledgments
   - Repeated information already in the summary
   - Exploratory questions that were fully resolved
3. Keep the summary in present tense where possible ("User wants X" not "User wanted X")
4. If the user corrected or changed their mind, reflect the LATEST state
5. Maximum length: {max_tokens} tokens

EXISTING SUMMARY:
{existing_summary}

NEW CONVERSATION TURNS:
{new_turns}

UPDATED SUMMARY:"""


class RunningSummarizer:
    def __init__(
        self,
        llm_client,
        model: str = "gpt-4o-mini",
        summarize_every: int = 5,     # Summarize every 5 turns
        window_size: int = 6,         # Keep last 6 messages verbatim
        max_summary_tokens: int = 400,
    ):
        self.client = llm_client
        self.model = model
        self.summarize_every = summarize_every
        self.window_size = window_size
        self.max_summary_tokens = max_summary_tokens

        self.summary: Optional[str] = None
        self.all_messages: List[Dict[str, str]] = []
        self.unsummarized_count: int = 0

    def add_message(self, role: str, content: str):
        self.all_messages.append({"role": role, "content": content})
        self.unsummarized_count += 1

    async def maybe_summarize(self):
        """Summarize if enough new turns have accumulated."""
        if self.unsummarized_count < self.summarize_every:
            return

        # Messages to summarize (everything except the recent window)
        to_summarize = self.all_messages[:-self.window_size]
        if not to_summarize:
            return

        # Format new turns (only unsummarized ones)
        summarized_count = len(to_summarize) - (self.unsummarized_count - self.window_size)
        new_turns = to_summarize[max(0, summarized_count):]

        if not new_turns:
            return

        turns_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in new_turns
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": RUNNING_SUMMARY_PROMPT.format(
                    existing_summary=self.summary or "(No existing summary -- this is the beginning)",
                    new_turns=turns_text,
                    max_tokens=self.max_summary_tokens,
                )
            }],
            max_tokens=self.max_summary_tokens,
            temperature=0.0,
        )

        self.summary = response.choices[0].message.content
        self.unsummarized_count = self.window_size  # Reset counter

    def get_context_messages(self) -> List[Dict[str, str]]:
        """Return context for LLM: summary + recent verbatim messages."""
        messages = []

        if self.summary:
            messages.append({
                "role": "system",
                "content": f"[Conversation summary so far]\n{self.summary}"
            })

        # Recent messages verbatim
        recent = self.all_messages[-self.window_size:]
        messages.extend(recent)

        return messages

    @property
    def compression_stats(self) -> dict:
        total_tokens_raw = sum(len(m["content"].split()) for m in self.all_messages) * 1.3
        summary_tokens = len(self.summary.split()) * 1.3 if self.summary else 0
        window_tokens = sum(
            len(m["content"].split()) for m in self.all_messages[-self.window_size:]
        ) * 1.3
        compressed_tokens = summary_tokens + window_tokens

        return {
            "total_messages": len(self.all_messages),
            "raw_tokens": int(total_tokens_raw),
            "compressed_tokens": int(compressed_tokens),
            "compression_ratio": round(total_tokens_raw / max(compressed_tokens, 1), 1),
            "token_reduction_pct": round((1 - compressed_tokens / max(total_tokens_raw, 1)) * 100, 1),
        }
```

### Entity Extraction Summarizer

```python
ENTITY_EXTRACTION_PROMPT = """Extract structured information from this conversation.
Return a JSON object with the following fields (omit fields with no information):

{{
  "user_name": "string or null",
  "user_id": "string or null",
  "entities": {{
    "<entity_type>": {{
      "id": "identifier",
      "attributes": {{}}
    }}
  }},
  "issue": "one-line description of the problem",
  "resolution": "what was decided or done",
  "resolution_status": "pending | in_progress | resolved | escalated",
  "action_items": ["list of pending actions"],
  "user_preferences": ["list of stated preferences"],
  "corrections": ["things the user corrected or changed"],
  "key_facts": ["other important facts"]
}}

Conversation:
{conversation}

JSON:"""


class EntityExtractionSummarizer:
    def __init__(self, llm_client, model: str = "gpt-4o-mini"):
        self.client = llm_client
        self.model = model
        self.entities: dict = {}
        self.all_messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        self.all_messages.append({"role": role, "content": content})

    async def extract(self) -> dict:
        """Extract entities from full conversation."""
        convo_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in self.all_messages
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": ENTITY_EXTRACTION_PROMPT.format(conversation=convo_text)
            }],
            max_tokens=800,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        self.entities = json.loads(response.choices[0].message.content)
        return self.entities

    def get_context_injection(self) -> str:
        """Format entities as a concise context string."""
        if not self.entities:
            return ""

        parts = []
        if self.entities.get("user_name"):
            parts.append(f"Customer: {self.entities['user_name']}")
        if self.entities.get("issue"):
            parts.append(f"Issue: {self.entities['issue']}")
        if self.entities.get("resolution"):
            parts.append(f"Resolution: {self.entities['resolution']} ({self.entities.get('resolution_status', 'unknown')})")
        if self.entities.get("action_items"):
            parts.append(f"Pending actions: {', '.join(self.entities['action_items'])}")
        if self.entities.get("user_preferences"):
            parts.append(f"Preferences: {', '.join(self.entities['user_preferences'])}")
        if self.entities.get("key_facts"):
            parts.append(f"Key facts: {', '.join(self.entities['key_facts'])}")
        for etype, edata in self.entities.get("entities", {}).items():
            parts.append(f"{etype}: {edata.get('id', 'unknown')}")

        return " | ".join(parts)
```

### Hierarchical Summary (For 50+ Turn Conversations)

```python
class HierarchicalSummarizer:
    def __init__(
        self,
        llm_client,
        model: str = "gpt-4o-mini",
        chunk_size: int = 10,         # Summarize every 10 turns
        mega_chunk_size: int = 3,     # Merge every 3 chunk summaries
        window_size: int = 6,
    ):
        self.client = llm_client
        self.model = model
        self.chunk_size = chunk_size
        self.mega_chunk_size = mega_chunk_size
        self.window_size = window_size

        self.all_messages: List[Dict[str, str]] = []
        self.chunk_summaries: List[str] = []  # Level 1
        self.mega_summary: Optional[str] = None  # Level 2
        self.current_chunk: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        self.all_messages.append({"role": role, "content": content})
        self.current_chunk.append({"role": role, "content": content})

    async def maybe_summarize(self):
        """Create chunk summaries and mega summaries as needed."""
        # Level 1: Chunk summary
        if len(self.current_chunk) >= self.chunk_size:
            chunk_text = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in self.current_chunk
            )
            summary = await self._summarize(
                f"Summarize this conversation chunk:\n{chunk_text}"
            )
            self.chunk_summaries.append(summary)
            self.current_chunk = []

            # Level 2: Mega summary (merge chunk summaries)
            if len(self.chunk_summaries) >= self.mega_chunk_size:
                chunks_text = "\n---\n".join(self.chunk_summaries)
                self.mega_summary = await self._summarize(
                    f"Merge these conversation summaries into one coherent summary:\n\n"
                    f"{'Existing mega summary: ' + self.mega_summary + chr(10) + chr(10) if self.mega_summary else ''}"
                    f"{chunks_text}"
                )
                self.chunk_summaries = []  # Reset level 1

    def get_context_messages(self) -> List[Dict[str, str]]:
        """Return hierarchical context."""
        messages = []

        # Level 2: Mega summary (oldest context)
        if self.mega_summary:
            messages.append({
                "role": "system",
                "content": f"[Earlier conversation summary]\n{self.mega_summary}"
            })

        # Level 1: Recent chunk summaries
        if self.chunk_summaries:
            messages.append({
                "role": "system",
                "content": f"[Recent conversation summary]\n{' '.join(self.chunk_summaries)}"
            })

        # Level 0: Current chunk (verbatim)
        recent = self.all_messages[-self.window_size:]
        messages.extend(recent)

        return messages

    async def _summarize(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0,
        )
        return response.choices[0].message.content
```

## Decision Tree

```
Which summarization strategy should I use?
  │
  ├─ How long are your conversations?
  │   ├─ < 15 turns → No summarization needed
  │   ├─ 15-50 turns → Running summary
  │   └─ 50+ turns → Hierarchical summary
  │
  ├─ What type of information matters most?
  │   ├─ Facts (IDs, amounts, statuses) → Entity extraction
  │   ├─ Narrative (reasoning, context) → Running summary
  │   └─ Both → Entity extraction + running summary
  │
  ├─ Latency budget for compression?
  │   ├─ < 100ms → No summarization (use sliding window)
  │   ├─ 100-500ms → Running summary with gpt-4o-mini
  │   └─ 500ms+ → Hierarchical summary
  │
  └─ How important is exact recall?
      ├─ Critical → Wide window (15+ turns) + entity extraction
      ├─ Important → Running summary + 10-turn window
      └─ Nice-to-have → Aggressive summarization + 5-turn window
```

## When NOT to Use

- **Short conversations** (< 15 turns) -- the overhead of summarization exceeds the benefit.
- **When the full transcript is needed for compliance** -- summarize for the LLM context but store the full transcript in your database.
- **Real-time streaming conversations** -- summarization latency (200-500ms) may interrupt the flow. Use sliding window instead and summarize asynchronously.
- **When precision matters more than compression** -- a running summary may lose the exact phrasing of a user's complaint. Keep verbatim for legal/compliance scenarios.

## Tradeoffs

| Strategy | Compression | Fidelity | Latency | Cost | Best For |
|---|---|---|---|---|---|
| Running summary | 5-10x | Medium | 200-500ms/step | ~$0.0001/step | Most applications |
| Hierarchical | 10-15x | Medium | 500-1500ms/step | ~$0.001/step | Very long sessions |
| Entity extraction | 10-20x | High (for facts) | 300-800ms | ~$0.0003/extract | Fact-heavy workflows |
| Combined (entity + running) | 8-12x | High | 400-800ms/step | ~$0.0004/step | Production agents |

## Real-World Examples

1. **IT helpdesk agent** -- Average 25 turns per ticket. Running summary triggers at turn 10, then every 5 turns. Summary preserves: ticket ID, error messages, steps tried, current status. Window keeps last 6 messages. Result: 25 turns (12K tokens) compressed to ~2.5K (5x reduction).

2. **Sales demo agent** -- 40-turn product demo. Entity extraction captures: prospect name, company, budget, requirements, objections, next steps. Running summary captures: demo flow, questions asked, features shown. Both injected as context. 40 turns (20K tokens) compressed to ~2K (10x reduction).

3. **Therapy chatbot** -- 60+ turn sessions. Hierarchical summarization preserves emotional arc and topics discussed across levels. Window of 10 turns preserves current emotional state. Caution: summarization can lose emotional nuance -- wider window and careful prompt engineering required.

## Failure Modes

| Failure | Symptom | Fix |
|---|---|---|
| Summary drift | After many iterations, summary diverges from actual conversation | Reset summary periodically from entity extraction |
| Lost corrections | User corrected a fact but summary kept the old one | Prompt: "If the user corrected or changed their mind, reflect the LATEST state" |
| Over-summarization | Summary becomes too terse, loses useful context | Increase max_summary_tokens, reduce summarize_every |
| Entity extraction misses | Structured extraction misses implicit information | Supplement with running summary for narrative context |
| Latency spikes | Summarization takes 2s+ on complex conversations | Use gpt-4o-mini, cap input to summarization calls, async processing |

## Source(s) and Further Reading

- [LangChain ConversationSummaryMemory](https://python.langchain.com/docs/modules/memory/types/summary/) -- LangChain docs
- [State of AI Agent Memory 2026](https://mem0.ai) -- Mem0 research report
- [Mem0 Paper: Memory Layer for AI Agents](https://arxiv.org/abs/2504.19413) -- arXiv
