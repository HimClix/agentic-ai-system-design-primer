# Memory Eviction Strategies

> Deciding what to forget when the agent's context fills up: FIFO, LRU, importance-based, semantic deduplication, and temporal decay. Eviction strategies are the garbage collection of agent memory.

## What It Is

Memory eviction determines which memories (messages, observations, tool results) to remove when the agent's context window approaches its limit. Without eviction, agents either crash (context overflow) or degrade (lose important early context to simple truncation).

Eviction strategies for agent context:

| Strategy | Principle | Best For |
|----------|-----------|----------|
| **FIFO** | Remove oldest first | Conversation history |
| **LRU** | Remove least recently used | Cached lookups |
| **Importance-based** | Score each memory, evict lowest | Multi-step reasoning |
| **Semantic dedup** | Merge similar memories | Redundant observations |
| **Temporal decay** | Memories lose weight over time | Long-running sessions |
| **Summarize-and-evict** | Compress old memories into summaries | All use cases |

## How It Works

### Context Window Lifecycle

```
     Context Window (200K tokens)
     ┌────────────────────────────────────────────────┐
     │ System Prompt (fixed: ~2,000 tokens)           │
     │────────────────────────────────────────────────│
     │ Workspace (managed: ~5,000 tokens)             │
     │────────────────────────────────────────────────│
     │ Memory Buffer (grows with each step)           │
     │                                                 │
     │ Step 1: User msg + Response        (+800)      │
     │ Step 2: Tool call + Result         (+1,200)    │
     │ Step 3: User msg + Response        (+900)      │
     │ Step 4: Tool call + Result         (+2,500)    │
     │ ...                                             │
     │ Step 47: Tool call + Result        (+1,800)    │
     │                                                 │
     │ [EVICTION TRIGGER: 80% capacity reached]       │
     │────────────────────────────────────────────────│
     │ Safety margin (~10,000 tokens reserved)        │
     └────────────────────────────────────────────────┘
```

### Eviction Trigger Flow

```
    After each step:
         │
    ┌────▼──────────────────┐
    │ Count tokens in buffer │
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │ tokens > 80% of limit? │
    └────┬────────────┬─────┘
        Yes           No → continue
         │
    ┌────▼──────────────────┐
    │ Run eviction strategy  │
    │ until tokens < 60%     │
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │ Log eviction metrics   │
    │ (what was removed,     │
    │  new token count)      │
    └───────────────────────┘
```

## Production Implementation

```python
from __future__ import annotations
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


@dataclass
class MemoryEntry:
    """A single memory item in the agent's context buffer."""
    id: str
    content: str
    token_count: int
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    importance_score: float = 0.5  # 0.0 - 1.0
    memory_type: str = "message"   # message, tool_result, observation, knowledge
    metadata: dict = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def access(self):
        self.last_accessed_at = time.time()
        self.access_count += 1


class EvictionStrategy(ABC):
    """Base class for memory eviction strategies."""

    @abstractmethod
    def select_for_eviction(
        self,
        memories: list[MemoryEntry],
        tokens_to_free: int,
    ) -> list[str]:
        """Return IDs of memories to evict to free the required tokens."""
        ...


class FIFOEviction(EvictionStrategy):
    """
    First In, First Out: Remove the oldest memories first.
    
    Pros: Simple, predictable, O(n log n) sort
    Cons: May remove important early context
    Best for: Conversation history where recent context matters most
    """

    def select_for_eviction(
        self,
        memories: list[MemoryEntry],
        tokens_to_free: int,
    ) -> list[str]:
        sorted_memories = sorted(memories, key=lambda m: m.created_at)
        to_evict = []
        freed = 0

        for memory in sorted_memories:
            if freed >= tokens_to_free:
                break
            to_evict.append(memory.id)
            freed += memory.token_count

        return to_evict


class LRUEviction(EvictionStrategy):
    """
    Least Recently Used: Remove memories that haven't been accessed recently.
    
    Pros: Keeps frequently referenced information
    Cons: Rarely-needed-but-critical context may be evicted
    Best for: Agent workspaces with cached lookups
    """

    def select_for_eviction(
        self,
        memories: list[MemoryEntry],
        tokens_to_free: int,
    ) -> list[str]:
        sorted_memories = sorted(memories, key=lambda m: m.last_accessed_at)
        to_evict = []
        freed = 0

        for memory in sorted_memories:
            if freed >= tokens_to_free:
                break
            to_evict.append(memory.id)
            freed += memory.token_count

        return to_evict


class ImportanceEviction(EvictionStrategy):
    """
    Importance-based: Score each memory and evict the least important.
    
    Importance scoring heuristics:
    - User messages: high importance (contain the task)
    - Tool results with errors: lower importance (retry will get new results)
    - Final answers: high importance (may be referenced)
    - Intermediate observations: medium importance
    
    Pros: Keeps the most valuable context
    Cons: Requires scoring logic, more complex
    Best for: Multi-step reasoning where some steps matter more than others
    """

    # Importance weights by memory type
    TYPE_WEIGHTS = {
        "user_message": 0.9,
        "system_message": 1.0,
        "tool_result": 0.5,
        "tool_error": 0.2,
        "observation": 0.4,
        "knowledge": 0.7,
        "assistant_response": 0.6,
        "plan": 0.8,
    }

    def _score_memory(self, memory: MemoryEntry) -> float:
        """Calculate eviction priority (lower = evict first)."""
        base = self.TYPE_WEIGHTS.get(memory.memory_type, 0.5)

        # Recency bonus: recent memories get +0.2
        age_hours = memory.age_seconds / 3600
        recency_bonus = max(0, 0.2 - (age_hours * 0.05))

        # Access frequency bonus
        access_bonus = min(0.2, memory.access_count * 0.05)

        # Explicit importance override
        explicit = memory.importance_score

        return (base + recency_bonus + access_bonus + explicit) / 2

    def select_for_eviction(
        self,
        memories: list[MemoryEntry],
        tokens_to_free: int,
    ) -> list[str]:
        scored = [(m, self._score_memory(m)) for m in memories]
        scored.sort(key=lambda x: x[1])  # Lowest score = evict first

        to_evict = []
        freed = 0

        for memory, score in scored:
            if freed >= tokens_to_free:
                break
            to_evict.append(memory.id)
            freed += memory.token_count

        return to_evict


class TemporalDecayEviction(EvictionStrategy):
    """
    Temporal decay: Memories lose importance weight over time.
    
    Uses exponential decay: weight = base_weight * e^(-lambda * age_hours)
    
    Pros: Natural aging of information
    Cons: May evict old-but-critical context
    Best for: Long-running agent sessions (hours/days)
    """

    def __init__(self, decay_lambda: float = 0.1):
        self.decay_lambda = decay_lambda

    def _decayed_score(self, memory: MemoryEntry) -> float:
        import math
        age_hours = memory.age_seconds / 3600
        decay_factor = math.exp(-self.decay_lambda * age_hours)
        return memory.importance_score * decay_factor

    def select_for_eviction(
        self,
        memories: list[MemoryEntry],
        tokens_to_free: int,
    ) -> list[str]:
        scored = [(m, self._decayed_score(m)) for m in memories]
        scored.sort(key=lambda x: x[1])

        to_evict = []
        freed = 0

        for memory, score in scored:
            if freed >= tokens_to_free:
                break
            to_evict.append(memory.id)
            freed += memory.token_count

        return to_evict


class SemanticDedup(EvictionStrategy):
    """
    Semantic deduplication: Merge similar memories before eviction.
    
    If two tool results say the same thing, keep only one.
    Reduces redundancy without losing information.
    
    Pros: Removes actual redundancy
    Cons: Requires embedding computation, possible information loss
    Best for: RAG-heavy agents that retrieve overlapping chunks
    """

    def __init__(self, similarity_threshold: float = 0.92):
        self.threshold = similarity_threshold

    def _simple_hash(self, text: str) -> str:
        """Content hash for exact dedup."""
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def select_for_eviction(
        self,
        memories: list[MemoryEntry],
        tokens_to_free: int,
    ) -> list[str]:
        # Phase 1: Exact dedup (fast, no embeddings)
        seen_hashes = {}
        duplicates = []

        for memory in memories:
            h = self._simple_hash(memory.content)
            if h in seen_hashes:
                duplicates.append(memory.id)
            else:
                seen_hashes[h] = memory.id

        freed = sum(m.token_count for m in memories if m.id in duplicates)
        if freed >= tokens_to_free:
            return duplicates[:len(duplicates)]  # Return enough duplicates

        # Phase 2: Fall back to FIFO for remaining
        remaining = [m for m in memories if m.id not in duplicates]
        fifo = FIFOEviction()
        additional = fifo.select_for_eviction(remaining, tokens_to_free - freed)

        return duplicates + additional


# --- Memory Buffer with Eviction ---

class AgentMemoryBuffer:
    """
    Token-bounded memory buffer with configurable eviction.
    
    Usage:
        buffer = AgentMemoryBuffer(max_tokens=150_000)
        buffer.add("user_msg_1", "Hello!", 10, memory_type="user_message")
        buffer.add("tool_1", "Search result: ...", 500, memory_type="tool_result")
        # Eviction happens automatically when buffer approaches capacity
    """

    def __init__(
        self,
        max_tokens: int = 150_000,
        eviction_trigger_pct: float = 0.80,
        eviction_target_pct: float = 0.60,
        strategy: EvictionStrategy = None,
        protected_types: list[str] = None,
    ):
        self.max_tokens = max_tokens
        self.trigger_pct = eviction_trigger_pct
        self.target_pct = eviction_target_pct
        self.strategy = strategy or ImportanceEviction()
        self.protected_types = protected_types or ["system_message", "plan"]
        self.memories: dict[str, MemoryEntry] = {}
        self._total_tokens = 0
        self._eviction_log: list[dict] = []

    @property
    def current_tokens(self) -> int:
        return self._total_tokens

    @property
    def utilization(self) -> float:
        return self._total_tokens / self.max_tokens

    def add(
        self,
        id: str,
        content: str,
        token_count: int,
        memory_type: str = "message",
        importance: float = 0.5,
    ) -> None:
        """Add a memory entry. Triggers eviction if needed."""
        entry = MemoryEntry(
            id=id,
            content=content,
            token_count=token_count,
            memory_type=memory_type,
            importance_score=importance,
        )
        self.memories[id] = entry
        self._total_tokens += token_count

        # Check eviction trigger
        if self.utilization >= self.trigger_pct:
            self._evict()

    def get(self, id: str) -> Optional[MemoryEntry]:
        """Get a memory entry and update its access time."""
        entry = self.memories.get(id)
        if entry:
            entry.access()
        return entry

    def _evict(self) -> None:
        """Run eviction strategy until below target utilization."""
        target_tokens = int(self.max_tokens * self.target_pct)
        tokens_to_free = self._total_tokens - target_tokens

        if tokens_to_free <= 0:
            return

        # Get evictable memories (exclude protected types)
        evictable = [
            m for m in self.memories.values()
            if m.memory_type not in self.protected_types
        ]

        to_evict_ids = self.strategy.select_for_eviction(evictable, tokens_to_free)

        evicted_count = 0
        evicted_tokens = 0
        for mid in to_evict_ids:
            if mid in self.memories:
                entry = self.memories.pop(mid)
                self._total_tokens -= entry.token_count
                evicted_count += 1
                evicted_tokens += entry.token_count

        self._eviction_log.append({
            "timestamp": time.time(),
            "evicted_count": evicted_count,
            "evicted_tokens": evicted_tokens,
            "strategy": type(self.strategy).__name__,
            "utilization_before": (self._total_tokens + evicted_tokens) / self.max_tokens,
            "utilization_after": self._total_tokens / self.max_tokens,
        })

    def get_ordered_content(self) -> list[str]:
        """Get all memory content in chronological order."""
        sorted_memories = sorted(self.memories.values(), key=lambda m: m.created_at)
        return [m.content for m in sorted_memories]

    def get_stats(self) -> dict:
        return {
            "total_entries": len(self.memories),
            "total_tokens": self._total_tokens,
            "utilization": f"{self.utilization:.1%}",
            "evictions_performed": len(self._eviction_log),
            "by_type": {
                t: sum(1 for m in self.memories.values() if m.memory_type == t)
                for t in set(m.memory_type for m in self.memories.values())
            },
        }
```

## Decision Tree: Choosing an Eviction Strategy

```
    Which eviction strategy should I use?
                        │
             ┌──────────▼──────────┐
             │ Is this a short      │
             │ session (<10 steps)? │
             └──┬──────────────┬──┘
               Yes             No
                │               │
           FIFO (simple,   ┌───▼───────────────┐
           predictable)    │ Does the agent     │
                           │ re-reference old   │
                           │ context?            │
                           └──┬────────────┬───┘
                             Yes           No
                              │             │
                          LRU or        ┌───▼───────────┐
                          Importance    │ Are there many │
                                        │ redundant tool │
                                        │ results?       │
                                        └──┬────────┬───┘
                                          Yes       No
                                           │        │
                                      Semantic   Temporal
                                      Dedup +    Decay
                                      FIFO
```

## When NOT to Use

1. **Short conversations (< 10 turns)**: Context window will not fill up. Eviction adds unnecessary complexity.
2. **Small models with 4K context**: At 4K tokens, there is almost nothing to evict. Use summarization instead.
3. **Stateless request/response**: If each request is independent, there is no memory to evict.
4. **When prompt caching is active**: Evicting cached content invalidates the cache, increasing costs. Keep cached sections stable.
5. **Human-in-the-loop applications**: Users may reference early messages; evicting them causes confusion.

## Tradeoffs

| Strategy | Speed | Accuracy | Complexity | Memory Loss Risk |
|----------|-------|----------|------------|------------------|
| FIFO | O(n) | Low | Very low | High (important early context) |
| LRU | O(n) | Medium | Low | Medium (rarely-used but critical) |
| Importance-based | O(n) | High | Medium | Low |
| Semantic dedup | O(n^2) | High | High | Very low (only removes redundancy) |
| Temporal decay | O(n) | Medium | Medium | Medium (old = important sometimes) |
| Summarize + evict | O(n) + LLM call | Highest | Highest | Lowest |

## Failure Modes

### 1. Critical Context Evicted
The original user query or a key constraint is evicted, causing the agent to forget the task.
**Mitigation**: Protect critical memory types (user messages, plan) from eviction.

### 2. Summarization Information Loss
Summarizing 10 tool results into a 2-sentence summary loses important details.
**Mitigation**: Keep summaries at sufficient detail. Include key data points (numbers, names, IDs).

### 3. Eviction Oscillation
Agent re-fetches evicted information (calling the same tool again), which gets evicted again, creating a loop.
**Mitigation**: Track evicted content. Promote re-fetched information to higher importance.

### 4. Token Count Inaccuracy
Estimated token count differs from actual tokenizer count, causing premature or late eviction.
**Mitigation**: Use the actual tokenizer (tiktoken for OpenAI, count_tokens for Anthropic).

### 5. Protected Category Bloat
Too many memory types are marked as "protected," leaving nothing eligible for eviction.
**Mitigation**: Keep protected types minimal (only system prompt and current plan).

## Sources and Further Reading

- [Cognitive Architectures for Language Agents (CoALA)](https://arxiv.org/abs/2309.02427)
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
- [Context Window Management - LangChain](https://python.langchain.com/docs/how_to/trim_messages/)
- [Token Counting - Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/token-counting)
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents)
