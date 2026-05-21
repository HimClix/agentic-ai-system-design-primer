# Context Engineering
> Context engineering is now a job title, not just a prompting technique. It's the discipline of designing what information enters the LLM's context window, in what order, and at what cost.

## What It Is

Context engineering is the systematic practice of curating, structuring, and managing the information provided to an LLM at inference time. While "prompt engineering" focuses on writing instructions, context engineering focuses on the entire information environment: system prompts, retrieved documents, tool schemas, conversation history, cached prefixes, and dynamically injected context.

The context window is the LLM's only interface with the world. Everything the model knows about the current task, the user's history, available tools, and domain knowledge must fit in this window. Context engineering is the discipline of making every token in that window count.

## The 6 Core Techniques

| # | Technique | What It Does | Impact |
|---|-----------|-------------|--------|
| 1 | **Curation** | Select what enters the context | Prevents noise, hallucination |
| 2 | **Structuring** | Order and format context for attention | Exploits primacy/recency bias |
| 3 | **Caching** | Reuse expensive prefix computations | 50-90% cost reduction |
| 4 | **Versioning** | Track and A/B test prompt changes | Measurable prompt quality |
| 5 | **Compression** | Reduce token count without losing information | More room for useful content |
| 6 | **Dynamic Assembly** | Build context per-request based on conditions | Relevance per interaction |

## How It Works

### The Context Window Budget

```
Model: Claude 3.5 Sonnet (200K context window)
Optimal fill: 60-80% (120K-160K tokens)

Budget allocation:
  ┌──────────────────────────────────────────────┐
  │ System Prompt (instructions, persona)  5-10% │  <- Cached (90% savings)
  │ Tool Schemas (available tools)          5-15% │  <- Cached (90% savings)  
  │ Retrieved Context (RAG documents)      20-40% │  <- Dynamic per query
  │ Conversation History                   10-20% │  <- Sliding window
  │ Current User Message                    1-5%  │  <- Always included
  │ [Headroom for generation]              20-40% │  <- Reserved for output
  └──────────────────────────────────────────────┘

WARNING: U-shaped attention curve
  - Information at positions 0-20% and 80-100% gets high attention
  - Information at positions 40-60% ("the middle") gets low attention
  - Place most important context at the BEGINNING and END
```

### The U-Shaped Attention Problem

Research (Liu et al., "Lost in the Middle," 2023) shows that LLM performance degrades when key information is placed in the middle of long contexts:

```
Attention Distribution in Long Context:

Position:   |  Start  |  25%  |  50%  |  75%  |  End  |
Attention:  |  HIGH   |  Med  |  LOW  |  Med  |  HIGH |
            |=========|-------|.......|-------|=======|

Place here: | Critical|       | Least | Suppl.| Recent|
            | context | Suppl.| import| info  | user  |
            | + rules |       | ant   |       | msg   |
```

**Practical rule**: Keep context window fill at 60-80% of maximum. Beyond 80%, attention degradation becomes measurable. Below 60%, you're not using the model's capacity.

## Production Implementation

```python
"""Context engineering: build optimized context windows."""
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
import tiktoken
import logging

logger = logging.getLogger(__name__)


class ContextPriority(int, Enum):
    """Priority levels for context components."""
    CRITICAL = 1    # Always include (system prompt, current message)
    HIGH = 2        # Include unless space is tight (tool schemas, key docs)
    MEDIUM = 3      # Include if space available (conversation history)
    LOW = 4         # Include only with remaining space (supplementary context)


@dataclass
class ContextBlock:
    """A block of content to potentially include in the context."""
    name: str
    content: str
    priority: ContextPriority
    token_count: int = 0
    cacheable: bool = False
    position_preference: str = "any"  # "start", "end", "any"
    
    def __post_init__(self):
        if self.token_count == 0:
            encoder = tiktoken.encoding_for_model("gpt-4")
            self.token_count = len(encoder.encode(self.content))


@dataclass
class ContextBudget:
    """Token budget for context assembly."""
    total_window: int = 128000       # Model's context window
    target_fill_ratio: float = 0.65  # Target 65% fill
    generation_reserve: float = 0.20 # Reserve 20% for output
    
    @property
    def available_tokens(self) -> int:
        return int(self.total_window * (self.target_fill_ratio))
    
    @property
    def max_tokens(self) -> int:
        """Absolute maximum (never exceed)."""
        return int(self.total_window * (1 - self.generation_reserve))


class ContextAssembler:
    """Assembles the optimal context window from available blocks."""
    
    def __init__(self, budget: ContextBudget = None):
        self.budget = budget or ContextBudget()
    
    def assemble(self, blocks: list[ContextBlock]) -> list[ContextBlock]:
        """Select and order context blocks within budget.
        
        Algorithm:
        1. Sort by priority (CRITICAL first)
        2. Add blocks until budget is reached
        3. Order: start-preferring blocks first, then by priority, 
           then end-preferring blocks last
        """
        # Sort by priority
        sorted_blocks = sorted(blocks, key=lambda b: b.priority.value)
        
        selected = []
        used_tokens = 0
        available = self.budget.available_tokens
        
        for block in sorted_blocks:
            if block.priority == ContextPriority.CRITICAL:
                # Always include critical blocks
                selected.append(block)
                used_tokens += block.token_count
            elif used_tokens + block.token_count <= available:
                selected.append(block)
                used_tokens += block.token_count
            else:
                logger.info(
                    f"Skipped context block '{block.name}' "
                    f"({block.token_count} tokens, priority={block.priority.name}). "
                    f"Budget: {used_tokens}/{available} tokens used."
                )
        
        # Order: start-preferring, then by priority, then end-preferring
        ordered = self._order_blocks(selected)
        
        fill_ratio = used_tokens / self.budget.total_window
        logger.info(
            f"Context assembled: {len(ordered)} blocks, "
            f"{used_tokens} tokens, {fill_ratio:.0%} fill"
        )
        
        if fill_ratio > 0.80:
            logger.warning(
                f"Context fill ratio {fill_ratio:.0%} exceeds 80%. "
                f"Attention degradation likely in middle sections."
            )
        
        return ordered
    
    def _order_blocks(self, blocks: list[ContextBlock]) -> list[ContextBlock]:
        """Order blocks for optimal attention distribution."""
        start_blocks = [b for b in blocks if b.position_preference == "start"]
        end_blocks = [b for b in blocks if b.position_preference == "end"]
        any_blocks = [b for b in blocks if b.position_preference == "any"]
        
        # Sort "any" blocks by priority (most important first)
        any_blocks.sort(key=lambda b: b.priority.value)
        
        return start_blocks + any_blocks + end_blocks
    
    def build_messages(self, blocks: list[ContextBlock]) -> list[dict]:
        """Convert assembled context blocks to LLM message format."""
        selected = self.assemble(blocks)
        
        messages = []
        system_parts = []
        
        for block in selected:
            if block.name == "system_prompt":
                system_parts.insert(0, block.content)  # System always first
            elif block.name == "user_message":
                messages.append({"role": "user", "content": block.content})
            elif block.name.startswith("conversation_"):
                # Preserve conversation order
                role = "assistant" if "assistant" in block.name else "user"
                messages.append({"role": role, "content": block.content})
            else:
                system_parts.append(f"\n## {block.name}\n{block.content}")
        
        # Build system message from all system parts
        if system_parts:
            system_content = "\n\n".join(system_parts)
            messages.insert(0, {"role": "system", "content": system_content})
        
        return messages


# Usage example:

assembler = ContextAssembler(ContextBudget(
    total_window=200000,
    target_fill_ratio=0.65,
))

blocks = [
    ContextBlock(
        name="system_prompt",
        content="You are a customer support agent for Acme Payments...",
        priority=ContextPriority.CRITICAL,
        cacheable=True,
        position_preference="start",
    ),
    ContextBlock(
        name="tool_schemas",
        content="[tool schemas JSON here]",
        priority=ContextPriority.CRITICAL,
        cacheable=True,
        position_preference="start",
    ),
    ContextBlock(
        name="retrieved_context",
        content="[RAG results here]",
        priority=ContextPriority.HIGH,
        position_preference="start",  # Primacy bias for important docs
    ),
    ContextBlock(
        name="conversation_history",
        content="[last 10 messages]",
        priority=ContextPriority.MEDIUM,
    ),
    ContextBlock(
        name="user_message",
        content="How do I process a refund for order #12345?",
        priority=ContextPriority.CRITICAL,
        position_preference="end",  # Recency bias for current request
    ),
]

messages = assembler.build_messages(blocks)
```

## Decision Tree: Context Strategy

```
What type of application are you building?
  |
  |-- Chat agent (multi-turn conversation)?
  |     --> Sliding window history + cached system prompt + dynamic RAG
  |     --> Cache: system prompt + tool schemas (90% savings)
  |     --> Compress: summarize old conversation turns
  |     --> Dynamic: RAG only when needed
  |
  |-- Single-turn Q&A?
  |     --> Cached system prompt + RAG context + user query
  |     --> Cache: system prompt (high reuse)
  |     --> No history management needed
  |
  |-- Code generation agent?
  |     --> Cached instructions + dynamic code context + tools
  |     --> Cache: coding instructions + tool schemas
  |     --> Dynamic: relevant code files (codebase search)
  |
  |-- Multi-agent system?
  |     --> Per-agent context + shared memory + task context
  |     --> Cache: each agent's system prompt separately
  |     --> Dynamic: shared memory injected per turn
```

## When NOT to Over-Engineer Context

- **Simple chatbots**: A system prompt + user message is fine. Don't add RAG, caching, and compression if you don't need them.
- **Low-volume applications**: Prompt caching ROI requires volume. Under 100 queries/day, the savings don't justify the complexity.
- **Short conversations**: If conversations rarely exceed 5 turns, don't build history compression.
- **Single-model applications**: Context engineering complexity increases with multi-model architectures. Start simple.

## Tradeoffs

| Technique | Benefit | Cost | When to Add |
|-----------|---------|------|-------------|
| Prompt Caching | 50-90% cost reduction | Cache management complexity | >100 queries/day with stable system prompt |
| Dynamic Injection | Relevant context per request | Increased latency, complexity | Multi-domain or multi-user applications |
| Version Control | Measurable prompt quality | CI/CD for prompts, eval infra | Production applications with quality SLAs |
| Compression | More room for useful content | Information loss risk | Context exceeding 60% fill regularly |
| Structuring | Better attention to key info | Manual ordering effort | When answer quality depends on specific docs |
| Curation | Reduces noise and hallucination | Must decide what to exclude | When unfiltered context hurts quality |

## Real-World Examples

### Production Context Budget: Customer Support Agent
```
Model: Claude 3.5 Sonnet (200K window)
Budget: 130K tokens (65% fill)

Allocation:
  System prompt:        2,500 tokens  ( 2%)  [CACHED - 90% savings]
  Tool schemas:         4,000 tokens  ( 3%)  [CACHED - 90% savings]
  Company policies:     8,000 tokens  ( 6%)  [CACHED - 90% savings]
  Customer profile:       500 tokens  ( 0.4%)  [DYNAMIC per customer]
  RAG documents:       15,000 tokens  (12%)  [DYNAMIC per query]
  Conversation history: 10,000 tokens  ( 8%)  [SLIDING WINDOW, last 20 msgs]
  Current message:        200 tokens  ( 0.2%)  [ALWAYS included]
  Generation headroom:  40,000 tokens  (20%)  [RESERVED]
  Unused:              120,000 tokens  (51%)  [Available for expansion]

Monthly cost savings from caching: ~$2,400 (14,500 cached tokens * ~1M queries)
```

## Failure Modes

| Failure | Cause | Impact | Fix |
|---------|-------|--------|-----|
| Context overflow | Too much included | Attention degradation, wrong answers | Budget management, compression |
| Lost in the middle | Key info at position 40-60% | Model ignores critical context | Move important context to start/end |
| Stale cache | Cached prompt out of date | Old instructions, wrong behavior | Cache invalidation on prompt update |
| Context starvation | Too little included | Model lacks information to answer | Include more RAG context, relax budget |
| Priority inversion | Low-priority content displaces high | Important context excluded | Strict priority ordering |
| Prompt injection via context | RAG doc contains adversarial text | Model follows injected instructions | Sanitize retrieved content |

## Source(s) and Further Reading

- Andrej Karpathy, "Context Engineering" talk (2025) -- the discipline defined
- Liu et al., "Lost in the Middle" (2023) -- attention degradation in long contexts
- Anthropic, "Prompt Caching" (2024) -- 90% cost savings on cached prefixes
- OpenAI, "Automatic Prompt Caching" (2024) -- 50% savings
- Simon Willison, "Context Engineering" blog series (2025) -- practical patterns
- LangChain, "Context Management" -- framework implementation
