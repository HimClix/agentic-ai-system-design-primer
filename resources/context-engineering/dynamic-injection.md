# Dynamic Context Injection
> Not every request needs the same context. Dynamic injection assembles the context window per-request, including only what's relevant and excluding what's noise.

## What It Is

Dynamic context injection is the practice of assembling the LLM's context window at runtime based on the specific request, user, conversation state, and retrieved information. Instead of a static system prompt that tries to cover every scenario, the context is built from modular components that are conditionally included.

The principle: include what the LLM needs for THIS request. Exclude everything else. Extra context is not free -- it costs tokens, degrades attention, and increases hallucination risk.

## How It Works

### The Three Tiers of Context

```
Tier 1: ALWAYS Include (every request)
  - Core system instructions (persona, rules, guardrails)
  - Available tool schemas
  - Current user message
  
Tier 2: CONDITIONALLY Include (based on request)
  - RAG-retrieved documents (only when knowledge is needed)
  - User profile/preferences (only when personalization matters)
  - Recent conversation history (only in multi-turn)
  - Domain-specific guidelines (only for that domain)
  
Tier 3: ON-DEMAND Include (based on specific triggers)
  - Error recovery context (only after a tool failure)
  - Escalation procedures (only when escalation detected)
  - Compliance text (only for regulated operations)
  - Few-shot examples (only for complex/rare tasks)
```

### Context Assembly Pipeline

```
User Request arrives
  |
  v
[Request Analyzer]
  |-- Detect intent (question, action, clarification)
  |-- Detect domain (payments, refunds, onboarding)
  |-- Detect complexity (simple lookup, multi-step, edge case)
  |-- Detect user segment (enterprise, SMB, developer)
  |
  v
[Context Builder]
  |-- Load Tier 1 (always-include base)
  |-- Evaluate Tier 2 conditions:
  |     |-- Need RAG? -> Query vector DB, inject results
  |     |-- Multi-turn? -> Include last N messages
  |     |-- User-specific? -> Load user profile
  |-- Evaluate Tier 3 triggers:
  |     |-- Previous tool failure? -> Add error recovery guide
  |     |-- Compliance domain? -> Add regulatory text
  |     |-- Complex task? -> Add few-shot examples
  |
  v
[Budget Manager]
  |-- Total tokens < 65% of context window?
  |     YES -> Include all selected blocks
  |     NO  -> Trim lowest-priority blocks
  |          -> Summarize conversation history
  |          -> Truncate RAG results
  |
  v
[Position Optimizer]
  |-- Place critical context at start (primacy bias)
  |-- Place user message at end (recency bias)
  |-- Keep supplementary context in between
  |
  v
Assembled Context -> LLM API Call
```

## Production Implementation

```python
"""Dynamic context injection system for production agents."""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum
import tiktoken
import logging

logger = logging.getLogger(__name__)


class InclusionTier(str, Enum):
    ALWAYS = "always"           # Every request
    CONDITIONAL = "conditional" # Based on request analysis
    ON_DEMAND = "on_demand"     # Triggered by specific signals


@dataclass
class ContextComponent:
    """A modular piece of context that can be injected."""
    name: str
    tier: InclusionTier
    content_fn: Callable  # Function that returns the content string
    condition_fn: Optional[Callable] = None  # Returns True if should include
    priority: int = 50  # 1=highest, 100=lowest
    max_tokens: int = 5000  # Maximum tokens for this component
    position: str = "middle"  # "start", "middle", "end"
    cacheable: bool = False


@dataclass
class RequestContext:
    """Analyzed request metadata for injection decisions."""
    user_message: str
    user_id: str
    conversation_id: str
    turn_number: int
    detected_intent: str           # "question", "action", "clarification"
    detected_domain: str           # "payments", "refunds", "general"
    detected_complexity: str       # "simple", "moderate", "complex"
    user_segment: str              # "enterprise", "smb", "developer"
    previous_tool_failure: bool = False
    escalation_detected: bool = False
    has_conversation_history: bool = False
    metadata: dict = field(default_factory=dict)


class DynamicContextInjector:
    """Assembles context per-request from modular components."""

    def __init__(
        self,
        context_window_size: int = 200000,
        target_fill_ratio: float = 0.65,
    ):
        self.components: list[ContextComponent] = []
        self.window_size = context_window_size
        self.target_fill = target_fill_ratio
        self.encoder = tiktoken.encoding_for_model("gpt-4")

    def register(self, component: ContextComponent):
        """Register a context component."""
        self.components.append(component)

    def _count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    async def build(self, request: RequestContext) -> list[dict]:
        """Build the complete context for a request.
        
        Returns list of message dicts ready for the LLM API.
        """
        budget = int(self.window_size * self.target_fill)
        used_tokens = 0
        selected_blocks = []

        # Sort by tier (ALWAYS first), then by priority
        sorted_components = sorted(
            self.components,
            key=lambda c: (c.tier.value, c.priority),
        )

        for component in sorted_components:
            # Tier filtering
            if component.tier == InclusionTier.ALWAYS:
                should_include = True
            elif component.tier == InclusionTier.CONDITIONAL:
                should_include = (
                    component.condition_fn(request) 
                    if component.condition_fn 
                    else False
                )
            elif component.tier == InclusionTier.ON_DEMAND:
                should_include = (
                    component.condition_fn(request) 
                    if component.condition_fn 
                    else False
                )
            else:
                should_include = False

            if not should_include:
                continue

            # Get content
            try:
                content = await component.content_fn(request)
                if not content:
                    continue
            except Exception as e:
                logger.warning(f"Failed to load context component '{component.name}': {e}")
                continue

            # Token budget check
            token_count = self._count_tokens(content)
            truncated = False

            if token_count > component.max_tokens:
                # Truncate to max_tokens
                tokens = self.encoder.encode(content)[:component.max_tokens]
                content = self.encoder.decode(tokens)
                token_count = component.max_tokens
                truncated = True

            if used_tokens + token_count > budget:
                if component.tier == InclusionTier.ALWAYS:
                    # Always-include blocks override budget (but log warning)
                    logger.warning(
                        f"Budget exceeded by ALWAYS component '{component.name}'"
                    )
                else:
                    logger.info(
                        f"Skipped '{component.name}' ({token_count} tokens): "
                        f"budget {used_tokens}/{budget}"
                    )
                    continue

            selected_blocks.append({
                "name": component.name,
                "content": content,
                "tokens": token_count,
                "position": component.position,
                "cacheable": component.cacheable,
                "truncated": truncated,
            })
            used_tokens += token_count

        # Order blocks by position preference
        start_blocks = [b for b in selected_blocks if b["position"] == "start"]
        middle_blocks = [b for b in selected_blocks if b["position"] == "middle"]
        end_blocks = [b for b in selected_blocks if b["position"] == "end"]

        ordered = start_blocks + middle_blocks + end_blocks
        
        fill_pct = used_tokens / self.window_size
        logger.info(
            f"Context built: {len(ordered)} components, "
            f"{used_tokens} tokens, {fill_pct:.0%} fill"
        )

        return self._format_messages(ordered)

    def _format_messages(self, blocks: list[dict]) -> list[dict]:
        """Convert context blocks to LLM message format."""
        system_parts = []
        messages = []

        for block in blocks:
            if block["name"] in ("system_prompt", "tool_guidelines", "domain_context",
                                 "rag_context", "user_profile", "few_shot_examples",
                                 "error_recovery", "compliance_text"):
                system_parts.append(f"## {block['name']}\n{block['content']}")
            elif block["name"] == "conversation_history":
                # Parse and add as separate messages
                messages.extend(self._parse_history(block["content"]))
            elif block["name"] == "user_message":
                messages.append({"role": "user", "content": block["content"]})

        if system_parts:
            messages.insert(0, {
                "role": "system",
                "content": "\n\n".join(system_parts),
            })

        return messages

    def _parse_history(self, history_text: str) -> list[dict]:
        """Parse conversation history into message format."""
        # Simplified; in production, history would be structured data
        return [{"role": "user", "content": history_text}]


# --- Register Components ---

injector = DynamicContextInjector()

# Tier 1: Always included
injector.register(ContextComponent(
    name="system_prompt",
    tier=InclusionTier.ALWAYS,
    content_fn=lambda req: load_system_prompt(),
    priority=1,
    position="start",
    cacheable=True,
))

injector.register(ContextComponent(
    name="user_message",
    tier=InclusionTier.ALWAYS,
    content_fn=lambda req: req.user_message,
    priority=1,
    position="end",
))

# Tier 2: Conditional
injector.register(ContextComponent(
    name="rag_context",
    tier=InclusionTier.CONDITIONAL,
    content_fn=lambda req: retrieve_rag_context(req.user_message),
    condition_fn=lambda req: req.detected_intent == "question",
    priority=10,
    max_tokens=3000,
    position="start",  # Important docs near the beginning
))

injector.register(ContextComponent(
    name="conversation_history",
    tier=InclusionTier.CONDITIONAL,
    content_fn=lambda req: load_conversation_history(req.conversation_id, max_turns=10),
    condition_fn=lambda req: req.has_conversation_history and req.turn_number > 1,
    priority=30,
    max_tokens=4000,
    position="middle",
))

injector.register(ContextComponent(
    name="user_profile",
    tier=InclusionTier.CONDITIONAL,
    content_fn=lambda req: load_user_profile(req.user_id),
    condition_fn=lambda req: req.user_segment in ("enterprise", "vip"),
    priority=20,
    max_tokens=500,
    position="start",
))

injector.register(ContextComponent(
    name="domain_context",
    tier=InclusionTier.CONDITIONAL,
    content_fn=lambda req: load_domain_guidelines(req.detected_domain),
    condition_fn=lambda req: req.detected_domain != "general",
    priority=15,
    max_tokens=2000,
    position="start",
    cacheable=True,  # Domain guidelines don't change often
))

# Tier 3: On-demand
injector.register(ContextComponent(
    name="error_recovery",
    tier=InclusionTier.ON_DEMAND,
    content_fn=lambda req: load_error_recovery_guide(),
    condition_fn=lambda req: req.previous_tool_failure,
    priority=5,  # High priority when triggered
    max_tokens=1000,
    position="start",
))

injector.register(ContextComponent(
    name="few_shot_examples",
    tier=InclusionTier.ON_DEMAND,
    content_fn=lambda req: load_few_shot_examples(req.detected_domain, req.detected_complexity),
    condition_fn=lambda req: req.detected_complexity == "complex",
    priority=25,
    max_tokens=2000,
    position="start",
))

injector.register(ContextComponent(
    name="compliance_text",
    tier=InclusionTier.ON_DEMAND,
    content_fn=lambda req: load_compliance_requirements(req.detected_domain),
    condition_fn=lambda req: req.detected_domain in ("payments", "kyc", "banking"),
    priority=10,
    max_tokens=1500,
    position="start",
))
```

## Decision Tree: What to Include

```
For each piece of context, ask:
  |
  |-- Does the LLM need this for EVERY request?
  |     YES -> Tier 1 (always include, cache if possible)
  |     NO  --> Does the LLM need this for SOME requests?
  |               |
  |               YES -> Define the condition (Tier 2)
  |               |      Examples:
  |               |        - RAG: when intent is "question"
  |               |        - History: when turn > 1
  |               |        - User profile: when segment is "enterprise"
  |               |
  |               NO  --> Is this triggered by a specific signal?
  |                        |
  |                        YES -> Tier 3 (on-demand, e.g., after tool failure)
  |                        NO  -> Don't include it at all
```

## When NOT to Use Dynamic Injection

- **Simple single-turn applications**: A fixed system prompt + user message is fine
- **Over-optimization**: Don't build a complex injection system for 5 context components
- **When latency matters more than precision**: Dynamic assembly adds ~10-50ms
- **Prototype phase**: Get the prompt working first, then optimize with dynamic injection

## Tradeoffs

| Aspect | Dynamic Injection | Static Prompt |
|--------|-------------------|---------------|
| Token efficiency | High (only relevant content) | Low (everything always) |
| Complexity | Higher (condition logic) | Lower (fixed template) |
| Latency | +10-50ms for assembly | None |
| Relevance | High per-request | Same for all requests |
| Maintenance | Each component independent | One large prompt to maintain |
| Testing | Can test each component separately | Test entire prompt |
| Debugging | Can see exactly what was included | Always the same |

## Real-World Examples

### Simple Request (3 components included)
```
Request: "What's your pricing?"
Analyzed: intent=question, domain=general, complexity=simple

Included:
  [ALWAYS] system_prompt (2,500 tokens)
  [CONDITIONAL] rag_context: pricing page docs (1,200 tokens)
  [ALWAYS] user_message (10 tokens)
  Total: 3,710 tokens (1.9% of 200K window)
```

### Complex Request (7 components included)
```
Request: "Process a refund for order_12345, the customer is upset"
Analyzed: intent=action, domain=payments, complexity=complex,
          segment=enterprise, previous_tool_failure=True

Included:
  [ALWAYS] system_prompt (2,500 tokens)
  [ON-DEMAND] error_recovery (800 tokens) -- previous failure
  [ON-DEMAND] compliance_text (1,200 tokens) -- payments domain
  [CONDITIONAL] domain_context: payments guidelines (1,800 tokens)
  [CONDITIONAL] user_profile: enterprise customer (400 tokens)
  [CONDITIONAL] conversation_history: last 5 messages (1,500 tokens)
  [ALWAYS] user_message (30 tokens)
  Total: 8,230 tokens (4.1% of 200K window)
```

## Failure Modes

| Failure | Cause | Impact | Fix |
|---------|-------|--------|-----|
| Wrong component included | Condition logic error | Irrelevant context, confusion | Test conditions with edge cases |
| Needed component excluded | Missing condition | LLM lacks information to answer | Monitor "I don't know" responses |
| Budget overflow | Too many components triggered | Context truncation | Priority-based trimming |
| Slow assembly | Too many async content loaders | High latency | Parallel loading, caching |
| Stale component content | Dynamic loader returns old data | Outdated information in context | TTL on component content |

## Source(s) and Further Reading

- Andrej Karpathy, "Context Engineering" (2025) -- per-request context assembly
- Simon Willison, "Building context-aware AI systems" (2025) -- dynamic injection patterns
- LangChain, "Dynamic Prompt Templates" -- framework implementation
- LlamaIndex, "Context Augmentation" -- RAG-specific context injection
