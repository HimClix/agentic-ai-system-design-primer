# Model Routing
> Route 60-70% of requests to cheap models, reserve frontier models for complex reasoning -- same quality, fraction of the cost.

## What It Is

Model routing is the practice of dynamically selecting which LLM to use for each step of an agent's execution based on the complexity, risk, and nature of the task. Instead of using a single expensive frontier model for everything, a lightweight classifier routes each request to the cheapest model capable of handling it correctly.

The economic insight: in a typical agentic workflow, only 15-25% of LLM calls require frontier-level reasoning. The rest -- classification, formatting, extraction, summarization -- can be handled by models that cost 5-20x less.

## How It Works

### The Model Tier System

```
Tier 1 - CHEAP ($0.25-0.80/M input)
├── Claude Haiku 3.5: $0.80/$4.00
├── GPT-4o-mini: $0.15/$0.60
├── Gemini 2.0 Flash: $0.10/$0.40
└── Use for: classification, extraction, formatting, simple Q&A

Tier 2 - STANDARD ($3.00/M input)  
├── Claude Sonnet 4: $3.00/$15.00
├── GPT-4o: $2.50/$10.00
├── Gemini 2.5 Pro: $1.25/$10.00
└── Use for: reasoning, planning, code generation, analysis

Tier 3 - PREMIUM ($15.00/M input)
├── Claude Opus 4: $15.00/$75.00
├── o1/o3: $15.00/$60.00
└── Use for: complex multi-step reasoning, novel problems, safety-critical
```

### Savings Calculation

```
Scenario: 1,000 agent sessions/day, 8 LLM calls per session

Without routing (all Sonnet):
  8,000 calls x 2,000 avg tokens = 16M tokens/day
  Cost: 16 x $3.00 = $48/day input + output ≈ $96/day
  Monthly: $2,880

With routing (60% Haiku, 30% Sonnet, 10% Opus):
  4,800 Haiku calls: 9.6M tokens x $0.80/M  = $7.68
  2,400 Sonnet calls: 4.8M tokens x $3.00/M  = $14.40
  800 Opus calls:     1.6M tokens x $15.00/M = $24.00
  Input cost: $46.08/day + output ≈ $92/day
  
  Wait -- that's almost the same! The savings come from output tokens:
  
  Haiku output:  4,800 x 400 tokens x $4.00/M  = $7.68
  Sonnet output: 2,400 x 400 tokens x $15.00/M = $14.40
  Opus output:   800 x 400 tokens x $75.00/M   = $24.00
  
  vs All-Sonnet output: 8,000 x 400 x $15.00/M = $48.00
  
  Total with routing: ~$138/day = $4,140/month
  Total all-Sonnet:   ~$144/day = $4,320/month
  
  Real savings come from replacing Opus-for-everything:
  All-Opus: 16M input x $15 + 3.2M output x $75 = $480/day = $14,400/month
  With routing: $138/day = $4,140/month → 71% savings
```

## Production Implementation

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import re


class ModelTier(Enum):
    CHEAP = "cheap"
    STANDARD = "standard"  
    PREMIUM = "premium"


@dataclass
class RoutingDecision:
    tier: ModelTier
    model: str
    reason: str
    confidence: float


# Task classification patterns
TASK_PATTERNS = {
    ModelTier.CHEAP: [
        "classify", "extract", "format", "parse", "summarize_short",
        "yes_no", "multiple_choice", "template_fill", "translate",
        "sentiment", "entity_extraction", "slot_filling",
    ],
    ModelTier.STANDARD: [
        "reason", "analyze", "plan", "code_generate", "explain",
        "compare", "recommend", "debug", "synthesize", "draft",
    ],
    ModelTier.PREMIUM: [
        "complex_reasoning", "novel_problem", "safety_critical",
        "multi_step_math", "legal_analysis", "medical_advice",
        "ambiguous_requirements", "creative_writing_long",
    ],
}

MODEL_MAP = {
    ModelTier.CHEAP: "claude-haiku-3.5",
    ModelTier.STANDARD: "claude-sonnet-4",
    ModelTier.PREMIUM: "claude-opus-4",
}


class ModelRouter:
    """Routes LLM requests to the appropriate model tier."""
    
    def __init__(self, default_tier: ModelTier = ModelTier.STANDARD):
        self.default_tier = default_tier
        self._step_overrides: dict[str, ModelTier] = {}
    
    def register_step(self, step_name: str, tier: ModelTier):
        """Pre-register a workflow step to always use a specific tier."""
        self._step_overrides[step_name] = tier
    
    def route(
        self,
        step_name: Optional[str] = None,
        task_description: Optional[str] = None,
        token_count: Optional[int] = None,
        requires_tool_use: bool = False,
        is_final_output: bool = False,
    ) -> RoutingDecision:
        """Determine which model tier to use for this request."""
        
        # 1. Check pre-registered step overrides
        if step_name and step_name in self._step_overrides:
            tier = self._step_overrides[step_name]
            return RoutingDecision(
                tier=tier,
                model=MODEL_MAP[tier],
                reason=f"step_override:{step_name}",
                confidence=1.0,
            )
        
        # 2. Route based on task type classification
        if task_description:
            tier = self._classify_task(task_description)
            return RoutingDecision(
                tier=tier,
                model=MODEL_MAP[tier],
                reason=f"task_classification:{task_description[:50]}",
                confidence=0.85,
            )
        
        # 3. Heuristic routing
        if token_count and token_count < 500 and not requires_tool_use:
            return RoutingDecision(
                tier=ModelTier.CHEAP,
                model=MODEL_MAP[ModelTier.CHEAP],
                reason="small_request_no_tools",
                confidence=0.7,
            )
        
        if is_final_output:
            # Final user-facing output deserves a better model
            return RoutingDecision(
                tier=ModelTier.STANDARD,
                model=MODEL_MAP[ModelTier.STANDARD],
                reason="final_output",
                confidence=0.8,
            )
        
        # 4. Default
        return RoutingDecision(
            tier=self.default_tier,
            model=MODEL_MAP[self.default_tier],
            reason="default",
            confidence=0.5,
        )
    
    def _classify_task(self, description: str) -> ModelTier:
        """Simple keyword-based task classification."""
        desc_lower = description.lower()
        
        for tier in [ModelTier.PREMIUM, ModelTier.CHEAP]:
            for pattern in TASK_PATTERNS[tier]:
                if pattern in desc_lower:
                    return tier
        
        return ModelTier.STANDARD


# --- LiteLLM integration ---

def get_litellm_router_config() -> dict:
    """Generate LiteLLM router configuration for model routing."""
    return {
        "model_list": [
            {
                "model_name": "agent-cheap",
                "litellm_params": {
                    "model": "anthropic/claude-3-5-haiku-latest",
                    "api_key": "os.environ/ANTHROPIC_API_KEY",
                },
                "model_info": {"max_tokens": 8192},
            },
            {
                "model_name": "agent-standard",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-20250514",
                    "api_key": "os.environ/ANTHROPIC_API_KEY",
                },
                "model_info": {"max_tokens": 16384},
            },
            {
                "model_name": "agent-premium",
                "litellm_params": {
                    "model": "anthropic/claude-opus-4-20250514",
                    "api_key": "os.environ/ANTHROPIC_API_KEY",
                },
                "model_info": {"max_tokens": 32768},
            },
        ],
        "router_settings": {
            "routing_strategy": "simple-shuffle",  # or "least-busy"
            "num_retries": 2,
            "timeout": 60,
            "fallbacks": [
                {"agent-premium": ["agent-standard"]},
                {"agent-standard": ["agent-cheap"]},
            ],
        },
    }


# --- Usage in LangGraph ---

from langgraph.graph import StateGraph

router = ModelRouter()

# Pre-register workflow steps
router.register_step("classify_intent", ModelTier.CHEAP)
router.register_step("extract_entities", ModelTier.CHEAP)
router.register_step("generate_plan", ModelTier.STANDARD)
router.register_step("execute_tool_call", ModelTier.STANDARD)
router.register_step("synthesize_response", ModelTier.STANDARD)
router.register_step("safety_check", ModelTier.CHEAP)
router.register_step("complex_reasoning", ModelTier.PREMIUM)
```

## Decision Tree: Which Model for Which Step

```
Agent Step
│
├── Classification / Intent Detection
│   └── CHEAP (Haiku) -- accuracy matches larger models for classification
│
├── Entity Extraction / Slot Filling  
│   └── CHEAP (Haiku) -- structured extraction is well-handled
│
├── Tool Call Formatting
│   └── CHEAP (Haiku) -- generating JSON tool calls is pattern-matching
│
├── Simple Q&A (FAQ, known answers)
│   └── CHEAP (Haiku) -- retrieval + template response
│
├── Summarization (< 2000 tokens input)
│   └── CHEAP (Haiku) -- extractive summarization works fine
│
├── Planning / Task Decomposition
│   └── STANDARD (Sonnet) -- needs reasoning about task structure
│
├── Code Generation / Debugging
│   └── STANDARD (Sonnet) -- reasoning + knowledge required
│
├── Analysis / Comparison
│   └── STANDARD (Sonnet) -- multi-factor reasoning
│
├── Final User-Facing Response
│   └── STANDARD (Sonnet) -- quality of output matters most here
│
├── Complex Multi-Step Reasoning
│   └── PREMIUM (Opus) -- novel logic chains
│
├── Ambiguous / Underspecified Tasks
│   └── PREMIUM (Opus) -- needs to ask clarifying questions
│
└── Safety-Critical Decisions  
    └── PREMIUM (Opus) -- financial, medical, legal implications
```

## When NOT to Use Model Routing

- **Latency-sensitive applications**: Router adds 20-100ms per decision. If you need <200ms total latency, use a single model.
- **When quality is paramount**: If a 1% quality drop is unacceptable (medical, legal), use the best model everywhere.
- **Simple single-step agents**: Routing overhead isn't worth it for one LLM call.

## Tradeoffs

| Approach | Savings | Quality Risk | Complexity | Latency |
|----------|---------|-------------|------------|---------|
| Static step mapping | 50-60% | Low (predictable) | Low | +0ms |
| LLM-based router | 60-70% | Medium (router errors) | High | +100-200ms |
| Keyword heuristic | 40-50% | Medium | Low | +5ms |
| Cascade (try cheap first) | 30-40% | Low (falls back) | Medium | +200ms on fallback |

## Real-World Examples

- **Document analysis pipeline**: Routing entity extraction to Haiku and synthesis to Sonnet: $0.34/doc vs $1.40 all-frontier (76% savings).
- **Customer support agent**: Intent classification (Haiku) -> knowledge retrieval (Haiku) -> response generation (Sonnet) -> sentiment check (Haiku): 60% of calls on Haiku.
- **Code review agent**: Diff parsing (Haiku) -> pattern matching (Haiku) -> complex analysis (Sonnet) -> security review (Opus): tiered by risk level.

## Failure Modes

1. **Misrouting complex queries to cheap models**: User asks a nuanced question, router sends to Haiku, poor answer. Mitigation: err toward higher tier when confidence < 0.7.
2. **Router itself consuming tokens**: Using an LLM to route adds cost. Mitigation: use heuristic or small classifier, not another LLM call.
3. **Fallback cascades**: Premium unavailable -> Standard unavailable -> everything on Cheap. Mitigation: circuit breakers per tier, alert on cascade.
4. **Inconsistent quality**: User notices quality varies between messages. Mitigation: use Standard minimum for user-facing outputs.

## Source(s) and Further Reading

- LiteLLM Router: https://docs.litellm.ai/docs/routing
- Anthropic Model Selection Guide: https://docs.anthropic.com/en/docs/about-claude/models
- "Building Effective Agents" - Anthropic (2024)
- Martian Model Router: https://withmartian.com/ (commercial routing service)
