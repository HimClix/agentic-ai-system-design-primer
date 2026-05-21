# Guardrails AI

> Validate LLM inputs and outputs with composable Python validators -- think Pydantic for AI responses.

## What It Is

Guardrails AI is an open-source Python framework for validating LLM inputs and outputs. It provides a library of pre-built validators (PII detection, hallucination checks, schema compliance, toxicity filtering) and a decorator-based pattern for composing them into validation pipelines. Unlike NeMo Guardrails (which focuses on dialog flow control) or LlamaFirewall (which focuses on security), Guardrails AI focuses on **data quality and structural validation**.

**Key facts:**
- **License**: Apache 2.0
- **Hub**: 60+ community validators at hub.guardrailsai.com
- **Pattern**: Python decorators + RAIL spec (XML-based output schema)
- **Best for**: Structured output validation, PII protection, hallucination reduction
- **Latency**: 10-50ms per validation (fast, minimal overhead)

## How It Works

### Core Concept: Guard + Validators

```
User Input → [Input Validators] → LLM Call → [Output Validators] → Validated Output
                                                      │
                                                      ▼
                                              Validation Failed?
                                              → Re-ask LLM (up to N retries)
                                              → Return error
                                              → Return with fix applied
```

A **Guard** wraps an LLM call and applies **Validators** before and after the call:

- **Input validators**: Run on the prompt before it goes to the LLM
- **Output validators**: Run on the response before it reaches the user
- **On-fail actions**: What to do when validation fails (reask, fix, filter, refrain, exception, noop)

### Validator Categories

| Category | Validators | Use Case |
|----------|-----------|----------|
| **PII** | DetectPII, AnonymizePII | Prevent leaking personal data |
| **Hallucination** | FactualConsistency, ProvenanceCheck | Verify claims against sources |
| **Schema** | JsonSchema, PydanticModel | Enforce structured output format |
| **Toxicity** | ToxicLanguage, Profanity | Block harmful content |
| **Quality** | ReadabilityScore, GibberishDetector | Ensure response quality |
| **Relevance** | RelevancyEvaluator, AnswerRelevancy | Ensure on-topic responses |
| **Code** | ValidSQL, ValidPython, SQLInjection | Validate generated code |
| **Custom** | Any Python function | Domain-specific validation |

## Production Implementation

### Installation

```bash
pip install guardrails-ai

# Install specific validators from the hub
guardrails hub install hub://guardrails/detect_pii
guardrails hub install hub://guardrails/toxic_language
guardrails hub install hub://guardrails/provenance_llm
guardrails hub install hub://guardrails/valid_json
```

### Basic Usage: Output Validation

```python
"""
Guardrails AI -- Basic output validation with composable validators.
"""
from guardrails import Guard
from guardrails.hub import DetectPII, ToxicLanguage, ValidJson
from pydantic import BaseModel, Field
from typing import Optional


# ============================================================
# Pattern 1: Simple text validation
# ============================================================

guard = Guard().use_many(
    DetectPII(
        pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "SSN", "CREDIT_CARD"],
        on_fail="fix",  # Auto-redact PII
    ),
    ToxicLanguage(
        threshold=0.8,
        on_fail="refrain",  # Return empty if toxic
    ),
)

# Wrap any LLM call
result = guard(
    llm_api=openai.chat.completions.create,
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize the customer complaint"}],
)

print(result.validated_output)  # PII redacted, toxicity filtered
print(result.validation_passed)  # True/False
print(result.raw_llm_output)     # Original LLM response (for audit)


# ============================================================
# Pattern 2: Structured output with Pydantic
# ============================================================

class CustomerResponse(BaseModel):
    """Validated customer service response."""
    greeting: str = Field(description="Professional greeting")
    answer: str = Field(description="Answer to customer question")
    next_steps: list[str] = Field(description="Suggested next actions")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    sources: list[str] = Field(description="Sources referenced")

guard = Guard.from_pydantic(
    output_class=CustomerResponse,
    num_reasks=2,  # Retry up to 2 times if validation fails
)

# Add validators to specific fields
guard.use(
    DetectPII(
        pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"],
        on_fail="fix",
    ),
    on="answer",  # Only apply to the 'answer' field
)

result = guard(
    llm_api=openai.chat.completions.create,
    model="gpt-4o",
    messages=[{"role": "user", "content": "How do I reset my password?"}],
)

# result.validated_output is a CustomerResponse instance
response: CustomerResponse = result.validated_output
print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence}")


# ============================================================
# Pattern 3: Input validation
# ============================================================

input_guard = Guard().use(
    DetectPII(
        pii_entities=["SSN", "CREDIT_CARD"],
        on_fail="exception",  # Raise exception if PII in input
    ),
)

try:
    # Validate user input BEFORE sending to LLM
    input_result = input_guard.validate(user_input)
except Exception as e:
    print(f"Input rejected: {e}")
    # Return safe error message to user
```

### Production Setup: Custom Validators

```python
"""
Custom validators for domain-specific validation.
"""
from guardrails.validators import Validator, register_validator, ValidationResult
from guardrails import Guard
from typing import Any, Optional
import re


@register_validator(name="financial-amount-check", data_type="string")
class FinancialAmountCheck(Validator):
    """
    Validate that financial amounts in responses are within expected ranges.
    Catches hallucinated numbers (LLM saying "$1 billion" when data says "$1 million").
    """
    
    def __init__(
        self,
        max_amount: float = 1_000_000,
        currency: str = "USD",
        on_fail: str = "reask",
        **kwargs,
    ):
        super().__init__(
            max_amount=max_amount,
            currency=currency,
            on_fail=on_fail,
            **kwargs,
        )
        self.max_amount = max_amount
        self.currency = currency
    
    def validate(self, value: str, metadata: dict) -> ValidationResult:
        """Check all monetary amounts in the response."""
        # Find all dollar amounts
        amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?', value)
        
        for amount_str in amounts:
            # Parse the amount
            numeric = amount_str.replace('$', '').replace(',', '')
            multiplier = 1
            if 'billion' in amount_str.lower():
                multiplier = 1_000_000_000
            elif 'million' in amount_str.lower():
                multiplier = 1_000_000
            elif 'trillion' in amount_str.lower():
                multiplier = 1_000_000_000_000
            
            try:
                numeric_cleaned = re.sub(r'[^\d.]', '', numeric)
                parsed_amount = float(numeric_cleaned) * multiplier
            except ValueError:
                continue
            
            if parsed_amount > self.max_amount:
                return ValidationResult(
                    outcome="fail",
                    error_message=(
                        f"Amount {amount_str} exceeds maximum expected "
                        f"value of ${self.max_amount:,.2f}. "
                        f"Please verify this figure against source data."
                    ),
                )
        
        return ValidationResult(outcome="pass")


@register_validator(name="no-competitor-mentions", data_type="string")
class NoCompetitorMentions(Validator):
    """
    Ensure the response doesn't mention competitor products.
    """
    
    def __init__(
        self,
        competitors: list[str],
        on_fail: str = "fix",
        **kwargs,
    ):
        super().__init__(competitors=competitors, on_fail=on_fail, **kwargs)
        self.competitors = [c.lower() for c in competitors]
    
    def validate(self, value: str, metadata: dict) -> ValidationResult:
        value_lower = value.lower()
        found = [c for c in self.competitors if c in value_lower]
        
        if found:
            return ValidationResult(
                outcome="fail",
                error_message=f"Response mentions competitors: {found}. Remove these references.",
                fix_value=self._remove_competitors(value, found),
            )
        
        return ValidationResult(outcome="pass")
    
    def _remove_competitors(self, text: str, competitors: list[str]) -> str:
        for comp in competitors:
            text = re.sub(
                rf'\b{re.escape(comp)}\b',
                '[alternative solution]',
                text,
                flags=re.IGNORECASE,
            )
        return text


# ============================================================
# Compose custom + built-in validators
# ============================================================

production_guard = Guard().use_many(
    # Built-in
    DetectPII(pii_entities=["EMAIL_ADDRESS", "SSN"], on_fail="fix"),
    ToxicLanguage(threshold=0.8, on_fail="refrain"),
    # Custom
    FinancialAmountCheck(max_amount=10_000_000, on_fail="reask"),
    NoCompetitorMentions(competitors=["CompetitorA", "CompetitorB"], on_fail="fix"),
)
```

### Integration with LangGraph

```python
"""
Guardrails AI integrated into a LangGraph agent.
"""
from langgraph.graph import StateGraph, END
from guardrails import Guard
from guardrails.hub import DetectPII, ToxicLanguage
from typing import TypedDict


class AgentState(TypedDict):
    messages: list
    validated_response: str
    validation_passed: bool


# Create guard
guard = Guard().use_many(
    DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "SSN"], on_fail="fix"),
    ToxicLanguage(threshold=0.8, on_fail="refrain"),
)


def validate_input_node(state: AgentState) -> AgentState:
    """Input validation node."""
    last_message = state["messages"][-1]["content"]
    
    try:
        result = guard.validate(last_message)
        if not result.validation_passed:
            return {
                **state,
                "validated_response": "I noticed some issues with the input. Could you rephrase?",
                "validation_passed": False,
            }
    except Exception:
        return {
            **state,
            "validated_response": "Input validation failed. Please try again.",
            "validation_passed": False,
        }
    
    return {**state, "validation_passed": True}


def llm_node(state: AgentState) -> AgentState:
    """LLM call with output validation."""
    result = guard(
        llm_api=openai.chat.completions.create,
        model="gpt-4o",
        messages=state["messages"],
    )
    
    return {
        **state,
        "validated_response": result.validated_output or "I couldn't generate a valid response.",
        "validation_passed": result.validation_passed,
    }


def should_continue(state: AgentState) -> str:
    return "llm" if state["validation_passed"] else END


# Build graph
graph = StateGraph(AgentState)
graph.add_node("validate_input", validate_input_node)
graph.add_node("llm", llm_node)
graph.set_entry_point("validate_input")
graph.add_conditional_edges("validate_input", should_continue)
graph.add_edge("llm", END)
agent = graph.compile()
```

## Decision Tree / When to Use

```
Do you need to validate LLM OUTPUT STRUCTURE (JSON, Pydantic)?
  YES --> Guardrails AI (purpose-built for this)

Do you need PII DETECTION/REDACTION?
  YES --> Guardrails AI (hub://guardrails/detect_pii)

Do you need HALLUCINATION CHECKING against sources?
  YES --> Guardrails AI (hub://guardrails/provenance_llm)

Do you need DIALOG FLOW CONTROL?
  NO  --> Guardrails AI is NOT the right tool (use NeMo Guardrails)

Do you need PROMPT INJECTION DEFENSE?
  YES --> Guardrails AI is supplementary (use LlamaFirewall as primary)
```

## When NOT to Use

- **Dialog flow management** -- Use NeMo Guardrails (Colang is purpose-built for flows)
- **Prompt injection defense as primary** -- Use LlamaFirewall (AlignmentCheck is superior)
- **Real-time streaming validation** -- Guardrails AI validates complete outputs, not streams
- **Very high throughput (>1000 rps)** -- Consider async batch validation instead

## Tradeoffs

| Aspect | Guardrails AI | NeMo Guardrails | LlamaFirewall |
|--------|-------------|----------------|---------------|
| **Primary use** | I/O validation | Dialog control | Security |
| **Learning curve** | Very low (Python) | Medium (Colang DSL) | Low (Python API) |
| **Latency** | 10-50ms | 100-300ms | 100-500ms |
| **Retry/reask** | Built-in | Limited | No |
| **Schema enforcement** | Excellent | Limited | No |
| **PII detection** | Excellent | Basic | No |
| **Injection defense** | Basic | Good | Excellent |
| **Community validators** | 60+ on hub | Limited | Limited |

## Real-World Examples

1. **Healthcare chatbot** -- DetectPII removes patient identifiers from LLM responses before displaying. FinancialAmountCheck prevents hallucinated dosage numbers.

2. **Legal document generator** -- Pydantic schema ensures every generated contract has required clauses. ValidJson guarantees parseable output.

3. **Customer service agent** -- ToxicLanguage filter prevents frustrated agents from receiving aggressive AI responses. NoCompetitorMentions keeps responses brand-safe.

## Failure Modes

| Failure | Cause | Impact | Prevention |
|---------|-------|--------|------------|
| **Reask loop** | LLM cannot produce valid output | Infinite retries, high cost | Set max retries (num_reasks=2-3) |
| **PII false positive** | Validator flags non-PII as PII | Legitimate data redacted | Tune PII entity types + thresholds |
| **Schema too strict** | Pydantic model rejects valid responses | High failure rate | Start loose, tighten gradually |
| **Validator conflict** | Two validators produce contradictory fixes | Unpredictable output | Test validator combinations |
| **Performance at scale** | Many validators per call | Latency accumulation | Profile and prioritize validators |

## Source(s) and Further Reading

- Guardrails AI Documentation: https://docs.guardrailsai.com/
- Guardrails Hub (60+ validators): https://hub.guardrailsai.com/
- Guardrails AI GitHub: https://github.com/guardrails-ai/guardrails
- "Getting Started with Guardrails AI" (2024): https://docs.guardrailsai.com/getting_started/
