# Production Reliability Playbook
> The complete mitigation stack that reduces agent failure rates from 41-87% to single digits -- every technique, when to apply it, and the code to implement it.

## What It Is

This is the operational playbook for making agentic systems production-reliable. It combines every proven mitigation technique into a layered defense strategy. Each layer catches failures that slip through the previous one.

The goal: reduce the 41-87% baseline failure rate to <5% for production workloads by stacking mitigations that address different failure categories.

## How It Works

### The Reliability Stack (Layer by Layer)

```
Layer 7: Human Escalation      ← Catches: novel failures, edge cases
Layer 6: Output Validation      ← Catches: FM-3.1, FM-3.2, FM-3.3
Layer 5: Confidence Scoring     ← Catches: uncertain outputs
Layer 4: Fallback Ladders       ← Catches: model/tool failures
Layer 3: Circuit Breakers       ← Catches: FM-2.1, FM-2.2 (tool failures)
Layer 2: Bounded Autonomy       ← Catches: FM-1.3, FM-1.5 (loops)
Layer 1: Deterministic Guards   ← Catches: FM-1.1 (requirement violations)
Layer 0: State Machine          ← Catches: FM-1.4, FM-2.3, FM-2.4
```

## Production Implementation

### Layer 0: State Machine Design

The foundation. Use LangGraph to make agent behavior deterministic at the macro level.

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Literal
import operator


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    step_count: int
    current_phase: str
    errors: list[str]
    confidence: float
    requires_approval: bool


def build_reliable_agent() -> StateGraph:
    """Build an agent with reliability layers baked into the graph."""
    graph = StateGraph(AgentState)
    
    # Nodes
    graph.add_node("parse_task", parse_task)
    graph.add_node("plan", create_plan)
    graph.add_node("execute", execute_step)
    graph.add_node("verify", verify_output)
    graph.add_node("format_response", format_response)
    graph.add_node("escalate", escalate_to_human)
    graph.add_node("error_recovery", handle_error)
    
    # Edges with conditional routing
    graph.set_entry_point("parse_task")
    graph.add_edge("parse_task", "plan")
    graph.add_conditional_edges("plan", route_after_plan)
    graph.add_conditional_edges("execute", route_after_execute)
    graph.add_conditional_edges("verify", route_after_verify)
    graph.add_edge("format_response", END)
    graph.add_edge("escalate", END)
    graph.add_conditional_edges("error_recovery", route_after_error)
    
    return graph.compile()


def route_after_execute(state: AgentState) -> Literal["verify", "error_recovery", "escalate"]:
    if state["step_count"] > 25:
        return "escalate"  # Bounded autonomy
    if state["errors"] and len(state["errors"]) > 3:
        return "escalate"  # Too many errors
    if state["errors"]:
        return "error_recovery"
    return "verify"


def route_after_verify(state: AgentState) -> Literal["format_response", "execute", "escalate"]:
    if state["confidence"] >= 0.85:
        return "format_response"  # Output is good
    if state["step_count"] < 20:
        return "execute"  # Try again
    return "escalate"  # Can't get good output, ask human
```

### Layer 1: Deterministic Guards

Rules that never need LLM judgment. Enforced before and after every step.

```python
from dataclasses import dataclass

@dataclass 
class GuardResult:
    passed: bool
    violations: list[str]


class DeterministicGuards:
    """Hard rules that must be satisfied regardless of LLM output."""
    
    def __init__(self):
        self.rules: list[callable] = []
    
    def add_rule(self, name: str, check_fn: callable):
        self.rules.append({"name": name, "check": check_fn})
    
    def evaluate(self, output: str, context: dict) -> GuardResult:
        violations = []
        for rule in self.rules:
            try:
                if not rule["check"](output, context):
                    violations.append(rule["name"])
            except Exception as e:
                violations.append(f"{rule['name']}: {str(e)}")
        
        return GuardResult(passed=len(violations) == 0, violations=violations)


# Example guards
guards = DeterministicGuards()

# Format guard: output must be valid JSON if JSON was requested
guards.add_rule("json_format", lambda output, ctx: 
    not ctx.get("require_json") or is_valid_json(output))

# Length guard: output must not exceed max length
guards.add_rule("max_length", lambda output, ctx:
    len(output) <= ctx.get("max_output_chars", 10_000))

# PII guard: output must not contain PII patterns
guards.add_rule("no_pii", lambda output, ctx:
    not contains_pii(output))

# SQL injection guard: tool parameters must not contain SQL injection
guards.add_rule("no_sql_injection", lambda output, ctx:
    not contains_sql_injection(output))
```

### Layer 2: Bounded Autonomy

Limit what the agent can do and for how long.

```python
@dataclass
class AutonomyBounds:
    max_steps: int = 25
    max_tool_calls: int = 15
    max_llm_calls: int = 20
    max_wall_time_seconds: int = 300  # 5 minutes
    max_tokens_per_session: int = 50_000
    allowed_tools: set[str] = None  # None = all allowed
    blocked_actions: set[str] = None  # e.g., {"delete_file", "send_email"}
    require_approval_for: set[str] = None  # e.g., {"execute_payment"}


class BoundedAutonomyEnforcer:
    def __init__(self, bounds: AutonomyBounds):
        self.bounds = bounds
        self.step_count = 0
        self.tool_calls = 0
        self.llm_calls = 0
        self.start_time = time.time()
        self.tokens_used = 0
    
    def check_before_step(self, action: str, tool: str = None) -> tuple[bool, str]:
        """Check if the next action is allowed within bounds."""
        # Step limit
        if self.step_count >= self.bounds.max_steps:
            return False, f"Step limit reached ({self.bounds.max_steps})"
        
        # Wall time
        elapsed = time.time() - self.start_time
        if elapsed > self.bounds.max_wall_time_seconds:
            return False, f"Time limit reached ({self.bounds.max_wall_time_seconds}s)"
        
        # Token budget
        if self.tokens_used >= self.bounds.max_tokens_per_session:
            return False, f"Token budget exceeded ({self.bounds.max_tokens_per_session})"
        
        # Tool allowlist
        if tool and self.bounds.allowed_tools and tool not in self.bounds.allowed_tools:
            return False, f"Tool '{tool}' not in allowed list"
        
        # Blocked actions
        if self.bounds.blocked_actions and action in self.bounds.blocked_actions:
            return False, f"Action '{action}' is blocked"
        
        # Approval required
        if self.bounds.require_approval_for and action in self.bounds.require_approval_for:
            return False, f"Action '{action}' requires human approval"
        
        self.step_count += 1
        return True, "OK"
```

### Layer 3: Per-Tool Circuit Breakers

```python
import time
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


class ToolCircuitBreaker:
    """Per-tool circuit breaker to prevent cascading failures."""
    
    def __init__(
        self,
        tool_name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        self.tool_name = tool_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
    
    def can_call(self) -> tuple[bool, str]:
        """Check if the tool can be called."""
        if self.state == CircuitState.CLOSED:
            return True, "OK"
        
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True, "Circuit half-open, testing recovery"
            return False, f"Circuit OPEN for {self.tool_name}, retry after {self.recovery_timeout}s"
        
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True, "Half-open test call"
            return False, f"Circuit HALF_OPEN, max test calls reached for {self.tool_name}"
        
        return False, "Unknown state"
    
    def record_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
        self.failure_count = 0
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

### Layer 4: Fallback Ladders

```python
class FallbackLadder:
    """Try increasingly degraded approaches when primary fails."""
    
    def __init__(self):
        self.strategies: list[dict] = []
    
    def add_strategy(self, name: str, fn: callable, description: str):
        self.strategies.append({"name": name, "fn": fn, "desc": description})
    
    async def execute(self, context: dict) -> dict:
        """Try each strategy in order until one succeeds."""
        errors = []
        for strategy in self.strategies:
            try:
                result = await strategy["fn"](context)
                return {
                    "success": True,
                    "result": result,
                    "strategy_used": strategy["name"],
                    "strategies_tried": len(errors) + 1,
                }
            except Exception as e:
                errors.append({"strategy": strategy["name"], "error": str(e)})
        
        return {
            "success": False,
            "errors": errors,
            "message": "All strategies exhausted",
        }


# Example: RAG fallback ladder
rag_fallback = FallbackLadder()
rag_fallback.add_strategy("vector_search", vector_search, "Primary: semantic search")
rag_fallback.add_strategy("keyword_search", keyword_search, "Fallback 1: BM25 keyword search")
rag_fallback.add_strategy("web_search", web_search, "Fallback 2: web search")
rag_fallback.add_strategy("admit_unknown", lambda ctx: "I don't have information about this.", 
                          "Final: admit lack of knowledge")
```

### Layer 5: Confidence Scoring

```python
@dataclass
class ConfidenceScore:
    value: float  # 0.0 to 1.0
    factors: dict[str, float]
    action: str  # "proceed", "verify", "escalate"


class ConfidenceScorer:
    """Score confidence based on multiple signals, not just LLM self-report."""
    
    def score(
        self,
        llm_confidence: float,
        tool_success_rate: float,
        retrieval_relevance: float,
        output_consistency: float,  # Same output on retry?
    ) -> ConfidenceScore:
        factors = {
            "llm_confidence": llm_confidence * 0.2,      # Low weight: LLMs overestimate
            "tool_success": tool_success_rate * 0.3,      # Tools worked correctly
            "retrieval_relevance": retrieval_relevance * 0.3,  # RAG found relevant docs
            "output_consistency": output_consistency * 0.2,    # Stable output
        }
        
        total = sum(factors.values())
        
        if total >= 0.85:
            action = "proceed"
        elif total >= 0.60:
            action = "verify"
        else:
            action = "escalate"
        
        return ConfidenceScore(value=total, factors=factors, action=action)
```

### Layer 6: Output Validation

```python
class OutputValidator:
    """Deterministic + LLM validation of agent outputs."""
    
    def validate(self, output: str, task: dict) -> dict:
        results = {}
        
        # Deterministic checks (always run)
        results["format_valid"] = self._check_format(output, task.get("output_format"))
        results["length_ok"] = len(output) <= task.get("max_length", 10_000)
        results["no_hallucination_markers"] = not self._has_hallucination_markers(output)
        
        # Content checks
        if task.get("must_contain"):
            results["contains_required"] = all(
                term in output for term in task["must_contain"]
            )
        
        if task.get("must_not_contain"):
            results["excludes_forbidden"] = not any(
                term in output for term in task["must_not_contain"]
            )
        
        all_passed = all(results.values())
        return {"passed": all_passed, "checks": results}
    
    def _has_hallucination_markers(self, text: str) -> bool:
        """Check for common hallucination indicators."""
        markers = [
            "I don't have access to",
            "As an AI language model",
            "I cannot verify",
            "Based on my training data",
        ]
        return any(m.lower() in text.lower() for m in markers)
```

## Production Readiness Checklist

```
RELIABILITY LAYER 0: State Machine
[ ] Agent uses explicit state graph (LangGraph/similar)
[ ] All valid state transitions are defined
[ ] No implicit state transitions possible
[ ] Terminal states are explicitly defined

RELIABILITY LAYER 1: Deterministic Guards
[ ] Input validation on all user inputs
[ ] Output format validation (JSON, length, etc.)
[ ] PII detection and scrubbing
[ ] SQL/command injection prevention
[ ] Content policy enforcement

RELIABILITY LAYER 2: Bounded Autonomy
[ ] Maximum step count per session (e.g., 25)
[ ] Maximum wall-clock time per session (e.g., 5 min)
[ ] Maximum token budget per session
[ ] Tool allowlist per agent role
[ ] Blocked action list for dangerous operations
[ ] Approval gates for write operations

RELIABILITY LAYER 3: Circuit Breakers
[ ] Per-tool circuit breaker with failure threshold
[ ] Recovery timeout and half-open testing
[ ] Circuit breaker state exposed in metrics
[ ] Alert on circuit breaker OPEN events

RELIABILITY LAYER 4: Fallback Ladders
[ ] Primary → secondary → tertiary strategy per capability
[ ] "Admit unknown" as final fallback (never hallucinate)
[ ] Fallback strategy logged for analysis
[ ] Degraded mode notification to user

RELIABILITY LAYER 5: Confidence Scoring
[ ] Multi-signal confidence (not just LLM self-report)
[ ] Per-action-type confidence thresholds
[ ] Low-confidence outputs routed to verification
[ ] Confidence scores logged for threshold tuning

RELIABILITY LAYER 6: Output Validation
[ ] Deterministic format validation
[ ] Content completeness check
[ ] Hallucination marker detection
[ ] Required/forbidden content check
[ ] Validation failures trigger retry or escalation

RELIABILITY LAYER 7: Human Escalation
[ ] Escalation queue with SLA monitoring
[ ] Escalation criteria clearly defined
[ ] User notified of escalation with ETA
[ ] Escalation context includes full agent trace
[ ] Feedback from escalation fed back to improve agent

OBSERVABILITY
[ ] Every agent step logged with latency
[ ] Tool call success/failure rates tracked
[ ] Token usage per step and per session tracked
[ ] Error categorization by MAST failure mode
[ ] Dashboard with reliability metrics
[ ] Alerts on failure rate > threshold

TESTING
[ ] Unit tests for each tool with mocked LLM
[ ] Integration tests with real LLM calls
[ ] Regression test suite for known failure cases
[ ] Trajectory evaluation on golden datasets
[ ] Chaos testing (tool failures, timeout, rate limits)
[ ] Load testing with concurrent sessions

DEPLOYMENT
[ ] Canary deployments for prompt changes
[ ] Rollback plan for prompt regressions
[ ] Feature flags for new tool integrations
[ ] Staged rollout (1% → 10% → 50% → 100%)
```

## When NOT to Use the Full Stack

- **MVP/Prototype**: Use Layers 0-2 only. Skip circuit breakers and confidence scoring.
- **Internal tools**: Skip Layers 5-7 if users are engineers who can handle failures.
- **Single-turn agents**: Layers 0 and 1 are sufficient. No session management needed.

## Tradeoffs

| Full Stack | Minimal Stack |
|-----------|--------------|
| <5% failure rate | 20-40% failure rate |
| 4-6 weeks to implement | 1 week |
| +100-300ms latency | Minimal overhead |
| Complex debugging | Simple debugging |
| Suitable for customer-facing | Suitable for internal |

## Source(s) and Further Reading

- MAST Taxonomy (NeurIPS 2025)
- "Building Effective Agents" - Anthropic (2024)
- LangGraph Reliability Patterns: https://langchain-ai.github.io/langgraph/
- Circuit Breaker Pattern: Martin Fowler's "CircuitBreaker" (2014)
- "Production-Ready LLM Applications" - Chip Huyen (2024)
