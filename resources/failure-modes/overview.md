# Failure Modes in Agentic Systems
> 41-86.7% of agent runs fail. This is an architecture problem, not a model problem -- the MAST taxonomy identifies 14 distinct failure modes from 1,642 real traces.

## What It Is

Agent failure modes are the systematic ways agentic AI systems fail in production. Unlike traditional software where failures are primarily bugs in deterministic code, agent failures emerge from the interaction between nondeterministic LLM reasoning, tool execution, and system architecture.

The MAST (Multi-Agent System Taxonomy) study from NeurIPS 2025 analyzed 1,642 execution traces across 7 agent frameworks (AutoGPT, BabyAGI, CrewAI, LangGraph, MetaGPT, CAMEL, ChatDev) and found:

- **41-86.7% failure rate** across frameworks
- **14 distinct failure modes** categorized into 3 families
- Most failures are **architecture-caused**, not model-caused

The key insight: upgrading to a better model fixes only ~30% of failures. The remaining 70% require architectural mitigations.

## How It Works

### Three Failure Categories

```
Total Agent Failures (100%)
│
├── Specification Failures (41.77%)
│   ├── How the task is defined and decomposed
│   ├── Agent misunderstands requirements
│   └── Agent doesn't know when it's done
│
├── Coordination Failures (36.94%)
│   ├── How agents work together or sequence steps
│   ├── Tool selection errors
│   └── Infinite loops and repetitive behavior
│
└── Verification Failures (21.30%)
    ├── How results are checked
    ├── Wrong outputs accepted as correct
    └── Correct outputs rejected
```

### Failure Rate by Framework

| Framework | Failure Rate | Dominant Failure Type |
|-----------|-------------|----------------------|
| AutoGPT | 86.7% | Coordination (step repetition) |
| BabyAGI | 78.3% | Specification (task decomposition) |
| CAMEL | 72.1% | Coordination (role confusion) |
| ChatDev | 58.4% | Specification (requirement misparse) |
| CrewAI | 53.2% | Coordination (delegation loops) |
| MetaGPT | 47.8% | Verification (output quality) |
| LangGraph | 41.0% | Specification (state management) |

LangGraph's lower failure rate correlates with its explicit state machine design, which prevents many coordination failures by making transitions deterministic.

## Production Implementation

```python
from enum import Enum
from dataclasses import dataclass


class FailureCategory(Enum):
    SPECIFICATION = "specification"    # 41.77%
    COORDINATION = "coordination"      # 36.94%
    VERIFICATION = "verification"      # 21.30%


class FailureMode(Enum):
    # Specification failures
    FM_1_1_DISOBEY_REQUIREMENTS = "disobey_task_requirements"      # 11.8%
    FM_1_2_WRONG_DECOMPOSITION = "wrong_task_decomposition"        # 6.2%
    FM_1_3_STEP_REPETITION = "step_repetition"                     # 15.7%
    FM_1_4_WRONG_STEP_ORDER = "wrong_step_order"                   # 3.8%
    FM_1_5_NOT_RECOGNIZING_COMPLETION = "not_recognizing_completion" # 12.4%
    
    # Coordination failures
    FM_2_1_WRONG_TOOL_SELECTION = "wrong_tool_selection"            # 8.3%
    FM_2_2_TOOL_MISUSE = "tool_parameter_misuse"                   # 7.1%
    FM_2_3_DELEGATION_ERROR = "delegation_error"                   # 5.9%
    FM_2_4_ROLE_CONFUSION = "role_confusion"                       # 4.8%
    FM_2_5_COMMUNICATION_BREAKDOWN = "communication_breakdown"     # 6.2%
    FM_2_6_RESOURCE_CONFLICT = "resource_conflict"                 # 4.7%
    
    # Verification failures
    FM_3_1_FALSE_POSITIVE = "false_positive_verification"          # 8.4%
    FM_3_2_FALSE_NEGATIVE = "false_negative_verification"          # 5.1%
    FM_3_3_NO_VERIFICATION = "no_verification_performed"           # 7.8%


@dataclass
class FailureEvent:
    mode: FailureMode
    category: FailureCategory
    description: str
    step_number: int
    recoverable: bool
    mitigation_applied: str = ""


class FailureDetector:
    """Detect common failure modes during agent execution."""
    
    def __init__(self, max_steps: int = 25, max_tool_retries: int = 3):
        self.max_steps = max_steps
        self.max_tool_retries = max_tool_retries
        self._step_history: list[str] = []
        self._tool_call_counts: dict[str, int] = {}
    
    def check_step_repetition(self, current_step: str) -> bool:
        """FM-1.3: Detect if the agent is repeating the same step."""
        self._step_history.append(current_step)
        if len(self._step_history) >= 3:
            last_three = self._step_history[-3:]
            if len(set(last_three)) == 1:
                return True  # Same step 3 times in a row
        return False
    
    def check_completion_recognition(
        self, step_count: int, task_outputs: list
    ) -> bool:
        """FM-1.5: Detect if agent should have stopped but hasn't."""
        if step_count > self.max_steps:
            return True
        # Check if the last 3 outputs are substantially similar
        if len(task_outputs) >= 3:
            last_three = task_outputs[-3:]
            if self._outputs_similar(last_three):
                return True
        return False
    
    def check_tool_misuse(self, tool_name: str, error_count: int) -> bool:
        """FM-2.2: Detect repeated tool failures."""
        self._tool_call_counts[tool_name] = self._tool_call_counts.get(tool_name, 0) + 1
        return error_count >= self.max_tool_retries
    
    def _outputs_similar(self, outputs: list) -> bool:
        """Check if outputs are too similar (indicating a loop)."""
        # Simple heuristic: compare string lengths within 10%
        lengths = [len(str(o)) for o in outputs]
        avg = sum(lengths) / len(lengths)
        return all(abs(l - avg) / max(avg, 1) < 0.1 for l in lengths)
```

## Decision Tree: Diagnosing Agent Failures

```
Agent returned wrong result or didn't complete:
│
├── Did the agent understand the task correctly?
│   ├── NO → Specification failure (FM-1.1, FM-1.2)
│   │   └── Fix: Improve system prompt, add examples, validate task parsing
│   └── YES → Continue diagnosis
│
├── Did the agent get stuck in a loop?
│   ├── YES → Step repetition (FM-1.3) or completion blindness (FM-1.5)
│   │   └── Fix: Add step counter, loop detector, explicit completion criteria
│   └── NO → Continue diagnosis
│
├── Did the agent use the right tools?
│   ├── Wrong tool selected → FM-2.1
│   │   └── Fix: Better tool descriptions, fewer similar tools
│   ├── Right tool, wrong parameters → FM-2.2
│   │   └── Fix: Stricter tool schemas, parameter validation
│   └── Tools worked correctly → Continue diagnosis
│
├── Was the result verified?
│   ├── No verification done → FM-3.3
│   │   └── Fix: Add deterministic validators
│   ├── Verified but wrong accepted → FM-3.1
│   │   └── Fix: Stricter validation criteria
│   └── Verified correctly → The system worked, check for edge cases
│
└── Multi-agent specific:
    ├── Delegation went to wrong agent → FM-2.3
    ├── Agents confused about roles → FM-2.4
    └── Agents can't communicate results → FM-2.5
```

## When NOT to Focus on Failure Modes

- **Prototype stage**: Accept failures, iterate on the happy path first.
- **Single-turn classification**: Failure modes are mostly relevant to multi-step agents.
- **Human-in-the-loop always on**: If a human reviews every output, many failure modes are caught naturally.

## Tradeoffs

| Mitigation Strategy | Failure Modes Addressed | Cost | Latency Impact |
|---------------------|------------------------|------|---------------|
| Step counter / max steps | FM-1.3, FM-1.5 | None | None |
| Tool call circuit breaker | FM-2.1, FM-2.2 | None | None |
| Deterministic validators | FM-3.1, FM-3.2, FM-3.3 | Low | +50-100ms |
| State machine (LangGraph) | FM-1.3, FM-1.4, FM-2.3 | Medium | None |
| Explicit completion criteria | FM-1.5 | Low | None |
| Tool description optimization | FM-2.1, FM-2.2 | Low | None |
| Output verification LLM call | FM-3.1, FM-3.2 | High ($) | +500-1000ms |

## Real-World Examples

- **AutoGPT (2023)**: Step repetition (FM-1.3) was the #1 reported issue. Agents would "web search → read page → web search the same thing" indefinitely.
- **ChatDev code generation**: Would generate code, run tests, see failures, and regenerate identical code -- completion blindness (FM-1.5).
- **Enterprise RAG agents**: Would retrieve wrong documents but proceed to answer confidently -- false positive verification (FM-3.1).

## Failure Modes

1. **Over-engineering mitigation**: Adding so many checks that the agent can't complete simple tasks.
2. **False positive failure detection**: Detector flags legitimate exploratory behavior as a loop.
3. **Mitigation masking root causes**: Circuit breakers hide broken tool integrations instead of fixing them.

## Source(s) and Further Reading

- MAST Taxonomy (NeurIPS 2025): "A Taxonomy of Agent Failure Modes in Multi-Agent Systems"
- "Multi-Agent Failure Analysis: A Study of 1,642 Execution Traces" (2025)
- AutoGPT Issues: https://github.com/Significant-Gravitas/AutoGPT/issues
- "Building Effective Agents" - Anthropic (2024)
- "Why Agent Benchmarks Don't Predict Production Performance" - LangChain Blog (2025)
