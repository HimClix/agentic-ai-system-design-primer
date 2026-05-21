# MAST Failure Taxonomy Deep Dive
> 14 failure modes, 1,642 traces, 7 frameworks -- the most comprehensive categorization of how agents actually fail in practice.

## What It Is

MAST (Multi-Agent System Taxonomy) is a failure classification framework from NeurIPS 2025 that provides the first systematic categorization of agentic system failures. It was developed by analyzing 1,642 real execution traces across AutoGPT, BabyAGI, CrewAI, LangGraph, MetaGPT, CAMEL, and ChatDev.

Each failure mode has a code (FM-X.Y), prevalence percentage, detection method, and recommended mitigation. This is the vocabulary for discussing agent reliability in technical conversations.

## Category 1: Specification Failures (41.77%)

Failures in understanding, decomposing, or tracking the task.

### FM-1.1: Disobey Task Requirements (11.8%)

**What happens**: Agent ignores explicit constraints or requirements stated in the task.

**Example**: Task says "respond in JSON format only." Agent outputs markdown with an explanation.

**Detection**: Compare output against requirement checklist extracted from the task.

```python
def detect_requirement_violation(task: str, output: str, llm) -> list[str]:
    """Extract requirements from task and check output compliance."""
    # Step 1: Extract requirements deterministically
    requirements = extract_requirements(task)  # regex + LLM extraction
    
    # Step 2: Check each requirement
    violations = []
    for req in requirements:
        if req.type == "format":
            if not validate_format(output, req.expected_format):
                violations.append(f"Format violation: expected {req.expected_format}")
        elif req.type == "constraint":
            if not check_constraint(output, req.constraint):
                violations.append(f"Constraint violation: {req.constraint}")
    
    return violations
```

**Mitigation**: Structured output enforcement (JSON mode), explicit requirement checklist in system prompt, post-generation validation.

---

### FM-1.2: Wrong Task Decomposition (6.2%)

**What happens**: Agent breaks down a complex task into wrong subtasks, missing critical steps or adding unnecessary ones.

**Example**: Task: "Deploy the application." Agent decomposes into: [write code, run tests] -- missing build, containerize, deploy steps.

**Detection**: Compare agent's plan against a reference decomposition or minimum required steps.

**Mitigation**: 
- Provide decomposition examples in the prompt (few-shot)
- Use a planning step with explicit validation before execution
- For known task types, provide canonical decomposition templates

---

### FM-1.3: Step Repetition (15.7%) -- MOST COMMON

**What happens**: Agent repeats the same action multiple times without progress. The #1 failure mode by prevalence.

**Example**: Agent calls `web_search("best restaurants")` three times in a row with the same query, getting the same results each time.

**Detection**:
```python
class StepRepetitionDetector:
    def __init__(self, window_size: int = 3, similarity_threshold: float = 0.9):
        self.history: list[dict] = []
        self.window = window_size
        self.threshold = similarity_threshold
    
    def check(self, action: dict) -> bool:
        """Returns True if repetition detected."""
        self.history.append(action)
        
        if len(self.history) < self.window:
            return False
        
        recent = self.history[-self.window:]
        
        # Check if all recent actions are the same tool + similar args
        tools = [a.get("tool") for a in recent]
        if len(set(tools)) == 1:  # Same tool
            args = [str(a.get("args", {})) for a in recent]
            if len(set(args)) == 1:  # Same arguments
                return True
        
        return False
    
    def get_mitigation(self) -> str:
        """Suggest what to do when repetition detected."""
        repeated_tool = self.history[-1].get("tool", "unknown")
        return (
            f"Step repetition detected: {repeated_tool} called {self.window}x "
            f"with same args. Options: (1) try different parameters, "
            f"(2) use a different tool, (3) summarize what you have and proceed."
        )
```

**Mitigation**: Step counter with hard limit, deduplicated action history, inject "you already tried X, try something different" into context.

---

### FM-1.4: Wrong Step Order (3.8%)

**What happens**: Agent executes steps in the wrong sequence, causing failures in dependent steps.

**Example**: Agent tries to query a database before connecting to it, or deploys code before running tests.

**Detection**: Define dependency graph for steps, check execution order against it.

**Mitigation**: Use a state machine (LangGraph) with explicit transition rules. Make dependencies explicit in the agent's planning prompt.

---

### FM-1.5: Not Recognizing Completion (12.4%) -- SECOND MOST COMMON

**What happens**: Agent has achieved the goal but doesn't recognize it, continuing to perform unnecessary actions.

**Example**: Agent successfully generated the report but then starts "improving" it in an endless loop, or searches for more information after already answering the question.

**Detection**:
```python
class CompletionDetector:
    def __init__(self, max_steps: int = 25):
        self.max_steps = max_steps
        self.outputs: list[str] = []
    
    def check(self, step_num: int, current_output: str, task_goal: str) -> bool:
        """Returns True if task appears complete."""
        self.outputs.append(current_output)
        
        # Hard limit
        if step_num >= self.max_steps:
            return True
        
        # Check if output satisfies goal (deterministic checks)
        if self._goal_met(current_output, task_goal):
            return True
        
        # Check if agent is producing diminishing returns
        if len(self.outputs) >= 3:
            recent_lengths = [len(o) for o in self.outputs[-3:]]
            if max(recent_lengths) - min(recent_lengths) < 50:
                return True  # Output not changing meaningfully
        
        return False
    
    def _goal_met(self, output: str, goal: str) -> bool:
        """Check if explicit completion criteria are met."""
        # e.g., for "generate a JSON report": check if output is valid JSON
        # e.g., for "find the answer": check if output contains a definitive statement
        return False  # Override per task type
```

**Mitigation**: Explicit completion criteria in the system prompt ("you are done when..."), output stability detection, step budget with forced termination.

## Category 2: Coordination Failures (36.94%)

Failures in how the agent coordinates tool use, delegation, and multi-agent interaction.

### FM-2.1: Wrong Tool Selection (8.3%)

**What happens**: Agent selects an inappropriate tool for the current step.

**Example**: Agent uses `web_search` when the answer is in the local knowledge base, or uses `calculator` for a question requiring database lookup.

**Detection**: Log tool selection rationale, compare against ground-truth tool mapping for known task types.

**Mitigation**: 
- Fewer tools with clear, distinct descriptions
- Tool descriptions that include "use this when..." and "do NOT use this when..."
- Tool selection as a separate reasoning step (not embedded in the response)

---

### FM-2.2: Tool Parameter Misuse (7.1%)

**What happens**: Agent selects the right tool but provides wrong parameters.

**Example**: `search_database(query="SELECT * FROM users", limit=-1)` -- SQL injection in the query field, invalid limit.

**Detection**: Schema validation on tool parameters before execution.

```python
from pydantic import BaseModel, Field, validator

class SearchDatabaseParams(BaseModel):
    query: str = Field(..., max_length=500)
    limit: int = Field(default=10, ge=1, le=100)
    
    @validator("query")
    def no_sql_injection(cls, v):
        dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "--", ";"]
        for keyword in dangerous:
            if keyword.upper() in v.upper():
                raise ValueError(f"Dangerous SQL keyword detected: {keyword}")
        return v
```

**Mitigation**: Pydantic schema validation, parameter range constraints, input sanitization.

---

### FM-2.3: Delegation Error (5.9%)

**What happens**: In multi-agent systems, task is delegated to the wrong specialist agent.

**Example**: A coding task gets routed to the research agent instead of the developer agent.

**Detection**: Track delegation decisions, compare delegated agent's expertise against task requirements.

**Mitigation**: Clear role descriptions per agent, delegation classifier, allow agents to refuse tasks outside their expertise.

---

### FM-2.4: Role Confusion (4.8%)

**What happens**: Agent acts outside its assigned role, performing tasks meant for other agents.

**Example**: In a researcher + coder pair, the researcher starts writing code instead of passing findings to the coder.

**Detection**: Monitor action types per agent against allowed actions for their role.

**Mitigation**: Strict role boundaries in system prompts, action allowlists per agent role.

---

### FM-2.5: Communication Breakdown (6.2%)

**What happens**: Information is lost or corrupted when passed between agents or between agent steps.

**Example**: Agent A finds three candidate solutions but only passes one to Agent B, or passes them in a format Agent B can't parse.

**Detection**: Compare information content at handoff points.

**Mitigation**: Structured handoff formats (JSON schemas), explicit "I am passing you the following information" prompts.

---

### FM-2.6: Resource Conflict (4.7%)

**What happens**: Multiple agents or steps try to use the same resource simultaneously, causing conflicts.

**Example**: Two agents both try to write to the same file, or both try to book the same meeting slot.

**Detection**: Resource lock monitoring, conflict detection in shared state.

**Mitigation**: Resource locking, sequential execution for shared resources, optimistic concurrency with retry.

## Category 3: Verification Failures (21.30%)

Failures in validating the quality and correctness of outputs.

### FM-3.1: False Positive Verification (8.4%)

**What happens**: Agent's self-verification approves a wrong answer.

**Example**: Agent generates incorrect Python code, "verifies" it by re-reading it (not executing it), and reports success.

**Detection**: Compare LLM verification against deterministic checks (test execution, schema validation, rule matching).

**Mitigation**: Never rely solely on LLM self-verification. Use deterministic validators: run the code, validate the schema, check against rules.

---

### FM-3.2: False Negative Verification (5.1%)

**What happens**: Agent rejects a correct answer, triggering unnecessary retries.

**Example**: Agent generates a correct SQL query but then decides it's wrong because it "looks complex" and regenerates a simpler (incorrect) version.

**Detection**: Track retry patterns where original answer was correct.

**Mitigation**: Confidence scoring with thresholds, limit self-correction iterations (max 2 retries).

---

### FM-3.3: No Verification Performed (7.8%)

**What happens**: Agent produces output without any verification step.

**Example**: Agent generates a financial report and returns it without checking totals, date ranges, or data completeness.

**Detection**: Check execution trace for verification step presence.

**Mitigation**: Mandatory verification node in the state graph. Never allow the "generate" node to be the terminal node.

## Summary Table

| Code | Failure Mode | Prevalence | Detection | Key Mitigation |
|------|-------------|-----------|-----------|---------------|
| FM-1.1 | Disobey requirements | 11.8% | Requirement checklist | Structured output, validation |
| FM-1.2 | Wrong decomposition | 6.2% | Plan comparison | Few-shot examples, templates |
| FM-1.3 | Step repetition | **15.7%** | Action dedup | Step counter, loop detector |
| FM-1.4 | Wrong step order | 3.8% | Dependency graph | State machine transitions |
| FM-1.5 | Completion blindness | **12.4%** | Output stability | Explicit completion criteria |
| FM-2.1 | Wrong tool selection | 8.3% | Tool-task mapping | Better descriptions, fewer tools |
| FM-2.2 | Tool parameter misuse | 7.1% | Schema validation | Pydantic models, constraints |
| FM-2.3 | Delegation error | 5.9% | Expertise matching | Role clarity, refusal option |
| FM-2.4 | Role confusion | 4.8% | Action allowlists | Strict role boundaries |
| FM-2.5 | Communication breakdown | 6.2% | Handoff comparison | Structured formats |
| FM-2.6 | Resource conflict | 4.7% | Lock monitoring | Resource locking |
| FM-3.1 | False positive verify | 8.4% | Deterministic checks | Never LLM-only verification |
| FM-3.2 | False negative verify | 5.1% | Retry tracking | Confidence thresholds |
| FM-3.3 | No verification | 7.8% | Trace inspection | Mandatory verify node |

## Source(s) and Further Reading

- MAST Taxonomy Paper (NeurIPS 2025): "A Taxonomy of Agent Failure Modes"
- Dataset: 1,642 execution traces across 7 frameworks
- "On the Reliability of LLM-Based Agents" - Stanford HAI (2025)
- AutoGPT Failure Analysis: https://github.com/Significant-Gravitas/AutoGPT/issues
