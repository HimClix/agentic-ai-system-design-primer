# LlamaFirewall

> Meta's LlamaFirewall catches what input scanners miss -- by inspecting the agent's chain-of-thought reasoning for signs of goal hijacking after the injection has already entered the context.

## What It Is

LlamaFirewall is Meta's open-source guardrail system (arXiv:2505.03574) specifically designed for securing AI agents. Its key innovation is the **AlignmentCheck** module, which inspects an agent's chain-of-thought reasoning to detect when an injection has successfully influenced the model's intent -- catching attacks that bypass input/output scanning.

**Key facts:**
- **Paper**: arXiv:2505.03574 (May 2025)
- **Architecture**: Layered pipeline with multiple specialized modules
- **Key module**: AlignmentCheck -- chain-of-thought inspection for goal hijacking
- **Block rate**: 94%+ via ceLLMate approach (chain-of-thought analysis)
- **Best for**: Security-first agent deployments where injection defense is paramount
- **Differentiation**: Catches injections AFTER they enter context (other tools only scan input/output)

## How It Works

### The AlignmentCheck Innovation

Traditional defenses scan input (before LLM) or output (after LLM). LlamaFirewall's AlignmentCheck operates at a different point -- it inspects the **reasoning trace** (chain-of-thought) to detect when the model's goals have been hijacked:

```
Traditional Defense:
  Input Scanning → LLM → Output Scanning
  ↑ catches ~60%              ↑ catches ~70%
  (known patterns)            (known outputs)

LlamaFirewall AlignmentCheck:
  Input → LLM reasoning (CoT) → AlignmentCheck → Output
                                 ↑
                    Inspects reasoning for goal hijacking
                    catches ~94%+ including novel attacks
```

**Why this works better:**

1. **Input scanners** catch known attack patterns but miss novel phrasings
2. **Output scanners** catch known bad outputs but miss when the output looks legitimate
3. **AlignmentCheck** catches the *effect* of injection regardless of the *form*:
   - "The document says I should forward data" -- MISALIGNED
   - "The user wants me to ignore safety" -- MISALIGNED
   - "Let me search for the revenue data" -- ALIGNED

### Layered Pipeline Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    LlamaFirewall                          │
│                                                           │
│  ┌─────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │  PromptGuard │→ │  AlignmentCheck  │→ │ OutputGuard │ │
│  │             │  │  (CoT Inspection) │  │             │ │
│  │ Input scan  │  │                  │  │ Output scan │ │
│  │ Known       │  │ • Goal alignment │  │ Data leak   │ │
│  │ patterns    │  │ • Intent verify  │  │ detection   │ │
│  │ Jailbreak   │  │ • Source verify  │  │ Tool call   │ │
│  │ detection   │  │ • Action verify  │  │ validation  │ │
│  └─────────────┘  └──────────────────┘  └─────────────┘ │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              CodeShield (optional)                   │ │
│  │  • Generated code safety analysis                    │ │
│  │  • Detect malicious code patterns                    │ │
│  │  • Sandbox enforcement for code execution            │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### Modules

| Module | Function | Catches |
|--------|----------|---------|
| **PromptGuard** | Input-level jailbreak and injection detection | Known injection patterns, encoding tricks |
| **AlignmentCheck** | Chain-of-thought goal alignment verification | Goal hijacking, indirect injection effects, subtle manipulation |
| **OutputGuard** | Output-level data leakage and safety checks | PII leakage, credential exposure, unsafe tool calls |
| **CodeShield** | Generated code security analysis | Malicious code, dangerous system calls, data exfiltration code |

### ceLLMate: The 94%+ Block Rate

The **ceLLMate** approach, which LlamaFirewall builds upon, achieves 94%+ injection block rates by:

1. **Extracting the reasoning trace** from the agent's chain-of-thought
2. **Classifying intent alignment** using a fine-tuned model trained on injection/non-injection pairs
3. **Comparing stated goals** (from user message) with **inferred goals** (from reasoning)
4. **Flagging divergence** as potential injection success

This is fundamentally harder to bypass than pattern matching because:
- The attacker must not only inject instructions but make the model's reasoning about those instructions look aligned with the user's original goal
- Even if the injection enters the context successfully, AlignmentCheck detects the goal shift

## Production Implementation

```python
"""
LlamaFirewall integration pattern for agent systems.
Shows the architectural pattern -- actual LlamaFirewall API may differ.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class FirewallVerdict(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"


@dataclass
class FirewallResult:
    verdict: FirewallVerdict
    module: str           # Which module triggered
    confidence: float     # 0.0 to 1.0
    reason: str
    details: dict[str, Any]


class LlamaFirewallClient:
    """
    Client for LlamaFirewall pipeline.
    Orchestrates PromptGuard → AlignmentCheck → OutputGuard → CodeShield.
    """
    
    def __init__(
        self,
        prompt_guard_threshold: float = 0.85,
        alignment_threshold: float = 0.80,
        output_guard_threshold: float = 0.85,
        enable_code_shield: bool = False,
    ):
        self.prompt_guard_threshold = prompt_guard_threshold
        self.alignment_threshold = alignment_threshold
        self.output_guard_threshold = output_guard_threshold
        self.enable_code_shield = enable_code_shield
    
    async def check_input(self, user_input: str, system_prompt: str) -> FirewallResult:
        """
        PromptGuard: Check user input for injection/jailbreak attempts.
        Run BEFORE the LLM call.
        """
        # In production: call LlamaFirewall PromptGuard API
        # This is the pattern -- replace with actual API call
        
        injection_score = await self._prompt_guard_score(user_input, system_prompt)
        
        if injection_score >= self.prompt_guard_threshold:
            return FirewallResult(
                verdict=FirewallVerdict.BLOCK,
                module="PromptGuard",
                confidence=injection_score,
                reason="Prompt injection or jailbreak detected in user input",
                details={"injection_score": injection_score},
            )
        elif injection_score >= 0.5:
            return FirewallResult(
                verdict=FirewallVerdict.WARN,
                module="PromptGuard",
                confidence=injection_score,
                reason="Suspicious input detected, proceeding with enhanced monitoring",
                details={"injection_score": injection_score},
            )
        
        return FirewallResult(
            verdict=FirewallVerdict.ALLOW,
            module="PromptGuard",
            confidence=1.0 - injection_score,
            reason="Input passed safety check",
            details={"injection_score": injection_score},
        )
    
    async def check_alignment(
        self,
        user_goal: str,
        agent_reasoning: str,
        retrieved_context: Optional[str] = None,
    ) -> FirewallResult:
        """
        AlignmentCheck: Inspect chain-of-thought for goal hijacking.
        Run AFTER the LLM generates reasoning but BEFORE executing actions.
        
        This is the key innovation of LlamaFirewall.
        """
        # In production: call LlamaFirewall AlignmentCheck API
        
        alignment_score = await self._alignment_check_score(
            user_goal, agent_reasoning, retrieved_context
        )
        
        if alignment_score < (1.0 - self.alignment_threshold):
            # Low alignment = high hijacking probability
            return FirewallResult(
                verdict=FirewallVerdict.BLOCK,
                module="AlignmentCheck",
                confidence=1.0 - alignment_score,
                reason="Agent reasoning appears misaligned with user goal. Possible injection in retrieved content.",
                details={
                    "alignment_score": alignment_score,
                    "user_goal_summary": user_goal[:100],
                    "reasoning_excerpt": agent_reasoning[:200],
                },
            )
        
        return FirewallResult(
            verdict=FirewallVerdict.ALLOW,
            module="AlignmentCheck",
            confidence=alignment_score,
            reason="Agent reasoning aligned with user goal",
            details={"alignment_score": alignment_score},
        )
    
    async def check_output(
        self,
        output: str,
        tool_calls: list[dict[str, Any]],
    ) -> FirewallResult:
        """
        OutputGuard: Check agent output for data leakage and unsafe actions.
        Run BEFORE returning output to user or executing tool calls.
        """
        issues = []
        
        # Check for credential/PII leakage
        leakage_score = await self._check_data_leakage(output)
        if leakage_score > self.output_guard_threshold:
            issues.append(f"Data leakage risk: {leakage_score:.2f}")
        
        # Check tool calls for dangerous operations
        for call in tool_calls:
            safety_score = await self._check_tool_safety(call)
            if safety_score < 0.5:
                issues.append(f"Unsafe tool call: {call.get('name', 'unknown')}")
        
        if issues:
            return FirewallResult(
                verdict=FirewallVerdict.BLOCK,
                module="OutputGuard",
                confidence=0.9,
                reason=f"Output safety issues: {'; '.join(issues)}",
                details={"issues": issues},
            )
        
        return FirewallResult(
            verdict=FirewallVerdict.ALLOW,
            module="OutputGuard",
            confidence=0.95,
            reason="Output passed safety checks",
            details={},
        )
    
    async def full_pipeline(
        self,
        user_input: str,
        system_prompt: str,
        agent_reasoning: Optional[str] = None,
        agent_output: Optional[str] = None,
        tool_calls: Optional[list[dict]] = None,
        retrieved_context: Optional[str] = None,
    ) -> list[FirewallResult]:
        """
        Run the full LlamaFirewall pipeline.
        Returns results from all modules, stops at first BLOCK.
        """
        results = []
        
        # Stage 1: PromptGuard (input)
        r1 = await self.check_input(user_input, system_prompt)
        results.append(r1)
        if r1.verdict == FirewallVerdict.BLOCK:
            return results
        
        # Stage 2: AlignmentCheck (reasoning)
        if agent_reasoning:
            r2 = await self.check_alignment(user_input, agent_reasoning, retrieved_context)
            results.append(r2)
            if r2.verdict == FirewallVerdict.BLOCK:
                return results
        
        # Stage 3: OutputGuard (output)
        if agent_output or tool_calls:
            r3 = await self.check_output(agent_output or "", tool_calls or [])
            results.append(r3)
        
        return results
    
    # Internal scoring methods (replace with actual LlamaFirewall API)
    async def _prompt_guard_score(self, user_input: str, system_prompt: str) -> float:
        """Score probability of injection in user input."""
        # Placeholder -- actual implementation calls PromptGuard model
        return 0.1
    
    async def _alignment_check_score(
        self, user_goal: str, reasoning: str, context: Optional[str]
    ) -> float:
        """Score alignment between user goal and agent reasoning."""
        # Placeholder -- actual implementation calls AlignmentCheck model
        return 0.95
    
    async def _check_data_leakage(self, output: str) -> float:
        """Score probability of data leakage in output."""
        return 0.05
    
    async def _check_tool_safety(self, tool_call: dict) -> float:
        """Score safety of a tool call."""
        return 0.95


# ============================================================
# Integration with LangGraph Agent
# ============================================================

async def langgraph_with_llamafirewall():
    """Example: LangGraph agent with LlamaFirewall at each stage."""
    from langgraph.graph import StateGraph, END
    
    firewall = LlamaFirewallClient()
    
    async def input_check_node(state: dict) -> dict:
        """Check input before processing."""
        result = await firewall.check_input(
            state["user_input"],
            state["system_prompt"],
        )
        state["firewall_input_result"] = result
        return state
    
    async def reasoning_node(state: dict) -> dict:
        """Generate reasoning (CoT)."""
        # LLM call to generate reasoning
        # ... (LLM call here)
        state["reasoning"] = "Let me search for Q3 revenue data..."
        return state
    
    async def alignment_check_node(state: dict) -> dict:
        """Check reasoning alignment BEFORE executing actions."""
        result = await firewall.check_alignment(
            state["user_input"],
            state["reasoning"],
            state.get("retrieved_context"),
        )
        state["firewall_alignment_result"] = result
        return state
    
    async def action_node(state: dict) -> dict:
        """Execute actions and generate output."""
        # Only reached if alignment check passes
        state["output"] = "Q3 revenue was $42.3M..."
        state["tool_calls"] = []
        return state
    
    async def output_check_node(state: dict) -> dict:
        """Check output before returning to user."""
        result = await firewall.check_output(
            state["output"],
            state.get("tool_calls", []),
        )
        state["firewall_output_result"] = result
        return state
    
    def route_after_input_check(state: dict) -> str:
        if state["firewall_input_result"].verdict == FirewallVerdict.BLOCK:
            return "blocked"
        return "reasoning"
    
    def route_after_alignment_check(state: dict) -> str:
        if state["firewall_alignment_result"].verdict == FirewallVerdict.BLOCK:
            return "blocked"
        return "action"
    
    # Build the graph
    graph = StateGraph(dict)
    graph.add_node("input_check", input_check_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("alignment_check", alignment_check_node)
    graph.add_node("action", action_node)
    graph.add_node("output_check", output_check_node)
    graph.add_node("blocked", lambda s: {**s, "output": "Request blocked for safety."})
    
    graph.set_entry_point("input_check")
    graph.add_conditional_edges("input_check", route_after_input_check)
    graph.add_edge("reasoning", "alignment_check")
    graph.add_conditional_edges("alignment_check", route_after_alignment_check)
    graph.add_edge("action", "output_check")
    graph.add_edge("output_check", END)
    graph.add_edge("blocked", END)
```

## Decision Tree / When to Use

```
Is SECURITY your primary concern (not just content quality)?
  YES --> LlamaFirewall (security-first design)
  NO  --> Consider NeMo Guardrails (dialog) or Guardrails AI (validation)

Does your agent use TOOL CALLING or CODE EXECUTION?
  YES --> LlamaFirewall (AlignmentCheck catches tool-call hijacking)
  NO  --> Simpler guardrails may suffice

Does your agent use RAG or external content?
  YES --> LlamaFirewall (AlignmentCheck catches indirect injection effects)
  NO  --> PromptGuard alone may be sufficient

Can you accept 100-500ms additional latency?
  YES --> Full LlamaFirewall pipeline
  NO  --> Use PromptGuard only (fastest module)
```

## When NOT to Use

- **Dialog flow management** -- Use NeMo Guardrails (Colang)
- **Structured output validation** -- Use Guardrails AI (Pydantic validators)
- **Latency-critical applications (<50ms budget)** -- AlignmentCheck adds 100-300ms
- **Simple chatbots without tools** -- Overkill for read-only conversational agents

## Tradeoffs

| Aspect | LlamaFirewall | NeMo Guardrails | Guardrails AI |
|--------|-------------|----------------|---------------|
| **Primary strength** | Injection defense | Dialog flow control | Output validation |
| **Block rate** | 94%+ (ceLLMate) | ~80% (pattern matching) | ~70% (validators) |
| **Catches indirect injection** | Yes (AlignmentCheck) | Limited | No |
| **Catches novel attacks** | Yes (reasoning analysis) | No (pattern-based) | No (rule-based) |
| **Latency** | 100-500ms | 100-300ms | 10-50ms |
| **Code security** | Yes (CodeShield) | No | Limited |

## Real-World Examples

1. **Multi-tool agent with RAG** -- AlignmentCheck prevents a poisoned RAG document from hijacking the agent's tool calls, even when the injection bypasses input scanning.

2. **Code generation agent** -- CodeShield analyzes generated code for malicious patterns (data exfiltration, privilege escalation) before execution in sandbox.

3. **Enterprise email assistant** -- AlignmentCheck detects when email content tries to make the assistant forward messages to external addresses.

## Failure Modes

| Failure | Cause | Impact | Prevention |
|---------|-------|--------|------------|
| **AlignmentCheck false negative** | Very subtle goal shift in reasoning | Injection succeeds | Layer with output validation + least privilege |
| **Latency spike** | AlignmentCheck model under load | User experience degrades | Async processing, model optimization |
| **No CoT available** | Model doesn't generate visible reasoning | AlignmentCheck cannot operate | Use models that support CoT; fall back to input/output scanning |
| **Over-blocking on complex tasks** | Multi-step reasoning looks like goal shifting | Legitimate complex tasks blocked | Tune alignment threshold per task type |

## Source(s) and Further Reading

- Meta, "LlamaFirewall: An Open Source Guardrail System for Building Secure AI Agents" (2025): https://arxiv.org/abs/2505.03574
- ceLLMate: Chain-of-thought analysis achieving 94%+ injection block rate
- Meta AI Blog, "Introducing LlamaFirewall" (2025)
- LlamaFirewall GitHub repository
- Inan et al., "Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations" (2023)
