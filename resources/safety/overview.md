# Safety Overview: The 4-Layer Guardrail Taxonomy

> System prompts are suggestions to a sufficiently capable model -- real safety comes from scoped credentials, sandboxes, and forced approval gates.

## What It Is

Agent safety is the discipline of constraining autonomous AI systems so they cannot cause harm even when they are compromised, confused, or deliberately attacked. Unlike traditional software security (firewall + auth), agent safety must handle a fundamentally new threat: the agent itself can be manipulated into becoming the attacker.

The **4-Layer Guardrail Taxonomy** provides a defense-in-depth architecture:

| Layer | What It Protects | Mechanism | Failure Mode If Missing |
|-------|-----------------|-----------|------------------------|
| **1. Content Layer** | Users from harmful output | Input/output filters, toxicity classifiers, PII redaction | Agent produces harmful, biased, or privacy-violating content |
| **2. Evaluation Layer** | Decision quality | LLM-as-judge, trajectory eval, chain-of-thought inspection | Agent takes wrong actions that look plausible |
| **3. Sandbox Layer** | Infrastructure from agent | Isolated execution, network controls, filesystem restrictions | Compromised agent gains access to production systems |
| **4. Action Layer** | Real-world consequences | Human-in-the-loop, approval gates, scoped credentials, undo mechanisms | Agent executes irreversible harmful actions |

## How It Works

### Why System Prompts Cannot Stop Determined Models

System prompts are processed by the same model that processes user input. They exist in the same context window, subject to the same attention mechanism. A sufficiently clever prompt injection can override, ignore, or reinterpret system instructions because:

1. **No architectural separation**: System prompt and user input are both token sequences processed identically by the transformer.
2. **Instruction hierarchy is soft**: Models learn to follow system prompts during training, but this is a statistical tendency, not a hard constraint.
3. **Context window competition**: As conversations grow, system prompt influence dilutes.
4. **Novel attacks bypass training**: Adversarial inputs exploit distribution gaps the model has never seen.

**The real stops are:**

```
System Prompt ("Don't do bad things")     -- ~85% effective, trivially bypassed
     +
Input/Output Scanning                     -- ~93% effective, catches known patterns
     +
Scoped Credentials (read-only tokens)     -- 100% effective for its scope
     +
Sandbox (no network, no filesystem)       -- 100% effective for its scope
     +
Forced Human Approval (irreversible ops)  -- 100% effective when enforced
```

### Decision Tree: Which Guardrail Layer for Which Risk

```
Is the risk about CONTENT the agent produces?
  YES --> Layer 1: Content filters (NeMo Guardrails, Guardrails AI)
  NO  --> 

Is the risk about REASONING quality?
  YES --> Layer 2: Evaluation (LLM-as-judge, trajectory eval, AlignmentCheck)
  NO  -->

Is the risk about CODE EXECUTION or SYSTEM ACCESS?
  YES --> Layer 3: Sandbox (E2B, gVisor, credential scoping)
  NO  -->

Is the risk about IRREVERSIBLE REAL-WORLD ACTIONS?
  YES --> Layer 4: Action controls (HITL, approval gates, undo mechanisms)
```

### OWASP Top 10 for LLM Applications (2025)

The OWASP Foundation published the definitive threat catalog for LLM-powered systems:

| Rank | Vulnerability | Agent Impact | Primary Defense Layer |
|------|--------------|-------------|----------------------|
| LLM01 | **Prompt Injection** | Agent executes attacker instructions | Layers 1 + 3 + 4 |
| LLM02 | **Sensitive Information Disclosure** | Agent leaks PII, credentials, internal data | Layer 1 (output filtering) |
| LLM03 | **Supply Chain** | Poisoned models, plugins, training data | Layer 3 (sandbox) |
| LLM04 | **Data and Model Poisoning** | Corrupted RAG corpus, fine-tuning attacks | Layer 2 (eval) |
| LLM05 | **Improper Output Handling** | Unvalidated agent output causes downstream injection | Layer 1 (output validation) |
| LLM06 | **Excessive Agency** | Agent has more permissions than needed | Layer 4 (least privilege) |
| LLM07 | **System Prompt Leakage** | Attacker extracts system instructions | Layer 1 (output filtering) |
| LLM08 | **Vector and Embedding Weaknesses** | RAG retrieval manipulation | Layer 2 (eval) |
| LLM09 | **Misinformation** | Agent generates plausible but false information | Layer 2 (eval + grounding) |
| LLM10 | **Unbounded Consumption** | Agent runs infinite loops, burns tokens | Layer 3 (resource limits) |

### EU AI Act Timeline (Critical for Production Systems)

```
Feb 2, 2025   -- Prohibited practices banned (social scoring, etc.)
Aug 2, 2025   -- GPAI model obligations apply
Aug 2, 2026   -- HIGH-RISK SYSTEM OBLIGATIONS TAKE EFFECT <-- KEY DEADLINE
                 Autonomous agents making consequential decisions 
                 likely classified as high-risk
Aug 2, 2027   -- Full enforcement for all systems
```

**What "high-risk" means for agent builders:**
- Mandatory risk management system
- Data governance requirements
- Technical documentation
- Record-keeping (audit trails)
- Human oversight mechanisms
- Accuracy, robustness, cybersecurity standards
- Conformity assessment before deployment

## Production Implementation

```python
"""
4-Layer Guardrail Stack -- Production Architecture
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any

class RiskLevel(Enum):
    LOW = "low"          # Read-only queries, no side effects
    MEDIUM = "medium"    # Writes to non-critical systems
    HIGH = "high"        # Financial transactions, PII access
    CRITICAL = "critical" # Irreversible actions, compliance-sensitive

@dataclass
class GuardrailConfig:
    """Configure guardrails based on risk level."""
    risk_level: RiskLevel
    
    @property
    def content_filter(self) -> bool:
        """Layer 1: Always on."""
        return True
    
    @property
    def eval_scoring(self) -> bool:
        """Layer 2: Medium and above."""
        return self.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)
    
    @property
    def sandbox_execution(self) -> bool:
        """Layer 3: High and above."""
        return self.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
    
    @property 
    def human_approval(self) -> bool:
        """Layer 4: Critical only."""
        return self.risk_level == RiskLevel.CRITICAL

    @property
    def credential_scope(self) -> str:
        """Credential scope narrows with risk level."""
        return {
            RiskLevel.LOW: "read-only",
            RiskLevel.MEDIUM: "read-write-scoped",
            RiskLevel.HIGH: "read-write-scoped-audited",
            RiskLevel.CRITICAL: "read-only-until-approved",
        }[self.risk_level]


class GuardrailStack:
    """Composable 4-layer guardrail stack."""
    
    def __init__(self, config: GuardrailConfig):
        self.config = config
        self.layers = []
        self._build_stack()
    
    def _build_stack(self):
        # Layer 1: Always present
        self.layers.append(ContentFilter())
        
        # Layer 2: Evaluation scoring
        if self.config.eval_scoring:
            self.layers.append(EvalScorer())
        
        # Layer 3: Sandbox
        if self.config.sandbox_execution:
            self.layers.append(SandboxEnforcer())
        
        # Layer 4: Human approval gate
        if self.config.human_approval:
            self.layers.append(HumanApprovalGate())
    
    async def check(self, action: dict[str, Any]) -> dict[str, Any]:
        """Run action through all guardrail layers."""
        result = {"allowed": True, "action": action, "checks": []}
        
        for layer in self.layers:
            check = await layer.evaluate(action)
            result["checks"].append(check)
            
            if not check["passed"]:
                result["allowed"] = False
                result["blocked_by"] = layer.__class__.__name__
                result["reason"] = check["reason"]
                break
        
        return result


# Usage
config = GuardrailConfig(risk_level=RiskLevel.CRITICAL)
stack = GuardrailStack(config)
# Every agent action flows through: content -> eval -> sandbox -> human approval
```

## Decision Tree / When to Use

**Use all 4 layers when:**
- Agent handles financial transactions
- Agent accesses PII or health data
- Agent operates in regulated industries
- Agent can modify production infrastructure

**Use Layers 1-3 (skip human approval) when:**
- Agent performs automatable but reversible actions
- Low-value transactions with automatic rollback
- Internal tooling with audit trails

**Use Layers 1-2 (skip sandbox + approval) when:**
- Agent is purely conversational (no tool use)
- Agent only reads data, never writes

**Use Layer 1 only when:**
- Simple chatbot with no tool access
- Internal Q&A over trusted documents

## When NOT to Use

- **Never skip Layer 1 (content)** -- even internal tools need output filtering for PII leakage
- **Never give agents permanent credentials** -- even trusted agents get compromised
- **Never rely solely on system prompts** for safety-critical operations

## Tradeoffs

| Approach | Safety | Latency | User Experience | Complexity |
|----------|--------|---------|-----------------|------------|
| All 4 layers | Highest | +2-5s per action | Approval fatigue | High |
| Layers 1-3 | High | +0.5-2s per action | Smooth | Medium |
| Layers 1-2 | Medium | +0.1-0.5s | Smooth | Low |
| Layer 1 only | Low | +50ms | Smooth | Low |
| No guardrails | None | 0 | Best | None |

## Real-World Examples

1. **GitHub Copilot Workspace** -- Uses Layer 3 (sandbox) for code execution + Layer 4 (human approval before commit). No code reaches production without explicit human merge.

2. **Stripe Agent Toolkit** -- Layer 4 (scoped API keys with read-only default). Agents can read payment data but cannot create charges without elevated, session-scoped tokens.

3. **Anthropic Claude Computer Use** -- Layer 2 (eval/monitoring) + Layer 4 (human-in-the-loop). Every computer action is logged and reviewed.

4. **LangGraph Platform** -- Layer 2 (evaluation) + Layer 4 (interrupt_before on sensitive nodes). Agents pause at designated checkpoints for human review.

## Failure Modes

| Failure | Cause | Impact | Prevention |
|---------|-------|--------|------------|
| **Guardrail bypass** | Single-layer defense | Complete compromise | Defense in depth (all 4 layers) |
| **Approval fatigue** | Too many human checkpoints | Humans rubber-stamp approvals | Risk-based approval routing |
| **False positive blocking** | Overly aggressive content filters | Agent becomes unusable | Tuned thresholds + override audit trail |
| **Latency death spiral** | Sequential guardrail checks | User abandonment | Parallel checks where possible |
| **Credential over-scoping** | Agent gets admin tokens for convenience | Blast radius of compromise = everything | Least privilege always |

## Source(s) and Further Reading

- OWASP Top 10 for LLM Applications (2025): https://owasp.org/www-project-top-10-for-large-language-model-applications/
- EU AI Act Official Text: https://artificialintelligenceact.eu/
- NIST AI Risk Management Framework: https://www.nist.gov/artificial-intelligence/ai-risk-management-framework
- Anthropic, "Core Views on AI Safety" (2024)
- Meta LlamaFirewall: arXiv:2505.03574
- Simon Willison, "Prompt Injection: What's the worst that can happen?" (2023)
- Greshake et al., "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection" (2023)
