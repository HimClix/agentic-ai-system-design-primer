# NIST AI Risk Management Framework

> NIST AI RMF is not a checklist -- it is a structured way to think about AI risk across the entire lifecycle, from design through deployment to decommissioning.

## What It Is

The NIST AI Risk Management Framework (AI RMF 1.0, published January 2023) is a voluntary framework from the National Institute of Standards and Technology that provides guidance for managing risks associated with AI systems. For agentic AI systems, it serves as the most comprehensive risk management structure available, influencing EU AI Act compliance and enterprise AI governance programs.

Unlike prescriptive regulations, NIST AI RMF is **outcome-based** -- it tells you what risks to manage, not exactly how to manage them.

## How It Works

### The 4 Core Functions

| Function | Purpose | Agent Application |
|----------|---------|-------------------|
| **GOVERN** | Establish AI risk management culture and accountability | Who owns agent safety? What policies exist? |
| **MAP** | Identify and document AI risks in context | What can go wrong with this agent? In what context? |
| **MEASURE** | Assess and track identified risks | How bad is each risk? How often? How do we detect? |
| **MANAGE** | Treat, mitigate, and monitor prioritized risks | What controls are in place? Are they working? |

### Key Controls for Agent Systems

**GOVERN:**
- Designate an AI risk owner for each agent system
- Establish acceptable risk thresholds for agent autonomy
- Define escalation procedures when agents fail
- Maintain an inventory of deployed agents and their capabilities

**MAP:**
- Document all tools the agent can access and their blast radius
- Identify populations affected by agent decisions
- Map data flows through the agent (what data enters, what leaves)
- Identify third-party dependencies (LLM providers, tool APIs)

**MEASURE:**
- Track agent accuracy, hallucination rate, and error frequency
- Measure fairness and bias in agent decisions across user groups
- Monitor for prompt injection attempts and successful attacks
- Evaluate agent behavior against intended purpose

**MANAGE:**
- Implement guardrails (content, eval, sandbox, action layers)
- Establish human-in-the-loop for high-risk decisions
- Maintain incident response procedures for agent failures
- Plan for agent decommissioning (data retention, model versioning)

## Production Implementation

```python
"""
NIST AI RMF compliance tracker for agent systems.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from datetime import datetime


class RiskSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ControlStatus(Enum):
    NOT_IMPLEMENTED = "not_implemented"
    PARTIAL = "partial"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"


@dataclass
class RiskEntry:
    risk_id: str
    nist_function: str        # GOVERN, MAP, MEASURE, MANAGE
    nist_category: str        # e.g., "GOVERN 1.1", "MAP 2.3"
    description: str
    severity: RiskSeverity
    likelihood: RiskSeverity
    control_description: str
    control_status: ControlStatus
    owner: str
    last_reviewed: datetime
    next_review: datetime
    evidence: str             # Link to evidence of control effectiveness
    notes: str = ""


# Example risk register for an agentic AI system
AGENT_RISK_REGISTER = [
    RiskEntry(
        risk_id="RISK-001",
        nist_function="MAP",
        nist_category="MAP 2.1 - Context of Use",
        description="Agent makes decisions outside its intended scope due to prompt injection",
        severity=RiskSeverity.CRITICAL,
        likelihood=RiskSeverity.MEDIUM,
        control_description="LlamaFirewall AlignmentCheck + input scanning + scoped credentials",
        control_status=ControlStatus.IMPLEMENTED,
        owner="security-team",
        last_reviewed=datetime(2025, 4, 1),
        next_review=datetime(2025, 7, 1),
        evidence="link-to-penetration-test-results",
    ),
    RiskEntry(
        risk_id="RISK-002",
        nist_function="MEASURE",
        nist_category="MEASURE 2.6 - Bias Testing",
        description="Agent provides different quality of service based on user demographics",
        severity=RiskSeverity.HIGH,
        likelihood=RiskSeverity.LOW,
        control_description="Quarterly bias audits on agent responses across demographic segments",
        control_status=ControlStatus.PARTIAL,
        owner="ml-fairness-team",
        last_reviewed=datetime(2025, 3, 15),
        next_review=datetime(2025, 6, 15),
        evidence="link-to-bias-audit-q1-2025",
    ),
    RiskEntry(
        risk_id="RISK-003",
        nist_function="GOVERN",
        nist_category="GOVERN 1.7 - Audit Trail",
        description="Insufficient logging of agent reasoning for compliance audits",
        severity=RiskSeverity.HIGH,
        likelihood=RiskSeverity.HIGH,
        control_description="Full audit trail with reasoning traces, stored for 7 years",
        control_status=ControlStatus.VERIFIED,
        owner="compliance-team",
        last_reviewed=datetime(2025, 5, 1),
        next_review=datetime(2025, 8, 1),
        evidence="link-to-soc2-audit-report",
    ),
]
```

## Decision Tree / When to Use

- **US-based companies deploying AI agents**: Use NIST AI RMF as primary governance framework
- **Companies targeting EU markets**: Use NIST AI RMF as foundation, layer EU AI Act requirements on top
- **Enterprise AI programs**: Use NIST AI RMF for risk assessment structure
- **Startups**: Use NIST AI RMF categories as a thinking framework, even without formal implementation

## When NOT to Use

- **As a replacement for EU AI Act compliance** -- NIST is voluntary, EU AI Act is law
- **As a checklist** -- It is a framework for thinking, not a pass/fail test
- **For simple chatbots** -- The overhead may not be justified for low-risk systems

## Tradeoffs

| Aspect | NIST AI RMF | ISO 42001 | EU AI Act |
|--------|------------|-----------|-----------|
| **Binding** | Voluntary | Voluntary (certification) | Mandatory (EU) |
| **Scope** | All AI | AI management system | High-risk AI |
| **Prescriptive** | Outcome-based | Process-based | Requirement-based |
| **Cost to implement** | Low-Medium | Medium-High | High |
| **Best for** | Risk thinking framework | Formal management system | Legal compliance |

## Source(s) and Further Reading

- NIST AI RMF 1.0: https://www.nist.gov/artificial-intelligence/ai-risk-management-framework
- NIST AI RMF Playbook: https://airc.nist.gov/AI_RMF_Knowledge_Base/Playbook
- NIST AI 600-1: Generative AI Profile (2024)
- NIST Trustworthy AI characteristics: Valid, Reliable, Safe, Secure, Resilient, Accountable, Transparent, Explainable, Privacy-Enhanced, Fair
