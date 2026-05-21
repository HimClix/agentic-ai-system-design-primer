# EU AI Act

> August 2, 2026 is the deadline that matters -- that is when high-risk AI system obligations take effect, and autonomous agents making consequential decisions will likely be classified as high-risk.

## What It Is

The EU AI Act (Regulation 2024/1689) is the world's first comprehensive AI regulation, adopted in 2024. It classifies AI systems by risk level and imposes escalating obligations. For agent builders, the critical date is **August 2, 2026**, when high-risk system obligations become enforceable with penalties up to 35 million EUR or 7% of global turnover.

## How It Works

### Risk Classification for Agent Systems

| Risk Level | Examples | Agent Applicability | Obligations |
|-----------|---------|---------------------|-------------|
| **Unacceptable** (banned) | Social scoring, subliminal manipulation | Agent that manipulates user decisions without awareness | Prohibited |
| **High-Risk** | Employment decisions, credit scoring, critical infrastructure | Agent that makes or influences consequential decisions | Full compliance required |
| **Limited Risk** | Chatbots, content generation | Conversational agents, content agents | Transparency obligations |
| **Minimal Risk** | Spam filters, AI in games | Internal tooling, simple assistants | No specific obligations |

### Why Most Production Agents Will Be "High-Risk"

An agent is likely high-risk if it:
- Makes decisions affecting employment, credit, insurance, or education
- Operates critical infrastructure (energy, transport, water, finance)
- Interacts with law enforcement or border control systems
- Provides medical advice or triage
- Makes decisions about access to essential services

### Key Obligations for High-Risk AI Agents (Aug 2, 2026)

| Obligation | What It Means | Implementation |
|-----------|---------------|----------------|
| **Risk Management System** | Ongoing risk identification and mitigation | NIST AI RMF-style risk register |
| **Data Governance** | Quality requirements for training/fine-tuning data | Document data sources, quality metrics, bias checks |
| **Technical Documentation** | Complete system description | Architecture docs, model cards, capability boundaries |
| **Record-Keeping** | Automatic logging of operations | Full audit trails (see audit-trails.md) |
| **Transparency** | Users know they interact with AI | Clear disclosure, explainable decisions |
| **Human Oversight** | Humans can intervene and override | HITL gates, kill switches, override mechanisms |
| **Accuracy & Robustness** | System performs reliably | Evaluation pipelines, monitoring, red-teaming |
| **Cybersecurity** | Protection against attacks | Prompt injection defense, credential scoping |

### Timeline

```
Feb 2, 2025   ✓  Prohibited practices banned
Aug 2, 2025      GPAI model obligations apply (foundation model providers)
Aug 2, 2026   ⚠  HIGH-RISK SYSTEM OBLIGATIONS -- the deadline for agent builders
Aug 2, 2027      Full enforcement for ALL systems including existing ones
```

### GPAI (General-Purpose AI) Model Obligations

If you use a GPAI model (GPT-4, Claude, Gemini, Llama), the **model provider** has obligations (technical docs, copyright compliance, training data summaries). But as the **deployer** (agent builder), YOU have additional obligations for how you use the model in your high-risk system.

## Production Implementation

```python
"""
EU AI Act compliance checklist for agent systems.
Track readiness for Aug 2, 2026 deadline.
"""
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, date


class ComplianceStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    AUDITED = "audited"


@dataclass
class ComplianceItem:
    article: str
    requirement: str
    status: ComplianceStatus
    owner: str
    evidence: str
    deadline: date
    notes: str = ""


# Compliance checklist for an agentic AI system
EU_AI_ACT_CHECKLIST = [
    ComplianceItem(
        article="Art. 9 - Risk Management",
        requirement="Establish and maintain a risk management system throughout the AI system lifecycle",
        status=ComplianceStatus.IN_PROGRESS,
        owner="ai-governance-team",
        evidence="Risk register maintained in Confluence",
        deadline=date(2026, 8, 2),
    ),
    ComplianceItem(
        article="Art. 10 - Data Governance",
        requirement="Training, validation, and testing datasets must meet quality criteria; bias examination required",
        status=ComplianceStatus.NOT_STARTED,
        owner="ml-platform-team",
        evidence="",
        deadline=date(2026, 8, 2),
        notes="Need to document all fine-tuning datasets and RAG corpus sources",
    ),
    ComplianceItem(
        article="Art. 11 - Technical Documentation",
        requirement="Maintain technical documentation demonstrating compliance before placing on market",
        status=ComplianceStatus.IN_PROGRESS,
        owner="engineering-team",
        evidence="Architecture docs + model cards in GitHub",
        deadline=date(2026, 8, 2),
    ),
    ComplianceItem(
        article="Art. 12 - Record-Keeping",
        requirement="Automatic logging of events throughout the AI system lifecycle",
        status=ComplianceStatus.IMPLEMENTED,
        owner="platform-team",
        evidence="Audit trail system deployed (see audit-trails.md)",
        deadline=date(2026, 8, 2),
    ),
    ComplianceItem(
        article="Art. 13 - Transparency",
        requirement="Enable deployers to interpret output and use appropriately",
        status=ComplianceStatus.IN_PROGRESS,
        owner="product-team",
        evidence="AI disclosure banner implemented",
        deadline=date(2026, 8, 2),
    ),
    ComplianceItem(
        article="Art. 14 - Human Oversight",
        requirement="Designed to be effectively overseen by natural persons",
        status=ComplianceStatus.IMPLEMENTED,
        owner="product-team",
        evidence="HITL gates on all consequential actions",
        deadline=date(2026, 8, 2),
    ),
    ComplianceItem(
        article="Art. 15 - Accuracy, Robustness, Cybersecurity",
        requirement="Appropriate levels of accuracy, robustness, and cybersecurity",
        status=ComplianceStatus.IN_PROGRESS,
        owner="security-team",
        evidence="Eval pipeline + red-teaming program",
        deadline=date(2026, 8, 2),
    ),
]


def compliance_report():
    """Generate compliance readiness report."""
    total = len(EU_AI_ACT_CHECKLIST)
    by_status = {}
    for item in EU_AI_ACT_CHECKLIST:
        by_status.setdefault(item.status.value, []).append(item.article)
    
    days_until_deadline = (date(2026, 8, 2) - date.today()).days
    
    print(f"\n=== EU AI Act Compliance Report ===")
    print(f"Days until deadline: {days_until_deadline}")
    print(f"Total requirements: {total}")
    for status, articles in by_status.items():
        print(f"  {status}: {len(articles)} ({', '.join(articles)})")
    
    not_ready = [i for i in EU_AI_ACT_CHECKLIST
                 if i.status in (ComplianceStatus.NOT_STARTED, ComplianceStatus.IN_PROGRESS)]
    if not_ready:
        print(f"\n⚠ {len(not_ready)} items not yet compliant")
```

## Decision Tree / When to Use

```
Does your AI agent serve users in the EU?
  YES --> EU AI Act applies to you
  
Does your agent make or influence CONSEQUENTIAL decisions?
  YES --> Likely HIGH-RISK --> Full compliance required by Aug 2, 2026
  NO  --> Likely LIMITED RISK --> Transparency obligations only

Are you a model PROVIDER or DEPLOYER?
  DEPLOYER (using GPT-4/Claude/etc. in your agent) --> Deployer obligations
  PROVIDER (training/fine-tuning foundation models) --> Provider + deployer obligations
```

## When NOT to Use

- **Agents serving only non-EU users** -- EU AI Act does not apply (but other regulations may)
- **Internal tools with no EU user impact** -- Minimal risk, minimal obligations
- **Research/academic use** -- Exempted from most obligations

## Tradeoffs

| Compliance Level | Legal Risk | Implementation Cost | Time to Market | User Trust |
|-----------------|-----------|--------------------|--------------:|-----------|
| Full compliance | Minimal | High (~6-12 months) | Slower | Highest |
| Partial compliance | Moderate | Medium (~3-6 months) | Moderate | Good |
| Transparency only | Moderate-High | Low (~1-2 months) | Fast | Basic |
| Non-compliant | Very high (up to 7% turnover) | None | Fastest | Low |

## Failure Modes

| Failure | Cause | Impact | Prevention |
|---------|-------|--------|------------|
| **Misclassification** | Assuming agent is minimal risk when it is high-risk | Non-compliance, fines | Conservative classification; legal review |
| **Deadline miss** | Underestimating compliance effort | Fines, market access loss | Start now; 14+ months is not much time |
| **Documentation gap** | No technical docs or model cards | Fails conformity assessment | Maintain docs alongside development |
| **Audit trail missing** | No automatic logging | Art. 12 violation | Implement audit system early |

## Source(s) and Further Reading

- EU AI Act Full Text: https://artificialintelligenceact.eu/
- EU AI Act Explorer: https://artificialintelligenceact.eu/ai-act-explorer/
- European Commission AI Act page: https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- AI Act compliance timeline: https://artificialintelligenceact.eu/implementation/
- NIST AI RMF crosswalk to EU AI Act: https://airc.nist.gov/
