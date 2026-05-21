# Human-in-the-Loop (HITL) for Agentic Systems
> HITL is a feature, not a limitation -- production agents that never involve humans are either trivially simple or dangerously overconfident.

## What It Is

Human-in-the-Loop (HITL) is the architectural pattern of incorporating human oversight and approval into agent workflows. Instead of treating human involvement as a failure state, production systems design it as a deliberate feature for high-risk, high-stakes, or uncertain decisions.

The key insight: the goal is not to eliminate human involvement but to **maximize the value of human attention** by having the agent handle routine work autonomously and escalate only when human judgment is truly needed.

## How It Works

### Where Approvals Are Required

```
Action Risk Matrix:

                    Low Confidence    High Confidence
                    ─────────────    ────────────────
Read-only          Proceed (log)     Proceed (log)
Internal write     Review queue      Auto-approve
External comms     Always approve    Review queue  
Financial          Always approve    Always approve
Production deploy  Always approve    Always approve
Data deletion      Always approve    Always approve
```

### Decision Table by Action Type

| Action Type | Risk Level | Default Policy | Override Conditions |
|------------|-----------|---------------|-------------------|
| Search / read data | Low | Auto-approve | None |
| Update internal record | Medium | Auto if confidence > 0.7 | New field values, bulk updates |
| Send email to customer | High | Always approve | None (regulatory requirement) |
| Process refund < $50 | Medium | Auto if confidence > 0.85 | First-time customer, flagged account |
| Process refund > $500 | Critical | Always approve | None |
| Deploy to production | Critical | Always approve | None |
| Delete user data | Critical | Always approve | GDPR request with verified identity |
| Create support ticket | Low | Auto-approve | Sensitive topics (legal, security) |
| Modify subscription | High | Auto if confidence > 0.9 | Downgrade, cancellation |

### HITL Architecture

```
Agent Loop
    │
    ├── Step 1: Analyze request (auto)
    │
    ├── Step 2: Search knowledge base (auto)
    │
    ├── Step 3: Draft response (auto)
    │
    ├── Step 4: APPROVAL GATE ◄─── interrupt()
    │       │
    │       ▼
    │   ┌──────────────────┐
    │   │  Approval Queue   │
    │   │                   │
    │   │  - Slack notify   │
    │   │  - Email notify   │
    │   │  - Dashboard view │
    │   │                   │
    │   │  Human reviews:   │
    │   │  ✓ Approve        │
    │   │  ✗ Reject         │
    │   │  ✎ Edit & approve │
    │   └────────┬──────────┘
    │            │ resume()
    │            ▼
    ├── Step 5: Execute approved action
    │
    └── Step 6: Send response (auto)
```

## Production Implementation

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited_and_approved"
    EXPIRED = "expired"


class ActionType(Enum):
    READ_ONLY = "read_only"
    INTERNAL_WRITE = "internal_write"
    EXTERNAL_COMMS = "external_comms"
    FINANCIAL = "financial"
    PRODUCTION = "production"
    DATA_DELETION = "data_deletion"


@dataclass
class ApprovalRequest:
    """Request submitted to the approval queue."""
    run_id: str
    session_id: str
    tenant_id: str
    action_type: ActionType
    action_description: str
    agent_reasoning: str
    proposed_action: dict
    confidence: float
    context: dict  # Relevant context for the human reviewer
    requested_at: float
    timeout_seconds: int = 3600  # 1 hour default


@dataclass 
class ApprovalDecision:
    """Human's decision on an approval request."""
    status: ApprovalStatus
    reviewer_id: str
    notes: Optional[str] = None
    edited_action: Optional[dict] = None  # Modified action if edited
    decided_at: Optional[float] = None


class HITLPolicy:
    """Policy engine that determines when human approval is needed."""
    
    def __init__(self):
        self.policies: dict[ActionType, dict] = {
            ActionType.READ_ONLY: {
                "always_approve": True,
                "confidence_threshold": 0.0,
            },
            ActionType.INTERNAL_WRITE: {
                "always_approve": False,
                "confidence_threshold": 0.7,
            },
            ActionType.EXTERNAL_COMMS: {
                "always_approve": False,
                "confidence_threshold": 1.1,  # > 1.0 = always needs approval
            },
            ActionType.FINANCIAL: {
                "always_approve": False,
                "confidence_threshold": 1.1,  # Always needs approval
            },
            ActionType.PRODUCTION: {
                "always_approve": False,
                "confidence_threshold": 1.1,
            },
            ActionType.DATA_DELETION: {
                "always_approve": False,
                "confidence_threshold": 1.1,
            },
        }
    
    def needs_approval(
        self, action_type: ActionType, confidence: float, context: dict = None
    ) -> tuple[bool, str]:
        """Determine if an action needs human approval."""
        policy = self.policies[action_type]
        
        if policy["always_approve"]:
            return False, "Auto-approved: read-only action"
        
        if confidence >= policy["confidence_threshold"]:
            return False, f"Auto-approved: confidence {confidence:.2f} >= {policy['confidence_threshold']}"
        
        # Context-based overrides
        if context:
            if context.get("amount_usd", 0) > 500:
                return True, f"Requires approval: amount ${context['amount_usd']} > $500"
            if context.get("is_new_customer"):
                return True, "Requires approval: new customer"
            if context.get("flagged_account"):
                return True, "Requires approval: flagged account"
        
        return True, f"Requires approval: confidence {confidence:.2f} < {policy['confidence_threshold']}"
```

## Decision Tree: When to Add HITL

```
Is the action reversible?
│
├── YES (e.g., update a draft, internal note)
│   └── Consider auto-approve with confidence > 0.7
│       └── Log for post-hoc review
│
└── NO (e.g., send email, process payment, delete data)
    │
    ├── Low risk (< $50, internal only)?
    │   └── Auto-approve with confidence > 0.85
    │
    ├── Medium risk ($50-500, external comms)?
    │   └── Require approval, allow timeout auto-approval
    │
    └── High risk (> $500, production, legal)?
        └── Always require approval, no timeout auto-approval
```

## When NOT to Use HITL

- **Latency-critical chatbot**: If users expect sub-second responses, HITL adds unacceptable delay. Use guardrails instead.
- **Fully autonomous batch processing**: If the agent processes 10,000 documents, approving each one is infeasible. Use sampling + spot-checks.
- **Read-only agents**: If the agent only searches and summarizes, no approval gates are needed.

## Tradeoffs

| HITL Strategy | User Experience | Safety | Operational Cost |
|--------------|----------------|--------|-----------------|
| No HITL (fully autonomous) | Best (instant) | Lowest | Lowest |
| Approval for high-risk only | Good (most actions instant) | High | Low-Medium |
| Approval for all writes | Moderate (delays on writes) | Very High | High |
| Approval for everything | Poor (constant waiting) | Highest | Very High |

## Real-World Examples

- **Intercom's Fin**: Auto-resolves simple questions, escalates to human agents for complex issues or when confidence is low.
- **GitHub Copilot Workspace**: Generates code changes but requires human review before committing.
- **Financial trading agents**: All trade executions require human confirmation regardless of confidence.

## Failure Modes

1. **Approval fatigue**: Humans start rubber-stamping everything. Mitigation: risk-based approval (not everything), clear context in requests.
2. **SLA breach on approvals**: Approval takes too long, customer waits. Mitigation: timeout policies, escalation chains.
3. **Stale approval context**: By the time a human reviews, the situation has changed. Mitigation: include timestamp, re-validate before execution.

## Source(s) and Further Reading

- LangGraph Human-in-the-Loop: https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
- "Building Effective Agents" - Anthropic (2024)
- "The Last Mile of AI Automation" - Harvard Business Review (2024)
