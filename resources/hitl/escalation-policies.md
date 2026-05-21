# Escalation Policies
> When to escalate, who to escalate to, and how long to wait at each level -- the chain from agent to approval queue to human to manager.

## What It Is

Escalation policies define the rules for when an agent should stop trying to handle a request autonomously and hand off to a human. They specify trigger conditions, escalation chains, notification methods, and SLAs for response times at each level.

The key principle: escalation is not failure. It is the agent correctly recognizing the limits of its capability and routing to the right human at the right time.

## How It Works

### Escalation Triggers

```
Trigger Category 1: Confidence-Based
├── Agent confidence < threshold for action type
├── Multiple conflicting tool results
└── RAG retrieval relevance < 0.5

Trigger Category 2: Complexity-Based
├── Task requires > 25 steps
├── Task involves multiple departments
└── Novel problem not in knowledge base

Trigger Category 3: Risk-Based
├── Financial amount > $500
├── Legal/compliance implications
├── Customer VIP flag
└── Account flagged for fraud

Trigger Category 4: Failure-Based
├── > 3 consecutive tool call failures
├── Agent stuck in loop (FM-1.3)
├── Conflicting requirements detected
└── Agent explicitly says "I don't know"
```

### Escalation Chain

```
Level 0: Agent (autonomous)
    │ trigger condition met
    ▼
Level 1: Approval Queue (self-serve)
    │ SLA: 15 minutes
    │ Who: Available team member via Slack/dashboard
    ▼
Level 2: Domain Expert
    │ SLA: 1 hour
    │ Who: Subject matter expert for the topic
    ▼
Level 3: Team Lead / Manager
    │ SLA: 4 hours
    │ Who: Escalation manager on-call
    ▼
Level 4: Incident Response
    │ SLA: Immediate
    │ Who: On-call engineer + product manager
    └── For: system failures, security incidents, data breaches
```

## Production Implementation

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time
import asyncio


class EscalationLevel(Enum):
    AGENT = 0           # Agent handles autonomously
    APPROVAL_QUEUE = 1  # First available team member
    DOMAIN_EXPERT = 2   # Subject matter expert
    TEAM_LEAD = 3       # Management escalation
    INCIDENT = 4        # Incident response team


class EscalationTrigger(Enum):
    LOW_CONFIDENCE = "low_confidence"
    HIGH_RISK_ACTION = "high_risk_action"
    COMPLEXITY_EXCEEDED = "complexity_exceeded"
    TOOL_FAILURES = "tool_failures"
    AGENT_LOOP = "agent_loop"
    EXPLICIT_UNKNOWN = "explicit_unknown"
    SLA_BREACH = "sla_breach"
    CUSTOMER_REQUEST = "customer_request"
    VIP_CUSTOMER = "vip_customer"


@dataclass
class EscalationConfig:
    """Configuration for escalation at each level."""
    level: EscalationLevel
    sla_seconds: int                    # Max time before auto-escalating to next level
    notification_channels: list[str]    # ["slack", "email", "pagerduty"]
    assignee_group: str                 # Team or role to assign to
    auto_escalate_on_timeout: bool = True
    max_reassignments: int = 3


@dataclass
class EscalationEvent:
    """Record of an escalation event."""
    run_id: str
    session_id: str
    tenant_id: str
    trigger: EscalationTrigger
    current_level: EscalationLevel
    context: dict
    created_at: float = field(default_factory=time.time)
    assigned_to: Optional[str] = None
    resolved_at: Optional[float] = None
    resolution: Optional[str] = None


class EscalationManager:
    """Manages the escalation chain for agent runs."""
    
    def __init__(self):
        self.configs = {
            EscalationLevel.APPROVAL_QUEUE: EscalationConfig(
                level=EscalationLevel.APPROVAL_QUEUE,
                sla_seconds=900,           # 15 minutes
                notification_channels=["slack"],
                assignee_group="support-team",
            ),
            EscalationLevel.DOMAIN_EXPERT: EscalationConfig(
                level=EscalationLevel.DOMAIN_EXPERT,
                sla_seconds=3600,          # 1 hour
                notification_channels=["slack", "email"],
                assignee_group="domain-experts",
            ),
            EscalationLevel.TEAM_LEAD: EscalationConfig(
                level=EscalationLevel.TEAM_LEAD,
                sla_seconds=14400,         # 4 hours
                notification_channels=["slack", "email", "pagerduty"],
                assignee_group="team-leads",
            ),
            EscalationLevel.INCIDENT: EscalationConfig(
                level=EscalationLevel.INCIDENT,
                sla_seconds=0,             # Immediate
                notification_channels=["pagerduty", "slack"],
                assignee_group="incident-response",
            ),
        }
        self._active_escalations: dict[str, EscalationEvent] = {}
    
    async def evaluate_escalation(
        self,
        run_id: str,
        confidence: float,
        action_type: str,
        step_count: int,
        error_count: int,
        context: dict,
    ) -> Optional[EscalationEvent]:
        """Evaluate if escalation is needed based on current state."""
        
        trigger = self._determine_trigger(
            confidence=confidence,
            action_type=action_type,
            step_count=step_count,
            error_count=error_count,
            context=context,
        )
        
        if trigger is None:
            return None  # No escalation needed
        
        level = self._determine_level(trigger, context)
        
        event = EscalationEvent(
            run_id=run_id,
            session_id=context.get("session_id", ""),
            tenant_id=context.get("tenant_id", ""),
            trigger=trigger,
            current_level=level,
            context=context,
        )
        
        # Send notifications
        config = self.configs[level]
        await self._notify(event, config)
        
        self._active_escalations[run_id] = event
        return event
    
    def _determine_trigger(
        self,
        confidence: float,
        action_type: str,
        step_count: int,
        error_count: int,
        context: dict,
    ) -> Optional[EscalationTrigger]:
        """Determine which trigger condition is met."""
        
        # VIP customer always escalates
        if context.get("is_vip"):
            return EscalationTrigger.VIP_CUSTOMER
        
        # Customer explicitly requested human
        if context.get("customer_requested_human"):
            return EscalationTrigger.CUSTOMER_REQUEST
        
        # High-risk action
        if action_type in ("financial", "production", "data_deletion"):
            return EscalationTrigger.HIGH_RISK_ACTION
        
        # Agent stuck
        if step_count > 25:
            return EscalationTrigger.COMPLEXITY_EXCEEDED
        
        # Too many failures
        if error_count > 3:
            return EscalationTrigger.TOOL_FAILURES
        
        # Low confidence
        confidence_thresholds = {
            "read_only": 0.0,       # Never escalate for reads
            "internal_write": 0.7,
            "external_comms": 0.9,
            "financial": 0.95,
        }
        threshold = confidence_thresholds.get(action_type, 0.85)
        if confidence < threshold:
            return EscalationTrigger.LOW_CONFIDENCE
        
        return None
    
    def _determine_level(
        self, trigger: EscalationTrigger, context: dict
    ) -> EscalationLevel:
        """Determine escalation level based on trigger."""
        level_map = {
            EscalationTrigger.LOW_CONFIDENCE: EscalationLevel.APPROVAL_QUEUE,
            EscalationTrigger.HIGH_RISK_ACTION: EscalationLevel.APPROVAL_QUEUE,
            EscalationTrigger.COMPLEXITY_EXCEEDED: EscalationLevel.DOMAIN_EXPERT,
            EscalationTrigger.TOOL_FAILURES: EscalationLevel.DOMAIN_EXPERT,
            EscalationTrigger.AGENT_LOOP: EscalationLevel.DOMAIN_EXPERT,
            EscalationTrigger.EXPLICIT_UNKNOWN: EscalationLevel.APPROVAL_QUEUE,
            EscalationTrigger.SLA_BREACH: EscalationLevel.TEAM_LEAD,
            EscalationTrigger.CUSTOMER_REQUEST: EscalationLevel.APPROVAL_QUEUE,
            EscalationTrigger.VIP_CUSTOMER: EscalationLevel.DOMAIN_EXPERT,
        }
        return level_map.get(trigger, EscalationLevel.APPROVAL_QUEUE)
    
    async def _notify(self, event: EscalationEvent, config: EscalationConfig):
        """Send notifications through configured channels."""
        for channel in config.notification_channels:
            if channel == "slack":
                await self._notify_slack(event, config)
            elif channel == "email":
                await self._notify_email(event, config)
            elif channel == "pagerduty":
                await self._notify_pagerduty(event, config)
    
    async def check_sla_breaches(self):
        """Background job: auto-escalate if SLA is breached."""
        while True:
            now = time.time()
            for run_id, event in list(self._active_escalations.items()):
                if event.resolved_at:
                    continue
                
                config = self.configs[event.current_level]
                elapsed = now - event.created_at
                
                if elapsed > config.sla_seconds and config.auto_escalate_on_timeout:
                    # Escalate to next level
                    next_level = EscalationLevel(event.current_level.value + 1)
                    if next_level.value <= EscalationLevel.INCIDENT.value:
                        event.current_level = next_level
                        next_config = self.configs[next_level]
                        await self._notify(event, next_config)
            
            await asyncio.sleep(60)
```

## Decision Tree: Escalation Level Selection

```
What triggered the escalation?
│
├── Customer requested human → Level 1 (Approval Queue)
│
├── Low confidence → Level 1 (Approval Queue)
│   └── If not resolved in 15 min → Level 2 (Domain Expert)
│
├── Complex/novel problem → Level 2 (Domain Expert)
│   └── If not resolved in 1 hour → Level 3 (Team Lead)
│
├── Repeated tool failures → Level 2 (Domain Expert)
│   └── If systemic issue → Level 4 (Incident)
│
├── VIP customer → Level 2 (Domain Expert, immediate)
│
└── System failure → Level 4 (Incident, immediate)
```

## SLA Reference Table

| Level | Response SLA | Resolution SLA | Auto-Escalate After |
|-------|-------------|---------------|-------------------|
| L1 - Approval Queue | 5 min | 15 min | 15 min |
| L2 - Domain Expert | 15 min | 1 hour | 1 hour |
| L3 - Team Lead | 30 min | 4 hours | 4 hours |
| L4 - Incident | Immediate | Best effort | Never (already top) |

## When NOT to Use Escalation Policies

- **Batch processing**: No human waiting for a response. Log issues for review.
- **Internal tools with tolerant users**: Engineers can retry or debug themselves.
- **Prototype/demo**: Over-engineering escalation for a demo is premature.

## Failure Modes

1. **Escalation storm**: Many requests escalate simultaneously, overwhelming the queue. Mitigation: rate-limit escalations, batch similar issues.
2. **Wrong expert routed**: Domain expert gets a question outside their area. Mitigation: include topic classification in escalation, allow reassignment.
3. **SLA timer race condition**: Timer fires right as human is responding. Mitigation: check for pending responses before auto-escalating.
4. **Notification fatigue**: Too many Slack messages, humans start ignoring them. Mitigation: batch notifications, digest mode for low-priority.

## Source(s) and Further Reading

- PagerDuty Escalation Policies: https://support.pagerduty.com/docs/escalation-policies
- "Incident Management at Google" - SRE Book
- LangGraph HITL: https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
