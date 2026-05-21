# Approval Gates with LangGraph
> LangGraph's interrupt/resume mechanism pauses the agent at decision points, sends approval requests to humans, and resumes execution with the human's decision.

## What It Is

Approval gates are checkpoints in the agent execution flow where the system pauses, sends the proposed action to a human for review, and waits for approval before continuing. LangGraph natively supports this through its `interrupt()` function and checkpoint-based resume mechanism.

The execution flow: agent reaches approval node -> state is checkpointed -> human is notified -> human approves/rejects/edits -> execution resumes from checkpoint.

## How It Works

### LangGraph Interrupt/Resume Flow

```
Agent execution:
  Step 1: analyze ✓
  Step 2: plan ✓
  Step 3: prepare action ✓
  Step 4: INTERRUPT (approval needed)
          │
          ├── State checkpointed to PostgreSQL
          ├── Approval request sent to queue
          ├── Notification sent (Slack/email)
          │
          │   ... time passes (seconds to hours) ...
          │
          ├── Human approves via dashboard/Slack
          ├── State resumed from checkpoint
          │
  Step 5: execute approved action ✓
  Step 6: respond ✓
```

## Production Implementation

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import interrupt, Command
import operator
import json
import time
import httpx


# --- State ---

class ApprovalState(TypedDict):
    messages: Annotated[list, operator.add]
    proposed_action: dict
    action_type: str
    confidence: float
    approval_status: str  # "pending", "approved", "rejected", "edited"
    approval_notes: str
    reviewer_id: str


# --- Approval Node ---

async def request_approval(state: ApprovalState) -> dict:
    """
    Pause execution and request human approval.
    Uses LangGraph's interrupt() to checkpoint and wait.
    """
    proposed = state["proposed_action"]
    action_type = state["action_type"]
    confidence = state["confidence"]
    
    # Prepare the approval request context
    approval_context = {
        "action_type": action_type,
        "proposed_action": proposed,
        "confidence": confidence,
        "agent_reasoning": state["messages"][-1].content if state["messages"] else "",
        "timestamp": time.time(),
    }
    
    # Send notification (non-blocking, before interrupt)
    await send_approval_notification(approval_context)
    
    # INTERRUPT: Pause execution, checkpoint state
    # The value passed to interrupt() is what the human sees
    human_decision = interrupt(
        {
            "type": "approval_required",
            "action": proposed,
            "action_type": action_type,
            "confidence": confidence,
            "message": f"Agent wants to {action_type}: {json.dumps(proposed, indent=2)}",
        }
    )
    
    # Execution resumes here when human provides decision via Command
    return {
        "approval_status": human_decision.get("status", "rejected"),
        "approval_notes": human_decision.get("notes", ""),
        "reviewer_id": human_decision.get("reviewer_id", "unknown"),
        "proposed_action": human_decision.get("edited_action", proposed),
    }


async def execute_action(state: ApprovalState) -> dict:
    """Execute the approved action."""
    action = state["proposed_action"]
    
    # Execute the action
    result = await execute_tool(action["tool"], action["args"])
    
    return {
        "messages": [{"role": "assistant", "content": f"Action completed: {result}"}],
    }


async def handle_rejection(state: ApprovalState) -> dict:
    """Handle when human rejects the proposed action."""
    notes = state.get("approval_notes", "No reason provided")
    return {
        "messages": [{"role": "assistant", "content": f"Action was not approved. Reviewer notes: {notes}"}],
    }


# --- Routing ---

def route_after_approval(state: ApprovalState) -> Literal["execute_action", "handle_rejection"]:
    if state["approval_status"] in ("approved", "edited"):
        return "execute_action"
    return "handle_rejection"


# --- Graph ---

def build_approval_graph():
    graph = StateGraph(ApprovalState)
    
    graph.add_node("prepare_action", prepare_action)
    graph.add_node("request_approval", request_approval)
    graph.add_node("execute_action", execute_action)
    graph.add_node("handle_rejection", handle_rejection)
    
    graph.set_entry_point("prepare_action")
    graph.add_edge("prepare_action", "request_approval")
    graph.add_conditional_edges("request_approval", route_after_approval)
    graph.add_edge("execute_action", END)
    graph.add_edge("handle_rejection", END)
    
    checkpointer = AsyncPostgresSaver.from_conn_string(
        "postgresql://user:pass@localhost:5432/agents"
    )
    
    return graph.compile(checkpointer=checkpointer)


# --- API for Human Approval ---

from fastapi import FastAPI

app = FastAPI()

@app.post("/api/v1/approvals/{thread_id}/decide")
async def submit_approval(thread_id: str, decision: dict):
    """Human submits approval decision, resuming the agent."""
    graph = build_approval_graph()
    
    # Resume execution from the interrupt point
    result = await graph.ainvoke(
        Command(resume=decision),
        config={"configurable": {"thread_id": thread_id}},
    )
    
    return {"status": "resumed", "result": result}


# --- Notification Integration ---

async def send_approval_notification(context: dict):
    """Send approval notification via Slack."""
    slack_message = {
        "text": f"Approval Required: {context['action_type']}",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Agent needs approval*\n"
                            f"Action: `{context['action_type']}`\n"
                            f"Confidence: {context['confidence']:.0%}\n"
                            f"```{json.dumps(context['proposed_action'], indent=2)}```",
                },
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "style": "primary",
                        "action_id": "approve",
                        "value": json.dumps({"thread_id": context.get("thread_id")}),
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Reject"},
                        "style": "danger",
                        "action_id": "reject",
                        "value": json.dumps({"thread_id": context.get("thread_id")}),
                    },
                ],
            },
        ],
    }
    
    async with httpx.AsyncClient() as client:
        await client.post(
            "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
            json=slack_message,
        )


# --- Timeout Handling ---

import asyncio

class ApprovalTimeoutHandler:
    """Handle approval requests that don't get a response."""
    
    def __init__(self, default_timeout: int = 3600):
        self.default_timeout = default_timeout
        self.timeout_policies = {
            "read_only": 0,           # Never needs approval
            "internal_write": 3600,   # 1 hour
            "external_comms": 7200,   # 2 hours
            "financial": 86400,       # 24 hours (never auto-approve)
        }
    
    async def check_expired_approvals(self):
        """Background job to handle expired approval requests."""
        while True:
            expired = await self._get_expired_requests()
            
            for request in expired:
                action_type = request["action_type"]
                policy = self.timeout_policies.get(action_type, self.default_timeout)
                
                if policy == 0:
                    # Auto-approve on timeout (low risk)
                    await self._auto_approve(request)
                else:
                    # Escalate to manager
                    await self._escalate(request)
            
            await asyncio.sleep(60)  # Check every minute
```

## Decision Tree: Approval Gate Placement

```
Where to place approval gates in the graph?
│
├── Before every tool call? → Too aggressive, slows everything down
│
├── Before write tool calls only? → Good default for most systems
│
├── Before the final response? → Good for customer-facing agents
│
└── At specific decision points? → Best: identify the 2-3 highest-risk points
    ├── After "prepare action" but before "execute action"
    ├── After "draft email" but before "send email"
    └── After "calculate refund" but before "process refund"
```

## When NOT to Use Approval Gates

- **Internal development tools**: Engineers can handle mistakes. Auto-log instead of approve.
- **High-volume automated pipelines**: Can't approve 10,000 actions/hour. Use sampling.
- **Read-only agents**: No state changes = no approval needed.

## Tradeoffs

| Approach | Latency | Safety | Scalability |
|----------|---------|--------|-------------|
| No gates | Fastest | Lowest | Best |
| Selective gates (high-risk only) | Minimal impact | High | Good |
| All-writes gates | Moderate delay | Very high | Limited by human bandwidth |
| All-actions gates | Highest delay | Highest | Not scalable |

## Failure Modes

1. **Approval queue overflow**: Too many requests, humans can't keep up. Mitigation: auto-approve low-risk after timeout, hire more reviewers.
2. **Checkpoint corruption**: PostgreSQL checkpoint can't be resumed. Mitigation: checkpoint verification, fallback to restart run.
3. **Notification delivery failure**: Slack webhook down, no one knows approval is pending. Mitigation: multi-channel notification (Slack + email), dashboard polling.

## Source(s) and Further Reading

- LangGraph Interrupt: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/
- LangGraph Command: https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
- Slack Interactive Messages: https://api.slack.com/messaging/interactivity
