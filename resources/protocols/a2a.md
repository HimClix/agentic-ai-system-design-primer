# Agent-to-Agent Protocol (A2A)
> A2A is for agent-to-agent communication across organizational boundaries -- when your agent needs to talk to another company's agent, not just call a tool.

## What It Is

The Agent-to-Agent (A2A) protocol, developed by Google, enables AI agents to discover, communicate with, and delegate tasks to other AI agents across different frameworks, organizations, and platforms. While MCP connects agents to tools, A2A connects agents to agents.

A2A addresses the challenge of distributed agent intelligence: when no single agent has all the capabilities needed, and agents from different systems must collaborate to complete a task.

## How It Works

### Core Concepts

```
Agent A (Client)                          Agent B (Server)
┌──────────────────┐                     ┌──────────────────┐
│                  │  1. Discover         │                  │
│  "I need help    │────────────────────►│  Agent Card      │
│   with X"        │  (GET /agent-card)  │  {name, skills,  │
│                  │◄────────────────────│   capabilities}  │
│                  │                     │                  │
│  2. Create Task  │  3. Send Task       │                  │
│  {id, desc,      │────────────────────►│  4. Process Task │
│   context}       │  (POST /tasks)      │  - Plan          │
│                  │                     │  - Execute       │
│                  │  5. Stream Updates  │  - Verify        │
│                  │◄────────────────────│                  │
│                  │  (SSE / webhook)    │  6. Return Result│
│  7. Use Result   │◄────────────────────│  {output, status}│
└──────────────────┘                     └──────────────────┘
```

### Key Components

| Component | Purpose | Format |
|-----------|---------|--------|
| **Agent Card** | Agent's identity and capabilities | JSON at `/.well-known/agent.json` |
| **Task** | Unit of work delegated between agents | JSON with id, description, context |
| **Message** | Communication within a task | Text/structured data with parts |
| **Artifact** | Output produced by the agent | Files, structured data, references |

### Agent Card Example

```json
{
  "name": "travel-booking-agent",
  "description": "Handles flight and hotel booking, itinerary planning",
  "url": "https://travel-agent.example.com",
  "version": "1.0",
  "capabilities": {
    "streaming": true,
    "pushNotifications": true,
    "stateTransitionHistory": true
  },
  "skills": [
    {
      "id": "book-flight",
      "name": "Flight Booking",
      "description": "Search and book flights",
      "inputModes": ["text", "structured"],
      "outputModes": ["text", "structured"]
    },
    {
      "id": "book-hotel", 
      "name": "Hotel Booking",
      "description": "Search and book hotels",
      "inputModes": ["text"],
      "outputModes": ["text", "structured"]
    }
  ],
  "authentication": {
    "schemes": ["oauth2", "apiKey"]
  }
}
```

### Task Lifecycle

```
submitted → working → [input_required] → working → completed
                 │                                      │
                 └──→ failed                           │
                 └──→ cancelled                        │
                                                       └── artifacts available
```

## Production Implementation

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import httpx
import json


class TaskStatus(Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentCard:
    name: str
    description: str
    url: str
    skills: list[dict]
    version: str = "1.0"
    authentication: dict = field(default_factory=dict)


@dataclass 
class A2ATask:
    id: str
    description: str
    status: TaskStatus = TaskStatus.SUBMITTED
    context: dict = field(default_factory=dict)
    artifacts: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)


class A2AClient:
    """Client for communicating with remote A2A agents."""
    
    def __init__(self, timeout: int = 120):
        self.timeout = timeout
        self._agent_cards: dict[str, AgentCard] = {}
    
    async def discover_agent(self, base_url: str) -> AgentCard:
        """Discover a remote agent's capabilities via its Agent Card."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{base_url}/.well-known/agent.json",
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
        
        card = AgentCard(
            name=data["name"],
            description=data["description"],
            url=data["url"],
            skills=data.get("skills", []),
            version=data.get("version", "1.0"),
            authentication=data.get("authentication", {}),
        )
        self._agent_cards[base_url] = card
        return card
    
    async def send_task(
        self,
        agent_url: str,
        task_description: str,
        context: dict = None,
        auth_token: str = None,
    ) -> A2ATask:
        """Send a task to a remote agent."""
        import uuid
        task_id = str(uuid.uuid4())
        
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "id": task_id,
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": task_description}],
                },
            },
        }
        
        if context:
            payload["params"]["metadata"] = context
        
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                agent_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
        
        return A2ATask(
            id=task_id,
            description=task_description,
            status=TaskStatus(result.get("result", {}).get("status", "submitted")),
            artifacts=result.get("result", {}).get("artifacts", []),
        )
    
    async def get_task_status(
        self, agent_url: str, task_id: str
    ) -> A2ATask:
        """Poll for task status."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"id": task_id},
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                agent_url,
                json=payload,
                timeout=30,
            )
            result = response.json()
        
        task_data = result.get("result", {})
        return A2ATask(
            id=task_id,
            description="",
            status=TaskStatus(task_data.get("status", "unknown")),
            artifacts=task_data.get("artifacts", []),
            history=task_data.get("history", []),
        )


# --- A2A Server (your agent as a server) ---

from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/.well-known/agent.json")
async def agent_card():
    """Serve the Agent Card for discovery."""
    return {
        "name": "support-agent",
        "description": "Handles customer support queries with knowledge base access",
        "url": "https://support-agent.example.com",
        "version": "1.0",
        "skills": [
            {
                "id": "answer-question",
                "name": "Answer Support Question",
                "description": "Answer customer questions using the knowledge base",
                "inputModes": ["text"],
                "outputModes": ["text"],
            },
            {
                "id": "create-ticket",
                "name": "Create Support Ticket",
                "description": "Create a support ticket for unresolved issues",
                "inputModes": ["text", "structured"],
                "outputModes": ["structured"],
            },
        ],
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
        },
    }

@app.post("/")
async def handle_jsonrpc(request: Request):
    """Handle A2A JSON-RPC requests."""
    body = await request.json()
    method = body.get("method")
    
    if method == "tasks/send":
        return await handle_task_send(body)
    elif method == "tasks/get":
        return await handle_task_get(body)
    elif method == "tasks/cancel":
        return await handle_task_cancel(body)
    else:
        return {"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}}
```

## Decision Tree: When You Need A2A

```
Does your agent need to communicate with other agents?
│
├── NO → You don't need A2A. Use MCP for tool access.
│
└── YES → Are the other agents in your organization?
    │
    ├── YES → Can you use shared state (database, queue)?
    │   ├── YES → Direct integration is simpler than A2A
    │   └── NO → A2A for loose coupling between teams
    │
    └── NO → Cross-organization agents?
        └── YES → A2A is the right protocol
            ├── Partner APIs exposing agent capabilities
            ├── Marketplace of specialized agents
            └── Federated agent systems
```

## When NOT to Use A2A

- **Single-org, single-codebase**: Direct function calls or shared state is simpler.
- **Agent calling tools**: Use MCP, not A2A. A2A is for agent-agent, not agent-tool.
- **Tight coupling needed**: A2A adds latency and complexity. If agents must be tightly synchronized, co-locate them.

## Tradeoffs

| Aspect | A2A | Direct Integration |
|--------|-----|-------------------|
| Decoupling | High (agents are independent services) | Low (shared codebase) |
| Latency | Higher (network + protocol overhead) | Lower |
| Discoverability | Built-in (Agent Cards) | Requires documentation |
| Cross-org support | Native | Requires custom API design |
| Complexity | Higher | Lower |

## Failure Modes

1. **Agent Card stale**: Remote agent updated its capabilities but card is cached. Mitigation: TTL-based card refresh.
2. **Task timeout**: Remote agent takes too long. Mitigation: timeout configuration, async polling with max attempts.
3. **Trust boundary**: How to verify the remote agent is who it claims? Mitigation: mutual TLS, OAuth 2.0 verification.

## Source(s) and Further Reading

- A2A Protocol: https://github.com/google/A2A
- A2A Specification: https://google.github.io/A2A/
- "Agent-to-Agent Communication in Multi-Agent Systems" - Google (2025)
