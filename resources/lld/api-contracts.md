# Agent API Contracts
> Agent APIs differ from traditional REST: they return run IDs for async operations, stream intermediate steps via SSE, and include token usage in every response.

## What It Is

API contracts for agent services define the HTTP endpoints, request/response schemas, streaming formats, and error handling for interacting with the agent. Agent APIs have unique requirements: long-running operations, streaming intermediate results, and rich metadata (token usage, tool calls, confidence scores).

## How It Works

### Core Endpoints

```
POST   /api/v1/agent/run           Create a new agent run
GET    /api/v1/agent/run/{run_id}  Get run status and result
GET    /api/v1/agent/run/{run_id}/stream  Stream run events (SSE)
POST   /api/v1/agent/run/{run_id}/cancel  Cancel a running run
GET    /api/v1/agent/sessions/{session_id}/history  Get conversation history
DELETE /api/v1/agent/sessions/{session_id}  Delete session
GET    /api/v1/agent/health        Health check
```

## Production Implementation

### Request/Response Schemas

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
import uuid


# --- Request Schemas ---

class AgentRunRequest(BaseModel):
    """POST /api/v1/agent/run"""
    message: str = Field(..., min_length=1, max_length=10_000, description="User message")
    session_id: Optional[str] = Field(
        default=None, 
        description="Session ID for conversation continuity. Omit to start new session."
    )
    model_preference: Optional[str] = Field(
        default=None, 
        description="Preferred model: 'fast', 'balanced', 'powerful'"
    )
    stream: bool = Field(
        default=False, 
        description="If true, returns SSE stream instead of waiting for completion"
    )
    metadata: Optional[dict] = Field(
        default=None, 
        description="Arbitrary metadata attached to the run"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is the refund policy for premium plans?",
                "session_id": "sess_abc123",
                "stream": True,
            }
        }


# --- Response Schemas ---

class ToolCallRecord(BaseModel):
    """Record of a single tool call made during the run."""
    tool_name: str
    arguments: dict
    result: Optional[str] = None
    status: Literal["success", "error", "timeout"]
    latency_ms: int
    timestamp: datetime


class UsageInfo(BaseModel):
    """Token usage for the run."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    model_used: str
    cache_hit: bool = False


class AgentRunResponse(BaseModel):
    """GET /api/v1/agent/run/{run_id}"""
    run_id: str
    session_id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled", "escalated"]
    message: Optional[str] = Field(None, description="Agent's response message")
    
    # Execution details
    steps_completed: int = 0
    tool_calls: list[ToolCallRecord] = []
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Metadata
    usage: Optional[UsageInfo] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    # Error info
    error: Optional[str] = None
    error_code: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "run_id": "run_abc123",
                "session_id": "sess_abc123",
                "status": "completed",
                "message": "Our refund policy for premium plans allows full refunds within 30 days...",
                "steps_completed": 3,
                "tool_calls": [{
                    "tool_name": "search_knowledge_base",
                    "arguments": {"query": "refund policy premium"},
                    "result": "Premium plan refund policy...",
                    "status": "success",
                    "latency_ms": 234,
                    "timestamp": "2025-01-15T10:30:00Z",
                }],
                "confidence": 0.92,
                "usage": {
                    "input_tokens": 2800,
                    "output_tokens": 350,
                    "total_tokens": 3150,
                    "estimated_cost_usd": 0.0137,
                    "model_used": "claude-sonnet-4",
                    "cache_hit": False,
                },
                "created_at": "2025-01-15T10:30:00Z",
                "completed_at": "2025-01-15T10:30:08Z",
            }
        }


# --- SSE Streaming Format ---

class StreamEvent(BaseModel):
    """Single event in the SSE stream."""
    event: Literal[
        "run.started",
        "step.started",
        "step.tool_call",
        "step.tool_result",
        "step.completed",
        "message.delta",      # Partial response token
        "message.complete",   # Full response
        "run.completed",
        "run.failed",
        "run.escalated",
    ]
    data: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# --- Error Responses ---

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    error_code: str
    message: str
    details: Optional[dict] = None
    retry_after: Optional[int] = Field(None, description="Seconds to wait before retry")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "rate_limit_exceeded",
                "error_code": "RATE_LIMIT",
                "message": "Token budget exceeded for this session. Please start a new session.",
                "retry_after": 60,
            }
        }


# Error codes
ERROR_CODES = {
    "RATE_LIMIT": (429, "Rate limit exceeded"),
    "BUDGET_EXCEEDED": (429, "Token budget exceeded"),
    "SESSION_NOT_FOUND": (404, "Session not found"),
    "RUN_NOT_FOUND": (404, "Run not found"),
    "INVALID_INPUT": (400, "Invalid input"),
    "TOOL_FAILURE": (502, "Tool execution failed"),
    "LLM_UNAVAILABLE": (503, "LLM provider unavailable"),
    "TIMEOUT": (504, "Agent run timed out"),
    "INTERNAL_ERROR": (500, "Internal server error"),
}


# --- Conversation History ---

class ConversationTurn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime
    run_id: Optional[str] = None
    tool_calls: list[ToolCallRecord] = []


class ConversationHistoryResponse(BaseModel):
    """GET /api/v1/agent/sessions/{session_id}/history"""
    session_id: str
    turns: list[ConversationTurn]
    total_turns: int
    total_tokens: int
    created_at: datetime
    last_activity: datetime
    
    # Pagination
    has_more: bool = False
    next_cursor: Optional[str] = None
```

### SSE Stream Format

```
# Client sends: POST /api/v1/agent/run {"message": "...", "stream": true}
# Server responds with SSE stream:

event: run.started
data: {"run_id": "run_abc123", "session_id": "sess_abc123"}

event: step.started
data: {"step": 1, "phase": "planning"}

event: step.tool_call
data: {"tool": "search_knowledge_base", "args": {"query": "refund policy"}}

event: step.tool_result
data: {"tool": "search_knowledge_base", "status": "success", "latency_ms": 234}

event: step.completed
data: {"step": 1, "phase": "planning"}

event: message.delta
data: {"token": "Our "}

event: message.delta
data: {"token": "refund "}

event: message.delta
data: {"token": "policy "}

event: message.complete
data: {"message": "Our refund policy for premium plans allows..."}

event: run.completed
data: {"run_id": "run_abc123", "steps": 3, "tokens": 3150, "cost_usd": 0.0137}
```

### FastAPI Implementation

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import asyncio


app = FastAPI(title="Agent API", version="1.0.0")


@app.post("/api/v1/agent/run", response_model=AgentRunResponse)
async def create_run(request: AgentRunRequest):
    """Create a new agent run."""
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"
    
    if request.stream:
        return StreamingResponse(
            stream_agent_run(run_id, session_id, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    
    # Synchronous execution
    result = await execute_agent_run(run_id, session_id, request)
    return result


async def stream_agent_run(run_id: str, session_id: str, request: AgentRunRequest):
    """Generator that yields SSE events during agent execution."""
    yield f"event: run.started\ndata: {json.dumps({'run_id': run_id, 'session_id': session_id})}\n\n"
    
    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=request.message)]},
        config={"configurable": {"thread_id": session_id}},
    ):
        event_type = event.get("event")
        event_data = event.get("data", {})
        
        if event_type == "on_chat_model_stream":
            token = event_data.get("chunk", {}).get("content", "")
            if token:
                yield f"event: message.delta\ndata: {json.dumps({'token': token})}\n\n"
        
        elif event_type == "on_tool_start":
            yield f"event: step.tool_call\ndata: {json.dumps({'tool': event_data.get('name')})}\n\n"
    
    yield f"event: run.completed\ndata: {json.dumps({'run_id': run_id})}\n\n"
```

## When NOT to Use This API Design

- **Internal tool with single user**: A simple function call is easier than HTTP APIs.
- **Batch processing**: Use a batch API (submit list, get results later) instead of per-request API.
- **WebSocket-only apps**: If you're already using WebSocket, send events through that instead of SSE.

## Tradeoffs

| Pattern | Latency | Complexity | Client Compatibility |
|---------|---------|-----------|---------------------|
| Synchronous POST | Highest (wait for completion) | Lowest | Universal |
| SSE streaming | Lowest (real-time) | Medium | Most clients |
| WebSocket | Lowest (bidirectional) | High | Requires WS support |
| Polling | Variable (poll interval) | Low | Universal |

## Failure Modes

1. **SSE connection drop mid-stream**: Client loses connection during streaming. Mitigation: client-side reconnection with `Last-Event-ID`, server-side event replay from checkpoint.
2. **Timeout on synchronous endpoint**: Agent takes > 30s, gateway times out. Mitigation: use streaming or async pattern, increase gateway timeout for agent routes.
3. **Pagination token expiry**: Token expires between paginated history requests. Mitigation: cursor-based pagination with no expiry.

## Source(s) and Further Reading

- LangGraph API Reference: https://langchain-ai.github.io/langgraph/cloud/reference/api/api_ref/
- SSE Specification: https://html.spec.whatwg.org/multipage/server-sent-events.html
- OpenAI API Design (reference): https://platform.openai.com/docs/api-reference
- FastAPI Streaming: https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse
