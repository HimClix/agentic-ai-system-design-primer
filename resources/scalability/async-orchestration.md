# Async Orchestration
> Decouple scheduling from execution -- agents that take minutes to complete shouldn't hold HTTP connections open.

## What It Is

Async orchestration separates the act of *requesting* an agent run from *waiting for its completion*. Instead of blocking an HTTP request while the agent thinks, plans, executes tools, and generates a response, the system immediately returns a run ID and delivers results asynchronously via webhooks, SSE, or polling.

This is essential when agent sessions take longer than typical HTTP timeout windows (30-60 seconds) or when you need to handle bursts of requests that exceed your LLM rate limits.

## How It Works

### Sync vs Async Agent Execution

```
SYNCHRONOUS (problematic for agents):
Client ──POST /run──▶ Server ──LLM call──▶ Tool call──▶ LLM──▶ Response
         │                                                          │
         └──────────── blocks for 30-120 seconds ──────────────────┘

ASYNCHRONOUS (production pattern):
Client ──POST /run──▶ Server ──▶ {run_id: "abc", status: "queued"} ←── 200ms
                         │
                         ▼
                    Task Queue (SQS/Redis)
                         │
                         ▼
                    Worker picks up task
                    ├── LLM call (5s)
                    ├── Tool call (2s)
                    ├── LLM call (5s)
                    └── Done
                         │
                         ▼
                    Deliver result:
                    ├── Webhook callback
                    ├── SSE stream
                    └── Polling endpoint
```

### Task Queue Architecture

```
                    ┌─────────────────┐
                    │   API Server    │
                    │ POST /runs      │
                    │ GET /runs/{id}  │
                    └────────┬────────┘
                             │ enqueue
                             ▼
                    ┌─────────────────┐
                    │   Task Queue    │
                    │  (SQS / Redis   │
                    │   Streams /     │
                    │   BullMQ)       │
                    └────────┬────────┘
                             │ dequeue
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Worker 1 │  │ Worker 2 │  │ Worker N │
        │ (agent   │  │ (agent   │  │ (agent   │
        │  loop)   │  │  loop)   │  │  loop)   │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │              │              │
             └──────────────┼──────────────┘
                            │ publish results
                            ▼
                   ┌─────────────────┐
                   │  Result Store   │
                   │  (PostgreSQL)   │
                   └────────┬────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         Webhook        SSE Stream    Poll API
```

## Production Implementation

```python
import uuid
import json
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import redis.asyncio as redis
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
import asyncio


class RunStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class AgentRun:
    run_id: str
    tenant_id: str
    session_id: str
    status: RunStatus
    input_message: str
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    steps_completed: int = 0
    tokens_used: int = 0


# --- API Layer ---

app = FastAPI()
redis_client = redis.Redis()


@app.post("/api/v1/runs")
async def create_run(request: dict):
    """Submit an agent run. Returns immediately with run_id."""
    run_id = str(uuid.uuid4())
    
    run = AgentRun(
        run_id=run_id,
        tenant_id=request["tenant_id"],
        session_id=request.get("session_id", str(uuid.uuid4())),
        status=RunStatus.QUEUED,
        input_message=request["message"],
    )
    
    # Store run metadata
    await redis_client.setex(
        f"run:{run_id}", 3600, json.dumps(run.__dict__, default=str)
    )
    
    # Enqueue for processing
    await redis_client.lpush("queue:agent_runs", json.dumps({
        "run_id": run_id,
        "tenant_id": run.tenant_id,
        "session_id": run.session_id,
        "message": run.input_message,
    }))
    
    return {
        "run_id": run_id,
        "status": "queued",
        "poll_url": f"/api/v1/runs/{run_id}",
        "stream_url": f"/api/v1/runs/{run_id}/stream",
    }


@app.get("/api/v1/runs/{run_id}")
async def get_run(run_id: str):
    """Poll for run status and result."""
    data = await redis_client.get(f"run:{run_id}")
    if not data:
        return {"error": "Run not found"}, 404
    return json.loads(data)


@app.get("/api/v1/runs/{run_id}/stream")
async def stream_run(run_id: str):
    """SSE stream for real-time updates."""
    async def event_generator():
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(f"stream:{run_id}")
        
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    yield f"data: {json.dumps(data)}\n\n"
                    
                    if data.get("status") in ["completed", "failed", "timeout"]:
                        break
        finally:
            await pubsub.unsubscribe(f"stream:{run_id}")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# --- Worker Layer ---

class AgentWorker:
    """Worker that processes agent runs from the queue."""
    
    def __init__(self, redis_client: redis.Redis, max_run_time: int = 300):
        self.redis = redis_client
        self.max_run_time = max_run_time
    
    async def run_forever(self):
        """Main worker loop -- pull tasks from queue and execute."""
        while True:
            # Blocking pop from queue (waits for work)
            task_data = await self.redis.brpop("queue:agent_runs", timeout=30)
            if not task_data:
                continue
            
            _, task_json = task_data
            task = json.loads(task_json)
            
            try:
                await self._execute_run(task)
            except Exception as e:
                await self._mark_failed(task["run_id"], str(e))
    
    async def _execute_run(self, task: dict):
        """Execute a single agent run with timeout."""
        run_id = task["run_id"]
        
        # Update status to running
        await self._update_status(run_id, RunStatus.RUNNING)
        await self._publish_event(run_id, {"status": "running", "step": 0})
        
        try:
            result = await asyncio.wait_for(
                self._run_agent(task),
                timeout=self.max_run_time,
            )
            
            await self._update_status(run_id, RunStatus.COMPLETED, result=result)
            await self._publish_event(run_id, {
                "status": "completed",
                "result": result,
            })
            
            # Deliver via webhook if configured
            await self._deliver_webhook(task.get("webhook_url"), run_id, result)
            
        except asyncio.TimeoutError:
            await self._update_status(run_id, RunStatus.TIMEOUT)
            await self._publish_event(run_id, {"status": "timeout"})
    
    async def _run_agent(self, task: dict) -> str:
        """Run the actual agent logic (LangGraph invocation)."""
        config = {"configurable": {"thread_id": task["session_id"]}}
        
        step = 0
        async for event in self.graph.astream(
            {"messages": [{"role": "user", "content": task["message"]}]},
            config=config,
        ):
            step += 1
            # Stream intermediate steps
            await self._publish_event(task["run_id"], {
                "status": "running",
                "step": step,
                "event_type": list(event.keys())[0] if event else "unknown",
            })
        
        # Extract final response
        state = await self.graph.aget_state(config)
        messages = state.values.get("messages", [])
        return messages[-1].content if messages else ""
    
    async def _publish_event(self, run_id: str, data: dict):
        """Publish SSE event via Redis Pub/Sub."""
        await self.redis.publish(f"stream:{run_id}", json.dumps(data))
    
    async def _update_status(self, run_id: str, status: RunStatus, result: str = None):
        """Update run status in Redis."""
        run_data = json.loads(await self.redis.get(f"run:{run_id}"))
        run_data["status"] = status.value
        if result:
            run_data["result"] = result
        run_data["completed_at"] = time.time()
        await self.redis.setex(f"run:{run_id}", 3600, json.dumps(run_data))
    
    async def _deliver_webhook(self, url: str, run_id: str, result: str):
        """Deliver result via webhook callback."""
        if not url:
            return
        import httpx
        async with httpx.AsyncClient() as client:
            await client.post(url, json={
                "run_id": run_id,
                "status": "completed",
                "result": result,
            }, timeout=10)
```

### Worker Auto-Scaling

```yaml
# Kubernetes KEDA ScaledObject for queue-based auto-scaling
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: agent-worker-scaler
spec:
  scaleTargetRef:
    name: agent-workers
  minReplicaCount: 2
  maxReplicaCount: 30
  pollingInterval: 15
  cooldownPeriod: 120
  triggers:
  - type: redis
    metadata:
      address: redis:6379
      listName: queue:agent_runs
      listLength: "5"  # Scale up when > 5 pending tasks per worker
```

## Decision Tree: Result Delivery Method

```
How does the client want results?
│
├── Real-time UI (chatbot)?
│   └── SSE streaming
│       ├── Client opens EventSource connection
│       ├── Server streams intermediate steps + final result
│       └── Connection auto-closes on completion
│
├── Backend integration (API to API)?
│   └── Webhook callback
│       ├── Client provides callback URL
│       ├── Server POSTs result when done
│       └── Include HMAC signature for verification
│
├── Mobile app / unreliable connection?
│   └── Polling
│       ├── Client polls GET /runs/{id} every 2-5 seconds
│       ├── Server returns current status + partial results
│       └── Simple, works with any client
│
└── Batch processing?
    └── Queue + notification
        ├── Results written to S3/database
        ├── Notification via SNS/email when batch completes
        └── No real-time requirement
```

## When NOT to Use Async Orchestration

- **Simple single-turn agents**: If the agent always responds in < 5 seconds, sync is simpler.
- **Interactive debugging**: When developing/testing, sync responses are easier to debug.
- **Very low volume**: < 10 requests/minute doesn't justify queue infrastructure.

## Tradeoffs

| Approach | Latency | Complexity | Reliability | Cost |
|----------|---------|-----------|-------------|------|
| Synchronous | Low (but timeout risk) | Low | Medium | Low |
| Async + polling | Medium (poll interval) | Medium | High | Medium |
| Async + SSE | Low (real-time) | Medium-High | High | Medium |
| Async + webhook | Variable (callback delay) | Medium | High (with retry) | Medium |

## Failure Modes

1. **Lost tasks**: Worker crashes after dequeuing but before completing. Mitigation: visibility timeout + redelivery (SQS), or Redis RPOPLPUSH with processing list.
2. **Zombie runs**: Run marked as "running" but worker died. Mitigation: heartbeat mechanism, timeout reaper job.
3. **SSE connection memory leak**: Thousands of open SSE connections consuming server memory. Mitigation: connection limits per user, idle timeout.
4. **Webhook delivery failure**: Target server is down when result is ready. Mitigation: retry with exponential backoff, dead letter queue.

## Source(s) and Further Reading

- LangGraph Async Streaming: https://langchain-ai.github.io/langgraph/how-tos/streaming/
- AWS SQS: https://aws.amazon.com/sqs/
- Redis Streams: https://redis.io/docs/data-types/streams/
- KEDA Autoscaler: https://keda.sh/
- Server-Sent Events Spec: https://html.spec.whatwg.org/multipage/server-sent-events.html
