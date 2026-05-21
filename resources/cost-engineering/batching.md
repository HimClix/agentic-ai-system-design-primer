# Batching Strategies for Agent Systems

> Batch multiple requests into fewer LLM calls to cut costs by 30-70%. Trade latency for throughput.

## What It Is

Batching in agent systems means collecting multiple independent requests and processing them together in fewer LLM calls. This reduces per-request overhead and, with some providers, unlocks significant discounts.

Three levels of batching:

1. **Request-level batching**: Combine multiple user queries into a single LLM prompt
2. **Embedding batching**: Batch embedding generation for multiple texts
3. **API-level batching**: Use provider batch APIs for async processing (OpenAI Batch API: 50% discount)

## How It Works

### Request-Level Batching

```
WITHOUT batching (5 separate LLM calls):
  Call 1: Classify("How do I reset my password?")     → support
  Call 2: Classify("I want to upgrade my plan")        → sales
  Call 3: Classify("API returning 500 errors")         → technical
  Call 4: Classify("Invoice question")                 → billing
  Call 5: Classify("Feature request: dark mode")       → product

  Total: 5 API calls, 5× round-trip latency, 5× overhead tokens (system prompt repeated)

WITH batching (1 LLM call):
  Call 1: Classify([
    "1. How do I reset my password?",
    "2. I want to upgrade my plan",
    "3. API returning 500 errors",
    "4. Invoice question",
    "5. Feature request: dark mode"
  ]) → [support, sales, technical, billing, product]

  Total: 1 API call, 1× round-trip, 1× overhead tokens
  Token savings: ~60% (system prompt counted once instead of 5 times)
```

### Cost Math

```
Classification task with 100-token system prompt, 50-token query, 10-token response:

WITHOUT batching (per query):
  Input: 100 (system) + 50 (query) = 150 tokens
  Output: 10 tokens
  Total for 10 queries: 1,500 input + 100 output = 1,600 tokens

WITH batching (10 queries in one call):
  Input: 100 (system) + 500 (10 queries) = 600 tokens
  Output: 100 tokens (10 classifications)
  Total: 600 input + 100 output = 700 tokens

  Savings: 56% fewer tokens

At Claude Sonnet pricing ($3/M input, $15/M output):
  Without: $0.0060/batch
  With:    $0.0033/batch
  Savings: $0.0027/batch → $81/month at 1,000 batches/day
```

### OpenAI Batch API (50% Discount)

```
Standard API:  $2.50/M input, $10.00/M output (GPT-4o)
Batch API:     $1.25/M input,  $5.00/M output (50% off)

Constraint: Results delivered within 24 hours (not real-time)

Use cases:
  - Nightly evaluation runs
  - Bulk classification/extraction
  - Content moderation backlog
  - Training data generation

NOT suitable for:
  - Real-time user-facing agents
  - Interactive conversations
  - Latency-sensitive tool calls
```

## Production Implementation

### Request-Level Batching with Python

```python
import asyncio
import time
from typing import TypeVar, Generic
from dataclasses import dataclass, field
from collections import defaultdict
from anthropic import Anthropic

T = TypeVar("T")


@dataclass
class BatchItem(Generic[T]):
    """A single item waiting to be batched."""
    input_data: str
    future: asyncio.Future
    submitted_at: float = field(default_factory=time.monotonic)


class LLMBatcher:
    """
    Collects individual LLM requests and batches them into single calls.
    
    Two trigger conditions:
    1. Batch size reached (e.g., 10 items)
    2. Max wait time exceeded (e.g., 200ms)
    
    Ensures no request waits longer than max_wait_ms.
    """

    def __init__(
        self,
        system_prompt: str,
        model: str = "claude-haiku-3.5",
        max_batch_size: int = 10,
        max_wait_ms: float = 200,
    ):
        self.client = Anthropic()
        self.system_prompt = system_prompt
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: list[BatchItem] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None

    async def submit(self, input_text: str) -> str:
        """Submit a single request. Returns when the batch is processed."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        item = BatchItem(input_data=input_text, future=future)

        async with self._lock:
            self._queue.append(item)

            if len(self._queue) >= self.max_batch_size:
                # Batch full -- flush immediately
                await self._flush()
            elif self._flush_task is None:
                # Start timer for max wait
                self._flush_task = asyncio.create_task(self._delayed_flush())

        return await future

    async def _delayed_flush(self):
        """Flush after max_wait_ms if batch hasn't filled."""
        await asyncio.sleep(self.max_wait_ms / 1000)
        async with self._lock:
            if self._queue:
                await self._flush()
            self._flush_task = None

    async def _flush(self):
        """Process all queued items in a single LLM call."""
        if not self._queue:
            return

        items = self._queue[:]
        self._queue.clear()
        if self._flush_task:
            self._flush_task.cancel()
            self._flush_task = None

        # Build batched prompt
        numbered_inputs = "\n".join(
            f"{i+1}. {item.input_data}" for i, item in enumerate(items)
        )
        batched_prompt = (
            f"Process each of the following {len(items)} items. "
            f"Respond with a numbered list matching the input numbers.\n\n"
            f"{numbered_inputs}"
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=self.system_prompt,
                messages=[{"role": "user", "content": batched_prompt}],
            )
            raw_output = response.content[0].text

            # Parse numbered results
            results = self._parse_numbered_results(raw_output, len(items))

            for item, result in zip(items, results):
                if not item.future.done():
                    item.future.set_result(result)

        except Exception as e:
            for item in items:
                if not item.future.done():
                    item.future.set_exception(e)

    def _parse_numbered_results(self, text: str, expected_count: int) -> list[str]:
        """Parse numbered results from batched LLM output."""
        lines = text.strip().split("\n")
        results = []
        current = []

        for line in lines:
            # Check if line starts with a number
            stripped = line.strip()
            if stripped and stripped[0].isdigit() and "." in stripped[:4]:
                if current:
                    results.append("\n".join(current).strip())
                    current = []
                # Remove the number prefix
                content = stripped.split(".", 1)[1].strip() if "." in stripped else stripped
                current.append(content)
            else:
                current.append(stripped)

        if current:
            results.append("\n".join(current).strip())

        # Pad or truncate to expected count
        while len(results) < expected_count:
            results.append("[PARSING_ERROR: missing result]")

        return results[:expected_count]


# --- Embedding Batching ---

class EmbeddingBatcher:
    """
    Batch embedding generation for maximum throughput.
    
    Most embedding APIs support batch inputs:
    - OpenAI: up to 2048 texts per call
    - Voyage: up to 128 texts per call
    - Cohere: up to 96 texts per call
    """

    def __init__(self, batch_size: int = 128):
        import voyageai
        self.client = voyageai.Client()
        self.batch_size = batch_size

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in batches."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            result = self.client.embed(
                batch,
                model="voyage-3",
                input_type="document",
            )
            all_embeddings.extend(result.embeddings)

        return all_embeddings

    def cost_comparison(self, num_texts: int) -> dict:
        """Compare batched vs individual embedding costs."""
        # Voyage-3: $0.06/M tokens
        avg_tokens_per_text = 200
        total_tokens = num_texts * avg_tokens_per_text

        # Individual calls have per-request overhead
        individual_api_calls = num_texts
        batched_api_calls = (num_texts + self.batch_size - 1) // self.batch_size

        return {
            "individual_api_calls": individual_api_calls,
            "batched_api_calls": batched_api_calls,
            "api_call_reduction": f"{(1 - batched_api_calls/individual_api_calls) * 100:.0f}%",
            "token_cost": f"${total_tokens / 1_000_000 * 0.06:.4f}",
            "note": "Token cost is the same; batching reduces API call overhead and latency",
        }


# --- OpenAI Batch API Integration ---

import json
from pathlib import Path


def create_openai_batch_file(
    requests: list[dict],
    output_path: str = "batch_input.jsonl",
) -> str:
    """
    Create a JSONL file for OpenAI's Batch API.
    
    Each request is a standard chat completion request.
    50% cost discount, results within 24 hours.
    """
    with open(output_path, "w") as f:
        for i, req in enumerate(requests):
            batch_request = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": req.get("model", "gpt-4o"),
                    "messages": req["messages"],
                    "max_tokens": req.get("max_tokens", 1024),
                },
            }
            f.write(json.dumps(batch_request) + "\n")

    return output_path


def submit_openai_batch(file_path: str) -> str:
    """Submit a batch job to OpenAI. Returns batch ID."""
    from openai import OpenAI
    client = OpenAI()

    # Upload the file
    batch_file = client.files.create(
        file=open(file_path, "rb"),
        purpose="batch",
    )

    # Create the batch
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    return batch.id


# --- Celery-Based Batching (Production) ---

CELERY_BATCH_EXAMPLE = """
# tasks.py
from celery import Celery
from celery.contrib.batches import Batches

app = Celery('tasks', broker='redis://localhost:6379/0')

@app.task(base=Batches, flush_every=10, flush_interval=0.5)
def classify_batch(requests):
    \"\"\"
    Celery collects up to 10 requests or waits 500ms, 
    then calls this function with all collected requests.
    \"\"\"
    texts = [req.args[0] for req in requests]
    
    # Single LLM call for all texts
    results = batch_classify(texts)
    
    # Return results to individual callers
    for req, result in zip(requests, results):
        req.return_value = result
"""
```

## Decision Tree: When to Batch

```
    Should I batch these LLM calls?
                    │
         ┌──────────▼──────────┐
         │ Are the requests    │
         │ real-time (user     │
         │ waiting)?           │
         └───┬────────────┬───┘
            Yes           No
             │             │
     ┌───────▼──────┐  ┌──▼──────────────┐
     │ Is 200ms      │  │ Use OpenAI Batch │
     │ extra latency │  │ API (50% off)    │
     │ acceptable?   │  │ or batch with    │
     └──┬────────┬──┘  │ Celery/SQS       │
       Yes      No     └─────────────────┘
        │        │
   Micro-batch  Don't batch
   (10 items,   (process
    200ms max)   individually)
        │
  ┌─────▼──────────────┐
  │ Are the requests    │
  │ homogeneous (same   │
  │ task type)?         │
  └──┬────────────┬────┘
    Yes           No ──► Don't batch
     │                    (mixed tasks are hard
     │                     to batch into one prompt)
  ┌──▼───────────┐
  │  BATCH IT    │
  └──────────────┘
```

## When NOT to Use

1. **Real-time interactive agents**: Chatbots cannot wait 200ms to collect batch items. Each user expects immediate response.
2. **Heterogeneous tasks**: If each request requires different system prompts or tools, batching provides minimal benefit.
3. **Low volume**: < 10 requests/second means batches rarely fill, and you are just adding latency.
4. **Streaming responses**: Streaming is per-request by nature and cannot be batched.
5. **Tasks requiring conversation context**: Multi-turn conversations cannot be batched across users.

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| 30-60% token cost reduction (shared system prompt) | Added latency (batch wait time) |
| 50% discount with OpenAI Batch API | Parsing batched output is error-prone |
| Fewer API calls (reduced rate limit pressure) | One failure can affect all items in batch |
| Higher throughput (more requests per second) | Only works for homogeneous request types |
| Reduces per-request overhead | Complexity in request-result matching |

## Batching Cost Comparison Table

| Method | Cost Reduction | Latency Impact | Best For |
|--------|---------------|----------------|----------|
| Micro-batching (10 items, 200ms) | 30-40% | +200ms | Real-time classification, routing |
| Macro-batching (100 items, 5min) | 50-60% | +5 minutes | Bulk extraction, content moderation |
| OpenAI Batch API | 50% flat | +up to 24 hours | Evaluations, training data generation |
| Embedding batching | API calls only | Minimal | Ingestion pipelines, RAG indexing |
| SQS/Celery batching | 40-50% | +0.5-5 seconds | Background processing, async tasks |

## Failure Modes

### 1. Batch Parsing Errors
The LLM does not follow the numbered output format, causing results to map to wrong requests.
**Mitigation**: Use structured output (JSON mode). Validate output count matches input count. Fall back to individual processing on parse failure.

### 2. Partial Batch Failures
The LLM processes items 1-7 correctly but item 8 causes a malformed response.
**Mitigation**: Detect missing/malformed items and re-process only the failed items individually.

### 3. Latency Spikes
Batch size is set to 50 but items arrive slowly, so items wait up to the max wait time every time.
**Mitigation**: Adaptive batch sizing -- reduce batch size during low-traffic periods.

### 4. Queue Overflow
During traffic spikes, the batch queue grows unboundedly, consuming memory.
**Mitigation**: Set a max queue size. Reject new requests (with backpressure) when queue is full.

### 5. Batch Poison Pill
One malformed input in the batch causes the entire LLM call to fail.
**Mitigation**: Input validation before queuing. Sanitize all inputs.

## Sources and Further Reading

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch)
- [Celery Batches](https://docs.celeryq.dev/en/stable/userguide/tasks.html)
- [LiteLLM Batching](https://docs.litellm.ai/docs/batching)
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents)
- [Prompt Caching - Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) -- Complementary cost optimization
