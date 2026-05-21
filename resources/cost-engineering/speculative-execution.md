# Speculative Execution for Agents

> Start multiple reasoning paths simultaneously, use the first one that succeeds. Trade compute cost for latency.

## What It Is

Speculative execution in agent systems means launching multiple LLM calls (or full reasoning paths) in parallel and using whichever finishes first with acceptable quality. This is borrowed from CPU architecture, where processors execute both branches of an `if` statement speculatively and discard the wrong one.

In agent contexts, this manifests as:

1. **Model racing**: Send the same prompt to fast + slow models simultaneously. Use the fast model's answer if good enough; fall back to the slow model's answer otherwise.
2. **Path racing**: Start multiple reasoning approaches in parallel (e.g., direct answer vs. tool-assisted answer). Use whichever converges first.
3. **Speculative decoding**: Generate draft tokens with a small model, verify with a large model (inference-level optimization).

## How It Works

### Model Racing Architecture

```
                     User Query
                         │
              ┌──────────┴──────────┐
              │                     │
     ┌────────▼────────┐  ┌────────▼────────┐
     │  Fast Model      │  │  Slow Model      │
     │  (Haiku, 300ms)  │  │  (Sonnet, 1.5s)  │
     └────────┬────────┘  └────────┬────────┘
              │                     │
     ┌────────▼────────┐           │
     │  Quality Check   │           │
     │  (confidence,    │           │
     │   complexity)    │           │
     └───┬──────────┬──┘           │
         │          │               │
      PASS        FAIL              │
         │          │               │
    Use Haiku   ┌───▼───────────────▼──┐
    (300ms,     │  Wait for Sonnet      │
     $0.001)    │  (1.5s total, $0.01)  │
                └──────────────────────┘
```

### Real Numbers: Latency vs Cost

```
Scenario: Customer support agent, 10,000 queries/day

Strategy 1: Always use Sonnet
  Latency: 1,500ms (p50), 3,000ms (p95)
  Cost: $0.015/query × 10K = $150/day

Strategy 2: Always use Haiku
  Latency: 300ms (p50), 600ms (p95)
  Cost: $0.001/query × 10K = $10/day
  Quality: 70% acceptable (30% need escalation)

Strategy 3: Speculative (Haiku + Sonnet in parallel)
  Latency: 350ms (p50, Haiku wins 70%), 1,600ms (p50, Sonnet fallback 30%)
  Effective p50: ~700ms (weighted average)
  Cost: $0.001 (Haiku always) + $0.015 × 30% (Sonnet when needed) = $0.0055/query
  Total: $55/day
  Quality: Same as Sonnet (100% acceptable)

  Savings vs always-Sonnet: 63% cost reduction, 53% latency reduction
  Extra cost vs always-Haiku: 5.5× more, but 100% quality instead of 70%
```

### Path Racing Architecture

```
                     User Query
                         │
              ┌──────────┼──────────┐
              │          │          │
     ┌────────▼──┐  ┌───▼────┐  ┌──▼────────┐
     │ Direct    │  │ RAG    │  │ Tool-     │
     │ Answer    │  │ Path   │  │ Assisted  │
     │ (1 call)  │  │ (2     │  │ (3+       │
     │           │  │  calls)│  │  calls)   │
     └────┬──────┘  └───┬───┘  └──┬────────┘
          │              │         │
     ┌────▼──────────────▼─────────▼────┐
     │        First Acceptable Result    │
     │        (cancel remaining paths)   │
     └──────────────────────────────────┘
```

## Production Implementation

```python
import asyncio
import time
from dataclasses import dataclass
from typing import Optional
from anthropic import AsyncAnthropic


@dataclass
class SpeculativeResult:
    content: str
    model: str
    latency_ms: float
    cost_usd: float
    quality_score: float
    was_speculative: bool  # True if the fast path was used


class SpeculativeExecutor:
    """
    Execute LLM calls speculatively across multiple models.
    
    Strategy: Launch fast + slow models simultaneously.
    If fast model passes quality threshold, cancel slow model.
    Otherwise, wait for slow model.
    """

    # Pricing per million tokens (as of 2025)
    PRICING = {
        "claude-haiku-3.5": {"input": 0.80, "output": 4.00},
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    }

    def __init__(
        self,
        fast_model: str = "claude-haiku-3.5",
        slow_model: str = "claude-sonnet-4-20250514",
        quality_threshold: float = 0.8,
    ):
        self.client = AsyncAnthropic()
        self.fast_model = fast_model
        self.slow_model = slow_model
        self.quality_threshold = quality_threshold

    async def _call_model(
        self,
        model: str,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
    ) -> tuple[str, float]:
        """Call a model and return (response, latency_ms)."""
        start = time.monotonic()
        response = await self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        latency = (time.monotonic() - start) * 1000
        return response.content[0].text, latency

    async def _check_quality(self, response: str, query: str) -> float:
        """
        Quick quality check using the fast model.
        Returns score 0.0 - 1.0.
        
        Heuristics (fast, no LLM call):
        - Response length: too short = low quality
        - Contains hedging language ("I'm not sure") = low confidence
        - Contains specific data/numbers = higher quality
        
        For more accuracy, use LLM-as-judge (adds ~200ms).
        """
        score = 0.5  # Base score

        # Length heuristic
        word_count = len(response.split())
        if word_count < 10:
            score -= 0.3  # Too short
        elif word_count > 30:
            score += 0.1

        # Confidence heuristic
        hedging_phrases = [
            "i'm not sure", "i don't know", "i cannot", "i'm unable",
            "might be", "possibly", "it's unclear",
        ]
        response_lower = response.lower()
        if any(phrase in response_lower for phrase in hedging_phrases):
            score -= 0.2

        # Specificity heuristic
        if any(char.isdigit() for char in response):
            score += 0.1  # Contains specific numbers

        # Code block heuristic (good for technical queries)
        if "```" in response:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD."""
        pricing = self.PRICING.get(model, self.PRICING["claude-sonnet-4-20250514"])
        return (
            input_tokens / 1_000_000 * pricing["input"]
            + output_tokens / 1_000_000 * pricing["output"]
        )

    async def execute(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
    ) -> SpeculativeResult:
        """
        Speculative execution: race fast model against slow model.
        
        1. Launch both models simultaneously
        2. When fast model returns, check quality
        3. If quality >= threshold, cancel slow model and return fast result
        4. Otherwise, wait for slow model
        """
        # Create tasks
        fast_task = asyncio.create_task(
            self._call_model(self.fast_model, system, messages, max_tokens)
        )
        slow_task = asyncio.create_task(
            self._call_model(self.slow_model, system, messages, max_tokens)
        )

        # Wait for fast model first
        fast_response, fast_latency = await fast_task
        fast_quality = await self._check_quality(
            fast_response, messages[-1]["content"]
        )

        if fast_quality >= self.quality_threshold:
            # Fast model is good enough -- cancel slow model
            slow_task.cancel()
            try:
                await slow_task
            except asyncio.CancelledError:
                pass

            return SpeculativeResult(
                content=fast_response,
                model=self.fast_model,
                latency_ms=fast_latency,
                cost_usd=self._estimate_cost(self.fast_model, 500, 200),
                quality_score=fast_quality,
                was_speculative=True,
            )

        # Fast model not good enough -- wait for slow model
        slow_response, slow_latency = await slow_task

        # Cost: we paid for BOTH models (fast was wasted)
        total_cost = (
            self._estimate_cost(self.fast_model, 500, 200)
            + self._estimate_cost(self.slow_model, 500, 200)
        )

        return SpeculativeResult(
            content=slow_response,
            model=self.slow_model,
            latency_ms=slow_latency,
            cost_usd=total_cost,
            quality_score=1.0,  # Assume slow model is always good enough
            was_speculative=False,
        )


# --- Multi-Path Speculative Execution ---

class PathRacer:
    """
    Race multiple reasoning paths. Return the first acceptable result.
    
    Use case: Some queries can be answered directly, some need RAG,
    some need tool calls. Race all three.
    """

    def __init__(self):
        self.client = AsyncAnthropic()

    async def _direct_answer(self, query: str) -> Optional[str]:
        """Path 1: Try to answer directly (fastest)."""
        response = await self.client.messages.create(
            model="claude-haiku-3.5",
            max_tokens=512,
            system=(
                "Answer the question directly. If you are not confident "
                "in your answer, respond with exactly 'UNCERTAIN'."
            ),
            messages=[{"role": "user", "content": query}],
        )
        result = response.content[0].text
        if "UNCERTAIN" in result:
            return None
        return result

    async def _rag_answer(self, query: str) -> Optional[str]:
        """Path 2: Answer using RAG (medium speed)."""
        # Simulate retrieval
        chunks = await retrieve_relevant_chunks(query)
        if not chunks:
            return None

        context = "\n".join(chunks)
        response = await self.client.messages.create(
            model="claude-haiku-3.5",
            max_tokens=512,
            system=f"Answer using ONLY this context:\n{context}",
            messages=[{"role": "user", "content": query}],
        )
        return response.content[0].text

    async def _tool_answer(self, query: str) -> Optional[str]:
        """Path 3: Answer using tools (slowest but most reliable)."""
        # Full ReAct loop with tools
        return await run_react_agent(query)

    async def race(self, query: str) -> dict:
        """Race all paths, return first acceptable result."""
        tasks = {
            "direct": asyncio.create_task(self._direct_answer(query)),
            "rag": asyncio.create_task(self._rag_answer(query)),
            "tool": asyncio.create_task(self._tool_answer(query)),
        }

        # Wait for tasks as they complete
        for coro in asyncio.as_completed(tasks.values()):
            result = await coro

            if result is not None:
                # Cancel remaining tasks
                for name, task in tasks.items():
                    if not task.done():
                        task.cancel()

                # Find which path won
                winning_path = next(
                    name for name, task in tasks.items()
                    if task.done() and not task.cancelled() and task.result() == result
                )

                return {
                    "answer": result,
                    "path": winning_path,
                }

        raise RuntimeError("All paths failed")
```

## Decision Tree: When to Use Speculative Execution

```
    Should I use speculative execution?
                    │
         ┌──────────▼──────────┐
         │ Is latency your      │
         │ primary concern      │
         │ (SLA < 1 second)?    │
         └───┬────────────┬────┘
            Yes           No ──► Use simple model routing
             │                    (sequential, not parallel)
         ┌───▼────────────────┐
         │ Can you afford      │
         │ 1.5-2x token cost  │
         │ per query?          │
         └───┬────────────┬───┘
            Yes           No ──► Use fast model with
             │                    async escalation
         ┌───▼────────────────┐
         │ Is the fast model   │
         │ adequate for >50%   │
         │ of queries?         │
         └───┬────────────┬───┘
            Yes           No ──► Just use the slow model
             │                    (speculation wastes money)
         ┌───▼──────────────┐
         │ USE SPECULATIVE   │
         │ EXECUTION         │
         └──────────────────┘
```

## When NOT to Use

1. **Cost-constrained environments**: Speculative execution always pays for the fast model. If the fast model fails quality checks >50% of the time, you are paying for both models on most queries.
2. **Batch processing**: No latency requirement means no benefit from speculation. Just use the best model sequentially.
3. **When fast model success rate is < 50%**: If you need the slow model most of the time anyway, speculation just adds the fast model cost on top.
4. **Streaming responses**: You cannot speculate when streaming -- the user sees the first token immediately. Use model routing instead.
5. **Tasks requiring the large model's capabilities**: If the task inherently requires Opus/Sonnet (complex reasoning, long code generation), Haiku will almost always fail quality checks.

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Dramatic latency reduction (50-70% on average) | Always pays for the fast model (even when unused) |
| Quality matches the slow model (fallback) | Complex implementation (parallel tasks, cancellation) |
| Adaptive -- fast path used only when sufficient | Quality check adds latency to the fast path |
| Works across providers (cross-provider racing) | Debugging is harder (which path ran?) |
| Predictable worst-case latency (= slow model) | Wasted compute on failed speculations |

## Real-World Numbers

| Metric | Always Sonnet | Always Haiku | Speculative (Haiku→Sonnet) |
|--------|-------------|-------------|---------------------------|
| p50 latency | 1,500ms | 300ms | 400ms |
| p95 latency | 3,000ms | 600ms | 1,800ms |
| Cost/query | $0.015 | $0.001 | $0.005-0.008 |
| Quality (support tasks) | 95% | 70% | 95% |
| Monthly cost @ 10K/day | $4,500 | $300 | $1,500-2,400 |

## Failure Modes

### 1. Quality Check Too Lenient
The fast model's response passes quality checks but is actually wrong.
**Mitigation**: Include domain-specific quality checks (not just heuristics). Sample and audit fast-path responses. Track speculative accuracy separately.

### 2. Quality Check Too Strict
The fast model's response is always rejected, making speculation pure waste.
**Mitigation**: Monitor fast-path acceptance rate. If < 40%, switch to sequential model routing.

### 3. Resource Exhaustion
Under high load, speculative execution doubles the concurrent LLM calls, potentially hitting rate limits.
**Mitigation**: Disable speculation when rate limit utilization > 70%. Implement adaptive speculation.

### 4. Cost Overshoot
During traffic spikes, speculation costs more than expected because both models run for every query.
**Mitigation**: Set cost budgets with automatic speculation disabling. Track speculation cost separately.

### 5. Cancellation Race Condition
The slow model finishes just before cancellation, and you pay for both.
**Mitigation**: This is inherent. Budget for worst-case (both models always run).

## Sources and Further Reading

- [Speculative Decoding (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192) -- Token-level speculative execution
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents)
- [Model Routing - Martian](https://withmartian.com/) -- Production model routing with speculation
- [LiteLLM Model Fallbacks](https://docs.litellm.ai/docs/routing) -- Fallback routing implementation
- [Speculative RAG (2024)](https://arxiv.org/abs/2407.08223) -- Applying speculation to RAG pipelines
