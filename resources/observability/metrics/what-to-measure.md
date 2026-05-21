# What to Measure: Agent Metrics That Matter

> Measure P95 latency, not average -- an average of 200ms hides the fact that 1 in 20 users waits 8 seconds.

## What It Is

Agent metrics are the quantitative signals that tell you whether your agent is fast enough, cheap enough, accurate enough, and safe enough for production. Unlike traditional service metrics (throughput, error rate), agent metrics must capture **reasoning quality, token economics, and behavioral correctness**.

## How It Works

### The 7 Essential Agent Metrics

| Metric | What It Measures | Target | Alert Threshold | Why P95, Not Average |
|--------|-----------------|--------|-----------------|---------------------|
| **P95 Latency** | End-to-end response time | <5s for interactive, <30s for background | P95 > 10s | Average hides tail latency |
| **Token Usage / Request** | Cost efficiency | Varies by model | >2x baseline | Sudden spikes = prompt injection or loop |
| **Tool Success Rate** | Per-tool reliability | >99% per tool | <95% for any tool | Failing tools cause wrong agent decisions |
| **Finish Reason Distribution** | Why model stopped generating | >95% "stop" | >5% "length" or "content_filter" | "length" = truncated answers |
| **Cost / Request** | Dollar cost per agent run | Varies | >3x baseline | Token explosion = runaway loop |
| **Hallucination Rate** (sampled) | Factual accuracy | <5% | >10% | Requires LLM-as-judge sampling |
| **Human Escalation Rate** | How often agent cannot resolve | <20% | >30% | Rising = agent quality degrading |

### Detailed Metric Specifications

#### 1. P95 Latency

```python
"""Track P95 latency, not average."""
import time
from collections import deque
import statistics

class LatencyTracker:
    def __init__(self, window_size: int = 1000):
        self.latencies = deque(maxlen=window_size)
    
    def record(self, latency_ms: float):
        self.latencies.append(latency_ms)
    
    @property
    def p50(self) -> float:
        return self._percentile(50)
    
    @property
    def p95(self) -> float:
        return self._percentile(95)
    
    @property
    def p99(self) -> float:
        return self._percentile(99)
    
    def _percentile(self, pct: int) -> float:
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * pct / 100)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

# Alert thresholds:
# P95 > 5s for interactive agents (chatbots, copilots)
# P95 > 30s for background agents (data processing, analysis)
# P99 > 15s for interactive agents
```

#### 2. Token Usage Per Request

```python
"""Track token usage per request to detect anomalies."""

class TokenTracker:
    def __init__(self):
        self.baseline_input = 0    # Establish from first 100 requests
        self.baseline_output = 0
        self.samples = []
    
    def record(self, input_tokens: int, output_tokens: int):
        total = input_tokens + output_tokens
        self.samples.append(total)
        
        # Update baseline from first 100 samples
        if len(self.samples) == 100:
            self.baseline_input = statistics.median(self.samples)
    
    def is_anomalous(self, input_tokens: int, output_tokens: int) -> bool:
        total = input_tokens + output_tokens
        if self.baseline_input > 0:
            return total > self.baseline_input * 2  # Alert at 2x
        return False

# Alert thresholds:
# Total tokens > 2x baseline median = WARN
# Total tokens > 5x baseline median = CRITICAL (likely injection or loop)
# Input tokens > 100K in single request = CRITICAL
```

#### 3. Tool Success Rate (Per Tool)

```python
"""Per-tool success rate tracking."""

class ToolMetrics:
    def __init__(self):
        self.tool_stats: dict[str, dict] = {}
    
    def record(self, tool_name: str, success: bool, latency_ms: float):
        if tool_name not in self.tool_stats:
            self.tool_stats[tool_name] = {
                "total": 0, "success": 0, "latencies": deque(maxlen=100)
            }
        
        stats = self.tool_stats[tool_name]
        stats["total"] += 1
        if success:
            stats["success"] += 1
        stats["latencies"].append(latency_ms)
    
    def success_rate(self, tool_name: str) -> float:
        stats = self.tool_stats.get(tool_name, {"total": 0, "success": 0})
        return stats["success"] / stats["total"] if stats["total"] > 0 else 1.0
    
    def p95_latency(self, tool_name: str) -> float:
        stats = self.tool_stats.get(tool_name)
        if not stats or not stats["latencies"]:
            return 0
        sorted_lat = sorted(stats["latencies"])
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

# Alert thresholds per tool:
# Success rate < 99% = WARN
# Success rate < 95% = CRITICAL
# P95 latency > 2x baseline = WARN
# Any tool at 0% success = CRITICAL (tool is down)
```

#### 4. Finish Reason Distribution

```python
"""Track why the model stopped generating."""

class FinishReasonTracker:
    def __init__(self):
        self.counts: dict[str, int] = {}
        self.total = 0
    
    def record(self, finish_reason: str):
        self.counts[finish_reason] = self.counts.get(finish_reason, 0) + 1
        self.total += 1
    
    def distribution(self) -> dict[str, float]:
        return {k: v / self.total for k, v in self.counts.items()} if self.total > 0 else {}

# Alert thresholds:
# "stop" < 90% = WARN (model is being cut off too often)
# "length" > 5% = WARN (max_tokens too low, truncated responses)
# "content_filter" > 1% = CRITICAL (safety filter triggering frequently)
# "length" > 15% = CRITICAL (agent producing overly long responses)
```

#### 5. Cost Per Request

```python
"""Track dollar cost per agent run."""

MODEL_PRICING = {
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "claude-sonnet-4-20250514": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
}

class CostTracker:
    def __init__(self):
        self.costs = deque(maxlen=10000)
    
    def record(self, model: str, input_tokens: int, output_tokens: int):
        pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
        cost = input_tokens * pricing["input"] + output_tokens * pricing["output"]
        self.costs.append(cost)
        return cost
    
    @property
    def avg_cost(self) -> float:
        return sum(self.costs) / len(self.costs) if self.costs else 0
    
    @property
    def p95_cost(self) -> float:
        if not self.costs:
            return 0
        sorted_costs = sorted(self.costs)
        idx = int(len(sorted_costs) * 0.95)
        return sorted_costs[min(idx, len(sorted_costs) - 1)]

# Alert thresholds:
# P95 cost > 3x baseline = WARN
# Single request cost > $1 = WARN (for most agents)
# Daily cost > 2x projected = CRITICAL
```

#### 6. Hallucination Rate (Sampled)

```python
"""Sample-based hallucination detection using LLM-as-judge."""

class HallucinationTracker:
    def __init__(self, sample_rate: float = 0.05):  # Check 5% of responses
        self.sample_rate = sample_rate
        self.checked = 0
        self.hallucinations = 0
    
    async def check(self, response: str, context: str, judge_llm) -> bool:
        """Sample and check for hallucinations."""
        import random
        if random.random() > self.sample_rate:
            return False  # Not sampled
        
        self.checked += 1
        
        judge_response = await judge_llm.ainvoke(
            f"""Does this response contain claims not supported by the provided context?
            
            Context: {context}
            Response: {response}
            
            Reply YES or NO only."""
        )
        
        is_hallucination = "yes" in judge_response.content.lower()
        if is_hallucination:
            self.hallucinations += 1
        
        return is_hallucination
    
    @property
    def rate(self) -> float:
        return self.hallucinations / self.checked if self.checked > 0 else 0

# Alert thresholds:
# Hallucination rate > 5% = WARN
# Hallucination rate > 10% = CRITICAL
# Note: sampling at 5% means checking 5 out of every 100 responses
```

#### 7. Human Escalation Rate

```python
"""Track how often the agent cannot resolve and escalates to humans."""

class EscalationTracker:
    def __init__(self):
        self.total_sessions = 0
        self.escalated = 0
        self.escalation_reasons: dict[str, int] = {}
    
    def record_session(self, escalated: bool, reason: str = ""):
        self.total_sessions += 1
        if escalated:
            self.escalated += 1
            self.escalation_reasons[reason] = self.escalation_reasons.get(reason, 0) + 1
    
    @property
    def rate(self) -> float:
        return self.escalated / self.total_sessions if self.total_sessions > 0 else 0

# Alert thresholds:
# Escalation rate > 20% = WARN (agent struggling with too many queries)
# Escalation rate > 30% = CRITICAL (agent may be broken)
# Sudden increase > 2x = CRITICAL (something changed)
```

## Decision Tree / When to Use

- **All 7 metrics**: Production agents with tool calling in customer-facing scenarios
- **Metrics 1-5**: Internal tools, non-customer-facing agents
- **Metrics 1-3 only**: Development and staging environments

## When NOT to Use

- Never skip latency and cost tracking in production
- Hallucination sampling can be deferred for purely internal tools

## Tradeoffs

| Metric | Collection Cost | Value | Overhead |
|--------|----------------|-------|----------|
| Latency | Free (timer) | Critical | None |
| Token usage | Free (API response) | Critical | None |
| Tool success | Free (try/catch) | High | None |
| Finish reason | Free (API response) | Medium | None |
| Cost per request | Free (calculation) | High | None |
| Hallucination rate | ~$0.01 per check | Very high | LLM call per sample |
| Escalation rate | Free (counter) | High | None |

## Failure Modes

| Failure | Cause | Impact | Prevention |
|---------|-------|--------|------------|
| **Alert fatigue** | Thresholds too low | Real alerts ignored | Baseline first, then set thresholds |
| **Missing baseline** | No historical data for comparison | Cannot detect anomalies | Run for 1 week before setting alerts |
| **Metric cardinality explosion** | Too many dimensions per metric | Storage/query cost | Limit label values |
| **Sampling bias** | Hallucination check only on short responses | Miss hallucinations in long responses | Stratified sampling |

## Source(s) and Further Reading

- Google SRE Book, Chapter 6: "Service Level Objectives"
- Langfuse Metrics Guide: https://langfuse.com/docs/analytics
- LangSmith Monitoring: https://docs.smith.langchain.com/monitoring
- EDDOps paper (arXiv:2411.13768): Eval-driven development for LLM systems
