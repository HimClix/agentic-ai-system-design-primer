# Prometheus + Grafana for AI Agent Infrastructure Monitoring
> Langfuse/LangSmith give you agent traces. Prometheus + Grafana give you infrastructure metrics. You need both: one tells you why the agent gave a bad answer, the other tells you why the agent was slow.

## What It Is

Prometheus + Grafana provides the infrastructure observability layer for AI agent systems -- scraping metrics from LLM inference servers (vLLM, TGI), application-level agent instrumentation (token counts, tool call durations, step counts), and standard compute metrics (CPU, memory, GPU utilization). This is complementary to LLM-specific observability platforms (Langfuse, LangSmith) which handle agent traces, prompt debugging, and evaluation. In production, you run both.

### Observability Stack Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Grafana Dashboards                        │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│   │ Agent Ops    │  │ LLM Infra    │  │ Cost & Budget        │  │
│   │ Dashboard    │  │ Dashboard    │  │ Dashboard            │  │
│   └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│          │                 │                      │              │
│   ┌──────┴─────────────────┴──────────────────────┴──────────┐  │
│   │                    Grafana Query Layer                    │  │
│   │   Prometheus (metrics) │ Tempo (traces) │ Loki (logs)    │  │
│   └────────┬───────────────┼───────────────┬─────────────────┘  │
└────────────┼───────────────┼───────────────┼─────────────────────┘
             │               │               │
┌────────────▼──────┐ ┌──────▼──────┐ ┌──────▼──────────────┐
│   Prometheus      │ │   Tempo     │ │   Loki              │
│   (TSDB)          │ │  (Traces)   │ │  (Logs)             │
│                   │ │             │ │                     │
│  Scrape targets:  │ │  OTel spans │ │  Structured logs    │
│  - vLLM /metrics  │ │  from       │ │  from agent app     │
│  - TGI /metrics   │ │  OpenLLMetry│ │  + inference server │
│  - App /metrics   │ │             │ │                     │
│  - Node exporter  │ │             │ │                     │
│  - GPU exporter   │ │             │ │                     │
└───────────────────┘ └─────────────┘ └─────────────────────┘
```

### Where Prometheus+Grafana Fits vs. Langfuse/LangSmith

| Concern | Prometheus + Grafana | Langfuse / LangSmith |
|---------|---------------------|---------------------|
| Token usage over time | `llm_tokens_total` counter | Per-trace token breakdown |
| LLM latency percentiles | `histogram_quantile(0.99, llm_request_duration)` | Per-call latency in trace waterfall |
| GPU utilization | `nvidia_gpu_utilization` from DCGM exporter | Not available |
| Agent success rate | `agent_task_success_total / agent_task_total` | Eval scores per trace |
| Tool call errors | `tool_call_errors_total` by tool name | Tool call details in trace |
| Cost tracking | `llm_tokens_total * per_token_price` | Built-in cost tracking |
| Prompt debugging | Not applicable | Full prompt/response pairs |
| Alerting on SLO breach | Prometheus Alertmanager | Limited alerting |
| Queue depth / backpressure | `vllm_num_requests_waiting` | Not available |

**Rule of thumb**: Prometheus for "is it healthy and fast?" Langfuse for "is it correct and useful?"

## How It Works

### Metric Collection Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  vLLM Server    │     │  Agent App      │     │  GPU Node       │
│  :8000/metrics  │     │  :9090/metrics  │     │  :9400/metrics  │
│                 │     │                 │     │                 │
│ vllm_request_*  │     │ agent_*         │     │ nvidia_gpu_*    │
│ vllm_cache_*    │     │ llm_*           │     │ DCGM_FI_*       │
│ vllm_num_*      │     │ tool_*          │     │                 │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────┬───────────┴───────────┬───────────┘
                     │                       │
              ┌──────▼───────────────────────▼──────┐
              │          Prometheus Server           │
              │   scrape_interval: 15s               │
              │   retention: 30d                     │
              │   storage: 50GB                      │
              └──────────────────┬───────────────────┘
                                 │
                          ┌──────▼──────┐
                          │   Grafana   │
                          │  Dashboards │
                          │  + Alerts   │
                          └─────────────┘
```

## Production Implementation

### 1. Prometheus Scrape Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/agent_alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]

scrape_configs:
  # --- vLLM Inference Server Metrics ---
  - job_name: "vllm"
    metrics_path: /metrics
    scrape_interval: 10s          # More frequent for latency-sensitive metrics
    static_configs:
      - targets:
          - "vllm-server-0:8000"
          - "vllm-server-1:8000"
        labels:
          service: "llm-inference"
          model: "claude-sonnet-equivalent"

  # --- Text Generation Inference (TGI) Metrics ---
  - job_name: "tgi"
    metrics_path: /metrics
    scrape_interval: 10s
    static_configs:
      - targets:
          - "tgi-server-0:8080"
        labels:
          service: "llm-inference"
          model: "llama-3.1-70b"

  # --- Agent Application Metrics ---
  - job_name: "agent-app"
    metrics_path: /metrics
    scrape_interval: 15s
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: ["agent-platform"]
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__

  # --- NVIDIA GPU Metrics (DCGM Exporter) ---
  - job_name: "nvidia-gpu"
    metrics_path: /metrics
    scrape_interval: 15s
    static_configs:
      - targets:
          - "gpu-node-0:9400"
          - "gpu-node-1:9400"

  # --- Node Exporter (standard compute) ---
  - job_name: "node"
    static_configs:
      - targets:
          - "node-exporter:9100"
```

### 2. Custom Agent Application Metrics (Python Instrumentation)

```python
"""
Agent application metrics instrumentation using prometheus_client.
Expose via /metrics endpoint for Prometheus to scrape.
"""
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    start_http_server,
    REGISTRY,
)
import time
from functools import wraps

# ========================================================================
# METRIC DEFINITIONS
# ========================================================================

# --- Token Metrics ---
LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total LLM tokens consumed",
    ["model", "direction", "tenant_id"],
    # direction: "input" | "output"
)

LLM_TOKENS_COST_DOLLARS = Counter(
    "llm_tokens_cost_dollars_total",
    "Estimated cost in USD of LLM tokens consumed",
    ["model", "tenant_id"],
)

# --- LLM Request Metrics ---
LLM_REQUEST_DURATION_SECONDS = Histogram(
    "llm_request_duration_seconds",
    "LLM API request duration in seconds",
    ["model", "provider", "status"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

LLM_TIME_TO_FIRST_TOKEN_SECONDS = Histogram(
    "llm_time_to_first_token_seconds",
    "Time to first token from LLM in seconds",
    ["model", "provider"],
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
)

LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total LLM API requests",
    ["model", "provider", "status"],
    # status: "success" | "error" | "timeout" | "rate_limited"
)

# --- Agent Execution Metrics ---
AGENT_STEP_COUNT = Histogram(
    "agent_step_count",
    "Number of steps (LLM calls) per agent invocation",
    ["agent_type", "tenant_id"],
    buckets=[1, 2, 3, 5, 8, 10, 15, 20, 30, 50],
)

AGENT_TURN_DURATION_SECONDS = Histogram(
    "agent_turn_duration_seconds",
    "End-to-end duration of a single agent turn",
    ["agent_type", "tenant_id"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

AGENT_TASK_TOTAL = Counter(
    "agent_task_total",
    "Total agent tasks attempted",
    ["agent_type", "tenant_id", "outcome"],
    # outcome: "success" | "failure" | "timeout" | "human_escalation"
)

AGENT_ACTIVE_SESSIONS = Gauge(
    "agent_active_sessions",
    "Currently active agent sessions",
    ["agent_type"],
)

# --- Tool Call Metrics ---
TOOL_CALL_DURATION_SECONDS = Histogram(
    "tool_call_duration_seconds",
    "Duration of individual tool calls",
    ["tool_name", "status"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

TOOL_CALL_TOTAL = Counter(
    "tool_call_total",
    "Total tool calls made by agents",
    ["tool_name", "status"],
    # status: "success" | "error" | "timeout"
)

# --- Memory / RAG Metrics ---
RAG_RETRIEVAL_DURATION_SECONDS = Histogram(
    "rag_retrieval_duration_seconds",
    "Duration of RAG retrieval operations",
    ["retriever_type"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

RAG_DOCUMENTS_RETRIEVED = Histogram(
    "rag_documents_retrieved",
    "Number of documents retrieved per query",
    ["retriever_type"],
    buckets=[0, 1, 3, 5, 10, 20, 50],
)

# --- Checkpoint Metrics ---
CHECKPOINT_SAVE_DURATION_SECONDS = Histogram(
    "checkpoint_save_duration_seconds",
    "Duration of checkpoint save operations",
    [],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

CHECKPOINT_SIZE_BYTES = Histogram(
    "checkpoint_size_bytes",
    "Size of checkpoint data in bytes",
    [],
    buckets=[1024, 10240, 102400, 1048576, 10485760],  # 1KB to 10MB
)


# ========================================================================
# INSTRUMENTATION HELPERS
# ========================================================================

# Per-token pricing (as of mid-2025, approximate)
TOKEN_PRICES = {
    "claude-sonnet-4-20250514":    {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-haiku-4-20250514":     {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000},
    "gpt-4o":                      {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini":                 {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "meta-llama/Llama-3.1-70B":    {"input": 0.00, "output": 0.00},  # Self-hosted
}


def record_llm_usage(
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    duration_seconds: float,
    ttft_seconds: float | None,
    status: str,
    tenant_id: str = "default",
):
    """Record all metrics for a single LLM API call."""
    LLM_TOKENS_TOTAL.labels(
        model=model, direction="input", tenant_id=tenant_id,
    ).inc(input_tokens)
    LLM_TOKENS_TOTAL.labels(
        model=model, direction="output", tenant_id=tenant_id,
    ).inc(output_tokens)

    LLM_REQUEST_DURATION_SECONDS.labels(
        model=model, provider=provider, status=status,
    ).observe(duration_seconds)

    LLM_REQUESTS_TOTAL.labels(
        model=model, provider=provider, status=status,
    ).inc()

    if ttft_seconds is not None:
        LLM_TIME_TO_FIRST_TOKEN_SECONDS.labels(
            model=model, provider=provider,
        ).observe(ttft_seconds)

    # Calculate cost
    prices = TOKEN_PRICES.get(model, {"input": 0, "output": 0})
    cost = (input_tokens * prices["input"]) + (output_tokens * prices["output"])
    LLM_TOKENS_COST_DOLLARS.labels(
        model=model, tenant_id=tenant_id,
    ).inc(cost)


def instrument_tool_call(tool_name: str):
    """Decorator to instrument a tool function."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            status = "success"
            try:
                result = await func(*args, **kwargs)
                return result
            except TimeoutError:
                status = "timeout"
                raise
            except Exception:
                status = "error"
                raise
            finally:
                duration = time.time() - start
                TOOL_CALL_DURATION_SECONDS.labels(
                    tool_name=tool_name, status=status,
                ).observe(duration)
                TOOL_CALL_TOTAL.labels(
                    tool_name=tool_name, status=status,
                ).inc()
        return wrapper
    return decorator


# ========================================================================
# START METRICS SERVER
# ========================================================================

def start_metrics_endpoint(port: int = 9090):
    """Start Prometheus metrics HTTP server on /metrics."""
    start_http_server(port)
    print(f"Prometheus metrics available at http://0.0.0.0:{port}/metrics")
```

### 3. Prometheus Alerting Rules for Agent SLOs

```yaml
# rules/agent_alerts.yml
groups:
  - name: agent_slo_alerts
    interval: 30s
    rules:

      # --- LLM Latency SLO ---
      - alert: LLMLatencyP99High
        expr: |
          histogram_quantile(0.99,
            rate(llm_request_duration_seconds_bucket[5m])
          ) > 10
        for: 5m
        labels:
          severity: warning
          team: agent-platform
        annotations:
          summary: "LLM P99 latency > 10s for 5 minutes"
          description: |
            P99 latency is {{ $value | printf "%.1f" }}s.
            Check LLM provider status and queue depth.

      - alert: LLMLatencyP99Critical
        expr: |
          histogram_quantile(0.99,
            rate(llm_request_duration_seconds_bucket[5m])
          ) > 30
        for: 2m
        labels:
          severity: critical
          team: agent-platform
        annotations:
          summary: "LLM P99 latency > 30s for 2 minutes"

      # --- Time to First Token SLO ---
      - alert: TTFTHighLatency
        expr: |
          histogram_quantile(0.95,
            rate(llm_time_to_first_token_seconds_bucket[5m])
          ) > 2
        for: 5m
        labels:
          severity: warning
          team: agent-platform
        annotations:
          summary: "TTFT P95 > 2s -- users are waiting too long for first response"

      # --- LLM Error Rate ---
      - alert: LLMErrorRateHigh
        expr: |
          sum(rate(llm_requests_total{status=~"error|timeout"}[5m]))
          /
          sum(rate(llm_requests_total[5m]))
          > 0.05
        for: 3m
        labels:
          severity: critical
          team: agent-platform
        annotations:
          summary: "LLM error rate > 5% for 3 minutes"
          description: |
            Error rate is {{ $value | printf "%.1f" }}%.
            Check provider status. Circuit breaker may activate.

      # --- Agent Task Success Rate ---
      - alert: AgentSuccessRateLow
        expr: |
          sum(rate(agent_task_total{outcome="success"}[15m]))
          /
          sum(rate(agent_task_total[15m]))
          < 0.85
        for: 10m
        labels:
          severity: warning
          team: agent-platform
        annotations:
          summary: "Agent success rate < 85% for 10 minutes"

      # --- Tool Call Failures ---
      - alert: ToolCallFailureRateHigh
        expr: |
          sum by (tool_name) (
            rate(tool_call_total{status="error"}[5m])
          )
          /
          sum by (tool_name) (
            rate(tool_call_total[5m])
          )
          > 0.10
        for: 5m
        labels:
          severity: warning
          team: agent-platform
        annotations:
          summary: "Tool {{ $labels.tool_name }} failure rate > 10%"

      # --- Cost Alerts ---
      - alert: DailyTokenBudgetExceeded
        expr: |
          sum(increase(llm_tokens_cost_dollars_total[24h])) > 500
        labels:
          severity: warning
          team: agent-platform
        annotations:
          summary: "Daily LLM cost exceeded $500"
          description: |
            24h cost is ${{ $value | printf "%.2f" }}.
            Review high-token-usage tenants.

      - alert: TenantTokenBudgetExceeded
        expr: |
          sum by (tenant_id) (
            increase(llm_tokens_cost_dollars_total[24h])
          ) > 50
        labels:
          severity: warning
          team: agent-platform
        annotations:
          summary: "Tenant {{ $labels.tenant_id }} exceeded $50/day LLM budget"

      # --- vLLM Inference Server ---
      - alert: VLLMQueueDepthHigh
        expr: |
          vllm:num_requests_waiting > 50
        for: 2m
        labels:
          severity: warning
          team: ml-infra
        annotations:
          summary: "vLLM request queue depth > 50"
          description: "Inference server is backlogged. Consider scaling up."

      - alert: VLLMGPUUtilizationHigh
        expr: |
          DCGM_FI_DEV_GPU_UTIL > 95
        for: 5m
        labels:
          severity: warning
          team: ml-infra
        annotations:
          summary: "GPU utilization > 95% for 5 minutes"

      # --- Agent Step Count Anomaly ---
      - alert: AgentStepCountAnomaly
        expr: |
          histogram_quantile(0.95,
            rate(agent_step_count_bucket[15m])
          ) > 20
        for: 10m
        labels:
          severity: warning
          team: agent-platform
        annotations:
          summary: "P95 agent step count > 20 -- possible infinite loop"
```

### 4. PromQL Queries for Agent Operations

```promql
# ========================================================================
# KEY PROMQL QUERIES FOR AGENT MONITORING
# ========================================================================

# --- Token Usage ---

# Total tokens consumed in last 24h by model
sum by (model) (increase(llm_tokens_total[24h]))

# Token consumption rate (tokens/second) over last 5 minutes
sum(rate(llm_tokens_total[5m]))

# Input vs output token ratio (high output ratio = expensive)
sum(rate(llm_tokens_total{direction="output"}[1h]))
/
sum(rate(llm_tokens_total{direction="input"}[1h]))

# --- Cost ---

# Estimated daily cost in USD
sum(increase(llm_tokens_cost_dollars_total[24h]))

# Cost per tenant per day
sum by (tenant_id) (increase(llm_tokens_cost_dollars_total[24h]))

# Cost trend: compare today vs yesterday
sum(increase(llm_tokens_cost_dollars_total[24h]))
-
sum(increase(llm_tokens_cost_dollars_total[24h] offset 1d))

# --- Latency ---

# LLM request latency P50 / P95 / P99
histogram_quantile(0.50, rate(llm_request_duration_seconds_bucket[5m]))
histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))
histogram_quantile(0.99, rate(llm_request_duration_seconds_bucket[5m]))

# Time to first token P95 by model
histogram_quantile(0.95,
  sum by (le, model) (rate(llm_time_to_first_token_seconds_bucket[5m]))
)

# End-to-end agent turn latency P95
histogram_quantile(0.95, rate(agent_turn_duration_seconds_bucket[5m]))

# Average tool call latency by tool
sum by (tool_name) (rate(tool_call_duration_seconds_sum[5m]))
/
sum by (tool_name) (rate(tool_call_duration_seconds_count[5m]))

# --- Throughput ---

# Agent requests per second
sum(rate(agent_task_total[5m]))

# LLM calls per second by provider
sum by (provider) (rate(llm_requests_total[5m]))

# Tool calls per second by tool
sum by (tool_name) (rate(tool_call_total[5m]))

# --- Error Rates ---

# LLM error rate by provider
sum by (provider) (rate(llm_requests_total{status=~"error|timeout"}[5m]))
/
sum by (provider) (rate(llm_requests_total[5m]))

# Agent success rate
sum(rate(agent_task_total{outcome="success"}[5m]))
/
sum(rate(agent_task_total[5m]))

# Tool-specific error rate
sum by (tool_name) (rate(tool_call_total{status="error"}[5m]))
/
sum by (tool_name) (rate(tool_call_total[5m]))

# --- vLLM Inference Server ---

# Requests waiting in queue
vllm:num_requests_waiting

# Request throughput
rate(vllm:num_requests_running[5m])

# KV cache utilization (memory pressure indicator)
vllm:gpu_cache_usage_perc

# Average generation length
rate(vllm:generation_tokens_total[5m])
/
rate(vllm:num_requests_finished[5m])

# --- GPU ---

# GPU utilization by node
DCGM_FI_DEV_GPU_UTIL

# GPU memory usage
DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE) * 100

# GPU temperature
DCGM_FI_DEV_GPU_TEMP
```

### 5. Grafana Dashboard JSON Template (Agent Ops)

```json
{
  "dashboard": {
    "title": "Agent Platform - Operations Dashboard",
    "tags": ["agent-platform", "llm", "production"],
    "timezone": "browser",
    "refresh": "30s",
    "panels": [
      {
        "title": "LLM Request Rate (req/s)",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 8, "x": 0, "y": 0 },
        "targets": [
          {
            "expr": "sum by (provider) (rate(llm_requests_total[5m]))",
            "legendFormat": "{{ provider }}"
          }
        ]
      },
      {
        "title": "LLM Latency (P50 / P95 / P99)",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 8, "x": 8, "y": 0 },
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(llm_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(llm_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P99"
          }
        ]
      },
      {
        "title": "LLM Error Rate (%)",
        "type": "stat",
        "gridPos": { "h": 8, "w": 8, "x": 16, "y": 0 },
        "targets": [
          {
            "expr": "sum(rate(llm_requests_total{status=~'error|timeout'}[5m])) / sum(rate(llm_requests_total[5m])) * 100",
            "legendFormat": "Error Rate %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                { "color": "green", "value": null },
                { "color": "yellow", "value": 1 },
                { "color": "red", "value": 5 }
              ]
            },
            "unit": "percent"
          }
        }
      },
      {
        "title": "Daily LLM Cost (USD)",
        "type": "stat",
        "gridPos": { "h": 8, "w": 8, "x": 0, "y": 8 },
        "targets": [
          {
            "expr": "sum(increase(llm_tokens_cost_dollars_total[24h]))",
            "legendFormat": "Daily Cost"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD",
            "thresholds": {
              "steps": [
                { "color": "green", "value": null },
                { "color": "yellow", "value": 300 },
                { "color": "red", "value": 500 }
              ]
            }
          }
        }
      },
      {
        "title": "Token Usage by Model (24h)",
        "type": "piechart",
        "gridPos": { "h": 8, "w": 8, "x": 8, "y": 8 },
        "targets": [
          {
            "expr": "sum by (model) (increase(llm_tokens_total[24h]))",
            "legendFormat": "{{ model }}"
          }
        ]
      },
      {
        "title": "Agent Success Rate (%)",
        "type": "gauge",
        "gridPos": { "h": 8, "w": 8, "x": 16, "y": 8 },
        "targets": [
          {
            "expr": "sum(rate(agent_task_total{outcome='success'}[1h])) / sum(rate(agent_task_total[1h])) * 100",
            "legendFormat": "Success %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100,
            "thresholds": {
              "steps": [
                { "color": "red", "value": null },
                { "color": "yellow", "value": 80 },
                { "color": "green", "value": 95 }
              ]
            }
          }
        }
      },
      {
        "title": "Tool Call Latency by Tool (P95)",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 16 },
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum by (le, tool_name) (rate(tool_call_duration_seconds_bucket[5m])))",
            "legendFormat": "{{ tool_name }}"
          }
        ]
      },
      {
        "title": "Active Agent Sessions",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 16 },
        "targets": [
          {
            "expr": "sum by (agent_type) (agent_active_sessions)",
            "legendFormat": "{{ agent_type }}"
          }
        ]
      },
      {
        "title": "Agent Step Count Distribution",
        "type": "heatmap",
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 24 },
        "targets": [
          {
            "expr": "sum(increase(agent_step_count_bucket[5m])) by (le)",
            "legendFormat": "{{ le }}",
            "format": "heatmap"
          }
        ]
      },
      {
        "title": "TTFT - Time to First Token (P95)",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 24 },
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum by (le, model) (rate(llm_time_to_first_token_seconds_bucket[5m])))",
            "legendFormat": "{{ model }}"
          }
        ]
      }
    ]
  }
}
```

## Decision Tree: Do You Need Prometheus+Grafana?

```
Are you running AI agents in production?
│
├── Yes, using hosted LLM APIs only (no self-hosted inference)
│   ├── Need infra metrics (latency, errors, cost)? → Yes, add Prometheus
│   ├── Already have Datadog/New Relic? → Use their Prometheus integration
│   └── Prototype stage? → Skip. Langfuse alone is sufficient.
│
├── Yes, with self-hosted inference (vLLM/TGI)
│   └── Prometheus is essential. GPU metrics, queue depth, cache utilization
│       are only available via Prometheus scraping.
│
├── Yes, multi-tenant platform
│   └── Prometheus is essential for per-tenant cost tracking and SLO monitoring.
│       Langfuse alone cannot do per-tenant infra cost attribution.
│
└── No, still in development
    └── Skip Prometheus. Use Langfuse/LangSmith for development-time observability.
```

## When NOT to Use Prometheus+Grafana

- **Prototype / hackathon**: Langfuse alone is faster to set up and gives you traces.
- **Serverless-only deployment** (Lambda, Cloud Functions): No long-running process to scrape. Use CloudWatch or Datadog instead.
- **< 10 users, no SLA**: Manual log checking is sufficient at this scale.
- **Already have Datadog/New Relic**: They have Prometheus-compatible ingestion. Use their agent integration rather than running a separate Prometheus.
- **Only need prompt debugging**: That is Langfuse/LangSmith territory, not Prometheus.

## Tradeoffs

| Approach | Setup Time | Monthly Cost | Best For |
|----------|-----------|-------------|----------|
| Prometheus + Grafana (self-hosted) | 4-8 hours | $50-200 (compute) | Full control, self-hosted inference |
| Grafana Cloud (managed) | 1-2 hours | $100-500 | Faster setup, no ops burden |
| Datadog with Prometheus integration | 1-2 hours | $200-1000+ | Teams already using Datadog |
| CloudWatch only | 30 min | $50-150 | AWS-native, serverless deployments |
| Langfuse only (no infra metrics) | 30 min | $0-100 | Prototypes, traces-only needs |

| Metric Category | Prometheus | Langfuse | Both Needed? |
|----------------|-----------|---------|-------------|
| Token count trends | Counter over time | Per-trace breakdown | Both useful |
| LLM latency percentiles | Yes (histograms) | Yes (per-trace) | Prometheus for SLOs |
| GPU utilization | Yes | No | Prometheus only |
| Agent trace waterfall | No | Yes | Langfuse only |
| Cost dashboards | Yes (calculated) | Yes (built-in) | Either works |
| Alerting on SLO breach | Yes (Alertmanager) | Limited | Prometheus preferred |
| Queue depth / backpressure | Yes | No | Prometheus only |

## Failure Modes

1. **Metric cardinality explosion**: Adding `user_id` as a label on high-throughput metrics creates millions of time series. Mitigation: use `tenant_id` (bounded set), not `user_id` (unbounded).

2. **Scrape target disappearing**: vLLM pod restarts and Prometheus loses the target. Mitigation: use Kubernetes service discovery (`kubernetes_sd_configs`) instead of static targets.

3. **Stale cost estimates**: Token prices change but `TOKEN_PRICES` dict is hardcoded. Mitigation: load prices from config map, update quarterly.

4. **Alert fatigue**: Too many low-severity alerts fire during normal operation. Mitigation: tune thresholds using 2 weeks of baseline data; use `for:` duration to avoid flapping alerts.

5. **Missing TTFT metric**: Not all LLM SDKs expose time-to-first-token. You may need to instrument streaming manually.

6. **Prometheus storage full**: 30 days of high-cardinality metrics fills disk. Mitigation: use Thanos or Cortex for long-term storage; reduce retention to 15 days for raw metrics.

## Source(s) and Further Reading

- Prometheus Documentation: https://prometheus.io/docs/
- Grafana Dashboards: https://grafana.com/docs/grafana/latest/dashboards/
- vLLM Metrics: https://docs.vllm.ai/en/latest/serving/metrics.html
- DCGM Exporter (GPU Metrics): https://github.com/NVIDIA/dcgm-exporter
- OpenLLMetry (OTel for LLMs): https://github.com/traceloop/openllmetry
- Prometheus Client Python: https://prometheus.github.io/client_python/
- "Monitoring LLM Applications" - Chip Huyen (2024)
- Grafana Loki: https://grafana.com/docs/loki/latest/
- Grafana Tempo: https://grafana.com/docs/tempo/latest/
