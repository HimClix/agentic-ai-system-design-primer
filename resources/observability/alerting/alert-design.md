# Alert Design for AI Agents

> The five agent-specific alerts you must have on day one: runaway loops, token explosions, repeated tool failures, unsafe actions, and quality degradation.

## What It Is

Agent alerting detects behavioral anomalies that traditional APM alerts miss: an agent stuck in a reasoning loop, a prompt injection causing a token explosion, a tool failing silently while the agent confidently hallucinates, or gradual quality degradation that no single request reveals.

## How It Works

### The 5 Critical Agent Alerts

| Alert | What It Detects | Detection Method | Severity |
|-------|----------------|-----------------|----------|
| **Runaway loop** | Agent repeating same action >N times | Step counter per trace | CRITICAL |
| **Token explosion** | Token usage >5x baseline per request | Token count vs rolling median | CRITICAL |
| **Repeated tool failure** | Same tool failing >N times in window | Per-tool error rate | HIGH |
| **Unsafe action** | Agent attempting blocked/unauthorized action | Guardrail trigger count | CRITICAL |
| **Quality degradation** | Average eval score dropping over time | Rolling average of LLM-as-judge scores | HIGH |

### Detection Methods and Thresholds

```python
"""
Production alert definitions for AI agents.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import time
from collections import deque


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    name: str
    severity: Severity
    message: str
    value: float
    threshold: float
    trace_id: Optional[str] = None
    timestamp: float = 0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class AgentAlertManager:
    """Detects and fires alerts for agent behavioral anomalies."""
    
    def __init__(self, alert_handler=None):
        self.handler = alert_handler or self._default_handler
        # Rolling windows for baselines
        self._token_history = deque(maxlen=1000)
        self._cost_history = deque(maxlen=1000)
        self._quality_scores = deque(maxlen=500)
        self._tool_errors: dict[str, deque] = {}
        self._loop_counters: dict[str, dict] = {}  # trace_id -> {action: count}
    
    # ─── Alert 1: Runaway Loop Detection ─────────────────────
    
    def check_runaway_loop(
        self,
        trace_id: str,
        action: str,
        max_repeats: int = 5,
    ) -> Optional[Alert]:
        """
        Detect when an agent repeats the same action too many times.
        Catches: infinite reasoning loops, repeated failed tool calls,
        agent stuck in a cycle.
        """
        if trace_id not in self._loop_counters:
            self._loop_counters[trace_id] = {}
        
        counts = self._loop_counters[trace_id]
        counts[action] = counts.get(action, 0) + 1
        
        if counts[action] > max_repeats:
            alert = Alert(
                name="runaway_loop",
                severity=Severity.CRITICAL,
                message=f"Agent repeated action '{action}' {counts[action]} times "
                        f"(threshold: {max_repeats}). Possible infinite loop.",
                value=counts[action],
                threshold=max_repeats,
                trace_id=trace_id,
            )
            self.handler(alert)
            return alert
        return None
    
    # ─── Alert 2: Token Explosion ────────────────────────────
    
    def check_token_explosion(
        self,
        trace_id: str,
        tokens_used: int,
        multiplier_threshold: float = 5.0,
    ) -> Optional[Alert]:
        """
        Detect when a single request uses vastly more tokens than normal.
        Catches: prompt injection causing verbose output, context stuffing,
        agent including full documents in response.
        """
        self._token_history.append(tokens_used)
        
        if len(self._token_history) < 50:
            return None  # Need baseline
        
        sorted_tokens = sorted(self._token_history)
        median = sorted_tokens[len(sorted_tokens) // 2]
        
        if median > 0 and tokens_used > median * multiplier_threshold:
            alert = Alert(
                name="token_explosion",
                severity=Severity.CRITICAL,
                message=f"Token usage {tokens_used} is {tokens_used/median:.1f}x "
                        f"the median ({median}). Possible injection or loop.",
                value=tokens_used,
                threshold=median * multiplier_threshold,
                trace_id=trace_id,
            )
            self.handler(alert)
            return alert
        return None
    
    # ─── Alert 3: Repeated Tool Failure ──────────────────────
    
    def check_tool_failure(
        self,
        tool_name: str,
        success: bool,
        window_size: int = 100,
        min_success_rate: float = 0.95,
    ) -> Optional[Alert]:
        """
        Detect when a specific tool starts failing repeatedly.
        Catches: API outages, auth token expiration, rate limiting,
        breaking changes in external APIs.
        """
        if tool_name not in self._tool_errors:
            self._tool_errors[tool_name] = deque(maxlen=window_size)
        
        self._tool_errors[tool_name].append(1 if success else 0)
        
        if len(self._tool_errors[tool_name]) < 20:
            return None  # Need baseline
        
        success_rate = sum(self._tool_errors[tool_name]) / len(self._tool_errors[tool_name])
        
        if success_rate < min_success_rate:
            severity = Severity.CRITICAL if success_rate < 0.8 else Severity.HIGH
            alert = Alert(
                name="tool_failure",
                severity=severity,
                message=f"Tool '{tool_name}' success rate is {success_rate:.1%} "
                        f"(threshold: {min_success_rate:.1%}). "
                        f"Recent {len(self._tool_errors[tool_name])} calls examined.",
                value=success_rate,
                threshold=min_success_rate,
            )
            self.handler(alert)
            return alert
        return None
    
    # ─── Alert 4: Unsafe Action Attempted ────────────────────
    
    def check_unsafe_action(
        self,
        trace_id: str,
        action_name: str,
        guardrail_result: str,  # "blocked", "warned", "passed"
    ) -> Optional[Alert]:
        """
        Alert when guardrails block an agent action.
        Catches: prompt injection success, privilege escalation attempts,
        data exfiltration attempts.
        """
        if guardrail_result == "blocked":
            alert = Alert(
                name="unsafe_action_blocked",
                severity=Severity.CRITICAL,
                message=f"Guardrail BLOCKED action '{action_name}'. "
                        f"Possible injection or attack in progress.",
                value=1,
                threshold=0,
                trace_id=trace_id,
            )
            self.handler(alert)
            return alert
        
        elif guardrail_result == "warned":
            alert = Alert(
                name="unsafe_action_warned",
                severity=Severity.WARNING,
                message=f"Guardrail WARNING on action '{action_name}'. "
                        f"Monitoring for escalation.",
                value=0.5,
                threshold=0,
                trace_id=trace_id,
            )
            self.handler(alert)
            return alert
        
        return None
    
    # ─── Alert 5: Quality Degradation ────────────────────────
    
    def check_quality_degradation(
        self,
        eval_score: float,  # 0.0 to 1.0
        min_rolling_avg: float = 0.7,
        window_size: int = 100,
    ) -> Optional[Alert]:
        """
        Detect gradual quality degradation over time.
        Catches: model regression after updates, prompt drift,
        RAG corpus quality degradation, changing user patterns.
        """
        self._quality_scores.append(eval_score)
        
        if len(self._quality_scores) < window_size:
            return None
        
        recent = list(self._quality_scores)[-window_size:]
        rolling_avg = sum(recent) / len(recent)
        
        if rolling_avg < min_rolling_avg:
            alert = Alert(
                name="quality_degradation",
                severity=Severity.HIGH,
                message=f"Rolling average quality score is {rolling_avg:.2f} "
                        f"(threshold: {min_rolling_avg}). "
                        f"Agent quality may be degrading.",
                value=rolling_avg,
                threshold=min_rolling_avg,
            )
            self.handler(alert)
            return alert
        return None
    
    def _default_handler(self, alert: Alert):
        """Default: log the alert. In production: PagerDuty, Slack, etc."""
        import logging
        logger = logging.getLogger("agent.alerts")
        log_fn = {
            Severity.INFO: logger.info,
            Severity.WARNING: logger.warning,
            Severity.HIGH: logger.error,
            Severity.CRITICAL: logger.critical,
        }[alert.severity]
        log_fn(f"[{alert.name}] {alert.message}")
```

### Integration Patterns

```python
# PagerDuty integration
def pagerduty_handler(alert: Alert):
    if alert.severity in (Severity.HIGH, Severity.CRITICAL):
        # Trigger PagerDuty incident
        requests.post("https://events.pagerduty.com/v2/enqueue", json={
            "routing_key": "your-integration-key",
            "event_action": "trigger",
            "payload": {
                "summary": f"[Agent] {alert.name}: {alert.message}",
                "severity": alert.severity.value,
                "source": "agent-monitor",
                "custom_details": {
                    "trace_id": alert.trace_id,
                    "value": alert.value,
                    "threshold": alert.threshold,
                },
            },
        })

# Grafana annotation (for correlating with dashboards)
def grafana_handler(alert: Alert):
    requests.post("http://grafana:3000/api/annotations", json={
        "text": f"{alert.name}: {alert.message}",
        "tags": ["agent", alert.severity.value, alert.name],
    }, headers={"Authorization": "Bearer ..."})
```

## Decision Tree / When to Use

- **Day 1**: Runaway loops + token explosions (prevent cost disasters)
- **Week 1**: + Tool failures + unsafe actions (prevent quality and safety issues)
- **Month 1**: + Quality degradation (long-term monitoring)

## When NOT to Use

- Never skip runaway loop and token explosion alerts -- they prevent cost disasters
- Quality degradation alerts can be deferred during initial development

## Tradeoffs

| Alert | False Positive Risk | Miss Risk If Absent | Effort to Implement |
|-------|-------------------|--------------------|--------------------|
| Runaway loop | Low | Cost explosion | Low (counter) |
| Token explosion | Medium (long legit responses) | Cost explosion | Low (comparison) |
| Tool failure | Low | Silent wrong answers | Low (success tracking) |
| Unsafe action | Low (guardrails decide) | Security breach | Medium (guardrail integration) |
| Quality degradation | Medium (score variance) | Gradual quality loss | High (requires eval pipeline) |

## Failure Modes

| Failure | Cause | Impact | Prevention |
|---------|-------|--------|------------|
| **Alert fatigue** | Thresholds too aggressive | Real alerts ignored | Tune based on observed baselines |
| **Missing baseline** | Alerts enabled before calibration | Every request triggers alert | Wait for 100+ requests before enabling |
| **Silent failure** | Alert pipeline itself fails | No alerts for real issues | Alert on alerting system health |
| **Partial coverage** | Only monitor some agents/tools | Blind spots | Centralized alert configuration |

## Source(s) and Further Reading

- Google SRE Book, Chapter 6: Monitoring Distributed Systems
- PagerDuty Events API: https://developer.pagerduty.com/docs/events-api-v2/overview/
- Grafana Alerting: https://grafana.com/docs/grafana/latest/alerting/
- Langfuse Alerting: https://langfuse.com/docs/analytics
