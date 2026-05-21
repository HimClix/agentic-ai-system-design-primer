# Audit Trails for AI Agents

> If you can't replay exactly what an agent did, why it did it, and what it saw at each step, you don't have an audit trail -- you have a liability.

## What It Is

An audit trail for AI agents is a complete, immutable record of every decision, action, observation, and reasoning step the agent performed during a session. Unlike traditional application logs (which record system events), agent audit trails must capture **the reasoning process** -- what the agent observed, what it considered, why it chose a specific action, and what the outcome was.

This is required for compliance (EU AI Act, SOC 2, HIPAA), incident investigation, and debugging non-deterministic behavior.

## How It Works

### What to Log Per Agent Step

| Field | Description | Example |
|-------|------------|---------|
| `trace_id` | Unique ID for the entire agent session | `trace_abc123` |
| `step_id` | Unique ID for this specific step | `step_007` |
| `timestamp` | ISO 8601 with timezone | `2025-05-19T14:30:22.543Z` |
| `agent_name` | Which agent is acting | `customer_support_agent` |
| `agent_version` | Deployed version | `v2.3.1` |
| `step_type` | Category of step | `llm_call`, `tool_call`, `decision`, `human_approval` |
| `input_state` | State entering this step | `{"messages": [...], "context": {...}}` |
| `model_name` | LLM model used | `gpt-4o-2024-08-06` |
| `prompt_hash` | Hash of system prompt version | `sha256:a1b2c3...` |
| `reasoning` | Agent's chain-of-thought | `"Customer asked about refund. Checking order status..."` |
| `tool_calls` | Tools invoked | `[{"name": "get_order", "args": {"id": "ORD-123"}}]` |
| `tool_results` | Tool return values | `[{"order_status": "delivered"}]` |
| `output_state` | State after this step | `{"response": "...", "next_action": "..."}` |
| `tokens_used` | Input + output tokens | `{"input": 1500, "output": 350}` |
| `latency_ms` | Step duration | `2340` |
| `guardrail_results` | Safety check results | `{"input_scan": "pass", "alignment": "pass"}` |
| `credentials_used` | Token IDs used (not values) | `["tok_read_abc"]` |
| `error` | Any error that occurred | `null` or `{"type": "ToolError", "message": "..."}` |
| `user_id` | Who initiated this session | `user_jane_doe` (hashed for PII) |
| `approval_id` | Human who approved (if HITL) | `approver_john` |

## Production Implementation

```python
"""
Agent audit trail system for compliance and debugging.
"""
import json
import time
import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Single audit trail entry for one agent step."""
    trace_id: str
    step_id: str
    timestamp: str
    agent_name: str
    agent_version: str
    step_type: str  # llm_call, tool_call, decision, human_approval, error
    
    # Input/Output
    input_state_hash: str  # Hash of input state (not raw state for size)
    output_state_hash: str = ""
    
    # LLM details
    model_name: str = ""
    prompt_hash: str = ""
    reasoning_summary: str = ""  # Truncated CoT
    tokens_input: int = 0
    tokens_output: int = 0
    
    # Tool details
    tool_name: str = ""
    tool_args_hash: str = ""  # Hash, not raw args (may contain PII)
    tool_success: bool = True
    tool_latency_ms: float = 0
    
    # Safety
    guardrail_results: dict = field(default_factory=dict)
    credential_token_ids: list[str] = field(default_factory=list)
    
    # Metadata
    latency_ms: float = 0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    user_id_hash: str = ""  # Always hashed
    approval_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class AuditTrailRecorder:
    """
    Records audit trail entries for agent sessions.
    Designed for append-only, immutable storage.
    """
    
    def __init__(
        self,
        agent_name: str,
        agent_version: str,
        storage_backend: str = "stdout",  # stdout, file, s3, database
    ):
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.storage_backend = storage_backend
        self._buffer: list[AuditEntry] = []
    
    def start_trace(self, user_id: str) -> str:
        """Start a new audit trace for an agent session."""
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        user_id_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        
        entry = AuditEntry(
            trace_id=trace_id,
            step_id=f"{trace_id}_init",
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_name=self.agent_name,
            agent_version=self.agent_version,
            step_type="session_start",
            input_state_hash="",
            user_id_hash=user_id_hash,
        )
        
        self._write(entry)
        return trace_id
    
    def record_llm_call(
        self,
        trace_id: str,
        model_name: str,
        input_state: dict,
        output_state: dict,
        reasoning: str,
        tokens_input: int,
        tokens_output: int,
        latency_ms: float,
        prompt_version: str = "",
        guardrail_results: Optional[dict] = None,
    ) -> str:
        """Record an LLM call step."""
        step_id = f"{trace_id}_llm_{uuid.uuid4().hex[:8]}"
        
        entry = AuditEntry(
            trace_id=trace_id,
            step_id=step_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_name=self.agent_name,
            agent_version=self.agent_version,
            step_type="llm_call",
            input_state_hash=self._hash_state(input_state),
            output_state_hash=self._hash_state(output_state),
            model_name=model_name,
            prompt_hash=hashlib.sha256(prompt_version.encode()).hexdigest()[:16],
            reasoning_summary=reasoning[:500],  # Truncate for storage
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
            guardrail_results=guardrail_results or {},
        )
        
        self._write(entry)
        return step_id
    
    def record_tool_call(
        self,
        trace_id: str,
        tool_name: str,
        tool_args: dict,
        tool_result: Any,
        success: bool,
        latency_ms: float,
        credential_token_id: Optional[str] = None,
        error: Optional[Exception] = None,
    ) -> str:
        """Record a tool call step."""
        step_id = f"{trace_id}_tool_{uuid.uuid4().hex[:8]}"
        
        entry = AuditEntry(
            trace_id=trace_id,
            step_id=step_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_name=self.agent_name,
            agent_version=self.agent_version,
            step_type="tool_call",
            input_state_hash=self._hash_state(tool_args),
            output_state_hash=self._hash_state({"result": str(tool_result)[:1000]}),
            tool_name=tool_name,
            tool_args_hash=self._hash_state(tool_args),
            tool_success=success,
            tool_latency_ms=latency_ms,
            credential_token_ids=[credential_token_id] if credential_token_id else [],
            error_type=type(error).__name__ if error else None,
            error_message=str(error)[:500] if error else None,
        )
        
        self._write(entry)
        return step_id
    
    def record_human_approval(
        self,
        trace_id: str,
        action_requested: str,
        approver_id: str,
        approved: bool,
        reason: str = "",
    ) -> str:
        """Record a human-in-the-loop approval decision."""
        step_id = f"{trace_id}_hitl_{uuid.uuid4().hex[:8]}"
        
        entry = AuditEntry(
            trace_id=trace_id,
            step_id=step_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_name=self.agent_name,
            agent_version=self.agent_version,
            step_type="human_approval",
            input_state_hash=self._hash_state({"action": action_requested}),
            output_state_hash=self._hash_state({"approved": approved, "reason": reason}),
            approval_id=approver_id,
            reasoning_summary=f"Action: {action_requested}, Approved: {approved}, Reason: {reason}",
        )
        
        self._write(entry)
        return step_id
    
    def _hash_state(self, state: Any) -> str:
        """Hash state for storage (don't store raw state with potential PII)."""
        return hashlib.sha256(
            json.dumps(state, default=str, sort_keys=True).encode()
        ).hexdigest()[:16]
    
    def _write(self, entry: AuditEntry):
        """Write entry to storage backend."""
        self._buffer.append(entry)
        
        if self.storage_backend == "stdout":
            logger.info(f"AUDIT: {entry.to_json()}")
        # In production: write to S3, database, or dedicated audit service
    
    def flush(self):
        """Flush buffered entries to storage."""
        # In production: batch write to storage backend
        self._buffer.clear()
```

### Compliance Requirements by Framework

| Framework | Requirement | What to Store |
|-----------|------------|---------------|
| **EU AI Act** (Aug 2026) | Automatic logging of high-risk AI operations | Full trace with reasoning, inputs, outputs |
| **SOC 2** | Audit trail for system actions | Who, what, when, outcome |
| **HIPAA** | Access logs for PHI | Every data access with user ID, timestamp |
| **GDPR** | Data processing records | What personal data was processed, purpose |
| **PCI DSS** | Access to cardholder data | Every access to payment data |

### Storage Pattern

```
Hot Storage (0-30 days):  PostgreSQL/ClickHouse -- fast queries for debugging
Warm Storage (30-365 days): S3 + Parquet -- cost-effective, queryable with Athena
Cold Storage (1-7 years):   S3 Glacier -- compliance retention, rarely accessed
```

## Decision Tree / When to Use

- **Always** -- Every production agent needs audit trails. The question is how detailed.
- **Detailed (all fields)**: Regulated industries, financial operations, PII handling
- **Standard (core fields)**: Internal tools, non-regulated environments
- **Minimal (trace_id + outcome)**: Development, testing, non-production

## When NOT to Use

- Never skip audit trails for production agents
- Development environments can use simplified logging

## Tradeoffs

| Storage Level | Compliance | Storage Cost | Query Speed | Debugging Value |
|--------------|------------|-------------|-------------|-----------------|
| Full state capture | Excellent | High (~$50/M traces/mo) | Slow | Excellent |
| State hashes + summaries | Good | Medium (~$10/M traces/mo) | Fast | Good |
| Step IDs + outcomes only | Minimal | Low (~$2/M traces/mo) | Very fast | Limited |

## Failure Modes

| Failure | Cause | Impact | Prevention |
|---------|-------|--------|------------|
| **Audit gap** | Logger fails silently | Missing compliance records | Health checks on audit pipeline |
| **PII in logs** | Raw state logged without masking | Privacy violation | Always hash/mask PII fields |
| **Storage overflow** | High-volume agent creates too many entries | Cost explosion | Sampling for low-risk operations |
| **Tampered logs** | Logs modified after the fact | Audit integrity compromised | Append-only storage + checksums |

## Source(s) and Further Reading

- EU AI Act, Article 12: Record-keeping obligations for high-risk AI systems
- NIST AI RMF, GOVERN 1.7: Audit trail requirements
- SOC 2 Type II, CC7.2: System monitoring and logging
- OpenTelemetry Logging specification: https://opentelemetry.io/docs/specs/otel/logs/
