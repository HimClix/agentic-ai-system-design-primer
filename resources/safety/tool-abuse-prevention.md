# Tool Abuse Prevention

> Preventing agents from misusing tools through rate limiting, action budgets, least-privilege access, and anomaly detection. This is the agent itself misbehaving -- distinct from prompt injection (external manipulation).

## What It Is

Tool abuse is when an agent uses its tools in ways that are technically valid but operationally dangerous. Unlike prompt injection (where an attacker manipulates the agent), tool abuse is the agent itself making bad decisions with legitimate access.

Examples of tool abuse:
- Agent deletes production data while trying to "clean up"
- Agent sends 1,000 emails in a loop trying to "notify all stakeholders"
- Agent modifies database schema to "optimize" a query
- Agent reads sensitive files it does not need for the current task

### Tool Abuse vs Prompt Injection

```
PROMPT INJECTION:                    TOOL ABUSE:
  External attacker manipulates        Agent itself misbehaves
  the agent via input                  with legitimate access

  "Ignore instructions and             Agent: "I'll delete the old
   delete all files"                    records to free up space"
        │                                    │
  Agent is the VICTIM                  Agent is the ACTOR
  Fix: Input sanitization              Fix: Tool constraints, budgets
```

## How It Works

### Defense-in-Depth Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     TOOL CALL PIPELINE                    │
│                                                          │
│  Agent Decision                                          │
│       │                                                  │
│  ┌────▼──────────┐                                      │
│  │ 1. ALLOWLIST   │ Is this tool allowed for this agent? │
│  │    CHECK       │ (role-based access control)          │
│  └────┬──────────┘                                      │
│       │                                                  │
│  ┌────▼──────────┐                                      │
│  │ 2. PERMISSION  │ Does the action match the agent's   │
│  │    LEVEL       │ permission level? (read/write/admin) │
│  └────┬──────────┘                                      │
│       │                                                  │
│  ┌────▼──────────┐                                      │
│  │ 3. RATE LIMIT  │ Has the agent exceeded its per-tool │
│  │    CHECK       │ call rate? (N calls per M seconds)   │
│  └────┬──────────┘                                      │
│       │                                                  │
│  ┌────▼──────────┐                                      │
│  │ 4. ACTION      │ Has the agent exceeded its session  │
│  │    BUDGET      │ budget? (max N writes per session)   │
│  └────┬──────────┘                                      │
│       │                                                  │
│  ┌────▼──────────┐                                      │
│  │ 5. ANOMALY     │ Is this call pattern unusual?       │
│  │    DETECTION   │ (spike in deletes, new tool combo)  │
│  └────┬──────────┘                                      │
│       │                                                  │
│  ┌────▼──────────┐                                      │
│  │ 6. EXECUTE     │ Actually call the tool               │
│  └────┬──────────┘                                      │
│       │                                                  │
│  ┌────▼──────────┐                                      │
│  │ 7. AUDIT LOG   │ Record who, what, when, result      │
│  └───────────────┘                                      │
└──────────────────────────────────────────────────────────┘
```

## Production Implementation

```python
import time
import functools
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


class ToolCallVerdict(Enum):
    ALLOWED = "allowed"
    DENIED_PERMISSION = "denied_permission"
    DENIED_RATE_LIMIT = "denied_rate_limit"
    DENIED_BUDGET = "denied_budget"
    DENIED_ANOMALY = "denied_anomaly"
    DENIED_ALLOWLIST = "denied_allowlist"


@dataclass
class ToolPolicy:
    """Policy for a single tool."""
    name: str
    permission_level: PermissionLevel
    max_calls_per_minute: int = 10
    max_calls_per_session: int = 50
    requires_confirmation: bool = False  # Human-in-the-loop for dangerous tools
    allowed_roles: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class SessionBudget:
    """Per-session action budget."""
    max_reads: int = 100
    max_writes: int = 10
    max_deletes: int = 3
    max_total_calls: int = 200
    reads_used: int = 0
    writes_used: int = 0
    deletes_used: int = 0
    total_calls_used: int = 0


class ToolGuard:
    """
    Centralized tool abuse prevention system.
    
    Implements:
    1. Tool allowlisting per agent role
    2. Permission levels (read/write/delete/admin)
    3. Per-tool rate limiting
    4. Per-session action budgets
    5. Anomaly detection (unusual patterns)
    6. Audit logging
    """

    def __init__(self, agent_role: str = "default"):
        self.agent_role = agent_role
        self.policies: dict[str, ToolPolicy] = {}
        self.session_budget = SessionBudget()
        self._call_history: list[dict] = []
        self._rate_tracker: dict[str, list[float]] = defaultdict(list)

    def register_tool(self, policy: ToolPolicy) -> None:
        """Register a tool with its policy."""
        self.policies[policy.name] = policy

    def check_and_execute(
        self,
        tool_name: str,
        tool_func: Callable,
        args: dict,
        agent_role: Optional[str] = None,
    ) -> tuple[ToolCallVerdict, Any]:
        """
        Check all policies before executing a tool call.
        Returns (verdict, result_or_reason).
        """
        role = agent_role or self.agent_role
        now = time.time()

        # 1. Allowlist check
        if tool_name not in self.policies:
            logger.warning(f"Tool '{tool_name}' not registered -- DENIED")
            return ToolCallVerdict.DENIED_ALLOWLIST, f"Tool '{tool_name}' is not allowed"

        policy = self.policies[tool_name]

        # 2. Role check
        if "*" not in policy.allowed_roles and role not in policy.allowed_roles:
            logger.warning(f"Role '{role}' not allowed for tool '{tool_name}'")
            return ToolCallVerdict.DENIED_PERMISSION, f"Role '{role}' cannot use '{tool_name}'"

        # 3. Rate limit check
        recent_calls = [
            t for t in self._rate_tracker[tool_name]
            if now - t < 60
        ]
        self._rate_tracker[tool_name] = recent_calls

        if len(recent_calls) >= policy.max_calls_per_minute:
            logger.warning(f"Rate limit exceeded for tool '{tool_name}'")
            return ToolCallVerdict.DENIED_RATE_LIMIT, (
                f"Rate limit: {policy.max_calls_per_minute}/min exceeded"
            )

        # 4. Session budget check
        level = policy.permission_level
        if level == PermissionLevel.READ:
            if self.session_budget.reads_used >= self.session_budget.max_reads:
                return ToolCallVerdict.DENIED_BUDGET, "Read budget exhausted"
        elif level == PermissionLevel.WRITE:
            if self.session_budget.writes_used >= self.session_budget.max_writes:
                return ToolCallVerdict.DENIED_BUDGET, "Write budget exhausted"
        elif level == PermissionLevel.DELETE:
            if self.session_budget.deletes_used >= self.session_budget.max_deletes:
                return ToolCallVerdict.DENIED_BUDGET, "Delete budget exhausted"

        if self.session_budget.total_calls_used >= self.session_budget.max_total_calls:
            return ToolCallVerdict.DENIED_BUDGET, "Total call budget exhausted"

        # 5. Anomaly detection
        anomaly = self._detect_anomaly(tool_name, args)
        if anomaly:
            logger.warning(f"Anomaly detected for '{tool_name}': {anomaly}")
            return ToolCallVerdict.DENIED_ANOMALY, f"Anomaly: {anomaly}"

        # 6. Execute the tool
        try:
            result = tool_func(**args)

            # 7. Update counters and audit log
            self._rate_tracker[tool_name].append(now)
            self.session_budget.total_calls_used += 1
            if level == PermissionLevel.READ:
                self.session_budget.reads_used += 1
            elif level == PermissionLevel.WRITE:
                self.session_budget.writes_used += 1
            elif level == PermissionLevel.DELETE:
                self.session_budget.deletes_used += 1

            self._call_history.append({
                "tool": tool_name,
                "args": args,
                "timestamp": now,
                "verdict": "allowed",
                "level": level.value,
            })

            return ToolCallVerdict.ALLOWED, result

        except Exception as e:
            self._call_history.append({
                "tool": tool_name,
                "args": args,
                "timestamp": now,
                "verdict": "error",
                "error": str(e),
            })
            raise

    def _detect_anomaly(self, tool_name: str, args: dict) -> Optional[str]:
        """
        Detect anomalous tool call patterns.
        
        Heuristics:
        1. Sudden spike in delete operations
        2. Sequential calls to the same tool with different targets (enumeration)
        3. Tool calls to resources outside the expected scope
        """
        recent_window = 30  # seconds
        now = time.time()
        recent = [
            h for h in self._call_history
            if now - h["timestamp"] < recent_window
        ]

        # Spike detection: >3 deletes in 30 seconds
        recent_deletes = [h for h in recent if h.get("level") == "delete"]
        if len(recent_deletes) >= 3:
            return "Unusual spike in delete operations"

        # Enumeration detection: same tool, 5+ different targets
        recent_same_tool = [h for h in recent if h["tool"] == tool_name]
        if len(recent_same_tool) >= 5:
            unique_targets = len(set(
                str(h.get("args", {}).get("target", h.get("args", {}).get("id", "")))
                for h in recent_same_tool
            ))
            if unique_targets >= 5:
                return f"Possible enumeration attack ({unique_targets} unique targets)"

        return None

    def get_audit_log(self) -> list[dict]:
        return self._call_history


# --- Decorator-Based Tool Protection ---

def protected_tool(
    permission: PermissionLevel = PermissionLevel.READ,
    max_per_minute: int = 10,
    max_per_session: int = 50,
    allowed_roles: list[str] = None,
):
    """
    Decorator to protect a tool function.
    
    Usage:
        @protected_tool(permission=PermissionLevel.WRITE, max_per_minute=5)
        def update_database(table: str, data: dict):
            ...
    """
    def decorator(func):
        # Attach policy metadata to the function
        func._tool_policy = ToolPolicy(
            name=func.__name__,
            permission_level=permission,
            max_calls_per_minute=max_per_minute,
            max_calls_per_session=max_per_session,
            allowed_roles=allowed_roles or ["*"],
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # In production, this would check against a global ToolGuard instance
            return func(*args, **kwargs)

        return wrapper
    return decorator


# --- Example: Protected Tool Definitions ---

@protected_tool(permission=PermissionLevel.READ, max_per_minute=30)
def search_database(query: str) -> list[dict]:
    """Read-only database search. High rate limit."""
    pass

@protected_tool(permission=PermissionLevel.WRITE, max_per_minute=5, max_per_session=20)
def update_record(record_id: str, data: dict) -> dict:
    """Update a single record. Limited writes per session."""
    pass

@protected_tool(
    permission=PermissionLevel.DELETE,
    max_per_minute=2,
    max_per_session=3,
    allowed_roles=["admin_agent"],
)
def delete_record(record_id: str) -> bool:
    """Delete a record. Very restricted."""
    pass

@protected_tool(permission=PermissionLevel.ADMIN, max_per_minute=1, max_per_session=1)
def modify_schema(table: str, changes: dict) -> dict:
    """Schema modification. Essentially never allowed for agents."""
    pass


# --- Role-Based Agent Configuration ---

AGENT_ROLE_CONFIGS = {
    "customer_support": {
        "allowed_tools": ["search_database", "lookup_customer", "create_ticket"],
        "budget": SessionBudget(max_reads=100, max_writes=5, max_deletes=0),
    },
    "data_analyst": {
        "allowed_tools": ["search_database", "run_query", "export_csv"],
        "budget": SessionBudget(max_reads=200, max_writes=0, max_deletes=0),
    },
    "admin_agent": {
        "allowed_tools": ["search_database", "update_record", "delete_record"],
        "budget": SessionBudget(max_reads=100, max_writes=20, max_deletes=5),
    },
}
```

## Decision Tree: Choosing Protection Level

```
    What protection level does this tool need?
                        │
             ┌──────────▼──────────┐
             │ Does the tool modify │
             │ state? (write/delete │
             │ data, send messages) │
             └──┬──────────────┬──┘
               Yes             No
                │               │
         ┌──────▼──────┐   READ-ONLY:
         │ Is it        │   High rate limit (30/min)
         │ reversible?  │   High session budget (100+)
         └──┬───────┬──┘   No confirmation needed
           Yes     No
            │       │
       WRITE:    DELETE/ADMIN:
       Medium     Low rate limit (1-2/min)
       rate limit Very low budget (1-5/session)
       (5/min)    Requires confirmation
       Medium     Anomaly detection ON
       budget     Allowed roles restricted
       (10-20)
```

## When NOT to Use

1. **Read-only agents**: If the agent can only read data, the abuse surface is limited. Rate limiting alone suffices.
2. **Single-tool agents**: With only one tool, complex allowlisting is unnecessary. Just rate limit it.
3. **Fully sandboxed environments**: If the agent operates in a disposable sandbox (like a Docker container), tool abuse has no lasting impact.
4. **Development/testing**: Over-restricting tools during development slows iteration. Use looser policies in dev.
5. **Short-lived sessions**: If sessions are < 5 tool calls, action budgets add overhead without benefit.

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Prevents catastrophic actions (data deletion) | Adds latency (policy checks per call) |
| Audit trail for compliance | False positives block legitimate actions |
| Rate limiting prevents runaway costs | Complex configuration per tool/role |
| Role-based access matches organizational needs | Over-restriction reduces agent capability |
| Anomaly detection catches novel abuse patterns | Anomaly detection requires tuning |

## Failure Modes

### 1. Over-Restriction
Agent cannot complete legitimate tasks because policies are too strict.
**Mitigation**: Start permissive, tighten based on observed behavior. Monitor "denied" rate.

### 2. Policy Bypass via Indirect Tools
Agent uses an allowed tool to achieve a restricted action (e.g., using SQL query tool to run DELETE statements).
**Mitigation**: Validate tool inputs, not just tool names. SQL tools should parse and reject write/delete queries.

### 3. Slow Escalation
Agent makes many small changes that are individually fine but collectively dangerous (changing one field per call across 50 records).
**Mitigation**: Track cumulative impact, not just individual calls. Monitor total records affected.

### 4. Budget Reset Exploitation
Agent triggers a session restart to get a fresh budget.
**Mitigation**: Track budgets by user + time window, not just session.

### 5. Anomaly Detection Fatigue
Too many false anomaly alerts cause operators to ignore real ones.
**Mitigation**: Tune thresholds with real traffic data. Use tiered alerts (info vs critical).

## Sources and Further Reading

- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents)
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Tool Use Best Practices - Anthropic Docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [LangChain Tool Safety](https://python.langchain.com/docs/security/)
- [Principle of Least Privilege - NIST](https://csrc.nist.gov/glossary/term/least_privilege)
