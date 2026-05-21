# Credential Scoping for AI Agents

> An agent should never hold a credential more powerful than the specific action it needs to perform right now -- session-scoped, read-only by default, write only when explicitly approved.

## What It Is

Credential scoping is the practice of issuing AI agents the **minimum-privilege, shortest-lived credentials** possible for each operation. Unlike human users who receive persistent role-based access, agents receive **session-scoped tokens** that grant only the specific permissions needed for the current task, expire quickly, and cannot be escalated.

This is the **last line of defense** -- even when all other safety measures fail (input scanning, alignment checks, output validation), scoped credentials limit what a compromised agent can actually do.

## How It Works

### The Principle: No Permanent Credentials for Agents

```
WRONG (what most teams do):
  Agent → uses admin API key → can read/write/delete everything
  If agent is compromised → blast radius = EVERYTHING

RIGHT (credential scoping):
  Agent → requests session token → gets read-only for specific resource
  Agent needs write? → separate approval → short-lived write token
  If agent is compromised → blast radius = one resource, read-only, expires in 5 min
```

### Credential Hierarchy

```
Level 0: NO CREDENTIALS (default)
  Agent starts with nothing. Must request what it needs.
  
Level 1: READ-ONLY, RESOURCE-SCOPED
  Can read specific resources. Cannot modify anything.
  Example: Read customer order #12345, nothing else.

Level 2: WRITE, RESOURCE-SCOPED, TIME-LIMITED
  Can modify specific resources for limited time.
  Requires explicit approval (HITL or policy engine).
  Example: Update order #12345 status. Expires in 5 minutes.

Level 3: ELEVATED (rare, always audited)
  Can perform admin operations on specific resources.
  Always requires human approval. Always audited.
  Example: Issue refund for order #12345. Single-use token.
```

## Production Implementation

```python
"""
Credential scoping system for AI agents.
Issues minimum-privilege, session-scoped tokens.
"""
import secrets
import time
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    NONE = "none"
    READ = "read"
    WRITE = "write"
    ELEVATED = "elevated"


@dataclass
class ScopedToken:
    """A time-limited, resource-scoped credential for an agent."""
    token_id: str
    agent_id: str
    session_id: str
    permission: PermissionLevel
    resource_type: str        # e.g., "orders", "users", "payments"
    resource_id: Optional[str]  # Specific resource, or None for type-level
    allowed_actions: list[str]  # e.g., ["read", "update_status"]
    created_at: float
    expires_at: float
    single_use: bool = False
    used: bool = False
    approval_id: Optional[str] = None  # Who approved this token
    
    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at
    
    @property
    def is_valid(self) -> bool:
        if self.is_expired:
            return False
        if self.single_use and self.used:
            return False
        return True
    
    def ttl_seconds(self) -> float:
        return max(0, self.expires_at - time.time())


class CredentialManager:
    """
    Issues and validates scoped credentials for AI agents.
    """
    
    # Default TTLs by permission level
    DEFAULT_TTLS = {
        PermissionLevel.READ: 300,       # 5 minutes
        PermissionLevel.WRITE: 120,      # 2 minutes
        PermissionLevel.ELEVATED: 60,    # 1 minute
    }
    
    def __init__(self):
        self._active_tokens: dict[str, ScopedToken] = {}
        self._audit_log: list[dict] = []
    
    def issue_read_token(
        self,
        agent_id: str,
        session_id: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> ScopedToken:
        """
        Issue a read-only, resource-scoped token.
        No approval needed for read access.
        """
        ttl = ttl_seconds or self.DEFAULT_TTLS[PermissionLevel.READ]
        
        token = ScopedToken(
            token_id=f"tok_{secrets.token_urlsafe(16)}",
            agent_id=agent_id,
            session_id=session_id,
            permission=PermissionLevel.READ,
            resource_type=resource_type,
            resource_id=resource_id,
            allowed_actions=["read", "list", "search"],
            created_at=time.time(),
            expires_at=time.time() + ttl,
        )
        
        self._active_tokens[token.token_id] = token
        self._log("token_issued", token)
        
        logger.info(
            f"Read token issued: {token.token_id} for {resource_type}"
            f"{'/' + resource_id if resource_id else ''} "
            f"(TTL: {ttl}s)"
        )
        
        return token
    
    def issue_write_token(
        self,
        agent_id: str,
        session_id: str,
        resource_type: str,
        resource_id: str,  # Must be specific for writes
        allowed_actions: list[str],
        approval_id: str,  # Who approved this
        ttl_seconds: Optional[int] = None,
        single_use: bool = True,
    ) -> ScopedToken:
        """
        Issue a write token for a specific resource.
        REQUIRES approval_id -- someone must have approved this.
        """
        ttl = ttl_seconds or self.DEFAULT_TTLS[PermissionLevel.WRITE]
        
        token = ScopedToken(
            token_id=f"tok_{secrets.token_urlsafe(16)}",
            agent_id=agent_id,
            session_id=session_id,
            permission=PermissionLevel.WRITE,
            resource_type=resource_type,
            resource_id=resource_id,
            allowed_actions=allowed_actions,
            created_at=time.time(),
            expires_at=time.time() + ttl,
            single_use=single_use,
            approval_id=approval_id,
        )
        
        self._active_tokens[token.token_id] = token
        self._log("write_token_issued", token)
        
        logger.info(
            f"Write token issued: {token.token_id} for {resource_type}/{resource_id} "
            f"actions={allowed_actions} (TTL: {ttl}s, single_use: {single_use})"
        )
        
        return token
    
    def validate_action(
        self,
        token_id: str,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Validate whether a token permits a specific action.
        Returns validation result with detailed reason.
        """
        token = self._active_tokens.get(token_id)
        
        if not token:
            self._log("action_denied", None, reason="token_not_found")
            return {"allowed": False, "reason": "Token not found"}
        
        if not token.is_valid:
            reason = "expired" if token.is_expired else "already_used"
            self._log("action_denied", token, reason=reason)
            return {"allowed": False, "reason": f"Token {reason}"}
        
        if resource_type != token.resource_type:
            self._log("action_denied", token, reason="wrong_resource_type")
            return {"allowed": False, "reason": f"Token scoped to {token.resource_type}, not {resource_type}"}
        
        if token.resource_id and resource_id != token.resource_id:
            self._log("action_denied", token, reason="wrong_resource_id")
            return {"allowed": False, "reason": f"Token scoped to resource {token.resource_id}"}
        
        if action not in token.allowed_actions:
            self._log("action_denied", token, reason="action_not_allowed")
            return {"allowed": False, "reason": f"Action '{action}' not in allowed actions: {token.allowed_actions}"}
        
        # Mark as used if single-use
        if token.single_use:
            token.used = True
        
        self._log("action_allowed", token, action=action)
        return {"allowed": True, "token_id": token_id, "ttl_remaining": token.ttl_seconds()}
    
    def revoke_token(self, token_id: str) -> bool:
        """Explicitly revoke a token before expiry."""
        token = self._active_tokens.pop(token_id, None)
        if token:
            self._log("token_revoked", token)
            return True
        return False
    
    def revoke_session(self, session_id: str) -> int:
        """Revoke all tokens for a session."""
        to_revoke = [
            tid for tid, t in self._active_tokens.items()
            if t.session_id == session_id
        ]
        for tid in to_revoke:
            self.revoke_token(tid)
        return len(to_revoke)
    
    def cleanup_expired(self) -> int:
        """Remove expired tokens from memory."""
        expired = [
            tid for tid, t in self._active_tokens.items()
            if t.is_expired
        ]
        for tid in expired:
            del self._active_tokens[tid]
        return len(expired)
    
    def _log(self, event: str, token: Optional[ScopedToken], **extra):
        """Append to audit log."""
        entry = {
            "event": event,
            "timestamp": time.time(),
            "token_id": token.token_id if token else None,
            "agent_id": token.agent_id if token else None,
            "session_id": token.session_id if token else None,
            "resource": f"{token.resource_type}/{token.resource_id}" if token else None,
            **extra,
        }
        self._audit_log.append(entry)


# ============================================================
# Integration with Agent Tool Calls
# ============================================================

class ScopedToolExecutor:
    """
    Wraps tool execution with credential validation.
    Every tool call must present a valid scoped token.
    """
    
    def __init__(self, credential_manager: CredentialManager):
        self.cred_mgr = credential_manager
    
    async def execute_tool(
        self,
        tool_name: str,
        tool_args: dict,
        token_id: str,
    ) -> dict:
        """
        Execute a tool call only if the token permits it.
        """
        # Map tool names to resource types and actions
        tool_mapping = {
            "search_orders": ("orders", "read"),
            "get_order": ("orders", "read"),
            "update_order_status": ("orders", "update_status"),
            "issue_refund": ("payments", "refund"),
            "get_customer": ("customers", "read"),
            "send_email": ("communications", "send"),
        }
        
        if tool_name not in tool_mapping:
            return {"error": f"Unknown tool: {tool_name}"}
        
        resource_type, action = tool_mapping[tool_name]
        resource_id = tool_args.get("id") or tool_args.get("order_id")
        
        # Validate token
        validation = self.cred_mgr.validate_action(
            token_id=token_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
        )
        
        if not validation["allowed"]:
            logger.warning(
                f"Tool call blocked: {tool_name} - {validation['reason']}"
            )
            return {
                "error": f"Permission denied: {validation['reason']}",
                "tool": tool_name,
            }
        
        # Execute the actual tool
        # result = await actual_tool_function(tool_name, tool_args)
        result = {"status": "executed", "tool": tool_name}
        
        return result


# ============================================================
# Usage Example
# ============================================================

async def agent_workflow_example():
    cred_mgr = CredentialManager()
    executor = ScopedToolExecutor(cred_mgr)
    
    agent_id = "agent-cs-001"
    session_id = "session-abc123"
    
    # Step 1: Agent gets read token (no approval needed)
    read_token = cred_mgr.issue_read_token(
        agent_id=agent_id,
        session_id=session_id,
        resource_type="orders",
    )
    
    # Step 2: Agent reads order (allowed)
    result = await executor.execute_tool(
        "get_order",
        {"order_id": "ORD-12345"},
        read_token.token_id,
    )
    print(f"Read order: {result}")  # Works
    
    # Step 3: Agent tries to update with read token (blocked)
    result = await executor.execute_tool(
        "update_order_status",
        {"order_id": "ORD-12345", "status": "shipped"},
        read_token.token_id,
    )
    print(f"Update attempt: {result}")  # Blocked: action not allowed
    
    # Step 4: After human approval, agent gets write token
    write_token = cred_mgr.issue_write_token(
        agent_id=agent_id,
        session_id=session_id,
        resource_type="orders",
        resource_id="ORD-12345",
        allowed_actions=["update_status"],
        approval_id="human-jane-doe",
        single_use=True,
    )
    
    # Step 5: Agent updates order (allowed, single use)
    result = await executor.execute_tool(
        "update_order_status",
        {"order_id": "ORD-12345", "status": "shipped"},
        write_token.token_id,
    )
    print(f"Update with write token: {result}")  # Works
    
    # Step 6: Agent tries to update again (blocked, token used)
    result = await executor.execute_tool(
        "update_order_status",
        {"order_id": "ORD-12345", "status": "delivered"},
        write_token.token_id,
    )
    print(f"Second update: {result}")  # Blocked: already used
    
    # Cleanup
    cred_mgr.revoke_session(session_id)
```

## Decision Tree / When to Use

```
Does your agent call ANY external APIs or services?
  YES --> Credential scoping is MANDATORY
  
Does the agent modify data (write, update, delete)?
  YES --> Write tokens require approval + single-use + short TTL
  NO  --> Read-only tokens still needed (scope to specific resources)

Is the agent multi-tenant (handles multiple users)?
  YES --> Session-scoped tokens MUST be per-user, per-session
  NO  --> Still scope tokens, but simpler management
```

## When NOT to Use

- **Never skip credential scoping for production agents** -- even internal tools
- **Purely conversational agents with no tool access** -- no credentials needed at all
- **Development/testing environments** -- use broader tokens but still practice scoping

## Tradeoffs

| Approach | Security | Complexity | Latency | Developer Experience |
|----------|----------|-----------|---------|---------------------|
| Admin API key for agent | None | None | 0ms | Easy but dangerous |
| Role-based tokens | Medium | Low | ~5ms | Moderate |
| Session-scoped read-only | High | Medium | ~10ms | Good |
| Per-action single-use tokens | Very high | High | ~15ms | Requires workflow changes |
| Full credential manager | Highest | High | ~20ms | Requires infrastructure |

## Real-World Examples

1. **Stripe Agent Toolkit** -- Issues read-only API keys by default. Write operations (creating charges) require separate, scoped API keys with explicit capability grants.

2. **GitHub Copilot** -- Never receives user's GitHub token directly. Operations are mediated through a proxy that scopes permissions to the current repository.

3. **Salesforce Einstein** -- Agent tokens are scoped to the specific Salesforce org and object types the user has access to, with additional agent-specific restrictions.

## Failure Modes

| Failure | Cause | Impact | Prevention |
|---------|-------|--------|------------|
| **Over-scoped token** | Developer gives agent broad permissions for convenience | Compromised agent accesses everything | Audit token scopes regularly |
| **Token not rotated** | Long-lived tokens persist across sessions | Stolen token remains valid | Short TTLs + session-scoped |
| **Approval bypass** | Write token issued without proper approval flow | Unauthorized modifications | Enforce approval_id requirement |
| **Token leaked in logs** | Token appears in debug output | Token can be reused | Mask tokens in logs; short TTL limits window |

## Source(s) and Further Reading

- OWASP LLM06, Excessive Agency: https://genai.owasp.org/llmrisk/llm06-excessive-agency/
- Principle of Least Privilege (NIST): https://csrc.nist.gov/glossary/term/least_privilege
- Stripe Agent Toolkit: https://github.com/stripe/agent-toolkit
- OAuth 2.0 Token Best Practices (RFC 6819)
- Google Cloud, "Securing AI Agents with Workload Identity" (2024)
