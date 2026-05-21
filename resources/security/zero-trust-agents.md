# Zero-Trust Principles for Agents
> Never trust the LLM's decisions implicitly -- verify every tool call, scope every credential, log every action, and approve every write.

## What It Is

Zero-trust for agents applies the "never trust, always verify" principle to every aspect of agent execution. Unlike traditional zero-trust networking (which focuses on network perimeters), agent zero-trust focuses on the decision boundary between the LLM's nondeterministic outputs and the deterministic execution of tools and actions.

The 5 principles: (1) no permanent credentials, (2) all tool calls through auth layer, (3) network egress allowlisted, (4) every action logged with reasoning, (5) write ops require approval or idempotency.

## How It Works

### The 5 Zero-Trust Principles

```
Principle 1: No Permanent Credentials
├── Agent receives session-scoped tokens, not long-lived API keys
├── Tokens expire with the session (max 1 hour)
├── Credentials never appear in conversation context or memory
└── Rotated on every session start

Principle 2: All Tool Calls Through Auth Layer
├── Every tool call passes through authorization middleware
├── Tool access is scoped per-role, per-tenant, per-session
├── Rate limits per tool per session
└── No direct tool invocation bypassing the auth layer

Principle 3: Network Egress Allowlisted
├── Agent cannot make arbitrary outbound HTTP requests
├── Allowed domains explicitly listed
├── DNS-level enforcement (not just app-level)
└── Blocked: agent cannot exfiltrate data to unknown endpoints

Principle 4: Every Action Logged with Reasoning
├── Every tool call logged with: who, what, when, why
├── "Why" = the LLM's reasoning that led to the tool call
├── Immutable audit log (append-only, tamper-evident)
└── Queryable for incident investigation

Principle 5: Write Ops Require Approval or Idempotency
├── Read operations: auto-approved (but logged)
├── Idempotent writes: auto-approved with idempotency key
├── Non-idempotent writes: require human approval
├── Destructive operations: always require human approval
└── Financial operations: always require human approval
```

## Production Implementation

```python
import time
import uuid
import hashlib
import hmac
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# --- Principle 1: Session-Scoped Credentials ---

@dataclass
class SessionCredential:
    token: str
    tool_name: str
    tenant_id: str
    session_id: str
    scopes: list[str]  # e.g., ["read", "write:limited"]
    expires_at: float
    created_at: float = field(default_factory=time.time)
    
    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class CredentialBroker:
    """Issues short-lived, scoped credentials for tool access."""
    
    def __init__(self, vault_client, default_ttl: int = 3600):
        self.vault = vault_client
        self.default_ttl = default_ttl
    
    async def issue_credential(
        self,
        tool_name: str,
        tenant_id: str,
        session_id: str,
        scopes: list[str],
    ) -> SessionCredential:
        """Issue a session-scoped credential for a specific tool."""
        # Fetch base credential from Vault
        base_cred = await self.vault.read(
            f"secret/agents/{tenant_id}/tools/{tool_name}"
        )
        
        # Generate session-scoped token
        # Option A: Use Vault dynamic secrets (preferred)
        # dynamic_cred = await self.vault.write(
        #     f"database/creds/{tenant_id}-{tool_name}",
        #     ttl=self.default_ttl
        # )
        
        # Option B: Wrap the base credential with session metadata
        token = self._create_session_token(
            base_cred, tenant_id, session_id, scopes
        )
        
        return SessionCredential(
            token=token,
            tool_name=tool_name,
            tenant_id=tenant_id,
            session_id=session_id,
            scopes=scopes,
            expires_at=time.time() + self.default_ttl,
        )
    
    def _create_session_token(
        self, base_cred: dict, tenant_id: str, session_id: str, scopes: list
    ) -> str:
        """Create a signed, scoped session token."""
        payload = f"{tenant_id}:{session_id}:{','.join(scopes)}:{time.time()}"
        signature = hmac.new(
            base_cred["signing_key"].encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()
        return f"{payload}:{signature}"


# --- Principle 2: Tool Authorization Layer ---

class ToolAction(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    COMMUNICATE = "communicate"  # Send email, post message


TOOL_ACTION_MAP = {
    "search_knowledge_base": ToolAction.READ,
    "get_user_profile": ToolAction.READ,
    "update_ticket": ToolAction.WRITE,
    "send_email": ToolAction.COMMUNICATE,
    "execute_code": ToolAction.EXECUTE,
    "delete_record": ToolAction.DELETE,
    "create_payment": ToolAction.WRITE,
}


class ToolAuthorizationLayer:
    """Central authorization for all tool calls."""
    
    def __init__(self, credential_broker: CredentialBroker):
        self.broker = credential_broker
        self._session_creds: dict[str, SessionCredential] = {}
    
    async def authorize_and_execute(
        self,
        tool_name: str,
        args: dict,
        tenant_id: str,
        session_id: str,
        agent_reasoning: str,
    ) -> dict:
        """Authorize then execute a tool call. Returns result or rejection."""
        
        # 1. Get or issue session credential
        cred_key = f"{session_id}:{tool_name}"
        cred = self._session_creds.get(cred_key)
        
        if not cred or cred.is_expired:
            action = TOOL_ACTION_MAP.get(tool_name, ToolAction.READ)
            scopes = self._scopes_for_action(action)
            cred = await self.broker.issue_credential(
                tool_name, tenant_id, session_id, scopes
            )
            self._session_creds[cred_key] = cred
        
        # 2. Check scopes
        action = TOOL_ACTION_MAP.get(tool_name, ToolAction.READ)
        required_scope = action.value
        if required_scope not in cred.scopes:
            return {
                "error": "insufficient_scope",
                "message": f"Credential lacks '{required_scope}' scope for {tool_name}",
            }
        
        # 3. Log the authorized call (Principle 4)
        await self._log_tool_call(
            tool_name=tool_name,
            args=args,
            tenant_id=tenant_id,
            session_id=session_id,
            reasoning=agent_reasoning,
            action=action,
        )
        
        # 4. Execute
        result = await execute_tool(tool_name, args, credential=cred.token)
        return result
    
    def _scopes_for_action(self, action: ToolAction) -> list[str]:
        scope_map = {
            ToolAction.READ: ["read"],
            ToolAction.WRITE: ["read", "write"],
            ToolAction.DELETE: ["read", "write", "delete"],
            ToolAction.EXECUTE: ["execute"],
            ToolAction.COMMUNICATE: ["communicate"],
        }
        return scope_map.get(action, ["read"])
    
    async def _log_tool_call(self, **kwargs):
        """Immutable audit log entry."""
        log_entry = {
            "timestamp": time.time(),
            "trace_id": str(uuid.uuid4()),
            **kwargs,
        }
        # Write to append-only audit log
        await self.audit_log.append(log_entry)


# --- Principle 3: Network Egress Control ---

class EgressController:
    """Control outbound network access from agent tools."""
    
    def __init__(self, allowed_domains: list[str]):
        self.allowed = set(allowed_domains)
    
    def check_url(self, url: str) -> tuple[bool, str]:
        """Check if an outbound URL is allowed."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.hostname
        
        if not domain:
            return False, "Invalid URL: no hostname"
        
        # Check exact match and wildcard subdomains
        for allowed in self.allowed:
            if domain == allowed or domain.endswith(f".{allowed}"):
                return True, "OK"
        
        return False, f"Domain '{domain}' not in egress allowlist"


# Default allowlist for a typical agent
DEFAULT_EGRESS_ALLOWLIST = [
    "api.anthropic.com",        # LLM provider
    "api.openai.com",           # LLM provider fallback
    "*.pinecone.io",            # Vector DB
    "api.github.com",           # Code tools
    "googleapis.com",           # Google APIs
    # Add tenant-specific domains
]


# --- Principle 5: Write Operation Approval ---

class WriteApprovalPolicy:
    """Determine if a write operation needs human approval."""
    
    # Operations that are always auto-approved (idempotent reads)
    AUTO_APPROVE = {ToolAction.READ}
    
    # Operations that need approval based on context
    CONTEXT_DEPENDENT = {ToolAction.WRITE, ToolAction.EXECUTE}
    
    # Operations that always need approval
    ALWAYS_APPROVE = {ToolAction.DELETE, ToolAction.COMMUNICATE}
    
    def needs_approval(
        self,
        action: ToolAction,
        tool_name: str,
        args: dict,
        idempotency_key: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Check if this operation needs human approval."""
        
        if action in self.AUTO_APPROVE:
            return False, "Read operations are auto-approved"
        
        if action in self.ALWAYS_APPROVE:
            return True, f"{action.value} operations always require approval"
        
        # Context-dependent: check if idempotent
        if idempotency_key:
            return False, "Idempotent write with key, auto-approved"
        
        # Check for high-value operations
        if tool_name in {"create_payment", "modify_subscription", "update_billing"}:
            return True, "Financial operations require approval"
        
        return False, "Standard write operation, auto-approved"
```

## Decision Tree: Applying Zero-Trust

```
New agent capability being added:
│
├── Does it read data?
│   └── Apply: Principle 2 (auth layer) + Principle 4 (logging)
│
├── Does it write data?
│   ├── Idempotent? → Auto-approve with idempotency key
│   └── Non-idempotent? → Require human approval
│
├── Does it access external APIs?
│   └── Apply: Principle 1 (session creds) + Principle 3 (egress allowlist)
│
├── Does it handle sensitive data?
│   └── Apply: All 5 principles + PII scrubbing + encryption at rest
│
└── Does it make financial transactions?
    └── Apply: All 5 principles + mandatory approval + dual-control
```

## When NOT to Apply Full Zero-Trust

- **Development environments**: Session-scoped credentials and approval gates slow down development. Use a permissive "dev mode" with logging.
- **Internal research tools**: If the agent is only used by the team building it, full zero-trust is overkill.
- **Hackathons/POCs**: Ship first, secure later. But plan for security from the start.

## Tradeoffs

| Principle | Security Benefit | UX/Performance Cost | Implementation Effort |
|-----------|-----------------|---------------------|----------------------|
| Session-scoped creds | Limits blast radius | Credential refresh latency | High (Vault integration) |
| Auth layer | Prevents unauthorized actions | +10-50ms per tool call | Medium |
| Egress allowlist | Prevents data exfiltration | Limits agent capabilities | Low |
| Full audit logging | Incident investigation, compliance | Storage costs, write latency | Medium |
| Write approval | Prevents destructive actions | User waits for approval | Medium |

## Failure Modes

1. **Overly restrictive egress**: Agent can't access a legitimate API because it wasn't in the allowlist. Mitigation: monitoring for blocked requests, easy allowlist updates.
2. **Session credential clock skew**: Token expires mid-session due to clock drift between services. Mitigation: 30-second grace period, automatic refresh.
3. **Audit log storage explosion**: Logging every reasoning chain consumes massive storage. Mitigation: sampling for low-risk reads, full logging for writes.
4. **Approval fatigue**: Human approvers start rubber-stamping everything. Mitigation: risk-based approval (only high-risk), clear context in approval request.

## Source(s) and Further Reading

- NIST Zero Trust Architecture (SP 800-207): https://csrc.nist.gov/publications/detail/sp/800-207/final
- HashiCorp Vault Dynamic Secrets: https://developer.hashicorp.com/vault/docs/secrets
- OWASP Top 10 for LLMs: https://owasp.org/www-project-top-10-for-large-language-model-applications/
- "Zero Trust Architecture for AI Systems" - MITRE (2024)
