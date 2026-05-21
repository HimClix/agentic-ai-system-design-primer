# Credential Management for Agents
> Never store credentials in agent memory or conversation context -- use session-scoped tokens from a secrets manager with per-tool scoping.

## What It Is

Credential management for agents addresses the unique challenge of providing LLM-powered agents with access to external services without exposing long-lived secrets. Traditional apps read credentials once at startup; agents need credentials dynamically based on which tools the LLM decides to call, scoped to the current tenant and session.

The cardinal rule: **credentials must never appear in the LLM's context window**. If a credential enters the conversation history, it can be leaked in responses, logged in observability tools, or persisted in memory stores.

## How It Works

### Credential Flow Architecture

```
Agent decides to call a tool
         │
         ▼
┌─────────────────────┐
│ Tool Execution Layer │ (not the LLM)
│ "I need creds for    │
│  tool X, tenant Y"   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Credential Broker    │
│ 1. Authenticate      │
│    agent session      │
│ 2. Check entitlements │
│ 3. Fetch from Vault   │
│ 4. Issue scoped token │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ HashiCorp Vault      │
│ / AWS Secrets Manager│
│ / GCP Secret Manager │
└────────┬────────────┘
         │ session-scoped credential
         ▼
┌─────────────────────┐
│ Tool Executor        │
│ Uses credential to   │
│ call external API     │
│ Credential NEVER sent │
│ to LLM context       │
└─────────────────────┘
```

### What the LLM Sees vs What Actually Happens

```
LLM's perspective:
  "I'll call search_database(query='SELECT name FROM users WHERE id=123')"
  → Result: {"name": "Jane Doe"}

Actual execution:
  1. LLM outputs tool call JSON (no credentials in it)
  2. Tool executor intercepts
  3. Credential broker fetches DB password from Vault
  4. Tool executor connects to DB with credential
  5. Executes query
  6. Returns result to LLM (credential not included)
```

## Production Implementation

```python
import time
import asyncio
from dataclasses import dataclass
from typing import Optional, Any
from contextlib import asynccontextmanager


@dataclass
class ScopedCredential:
    """A short-lived, scoped credential for tool access."""
    service: str
    credential_type: str  # "api_key", "oauth_token", "db_password"
    value: str  # The actual secret (never expose to LLM)
    scopes: list[str]
    tenant_id: str
    session_id: str
    expires_at: float
    lease_id: Optional[str] = None  # Vault lease for revocation
    
    @property
    def is_valid(self) -> bool:
        return time.time() < self.expires_at


class VaultCredentialManager:
    """Manages agent credentials via HashiCorp Vault."""
    
    def __init__(self, vault_addr: str, vault_token: str):
        import hvac
        self.client = hvac.Client(url=vault_addr, token=vault_token)
        self._cache: dict[str, ScopedCredential] = {}
    
    async def get_credential(
        self,
        service: str,
        tenant_id: str,
        session_id: str,
        credential_type: str = "api_key",
        ttl: int = 3600,
    ) -> ScopedCredential:
        """Fetch or cache a scoped credential."""
        cache_key = f"{tenant_id}:{session_id}:{service}"
        
        # Check cache
        cached = self._cache.get(cache_key)
        if cached and cached.is_valid:
            return cached
        
        # Fetch from Vault
        if credential_type == "db_password":
            cred = await self._get_dynamic_db_credential(service, tenant_id, ttl)
        elif credential_type == "oauth_token":
            cred = await self._get_oauth_token(service, tenant_id, ttl)
        else:
            cred = await self._get_static_secret(service, tenant_id, ttl)
        
        cred.session_id = session_id
        self._cache[cache_key] = cred
        return cred
    
    async def _get_static_secret(
        self, service: str, tenant_id: str, ttl: int
    ) -> ScopedCredential:
        """Read a static secret from Vault KV store."""
        path = f"secret/data/agents/{tenant_id}/{service}"
        response = self.client.secrets.kv.v2.read_secret_version(path=path)
        data = response["data"]["data"]
        
        return ScopedCredential(
            service=service,
            credential_type="api_key",
            value=data["api_key"],
            scopes=data.get("scopes", ["read"]),
            tenant_id=tenant_id,
            session_id="",
            expires_at=time.time() + ttl,
        )
    
    async def _get_dynamic_db_credential(
        self, service: str, tenant_id: str, ttl: int
    ) -> ScopedCredential:
        """Get a dynamic database credential (auto-rotated by Vault)."""
        role = f"{tenant_id}-{service}-readonly"
        response = self.client.secrets.database.generate_credentials(name=role)
        
        return ScopedCredential(
            service=service,
            credential_type="db_password",
            value=f"{response['data']['username']}:{response['data']['password']}",
            scopes=["read"],
            tenant_id=tenant_id,
            session_id="",
            expires_at=time.time() + response["lease_duration"],
            lease_id=response["lease_id"],
        )
    
    async def revoke_session_credentials(self, session_id: str):
        """Revoke all credentials issued for a session."""
        to_revoke = [
            k for k, v in self._cache.items() 
            if v.session_id == session_id
        ]
        for key in to_revoke:
            cred = self._cache.pop(key)
            if cred.lease_id:
                try:
                    self.client.sys.revoke_lease(cred.lease_id)
                except Exception:
                    pass  # Best effort revocation


# --- Kubernetes Secrets Integration ---

class K8sSecretManager:
    """For deployments using Kubernetes secrets instead of Vault."""
    
    def __init__(self):
        from kubernetes import client, config
        config.load_incluster_config()
        self.v1 = client.CoreV1Api()
    
    async def get_credential(
        self, service: str, tenant_id: str, namespace: str = "agents"
    ) -> dict:
        """Read credential from Kubernetes secret."""
        secret_name = f"agent-creds-{tenant_id}-{service}"
        
        try:
            secret = self.v1.read_namespaced_secret(secret_name, namespace)
            import base64
            return {
                k: base64.b64decode(v).decode()
                for k, v in secret.data.items()
            }
        except Exception as e:
            raise ValueError(f"Secret {secret_name} not found: {e}")


# --- Tool Executor with Credential Injection ---

class SecureToolExecutor:
    """Executes tools with credential injection -- LLM never sees credentials."""
    
    def __init__(self, cred_manager: VaultCredentialManager):
        self.cred_manager = cred_manager
        self._tool_registry: dict[str, dict] = {}
    
    def register_tool(self, name: str, fn: callable, service: str, cred_type: str):
        self._tool_registry[name] = {
            "fn": fn,
            "service": service,
            "cred_type": cred_type,
        }
    
    async def execute(
        self, tool_name: str, args: dict, tenant_id: str, session_id: str
    ) -> Any:
        """Execute a tool with injected credentials."""
        tool_info = self._tool_registry.get(tool_name)
        if not tool_info:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        # Get credential (LLM never sees this)
        credential = await self.cred_manager.get_credential(
            service=tool_info["service"],
            tenant_id=tenant_id,
            session_id=session_id,
            credential_type=tool_info["cred_type"],
        )
        
        # Execute with credential injected
        result = await tool_info["fn"](
            **args,
            _credential=credential.value,  # Internal parameter, not from LLM
        )
        
        # Scrub any credential leaks from result
        result_str = str(result)
        if credential.value in result_str:
            raise SecurityError("Credential detected in tool output!")
        
        return result
    
    @asynccontextmanager
    async def session_scope(self, session_id: str):
        """Context manager that revokes all session credentials on exit."""
        try:
            yield
        finally:
            await self.cred_manager.revoke_session_credentials(session_id)
```

### Vault Configuration for Agent Credentials

```hcl
# Vault policy for agent credential access
path "secret/data/agents/{{identity.entity.metadata.tenant_id}}/*" {
  capabilities = ["read"]
}

# Dynamic database credentials
path "database/creds/{{identity.entity.metadata.tenant_id}}-*-readonly" {
  capabilities = ["read"]
}

# Deny access to other tenants' credentials
path "secret/data/agents/*" {
  capabilities = ["deny"]
}
```

## Decision Tree: Credential Strategy

```
What kind of external service?
│
├── Database (SQL/NoSQL)
│   └── Vault dynamic credentials (auto-rotating passwords)
│       ├── Read-only role for queries
│       └── Read-write role for mutations (if approved)
│
├── SaaS API (Slack, GitHub, etc.)
│   └── OAuth tokens stored in Vault KV
│       ├── Refresh token in Vault
│       └── Short-lived access token issued per session
│
├── Internal microservice
│   └── mTLS or service mesh identity
│       └── No credential management needed
│
└── Cloud provider (AWS, GCP)
    └── IAM role assumption
        ├── EKS IRSA (AWS)
        └── Workload Identity (GCP)
```

## When NOT to Use Vault-Based Credential Management

- **Single-tenant self-hosted**: Customer manages their own credentials. Pass via environment variables.
- **Development/testing**: Use .env files or Kubernetes secrets. Vault is overkill for dev.
- **No external integrations**: Agent only calls internal services with service mesh auth.

## Tradeoffs

| Approach | Security | Complexity | Latency | Cost |
|----------|----------|-----------|---------|------|
| Environment variables | Low | Lowest | None | Free |
| Kubernetes secrets | Medium | Low | None | Free |
| AWS Secrets Manager | High | Medium | +10ms | $0.40/secret/month |
| HashiCorp Vault | Highest | High | +20ms | $0 (OSS) or enterprise |

## Failure Modes

1. **Vault unavailable**: Agent can't get credentials, all tool calls fail. Mitigation: credential caching with TTL, fallback to cached credentials.
2. **Credential leak in logs**: Tool output includes credentials. Mitigation: output scrubbing, log filtering.
3. **Dynamic credential exhaustion**: Vault rate-limits dynamic credential generation. Mitigation: credential caching, connection pooling.
4. **Session cleanup failure**: Credentials not revoked after session ends. Mitigation: TTL-based auto-expiry as safety net, cleanup job.

## Source(s) and Further Reading

- HashiCorp Vault: https://developer.hashicorp.com/vault
- AWS Secrets Manager: https://aws.amazon.com/secrets-manager/
- Kubernetes Secrets: https://kubernetes.io/docs/concepts/configuration/secret/
- "Secrets Management Best Practices" - HashiCorp
- OWASP Secrets Management Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html
