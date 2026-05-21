# Signed Tool Calls

> HMAC signing of tool call payloads for auditability, tamper-proof trails, and replay attack prevention. Ensures tool calls originated from authorized agents and have not been modified.

## What It Is

Signed tool calls add a cryptographic signature to every tool invocation, proving:
1. **Authenticity**: The tool call was made by an authorized agent (not a spoofed request)
2. **Integrity**: The tool call payload was not modified in transit or storage
3. **Non-repudiation**: The agent cannot deny having made the call (audit trail)
4. **Replay protection**: A captured tool call cannot be re-executed (timestamp + nonce)

This matters for:
- **Financial tools**: Payment APIs, fund transfers, account modifications
- **Infrastructure tools**: Production deployment, database operations
- **Compliance-sensitive tools**: Data access in regulated industries (healthcare, finance)

### Why Sign Tool Calls?

```
WITHOUT signing:
  Agent → "delete_user(id=123)" → Tool executes
  
  Problems:
  1. Was it really the agent or a spoofed request?
  2. Was the original call "delete_user(id=123)" or was it "delete_user(id=456)" 
     and someone changed it?
  3. In the audit log, how do you prove the logged call is genuine?
  4. Can someone replay a captured "transfer_funds" call?

WITH signing:
  Agent → "delete_user(id=123)" + HMAC signature + timestamp + nonce → Verify → Execute
  
  Guarantees:
  1. Signature proves origin (agent had the signing key)
  2. Any payload modification invalidates the signature
  3. Audit log includes tamper-proof signatures
  4. Timestamp + nonce prevent replay attacks
```

## How It Works

### Signing Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    TOOL CALL SIGNING FLOW                     │
│                                                              │
│  1. Agent decides to call a tool                             │
│     │                                                        │
│  2. ┌──────────────────────────┐                            │
│     │ Build canonical payload   │                            │
│     │ {                         │                            │
│     │   "tool": "transfer_funds"│                            │
│     │   "args": {"amount": 100} │                            │
│     │   "timestamp": 1705312000 │                            │
│     │   "nonce": "abc123"       │                            │
│     │   "agent_id": "agent-42"  │                            │
│     │ }                         │                            │
│     └───────────┬──────────────┘                            │
│                 │                                            │
│  3. ┌───────────▼──────────────┐                            │
│     │ HMAC-SHA256(key, payload) │                            │
│     │ → signature: "7f3a..."    │                            │
│     └───────────┬──────────────┘                            │
│                 │                                            │
│  4. ┌───────────▼──────────────┐                            │
│     │ Send: payload + signature │──► Tool Executor           │
│     └──────────────────────────┘    │                        │
│                                      │                       │
│  5. Tool Executor:                   │                       │
│     ┌────────────────────────────┐  │                       │
│     │ Verify signature            │  │                       │
│     │ Check timestamp (±30s)      │  │                       │
│     │ Check nonce uniqueness      │  │                       │
│     │ Execute if all pass         │  │                       │
│     └────────────────────────────┘  │                       │
│                                      │                       │
│  6. Store in audit log:              │                       │
│     {payload, signature, result,     │                       │
│      verified: true, executed_at}    │                       │
└──────────────────────────────────────────────────────────────┘
```

## Production Implementation

```python
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional
from functools import wraps


@dataclass
class SignedToolCall:
    """A cryptographically signed tool call."""
    tool_name: str
    arguments: dict
    agent_id: str
    session_id: str
    timestamp: float
    nonce: str
    signature: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "signature": self.signature,
            "metadata": self.metadata,
        }


class ToolCallSigner:
    """
    HMAC-SHA256 signing and verification for tool calls.
    
    Features:
    - HMAC-SHA256 signatures on canonical payloads
    - Timestamp validation (prevents replay after window)
    - Nonce tracking (prevents replay within window)
    - Per-agent signing keys
    - Audit log integration
    """

    def __init__(
        self,
        signing_key: bytes,
        replay_window_seconds: float = 30.0,
        max_nonce_cache: int = 10_000,
    ):
        self.signing_key = signing_key
        self.replay_window = replay_window_seconds
        self.max_nonce_cache = max_nonce_cache
        self._used_nonces: dict[str, float] = {}  # nonce → timestamp
        self._audit_log: list[dict] = []

    def _canonicalize(self, tool_name: str, arguments: dict,
                      agent_id: str, session_id: str,
                      timestamp: float, nonce: str) -> bytes:
        """
        Create a canonical representation of the tool call for signing.
        JSON with sorted keys ensures deterministic serialization.
        """
        canonical = {
            "tool_name": tool_name,
            "arguments": arguments,
            "agent_id": agent_id,
            "session_id": session_id,
            "timestamp": timestamp,
            "nonce": nonce,
        }
        return json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()

    def sign(
        self,
        tool_name: str,
        arguments: dict,
        agent_id: str,
        session_id: str,
    ) -> SignedToolCall:
        """Sign a tool call, producing a signature."""
        timestamp = time.time()
        nonce = str(uuid.uuid4())

        payload = self._canonicalize(
            tool_name, arguments, agent_id, session_id, timestamp, nonce
        )
        signature = hmac.new(self.signing_key, payload, hashlib.sha256).hexdigest()

        return SignedToolCall(
            tool_name=tool_name,
            arguments=arguments,
            agent_id=agent_id,
            session_id=session_id,
            timestamp=timestamp,
            nonce=nonce,
            signature=signature,
        )

    def verify(self, signed_call: SignedToolCall) -> tuple[bool, str]:
        """
        Verify a signed tool call.
        
        Checks:
        1. Signature is valid (payload not tampered)
        2. Timestamp is within replay window
        3. Nonce has not been used before
        
        Returns: (is_valid, reason)
        """
        # 1. Timestamp check (prevent old replays)
        age = time.time() - signed_call.timestamp
        if age > self.replay_window:
            return False, f"Timestamp too old ({age:.0f}s > {self.replay_window}s)"
        if age < -5:  # Allow 5s clock skew
            return False, f"Timestamp in future ({age:.0f}s)"

        # 2. Nonce check (prevent replay within window)
        if signed_call.nonce in self._used_nonces:
            return False, f"Nonce already used: {signed_call.nonce}"

        # 3. Signature verification
        payload = self._canonicalize(
            signed_call.tool_name,
            signed_call.arguments,
            signed_call.agent_id,
            signed_call.session_id,
            signed_call.timestamp,
            signed_call.nonce,
        )
        expected_sig = hmac.new(
            self.signing_key, payload, hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signed_call.signature, expected_sig):
            return False, "Invalid signature (payload may have been tampered)"

        # Mark nonce as used
        self._used_nonces[signed_call.nonce] = signed_call.timestamp
        self._cleanup_nonces()

        return True, "Valid"

    def _cleanup_nonces(self):
        """Remove expired nonces to prevent memory growth."""
        now = time.time()
        expired = [
            nonce for nonce, ts in self._used_nonces.items()
            if now - ts > self.replay_window * 2
        ]
        for nonce in expired:
            del self._used_nonces[nonce]

        # Hard cap
        if len(self._used_nonces) > self.max_nonce_cache:
            oldest = sorted(self._used_nonces.items(), key=lambda x: x[1])
            for nonce, _ in oldest[:len(oldest) // 2]:
                del self._used_nonces[nonce]

    def execute_signed(
        self,
        signed_call: SignedToolCall,
        tool_registry: dict[str, callable],
    ) -> dict:
        """Verify and execute a signed tool call."""
        is_valid, reason = self.verify(signed_call)

        audit_entry = {
            "tool_name": signed_call.tool_name,
            "agent_id": signed_call.agent_id,
            "session_id": signed_call.session_id,
            "timestamp": signed_call.timestamp,
            "signature": signed_call.signature,
            "verified": is_valid,
            "verification_reason": reason,
            "executed": False,
            "result": None,
            "error": None,
        }

        if not is_valid:
            audit_entry["error"] = f"Verification failed: {reason}"
            self._audit_log.append(audit_entry)
            raise PermissionError(f"Tool call verification failed: {reason}")

        # Execute the tool
        tool_func = tool_registry.get(signed_call.tool_name)
        if not tool_func:
            audit_entry["error"] = f"Unknown tool: {signed_call.tool_name}"
            self._audit_log.append(audit_entry)
            raise ValueError(f"Unknown tool: {signed_call.tool_name}")

        try:
            result = tool_func(**signed_call.arguments)
            audit_entry["executed"] = True
            audit_entry["result"] = str(result)[:500]
            self._audit_log.append(audit_entry)
            return {"success": True, "result": result}
        except Exception as e:
            audit_entry["executed"] = True
            audit_entry["error"] = str(e)
            self._audit_log.append(audit_entry)
            raise

    def get_audit_log(self) -> list[dict]:
        return self._audit_log


# --- Decorator for Automatic Signing ---

def signed_tool(signer: ToolCallSigner, agent_id: str, session_id: str):
    """
    Decorator that automatically signs and verifies tool calls.
    
    Usage:
        signer = ToolCallSigner(signing_key=b"secret")
        
        @signed_tool(signer, agent_id="agent-1", session_id="sess-1")
        def transfer_funds(from_account: str, to_account: str, amount: float):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(**kwargs):
            # Sign
            signed = signer.sign(
                tool_name=func.__name__,
                arguments=kwargs,
                agent_id=agent_id,
                session_id=session_id,
            )

            # Verify and execute
            registry = {func.__name__: func}
            result = signer.execute_signed(signed, registry)
            return result["result"]

        return wrapper
    return decorator


# --- Integration with Agent Loop ---

class SignedToolExecutor:
    """
    Drop-in replacement for tool execution that adds signing.
    Integrates with LangChain/LangGraph tool execution.
    """

    def __init__(self, signing_key: bytes, agent_id: str):
        self.signer = ToolCallSigner(signing_key)
        self.agent_id = agent_id
        self.session_id = str(uuid.uuid4())

    def execute(
        self,
        tool_name: str,
        arguments: dict,
        tool_registry: dict[str, callable],
    ) -> Any:
        """Sign and execute a tool call."""
        signed = self.signer.sign(
            tool_name=tool_name,
            arguments=arguments,
            agent_id=self.agent_id,
            session_id=self.session_id,
        )

        result = self.signer.execute_signed(signed, tool_registry)
        return result["result"]

    def export_audit_log(self) -> list[dict]:
        """Export the tamper-proof audit log."""
        return self.signer.get_audit_log()
```

## Decision Tree: When to Sign Tool Calls

```
    Should I sign this tool call?
                    │
         ┌──────────▼──────────┐
         │ Does the tool modify │
         │ external state?      │
         │ (write, delete,     │
         │  send, deploy)       │
         └──┬──────────────┬──┘
           Yes             No
            │               │
    ┌───────▼──────┐   Not required
    │ Is there a    │   (but still good
    │ compliance    │    practice for
    │ requirement?  │    observability)
    │ (SOC2, HIPAA, │
    │  PCI-DSS)     │
    └──┬────────┬──┘
      Yes      No
       │        │
   REQUIRED  ┌──▼────────────┐
              │ Is the tool   │
              │ high-impact?  │
              │ (payments,    │
              │  deployments) │
              └──┬────────┬──┘
                Yes      No
                 │        │
           RECOMMENDED  Optional
```

## When NOT to Use

1. **Read-only tools**: Search, lookup, and query tools do not modify state. Signing adds overhead without security benefit.
2. **Internal development**: During development, signing adds friction. Enable only in staging/production.
3. **High-frequency low-impact tools**: Tools called 1000+/second (metrics collection, logging) where signing overhead matters.
4. **Trusted internal networks**: If the agent and tools run in the same process with no network boundary, signing is unnecessary.
5. **Ephemeral/disposable environments**: Sandbox environments where tool calls have no lasting impact.

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Tamper-proof audit trail | Adds ~0.1ms latency per tool call |
| Replay attack prevention | Key management complexity |
| Compliance-friendly (SOC2, HIPAA) | Nonce storage requires cleanup |
| Proves tool call origin | Over-engineering for low-risk tools |
| Integrates with existing HMAC infrastructure | Signing key compromise = security bypass |

## Failure Modes

### 1. Clock Skew
Agent and verifier have different system clocks, causing valid calls to be rejected.
**Mitigation**: Allow 5-10 second clock skew tolerance. Use NTP synchronization.

### 2. Signing Key Compromise
If the signing key leaks, anyone can forge tool calls.
**Mitigation**: Key rotation schedule (monthly). Per-agent keys (limit blast radius). Store keys in KMS.

### 3. Nonce Storage Overflow
In long-running systems, the nonce cache grows unboundedly.
**Mitigation**: TTL-based cleanup. Hard cap with LRU eviction. Store nonces in Redis with TTL.

### 4. Canonical Form Divergence
Signer and verifier serialize arguments differently, causing signature mismatch.
**Mitigation**: Use `json.dumps(sort_keys=True, separators=(',',':'))` everywhere. Test canonical form.

### 5. Performance Impact at Scale
Signing adds HMAC computation + nonce lookup per call.
**Mitigation**: HMAC-SHA256 is ~0.01ms. Only sign write/delete tools, not reads.

## Sources and Further Reading

- [HMAC - RFC 2104](https://datatracker.ietf.org/doc/html/rfc2104)
- [AWS Request Signing (SigV4)](https://docs.aws.amazon.com/general/latest/gr/signing-aws-api-requests.html)
- [Stripe API Request Signing](https://stripe.com/docs/webhooks/signatures)
- [OWASP API Security - Integrity](https://owasp.org/API-Security/)
- [SOC 2 Type II - Audit Logging Requirements](https://us.aicpa.org/interestareas/frc/assuranceadvisoryservices/sorhome)
