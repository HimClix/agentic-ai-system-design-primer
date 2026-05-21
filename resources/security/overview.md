# Security for Agentic Systems
> Agent security is fundamentally different from traditional app security -- agents make autonomous decisions about which tools to call, what data to access, and what actions to take.

## What It Is

Security for agentic AI systems addresses the unique risks that arise when an LLM-powered agent has access to tools, databases, APIs, and can take actions on behalf of users. Traditional application security assumes deterministic code paths; agent security must account for nondeterministic decision-making by the LLM.

The core difference: in traditional apps, **code decides** what to do. In agents, the **LLM decides**, and the LLM can be manipulated via prompt injection, confused by ambiguous inputs, or hallucinate tool calls that shouldn't happen.

## How It Works

### Traditional vs Agent Security Comparison

| Dimension | Traditional App | Agentic System |
|-----------|---------------|---------------|
| **Auth** | User authenticates, app uses fixed credentials | Agent needs per-tool, per-tenant, session-scoped credentials |
| **Input validation** | Validate form fields against schema | Validate natural language (prompt injection, jailbreak) |
| **Data access** | Query builder with parameterized SQL | LLM generates queries -- must constrain scope |
| **Action authorization** | Role-based on endpoints | Per-tool-call authorization, confidence-based gates |
| **Audit** | Log API calls | Log every LLM decision + reasoning chain |
| **Secrets** | Env vars / Vault, app reads once | Never put secrets in agent memory/context |
| **Execution sandbox** | Docker containers | Sandboxed code execution + network egress control |
| **Attack surface** | Known endpoints | Open-ended natural language input |

### Agent-Specific Threat Model

```
THREAT CATEGORIES:

1. Prompt Injection (Direct + Indirect)
   ├── Direct: User crafts input to override system prompt
   ├── Indirect: Malicious content in retrieved documents/tools
   └── Impact: Agent takes unauthorized actions

2. Credential Exposure
   ├── Agent leaks API keys in responses
   ├── Credentials stored in conversation history
   └── Impact: Unauthorized access to external services

3. Excessive Tool Access
   ├── Agent calls tools it shouldn't have access to
   ├── Agent escalates privileges via tool chaining
   └── Impact: Data breach, unauthorized mutations

4. Data Exfiltration
   ├── Agent extracts sensitive data via tool responses
   ├── Agent passes PII to external APIs
   └── Impact: Privacy violation, compliance breach

5. Denial of Service
   ├── Agent enters infinite loop (FM-1.3)
   ├── Agent makes excessive API calls
   └── Impact: Cost overrun, service degradation
```

## Production Implementation

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import re


class RiskLevel(Enum):
    LOW = "low"          # Read-only, internal data
    MEDIUM = "medium"    # Write to internal systems
    HIGH = "high"        # External communications, financial
    CRITICAL = "critical" # Production deployments, data deletion


@dataclass
class SecurityPolicy:
    """Per-agent security policy."""
    allowed_tools: set[str]
    blocked_tools: set[str] = None
    max_tool_calls_per_session: int = 50
    require_approval_for_risk: RiskLevel = RiskLevel.HIGH
    allow_network_egress: bool = False
    egress_allowlist: list[str] = None  # Domains allowed for outbound calls
    pii_scrub_outputs: bool = True
    log_all_reasoning: bool = True


class AgentSecurityLayer:
    """Security middleware for agent execution."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.tool_call_count = 0
    
    def validate_input(self, user_input: str) -> tuple[bool, str]:
        """Check user input for prompt injection attempts."""
        # Check for common injection patterns
        injection_patterns = [
            r"ignore\s+(previous|above|all)\s+(instructions|prompts)",
            r"you\s+are\s+now\s+",
            r"new\s+instructions?\s*:",
            r"system\s*:\s*",
            r"<\s*system\s*>",
            r"ADMIN\s*MODE",
            r"jailbreak",
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return False, f"Potential prompt injection detected: {pattern}"
        
        return True, "OK"
    
    def authorize_tool_call(
        self, tool_name: str, args: dict, risk_level: RiskLevel
    ) -> tuple[bool, str]:
        """Check if a tool call is authorized."""
        # Tool allowlist
        if tool_name not in self.policy.allowed_tools:
            return False, f"Tool '{tool_name}' not in allowed list"
        
        # Tool blocklist
        if self.policy.blocked_tools and tool_name in self.policy.blocked_tools:
            return False, f"Tool '{tool_name}' is explicitly blocked"
        
        # Rate limit
        self.tool_call_count += 1
        if self.tool_call_count > self.policy.max_tool_calls_per_session:
            return False, f"Tool call limit exceeded ({self.policy.max_tool_calls_per_session})"
        
        # Risk-based approval
        if risk_level.value >= self.policy.require_approval_for_risk.value:
            return False, f"Tool call requires human approval (risk: {risk_level.value})"
        
        # Network egress check
        if not self.policy.allow_network_egress and self._requires_egress(tool_name, args):
            return False, "Network egress not allowed for this agent"
        
        return True, "OK"
    
    def scrub_output(self, output: str) -> str:
        """Remove PII and sensitive data from agent output."""
        if not self.policy.pii_scrub_outputs:
            return output
        
        # Email addresses
        output = re.sub(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b', '[EMAIL_REDACTED]', output)
        # Phone numbers
        output = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REDACTED]', output)
        # Credit card numbers
        output = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CC_REDACTED]', output)
        # SSN
        output = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', output)
        
        return output
    
    def _requires_egress(self, tool_name: str, args: dict) -> bool:
        egress_tools = {"web_search", "http_request", "api_call", "send_email"}
        return tool_name in egress_tools
```

## Decision Tree: Security Architecture

```
What does the agent access?
│
├── Read-only internal data
│   └── Minimal security: input validation + audit logging
│
├── Write to internal systems
│   └── Medium security: + tool authorization + approval gates for writes
│
├── External APIs / communications
│   └── High security: + credential management + egress control + PII scrubbing
│
└── Financial / production systems
    └── Maximum security: + zero-trust + mandatory human approval + full audit
```

## When NOT to Over-Invest in Agent Security

- **Internal-only tools**: Engineering productivity agents used by trusted employees. Basic auth + audit is sufficient.
- **Read-only agents**: Agents that only search and summarize have lower risk profiles.
- **Development/testing**: Security overhead slows iteration. Use a permissive policy in dev, strict in prod.

## Tradeoffs

| Security Level | Protection | UX Impact | Engineering Cost |
|---------------|-----------|-----------|-----------------|
| Minimal (input validation only) | Low | None | 1 day |
| Standard (+ tool auth + audit) | Medium | Low | 1 week |
| High (+ zero-trust + approval) | High | Medium (approval delays) | 2-3 weeks |
| Maximum (+ network isolation) | Highest | High (restricted capabilities) | 4-6 weeks |

## Real-World Examples

- **GitHub Copilot**: Sandboxed code execution, no network access from generated code.
- **ChatGPT Plugins (deprecated)**: OAuth-based per-tool auth, user consent for each plugin.
- **Enterprise agent frameworks**: Vault-integrated credential management, RLS database access.

## Failure Modes

1. **Over-permissive tool access**: Agent given `admin_delete_all` tool "just in case." It gets used on a hallucination. Mitigation: principle of least privilege, explicit tool allowlists.
2. **Indirect prompt injection via RAG**: Malicious content in indexed documents instructs the agent to exfiltrate data. Mitigation: separate data plane from control plane, never execute instructions from retrieved content.
3. **Credential in conversation history**: User pastes an API key, it enters conversation memory, later retrieved and displayed. Mitigation: credential detection + scrubbing in input and memory.

## Source(s) and Further Reading

- OWASP Top 10 for LLMs: https://owasp.org/www-project-top-10-for-large-language-model-applications/
- "Prompt Injection Attacks" - Simon Willison (2023-2025)
- Anthropic Safety Best Practices: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/mitigate-prompt-injections
- "Securing LLM Applications" - NIST AI 600-1
