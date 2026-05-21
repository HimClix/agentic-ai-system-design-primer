# E2B Sandboxed Code Execution

> Never let an AI agent execute code on your infrastructure -- E2B provides isolated cloud sandboxes that spin up in ~150ms, run arbitrary code, and destroy themselves on timeout.

## What It Is

E2B (pronounced "e-to-b") is a cloud platform providing sandboxed code execution environments for AI agents. Each sandbox is an isolated microVM (based on Firecracker) with its own filesystem, network stack, and resource limits. When an agent needs to run code, the code executes inside the sandbox -- not on your servers.

**Key facts:**
- **Isolation**: Firecracker microVMs (same tech as AWS Lambda)
- **Startup time**: ~150ms per sandbox
- **Languages**: Python, JavaScript/TypeScript, Bash, and any language via custom Dockerfiles
- **Network**: Configurable egress controls (block, allowlist, or full access)
- **Persistence**: Sandboxes are ephemeral by default, destroyed on timeout
- **Pricing**: Pay-per-second compute
- **SDK**: Python and TypeScript

## How It Works

```
Agent decides to run code
         │
         ▼
┌──────────────────┐     ┌─────────────────────────────────┐
│   Your Server    │     │        E2B Cloud                 │
│                  │     │                                   │
│  Agent says:     │ API │  ┌──────────────────────────┐    │
│  "Run this       │────>│  │   Firecracker microVM    │    │
│   Python code"   │     │  │                          │    │
│                  │     │  │  - Isolated filesystem    │    │
│  Gets back:      │<────│  │  - No host access         │    │
│  stdout, stderr, │     │  │  - Configurable network   │    │
│  files, artifacts│     │  │  - Resource limits         │    │
│                  │     │  │  - Destroyed on timeout    │    │
└──────────────────┘     │  └──────────────────────────┘    │
                         └─────────────────────────────────┘

Even if the agent is compromised and generates malicious code,
the blast radius is limited to the disposable sandbox.
```

## Production Implementation

### Installation

```bash
pip install e2b-code-interpreter
```

### Basic Code Execution

```python
"""
E2B sandboxed code execution for AI agents.
"""
from e2b_code_interpreter import Sandbox
import os

# Set API key
os.environ["E2B_API_KEY"] = "your-api-key"


async def run_code_in_sandbox(code: str, language: str = "python") -> dict:
    """
    Execute code in an isolated E2B sandbox.
    Returns stdout, stderr, and any generated artifacts.
    """
    sandbox = Sandbox()
    
    try:
        # Execute code
        execution = sandbox.run_code(code)
        
        return {
            "success": execution.error is None,
            "stdout": execution.logs.stdout,
            "stderr": execution.logs.stderr,
            "error": str(execution.error) if execution.error else None,
            "artifacts": [
                {
                    "name": result.png and "chart.png" or "output",
                    "type": "image/png" if result.png else "text",
                }
                for result in execution.results
            ],
        }
    finally:
        sandbox.kill()  # Always destroy sandbox


# Example usage
result = await run_code_in_sandbox("""
import pandas as pd
import matplotlib.pyplot as plt

data = {'Q1': 42, 'Q2': 48, 'Q3': 55, 'Q4': 61}
df = pd.DataFrame(list(data.items()), columns=['Quarter', 'Revenue'])

plt.figure(figsize=(8, 5))
plt.bar(df['Quarter'], df['Revenue'])
plt.title('Quarterly Revenue')
plt.savefig('revenue.png')
print(f"Total: ${sum(data.values())}M")
""")
```

### Production Configuration with Network Controls

```python
"""
Production E2B setup with security controls.
"""
from e2b_code_interpreter import Sandbox
from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)


class SecureSandboxManager:
    """
    Manages E2B sandboxes with production security controls.
    """
    
    def __init__(
        self,
        timeout_seconds: int = 30,
        max_memory_mb: int = 512,
        max_concurrent: int = 10,
        allow_network: bool = False,
        allowed_domains: Optional[list[str]] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
        self.max_concurrent = max_concurrent
        self.allow_network = allow_network
        self.allowed_domains = allowed_domains or []
        self._active_sandboxes = 0
    
    async def execute(
        self,
        code: str,
        language: str = "python",
        session_id: str = "",
        user_id: str = "",
    ) -> dict:
        """
        Execute code in a secure sandbox with full audit trail.
        """
        # Concurrency limit
        if self._active_sandboxes >= self.max_concurrent:
            return {
                "success": False,
                "error": "Sandbox concurrency limit reached",
            }
        
        # Pre-execution code analysis (fast heuristic)
        safety_check = self._pre_check_code(code)
        if not safety_check["safe"]:
            logger.warning(
                f"Code blocked by pre-check: {safety_check['reason']}",
                extra={"user_id": user_id, "session_id": session_id},
            )
            return {
                "success": False,
                "error": f"Code blocked: {safety_check['reason']}",
            }
        
        self._active_sandboxes += 1
        start_time = time.time()
        
        try:
            sandbox = Sandbox(
                timeout=self.timeout_seconds,
            )
            
            # Execute with timeout
            execution = sandbox.run_code(code)
            
            elapsed = time.time() - start_time
            
            result = {
                "success": execution.error is None,
                "stdout": execution.logs.stdout[:10000],  # Cap output size
                "stderr": execution.logs.stderr[:5000],
                "error": str(execution.error) if execution.error else None,
                "execution_time_seconds": elapsed,
                "sandbox_id": sandbox.id if hasattr(sandbox, 'id') else "unknown",
            }
            
            # Audit log
            logger.info(
                "Sandbox execution completed",
                extra={
                    "user_id": user_id,
                    "session_id": session_id,
                    "success": result["success"],
                    "elapsed_seconds": elapsed,
                    "code_length": len(code),
                },
            )
            
            return result
            
        except TimeoutError:
            logger.warning(
                f"Sandbox execution timed out after {self.timeout_seconds}s",
                extra={"user_id": user_id, "session_id": session_id},
            )
            return {
                "success": False,
                "error": f"Execution timed out after {self.timeout_seconds}s",
            }
        except Exception as e:
            logger.error(f"Sandbox error: {e}", extra={"user_id": user_id})
            return {"success": False, "error": str(e)}
        finally:
            self._active_sandboxes -= 1
            try:
                sandbox.kill()
            except Exception:
                pass
    
    def _pre_check_code(self, code: str) -> dict:
        """
        Fast heuristic check before sandbox execution.
        This is NOT a security boundary -- the sandbox is the security boundary.
        This just catches obvious mistakes and reduces unnecessary sandbox usage.
        """
        import re
        
        # Block obvious dangerous patterns
        dangerous_patterns = [
            (r"subprocess\.(call|run|Popen)", "subprocess execution"),
            (r"os\.system\(", "os.system call"),
            (r"shutil\.rmtree\(['\"]\/", "filesystem deletion"),
            (r"(rm\s+-rf|dd\s+if=)", "destructive shell command"),
            (r"requests\.(get|post).*\b(169\.254|metadata)", "cloud metadata access"),
            (r"socket\.connect", "raw socket connection"),
        ]
        
        for pattern, reason in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return {"safe": False, "reason": reason}
        
        # Check code length (prevent abuse)
        if len(code) > 50000:
            return {"safe": False, "reason": "Code too long (>50KB)"}
        
        return {"safe": True, "reason": ""}


# Integration with LangGraph tool
from langchain_core.tools import tool

sandbox_manager = SecureSandboxManager(
    timeout_seconds=30,
    allow_network=False,
)

@tool
async def execute_python(code: str) -> str:
    """Execute Python code in a secure sandbox. Returns stdout output."""
    result = await sandbox_manager.execute(code, language="python")
    
    if result["success"]:
        return result["stdout"]
    else:
        return f"Error: {result['error']}\nStderr: {result.get('stderr', '')}"
```

## Decision Tree / When to Use

```
Does your agent generate or execute CODE?
  YES --> You MUST sandbox it (E2B or equivalent)
  NO  --> Not needed

Does the agent need to install packages dynamically?
  YES --> E2B (full OS environment per sandbox)
  NO  --> E2B still recommended, but lighter options exist

Do you need sub-200ms sandbox startup?
  YES --> E2B (~150ms) or Firecracker directly
  NO  --> Docker containers are also viable

Do you need network access from sandboxes?
  YES --> E2B with domain allowlist
  NO  --> E2B with network disabled (safest)
```

## When NOT to Use

- **Agent only makes API calls, no code execution** -- Sandboxing API calls is handled by credential scoping
- **Static code analysis only** -- If you only need to analyze code, not run it, use a linter
- **Deterministic tool calls** -- Pre-defined tool functions don't need sandboxing (they're already controlled)
- **Budget-constrained prototypes** -- E2B costs per sandbox-second; use local Docker for development

## Tradeoffs

| Approach | Isolation | Startup | Cost | Complexity |
|----------|----------|---------|------|------------|
| **E2B** | Excellent (microVM) | ~150ms | Pay-per-second | Low (SDK) |
| **Docker container** | Good | 1-5s | Self-hosted | Medium |
| **gVisor** | Good | <100ms | Self-hosted | High |
| **Firecracker (raw)** | Excellent | ~125ms | Self-hosted | Very high |
| **No sandbox** | None | 0ms | Free | None |

## Real-World Examples

1. **OpenAI Code Interpreter** -- Uses sandboxed environments (similar architecture) to execute user code safely. Each session gets an isolated environment.

2. **Data analysis agents** -- Agent generates pandas/matplotlib code, executes in E2B sandbox, returns charts as images. No data ever touches the host.

3. **CI/CD automation agents** -- Agent generates build scripts, runs them in sandbox to validate before applying to real CI pipeline.

## Failure Modes

| Failure | Cause | Impact | Prevention |
|---------|-------|--------|------------|
| **Sandbox escape** | microVM vulnerability | Code reaches host | Keep E2B SDK updated; defense in depth |
| **Resource exhaustion** | Code runs infinite loop / allocates memory | Sandbox hangs, costs increase | Timeout + memory limits |
| **Network exfiltration** | Code sends data to attacker | Data leak from sandbox | Disable network or use strict allowlist |
| **Sandbox reuse** | Leftover state from previous execution | Data leakage between sessions | Always use fresh sandboxes |
| **API key exposure** | API key leaked in sandbox environment | Unauthorized sandbox creation | Rotate keys; use IAM-scoped tokens |

## Source(s) and Further Reading

- E2B Documentation: https://e2b.dev/docs
- E2B GitHub: https://github.com/e2b-dev/e2b
- Firecracker (underlying technology): https://firecracker-microvm.github.io/
- AWS Lambda isolation model (similar approach): https://docs.aws.amazon.com/lambda/latest/dg/lambda-security.html
