# Code Execution Tools for Agents
> Never run LLM-generated code on your host machine. Sandboxed execution in E2B, Docker, or Pyodide is the only production-safe pattern.

## What It Is

Code execution tools allow agents to write and run code dynamically -- performing data analysis, generating visualizations, testing hypotheses, or executing multi-step computations that would be unreliable via LLM text generation alone. The agent writes code (typically Python), a sandboxed runtime executes it, and the results (stdout, files, errors) are returned to the agent.

The critical constraint: LLM-generated code is untrusted code. It can contain anything -- malicious commands, infinite loops, memory-exhausting operations, or network exfiltration. Every production deployment requires security isolation, resource limits, and network controls.

## How It Works

### Architecture

```
LLM generates Python code
  |
  v
[Code Sanitizer] -- basic checks (no os.system, no imports of subprocess)
  |
  v
[Sandbox Orchestrator] -- selects/creates sandbox
  |
  v
[Sandboxed Runtime]
  |-- E2B Cloud Sandbox (managed, recommended)
  |-- Docker Container (self-hosted)
  |-- Pyodide (in-browser, limited)
  |
  Execution with:
  |-- Timeout: 30s default
  |-- Memory limit: 256MB-1GB
  |-- No network egress (or allowlisted domains only)
  |-- No filesystem persistence
  |-- No privilege escalation
  |
  v
[Result Collector]
  |-- stdout/stderr
  |-- Generated files (charts, CSVs)
  |-- Execution time
  |-- Exit code
  |
  v
Return to LLM as structured result
```

### Sandbox Options Comparison

| Feature | E2B | Docker | Pyodide |
|---------|-----|--------|---------|
| Isolation level | Cloud VM | Container | Browser WASM |
| Setup complexity | API key only | Docker daemon | JS runtime |
| Cold start | ~2s | ~1-5s | ~3s (first load) |
| Python packages | pip install at runtime | Pre-built image | Limited (pure Python) |
| Network control | Configurable | iptables/network policies | No network by default |
| File I/O | Sandboxed filesystem | Volume mounts | Virtual filesystem |
| Cost | $0.10/hour per sandbox | Your infrastructure | Free (client-side) |
| Max execution time | Configurable (up to hours) | Configurable | ~30s (browser limit) |
| Best for | Production agents | Self-hosted deployments | Client-side demos |

## Production Implementation

### E2B Sandbox (Recommended for Production)

```python
"""Code execution tool using E2B sandboxes."""
from e2b_code_interpreter import Sandbox
from typing import Optional
import base64
import logging
import time

logger = logging.getLogger(__name__)

# E2B sandbox configuration
E2B_TIMEOUT = 30  # seconds
E2B_TEMPLATE = "Python3"  # Pre-configured Python sandbox


class CodeExecutionTool:
    name = "execute_python"
    description = (
        "Execute Python code in a secure sandbox. Use for data analysis, "
        "calculations, file generation, and any task requiring computation. "
        "The sandbox has pandas, numpy, matplotlib, and requests pre-installed. "
        "Code runs with a 30-second timeout and no persistent state between calls. "
        "Print results to stdout -- they will be returned to you."
    )

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": (
                                "Python code to execute. Must print results to stdout. "
                                "Available packages: pandas, numpy, matplotlib, scipy, requests. "
                                "For charts, save to '/tmp/chart.png' and it will be returned. "
                                "Example: print(2 + 2)"
                            ),
                        },
                        "install_packages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Additional pip packages to install before execution. "
                                "Example: ['beautifulsoup4', 'openpyxl']"
                            ),
                        },
                    },
                    "required": ["code"],
                    "additionalProperties": False,
                },
            },
        }

    async def execute(self, code: str, install_packages: list[str] = None) -> dict:
        """Execute Python code in E2B sandbox."""
        start_time = time.time()

        # Basic safety check (defense in depth, sandbox is the real protection)
        safety_result = self._safety_check(code)
        if safety_result:
            return safety_result

        try:
            sandbox = Sandbox(api_key=self.api_key, timeout=E2B_TIMEOUT)
            
            try:
                # Install additional packages if requested
                if install_packages:
                    for pkg in install_packages[:5]:  # Max 5 packages
                        sandbox.process.start_and_wait(f"pip install {pkg}", timeout=30)

                # Execute the code
                execution = sandbox.run_code(code)
                
                elapsed = time.time() - start_time
                
                result = {
                    "success": True,
                    "stdout": execution.text or "",
                    "stderr": execution.error or "",
                    "execution_time_seconds": round(elapsed, 2),
                }
                
                # Check for generated files (charts, CSVs, etc.)
                if execution.results:
                    result["outputs"] = []
                    for output in execution.results:
                        if hasattr(output, "png") and output.png:
                            result["outputs"].append({
                                "type": "image",
                                "format": "png",
                                "data": output.png,  # base64
                            })
                        elif hasattr(output, "text") and output.text:
                            result["outputs"].append({
                                "type": "text",
                                "data": output.text,
                            })
                
                return result
                
            finally:
                sandbox.close()

        except TimeoutError:
            return {
                "success": False,
                "error_type": "timeout",
                "message": f"Code execution timed out after {E2B_TIMEOUT} seconds. "
                           f"Simplify the code or break it into smaller steps.",
            }
        except Exception as e:
            logger.error(f"Code execution failed: {str(e)}")
            return {
                "success": False,
                "error_type": "execution_error",
                "message": f"Code execution failed: {str(e)}. "
                           f"Check your code for syntax errors or missing imports.",
            }

    def _safety_check(self, code: str) -> Optional[dict]:
        """Defense-in-depth check. The sandbox is the real protection."""
        dangerous_patterns = [
            ("os.system", "Use subprocess-free alternatives"),
            ("subprocess", "Direct process execution is not allowed"),
            ("__import__('os')", "Dynamic OS imports are not allowed"),
            ("eval(input", "Interactive input is not available in sandbox"),
            ("open('/etc", "System file access is not allowed"),
        ]
        
        for pattern, suggestion in dangerous_patterns:
            if pattern in code:
                return {
                    "success": False,
                    "error_type": "validation",
                    "message": f"Code contains blocked pattern: '{pattern}'. {suggestion}.",
                }
        return None
```

### Docker-Based Execution (Self-Hosted)

```python
"""Code execution using Docker containers for self-hosted deployments."""
import docker
import tempfile
import os
import time
from pathlib import Path

DOCKER_IMAGE = "python:3.11-slim"
MEMORY_LIMIT = "256m"
CPU_QUOTA = 50000  # 50% of one CPU
NETWORK_MODE = "none"  # No network access
TIMEOUT = 30


class DockerCodeExecutor:
    def __init__(self):
        self.client = docker.from_env()
        self._ensure_image()

    def _ensure_image(self):
        """Pull the execution image if not present."""
        try:
            self.client.images.get(DOCKER_IMAGE)
        except docker.errors.ImageNotFound:
            self.client.images.pull(DOCKER_IMAGE)

    async def execute(self, code: str) -> dict:
        """Run code in an isolated Docker container."""
        start_time = time.time()
        
        # Write code to a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir="/tmp"
        ) as f:
            f.write(code)
            code_path = f.name

        try:
            container = self.client.containers.run(
                image=DOCKER_IMAGE,
                command=f"python /code/script.py",
                volumes={
                    code_path: {"bind": "/code/script.py", "mode": "ro"},
                },
                mem_limit=MEMORY_LIMIT,
                cpu_quota=CPU_QUOTA,
                network_mode=NETWORK_MODE,
                detach=True,
                # Security: drop all capabilities, read-only root filesystem
                cap_drop=["ALL"],
                read_only=True,
                # Writable /tmp for the script
                tmpfs={"/tmp": "size=50m"},
                user="nobody",
            )

            # Wait with timeout
            result = container.wait(timeout=TIMEOUT)
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8")
            exit_code = result["StatusCode"]
            
            elapsed = time.time() - start_time

            return {
                "success": exit_code == 0,
                "stdout": stdout[:10000],  # Cap output size
                "stderr": stderr[:5000] if exit_code != 0 else "",
                "exit_code": exit_code,
                "execution_time_seconds": round(elapsed, 2),
            }

        except docker.errors.ContainerError as e:
            return {
                "success": False,
                "error_type": "execution_error",
                "message": f"Code execution failed: {str(e)[:500]}",
            }
        except Exception as e:
            if "timed out" in str(e).lower() or "read timeout" in str(e).lower():
                return {
                    "success": False,
                    "error_type": "timeout",
                    "message": f"Code execution timed out after {TIMEOUT}s.",
                }
            raise
        finally:
            # Cleanup
            os.unlink(code_path)
            try:
                container.remove(force=True)
            except Exception:
                pass
```

## Decision Tree: Choosing a Code Execution Backend

```
Are you building a production SaaS agent?
  |
  YES --> E2B (managed, scalable, secure by default)
  |
  NO --> Are you self-hosting with Docker available?
          |
          YES --> Docker containers (full control, your infrastructure)
          |
          NO --> Is it client-side / demo only?
                  |
                  YES --> Pyodide (browser-based, limited but zero-infra)
                  |
                  NO --> Do you need GPU?
                          |
                          YES --> Modal.com or E2B with GPU templates
                          NO --> E2B or Docker
```

## When NOT to Use Code Execution

- **Simple calculations**: `2 + 2`, unit conversions, date math -- the LLM can do these natively
- **Text processing**: Summarization, extraction, formatting -- LLM native capability
- **When the LLM can use existing tools**: If you have a `query_database` tool, don't generate SQL and run it via code execution
- **Sensitive data processing**: Code execution sandboxes may log code -- don't pass secrets through generated code

## Tradeoffs

| Aspect | E2B | Docker | Pyodide |
|--------|-----|--------|---------|
| Security | High (cloud isolation) | Medium (container escape risk) | High (WASM sandbox) |
| Cost | $0.10/hr per sandbox | Your compute costs | Free |
| Package availability | Full pip | Full pip | Limited (WASM-compiled) |
| Network access | Configurable | Configurable | None |
| Persistent state | Per-sandbox session | Per-container | Per-page |
| Scaling | Managed | Manual | Client-side |
| Debugging | Logs via API | Container logs | Browser console |

## Real-World Examples

### Data Analysis Agent
```python
# Agent generates and executes:
code = """
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/tmp/sales.csv')
monthly = data.groupby('month')['revenue'].sum()
monthly.plot(kind='bar', title='Monthly Revenue')
plt.savefig('/tmp/chart.png')
print(f"Total revenue: ${monthly.sum():,.2f}")
print(f"Best month: {monthly.idxmax()} (${monthly.max():,.2f})")
"""
result = await code_executor.execute(code)
# Returns: stdout with revenue stats + chart PNG
```

## Failure Modes

| Failure | Cause | Impact | Fix |
|---------|-------|--------|-----|
| Infinite loop | LLM generates `while True` | Sandbox hangs until timeout | Enforce timeout (30s) |
| Memory exhaustion | LLM loads huge dataset | Container OOM killed | Memory limit (256MB-1GB) |
| Network exfiltration | LLM sends data to external URL | Data leak | network_mode="none" |
| Privilege escalation | LLM tries sudo/root operations | Container breakout | Drop all capabilities, run as nobody |
| Disk filling | LLM writes large files | Host disk full | tmpfs with size limit |
| Package supply chain | LLM installs malicious package | Code execution in sandbox | Allowlist packages, scan before install |

## Source(s) and Further Reading

- E2B Documentation -- managed code sandbox for AI agents
- Docker Security Best Practices -- container isolation patterns
- Pyodide Documentation -- WebAssembly Python runtime
- OpenAI Code Interpreter -- code execution agent architecture reference
- "Securing AI-Generated Code Execution" (USENIX, 2024) -- security research
