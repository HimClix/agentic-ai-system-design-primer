# Containerization for Agents
> `langgraph build` produces a Docker image -- optimize it with multi-stage builds, slim base images, and proper secrets injection.

## What It Is

Containerization packages agent applications into portable Docker images that include the agent code, dependencies, and runtime configuration. For LangGraph applications, the `langgraph build` CLI generates a Docker image directly. For custom deployments, multi-stage builds produce slim, secure images.

## How It Works

### LangGraph Build Pipeline

```
langgraph.json (config)
    │
    ▼
langgraph build --tag my-agent:v1
    │
    ├── Reads langgraph.json for graph configuration
    ├── Installs Python dependencies
    ├── Copies application code
    ├── Sets up LangGraph server runtime
    └── Produces Docker image
    
    ▼
Docker image: my-agent:v1
    │
    ▼
langgraph up  (local)
    OR
docker push → K8s deploy (production)
```

### langgraph.json Configuration

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agent/graph.py:graph"
  },
  "env": ".env",
  "python_version": "3.11",
  "pip_config_file": "pip.conf",
  "dockerfile_lines": []
}
```

## Production Implementation

### Custom Multi-Stage Dockerfile

```dockerfile
# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production image
FROM python:3.11-slim AS production

# Security: run as non-root
RUN groupadd -r agent && useradd -r -g agent agent

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/agent/.local

# Copy application code
COPY src/ ./src/
COPY langgraph.json .

# Set PATH for user-installed packages
ENV PATH=/home/agent/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/healthz')" || exit 1

# Switch to non-root user
USER agent

EXPOSE 8000

# Graceful shutdown support
STOPSIGNAL SIGTERM

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--timeout-graceful-shutdown", "30"]
```

### Environment Variable Management

```python
from pydantic_settings import BaseSettings
from typing import Optional


class AgentSettings(BaseSettings):
    """Agent configuration from environment variables."""
    
    # LLM providers (REQUIRED)
    anthropic_api_key: str
    openai_api_key: Optional[str] = None
    
    # Infrastructure (REQUIRED)
    redis_url: str = "redis://localhost:6379"
    database_url: str = "postgresql://localhost:5432/agents"
    
    # Vector database
    pinecone_api_key: Optional[str] = None
    pinecone_index: str = "agent-knowledge"
    
    # Agent configuration
    default_model: str = "claude-sonnet-4"
    max_steps: int = 25
    max_session_time: int = 300
    
    # Observability
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "production"
    log_level: str = "INFO"
    
    # Feature flags
    enable_semantic_cache: bool = True
    enable_model_routing: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = AgentSettings()
```

### Kubernetes Secret Injection

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: agent-secrets
type: Opaque
stringData:
  anthropic-api-key: "sk-ant-..."
  database-url: "postgresql://user:pass@host:5432/db"
  redis-url: "redis://:password@redis:6379"

---
# deployment.yaml - referencing secrets
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-service
spec:
  template:
    spec:
      containers:
      - name: agent
        image: your-registry/agent:v1
        envFrom:
        - secretRef:
            name: agent-secrets
        env:
        - name: DEFAULT_MODEL
          value: "claude-sonnet-4"
        - name: LOG_LEVEL
          value: "INFO"
```

## Decision Tree: Image Optimization

```
Image size matters?
│
├── YES (deploying frequently, cold starts matter)
│   ├── Use multi-stage build
│   ├── Use python:3.11-slim (not full python:3.11)
│   ├── Remove build tools from final image
│   └── Target: < 500MB
│
└── NO (internal tool, deploys weekly)
    └── Single-stage Dockerfile is fine
```

## When NOT to Use Custom Dockerfiles

- **LangGraph Platform**: The managed service handles containerization for you.
- **Quick prototypes**: Use `langgraph build` directly; customizing Dockerfiles is premature.
- **Lambda deployment**: Use Lambda container image support or ZIP-based packaging instead.

## Tradeoffs

| Approach | Image Size | Build Time | Security | Flexibility |
|----------|-----------|-----------|----------|------------|
| `langgraph build` | ~800MB | Fast | Good | Low |
| Single-stage Dockerfile | ~1.2GB | Fast | Medium | High |
| Multi-stage Dockerfile | ~400MB | Slower | Good | High |
| Distroless base | ~250MB | Slowest | Best | Medium |

## Failure Modes

1. **Secrets in Docker layers**: Building with `ENV API_KEY=sk-...` bakes secrets into image layers. Mitigation: always inject at runtime via K8s secrets or Vault.
2. **Missing dependencies at runtime**: Multi-stage build forgets to copy a system library. Mitigation: test image locally before deploying.
3. **Python version mismatch**: Building on 3.12 but production uses 3.11. Mitigation: pin Python version in Dockerfile.
4. **Oversized images causing slow deploys**: 2GB+ images take minutes to pull. Mitigation: multi-stage builds, .dockerignore file.

## Source(s) and Further Reading

- LangGraph Docker: https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/
- Docker Multi-Stage Builds: https://docs.docker.com/build/building/multi-stage/
- Kubernetes Secrets: https://kubernetes.io/docs/concepts/configuration/secret/
- Pydantic Settings: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
