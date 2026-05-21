# Model Serving Infrastructure
> Use API providers until GPU costs are justified at scale -- then vLLM with PagedAttention for 2-4x throughput improvement over naive serving.

## What It Is

Model serving is the infrastructure for running LLM inference, either through cloud API providers (Anthropic, OpenAI) or self-hosted inference servers (vLLM, TGI, Triton). The decision to self-host is primarily economic: at sufficient scale, the amortized cost of GPU infrastructure is lower than per-token API pricing.

## How It Works

### Serving Options Comparison

| Feature | vLLM | TGI (HuggingFace) | Triton (NVIDIA) | Ollama | API Providers |
|---------|------|-------------------|-----------------|--------|--------------|
| **Primary use** | High-throughput serving | HF model serving | Multi-framework | Local dev | Production SaaS |
| **PagedAttention** | Yes (invented it) | Yes | Via backend | No | N/A |
| **Tensor parallelism** | Yes (multi-GPU) | Yes | Yes | Limited | N/A |
| **Continuous batching** | Yes | Yes | Yes | No | N/A |
| **OpenAI-compatible API** | Yes | Yes | Plugin | Yes | Native |
| **Quantization** | AWQ, GPTQ, FP8 | GPTQ, EETQ | All | GGUF | N/A |
| **Speculative decoding** | Yes | Yes | Custom | No | Provider-managed |
| **Prefix caching** | Yes | Partial | Custom | No | Provider-managed |
| **Ease of setup** | Medium | Easy | Hard | Easiest | Easiest |
| **Production maturity** | High | High | Highest | Low | Highest |
| **Best for** | Scale inference | Quick deployment | Multi-model | Dev/testing | Most teams |

### When Self-Hosting Makes Economic Sense

```
Monthly API Cost vs Self-Hosted Cost:

API (Sonnet at $3/$15 per M tokens):
  1M tokens/day input + 200K output = ($3 + $3)/day = $6/day = $180/month
  10M tokens/day = $60/day = $1,800/month
  100M tokens/day = $600/day = $18,000/month

Self-Hosted (Llama 3.3 70B on A100 80GB):
  GPU instance: $2.50/hr (A100 on-demand) = $1,800/month
  Or reserved: ~$1,200/month
  Throughput: ~500 tokens/sec with vLLM
  Capacity: 500 * 86400 = 43.2M tokens/day
  
  Cost per M tokens: $1,200 / (43.2 * 30) = ~$0.93/M tokens (input = output)

Break-even point: ~5-10M tokens/day (varies by model quality requirements)
```

**Critical caveat**: Self-hosted open-source models (Llama 70B, Mixtral) are not equivalent to frontier models (Claude Opus, GPT-4o) in quality. The comparison is only valid if the open-source model is sufficient for your use case.

## Production Implementation

### vLLM Production Configuration

```bash
# Start vLLM server with production settings
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --disable-log-requests \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key $VLLM_API_KEY
```

### vLLM Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 1  # One replica per GPU set
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command:
        - python
        - -m
        - vllm.entrypoints.openai.api_server
        args:
        - --model=meta-llama/Llama-3.3-70B-Instruct
        - --tensor-parallel-size=4
        - --gpu-memory-utilization=0.90
        - --max-model-len=8192
        - --enable-prefix-caching
        resources:
          limits:
            nvidia.com/gpu: 4  # 4x A100 80GB
            memory: 320Gi
          requests:
            nvidia.com/gpu: 4
            memory: 256Gi
        ports:
        - containerPort: 8000
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120  # Model loading takes time
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 180
          periodSeconds: 30
      nodeSelector:
        gpu-type: a100-80gb
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

### Hybrid Setup: API + Self-Hosted

```python
# LiteLLM config for hybrid model routing
LITELLM_CONFIG = {
    "model_list": [
        # Frontier models via API
        {
            "model_name": "agent-premium",
            "litellm_params": {
                "model": "anthropic/claude-opus-4-20250514",
                "api_key": "os.environ/ANTHROPIC_API_KEY",
            },
        },
        {
            "model_name": "agent-standard",
            "litellm_params": {
                "model": "anthropic/claude-sonnet-4-20250514",
                "api_key": "os.environ/ANTHROPIC_API_KEY",
            },
        },
        # Self-hosted for high-volume cheap tasks
        {
            "model_name": "agent-cheap",
            "litellm_params": {
                "model": "openai/meta-llama/Llama-3.3-70B-Instruct",
                "api_base": "http://vllm-server:8000/v1",
                "api_key": "os.environ/VLLM_API_KEY",
            },
        },
    ],
    "router_settings": {
        "fallbacks": [
            {"agent-cheap": ["agent-standard"]},  # If self-hosted is down, fall back to API
        ],
    },
}
```

## Decision Tree: API vs Self-Hosted

```
Do you need frontier-quality models (Claude Opus, GPT-4o)?
│
├── YES → Use API providers
│   └── Self-hosting frontier-equivalent models is not possible
│
└── NO → Open-source models are sufficient?
    │
    ├── Token volume < 5M/day?
    │   └── Use API (cheaper than GPU infra)
    │
    ├── Token volume 5-50M/day?
    │   ├── Have GPU expertise? → Self-host with vLLM
    │   └── No GPU expertise? → Use API (operational cost matters)
    │
    └── Token volume > 50M/day?
        └── Self-host with vLLM (significant cost savings)
            ├── Data residency requirements? → Required to self-host
            └── Latency requirements? → Self-host reduces network latency
```

## When NOT to Self-Host

- **You need Claude Opus or GPT-4o quality**: These aren't available for self-hosting.
- **< 5M tokens/day**: GPU costs exceed API costs at low volumes.
- **No ML/infra expertise**: vLLM requires GPU management, CUDA debugging, OOM troubleshooting.
- **Rapid model iteration**: Switching between models is trivial with APIs, expensive with self-hosting (different GPU requirements).

## Tradeoffs

| Factor | API Providers | Self-Hosted (vLLM) |
|--------|-------------|-------------------|
| Setup time | Minutes | Days-Weeks |
| Cost at low volume | Lower | Higher (fixed GPU cost) |
| Cost at high volume | Higher | Lower (amortized) |
| Model quality (frontier) | Best available | Limited to open-source |
| Latency | +50-100ms (network) | Lower (local inference) |
| Data privacy | Data sent to third party | Data stays on your infra |
| Scaling | Provider handles it | You handle GPU provisioning |
| Maintenance | Zero | Significant (CUDA, drivers, OOM) |

## Real-World Examples

- **Anthropic API users**: Most production agents use API providers. Per-token pricing makes costs predictable.
- **Meta AI**: Self-hosts Llama models on their own GPU clusters for internal tools.
- **Anyscale**: Offers managed vLLM hosting as a middle ground between API and self-hosted.

## Failure Modes

1. **OOM during inference**: Long context request exceeds GPU memory. Mitigation: set `--max-model-len` to prevent oversized requests.
2. **Model loading timeout**: 70B model takes 5+ minutes to load on cold start. Mitigation: pre-warm instances, use readiness probes with long initial delay.
3. **GPU thermal throttling**: Sustained high utilization causes thermal throttling, reducing throughput. Mitigation: monitor GPU temperature, ensure adequate cooling.
4. **API rate limits during spikes**: Provider rate-limits during peak usage. Mitigation: request rate limit increases, implement queuing, multi-provider fallback.

## Source(s) and Further Reading

- vLLM: https://docs.vllm.ai/
- vLLM PagedAttention paper: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023)
- Text Generation Inference: https://huggingface.co/docs/text-generation-inference/
- NVIDIA Triton: https://developer.nvidia.com/triton-inference-server
- Ollama: https://ollama.ai/
- LiteLLM: https://docs.litellm.ai/
