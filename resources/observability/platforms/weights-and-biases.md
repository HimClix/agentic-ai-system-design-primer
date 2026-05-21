# Weights & Biases (W&B) for Agent Observability

> W&B Weave for LLM tracing, experiment tracking for prompt optimization, and dataset management for evaluation. Best for teams that also do model training/fine-tuning and want one platform for everything.

## What It Is

Weights & Biases (W&B) is an ML experiment tracking platform that expanded into LLM observability with **W&B Weave**. It provides:

1. **Weave (LLM tracing)**: Trace LLM calls, agent steps, and tool invocations with full input/output logging
2. **Experiment tracking**: Track prompt variants, model comparisons, and hyperparameter sweeps
3. **Dataset management**: Version and manage evaluation datasets
4. **Evaluation**: Run evaluations against datasets and visualize results
5. **Model registry**: Track fine-tuned model versions

### W&B vs Langfuse vs LangSmith

| Feature | W&B Weave | Langfuse | LangSmith |
|---------|-----------|----------|-----------|
| LLM tracing | Yes | Yes | Yes |
| Experiment tracking (ML) | Yes (core product) | No | No |
| Model training integration | Yes (native) | No | No |
| Fine-tuning tracking | Yes | No | No |
| Prompt management | Via Experiments | Yes | Yes (Hub) |
| Open source | Weave is OSS | Yes (fully) | No |
| Self-hosted | Yes (enterprise) | Yes | No |
| Pricing (hobby) | Free tier | Free tier | Free tier |
| Pricing (team) | $50/user/month | $59/month base | $39/user/month |
| Best for | Teams doing training + inference | Pure inference monitoring | LangChain-heavy teams |

### When to Choose W&B

```
Use W&B if:
  ✓ You fine-tune models (W&B is the gold standard for training tracking)
  ✓ You want ML experiment tracking + LLM observability in one platform
  ✓ You run prompt optimization experiments (A/B testing prompts)
  ✓ Your team already uses W&B for non-LLM ML work

Use Langfuse instead if:
  ✓ You only need inference monitoring (no training)
  ✓ You want full open-source and self-hosted
  ✓ Cost-sensitive (Langfuse is cheaper for pure observability)

Use LangSmith instead if:
  ✓ You are heavily invested in LangChain/LangGraph
  ✓ You need the deepest LangChain integration
```

## How It Works

### W&B Weave Architecture

```
                     Agent Application
                           │
                    ┌──────▼──────┐
                    │  @weave.op() │  ← Decorators on functions
                    │  decorators  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Weave SDK    │  Auto-captures:
                    │              │  - Inputs/outputs
                    │              │  - Latency
                    │              │  - Token counts
                    │              │  - Costs
                    │              │  - Parent-child relations
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ W&B Backend  │  Stores traces, metrics,
                    │ (cloud or    │  datasets, experiments
                    │  self-hosted)│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ W&B Dashboard│  Visualization, filtering,
                    │              │  comparison, export
                    └─────────────┘
```

## Production Implementation

### Basic Weave Tracing

```python
import weave
import anthropic

# Initialize Weave (creates project in W&B)
weave.init("agent-project")

client = anthropic.Anthropic()


# --- Trace any function with @weave.op() ---

@weave.op()
def classify_intent(user_message: str) -> str:
    """Classify user intent. Automatically traced by Weave."""
    response = client.messages.create(
        model="claude-haiku-3.5",
        max_tokens=50,
        system="Classify as: support, sales, technical. Output only the category.",
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text.strip()


@weave.op()
def search_knowledge_base(query: str) -> list[str]:
    """Search KB. Traced with inputs, outputs, and latency."""
    # Your search implementation
    results = vector_search(query, top_k=5)
    return [r["content"] for r in results]


@weave.op()
def generate_response(
    user_message: str,
    context: list[str],
    intent: str,
) -> str:
    """Generate final response. Traced as part of the agent pipeline."""
    context_text = "\n".join(context)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=f"You are a {intent} agent. Use this context:\n{context_text}",
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


@weave.op()
def agent_pipeline(user_message: str) -> dict:
    """
    Full agent pipeline. Weave creates a trace tree:
    
    agent_pipeline
    ├── classify_intent
    ├── search_knowledge_base
    └── generate_response
    """
    intent = classify_intent(user_message)
    context = search_knowledge_base(user_message)
    response = generate_response(user_message, context, intent)
    
    return {
        "response": response,
        "intent": intent,
        "context_used": len(context),
    }


# --- Auto-patching for LLM clients ---
# Weave can automatically trace all Anthropic/OpenAI calls

weave.init("agent-project")

# After this, ALL client.messages.create() calls are traced automatically
# No decorators needed on the LLM calls themselves


# --- Evaluation with Weave ---

class AgentEvaluator(weave.Scorer):
    """Custom scorer for agent evaluation."""

    @weave.op()
    def score(self, output: dict, expected: dict) -> dict:
        """Score agent output against expected values."""
        intent_correct = output.get("intent") == expected.get("intent")
        response_length = len(output.get("response", ""))

        # LLM-as-Judge for response quality
        judge_response = client.messages.create(
            model="claude-haiku-3.5",
            max_tokens=10,
            system="Rate the response quality 1-5. Output only the number.",
            messages=[{
                "role": "user",
                "content": (
                    f"Question: {expected.get('question', '')}\n"
                    f"Response: {output.get('response', '')}\n"
                    f"Rate 1-5:"
                ),
            }],
        )
        quality_score = int(judge_response.content[0].text.strip())

        return {
            "intent_correct": intent_correct,
            "quality_score": quality_score,
            "response_length": response_length,
        }


# Run evaluation
async def run_evaluation():
    # Create or load a dataset
    dataset = weave.Dataset(
        name="support-eval-v1",
        rows=[
            {
                "question": "How do I reset my password?",
                "expected": {"intent": "support"},
            },
            {
                "question": "I want to upgrade to enterprise",
                "expected": {"intent": "sales"},
            },
            {
                "question": "API returning 500 errors",
                "expected": {"intent": "technical"},
            },
        ],
    )
    weave.publish(dataset)

    # Run evaluation
    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=[AgentEvaluator()],
    )

    results = await evaluation.evaluate(agent_pipeline)
    # Results are automatically logged to W&B with visualizations
    return results


# --- Prompt Experiment Tracking ---

@weave.op()
def run_prompt_experiment(
    system_prompt: str,
    test_queries: list[str],
    experiment_name: str,
) -> dict:
    """
    Track prompt experiments.
    Compare different system prompts on the same test queries.
    """
    import wandb

    # Start a W&B run for this experiment
    run = wandb.init(
        project="agent-prompt-experiments",
        name=experiment_name,
        config={
            "system_prompt": system_prompt,
            "model": "claude-sonnet-4-20250514",
            "num_queries": len(test_queries),
        },
    )

    results = []
    total_tokens = 0
    total_latency = 0

    for query in test_queries:
        import time
        start = time.time()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": query}],
        )
        latency = time.time() - start

        tokens = response.usage.input_tokens + response.usage.output_tokens
        total_tokens += tokens
        total_latency += latency

        results.append({
            "query": query,
            "response": response.content[0].text,
            "tokens": tokens,
            "latency": latency,
        })

    # Log aggregate metrics
    wandb.log({
        "avg_tokens": total_tokens / len(test_queries),
        "avg_latency": total_latency / len(test_queries),
        "total_cost": total_tokens / 1_000_000 * 18,  # Sonnet pricing
    })

    # Log individual results as a table
    table = wandb.Table(
        columns=["query", "response", "tokens", "latency"],
        data=[[r["query"], r["response"], r["tokens"], r["latency"]] for r in results],
    )
    wandb.log({"results": table})

    run.finish()
    return {"avg_tokens": total_tokens / len(test_queries)}
```

## Decision Tree: When to Use W&B

```
    Should I use W&B for agent observability?
                        │
             ┌──────────▼──────────┐
             │ Does your team do    │
             │ model training or    │
             │ fine-tuning?         │
             └──┬──────────────┬──┘
               Yes             No
                │               │
         W&B is ideal      ┌───▼────────────┐
         (one platform     │ Do you need     │
          for everything)  │ experiment      │
                           │ tracking for    │
                           │ prompt optim?   │
                           └──┬────────┬───┘
                             Yes      No
                              │        │
                         W&B Weave   Langfuse or
                         is good     LangSmith
                                     may be simpler
```

## When NOT to Use

1. **Pure inference monitoring (no training)**: If you only need trace visualization and cost tracking, Langfuse is simpler and cheaper.
2. **LangChain-first teams**: LangSmith has deeper LangChain integration.
3. **Budget-constrained**: W&B Teams at $50/user/month is more expensive than alternatives for pure observability.
4. **When you need full open-source**: Weave is open-source but the full W&B platform (experiments, artifacts) is proprietary. Langfuse is fully open-source.
5. **Simple applications**: If your agent is a single LLM call, any logging solution works. W&B is for complex pipelines.

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Unified ML + LLM platform | More expensive than observation-only tools |
| Best-in-class experiment tracking | Steeper learning curve than Langfuse |
| Weave is open source | Full platform is proprietary |
| Excellent visualization and comparison | Overkill for simple agent pipelines |
| Strong community and documentation | Less LangChain-specific than LangSmith |
| Dataset versioning built in | Self-hosted requires enterprise license |

## Failure Modes

### 1. Data Volume Overload
Tracing every LLM call at high volume generates terabytes of trace data.
**Mitigation**: Sample traces (e.g., log 10% of production traffic). Use Weave's sampling configuration.

### 2. Latency Impact
The Weave SDK adds overhead to traced functions.
**Mitigation**: Async trace submission. Weave's overhead is typically < 5ms per traced call.

### 3. Cost Creep
W&B storage costs grow with trace volume.
**Mitigation**: Set retention policies. Delete old traces. Monitor W&B storage usage monthly.

### 4. Vendor Lock-in
Traces stored in W&B format are not easily portable.
**Mitigation**: Export traces to OpenTelemetry format periodically. Keep raw logs in your own storage as backup.

## Sources and Further Reading

- [W&B Weave Documentation](https://wandb.me/weave)
- [W&B Weave GitHub](https://github.com/wandb/weave)
- [W&B Experiment Tracking](https://docs.wandb.ai/guides/track)
- [W&B Pricing](https://wandb.ai/pricing)
- [Weave LLM Tracing Quickstart](https://weave-docs.wandb.ai/quickstart)
- [W&B + Anthropic Integration](https://docs.wandb.ai/guides/integrations/anthropic)
