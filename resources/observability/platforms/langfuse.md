# Langfuse: Open-Source LLM Observability

> Langfuse is the open-source observability platform that 2,300+ companies chose because they need full data control, self-hosting, and framework-agnostic tracing -- without vendor lock-in.

## What It Is

Langfuse is an open-source (MIT license) LLM engineering platform providing tracing, evaluation, prompt management, and metrics for LLM applications. It is framework-agnostic (works with LangChain, LlamaIndex, OpenAI SDK, or custom code), self-hostable (Postgres + ClickHouse), and used by 2,300+ companies in production.

**Key facts:**
- **License**: MIT (fully open-source)
- **Self-hosted**: Postgres + ClickHouse (or managed cloud)
- **Adoption**: 2,300+ companies
- **Framework support**: LangChain, LangGraph, LlamaIndex, OpenAI SDK, custom
- **Data model**: Traces > Observations (generations, spans, events) > Scores
- **Key features**: Tracing, prompt management + versioning, evaluation, datasets, sessions

## How It Works

### Data Model

```
Trace (one agent session)
├── Generation (LLM call)
│   ├── model: gpt-4o
│   ├── input: [messages]
│   ├── output: response
│   ├── tokens: {input: 1500, output: 300}
│   ├── cost: $0.004
│   └── Score (quality rating)
│       ├── name: "helpfulness"
│       └── value: 0.85
│
├── Span (tool call or processing step)
│   ├── name: "search_orders"
│   ├── input: {"customer_id": "C-123"}
│   ├── output: [order_list]
│   └── duration: 234ms
│
├── Span (retrieval step)
│   ├── name: "rag_retrieval"
│   ├── input: "refund policy"
│   └── output: [relevant_chunks]
│
└── Session (groups related traces)
    └── session_id: "chat_session_abc"
```

### Prompt Management + Versioning

Langfuse stores prompt templates with version control, allowing you to:
- Version prompts (v1, v2, v3...)
- Link traces to specific prompt versions
- Compare performance across prompt versions
- Roll back to previous versions in production

## Production Implementation

### Self-Hosted Setup

```yaml
# docker-compose.yml for Langfuse self-hosted
version: "3.8"
services:
  langfuse:
    image: langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/langfuse
      - NEXTAUTH_SECRET=your-secret-here
      - NEXTAUTH_URL=http://localhost:3000
      - SALT=your-salt-here
    depends_on:
      - postgres
  
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=langfuse
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

### Python SDK Integration

```python
"""
Langfuse integration for production agent tracing.
"""
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import openai

# Initialize (reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST from env)
langfuse = Langfuse()


# ============================================================
# Pattern 1: Decorator-based tracing (simplest)
# ============================================================

@observe()
def my_agent(user_input: str) -> str:
    """Automatically traced agent function."""
    
    # This LLM call is automatically captured as a generation
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": user_input}],
    )
    
    return response.choices[0].message.content


# ============================================================
# Pattern 2: Manual tracing (full control)
# ============================================================

def traced_agent_manual(user_input: str, user_id: str, session_id: str):
    """Agent with full manual tracing for maximum control."""
    
    # Create trace
    trace = langfuse.trace(
        name="customer_support_agent",
        user_id=user_id,
        session_id=session_id,
        metadata={"agent_version": "v2.3.1"},
        tags=["production", "customer_support"],
    )
    
    # Step 1: Retrieval span
    retrieval_span = trace.span(
        name="rag_retrieval",
        input={"query": user_input},
    )
    # ... perform retrieval ...
    chunks = ["refund policy: 30 day window..."]
    retrieval_span.end(output={"chunks": chunks, "count": len(chunks)})
    
    # Step 2: LLM generation
    generation = trace.generation(
        name="main_llm_call",
        model="gpt-4o",
        model_parameters={"temperature": 0.1, "max_tokens": 1024},
        input=[
            {"role": "system", "content": "You are a support agent."},
            {"role": "user", "content": user_input},
        ],
        metadata={"prompt_version": "v12"},
    )
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a support agent."},
            {"role": "user", "content": user_input},
        ],
        temperature=0.1,
    )
    
    output = response.choices[0].message.content
    
    generation.end(
        output=output,
        usage={
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
        },
        metadata={"finish_reason": response.choices[0].finish_reason},
    )
    
    # Step 3: Score the trace (for evaluation)
    trace.score(
        name="user_satisfaction",
        value=1,  # Will be updated when user provides feedback
        comment="Pending user feedback",
    )
    
    return output


# ============================================================
# Pattern 3: LangGraph integration
# ============================================================

from langfuse.callback import CallbackHandler

def create_langgraph_with_langfuse():
    """LangGraph agent with Langfuse tracing via callback handler."""
    
    langfuse_handler = CallbackHandler(
        public_key="pk-...",
        secret_key="sk-...",
        host="https://your-langfuse.com",
    )
    
    # Pass to LangGraph invoke
    # result = graph.invoke(
    #     {"messages": [user_message]},
    #     config={"callbacks": [langfuse_handler]},
    # )


# ============================================================
# Pattern 4: Prompt management
# ============================================================

def use_managed_prompt():
    """Fetch prompt from Langfuse prompt management."""
    
    # Get the latest production prompt
    prompt = langfuse.get_prompt(
        name="customer_support_system",
        version=None,  # Latest version
        label="production",  # Only production-labeled prompts
    )
    
    # Use the prompt template
    compiled = prompt.compile(
        customer_name="Jane",
        product_name="Widget Pro",
    )
    
    # The trace will automatically link to this prompt version
    # enabling per-prompt-version performance comparison
    
    return compiled


# ============================================================
# Pattern 5: Evaluation scoring
# ============================================================

def score_trace_with_llm_judge(trace_id: str, output: str, expected: str):
    """Score a trace using LLM-as-judge pattern."""
    
    # Run LLM judge
    judge_response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "system",
            "content": "Rate this response 1-5 for helpfulness.",
        }, {
            "role": "user",
            "content": f"Response: {output}\nExpected: {expected}",
        }],
    )
    
    score = int(judge_response.choices[0].message.content.strip())
    
    # Attach score to the trace
    langfuse.score(
        trace_id=trace_id,
        name="helpfulness",
        value=score / 5.0,  # Normalize to 0-1
        comment=f"LLM judge score: {score}/5",
    )
```

## Decision Tree / When to Use

```
Need OPEN-SOURCE + SELF-HOSTED?
  YES --> Langfuse (MIT license, full data control)

Need FRAMEWORK-AGNOSTIC tracing?
  YES --> Langfuse (works with any framework)

Need PROMPT MANAGEMENT + VERSIONING?
  YES --> Langfuse (built-in prompt management)

Need DATA RESIDENCY compliance?
  YES --> Langfuse self-hosted (data never leaves your infra)

Using LangChain/LangGraph exclusively?
  Consider LangSmith for deeper native integration
```

## When NOT to Use

- **Need zero-overhead tracing** -- LangSmith's async export is slightly lower overhead for LangChain
- **Simple proxy-based monitoring** -- Helicone is simpler for basic API monitoring
- **Don't want to manage infrastructure** -- Use Langfuse Cloud instead of self-hosted

## Tradeoffs

| Aspect | Langfuse | LangSmith | Phoenix |
|--------|---------|-----------|---------|
| **Open-source** | Yes (MIT) | No (proprietary) | Yes (Apache 2.0) |
| **Self-hosted** | Yes | No | Yes (local only) |
| **Framework** | Agnostic | LangChain-native | Agnostic |
| **Prompt management** | Yes | Yes | No |
| **Eval integration** | Scores, datasets | Datasets, CI gates | Evals |
| **Production scale** | Yes (ClickHouse) | Yes | Dev/debug focused |
| **Companies** | 2,300+ | Not disclosed | Growing |

## Real-World Examples

1. **E-commerce support agent**: Traces every conversation, scores with LLM-as-judge, compares prompt versions to optimize resolution rate.
2. **RAG application**: Traces retrieval quality, scores chunk relevance, identifies when retrieval failures cause hallucinations.
3. **Multi-agent workflow**: Hierarchical traces showing orchestrator delegation to specialist agents.

## Failure Modes

| Failure | Cause | Impact | Prevention |
|---------|-------|--------|------------|
| **Trace ingestion lag** | High volume overwhelms Postgres | Delayed traces in UI | ClickHouse for analytics, Postgres for metadata |
| **Storage growth** | Full I/O capture for every trace | Disk costs grow | Sampling for high-volume; retention policies |
| **SDK version mismatch** | Old SDK with new server | Missing features, errors | Pin SDK version; test upgrades |

## Source(s) and Further Reading

- Langfuse Documentation: https://langfuse.com/docs
- Langfuse GitHub: https://github.com/langfuse/langfuse
- Langfuse LangGraph Cookbook: https://langfuse.com/docs/integrations/langchain/example-langchain-langgraph
- Langfuse Self-Hosting Guide: https://langfuse.com/docs/deployment/self-host
