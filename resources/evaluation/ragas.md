# RAGAS: RAG Assessment Framework

> RAGAS decomposes RAG quality into four independent metrics -- faithfulness, context precision, context recall, and answer relevancy -- so you know whether the retriever or the generator is broken.

## What It Is

RAGAS (Retrieval Augmented Generation Assessment) is an open-source evaluation framework specifically designed for RAG pipelines. It provides four core metrics that independently measure retrieval quality and generation quality, enabling you to diagnose whether your RAG system fails because of bad retrieval (wrong documents) or bad generation (wrong synthesis).

**Key facts:**
- **License**: Apache 2.0
- **Metrics**: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- **Integration**: Works with Langfuse, LangSmith, and custom pipelines
- **CI support**: Can be run as pytest assertions for CI eval gates

## How It Works

### The 4 Core Metrics

| Metric | Measures | Range | Question It Answers |
|--------|---------|-------|-------------------|
| **Faithfulness** | Is the answer grounded in the context? | 0-1 | "Did the generator hallucinate beyond what the retrieved documents say?" |
| **Answer Relevancy** | Does the answer address the question? | 0-1 | "Did the generator actually answer what was asked?" |
| **Context Precision** | Are the retrieved documents relevant? | 0-1 | "Did the retriever find the right documents?" |
| **Context Recall** | Did retrieval find all needed information? | 0-1 | "Did the retriever miss any important documents?" |

### Diagnostic Matrix

```
Faithfulness LOW + Context Precision HIGH
  → Generator is hallucinating despite good retrieval
  → Fix: Better prompt, lower temperature, add grounding instructions

Faithfulness HIGH + Context Precision LOW
  → Generator faithfully represents bad documents
  → Fix: Improve retrieval (better embeddings, reranking)

Answer Relevancy LOW + Context Recall HIGH
  → All information retrieved but generator misses the point
  → Fix: Better prompt engineering for answer formatting

Context Recall LOW
  → Important documents missing from retrieval
  → Fix: More/better embeddings, expand chunk overlap, add metadata
```

## Production Implementation

### Installation

```bash
pip install ragas langchain-openai
```

### Basic RAGAS Evaluation

```python
"""
RAGAS evaluation for RAG pipeline.
"""
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset


# Prepare evaluation dataset
eval_data = {
    "question": [
        "What is the refund policy?",
        "How do I reset my password?",
        "What are the shipping options?",
    ],
    "answer": [
        "You can request a refund within 30 days of purchase.",
        "Click 'Forgot Password' on the login page to reset.",
        "We offer standard (5-7 days) and express (1-2 days) shipping.",
    ],
    "contexts": [
        ["Our refund policy allows returns within 30 days. Items must be unused."],
        ["To reset your password, go to login and click 'Forgot Password'. Enter your email."],
        ["Shipping options: Standard delivery takes 5-7 business days. Express is 1-2 days."],
    ],
    "ground_truth": [
        "Refunds are available within 30 days for unused items.",
        "Use the 'Forgot Password' link on the login page.",
        "Standard (5-7 days) and express (1-2 days) shipping available.",
    ],
}

dataset = Dataset.from_dict(eval_data)

# Run evaluation
result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
)

print(result)
# {'faithfulness': 0.92, 'answer_relevancy': 0.88,
#  'context_precision': 0.95, 'context_recall': 0.85}
```

### Integration with Langfuse

```python
"""
RAGAS + Langfuse integration for production RAG evaluation.
"""
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.integrations.langfuse import LangfuseTraceHandler
from langfuse import Langfuse

langfuse = Langfuse()

# Evaluate and log results to Langfuse
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy],
    callbacks=[LangfuseTraceHandler()],
)

# Scores appear as evaluations on Langfuse traces
# enabling per-trace quality scoring and dashboarding
```

### CI Integration (pytest)

```python
"""
tests/eval/test_rag_quality.py -- RAGAS eval as CI gate.
"""
import pytest
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset


@pytest.fixture
def eval_dataset():
    """Load evaluation dataset."""
    return Dataset.from_dict({
        "question": [
            "What is the return window?",
            "How do I contact support?",
        ],
        "answer": [],      # Filled by running the RAG pipeline
        "contexts": [],    # Filled by running retrieval
        "ground_truth": [
            "30-day return window for all purchases.",
            "Contact support via email at support@company.com or chat.",
        ],
    })


@pytest.fixture
def rag_pipeline():
    """Initialize the RAG pipeline under test."""
    from my_app.rag import RAGPipeline
    return RAGPipeline()


def test_faithfulness_above_threshold(eval_dataset, rag_pipeline):
    """Block deployment if faithfulness drops below 0.8."""
    # Run RAG pipeline on eval questions
    answers = []
    contexts = []
    for q in eval_dataset["question"]:
        result = rag_pipeline.query(q)
        answers.append(result["answer"])
        contexts.append(result["retrieved_chunks"])
    
    eval_dataset = eval_dataset.add_column("answer", answers)
    eval_dataset = eval_dataset.add_column("contexts", contexts)
    
    result = evaluate(dataset=eval_dataset, metrics=[faithfulness])
    
    assert result["faithfulness"] >= 0.8, (
        f"Faithfulness score {result['faithfulness']:.2f} below threshold 0.80. "
        f"The generator may be hallucinating."
    )


def test_context_precision_above_threshold(eval_dataset, rag_pipeline):
    """Block deployment if retrieval quality drops below 0.75."""
    answers = []
    contexts = []
    for q in eval_dataset["question"]:
        result = rag_pipeline.query(q)
        answers.append(result["answer"])
        contexts.append(result["retrieved_chunks"])
    
    eval_dataset = eval_dataset.add_column("answer", answers)
    eval_dataset = eval_dataset.add_column("contexts", contexts)
    
    result = evaluate(dataset=eval_dataset, metrics=[context_precision])
    
    assert result["context_precision"] >= 0.75, (
        f"Context precision {result['context_precision']:.2f} below threshold 0.75. "
        f"Retrieval is returning irrelevant documents."
    )
```

## Decision Tree / When to Use

```
Do you have a RAG pipeline?
  YES --> Use RAGAS (purpose-built for RAG evaluation)

Is your RAG answer quality dropping?
  Run RAGAS to diagnose:
  ├── Low faithfulness? → Generator problem (prompt, temperature)
  ├── Low context precision? → Retriever returning wrong docs
  ├── Low context recall? → Retriever missing relevant docs
  └── Low answer relevancy? → Generator not addressing the question
```

## When NOT to Use

- **Non-RAG agents** -- RAGAS is specifically for retrieval-augmented generation
- **Pure tool-calling agents** -- Use trajectory eval instead
- **Simple chatbots without knowledge base** -- Use LLM-as-judge instead

## Tradeoffs

| Approach | Cost Per Eval | What It Catches | What It Misses |
|----------|-------------|-----------------|----------------|
| **RAGAS (full 4 metrics)** | ~$0.02/question | Retrieval + generation quality | Multi-step reasoning, tool usage |
| **Faithfulness only** | ~$0.005/question | Hallucinations | Retrieval quality |
| **Human eval** | ~$0.50/question | Everything | Expensive, slow, not scalable |

## Real-World Examples

1. **Customer support RAG**: RAGAS revealed 60% context precision -- retriever returning FAQ answers for billing questions. Fix: metadata filtering by category.
2. **Legal document search**: Faithfulness score dropped from 0.92 to 0.71 after prompt change. RAGAS CI gate blocked deployment, preventing hallucinated legal advice.

## Failure Modes

| Failure | Cause | Impact | Prevention |
|---------|-------|--------|------------|
| **RAGAS judge inconsistency** | LLM-based metrics have variance | Scores fluctuate run-to-run | Average over multiple runs; use fixed seed |
| **Eval data mismatch** | Eval questions don't match production queries | High eval scores, low production quality | Regularly add production queries to eval set |
| **Cost at scale** | RAGAS uses LLM calls for scoring | Expensive for large eval sets | Sample strategically; run full suite weekly |

## Source(s) and Further Reading

- RAGAS Documentation: https://docs.ragas.io/
- RAGAS GitHub: https://github.com/explodinggradients/ragas
- RAGAS Paper: "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (2023)
- Langfuse RAGAS Integration: https://langfuse.com/docs/scores/model-based-evals/ragas
