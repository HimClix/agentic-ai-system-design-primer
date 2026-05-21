# DSPy

> DSPy replaces hand-written prompts with compiled, optimized programs -- instead of prompt engineering, you define signatures and let the compiler find the best prompting strategy.

## What It Is

DSPy (Declarative Self-improving Language Programs) from Stanford NLP takes a radically different approach to building LLM applications:

- **Instead of prompts**: You write Python signatures (input/output specs)
- **Instead of prompt engineering**: A compiler optimizes prompts automatically
- **Instead of few-shot examples**: The optimizer selects the best examples from your data
- **Instead of manual tuning**: Teleprompters (optimizers) search for the best configuration

The core philosophy: prompts are like assembly code -- let a compiler handle them.

## How It Works

### Traditional vs DSPy Approach

```
TRADITIONAL:
  Developer writes prompt -> LLM generates output -> Developer tweaks prompt -> Repeat
  (Manual, fragile, model-specific)

DSPy:
  Developer writes signature -> Compiler optimizes prompts -> Evaluator scores output
  (Automated, robust, model-portable)
```

### Architecture

```
     +------------------+
     |  Signature       |    "question -> answer"
     |  (Input/Output)  |    (What, not how)
     +--------+---------+
              |
     +--------v---------+
     |  Module           |    dspy.ChainOfThought(signature)
     |  (How to solve)  |    dspy.ReAct(signature, tools=[...])
     +--------+---------+
              |
     +--------v---------+
     |  Program          |    Composition of modules
     |  (Pipeline)       |    (like PyTorch nn.Module)
     +--------+---------+
              |
     +--------v---------+     +----------------+
     |  Teleprompter     |<----|  Training Set   |
     |  (Optimizer)      |     |  (examples)     |
     |  Finds best       |     +----------------+
     |  prompts/examples |
     +--------+---------+
              |
     +--------v---------+
     |  Compiled         |    Optimized prompts baked in
     |  Program          |
     +------------------+
```

## Production Implementation

### Basic DSPy Program

```python
import dspy

# Configure LLM
lm = dspy.LM("anthropic/claude-sonnet-4-20250514")
dspy.configure(lm=lm)

# -- Define Signature --
class AnswerQuestion(dspy.Signature):
    """Answer a factual question with a short, precise answer."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="A concise factual answer")

# -- Define Module --
# ChainOfThought automatically adds reasoning before answering
cot = dspy.ChainOfThought(AnswerQuestion)

# -- Use (unoptimized) --
result = cot(question="What is the capital of France?")
print(result.answer)  # "Paris"
```

### Compiled/Optimized Program

```python
import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

# -- Define Signature --
class ExtractEntities(dspy.Signature):
    """Extract named entities from text."""
    text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="List of named entities")

# -- Define Program --
class EntityExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractEntities)

    def forward(self, text: str):
        return self.extract(text=text)

# -- Training Data --
trainset = [
    dspy.Example(
        text="Apple CEO Tim Cook announced new products in Cupertino.",
        entities=["Apple", "Tim Cook", "Cupertino"]
    ).with_inputs("text"),
    dspy.Example(
        text="The European Union and Japan signed a trade agreement in Tokyo.",
        entities=["European Union", "Japan", "Tokyo"]
    ).with_inputs("text"),
    # ... more examples
]

# -- Define Metric --
def entity_match_metric(example, pred, trace=None):
    """Score: what fraction of true entities were found?"""
    true_entities = set(example.entities)
    pred_entities = set(pred.entities)
    if not true_entities:
        return 1.0
    return len(true_entities & pred_entities) / len(true_entities)

# -- Compile (Optimize) --
teleprompter = BootstrapFewShot(
    metric=entity_match_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4
)

compiled_extractor = teleprompter.compile(
    EntityExtractor(),
    trainset=trainset
)

# -- Use Optimized Version --
result = compiled_extractor(text="Microsoft CEO Satya Nadella visited London.")
print(result.entities)  # ["Microsoft", "Satya Nadella", "London"]
```

### DSPy with Agents (ReAct Module)

```python
import dspy

# Define tools
def search(query: str) -> str:
    """Search the web for information."""
    # Implementation
    return "search results..."

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

# DSPy ReAct agent
class ResearchAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.react = dspy.ReAct(
            "question -> answer",
            tools=[search, calculate],
            max_iters=5
        )

    def forward(self, question: str):
        return self.react(question=question)

agent = ResearchAgent()
result = agent(question="What is the GDP of France divided by its population?")
```

## When to Use Alongside Other Frameworks

DSPy is not a replacement for LangGraph or CrewAI -- it is a complement:

```
LangGraph + DSPy:
  - Use LangGraph for agent flow control (graph, state, persistence)
  - Use DSPy for optimizing the LLM calls within each node
  - Each LangGraph node can be a compiled DSPy module

CrewAI + DSPy:
  - Use CrewAI for agent role definition and task orchestration
  - Use DSPy to optimize the prompts each agent uses internally

Standalone DSPy:
  - For RAG pipelines (optimize retrieval + generation together)
  - For classification tasks (optimize few-shot selection)
  - For any LLM call where you have training data and a metric
```

## Decision Tree: When to Use

```
     Should I use DSPy?
                |
     +----------v----------+
     | Do you have labeled |
     | examples (training  |
     | data) for your task?|
     +---+------------+----+
         |            |
        Yes          No --> Stick with manual prompting or other frameworks
         |
     +---v-----------+----+
     | Can you define a   |
     | quantitative       |
     | evaluation metric? |
     +---+------------+---+
         |            |
        Yes          No --> DSPy optimizer needs a metric
         |
     +---v-----------+----+
     | Are you spending   |
     | significant time   |
     | on prompt          |
     | engineering?       |
     +---+------------+---+
         |            |
        Yes          No --> Manual prompts are working fine
         |
     +---v-------------------+
     | USE DSPy              |
     | (standalone or with   |
     |  LangGraph/CrewAI)    |
     +------------------------+
```

## When NOT to Use

1. **No training data**: DSPy's optimizers need examples to learn from
2. **No evaluation metric**: Without a metric, the compiler cannot optimize
3. **Simple one-off prompts**: Overkill for "summarize this text" type tasks
4. **Rapid prototyping**: DSPy adds a compilation step that slows iteration
5. **Full agent orchestration**: DSPy handles LLM calls, not agent flow -- use LangGraph for that

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Automated prompt optimization | Requires training data and metrics |
| Model-portable (switch models without reprompting) | Learning curve (different mental model) |
| Systematic improvement over manual prompting | Compilation step adds development time |
| Composable modules (like PyTorch) | Smaller community than LangChain |
| Research-backed (Stanford NLP) | Less enterprise adoption (1.8% of listings) |

## Sources and Further Reading

- [DSPy Documentation](https://dspy.ai/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [DSPy: Compiling Declarative Language Model Calls](https://arxiv.org/abs/2310.03714) (Khattab et al., 2023)
- [DSPy Cookbook](https://dspy.ai/learn/)
