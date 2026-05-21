# LlamaIndex

> LlamaIndex is the data-centric agent framework -- if your agent's primary job is querying, indexing, and reasoning over data, LlamaIndex's RAG-first architecture is unmatched.

## What It Is

LlamaIndex (formerly GPT Index) is a framework centered on connecting LLMs to data. While LangChain/LangGraph focus on orchestration, LlamaIndex focuses on:

- **Data ingestion**: Loading from 160+ data sources (PDFs, databases, APIs, etc.)
- **Indexing**: Vector, keyword, knowledge graph, and summary indexes
- **Querying**: Sophisticated query engines with routing, sub-questions, and multi-step
- **Agents**: ReAct agents that use query engines as tools

The key differentiator: LlamaIndex treats data retrieval as a first-class concern, not just another tool.

## How It Works

### Architecture

```
     Data Sources              Index Layer              Agent Layer
     +---------+              +-----------+            +----------+
     | PDFs    |--+           | Vector    |            |          |
     | DBs     |  +---------->| Index     +----------->| ReAct    |
     | APIs    |--+    Load   |           |   Query    | Agent    |
     | Notion  |  +---------->| Summary   |   Engine   |          |
     | Slack   |--+           | Index     +----------->| Uses     |
     | S3      |              |           |            | indices  |
     +---------+              | Knowledge |            | as tools |
                              | Graph     |            +----------+
                              +-----------+
```

### LlamaIndex vs LangChain Mental Model

```
LangChain:  Prompt Template --> LLM --> Output Parser --> Next Step
            (Orchestration-first: how do I chain LLM calls?)

LlamaIndex: Data Source --> Index --> Query Engine --> Response
            (Data-first: how do I get the right data to the LLM?)
```

## Production Implementation

### RAG Agent with Query Engine Tools

```python
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.anthropic import Anthropic

# Configure LLM
Settings.llm = Anthropic(model="claude-sonnet-4-20250514")

# -- Load and Index Data --
# Index 1: Product documentation
product_docs = SimpleDirectoryReader("./data/product_docs").load_data()
product_index = VectorStoreIndex.from_documents(product_docs)
product_engine = product_index.as_query_engine(similarity_top_k=5)

# Index 2: Financial reports
financial_docs = SimpleDirectoryReader("./data/financial").load_data()
financial_index = VectorStoreIndex.from_documents(financial_docs)
financial_engine = financial_index.as_query_engine(similarity_top_k=3)

# Index 3: HR policies
hr_docs = SimpleDirectoryReader("./data/hr_policies").load_data()
hr_index = VectorStoreIndex.from_documents(hr_docs)
hr_engine = hr_index.as_query_engine(similarity_top_k=3)

# -- Wrap as Tools --
tools = [
    QueryEngineTool(
        query_engine=product_engine,
        metadata=ToolMetadata(
            name="product_docs",
            description="Search product documentation for features, specs, and guides."
        )
    ),
    QueryEngineTool(
        query_engine=financial_engine,
        metadata=ToolMetadata(
            name="financial_reports",
            description="Search financial reports for revenue, costs, and projections."
        )
    ),
    QueryEngineTool(
        query_engine=hr_engine,
        metadata=ToolMetadata(
            name="hr_policies",
            description="Search HR policies for leave, benefits, and compliance."
        )
    )
]

# -- Create Agent --
agent = ReActAgent.from_tools(
    tools,
    llm=Anthropic(model="claude-sonnet-4-20250514"),
    verbose=True,
    max_iterations=10
)

# -- Query --
response = agent.chat("What was our Q3 revenue and does our leave policy allow sabbaticals?")
print(response)
```

### Sub-Question Query Engine

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# Automatically decomposes complex questions
sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[
        QueryEngineTool(
            query_engine=product_engine,
            metadata=ToolMetadata(name="products", description="Product info")
        ),
        QueryEngineTool(
            query_engine=financial_engine,
            metadata=ToolMetadata(name="finance", description="Financial data")
        )
    ]
)

# This will automatically break into sub-questions and route each
response = sub_question_engine.query(
    "Compare our product revenue to competitor benchmarks"
)
# Internally generates:
# Sub-Q1: "What is our product revenue?" -> financial_engine
# Sub-Q2: "What are competitor benchmarks?" -> product_engine
# Synthesizes: Combined answer
```

## How It Differs from LangChain

| Dimension | LlamaIndex | LangChain |
|-----------|-----------|-----------|
| **Core focus** | Data indexing and retrieval | LLM orchestration and chains |
| **RAG quality** | Superior (built-in advanced retrieval) | Good (requires more setup) |
| **Agent patterns** | ReAct with query engine tools | Multiple patterns via LangGraph |
| **Data connectors** | 160+ built-in (LlamaHub) | Fewer built-in, community-driven |
| **Multi-agent** | Limited (use LangGraph for this) | Full support via LangGraph |
| **Workflow control** | Query pipelines (linear) | Full graph with cycles (LangGraph) |
| **Best for** | Data-heavy RAG agents | Complex multi-step agents |

## Decision Tree: When to Use

```
     Should I use LlamaIndex?
                |
     +----------v----------+
     | Is the agent's      |
     | primary job querying |
     | and reasoning over   |
     | DATA?               |
     +---+------------+----+
         |            |
        Yes          No --> LangGraph or CrewAI
         |
     +---v-----------+----+
     | Do you need        |
     | advanced retrieval? |
     | (sub-questions,    |
     |  re-ranking, KG)   |
     +---+------------+---+
         |            |
        Yes          No --> LangChain RAG may suffice
         |
     +---v-------------------+
     | USE LlamaIndex        |
     | (+ LangGraph for      |
     |  complex agent flow)  |
     +------------------------+
```

## When NOT to Use

1. **Complex multi-agent coordination**: LangGraph or AutoGen are better
2. **Non-data tasks**: If the agent mostly calls APIs or generates code, LangChain is better
3. **Streaming-first applications**: LangGraph's streaming is more mature
4. **Custom execution graphs**: LlamaIndex's workflows are more linear

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Best-in-class data indexing and retrieval | Limited multi-agent support |
| 160+ data connectors (LlamaHub) | Less mature agent patterns than LangGraph |
| Sub-question decomposition built-in | Streaming and HITL less developed |
| Knowledge graph integration | Smaller job market share than LangChain |
| Production RAG with re-ranking | Less community content than LangChain |

## Sources and Further Reading

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [LlamaIndex GitHub](https://github.com/run-llama/llama_index)
- [LlamaHub (Data Connectors)](https://llamahub.ai/)
- [Building Agentic RAG with LlamaIndex](https://docs.llamaindex.ai/en/stable/understanding/agent/)
