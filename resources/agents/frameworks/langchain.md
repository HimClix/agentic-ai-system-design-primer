# LangChain

> The ecosystem glue for LLM applications — chains, LCEL, runnables, and 700+ integrations. LangGraph builds on top of LangChain; you need to understand LangChain first.

## What It Is

LangChain is the foundational framework for building LLM-powered applications. It provides abstractions for:
- **Chains** — composable sequences of LLM calls and transformations
- **LCEL (LangChain Expression Language)** — declarative pipe syntax for building chains
- **Runnables** — the base interface everything implements (invoke, stream, batch)
- **Document Loaders** — ingest from 100+ sources (PDF, web, DB, S3, Confluence)
- **Text Splitters** — chunking strategies for RAG
- **Retrievers** — interface to vector stores, BM25, hybrid search
- **Output Parsers** — structured output extraction
- **Callbacks** — hooks for logging, tracing, streaming

LangChain appears in **34.3% of all agentic job postings** — the #1 framework by market share.

## How It Works

### LCEL (LangChain Expression Language)

The modern way to compose LangChain components using the pipe (`|`) operator:

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specialized in {domain}"),
    ("human", "{question}")
])

model = ChatAnthropic(model="claude-sonnet-4-20250514")
parser = StrOutputParser()

# LCEL chain — pipe syntax
chain = prompt | model | parser

# Invoke
result = chain.invoke({"domain": "finance", "question": "What is DCF?"})

# Stream
for chunk in chain.stream({"domain": "finance", "question": "What is DCF?"}):
    print(chunk, end="")

# Batch
results = chain.batch([
    {"domain": "finance", "question": "What is DCF?"},
    {"domain": "legal", "question": "What is tort?"}
])
```

### Runnables Interface

Every LangChain component implements the `Runnable` interface:

```python
# All runnables support:
runnable.invoke(input)        # Single input → single output
runnable.stream(input)        # Single input → streaming output
runnable.batch([inputs])      # Multiple inputs → multiple outputs
await runnable.ainvoke(input) # Async version of invoke
```

### RAG Chain Example

```python
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough

# Components
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Qdrant.from_existing_collection(
    embedding=embeddings,
    collection_name="docs",
    url="http://localhost:6333"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# RAG chain with LCEL
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

answer = rag_chain.invoke("How does our refund policy work?")
```

### Document Loading + Splitting

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(docs)
```

### Callbacks for Observability

```python
from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler(
    public_key="pk-...",
    secret_key="sk-..."
)

# Pass to any chain invocation
result = chain.invoke(
    {"question": "..."},
    config={"callbacks": [langfuse_handler]}
)
```

## LangChain vs LangGraph

| | LangChain | LangGraph |
|---|---|---|
| **Purpose** | Linear chains, RAG pipelines, simple workflows | Stateful agent workflows with cycles and branching |
| **Control flow** | Sequential (chain A → B → C) | Graph-based (conditional edges, loops, fan-out) |
| **State** | Stateless — each invocation is independent | Stateful — persistent state across steps |
| **Checkpointing** | None built-in | Redis/Postgres checkpointers |
| **HITL** | Not built-in | Native interrupt/resume |
| **When to use** | RAG pipelines, prompt chains, simple tool use | Multi-step agents, supervisor patterns, complex workflows |

**Rule of thumb:** Start with LangChain for RAG and simple chains. Move to LangGraph when you need loops, state, or multi-agent coordination.

## LangChain Ecosystem

```
LangChain (core)
├── langchain-core          — base abstractions (runnables, prompts, parsers)
├── langchain-community     — 700+ integrations (vector stores, loaders, tools)
├── langchain-openai        — OpenAI-specific integrations
├── langchain-anthropic     — Anthropic-specific integrations
├── langchain-google        — Google AI integrations
├── LangGraph               — stateful agent workflows (separate package)
├── LangSmith               — observability + eval platform
├── LangServe               — deploy chains as REST APIs
└── LangFlow                — visual builder (no-code)
```

## Production Implementation

### FastAPI + LangChain RAG Service

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

model = ChatAnthropic(model="claude-sonnet-4-20250514", streaming=True)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on this context:\n{context}"),
    ("human", "{question}")
])
chain = prompt | model | StrOutputParser()

@app.post("/api/ask")
async def ask(question: str):
    async def stream():
        async for chunk in chain.astream({
            "context": await retrieve_context(question),
            "question": question
        }):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(stream(), media_type="text/event-stream")
```

## Decision Tree / When to Use

```
Building a RAG pipeline?
├── YES → LangChain (document loaders, splitters, retrievers, chains)
│
Building a simple prompt chain (A → B → C)?
├── YES → LangChain LCEL
│
Need to call tools in a loop with state?
├── YES → LangGraph (but you'll still use LangChain components inside)
│
Need multi-agent coordination?
├── YES → LangGraph (supervisor/hierarchical) or CrewAI
│
Need data-centric RAG with complex indexing?
├── YES → Consider LlamaIndex as alternative
│
Just need a quick prototype?
├── YES → LangChain — fastest path to working demo
```

## When NOT to Use

- **When you need stateful agent workflows with cycles** — use LangGraph instead (LangChain chains are linear)
- **When you want zero abstraction** — use raw Anthropic/OpenAI SDKs if you want full control
- **When you want type-safe agents** — consider Pydantic AI for strict typing
- **When the framework's abstractions fight your architecture** — Anthropic's advice: "reduce abstraction layers as you move to production"

## Tradeoffs

| Advantage(s) | Disadvantage(s) |
|---|---|
| Largest ecosystem — 700+ integrations | Abstraction overhead — layers between you and the LLM |
| LCEL is elegant and composable | Breaking changes between versions (v0.1 → v0.2 was painful) |
| Best RAG tooling (loaders, splitters, retrievers) | Over-abstraction for simple use cases |
| LangSmith integration for observability | Lock-in to LangChain ecosystem patterns |
| Huge community — most tutorials and examples | "LangChain-specific" jobs pay ~$80K less than framework-agnostic roles |

## Real-World Examples

- **RAG pipelines** — most production RAG systems use LangChain's document loaders + splitters + retrievers even if agents use LangGraph
- **Chatbots** — LangChain's conversation memory + prompt templates for simple Q&A bots
- **Data processing** — document loaders for ingesting from Confluence, Notion, Slack, PDFs at scale
- **Prototyping** — fastest path from idea to working demo because of pre-built integrations

## Failure Modes

| Failure | Cause | Mitigation |
|---|---|---|
| Abstraction leakage | LangChain wraps provider APIs; errors surface as LangChain errors, not provider errors | Learn the underlying APIs, use raw SDKs for debugging |
| Version breakage | Frequent API changes across versions | Pin versions, test before upgrading |
| Performance overhead | Extra abstraction layers add latency | Profile with LangSmith; drop to raw SDK for hot paths |
| Over-engineering | Using LangChain for tasks that need 5 lines of raw API code | Ask: "Would raw SDK be simpler?" If yes, skip LangChain for that component |

## Source(s) and Further Reading

- [LangChain Python docs](https://python.langchain.com/docs/)
- [LangChain Academy](https://academy.langchain.com/) — free courses
- [GitHub: langchain-ai/langchain](https://github.com/langchain-ai/langchain) — 105k+ stars
- [LangChain Job Market 2026](https://agentic-engineering-jobs.com/langchain-job-market-2026) — 34.3% of agentic listings
- [O'Reilly: Learning LangChain](https://github.com/langchain-ai/learning-langchain)
- [Anthropic: "reduce abstraction layers as you move to production"](https://www.anthropic.com/research/building-effective-agents)
