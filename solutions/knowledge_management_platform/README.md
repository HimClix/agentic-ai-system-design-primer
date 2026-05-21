# Design an AI Knowledge Management Platform

> Note: This document links directly to relevant areas of the
> [Agentic AI System Design Primer](../../README.md)

## Step 1: Outline use cases and constraints

### Use cases

- **Semantic search** -- employee types "how do we handle GDPR data deletion requests?" -> system finds the most relevant internal documents, policies, and Slack discussions, returning a synthesized answer with citations
- **Q&A with citations** -- "what's our refund policy for enterprise customers?" -> system retrieves the exact policy document, quotes the relevant section, and provides a link to the source
- **Document ingestion** -- automatically ingest and index documents from Confluence, Google Drive, Slack channels, PDF uploads, and code repositories
- **Document summarization** -- "summarize this 40-page compliance report" -> system generates a structured summary with key takeaways
- **Knowledge graph building** -- automatically discover relationships between concepts, people, teams, and documents ("who owns the payments integration?" -> traces through docs to find the owner)
- **Access control** -- Finance department documents are only visible to Finance team. Engineering docs are visible to Engineering. Cross-department sharing requires explicit grants.
- **Multi-hop questions** -- "What was the decision on the pricing change that the payments team discussed in Q1, and how does it affect the enterprise contract template?" -> agent chains multiple retrievals across different document sources

### Constraints and assumptions

#### State assumptions

- **Users:** 10,000 employees across 20 departments
- **Queries/day:** 50,000 search/Q&A queries
- **Documents:** 1,000,000 documents indexed (growing at ~10K/month)
- **Document types:** PDF, DOCX, Confluence pages, Slack threads, Google Docs, Markdown, code files
- **Latency:** Simple search: < 2 seconds. Q&A with synthesis: < 5 seconds. Multi-hop: < 15 seconds.
- **Accuracy:** 90%+ relevance for top-3 results. Citations must link to correct source document.
- **Access control:** Department-level RBAC on all search results. No document should ever be returned to a user without read permission.
- **Freshness:** Documents updated in source should reflect in search within 1 hour.

#### Calculate usage

```
Queries/day: 50,000
Query breakdown:
  Simple search (keyword + semantic): 30,000 (60%)
  Q&A with synthesis:                 15,000 (30%)
  Multi-hop complex questions:         3,000 (6%)
  Document summarization:              2,000 (4%)

Per query token usage:
  Simple search (no LLM for generation, just retrieval + reranking):
    Reranking: 50 chunks * 200 tokens = 10K tokens (Cohere Rerank API)
    No generation LLM call needed
    Cost: $0.002/query (reranking only)

  Q&A with synthesis:
    Retrieved context:    3,000 tokens (top 5 chunks)
    System prompt:        1,000 tokens
    User query:             100 tokens
    Generated answer:       500 tokens (with citations)
    Total: 4,100 input + 500 output (Sonnet)
    Cost: $0.02/query

  Multi-hop (agent loop, avg 3 retrieval steps):
    3 * (retrieval + reasoning) = 3 * $0.02 = $0.06/query

  Summarization:
    Document: 10,000 tokens input
    Summary: 2,000 tokens output
    Cost: $0.06/query

Daily cost:
  Search:       30,000 * $0.002 = $60
  Q&A:          15,000 * $0.02  = $300
  Multi-hop:     3,000 * $0.06  = $180
  Summarization: 2,000 * $0.06  = $120
  Total: ~$660/day = ~$19,800/month

Document ingestion (one-time + ongoing):
  Initial: 1M docs * avg 5 chunks/doc = 5M chunks
  Embedding: 5M * 500 tokens * $0.10/M tokens (text-embedding-3-small) = $250
  Monthly new: 10K docs * 5 chunks * 500 tokens * $0.10/M = $2.50/month
  Storage: 5M vectors * 1536 dims * 4 bytes = ~30GB in vector DB

Reranking cost:
  45,000 queries/day needing reranking * $0.002/query = $90/day = $2,700/month
```

### Out of scope

- Building document editors (users create documents in existing tools)
- Real-time collaboration features
- Document versioning (rely on source system versioning)
- Translation/multi-language (English only for v1)
- Training custom embedding models
- Building the identity provider (integrate with existing SSO/SAML)

---

## Step 2: Create a high-level design

```
                          ┌─────────────────────┐
                          │  Employees (Web UI)  │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │    API Gateway       │
                          │  Auth · RBAC · Rate  │
                          │  Limit · Session     │
                          └──────────┬──────────┘
                                     │
              ┌──────────────────────▼──────────────────────┐
              │              QUERY ROUTER                    │
              │  Classifies query type:                      │
              │  - Simple search (keyword/semantic)          │
              │  - Q&A (needs synthesis)                     │
              │  - Multi-hop (needs agent reasoning)         │
              │  - Summarization (needs doc + LLM)           │
              └─────┬──────────┬──────────┬────────────────┘
                    │          │          │
         ┌──────────▼┐  ┌─────▼─────┐  ┌▼──────────────┐
         │  Search    │  │ Q&A       │  │ Multi-Hop     │
         │  Pipeline  │  │ Pipeline  │  │ Agent (ReAct) │
         │            │  │           │  │               │
         │ No LLM     │  │ Retrieve  │  │ Iterative     │
         │ generation │  │ + Generate│  │ retrieve +    │
         │            │  │ + Cite    │  │ reason +      │
         │            │  │           │  │ synthesize    │
         └──────┬─────┘  └─────┬─────┘  └──────┬────────┘
                │              │               │
         ┌──────▼──────────────▼───────────────▼──────────┐
         │              RAG PIPELINE                        │
         │                                                  │
         │  ┌──────────────────────────────────────────┐  │
         │  │           RETRIEVAL LAYER                  │  │
         │  │                                            │  │
         │  │  ┌─────────┐  ┌──────────┐  ┌─────────┐ │  │
         │  │  │ BM25    │  │ Vector   │  │ Hybrid  │ │  │
         │  │  │ (keyword)│  │ (semantic)│  │ (RRF)  │ │  │
         │  │  └────┬────┘  └────┬─────┘  └────┬────┘ │  │
         │  │       └────────────┼──────────────┘      │  │
         │  │                    ▼                      │  │
         │  │  ┌─────────────────────────────────────┐ │  │
         │  │  │        RE-RANKING (Cohere)           │ │  │
         │  │  │  50 candidates → top 5 results       │ │  │
         │  │  └─────────────────────────────────────┘ │  │
         │  └──────────────────────────────────────────┘  │
         │                                                  │
         │  ┌──────────────────────────────────────────┐  │
         │  │      ACCESS CONTROL FILTER                │  │
         │  │  Filter results by user's department      │  │
         │  │  and document-level permissions            │  │
         │  └──────────────────────────────────────────┘  │
         │                                                  │
         │  ┌──────────────────────────────────────────┐  │
         │  │      GENERATION + CITATION                │  │
         │  │  LLM synthesizes answer from top chunks   │  │
         │  │  with inline citations [1], [2], [3]      │  │
         │  └──────────────────────────────────────────┘  │
         └────────────────────┬───────────────────────────┘
                              │
         ┌────────────────────▼───────────────────────────┐
         │           INGESTION PIPELINE                     │
         │                                                  │
         │  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
         │  │ Source    │  │ Document │  │ Chunking     │ │
         │  │ Connectors│  │ Parser   │  │ Engine       │ │
         │  │           │  │          │  │              │ │
         │  │ -Confluence│  │ -PDF     │  │ -Recursive  │ │
         │  │ -GDrive   │  │ -DOCX    │  │ -Semantic   │ │
         │  │ -Slack    │  │ -HTML    │  │ -512 tokens │ │
         │  │ -GitHub   │  │ -Markdown│  │  with 10%   │ │
         │  │ -Manual   │  │ -Code    │  │  overlap    │ │
         │  └──────────┘  └──────────┘  └──────┬───────┘ │
         │                                      │         │
         │  ┌──────────┐  ┌──────────┐  ┌──────▼───────┐ │
         │  │ Metadata │  │ Access   │  │ Embedding   │ │
         │  │ Enrichment│  │ Control  │  │ (OpenAI     │ │
         │  │ (dept,   │  │ Tag      │  │  text-embed │ │
         │  │  author, │  │          │  │  -3-small)  │ │
         │  │  date)   │  │          │  │             │ │
         │  └──────────┘  └──────────┘  └──────┬──────┘ │
         │                                      │        │
         │                            ┌─────────▼──────┐ │
         │                            │ Vector DB      │ │
         │                            │ (Qdrant/       │ │
         │                            │  Weaviate)     │ │
         │                            └────────────────┘ │
         └────────────────────────────────────────────────┘
                              │
         ┌────────────────────▼───────────────────────────┐
         │              DATA LAYER                          │
         │  ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
         │  │ Qdrant   │ │PostgreSQL│ │ Redis        │  │
         │  │ (vectors)│ │(metadata │ │ (query cache │  │
         │  │ 5M chunks│ │ ACL,     │ │  user prefs) │  │
         │  │          │ │ kg nodes)│ │              │  │
         │  └──────────┘ └──────────┘ └──────────────┘  │
         │  ┌──────────────────────────────────────────┐ │
         │  │ Elasticsearch (BM25 keyword search)       │ │
         │  └──────────────────────────────────────────┘ │
         └────────────────────────────────────────────────┘
```

---

## Step 3: Design core components

### Agent Architecture Decision

**Routing pattern** for query classification, with a **ReAct agent** only for multi-hop questions.

Cross-reference: [Workflow vs Agent](../../README.md#workflow-vs-agent) -- "Start with workflows. Move to agents only when workflows can't handle the task's dynamic nature."

**Why routing (not a full agent for everything):**
- 60% of queries are simple search -- no LLM generation needed, just retrieval + reranking
- 30% are Q&A -- a single retrieve-then-generate pipeline, not a reasoning loop
- Only 6% are multi-hop questions that need iterative reasoning (true agent behavior)

Using a ReAct agent for every query would be 5-10x more expensive for no benefit on simple queries.

### Pattern Selection

**Routing workflow** (Anthropic pattern #2) + **ReAct** for complex queries only.

Cross-reference: [Agent Architecture Patterns](../../README.md#agent-architecture-patterns)

```python
# Query router (Haiku, cheap classification)
def route_query(query: str) -> str:
    """Classify into: search, qa, multi_hop, summarize"""
    response = haiku.invoke(f"""
    Classify this knowledge base query:
    - "search": Looking for documents, simple factual lookup
    - "qa": Needs a synthesized answer with citations
    - "multi_hop": Requires combining info from multiple sources or reasoning chains
    - "summarize": Asking to summarize a specific document
    
    Query: {query}
    """)
    return response  # "search" | "qa" | "multi_hop" | "summarize"
```

**Multi-hop ReAct example:**
```
User: "What was the decision on the pricing change that the payments team
       discussed in Q1, and how does it affect the enterprise contract template?"

Thought: This requires finding 2 pieces of info and connecting them.
  First, I need to find the pricing decision from Q1.

Action: search_knowledge_base(query="payments team pricing change decision Q1 2026",
        departments=["payments", "product"])
Observation: Found: "Pricing Committee Notes - March 2026" — decided to increase
  enterprise tier from $50/seat to $65/seat effective June 1.

Thought: Now I need to find how this affects the enterprise contract template.

Action: search_knowledge_base(query="enterprise contract template pricing terms",
        departments=["legal", "sales"])
Observation: Found: "Enterprise Contract Template v4.2" — Section 5.3 specifies
  pricing is locked for the contract duration. Amendment needed for renewals.

Thought: I can now synthesize the answer.

Action: respond("""
  The payments team decided in Q1 2026 to increase enterprise pricing from
  $50/seat to $65/seat, effective June 1 [1]. This affects the enterprise
  contract template in Section 5.3 [2], which locks pricing for the contract
  duration. Existing contracts are unaffected, but renewals after June 1 will
  need to reference the new pricing schedule. An amendment process will be
  needed for mid-term price adjustments.

  [1] Pricing Committee Notes - March 2026 (link)
  [2] Enterprise Contract Template v4.2, Section 5.3 (link)
""")
```

### RAG Pipeline Design

Cross-reference: [RAG](../../README.md#rag-retrieval-augmented-generation)

#### Ingestion Pipeline

```
Document arrives (new or updated)
    │
    ▼
┌─────────────────────────────────────┐
│ 1. SOURCE CONNECTOR                 │
│    Confluence API / GDrive API /    │
│    Slack API / manual upload        │
│    Detects: new, updated, deleted   │
└──────────────┬──────────────────────┘
               │
    ┌──────────▼──────────┐
    │ 2. DOCUMENT PARSER  │
    │    PDF → text (PyMuPDF)         │
    │    DOCX → text (python-docx)    │
    │    HTML → text (BeautifulSoup)  │
    │    Slack → thread text          │
    │    Code → AST-aware chunking    │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ 3. CHUNKING          │
    │    Strategy: Recursive         │
    │    Size: 512 tokens            │
    │    Overlap: 50 tokens (10%)    │
    │    Respect: headers, sections  │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ 4. METADATA ENRICHMENT│
    │    - department (from source)  │
    │    - author                    │
    │    - created_date / updated    │
    │    - document_type             │
    │    - access_level              │
    │    - source_url                │
    │    - parent_document_id        │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ 5. EMBEDDING          │
    │    Model: text-embedding-3-small│
    │    Dimensions: 1536            │
    │    Batch size: 100 chunks      │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ 6. INDEX              │
    │    Vector DB: Qdrant           │
    │    + Elasticsearch (BM25)      │
    │    + PostgreSQL (metadata)     │
    └─────────────────────┘
```

#### Retrieval Pipeline

```
User query arrives
    │
    ▼
┌─────────────────────────────────────┐
│ 1. QUERY PROCESSING                 │
│    - Expand abbreviations           │
│    - Spell correction               │
│    - Query rewriting for multi-hop  │
│      "our refund policy" →          │
│      "company refund return policy  │
│       customer enterprise"          │
└──────────────┬──────────────────────┘
               │
    ┌──────────▼──────────┐
    │ 2. HYBRID RETRIEVAL  │
    │    BM25 (Elasticsearch): top 30│
    │    Vector (Qdrant): top 30     │
    │    Fuse with RRF (k=60)        │
    │    Result: top 50 candidates   │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ 3. ACCESS FILTER     │
    │    Filter candidates by:       │
    │    - user's department         │
    │    - document access_level     │
    │    - explicit shares           │
    │    MUST happen BEFORE reranking│
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ 4. RE-RANKING         │
    │    Cohere Rerank v3            │
    │    Input: 50 candidates        │
    │    Output: top 5 by relevance  │
    │    Latency: ~100ms             │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ 5. GENERATION         │
    │    Claude Sonnet with context: │
    │    - Top 5 chunks (3K tokens)  │
    │    - System prompt (1K tokens) │
    │    - Instruction: cite sources │
    │    Output: answer + [1][2][3]  │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ 6. CITATION ASSEMBLY  │
    │    Map [1] → source URL        │
    │    Map [2] → source URL        │
    │    Verify citations are valid  │
    └─────────────────────┘
```

### Tool Design

```json
{
  "tools": [
    {
      "name": "search_knowledge_base",
      "description": "Search the enterprise knowledge base using hybrid search (keyword + semantic). Returns ranked document chunks with source citations. Results are filtered by the requesting user's access permissions.",
      "input_schema": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "Search query in natural language"},
          "departments": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Filter by department. Omit to search all accessible departments."
          },
          "document_types": {
            "type": "array",
            "items": {"type": "string", "enum": ["confluence", "gdrive", "slack", "pdf", "code", "all"]},
            "default": ["all"]
          },
          "date_range": {
            "type": "object",
            "properties": {
              "start": {"type": "string", "format": "date"},
              "end": {"type": "string", "format": "date"}
            }
          },
          "max_results": {"type": "integer", "default": 5, "maximum": 20}
        },
        "required": ["query"]
      }
    },
    {
      "name": "get_document",
      "description": "Retrieve the full content of a specific document by ID. Use when you need more context than the search chunk provides.",
      "input_schema": {
        "type": "object",
        "properties": {
          "document_id": {"type": "string"},
          "sections": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific sections to retrieve. Omit for full document."
          }
        },
        "required": ["document_id"]
      }
    },
    {
      "name": "summarize_document",
      "description": "Generate a structured summary of a document. Returns key points, decisions, action items, and relevant dates.",
      "input_schema": {
        "type": "object",
        "properties": {
          "document_id": {"type": "string"},
          "summary_type": {
            "type": "string",
            "enum": ["executive", "detailed", "key_decisions", "action_items"],
            "default": "executive"
          },
          "max_length": {"type": "integer", "default": 500, "description": "Max words in summary"}
        },
        "required": ["document_id"]
      }
    },
    {
      "name": "query_knowledge_graph",
      "description": "Query the organizational knowledge graph for relationships between people, teams, projects, and concepts.",
      "input_schema": {
        "type": "object",
        "properties": {
          "entity": {"type": "string", "description": "Person, team, project, or concept to query about"},
          "relationship_type": {
            "type": "string",
            "enum": ["owns", "authored", "related_to", "depends_on", "reports_to", "member_of"],
            "description": "Type of relationship to explore"
          },
          "depth": {"type": "integer", "default": 1, "maximum": 3, "description": "How many hops to traverse"}
        },
        "required": ["entity"]
      }
    },
    {
      "name": "format_citation",
      "description": "Format a citation reference for inclusion in a response. Returns a structured citation with title, URL, author, and date.",
      "input_schema": {
        "type": "object",
        "properties": {
          "document_id": {"type": "string"},
          "chunk_id": {"type": "string"},
          "highlight_text": {"type": "string", "description": "The specific text being cited"}
        },
        "required": ["document_id"]
      }
    }
  ]
}
```

### Access Control Architecture (Critical Design)

**This is the highest-stakes component.** A RAG retrieval bypass that leaks HR compensation data to engineering is a fireable security incident.

```
Access Control Layers:

1. INGESTION TIME (proactive):
   - Every chunk tagged with: department, access_level, owner
   - Access levels: "public", "department", "restricted", "confidential"
   - Inherited from source: Confluence space permissions, GDrive folder sharing

2. QUERY TIME (enforced):
   - User's department and role extracted from JWT
   - Vector DB query includes metadata filter:
     qdrant.search(
       query_vector=embed(query),
       query_filter=Filter(
         must=[
           FieldCondition(key="access_level", match=MatchAny(any=["public", "department"])),
           FieldCondition(key="department", match=MatchAny(any=user.departments))
         ]
       )
     )
   - Access filter applied BEFORE reranking (never score a document the user cannot see)

3. RESPONSE TIME (defensive):
   - PII scanner on generated output
   - Citation URLs validated against user permissions
   - Audit log: who searched what, what results were shown
```

**Why filter before reranking (not after):**
If you rerank first and filter after, the reranker has seen confidential documents in its scoring context. Even if you remove them from the final results, the reranker's scoring of visible documents may be influenced by the confidential ones (position bias). Filtering first ensures clean separation.

### Memory Architecture

| Memory Type | Store | What It Holds | TTL |
|---|---|---|---|
| **Session** | Redis | Current search context, conversation history, recently viewed documents | 30 min idle timeout |
| **Long-term** (knowledge graph) | PostgreSQL + Neo4j | Entity relationships: teams, projects, owners, dependencies. Built from document analysis. | Continuously updated |
| **Episodic** (search history) | PostgreSQL | Per-user search history for personalization. "You frequently search for payments-related docs" -> boost payments results. | 90 days |
| **Semantic cache** | Redis + vector similarity | Cache answers to frequently asked questions. "What's our refund policy?" is asked 50x/week. | 24 hours (documents may update) |

### API Design

```
POST /api/v1/knowledge/query
Content-Type: application/json
Authorization: Bearer <jwt_token>

Request:
{
  "query": "What's our refund policy for enterprise customers?",
  "session_id": "sess_abc123",
  "stream": true,
  "include_sources": true,
  "departments": null,         // null = search all accessible
  "max_sources": 5
}

Response (SSE stream):
data: {"type": "query_type", "classification": "qa"}
data: {"type": "sources_found", "count": 5, "top_source": "Enterprise Refund Policy v3.1"}
data: {"type": "answer_chunk", "content": "Our enterprise refund policy allows "}
data: {"type": "answer_chunk", "content": "full refunds within 30 days of purchase "}
data: {"type": "answer_chunk", "content": "for annual contracts [1]. For monthly "}
data: {"type": "answer_chunk", "content": "contracts, prorated refunds are available "}
data: {"type": "answer_chunk", "content": "at any time [2]."}
data: {"type": "citations", "sources": [
  {"id": "[1]", "title": "Enterprise Refund Policy v3.1", "url": "https://confluence.internal/...", "author": "Legal Team", "updated": "2026-03-15", "relevance_score": 0.96},
  {"id": "[2]", "title": "Billing FAQ - Enterprise", "url": "https://confluence.internal/...", "author": "Finance Team", "updated": "2026-04-02", "relevance_score": 0.91}
]}
data: {"type": "done", "usage": {"retrieval_ms": 180, "rerank_ms": 95, "generation_ms": 1200, "total_ms": 1475, "tokens": 4600, "cost_usd": 0.019}}
```

---

## Step 4: Scale the design

### Bottleneck Analysis

| Bottleneck | At 10K Users | At 100K Users | Mitigation |
|---|---|---|---|
| **Vector search latency** | Qdrant handles 50K queries/day easily | 500K queries/day, search latency increases | Qdrant cluster with 3 shards. Quantization (binary quantization reduces memory 32x with ~2% accuracy loss). |
| **Embedding cost at ingestion** | 1M docs = $250 one-time | 10M docs = $2,500 + reprocessing on model change | Batch embedding pipeline. Cache embeddings. Only re-embed on document update. |
| **Reranking** | 45K rerank calls/day ($2,700/month) | 450K calls/day ($27K/month) | Self-hosted reranker (BGE Reranker) for cost reduction. Cache reranking results for identical query-document pairs (1-hour TTL). |
| **Access control filtering** | Manageable with metadata filters | Complex cross-department sharing | Pre-compute user access bitmasks. Store in Redis. Filter at the vector DB level using payload filters. |
| **Document freshness** | Hourly sync from sources | Near-real-time needed | Webhook-based ingestion from Confluence/GDrive. Slack real-time events API. Incremental updates (only re-embed changed sections). |

### Cost Estimation

| Component | Monthly Cost | Notes |
|---|---|---|
| **LLM - Sonnet (Q&A + multi-hop)** | $14,400 | 18K synthesis calls/day at avg $0.027/call |
| **LLM - Haiku (routing)** | $150 | 50K classification calls/day at $0.0001/call |
| **Cohere Rerank** | $2,700 | 45K calls/day at $0.002/call |
| **Embedding (ongoing)** | $3 | 10K new docs/month, negligible |
| **Qdrant (managed)** | $1,200 | 5M vectors, 30GB, 3-node cluster |
| **Elasticsearch** | $800 | BM25 index, 1M documents |
| **PostgreSQL** | $600 | Metadata, access control, knowledge graph |
| **Redis** | $300 | Session cache, query cache |
| **S3 (document storage)** | $200 | Original documents + processed chunks |
| **Infrastructure** | $2,000 | 4 app nodes, ingestion workers |
| **Langfuse** | $59 | Observability |
| **Total** | **~$22,412/month** | **$0.015 per query average** |

**Cost per query type:**

| Query Type | Cost | Volume/Day | Daily Cost |
|---|---|---|---|
| Simple search | $0.002 | 30,000 | $60 |
| Q&A with synthesis | $0.022 | 15,000 | $330 |
| Multi-hop (agent) | $0.068 | 3,000 | $204 |
| Summarization | $0.063 | 2,000 | $126 |

### Failure Modes

Cross-reference: [Failure Modes & Mitigation](../../README.md#failure-modes--mitigation)

| Failure Mode | Severity | Detection | Mitigation |
|---|---|---|---|
| **Stale documents** | Medium | User reports outdated answer. Document shows old version. | Source sync health check: alert if any connector hasn't synced in 2 hours. Show "last updated" date on every citation. |
| **Access control bypass via RAG** | **Critical** | Confidential document chunk appears in results for unauthorized user | Defense in depth: 1) Filter at ingestion (tag every chunk), 2) Filter at query (metadata filter in vector search), 3) Filter at response (verify citation URLs against user permissions). Regular audit: random sample of 100 queries/day checked for ACL violations. |
| **Citation to wrong document** | High | Citation [1] links to a document that doesn't contain the quoted text | Citation verification step: before returning, check that the cited text actually exists in the linked document. If not, flag as "unverified citation." |
| **Confidential data in generated response** | **Critical** | LLM synthesizes an answer that includes confidential information from a document the user can access but shouldn't quote externally | PII/confidentiality scanner on generated output. Classify documents as "quotable" vs "reference-only." |
| **Knowledge graph staleness** | Low | "Who owns the payments service?" returns someone who left 3 months ago | Periodic reconciliation against HR system. Flag entities not updated in 90 days. |
| **Retrieval failure (relevant doc not found)** | Medium | User gets "I don't have information about that" when the doc exists | Monitor "no results" rate. If > 10%, investigate: is it a chunking issue? Embedding quality? Query rewriting needed? |
| **Hallucination (answer not grounded in sources)** | High | Answer contains information not present in retrieved chunks | Faithfulness check: score generated answer against retrieved context using Sonnet-as-judge. Flag answers scoring < 0.8 with "This answer may not be fully supported by sources." |

### Observability Setup

**Tracing (Langfuse):**
- Every query gets a trace: query_type classification, retrieval step (chunks retrieved with scores), reranking step (before/after order), generation step (prompt + response), citation assembly
- RAG-specific metrics per trace: retrieval recall (did we find the right docs?), faithfulness score (is the answer grounded?)

**Metrics:**

| Metric | Alert Threshold |
|---|---|
| `km_query_latency_p95{type="search"}` | > 3s |
| `km_query_latency_p95{type="qa"}` | > 8s |
| `km_retrieval_empty_rate` | > 15% (retrieval quality issue) |
| `km_reranker_latency_p95` | > 500ms |
| `km_faithfulness_score_p50` | < 0.85 (hallucination risk) |
| `km_acl_filter_ratio` | Monitor but no alert (understand how much gets filtered) |
| `km_citation_verification_fail_rate` | > 5% (citation quality issue) |
| `km_source_sync_lag_hours` | > 2 hours for any connector |
| `km_cache_hit_rate` | < 20% (caching not effective) |
| `km_cost_per_query_p95` | > $0.10 (cost anomaly) |

**RAG Quality Dashboard:**
- Daily RAGAS scores (sampled 200 queries/day):
  - Faithfulness: is the answer grounded in context?
  - Context Precision: are retrieved chunks relevant?
  - Context Recall: did we find all relevant info?
  - Answer Relevancy: does the answer address the question?

Cross-reference: [Evaluation Frameworks -- RAGAS](../../README.md#ragas-rag-assessment)

---

## Additional Talking Points

### Chunking Strategy Matters More Than Embedding Model

At 1M documents, chunking quality has 3x more impact on retrieval accuracy than switching between embedding models. Our approach:

1. **Recursive chunking** -- respect document structure (headers, sections, paragraphs)
2. **512-token chunks** -- sweet spot between context richness and retrieval precision
3. **50-token overlap** -- prevents losing context at chunk boundaries
4. **Parent-child linking** -- each chunk knows its parent document and section header. If a chunk is retrieved, we can optionally include the section header for context.

### Hybrid Search Is Non-Negotiable

Neither BM25 nor vector search alone is sufficient:

```
Query: "SOC2 compliance audit checklist"

BM25 finds: Documents with exact term "SOC2" (keyword match)
Vector finds: Documents about "security compliance frameworks" (semantic match)
Hybrid (RRF): Both types of relevant documents, ranked by combined relevance

Without BM25: Miss documents that use the exact acronym "SOC2"
Without Vector: Miss documents about compliance that use different terminology
```

Cross-reference: [RAG -- Hybrid Search](../../README.md#hybrid-search-bm25--vector)

### Knowledge Graph as a Differentiator

The knowledge graph transforms the system from "document search" to "organizational intelligence":

```
Traditional search: "Who owns the payments integration?"
→ Returns documents that mention "payments integration" and hope one mentions the owner

Knowledge graph search:
→ Entity: "Payments Integration"
→ Relationship: "owned_by" → "Payments Platform Team"
→ Relationship: "tech_lead" → "Alex Chen"
→ Direct answer: "The Payments Integration is owned by the Payments Platform Team, with Alex Chen as tech lead."
```

Building the knowledge graph: LLM-based entity extraction during ingestion. Extract (entity, relationship, entity) triples from every document. Store in a graph database (Neo4j) or in PostgreSQL with adjacency lists.

### Access Control is the #1 Deployment Blocker

In every enterprise KM deployment, access control is the feature that takes the longest and causes the most deployment delays. Our architecture handles this by making ACL a first-class concern at every layer, not an afterthought filter on top of search results.

### Interview Cross-References
- RAG pipeline architecture: [RAG](../../README.md#rag-retrieval-augmented-generation)
- Hybrid search: [RAG -- Hybrid Search](../../README.md#hybrid-search-bm25--vector)
- Reranking: [RAG -- Re-ranking](../../README.md#re-ranking)
- Chunking strategies: [RAG -- Chunking Strategies](../../README.md#chunking-strategies)
- RAGAS evaluation: [Evaluation Frameworks -- RAGAS](../../README.md#ragas-rag-assessment)
- Routing as workflow pattern: [Workflow vs Agent](../../README.md#workflow-vs-agent)
- Multi-tenancy / access control: [Scalability -- Multi-Tenancy](../../README.md#multi-tenancy)
- Faithfulness scoring: [Evaluation Frameworks -- LLM-as-a-Judge](../../README.md#llm-as-a-judge)
- Cost of embeddings at scale: [Cost Engineering](../../README.md#cost-engineering)
- Prompt injection via retrieved documents: [Safety & Guardrails -- Prompt Injection](../../README.md#prompt-injection)
