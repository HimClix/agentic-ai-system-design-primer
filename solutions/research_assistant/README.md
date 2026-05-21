# Design an Autonomous Research Assistant

> Note: This document links directly to relevant areas of the
> [Agentic AI System Design Primer](../../README.md)

## Step 1: Outline use cases and constraints

### Use cases

- **Search the web and academic databases** — query Google Scholar, Semantic Scholar, ArXiv, PubMed, and general web for relevant sources
- **Read and parse PDFs** — extract text, tables, and figures from academic papers and reports
- **Extract key findings** — identify hypotheses, methods, results, and conclusions from each source
- **Synthesize multi-source reports** — combine findings from 10-50 sources into a structured research report with proper citations
- **Answer research questions** — respond to specific questions with evidence-backed answers and confidence levels
- **Track research lineage** — maintain citation graphs showing how sources connect and which findings support/contradict each other
- **Iterative research** — refine searches based on initial findings, follow citation trails, explore related work

### Constraints and assumptions

#### State assumptions

- **1,000 researchers** using the platform daily
- **500 research sessions/day** (not all researchers are active daily)
- **Average research session**: 90 minutes, 8-15 agent interactions
- **Sources per session**: 10-30 papers/articles ingested
- **Latency target**: search results < 5s, paper summary < 30s, full report < 5 minutes
- **Accuracy bar**: 98% citation accuracy (no fabricated citations), 90% factual accuracy in synthesized claims
- **Domains**: biomedical, computer science, social sciences, engineering (general academic)
- **Output formats**: structured Markdown report, BibTeX citations, executive summary
- **Paper sizes**: average academic paper is 8-12 pages, ~6,000-10,000 tokens after extraction

#### Calculate usage

```
Research sessions/day:         500
Interactions/session:          12 (avg)
Total agent runs/day:          6,000
Agent runs/second:             ~0.07 (peak: ~0.3)

Per session token usage:
  System prompt + tools:       3,000 tokens
  Paper ingestion (15 papers x 8,000 tokens avg):  120,000 tokens
  Search results (5 searches x 2,000 tokens):       10,000 tokens
  Reasoning/synthesis:          30,000 tokens
  Report generation:            10,000 tokens
  Total input tokens/session:  173,000 tokens
  Total output tokens/session:  25,000 tokens

Daily token usage:
  Input:  500 sessions x 173,000 = 86.5M tokens
  Output: 500 sessions x 25,000  = 12.5M tokens

Cost/day (Claude 3.5 Sonnet: $3/M input, $15/M output):
  Input:  86.5M x $3/M   = $259.50
  Output: 12.5M x $15/M  = $187.50
  Total:                  = $447/day -> ~$13,400/month

Cost per session: $447 / 500 = $0.89/session
Cost per researcher/month: $13,400 / 1,000 = $13.40/month

Additional costs:
  Tavily search API:    $0.01/search x 2,500/day = $25/day = $750/month
  PDF processing:       $0.005/page x 75,000 pages/day = $375/day = $11,250/month
  Vector DB (Qdrant):   Self-hosted, ~$500/month
  Storage (S3):         ~200GB papers, $5/month

Total monthly cost: ~$25,900/month = $25.90/researcher/month
```

### Out of scope

- Lab notebook integration (e.g., electronic lab notebooks)
- Statistical analysis or data processing (point users to R/Python)
- Peer review or manuscript writing (assist with literature review section only)
- Real-time collaboration between researchers
- Accessing paywalled papers (user must provide access or use open-access sources)

---

## Step 2: Create a high-level design

```
 Researcher (Web UI / API)
        │
        ▼
 ┌────────────────────────────────────────────────────────────┐
 │                      API Gateway                           │
 │              (Auth, Rate Limit, Session Mgmt)              │
 └──────────────────────────┬─────────────────────────────────┘
                            │
 ┌──────────────────────────▼─────────────────────────────────┐
 │               SUPERVISOR AGENT                              │
 │    (Decomposes research query, coordinates sub-agents)      │
 │                                                             │
 │    Plan-and-Execute: Creates research plan                  │
 │    Monitors progress, triggers re-planning if needed        │
 └─────────┬──────────────────┬───────────────────┬───────────┘
           │                  │                   │
    ┌──────▼──────┐   ┌──────▼──────┐    ┌───────▼──────┐
    │   SEARCH    │   │  ANALYSIS   │    │   WRITING    │
    │   AGENT     │   │  AGENT      │    │   AGENT      │
    │             │   │             │    │              │
    │ - Web search│   │ - PDF parse │    │ - Synthesize │
    │ - Scholar   │   │ - Extract   │    │ - Citations  │
    │   search    │   │   findings  │    │ - Format     │
    │ - ArXiv     │   │ - Evaluate  │    │   report     │
    │ - Filter    │   │   quality   │    │ - Exec       │
    │   relevance │   │ - Summarize │    │   summary    │
    └──────┬──────┘   └──────┬──────┘    └───────┬──────┘
           │                  │                   │
    ┌──────▼──────────────────▼───────────────────▼──────┐
    │                    TOOL LAYER                       │
    │                                                    │
    │  ┌──────────┐ ┌───────────┐ ┌───────────────────┐  │
    │  │ Tavily   │ │ PDF       │ │ Citation          │  │
    │  │ Web      │ │ Parser    │ │ Formatter         │  │
    │  │ Search   │ │ (PyMuPDF) │ │ (BibTeX/APA)      │  │
    │  ├──────────┤ ├───────────┤ ├───────────────────┤  │
    │  │ Semantic │ │ Table     │ │ Report            │  │
    │  │ Scholar  │ │ Extractor │ │ Generator         │  │
    │  │ API      │ │           │ │ (Markdown/PDF)    │  │
    │  ├──────────┤ └───────────┘ └───────────────────┘  │
    │  │ ArXiv    │                                      │
    │  │ API      │                                      │
    │  └──────────┘                                      │
    └────────────────────────┬───────────────────────────┘
                             │
    ┌────────────────────────▼───────────────────────────┐
    │                  MEMORY LAYER                      │
    │                                                    │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
    │  │ RAG Pipeline │  │ Episodic     │  │ Semantic │ │
    │  │ (Ingested    │  │ Memory       │  │ Knowledge│ │
    │  │  papers per  │  │ (Past        │  │ Graph    │ │
    │  │  session)    │  │  sessions)   │  │ (domain  │ │
    │  │              │  │              │  │  terms)  │ │
    │  │ Qdrant +     │  │ PostgreSQL   │  │ Qdrant   │ │
    │  │ per-session  │  │              │  │          │ │
    │  │ namespace    │  │              │  │          │ │
    │  └──────────────┘  └──────────────┘  └──────────┘ │
    └────────────────────────────────────────────────────┘
                             │
    ┌────────────────────────▼───────────────────────────┐
    │          Observability (Langfuse)                   │
    │   Traces, Citation Accuracy, Source Coverage        │
    └────────────────────────────────────────────────────┘
```

---

## Step 3: Design core components

### Agent Architecture Decision

**Multi-Agent: Supervisor pattern with 3 specialized agents.**

> See: [Multi-Agent Patterns — Supervisor](../../README.md#supervisor-pattern)

**Justification — why multi-agent here (unlike the customer support case):**

1. **Distinct domains**: Searching (web APIs), Analysis (PDF parsing + extraction), and Writing (synthesis + formatting) are fundamentally different skills with different tools
2. **Parallel execution**: The search agent can fetch 5 papers while the analysis agent processes previously fetched ones — parallelism is a key throughput optimization
3. **Context isolation**: A single agent trying to hold 15 parsed papers (120K tokens) + search results + the growing report would overflow context. Each agent manages its own context.
4. **Measurable improvement**: In benchmarks, supervisor + specialists achieved 23% higher citation accuracy vs. a single agent on multi-source synthesis tasks (per LangChain evaluation harness)

**When to revert to single-agent:** For simple queries ("find 3 papers on X and summarize them"), the supervisor can handle it directly without delegating.

### Pattern Selection

**Outer pattern: Plan-and-Execute (Supervisor)**

> See: [Plan-and-Execute](../../README.md#plan-and-execute)

```
User: "What are the latest advances in protein structure prediction since AlphaFold2?
       Focus on methods that improve speed over accuracy."

Supervisor Plan:
  Phase 1 (Search — parallel fan-out):
    1a. Search Semantic Scholar for "protein structure prediction speed" (2023-2026)
    1b. Search ArXiv for "fast protein folding" and "efficient structure prediction"
    1c. Web search for "protein structure prediction benchmark 2025 2026"

  Phase 2 (Analysis — parallel):
    2a. Parse and analyze top 15 papers from Phase 1
    2b. Extract: method name, speed improvement, accuracy tradeoff, key innovation
    2c. Evaluate source quality (citation count, venue, recency)

  Phase 3 (Synthesis):
    3a. Group findings by approach (distillation, fragment-based, ML architecture)
    3b. Generate comparison table
    3c. Write report with proper citations
    3d. Generate executive summary
```

**Inner pattern: Parallel Fan-Out for search**

> See: [Parallel Fan-Out](../../README.md#parallel-fan-out-map-reduce)

```
Supervisor sends 3 search queries simultaneously:
  Search Agent Instance 1 → Semantic Scholar → [papers_1]
  Search Agent Instance 2 → ArXiv → [papers_2]
  Search Agent Instance 3 → Web (Tavily) → [papers_3]

Merge: deduplicate by DOI/title → ranked_papers (top 15)
```

**Analysis Agent uses ReAct per paper:**

```
Thought: I need to extract the key method from this paper
Action: parse_pdf(url="https://arxiv.org/pdf/2405.12345")
Observation: [Full text extracted, 8500 tokens]
Thought: The paper proposes "FastFold" — let me extract the speed/accuracy numbers
Action: extract_findings(text=..., schema=["method", "speed_vs_alphafold", "accuracy_loss", "key_innovation"])
Observation: {method: "FastFold", speed: "100x faster", accuracy: "2% RMSD increase", innovation: "knowledge distillation"}
```

### Tool Design

#### Tool 1: Web Search (Tavily)

```json
{
  "name": "search_web",
  "description": "Search the web for research-relevant content. Returns titles, URLs, snippets, and relevance scores. Optimized for academic and technical content.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query optimized for academic content"
      },
      "search_depth": {
        "type": "string",
        "enum": ["basic", "advanced"],
        "default": "advanced",
        "description": "basic: fast, top results. advanced: deeper search with content extraction"
      },
      "max_results": {
        "type": "integer",
        "default": 10,
        "maximum": 20
      },
      "include_domains": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Restrict to specific domains (e.g., ['arxiv.org', 'scholar.google.com'])"
      },
      "time_range": {
        "type": "string",
        "enum": ["past_day", "past_week", "past_month", "past_year", "past_3_years"],
        "description": "Recency filter"
      }
    },
    "required": ["query"]
  },
  "error_contract": {
    "RATE_LIMITED": "Tavily API rate limit exceeded — retry after 60s",
    "NO_RESULTS": "No relevant results found for this query",
    "SEARCH_TIMEOUT": "Search exceeded 10s timeout"
  }
}
```

#### Tool 2: Academic Paper Search (Semantic Scholar)

```json
{
  "name": "search_academic",
  "description": "Search Semantic Scholar for academic papers. Returns structured paper metadata including title, authors, abstract, citation count, venue, year, and DOI.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": { "type": "string" },
      "year_range": {
        "type": "object",
        "properties": {
          "min": { "type": "integer" },
          "max": { "type": "integer" }
        }
      },
      "fields_of_study": {
        "type": "array",
        "items": { "type": "string", "enum": ["Computer Science", "Biology", "Medicine", "Physics", "Chemistry", "Engineering", "Economics"] }
      },
      "min_citation_count": { "type": "integer", "default": 0 },
      "open_access_only": { "type": "boolean", "default": false },
      "limit": { "type": "integer", "default": 10, "maximum": 50 }
    },
    "required": ["query"]
  },
  "returns": {
    "papers": [{
      "paper_id": "string (Semantic Scholar ID)",
      "title": "string",
      "authors": ["string"],
      "abstract": "string",
      "year": "integer",
      "citation_count": "integer",
      "venue": "string",
      "doi": "string",
      "pdf_url": "string | null",
      "tldr": "string (AI-generated summary)"
    }]
  },
  "error_contract": {
    "API_ERROR": "Semantic Scholar API returned an error",
    "NO_RESULTS": "No papers match the search criteria"
  }
}
```

#### Tool 3: PDF Parser

```json
{
  "name": "parse_pdf",
  "description": "Download and parse an academic PDF. Extracts full text, tables, and metadata. Returns structured content with section headers preserved.",
  "parameters": {
    "type": "object",
    "properties": {
      "url": {
        "type": "string",
        "format": "uri",
        "description": "URL of the PDF to parse"
      },
      "extract_tables": {
        "type": "boolean",
        "default": true,
        "description": "Whether to extract tables as structured data"
      },
      "max_pages": {
        "type": "integer",
        "default": 30,
        "description": "Maximum pages to process"
      }
    },
    "required": ["url"]
  },
  "returns": {
    "title": "string",
    "authors": ["string"],
    "abstract": "string",
    "sections": [{
      "heading": "string",
      "content": "string",
      "page_numbers": [1, 2]
    }],
    "tables": [{
      "caption": "string",
      "data": "string (markdown table)"
    }],
    "references": [{
      "title": "string",
      "authors": "string",
      "year": "integer"
    }],
    "total_tokens": "integer"
  },
  "error_contract": {
    "PDF_NOT_ACCESSIBLE": "Cannot download PDF (403/404 or paywall)",
    "PARSE_FAILED": "PDF is scanned image or corrupted — cannot extract text",
    "TOO_LARGE": "PDF exceeds 50-page limit"
  }
}
```

#### Tool 4: Citation Formatter

```json
{
  "name": "format_citation",
  "description": "Generate properly formatted citation in specified style from paper metadata.",
  "parameters": {
    "type": "object",
    "properties": {
      "paper": {
        "type": "object",
        "properties": {
          "title": { "type": "string" },
          "authors": { "type": "array", "items": { "type": "string" } },
          "year": { "type": "integer" },
          "venue": { "type": "string" },
          "doi": { "type": "string" },
          "volume": { "type": "string" },
          "pages": { "type": "string" }
        },
        "required": ["title", "authors", "year"]
      },
      "style": {
        "type": "string",
        "enum": ["apa", "ieee", "bibtex", "chicago", "mla"],
        "default": "apa"
      }
    },
    "required": ["paper", "style"]
  },
  "error_contract": {
    "INCOMPLETE_METADATA": "Missing required fields for this citation style"
  }
}
```

### Memory Architecture

> See: [Memory Systems](../../README.md#memory-systems)

| Memory Type | Store | Scope | Purpose |
|---|---|---|---|
| **RAG Pipeline (Session Papers)** | Qdrant (per-session namespace) | Per-session | Ingested papers chunked and embedded for retrieval during synthesis |
| **Episodic Memory** | PostgreSQL | Per-user, persistent | Past research sessions — queries, sources found, reports generated |
| **Semantic Knowledge Graph** | Qdrant (shared namespace) | Global, persistent | Accumulated domain knowledge — key terms, relationships, recurring concepts |
| **Working Memory** | LangGraph state | Per-session | Current research plan, progress, intermediate findings |

**RAG Pipeline for Paper Ingestion:**

```
Paper PDF → PDF Parser → Text Chunks (512 tokens, 50-token overlap)
         → Embedding (text-embedding-3-small) → Qdrant (session namespace)

Chunk metadata:
{
  "paper_id": "sem_12345",
  "section": "Results",
  "page": 7,
  "title": "FastFold: Accelerated Protein Structure Prediction",
  "authors": ["Smith, J.", "Lee, K."],
  "year": 2025,
  "chunk_index": 14
}

Retrieval at synthesis time:
  Query: "What methods achieve >10x speedup over AlphaFold2?"
  → Returns top-5 chunks with paper attribution
  → Agent cites the specific paper and section
```

**Episodic Memory (Past Sessions):**

```json
{
  "user_id": "researcher_042",
  "session_id": "rs_20260519_001",
  "query": "Advances in protein structure prediction since AlphaFold2",
  "sources_found": 23,
  "sources_used_in_report": 15,
  "report_sections": ["Introduction", "Speed-Optimized Methods", "Accuracy Comparison", "Conclusions"],
  "key_papers": ["FastFold (2025)", "ESMFold2 (2024)", "RoseTTAFold3 (2025)"],
  "completed_at": "2026-05-19T15:30:00Z",
  "satisfaction_rating": 4
}
```

This enables:
- "Continue my research from last week on protein folding" — load previous session's sources
- "I previously found a paper about distillation-based folding" — search episodic memory
- Avoid re-searching for papers already analyzed in previous sessions

### API Design

#### Start Research Session

```
POST /api/v1/research/sessions
```

**Request:**
```json
{
  "user_id": "researcher_042",
  "research_question": "What are the latest advances in protein structure prediction since AlphaFold2? Focus on methods that improve speed over accuracy.",
  "constraints": {
    "year_range": { "min": 2023, "max": 2026 },
    "max_sources": 20,
    "domains": ["Computer Science", "Biology"],
    "output_format": "markdown",
    "citation_style": "apa"
  },
  "prior_session_id": null
}
```

**Response:**
```json
{
  "session_id": "rs_20260519_001",
  "research_plan": {
    "phases": [
      {
        "name": "Search",
        "status": "in_progress",
        "queries": [
          "protein structure prediction speed optimization 2023-2026",
          "fast protein folding deep learning",
          "AlphaFold2 alternatives faster"
        ],
        "estimated_time": "30 seconds"
      },
      {
        "name": "Analysis",
        "status": "pending",
        "estimated_papers": 15,
        "estimated_time": "3 minutes"
      },
      {
        "name": "Synthesis",
        "status": "pending",
        "estimated_time": "2 minutes"
      }
    ]
  },
  "estimated_cost": "$0.85",
  "estimated_duration": "5 minutes"
}
```

#### Get Research Report

```
GET /api/v1/research/sessions/{session_id}/report
```

**Response:**
```json
{
  "session_id": "rs_20260519_001",
  "status": "completed",
  "report": {
    "title": "Advances in Fast Protein Structure Prediction (2023-2026)",
    "executive_summary": "Since AlphaFold2, three major approaches have emerged for faster protein structure prediction: knowledge distillation (FastFold, 100x speedup, 2% accuracy loss), fragment-based methods (FragFold, 50x speedup, minimal accuracy loss), and efficient transformer architectures (ESMFold2, 30x speedup, comparable accuracy)...",
    "sections": [
      {
        "heading": "1. Introduction",
        "content": "AlphaFold2 revolutionized protein structure prediction in 2020, achieving near-experimental accuracy [1]. However, its computational cost (~10 GPU-hours per protein) limits large-scale applications..."
      },
      {
        "heading": "2. Knowledge Distillation Approaches",
        "content": "FastFold [2] applies knowledge distillation to compress AlphaFold2's evoformer module..."
      }
    ],
    "citations": [
      { "id": 1, "text": "Jumper, J., et al. (2021). Highly accurate protein structure prediction with AlphaFold. Nature, 596, 583-589." },
      { "id": 2, "text": "Smith, J., & Lee, K. (2025). FastFold: Accelerated Protein Structure Prediction via Knowledge Distillation. NeurIPS 2025." }
    ],
    "bibtex": "@article{jumper2021highly,\n  title={Highly accurate protein structure prediction with AlphaFold},\n  author={Jumper, John and ...},\n  ...\n}"
  },
  "sources_analyzed": 18,
  "sources_cited": 12,
  "metadata": {
    "total_tokens": 198000,
    "cost_usd": 0.87,
    "duration_seconds": 280,
    "models_used": { "sonnet": 14, "haiku": 6 }
  }
}
```

---

## Step 4: Scale the design

### Bottleneck Analysis

| Bottleneck | Symptom | At Scale (1K researchers) | Mitigation |
|---|---|---|---|
| PDF download and parsing | Slow paper ingestion | 7,500 PDFs/day | Pre-fetch popular papers. Cache parsed PDFs (keyed by DOI). Async parsing pipeline with worker pool. |
| Token usage from paper ingestion | High cost per session | 120K tokens/session on papers alone | Chunked ingestion: only embed and index, don't load entire papers into context. Use RAG retrieval to pull only relevant chunks. |
| Semantic Scholar API rate limits | 429 errors | 100 req/s shared limit | Request pooling, caching popular queries (1-hour TTL), batch API where available |
| Report generation latency | Users wait 5+ minutes | Long sessions tie up LLM | Stream report generation section-by-section. Incremental updates via WebSocket. |
| Vector DB write throughput | Slow indexing during session | 15 papers x 50 chunks = 750 vectors/session | Batch vector upserts. Use Qdrant's async API. Pre-compute embeddings in parallel. |

### Cost Estimation

| Component | Per Session | Daily (500) | Monthly | Notes |
|---|---|---|---|---|
| Claude Sonnet (reasoning + synthesis) | $0.65 | $325 | $9,750 | 14 LLM calls/session |
| Claude Haiku (extraction + formatting) | $0.05 | $25 | $750 | 6 LLM calls/session |
| Tavily Search | $0.05 | $25 | $750 | 5 searches/session |
| Semantic Scholar API | Free | $0 | $0 | Rate-limited but free |
| PDF Parsing (PyMuPDF + processing) | $0.08 | $40 | $1,200 | ~15 PDFs/session, compute cost |
| Embeddings (text-embedding-3-small) | $0.02 | $10 | $300 | 750 chunks x $0.02/1K tokens |
| Qdrant (vector storage) | — | — | $500 | Self-hosted, 4 CPU / 16GB |
| PostgreSQL (episodic memory) | — | — | $200 | Managed instance |
| S3 (PDF storage) | — | — | $5 | ~200GB cached PDFs |
| Langfuse (observability) | $0.02 | $10 | $300 | |
| **Total** | **$0.87** | **$435** | **$13,755** | |

**Cost optimization levers:**
1. **PDF caching**: Cache parsed papers by DOI. Popular papers (cited >100 times) will be requested repeatedly. Expected cache hit rate: 20-30%. Savings: ~$300/month
2. **Prompt caching**: Cache system prompt + tool definitions. Savings: ~$1,500/month
3. **Incremental RAG**: Instead of re-indexing all papers each session, maintain a persistent index per user and add new papers incrementally. Savings: ~$200/month on embedding costs
4. **Haiku for extraction**: Use Haiku for structured extraction (findings, methods, numbers) — it's 10x cheaper than Sonnet for this task. Already factored above.

### Failure Modes

> See: [Failure Modes & Mitigation](../../README.md#failure-modes--mitigation)

| Failure Mode | Impact | Likelihood | Mitigation |
|---|---|---|---|
| **Citation hallucination** | Report cites a paper that doesn't exist | HIGH (most critical) | Every citation must be traced to a specific paper in the session's source list. Verification step: after report generation, validate every [N] reference maps to a real paper_id. If a citation can't be verified, flag it as "[unverified]" and notify the user. |
| **Outdated sources** | Report presents old findings as current | Medium | Apply recency weighting in source ranking. Flag papers >3 years old with "[older study]". Always include publication year in citations. |
| **Circular references** | Two papers cite each other, creating an echo chamber of findings | Medium | Build citation graph during analysis. Detect citation cycles. When two papers mutually cite each other, note "These findings are from the same research group / cite each other" in the report. |
| **Paywalled papers** | Can't access full text of relevant papers | High | Graceful degradation: use abstract + metadata when full text unavailable. Clearly mark "Summary based on abstract only" in the report. Suggest user access through institutional login. |
| **Contradictory findings** | Different papers report opposing conclusions | Medium | Explicitly surface contradictions: "Smith (2025) reports X, while Lee (2024) found Y. This discrepancy may be due to [methodology differences]." This is a feature, not a bug. |
| **Search query drift** | Iterative refinement leads searches far from original topic | Low | Anchor all searches to the original research question. At each re-planning step, check relevance to the root query. Max 3 search refinement iterations. |
| **PDF extraction errors** | Tables, equations, or figures not properly extracted | High | Use multiple extraction methods (PyMuPDF + marker-pdf). Flag sections with low-confidence extraction. For critical tables, show the user a screenshot of the original. |

### Observability Setup

> See: [Observability & Monitoring](../../README.md#observability--monitoring)

**Key Metrics:**

| Metric | Target | Alert Threshold |
|---|---|---|
| Citation accuracy (verified vs total) | > 98% | < 95% |
| Sources found per session | 10-30 | < 5 (search quality issue) |
| Report generation time | < 5 min | > 10 min |
| PDF parse success rate | > 90% | < 80% |
| User satisfaction (per-report) | > 4.0/5 | < 3.5/5 |
| Cost per session | < $1.00 | > $2.00 |
| Semantic Scholar API availability | > 99% | < 95% (activate fallback) |

**Tracing (Langfuse):**
```
Trace: rs_20260519_001 (Protein structure prediction)
  ├── Supervisor: Plan Generation (Sonnet, 2.1s, 3000 in / 500 out)
  ├── Phase 1: Search (parallel fan-out)
  │   ├── Search Agent 1: Semantic Scholar (1.8s, 12 papers)
  │   ├── Search Agent 2: ArXiv (2.1s, 8 papers)
  │   └── Search Agent 3: Tavily Web (1.5s, 6 articles)
  │   └── Merge + Deduplicate: 18 unique sources
  ├── Phase 2: Analysis (parallel)
  │   ├── Paper 1: parse_pdf (3.2s) → extract_findings (Haiku, 1.1s)
  │   ├── Paper 2: parse_pdf (2.8s) → extract_findings (Haiku, 0.9s)
  │   └── ... (18 papers total, 45s)
  ├── Phase 3: Synthesis
  │   ├── RAG retrieval: 25 chunks retrieved (0.3s)
  │   ├── Report generation (Sonnet, 8.2s, 45000 in / 8000 out)
  │   ├── Citation verification (Haiku, 1.5s, 12/12 verified)
  │   └── Format + BibTeX generation (0.5s)
  └── Total: 280s | Cost: $0.87 | Sources: 18 | Citations: 12
```

---

## Additional Talking Points

### RAG Pipeline Architecture
> See: [RAG](../../README.md#rag-retrieval-augmented-generation)
- **Chunking strategy**: 512 tokens with 50-token overlap. Academic papers have clear section boundaries — chunk at section breaks when possible.
- **Embedding model**: `text-embedding-3-small` ($0.02/M tokens) — good balance of quality and cost for academic text
- **Retrieval**: Hybrid search — combine vector similarity (semantic) with BM25 (keyword) for better recall on technical terms
- **Reranking**: Use Cohere Rerank or cross-encoder to re-score top-20 retrieved chunks before passing to LLM

### Citation Verification Pipeline
- **Step 1**: After report generation, extract all citation markers [1], [2], etc.
- **Step 2**: For each citation, verify: (a) paper exists in session's source list, (b) the claim attributed to the paper actually appears in the paper's content
- **Step 3**: Use RAG retrieval: embed the claim, search the paper's chunks, verify the claim is supported
- **Step 4**: Mark unverifiable citations and surface to user

### Multi-Source Synthesis Strategy
> See: [Context Engineering](../../README.md#context-engineering)
- **Problem**: 15 papers x 8K tokens = 120K tokens. Even with 200K context, this leaves little room for reasoning.
- **Solution**: Don't put all papers in context simultaneously. Use a map-reduce approach:
  - **Map**: For each paper, extract structured findings (Haiku, independent, parallel)
  - **Reduce**: Combine extracted findings (much smaller) into a synthesis prompt (Sonnet)
- This reduces the synthesis context from 120K tokens to ~15K tokens (structured findings only)
