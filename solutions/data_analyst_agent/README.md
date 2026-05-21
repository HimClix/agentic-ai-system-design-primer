# Design an AI Data Analyst Agent

> Note: This document links directly to relevant areas of the
> [Agentic AI System Design Primer](../../README.md)

## Step 1: Outline use cases and constraints

### Use cases

- **Natural language to SQL** -- analyst says "show me monthly revenue by product line for Q1 2026" -> agent generates SQL, executes against the data warehouse, returns results
- **Data visualization** -- "create a bar chart comparing these numbers" -> agent generates a chart using matplotlib/plotly and returns the image
- **Trend analysis** -- "what's the trend in customer churn over the last 6 months?" -> agent queries historical data, applies statistical analysis, identifies patterns
- **Report generation** -- "generate the weekly KPI report and email it to the leadership team" -> agent queries multiple tables, generates visualizations, formats into PDF, sends email
- **Scheduled reports** -- "every Monday at 9am, run this analysis and post results to #analytics Slack channel" -> agent creates a recurring job
- **Exploratory analysis** -- "what's driving the revenue drop in APAC?" -> agent autonomously explores multiple dimensions (product, channel, customer segment), identifies the most significant factors
- **Schema discovery** -- "what tables do we have about customer orders?" -> agent searches schema metadata, explains table relationships

### Constraints and assumptions

#### State assumptions

- **Users:** 200 data analysts across an organization
- **Sessions/day:** 1,000 analysis sessions (5 per analyst)
- **Latency:** Simple queries: < 10 seconds. Complex multi-step analysis: < 2 minutes. Report generation: < 5 minutes.
- **Accuracy:** SQL must be correct 95%+ of the time. Wrong aggregation logic is worse than no answer.
- **Data warehouse:** Snowflake/BigQuery with ~500 tables, 50TB+ of data
- **Safety:** **READ-ONLY access only**. No writes to any database, ever. SQL injection prevention is mandatory.
- **Compliance:** Query results may contain PII -- output filtering required. Query cost estimation before execution.

#### Calculate usage

```
Sessions/day: 1,000
Avg queries per session: 4 (NL -> SQL -> refine -> visualize)
Total queries/day: 4,000

Agent calls per query (Plan-and-Execute):
  Plan step:    1 call (Sonnet -- generate plan + SQL)
  Execute step: 1 call (Haiku -- format results)
  Visualize:    1 call (Sonnet -- generate chart code)
  Refine:       0.5 calls avg (user asks for adjustment 50% of the time)
  Total:        ~3.5 LLM calls per query

Daily LLM calls: 4,000 * 3.5 = 14,000

Token usage per call:
  Plan step (Sonnet):
    System prompt:        1,500 tokens
    Schema context (RAG): 3,000 tokens (relevant table schemas)
    User query:             200 tokens
    SQL generation:         500 tokens output
    Total: 4,700 input + 500 output

  Execute/Format (Haiku):
    Query results:        2,000 tokens (truncated result set)
    Formatting prompt:      500 tokens
    Formatted output:       800 tokens output
    Total: 2,500 input + 800 output

  Visualize (Sonnet):
    Data summary:         1,500 tokens
    Chart code prompt:      500 tokens
    Python code output:   1,000 tokens output
    Total: 2,000 input + 1,000 output

Cost calculation:
  Plan (Sonnet):     4,000/day * (4,700*$3/M + 500*$15/M) = $56.40 + $30.00 = $86.40/day
  Format (Haiku):    4,000/day * (2,500*$0.25/M + 800*$1.25/M) = $2.50 + $4.00 = $6.50/day
  Visualize (Sonnet):2,000/day * (2,000*$3/M + 1,000*$15/M) = $12.00 + $30.00 = $42.00/day
  Refine (Sonnet):   2,000/day * (5,000*$3/M + 500*$15/M) = $30.00 + $15.00 = $45.00/day

  Total LLM cost: ~$180/day = ~$5,400/month

Snowflake query costs:
  4,000 queries/day * avg 2 credits/query * $3/credit = $24K/month (this dominates!)
  (This is why query cost estimation before execution is critical)
```

### Out of scope

- Building the data warehouse (assume Snowflake/BigQuery exists)
- ETL/ELT pipeline management
- Data governance and catalog (integrate with existing tools like Alation/DataHub)
- Write operations (INSERT, UPDATE, DELETE, DDL)
- Real-time streaming analytics
- Building the chart rendering engine (use matplotlib/plotly)

---

## Step 2: Create a high-level design

```
                          ┌─────────────────────┐
                          │  Analyst (Web UI)    │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │    API Gateway       │
                          │  Auth · Rate Limit   │
                          │  Session Mgmt        │
                          └──────────┬──────────┘
                                     │
              ┌──────────────────────▼──────────────────────┐
              │           ANALYST AGENT (Plan-and-Execute)   │
              │                                              │
              │  ┌──────────────┐  ┌────────────────────┐  │
              │  │   Planner    │  │    Executor         │  │
              │  │ (Sonnet)     │  │  (Haiku for simple, │  │
              │  │              │  │   Sonnet for chart)  │  │
              │  │ - Understand │  │  - Run SQL           │  │
              │  │   intent     │  │  - Format results    │  │
              │  │ - Generate   │  │  - Generate charts   │  │
              │  │   plan + SQL │  │  - Compile report    │  │
              │  └──────┬───────┘  └────────┬───────────┘  │
              │         │                   │              │
              └─────────┼───────────────────┼──────────────┘
                        │                   │
         ┌──────────────▼───────────────────▼──────────────┐
         │                  TOOL LAYER                      │
         │  ┌──────────────┐  ┌──────────┐  ┌───────────┐│
         │  │ SQL Executor  │  │ Chart    │  │ Report    ││
         │  │ (READ ONLY)   │  │ Generator│  │ Formatter ││
         │  │               │  │ (plotly) │  │ (PDF/HTML)││
         │  │ - Query cost  │  └──────────┘  └───────────┘│
         │  │   estimator   │                              │
         │  │ - Timeout 30s │  ┌──────────┐  ┌──────────┐│
         │  │ - Max rows 10K│  │ Data     │  │ Schema   ││
         │  │ - SQL sanitize│  │ Export   │  │ Search   ││
         │  └──────────────┘  │ (CSV/XLS)│  │ (RAG)    ││
         │                     └──────────┘  └──────────┘│
         └─────────────────────────┬──────────────────────┘
                                   │
         ┌─────────────────────────▼──────────────────────┐
         │          SCHEMA-AWARE RAG LAYER                  │
         │                                                  │
         │  ┌──────────┐  ┌──────────────────────────────┐│
         │  │ Schema   │  │ Table Descriptions            ││
         │  │ Embeddings│  │ - Column names + types        ││
         │  │ (pgvector)│  │ - Business descriptions       ││
         │  │           │  │ - Common joins                ││
         │  │           │  │ - Sample queries              ││
         │  └──────────┘  │ - Known gotchas               ││
         │                 │   ("revenue table uses cents") ││
         │                 └──────────────────────────────┘│
         └─────────────────────────┬──────────────────────┘
                                   │
         ┌─────────────────────────▼──────────────────────┐
         │              MEMORY + STATE                      │
         │  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
         │  │ Session  │  │ Episodic │  │ Semantic     │ │
         │  │ (Redis)  │  │ (past    │  │ Schema Cache │ │
         │  │ Current  │  │ queries  │  │ (pgvector)   │ │
         │  │ analysis │  │ + results│  │              │ │
         │  │ context  │  │ per user)│  │              │ │
         │  └──────────┘  └──────────┘  └──────────────┘ │
         └─────────────────────────┬──────────────────────┘
                                   │
         ┌─────────────────────────▼──────────────────────┐
         │            SAFETY LAYER                          │
         │  ┌──────────────┐  ┌────────────────────────┐  │
         │  │ SQL Sanitizer │  │ Query Cost Estimator   │  │
         │  │ - No DML      │  │ - EXPLAIN before exec  │  │
         │  │ - No DDL      │  │ - Budget: 10 credits   │  │
         │  │ - Param bind  │  │ - User confirmation    │  │
         │  │ - Allowlist   │  │   for expensive queries│  │
         │  └──────────────┘  └────────────────────────┘  │
         │  ┌──────────────────────────────────────────┐  │
         │  │ PII Filter (output)                       │  │
         │  │ - Detect SSN, email, phone in results     │  │
         │  │ - Mask or redact based on user role        │  │
         │  └──────────────────────────────────────────┘  │
         └─────────────────────────────────────────────────┘
```

---

## Step 3: Design core components

### Agent Architecture Decision

**Single agent** with Plan-and-Execute pattern.

Cross-reference: [Single Agent vs Multi-Agent](../../README.md#single-agent-vs-multi-agent)

**Why single agent:** Data analysis is a linear process: understand query -> generate SQL -> execute -> visualize -> format. All steps share the same context (the user's question and the schema). A multi-agent system would fragment this context and add coordination overhead for no benefit.

### Pattern Selection

**Plan-and-Execute** -- Cross-reference: [Plan-and-Execute](../../README.md#plan-and-execute)

**Why Plan-and-Execute over ReAct:**

1. **Cost efficiency** -- analysis tasks follow predictable patterns. Generate one plan with Sonnet, execute each step with Haiku. This saves ~60% vs ReAct where every step uses full reasoning.
2. **Predictability** -- the plan is visible to the user before execution. They can approve or modify it.
3. **SQL quality** -- the planner generates SQL with full schema context in one focused call, rather than incrementally building SQL across multiple ReAct steps (which tends to produce worse SQL).

**Plan-and-Execute flow:**
```
User: "Compare monthly revenue by product line for Q1 vs Q2 2026"

PLANNER (Sonnet, one call):
  Plan:
    1. Query monthly revenue by product line for Q1 (Jan-Mar 2026)
    2. Query monthly revenue by product line for Q2 (Apr-Jun 2026)
    3. Combine results and calculate growth percentages
    4. Generate grouped bar chart comparing Q1 vs Q2
    5. Format summary with key insights

  SQL for step 1:
    SELECT product_line, DATE_TRUNC('month', order_date) as month,
           SUM(revenue_cents) / 100.0 as revenue
    FROM orders
    WHERE order_date BETWEEN '2026-01-01' AND '2026-03-31'
    GROUP BY product_line, month
    ORDER BY product_line, month

  SQL for step 2:
    [similar query for Q2]

EXECUTOR (Haiku, per step):
  Step 1: Execute SQL -> 15 rows returned
  Step 2: Execute SQL -> 15 rows returned
  Step 3: Calculate growth -> merged table
  Step 4: Generate chart code -> plotly bar chart
  Step 5: Format -> "Enterprise SaaS grew 23% Q1->Q2, while SMB declined 8%..."
```

**Re-planner:** If a SQL query fails (syntax error, timeout), the re-planner (Sonnet) generates corrected SQL based on the error message. Max 2 re-plan attempts.

### Tool Design

```json
{
  "tools": [
    {
      "name": "execute_sql",
      "description": "Execute a READ-ONLY SQL query against the data warehouse. Returns max 10,000 rows. Query is validated for safety (no DML/DDL) and cost-estimated before execution. Queries exceeding 10 credit budget require user confirmation.",
      "input_schema": {
        "type": "object",
        "properties": {
          "sql": {"type": "string", "description": "SELECT query only. No INSERT, UPDATE, DELETE, DROP, ALTER, CREATE."},
          "database": {"type": "string", "enum": ["analytics_warehouse", "marketing_db", "finance_db"]},
          "max_rows": {"type": "integer", "default": 1000, "maximum": 10000},
          "timeout_seconds": {"type": "integer", "default": 30, "maximum": 120}
        },
        "required": ["sql", "database"]
      }
    },
    {
      "name": "generate_chart",
      "description": "Generate a data visualization chart. Supports bar, line, scatter, pie, histogram, heatmap. Returns a PNG image URL.",
      "input_schema": {
        "type": "object",
        "properties": {
          "chart_type": {"type": "string", "enum": ["bar", "line", "scatter", "pie", "histogram", "heatmap", "grouped_bar", "stacked_bar"]},
          "data": {
            "type": "object",
            "description": "Data for the chart. Format: {x: [...], y: [...], labels: [...], series: [...]}"
          },
          "title": {"type": "string"},
          "x_label": {"type": "string"},
          "y_label": {"type": "string"},
          "style": {"type": "string", "enum": ["professional", "minimal", "colorful"], "default": "professional"}
        },
        "required": ["chart_type", "data", "title"]
      }
    },
    {
      "name": "search_schema",
      "description": "Search the data warehouse schema for relevant tables, columns, and relationships. Uses semantic search over table/column descriptions.",
      "input_schema": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "Natural language description of what data you're looking for. e.g., 'customer order history' or 'monthly recurring revenue'"},
          "max_results": {"type": "integer", "default": 5}
        },
        "required": ["query"]
      }
    },
    {
      "name": "format_report",
      "description": "Format analysis results into a structured report (PDF or HTML). Includes charts, tables, and narrative text.",
      "input_schema": {
        "type": "object",
        "properties": {
          "title": {"type": "string"},
          "sections": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "heading": {"type": "string"},
                "content_type": {"type": "string", "enum": ["text", "table", "chart"]},
                "content": {"type": "string", "description": "Text content, or reference to chart/table data"}
              }
            }
          },
          "format": {"type": "string", "enum": ["pdf", "html", "markdown"], "default": "html"}
        },
        "required": ["title", "sections"]
      }
    },
    {
      "name": "export_data",
      "description": "Export query results to a downloadable file format.",
      "input_schema": {
        "type": "object",
        "properties": {
          "data_reference": {"type": "string", "description": "Reference to the query result set"},
          "format": {"type": "string", "enum": ["csv", "xlsx", "json"]},
          "filename": {"type": "string"}
        },
        "required": ["data_reference", "format"]
      }
    },
    {
      "name": "estimate_query_cost",
      "description": "Estimate the cost of a SQL query before executing it. Uses EXPLAIN to estimate rows scanned and compute credits.",
      "input_schema": {
        "type": "object",
        "properties": {
          "sql": {"type": "string"},
          "database": {"type": "string"}
        },
        "required": ["sql", "database"]
      }
    }
  ]
}
```

**SQL Safety Pipeline (critical design):**

```python
def safe_execute_sql(sql: str, database: str, user_id: str) -> dict:
    # 1. PARSE: Use sqlparse to parse the SQL statement
    parsed = sqlparse.parse(sql)
    
    # 2. VALIDATE: Check for forbidden operations
    stmt_type = parsed[0].get_type()
    if stmt_type != 'SELECT':
        return {"error": "Only SELECT statements are allowed", "received": stmt_type}
    
    # 3. SCAN for dangerous patterns
    forbidden = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE',
                 'TRUNCATE', 'EXEC', 'EXECUTE', 'xp_', 'sp_', ';']
    sql_upper = sql.upper()
    for pattern in forbidden:
        if pattern in sql_upper and pattern != ';':  # semicolons checked separately
            return {"error": f"Forbidden SQL keyword: {pattern}"}
    
    # 4. CHECK for SQL injection via subqueries that modify data
    if re.search(r'\b(INTO|VALUES)\b', sql_upper):
        return {"error": "SELECT INTO is not allowed"}
    
    # 5. ESTIMATE COST before execution
    cost = estimate_query_cost(sql, database)
    if cost > 10:  # 10 credit budget
        return {
            "error": "query_too_expensive",
            "estimated_credits": cost,
            "budget": 10,
            "suggestion": "Add more filters or reduce date range to lower cost",
            "requires_user_confirmation": True
        }
    
    # 6. EXECUTE with timeout
    # Use the READ-ONLY database role (no write permissions at DB level)
    result = execute_with_timeout(sql, database, timeout=30, role="readonly_analyst")
    
    # 7. PII FILTER on output
    result = filter_pii(result, user_role=get_user_role(user_id))
    
    return result
```

### Schema-Aware RAG

Cross-reference: [RAG](../../README.md#rag-retrieval-augmented-generation)

The schema RAG layer is what makes the difference between a toy text-to-SQL demo and a production system.

**What gets indexed:**

```json
{
  "table_name": "orders",
  "schema": "analytics",
  "description": "All customer orders. One row per order. Includes line-item details as nested JSON in the 'items' column.",
  "columns": [
    {"name": "order_id", "type": "VARCHAR", "description": "Unique order identifier"},
    {"name": "customer_id", "type": "VARCHAR", "description": "FK to customers.id"},
    {"name": "order_date", "type": "TIMESTAMP", "description": "When the order was placed (UTC)"},
    {"name": "revenue_cents", "type": "BIGINT", "description": "IMPORTANT: Revenue stored in cents. Divide by 100 for dollars."},
    {"name": "product_line", "type": "VARCHAR", "description": "Product category: 'Enterprise', 'SMB', 'Startup', 'Free'"}
  ],
  "common_joins": ["JOIN customers ON orders.customer_id = customers.id"],
  "sample_queries": [
    "SELECT product_line, SUM(revenue_cents)/100.0 as revenue FROM orders GROUP BY product_line",
    "SELECT DATE_TRUNC('month', order_date) as month, COUNT(*) as order_count FROM orders GROUP BY month"
  ],
  "gotchas": [
    "revenue_cents is in CENTS, not dollars. Always divide by 100.",
    "order_date is in UTC. Convert to user's timezone for display.",
    "Deleted orders have status='cancelled' but are NOT removed from the table."
  ]
}
```

**RAG retrieval flow:**
1. User asks: "What's the average order value by customer segment?"
2. Embed the question and search schema descriptions
3. Return top 5 most relevant tables with column details and gotchas
4. Inject into the planner's context alongside the user's question
5. The planner generates SQL with full awareness of the schema quirks

### Memory Architecture

| Memory Type | Store | What It Holds | TTL |
|---|---|---|---|
| **Session** | Redis | Current analysis context: query history, result references, chart references | 2 hours idle timeout |
| **Episodic** (per-user) | PostgreSQL | Past queries with their SQL + results summaries. "Last time you asked about churn, this is the query that worked." | 90 days |
| **Semantic** (schema) | PostgreSQL + pgvector | Table descriptions, column meanings, common joins, gotchas. Updated by data engineering team. | Refreshed weekly from data catalog |

**Episodic memory value:** When a user asks "run the same churn analysis as last week," the agent retrieves the exact SQL from episodic memory instead of regenerating it. This is both faster and more accurate.

### API Design

```
POST /api/v1/analyst/query
Content-Type: application/json
Authorization: Bearer <jwt_token>

Request:
{
  "session_id": "sess_abc123",
  "query": "Compare monthly revenue by product line for Q1 vs Q2 2026",
  "database": "analytics_warehouse",
  "stream": true,
  "auto_visualize": true,
  "max_query_credits": 10
}

Response (SSE stream):
data: {"type": "plan", "steps": ["Query Q1 revenue", "Query Q2 revenue", "Calculate growth", "Generate chart", "Summarize insights"]}
data: {"type": "sql", "step": 1, "sql": "SELECT product_line, ...", "estimated_credits": 2.3}
data: {"type": "result", "step": 1, "rows": 15, "preview": [{"product_line": "Enterprise", "month": "2026-01", "revenue": 1250000}]}
data: {"type": "sql", "step": 2, "sql": "SELECT product_line, ...", "estimated_credits": 2.1}
data: {"type": "result", "step": 2, "rows": 15}
data: {"type": "chart", "url": "s3://charts/sess_abc123/comparison.png", "chart_type": "grouped_bar"}
data: {"type": "insight", "content": "Enterprise SaaS revenue grew 23% from Q1 to Q2 2026, driven primarily by expansion in existing accounts. SMB segment declined 8%, correlating with the pricing change in March."}
data: {"type": "done", "usage": {"queries_run": 2, "credits_used": 4.4, "tokens": 12000, "cost_usd": 0.058, "latency_ms": 8200}}
```

---

## Step 4: Scale the design

### Bottleneck Analysis

| Bottleneck | At 200 Users | At 2,000 Users | Mitigation |
|---|---|---|---|
| **Data warehouse query concurrency** | 4K queries/day, manageable | 40K queries/day, warehouse performance degrades | Query result caching (same SQL + same date range = cached result, 15 min TTL). Batch similar queries. |
| **Schema RAG retrieval** | 4K retrievals/day | 40K/day, embedding search latency | Pre-compute and cache schema embeddings. Refresh weekly. In-memory vector index for < 500 tables. |
| **Chart generation** | 2K charts/day | 20K/day, matplotlib is single-threaded | Dedicated chart generation workers (horizontally scaled). Use plotly for server-side rendering. |
| **LLM SQL generation quality** | Good for common patterns | Tail queries on rare tables fail | Expand schema RAG with more sample queries. A/B test SQL generation prompts. |

### Cost Estimation

| Component | Monthly Cost | Notes |
|---|---|---|
| **LLM - Sonnet (planning + SQL)** | $5,184 | Planner + visualize + refine calls |
| **LLM - Haiku (formatting)** | $195 | Cheap result formatting |
| **Snowflake compute** | $24,000 | **Dominant cost.** 4K queries/day at avg 2 credits. |
| **PostgreSQL (schema + episodic)** | $600 | db.r6g.large with pgvector |
| **Redis (sessions)** | $300 | r6g.medium |
| **S3 (charts + reports)** | $100 | ~10GB/month |
| **Infrastructure** | $1,500 | 3 app nodes |
| **Langfuse** | $59 | Observability |
| **Total** | **~$31,938/month** | **$1.06 per session** |

**Critical insight:** Snowflake compute is 75% of the total cost. LLM cost is secondary. The most impactful cost optimization is **query caching** and **query cost estimation**, not model routing.

**Query caching impact:**
```
Same-day cache hit rate for common queries: ~25%
  4,000 queries/day * 25% = 1,000 cached
  Savings: 1,000 * 2 credits * $3/credit * 30 days = $6,000/month (25% of Snowflake cost)
```

### Failure Modes

Cross-reference: [Failure Modes & Mitigation](../../README.md#failure-modes--mitigation)

| Failure Mode | Severity | Detection | Mitigation |
|---|---|---|---|
| **Slow query (no timeout)** | High | Query runs > 30 seconds | Hard timeout at 30s. EXPLAIN before execution to estimate runtime. Suggest adding filters. |
| **SQL injection via natural language** | **Critical** | User crafts input to generate DROP/DELETE | Multi-layer defense: 1) sqlparse validation, 2) regex scan, 3) READ-ONLY database role, 4) query allowlist at DB level. |
| **Wrong aggregation logic** | High | SUM where AVG was meant, missing GROUP BY, double-counting from JOINs | Reflexion: after generating SQL, run a validation step. "Does this SQL correctly answer the question? Check aggregation logic." |
| **Chart misrepresentation** | Medium | Y-axis doesn't start at zero, misleading scale, wrong chart type | Chart validation rules: bar charts start at zero, line charts have proper time axis, pie charts don't exceed 100%. |
| **PII in results** | High | Query returns email addresses, phone numbers, SSNs | PII detection on all result sets. Mask based on user role. Log all PII access attempts. |
| **Query cost explosion** | High | User asks "show me all transactions" (full table scan of 1B rows) | EXPLAIN-based cost estimation before execution. Hard budget of 10 credits per query. User confirmation for expensive queries. |
| **Stale schema metadata** | Medium | New table added to warehouse but not indexed in RAG | Weekly schema sync job. Alert if schema drift detected (new tables not indexed). |

### Observability Setup

**Tracing (Langfuse):**
- Per-session trace with all queries, plans, SQL generated, results (truncated), charts, and final output
- SQL quality scoring: after each query, log whether it needed re-planning (indicates initial SQL was wrong)

**Metrics:**

| Metric | Alert Threshold |
|---|---|
| `analyst_sql_first_try_success_rate` | < 90% (SQL quality degrading) |
| `analyst_query_latency_p95` | > 30s (warehouse performance issue) |
| `analyst_query_credits_p95` | > 8 credits (approaching budget limit) |
| `analyst_replan_rate` | > 20% (planner quality issue) |
| `analyst_session_cost_p95` | > $5 (cost anomaly) |
| `analyst_pii_detection_count` | > 0 (immediate security review) |
| `analyst_cache_hit_rate` | < 15% (caching not effective) |
| `analyst_schema_rag_miss_rate` | > 30% (schema metadata gaps) |

---

## Additional Talking Points

### Why Schema-Aware RAG Is the Secret Weapon

The difference between a demo text-to-SQL system and a production one is **schema awareness**. Without it:
- The agent doesn't know that `revenue_cents` is in cents, not dollars
- It joins the wrong tables because it doesn't know the correct foreign keys
- It doesn't know about soft deletes (`status != 'cancelled'`)
- It generates invalid SQL because it uses column names that don't exist

With schema RAG, the agent gets the most relevant table descriptions, column meanings, known gotchas, and sample queries injected into its context. This single feature improves SQL accuracy from ~70% to ~95%.

### Semantic Caching for SQL Results

```python
# Instead of exact-match caching, use semantic similarity
user_query_1 = "monthly revenue by product for Q1 2026"
user_query_2 = "Q1 2026 revenue broken down by product line"

# These are semantically identical -> same SQL -> serve cached result
# Similarity threshold: 0.92 (high, to avoid serving wrong cached result)
```

This is more aggressive than keyword-based caching. With a 0.92 similarity threshold, we catch ~25% of queries as cache hits while maintaining accuracy.

### Multi-Step Exploratory Analysis

The hardest use case: "What's driving the revenue drop in APAC?"

This requires the agent to autonomously explore multiple dimensions:
```
Step 1: Confirm revenue drop (query total APAC revenue, Q1 vs Q2)
Step 2: Break down by product line (which products dropped?)
Step 3: Break down by customer segment (which segments?)
Step 4: Break down by channel (direct vs partner?)
Step 5: Check for outliers (any single large customer churned?)
Step 6: Synthesize: "APAC revenue dropped 12%, driven primarily by Enterprise churn
        in the direct channel. 3 enterprise customers worth $2.1M churned in Q2."
```

This is where Plan-and-Execute shines: the planner generates all 6 steps upfront, and the executor runs them efficiently. If step 3 reveals the answer, the re-planner can skip remaining steps.

### Interview Cross-References
- Plan-and-Execute for structured analysis: [Plan-and-Execute](../../README.md#plan-and-execute)
- Schema RAG pipeline: [RAG](../../README.md#rag-retrieval-augmented-generation)
- SQL safety as tool guardrail: [Safety & Guardrails](../../README.md#safety--guardrails)
- Query cost as dominant cost factor: [Cost Engineering](../../README.md#cost-engineering)
- Episodic memory for query reuse: [Memory Systems -- Episodic Memory](../../README.md#episodic-memory)
- SQL injection via prompt: [Security](../../README.md#security)
