# Design an AI Coding Agent (Like Cursor / Claude Code)

> Note: This document links directly to relevant areas of the
> [Agentic AI System Design Primer](../../README.md)

## Step 1: Outline use cases and constraints

### Use cases

- **Read and understand code** — navigate file trees, read files, understand project structure and dependencies
- **Write and edit code** — generate new code, modify existing files, multi-file refactors
- **Run and debug** — execute code in sandboxed environments, run test suites, interpret errors and fix them
- **Explain code** — answer questions about codebases, trace control flow, explain architecture
- **Multi-file changes** — coordinated edits across multiple files (e.g., rename a function and update all callers)
- **Git operations** — create branches, stage changes, generate commit messages, create PRs
- **Test-driven fixes** — run tests, identify failing tests, fix code until tests pass (agentic loop)
- **Context-aware completions** — use project context (file tree, recent edits, open files) to generate relevant code

### Constraints and assumptions

#### State assumptions

- **5,000 developers** using the tool daily (enterprise deployment)
- **Concurrent sessions**: ~500 at peak (10% concurrent usage)
- **Average coding session**: 45 minutes, ~20 agent interactions
- **Latency target**: code completions < 2s, agent actions < 10s, test runs < 60s
- **Accuracy bar**: 85% of generated code compiles without modification; 70% passes existing tests first-try
- **Supported languages**: Python, TypeScript, Go, Java, Rust (top 5 by usage)
- **Repository sizes**: median 50K LOC, p95 500K LOC, max 2M LOC
- **Sandbox requirement**: ALL code execution must happen in isolated environments (security-critical)
- **Security**: no exfiltration of code to external services; all processing within enterprise boundary

#### Calculate usage

```
Developers:                   5,000 DAU
Sessions/developer/day:       2 (morning + afternoon)
Total sessions/day:           10,000
Interactions/session:         20
Total agent runs/day:         200,000
Agent runs/second:            ~2.3 (peak: ~8)

Tokens per interaction (avg):
  System prompt:              2,000 tokens (tool defs, project context)
  File context:               5,000 tokens (open files, relevant code)
  Conversation context:       3,000 tokens (recent messages + plan)
  Output:                     1,500 tokens (code generation)
  Total:                      11,500 tokens/interaction

  Complex interactions (multi-file, debugging):
  Input:                      30,000 tokens (multiple files + test output)
  Output:                     5,000 tokens
  Total:                      35,000 tokens

Weighted average:             ~15,000 tokens/interaction
  (70% simple x 11,500 + 30% complex x 35,000)

Tokens/day:                   200,000 x 15,000 = 3B tokens

Model routing split:
  Opus (complex reasoning, 20%):   40,000 runs x 35,000 tokens = 1.4B tokens
  Sonnet (standard coding, 50%):  100,000 runs x 11,500 tokens = 1.15B tokens
  Haiku (simple edits, 30%):       60,000 runs x 5,000 tokens = 0.3B tokens

Cost/day:
  Opus:   1.12B in x $15/M + 0.28B out x $75/M = $16,800 + $21,000 = $37,800
  Sonnet: 0.92B in x $3/M + 0.23B out x $15/M = $2,760 + $3,450 = $6,210
  Haiku:  0.24B in x $0.80/M + 0.06B out x $4/M = $192 + $240 = $432
  Total:  ~$44,442/day -> ~$1.33M/month

Cost per session:  $44,442 / 10,000 = $4.44/session
Cost per developer/month: $1.33M / 5,000 = $266/developer/month
```

**Benchmark comparison:**
- Cursor Pro: $20/month (subsidized, rate-limited)
- GitHub Copilot Enterprise: $39/month
- Our enterprise deployment at full capability: ~$266/month (unoptimized)
- After optimization target: ~$80-120/month per developer

### Out of scope

- IDE integration / extension development (assume we have a CLI or web interface)
- Real-time pair programming (collaborative editing)
- Custom model fine-tuning per organization
- Code review automation (separate system)
- Deployment and CI/CD pipeline management

---

## Step 2: Create a high-level design

```
 Developer (CLI/IDE)
        │
        ▼
 ┌──────────────────────────────────────────────────────┐
 │                    API Gateway                        │
 │         (Auth, Rate Limit, Session Management)        │
 └──────────────────────┬───────────────────────────────┘
                        │
 ┌──────────────────────▼───────────────────────────────┐
 │              SESSION MANAGER                          │
 │   (LangGraph Checkpointing, State Persistence)       │
 └──────────────────────┬───────────────────────────────┘
                        │
 ┌──────────────────────▼───────────────────────────────┐
 │              PLANNER AGENT (Plan-and-Execute)         │
 │    ┌─────────────────────────────────────────────┐   │
 │    │  1. Analyze request                          │   │
 │    │  2. Generate multi-step plan                 │   │
 │    │  3. Delegate steps to Executor               │   │
 │    │  4. Verify results, replan if needed          │   │
 │    └─────────────────────────────────────────────┘   │
 └──────────────────────┬───────────────────────────────┘
                        │
 ┌──────────────────────▼───────────────────────────────┐
 │              EXECUTOR AGENT (ReAct per step)          │
 │    ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
 │    │  LLM     │  │  Tool    │  │  Observation     │  │
 │    │  Router  │  │  Router  │  │  Validator       │  │
 │    └────┬─────┘  └────┬─────┘  └────────┬─────────┘  │
 │         │              │                 │            │
 └─────────┼──────────────┼─────────────────┼────────────┘
           │              │                 │
     ┌─────▼─────┐  ┌────▼─────────────────▼──────────┐
     │ LLM Pool  │  │          Tool Layer              │
     │           │  │                                  │
     │ ┌───────┐ │  │  ┌──────────┐  ┌──────────────┐ │
     │ │ Opus  │ │  │  │ File     │  │ Code         │ │
     │ │(hard) │ │  │  │ Read/    │  │ Execution    │ │
     │ ├───────┤ │  │  │ Write    │  │ (Sandbox)    │ │
     │ │Sonnet │ │  │  ├──────────┤  ├──────────────┤ │
     │ │(medium│ │  │  │ Search   │  │ Test Runner  │ │
     │ ├───────┤ │  │  │ (grep/   │  │ (Sandbox)    │ │
     │ │ Haiku │ │  │  │  ripgrep)│  ├──────────────┤ │
     │ │(fast) │ │  │  ├──────────┤  │ Git Ops      │ │
     │ └───────┘ │  │  │ Tree/    │  │ (branch,     │ │
     └───────────┘  │  │ Outline  │  │  commit, PR) │ │
                    │  └──────────┘  └──────────────┘ │
                    └──────────────────┬───────────────┘
                                       │
                    ┌──────────────────▼───────────────┐
                    │         SANDBOX LAYER            │
                    │   (E2B / Docker / Firecracker)   │
                    │                                  │
                    │  ┌────────────────────────────┐  │
                    │  │  Isolated per session       │  │
                    │  │  No network access          │  │
                    │  │  CPU/Memory limits          │  │
                    │  │  Filesystem snapshot/restore│  │
                    │  │  Timeout: 60s per execution │  │
                    │  └────────────────────────────┘  │
                    └──────────────────────────────────┘
                                       │
                    ┌──────────────────▼───────────────┐
                    │       PROJECT CONTEXT STORE       │
                    │  ┌────────┐  ┌────────────────┐  │
                    │  │File    │  │ Embeddings     │  │
                    │  │Tree    │  │ (code search)  │  │
                    │  │Cache   │  │                │  │
                    │  ├────────┤  ├────────────────┤  │
                    │  │Recent  │  │ Session State  │  │
                    │  │Changes │  │ (LangGraph     │  │
                    │  │Tracker │  │  checkpoints)  │  │
                    │  └────────┘  └────────────────┘  │
                    └──────────────────────────────────┘
                                       │
                    ┌──────────────────▼───────────────┐
                    │     Observability (Langfuse)      │
                    │  Traces, Token Usage, Latency,    │
                    │  Code Quality Metrics              │
                    └──────────────────────────────────┘
```

**Component Responsibilities:**

| Component | Role | What breaks without it |
|---|---|---|
| Session Manager | Persist state across interactions within a coding session | Agent loses context between turns; can't resume |
| Planner Agent | Break complex requests into steps | Multi-file changes fail; agent tries to do everything in one shot |
| Executor Agent | Execute each plan step using ReAct | No iterative problem-solving; can't recover from errors |
| LLM Router | Route to Opus/Sonnet/Haiku based on complexity | 3x cost overrun (everything goes to Opus) or quality drops |
| Sandbox Layer | Isolated code execution | Security breach — code runs on host; malicious code escapes |
| Project Context Store | File tree, embeddings, recent changes | Agent suggests code that doesn't fit the project; wrong imports |

---

## Step 3: Design core components

### Agent Architecture Decision

**Hybrid: Plan-and-Execute + ReAct** — a single agent with two operating modes.

> See: [Plan-and-Execute](../../README.md#plan-and-execute) and [ReAct](../../README.md#react-reasoning--acting)

**Justification:**
- **Plan-and-Execute** for the outer loop: complex requests ("refactor this module to use dependency injection") need a multi-step plan before execution
- **ReAct** for each step: each plan step is executed with observe-reason-act cycles (e.g., run tests, see failure, fix, re-run)
- This mirrors how developers actually work: plan the approach, then iteratively implement and debug

**Why NOT multi-agent:**
- A single agent with mode-switching is simpler than coordinator + planner + coder agents
- All operations share the same codebase context (splitting context across agents wastes tokens)
- Claude Code and Cursor both use single-agent architectures successfully
- Exception: for very large repos (>1M LOC), a "file finder" sub-agent could help scope the context window

### Pattern Selection

**Outer loop: Plan-and-Execute**

> See: [Plan-and-Execute](../../README.md#plan-and-execute)

```
User: "Add pagination to the /users API endpoint"

Plan:
  Step 1: Read the current /users endpoint implementation (GET handler + DB query)
  Step 2: Read the existing pagination utility if one exists (search for "paginate", "offset", "cursor")
  Step 3: Modify the DB query to accept page/limit parameters
  Step 4: Update the handler to parse pagination query params
  Step 5: Update the response schema to include pagination metadata
  Step 6: Add/update tests for the paginated endpoint
  Step 7: Run tests to verify
```

**Inner loop: ReAct per step**

```
Executing Step 7: Run tests to verify

Thought: I need to run the test suite for the users endpoint
Action: run_tests(path="tests/api/test_users.py")
Observation: FAILED - test_get_users_list expected 10 items, got 5
Thought: The default page size is 5, but the test expects the old behavior. I need to update the test.
Action: edit_file(path="tests/api/test_users.py", edits=[...])
Observation: File updated
Action: run_tests(path="tests/api/test_users.py")
Observation: 12/12 tests passed
Thought: All tests pass. Step 7 complete.
```

### Tool Design

#### Tool 1: File Read

```json
{
  "name": "read_file",
  "description": "Read the contents of a file from the project. Returns the full file content with line numbers. For large files (>500 lines), use the offset and limit parameters to read specific sections.",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {
        "type": "string",
        "description": "Relative path from project root (e.g., 'src/api/users.py')"
      },
      "offset": {
        "type": "integer",
        "description": "Starting line number (0-indexed). Default: 0"
      },
      "limit": {
        "type": "integer",
        "description": "Number of lines to read. Default: entire file. Max: 500"
      }
    },
    "required": ["path"]
  },
  "error_contract": {
    "FILE_NOT_FOUND": "File does not exist at the specified path",
    "PERMISSION_DENIED": "File is outside project boundary or is a binary file",
    "FILE_TOO_LARGE": "File exceeds 10MB limit — use offset/limit to read sections"
  }
}
```

#### Tool 2: File Write/Edit

```json
{
  "name": "edit_file",
  "description": "Apply targeted edits to an existing file using search-and-replace. Each edit specifies old content to find and new content to replace it with. The old_string must match exactly (including whitespace/indentation).",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {
        "type": "string",
        "description": "Relative path from project root"
      },
      "edits": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "old_string": { "type": "string", "description": "Exact text to find" },
            "new_string": { "type": "string", "description": "Replacement text" }
          },
          "required": ["old_string", "new_string"]
        },
        "description": "List of search-and-replace edits to apply"
      }
    },
    "required": ["path", "edits"]
  },
  "error_contract": {
    "FILE_NOT_FOUND": "File does not exist at the specified path",
    "OLD_STRING_NOT_FOUND": "The old_string was not found in the file — content may have changed",
    "MULTIPLE_MATCHES": "old_string matches multiple locations — provide more context to disambiguate",
    "WRITE_PROTECTED": "File is read-only or outside project boundary"
  }
}
```

#### Tool 3: Code Execution (Sandboxed)

```json
{
  "name": "execute_command",
  "description": "Execute a shell command in the project's sandboxed environment. Commands run in an isolated container with the project's dependencies installed. Network access is disabled. Timeout: 60 seconds.",
  "parameters": {
    "type": "object",
    "properties": {
      "command": {
        "type": "string",
        "description": "Shell command to execute (e.g., 'python -m pytest tests/', 'go test ./...')"
      },
      "working_directory": {
        "type": "string",
        "description": "Working directory relative to project root. Default: project root"
      },
      "timeout_seconds": {
        "type": "integer",
        "default": 60,
        "maximum": 120,
        "description": "Maximum execution time in seconds"
      }
    },
    "required": ["command"]
  },
  "returns": {
    "stdout": "string",
    "stderr": "string",
    "exit_code": "integer",
    "execution_time_ms": "integer",
    "truncated": "boolean (true if output exceeded 10KB limit)"
  },
  "error_contract": {
    "TIMEOUT": "Command exceeded timeout limit",
    "SANDBOX_ERROR": "Sandbox container failed to start",
    "COMMAND_BLOCKED": "Command is not allowed (e.g., rm -rf /, network access)"
  },
  "safety": {
    "blocked_commands": ["rm -rf /", "curl", "wget", "ssh", "scp", "nc"],
    "network": "disabled",
    "filesystem": "copy-on-write snapshot, restored after session",
    "resources": "2 CPU cores, 4GB RAM, 10GB disk"
  }
}
```

#### Tool 4: Code Search

```json
{
  "name": "search_code",
  "description": "Search for code patterns across the project using ripgrep. Returns matching lines with file paths and line numbers. Useful for finding function definitions, usages, imports, and patterns.",
  "parameters": {
    "type": "object",
    "properties": {
      "pattern": {
        "type": "string",
        "description": "Search pattern (regex supported)"
      },
      "path_filter": {
        "type": "string",
        "description": "Glob pattern to filter files (e.g., '*.py', 'src/**/*.ts')"
      },
      "max_results": {
        "type": "integer",
        "default": 20,
        "maximum": 50
      }
    },
    "required": ["pattern"]
  },
  "error_contract": {
    "INVALID_REGEX": "Search pattern is not a valid regex",
    "NO_RESULTS": "No matches found",
    "SEARCH_TIMEOUT": "Search exceeded 10s timeout (repo too large for this pattern)"
  }
}
```

#### Tool 5: Git Operations

```json
{
  "name": "git_operation",
  "description": "Perform git operations on the project repository.",
  "parameters": {
    "type": "object",
    "properties": {
      "operation": {
        "type": "string",
        "enum": ["status", "diff", "log", "create_branch", "add", "commit", "create_pr"]
      },
      "args": {
        "type": "object",
        "description": "Operation-specific arguments",
        "properties": {
          "branch_name": { "type": "string" },
          "commit_message": { "type": "string" },
          "files": { "type": "array", "items": { "type": "string" } },
          "pr_title": { "type": "string" },
          "pr_body": { "type": "string" }
        }
      }
    },
    "required": ["operation"]
  },
  "error_contract": {
    "NOT_A_GIT_REPO": "Project directory is not a git repository",
    "MERGE_CONFLICT": "Operation would create a merge conflict",
    "BRANCH_EXISTS": "Branch name already exists"
  }
}
```

### Memory Architecture

> See: [Memory Systems](../../README.md#memory-systems)

| Memory Type | Store | Scope | Purpose |
|---|---|---|---|
| **Project Context** | File system + embeddings (Qdrant) | Per-project, persistent | File tree, code structure, dependency graph |
| **Session State** | LangGraph checkpoints (PostgreSQL) | Per-session, ephemeral | Current plan, completed steps, open files, pending edits |
| **Conversation History** | In-context (sliding window) | Per-session | Last 10 interactions to maintain conversational coherence |
| **Tool Result Cache** | Redis | Per-session, 1-hour TTL | Cache file reads and search results within a session |

**LangGraph Checkpoint (Session State):**
```json
{
  "session_id": "sess_coding_001",
  "project": "acme-api",
  "state": {
    "current_plan": {
      "description": "Add pagination to /users endpoint",
      "steps": [
        {"id": 1, "description": "Read current handler", "status": "completed"},
        {"id": 2, "description": "Find pagination utility", "status": "completed"},
        {"id": 3, "description": "Modify DB query", "status": "in_progress"},
        {"id": 4, "description": "Update handler", "status": "pending"},
        {"id": 5, "description": "Update response schema", "status": "pending"},
        {"id": 6, "description": "Add tests", "status": "pending"},
        {"id": 7, "description": "Run tests", "status": "pending"}
      ]
    },
    "files_modified": ["src/api/users.py", "src/db/queries.py"],
    "files_read": ["src/api/users.py", "src/db/queries.py", "src/utils/pagination.py"],
    "test_results": null,
    "tokens_used": 45000,
    "cost_usd": 0.18
  }
}
```

**Context Window Management for Large Repos:**

```
Available context: 200K tokens (Claude)

Budget allocation:
  System prompt + tools:         2,000 tokens (fixed)
  Project structure summary:     1,000 tokens (file tree, condensed)
  Relevant files (auto-selected): 20,000-50,000 tokens (ranked by relevance)
  Conversation history:          5,000 tokens (sliding window, last 10 messages)
  Current plan + state:          2,000 tokens
  Reserved for output:           5,000 tokens
  Total used:                    35,000-85,000 tokens

File relevance ranking:
  1. Files explicitly mentioned by user
  2. Files currently being edited
  3. Files imported by edited files
  4. Files matching search patterns
  5. Recently viewed files (recency decay)

Overflow strategy:
  - Summarize files > 500 lines (keep function signatures, docstrings)
  - Drop old conversation turns (keep first + last 8)
  - For repos > 500K LOC: use code search to pull only relevant snippets
```

### LLM Routing Strategy

```python
def route_to_model(request: CodingRequest) -> str:
    """Route to appropriate model based on task complexity."""

    # Opus ($15/$75 per M tokens): Complex reasoning tasks
    if any([
        request.requires_multi_file_refactor,
        request.involves_architectural_decisions,
        request.debugging_complex_failure,
        request.token_estimate > 20000,
    ]):
        return "claude-opus-4"

    # Haiku ($0.80/$4 per M tokens): Simple, fast operations
    if any([
        request.is_simple_completion,     # Complete a function body
        request.is_commit_message,        # Generate commit message
        request.is_simple_rename,         # Rename variable/function
        request.token_estimate < 3000,
    ]):
        return "claude-haiku-3.5"

    # Sonnet ($3/$15 per M tokens): Default for standard coding tasks
    return "claude-sonnet-4"
```

### API Design

#### Start Coding Session

```
POST /api/v1/sessions
```

**Request:**
```json
{
  "project_path": "/home/dev/acme-api",
  "developer_id": "dev_123",
  "ide": "vscode",
  "context": {
    "open_files": ["src/api/users.py", "src/models/user.py"],
    "cursor_position": { "file": "src/api/users.py", "line": 42 },
    "git_branch": "feature/pagination"
  }
}
```

**Response:**
```json
{
  "session_id": "sess_coding_001",
  "sandbox_id": "sbx_abc123",
  "project_context": {
    "language": "python",
    "framework": "fastapi",
    "total_files": 127,
    "total_loc": 45000,
    "indexed": true
  }
}
```

#### Send Coding Request

```
POST /api/v1/sessions/{session_id}/messages
```

**Request:**
```json
{
  "message": "Add cursor-based pagination to the GET /users endpoint. Use the existing pagination utility in src/utils/pagination.py.",
  "context": {
    "selected_code": "def get_users(db: Session):\n    return db.query(User).all()",
    "file": "src/api/users.py",
    "line_range": [42, 43]
  }
}
```

**Response (streamed):**
```json
{
  "message_id": "msg_001",
  "plan": {
    "steps": [
      "Read src/utils/pagination.py to understand the existing pagination utility",
      "Modify get_users() in src/api/users.py to accept cursor and limit params",
      "Update the route handler to parse query parameters",
      "Add response model with pagination metadata",
      "Write tests for paginated endpoint",
      "Run tests"
    ]
  },
  "actions": [
    {
      "type": "file_read",
      "path": "src/utils/pagination.py",
      "status": "completed"
    },
    {
      "type": "file_edit",
      "path": "src/api/users.py",
      "diff": "--- a/src/api/users.py\n+++ b/src/api/users.py\n@@ -42,2 +42,15 @@\n-def get_users(db: Session):\n-    return db.query(User).all()\n+def get_users(db: Session, cursor: str | None = None, limit: int = 20):\n+    query = db.query(User).order_by(User.id)\n+    if cursor:\n+        query = query.filter(User.id > decode_cursor(cursor))\n+    results = query.limit(limit + 1).all()\n+    has_next = len(results) > limit\n+    items = results[:limit]\n+    next_cursor = encode_cursor(items[-1].id) if has_next else None\n+    return PaginatedResponse(\n+        items=items,\n+        next_cursor=next_cursor,\n+        has_next=has_next\n+    )",
      "status": "completed"
    },
    {
      "type": "test_run",
      "command": "python -m pytest tests/api/test_users.py -v",
      "result": "12/12 passed",
      "status": "completed"
    }
  ],
  "metadata": {
    "model_used": "claude-sonnet-4",
    "tokens_used": { "input": 18500, "output": 3200 },
    "cost_usd": 0.104,
    "latency_ms": 8400,
    "files_modified": ["src/api/users.py", "tests/api/test_users.py"],
    "sandbox_executions": 2
  }
}
```

---

## Step 4: Scale the design

### Bottleneck Analysis

| Bottleneck | Symptom | At Scale (5K devs) | Mitigation |
|---|---|---|---|
| LLM API rate limits | 429 errors during peak hours | ~8 req/s peak | Request queuing with priority (interactive > background), multiple API key pools |
| Sandbox provisioning | Slow first execution | 500 concurrent sandboxes | Pre-warm sandbox pool (E2B warm starts: ~300ms vs 5s cold start), session affinity |
| Context window overflow | Large repos don't fit | 500K+ LOC repos | Intelligent file selection, code summarization, chunked processing |
| Code embedding latency | Slow search on first index | 50K files to embed | Background indexing on project open, incremental updates on file save |
| Token costs | $1.3M/month unsustainable | 3B tokens/day | Aggressive model routing, prompt caching, semantic caching |

### Cost Estimation

| Component | Per Session | Daily (10K) | Monthly | Notes |
|---|---|---|---|---|
| Opus (20% of calls) | $3.78 | $37,800 | $1.13M | Complex reasoning, multi-file |
| Sonnet (50% of calls) | $0.62 | $6,210 | $186K | Standard coding tasks |
| Haiku (30% of calls) | $0.04 | $432 | $13K | Simple completions |
| E2B Sandbox | $0.10 | $1,000 | $30K | ~2 executions/session |
| Qdrant (code search) | $0.01 | $100 | $3K | Embedding search |
| LangGraph (state) | $0.005 | $50 | $1.5K | PostgreSQL checkpoints |
| Langfuse (observability) | $0.02 | $200 | $6K | Trace storage |
| **Total (unoptimized)** | **$4.58** | **$45,792** | **$1.37M** | |

**Cost optimization strategy:**

1. **Prompt caching** (Anthropic): Cache system prompt + project context across turns. Estimated 50% input token reduction for subsequent turns. Savings: ~$300K/month
2. **Model routing** (aggressive): Move 60% of Opus calls to Sonnet with better prompts. Savings: ~$400K/month
3. **Semantic caching**: Cache common code patterns (e.g., "add pagination" → template). Hit rate ~15%. Savings: ~$50K/month
4. **Diff-based context**: Send only changed lines, not full files, for subsequent turns. Savings: ~$100K/month

**Optimized monthly cost: ~$520K/month = $104/developer/month**

### Failure Modes

> See: [Failure Modes & Mitigation](../../README.md#failure-modes--mitigation)

| Failure Mode | Impact | Likelihood | Mitigation |
|---|---|---|---|
| **Infinite test loop** | Agent runs tests, fails, edits, runs, fails, edits endlessly | High | Max 5 fix-attempt cycles per test failure. After 5 failures, present the error to the developer and ask for guidance. Budget cap: 50K tokens per step. |
| **Destructive file operations** | Agent deletes important files or overwrites with garbage | Medium | Copy-on-write sandbox filesystem. Snapshot before each edit. Git diff review before applying to real workspace. Never allow `rm -rf` or overwriting files outside project boundary. |
| **Context overflow on large repos** | Agent reads too many files, exceeds context window | High | Token budget per file read (max 5K tokens). Auto-summarize large files. File relevance scoring — only load top-N most relevant files. Alert developer when context is >80% full. |
| **Hallucinated imports/APIs** | Agent generates code using non-existent libraries or functions | Medium | After generating code, validate imports against project's installed dependencies. Run type checker (mypy/tsc) in sandbox. If import fails, search for correct import. |
| **Sandbox escape** | Malicious prompt tricks agent into executing dangerous commands | Low | Allowlist of safe commands. No network access. Resource limits (CPU, memory, disk). Filesystem isolation (chroot/container). All commands logged and auditable. |
| **Git corruption** | Agent creates bad commits or force-pushes | Low | All git operations are preview-first (show diff before committing). No force-push allowed. Branch protection: never commit directly to main. All commits are reversible. |
| **Session state loss** | LangGraph checkpoint fails, losing plan progress | Low | Write-ahead log for state changes. Retry with last successful checkpoint. Graceful degradation: agent can re-read files to reconstruct state. |

### Observability Setup

> See: [Observability & Monitoring](../../README.md#observability--monitoring)

**Key Metrics:**

| Metric | Target | Alert Threshold |
|---|---|---|
| Time to first code suggestion (p50/p95) | p50 < 3s, p95 < 8s | p95 > 15s |
| Code compilation success rate | > 85% | < 75% |
| Test pass rate (first attempt) | > 70% | < 55% |
| Cost per session | < $5.00 | > $10.00 |
| Sandbox startup time (p95) | < 1s | > 3s |
| Context utilization (% of window used) | 30-60% | > 90% (overflow risk) |
| Fix loop iterations (avg) | < 2.5 | > 4 (potential infinite loop) |
| Model routing accuracy | Opus used for genuinely hard tasks | Opus usage > 30% of calls |

**Tracing (Langfuse):**
```
Trace: sess_coding_001 / msg_001 (Add pagination)
  ├── Planner LLM Call (Sonnet, 1.8s, 5000 in / 800 out)
  │   └── Plan: 7 steps generated
  ├── Step 1: read_file (pagination.py, 0.1s, success)
  ├── Step 2: read_file (users.py, 0.1s, success)
  ├── Step 3: LLM Call (Sonnet, 2.1s, 12000 in / 2000 out)
  │   └── Generated edit for users.py
  ├── Step 3: edit_file (users.py, 0.05s, success)
  ├── Step 6: LLM Call (Sonnet, 1.5s, 8000 in / 1500 out)
  │   └── Generated test file
  ├── Step 7: execute_command (pytest, 3.2s, success, 12/12 passed)
  └── Summary LLM Call (Haiku, 0.5s, 2000 in / 300 out)
  Total: 8.4s | Cost: $0.104 | Tokens: 21,700 | Files modified: 2
```

---

## Additional Talking Points

### Sandbox Architecture Deep Dive
> See: [Security](../../README.md#security)
- **E2B (preferred for cloud deployment)**: Pre-built sandboxes with language runtimes, ~300ms warm start, $0.05/execution
- **Docker (self-hosted)**: More control, slower startup (2-5s), free but requires infra management
- **Firecracker (high-security)**: VM-level isolation, used by AWS Lambda, <125ms cold start but complex setup
- **Key invariant**: The agent NEVER executes code on the developer's actual machine. All execution happens in a disposable sandbox with the project mounted read-only. Edits are applied as patches after review.

### Large Repository Strategy
- **Repos > 500K LOC**: Build a "code graph" (function call graph + import graph) using tree-sitter. When the agent needs to understand a function, it traverses the graph to find related code instead of reading entire files.
- **Repos > 1M LOC**: Add a "file finder" sub-agent (Haiku-powered) that answers "which files are relevant to this task?" before the main agent starts working. This agent uses the code graph + file path heuristics.

### Prompt Caching Strategy
> See: [Cost Engineering](../../README.md#cost-engineering)
- **Level 1**: Cache system prompt + tool definitions across all turns in a session (2,000 tokens saved per turn)
- **Level 2**: Cache project context (file tree, framework detection) across sessions for the same project (1,000 tokens saved)
- **Level 3**: Cache frequently-read files within a session using Redis (avoid re-reading unchanged files)
- **Expected savings**: 40-60% reduction in input tokens
