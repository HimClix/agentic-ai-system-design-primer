# Design a Browser Automation Agent

> Note: This document links directly to relevant areas of the
> [Agentic AI System Design Primer](../../README.md)

## Step 1: Outline use cases and constraints

### Use cases

- **Form filling** -- user says "apply to this job posting on LinkedIn," agent navigates to the page, fills in fields from the user's profile, and uploads a resume
- **Data extraction** -- "scrape all product prices from this competitor's catalog page" -- agent navigates pagination, handles lazy loading, extracts structured data
- **Auth flow handling** -- agent logs into websites using stored credentials, handles 2FA prompts by pausing for user input, manages session cookies
- **Screenshot verification** -- after each action, agent takes a screenshot and uses a vision model to verify the page state matches expectations
- **Multi-step web workflows** -- "book me the cheapest flight to Tokyo on Google Flights for June 15-22" -- agent navigates search, sets filters, compares results, selects the best option
- **Monitoring and alerting** -- "check this government website every hour for new visa appointment slots and alert me" -- recurring browser tasks
- **Testing and QA** -- "test the checkout flow on our staging site and report any broken steps" -- agent follows a test plan, screenshots each step, reports failures

### Constraints and assumptions

#### State assumptions

- **Concurrent sessions:** 1,000 simultaneous browser sessions
- **Session duration:** 30 seconds (simple extraction) to 10 minutes (complex multi-page workflow)
- **Tasks/day:** ~20K browser automation tasks
- **Latency:** Per-step latency of 2-5 seconds (navigate + screenshot + vision + action). Total task latency: 30s to 10 min.
- **Accuracy:** 95% task completion rate for known site patterns, 80% for unknown sites
- **Isolation:** Each browser session MUST be fully isolated (no cross-session data leakage)
- **Anti-bot:** Must handle common anti-bot measures gracefully (CAPTCHA detection, rate limiting)

#### Calculate usage

```
Concurrent sessions: 1,000
Tasks/day: 20,000
Avg steps/task: 8 (navigate, screenshot, reason, act, verify -- repeated)
Total steps/day: 160,000

Per step token usage:
  System prompt:              1,500 tokens
  Page DOM (cleaned):         2,000 tokens (stripped to relevant elements)
  Screenshot analysis (vision): 1,200 tokens (image input ~800 + reasoning ~400)
  Navigation history:           500 tokens
  Action generation:            200 tokens
  Total input:               ~5,400 tokens
  Total output:                ~300 tokens

Vision model calls: 160,000/day (one screenshot per step)
Text model calls:   160,000/day (reasoning + action selection)

Cost calculation (Claude Sonnet for reasoning, vision for screenshots):
  Text reasoning (Sonnet):
    Input:  160K * 5,400 = 864M tokens * $3/M = $2,592/day
    Output: 160K * 300   = 48M tokens * $15/M = $720/day

  Vision analysis (Sonnet with vision):
    Image tokens: 160K * 1,600 tokens/image = 256M tokens * $3/M = $768/day
    (Images are ~1,600 tokens per 1024x768 screenshot at medium resolution)

  Browser infrastructure:
    1,000 concurrent Docker containers * $0.02/hr = $20/hr = $480/day

  Total: ~$4,560/day = ~$137K/month
```

### Out of scope

- Building a browser engine (use Playwright/Puppeteer)
- CAPTCHA solving (detect and escalate to human or third-party solver)
- Credential management system (assume Vault integration exists)
- Mobile browser automation
- PDF download and processing within browser

---

## Step 2: Create a high-level design

```
                    ┌────────────────────────┐
                    │     User / API Client   │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │      API Gateway        │
                    │  Auth · Rate Limit      │
                    │  Task Queue Admission   │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │    Task Orchestrator    │
                    │  (Queue + Scheduler)    │
                    │  Assigns tasks to       │
                    │  browser workers        │
                    └───────────┬────────────┘
                                │
              ┌─────────────────▼─────────────────┐
              │         BROWSER WORKER POOL         │
              │  (Auto-scaling Docker containers)   │
              │                                     │
              │  ┌──────────────────────────────┐  │
              │  │     Browser Agent (ReAct)     │  │
              │  │                                │  │
              │  │  ┌──────┐    ┌──────────┐    │  │
              │  │  │ LLM  │◄──►│ Page     │    │  │
              │  │  │Router│    │ State    │    │  │
              │  │  └──┬───┘    └──────────┘    │  │
              │  │     │                         │  │
              │  │  ┌──▼───────────────────┐    │  │
              │  │  │    Tool Executor      │    │  │
              │  │  │  ┌───────────────┐   │    │  │
              │  │  │  │ Playwright    │   │    │  │
              │  │  │  │ (click, type, │   │    │  │
              │  │  │  │  navigate,    │   │    │  │
              │  │  │  │  scroll)      │   │    │  │
              │  │  │  ├───────────────┤   │    │  │
              │  │  │  │ Screenshot +  │   │    │  │
              │  │  │  │ Vision Model  │   │    │  │
              │  │  │  ├───────────────┤   │    │  │
              │  │  │  │ DOM Parser    │   │    │  │
              │  │  │  │ (clean HTML)  │   │    │  │
              │  │  │  └───────────────┘   │    │  │
              │  │  └──────────────────────┘    │  │
              │  └──────────────────────────────┘  │
              │                                     │
              │  ┌──────────────────────────────┐  │
              │  │  Sandbox Boundary (Docker)    │  │
              │  │  - Network egress allowlist   │  │
              │  │  - No host filesystem access  │  │
              │  │  - Session-scoped credentials  │  │
              │  │  - 10-minute max lifetime      │  │
              │  └──────────────────────────────┘  │
              └─────────────────┬─────────────────┘
                                │
              ┌─────────────────▼─────────────────┐
              │         MEMORY + STATE              │
              │  ┌──────────┐  ┌───────────────┐  │
              │  │ Redis    │  │ PostgreSQL    │  │
              │  │ (session │  │ (task results,│  │
              │  │  state)  │  │  site patterns│  │
              │  │          │  │  nav history) │  │
              │  └──────────┘  └───────────────┘  │
              └─────────────────┬─────────────────┘
                                │
              ┌─────────────────▼─────────────────┐
              │        OBSERVABILITY                │
              │  Langfuse traces · Screenshots S3   │
              │  Step-by-step replay · Cost tracking │
              └─────────────────────────────────────┘
```

**Component justification:**

| Component | Purpose | What breaks without it |
|---|---|---|
| **Task Orchestrator** | Queue management, worker assignment, retry scheduling | Tasks pile up, no backpressure, no retry |
| **Browser Worker Pool** | Auto-scaling isolated browser instances | Fixed capacity, resource waste or starvation |
| **Sandbox Boundary** | Isolation per session -- no cross-session leakage | Credential leakage, session hijacking |
| **Vision Model** | Verify page state matches expectations after each action | Agent acts on stale/wrong page, silent failures |
| **DOM Parser** | Extract clean, relevant HTML for LLM consumption | Raw HTML is too large (50K+ tokens), irrelevant noise |
| **Site Patterns DB** | Store learned navigation patterns for known sites | Re-discovers navigation on every visit, fragile |

---

## Step 3: Design core components

### Agent Architecture Decision

**Single agent per browser session**, with a shared site-pattern knowledge base.

Cross-reference: [Single Agent vs Multi-Agent](../../README.md#single-agent-vs-multi-agent)

**Why single agent:** Each browser task is a sequential process -- you can only interact with one page at a time. The agent needs to see the current page state, reason about it, and take the next action. A multi-agent system would not help because the bottleneck is the sequential nature of browser interaction, not reasoning capacity.

**Why NOT multi-agent:** There is no parallel execution opportunity within a single browser session. You cannot click two things at the same time. If a user needs to scrape 10 different sites simultaneously, you spin up 10 separate single-agent browser sessions, not one multi-agent system.

### Pattern Selection

**ReAct (Reasoning + Acting)** -- Cross-reference: [ReAct](../../README.md#react-reasoning--acting)

**Why ReAct:**
1. Each step genuinely depends on the previous step's result -- you cannot plan a click sequence without seeing the current page
2. Pages are dynamic -- content loads asynchronously, modals appear, layouts shift
3. The agent must adapt in real-time: "The button I expected isn't here, let me scroll down"

**Why NOT Plan-and-Execute:** Page layouts change between visits. A pre-planned click sequence ("click element #submit-btn at coordinates 450,300") would break whenever the site updates. ReAct observes the current state at every step.

**ReAct loop for browser automation:**
```
Thought: I need to navigate to the LinkedIn job posting
Action: navigate(url="https://linkedin.com/jobs/view/12345")
Observation: [screenshot] Page loaded. Login modal is blocking the content.

Thought: I need to log in first. Let me use the stored credentials.
Action: type(selector="input#username", text="user@email.com")
Observation: [screenshot] Email field filled. Password field is visible.

Thought: Now enter the password
Action: type(selector="input#password", text="***")
Observation: [screenshot] Password entered. Sign In button visible.

Thought: Click sign in
Action: click(selector="button[type='submit']")
Observation: [screenshot] 2FA prompt appeared. I need human input.

Thought: I cannot proceed without the 2FA code. Escalate to user.
Action: request_human_input(prompt="Please enter your LinkedIn 2FA code")
```

**Max iterations:** 25 (browser tasks can be long)
**Step timeout:** 30 seconds per step (prevents hanging on slow pages)
**Session timeout:** 10 minutes total

### Tool Design

```json
{
  "tools": [
    {
      "name": "navigate",
      "description": "Navigate the browser to a URL. Waits for page load (networkidle). Returns cleaned DOM snapshot + screenshot.",
      "input_schema": {
        "type": "object",
        "properties": {
          "url": {"type": "string", "format": "uri", "description": "Full URL to navigate to"},
          "wait_for": {
            "type": "string",
            "description": "CSS selector to wait for before considering page loaded. Optional.",
            "default": null
          },
          "timeout_ms": {"type": "integer", "default": 10000, "maximum": 30000}
        },
        "required": ["url"]
      }
    },
    {
      "name": "click",
      "description": "Click an element on the page. Use CSS selectors. If multiple elements match, clicks the first visible one. Returns screenshot after click.",
      "input_schema": {
        "type": "object",
        "properties": {
          "selector": {"type": "string", "description": "CSS selector for the element to click"},
          "text_content": {"type": "string", "description": "Alternative: find element by its visible text content"},
          "wait_after_ms": {"type": "integer", "default": 1000, "description": "Wait time after click for page updates"}
        }
      }
    },
    {
      "name": "type_text",
      "description": "Type text into an input field. Clears existing content first unless append=true.",
      "input_schema": {
        "type": "object",
        "properties": {
          "selector": {"type": "string", "description": "CSS selector for the input field"},
          "text": {"type": "string", "description": "Text to type"},
          "append": {"type": "boolean", "default": false},
          "press_enter": {"type": "boolean", "default": false, "description": "Press Enter after typing"}
        },
        "required": ["selector", "text"]
      }
    },
    {
      "name": "scroll",
      "description": "Scroll the page. Use to reveal lazy-loaded content or find elements below the fold.",
      "input_schema": {
        "type": "object",
        "properties": {
          "direction": {"type": "string", "enum": ["down", "up"]},
          "amount": {"type": "string", "enum": ["page", "half_page", "to_bottom", "to_top"]},
          "selector": {"type": "string", "description": "Scroll a specific scrollable container instead of the page"}
        },
        "required": ["direction", "amount"]
      }
    },
    {
      "name": "extract_data",
      "description": "Extract structured data from the current page. Specify a CSS selector for the repeating container and the fields to extract from each item.",
      "input_schema": {
        "type": "object",
        "properties": {
          "container_selector": {"type": "string", "description": "CSS selector for the repeating items (e.g., '.product-card')"},
          "fields": {
            "type": "object",
            "description": "Map of field_name -> CSS selector relative to each container",
            "additionalProperties": {"type": "string"}
          },
          "max_items": {"type": "integer", "default": 50}
        },
        "required": ["container_selector", "fields"]
      }
    },
    {
      "name": "screenshot_analyze",
      "description": "Take a screenshot and analyze it with the vision model. Use when DOM parsing is insufficient (e.g., canvas elements, complex visual layouts, CAPTCHA detection).",
      "input_schema": {
        "type": "object",
        "properties": {
          "question": {"type": "string", "description": "What to analyze in the screenshot. e.g., 'Is a CAPTCHA present?' or 'What is the total price shown?'"},
          "region": {
            "type": "object",
            "properties": {
              "x": {"type": "integer"}, "y": {"type": "integer"},
              "width": {"type": "integer"}, "height": {"type": "integer"}
            },
            "description": "Optional: crop to a specific region of the page"
          }
        },
        "required": ["question"]
      }
    },
    {
      "name": "get_page_state",
      "description": "Get the current page state: URL, title, cleaned DOM (interactive elements only), and any visible alerts/modals.",
      "input_schema": {
        "type": "object",
        "properties": {
          "include_full_dom": {"type": "boolean", "default": false, "description": "Include full cleaned DOM (more tokens but more context)"},
          "interactive_only": {"type": "boolean", "default": true, "description": "Only return clickable/typeable elements"}
        }
      }
    },
    {
      "name": "request_human_input",
      "description": "Pause execution and request input from the user. Use for 2FA codes, CAPTCHAs, or decisions the agent cannot make.",
      "input_schema": {
        "type": "object",
        "properties": {
          "prompt": {"type": "string", "description": "What to ask the user"},
          "input_type": {"type": "string", "enum": ["text", "captcha_image", "approval"], "default": "text"},
          "timeout_seconds": {"type": "integer", "default": 300}
        },
        "required": ["prompt"]
      }
    }
  ]
}
```

**DOM cleaning strategy (critical for token efficiency):**
```python
def clean_dom(page_html: str) -> str:
    """
    Reduce a 50K+ token raw DOM to ~2,000 tokens of relevant elements.
    """
    # 1. Remove all <script>, <style>, <svg>, <noscript> tags
    # 2. Remove all hidden elements (display:none, visibility:hidden)
    # 3. Keep only interactive elements: <a>, <button>, <input>, <select>, <textarea>
    # 4. Keep structural elements: <h1>-<h6>, <main>, <nav>, <form>
    # 5. For each element, keep: tag, id, class, href, placeholder, aria-label, text content
    # 6. Assign short numeric IDs: [1] <button>Submit</button>, [2] <input placeholder="Email">
    # 7. Truncate text content to 100 chars per element
    pass
```

### Memory Architecture

| Memory Type | Store | What It Holds | TTL |
|---|---|---|---|
| **Short-term** (page state) | In-memory (per worker) | Current URL, cleaned DOM, screenshot buffer, navigation history (last 5 pages) | Session lifetime (max 10 min) |
| **Long-term** (site patterns) | PostgreSQL | Learned CSS selectors per site, navigation flows, login page detection patterns | Updated on success, 30-day expiry for unused |
| **Credential store** | HashiCorp Vault | Per-user site credentials, OAuth tokens, session cookies | Per credential policy |

**Site pattern learning:**
After a successful task, the agent's navigation trace is saved as a "site pattern":
```json
{
  "site": "linkedin.com",
  "task_type": "job_application",
  "learned_at": "2026-05-19",
  "success_rate": 0.92,
  "steps": [
    {"action": "navigate", "url_pattern": "/jobs/view/*"},
    {"action": "click", "selector": "button.jobs-apply-button", "fallback": "a[contains(text(), 'Apply')]"},
    {"action": "wait_for", "selector": ".jobs-easy-apply-modal"},
    {"action": "fill_form", "field_mapping": {"name": "#first-name", "email": "#email"}}
  ]
}
```

On subsequent visits, the agent retrieves the pattern and uses it as a hint, falling back to vision-based navigation if the pattern fails.

### API Design

```
POST /api/v1/browser/task
Content-Type: application/json
Authorization: Bearer <jwt_token>

Request:
{
  "task_description": "Go to amazon.com, search for 'wireless headphones under $50', and extract the top 10 results with name, price, and rating",
  "start_url": "https://amazon.com",
  "max_steps": 15,
  "timeout_seconds": 300,
  "credentials_key": "vault://user123/amazon",    // optional
  "callback_url": "https://my-app.com/webhook",   // async result delivery
  "screenshot_mode": "every_step"                  // or "final_only" or "on_error"
}

Response (immediate):
{
  "task_id": "task_abc123",
  "status": "queued",
  "estimated_start": "2026-05-19T10:00:05Z",
  "polling_url": "/api/v1/browser/task/task_abc123/status"
}

GET /api/v1/browser/task/task_abc123/status
Response:
{
  "task_id": "task_abc123",
  "status": "running",
  "current_step": 5,
  "max_steps": 15,
  "current_url": "https://amazon.com/s?k=wireless+headphones",
  "screenshots": [
    {"step": 1, "url": "s3://screenshots/task_abc123/step_1.png"},
    {"step": 2, "url": "s3://screenshots/task_abc123/step_2.png"}
  ],
  "usage": {"steps_completed": 5, "tokens_used": 28000, "cost_usd": 0.12}
}

GET /api/v1/browser/task/task_abc123/result
Response:
{
  "task_id": "task_abc123",
  "status": "completed",
  "result": {
    "extracted_data": [
      {"name": "Sony WH-1000XM4", "price": "$48.99", "rating": "4.7"},
      {"name": "JBL Tune 510BT", "price": "$29.95", "rating": "4.5"}
    ]
  },
  "replay_url": "/api/v1/browser/task/task_abc123/replay",
  "total_steps": 8,
  "total_cost_usd": 0.19,
  "duration_seconds": 45
}
```

---

## Step 4: Scale the design

### Bottleneck Analysis

| Bottleneck | At 1K Sessions | At 10K Sessions | Mitigation |
|---|---|---|---|
| **Browser container startup** | ~3s cold start | 30s+ if pool exhausted | Pre-warmed container pool (keep 200 warm containers). Auto-scale based on queue depth. |
| **Vision model calls** | 160K/day | 1.6M/day, rate limits | Batch screenshots when possible. Use smaller model for simple verifications. Cache known page layouts. |
| **Memory per container** | ~500MB per browser | 5TB total RAM at 10K | Use headless Chrome with `--disable-gpu`, `--single-process`. Limit tab count to 1. Target 300MB per container. |
| **Network bandwidth** | Screenshots: 160K * 200KB = 32GB/day | 320GB/day | Compress screenshots (WebP at 80% quality). Store in S3 with lifecycle policy (delete after 7 days). |
| **Anti-bot blocking** | Occasional | Systematic at scale | Rotate user agents, use residential proxies for sensitive sites, implement request spacing. |

### Cost Estimation

| Component | Monthly Cost | Notes |
|---|---|---|
| **LLM - Text reasoning (Sonnet)** | $99,360 | 160K steps/day * 30 days * $0.0207/step |
| **LLM - Vision analysis** | $23,040 | 160K screenshots/day * 30 * $0.0048/screenshot |
| **Browser infra (Docker/K8s)** | $14,400 | 1K concurrent * $0.02/hr avg |
| **Screenshot storage (S3)** | $230 | 32GB/day * 30 * $0.023/GB + transfer |
| **PostgreSQL (site patterns)** | $400 | db.r6g.large |
| **Redis (session state)** | $300 | r6g.medium |
| **Proxy service** | $2,000 | Residential proxy for anti-bot |
| **Langfuse** | $59 | Observability |
| **Total** | **~$139,789/month** | **$0.233 per task** (20K tasks/day) |

**Cost optimization:**
1. **Skip vision for known pages** -- if site pattern exists and DOM matches expected structure, skip the screenshot analysis. Saves ~40% of vision costs (~$9K/month).
2. **Use Haiku for simple DOM analysis** -- "is the login button visible?" does not need Sonnet. Route simple checks to Haiku. Saves ~30% of text reasoning costs.
3. **Batch data extraction** -- extract all data from a page in one tool call instead of one element at a time.

### Failure Modes

Cross-reference: [Failure Modes & Mitigation](../../README.md#failure-modes--mitigation)

| Failure Mode | Severity | Detection | Mitigation |
|---|---|---|---|
| **Anti-bot detection** | High | Page shows CAPTCHA, "access denied," or Cloudflare challenge page | Vision model detects CAPTCHA. Options: escalate to human, use CAPTCHA service, or abort with clear error. |
| **Page layout change** | Medium | CSS selector not found, element not visible | Fallback strategy: try stored selector -> try aria-label -> try text content -> use vision model to locate element visually. |
| **Stale selectors** | Medium | `ElementNotFound` errors | Three-tier selector strategy: 1) data-testid (most stable), 2) aria attributes, 3) CSS class (least stable). Update site patterns on failure. |
| **Infinite navigation loop** | High | Agent visits the same URL > 2 times | Track URL history. If same URL appears 3x, abort step and try alternative approach. Hard max of 25 steps. |
| **Auth token expiry** | Medium | 401/403 response mid-session | Detect auth failure, attempt re-login using stored credentials. If re-login fails, escalate to user. |
| **Dynamic content not loading** | Medium | Expected elements not in DOM after wait | Incremental wait strategy: 1s -> 3s -> 5s. If still not loaded, take screenshot and use vision to check if content is actually present but rendered differently. |
| **Session hijacking (security)** | **Critical** | Cross-session data detected in container | Complete container isolation. No shared volumes. Ephemeral credentials. Container destroyed after each task. |
| **Cost runaway** | High | Token count per task > 50K (5x expected) | Per-task token budget. Kill session if budget exceeded. Alert on P95 cost per task. |

### Observability Setup

**Tracing (Langfuse):**
- Each task gets a trace containing: task_id, user_id, start_url, every step (navigate/click/type/extract) as child spans
- Each step span includes: action type, selector used, screenshot URL, DOM snapshot hash, vision model output, latency

**Step-by-step replay:**
- All screenshots saved to S3 with step numbers
- Replay UI shows screenshots in sequence with overlaid action indicators (where the agent clicked, what it typed)
- Invaluable for debugging: "why did the agent click the wrong button?"

**Metrics:**

| Metric | Alert Threshold |
|---|---|
| `browser_task_success_rate` | < 85% over 1 hour |
| `browser_step_latency_p95` | > 15s (page load + vision analysis) |
| `browser_vision_calls_per_task` | > 30 (possible loop) |
| `browser_antibot_detection_rate` | > 20% for any single site |
| `browser_container_pool_utilization` | > 80% (scale up) |
| `browser_container_startup_p95` | > 10s (pool too small) |
| `browser_cost_per_task_p95` | > $0.50 (cost runaway) |
| `browser_selector_fallback_rate` | > 30% (site patterns need updating) |

---

## Additional Talking Points

### DOM Cleaning vs Pure Vision Approach

There are two schools of thought for browser agents:

1. **DOM-first (our approach):** Parse the DOM, extract interactive elements, send as text to LLM. Use vision only for verification.
   - Pro: Much cheaper (text tokens vs image tokens), more precise selectors
   - Con: Breaks on canvas-rendered UIs, complex visual layouts

2. **Vision-first (e.g., Claude Computer Use):** Send screenshots, let the vision model decide coordinates to click.
   - Pro: Works on any visual layout, no DOM parsing needed
   - Con: 4-5x more expensive per step, less precise clicks, slower

**Our hybrid approach:** DOM-first for interaction, vision for verification and fallback. This gives us the cost efficiency of text-based reasoning with the robustness of visual verification.

### Anti-Bot Mitigation Architecture

```
Request goes through:
1. Random delay between actions (1-3s, human-like)
2. Realistic mouse movement simulation (bezier curves, not teleporting)
3. Rotate user-agent strings from a pool of 50 real browser fingerprints
4. Use residential proxies for high-value sites (rotate IP per session)
5. Respect robots.txt (ethical scraping)
6. Per-site rate limiting (max 1 request/5s for any single domain)
```

### Container Lifecycle Management

```
Container States:
  COLD    → WARMING (pre-loading browser)
  WARMING → WARM (browser ready, no task)
  WARM    → ACTIVE (task assigned)
  ACTIVE  → DRAINING (task complete, cleanup)
  DRAINING→ DESTROYED (container removed)

Pool Management:
  - Maintain min 200 warm containers
  - Scale up when queue depth > 100 or utilization > 70%
  - Scale down when utilization < 30% for 10 min
  - Max lifetime: 10 min (hard kill, prevents resource leaks)
  - Health check: browser process alive + responsive
```

### Interview Cross-References
- Why ReAct for sequential tasks: [ReAct](../../README.md#react-reasoning--acting)
- Container isolation as guardrail: [Safety & Guardrails -- Sandbox layer](../../README.md#safety--guardrails)
- Vision model cost impact: [Cost Engineering](../../README.md#cost-engineering)
- Step-by-step tracing: [Observability](../../README.md#observability--monitoring)
- Anti-bot as security concern: [Security](../../README.md#security)
- Async task queue architecture: [Scalability -- Async Orchestration](../../README.md#async-orchestration)
