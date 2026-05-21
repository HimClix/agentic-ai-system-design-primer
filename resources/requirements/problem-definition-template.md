# Problem Definition Template for Agentic AI Systems
> The first 10 minutes of a system design interview (or a real project kickoff) define whether you build the right thing. This template ensures you ask every question before you draw a single box.

## What It Is

A structured problem definition template for agentic AI systems, usable in both system design interviews and real-world project kickoffs. It forces you to define the business case, user workflow, constraints, scale, latency, accuracy, human involvement, and compliance requirements before starting architectural design. Skipping this step is the most common reason agent systems get redesigned.

## How It Works

Walk through each of the 8 sections below. In an interview, spend 5-8 minutes on this. In a real project, spend 1-2 days getting alignment with stakeholders on every answer.

### The 8 Sections

```
1. Business Use Case         → Why are we building this?
2. User Workflow              → How does the user interact?
3. Constraints                → What are the hard limits?
4. Scale Assumptions          → How big does it need to be?
5. Latency Expectations       → How fast does it need to respond?
6. Accuracy Requirements      → How correct does it need to be?
7. Human Involvement          → How much autonomy does the agent have?
8. Compliance & Security      → What regulations and data rules apply?
```

---

## Section Details and Guiding Questions

### 1. Business Use Case

**Purpose**: Define the problem the agent solves in business terms. This anchors all subsequent decisions.

**Questions to answer:**
- What manual process does this agent automate or augment?
- Who is the primary user? (developer, customer, internal operator, end consumer)
- What is the success metric? (time saved, cost reduced, accuracy improved, CSAT increase)
- What does failure look like? (user falls back to manual process, wrong answer given, money lost)
- What is the business value? (cost savings of $X/year, Y% productivity increase)
- Is this a new capability or replacing an existing system?

### 2. User Workflow

**Purpose**: Define the interaction model between user and agent.

**Questions to answer:**
- How does the user start? (typed query, button click, API call, event trigger)
- What does a single interaction look like? (one-shot query, multi-turn conversation, form-guided workflow)
- How long is a typical session? (30 seconds, 5 minutes, 1 hour, days)
- Does the user provide feedback? (thumbs up/down, corrections, approvals)
- How does the user know the agent succeeded? (direct output, notification, side effect in another system)
- What is the handoff to human like? (seamless transfer, escalation with context, email summary)

### 3. Constraints

**Purpose**: Define hard limits that cannot be negotiated.

**Questions to answer:**
- Budget: What is the maximum monthly LLM cost? (per user, total platform)
- Data sensitivity: What data does the agent access? (PII, financial, health, proprietary)
- Data residency: Where must data be stored and processed? (US only, EU, specific country)
- Vendor lock-in: Must we support multiple LLM providers? (portability requirement)
- Technology: Are there mandated tech stack choices? (cloud provider, language, framework)
- Timeline: When does this need to be in production? (affects build vs. buy decisions)

### 4. Scale Assumptions

**Purpose**: Define the expected load at launch and growth targets.

**Questions to answer:**
- How many users at launch? (day 1)
- How many users at 6 months? 12 months?
- How many concurrent sessions expected?
- How many agent turns per session?
- How much data does the agent need to access? (KB of documents, number of DB rows)
- What is the peak-to-average traffic ratio?
- Are there predictable traffic spikes? (Black Friday, tax season, product launch)

### 5. Latency Expectations

**Purpose**: Define speed requirements by interaction type.

**Questions to answer:**
- Is this synchronous (user waits) or asynchronous (user gets notified later)?
- If sync: what is the maximum acceptable wait time?
- Does the user see a streaming response? (required for waits > 3 seconds)
- Is time-to-first-token (TTFT) important? (chat = yes, batch processing = no)
- Are there different latency tiers for different task types?
- Is there a hard timeout? (user gives up and leaves after X seconds)

### 6. Accuracy Requirements

**Purpose**: Define how correct the agent needs to be, and what happens when it's wrong.

**Questions to answer:**
- Can the agent be wrong X% of the time? What is X?
- What is the cost of a wrong answer? (user confusion, financial loss, legal liability, safety risk)
- Is there a distinction between "wrong" and "incomplete"?
- How do you measure correctness? (human eval, automated eval, user feedback)
- Are there different accuracy tiers? (factual accuracy vs. tone vs. formatting)
- Is hallucination acceptable? (never for medical/legal, sometimes for brainstorming)

### 7. Human Involvement Requirements

**Purpose**: Define the spectrum from full autonomy to full human control.

**Questions to answer:**
- Can the agent take actions without human approval? Which actions?
- What actions always require human approval? (financial, communications, deletions)
- Is there a dollar threshold for automatic vs. approved actions?
- Who approves? (the user, a manager, an ops team)
- What is the approval latency budget? (sync: seconds, async: hours)
- Can the agent learn from human corrections? (feedback loop)
- What is the escalation path? (agent → human tier 1 → human tier 2)

### 8. Compliance & Security Assumptions

**Purpose**: Define regulatory and security requirements upfront -- these are expensive to retrofit.

**Questions to answer:**
- Which regulations apply? (GDPR, HIPAA, SOC 2, PCI DSS, SOX, AI Act)
- Does the LLM provider need specific certifications? (SOC 2, HIPAA BAA)
- Can data be sent to third-party LLM APIs? (some orgs require on-prem inference)
- What PII is involved? (names, emails, SSN, credit cards, health records)
- What is the data retention policy? (how long to keep conversations, logs, memories)
- Is audit trail required? (every agent action logged with reasoning)
- Is explainability required? (show users why the agent made a decision)
- Is there an AI ethics review board? (required for some enterprises)

---

## Filled-Out Examples

### Example 1: AI Coding Agent

```
SYSTEM: AI Coding Agent for IDE (like GitHub Copilot / Cursor)
```

**1. Business Use Case**
- **Problem**: Developers spend 40% of time on boilerplate code, debugging, and documentation
- **User**: Software developers (junior to senior)
- **Success metric**: 30% reduction in time-to-PR, 20% increase in code review velocity
- **Failure**: Agent generates incorrect code that passes tests but has bugs; user loses trust
- **Business value**: $50K/developer/year productivity gain across 500 developers = $25M/year

**2. User Workflow**
- **Start**: Inline code completion (tab-to-accept) or chat sidebar (natural language request)
- **Interaction**: Mix of one-shot (autocomplete) and multi-turn (chat for complex refactoring)
- **Session length**: Continuous during development (hours), individual interactions 10s-5min
- **Feedback**: Accept/reject completions, explicit corrections in chat
- **Success signal**: Code compiles, tests pass, PR merged

**3. Constraints**
- **Budget**: $50/developer/month LLM cost ceiling
- **Data sensitivity**: Source code is proprietary (cannot send to external API without DLP)
- **Data residency**: US-based processing (SOC 2 requirement)
- **Vendor**: Must support Claude + GPT-4o (avoid single-provider lock-in)
- **Timeline**: MVP in 3 months, GA in 6 months

**4. Scale Assumptions**
- **Launch**: 50 developers (internal dogfood)
- **6 months**: 500 developers (full engineering org)
- **12 months**: 2,000 developers (including contractors)
- **Concurrent sessions**: 200 peak (assuming 40% of devs active simultaneously)
- **Turns per session**: 50-200 autocomplete requests/hour, 5-10 chat messages/hour
- **Codebase size**: 500 repos, 10M+ lines of code across the org
- **Peak ratio**: 3x (morning standup → everyone starts coding)

**5. Latency Expectations**
- **Autocomplete TTFT**: < 200ms (must feel instant)
- **Autocomplete full response**: < 500ms
- **Chat TTFT**: < 500ms
- **Chat full response**: < 10s (streaming required)
- **Code edit application**: < 2s
- **Codebase search/RAG**: < 1s
- **Timeout**: 30s for chat, 1s for autocomplete (then show nothing)

**6. Accuracy Requirements**
- **Autocomplete**: > 30% acceptance rate (industry benchmark)
- **Chat code generation**: > 80% of generated code should compile without errors
- **Bug introduction rate**: < 1% (generated code must not introduce bugs at higher rate than human code)
- **Hallucination**: Zero tolerance for fabricated API names, library versions, or function signatures
- **Measurement**: A/B test acceptance rate, post-merge bug rate tracking

**7. Human Involvement**
- **Autonomy level**: Agent suggests, human always approves (no autonomous code commits)
- **Approval actions**: Every code change requires explicit accept (tab, click, or enter)
- **No approval needed for**: Reading code, searching codebase, explaining code
- **Learning from corrections**: Yes (fine-tune on accepted/rejected completions, anonymized)

**8. Compliance & Security**
- **Regulations**: SOC 2 Type II (enterprise customers require it)
- **LLM provider**: Must have SOC 2 certification, DPA signed, no training on our data
- **Data handling**: Source code filtered through DLP before sending to LLM; secrets/keys never sent
- **PII**: Minimal (variable names, comments might contain names -- accepted risk)
- **Audit trail**: Log every completion request (anonymized) for usage analytics
- **On-prem option**: Required for 3 enterprise customers (self-hosted vLLM with Llama 3.1)

---

### Example 2: Customer Support Agent

```
SYSTEM: AI Customer Support Agent for E-Commerce Platform
```

**1. Business Use Case**
- **Problem**: Support team handles 10,000 tickets/day, 60% are routine (order status, returns, FAQ)
- **User**: End customers contacting support via chat widget
- **Success metric**: Resolve 60% of tickets without human escalation; CSAT > 4.2/5.0
- **Failure**: Agent gives wrong order info, processes incorrect refund, or frustrates customer
- **Business value**: $2M/year savings (reduce support headcount from 50 to 25 for tier-1)

**2. User Workflow**
- **Start**: Customer clicks chat widget on website or app
- **Interaction**: Multi-turn conversation (average 5-8 turns)
- **Session length**: 3-10 minutes per interaction
- **Feedback**: Post-conversation CSAT rating (1-5 stars)
- **Success signal**: Issue resolved, customer confirms satisfaction, no follow-up ticket created
- **Handoff**: Seamless transfer to human agent with full conversation context

**3. Constraints**
- **Budget**: $0.10 average cost per support conversation (current human cost: $5.00)
- **Data sensitivity**: Customer PII (names, emails, addresses, order history, payment info)
- **Data residency**: EU data stays in EU (GDPR), US data in US
- **Vendor**: Prefer Claude (quality), GPT-4o as failover
- **Timeline**: Pilot in 2 months (100 customers), GA in 4 months

**4. Scale Assumptions**
- **Launch (pilot)**: 100 conversations/day
- **GA**: 6,000 conversations/day (60% of 10K tickets)
- **6 months**: 8,000 conversations/day (80% automated resolution)
- **Concurrent sessions**: 200 peak (lunch hour surge)
- **Turns per session**: 5-8 messages average
- **Knowledge base**: 500 FAQ articles, 3 policy documents, product catalog (50K SKUs)
- **Peak ratio**: 5x (post-marketing campaign, holiday season)

**5. Latency Expectations**
- **TTFT**: < 500ms
- **Full response**: < 5s (streaming required)
- **Tool calls** (order lookup, KB search): < 2s each
- **Total turn time**: < 8s (95th percentile)
- **Async operations**: Refund processing (confirmation within 30s, actual refund 1-3 business days)
- **Timeout**: 30s per turn, 15 minutes session idle timeout

**6. Accuracy Requirements**
- **Factual accuracy** (order status, pricing, policy): > 99% (wrong order info is unacceptable)
- **Resolution accuracy**: > 85% of automated resolutions are actually correct (measured by follow-up rate)
- **Tone**: Professional, empathetic, brand-consistent (evaluated by weekly human review of samples)
- **Hallucination**: Zero tolerance for fabricated order details, policies, or promotions
- **Escalation accuracy**: > 95% of escalations are genuinely needed (low false-escalation rate)

**7. Human Involvement**
- **Full autonomy**: Order status lookup, FAQ answers, tracking information
- **Requires approval**: Refunds > $50, address changes, account modifications
- **Always human**: Complaints about discrimination, legal threats, requests for manager
- **Escalation triggers**: Customer asks for human, 3+ failed resolution attempts, sentiment detection (anger)
- **Approval latency**: < 5 minutes (human support agent reviews in real-time queue)

**8. Compliance & Security**
- **Regulations**: GDPR (EU customers), CCPA (California customers), PCI DSS (payment data)
- **PII handling**: Mask credit card numbers before LLM, never store full card numbers
- **LLM provider**: DPA signed, SOC 2 certified, GDPR-compliant processing
- **Data retention**: Conversations retained 90 days, then anonymized for analytics
- **Audit trail**: Every agent action logged (tool calls, decisions, escalations)
- **Right to deletion**: Customer can request deletion via settings page; completed within 30 days

---

### Example 3: Data Analyst Agent

```
SYSTEM: AI Data Analyst Agent for Business Intelligence Team
```

**1. Business Use Case**
- **Problem**: Business stakeholders wait 2-5 days for data team to answer ad-hoc questions
- **User**: Product managers, marketing leads, finance team (non-technical)
- **Success metric**: Self-service resolution of 70% of data questions; average answer time < 5 minutes
- **Failure**: Agent generates incorrect SQL, returns wrong numbers, user makes bad business decision
- **Business value**: 3x data team productivity (focus on complex analysis, not ad-hoc queries)

**2. User Workflow**
- **Start**: Natural language question in Slack bot or web dashboard ("What was GMV last month by region?")
- **Interaction**: Multi-turn (clarify ambiguity → generate query → show results → refine → visualize)
- **Session length**: 2-15 minutes
- **Feedback**: "This looks right" / "This is wrong" + correction
- **Success signal**: User shares the result with their team (proxy for trust)
- **Output format**: Tables, charts (bar, line, pie), summary text, downloadable CSV

**3. Constraints**
- **Budget**: $500/month total LLM cost (shared across 30 users)
- **Data sensitivity**: Revenue data, customer counts, internal KPIs (confidential, not public)
- **Data residency**: US-only processing (all data in US data warehouse)
- **Database access**: Read-only (agent cannot write, update, or delete any data)
- **Technology**: Must query existing Snowflake data warehouse; no data duplication
- **Timeline**: MVP in 6 weeks, iterate based on user feedback

**4. Scale Assumptions**
- **Users**: 30 business stakeholders (launch), 100 (6 months), 300 (12 months)
- **Queries per day**: 50 (launch), 200 (6 months), 500 (12 months)
- **Concurrent sessions**: 10 peak
- **Data warehouse size**: 500 tables, 2TB total, largest table 50M rows
- **Query complexity**: 70% single-table, 25% 2-3 table joins, 5% complex (subqueries, CTEs)
- **Peak ratio**: 3x (Monday morning, month-end reporting)

**5. Latency Expectations**
- **TTFT**: < 1s (show "thinking..." immediately)
- **SQL generation**: < 5s
- **Query execution**: < 30s (Snowflake execution time, not agent's control)
- **Chart generation**: < 3s after data returns
- **Total turn time**: < 45s for simple queries, < 2 minutes for complex
- **Streaming**: Show SQL as it's generated; show "Running query..." during execution
- **Timeout**: 2 minutes for query execution (Snowflake timeout)

**6. Accuracy Requirements**
- **SQL correctness**: > 90% of generated SQL returns correct results (measured against golden queries)
- **Numeric accuracy**: 100% (if the query runs, the numbers must be right -- SQL is deterministic)
- **Schema understanding**: Agent must correctly identify which tables/columns to use > 95% of time
- **Hallucination**: Zero tolerance for fabricated column names, table names, or metric definitions
- **Wrong answer cost**: Medium-high (bad business decisions, but not financial transactions)
- **Confidence signal**: If agent is unsure about table/column mapping, ask user to confirm before running

**7. Human Involvement**
- **Full autonomy**: SELECT queries on any table, chart generation, data export
- **Never autonomous**: No INSERT/UPDATE/DELETE (enforced at DB level, read-only credentials)
- **Optional review**: Complex queries (> 3 joins) can optionally be reviewed by data team before running
- **Escalation**: If agent fails 2x on same question, auto-create Jira ticket for data team

**8. Compliance & Security**
- **Regulations**: SOC 2 (internal requirement), no external customer data involved
- **Access control**: RBAC (finance team sees revenue data, marketing sees campaign data)
- **LLM provider**: Table schemas and column names sent to LLM; actual data values NOT sent to LLM
- **Query results**: Returned directly to user, not stored by agent (except in session cache for 1 hour)
- **Audit trail**: Every query logged (who asked, what SQL ran, when) for compliance
- **PII in data**: Some tables contain customer emails -- agent never returns raw PII, only aggregates

---

## Blank Template

```markdown
# Problem Definition: [System Name]

## 1. Business Use Case
- **Problem**: ___
- **Primary user**: ___
- **Success metric**: ___
- **Failure scenario**: ___
- **Business value**: ___

## 2. User Workflow
- **Start trigger**: ___
- **Interaction type**: One-shot / Multi-turn / Workflow-guided
- **Session length**: ___
- **Feedback mechanism**: ___
- **Success signal**: ___
- **Human handoff**: ___

## 3. Constraints
- **Budget**: $___/month (total), $___/user/month
- **Data sensitivity**: ___
- **Data residency**: ___
- **Vendor requirements**: ___
- **Technology mandates**: ___
- **Timeline**: MVP by ___, GA by ___

## 4. Scale Assumptions
- **Users at launch**: ___
- **Users at 6 months**: ___
- **Users at 12 months**: ___
- **Concurrent sessions**: ___
- **Turns per session**: ___
- **Data volume**: ___
- **Peak-to-average ratio**: ___x

## 5. Latency Expectations
- **Sync or async**: ___
- **TTFT target**: < ___ ms
- **End-to-end turn target**: < ___ s
- **Streaming**: Required / Not required
- **Hard timeout**: ___ s

## 6. Accuracy Requirements
- **Acceptable error rate**: ___% 
- **Cost of wrong answer**: Low / Medium / High / Critical
- **Hallucination tolerance**: Zero / Low / Medium
- **Measurement method**: ___
- **Confidence signaling**: ___

## 7. Human Involvement
- **Autonomy level**: Full / Supervised / Approval-gated / Advisory-only
- **Actions needing approval**: ___
- **Actions always autonomous**: ___
- **Escalation trigger**: ___
- **Approval latency budget**: ___

## 8. Compliance & Security
- **Applicable regulations**: ___
- **LLM provider requirements**: ___
- **PII involved**: ___
- **Data retention policy**: ___
- **Audit trail required**: Yes / No
- **Explainability required**: Yes / No
```

## Decision Tree: How Deep to Go on Problem Definition

```
Is this a system design interview?
│
├── Yes
│   └── Spend 5-8 minutes on this template.
│       Cover sections 1-5 in detail.
│       Mention 6-8 briefly ("I'd clarify accuracy requirements and compliance needs").
│       Then move to architecture.
│
└── No (real project)
    │
    ├── Prototype / hackathon
    │   └── Fill out sections 1-3. Skip the rest. Ship fast.
    │
    ├── Internal tool
    │   └── Fill out sections 1-6. Simplify 7-8 (less compliance for internal).
    │
    └── Production external system
        └── Fill out ALL 8 sections completely. Get stakeholder sign-off.
            This document becomes the source of truth for the project.
```

## When NOT to Use This Template

- **You already have a product spec**: This template supplements a product spec with agent-specific questions. Don't duplicate.
- **Pure research / experimentation**: Just define the task and evaluation criteria.
- **Following a tutorial**: Skip the template. Build the thing, learn from it.

## Tradeoffs

| Section | Time to Define | Impact of Skipping |
|---------|---------------|-------------------|
| Business use case | 30 min | Build the wrong thing entirely |
| User workflow | 1-2 hours | Poor UX, user abandonment |
| Constraints | 1 hour | Budget overrun, compliance violations |
| Scale assumptions | 30 min | Under-provisioned at launch or over-engineered |
| Latency expectations | 30 min | Users leave if too slow, or over-optimize for speed |
| Accuracy requirements | 1-2 hours | Users don't trust the agent, or over-invest in accuracy |
| Human involvement | 1-2 hours | Uncontrolled agent actions or unnecessary bottlenecks |
| Compliance & security | 2-4 hours | Legal risk, failed security audits, costly retrofits |

## Failure Modes

1. **Skipping problem definition entirely**: Jump to architecture → build a beautifully engineered system that solves the wrong problem. Mitigation: mandate this template in project kickoff process.

2. **Vague accuracy requirements**: "It should be accurate" without a number. Mitigation: force a specific percentage and measurement method.

3. **Underestimating scale**: "We'll have 100 users" but marketing plans a launch campaign. Mitigation: ask about growth and peak scenarios explicitly.

4. **Ignoring compliance until late**: Build the system, then discover HIPAA applies and you need to re-architect everything. Mitigation: involve legal/compliance in problem definition.

5. **Overengineering constraints**: "We need active-active multi-region for 50 users." Mitigation: match infrastructure to actual scale, not hypothetical scale.

## Source(s) and Further Reading

- "System Design Interview" - Alex Xu (Chapter 1: Scale Estimation)
- Google DORA Research on Engineering Productivity: https://dora.dev/
- LangChain Agent Design Patterns: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/
- "Designing Machine Learning Systems" - Chip Huyen (Chapter 2: Requirements and Objectives)
- NIST AI Risk Management Framework: https://www.nist.gov/artificial-intelligence/ai-risk-management-framework
- GDPR Official Text: https://gdpr-info.eu/
- SOC 2 Overview: https://www.aicpa.org/soc2
