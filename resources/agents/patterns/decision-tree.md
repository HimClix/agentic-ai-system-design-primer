# Agent Pattern Decision Tree

> This is the most important file in the patterns directory. When an interviewer asks "how do you pick an agent pattern?" -- this is your answer.

## What It Is

A structured decision framework for choosing the right agent execution pattern based on task characteristics, cost constraints, latency requirements, and self-correction needs. Getting this wrong means either overpaying (ReAct for simple tasks) or under-delivering (ReWOO for sequential tasks).

## The Decision Flowchart

```
                            START: You need an agent
                                    |
                    +---------------v----------------+
                    | Can you define ALL steps       |
                    | before seeing any results?     |
                    +------+------------------+------+
                           |                  |
                          YES                 NO
                           |                  |
              +------------v-----------+      |
              | Are ALL steps          |      |
              | independent of         |      |
              | each other?            |      |
              +-----+----------+------+      |
                    |          |              |
                   YES         NO             |
                    |          |              |
            +-------v---+  +--v----------+   |
            |  ReWOO    |  | Plan-and-   |   |
            |           |  | Execute     |   |
            | Cheapest  |  |             |   |
            | Fastest   |  | ~70% less   |   |
            | Parallel  |  | than ReAct  |   |
            +-----------+  +------+------+   |
                                  |          |
                                  |   +------v-----------+
                                  |   | Does the task    |
                                  |   | require self-    |
                                  |   | correction?      |
                                  |   | (code, math,     |
                                  |   |  verifiable)     |
                                  |   +----+--------+----+
                                  |        |        |
                                  |       YES       NO
                                  |        |        |
                                  | +------v-----+  |
                                  | | Reflexion  |  |
                                  | | + ReAct    |  |
                                  | | 2-3x cost  |  |
                                  | | but higher |  |
                                  | | accuracy   |  |
                                  | +------------+  |
                                  |                 |
                                  |          +------v--------+
                                  |          |  ReAct        |
                                  |          |  (default)    |
                                  |          |  Most flexible|
                                  |          |  Highest cost |
                                  |          +---------------+
                                  |
                           +------v-----------+
                           | Does the task    |
                           | need self-       |
                           | correction too?  |
                           +----+--------+----+
                                |        |
                               YES       NO
                                |        |
                    +-----------v---+ +--v-----------+
                    | Reflexion +   | | Plan-and-    |
                    | Plan-Execute  | | Execute      |
                    | (rare combo)  | | (standard)   |
                    +---------------+ +--------------+
```

## Master Comparison Table

This is the table to memorize for interviews:

| Dimension | ReAct | Plan-and-Execute | Reflexion | ReWOO |
|-----------|-------|-------------------|-----------|-------|
| **LLM Calls** | N (grows with steps) | 1 plan + N exec + replans | 2-3x base pattern | 2 (fixed) |
| **Token Cost** | Highest (context accumulates) | ~30% of ReAct | 2-3x base pattern | Lowest (~20% of ReAct) |
| **Latency** | High (sequential) | Medium (executor can be fast) | Highest (multiple passes) | Lowest (parallel) |
| **Flexibility** | Highest | Medium | High (retries adapt) | Lowest |
| **Self-Correction** | None built-in | Via re-planner | Core feature | None |
| **Parallelism** | None | Executor steps can parallel | Each attempt is serial | Full parallelism |
| **Context Growth** | Linear with steps | Fixed per executor | Linear per attempt | Fixed |
| **Best Model Mix** | One model | Strong plan + cheap exec | Strong eval + any actor | One model |
| **Max Steps** | ~10 before context overflow | 20+ (independent contexts) | 2-3 retries | Unlimited tools |
| **Determinism** | Low | Medium | Medium | High (once planned) |

## Cost Comparison with Real Numbers

Assume: 5-step task, Claude Sonnet ($3/$15 per M tokens), ~500 tokens/step

```
                     ReAct         Plan-Execute    ReWOO
                     -----         ------------    -----
Input tokens:        11,500        3,800           1,400
Output tokens:        1,750          850             500
Cost per query:      $0.061        $0.024          $0.012
10K queries/day:     $610          $240            $120
Monthly (30 days):   $18,300       $7,200          $3,600

Savings vs ReAct:      --          61%             80%
```

For Reflexion, multiply the base pattern cost by 2-3x:
```
Reflexion + ReAct:   $0.061 * 2.5 = $0.153/query = $45,750/month
Reflexion + P&E:     $0.024 * 2.5 = $0.060/query = $18,000/month
```

## Decision Matrix by Task Type

| Task Type | Recommended Pattern | Why |
|-----------|-------------------|-----|
| Customer support (lookup + respond) | ReAct | Sequential, depends on what's found |
| Multi-entity comparison | ReWOO | All lookups are independent |
| Report generation | Plan-and-Execute | Clear structure, parallel research |
| Code generation | Reflexion + ReAct | Tests provide clear evaluation criteria |
| SQL generation | Reflexion + P&E | Schema validation is deterministic |
| Research synthesis | Plan-and-Execute | Plan sections, research independently |
| DevOps debugging | ReAct | Truly exploratory, next step depends on findings |
| Data pipeline building | Plan-and-Execute | Clear steps, test at each stage |
| Creative writing | ReAct or single LLM call | No clear evaluation criteria for Reflexion |
| API orchestration | ReWOO | Independent API calls |
| Mathematical proofs | Reflexion | Verifiable correctness |
| Travel planning | Plan-and-Execute | Plan components, research independently |

## The Five Questions to Ask

When deciding on a pattern, ask these in order:

### 1. "Can I predict the steps before execution?"
- **Yes** -> Plan-and-Execute or ReWOO
- **No** -> ReAct

### 2. "Are the steps independent?"
- **Yes** -> ReWOO (if steps are predictable) or Parallel Fan-out (multi-agent)
- **No** -> Plan-and-Execute or ReAct

### 3. "Is there a verifiable success criterion?"
- **Yes, deterministic** (tests, schema) -> Add Reflexion
- **Yes, qualitative** (rubric) -> Consider Reflexion with LLM evaluator
- **No** -> Skip Reflexion

### 4. "What's my cost tolerance?"
- **Low** -> ReWOO > Plan-Execute > ReAct (never Reflexion)
- **Medium** -> Plan-Execute (sweet spot)
- **High** -> Whatever fits best

### 5. "What's my latency tolerance?"
- **<2 seconds** -> ReWOO (parallel) or single LLM call (no agent)
- **2-10 seconds** -> Plan-Execute or ReAct with few steps
- **>10 seconds** -> Any pattern works

## Hybrid Patterns

In production, patterns are often combined:

### ReAct + Reflexion
Most common hybrid. Use ReAct for exploration, Reflexion for quality assurance.
```
[ReAct agent generates code] -> [Test evaluator] -> [Reflect] -> [ReAct retry]
```

### Plan-and-Execute + Reflexion
Plan once, execute with retry loops per step.
```
[Plan] -> [Execute Step 1 with Reflexion] -> [Execute Step 2 with Reflexion] -> ...
```

### Plan-and-Execute + ReWOO (per batch)
Plan the workflow, but within each phase, use ReWOO for parallel lookups.
```
[Plan: Phase 1 = gather data, Phase 2 = analyze, Phase 3 = synthesize]
  Phase 1: ReWOO (parallel data gathering)
  Phase 2: ReAct (sequential analysis)
  Phase 3: Single LLM call
```

## Anti-Pattern: Using the Wrong Pattern

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| ReAct for independent lookups | 5x cost overhead | Switch to ReWOO |
| ReWOO for sequential tasks | Wrong results (missing dependencies) | Switch to Plan-Execute or ReAct |
| Reflexion without eval criteria | Wasted 2-3x tokens with no accuracy gain | Remove Reflexion loop |
| ReAct for >10 step tasks | Context overflow, degraded reasoning | Switch to Plan-Execute |
| Plan-Execute for 2-step tasks | Overhead of planning exceeds savings | Use simple ReAct |
| Any agent for deterministic tasks | Unnecessary cost and latency | Use a workflow (no agent) |

## Interview Script

When asked "How do you choose an agent pattern?":

> "I start with the simplest approach that could work -- Anthropic's guidance is to avoid agents entirely if a workflow suffices. If I do need an agent, I ask three questions: First, can I predict the steps upfront? If yes, Plan-and-Execute saves ~70% on tokens. Second, are the steps independent? If yes, ReWOO gives me parallelism and the lowest cost. Third, do I need self-correction? If yes and I have a verifiable criterion like tests or schema validation, I add a Reflexion loop capped at 2-3 retries. ReAct is my default only when the task is truly exploratory -- where each step depends on the previous result and I cannot predict the path ahead."

## Sources and Further Reading

- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents)
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) (Yao et al., 2022)
- [Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091) (Wang et al., 2023)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) (Shinn et al., 2023)
- [ReWOO: Decoupling Reasoning from Observations](https://arxiv.org/abs/2305.18323) (Xu et al., 2023)
- [LangGraph Agent Architectures](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/)
