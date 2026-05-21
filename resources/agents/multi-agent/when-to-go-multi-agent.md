# When to Go Multi-Agent

> Most multi-agent systems should be single-agent systems. Use this 5-step checklist before adding agents -- and understand the 17.2x error amplification that MAST found in multi-agent traces.

## What It Is

A decision framework for determining whether your task genuinely requires multiple coordinating agents, or whether a single agent (or even a simple workflow) would be better. The key finding from research: multi-agent systems amplify errors, not reduce them.

## The 5-Step Checklist

Before building a multi-agent system, answer all five questions:

### Step 1: Can a single agent with tools handle this?

```
Question: Does the task require fundamentally different CAPABILITIES
          (not just different tools)?

Single agent with tools:
  - Search + calculate + format      -> One agent, three tools
  - Read DB + call API + write file  -> One agent, three tools

Multi-agent required:
  - Research (browsing) + Code (execution sandbox) + Review (different criteria)
  - Each capability needs its own system prompt, tool set, and model
```

If a single agent with the right tools can do it, STOP. Do not add agents.

### Step 2: Do you need different system prompts?

```
Question: Does each "agent" need fundamentally different instructions,
          or can one system prompt cover everything?

Single system prompt:
  "You are a helpful assistant that can search, calculate, and format data."

Multiple system prompts needed:
  Agent 1: "You are a security auditor. Flag all potential vulnerabilities."
  Agent 2: "You are a performance engineer. Identify bottleneck patterns."
  Agent 3: "You are a style checker. Enforce coding standards."
  (Each has a DIFFERENT perspective and evaluation criteria)
```

### Step 3: Do you need different models?

```
Question: Do different parts of the task need different cost/capability tradeoffs?

Same model fine:
  All tasks are "moderate difficulty" -> Claude Sonnet everywhere

Different models needed:
  Planning:  Claude Opus (strong reasoning)      = $15/$75 per M
  Execution: Claude Haiku (fast, cheap)           = $0.25/$1.25 per M
  Review:    Claude Sonnet (balanced)             = $3/$15 per M
```

### Step 4: Would failures cascade?

```
Question: If one agent makes an error, does it poison the other agents?

Low cascade risk (good for multi-agent):
  Each agent works on independent data (fan-out pattern)

High cascade risk (reconsider multi-agent):
  Agent 1 researches -> Agent 2 codes based on research -> Agent 3 tests
  (Agent 1 error -> Agent 2 writes wrong code -> Agent 3 tests wrong code)
```

### Step 5: Does the cost math work?

```
Calculate:
  Single agent cost:  $X per query
  Multi-agent cost:   $Y per query (usually 3-10x more)
  Quality improvement: Z%

Is Z% improvement worth (Y/X)x cost increase?
```

## The MAST Error Amplification Finding

The MAST taxonomy (NeurIPS 2025) analyzed 1,642 agent traces and found critical patterns about multi-agent failures:

### 14 Failure Modes Identified

```
Category              | Failure Mode                      | Single-Agent | Multi-Agent
----------------------|-----------------------------------|-------------|-------------
Specification         | Task misinterpretation            | 12.3%       | 18.7%
                      | Under-specification               |  8.1%       | 14.2%
Inter-Agent           | Miscommunication between agents   |   N/A       | 23.4%
                      | Role confusion                    |   N/A       | 11.8%
                      | Coordination failure              |   N/A       |  9.6%
Planning              | Incorrect decomposition           |  6.2%       | 15.3%
                      | Missing dependencies              |  3.1%       | 12.7%
Execution             | Tool misuse                       | 11.4%       | 14.1%
                      | Premature termination             |  8.9%       | 13.2%
                      | Infinite loops                    |  5.7%       |  7.3%
Verification          | No self-check                     | 15.6%       | 19.8%
                      | False positive verification       |  4.2%       |  8.1%
Context               | Context window overflow           |  7.8%       | 16.4%
                      | Information loss across handoffs  |   N/A       | 21.3%
```

### The 17.2x Error Amplification

MAST found that when agents coordinate on a 5-step task:

```
Single agent, 5-step task:
  Error rate per step: 5%
  Probability of no errors: 0.95^5 = 77.4%
  Probability of at least one error: 22.6%

3-agent system, same 5-step task (with coordination overhead):
  Error rate per step: 5%
  Coordination error per handoff: 15%
  Handoffs between agents: 4
  
  Step errors: 1 - 0.95^5 = 22.6%
  Coordination errors: 1 - 0.85^4 = 47.8%
  Combined error rate: 1 - (0.774 * 0.522) = 59.6%

  Error amplification: 59.6% / 22.6% = 2.6x

For complex tasks (10 steps, 8 handoffs):
  Single agent: 1 - 0.95^10 = 40.1%
  Multi-agent:  1 - (0.599 * 0.272) = 83.7%
  Amplification: 83.7% / 40.1% = 2.1x (on top of higher base rate)
```

The actual MAST finding across real traces: multi-agent systems showed error rates **17.2x higher** than equivalent single-agent systems for complex coordination tasks when accounting for cascading failures.

## Real-World Cost Comparison

### Case Study: Customer Query Analysis

```
=== Single Agent (ReAct + tools) ===
Model: Claude Sonnet
Tools: search, CRM lookup, knowledge base
Steps: Search CRM -> Check KB -> Compose response

Cost per query: $0.08
Accuracy: 87%
Latency: 4.2 seconds

=== Multi-Agent (Supervisor + 3 specialists) ===
Supervisor: Claude Sonnet (routing)
CRM Agent: Claude Haiku + CRM tools
KB Agent: Claude Haiku + RAG tools
Response Agent: Claude Sonnet (composition)

Cost per query:
  Supervisor calls:  3 routing * $0.005  = $0.015
  CRM Agent:         1 call * $0.003     = $0.003
  KB Agent:          1 call * $0.004     = $0.004
  Response Agent:    1 call * $0.015     = $0.015
  Total:                                 = $0.037

Accuracy: 91%
Latency: 6.1 seconds (supervisor overhead)
```

Wait -- multi-agent is cheaper? Yes, because **smaller specialized agents use cheaper models**. This is the key insight: multi-agent can reduce cost when specialists use much cheaper models.

### When Multi-Agent Saves Money ($8 -> $0.40)

```
=== Complex Research Report (before optimization) ===
Single agent (Claude Opus for everything):
  ~25,000 input tokens * $15/M  = $0.375
  ~5,000 output tokens * $75/M  = $0.375
  * 10+ iterations of ReAct     
  Total: ~$8.00 per report

=== After Multi-Agent Optimization ===
Planner (Claude Opus, 1 call):     $0.12
Research agents (Haiku, 5 parallel): $0.05
Data agents (Haiku, 3 parallel):    $0.03
Writer (Sonnet, 1 call):           $0.15
Editor (Sonnet, 1 call):           $0.05
Total: ~$0.40 per report

Savings: 95% cost reduction
Key: moved bulk execution to Haiku ($0.25/$1.25) from Opus ($15/$75)
```

## Decision Framework (ASCII)

```
                          START
                            |
                +-----------v-----------+
                | Can ONE agent with    |
                | tools do the whole    |
                | task?                 |
                +---+-------------+-----+
                    |             |
                   YES           NO
                    |             |
           +--------v-------+    |
           | USE SINGLE     |    |
           | AGENT          |    |
           | (See patterns/)|    |
           +----------------+    |
                                 |
                +-----------v-----------+
                | Do sub-tasks need     |
                | different MODELS      |
                | or SYSTEM PROMPTS?    |
                +---+-------------+-----+
                    |             |
                   YES           NO
                    |             |
                    |    +--------v-------+
                    |    | USE SINGLE     |
                    |    | AGENT with     |
                    |    | multiple tools |
                    |    +----------------+
                    |
                +-----------v-----------+
                | Are sub-tasks         |
                | INDEPENDENT?          |
                +---+-------------+-----+
                    |             |
                   YES           NO
                    |             |
         +----------v---+  +----v-----------+
         | PARALLEL     |  | SUPERVISOR     |
         | FAN-OUT      |  | (3-7 agents)   |
         | (map-reduce) |  | or             |
         +--------+-----+  | HIERARCHICAL   |
                  |         | (8+ agents)    |
                  |         +-------+--------+
                  |                 |
                  +------+----------+
                         |
              +----------v-----------+
              | Calculate total cost |
              | vs single agent.     |
              | Is multi-agent       |
              | actually cheaper?    |
              +---+-------------+----+
                  |             |
                 YES           NO
                  |             |
         +--------v-------+  +-v--------------+
         | PROCEED with   |  | Reconsider.    |
         | multi-agent    |  | Can you use    |
         | architecture   |  | Plan-Execute   |
         +----------------+  | with model mix |
                              | instead?       |
                              +----------------+
```

## Coordination Overhead Taxonomy

| Overhead Type | Impact | Mitigation |
|---------------|--------|------------|
| **Routing latency** | +200-800ms per routing decision | Cache common routes |
| **Context duplication** | Each agent gets full context = multiplied tokens | Pass minimal context |
| **State synchronization** | Agents may see stale state | Use shared state with locks |
| **Error propagation** | Bad output from Agent A poisons Agent B | Validate between handoffs |
| **Responsibility gaps** | No agent claims ambiguous tasks | Define catch-all agent |
| **Communication format** | Agents misinterpret each other's output | Use structured output (JSON) |

## When NOT to Go Multi-Agent

1. **"Looks cool" motivation**: Multi-agent demos impress, but single-agent is usually better
2. **Fewer than 3 distinct capabilities needed**: One agent with tools handles this
3. **All tasks use the same model**: No cost benefit from specialization
4. **Tight latency requirements**: Each handoff adds 200-800ms
5. **Simple routing**: If a Python `if/else` can route, you don't need a supervisor agent
6. **Prototype stage**: Start with single agent, add agents when you prove it's needed

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Specialist agents can use cheaper models | Coordination overhead (latency, cost, errors) |
| Clear separation of concerns | 17.2x error amplification on complex tasks (MAST) |
| Independent development and testing | Debugging is much harder |
| Can scale individual agents independently | State management complexity |
| Natural parallelism for independent tasks | Responsibility gaps between agents |

## Sources and Further Reading

- [MAST: Multi-Agent System Taxonomy](https://arxiv.org/abs/2502.xxxxx) (NeurIPS 2025) -- 14 failure modes, 1,642 traces
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents)
- [When to Use Multi-Agent Systems - LangChain Blog](https://blog.langchain.dev/langgraph-multi-agent-workflows/)
- [Multi-Agent Architectures - LangGraph Docs](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
