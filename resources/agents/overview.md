# Agents: Category Overview

> An agent is an LLM that runs in a loop, using tool results to decide its next action, until a stopping condition is met.

## What Is an Agent?

An agent is fundamentally different from a chain or a pipeline. In a chain, the control flow is predetermined by the developer. In an agent, the LLM itself decides the control flow at runtime based on the results of its actions.

Anthropic's "Building Effective Agents" (2024) defines it precisely:

- **Workflow**: Predetermined code paths where LLMs and tools are orchestrated through fixed logic (if/else, routing, parallel calls).
- **Agent**: An autonomous system where the LLM dynamically decides control flow -- which tools to call, in what order, and when to stop.

```
 WORKFLOW (Developer-controlled)              AGENT (LLM-controlled)
 +---------+    +--------+    +------+        +------+
 | Input   |--->| LLM 1  |--->| LLM 2|--->Out | Input|
 +---------+    +--------+    +------+        +--+---+
                                                 |
                                              +--v---+
                                              | LLM  |<---+
                                              +--+---+    |
                                                 |        |
                                            +----v---+    |
                                            | Tool   |----+
                                            | Result |
                                            +--------+
                                                 |
                                            Stop condition?
                                              Yes -> Output
```

## The Agent Execution Loop

Every agent, regardless of framework or pattern, runs the same fundamental loop:

```
                    +---------------------------+
                    |   Receive Task / Input     |
                    +------------+--------------+
                                 |
                    +------------v--------------+
                    |   Construct Prompt         |
                    |   (system + history +      |
                    |    tools + user message)   |
                    +------------+--------------+
                                 |
                    +------------v--------------+
                    |   Call LLM                 |
                    +------------+--------------+
                                 |
                         +-------v-------+
                         | Tool call?    |
                         +---+-------+---+
                             |       |
                          Yes|       |No
                             |       |
                    +--------v---+   +-------v--------+
                    | Execute    |   | Return final   |
                    | tool(s)    |   | response       |
                    +--------+---+   +----------------+
                             |
                    +--------v--------+
                    | Append result   |
                    | to conversation |
                    +--------+--------+
                             |
                             +---> (back to Call LLM)
```

Key properties of this loop:
1. **Stateful**: Each iteration appends to the conversation history
2. **Token-accumulating**: Cost grows with each iteration (context gets larger)
3. **Non-deterministic**: Same input can produce different tool sequences
4. **Terminates when**: LLM returns text without a tool call, or max iterations reached

## Workflow vs Agent: Decision Tree

Before building an agent, ask whether you actually need one. Anthropic's recommendation: start with the simplest solution.

```
                        START
                          |
              +-----------v-----------+
              | Is the task           |
              | deterministic?        |
              | (same input -> same   |
              |  steps every time)    |
              +---+-------------+-----+
                  |             |
                 Yes           No
                  |             |
         +--------v-------+    |
         | Use a WORKFLOW  |    |
         | (chain/router/ |    |
         |  pipeline)     |    |
         +----------------+    |
                               |
              +----------------v---------+
              | Does the task require    |
              | dynamic tool selection?  |
              | (LLM must decide WHICH   |
              |  tools and in what order)|
              +---+-------------+--------+
                  |             |
                 No            Yes
                  |             |
         +--------v-------+    |
         | Use a WORKFLOW  |    |
         | with routing   |    |
         +----------------+    |
                               |
              +----------------v---------+
              | Does the task need       |
              | multi-step reasoning     |
              | with intermediate tool   |
              | results informing next   |
              | actions?                 |
              +---+-------------+--------+
                  |             |
                 No            Yes
                  |             |
         +--------v-------+ +--v--------------+
         | Use a single   | | Use an AGENT    |
         | LLM call with  | | (pick pattern   |
         | tool use       | |  from patterns/ |
         +----------------+ |  decision-tree) |
                             +-----------------+
```

## Single Agent vs Multi-Agent Decision

```
              AGENT NEEDED (from above)
                          |
              +-----------v-----------+
              | Can ONE agent handle  |
              | the full task with    |
              | available tools?      |
              +---+-------------+-----+
                  |             |
                 Yes           No
                  |             |
    +-------------v---------+  |
    | SINGLE AGENT          |  |
    | Pick pattern:         |  |
    | - ReAct (flexible)    |  |
    | - Plan-and-Execute    |  |
    |   (complex, cheaper)  |  |
    | - ReWOO (parallel)    |  |
    | - Reflexion (needs    |  |
    |   self-correction)    |  |
    +-----------------------+  |
                               |
              +----------------v---------+
              | Are sub-tasks            |
              | INDEPENDENT of each      |
              | other?                   |
              +---+-------------+--------+
                  |             |
                 Yes           No
                  |             |
    +-------------v---------+  +----------v----------+
    | PARALLEL FAN-OUT      |  | Do you need >8      |
    | (map-reduce pattern)  |  | specialist agents?  |
    +-----------------------+  +---+----------+------+
                                   |          |
                                  No         Yes
                                   |          |
                      +------------v--+  +----v-----------+
                      | SUPERVISOR    |  | HIERARCHICAL   |
                      | (one router)  |  | (multi-level)  |
                      +---------------+  +----------------+
```

## The Five Workflow Patterns (Before You Need Agents)

From Anthropic's classification, use these before reaching for a full agent:

| Pattern | Description | When |
|---------|-------------|------|
| **Prompt Chaining** | Output of LLM 1 feeds into LLM 2 | Sequential tasks with validation gates |
| **Routing** | Classifier sends input to specialized handler | Different input types need different processing |
| **Parallelization** | Same input processed by multiple LLMs simultaneously | Voting, multi-perspective analysis |
| **Orchestrator-Workers** | Central LLM delegates to worker LLMs | Dynamic sub-task decomposition |
| **Evaluator-Optimizer** | One LLM generates, another evaluates and iterates | Quality-critical outputs (code, writing) |

## Key Agent Patterns (Deep Dives in patterns/ directory)

| Pattern | Token Cost | Latency | Best For |
|---------|-----------|---------|----------|
| **ReAct** | High (full context per step) | High (sequential) | Flexible, exploratory tasks |
| **Plan-and-Execute** | ~70% less than ReAct | Medium (parallel executors) | Complex multi-step tasks |
| **ReWOO** | Lowest | Lowest (all parallel) | Independent lookups |
| **Reflexion** | 2-3x overhead | Highest | Tasks needing self-correction |

## Production Considerations

### Token Cost Growth

Agent loops are token-expensive because context accumulates:

```
Iteration 1: system + tools + user msg           =  2,000 tokens
Iteration 2: + tool_call + tool_result            =  4,500 tokens
Iteration 3: + tool_call + tool_result            =  7,000 tokens
Iteration 4: + tool_call + tool_result            =  9,500 tokens
Iteration 5: + tool_call + tool_result + response = 12,000 tokens

Total input tokens billed: 2000 + 4500 + 7000 + 9500 + 12000 = 35,000
(Not 12,000 -- you pay for the growing context EACH iteration)
```

### Safety: Max Iterations

Always set a maximum iteration count. Without it, a confused agent will loop forever:

```python
MAX_ITERATIONS = 10  # Hard ceiling

for i in range(MAX_ITERATIONS):
    response = llm.invoke(messages)
    if response.tool_calls:
        # Execute tools, append results
        messages.append(response)
        for tc in response.tool_calls:
            result = execute_tool(tc)
            messages.append(ToolMessage(content=result, tool_call_id=tc.id))
    else:
        return response.content  # Agent chose to stop

raise AgentMaxIterationsError("Agent did not converge")
```

### Observability

Every production agent needs:
1. **Trace IDs**: Track the full execution path
2. **Step logging**: Record each LLM call, tool call, and result
3. **Token counting**: Per-step and cumulative
4. **Latency tracking**: Per-step and end-to-end
5. **Error classification**: Tool failures vs LLM hallucinations vs timeout

## Failure Modes (Common to All Agents)

From the MAST taxonomy (NeurIPS 2025, 1,642 agent traces analyzed):

1. **Infinite loops**: Agent calls the same tool repeatedly with same args
2. **Tool hallucination**: Agent invokes a tool that does not exist
3. **Premature termination**: Agent returns before completing the task
4. **Context window overflow**: Accumulated history exceeds model limit
5. **Error cascading**: One bad tool result poisons all subsequent reasoning

See `../failure-modes/` for the complete MAST taxonomy coverage.

## Directory Structure

```
agents/
  overview.md                          <-- You are here
  patterns/
    react.md                           # ReAct pattern deep dive
    plan-and-execute.md                # Plan-then-Execute pattern
    reflexion.md                       # Self-correction loop
    rewoo.md                           # Reasoning Without Observation
    decision-tree.md                   # How to pick a pattern
  multi-agent/
    supervisor.md                      # Supervisor pattern
    hierarchical.md                    # Multi-level supervision
    parallel-fan-out.md                # Map-reduce for agents
    when-to-go-multi-agent.md          # Decision framework
  frameworks/
    langgraph.md                       # LangGraph deep dive
    crewai.md                          # CrewAI framework
    autogen.md                         # Microsoft AutoGen
    llamaindex.md                      # LlamaIndex agents
    pydantic-ai.md                     # Pydantic AI
    dspy.md                            # DSPy
    comparison.md                      # Framework comparison
  sdks/
    claude-agent-sdk.md                # Anthropic Agent SDK
    openai-agents-sdk.md               # OpenAI Agents SDK
    google-adk.md                      # Google ADK
```

## Sources and Further Reading

- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents) (Dec 2024)
- [MAST: A Multi-Agent Systems Taxonomy for Evaluating Agentic LLM Systems](https://arxiv.org/abs/2502.xxxxx) (NeurIPS 2025)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (Yao et al., 2022)
- [Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091) (Wang et al., 2023)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) (Shinn et al., 2023)
- [ReWOO: Decoupling Reasoning from Observations](https://arxiv.org/abs/2305.18323) (Xu et al., 2023)
