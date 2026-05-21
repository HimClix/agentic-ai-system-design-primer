# ReAct: Reasoning + Acting

> ReAct interleaves chain-of-thought reasoning with tool actions -- the agent thinks, acts, observes, then thinks again before each step.

## What It Is

ReAct (Reason + Act) is the foundational agent pattern introduced by Yao et al. (2022) at Princeton. It combines:
- **Reasoning**: The LLM produces an explicit thought trace before each action
- **Acting**: The LLM selects and executes a tool
- **Observing**: The tool result is fed back for the next reasoning step

This creates a Thought-Action-Observation loop that mirrors how humans solve problems: think about what to do, do it, look at what happened, think again.

## How It Works

### Step-by-Step Execution

```
User Query: "What is the GDP per capita of the country where the Eiffel Tower is located?"

Step 1 - THOUGHT:  I need to find which country the Eiffel Tower is in.
         ACTION:   search("Eiffel Tower location")
         OBSERVE:  "The Eiffel Tower is in Paris, France."

Step 2 - THOUGHT:  The Eiffel Tower is in France. Now I need France's GDP per capita.
         ACTION:   search("France GDP per capita 2024")
         OBSERVE:  "France's GDP per capita is approximately $44,408 (2024)."

Step 3 - THOUGHT:  I have the answer. France's GDP per capita is $44,408.
         ACTION:   finish("The GDP per capita of France, where the Eiffel Tower
                   is located, is approximately $44,408 as of 2024.")
```

### Architecture

```
                         +-------------------+
                         |   User Query      |
                         +--------+----------+
                                  |
                    +-------------v-----------+
                    |  System Prompt:          |
                    |  "Use Thought/Action/    |
                    |   Observation format"    |
                    +-------------+-----------+
                                  |
                    +-------------v-----------+
              +---->|  LLM generates:         |
              |     |  Thought: <reasoning>   |
              |     |  Action: <tool + args>  |
              |     +-------------+-----------+
              |                   |
              |     +-------------v-----------+
              |     |  Execute Tool            |
              |     |  (search, calculator,    |
              |     |   API call, etc.)        |
              |     +-------------+-----------+
              |                   |
              |     +-------------v-----------+
              |     |  Observation: <result>   |
              +-----+  Append to history       |
                    +-------------+-----------+
                                  |
                         +--------v---------+
                         | Final Answer     |
                         +------------------+
```

## Production Implementation

### Using LangGraph (Production-Grade)

```python
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
import operator

# -- State Definition --
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    iteration_count: int

# -- Tools --
from langchain_community.tools import TavilySearchResults

search_tool = TavilySearchResults(max_results=3)
tools = [search_tool]

# -- LLM with tools bound --
llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0,
    max_tokens=4096
).bind_tools(tools)

# -- Agent Node --
MAX_ITERATIONS = 10

def agent_node(state: AgentState) -> dict:
    """The reasoning node -- LLM decides what to do next."""
    if state["iteration_count"] >= MAX_ITERATIONS:
        return {
            "messages": [AIMessage(content="Max iterations reached. Returning best answer.")],
            "iteration_count": state["iteration_count"]
        }

    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "iteration_count": state["iteration_count"] + 1
    }

# -- Routing Logic --
def should_continue(state: AgentState) -> str:
    """Decide: call tools or return final answer."""
    last_message = state["messages"][-1]

    if state["iteration_count"] >= MAX_ITERATIONS:
        return "end"

    if last_message.tool_calls:
        return "tools"

    return "end"

# -- Build Graph --
tool_node = ToolNode(tools)

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "end": END
})
graph.add_edge("tools", "agent")  # After tools, go back to agent

app = graph.compile()

# -- Run --
result = app.invoke({
    "messages": [HumanMessage(content="What is the GDP per capita of the Eiffel Tower's country?")],
    "iteration_count": 0
})
```

### Minimal ReAct Without Frameworks

```python
import json
from anthropic import Anthropic

client = Anthropic()

TOOLS = [
    {
        "name": "search",
        "description": "Search the web for information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
]

def run_react_agent(user_query: str, max_iterations: int = 10) -> str:
    messages = [{"role": "user", "content": user_query}]

    for i in range(max_iterations):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system="You are a helpful assistant. Think step by step before acting.",
            tools=TOOLS,
            messages=messages
        )

        # Check if LLM wants to use a tool
        if response.stop_reason == "tool_use":
            # Extract tool call
            tool_block = next(b for b in response.content if b.type == "tool_use")

            # Execute the tool
            tool_result = execute_tool(tool_block.name, tool_block.input)

            # Append assistant response + tool result
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": tool_result
                }]
            })
        else:
            # LLM returned final answer
            text_block = next(b for b in response.content if b.type == "text")
            return text_block.text

    raise RuntimeError(f"Agent did not converge in {max_iterations} iterations")
```

## Token Cost Analysis

ReAct is the most expensive pattern because the full conversation history is sent with every LLM call:

```
Assume: System prompt = 500 tokens, User query = 100 tokens
        Each tool call = 200 tokens, Each tool result = 500 tokens
        Each thought = 150 tokens

Iteration 1 INPUT:  500 + 100                                    =    600 tokens
Iteration 2 INPUT:  500 + 100 + 150 + 200 + 500                  =  1,450 tokens
Iteration 3 INPUT:  500 + 100 + 150 + 200 + 500 + 150 + 200 + 500 = 2,300 tokens
Iteration 4 INPUT:  ...                                           =  3,150 tokens
Iteration 5 INPUT:  ...                                           =  4,000 tokens

TOTAL INPUT TOKENS (5 iterations): ~11,500
TOTAL OUTPUT TOKENS: ~1,750 (thoughts + actions + final)

With Claude Sonnet at $3/M input, $15/M output:
  Input cost:  11,500 / 1M * $3  = $0.0345
  Output cost:  1,750 / 1M * $15 = $0.0263
  TOTAL PER QUERY: ~$0.06

At 10,000 queries/day: $600/day = $18,000/month
```

Compare this to Plan-and-Execute which achieves ~70% savings on the same tasks.

## Decision Tree: When to Use ReAct

```
         Should I use ReAct?
                |
     +----------v----------+
     | Is the task          |
     | exploratory?         |
     | (can't know steps    |
     |  in advance)         |
     +---+------------+----+
         |            |
        Yes          No --> Consider Plan-and-Execute
         |
     +---v-----------+----+
     | Are steps          |
     | sequential?        |
     | (each depends on   |
     |  prior result)     |
     +---+------------+---+
         |            |
        Yes          No --> Consider ReWOO (parallel)
         |
     +---v-----------+----+
     | Can you afford     |
     | higher token cost? |
     +---+------------+---+
         |            |
        Yes          No --> Consider Plan-and-Execute
         |
     +---v-----------+
     | USE ReAct     |
     +---------------+
```

## When NOT to Use

1. **Predictable multi-step tasks**: If you know the steps upfront, Plan-and-Execute saves ~70% on tokens
2. **Independent parallel lookups**: ReWOO is cheaper and faster
3. **Cost-sensitive production**: ReAct's token accumulation makes it expensive at scale
4. **Tasks requiring >10 steps**: Context window fills up; Plan-and-Execute handles better
5. **Batch processing**: Token costs multiply; use simpler patterns

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Most flexible -- handles any task | Highest token cost per query |
| Easy to implement and debug | Sequential execution (high latency) |
| Explicit reasoning trace (good for auditing) | Context grows linearly with steps |
| Works with any tool combination | Non-deterministic execution paths |
| Well-supported by all frameworks | Can get stuck in loops |
| Widely understood pattern | Overkill for simple/predictable tasks |

## Real-World Examples

1. **Customer support agent**: Looks up customer account, checks order status, queries knowledge base, composes response -- each step depends on prior results.

2. **Research assistant**: Searches for papers, reads abstracts, follows citations, synthesizes findings. Exploratory by nature.

3. **DevOps troubleshooting**: Checks logs, queries metrics, inspects configs -- investigation path depends on what each step reveals.

4. **Code debugging agent**: Reads error, searches codebase, reads related files, proposes fix. Cannot know which files to read without first seeing the error.

## Failure Modes

### 1. Infinite Loops
The agent calls the same tool with the same arguments repeatedly:
```
THOUGHT: I need to search for the population.
ACTION: search("France population")
OBSERVE: "67 million"
THOUGHT: Let me verify this.
ACTION: search("France population")  <-- SAME QUERY
OBSERVE: "67 million"
THOUGHT: Let me double-check...
```
**Mitigation**: Track (tool_name, args) pairs; reject duplicates after 2 attempts.

### 2. Tool Hallucination
Agent invokes a tool that doesn't exist:
```
ACTION: calculate_gdp(country="France")  <-- No such tool
```
**Mitigation**: Strict tool validation before execution. All frameworks handle this.

### 3. Reasoning Drift
Agent's thoughts gradually drift from the original question:
```
THOUGHT: While looking up France's GDP, I notice Paris has many museums...
ACTION: search("best museums in Paris")  <-- Off-topic
```
**Mitigation**: Include the original query in every system prompt iteration.

### 4. Context Window Overflow
After many iterations, the full history exceeds the context window:
**Mitigation**: Implement context summarization after N steps, or cap iterations.

### 5. Premature Termination
Agent returns an answer before completing all required steps, especially when the question has multiple parts.
**Mitigation**: Use structured output to verify all sub-questions are answered.

## Sources and Further Reading

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (Yao et al., ICLR 2023)
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents)
- [LangGraph ReAct Agent Tutorial](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [Tool Use with Claude - Anthropic Docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
