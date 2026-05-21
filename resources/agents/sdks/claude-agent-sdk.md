# Claude Agent SDK (Anthropic)

> Anthropic's official Agent SDK gives you the tool loop, subagents, permissions, and MCP integration that sit between the raw Messages API and full frameworks like LangGraph.

## What It Is

The Claude Agent SDK (also called `claude-agent-sdk`) is Anthropic's official Python SDK for building agents. It provides the primitives that frameworks like LangGraph build on top of, but with Anthropic-native ergonomics:

- **Tool loop**: Automatic tool execution loop (no manual while-loop)
- **Subagents**: First-class support for delegating to child agents
- **Permissions**: Built-in permission system for tool access control
- **MCP integration**: Native Model Context Protocol client support
- **Streaming**: Token-by-token and event-based streaming
- **Multi-turn**: Conversation management across turns

The SDK sits between the raw Claude API (maximum control, maximum boilerplate) and LangGraph (maximum abstraction, less Anthropic-specific optimization).

## How It Works

### Architecture

```
     +---------------------------+
     |   Claude Agent SDK        |
     |                           |
     |  +---------------------+  |
     |  | Agent               |  |       +------------------+
     |  | - system prompt     |  |       | MCP Servers      |
     |  | - tools             |  |<----->| (filesystem,     |
     |  | - subagents         |  |       |  database, etc.) |
     |  | - permissions       |  |       +------------------+
     |  +----------+----------+  |
     |             |              |
     |  +----------v----------+  |
     |  | Tool Loop           |  |       +------------------+
     |  | 1. Send to Claude   |  |       | Subagents        |
     |  | 2. Check tool calls |  |<----->| (specialist      |
     |  | 3. Execute tools    |  |       |  delegations)    |
     |  | 4. Send results     |  |       +------------------+
     |  | 5. Repeat until done|  |
     |  +---------------------+  |
     +---------------------------+
                 |
     +---------------------------+
     |   Claude Messages API     |
     |   (claude-sonnet, opus)   |
     +---------------------------+
```

### How It Differs from Raw API

```python
# === RAW API (manual tool loop) ===
messages = [{"role": "user", "content": query}]
while True:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        messages=messages,
        tools=tools
    )
    if response.stop_reason == "tool_use":
        # Manually extract tool call, execute, append result...
        tool_block = next(b for b in response.content if b.type == "tool_use")
        result = execute_tool(tool_block.name, tool_block.input)
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": [{"type": "tool_result", ...}]})
    else:
        break  # Done

# === AGENT SDK (automatic tool loop) ===
agent = Agent(
    model="claude-sonnet-4-20250514",
    tools=[search_tool, calculator_tool],
    system_prompt="You are a helpful assistant."
)
result = await agent.run("What is the GDP per capita of France?")
# Tool loop is handled automatically
```

## Production Implementation

### Basic Agent

```python
from claude_agent_sdk import Agent, Tool
from anthropic import Anthropic

client = Anthropic()

# Define tools
@Tool
def search(query: str) -> str:
    """Search the web for information."""
    # Implementation
    return search_api(query)

@Tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))  # Sandbox in production!

# Create agent
agent = Agent(
    client=client,
    model="claude-sonnet-4-20250514",
    system_prompt="You are a helpful research assistant.",
    tools=[search, calculate],
    max_iterations=10
)

# Run
result = await agent.run("What is France's GDP divided by its population?")
print(result.content)
```

### Subagents

```python
from claude_agent_sdk import Agent, Tool

# Define specialist subagents
research_agent = Agent(
    client=client,
    model="claude-sonnet-4-20250514",
    system_prompt="You are a research specialist. Search for data and return findings.",
    tools=[search_tool, scrape_tool]
)

code_agent = Agent(
    client=client,
    model="claude-sonnet-4-20250514",
    system_prompt="You are a code specialist. Write Python code.",
    tools=[code_exec_tool]
)

# Parent agent that can delegate to subagents
orchestrator = Agent(
    client=client,
    model="claude-opus-4-20250514",
    system_prompt="""You orchestrate tasks by delegating to specialist agents.
    Use the research_agent for data gathering and code_agent for code tasks.""",
    subagents={
        "research_agent": research_agent,
        "code_agent": code_agent
    }
)

# When orchestrator decides to delegate:
# It sends a message to the subagent, which runs its own tool loop,
# and returns the result back to the orchestrator.
result = await orchestrator.run("Research AI market size and plot a chart")
```

### Permissions

```python
from claude_agent_sdk import Agent, Permission

# Define permission boundaries
agent = Agent(
    client=client,
    model="claude-sonnet-4-20250514",
    tools=[file_read, file_write, shell_exec, web_search],
    permissions={
        "file_read": Permission.ALLOW,           # Always allowed
        "file_write": Permission.ASK_HUMAN,       # Requires human approval
        "shell_exec": Permission.DENY,            # Never allowed
        "web_search": Permission.ALLOW            # Always allowed
    }
)

# When the agent tries to use file_write:
# The SDK pauses and requests human approval before proceeding.
# When it tries shell_exec: immediately denied without LLM seeing it.
```

### MCP Integration

```python
from claude_agent_sdk import Agent
from claude_agent_sdk.mcp import MCPClient

# Connect to MCP servers
filesystem_mcp = MCPClient("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/data"])
postgres_mcp = MCPClient("npx", ["-y", "@modelcontextprotocol/server-postgres"])

agent = Agent(
    client=client,
    model="claude-sonnet-4-20250514",
    system_prompt="You can read files and query databases.",
    mcp_servers=[filesystem_mcp, postgres_mcp]
)

# Agent automatically discovers tools from MCP servers
# and can use them in its tool loop
result = await agent.run("Read the config file and query user count from the database")
```

### Streaming

```python
# Token-by-token streaming
async for event in agent.stream("Explain quantum computing"):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
    elif event.type == "tool_start":
        print(f"\n[Calling tool: {event.tool_name}]")
    elif event.type == "tool_result":
        print(f"\n[Tool result: {event.result[:100]}...]")
```

## Decision Tree: When to Use

```
     Should I use Claude Agent SDK?
                |
     +----------v----------+
     | Are you building    |
     | with Claude (not    |
     | OpenAI/Gemini)?     |
     +---+------------+----+
         |            |
        Yes          No --> Use that provider's SDK or a framework
         |
     +---v-----------+----+
     | Do you need        |
     | persistence, HITL, |
     | or complex graphs? |
     +---+------------+---+
         |            |
        Yes --> LangGraph   No
                           |
     +---v-----------+----+
     | Do you need        |
     | subagents or       |
     | permission control?|
     +---+------------+---+
         |            |
        Yes          No --> Raw Claude API may suffice
         |
     +---v-------------------+
     | USE Claude Agent SDK  |
     +------------------------+
```

## When NOT to Use

1. **Multi-provider**: If you need to switch between Claude/GPT/Gemini, use LangGraph
2. **Complex graphs**: State machines with cycles need LangGraph
3. **Persistence**: No built-in checkpointing -- use LangGraph for crash recovery
4. **Team collaboration**: LangGraph's graph model is easier for teams to reason about
5. **Non-Python**: SDK is Python-only

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Anthropic-native (optimized for Claude) | Claude-only (vendor lock-in) |
| Simpler than LangGraph for Claude agents | No persistence/checkpointing |
| First-class subagent support | No graph-based flow control |
| Built-in permission system | Smaller ecosystem than LangChain |
| Native MCP integration | Less community content |
| Automatic tool loop | Less mature than LangGraph |

## Real-World Examples

1. **Claude Code**: Anthropic's own coding assistant uses the Agent SDK internally for tool loops and subagent delegation.
2. **MCP-based assistants**: Agents that connect to local files, databases, and APIs via MCP servers.
3. **Permission-gated agents**: Enterprise agents where certain actions require human approval.

## Failure Modes

### 1. Subagent Context Loss
Parent agent sends insufficient context to subagent, causing it to fail.
**Mitigation**: Include relevant prior conversation in the subagent delegation message.

### 2. Permission Bottleneck
Too many actions require human approval, creating latency.
**Mitigation**: Default to ALLOW for read-only tools. Only ASK_HUMAN for destructive actions.

### 3. MCP Server Disconnection
MCP server crashes mid-agent-loop, causing tool calls to fail.
**Mitigation**: Implement retry logic and graceful degradation in the SDK configuration.

## Sources and Further Reading

- [Anthropic Agent SDK Documentation](https://docs.anthropic.com/en/docs/agents/overview)
- [Claude Agent SDK GitHub](https://github.com/anthropic/claude-agent-sdk)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents)
- [Tool Use with Claude](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
