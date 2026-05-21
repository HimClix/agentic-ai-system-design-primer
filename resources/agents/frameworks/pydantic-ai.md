# Pydantic AI

> Type-safe agents with dependency injection -- if you want Python-native, strongly typed agent development with MCP integration baked in, Pydantic AI is the growing alternative to LangChain.

## What It Is

Pydantic AI is an agent framework built by the team behind Pydantic (the most popular Python validation library). It brings Pydantic's philosophy to agents:

- **Type safety**: Agent inputs, outputs, and dependencies are all Pydantic models
- **Dependency injection**: Tools receive typed dependencies (DB connections, API clients) at runtime
- **Model agnostic**: Works with OpenAI, Anthropic, Gemini, Groq, and Ollama
- **MCP integration**: First-class support for Model Context Protocol servers
- **Logfire integration**: Built-in observability via Pydantic Logfire

## How It Works

### Architecture

```
     +------------------+
     |    Agent         |
     |  +------------+  |
     |  | System     |  |        +----------------+
     |  | Prompt     |  |        | Dependencies   |
     |  +------------+  |        | (injected at   |
     |  | Tools      |  |<-------| runtime)       |
     |  | (typed)    |  |        | - DB conn      |
     |  +------------+  |        | - API client   |
     |  | Result     |  |        | - Config       |
     |  | Type (T)   |  |        +----------------+
     |  +------------+  |
     +------------------+
             |
             v
     +------------------+
     |  Structured      |
     |  Output (T)      |
     |  Validated by    |
     |  Pydantic        |
     +------------------+
```

## Production Implementation

### Basic Agent

```python
from pydantic_ai import Agent
from pydantic import BaseModel

# Define structured output
class CityInfo(BaseModel):
    city: str
    country: str
    population: int
    notable_fact: str

# Create agent with typed output
agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    result_type=CityInfo,
    system_prompt="You are a geography expert. Return accurate city information."
)

# Run -- result is validated CityInfo, not raw text
result = agent.run_sync("Tell me about Tokyo")
print(result.data)  # CityInfo(city='Tokyo', country='Japan', population=13960000, ...)
print(result.data.population)  # 13960000 -- typed int, not string
```

### Dependency Injection

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

# Define dependencies
@dataclass
class AppDependencies:
    db_connection: DatabasePool
    api_client: HttpClient
    user_id: str

# Create agent with dependency type
agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    deps_type=AppDependencies,
    system_prompt="You are a customer support agent."
)

# Tools receive dependencies automatically via RunContext
@agent.tool
async def lookup_customer(ctx: RunContext[AppDependencies], customer_id: str) -> str:
    """Look up customer details from the database."""
    # ctx.deps is the injected AppDependencies instance
    result = await ctx.deps.db_connection.fetch(
        "SELECT * FROM customers WHERE id = $1", customer_id
    )
    return str(result)

@agent.tool
async def check_order_status(ctx: RunContext[AppDependencies], order_id: str) -> str:
    """Check order status via API."""
    response = await ctx.deps.api_client.get(f"/orders/{order_id}")
    return response.json()

# Run with dependencies injected at runtime
deps = AppDependencies(
    db_connection=await get_db_pool(),
    api_client=HttpClient(base_url="https://api.example.com"),
    user_id="user-123"
)

result = await agent.run("What's the status of order ORD-456?", deps=deps)
```

### MCP Integration

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerHTTP, MCPServerStdio

# Connect to MCP servers
weather_server = MCPServerHTTP(url="http://localhost:8080/mcp")
db_server = MCPServerStdio("npx", ["-y", "@modelcontextprotocol/server-postgres"])

agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    mcp_servers=[weather_server, db_server],
    system_prompt="You can check weather and query databases."
)

# Agent automatically discovers and uses MCP tools
async with agent.run_mcp_servers():
    result = await agent.run("What's the weather in Tokyo and how many users are in our DB?")
```

### Dynamic System Prompts

```python
@agent.system_prompt
async def dynamic_prompt(ctx: RunContext[AppDependencies]) -> str:
    """System prompt that adapts based on injected dependencies."""
    user = await ctx.deps.db_connection.fetch_user(ctx.deps.user_id)
    return f"""You are a support agent for {user.company}.
    The user's tier is {user.tier}. Adjust formality accordingly.
    {'Be concise.' if user.tier == 'enterprise' else 'Be friendly and detailed.'}"""
```

## Decision Tree: When to Use

```
     Should I use Pydantic AI?
                |
     +----------v----------+
     | Do you value type   |
     | safety and want     |
     | structured output?  |
     +---+------------+----+
         |            |
        Yes          No --> LangGraph or CrewAI
         |
     +---v-----------+----+
     | Do you need        |
     | dependency          |
     | injection (DB,     |
     | API clients)?      |
     +---+------------+---+
         |            |
        Yes          No --> Consider simpler frameworks
         |
     +---v-----------+----+
     | Are you already    |
     | using Pydantic     |
     | heavily?           |
     +---+------------+---+
         |            |
        Yes          No --> Still viable, but evaluate ecosystem maturity
         |
     +---v-------------------+
     | USE Pydantic AI       |
     +------------------------+
```

## When NOT to Use

1. **Complex multi-agent systems**: LangGraph or AutoGen handle this better
2. **Graph-based execution**: No built-in graph primitives like LangGraph
3. **Large existing LangChain codebase**: Migration cost may not be worth it
4. **Need HITL with persistence**: LangGraph's checkpointer + interrupt is more mature
5. **Maximum community support**: Smaller ecosystem than LangChain

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Strongest type safety of any framework | Smaller ecosystem and community |
| Dependency injection is production-ready | No built-in graph/state machine |
| First-class MCP support | Less mature multi-agent patterns |
| Clean, Pythonic API | Fewer tutorials and examples |
| Built-in Logfire observability | Newer, less battle-tested |
| Pydantic team backing | 2.7% job market share vs LangChain's 34.3% |

## Sources and Further Reading

- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [Pydantic AI GitHub](https://github.com/pydantic/pydantic-ai)
- [MCP Integration Guide](https://ai.pydantic.dev/mcp/)
- [Pydantic Logfire](https://pydantic.dev/logfire)
