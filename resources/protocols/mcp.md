# Model Context Protocol (MCP)
> MCP is the USB-C of agent-tool communication -- a universal protocol that lets any agent talk to any tool server without custom integration code.

## What It Is

The Model Context Protocol (MCP) is an open standard created by Anthropic for connecting AI agents to external tools and data sources. Instead of every agent framework implementing its own tool integration layer, MCP provides a universal client-server protocol with three primitives: **Tools** (actions the agent can take), **Resources** (data the agent can read), and **Prompts** (reusable prompt templates).

MCP solves the N x M integration problem: without MCP, N agent frameworks each need custom integrations for M tools = N x M integrations. With MCP, each framework implements one MCP client and each tool implements one MCP server = N + M integrations.

## How It Works

### Architecture

```
┌────────────────────┐       ┌────────────────────┐
│   Agent (Host)     │       │   MCP Server       │
│                    │       │   (Tool Provider)   │
│ ┌────────────────┐ │  MCP  │ ┌────────────────┐ │
│ │  MCP Client    │◄├───────┤►│  Tool Logic    │ │
│ │                │ │Protocol│ │                │ │
│ │ - List tools   │ │       │ │ - search_docs  │ │
│ │ - Call tools   │ │       │ │ - create_ticket│ │
│ │ - Read resources│ │       │ │ - run_query    │ │
│ └────────────────┘ │       │ └────────────────┘ │
│                    │       │                    │
│ ┌────────────────┐ │       │ ┌────────────────┐ │
│ │  LLM           │ │       │ │  External APIs │ │
│ │  (Claude, etc) │ │       │ │  (DB, SaaS)    │ │
│ └────────────────┘ │       │ └────────────────┘ │
└────────────────────┘       └────────────────────┘
```

### Three Primitives

| Primitive | Direction | Purpose | Example |
|-----------|-----------|---------|---------|
| **Tools** | Client calls server | Execute actions | `search_knowledge_base(query="refund policy")` |
| **Resources** | Client reads server | Access data | `resource://docs/refund-policy.md` |
| **Prompts** | Client gets templates | Reusable prompts | `prompt://summarize-ticket` |

### Transport Types

| Transport | Use Case | Connection | Latency |
|-----------|----------|-----------|---------|
| **stdio** | Local tools, CLI | Process stdin/stdout | <1ms |
| **SSE** | Remote servers, HTTP | Server-Sent Events | 10-50ms |
| **Streamable HTTP** | Production remote | HTTP + streaming | 10-50ms |

## Production Implementation

### Building an MCP Server in Python (FastMCP)

```python
from mcp.server.fastmcp import FastMCP
from pydantic import Field
import httpx

# Create MCP server
mcp = FastMCP(
    name="knowledge-base",
    description="Search and manage the company knowledge base",
)


# --- Tools ---

@mcp.tool()
async def search_knowledge_base(
    query: str = Field(description="Natural language search query"),
    top_k: int = Field(default=5, description="Number of results to return", ge=1, le=20),
    category: str = Field(default="all", description="Filter by category: all, product, billing, technical"),
) -> str:
    """Search the knowledge base for relevant articles.
    
    Use this tool when you need to find information about company products,
    policies, or technical documentation. Returns the most relevant articles
    with their content and relevance scores.
    """
    # Your search implementation
    results = await vector_search(query, top_k=top_k, category=category)
    
    formatted = []
    for r in results:
        formatted.append(f"## {r.title} (score: {r.score:.2f})\n{r.content}")
    
    return "\n\n---\n\n".join(formatted) if formatted else "No results found."


@mcp.tool()
async def create_support_ticket(
    title: str = Field(description="Ticket title"),
    description: str = Field(description="Detailed ticket description"),
    priority: str = Field(default="medium", description="Priority: low, medium, high, critical"),
    customer_email: str = Field(description="Customer email address"),
) -> str:
    """Create a new support ticket in the ticketing system.
    
    Use this tool when a customer issue cannot be resolved immediately
    and needs to be tracked. Always include relevant context from the
    conversation in the description.
    """
    ticket = await ticketing_api.create(
        title=title,
        description=description,
        priority=priority,
        customer_email=customer_email,
    )
    return f"Ticket created: {ticket.id} (URL: {ticket.url})"


# --- Resources ---

@mcp.resource("docs://policies/{policy_name}")
async def get_policy(policy_name: str) -> str:
    """Get a specific company policy document."""
    content = await load_policy(policy_name)
    return content


@mcp.resource("docs://faq")
async def get_faq() -> str:
    """Get the frequently asked questions document."""
    return await load_document("faq.md")


# --- Prompts ---

@mcp.prompt()
def summarize_ticket(ticket_content: str) -> str:
    """Generate a concise summary of a support ticket."""
    return f"""Summarize the following support ticket in 2-3 sentences.
Focus on: the customer's issue, what was tried, and current status.

Ticket content:
{ticket_content}"""


# Run the server
if __name__ == "__main__":
    mcp.run(transport="stdio")  # For local use
    # mcp.run(transport="sse", port=8080)  # For remote use
```

### Building an MCP Server in Go (mcp-go)

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    
    "github.com/mark3labs/mcp-go/mcp"
    "github.com/mark3labs/mcp-go/server"
)

func main() {
    s := server.NewMCPServer(
        "knowledge-base",
        "1.0.0",
        server.WithToolCapabilities(true),
        server.WithResourceCapabilities(true, false),
    )

    // Register tools
    searchTool := mcp.NewTool("search_knowledge_base",
        mcp.WithDescription("Search the knowledge base for relevant articles"),
        mcp.WithString("query",
            mcp.Required(),
            mcp.Description("Natural language search query"),
        ),
        mcp.WithNumber("top_k",
            mcp.Description("Number of results to return"),
        ),
    )

    s.AddTool(searchTool, func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
        query := req.Params.Arguments["query"].(string)
        topK := 5
        if v, ok := req.Params.Arguments["top_k"]; ok {
            topK = int(v.(float64))
        }

        results, err := vectorSearch(ctx, query, topK)
        if err != nil {
            return mcp.NewToolResultError(fmt.Sprintf("Search failed: %v", err)), nil
        }

        content, _ := json.Marshal(results)
        return mcp.NewToolResultText(string(content)), nil
    })

    // Start server
    if err := server.ServeStdio(s); err != nil {
        fmt.Printf("Server error: %v\n", err)
    }
}
```

### Connecting MCP Servers to LangGraph

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

async def build_agent_with_mcp():
    """Build a LangGraph agent connected to MCP tool servers."""
    
    async with MultiServerMCPClient({
        "knowledge_base": {
            "command": "python",
            "args": ["mcp_servers/knowledge_base.py"],
            "transport": "stdio",
        },
        "ticketing": {
            "url": "http://localhost:8081/sse",
            "transport": "sse",
        },
    }) as client:
        # MCP tools automatically converted to LangChain tools
        tools = client.get_tools()
        
        model = ChatAnthropic(model="claude-sonnet-4-20250514")
        agent = create_react_agent(model, tools)
        
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": "Find our refund policy"}]
        })
        return result
```

## Decision Tree: When to Build MCP Servers

```
Do you have tools that agents need to use?
│
├── YES → Are multiple agent frameworks using these tools?
│   ├── YES → Build MCP servers (universal compatibility)
│   └── NO → Single framework?
│       ├── LangGraph? → MCP or native LangChain tools (both work)
│       └── Other? → Check if framework supports MCP
│
└── NO → You don't need MCP yet
```

## When NOT to Use MCP

- **Simple single-tool agents**: Direct function calling is simpler for one or two tools.
- **Framework-locked**: If you're committed to one framework and won't switch, native tools may be simpler.
- **Performance-critical**: MCP adds serialization overhead. For sub-millisecond tool calls, use direct function invocation.

## Tradeoffs

| Aspect | MCP Tools | Native Framework Tools |
|--------|----------|----------------------|
| Portability | Works across frameworks | Framework-specific |
| Setup complexity | Medium (server + transport) | Low (just a function) |
| Type safety | JSON Schema | Language-native types |
| Debugging | Server logs + protocol traces | Standard debugging |
| Performance | +5-50ms (serialization + transport) | <1ms (function call) |
| Ecosystem | Growing MCP server registry | Framework-specific plugins |

## Real-World Examples

- **Claude Desktop**: Connects to local MCP servers for file access, database queries, and web search.
- **Cursor IDE**: Uses MCP for code intelligence tools.
- **Enterprise agents**: Build one MCP server for Jira, reuse across support agent, PM agent, and engineering agent.

## Failure Modes

1. **MCP server crash**: Tool becomes unavailable mid-conversation. Mitigation: health checks, automatic restart, fallback tools.
2. **Schema mismatch**: Server updates tool schema but client has cached old schema. Mitigation: schema versioning, client-side validation.
3. **Transport disconnection**: SSE connection drops during long agent sessions. Mitigation: reconnection logic, request-response fallback.

## Source(s) and Further Reading

- MCP Specification: https://modelcontextprotocol.io/
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- mcp-go: https://github.com/mark3labs/mcp-go
- FastMCP: https://github.com/jlowin/fastmcp
- LangChain MCP Adapters: https://github.com/langchain-ai/langchain-mcp-adapters
