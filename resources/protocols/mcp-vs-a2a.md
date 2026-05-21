# MCP vs A2A: When to Use Each
> MCP connects agents to tools. A2A connects agents to agents. They are complementary, not competing.

## What It Is

MCP (Model Context Protocol) and A2A (Agent-to-Agent) are two open protocols that address different communication needs in agentic systems. Understanding when to use each -- and when to use both together -- is critical for designing multi-agent architectures.

## Comparison Table

| Dimension | MCP | A2A |
|-----------|-----|-----|
| **Purpose** | Agent-to-tool communication | Agent-to-agent communication |
| **Target** | Tools, data sources, APIs | Other AI agents |
| **Direction** | Agent calls tool (client-server) | Agent delegates to agent (peer-to-peer) |
| **Protocol** | JSON-RPC over stdio/SSE/HTTP | JSON-RPC over HTTP |
| **Primitives** | Tools, Resources, Prompts | Tasks, Messages, Artifacts |
| **Discovery** | Client configuration | Agent Cards (/.well-known/agent.json) |
| **State** | Stateless tool calls | Stateful task lifecycle |
| **Streaming** | Supported | Supported |
| **Creator** | Anthropic | Google |
| **Maturity** | Production-ready (2024+) | Early adoption (2025) |
| **Adoption** | Claude, Cursor, LangChain, many IDEs | Google Agentspace, early adopters |
| **Complexity** | Low-Medium | Medium-High |

## How They Work Together

```
                    A2A Protocol
    ┌──────────────────────────────────────┐
    │                                      │
    ▼                                      ▼
┌──────────┐                        ┌──────────┐
│ Agent A  │                        │ Agent B  │
│          │                        │          │
│ ┌──────┐ │   MCP        MCP     │ ┌──────┐ │
│ │Client├─┤──►┌──────┐  ┌──────┐◄├─┤Client│ │
│ └──────┘ │   │Search │  │Email │ │ └──────┘ │
│          │   │Server │  │Server│ │          │
│ ┌──────┐ │   └──────┘  └──────┘ │ ┌──────┐ │
│ │Client├─┤──►┌──────┐           │ │Client├─┤──►┌──────┐
│ └──────┘ │   │DB     │           │ └──────┘ │   │CRM   │
│          │   │Server │           │          │   │Server │
└──────────┘   └──────┘           └──────────┘   └──────┘

Agent A uses MCP to access its tools (search, database)
Agent B uses MCP to access its tools (email, CRM)
Agent A uses A2A to delegate tasks to Agent B
```

### Concrete Example

```
Scenario: Customer wants a refund that requires approval

1. Support Agent (your org) receives customer request
   └── Uses MCP to search knowledge base (MCP tool call)

2. Support Agent determines refund needs finance team approval
   └── Uses A2A to send task to Finance Agent (different team/org)
       └── A2A Task: "Approve refund of $150 for order #12345"

3. Finance Agent processes the task
   └── Uses MCP to query financial database (MCP tool call)
   └── Uses MCP to check fraud rules (MCP tool call)
   └── Returns A2A result: "Approved" with artifact (approval ID)

4. Support Agent receives approval
   └── Uses MCP to process refund (MCP tool call)
   └── Uses MCP to send confirmation email (MCP tool call)
```

## Decision Guide

```
START: What are you connecting?
│
├── Agent needs to call a function/API/database
│   └── Use MCP
│       ├── Build an MCP server for the tool
│       └── Connect via MCP client in your agent
│
├── Agent needs to delegate work to another AI agent
│   ├── Same codebase / same team?
│   │   └── Direct function call or shared state (simplest)
│   │       └── Consider MCP if you want loose coupling
│   │
│   ├── Different team, same organization?
│   │   ├── Can share infrastructure? → Internal A2A or message queue
│   │   └── Independent services? → A2A protocol
│   │
│   └── Different organization?
│       └── A2A protocol (designed for cross-org agent communication)
│
├── Agent needs to read external data
│   └── Use MCP Resources
│
└── Both tool access AND agent delegation?
    └── Use both: MCP for tools, A2A for agent-agent
```

## Production Implementation

```python
class HybridAgentOrchestrator:
    """
    Agent that uses both MCP (for tools) and A2A (for other agents).
    """
    
    def __init__(self):
        self.mcp_clients: dict[str, MCPClient] = {}
        self.a2a_client = A2AClient()
        self.known_agents: dict[str, AgentCard] = {}
    
    async def setup(self):
        """Initialize connections to MCP servers and discover A2A agents."""
        # MCP: Connect to tool servers
        self.mcp_clients["knowledge"] = await MCPClient.connect(
            command="python", args=["mcp_servers/knowledge.py"]
        )
        self.mcp_clients["ticketing"] = await MCPClient.connect(
            url="http://ticketing-mcp:8080/sse"
        )
        
        # A2A: Discover partner agents
        self.known_agents["finance"] = await self.a2a_client.discover_agent(
            "https://finance-agent.internal.example.com"
        )
        self.known_agents["shipping"] = await self.a2a_client.discover_agent(
            "https://shipping-agent.partner.com"
        )
    
    async def handle_request(self, user_message: str) -> str:
        """Route request to appropriate protocol."""
        
        # Step 1: Use MCP tool to understand the request
        intent = await self.mcp_clients["knowledge"].call_tool(
            "classify_intent", {"message": user_message}
        )
        
        # Step 2: If simple, handle with MCP tools
        if intent["type"] == "simple_query":
            answer = await self.mcp_clients["knowledge"].call_tool(
                "search_knowledge_base", {"query": user_message}
            )
            return answer
        
        # Step 3: If requires another agent, use A2A
        if intent["type"] == "refund_approval":
            task = await self.a2a_client.send_task(
                agent_url=self.known_agents["finance"].url,
                task_description=f"Review refund request: {user_message}",
                context={"customer_id": intent["customer_id"]},
            )
            
            # Wait for result (with timeout)
            result = await self._wait_for_task(
                self.known_agents["finance"].url, task.id, timeout=60
            )
            return f"Refund status: {result.status.value}"
        
        return "I'm not sure how to help with that."
```

## When NOT to Use Either Protocol

- **Simple scripts**: A Python function calling an API doesn't need MCP or A2A.
- **Monolith agents**: Everything in one process doesn't need inter-process protocols.
- **Real-time gaming/robotics**: Protocols designed for conversational AI, not sub-millisecond control loops.

## Tradeoffs: Combined vs Separate

| Architecture | When | Pros | Cons |
|-------------|------|------|------|
| MCP only | Single agent, multiple tools | Simple, proven, well-supported | Can't delegate to other agents |
| A2A only | Agent orchestrating other agents, no direct tool access | Clean agent boundaries | Each agent needs its own tool access |
| MCP + A2A | Full enterprise agent ecosystem | Comprehensive, flexible | Highest complexity |
| Neither | Simple single-function agent | Minimal overhead | Not extensible |

## Real-World Examples

- **Enterprise support**: Support agent (MCP: knowledge base, CRM) delegates billing questions to billing agent (A2A).
- **Travel planning**: Planning agent (A2A) coordinates with flight agent, hotel agent, and activities agent -- each using MCP for their domain tools.
- **Development workflow**: PM agent (A2A) sends tasks to engineering agent, which uses MCP to interact with GitHub, CI/CD, and deployment tools.

## Source(s) and Further Reading

- MCP Specification: https://modelcontextprotocol.io/
- A2A Protocol: https://google.github.io/A2A/
- "MCP vs A2A: Complementary Protocols" - Google Cloud Blog (2025)
- Anthropic MCP Announcement: https://www.anthropic.com/news/model-context-protocol
