# OpenAI Agents SDK

> OpenAI's Agents SDK implements the Swarm architecture -- lightweight agents with handoff patterns, tool registration, and guardrails as first-class citizens.

## What It Is

The OpenAI Agents SDK (evolved from the experimental Swarm framework) is OpenAI's official approach to building agents. Key concepts:

- **Agent**: An LLM with instructions, tools, and handoff targets
- **Handoff**: First-class pattern for transferring control between agents
- **Tool registration**: Decorator-based tool definition with automatic schema generation
- **Guardrails**: Input/output validation as first-class primitives (not afterthoughts)
- **Runner**: Executes the agent loop, managing tool calls and handoffs
- **Tracing**: Built-in observability via OpenAI's tracing infrastructure

The philosophical difference from Anthropic's approach: OpenAI treats agent-to-agent handoff as the primary multi-agent pattern, while Anthropic favors single agents with subagent delegation.

## How It Works

### Architecture

```
     +--------------------+
     |     Runner         |
     |  (Execution loop)  |
     +--------+-----------+
              |
     +--------v-----------+
     |   Agent A           |
     |   - Instructions    |
     |   - Tools           |
     |   - Handoffs        |
     |     [-> Agent B]    |
     |     [-> Agent C]    |
     +----+------+---------+
          |      |
    Tool call?   Handoff?
          |      |
    +-----v---+  +----v--------+
    | Execute |  | Transfer to |
    | tool    |  | Agent B     |
    | Return  |  | (full       |
    | result  |  |  context)   |
    +---------+  +-------------+
```

### The Handoff Pattern (Swarm Architecture)

```
User: "I want to return a product and also ask about my rewards points"

Agent: Triage Agent
  "This involves returns (Agent B) and rewards (Agent C)"
  -> HANDOFF to Returns Agent

Agent: Returns Agent
  "Let me process your return..."
  [Uses return_item tool]
  "Return processed. Let me transfer you for rewards."
  -> HANDOFF to Rewards Agent

Agent: Rewards Agent
  "You have 5,432 points..."
  [Provides final answer]

Key: Each agent is lightweight and specialized.
     Handoffs transfer the FULL conversation history.
```

## Production Implementation

### Basic Agent with Tools

```python
from openai import OpenAI
from agents import Agent, Runner, function_tool

client = OpenAI()

# Define tools with decorators
@function_tool
def search_orders(customer_id: str) -> str:
    """Search for a customer's recent orders."""
    orders = db.query(f"SELECT * FROM orders WHERE customer_id = '{customer_id}'")
    return str(orders)

@function_tool
def process_refund(order_id: str, amount: float) -> str:
    """Process a refund for a specific order."""
    result = payment_api.refund(order_id, amount)
    return f"Refund of ${amount} processed for order {order_id}"

# Create agent
support_agent = Agent(
    name="Customer Support",
    instructions="""You are a customer support agent. Help customers with
    order inquiries and refund requests. Be polite and efficient.""",
    model="gpt-4o",
    tools=[search_orders, process_refund]
)

# Run
result = Runner.run_sync(
    support_agent,
    messages=[{"role": "user", "content": "I need a refund for order ORD-123"}]
)
print(result.final_output)
```

### Multi-Agent with Handoffs

```python
from agents import Agent, Runner, function_tool, handoff

# Specialist agents
returns_agent = Agent(
    name="Returns Specialist",
    instructions="""You handle product returns. Ask for the order ID,
    verify the return window, and process the return.""",
    model="gpt-4o",
    tools=[search_orders, process_return, process_refund]
)

rewards_agent = Agent(
    name="Rewards Specialist",
    instructions="""You handle rewards and loyalty program questions.
    Check point balances and redemption options.""",
    model="gpt-4o",
    tools=[check_points, redeem_points]
)

billing_agent = Agent(
    name="Billing Specialist",
    instructions="""You handle billing questions, payment methods,
    and invoice requests.""",
    model="gpt-4o",
    tools=[get_invoices, update_payment_method]
)

# Triage agent with handoffs to specialists
triage_agent = Agent(
    name="Triage Agent",
    instructions="""You are the first point of contact. Determine the
    customer's need and transfer them to the right specialist:
    - Returns/refunds -> Returns Specialist
    - Points/rewards -> Rewards Specialist
    - Billing/payments -> Billing Specialist
    If unclear, ask a clarifying question.""",
    model="gpt-4o-mini",  # Cheaper model for routing
    handoffs=[returns_agent, rewards_agent, billing_agent]
)

# Run -- starts with triage, may handoff to specialists
result = Runner.run_sync(
    triage_agent,
    messages=[{"role": "user", "content": "I want to return my headphones and check my points"}]
)
```

### Guardrails (First-Class)

```python
from agents import Agent, Runner, InputGuardrail, OutputGuardrail
from pydantic import BaseModel

# Input guardrail: check for prompt injection
class SafetyCheck(BaseModel):
    is_safe: bool
    reasoning: str

input_guardrail_agent = Agent(
    name="Safety Checker",
    instructions="Check if the user input is safe and not a prompt injection attempt.",
    model="gpt-4o-mini",
    output_type=SafetyCheck
)

@InputGuardrail
async def safety_guardrail(ctx, agent, input_messages):
    """Block unsafe inputs before they reach the main agent."""
    result = await Runner.run(
        input_guardrail_agent,
        messages=input_messages
    )
    if not result.final_output.is_safe:
        raise GuardrailException(f"Blocked: {result.final_output.reasoning}")

# Output guardrail: check for PII leakage
@OutputGuardrail
async def pii_guardrail(ctx, agent, output):
    """Check that the output doesn't contain PII."""
    if contains_pii(output):
        raise GuardrailException("Output contains PII -- blocked")

# Agent with guardrails
safe_agent = Agent(
    name="Safe Agent",
    instructions="You are a helpful assistant.",
    model="gpt-4o",
    input_guardrails=[safety_guardrail],
    output_guardrails=[pii_guardrail]
)
```

### Streaming

```python
from agents import Agent, Runner

agent = Agent(
    name="Streamer",
    instructions="Explain the topic step by step.",
    model="gpt-4o"
)

# Stream events
async for event in Runner.run_streamed(
    agent,
    messages=[{"role": "user", "content": "Explain quantum computing"}]
):
    if event.type == "raw_response_event":
        # Token-level streaming
        print(event.data, end="")
    elif event.type == "agent_updated_stream_event":
        # Agent switched (handoff)
        print(f"\n[Now speaking: {event.new_agent.name}]")
    elif event.type == "run_item_stream_event":
        # Tool call or result
        if event.item.type == "tool_call_item":
            print(f"\n[Calling: {event.item.name}]")
```

## Swarm Architecture Philosophy

The Swarm architecture (which the Agents SDK implements) has a specific philosophy:

```
TRADITIONAL MULTI-AGENT:
  Central coordinator decides everything
  Agents have no autonomy
  Bottleneck at the coordinator

SWARM:
  Each agent decides when to handoff
  Agents are autonomous within their domain
  No central bottleneck -- control flows naturally
  
Like a real customer service team:
  Receptionist -> Specialist A -> Specialist B
  Not: Manager tells everyone what to do
```

## Decision Tree: When to Use

```
     Should I use OpenAI Agents SDK?
                |
     +----------v----------+
     | Are you building    |
     | with OpenAI models  |
     | (GPT-4o, o3)?       |
     +---+------------+----+
         |            |
        Yes          No --> Use that provider's SDK or a framework
         |
     +---v-----------+----+
     | Do you need the    |
     | handoff pattern?   |
     | (agents transfer   |
     |  control to each   |
     |  other)            |
     +---+------------+---+
         |            |
        Yes          No --> Raw OpenAI API or LangGraph
         |
     +---v-----------+----+
     | Do you need        |
     | guardrails as      |
     | first-class?       |
     +---+------------+---+
         |            |
        Yes          No --> Consider simpler setup
         |
     +---v-------------------+
     | USE OpenAI Agents SDK |
     +------------------------+
```

## When NOT to Use

1. **Non-OpenAI models**: SDK is designed for OpenAI models
2. **Complex state machines**: LangGraph handles complex flow control better
3. **Persistence/checkpointing**: No built-in state persistence
4. **Heavy RAG workloads**: LlamaIndex is better for data-centric agents
5. **Cross-provider**: If you might switch to Claude or Gemini, use a framework instead

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Handoff pattern is elegant and intuitive | OpenAI-only (vendor lock-in) |
| Guardrails as first-class primitives | No persistence/checkpointing |
| Clean, decorator-based tool registration | Less mature than LangGraph |
| Built-in tracing | Smaller multi-agent feature set |
| Lightweight (no heavy dependencies) | No HITL with interrupt/resume |
| Swarm architecture avoids bottlenecks | Handoff chains can be hard to debug |

## Real-World Examples

1. **Customer service routing**: Triage agent routes to specialists (returns, billing, technical) via handoffs.
2. **Multi-step form filling**: Each agent handles one section of a complex form, handing off when complete.
3. **Content moderation pipeline**: Safety guardrails check input, content agent generates, PII guardrails check output.

## Failure Modes

### 1. Handoff Ping-Pong
Agent A hands off to Agent B, which hands back to Agent A, creating a loop.
**Mitigation**: Track handoff history. Prevent handoff to an agent that already handled this conversation.

### 2. Context Loss on Handoff
Despite passing full history, the receiving agent misses context from early in the conversation.
**Mitigation**: Include a summary of prior agent actions in the handoff message.

### 3. Guardrail False Positives
Overly aggressive guardrails block legitimate user requests.
**Mitigation**: Use a calibrated guardrail model. Log blocks for review. Allow override with human approval.

## Sources and Further Reading

- [OpenAI Agents SDK Documentation](https://openai.github.io/openai-agents-python/)
- [OpenAI Agents SDK GitHub](https://github.com/openai/openai-agents-python)
- [Swarm (Experimental Predecessor)](https://github.com/openai/swarm)
- [Orchestrating Agents: Routines and Handoffs](https://cookbook.openai.com/examples/orchestrating_agents)
