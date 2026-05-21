# Google Agent Development Kit (ADK)

> Google's ADK is the multi-language (Python/Java/Go) agent framework designed for GCP-native deployment -- if you are building agents for Vertex AI or Google Cloud, ADK is the native path.

## What It Is

Google's Agent Development Kit (ADK) is a framework for building, deploying, and managing AI agents. Key differentiators:

- **Multi-language**: Python, Java, and Go SDKs (most frameworks are Python-only)
- **GCP-native**: Designed for deployment on Vertex AI Agent Builder
- **Gemini-optimized**: Best experience with Gemini models (but supports others)
- **Agent Engine**: Managed runtime for deploying agents at scale
- **A2A Protocol**: Agent-to-Agent protocol for cross-agent communication
- **Grounding**: Built-in Google Search grounding for factual accuracy

## How It Works

### Architecture

```
     +---------------------------+
     |   Agent Development Kit   |
     |                           |
     |  +---------------------+  |          +------------------+
     |  | Agent               |  |          | Vertex AI        |
     |  | - model             |  |          | Agent Engine     |
     |  | - instructions      |  |--------->| (Managed deploy) |
     |  | - tools             |  |          +------------------+
     |  | - sub_agents        |  |
     |  +---------------------+  |          +------------------+
     |                           |          | Google Cloud     |
     |  +---------------------+  |          | Services         |
     |  | Tools               |  |--------->| - Search         |
     |  | - FunctionTool      |  |          | - BigQuery       |
     |  | - GoogleSearchTool  |  |          | - Cloud Storage  |
     |  | - CodeExecTool      |  |          +------------------+
     |  +---------------------+  |
     |                           |          +------------------+
     |  +---------------------+  |          | A2A Protocol     |
     |  | Orchestration       |  |--------->| (Agent-to-Agent  |
     |  | - SequentialAgent   |  |          |  communication)  |
     |  | - ParallelAgent     |  |          +------------------+
     |  | - LoopAgent         |  |
     |  +---------------------+  |
     +---------------------------+
```

### Built-in Agent Types

```
LlmAgent:        Single LLM agent with tools
SequentialAgent:  Runs sub-agents in order
ParallelAgent:    Runs sub-agents simultaneously
LoopAgent:        Repeats a sub-agent until condition met
CustomAgent:      User-defined agent logic
```

## Production Implementation

### Basic Agent (Python)

```python
from google.adk import Agent, Runner
from google.adk.tools import FunctionTool, google_search

# Define tools
@FunctionTool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return weather_api.get(city)

@FunctionTool
def book_hotel(city: str, check_in: str, check_out: str) -> str:
    """Book a hotel in the specified city."""
    return booking_api.book(city, check_in, check_out)

# Create agent
travel_agent = Agent(
    name="travel_agent",
    model="gemini-2.0-flash",
    instruction="""You are a travel planning assistant.
    Help users plan trips by checking weather and booking hotels.
    Always check the weather before recommending destinations.""",
    tools=[get_weather, book_hotel, google_search]
)

# Run
runner = Runner(agent=travel_agent)
result = runner.run("Plan a 3-day trip to Tokyo next week")
print(result.output)
```

### Multi-Agent Orchestration

```python
from google.adk import Agent, SequentialAgent, ParallelAgent

# Specialist agents
research_agent = Agent(
    name="researcher",
    model="gemini-2.0-flash",
    instruction="Research the given topic thoroughly using Google Search.",
    tools=[google_search]
)

data_agent = Agent(
    name="data_analyst",
    model="gemini-2.0-flash",
    instruction="Analyze data and generate insights.",
    tools=[bigquery_tool, sheets_tool]
)

writer_agent = Agent(
    name="writer",
    model="gemini-2.0-flash",
    instruction="Write a clear, well-structured report from the research and data."
)

# Sequential: research -> analyze -> write
report_pipeline = SequentialAgent(
    name="report_pipeline",
    sub_agents=[research_agent, data_agent, writer_agent]
)

# Or parallel: research AND analyze at the same time, then write
parallel_gather = ParallelAgent(
    name="data_gathering",
    sub_agents=[research_agent, data_agent]
)

efficient_pipeline = SequentialAgent(
    name="efficient_report",
    sub_agents=[parallel_gather, writer_agent]
)

runner = Runner(agent=efficient_pipeline)
result = runner.run("Create a market analysis report for electric vehicles")
```

### Loop Agent (Reflexion-like)

```python
from google.adk import Agent, LoopAgent

# Agent that iterates until quality is sufficient
code_gen_agent = Agent(
    name="code_generator",
    model="gemini-2.0-flash",
    instruction="""Write Python code for the given task.
    If tests fail, review the error and fix the code.
    Set 'quality_sufficient' to True when all tests pass.""",
    tools=[code_exec_tool, test_runner_tool]
)

iterative_coder = LoopAgent(
    name="iterative_coder",
    sub_agent=code_gen_agent,
    max_iterations=3,
    exit_condition=lambda state: state.get("quality_sufficient", False)
)
```

### Deployment to Vertex AI

```python
from google.adk import Agent
from google.adk.deploy import VertexAIDeployer

agent = Agent(
    name="production_agent",
    model="gemini-2.0-flash",
    instruction="...",
    tools=[...]
)

# Deploy to Vertex AI Agent Engine
deployer = VertexAIDeployer(
    project="my-gcp-project",
    location="us-central1"
)

endpoint = deployer.deploy(
    agent=agent,
    display_name="Production Travel Agent",
    scaling={
        "min_replicas": 1,
        "max_replicas": 10
    }
)

print(f"Agent deployed at: {endpoint.resource_name}")
```

### Java SDK Example

```java
import com.google.adk.Agent;
import com.google.adk.Runner;
import com.google.adk.tools.FunctionTool;

// Java SDK -- same concepts, different language
public class TravelAgent {
    @FunctionTool(description = "Get weather for a city")
    public static String getWeather(String city) {
        return WeatherAPI.get(city);
    }

    public static void main(String[] args) {
        Agent agent = Agent.builder()
            .name("travel_agent")
            .model("gemini-2.0-flash")
            .instruction("You are a travel planning assistant.")
            .tools(List.of(TravelAgent::getWeather))
            .build();

        Runner runner = new Runner(agent);
        var result = runner.run("Plan a trip to Paris");
        System.out.println(result.getOutput());
    }
}
```

## GCP Integration Points

| GCP Service | ADK Integration | Use Case |
|-------------|----------------|----------|
| **Vertex AI** | Agent Engine (managed runtime) | Production deployment |
| **Google Search** | Built-in GoogleSearchTool | Grounding / fact-checking |
| **BigQuery** | Tool wrapping | Data analysis agents |
| **Cloud Storage** | Tool wrapping | File processing agents |
| **Cloud Run** | Deployment target | Containerized agents |
| **Pub/Sub** | Event-driven triggers | Async agent invocation |
| **IAM** | Built-in auth | Permission management |

## Decision Tree: When to Use

```
     Should I use Google ADK?
                |
     +----------v----------+
     | Are you deploying   |
     | on GCP / Vertex AI? |
     +---+------------+----+
         |            |
        Yes          No --> LangGraph or provider SDK
         |
     +---v-----------+----+
     | Are you using      |
     | Gemini models      |
     | primarily?         |
     +---+------------+---+
         |            |
        Yes          No --> Consider LangGraph (model agnostic)
         |
     +---v-----------+----+
     | Do you need        |
     | Java or Go agents? |
     | (not just Python)  |
     +---+------------+---+
         |            |
        Yes          No --> Python frameworks have more options
         |
     +---v-------------------+
     | USE Google ADK        |
     +------------------------+
```

## When NOT to Use

1. **Non-GCP deployment**: ADK's value is in GCP integration; without it, use LangGraph
2. **Multi-provider models**: If you use Claude and GPT alongside Gemini, use a framework-agnostic tool
3. **Complex graph patterns**: LangGraph's state machine model is more flexible
4. **Maximum community support**: Smaller community than LangChain ecosystem
5. **HITL with persistence**: LangGraph's checkpointer + interrupt is more mature

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Multi-language (Python/Java/Go) | GCP-centric (less value outside GCP) |
| Native Vertex AI deployment | Gemini-first (other models are second-class) |
| Built-in Google Search grounding | Smaller community than LangChain |
| Managed Agent Engine (no infra) | Less mature than LangGraph |
| A2A protocol for cross-agent comms | Fewer tutorials and examples |
| Enterprise-grade GCP security (IAM) | Less flexible than LangGraph's graph model |

## Real-World Examples

1. **Google Customer Engineering**: Internal agents built on ADK for customer demos and POCs.
2. **Enterprise data agents**: BigQuery + Gemini agents for natural language data analysis on GCP.
3. **Multi-language microservices**: Go/Java backend services with embedded agent capabilities.

## Failure Modes

### 1. GCP Lock-in
Agent logic becomes deeply coupled to GCP services, making migration expensive.
**Mitigation**: Abstract GCP-specific tools behind interfaces. Keep core agent logic portable.

### 2. Agent Engine Cold Starts
Managed deployment may have cold start latency for scale-to-zero configurations.
**Mitigation**: Set min_replicas > 0 for latency-sensitive agents. Use keep-alive health checks.

### 3. A2A Protocol Immaturity
The Agent-to-Agent protocol is new and may have breaking changes.
**Mitigation**: Pin protocol versions. Maintain fallback direct-call patterns.

## Sources and Further Reading

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [Google ADK GitHub](https://github.com/google/adk-python)
- [Vertex AI Agent Builder](https://cloud.google.com/vertex-ai/docs/agents/overview)
- [Agent-to-Agent Protocol (A2A)](https://google.github.io/A2A/)
- [Gemini API Documentation](https://ai.google.dev/docs)
