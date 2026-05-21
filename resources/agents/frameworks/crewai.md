# CrewAI

> CrewAI is the highest-level agent framework -- define agents by their role, backstory, and goals, then let them collaborate in a "crew." Production without the graph boilerplate.

## What It Is

CrewAI is a role-based multi-agent framework that models agents as team members with defined roles, goals, and backstories. Instead of defining execution graphs, you define:
- **Agents**: Characters with roles and expertise
- **Tasks**: Work items assigned to agents
- **Crews**: Teams that execute tasks with a process (sequential, hierarchical, or consensual)
- **Flows**: Higher-level orchestration of multiple crews

CrewAI abstracts away the graph/state machine complexity of LangGraph in favor of a more intuitive team metaphor.

## How It Works

### Architecture

```
     +------------------+
     |      CREW        |
     |  (Process type:  |
     |   sequential /   |
     |   hierarchical)  |
     +---+----+----+----+
         |    |    |
    +----v-+  |  +-v----+
    |Agent 1| |  |Agent 3|
    |Role:  | |  |Role:  |
    |Researcher  |Writer |
    |Goal:  | |  |Goal:  |
    |Find   | |  |Draft  |
    |data   | |  |report |
    +-------+ |  +-------+
         +-----v-----+
         |  Agent 2   |
         |  Role:     |
         |  Analyst   |
         |  Goal:     |
         |  Analyze   |
         |  findings  |
         +-----------+
```

## Production Implementation

```python
from crewai import Agent, Task, Crew, Process
from crewai.tools import SerperDevTool, ScrapeWebsiteTool

# -- Define Agents --
researcher = Agent(
    role="Senior Research Analyst",
    goal="Find comprehensive market data on the given topic",
    backstory="""You are an expert research analyst with 15 years of
    experience in market research. You are known for your thorough
    analysis and ability to find obscure but relevant data points.""",
    tools=[SerperDevTool(), ScrapeWebsiteTool()],
    llm="claude-sonnet-4-20250514",
    verbose=True,
    memory=True,
    max_iter=5
)

analyst = Agent(
    role="Data Analyst",
    goal="Analyze research findings and extract actionable insights",
    backstory="""You are a quantitative analyst who excels at finding
    patterns in data and translating them into business insights.""",
    llm="claude-sonnet-4-20250514",
    verbose=True
)

writer = Agent(
    role="Report Writer",
    goal="Create a polished executive report from the analysis",
    backstory="""You are an experienced business writer who creates
    clear, compelling executive reports for C-level audiences.""",
    llm="claude-sonnet-4-20250514",
    verbose=True
)

# -- Define Tasks --
research_task = Task(
    description="""Research the current state of {topic}.
    Focus on: market size, growth rate, key players, and trends.
    Provide at least 5 data points with sources.""",
    expected_output="Structured research findings with data points and sources",
    agent=researcher
)

analysis_task = Task(
    description="""Analyze the research findings and identify:
    1. Top 3 opportunities
    2. Top 3 risks
    3. Recommended strategy""",
    expected_output="Structured analysis with opportunities, risks, and strategy",
    agent=analyst,
    context=[research_task]  # Receives output from research_task
)

report_task = Task(
    description="""Write an executive report that includes:
    - Executive summary (1 paragraph)
    - Key findings (bullet points)
    - Analysis and recommendations
    - Conclusion""",
    expected_output="A polished 2-page executive report",
    agent=writer,
    context=[research_task, analysis_task]  # Receives both prior outputs
)

# -- Create Crew --
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, report_task],
    process=Process.sequential,  # Tasks execute in order
    verbose=True,
    memory=True  # Enable crew-level memory
)

# -- Execute --
result = crew.kickoff(inputs={"topic": "generative AI in healthcare"})
print(result)
```

### CrewAI Flows (Multi-Crew Orchestration)

```python
from crewai.flow.flow import Flow, listen, start

class ContentPipeline(Flow):
    """Orchestrate multiple crews in a flow."""

    @start()
    def research_phase(self):
        """First phase: research crew gathers data."""
        research_crew = build_research_crew()
        result = research_crew.kickoff(inputs=self.state)
        self.state["research_results"] = result.raw
        return result

    @listen(research_phase)
    def analysis_phase(self, research_result):
        """Second phase: analysis crew processes research."""
        analysis_crew = build_analysis_crew()
        result = analysis_crew.kickoff(inputs=self.state)
        self.state["analysis_results"] = result.raw
        return result

    @listen(analysis_phase)
    def writing_phase(self, analysis_result):
        """Third phase: writing crew produces output."""
        writing_crew = build_writing_crew()
        result = writing_crew.kickoff(inputs=self.state)
        return result

# Run the flow
pipeline = ContentPipeline()
final = pipeline.kickoff()
```

## Decision Tree: When to Use

```
     Should I use CrewAI?
                |
     +----------v----------+
     | Do you think in     |
     | terms of ROLES and  |
     | TEAM COLLABORATION? |
     +---+------------+----+
         |            |
        Yes          No --> Consider LangGraph (graph-first thinking)
         |
     +---v-----------+----+
     | Do you need fine-   |
     | grained control     |
     | over execution      |
     | flow / state?       |
     +---+------------+---+
         |            |
        Yes --> LangGraph  No
                           |
     +---v-----------+----+
     | Is rapid                |
     | prototyping the         |
     | priority?               |
     +---+------------+---+
         |            |
        Yes          No --> Evaluate LangGraph for production robustness
         |
     +---v-------------------+
     | USE CrewAI            |
     +------------------------+
```

## When NOT to Use

1. **Complex conditional logic**: If your agent needs sophisticated branching, LangGraph's explicit edges are better
2. **Custom state management**: CrewAI's state is opinionated; LangGraph gives full control
3. **Low-level optimization**: CrewAI's abstractions add overhead that LangGraph avoids
4. **Non-Python environments**: CrewAI is Python-only
5. **Fine-grained streaming**: LangGraph's streaming is more granular

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Most intuitive API (role-based) | Less control over execution flow |
| Fastest time-to-prototype | Opinionated state management |
| Built-in memory and delegation | Harder to debug than explicit graphs |
| Flows for multi-crew orchestration | Smaller ecosystem than LangChain/LangGraph |
| Good documentation and community | Less enterprise adoption than LangGraph |
| Role-based prompting improves output quality | Abstraction can hide performance issues |

## Real-World Examples

1. **Content marketing pipeline**: Research agent finds trends, analyst identifies angles, writer produces content.
2. **Code review team**: Security auditor, performance reviewer, and style checker each review code from their perspective.
3. **Customer onboarding**: Qualification agent, KYC agent, and setup agent handle different onboarding stages.

## Failure Modes

### 1. Role Confusion
Agent backstories are too similar, causing overlapping or conflicting outputs.
**Mitigation**: Make roles sharply distinct. Test each agent in isolation before combining.

### 2. Context Bloat
Sequential process passes full output from every prior task, growing context.
**Mitigation**: Use `context=` to select only relevant prior tasks, not all of them.

### 3. Agent Delegation Loops
Agent A delegates to Agent B who delegates back to Agent A.
**Mitigation**: Disable delegation for agents that should not delegate. Set max_iter.

## Sources and Further Reading

- [CrewAI Documentation](https://docs.crewai.com/)
- [CrewAI GitHub](https://github.com/crewAIInc/crewAI)
- [CrewAI Flows Documentation](https://docs.crewai.com/concepts/flows)
