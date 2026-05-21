# Agent Workspace Design

> Structuring the working memory/scratchpad that agents use during multi-step tasks. The workspace is the agent's "desk" -- organized sections for plans, observations, tool results, and accumulated knowledge.

## What It Is

An agent workspace (or scratchpad) is a structured section of the context window that the agent reads and writes during task execution. It serves as the agent's working memory -- a place to track plans, record observations, accumulate intermediate results, and maintain state across multiple reasoning steps.

Unlike conversation history (which grows linearly), a well-designed workspace is **actively maintained**: sections are updated, summarized, and pruned as the agent works.

Key design goals:
1. **Structured for LLM parsing**: XML/JSON sections the model can reliably read and write
2. **Bounded in size**: Workspace cannot grow unboundedly (context window is finite)
3. **Actively managed**: Stale sections get summarized or evicted
4. **Task-aware**: Different task types use different workspace layouts

## How It Works

### Workspace Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CONTEXT WINDOW                            │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  SYSTEM PROMPT (fixed, ~500-1000 tokens)              │   │
│  │  - Agent identity, capabilities, constraints          │   │
│  │  - Tool descriptions                                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  WORKSPACE (dynamic, actively managed)                │   │
│  │                                                       │   │
│  │  <plan>                                               │   │
│  │    Current task plan with step status                 │   │
│  │  </plan>                                              │   │
│  │                                                       │   │
│  │  <current_step>                                       │   │
│  │    What the agent is doing right now                  │   │
│  │  </current_step>                                      │   │
│  │                                                       │   │
│  │  <observations>                                       │   │
│  │    Recent tool results and findings                   │   │
│  │  </observations>                                      │   │
│  │                                                       │   │
│  │  <accumulated_knowledge>                              │   │
│  │    Facts learned during this session                  │   │
│  │  </accumulated_knowledge>                             │   │
│  │                                                       │   │
│  │  <scratchpad>                                         │   │
│  │    Temporary calculations, draft text                 │   │
│  │  </scratchpad>                                        │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  CONVERSATION HISTORY (append-only)                   │   │
│  │  - User messages + assistant responses                │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  CURRENT USER MESSAGE                                 │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Token Budget Allocation

```
For a 200K context window model:

Fixed overhead:
  System prompt:                    1,000 tokens
  Tool descriptions:                2,000 tokens
  Current user message:               200 tokens
  Safety margin:                    5,000 tokens
  ─────────────────────────────────────────────
  Subtotal fixed:                   8,200 tokens

Dynamic workspace budget:
  Plan section:                       500 tokens (max)
  Current step:                       200 tokens (max)
  Observations (last 5 tool results): 2,500 tokens (max 500 each)
  Accumulated knowledge:            1,000 tokens (max, summarized)
  Scratchpad:                         500 tokens (max)
  ─────────────────────────────────────────────
  Subtotal workspace:              4,700 tokens

Conversation history:
  Remaining budget:              ~187,100 tokens
  (but practically, keeping last 10-20 turns: ~5,000 tokens)

Effective budget for RAG/tool results:  ~182,000 tokens
```

## Production Implementation

### XML-Based Workspace (Recommended for Claude)

```python
from dataclasses import dataclass, field
from typing import Optional
import tiktoken


@dataclass
class WorkspaceSection:
    name: str
    content: str
    max_tokens: int
    priority: int  # Higher = keep longer during eviction


@dataclass
class AgentWorkspace:
    """
    Structured workspace that agents read and write during task execution.
    
    Uses XML format because Claude parses XML very reliably.
    Implements token budget management and automatic summarization.
    """
    
    plan: str = ""
    current_step: str = ""
    observations: list[dict] = field(default_factory=list)
    accumulated_knowledge: list[str] = field(default_factory=list)
    scratchpad: str = ""
    
    # Configuration
    max_observations: int = 5
    max_knowledge_items: int = 10
    max_workspace_tokens: int = 5000
    
    def render(self) -> str:
        """Render workspace as XML for injection into the context."""
        observations_xml = ""
        for obs in self.observations[-self.max_observations:]:
            observations_xml += (
                f"  <observation step=\"{obs.get('step', '?')}\" "
                f"tool=\"{obs.get('tool', 'unknown')}\">\n"
                f"    {obs.get('result', '')}\n"
                f"  </observation>\n"
            )
        
        knowledge_xml = ""
        for item in self.accumulated_knowledge[-self.max_knowledge_items:]:
            knowledge_xml += f"  <fact>{item}</fact>\n"
        
        return f"""<workspace>
<plan>
{self.plan or "No plan created yet."}
</plan>

<current_step>
{self.current_step or "Awaiting task."}
</current_step>

<observations>
{observations_xml or "  No observations yet."}
</observations>

<accumulated_knowledge>
{knowledge_xml or "  No knowledge accumulated yet."}
</accumulated_knowledge>

<scratchpad>
{self.scratchpad or "Empty."}
</scratchpad>
</workspace>"""

    def update_plan(self, plan: str) -> None:
        self.plan = plan

    def set_current_step(self, step: str) -> None:
        self.current_step = step

    def add_observation(self, step: int, tool: str, result: str) -> None:
        """Add a tool result observation. Evicts oldest if over limit."""
        self.observations.append({
            "step": step,
            "tool": tool,
            "result": result[:500],  # Truncate long results
        })
        if len(self.observations) > self.max_observations:
            # Summarize evicted observations into accumulated knowledge
            evicted = self.observations.pop(0)
            self.accumulated_knowledge.append(
                f"[Step {evicted['step']}] {evicted['tool']}: {evicted['result'][:100]}..."
            )

    def add_knowledge(self, fact: str) -> None:
        """Add a learned fact. Deduplicates similar facts."""
        # Simple dedup: check if a similar fact already exists
        for existing in self.accumulated_knowledge:
            if fact.lower()[:50] == existing.lower()[:50]:
                return  # Skip duplicate
        self.accumulated_knowledge.append(fact)
        
        if len(self.accumulated_knowledge) > self.max_knowledge_items:
            self.accumulated_knowledge.pop(0)

    def update_scratchpad(self, content: str) -> None:
        self.scratchpad = content[:500]  # Bounded

    def clear_scratchpad(self) -> None:
        self.scratchpad = ""

    def estimate_tokens(self) -> int:
        """Estimate token count of the rendered workspace."""
        rendered = self.render()
        # Rough estimate: 1 token ≈ 4 characters for English text
        return len(rendered) // 4

    def compact(self, target_tokens: int = None) -> None:
        """
        Compact the workspace to fit within token budget.
        Called when workspace is approaching max_workspace_tokens.
        """
        target = target_tokens or self.max_workspace_tokens

        while self.estimate_tokens() > target:
            # Priority-based eviction:
            # 1. Clear scratchpad
            if self.scratchpad:
                self.scratchpad = ""
                continue

            # 2. Remove oldest observations (summarize to knowledge)
            if len(self.observations) > 2:
                evicted = self.observations.pop(0)
                summary = f"[Step {evicted['step']}] {evicted['tool']}: {evicted['result'][:60]}"
                self.accumulated_knowledge.append(summary)
                continue

            # 3. Truncate accumulated knowledge
            if len(self.accumulated_knowledge) > 3:
                self.accumulated_knowledge.pop(0)
                continue

            # 4. Truncate plan
            if len(self.plan) > 200:
                self.plan = self.plan[:200] + "... (truncated)"
                continue

            break  # Cannot compact further


# --- Agent Loop with Workspace ---

def run_agent_with_workspace(user_query: str) -> str:
    from anthropic import Anthropic
    
    client = Anthropic()
    workspace = AgentWorkspace(max_workspace_tokens=5000)
    
    SYSTEM_PROMPT = """You are a research assistant with a structured workspace.

Your workspace is provided in <workspace> tags. Use it to:
1. Track your plan (update <plan> as you progress)
2. Record key findings (add to <accumulated_knowledge>)
3. Use the scratchpad for temporary work

When you need to update the workspace, output your updates in <workspace_update> tags:
<workspace_update>
  <set_plan>Updated plan text</set_plan>
  <set_step>Current step description</set_step>
  <add_knowledge>New fact learned</add_knowledge>
  <set_scratchpad>Temporary work</set_scratchpad>
</workspace_update>

Always maintain your workspace to stay organized during complex tasks."""

    messages = [{"role": "user", "content": user_query}]
    
    for iteration in range(10):
        # Inject workspace into the system prompt
        full_system = f"{SYSTEM_PROMPT}\n\n{workspace.render()}"
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=full_system,
            messages=messages,
        )
        
        response_text = response.content[0].text
        
        # Parse workspace updates from response
        _apply_workspace_updates(workspace, response_text, iteration)
        
        # Compact workspace if it's getting too large
        if workspace.estimate_tokens() > workspace.max_workspace_tokens * 0.8:
            workspace.compact()
        
        # Check if agent is done
        if response.stop_reason == "end_turn" and not any(
            b.type == "tool_use" for b in response.content
        ):
            return response_text
        
        messages.append({"role": "assistant", "content": response.content})
    
    return "Max iterations reached."


def _apply_workspace_updates(workspace: AgentWorkspace, text: str, step: int):
    """Parse and apply workspace updates from LLM output."""
    import re
    
    plan_match = re.search(r"<set_plan>(.*?)</set_plan>", text, re.DOTALL)
    if plan_match:
        workspace.update_plan(plan_match.group(1).strip())
    
    step_match = re.search(r"<set_step>(.*?)</set_step>", text, re.DOTALL)
    if step_match:
        workspace.set_current_step(step_match.group(1).strip())
    
    for knowledge_match in re.finditer(r"<add_knowledge>(.*?)</add_knowledge>", text, re.DOTALL):
        workspace.add_knowledge(knowledge_match.group(1).strip())
    
    scratch_match = re.search(r"<set_scratchpad>(.*?)</set_scratchpad>", text, re.DOTALL)
    if scratch_match:
        workspace.update_scratchpad(scratch_match.group(1).strip())
```

### JSON-Based Workspace (Alternative)

```python
WORKSPACE_JSON_FORMAT = {
    "plan": {
        "goal": "Research and summarize recent AI agent papers",
        "steps": [
            {"id": 1, "description": "Search for papers", "status": "completed"},
            {"id": 2, "description": "Read abstracts", "status": "in_progress"},
            {"id": 3, "description": "Synthesize findings", "status": "pending"},
        ],
    },
    "current_step": {
        "id": 2,
        "description": "Reading abstracts of 3 papers found",
        "started_at": "2025-01-15T10:30:00Z",
    },
    "observations": [
        {
            "step": 1,
            "tool": "search",
            "result": "Found 3 relevant papers: GoT, Reflexion, ReWOO",
            "timestamp": "2025-01-15T10:29:00Z",
        },
    ],
    "knowledge": [
        "GoT extends ToT with merge/refine operations",
        "Reflexion uses verbal self-reflection for improvement",
    ],
    "scratchpad": "",
}
```

## Decision Tree: Workspace Design Choices

```
    How should I design the agent workspace?
                        │
             ┌──────────▼──────────┐
             │ What format should   │
             │ the workspace use?   │
             └──┬──────────────┬──┘
                │              │
           Claude/Anthropic   OpenAI/Others
                │              │
           Use XML         Use JSON
           (Claude parses  (structured
            XML reliably)   output mode)
                │
       ┌────────▼─────────┐
       │ How many steps     │
       │ will the agent     │
       │ typically take?    │
       └──┬──────────┬────┘
         <5         >10
          │          │
    Minimal       Full workspace
    workspace     with eviction
    (no eviction  and summarization
     needed)      needed
```

## When NOT to Use

1. **Single-turn tasks**: If the agent always completes in one LLM call, a workspace is unnecessary overhead.
2. **Simple chatbots**: Conversation history is sufficient; no need for structured workspace.
3. **Tasks under 3 steps**: The overhead of maintaining workspace sections is not justified.
4. **When context window is small**: On models with 4K-8K context, workspace competes with essential content. Use only on 100K+ models.
5. **Stateless operations**: If each request is independent (no multi-step reasoning), there is nothing to track.

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Organized reasoning (agent stays on track) | Consumes context window tokens |
| Prevents redundant work (knowledge persists) | Requires parsing logic for updates |
| Enables long-running tasks (10+ steps) | LLMs sometimes ignore workspace instructions |
| Debuggable (inspect workspace at any point) | XML/JSON formatting can be fragile |
| Bounded growth (explicit token budgets) | Adds system prompt complexity |
| Enables workspace-aware compaction | Requires careful section prioritization |

## Failure Modes

### 1. Workspace Bloat
Agent adds observations without ever summarizing, filling the context window.
**Mitigation**: Hard token limits per section. Automatic compaction at 80% capacity.

### 2. Workspace Neglect
Agent ignores the workspace entirely and reasons from scratch each step.
**Mitigation**: Explicitly instruct the agent to read and update workspace. Include workspace content in every prompt.

### 3. Parse Errors
Agent outputs malformed XML/JSON workspace updates that break parsing.
**Mitigation**: Graceful fallback -- if parsing fails, keep the previous workspace state. Log parse errors for prompt improvement.

### 4. Stale Information
Knowledge accumulated early in the session becomes outdated as the agent learns more.
**Mitigation**: Tag knowledge with step numbers. Allow the agent to mark items as outdated.

### 5. Section Confusion
Agent writes plan content in the scratchpad or observations in the knowledge section.
**Mitigation**: Provide clear examples of each section in the system prompt. Use strict section descriptions.

## Sources and Further Reading

- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents)
- [Claude Code Architecture](https://docs.anthropic.com/en/docs/claude-code) -- Uses internal workspace/scratchpad patterns
- [Context Engineering - LangChain](https://blog.langchain.dev/context-engineering/)
- [Agent Scratchpad - LangChain Docs](https://python.langchain.com/docs/modules/agents/concepts/#agentscratchpad)
- [Cognitive Architectures for Language Agents (CoALA)](https://arxiv.org/abs/2309.02427) -- Formal framework for agent memory/workspace
