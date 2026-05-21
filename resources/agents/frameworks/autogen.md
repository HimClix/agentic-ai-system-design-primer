# AutoGen (Microsoft)

> AutoGen models agents as conversable entities that talk to each other -- built for multi-agent conversations, code execution, and enterprise deployment. Adopted by BlackRock, Deloitte, and Accenture.

## What It Is

AutoGen is Microsoft's multi-agent framework where agents are defined as `ConversableAgent` instances that communicate through messages. Key concepts:

- **ConversableAgent**: Base class for all agents (can send/receive messages)
- **AssistantAgent**: AI-powered agent (wraps an LLM)
- **UserProxyAgent**: Represents a human or executes code on behalf of a human
- **GroupChat**: Manages multi-agent conversations with turn-taking
- **Code Execution**: Built-in sandboxed code execution (Docker or local)

AutoGen 0.4 (the current stable version) restructured around an event-driven, extensible architecture.

## How It Works

### Architecture

```
     +---------------------+
     |    GroupChat         |
     |    Manager           |
     |    (Turn selection)  |
     +---+----+----+-------+
         |    |    |
    +----v--+ | +--v----+
    |Assist | | |User   |
    |Agent  | | |Proxy  |
    |(LLM)  | | |(Human/|
    |       | | | Code) |
    +-------+ | +-------+
         +----v----+
         |Assist   |
         |Agent 2  |
         |(LLM)    |
         +---------+

Agents converse by sending messages to each other.
GroupChat manager decides who speaks next.
```

## Production Implementation

### Two-Agent Conversation

```python
from autogen import ConversableAgent

# Agent 1: AI Assistant
assistant = ConversableAgent(
    name="assistant",
    system_message="""You are a helpful AI assistant that can write
    and debug Python code. When you write code, include it in
    ```python blocks.""",
    llm_config={
        "config_list": [{
            "model": "claude-sonnet-4-20250514",
            "api_key": "...",
            "api_type": "anthropic"
        }]
    }
)

# Agent 2: User proxy with code execution
user_proxy = ConversableAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # or "ALWAYS" for HITL
    code_execution_config={
        "work_dir": "coding_workspace",
        "use_docker": True  # Sandboxed execution
    },
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", "")
)

# Start conversation
user_proxy.initiate_chat(
    assistant,
    message="Write a Python script that fetches the top 5 HN stories and saves to CSV."
)
```

### GroupChat (Multi-Agent)

```python
from autogen import GroupChat, GroupChatManager

# Define specialist agents
planner = ConversableAgent(
    name="planner",
    system_message="You break tasks into steps. Reply with a numbered plan.",
    llm_config=llm_config
)

coder = ConversableAgent(
    name="coder",
    system_message="You write Python code based on plans. Include code in ```python blocks.",
    llm_config=llm_config
)

reviewer = ConversableAgent(
    name="reviewer",
    system_message="You review code for bugs and suggest improvements.",
    llm_config=llm_config
)

executor = ConversableAgent(
    name="executor",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": True}
)

# Create group chat
group_chat = GroupChat(
    agents=[planner, coder, reviewer, executor],
    messages=[],
    max_round=12,
    speaker_selection_method="auto"  # LLM selects next speaker
)

manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config
)

# Start the group conversation
executor.initiate_chat(
    manager,
    message="Build a REST API that returns weather data for a given city."
)
```

### Enterprise Adoption

```
Known Production Users:
+-------------------+-------------------------------+
| Company           | Use Case                      |
+-------------------+-------------------------------+
| BlackRock         | Financial analysis agents     |
| Deloitte          | Consulting workflow automation|
| Accenture         | Enterprise process automation |
| Microsoft (internal)| Code generation, testing    |
| Walmart           | Supply chain optimization     |
+-------------------+-------------------------------+
```

## Decision Tree: When to Use

```
     Should I use AutoGen?
                |
     +----------v----------+
     | Do you need agents  |
     | to CONVERSE with    |
     | each other (chat)?  |
     +---+------------+----+
         |            |
        Yes          No --> LangGraph or CrewAI
         |
     +---v-----------+----+
     | Do you need        |
     | sandboxed CODE     |
     | EXECUTION?         |
     +---+------------+---+
         |            |
        Yes          No --> CrewAI might be simpler
         |
     +---v-----------+----+
     | Are you in a       |
     | Microsoft / Azure  |
     | ecosystem?         |
     +---+------------+---+
         |            |
        Yes          No --> Still viable, but evaluate LangGraph
         |
     +---v-------------------+
     | USE AutoGen           |
     +------------------------+
```

## When NOT to Use

1. **Deterministic workflows**: AutoGen's conversation model is inherently non-deterministic
2. **Fine-grained flow control**: LangGraph's explicit edges give more control
3. **Non-Python only**: AutoGen is primarily Python (C# version exists for .NET)
4. **Simple single-agent tasks**: The conversational framework is overkill
5. **Streaming-first UIs**: AutoGen's streaming support is less mature than LangGraph's

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Natural multi-agent conversation model | Less deterministic than graph-based frameworks |
| Built-in code execution (Docker sandboxed) | GroupChat turn-taking can be unpredictable |
| Enterprise backing (Microsoft) | API has undergone major breaking changes (0.2->0.4) |
| Strong Azure OpenAI integration | Smaller community than LangChain ecosystem |
| Handles human-agent-agent conversations | Configuration-heavy for complex setups |

## Failure Modes

### 1. GroupChat Monopolization
One agent dominates the conversation, preventing others from contributing.
**Mitigation**: Configure `speaker_selection_method` and `max_consecutive_auto_reply`.

### 2. Infinite Conversation Loops
Agents keep responding to each other without converging on a solution.
**Mitigation**: Set `max_round` and define clear `is_termination_msg` functions.

### 3. Code Execution Escapes
Malicious or buggy code escapes the Docker sandbox.
**Mitigation**: Always use `use_docker=True` in production. Restrict network access.

## Sources and Further Reading

- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [AutoGen GitHub](https://github.com/microsoft/autogen)
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) (Wu et al., 2023)
- [AutoGen 0.4 Migration Guide](https://microsoft.github.io/autogen/docs/migration-guide)
