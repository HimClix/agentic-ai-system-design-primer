# Single Responsibility: One Tool = One Action
> LLMs select the right tool 90%+ of the time when each tool does exactly one thing. Multi-action tools drop accuracy below 60%.

## What It Is

The single responsibility principle for agent tools means every tool performs exactly one well-defined action. A tool either reads data or writes data. It either creates a resource or deletes one. It never does "it depends on the `action` parameter."

This is the most impactful tool design principle because it directly affects the LLM's ability to select the correct tool. When a tool does multiple things based on a mode parameter, the LLM must reason about both "which tool?" AND "which mode?" simultaneously -- a compound decision that dramatically increases error rates.

## How It Works

### The Anti-Pattern: Multi-Action Tools

```python
# BAD: One tool that does everything
def manage_database(action: str, table: str, data: dict = None, 
                    query: str = None, record_id: str = None) -> dict:
    """Manage database operations.
    
    Args:
        action: One of "query", "insert", "update", "delete"
        table: Table name
        data: Data for insert/update operations
        query: SQL query for query operations
        record_id: Record ID for update/delete operations
    """
    if action == "query":
        return db.execute(query)
    elif action == "insert":
        return db.insert(table, data)
    elif action == "update":
        return db.update(table, record_id, data)
    elif action == "delete":
        return db.delete(table, record_id)
```

**Why this fails in production:**

1. **Parameter confusion**: The LLM passes `data` when calling with `action="query"`, or forgets `record_id` when calling `action="delete"`
2. **Description overload**: One `description` field must explain 4 different behaviors
3. **Validation complexity**: Which parameters are required depends on which action -- LLMs struggle with conditional requirements
4. **Error attribution**: When it fails, was it the wrong action, the wrong parameters, or the wrong tool entirely?

### The Correct Pattern: Split Into Specific Tools

```python
# GOOD: Four focused tools

def query_database(table: str, filters: dict = None, limit: int = 100) -> dict:
    """Query records from a database table.
    
    Use this to READ data. Returns matching records.
    This tool never modifies data.
    
    Args:
        table: Name of the database table (e.g., "users", "orders")
        filters: Key-value pairs to filter results (e.g., {"status": "active"})
        limit: Maximum number of records to return. Default 100, max 1000.
    """
    pass

def insert_record(table: str, data: dict) -> dict:
    """Insert a single new record into a database table.
    
    Use this to CREATE new data. Returns the created record with its ID.
    This tool is idempotent when called with the same data -- 
    duplicates are rejected.
    
    Args:
        table: Name of the database table (e.g., "users", "orders")  
        data: The record data as key-value pairs (e.g., {"name": "Alice", "email": "alice@example.com"})
    """
    pass

def update_record(table: str, record_id: str, updates: dict) -> dict:
    """Update an existing record in a database table.
    
    Use this to MODIFY existing data. Returns the updated record.
    Only the fields provided in 'updates' are changed; other fields are preserved.
    
    Args:
        table: Name of the database table
        record_id: The unique ID of the record to update
        updates: Key-value pairs of fields to update (e.g., {"status": "inactive"})
    """
    pass

def delete_record(table: str, record_id: str) -> dict:
    """Permanently delete a record from a database table.
    
    WARNING: This action is irreversible. Use only when the user 
    explicitly requests deletion. Consider update_record with 
    {"status": "deleted"} for soft deletes.
    
    Args:
        table: Name of the database table
        record_id: The unique ID of the record to delete
    """
    pass
```

## Why LLMs Struggle With Multi-Action Tools

### 1. Tool Selection Is a Classification Problem

When the LLM sees a list of tools, it performs a classification:
- Input: user intent + tool descriptions
- Output: which tool to call

With single-responsibility tools, this is a 1-of-N classification where N is the number of tools, and each tool name + description is distinctive.

With multi-action tools, the LLM must do two classifications:
1. Which tool? (less distinctive because the tool is generic)
2. Which action parameter value?

This compounds the error rate. If tool selection is 95% accurate and action selection is 90% accurate, the combined accuracy is 0.95 x 0.90 = 85.5%.

### 2. Parameter Confusion

Multi-action tools have parameters that are conditionally required:

```
action="query"  -> requires: query,    ignores: data, record_id
action="insert" -> requires: data,     ignores: query, record_id  
action="delete" -> requires: record_id, ignores: query, data
```

JSON Schema cannot express conditional requirements cleanly. The LLM sees all parameters and doesn't know which subset to use. Result: it passes extra parameters, misses required ones, or combines them incorrectly.

### 3. Description Length Limits

A tool description trying to explain 4 behaviors is 4x longer and 4x more confusing. The LLM's attention is diluted across multiple behaviors instead of focusing on one.

## Production Implementation

```python
"""Single-responsibility tool factory with shared infrastructure."""
from typing import Any, Optional, Callable
from pydantic import BaseModel, Field
import functools


class ToolConfig(BaseModel):
    """Configuration for a single-responsibility tool."""
    name: str = Field(..., description="Tool name, verb_noun format")
    description: str = Field(..., description="What this tool does (1-2 sentences)")
    category: str = Field(default="general", description="Tool category for grouping")
    is_destructive: bool = Field(default=False, description="Whether this modifies/deletes data")
    requires_confirmation: bool = Field(default=False, description="Whether to confirm before executing")
    timeout_seconds: int = Field(default=30, ge=1, le=300)


def single_responsibility_tool(config: ToolConfig):
    """Decorator that enforces single-responsibility tool patterns."""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(**kwargs):
            # Enforce: no 'action' or 'mode' parameters
            if "action" in kwargs or "mode" in kwargs or "operation" in kwargs:
                raise ValueError(
                    f"Tool '{config.name}' received a mode/action parameter. "
                    f"Single-responsibility tools should not have action selectors."
                )

            # Add metadata to result
            result = await func(**kwargs)
            return {
                "tool": config.name,
                "category": config.category,
                "result": result,
            }

        # Attach config for schema generation
        wrapper.tool_config = config
        wrapper.is_tool = True
        return wrapper
    return decorator


# Usage:

@single_responsibility_tool(ToolConfig(
    name="get_customer",
    description="Retrieve a customer's profile by their ID. Returns name, email, plan, and account status.",
    category="customers",
    is_destructive=False,
))
async def get_customer(customer_id: str) -> dict:
    """Retrieve a single customer by ID."""
    customer = await db.customers.find_one({"_id": customer_id})
    if not customer:
        return {"error": "Customer not found", "error_type": "not_found"}
    return {"customer": customer}


@single_responsibility_tool(ToolConfig(
    name="update_customer_email",
    description="Update a customer's email address. Sends a verification email to the new address.",
    category="customers",
    is_destructive=False,
))
async def update_customer_email(customer_id: str, new_email: str) -> dict:
    """Update a single field: the customer's email."""
    result = await db.customers.update_one(
        {"_id": customer_id},
        {"$set": {"email": new_email, "email_verified": False}},
    )
    if result.modified_count == 0:
        return {"error": "Customer not found", "error_type": "not_found"}
    return {"updated": True, "verification_sent_to": new_email}


@single_responsibility_tool(ToolConfig(
    name="cancel_customer_subscription",
    description="Cancel a customer's subscription immediately. This is irreversible. "
                "Use only when the customer explicitly requests cancellation.",
    category="customers",
    is_destructive=True,
    requires_confirmation=True,
))
async def cancel_customer_subscription(customer_id: str, reason: str) -> dict:
    """Cancel a subscription. Destructive, requires confirmation."""
    result = await billing.cancel(customer_id, reason=reason)
    return {"cancelled": True, "effective_date": result.effective_date}
```

## Decision Tree: When to Split a Tool

```
Does your tool have an 'action', 'mode', or 'operation' parameter?
  |
  YES --> Split it immediately. One tool per action value.
  |
  NO --> Does your tool do different things based on input combinations?
           |
           YES --> Split by behavior. Each behavior = one tool.
           |
           NO --> Does the description use "or" more than once?
                    |
                    YES --> It's probably doing too much. Consider splitting.
                    |
                    NO --> Single responsibility is satisfied.
```

## When NOT to Split

Not every operation needs its own tool. These are fine as single tools:

- **CRUD with uniform parameters**: A `search_records(table, filters)` that always takes the same shape is fine
- **Read operations with optional filters**: `list_orders(status?, date_range?, customer_id?)` is one behavior (listing) with optional filters
- **Format variations**: `export_report(format="pdf"|"csv")` is one action (export) with a format choice

The key question: **does the set of required parameters change based on other parameter values?** If yes, split.

## Tradeoffs

| Aspect | Split (Single Responsibility) | Combined (Multi-Action) |
|--------|-------------------------------|------------------------|
| Tool selection accuracy | 90-95% | 55-70% |
| Schema token cost | Higher (more tools) | Lower (fewer tools) |
| Maintenance | More files/functions | Fewer files |
| Error messages | Specific to the action | Generic |
| Testing | Clear unit test per tool | Complex conditional tests |
| LLM context window | ~50 tokens per tool schema | ~120 tokens for multi-action |
| Onboarding new tools | Easy (copy pattern) | Hard (add to switch statement) |

## Real-World Examples

### Slack Integration
```
BAD:  slack(action, channel, message, user, reaction, ...)
GOOD: send_slack_message(channel_id, text)
      add_slack_reaction(channel_id, timestamp, emoji)
      list_slack_channels(limit)
      get_slack_thread(channel_id, thread_ts)
```

### File Operations
```
BAD:  file_manager(operation, path, content, destination, ...)
GOOD: read_file(path)
      write_file(path, content)
      move_file(source_path, destination_path)
      delete_file(path)
      list_directory(path)
```

### Payment Processing
```
BAD:  process_payment(type, amount, currency, customer, refund_id, ...)
GOOD: create_payment_intent(amount, currency, customer_id)
      confirm_payment(payment_intent_id, payment_method_id)
      refund_payment(payment_id, amount, reason)
      get_payment_status(payment_id)
```

## Failure Modes

| Failure | Cause | Impact | Prevention |
|---------|-------|--------|------------|
| Wrong action selected | Multi-action tool, LLM picks wrong mode | Wrong data modified/deleted | Split into single-action tools |
| Extra parameters passed | LLM includes params for wrong action | API errors or unexpected behavior | Strict schema per tool |
| Missing required params | Conditional requirements confuse LLM | Tool execution fails | Each tool has fixed required set |
| Ambiguous tool match | Tool names too similar after splitting | LLM picks wrong sibling tool | Differentiate names and descriptions clearly |

## Source(s) and Further Reading

- Anthropic, "Building Effective Agents" (2024) -- tool granularity recommendations
- "9 Practices for Production Agentic Workflows" (arXiv, 2025) -- single-responsibility as top practice
- OpenAI Function Calling Guide -- schema design patterns
- Berkeley Function-Calling Leaderboard -- accuracy data on tool selection
