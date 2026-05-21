# Schema Design: Tool Descriptions as LLM Documentation
> The JSON Schema description field is the only documentation your LLM will ever read. Write it like you're onboarding a careful but literal junior developer.

## What It Is

Schema design for agent tools is the practice of writing JSON Schema definitions that serve as the LLM's primary (and often only) documentation for understanding what a tool does, what parameters it needs, and how to use it correctly. The `description` fields in your schema are not metadata -- they are the actual instruction set the LLM uses to decide when and how to call your tool.

A well-designed schema reduces tool selection errors by 40-60% compared to minimal schemas. This is the highest-leverage improvement you can make to agent reliability without changing any code.

## How It Works

### What the LLM Actually Sees

When you register tools with an LLM, the model receives the JSON Schema as part of its system context. Here's what the model processes:

```json
{
  "type": "function",
  "function": {
    "name": "search_orders",         <-- First thing LLM reads
    "description": "...",             <-- LLM's primary decision input
    "parameters": {
      "type": "object",
      "properties": {
        "start_date": {
          "type": "string",
          "description": "..."       <-- LLM's parameter guide
        }
      },
      "required": ["start_date"]     <-- LLM knows what's mandatory
    }
  }
}
```

The LLM uses this to answer three questions:
1. **Should I call this tool?** (based on `name` + function `description`)
2. **What arguments should I pass?** (based on parameter `description` + `type` + `enum`)
3. **What will I get back?** (inferred from function `description`)

### The Description Hierarchy

```
Tool name:         3-5 words, verb_noun format    -> "Which tool?"
Tool description:  2-4 sentences                   -> "When to use it?"
Parameter name:    1-3 words, snake_case            -> "What field?"
Parameter desc:    1-2 sentences + example          -> "What value?"
Enum values:       Self-documenting strings         -> "Which option?"
```

## Production Implementation

### Template: Complete Tool Schema

```python
"""Tool schema design patterns for LLM consumption."""


def create_order_tool_schema() -> dict:
    """Example of a production-quality tool schema."""
    return {
        "type": "function",
        "function": {
            "name": "create_order",
            # RULE 1: Description starts with what the tool does,
            # then when to use it, then important constraints.
            "description": (
                "Create a new order for a customer. "
                "Use this when the user wants to place an order or purchase items. "
                "Requires at least one item. Maximum 50 items per order. "
                "Returns the order ID and total amount."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        # RULE 2: Parameter descriptions include format and example.
                        "description": (
                            "The unique customer identifier. "
                            "Format: 'cust_' followed by alphanumeric characters. "
                            "Example: 'cust_abc123def456'"
                        ),
                    },
                    "items": {
                        "type": "array",
                        # RULE 3: Complex types get detailed structure descriptions.
                        "description": (
                            "List of items to include in the order. "
                            "Each item needs a product_id and quantity. "
                            "At least 1 item required, maximum 50 items."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "product_id": {
                                    "type": "string",
                                    "description": "Product identifier. Format: 'prod_xxx'. Example: 'prod_widget_001'",
                                },
                                "quantity": {
                                    "type": "integer",
                                    "description": "Number of units. Must be 1-999.",
                                    "minimum": 1,
                                    "maximum": 999,
                                },
                            },
                            "required": ["product_id", "quantity"],
                        },
                        "minItems": 1,
                        "maxItems": 50,
                    },
                    "shipping_address": {
                        "type": "object",
                        "description": "Delivery address. Required for physical products, ignored for digital products.",
                        "properties": {
                            "street": {
                                "type": "string",
                                "description": "Street address including number. Example: '123 Main Street, Apt 4B'",
                            },
                            "city": {"type": "string", "description": "City name"},
                            "state": {"type": "string", "description": "State or province code. Example: 'CA', 'NY'"},
                            "zip_code": {
                                "type": "string",
                                # RULE 4: Call out format gotchas.
                                "description": "Postal/ZIP code as a STRING, not a number. Example: '94102', '10001'",
                            },
                            "country": {
                                "type": "string",
                                "description": "Two-letter ISO country code. Default: 'US'. Example: 'US', 'CA', 'GB'",
                                "default": "US",
                            },
                        },
                        "required": ["street", "city", "state", "zip_code"],
                    },
                    "currency": {
                        "type": "string",
                        # RULE 5: Enums + description of what each value means.
                        "enum": ["usd", "eur", "gbp", "inr"],
                        "description": (
                            "Three-letter ISO currency code in lowercase. "
                            "Determines pricing and payment processing currency. "
                            "Default: 'usd'"
                        ),
                        "default": "usd",
                    },
                    "notes": {
                        "type": "string",
                        # RULE 6: Mark optional parameters clearly.
                        "description": "Optional free-text notes for the order. Max 500 characters. Leave empty if not needed.",
                    },
                },
                "required": ["customer_id", "items"],
                # RULE 7: additionalProperties false prevents LLM from inventing fields.
                "additionalProperties": False,
            },
        },
    }
```

### The 8 Rules of Schema Description Writing

```python
"""Schema description rules with examples."""

SCHEMA_RULES = {
    "1_start_with_action": {
        "bad": "This tool is for orders.",
        "good": "Create a new order for a customer.",
        "why": "Start with the verb. LLMs pattern-match on the first few words.",
    },
    "2_include_when_to_use": {
        "bad": "Creates orders.",
        "good": "Create a new order. Use this when the user wants to purchase items or place a new order.",
        "why": "The LLM needs to distinguish this from similar tools (update_order, cancel_order).",
    },
    "3_state_constraints": {
        "bad": "Search for users.",
        "good": "Search for users by name or email. Returns max 50 results. Use get_user for single user lookup by ID.",
        "why": "Constraints prevent the LLM from expecting impossible behavior.",
    },
    "4_include_format_and_example": {
        "bad": "The customer ID.",
        "good": "Customer identifier. Format: 'cust_' prefix + alphanumeric. Example: 'cust_abc123'",
        "why": "LLMs are 3x more likely to produce correct format when given an example.",
    },
    "5_call_out_gotchas": {
        "bad": "The zip code.",
        "good": "ZIP code as a STRING (not a number). Leading zeros matter. Example: '01234'",
        "why": "LLMs default to number types for numeric-looking strings.",
    },
    "6_distinguish_from_siblings": {
        "bad": "Get user information.",
        "good": "Get a single user's full profile by their user ID. For searching multiple users by name/email, use search_users instead.",
        "why": "Explicit cross-references reduce wrong-tool selection.",
    },
    "7_mark_optional_clearly": {
        "bad": "Notes for the order.",
        "good": "Optional free-text notes. Leave empty or omit if no notes needed.",
        "why": "LLMs sometimes feel compelled to fill every field. Say 'optional' explicitly.",
    },
    "8_describe_return_value": {
        "bad": "Creates an order.",
        "good": "Creates an order. Returns the order_id, total_amount, and estimated_delivery_date.",
        "why": "The LLM needs to know what data it will get back to plan next steps.",
    },
}
```

### Schema Validation Helper

```python
"""Validate tool schemas for LLM-readability before deployment."""
import json
from typing import Optional


class SchemaValidator:
    """Validate that tool schemas follow LLM-friendly best practices."""

    def validate(self, schema: dict) -> list[str]:
        """Return list of warnings for schema quality issues."""
        warnings = []
        func = schema.get("function", schema)
        
        # Check function name
        name = func.get("name", "")
        if not name:
            warnings.append("CRITICAL: Missing function name")
        elif "_" not in name:
            warnings.append(f"WARNING: Name '{name}' should be verb_noun format (e.g., 'get_user')")
        
        # Check function description
        desc = func.get("description", "")
        if not desc:
            warnings.append("CRITICAL: Missing function description")
        elif len(desc) < 20:
            warnings.append(f"WARNING: Description too short ({len(desc)} chars). Add when-to-use and constraints.")
        elif len(desc) > 500:
            warnings.append(f"WARNING: Description too long ({len(desc)} chars). LLMs lose focus after ~300 chars.")
        
        if desc and not desc[0].isupper():
            warnings.append("STYLE: Description should start with a capital letter")
        
        # Check parameters
        params = func.get("parameters", {})
        properties = params.get("properties", {})
        required = set(params.get("required", []))
        
        for param_name, param_schema in properties.items():
            param_desc = param_schema.get("description", "")
            
            if not param_desc:
                warnings.append(f"CRITICAL: Parameter '{param_name}' has no description")
            elif len(param_desc) < 10:
                warnings.append(f"WARNING: Parameter '{param_name}' description too short. Add format and example.")
            
            # Check for example in description
            if param_desc and "example" not in param_desc.lower() and "e.g." not in param_desc.lower():
                if param_schema.get("type") == "string" and param_name not in {"query", "message", "body", "notes"}:
                    warnings.append(f"SUGGESTION: Parameter '{param_name}' should include a format example")
            
            # Check string params that look like they need format
            if param_schema.get("type") == "string" and any(
                kw in param_name for kw in ["date", "time", "id", "code", "url", "email", "phone"]
            ):
                if "format" not in param_desc.lower() and "example" not in param_desc.lower():
                    warnings.append(
                        f"WARNING: Parameter '{param_name}' looks like it needs a format specification"
                    )
        
        # Check additionalProperties
        if "additionalProperties" not in params:
            warnings.append(
                "SUGGESTION: Add 'additionalProperties: false' to prevent LLM from inventing extra fields"
            )
        
        return warnings


# Usage
validator = SchemaValidator()
warnings = validator.validate(create_order_tool_schema())
for w in warnings:
    print(w)
```

## Decision Tree: Writing a Tool Description

```
Step 1: Start with the ACTION
  "Create a new ___" / "Search for ___" / "Delete ___"

Step 2: Add WHEN TO USE
  "Use this when the user wants to ___"
  "Use this to ___ (not for ___, use ___ instead)"

Step 3: Add CONSTRAINTS
  "Maximum 100 results" / "Only works for active users"
  "Requires ___ to be set up first"

Step 4: Add RETURN VALUE hint
  "Returns the ___ and ___"

Step 5: For each PARAMETER:
  - What is it? (1 sentence)
  - What format? (specific pattern)
  - Example value (concrete)
  - Edge case? (gotcha warning)

Step 6: VALIDATE
  - Run SchemaValidator
  - Test with LLM: "When would you use this tool?"
  - Check: does the LLM understand the distinction from similar tools?
```

## When NOT to Over-Document

- **Boolean parameters**: `"is_active": {"type": "boolean", "description": "Filter by active status"}` -- don't over-explain
- **Self-evident parameters**: `"message": {"type": "string", "description": "The message text"}` -- obvious from name
- **Internal implementation details**: Don't explain how the tool works internally

## Tradeoffs

| Aspect | Detailed Descriptions | Minimal Descriptions |
|--------|----------------------|---------------------|
| Tool selection accuracy | 90-95% | 60-70% |
| Token cost per tool | ~80-120 tokens | ~20-30 tokens |
| With 20 tools, context cost | ~2,000 tokens | ~500 tokens |
| Maintenance burden | Must update descriptions | Less to maintain |
| Parameter format accuracy | 90%+ with examples | 50-60% without |
| Debugging ease | Clear from schema what's expected | Must read code |

## Real-World Examples

### Bad Schema (Actual Production Failure)
```json
{
  "name": "db",
  "description": "Database operations",
  "parameters": {
    "properties": {
      "q": {"type": "string"},
      "t": {"type": "string"},
      "d": {"type": "object"}
    }
  }
}
```
**Result**: LLM had no idea what `q`, `t`, or `d` meant. Selected wrong tool 70% of the time.

### Good Schema (Fixed Version)
```json
{
  "name": "query_database",
  "description": "Run a read-only SQL query against the application database. Returns rows as JSON objects. Maximum 1000 rows per query. Use for reading data only -- for inserts/updates, use insert_record or update_record.",
  "parameters": {
    "properties": {
      "sql_query": {
        "type": "string",
        "description": "A SELECT SQL query. Must be read-only (no INSERT, UPDATE, DELETE). Example: \"SELECT id, name, email FROM users WHERE status = 'active' LIMIT 10\""
      },
      "timeout_seconds": {
        "type": "integer",
        "description": "Query timeout in seconds. Default 10, maximum 30. Increase for complex queries.",
        "default": 10,
        "maximum": 30
      }
    },
    "required": ["sql_query"],
    "additionalProperties": false
  }
}
```
**Result**: Tool selection accuracy improved from 70% to 95%.

## Failure Modes

| Failure | Cause | Impact | Fix |
|---------|-------|--------|-----|
| Wrong tool selected | Ambiguous description | Agent calls update instead of create | Add "Use this when..." + "Not for..." |
| Wrong parameter format | No example in description | Dates as "January 15" instead of "2024-01-15" | Add concrete example values |
| Extra fields invented | No additionalProperties:false | LLM adds fields that cause errors | Set additionalProperties to false |
| Confused by similar tools | No cross-references | Picks get_user when search_users needed | Add "For X, use Y instead" |
| Skips required parameters | Required not clearly marked | Tool fails on missing input | Keep required array accurate |

## Source(s) and Further Reading

- OpenAI, "Function Calling Guide" (2024) -- JSON Schema best practices
- Anthropic, "Tool Use Documentation" (2024) -- description writing guidelines
- Berkeley Function-Calling Leaderboard (BFCL) -- accuracy benchmarks by schema quality
- JSON Schema specification (2020-12) -- formal schema features
- "9 Practices for Production Agentic Workflows" (arXiv, 2025) -- schema design as top practice
