# Long-Term Memory for Agents

> Long-term memory is what makes an agent remember your name, your preferences, and the fact that last time it suggested PostgreSQL, you said you prefer MySQL.

## What It Is

Long-term memory is the durable storage layer that persists information across conversations, sessions, and even across different users. It is the agent equivalent of semantic memory in cognitive science -- the store of facts, concepts, and relationships that the agent has accumulated over time.

Long-term memory answers questions like:
- "What does this user prefer?" (preferences)
- "What happened last time?" (historical facts)
- "How is entity A related to entity B?" (relationships)
- "When did this fact become true/false?" (temporal knowledge)

### Three Storage Paradigms

| Paradigm | Storage | Query Style | Best For |
|---|---|---|---|
| **Vector Store** | Embeddings + metadata | Semantic similarity search | Facts, preferences, unstructured knowledge |
| **Knowledge Graph** | Nodes + edges + properties | Traversal, relationship queries | Entity relationships, temporal reasoning |
| **Key-Value Store** | Simple key→value pairs | Exact lookup | Session data, user profiles, flags |

Most production systems use a **hybrid** of all three.

## How It Works

### Architecture: The Dual-Layer Production Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT REQUEST                                │
│                                                                 │
│  "What's the status of that migration I asked about last week?" │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MEMORY ROUTER                                 │
│                                                                 │
│  1. Parse query intent                                          │
│  2. Decide which store(s) to query                              │
│  3. Merge results                                               │
│  4. Inject into context window                                  │
└────┬──────────────┬───────────────┬─────────────────────────────┘
     │              │               │
     ▼              ▼               ▼
┌─────────┐  ┌───────────┐  ┌──────────────┐
│ VECTOR  │  │ KNOWLEDGE │  │  KEY-VALUE   │
│ STORE   │  │ GRAPH     │  │  STORE       │
│         │  │           │  │              │
│ Semantic│  │ Entity    │  │ User profile │
│ search  │  │ relations │  │ Session data │
│         │  │ Temporal  │  │ Feature flags│
│ pgvector│  │ Neo4j     │  │ Redis        │
│ Pinecone│  │ FalkorDB  │  │ DynamoDB     │
│ Qdrant  │  │           │  │              │
└─────────┘  └───────────┘  └──────────────┘
```

### Vector Store: Semantic Memory

Vector stores hold embedded text chunks and retrieve them by semantic similarity.

```
WRITE: "User prefers dark mode and monospace fonts"
  → Embed → [0.12, -0.34, 0.89, ...] (1536 dimensions)
  → Store with metadata: {user_id: "u123", type: "preference", created: "2025-05-01"}

READ: "What are the user's UI preferences?"
  → Embed query → [0.15, -0.31, 0.85, ...]
  → Cosine similarity search → top-k results
  → Return: "User prefers dark mode and monospace fonts" (score: 0.92)
```

**When to use vector stores:**
- Storing and retrieving unstructured facts
- User preference recall
- Document/knowledge retrieval
- When you do not need relationship traversal

### Knowledge Graph: Relational Memory

Knowledge graphs store entities (nodes) and their relationships (edges) with properties.

```
WRITE: "John manages the Payments team, which owns the checkout service"

  [John] ──manages──▶ [Payments Team] ──owns──▶ [Checkout Service]
    │                       │
    ├─ role: "Engineering   ├─ size: 8
    │  Manager"             ├─ created: 2023-01
    └─ joined: 2022-03      └─ oncall_rotation: weekly

READ: "Who is responsible for the checkout service?"
  → Traverse: [Checkout Service] ◀──owns── [Payments Team] ◀──manages── [John]
  → Answer: "John manages the Payments team, which owns the checkout service"
```

**When to use knowledge graphs:**
- Multi-hop relationship queries ("who manages the person who owns X?")
- Temporal reasoning ("when did X start reporting to Y?")
- Entity-centric domains (CRM, org charts, product catalogs)
- When facts have valid_at / invalid_at temporal bounds

### Decision: Vector vs Graph vs Hybrid

```
What type of queries will your agent answer?
  │
  ├─ "What do I know about X?" (similarity search)
  │   └─ VECTOR STORE
  │
  ├─ "How is X related to Y?" (relationship traversal)
  │   └─ KNOWLEDGE GRAPH
  │
  ├─ "What changed between date A and date B?" (temporal)
  │   └─ KNOWLEDGE GRAPH (with temporal properties)
  │
  ├─ "Find everything relevant to this topic" (broad recall)
  │   └─ VECTOR STORE
  │
  └─ Multiple of the above?
      └─ HYBRID (vector + graph)
          Mem0 Pro, Zep Cloud, or custom build
```

## Production Implementation

### Minimal Vector Store with pgvector

```python
import asyncpg
from openai import OpenAI
from typing import List, Dict, Optional
import json

client = OpenAI()

class PgVectorMemory:
    """Production long-term memory using PostgreSQL + pgvector."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def initialize(self):
        """Create tables and indexes."""
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(1536),
                    memory_type TEXT DEFAULT 'fact',  -- fact, preference, event
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    expires_at TIMESTAMPTZ,
                    is_active BOOLEAN DEFAULT TRUE
                );
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_embedding 
                ON memories USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_user 
                ON memories (user_id, is_active);
            """)

    def _embed(self, text: str) -> List[float]:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    async def remember(
        self,
        user_id: str,
        content: str,
        memory_type: str = "fact",
        metadata: Optional[dict] = None,
    ):
        """Store a new memory."""
        embedding = self._embed(content)
        async with self.pool.acquire() as conn:
            # Check for duplicates (cosine similarity > 0.95)
            existing = await conn.fetchval("""
                SELECT id FROM memories 
                WHERE user_id = $1 
                  AND is_active = TRUE
                  AND (embedding <=> $2::vector) < 0.05
                LIMIT 1
            """, user_id, str(embedding))

            if existing:
                # Update existing memory instead of creating duplicate
                await conn.execute("""
                    UPDATE memories 
                    SET content = $1, embedding = $2::vector, 
                        metadata = $3, created_at = NOW()
                    WHERE id = $4
                """, content, str(embedding), json.dumps(metadata or {}), existing)
            else:
                await conn.execute("""
                    INSERT INTO memories (user_id, content, embedding, memory_type, metadata)
                    VALUES ($1, $2, $3::vector, $4, $5)
                """, user_id, content, str(embedding), memory_type, json.dumps(metadata or {}))

    async def recall(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve relevant memories by semantic similarity."""
        query_embedding = self._embed(query)

        type_filter = "AND memory_type = $4" if memory_type else ""

        async with self.pool.acquire() as conn:
            params = [user_id, str(query_embedding), top_k]
            if memory_type:
                params.append(memory_type)

            rows = await conn.fetch(f"""
                SELECT content, memory_type, metadata, 
                       1 - (embedding <=> $2::vector) as similarity,
                       created_at
                FROM memories
                WHERE user_id = $1
                  AND is_active = TRUE
                  AND (expires_at IS NULL OR expires_at > NOW())
                  {type_filter}
                ORDER BY embedding <=> $2::vector
                LIMIT $3
            """, *params)

            return [
                {
                    "content": row["content"],
                    "type": row["memory_type"],
                    "metadata": json.loads(row["metadata"]),
                    "similarity": float(row["similarity"]),
                    "created_at": row["created_at"].isoformat(),
                }
                for row in rows
            ]

    async def forget(self, user_id: str, memory_id: int):
        """Soft-delete a memory."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE memories SET is_active = FALSE 
                WHERE id = $1 AND user_id = $2
            """, memory_id, user_id)

    async def get_user_context(self, user_id: str, query: str) -> str:
        """Get formatted context string for injection into prompt."""
        memories = await self.recall(user_id, query, top_k=5)
        if not memories:
            return ""

        lines = ["[Known information about this user]"]
        for mem in memories:
            lines.append(f"- {mem['content']} (confidence: {mem['similarity']:.0%})")
        return "\n".join(lines)
```

### Memory-Aware Agent Loop

```python
async def agent_turn(
    user_id: str,
    user_message: str,
    memory: PgVectorMemory,
    conversation: List[Dict[str, str]],
):
    """Single agent turn with long-term memory integration."""

    # 1. Recall relevant memories
    user_context = await memory.get_user_context(user_id, user_message)

    # 2. Build prompt with memory context
    system_prompt = f"""You are a helpful assistant.

{user_context}

Use the above context to personalize your responses. 
If you learn new facts about the user, mention them naturally."""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation)
    messages.append({"role": "user", "content": user_message})

    # 3. Generate response
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    assistant_message = response.choices[0].message.content

    # 4. Extract and store new memories (background task)
    new_facts = await extract_facts(user_message, assistant_message)
    for fact in new_facts:
        await memory.remember(user_id, fact["content"], fact.get("type", "fact"))

    return assistant_message


async def extract_facts(user_message: str, assistant_message: str) -> List[dict]:
    """Extract storable facts from a conversation turn."""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Extract facts worth remembering from this exchange.
Return a JSON array of objects with "content" and "type" (fact/preference/event).
Only extract genuinely new information. Return [] if nothing worth storing.

User: {user_message}
Assistant: {assistant_message}

JSON array:"""
        }],
        max_tokens=300,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    return result.get("facts", result.get("items", []))
```

## Decision Tree

```
Setting up long-term memory?
  │
  ├─ Do you need it at all?
  │   ├─ Single-session only → No, use short-term memory
  │   ├─ Cross-session personalization → Yes
  │   └─ Multi-user knowledge sharing → Yes
  │
  ├─ Which framework?
  │   ├─ Want managed service → Mem0 Cloud or Zep Cloud
  │   ├─ Need graph + temporal → Zep (Graphiti)
  │   ├─ Already on LangGraph → LangMem
  │   ├─ Highest benchmark scores → Hindsight (91.4% LongMemEval)
  │   └─ Self-improving agent → Letta (83.2%)
  │
  └─ Build vs buy?
      ├─ < 1000 users → Build with pgvector (free, simple)
      ├─ 1K-100K users → Mem0 or Zep managed ($25-$249/mo)
      └─ 100K+ users → Custom build or enterprise tier
```

## When NOT to Use

- **Stateless tools** -- if every interaction is independent (e.g., a code formatter), memory adds complexity with no benefit.
- **Privacy-sensitive domains** -- storing user data long-term creates compliance obligations (GDPR, CCPA). Sometimes it is better not to remember.
- **Ephemeral agents** -- agents that spin up for one task and die. Use checkpointing instead.
- **When context window suffices** -- if your conversations never exceed 10 turns and never span sessions, short-term memory is enough.

## Tradeoffs

| Approach | Latency | Cost | Complexity | Recall Quality | Temporal Reasoning |
|---|---|---|---|---|---|
| pgvector DIY | 50-200ms | Low (Postgres only) | Medium | Good | None |
| Pinecone/Qdrant | 20-100ms | $70+/mo | Low | Good | None |
| Mem0 (vector only) | 50-150ms | Free tier / $99/mo | Low | 49% LongMemEval | Limited |
| Mem0 Pro (graph) | 100-300ms | $249/mo | Low | Better (graph) | Basic |
| Zep Cloud | 100-400ms | $25/mo | Low | 63.8% LongMemEval | Excellent (Graphiti) |
| LangMem | 200-600ms | Free (MIT) | Medium | 58.1% LOCOMO | Basic |
| Custom graph | 100-500ms | Neo4j + compute | High | Depends | Full control |

## Real-World Examples

1. **E-commerce assistant** -- Vector store holds user preferences (style, size, budget), past purchases, wish list. Graph holds product relationships. When user asks "something like that jacket I liked," memory retrieves the jacket and finds similar products through graph traversal.

2. **Healthcare chatbot** -- Vector store holds patient-reported symptoms and preferences. Graph holds medication interactions and treatment history with temporal bounds (started medication X on date Y, stopped on date Z). Critical for "what medications are you currently on?"

3. **Developer tools** -- Vector store holds code patterns the user prefers, past error resolutions, project-specific knowledge. When a similar error appears, memory surfaces the past resolution.

## Failure Modes

| Failure | Symptom | Fix |
|---|---|---|
| Memory bloat | Thousands of near-duplicate memories | Deduplication with similarity threshold (> 0.95 = duplicate) |
| Cross-user leakage | Agent uses User A's data for User B | Strict user_id filtering on every query, tenant isolation |
| Stale facts | "User lives in NYC" when they moved | Temporal validity windows, periodic user confirmation |
| Low recall precision | Retrieved memories are irrelevant | Better embedding model, metadata filtering, re-ranking |
| Write amplification | Every turn generates 5+ memories | Rate-limit memory writes, importance threshold |

## Source(s) and Further Reading

- [State of AI Agent Memory 2026](https://mem0.ai) -- Mem0 research report
- [Mem0 Paper](https://arxiv.org/abs/2504.19413) -- arXiv
- [pgvector Documentation](https://github.com/pgvector/pgvector) -- PostgreSQL vector extension
- [Mem0 vs Zep vs LangMem Comparison](https://dev.to) -- dev.to
- [8 Agent Memory Frameworks Compared](https://vectorize.io) -- vectorize.io
