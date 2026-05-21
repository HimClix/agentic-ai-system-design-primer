# Procedural Memory: Self-Improving Agents

> An agent that rewrites its own instructions based on feedback is the closest thing we have to an AI that truly learns -- and the most dangerous if left unsupervised.

## What It Is

Procedural memory is the mechanism by which an agent modifies its own behavior -- specifically its system prompts, decision rules, and operational instructions -- based on accumulated experience and feedback. In cognitive science, procedural memory is "muscle memory" or "knowing how." In agentic AI, it means **the agent gets better at its job over time without human intervention on every improvement**.

Currently, only two frameworks support procedural memory:
- **LangMem** -- explicit procedural memory via prompt optimization (MIT, free)
- **Letta** (formerly MemGPT) -- autonomous memory management where the agent decides what to remember and how to update its behavior (83.2% LongMemEval)

### Why This Matters

Traditional agents have static system prompts. A human writes the prompt once, and the agent follows it forever -- regardless of how many times users correct it, how many edge cases surface, or how much the domain evolves.

```
STATIC AGENT (traditional):
  Day 1:   System prompt v1.0 → Makes mistake X
  Day 30:  System prompt v1.0 → Makes mistake X again
  Day 90:  System prompt v1.0 → Still makes mistake X
  Day 180: Human manually updates prompt → v1.1

SELF-IMPROVING AGENT (procedural memory):
  Day 1:   System prompt v1.0 → Makes mistake X
  Day 1:   User corrects mistake → Agent updates prompt to v1.1
  Day 30:  System prompt v1.1 → Avoids mistake X, encounters new edge case Y
  Day 30:  Agent learns from failure → Prompt v1.2
  Day 90:  System prompt v1.2 → Handles both X and Y correctly
```

## How It Works

### The Feedback-to-Improvement Loop

```
┌─────────────────────────────────────────────────────────────────┐
│              PROCEDURAL MEMORY LOOP                             │
│                                                                 │
│  ┌────────────┐                                                 │
│  │ User sends │                                                 │
│  │ request    │                                                 │
│  └─────┬──────┘                                                 │
│        │                                                        │
│        ▼                                                        │
│  ┌────────────┐     ┌──────────────────┐                        │
│  │ Load latest│────▶│ Current system   │                        │
│  │ prompt from│     │ prompt (v1.2)    │                        │
│  │ store      │     └────────┬─────────┘                        │
│  └────────────┘              │                                  │
│                              ▼                                  │
│                    ┌──────────────────┐                          │
│                    │ Agent responds   │                          │
│                    └────────┬─────────┘                          │
│                             │                                   │
│                    ┌────────▼─────────┐                          │
│                    │ User reacts      │                          │
│                    └────────┬─────────┘                          │
│                             │                                   │
│              ┌──────────────┴──────────────┐                    │
│              │                             │                    │
│         ┌────▼────┐                 ┌──────▼──────┐             │
│         │ Positive│                 │ Negative    │             │
│         │ (no     │                 │ feedback or │             │
│         │ change) │                 │ correction  │             │
│         └─────────┘                 └──────┬──────┘             │
│                                            │                    │
│                                   ┌────────▼─────────┐          │
│                                   │ PROMPT OPTIMIZER  │          │
│                                   │                   │          │
│                                   │ Analyzes:         │          │
│                                   │ - What went wrong │          │
│                                   │ - User's feedback │          │
│                                   │ - Current prompt  │          │
│                                   │                   │          │
│                                   │ Generates:        │          │
│                                   │ - Improved prompt │          │
│                                   │   (v1.3)          │          │
│                                   └────────┬─────────┘          │
│                                            │                    │
│                                   ┌────────▼─────────┐          │
│                                   │ SAFETY GATE       │          │
│                                   │                   │          │
│                                   │ - Diff review     │          │
│                                   │ - Constraint      │          │
│                                   │   validation      │          │
│                                   │ - Optional human  │          │
│                                   │   approval        │          │
│                                   └────────┬─────────┘          │
│                                            │                    │
│                                   ┌────────▼─────────┐          │
│                                   │ PROMPT STORE      │          │
│                                   │                   │          │
│                                   │ v1.0 (original)   │          │
│                                   │ v1.1 (improved)   │          │
│                                   │ v1.2 (improved)   │          │
│                                   │ v1.3 ← NEW        │          │
│                                   └──────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### What Gets Improved

Procedural memory can modify several types of instructions:

| Component | Example Before | Example After | Trigger |
|---|---|---|---|
| **Task strategy** | "Debug by checking logs" | "Debug by checking metrics first, then logs" | User: "You should check Grafana before diving into logs" |
| **Output format** | "Respond in paragraphs" | "Respond with numbered steps, each under 2 sentences" | User: "Too verbose, give me bullet points" |
| **Domain rules** | "Use standard SQL" | "Use PostgreSQL-specific features (JSONB, CTEs)" | User: "We use PostgreSQL, not generic SQL" |
| **Safety constraints** | "Never modify production" | "Never modify production without explicit approval" | Post-incident: "Agent ran migration on prod without asking" |
| **Tool preferences** | "Search codebase with grep" | "Search with ripgrep for speed, fall back to grep" | User: "Use rg instead, it's faster" |

## Production Implementation

### Core Prompt Optimizer

```python
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
import difflib

OPTIMIZER_PROMPT = """You are a prompt engineering expert. Your job is to improve a system prompt based on user feedback about an agent's performance.

CURRENT SYSTEM PROMPT:
```
{current_prompt}
```

CONVERSATION WHERE THE ISSUE OCCURRED:
{conversation}

USER FEEDBACK:
{feedback}

RULES FOR IMPROVING THE PROMPT:
1. Make the minimum necessary change to address the feedback
2. Do NOT remove existing instructions unless they contradict the feedback
3. Do NOT add instructions unrelated to the feedback
4. Preserve the overall tone and structure
5. Be specific -- "check metrics first" is better than "be more thorough"
6. If the feedback suggests a new priority order, make it explicit with numbered steps
7. NEVER remove safety constraints or ethical guidelines
8. Keep the improved prompt similar in length (max 20% longer)

Return ONLY the improved system prompt, nothing else."""


@dataclass
class PromptVersion:
    version: int
    prompt: str
    created_at: str
    feedback_trigger: Optional[str] = None
    diff_from_previous: Optional[str] = None
    checksum: str = ""

    def __post_init__(self):
        self.checksum = hashlib.md5(self.prompt.encode()).hexdigest()[:8]


class ProceduralMemory:
    """Self-improving prompt management with safety controls."""

    def __init__(
        self,
        llm_client,
        optimizer_model: str = "gpt-4o",
        max_versions: int = 50,
        auto_approve: bool = False,  # If True, no human gate
        min_feedback_confidence: float = 0.7,
    ):
        self.client = llm_client
        self.optimizer_model = optimizer_model
        self.max_versions = max_versions
        self.auto_approve = auto_approve
        self.min_feedback_confidence = min_feedback_confidence

        self.versions: List[PromptVersion] = []
        self.pending_improvement: Optional[PromptVersion] = None
        self.constraints: List[str] = []  # Lines that must never be removed

    def initialize(self, initial_prompt: str, constraints: Optional[List[str]] = None):
        """Set the initial system prompt and immutable constraints."""
        self.versions = [PromptVersion(
            version=1,
            prompt=initial_prompt,
            created_at=datetime.utcnow().isoformat(),
            feedback_trigger="Initial prompt",
        )]
        self.constraints = constraints or []

    @property
    def current_prompt(self) -> str:
        return self.versions[-1].prompt if self.versions else ""

    @property
    def current_version(self) -> int:
        return self.versions[-1].version if self.versions else 0

    async def improve(
        self,
        conversation: List[Dict[str, str]],
        feedback: str,
    ) -> Dict:
        """Generate an improved prompt based on feedback."""

        # Format conversation for the optimizer
        convo_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in conversation[-10:]
        )

        response = await self.client.chat.completions.create(
            model=self.optimizer_model,
            messages=[{
                "role": "user",
                "content": OPTIMIZER_PROMPT.format(
                    current_prompt=self.current_prompt,
                    conversation=convo_text,
                    feedback=feedback,
                ),
            }],
            max_tokens=2000,
            temperature=0.0,
        )

        improved_prompt = response.choices[0].message.content.strip()

        # Safety checks
        safety_result = self._safety_check(improved_prompt)
        if not safety_result["safe"]:
            return {
                "status": "rejected",
                "reason": safety_result["reason"],
                "current_version": self.current_version,
            }

        # Generate diff
        diff = self._generate_diff(self.current_prompt, improved_prompt)

        new_version = PromptVersion(
            version=self.current_version + 1,
            prompt=improved_prompt,
            created_at=datetime.utcnow().isoformat(),
            feedback_trigger=feedback,
            diff_from_previous=diff,
        )

        if self.auto_approve:
            self._apply_version(new_version)
            return {
                "status": "applied",
                "version": new_version.version,
                "diff": diff,
            }
        else:
            self.pending_improvement = new_version
            return {
                "status": "pending_approval",
                "version": new_version.version,
                "diff": diff,
                "improved_prompt": improved_prompt,
            }

    def approve_pending(self) -> bool:
        """Approve and apply a pending prompt improvement."""
        if not self.pending_improvement:
            return False
        self._apply_version(self.pending_improvement)
        self.pending_improvement = None
        return True

    def reject_pending(self) -> bool:
        """Reject a pending prompt improvement."""
        if not self.pending_improvement:
            return False
        self.pending_improvement = None
        return True

    def rollback(self, to_version: Optional[int] = None) -> str:
        """Rollback to a previous prompt version."""
        if len(self.versions) <= 1:
            return "Cannot rollback: only one version exists"

        if to_version:
            target = next((v for v in self.versions if v.version == to_version), None)
            if not target:
                return f"Version {to_version} not found"
            # Trim versions after target
            self.versions = [v for v in self.versions if v.version <= to_version]
        else:
            # Rollback one version
            self.versions.pop()

        return f"Rolled back to version {self.current_version}"

    def _apply_version(self, version: PromptVersion):
        """Apply a new prompt version."""
        self.versions.append(version)
        # Trim old versions if exceeding max
        if len(self.versions) > self.max_versions:
            # Keep first (original) and last N versions
            self.versions = [self.versions[0]] + self.versions[-(self.max_versions-1):]

    def _safety_check(self, new_prompt: str) -> Dict:
        """Verify the improved prompt doesn't violate constraints."""
        # Check that immutable constraints are preserved
        for constraint in self.constraints:
            if constraint.lower() not in new_prompt.lower():
                return {
                    "safe": False,
                    "reason": f"Immutable constraint removed: '{constraint}'",
                }

        # Check prompt didn't shrink too much (possible destructive edit)
        if len(new_prompt) < len(self.current_prompt) * 0.5:
            return {
                "safe": False,
                "reason": "Prompt shrunk by more than 50% -- possible destructive edit",
            }

        # Check for dangerous patterns
        dangerous = ["ignore previous instructions", "disregard", "you are now", "forget all"]
        for pattern in dangerous:
            if pattern.lower() in new_prompt.lower():
                return {
                    "safe": False,
                    "reason": f"Dangerous pattern detected: '{pattern}'",
                }

        return {"safe": True, "reason": ""}

    def _generate_diff(self, old: str, new: str) -> str:
        """Generate a readable diff between prompt versions."""
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        diff = difflib.unified_diff(old_lines, new_lines, fromfile="previous", tofile="improved", lineterm="")
        return "".join(diff)

    def get_history(self) -> List[Dict]:
        """Get prompt version history."""
        return [
            {
                "version": v.version,
                "created_at": v.created_at,
                "trigger": v.feedback_trigger,
                "checksum": v.checksum,
                "prompt_length": len(v.prompt),
            }
            for v in self.versions
        ]


# ─── Usage ───

from openai import AsyncOpenAI

client = AsyncOpenAI()
procedural = ProceduralMemory(
    llm_client=client,
    optimizer_model="gpt-4o",
    auto_approve=False,  # Require human approval
)

# Initialize with immutable constraints
procedural.initialize(
    initial_prompt="""You are a database debugging assistant for Acme Payments.

When debugging database issues:
1. Ask about the database type and version
2. Check error messages and logs
3. Suggest solutions with explanations

SAFETY: Never execute DDL statements on production databases.
SAFETY: Always recommend taking a backup before any schema change.""",
    constraints=[
        "Never execute DDL statements on production",
        "Always recommend taking a backup",
    ],
)

# Agent makes a mistake, user provides feedback
result = await procedural.improve(
    conversation=[
        {"role": "user", "content": "My PostgreSQL queries are timing out"},
        {"role": "assistant", "content": "Let me check the error logs first..."},
        {"role": "user", "content": "You should check pg_stat_activity and connection pool metrics first, not logs"},
    ],
    feedback="Check pg_stat_activity and connection pool metrics before diving into logs for PostgreSQL timeout issues",
)

print(result)
# {
#     "status": "pending_approval",
#     "version": 2,
#     "diff": "...shows the changes...",
#     "improved_prompt": "...new prompt with pg_stat_activity prioritized..."
# }

# Human reviews and approves
procedural.approve_pending()
print(f"Now on version {procedural.current_version}")
```

### Persistent Prompt Store

```python
import asyncpg

class PostgresPromptStore:
    """Persist procedural memory in PostgreSQL."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def initialize(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS prompt_versions (
                    id SERIAL PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    version INT NOT NULL,
                    prompt TEXT NOT NULL,
                    feedback_trigger TEXT,
                    diff_from_previous TEXT,
                    checksum TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    approved_by TEXT,
                    approved_at TIMESTAMPTZ,
                    UNIQUE(agent_id, version)
                );
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prompt_active 
                ON prompt_versions (agent_id, is_active)
                WHERE is_active = TRUE;
            """)

    async def save_version(self, agent_id: str, version: PromptVersion, approved_by: Optional[str] = None):
        async with self.pool.acquire() as conn:
            # Deactivate current active version
            await conn.execute("""
                UPDATE prompt_versions SET is_active = FALSE 
                WHERE agent_id = $1 AND is_active = TRUE
            """, agent_id)

            # Insert new version as active
            await conn.execute("""
                INSERT INTO prompt_versions 
                    (agent_id, version, prompt, feedback_trigger, diff_from_previous, 
                     checksum, is_active, approved_by, approved_at)
                VALUES ($1, $2, $3, $4, $5, $6, TRUE, $7, NOW())
            """, agent_id, version.version, version.prompt, version.feedback_trigger,
                version.diff_from_previous, version.checksum, approved_by)

    async def get_active_prompt(self, agent_id: str) -> Optional[str]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT prompt FROM prompt_versions 
                WHERE agent_id = $1 AND is_active = TRUE
            """, agent_id)
            return row["prompt"] if row else None

    async def get_version_history(self, agent_id: str) -> List[Dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT version, feedback_trigger, checksum, is_active, created_at, approved_by
                FROM prompt_versions
                WHERE agent_id = $1
                ORDER BY version DESC
                LIMIT 20
            """, agent_id)
            return [dict(row) for row in rows]
```

## Decision Tree

```
Should I implement self-improving agents?
  │
  ├─ Does your agent receive regular user feedback?
  │   ├─ YES → Procedural memory can leverage that feedback
  │   └─ NO → Limited value -- no feedback means no improvement signal
  │
  ├─ Risk tolerance?
  │   ├─ Low (financial, medical, legal) → Manual approval gate required
  │   ├─ Medium (support, coding) → Auto-approve with safety checks
  │   └─ High (internal tools, experiments) → Auto-approve, rapid iteration
  │
  ├─ Which framework?
  │   ├─ On LangGraph → LangMem (native procedural memory)
  │   ├─ Want autonomous agent → Letta (self-managing)
  │   └─ Custom → Build with the pattern above
  │
  └─ Safeguards needed?
      ├─ Always: Immutable constraints, version history, rollback
      ├─ Usually: Human approval gate, diff review
      └─ Optional: A/B testing, automated eval before deploy
```

## Risks and Safeguards

### Risk Matrix

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| **Prompt injection via feedback** | Critical | Medium | Sanitize feedback, detect adversarial patterns |
| **Gradual prompt drift** | High | High | Periodic human review, constraint preservation |
| **Safety constraint removal** | Critical | Low | Immutable constraints that cannot be edited |
| **Degraded performance** | Medium | Medium | A/B test before full deployment, automated eval |
| **Infinite improvement loop** | Medium | Low | Rate limit improvements, minimum interval between changes |
| **Adversarial user manipulation** | High | Low | Aggregate feedback across users, weight by trust score |

### Essential Safeguards

```python
class SafeguardedProceduralMemory(ProceduralMemory):
    """Procedural memory with production safeguards."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.improvement_count_today = 0
        self.max_improvements_per_day = 5
        self.last_improvement_at = None
        self.min_improvement_interval_seconds = 3600  # 1 hour
        self.eval_function = None  # Optional automated evaluation

    async def improve(self, conversation, feedback) -> Dict:
        # Safeguard 1: Rate limiting
        if self.improvement_count_today >= self.max_improvements_per_day:
            return {"status": "rate_limited", "reason": "Max daily improvements reached"}

        # Safeguard 2: Minimum interval
        if self.last_improvement_at:
            elapsed = time.time() - self.last_improvement_at
            if elapsed < self.min_improvement_interval_seconds:
                return {"status": "too_soon", "wait_seconds": self.min_improvement_interval_seconds - elapsed}

        # Safeguard 3: Feedback sanitization
        sanitized_feedback = self._sanitize_feedback(feedback)
        if not sanitized_feedback:
            return {"status": "rejected", "reason": "Feedback appears adversarial"}

        result = await super().improve(conversation, sanitized_feedback)

        if result["status"] in ("applied", "pending_approval"):
            self.improvement_count_today += 1
            self.last_improvement_at = time.time()

            # Safeguard 4: Automated evaluation (if configured)
            if self.eval_function and result.get("improved_prompt"):
                eval_score = await self.eval_function(result["improved_prompt"])
                if eval_score < 0.8:  # Below 80% on eval
                    self.reject_pending()
                    return {
                        "status": "rejected_by_eval",
                        "eval_score": eval_score,
                        "reason": "Improved prompt scored below threshold on automated eval",
                    }

        return result

    def _sanitize_feedback(self, feedback: str) -> Optional[str]:
        """Detect and reject adversarial feedback."""
        adversarial_patterns = [
            "ignore previous",
            "disregard all",
            "you are now",
            "new instructions:",
            "system prompt:",
            "forget everything",
            "override",
        ]
        for pattern in adversarial_patterns:
            if pattern.lower() in feedback.lower():
                return None
        return feedback
```

## When NOT to Use

- **High-stakes domains without human oversight** -- a self-improving medical diagnosis agent that removes a safety check because a user complained it was "too cautious" is a disaster waiting to happen.
- **When feedback quality is low** -- if users give vague, contradictory, or adversarial feedback, the agent's prompts will degrade rather than improve.
- **Early stage / unstable product** -- if your agent's core behavior is still being designed, self-improvement adds a variable that makes debugging harder. Stabilize first, then enable learning.
- **When you cannot version and rollback** -- never allow self-improvement without the ability to instantly revert to any previous version.
- **When you lack evaluation** -- without automated evaluation, you cannot measure whether improvements are actually improvements or regressions.

## Tradeoffs

| Aspect | Static Prompts | Self-Improving (with approval) | Self-Improving (auto) |
|---|---|---|---|
| Quality over time | Constant (decays relative to evolving needs) | Improves (slowly, with review bottleneck) | Improves (fast, with risk) |
| Risk | Low | Low-Medium | Medium-High |
| Operational effort | None | Review effort (~5 min/improvement) | Monitoring effort |
| Speed of improvement | Weeks (manual rewrite cycles) | Hours (with approval pipeline) | Minutes (auto-apply) |
| Predictability | High | High | Lower |
| Debugging difficulty | Low | Medium (check version history) | Higher (multiple changes may interact) |

## Real-World Examples

1. **Internal DevOps assistant** -- Agent helps engineers debug infrastructure issues. After each incident, the on-call engineer provides feedback: "You should check PgBouncer stats before connection pool errors." Agent's debugging runbook evolves over 6 months from generic steps to a battle-tested playbook with company-specific knowledge. Auto-approve with safety checks (low risk, internal only).

2. **Customer-facing support agent** -- Agent handles billing inquiries. When agents give incorrect refund policy information, support leads correct via feedback. Human approval required for every prompt change (high risk, customer-facing). Over 3 months, the agent's understanding of edge cases in refund policy improves from 60% to 95% accuracy on internal eval.

3. **Code review assistant** -- Agent reviews PRs and suggests improvements. Developers give thumbs-down when suggestions are irrelevant. Procedural memory learns team-specific patterns: "This team uses dependency injection, don't suggest global state." A/B tested: new prompt version deployed only if it scores >= 85% on a held-out PR eval set.

## Failure Modes

| Failure | Symptom | Fix |
|---|---|---|
| Prompt injection via feedback | Agent starts behaving adversarially | Sanitize all feedback, detect injection patterns |
| Gradual drift | Prompt slowly diverges from intended behavior | Monthly human review, compare to original constraints |
| Contradictory improvements | One user says "be more concise," another says "explain more" | Aggregate feedback, use majority signal or user-specific prompts |
| Over-specialization | Prompt becomes too specific to one user's preferences | Separate user-specific vs agent-wide procedural memory |
| Version explosion | 500+ prompt versions, hard to track | Compact similar versions, keep meaningful changepoints only |
| Evaluation gaming | Improved prompt passes eval but fails on real tasks | Diverse eval set, update eval regularly, production monitoring |

## Source(s) and Further Reading

- [LangMem Procedural Memory Documentation](https://langchain-ai.github.io/langmem/) -- LangChain
- [Letta (MemGPT) GitHub](https://github.com/letta-ai/letta) -- Self-evolving agent architecture
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) -- Shinn et al., 2023
- [State of AI Agent Memory 2026](https://mem0.ai) -- Mem0 research report
- [8 Agent Memory Frameworks Compared](https://vectorize.io) -- vectorize.io
