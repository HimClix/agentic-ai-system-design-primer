# Graph of Thoughts (GoT)

> GoT models LLM reasoning as an arbitrary directed graph -- thoughts can branch, merge, and loop back, enabling the combination of partial solutions from different reasoning paths.

## What It Is

Graph of Thoughts (GoT), introduced by Besta et al. (2023) at ETH Zurich, extends the Tree of Thoughts (ToT) pattern by removing the tree constraint. Instead of a tree where thoughts only branch, GoT allows:

- **Branching**: One thought spawns multiple alternatives (like ToT)
- **Merging/Aggregation**: Multiple partial thoughts combine into one (unlike ToT)
- **Feedback loops**: Results from later stages refine earlier ones (unlike ToT)
- **Arbitrary connections**: Any thought can connect to any other thought

This makes GoT a superset of Chain-of-Thought (linear), Tree-of-Thoughts (tree), and other structured reasoning patterns.

### Evolution of Reasoning Structures

```
Chain-of-Thought:  A → B → C → D         (linear)
Tree-of-Thoughts:  A → B1 → C1           (branching)
                    ↘ B2 → C2
Graph-of-Thoughts: A → B1 → C1 ↘
                    ↘ B2 → C2 → D → E    (merge + branch + loop)
                         ↑___________|
```

## How It Works

### Core Operations

GoT defines three fundamental operations on the thought graph:

1. **Generate**: Create new thoughts from existing ones (1-to-N)
2. **Aggregate**: Combine multiple thoughts into one (N-to-1)
3. **Refine**: Improve a thought using feedback (1-to-1 with loop)

```
                    ┌──────────────────────────────────────────┐
                    │         Graph of Thoughts (GoT)          │
                    │                                          │
                    │   [Initial Query]                        │
                    │        │                                 │
                    │   ┌────▼─────┐    GENERATE (branch)      │
                    │   │ Thought A │                          │
                    │   └──┬───┬───┘                           │
                    │      │   │                               │
                    │  ┌───▼┐ ┌▼───┐    GENERATE (parallel)    │
                    │  │ B1 │ │ B2 │                           │
                    │  └──┬─┘ └─┬──┘                           │
                    │     │     │                               │
                    │  ┌──▼┐  ┌─▼──┐    Independent reasoning  │
                    │  │ C1│  │ C2 │                           │
                    │  └──┬┘  └─┬──┘                           │
                    │     │     │                               │
                    │  ┌──▼─────▼──┐    AGGREGATE (merge)      │
                    │  │  D (merged)│                          │
                    │  └──────┬────┘                           │
                    │         │                                │
                    │  ┌──────▼────┐    REFINE (loop)          │
                    │  │ D' (refined)│◄──── Score & refine     │
                    │  └──────┬────┘                           │
                    │         │                                │
                    │    [Final Answer]                         │
                    └──────────────────────────────────────────┘
```

### Concrete Example: Sorting Problem

The paper's key example -- sorting a list of numbers using GoT:

```
Input: [8, 3, 1, 7, 5, 2, 9, 4, 6]

Step 1 - GENERATE (decompose into sub-problems):
  Sub-list A: [8, 3, 1]
  Sub-list B: [7, 5, 2]
  Sub-list C: [9, 4, 6]

Step 2 - GENERATE (sort each independently):
  Sorted A: [1, 3, 8]
  Sorted B: [2, 5, 7]
  Sorted C: [4, 6, 9]

Step 3 - AGGREGATE (merge sorted sub-lists):
  Merged: [1, 2, 3, 4, 5, 6, 7, 8, 9]

Step 4 - REFINE (verify correctness):
  Check: Is [1, 2, 3, 4, 5, 6, 7, 8, 9] sorted?  YES.
  (If NO, loop back to Step 3 with error feedback)
```

CoT would try to sort the whole list at once (error-prone for LLMs).
ToT would try multiple sorting strategies but can't merge partial results.
GoT decomposes, solves parts independently, then merges -- like merge sort.

## Production Implementation

### GoT Framework

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum
import uuid
import anthropic


class OperationType(Enum):
    GENERATE = "generate"
    AGGREGATE = "aggregate"
    REFINE = "refine"
    SCORE = "score"


@dataclass
class Thought:
    """A single node in the thought graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    score: float = 0.0
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    operation: OperationType = OperationType.GENERATE
    metadata: dict = field(default_factory=dict)


class GraphOfThoughts:
    """
    Graph of Thoughts executor.

    Implements the three core operations:
    - generate: Branch one thought into multiple alternatives
    - aggregate: Merge multiple thoughts into one
    - refine: Improve a thought via self-critique loop
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_refinement_rounds: int = 3,
        branching_factor: int = 3,
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_refinements = max_refinement_rounds
        self.branching_factor = branching_factor
        self.thoughts: dict[str, Thought] = {}
        self.root_id: Optional[str] = None

    def _call_llm(self, system: str, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def add_thought(self, thought: Thought) -> str:
        self.thoughts[thought.id] = thought
        if not self.root_id:
            self.root_id = thought.id
        return thought.id

    def generate(self, parent_id: str, prompt_template: str, n: int = None) -> list[str]:
        """
        GENERATE operation: create N child thoughts from a parent.
        1-to-N operation (branching).
        """
        n = n or self.branching_factor
        parent = self.thoughts[parent_id]

        system = (
            "You are generating alternative approaches to a problem. "
            "Provide exactly one distinct approach. Be specific and concrete."
        )

        child_ids = []
        for i in range(n):
            prompt = prompt_template.format(
                content=parent.content,
                index=i + 1,
                total=n,
            )
            result = self._call_llm(system, prompt)

            child = Thought(
                content=result,
                parents=[parent_id],
                operation=OperationType.GENERATE,
                metadata={"branch_index": i},
            )
            child_id = self.add_thought(child)
            parent.children.append(child_id)
            child_ids.append(child_id)

        return child_ids

    def aggregate(self, parent_ids: list[str], merge_prompt: str) -> str:
        """
        AGGREGATE operation: merge N thoughts into one.
        N-to-1 operation (the key differentiator from ToT).
        """
        parents = [self.thoughts[pid] for pid in parent_ids]
        parent_contents = "\n\n---\n\n".join(
            f"Approach {i+1}:\n{p.content}" for i, p in enumerate(parents)
        )

        system = (
            "You are combining multiple partial solutions into one coherent, "
            "complete solution. Take the best elements from each approach."
        )
        prompt = merge_prompt.format(approaches=parent_contents)

        result = self._call_llm(system, prompt)

        merged = Thought(
            content=result,
            parents=parent_ids,
            operation=OperationType.AGGREGATE,
        )
        merged_id = self.add_thought(merged)

        for pid in parent_ids:
            self.thoughts[pid].children.append(merged_id)

        return merged_id

    def score(self, thought_id: str, scoring_prompt: str) -> float:
        """Score a thought on a 0-1 scale using LLM-as-judge."""
        thought = self.thoughts[thought_id]

        system = (
            "You are evaluating a solution. "
            "Respond with ONLY a number between 0.0 and 1.0 indicating quality."
        )
        prompt = scoring_prompt.format(content=thought.content)
        result = self._call_llm(system, prompt)

        try:
            score = float(result.strip())
            score = max(0.0, min(1.0, score))
        except ValueError:
            score = 0.5

        thought.score = score
        return score

    def refine(self, thought_id: str, refine_prompt: str) -> str:
        """
        REFINE operation: improve a thought via self-critique.
        1-to-1 with feedback loop.
        """
        current_id = thought_id

        for round_num in range(self.max_refinements):
            current = self.thoughts[current_id]

            system = (
                "You are refining a solution. Identify weaknesses and improve it. "
                "Output ONLY the improved solution, not commentary."
            )
            prompt = refine_prompt.format(
                content=current.content,
                round=round_num + 1,
            )
            result = self._call_llm(system, prompt)

            refined = Thought(
                content=result,
                parents=[current_id],
                operation=OperationType.REFINE,
                metadata={"refinement_round": round_num + 1},
            )
            refined_id = self.add_thought(refined)
            current.children.append(refined_id)

            # Score the refinement -- stop if good enough
            score = self.score(refined_id, "Rate this solution:\n{content}")
            if score >= 0.9:
                break

            current_id = refined_id

        return current_id

    def get_best_leaf(self) -> Thought:
        """Return the highest-scored leaf thought."""
        leaves = [
            t for t in self.thoughts.values()
            if not t.children
        ]
        if not leaves:
            raise ValueError("No leaf thoughts found")
        return max(leaves, key=lambda t: t.score)


# --- Example Usage: Essay Writing with GoT ---

def write_essay_with_got(topic: str) -> str:
    got = GraphOfThoughts(branching_factor=3, max_refinement_rounds=2)

    # Root thought
    root = Thought(content=topic)
    root_id = got.add_thought(root)

    # GENERATE: Create 3 different essay outlines
    outline_ids = got.generate(
        root_id,
        prompt_template=(
            "Write outline #{index} of {total} for an essay about: {content}\n"
            "Make this outline distinctly different from others. "
            "Include thesis, 3 body paragraphs, and conclusion."
        ),
    )

    # GENERATE: Expand each outline into a draft paragraph
    draft_ids = []
    for oid in outline_ids:
        expanded = got.generate(
            oid,
            prompt_template="Write the full essay body for this outline:\n{content}",
            n=1,
        )
        draft_ids.extend(expanded)

    # SCORE: Rate each draft
    for did in draft_ids:
        got.score(
            did,
            "Rate the quality of this essay draft (0-1):\n{content}",
        )

    # AGGREGATE: Merge the two best drafts
    scored_drafts = sorted(
        [(did, got.thoughts[did].score) for did in draft_ids],
        key=lambda x: x[1],
        reverse=True,
    )
    top_two = [scored_drafts[0][0], scored_drafts[1][0]]

    merged_id = got.aggregate(
        top_two,
        merge_prompt=(
            "Combine the best elements of these two essay drafts into one "
            "cohesive, well-structured essay:\n\n{approaches}"
        ),
    )

    # REFINE: Polish the merged essay
    final_id = got.refine(
        merged_id,
        refine_prompt=(
            "Improve this essay. Fix any logical gaps, improve transitions, "
            "strengthen arguments:\n\n{content}"
        ),
    )

    return got.thoughts[final_id].content
```

### Token Cost Analysis

```
GoT vs CoT vs ToT for a 3-branch, 2-level problem:

Chain-of-Thought (linear):
  1 LLM call = ~1,000 tokens
  Total: ~1,000 tokens

Tree-of-Thoughts (branching factor 3, depth 2):
  Level 1: 3 calls
  Level 2: 9 calls
  Scoring: 12 calls
  Total: 24 calls * ~500 tokens = ~12,000 tokens

Graph-of-Thoughts (3 branches, merge, refine):
  Generate: 3 calls (branches)
  Expand: 3 calls (drafts)
  Score: 3 calls
  Aggregate: 1 call (merge)
  Refine: 2 calls (polish loop)
  Total: 12 calls * ~700 tokens = ~8,400 tokens

GoT achieves better quality than ToT with ~30% fewer tokens
because aggregation recombines instead of expanding.
```

## Decision Tree: When to Use GoT

```
         Should I use Graph of Thoughts?
                    │
         ┌──────────▼──────────┐
         │ Can the problem be    │
         │ decomposed into parts │
         │ that can be solved    │
         │ independently?        │
         └───┬────────────┬─────┘
            Yes           No ──► Use ReAct or CoT
             │
         ┌───▼───────────────┐
         │ Do partial solutions │
         │ need to be MERGED   │
         │ (not just selected)? │
         └───┬────────────┬───┘
            Yes           No ──► Use Tree-of-Thoughts
             │                    (select best branch)
         ┌───▼───────────────┐
         │ Is iterative       │
         │ refinement valuable?│
         └───┬────────────┬───┘
            Yes           No ──► Use parallel generation
             │                    + simple voting
         ┌───▼───────────┐
         │   USE GoT      │
         └────────────────┘
```

## When NOT to Use

1. **Simple factual queries**: "What is the capital of France?" -- GoT overhead is unnecessary.
2. **Latency-critical applications**: GoT requires many sequential LLM calls (12+ for a basic problem). If you need < 2s response time, use CoT.
3. **Low-budget scenarios**: 8-12 LLM calls per query at $3/M tokens means $0.05-0.10 per query. At 10K queries/day, that is $500-1000/day.
4. **Tasks without decomposable sub-problems**: If the problem cannot be meaningfully split, GoT degenerates to expensive CoT.
5. **When ToT is sufficient**: If you only need to select the best branch (not merge branches), ToT is simpler and cheaper.

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Can merge partial solutions (unique to GoT) | High token cost (12+ LLM calls per query) |
| Feedback loops enable iterative improvement | Complex implementation and debugging |
| More expressive than ToT (arbitrary graphs) | Hard to define optimal graph structure |
| Decomposition improves accuracy on complex tasks | Latency: each call adds 500ms-2s |
| Natural fit for divide-and-conquer problems | Overkill for simple or linear tasks |
| Self-correction through refinement loops | Graph topology must be designed per task |

## Failure Modes

### 1. Infinite Refinement Loops
The refine operation never reaches a satisfactory score and loops indefinitely.
**Mitigation**: Hard cap on refinement rounds (2-3 max). Monotonically decreasing threshold.

### 2. Destructive Aggregation
Merging two good partial solutions produces a worse combined solution -- the LLM picks conflicting elements.
**Mitigation**: Score the merged result and fall back to the best individual branch if merged score is lower.

### 3. Decomposition Mismatch
The problem is split into parts that are not actually independent, causing incoherent partial solutions.
**Mitigation**: Include shared context in all branches. Use a "decomposition validation" step before branching.

### 4. Combinatorial Explosion
High branching factor + deep graphs = exponential LLM calls. 3 branches * 3 levels = 39 calls before aggregation.
**Mitigation**: Keep branching factor at 2-3 and depth at 2. Prune low-scoring branches early.

### 5. Score Calibration Failure
The scoring function is not well-calibrated, causing the wrong branches to be selected or merged.
**Mitigation**: Use relative ranking (which is better, A or B?) instead of absolute scores.

## Sources and Further Reading

- [Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/abs/2308.09687) (Besta et al., 2023, ETH Zurich)
- [GoT Official Implementation](https://github.com/spcl/graph-of-thoughts)
- [Tree of Thoughts (predecessor)](https://arxiv.org/abs/2305.10601) (Yao et al., 2023)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) (Wei et al., 2022)
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents)
