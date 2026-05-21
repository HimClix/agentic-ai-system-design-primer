# Hallucination Detection in Production

> Detecting when an agent fabricates information: faithfulness scoring, citation verification, confidence calibration, and LLM-as-Judge evaluation. Online (real-time) and offline (batch sampling) approaches.

## What It Is

Hallucination detection identifies cases where an LLM generates information that is:
1. **Not grounded in the provided context** (faithfulness hallucination): The answer includes facts not present in the retrieved documents
2. **Factually incorrect** (factual hallucination): The answer contradicts known facts
3. **Fabricated with false confidence**: The answer presents uncertain information as definitive

In agent systems, hallucination is particularly dangerous because agents can hallucinate tool names, fabricate tool results, or generate plausible-but-wrong intermediate reasoning.

### Types of Hallucination in Agent Systems

```
1. RESPONSE HALLUCINATION (most common):
   Agent says "Your order shipped on Jan 15" but the database says Jan 20.
   → Answer not faithful to retrieved context.

2. TOOL RESULT FABRICATION:
   Agent claims "I searched and found X" but actually never called the search tool.
   → Agent fabricated a tool interaction.

3. REASONING HALLUCINATION:
   Agent's chain-of-thought includes incorrect logic: "Since 2+2=5, ..."
   → Internal reasoning contains false statements.

4. CITATION HALLUCINATION:
   Agent cites "Smith et al., 2023, Nature" but the paper does not exist.
   → Fabricated source attribution.
```

## How It Works

### Detection Pipeline

```
          Agent Response
               │
    ┌──────────▼──────────────┐
    │  ONLINE DETECTION        │  (real-time, every response)
    │                          │
    │  1. Faithfulness Check   │  Is answer grounded in context?
    │  2. Confidence Score     │  How confident is the model?
    │  3. Citation Check       │  Do cited sources exist?
    │  4. Format Validation    │  Are tool results real?
    │                          │
    └──────┬──────────┬───────┘
           │          │
       PASS (>0.7)   FAIL (<0.7)
           │          │
    ┌──────▼──────┐  ┌▼──────────────┐
    │ Serve to    │  │ Action:        │
    │ user        │  │ - Block        │
    └─────────────┘  │ - Flag + serve │
                     │ - Regenerate   │
                     │ - Escalate     │
                     └───────────────┘

    ┌──────────────────────────────────┐
    │  OFFLINE DETECTION               │  (batch, sampling)
    │                                  │
    │  1. LLM-as-Judge (batch eval)    │
    │  2. Human review of flagged      │
    │  3. RAGAS faithfulness metric    │
    │  4. Statistical analysis         │
    └──────────────────────────────────┘
```

### RAGAS Faithfulness Metric

RAGAS (Retrieval Augmented Generation Assessment) decomposes faithfulness into verifiable claims:

```
Context: "Stripe was founded in 2010. It processes payments globally."
Answer:  "Stripe was founded in 2010 in San Francisco. It processes 
          payments globally and handles $1T annually."

Step 1 - Extract claims from the answer:
  Claim 1: "Stripe was founded in 2010"              → Supported (in context)
  Claim 2: "Stripe was founded in San Francisco"      → Not supported
  Claim 3: "It processes payments globally"            → Supported
  Claim 4: "It handles $1T annually"                   → Not supported

Step 2 - Calculate faithfulness:
  Faithfulness = supported_claims / total_claims
  Faithfulness = 2 / 4 = 0.50

  Score < 0.7 → FLAG AS POTENTIAL HALLUCINATION
```

## Production Implementation

```python
import re
from dataclasses import dataclass
from typing import Optional
from anthropic import Anthropic


@dataclass
class HallucinationCheck:
    """Result of a hallucination check."""
    faithfulness_score: float    # 0.0 - 1.0 (1.0 = fully grounded)
    confidence_score: float     # 0.0 - 1.0 (1.0 = highly confident)
    unsupported_claims: list[str]
    verdict: str                # "pass", "flag", "block"
    details: str


class HallucinationDetector:
    """
    Production hallucination detection for agent systems.
    
    Methods:
    1. Faithfulness scoring (is answer grounded in context?)
    2. Confidence calibration (does the model hedge appropriately?)
    3. Citation verification (do cited sources exist?)
    4. LLM-as-Judge (use a judge model to evaluate)
    """

    def __init__(
        self,
        judge_model: str = "claude-sonnet-4-20250514",
        block_threshold: float = 0.3,
        flag_threshold: float = 0.7,
    ):
        self.client = Anthropic()
        self.judge_model = judge_model
        self.block_threshold = block_threshold
        self.flag_threshold = flag_threshold

    def check_faithfulness(
        self,
        answer: str,
        context: str,
        question: str,
    ) -> HallucinationCheck:
        """
        Check if the answer is faithful to the provided context.
        Uses claim decomposition (RAGAS approach).
        """
        # Step 1: Extract claims from the answer
        claims = self._extract_claims(answer)

        if not claims:
            return HallucinationCheck(
                faithfulness_score=1.0,
                confidence_score=0.5,
                unsupported_claims=[],
                verdict="pass",
                details="No verifiable claims found in response.",
            )

        # Step 2: Verify each claim against context
        supported = []
        unsupported = []

        for claim in claims:
            is_supported = self._verify_claim(claim, context)
            if is_supported:
                supported.append(claim)
            else:
                unsupported.append(claim)

        # Step 3: Calculate faithfulness score
        faithfulness = len(supported) / len(claims) if claims else 1.0

        # Step 4: Determine verdict
        if faithfulness < self.block_threshold:
            verdict = "block"
        elif faithfulness < self.flag_threshold:
            verdict = "flag"
        else:
            verdict = "pass"

        return HallucinationCheck(
            faithfulness_score=round(faithfulness, 2),
            confidence_score=self._estimate_confidence(answer),
            unsupported_claims=unsupported,
            verdict=verdict,
            details=(
                f"{len(supported)}/{len(claims)} claims supported by context. "
                f"Unsupported: {unsupported}"
            ),
        )

    def _extract_claims(self, text: str) -> list[str]:
        """Extract verifiable claims from text using LLM."""
        response = self.client.messages.create(
            model="claude-haiku-3.5",
            max_tokens=1024,
            system=(
                "Extract all factual claims from the text. "
                "Each claim should be a single, verifiable statement. "
                "Output one claim per line, prefixed with '- '. "
                "Do not include opinions, questions, or meta-statements."
            ),
            messages=[{
                "role": "user",
                "content": f"Extract claims from:\n\n{text}",
            }],
        )

        claims = []
        for line in response.content[0].text.strip().split("\n"):
            line = line.strip()
            if line.startswith("- "):
                claims.append(line[2:].strip())
            elif line.startswith("*"):
                claims.append(line.lstrip("* ").strip())

        return claims

    def _verify_claim(self, claim: str, context: str) -> bool:
        """Check if a single claim is supported by the context."""
        response = self.client.messages.create(
            model="claude-haiku-3.5",
            max_tokens=10,
            system=(
                "You are a fact-checker. Given a context and a claim, "
                "determine if the claim is SUPPORTED by the context. "
                "Respond with only YES or NO."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Claim: {claim}\n\n"
                    f"Is this claim supported by the context? YES or NO."
                ),
            }],
        )
        return "YES" in response.content[0].text.upper()

    def _estimate_confidence(self, text: str) -> float:
        """
        Estimate model confidence from linguistic cues.
        Low-cost heuristic (no LLM call needed).
        """
        text_lower = text.lower()
        score = 0.7  # Default moderate confidence

        # High-confidence indicators
        confident_phrases = [
            "definitely", "certainly", "the answer is",
            "according to", "as stated in", "based on the data",
        ]
        for phrase in confident_phrases:
            if phrase in text_lower:
                score += 0.05

        # Low-confidence indicators (hedging)
        hedging_phrases = [
            "i think", "i believe", "probably", "might",
            "possibly", "i'm not sure", "it seems", "approximately",
            "around", "roughly", "may be", "could be",
        ]
        for phrase in hedging_phrases:
            if phrase in text_lower:
                score -= 0.05

        return max(0.0, min(1.0, round(score, 2)))

    def llm_as_judge(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> HallucinationCheck:
        """
        Use a strong LLM as a judge to evaluate hallucination.
        More accurate than claim decomposition but more expensive (~$0.005/eval).
        """
        response = self.client.messages.create(
            model=self.judge_model,
            max_tokens=512,
            system=(
                "You are an expert fact-checker evaluating an AI assistant's response. "
                "Score the response on faithfulness (is it grounded in the provided context?).\n\n"
                "Output format (strictly follow this):\n"
                "FAITHFULNESS_SCORE: <0.0-1.0>\n"
                "UNSUPPORTED_CLAIMS:\n- <claim 1>\n- <claim 2>\n"
                "REASONING: <brief explanation>"
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"QUESTION: {question}\n\n"
                    f"CONTEXT PROVIDED:\n{context}\n\n"
                    f"AI RESPONSE:\n{answer}\n\n"
                    f"Evaluate the faithfulness of the AI response."
                ),
            }],
        )

        output = response.content[0].text

        # Parse score
        score_match = re.search(r"FAITHFULNESS_SCORE:\s*([\d.]+)", output)
        score = float(score_match.group(1)) if score_match else 0.5

        # Parse unsupported claims
        unsupported = []
        in_claims = False
        for line in output.split("\n"):
            if "UNSUPPORTED_CLAIMS" in line:
                in_claims = True
                continue
            if "REASONING" in line:
                in_claims = False
                continue
            if in_claims and line.strip().startswith("- "):
                unsupported.append(line.strip()[2:])

        verdict = (
            "block" if score < self.block_threshold
            else "flag" if score < self.flag_threshold
            else "pass"
        )

        return HallucinationCheck(
            faithfulness_score=round(score, 2),
            confidence_score=self._estimate_confidence(answer),
            unsupported_claims=unsupported,
            verdict=verdict,
            details=output,
        )


# --- Online Detection Middleware ---

class HallucinationMiddleware:
    """
    Middleware that checks every agent response for hallucination.
    Integrates between the agent and the user.
    """

    def __init__(
        self,
        detector: HallucinationDetector,
        mode: str = "flag",  # "block" | "flag" | "monitor"
    ):
        self.detector = detector
        self.mode = mode
        self.stats = {"pass": 0, "flag": 0, "block": 0}

    def process_response(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> dict:
        """Process an agent response through hallucination detection."""
        check = self.detector.check_faithfulness(answer, context, question)
        self.stats[check.verdict] += 1

        if check.verdict == "block" and self.mode == "block":
            return {
                "action": "block",
                "original_answer": answer,
                "reason": check.details,
                "replacement": (
                    "I apologize, but I cannot provide a reliable answer "
                    "based on the available information. Please consult "
                    "the documentation directly or contact support."
                ),
            }
        elif check.verdict == "flag":
            return {
                "action": "serve_with_warning",
                "answer": answer,
                "warning": (
                    f"[Low confidence: {check.faithfulness_score:.0%} faithfulness] "
                    f"Some claims may not be fully supported by the source data."
                ),
                "unsupported_claims": check.unsupported_claims,
            }
        else:
            return {
                "action": "serve",
                "answer": answer,
            }

    def get_stats(self) -> dict:
        total = sum(self.stats.values()) or 1
        return {
            **self.stats,
            "hallucination_rate": round(
                (self.stats["flag"] + self.stats["block"]) / total * 100, 1
            ),
        }


# --- Offline Batch Evaluation ---

def batch_hallucination_eval(
    samples: list[dict],  # [{"question": ..., "answer": ..., "context": ...}]
    sample_rate: float = 0.1,
) -> dict:
    """
    Offline batch evaluation of hallucination rate.
    Run nightly on a sample of production traffic.
    """
    import random

    detector = HallucinationDetector()
    sampled = random.sample(samples, int(len(samples) * sample_rate))

    results = {
        "total_evaluated": len(sampled),
        "pass": 0,
        "flag": 0,
        "block": 0,
        "avg_faithfulness": 0.0,
        "worst_cases": [],
    }

    scores = []
    for sample in sampled:
        check = detector.llm_as_judge(
            question=sample["question"],
            answer=sample["answer"],
            context=sample["context"],
        )
        results[check.verdict] += 1
        scores.append(check.faithfulness_score)

        if check.faithfulness_score < 0.5:
            results["worst_cases"].append({
                "question": sample["question"],
                "faithfulness": check.faithfulness_score,
                "unsupported": check.unsupported_claims,
            })

    results["avg_faithfulness"] = round(sum(scores) / len(scores), 2) if scores else 0
    return results
```

## Decision Tree: When to Block vs Flag vs Let Through

```
    Hallucination detected -- what action to take?
                        │
             ┌──────────▼──────────┐
             │ Faithfulness score?  │
             └──┬───────┬──────┬──┘
               <0.3    0.3-0.7  >0.7
                │       │       │
             BLOCK    FLAG    PASS
                │       │       │
            ┌───▼──┐ ┌──▼──┐ ┌──▼──┐
            │Serve │ │Serve│ │Serve│
            │error │ │with │ │as-is│
            │msg   │ │warn │ │     │
            └──────┘ └─────┘ └─────┘

    Context-dependent overrides:

    ALWAYS BLOCK if:
    - Medical/legal/financial advice
    - Tool results were fabricated
    - Citations are to non-existent sources

    ALWAYS FLAG if:
    - Contains specific numbers not in context
    - Makes claims about recent events
    - References specific people/companies
```

## When NOT to Use

1. **Creative writing tasks**: Hallucination detection makes no sense for fiction, brainstorming, or creative content.
2. **Tasks without ground truth**: If there is no context to check against, faithfulness scoring is meaningless.
3. **Latency-critical paths**: LLM-as-Judge adds 500-2000ms per check. Use heuristic checks for real-time and batch-eval offline.
4. **Low-risk informational queries**: "What is Python?" -- hallucination here is low-stakes and detection is not worth the cost.
5. **Cost-constrained environments**: Each LLM-as-Judge call costs ~$0.005. At 100K queries/day, that is $500/day just for detection.

## Tradeoffs

| Advantage | Disadvantage |
|-----------|--------------|
| Catches fabricated information before users see it | Adds latency (200ms heuristic, 1.5s LLM judge) |
| RAGAS provides standardized metrics | Claim extraction can itself hallucinate |
| LLM-as-Judge is highly accurate (85%+ agreement with humans) | Expensive at scale ($0.005/eval) |
| Reduces liability for incorrect information | False positives block correct answers |
| Enables trust metrics for dashboards | Requires ground-truth context for each answer |

## Real-World Numbers

| Detection Method | Accuracy | Latency | Cost/Check | Best For |
|-----------------|----------|---------|------------|----------|
| Heuristic (hedging language) | 55-65% | <10ms | $0 | Pre-filter, always-on |
| Claim decomposition (RAGAS) | 75-85% | 500-800ms | $0.002 | Online, moderate volume |
| LLM-as-Judge (Sonnet) | 85-90% | 1-2s | $0.005 | Offline batch, high-stakes |
| Human review | 95%+ | Minutes-hours | $0.50-2.00 | Calibration, edge cases |

## Failure Modes

### 1. Judge Model Hallucinations
The LLM-as-Judge itself hallucinate claims that are "unsupported."
**Mitigation**: Use the strongest available model as judge. Cross-validate with multiple judges on critical cases.

### 2. Over-Blocking
Correct answers are blocked because the context is incomplete (the claim is true but not stated in the context).
**Mitigation**: Distinguish "not supported" from "contradicted." Only block on contradictions.

### 3. Missed Hallucinations
Subtle hallucinations (wrong date, swapped numbers) pass detection.
**Mitigation**: Use entity-specific checks for numbers, dates, names in high-stakes domains.

### 4. Gaming by the Agent
Agent learns to hedge everything ("I think," "approximately") to avoid hallucination flags.
**Mitigation**: Detect excessive hedging as a separate quality issue.

### 5. Context Window Limits
Very long contexts make claim verification unreliable (judge model misses details in long context).
**Mitigation**: Chunk context and verify claims against relevant chunks.

## Sources and Further Reading

- [RAGAS: Automated Evaluation of RAG](https://docs.ragas.io/en/stable/concepts/metrics/faithfulness.html)
- [Hallucination Leaderboard - Vectara](https://github.com/vectara/hallucination-leaderboard)
- [FActScore: Fine-grained Atomic Evaluation of Factual Precision](https://arxiv.org/abs/2305.14251)
- [LLM-as-Judge (Zheng et al., 2023)](https://arxiv.org/abs/2306.05685)
- [TruthfulQA Benchmark](https://arxiv.org/abs/2109.07958)
- [Anthropic: Reducing Hallucination](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/reduce-hallucinations)
