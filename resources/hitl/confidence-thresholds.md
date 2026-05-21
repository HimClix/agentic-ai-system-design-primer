# Confidence Thresholds by Action Type
> Set thresholds per action type: read_only=0.0, internal_write=0.7, financial=0.95 -- then tune with production data.

## What It Is

Confidence thresholds determine when an agent can act autonomously vs when it needs human approval. Different action types require different confidence levels because the cost of being wrong varies dramatically: a wrong search result is mildly annoying, a wrong financial transaction can be catastrophic.

The critical insight: never use a single threshold for everything. A one-size-fits-all threshold of 0.8 is too conservative for reads (blocking unnecessary) and too permissive for financial operations (dangerous).

## How It Works

### Default Threshold Table

| Action Type | Threshold | Rationale | Example Actions |
|------------|-----------|-----------|----------------|
| read_only | 0.0 | No risk, always auto-approve | Search, list, describe |
| classification | 0.3 | Low risk, easily verified | Intent classification, categorization |
| internal_write | 0.7 | Moderate risk, reversible | Update internal ticket, add note |
| external_comms | 0.9 | High risk, visible to customers | Send email, post response |
| financial | 0.95 | Critical, potentially irreversible | Process refund, charge customer |
| data_deletion | 0.95 | Critical, irreversible | Delete records, purge data |
| production_deploy | 1.1 | Always needs approval (threshold > 1.0) | Deploy code, modify infra |

### Confidence Scoring Inputs

Confidence is NOT just "how sure is the LLM?" -- that signal alone is unreliable. Production confidence combines multiple signals:

```
Confidence = weighted_average(
    llm_self_report      * 0.15,  # LLMs overestimate their confidence
    retrieval_relevance   * 0.25,  # How relevant was the RAG context?
    tool_success_rate     * 0.20,  # Did tools return valid results?
    output_consistency    * 0.20,  # Same answer on retry?
    task_familiarity      * 0.20,  # How similar to past successful tasks?
)
```

## Production Implementation

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import statistics


class ActionType(Enum):
    READ_ONLY = "read_only"
    CLASSIFICATION = "classification"
    INTERNAL_WRITE = "internal_write"
    EXTERNAL_COMMS = "external_comms"
    FINANCIAL = "financial"
    DATA_DELETION = "data_deletion"
    PRODUCTION_DEPLOY = "production_deploy"


@dataclass
class ConfidenceSignals:
    """Raw signals used to compute confidence."""
    llm_self_report: float = 0.5       # 0-1: LLM's stated confidence
    retrieval_relevance: float = 0.5   # 0-1: Avg reranker score of retrieved docs
    tool_success_rate: float = 1.0     # 0-1: Fraction of tools that succeeded
    output_consistency: float = 0.5    # 0-1: Semantic similarity across retries
    task_familiarity: float = 0.5      # 0-1: Similarity to past successful tasks


@dataclass
class ConfidenceResult:
    score: float
    action: str  # "auto_approve", "request_approval", "reject"
    signals: dict[str, float]
    threshold_used: float
    action_type: ActionType


class ConfidenceThresholdEngine:
    """
    Production confidence scoring with per-action-type thresholds.
    Thresholds are tuned with production data.
    """
    
    DEFAULT_THRESHOLDS = {
        ActionType.READ_ONLY: 0.0,
        ActionType.CLASSIFICATION: 0.3,
        ActionType.INTERNAL_WRITE: 0.7,
        ActionType.EXTERNAL_COMMS: 0.9,
        ActionType.FINANCIAL: 0.95,
        ActionType.DATA_DELETION: 0.95,
        ActionType.PRODUCTION_DEPLOY: 1.1,  # Always needs approval
    }
    
    SIGNAL_WEIGHTS = {
        "llm_self_report": 0.15,
        "retrieval_relevance": 0.25,
        "tool_success_rate": 0.20,
        "output_consistency": 0.20,
        "task_familiarity": 0.20,
    }
    
    def __init__(self, threshold_overrides: dict = None):
        self.thresholds = {**self.DEFAULT_THRESHOLDS}
        if threshold_overrides:
            self.thresholds.update(threshold_overrides)
        
        # For threshold tuning
        self._decision_log: list[dict] = []
    
    def evaluate(
        self,
        action_type: ActionType,
        signals: ConfidenceSignals,
        context: dict = None,
    ) -> ConfidenceResult:
        """
        Compute confidence score and determine action.
        
        Returns:
            ConfidenceResult with score, action, and decision metadata.
        """
        # Compute weighted confidence
        signal_dict = {
            "llm_self_report": signals.llm_self_report,
            "retrieval_relevance": signals.retrieval_relevance,
            "tool_success_rate": signals.tool_success_rate,
            "output_consistency": signals.output_consistency,
            "task_familiarity": signals.task_familiarity,
        }
        
        score = sum(
            signal_dict[name] * self.SIGNAL_WEIGHTS[name]
            for name in signal_dict
        )
        
        # Context-based adjustments
        if context:
            score = self._apply_context_adjustments(score, context)
        
        # Determine action based on threshold
        threshold = self.thresholds[action_type]
        
        if score >= threshold:
            action = "auto_approve"
        elif score >= threshold * 0.7:  # Within 70% of threshold
            action = "request_approval"
        else:
            action = "reject"
        
        result = ConfidenceResult(
            score=round(score, 4),
            action=action,
            signals=signal_dict,
            threshold_used=threshold,
            action_type=action_type,
        )
        
        # Log for threshold tuning
        self._decision_log.append({
            "action_type": action_type.value,
            "score": score,
            "threshold": threshold,
            "action": action,
            "signals": signal_dict,
        })
        
        return result
    
    def _apply_context_adjustments(self, score: float, context: dict) -> float:
        """Adjust score based on contextual signals."""
        adjusted = score
        
        # New customer: reduce confidence (less data to work with)
        if context.get("is_new_customer"):
            adjusted *= 0.9
        
        # VIP customer: increase threshold (higher stakes)
        if context.get("is_vip"):
            adjusted *= 0.85
        
        # High dollar amount: reduce confidence
        amount = context.get("amount_usd", 0)
        if amount > 1000:
            adjusted *= 0.8
        elif amount > 500:
            adjusted *= 0.9
        
        # Time pressure (customer waiting > 5 min): slightly increase
        wait_time = context.get("customer_wait_seconds", 0)
        if wait_time > 300:
            adjusted *= 1.05  # Slight boost to avoid long waits
        
        return max(0.0, min(1.0, adjusted))
    
    # --- Threshold Tuning ---
    
    def analyze_thresholds(self) -> dict:
        """
        Analyze decision log to suggest threshold adjustments.
        Run this periodically with production data.
        """
        analysis = {}
        
        for action_type in ActionType:
            decisions = [
                d for d in self._decision_log
                if d["action_type"] == action_type.value
            ]
            
            if not decisions:
                continue
            
            scores = [d["score"] for d in decisions]
            auto_approved = [d for d in decisions if d["action"] == "auto_approve"]
            approvals_requested = [d for d in decisions if d["action"] == "request_approval"]
            
            analysis[action_type.value] = {
                "total_decisions": len(decisions),
                "auto_approved": len(auto_approved),
                "approval_requested": len(approvals_requested),
                "auto_approve_rate": round(len(auto_approved) / max(len(decisions), 1) * 100, 1),
                "avg_score": round(statistics.mean(scores), 4),
                "median_score": round(statistics.median(scores), 4),
                "min_score": round(min(scores), 4),
                "max_score": round(max(scores), 4),
                "current_threshold": self.thresholds[action_type],
            }
        
        return analysis
    
    def suggest_threshold_adjustments(
        self, 
        human_feedback: list[dict],
        target_false_positive_rate: float = 0.02,
    ) -> dict:
        """
        Suggest threshold adjustments based on human feedback.
        
        Args:
            human_feedback: List of {"score": float, "human_correct": bool}
            target_false_positive_rate: Acceptable rate of wrong auto-approvals
        """
        if not human_feedback:
            return {}
        
        # Sort by score ascending
        sorted_feedback = sorted(human_feedback, key=lambda x: x["score"])
        
        # Find threshold where false positive rate <= target
        for i, entry in enumerate(sorted_feedback):
            remaining = sorted_feedback[i:]
            if not remaining:
                break
            
            false_positives = sum(
                1 for e in remaining if not e["human_correct"]
            )
            fp_rate = false_positives / len(remaining)
            
            if fp_rate <= target_false_positive_rate:
                return {
                    "suggested_threshold": round(entry["score"], 4),
                    "false_positive_rate": round(fp_rate, 4),
                    "auto_approve_rate": round(len(remaining) / len(sorted_feedback) * 100, 1),
                    "sample_size": len(human_feedback),
                }
        
        return {"suggested_threshold": 1.0, "message": "Cannot meet target FP rate"}


# --- Usage Example ---

engine = ConfidenceThresholdEngine()

# Example: Agent wants to process a refund
signals = ConfidenceSignals(
    llm_self_report=0.92,         # LLM says it's 92% confident
    retrieval_relevance=0.88,     # RAG found relevant refund policy
    tool_success_rate=1.0,        # All tool calls succeeded
    output_consistency=0.95,      # Same answer on retry
    task_familiarity=0.80,        # Similar to past refund tasks
)

result = engine.evaluate(
    action_type=ActionType.FINANCIAL,
    signals=signals,
    context={"amount_usd": 150, "is_vip": False},
)

print(f"Score: {result.score}")      # e.g., 0.89
print(f"Threshold: {result.threshold_used}")  # 0.95
print(f"Action: {result.action}")    # "request_approval" (0.89 < 0.95)
```

## Decision Tree: Setting Initial Thresholds

```
What happens if the agent is wrong?
│
├── Nothing bad (wrong search result)
│   └── Threshold: 0.0 (always auto-approve)
│
├── Minor inconvenience (wrong internal tag)
│   └── Threshold: 0.3-0.5
│
├── Fixable mistake (wrong ticket update)
│   └── Threshold: 0.7
│
├── Customer-visible error (wrong email)
│   └── Threshold: 0.9
│
├── Financial loss (wrong refund amount)
│   └── Threshold: 0.95
│
└── Irreversible damage (data deletion, production)
    └── Threshold: 1.1 (always require approval)
```

## When NOT to Use Confidence Thresholds

- **Binary decisions**: If the action is either always-approve or always-reject, thresholds add no value.
- **No confidence signals available**: If you can't measure retrieval relevance or tool success, a single threshold is meaningless. Fix the signals first.
- **All actions are low-risk**: If everything is read-only, skip the complexity.

## Tradeoffs

| Threshold Strategy | Auto-Approve Rate | False Positive Rate | Human Load |
|-------------------|-------------------|-------------------| ----------|
| Conservative (high thresholds) | 20-40% | < 1% | High |
| Balanced (recommended) | 50-70% | 2-5% | Medium |
| Aggressive (low thresholds) | 80-95% | 5-15% | Low |

## Tuning Process

```
1. Start with default thresholds (this page)
2. Run for 2 weeks, log all decisions
3. Have humans review a sample of auto-approved actions
4. Calculate false positive rate per action type
5. Adjust thresholds:
   - FP rate > 5%? → Raise threshold by 0.05
   - FP rate < 1% and auto-approve rate < 50%? → Lower threshold by 0.05
6. Repeat quarterly
```

## Failure Modes

1. **Calibration drift**: Model update changes confidence distribution, thresholds no longer calibrated. Mitigation: re-evaluate after model changes.
2. **Signal gaming**: LLM learns to output high confidence regardless of actual certainty. Mitigation: use multi-signal scoring, not just LLM self-report.
3. **Threshold fragility**: Small score changes cause big behavior changes at the boundary. Mitigation: soft boundaries with "request_approval" zone between auto-approve and reject.
4. **Cold start**: New action types have no data for tuning. Mitigation: start conservative (high threshold), lower based on data.

## Source(s) and Further Reading

- "Calibration of LLM Confidence" - Stanford (2024)
- "Building Effective Agents" - Anthropic (2024)
- "Machine Learning System Design" - Chip Huyen
- LangGraph Human-in-the-Loop: https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
