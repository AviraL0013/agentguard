"""Cost limit policy — blocks calls when spend exceeds thresholds."""

from __future__ import annotations

from agentguard.core.events import AgentEvent, LLMCallEvent, PolicyAction
from agentguard.policies.base import Policy, PolicyResult
from agentguard.tracking.cost import CostTracker


class CostPolicy(Policy):
    """Block LLM calls when cost exceeds configured limits.

    Args:
        tracker: The CostTracker instance to check against.
        max_run_cost: Max USD per run (0 = no limit).
        max_daily_cost: Max USD per day (0 = no limit).
        max_total_cost: Max USD total (0 = no limit).
    """

    name = "cost_limit"
    supported_events = [LLMCallEvent]

    def __init__(
        self,
        tracker: CostTracker,
        max_run_cost: float = 0,
        max_daily_cost: float = 0,
        max_total_cost: float = 0,
    ) -> None:
        self._tracker = tracker
        self._max_run = max_run_cost
        self._max_daily = max_daily_cost
        self._max_total = max_total_cost

    def evaluate(self, event: AgentEvent) -> PolicyResult:
        if not isinstance(event, LLMCallEvent):
            return PolicyResult(action=PolicyAction.ALLOW, policy_name=self.name)

        # Check total cost
        if self._max_total > 0:
            total = self._tracker.get_total_cost()
            if total >= self._max_total:
                return PolicyResult(
                    action=PolicyAction.BLOCK,
                    policy_name=self.name,
                    reason=f"Total cost ${total:.4f} exceeds limit ${self._max_total:.2f}",
                    details={"current_total": total, "limit": self._max_total},
                )

        # Check daily cost
        if self._max_daily > 0:
            daily = self._tracker.get_daily_cost()
            if daily >= self._max_daily:
                return PolicyResult(
                    action=PolicyAction.BLOCK,
                    policy_name=self.name,
                    reason=f"Daily cost ${daily:.4f} exceeds limit ${self._max_daily:.2f}",
                    details={"current_daily": daily, "limit": self._max_daily},
                )

        # Check run cost
        if self._max_run > 0 and event.run_id:
            run_cost = self._tracker.get_run_cost(event.run_id)
            if run_cost >= self._max_run:
                return PolicyResult(
                    action=PolicyAction.BLOCK,
                    policy_name=self.name,
                    reason=f"Run cost ${run_cost:.4f} exceeds limit ${self._max_run:.2f}",
                    details={"current_run_cost": run_cost, "limit": self._max_run},
                )

        return PolicyResult(action=PolicyAction.ALLOW, policy_name=self.name)
