"""Policy engine — base class, result types, and the engine that runs policies.

Each policy declares which event types it applies to via ``supported_events``.
The engine only evaluates relevant policies for each event.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from agentguard.core.events import AgentEvent, PolicyAction


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PolicyResult:
    """Result of a single policy evaluation."""

    action: PolicyAction
    policy_name: str
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.action == PolicyAction.ALLOW

    @property
    def blocked(self) -> bool:
        return self.action == PolicyAction.BLOCK

    @property
    def escalated(self) -> bool:
        return self.action == PolicyAction.ESCALATE


@dataclass
class CombinedPolicyResult:
    """Combined result from running multiple policies on one event."""

    results: list[PolicyResult] = field(default_factory=list)

    @property
    def action(self) -> PolicyAction:
        """The strictest action wins: BLOCK > ESCALATE > REDACT > ALLOW."""
        if any(r.action == PolicyAction.BLOCK for r in self.results):
            return PolicyAction.BLOCK
        if any(r.action == PolicyAction.ESCALATE for r in self.results):
            return PolicyAction.ESCALATE
        if any(r.action == PolicyAction.REDACT for r in self.results):
            return PolicyAction.REDACT
        return PolicyAction.ALLOW

    @property
    def allowed(self) -> bool:
        return self.action == PolicyAction.ALLOW

    @property
    def blocked(self) -> bool:
        return self.action == PolicyAction.BLOCK

    @property
    def reasons(self) -> list[str]:
        return [r.reason for r in self.results if r.reason and not r.allowed]

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serialize for storage in event.policy_results."""
        return [
            {
                "policy": r.policy_name,
                "action": r.action,
                "reason": r.reason,
                **r.details,
            }
            for r in self.results
        ]


# ---------------------------------------------------------------------------
# Base policy
# ---------------------------------------------------------------------------

class Policy(ABC):
    """Base class for all policies.

    Subclasses must implement ``evaluate`` and set ``name``
    and ``supported_events``.
    """

    name: str = "unnamed_policy"

    # Which event types this policy applies to.
    # Empty list = applies to ALL event types.
    supported_events: list[type[AgentEvent]] = []

    def applies_to(self, event: AgentEvent) -> bool:
        """Check if this policy should run for the given event type."""
        if not self.supported_events:
            return True
        return type(event) in self.supported_events

    @abstractmethod
    def evaluate(self, event: AgentEvent) -> PolicyResult:
        """Evaluate the event against this policy."""
        ...


# ---------------------------------------------------------------------------
# Policy Engine
# ---------------------------------------------------------------------------

class PolicyEngine:
    """Runs registered policies against events.

    Only evaluates policies whose ``supported_events`` match the event type.
    """

    def __init__(self, policies: Optional[list[Policy]] = None) -> None:
        self._policies: list[Policy] = list(policies or [])

    def add_policy(self, policy: Policy) -> None:
        self._policies.append(policy)

    def remove_policy(self, name: str) -> None:
        self._policies = [p for p in self._policies if p.name != name]

    def run(self, event: AgentEvent) -> CombinedPolicyResult:
        """Run all applicable policies on *event*."""
        results: list[PolicyResult] = []
        for policy in self._policies:
            if policy.applies_to(event):
                result = policy.evaluate(event)
                results.append(result)
        return CombinedPolicyResult(results=results)

    @property
    def policies(self) -> list[Policy]:
        return list(self._policies)
