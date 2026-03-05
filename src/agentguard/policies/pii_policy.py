"""PII policy — blocks or redacts events containing PII."""

from __future__ import annotations

from agentguard.core.events import AgentEvent, LLMCallEvent, PolicyAction, ToolCallEvent
from agentguard.detectors.pii import RegexPIIDetector
from agentguard.policies.base import Policy, PolicyResult


class PIIPolicy(Policy):
    """Block or redact if PII is detected in LLM inputs/outputs or tool arguments.

    Args:
        action: What to do when PII is found — BLOCK or REDACT.
        detector: Custom PIIDetector instance (defaults to RegexPIIDetector).
    """

    name = "pii"
    supported_events = [LLMCallEvent, ToolCallEvent]

    def __init__(
        self,
        action: PolicyAction = PolicyAction.BLOCK,
        detector: RegexPIIDetector | None = None,
    ) -> None:
        self._action = action
        self._detector = detector or RegexPIIDetector()

    def evaluate(self, event: AgentEvent) -> PolicyResult:
        texts_to_scan: list[str] = []

        if isinstance(event, LLMCallEvent):
            # Scan message contents
            for msg in event.messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    texts_to_scan.append(content)
            # Scan response
            if event.response_content:
                texts_to_scan.append(event.response_content)

        elif isinstance(event, ToolCallEvent):
            # Scan tool arguments
            for v in event.arguments.values():
                if isinstance(v, str):
                    texts_to_scan.append(v)
            # Scan result
            if isinstance(event.result, str):
                texts_to_scan.append(event.result)

        # Run PII detection
        all_matches = []
        for text in texts_to_scan:
            matches = self._detector.scan(text)
            all_matches.extend(matches)

        if all_matches:
            pii_types = list({m.pii_type for m in all_matches})
            return PolicyResult(
                action=self._action,
                policy_name=self.name,
                reason=f"PII detected: {', '.join(pii_types)}",
                details={
                    "pii_matches": [
                        {"type": m.pii_type, "value": m.value}
                        for m in all_matches
                    ]
                },
            )

        return PolicyResult(
            action=PolicyAction.ALLOW,
            policy_name=self.name,
        )
