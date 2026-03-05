"""Content filter policy — block messages matching harmful patterns."""

from __future__ import annotations

import re
from typing import Optional

from agentguard.core.events import AgentEvent, LLMCallEvent, PolicyAction
from agentguard.policies.base import Policy, PolicyResult


# Default patterns that indicate potentially harmful content
_DEFAULT_PATTERNS: list[tuple[str, str]] = [
    ("prompt_injection", r"(?i)ignore\s+(all\s+)?previous\s+instructions"),
    ("prompt_injection", r"(?i)you\s+are\s+now\s+(?:DAN|jailbroken|unrestricted)"),
    ("prompt_injection", r"(?i)disregard\s+(?:all\s+)?(?:your\s+)?(?:previous\s+)?(?:instructions|rules|guidelines)"),
    ("system_prompt_leak", r"(?i)(?:reveal|show|tell\s+me|output|print)\s+(?:your\s+)?system\s+prompt"),
    ("system_prompt_leak", r"(?i)what\s+(?:are|is)\s+your\s+(?:system\s+)?instructions"),
]


class ContentPolicy(Policy):
    """Block messages containing patterns that match harmful content.

    Detects prompt injection attempts and system prompt extraction.

    Args:
        custom_patterns: List of (label, regex_pattern) tuples to add.
        enable_defaults: Whether to include the default harmful patterns.
    """

    name = "content_filter"
    supported_events = [LLMCallEvent]

    def __init__(
        self,
        custom_patterns: list[tuple[str, str]] | None = None,
        enable_defaults: bool = True,
    ) -> None:
        self._patterns: list[tuple[str, re.Pattern[str]]] = []
        if enable_defaults:
            for label, pattern in _DEFAULT_PATTERNS:
                self._patterns.append((label, re.compile(pattern)))
        if custom_patterns:
            for label, pattern in custom_patterns:
                self._patterns.append((label, re.compile(pattern)))

    def evaluate(self, event: AgentEvent) -> PolicyResult:
        if not isinstance(event, LLMCallEvent):
            return PolicyResult(action=PolicyAction.ALLOW, policy_name=self.name)

        # Scan all message contents
        for msg in event.messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            for label, pattern in self._patterns:
                match = pattern.search(content)
                if match:
                    return PolicyResult(
                        action=PolicyAction.BLOCK,
                        policy_name=self.name,
                        reason=f"Content filter triggered: {label}",
                        details={
                            "filter_type": label,
                            "matched_text": match.group(0),
                        },
                    )

        return PolicyResult(action=PolicyAction.ALLOW, policy_name=self.name)
