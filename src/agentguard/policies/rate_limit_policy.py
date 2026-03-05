"""Rate limit policy — throttle agent calls per time window."""

from __future__ import annotations

import time
from collections import deque

from agentguard.core.events import AgentEvent, LLMCallEvent, PolicyAction, ToolCallEvent
from agentguard.policies.base import Policy, PolicyResult


class RateLimitPolicy(Policy):
    """Block calls that exceed a rate limit.

    Args:
        max_calls: Maximum number of calls allowed in the window.
        window_seconds: Time window in seconds (default 60).
    """

    name = "rate_limit"
    supported_events = [LLMCallEvent, ToolCallEvent]

    def __init__(self, max_calls: int = 60, window_seconds: float = 60.0) -> None:
        self._max_calls = max_calls
        self._window = window_seconds
        self._timestamps: deque[float] = deque()

    def evaluate(self, event: AgentEvent) -> PolicyResult:
        now = time.monotonic()

        # Purge old timestamps outside the window
        while self._timestamps and (now - self._timestamps[0]) > self._window:
            self._timestamps.popleft()

        if len(self._timestamps) >= self._max_calls:
            return PolicyResult(
                action=PolicyAction.BLOCK,
                policy_name=self.name,
                reason=f"Rate limit exceeded: {self._max_calls} calls per {self._window}s",
                details={
                    "max_calls": self._max_calls,
                    "window_seconds": self._window,
                    "current_count": len(self._timestamps),
                },
            )

        self._timestamps.append(now)
        return PolicyResult(action=PolicyAction.ALLOW, policy_name=self.name)
