"""Tool policy — restrict which tools an agent is allowed to call."""

from __future__ import annotations

from agentguard.core.events import AgentEvent, PolicyAction, ToolCallEvent
from agentguard.policies.base import Policy, PolicyResult


class ToolPolicy(Policy):
    """Block or allow specific tools.

    Operates in one of two modes:
    - **blocklist**: block calls to the listed tools, allow everything else.
    - **allowlist**: only allow calls to the listed tools, block everything else.

    Args:
        blocked_tools: Tool names to block.
        allowed_tools: Tool names to allow (everything else is blocked).
        If both are provided, blocked_tools takes precedence.
    """

    name = "tool_restriction"
    supported_events = [ToolCallEvent]

    def __init__(
        self,
        blocked_tools: list[str] | None = None,
        allowed_tools: list[str] | None = None,
    ) -> None:
        self._blocked = set(blocked_tools or [])
        self._allowed = set(allowed_tools or [])

    def evaluate(self, event: AgentEvent) -> PolicyResult:
        if not isinstance(event, ToolCallEvent):
            return PolicyResult(action=PolicyAction.ALLOW, policy_name=self.name)

        tool = event.tool_name

        # Blocklist mode
        if self._blocked and tool in self._blocked:
            return PolicyResult(
                action=PolicyAction.BLOCK,
                policy_name=self.name,
                reason=f"Tool '{tool}' is blocked by policy",
                details={"tool": tool, "mode": "blocklist"},
            )

        # Allowlist mode
        if self._allowed and tool not in self._allowed:
            return PolicyResult(
                action=PolicyAction.BLOCK,
                policy_name=self.name,
                reason=f"Tool '{tool}' is not in the allowed list",
                details={"tool": tool, "mode": "allowlist"},
            )

        return PolicyResult(action=PolicyAction.ALLOW, policy_name=self.name)
