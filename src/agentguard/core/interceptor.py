"""Core interceptor — before/after hooks for LLM and tool calls.

The interceptor connects the policy engine, PII detector, cost tracker,
and audit logger. It is the central nervous system of AgentGuard.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Optional

from agentguard.core.events import (
    AgentEvent,
    EscalationEvent,
    LLMCallEvent,
    PolicyAction,
    PolicyViolationEvent,
    ToolCallEvent,
)
from agentguard.detectors.pii import PIIMatch, RegexPIIDetector
from agentguard.logging.audit import AuditLogger
from agentguard.policies.base import CombinedPolicyResult, PolicyEngine
from agentguard.tracking.cost import CostTracker


class PolicyViolationError(Exception):
    """Raised when a policy blocks an action."""

    def __init__(self, reason: str, policy_result: Optional[CombinedPolicyResult] = None):
        self.reason = reason
        self.policy_result = policy_result
        super().__init__(reason)


class Interceptor:
    """Before/after hooks for LLM and tool calls.

    Orchestrates policy checks, PII scanning, cost tracking,
    and audit logging for every agent action.
    """

    def __init__(
        self,
        policy_engine: PolicyEngine,
        audit_logger: AuditLogger,
        cost_tracker: CostTracker,
        pii_detector: Optional[RegexPIIDetector] = None,
        on_escalation: Optional[Callable[[EscalationEvent], Any]] = None,
    ) -> None:
        self._policy_engine = policy_engine
        self._audit = audit_logger
        self._cost = cost_tracker
        self._pii = pii_detector
        self._on_escalation = on_escalation

    # -- LLM calls ---------------------------------------------------------

    def before_llm_call(self, event: LLMCallEvent) -> LLMCallEvent:
        """Run before an LLM API call. May block or modify the event."""
        # PII scan on input messages
        if self._pii:
            for msg in event.messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    matches = self._pii.scan(content)
                    if matches:
                        event.input_pii_found.extend(
                            [{"type": m.pii_type, "value": m.value} for m in matches]
                        )

        # Run policies
        result = self._policy_engine.run(event)
        event.policy_results = result.to_dicts()

        if result.blocked:
            event.blocked = True
            event.block_reason = "; ".join(result.reasons)
            self._log_violation(event, result)
            self._audit.log(event)
            raise PolicyViolationError(event.block_reason, result)

        if result.action == PolicyAction.ESCALATE:
            self._handle_escalation(event, result)

        return event

    def after_llm_call(self, event: LLMCallEvent, response: Any) -> LLMCallEvent:
        """Run after an LLM API call. Logs results and scans output."""
        # Extract response data
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                event.response_content = choice.message.content or ""
            if hasattr(choice, "finish_reason"):
                event.finish_reason = choice.finish_reason

        # Extract usage (only if not already populated by a specific integration)
        if event.tokens_in == 0 and event.tokens_out == 0:
            if hasattr(response, "usage") and response.usage:
                event.tokens_in = getattr(response.usage, "prompt_tokens", 0)
                event.tokens_out = getattr(response.usage, "completion_tokens", 0)

        # Track cost
        event.cost_usd = self._cost.track(
            model=event.model,
            tokens_in=event.tokens_in,
            tokens_out=event.tokens_out,
            run_id=event.run_id,
            session_id=event.session_id,
        )

        # PII scan on output
        if self._pii and event.response_content:
            matches = self._pii.scan(event.response_content)
            if matches:
                event.output_pii_found = [
                    {"type": m.pii_type, "value": m.value} for m in matches
                ]

        # Run post-call policies (with response data now populated)
        result = self._policy_engine.run(event)
        event.policy_results = result.to_dicts()

        # Log the complete event
        self._audit.log(event)

        return event

    # -- Tool calls --------------------------------------------------------

    def before_tool_call(self, event: ToolCallEvent) -> ToolCallEvent:
        """Run before a tool call. May block the call."""
        # Run policies
        result = self._policy_engine.run(event)
        event.policy_results = result.to_dicts()

        if result.blocked:
            event.blocked = True
            event.block_reason = "; ".join(result.reasons)
            self._log_violation(event, result)
            self._audit.log(event)
            raise PolicyViolationError(event.block_reason, result)

        if result.action == PolicyAction.ESCALATE:
            self._handle_escalation(event, result)

        return event

    def after_tool_call(self, event: ToolCallEvent, result: Any) -> ToolCallEvent:
        """Run after a tool call. Logs the result."""
        event.result = result
        self._audit.log(event)
        return event

    # -- async variants ----------------------------------------------------

    async def abefore_llm_call(self, event: LLMCallEvent) -> LLMCallEvent:
        """Async version of before_llm_call."""
        # PII and policy logic is CPU-bound, reuse the sync implementation
        return self.before_llm_call(event)

    async def aafter_llm_call(self, event: LLMCallEvent, response: Any) -> LLMCallEvent:
        """Async version of after_llm_call."""
        return self.after_llm_call(event, response)

    async def abefore_tool_call(self, event: ToolCallEvent) -> ToolCallEvent:
        """Async version of before_tool_call."""
        return self.before_tool_call(event)

    async def aafter_tool_call(self, event: ToolCallEvent, result: Any) -> ToolCallEvent:
        """Async version of after_tool_call."""
        return self.after_tool_call(event, result)

    # -- internals ---------------------------------------------------------

    def _log_violation(self, event: AgentEvent, result: CombinedPolicyResult) -> None:
        """Log a PolicyViolationEvent."""
        for pr in result.results:
            if pr.blocked or pr.escalated:
                violation = PolicyViolationEvent(
                    run_id=event.run_id,
                    agent_id=event.agent_id,
                    session_id=event.session_id,
                    policy_name=pr.policy_name,
                    violation_reason=pr.reason,
                    action_taken=pr.action,
                    original_event_id=event.event_id,
                    original_event_type=event.event_type,
                )
                self._audit.log(violation)

    def _handle_escalation(self, event: AgentEvent, result: CombinedPolicyResult) -> None:
        """Handle escalation to human-in-the-loop."""
        escalation = EscalationEvent(
            run_id=event.run_id,
            agent_id=event.agent_id,
            session_id=event.session_id,
            reason="; ".join(result.reasons),
            original_event_id=event.event_id,
        )
        self._audit.log(escalation)

        if self._on_escalation:
            if inspect.iscoroutinefunction(self._on_escalation):
                # Schedule the async callback
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._on_escalation(escalation))
                except RuntimeError:
                    asyncio.run(self._on_escalation(escalation))
            else:
                self._on_escalation(escalation)
