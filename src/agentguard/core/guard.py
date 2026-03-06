"""AgentGuard — the main orchestrator.

This is the single class that users interact with.
It configures policies, creates the interceptor, and provides
methods to wrap OpenAI clients and tools.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional, TypeVar, Union

from agentguard.core.events import LLMCallEvent, ToolCallEvent
from agentguard.core.interceptor import Interceptor, PolicyViolationError
from agentguard.detectors.content import CompositeContentDetector, ContentDetector
from agentguard.detectors.pii import PIIDetector, RegexPIIDetector
from agentguard.logging.audit import AuditLogger
from agentguard.logging.backends.base import AuditBackend
from agentguard.logging.reader import AuditReader
from agentguard.policies.base import Policy, PolicyEngine
from agentguard.policies.content_policy import ContentPolicy
from agentguard.policies.cost_policy import CostPolicy
from agentguard.policies.pii_policy import PIIPolicy
from agentguard.policies.rate_limit_policy import RateLimitPolicy
from agentguard.policies.tool_policy import ToolPolicy
from agentguard.tracking.cost import CostTracker

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Policy shortcuts (string aliases)
# ---------------------------------------------------------------------------

_POLICY_ALIASES = {
    "pii",
    "cost_limit",
    "rate_limit",
    "content_filter",
    "tool_restriction",
}


class AgentGuard:
    """The main entry point for AgentGuard.

    Usage::

        guard = AgentGuard(
            policies=["pii", "cost_limit"],
            audit_path="audit.jsonl",
            cost_limit=5.00,
        )
        safe_client = guard.wrap_openai(client)
        safe_tool = guard.wrap_tool(my_function)

    Args:
        policies: List of Policy instances or string aliases
            ("pii", "cost_limit", "rate_limit", "content_filter", "tool_restriction").
        audit_path: Path for the JSON-lines audit log file.
        agent_id: Identifier for this agent.
        cost_limit: Max USD per run (used with "cost_limit" policy).
        daily_cost_limit: Max USD per day.
        total_cost_limit: Max USD total.
        rate_limit: Max calls per minute (used with "rate_limit" policy).
        blocked_tools: Tool names to block (used with "tool_restriction" policy).
        allowed_tools: Tool names to allow (used with "tool_restriction" policy).
        on_escalation: Callback for human-in-the-loop escalations.
    """

    def __init__(
        self,
        *,
        policies: Optional[list[Union[str, Policy]]] = None,
        audit_path: str | Path = "agentguard_audit.jsonl",
        audit_backends: list[AuditBackend] | None = None,
        audit_encryption_key: str | None = None,
        agent_id: str = "default",
        pii_detector: PIIDetector | None = None,
        content_detectors: list[ContentDetector] | None = None,
        content_threshold: float = 0.5,
        cost_limit: float = 0,
        daily_cost_limit: float = 0,
        total_cost_limit: float = 0,
        rate_limit: int = 0,
        blocked_tools: Optional[list[str]] = None,
        allowed_tools: Optional[list[str]] = None,
        on_escalation: Optional[Callable] = None,
    ) -> None:
        self._agent_id = agent_id

        # Core components
        self._pii_detector: PIIDetector = pii_detector or RegexPIIDetector()
        self._content_detectors = content_detectors
        self._content_threshold = content_threshold
        self._cost_tracker = CostTracker()
        self._audit_logger = AuditLogger(
            audit_path,
            backends=audit_backends,
            encryption_key=audit_encryption_key,
        )
        self._audit_path = Path(audit_path)

        # Build policy list
        resolved_policies = self._resolve_policies(
            policies or ["pii", "content_filter"],
            cost_limit=cost_limit,
            daily_cost_limit=daily_cost_limit,
            total_cost_limit=total_cost_limit,
            rate_limit_val=rate_limit,
            blocked_tools=blocked_tools,
            allowed_tools=allowed_tools,
        )
        self._policy_engine = PolicyEngine(resolved_policies)

        # Create interceptor
        self._interceptor = Interceptor(
            policy_engine=self._policy_engine,
            audit_logger=self._audit_logger,
            cost_tracker=self._cost_tracker,
            pii_detector=self._pii_detector,
            on_escalation=on_escalation,
        )

        # Run tracking
        self._current_run_id: str = uuid.uuid4().hex[:16]

    # -- public API --------------------------------------------------------

    def new_run(self) -> str:
        """Start a new run and return the run_id."""
        self._current_run_id = uuid.uuid4().hex[:16]
        return self._current_run_id

    @property
    def run_id(self) -> str:
        """Current run ID."""
        return self._current_run_id

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker

    @property
    def policy_engine(self) -> PolicyEngine:
        return self._policy_engine

    def wrap_openai(self, client: Any) -> Any:
        """Wrap a sync OpenAI client with a guarded proxy.

        Returns a ``GuardedOpenAI`` instance that behaves identically
        to the original client but runs all calls through AgentGuard.
        """
        from agentguard.integrations.openai import GuardedOpenAI

        return GuardedOpenAI(client, self._interceptor, self._agent_id, self._current_run_id, self)

    def wrap_openai_async(self, client: Any) -> Any:
        """Wrap an async OpenAI client (``AsyncOpenAI``) with a guarded proxy.

        Returns an ``AsyncGuardedOpenAI`` instance::

            from openai import AsyncOpenAI
            client = AsyncOpenAI()
            safe = guard.wrap_openai_async(client)
            response = await safe.chat.completions.create(...)
        """
        from agentguard.integrations.openai import AsyncGuardedOpenAI

        return AsyncGuardedOpenAI(client, self._interceptor, self._agent_id, self._current_run_id, self)

    def wrap_anthropic(self, client: Any) -> Any:
        """Wrap a sync Anthropic client (``anthropic.Anthropic``) with a guarded proxy.

        Returns a ``GuardedAnthropic`` instance that behaves identically
        to the original client but runs all calls through AgentGuard::

            import anthropic
            client = anthropic.Anthropic()
            safe = guard.wrap_anthropic(client)
            response = safe.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello!"}],
            )
        """
        from agentguard.integrations.anthropic import GuardedAnthropic

        return GuardedAnthropic(client, self._interceptor, self._agent_id, self._current_run_id, self)

    def wrap_anthropic_async(self, client: Any) -> Any:
        """Wrap an async Anthropic client (``anthropic.AsyncAnthropic``) with a guarded proxy.

        Returns an ``AsyncGuardedAnthropic`` instance::

            import anthropic
            client = anthropic.AsyncAnthropic()
            safe = guard.wrap_anthropic_async(client)
            response = await safe.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello!"}],
            )
        """
        from agentguard.integrations.anthropic import AsyncGuardedAnthropic

        return AsyncGuardedAnthropic(client, self._interceptor, self._agent_id, self._current_run_id, self)

    def wrap_tool(self, func: F, *, tool_name: Optional[str] = None) -> F:
        """Wrap a sync tool function with policy enforcement.

        The wrapped function will be checked against tool policies
        before execution and logged after execution.

        Args:
            func: The tool function to wrap.
            tool_name: Override name (defaults to func.__name__).
        """
        name = tool_name or getattr(func, "__name__", "unknown_tool")

        # Auto-detect async functions and use the async wrapper
        if inspect.iscoroutinefunction(func):
            return self.wrap_tool_async(func, tool_name=name)  # type: ignore

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            event = ToolCallEvent(
                run_id=self._current_run_id,
                agent_id=self._agent_id,
                tool_name=name,
                arguments=kwargs if kwargs else {"args": list(args)},
            )

            # Before — may raise PolicyViolationError
            self._interceptor.before_tool_call(event)

            # Execute the tool
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                event.result = f"ERROR: {type(e).__name__}: {e}"
                event.duration_ms = (time.perf_counter() - start) * 1000
                self._interceptor.after_tool_call(event, event.result)
                raise

            event.duration_ms = (time.perf_counter() - start) * 1000
            self._interceptor.after_tool_call(event, result)
            return result

        return wrapper  # type: ignore

    def wrap_tool_async(self, func: F, *, tool_name: Optional[str] = None) -> F:
        """Wrap an async tool function with policy enforcement.

        Usage::

            async def send_email(to: str, body: str) -> str:
                ...

            safe_send = guard.wrap_tool_async(send_email)
            result = await safe_send(to="user@test.com", body="Hi")
        """
        name = tool_name or getattr(func, "__name__", "unknown_tool")

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            event = ToolCallEvent(
                run_id=self._current_run_id,
                agent_id=self._agent_id,
                tool_name=name,
                arguments=kwargs if kwargs else {"args": list(args)},
            )

            # Before — may raise PolicyViolationError
            await self._interceptor.abefore_tool_call(event)

            # Execute the async tool
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                event.result = f"ERROR: {type(e).__name__}: {e}"
                event.duration_ms = (time.perf_counter() - start) * 1000
                await self._interceptor.aafter_tool_call(event, event.result)
                raise

            event.duration_ms = (time.perf_counter() - start) * 1000
            await self._interceptor.aafter_tool_call(event, result)
            return result

        return async_wrapper  # type: ignore

    def get_reader(self) -> AuditReader:
        """Get an AuditReader for the current audit log."""
        self._audit_logger.close()
        return AuditReader(self._audit_path)

    def get_report(self) -> dict[str, Any]:
        """Get a summary report of all activity."""
        t_in, t_out = self._cost_tracker.get_total_tokens()
        return {
            "total_cost_usd": self._cost_tracker.get_total_cost(),
            "total_tokens_in": t_in,
            "total_tokens_out": t_out,
            "daily_cost_usd": self._cost_tracker.get_daily_cost(),
            "run_id": self._current_run_id,
            "run_cost_usd": self._cost_tracker.get_run_cost(self._current_run_id),
            "policies_active": [p.name for p in self._policy_engine.policies],
        }

    def close(self) -> None:
        """Close the audit logger."""
        self._audit_logger.close()

    def __enter__(self) -> AgentGuard:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    async def __aenter__(self) -> AgentGuard:
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _resolve_policies(
        self,
        policies: list[Union[str, Policy]],
        *,
        cost_limit: float,
        daily_cost_limit: float,
        total_cost_limit: float,
        rate_limit_val: int,
        blocked_tools: Optional[list[str]],
        allowed_tools: Optional[list[str]],
    ) -> list[Policy]:
        """Resolve string aliases to Policy instances."""
        resolved: list[Policy] = []
        for p in policies:
            if isinstance(p, Policy):
                resolved.append(p)
            elif p == "pii":
                resolved.append(PIIPolicy(detector=self._pii_detector))
            elif p == "cost_limit":
                resolved.append(
                    CostPolicy(
                        tracker=self._cost_tracker,
                        max_run_cost=cost_limit,
                        max_daily_cost=daily_cost_limit,
                        max_total_cost=total_cost_limit,
                    )
                )
            elif p == "rate_limit":
                resolved.append(
                    RateLimitPolicy(max_calls=rate_limit_val or 60)
                )
            elif p == "content_filter":
                if self._content_detectors:
                    detector = CompositeContentDetector(
                        detectors=list(self._content_detectors),
                        threshold=self._content_threshold,
                    )
                    resolved.append(ContentPolicy(detector=detector, threshold=self._content_threshold))
                else:
                    resolved.append(ContentPolicy(threshold=self._content_threshold))
            elif p == "tool_restriction":
                resolved.append(
                    ToolPolicy(
                        blocked_tools=blocked_tools,
                        allowed_tools=allowed_tools,
                    )
                )
            else:
                raise ValueError(
                    f"Unknown policy alias: '{p}'. "
                    f"Valid aliases: {sorted(_POLICY_ALIASES)}"
                )
        return resolved
