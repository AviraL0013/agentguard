"""Anthropic proxy wrapper — drop-in replacement for sync and async clients.

Uses the proxy pattern (NOT monkey-patching) for version safety.
Delegates all calls to the real Anthropic client while intercepting them
through AgentGuard's policy engine.

Supports both ``anthropic.Anthropic`` (sync) and ``anthropic.AsyncAnthropic`` (async).

Key differences from the OpenAI integration:

* Anthropic's call is ``client.messages.create()`` (not ``chat.completions.create``).
* The ``system`` prompt is a top-level string parameter, not a message dict.
* Response text lives in ``response.content[0].text``.
* Token counts are ``response.usage.input_tokens`` / ``response.usage.output_tokens``.
* Stop reason is ``response.stop_reason`` (not ``finish_reason``).

All of these are normalised into the shared ``LLMCallEvent`` transparently.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from agentguard.core.events import LLMCallEvent
from agentguard.core.interceptor import Interceptor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_llm_event(run_id: str, agent_id: str, kwargs: dict[str, Any]) -> LLMCallEvent:
    """Build an ``LLMCallEvent`` from Anthropic ``messages.create`` kwargs.

    Anthropic allows a top-level ``system`` string alongside the ``messages``
    list.  We fold it into the messages list as a synthetic ``{"role": "system",
    "content": ...}`` entry so the rest of AgentGuard (PII scan, audit log,
    etc.) sees a unified format.
    """
    messages: list[dict[str, Any]] = list(kwargs.get("messages", []))

    # Fold the optional top-level `system` prompt into the messages list
    system_prompt: Optional[str] = kwargs.get("system")
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    return LLMCallEvent(
        run_id=run_id,
        agent_id=agent_id,
        model=kwargs.get("model", "unknown"),
        messages=messages,
        temperature=kwargs.get("temperature"),
    )


def _extract_response(event: LLMCallEvent, response: Any) -> None:
    """Populate ``event`` fields from an Anthropic ``Message`` response object."""
    # Response text — Anthropic returns a list of content blocks
    if hasattr(response, "content") and response.content:
        first = response.content[0]
        if hasattr(first, "text"):
            event.response_content = first.text

    # Stop reason (maps to OpenAI's finish_reason)
    if hasattr(response, "stop_reason"):
        event.finish_reason = response.stop_reason

    # Token usage
    if hasattr(response, "usage") and response.usage:
        event.tokens_in = getattr(response.usage, "input_tokens", 0)
        event.tokens_out = getattr(response.usage, "output_tokens", 0)


# ---------------------------------------------------------------------------
# Sync wrappers
# ---------------------------------------------------------------------------

class GuardedAnthropicMessages:
    """Proxy for ``client.messages``."""

    def __init__(
        self,
        real_messages: Any,
        interceptor: Interceptor,
        agent_id: str,
        run_id_ref: Any,
    ) -> None:
        self._real = real_messages
        self._interceptor = interceptor
        self._agent_id = agent_id
        self._run_id_ref = run_id_ref

    def create(self, **kwargs: Any) -> Any:
        """Guarded version of ``client.messages.create()``."""
        run_id = (
            self._run_id_ref.run_id
            if hasattr(self._run_id_ref, "run_id")
            else "unknown"
        )

        event = _build_llm_event(run_id, self._agent_id, kwargs)

        # Before — may raise PolicyViolationError
        self._interceptor.before_llm_call(event)

        # Execute the real API call
        start = time.perf_counter()
        response = self._real.create(**kwargs)
        event.latency_ms = (time.perf_counter() - start) * 1000

        # Populate response fields
        _extract_response(event, response)

        # After — cost tracking + output PII scan + audit log
        self._interceptor.after_llm_call(event, response)
        return response


class GuardedAnthropic:
    """Drop-in proxy for ``anthropic.Anthropic`` (sync).

    Usage::

        import anthropic
        from agentguard import AgentGuard

        client = anthropic.Anthropic()
        guard = AgentGuard(policies=["pii", "cost_limit"])
        safe = guard.wrap_anthropic(client)

        response = safe.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello, Claude!"}],
        )
        print(response.content[0].text)
    """

    def __init__(
        self,
        client: Any,
        interceptor: Interceptor,
        agent_id: str,
        run_id: str,
        guard_ref: Any = None,
    ) -> None:
        self._client = client
        self._interceptor = interceptor
        self._agent_id = agent_id
        self._run_id = run_id
        self._guard_ref = guard_ref
        self.messages = GuardedAnthropicMessages(
            client.messages, interceptor, agent_id, self
        )

    @property
    def run_id(self) -> str:
        if self._guard_ref:
            return self._guard_ref.run_id
        return self._run_id

    def __getattr__(self, name: str) -> Any:
        """Pass through any attribute not explicitly handled (e.g. ``models``)."""
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------

class AsyncGuardedAnthropicMessages:
    """Async proxy for ``client.messages``."""

    def __init__(
        self,
        real_messages: Any,
        interceptor: Interceptor,
        agent_id: str,
        run_id_ref: Any,
    ) -> None:
        self._real = real_messages
        self._interceptor = interceptor
        self._agent_id = agent_id
        self._run_id_ref = run_id_ref

    async def create(self, **kwargs: Any) -> Any:
        """Async guarded version of ``client.messages.create()``."""
        run_id = (
            self._run_id_ref.run_id
            if hasattr(self._run_id_ref, "run_id")
            else "unknown"
        )

        event = _build_llm_event(run_id, self._agent_id, kwargs)

        # Before — may raise PolicyViolationError
        await self._interceptor.abefore_llm_call(event)

        # Execute the real async API call
        start = time.perf_counter()
        response = await self._real.create(**kwargs)
        event.latency_ms = (time.perf_counter() - start) * 1000

        # Populate response fields
        _extract_response(event, response)

        # After — cost tracking + output PII scan + audit log
        await self._interceptor.aafter_llm_call(event, response)
        return response


class AsyncGuardedAnthropic:
    """Drop-in proxy for ``anthropic.AsyncAnthropic``.

    Usage::

        import anthropic
        from agentguard import AgentGuard

        async with AgentGuard(policies=["pii", "cost_limit"]) as guard:
            client = anthropic.AsyncAnthropic()
            safe = guard.wrap_anthropic_async(client)

            response = await safe.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello, Claude!"}],
            )
            print(response.content[0].text)
    """

    def __init__(
        self,
        client: Any,
        interceptor: Interceptor,
        agent_id: str,
        run_id: str,
        guard_ref: Any = None,
    ) -> None:
        self._client = client
        self._interceptor = interceptor
        self._agent_id = agent_id
        self._run_id = run_id
        self._guard_ref = guard_ref
        self.messages = AsyncGuardedAnthropicMessages(
            client.messages, interceptor, agent_id, self
        )

    @property
    def run_id(self) -> str:
        if self._guard_ref:
            return self._guard_ref.run_id
        return self._run_id

    def __getattr__(self, name: str) -> Any:
        """Pass through any attribute not explicitly handled."""
        return getattr(self._client, name)
