"""OpenAI proxy wrapper — drop-in replacement for sync and async clients.

Uses the proxy pattern (NOT monkey-patching) for version safety.
Delegates all calls to the real client while intercepting them
through AgentGuard's policy engine.

Supports both ``openai.OpenAI`` (sync) and ``openai.AsyncOpenAI`` (async).
"""

from __future__ import annotations

import time
from typing import Any, Optional

from agentguard.core.events import LLMCallEvent
from agentguard.core.interceptor import Interceptor


# ---------------------------------------------------------------------------
# Sync wrappers
# ---------------------------------------------------------------------------

class GuardedCompletions:
    """Proxy for ``client.chat.completions``."""

    def __init__(
        self,
        real_completions: Any,
        interceptor: Interceptor,
        agent_id: str,
        run_id_ref: Any,
    ) -> None:
        self._real = real_completions
        self._interceptor = interceptor
        self._agent_id = agent_id
        self._run_id_ref = run_id_ref

    def create(self, **kwargs: Any) -> Any:
        """Guarded version of ``client.chat.completions.create()``."""
        run_id = self._run_id_ref.run_id if hasattr(self._run_id_ref, 'run_id') else "unknown"

        event = LLMCallEvent(
            run_id=run_id,
            agent_id=self._agent_id,
            model=kwargs.get("model", "unknown"),
            messages=kwargs.get("messages", []),
            temperature=kwargs.get("temperature"),
        )

        self._interceptor.before_llm_call(event)

        start = time.perf_counter()
        response = self._real.create(**kwargs)
        event.latency_ms = (time.perf_counter() - start) * 1000

        self._interceptor.after_llm_call(event, response)
        return response


class GuardedChat:
    """Proxy for ``client.chat``."""

    def __init__(
        self,
        real_chat: Any,
        interceptor: Interceptor,
        agent_id: str,
        run_id_ref: Any,
    ) -> None:
        self._real = real_chat
        self.completions = GuardedCompletions(
            real_chat.completions, interceptor, agent_id, run_id_ref
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class GuardedOpenAI:
    """Drop-in proxy for the OpenAI Python client (sync).

    Usage::

        from openai import OpenAI
        from agentguard import AgentGuard

        client = OpenAI()
        guard = AgentGuard(policies=["pii", "cost_limit"])
        safe = guard.wrap_openai(client)

        response = safe.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
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
        self.chat = GuardedChat(client.chat, interceptor, agent_id, self)

    @property
    def run_id(self) -> str:
        if self._guard_ref:
            return self._guard_ref.run_id
        return self._run_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------

class AsyncGuardedCompletions:
    """Async proxy for ``client.chat.completions``."""

    def __init__(
        self,
        real_completions: Any,
        interceptor: Interceptor,
        agent_id: str,
        run_id_ref: Any,
    ) -> None:
        self._real = real_completions
        self._interceptor = interceptor
        self._agent_id = agent_id
        self._run_id_ref = run_id_ref

    async def create(self, **kwargs: Any) -> Any:
        """Async guarded version of ``client.chat.completions.create()``."""
        run_id = self._run_id_ref.run_id if hasattr(self._run_id_ref, 'run_id') else "unknown"

        event = LLMCallEvent(
            run_id=run_id,
            agent_id=self._agent_id,
            model=kwargs.get("model", "unknown"),
            messages=kwargs.get("messages", []),
            temperature=kwargs.get("temperature"),
        )

        # Before — may raise PolicyViolationError
        await self._interceptor.abefore_llm_call(event)

        # Execute the real async API call
        start = time.perf_counter()
        response = await self._real.create(**kwargs)
        event.latency_ms = (time.perf_counter() - start) * 1000

        # After — logs everything
        await self._interceptor.aafter_llm_call(event, response)

        return response


class AsyncGuardedChat:
    """Async proxy for ``client.chat``."""

    def __init__(
        self,
        real_chat: Any,
        interceptor: Interceptor,
        agent_id: str,
        run_id_ref: Any,
    ) -> None:
        self._real = real_chat
        self.completions = AsyncGuardedCompletions(
            real_chat.completions, interceptor, agent_id, run_id_ref
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class AsyncGuardedOpenAI:
    """Drop-in proxy for the async OpenAI Python client.

    Usage::

        from openai import AsyncOpenAI
        from agentguard import AgentGuard

        client = AsyncOpenAI()
        guard = AgentGuard(policies=["pii", "cost_limit"])
        safe = guard.wrap_openai_async(client)

        response = await safe.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
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
        self.chat = AsyncGuardedChat(client.chat, interceptor, agent_id, self)

    @property
    def run_id(self) -> str:
        if self._guard_ref:
            return self._guard_ref.run_id
        return self._run_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
