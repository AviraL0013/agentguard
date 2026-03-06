"""Audit reader and replay engine — the killer feature.

Read JSON-lines audit logs, group events by run_id, and replay
agent runs step-by-step with a pretty-printed trace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RunTrace:
    """A single agent run — all events grouped by run_id."""

    run_id: str
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """Total duration of the run (first event to last event)."""
        if len(self.events) < 2:
            return 0.0
        # Approximate from latency fields
        return sum(e.get("latency_ms", 0) + e.get("duration_ms", 0) for e in self.events)

    @property
    def total_cost(self) -> float:
        return sum(e.get("cost_usd", 0) for e in self.events)

    @property
    def total_tokens(self) -> tuple[int, int]:
        t_in = sum(e.get("tokens_in", 0) for e in self.events)
        t_out = sum(e.get("tokens_out", 0) for e in self.events)
        return t_in, t_out

    @property
    def violations(self) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event_type") == "policy_violation"]

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0

    def print_trace(self, *, show_content: bool = True, max_content_length: int = 200) -> str:
        """Pretty-print the run trace and return it as a string."""
        lines: list[str] = []
        lines.append(f"╔══ AgentGuard Run Trace ══════════════════════════")
        lines.append(f"║ Run ID:  {self.run_id}")
        lines.append(f"║ Events:  {len(self.events)}")
        t_in, t_out = self.total_tokens
        lines.append(f"║ Tokens:  {t_in:,} in / {t_out:,} out")
        lines.append(f"║ Cost:    ${self.total_cost:.4f}")
        if self.has_violations:
            lines.append(f"║ ⚠️  Violations: {len(self.violations)}")
        lines.append(f"╠══════════════════════════════════════════════════")

        for i, event in enumerate(self.events, 1):
            event_type = event.get("event_type", "unknown")
            lines.append(f"║")

            if event_type == "llm_call":
                model = event.get("model", "unknown")
                tokens_in = event.get("tokens_in", 0)
                tokens_out = event.get("tokens_out", 0)
                cost = event.get("cost_usd", 0)
                blocked = event.get("blocked", False)

                status = "🚫 BLOCKED" if blocked else "✅"
                lines.append(f"║ Step {i}: [LLM Call] {status}")
                lines.append(f"║   Model:  {model}")
                lines.append(f"║   Tokens: {tokens_in} in / {tokens_out} out | Cost: ${cost:.4f}")

                if show_content:
                    messages = event.get("messages", [])
                    if messages:
                        last_msg = messages[-1].get("content", "")
                        if isinstance(last_msg, str) and last_msg:
                            truncated = last_msg[:max_content_length]
                            if len(last_msg) > max_content_length:
                                truncated += "..."
                            lines.append(f"║   Input:  {truncated}")

                    response = event.get("response_content", "")
                    if response:
                        truncated = response[:max_content_length]
                        if len(response) > max_content_length:
                            truncated += "..."
                        lines.append(f"║   Output: {truncated}")

                if blocked:
                    reason = event.get("block_reason", "")
                    lines.append(f"║   ⛔ Reason: {reason}")

                # Show PII detections
                pii_in = event.get("input_pii_found", [])
                pii_out = event.get("output_pii_found", [])
                if pii_in:
                    types = [p.get("type", "unknown") for p in pii_in]
                    lines.append(f"║   ⚠️  Input PII: {', '.join(types)}")
                if pii_out:
                    types = [p.get("type", "unknown") for p in pii_out]
                    lines.append(f"║   ⚠️  Output PII: {', '.join(types)}")

            elif event_type == "tool_call":
                tool = event.get("tool_name", "unknown")
                blocked = event.get("blocked", False)
                duration = event.get("duration_ms", 0)

                status = "🚫 BLOCKED" if blocked else "✅"
                lines.append(f"║ Step {i}: [Tool Call] {status}")
                lines.append(f"║   Tool:     {tool}")

                args = event.get("arguments", {})
                if args and show_content:
                    args_str = json.dumps(args, default=str)
                    if len(args_str) > max_content_length:
                        args_str = args_str[:max_content_length] + "..."
                    lines.append(f"║   Args:     {args_str}")

                if duration:
                    lines.append(f"║   Duration: {duration:.0f}ms")

                if blocked:
                    reason = event.get("block_reason", "")
                    lines.append(f"║   ⛔ Reason: {reason}")

            elif event_type == "policy_violation":
                policy = event.get("policy_name", "unknown")
                reason = event.get("violation_reason", "")
                action = event.get("action_taken", "block")
                lines.append(f"║ Step {i}: [Policy Violation] 🚨")
                lines.append(f"║   Policy: {policy}")
                lines.append(f"║   Reason: {reason}")
                lines.append(f"║   Action: {action}")

            elif event_type == "escalation":
                reason = event.get("reason", "")
                lines.append(f"║ Step {i}: [Escalation] 🔔")
                lines.append(f"║   Reason: {reason}")

            else:
                lines.append(f"║ Step {i}: [{event_type}]")

            # Show policy results for any event
            policy_results = event.get("policy_results", [])
            for pr in policy_results:
                if pr.get("action") != "allow":
                    lines.append(f"║   📋 Policy '{pr.get('policy', '')}': {pr.get('action')} — {pr.get('reason', '')}")

        lines.append(f"║")
        lines.append(f"╚══════════════════════════════════════════════════")

        output = "\n".join(lines)
        print(output)
        return output


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class AuditReader:
    """Read and query JSON-lines audit logs.

    Usage::

        reader = AuditReader("audit.jsonl")
        run = reader.get_run("run_abc123")
        run.print_trace()
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._events: list[dict[str, Any]] = []
        self._loaded = False

    # -- loading -----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.reload()

    def reload(self) -> None:
        """(Re)load events from the log file."""
        self._events = []
        if not self._path.exists():
            self._loaded = True
            return
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, dict) and "data" in parsed and "hash" in parsed and "seq" in parsed:
                            # Unwrap new AuditEntry format
                            try:
                                event_data = json.loads(parsed["data"])
                                if isinstance(event_data, dict):
                                    self._events.append(event_data)
                            except json.JSONDecodeError:
                                pass # Skip malformed inner json
                        else:
                            # Backward compatible flat JSON
                            self._events.append(parsed)
                    except json.JSONDecodeError:
                        continue  # skip malformed lines
        self._loaded = True

    # -- queries -----------------------------------------------------------

    def get_run(self, run_id: str) -> RunTrace:
        """Get all events for a specific run as a RunTrace."""
        self._ensure_loaded()
        events = [e for e in self._events if e.get("run_id") == run_id]
        return RunTrace(run_id=run_id, events=events)

    def get_all_runs(self) -> list[RunTrace]:
        """Get all runs, grouped by run_id."""
        self._ensure_loaded()
        runs: dict[str, list[dict[str, Any]]] = {}
        for event in self._events:
            rid = event.get("run_id", "unknown")
            runs.setdefault(rid, []).append(event)
        return [RunTrace(run_id=rid, events=evts) for rid, evts in runs.items()]

    def get_violations(self) -> list[dict[str, Any]]:
        """Get all policy violation events."""
        self._ensure_loaded()
        return [e for e in self._events if e.get("event_type") == "policy_violation"]

    def get_events(
        self,
        event_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query events with optional filters."""
        self._ensure_loaded()
        results = self._events
        if event_type:
            results = [e for e in results if e.get("event_type") == event_type]
        if agent_id:
            results = [e for e in results if e.get("agent_id") == agent_id]
        return results[:limit]

    @property
    def total_events(self) -> int:
        self._ensure_loaded()
        return len(self._events)

    @property
    def total_runs(self) -> int:
        self._ensure_loaded()
        return len({e.get("run_id") for e in self._events})

    @property
    def total_cost(self) -> float:
        self._ensure_loaded()
        return sum(e.get("cost_usd", 0) for e in self._events)
