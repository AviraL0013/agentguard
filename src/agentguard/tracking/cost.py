"""Token and cost tracking for LLM API calls.

Maintains per-model pricing and tracks spend per run, per session,
and in aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Pricing table (USD per 1M tokens)
# ---------------------------------------------------------------------------

# Source: official pricing pages as of early 2026.
# Users can override / extend via CostTracker.set_pricing().

_DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    # model_name: (input_cost_per_1M, output_cost_per_1M)

    # --- OpenAI ---
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-2024-11-20": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3-mini": (1.10, 4.40),

    # --- Anthropic Claude (generic aliases + versioned IDs) ---
    # Generic aliases (prefix-matched for older model strings)
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    # Versioned IDs (exact match, takes priority over prefix match)
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-3-sonnet-20240229": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),

    # --- Google Gemini ---
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-pro": (1.25, 5.00),
}


@dataclass
class CostRecord:
    """A single cost record for one LLM call."""

    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    run_id: str = ""
    session_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CostTracker:
    """Track token usage and costs across all LLM calls.

    Usage::

        tracker = CostTracker()
        cost = tracker.track("gpt-4o", tokens_in=500, tokens_out=200, run_id="run_1")
        print(f"This call cost ${cost:.4f}")
        print(f"Run total: ${tracker.get_run_cost('run_1'):.4f}")
        print(f"All time:  ${tracker.get_total_cost():.4f}")
    """

    def __init__(self) -> None:
        self._pricing: dict[str, tuple[float, float]] = dict(_DEFAULT_PRICING)
        self._records: list[CostRecord] = []

    # -- configuration -----------------------------------------------------

    def set_pricing(
        self,
        model: str,
        input_cost_per_million: float,
        output_cost_per_million: float,
    ) -> None:
        """Set or override pricing for a model."""
        self._pricing[model] = (input_cost_per_million, output_cost_per_million)

    # -- tracking ----------------------------------------------------------

    def track(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
        run_id: str = "",
        session_id: str = "",
    ) -> float:
        """Record token usage and return the cost in USD."""
        cost = self.calculate_cost(model, tokens_in, tokens_out)
        self._records.append(
            CostRecord(
                model=model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
                run_id=run_id,
                session_id=session_id,
            )
        )
        return cost

    def calculate_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Calculate cost without recording it."""
        pricing = self._pricing.get(model)
        if pricing is None:
            # Try partial match (e.g. "gpt-4o-2024-05-13" → "gpt-4o")
            for known_model, p in self._pricing.items():
                if model.startswith(known_model):
                    pricing = p
                    break
        if pricing is None:
            return 0.0  # unknown model — can't price
        in_cost, out_cost = pricing
        return (tokens_in * in_cost / 1_000_000) + (tokens_out * out_cost / 1_000_000)

    # -- queries -----------------------------------------------------------

    def get_run_cost(self, run_id: str) -> float:
        """Total cost for a specific run."""
        return sum(r.cost_usd for r in self._records if r.run_id == run_id)

    def get_session_cost(self, session_id: str) -> float:
        """Total cost for a session."""
        return sum(r.cost_usd for r in self._records if r.session_id == session_id)

    def get_total_cost(self) -> float:
        """Total cost across all recorded calls."""
        return sum(r.cost_usd for r in self._records)

    def get_daily_cost(self, day: Optional[date] = None) -> float:
        """Total cost for a specific day (defaults to today UTC)."""
        target = day or datetime.now(timezone.utc).date()
        return sum(
            r.cost_usd for r in self._records if r.timestamp.date() == target
        )

    def get_run_tokens(self, run_id: str) -> tuple[int, int]:
        """(total_in, total_out) tokens for a run."""
        t_in = sum(r.tokens_in for r in self._records if r.run_id == run_id)
        t_out = sum(r.tokens_out for r in self._records if r.run_id == run_id)
        return t_in, t_out

    def get_total_tokens(self) -> tuple[int, int]:
        """(total_in, total_out) tokens across all calls."""
        return (
            sum(r.tokens_in for r in self._records),
            sum(r.tokens_out for r in self._records),
        )

    @property
    def records(self) -> list[CostRecord]:
        """All cost records (read-only copy)."""
        return list(self._records)
