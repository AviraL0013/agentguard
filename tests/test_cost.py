"""Tests for cost tracking."""

from agentguard.tracking.cost import CostTracker


class TestCostTracker:
    def test_track_returns_cost(self):
        tracker = CostTracker()
        cost = tracker.track("gpt-4o", tokens_in=1000, tokens_out=500)
        assert cost > 0

    def test_gpt4o_pricing(self):
        tracker = CostTracker()
        # GPT-4o: $2.50 per 1M input, $10.00 per 1M output
        cost = tracker.calculate_cost("gpt-4o", tokens_in=1_000_000, tokens_out=1_000_000)
        assert abs(cost - 12.50) < 0.01  # $2.50 + $10.00

    def test_gpt4o_mini_pricing(self):
        tracker = CostTracker()
        # GPT-4o-mini: $0.15 per 1M input, $0.60 per 1M output
        cost = tracker.calculate_cost("gpt-4o-mini", tokens_in=1_000_000, tokens_out=1_000_000)
        assert abs(cost - 0.75) < 0.01  # $0.15 + $0.60

    def test_run_cost_tracking(self):
        tracker = CostTracker()
        tracker.track("gpt-4o", tokens_in=1000, tokens_out=500, run_id="run1")
        tracker.track("gpt-4o", tokens_in=2000, tokens_out=1000, run_id="run1")
        tracker.track("gpt-4o", tokens_in=500, tokens_out=100, run_id="run2")

        run1_cost = tracker.get_run_cost("run1")
        run2_cost = tracker.get_run_cost("run2")
        assert run1_cost > run2_cost

    def test_total_cost(self):
        tracker = CostTracker()
        tracker.track("gpt-4o", tokens_in=1000, tokens_out=500)
        tracker.track("gpt-4o-mini", tokens_in=2000, tokens_out=1000)
        assert tracker.get_total_cost() > 0

    def test_total_tokens(self):
        tracker = CostTracker()
        tracker.track("gpt-4o", tokens_in=1000, tokens_out=500)
        tracker.track("gpt-4o", tokens_in=2000, tokens_out=1000)
        t_in, t_out = tracker.get_total_tokens()
        assert t_in == 3000
        assert t_out == 1500

    def test_run_tokens(self):
        tracker = CostTracker()
        tracker.track("gpt-4o", tokens_in=100, tokens_out=50, run_id="r1")
        tracker.track("gpt-4o", tokens_in=200, tokens_out=100, run_id="r1")
        t_in, t_out = tracker.get_run_tokens("r1")
        assert t_in == 300
        assert t_out == 150

    def test_custom_pricing(self):
        tracker = CostTracker()
        tracker.set_pricing("custom-model", 5.0, 15.0)
        cost = tracker.calculate_cost("custom-model", tokens_in=1_000_000, tokens_out=1_000_000)
        assert abs(cost - 20.0) < 0.01

    def test_unknown_model_returns_zero(self):
        tracker = CostTracker()
        cost = tracker.calculate_cost("unknown-model-xyz", tokens_in=1000, tokens_out=500)
        assert cost == 0.0

    def test_partial_model_match(self):
        tracker = CostTracker()
        # "gpt-4o-2024-05-13" should match "gpt-4o" pricing
        cost = tracker.calculate_cost("gpt-4o-2024-05-13", tokens_in=1_000_000, tokens_out=1_000_000)
        assert cost > 0

    def test_daily_cost(self):
        tracker = CostTracker()
        tracker.track("gpt-4o", tokens_in=1000, tokens_out=500)
        assert tracker.get_daily_cost() > 0
