"""Tests for event models."""

from agentguard.core.events import (
    AgentEvent,
    EscalationEvent,
    EventType,
    LLMCallEvent,
    PolicyAction,
    PolicyViolationEvent,
    ToolCallEvent,
)


class TestEventModels:
    def test_llm_call_event_defaults(self):
        event = LLMCallEvent()
        assert event.event_type == "llm_call"
        assert event.run_id
        assert event.event_id
        assert event.tokens_in == 0
        assert event.blocked is False

    def test_llm_call_event_with_data(self):
        event = LLMCallEvent(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            tokens_in=10,
            tokens_out=20,
            cost_usd=0.001,
        )
        assert event.model == "gpt-4o"
        assert event.tokens_in == 10
        assert event.cost_usd == 0.001

    def test_tool_call_event(self):
        event = ToolCallEvent(
            tool_name="send_email",
            arguments={"to": "test@example.com", "body": "Hi"},
        )
        assert event.event_type == "tool_call"
        assert event.tool_name == "send_email"
        assert event.blocked is False

    def test_policy_violation_event(self):
        event = PolicyViolationEvent(
            policy_name="pii",
            violation_reason="Email detected",
            action_taken=PolicyAction.BLOCK,
        )
        assert event.event_type == "policy_violation"
        assert event.policy_name == "pii"

    def test_escalation_event(self):
        event = EscalationEvent(
            reason="High-risk decision",
            escalated_to="admin@company.com",
        )
        assert event.event_type == "escalation"
        assert event.resolved is False

    def test_event_serialization(self):
        event = LLMCallEvent(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
        )
        json_str = event.model_dump_json()
        assert "gpt-4o" in json_str
        assert "llm_call" in json_str

    def test_run_id_grouping(self):
        """Events with same run_id belong to same workflow."""
        run_id = "test_run_123"
        e1 = LLMCallEvent(run_id=run_id)
        e2 = ToolCallEvent(run_id=run_id)
        e3 = LLMCallEvent(run_id=run_id)
        assert e1.run_id == e2.run_id == e3.run_id == run_id

    def test_unique_event_ids(self):
        e1 = LLMCallEvent()
        e2 = LLMCallEvent()
        assert e1.event_id != e2.event_id
