"""Pydantic event models for AgentGuard.

Every agent action (LLM call, tool call, policy violation, escalation)
is captured as a typed, serializable event with a run_id for grouping.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    POLICY_VIOLATION = "policy_violation"
    ESCALATION = "escalation"


class PolicyAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    ESCALATE = "escalate"
    REDACT = "redact"


# ---------------------------------------------------------------------------
# Base Event
# ---------------------------------------------------------------------------

class AgentEvent(BaseModel):
    """Base event — every event carries these fields."""

    event_id: str = Field(default_factory=_new_id)
    event_type: EventType
    run_id: str = Field(default_factory=_new_id, description="Groups multi-step agent workflows")
    agent_id: str = "default"
    session_id: str = Field(default_factory=_new_id)
    timestamp: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# LLM Call Event
# ---------------------------------------------------------------------------

class LLMCallEvent(AgentEvent):
    """Captures a single LLM API call."""

    event_type: EventType = EventType.LLM_CALL

    # Request
    model: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    temperature: Optional[float] = None

    # Response
    response_content: Optional[str] = None
    finish_reason: Optional[str] = None

    # Metrics
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    # Guard results
    input_pii_found: list[dict[str, Any]] = Field(default_factory=list)
    output_pii_found: list[dict[str, Any]] = Field(default_factory=list)
    policy_results: list[dict[str, Any]] = Field(default_factory=list)
    blocked: bool = False
    block_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Tool Call Event
# ---------------------------------------------------------------------------

class ToolCallEvent(AgentEvent):
    """Captures a tool/function call made by an agent."""

    event_type: EventType = EventType.TOOL_CALL

    tool_name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    duration_ms: float = 0.0

    # Guard results
    policy_results: list[dict[str, Any]] = Field(default_factory=list)
    blocked: bool = False
    block_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Policy Violation Event
# ---------------------------------------------------------------------------

class PolicyViolationEvent(AgentEvent):
    """Logged when a policy is violated."""

    event_type: EventType = EventType.POLICY_VIOLATION

    policy_name: str = ""
    violation_reason: str = ""
    action_taken: PolicyAction = PolicyAction.BLOCK
    original_event_id: Optional[str] = None
    original_event_type: Optional[str] = None


# ---------------------------------------------------------------------------
# Escalation Event
# ---------------------------------------------------------------------------

class EscalationEvent(AgentEvent):
    """Logged when a decision is escalated to a human."""

    event_type: EventType = EventType.ESCALATION

    reason: str = ""
    escalated_to: Optional[str] = None
    original_event_id: Optional[str] = None
    resolved: bool = False
    resolution: Optional[str] = None
