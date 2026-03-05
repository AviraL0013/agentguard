"""AgentGuard — Compliance, governance & observability layer for AI agents.

The governance layer that lets companies trust their AI agents
enough to actually deploy them.

Usage::

    from agentguard import AgentGuard, AuditReader

    guard = AgentGuard(policies=["pii", "content_filter", "cost_limit"])
    safe_client = guard.wrap_openai(client)        # sync
    safe_client = guard.wrap_openai_async(client)   # async
    safe_tool = guard.wrap_tool(my_function)        # auto-detects sync/async
"""

# Core
from agentguard.core.guard import AgentGuard
from agentguard.core.events import (
    AgentEvent,
    EventType,
    LLMCallEvent,
    ToolCallEvent,
    PolicyViolationEvent,
    EscalationEvent,
    PolicyAction,
)
from agentguard.core.interceptor import PolicyViolationError

# Policies
from agentguard.policies.base import Policy, PolicyResult, PolicyEngine, CombinedPolicyResult
from agentguard.policies.pii_policy import PIIPolicy
from agentguard.policies.cost_policy import CostPolicy
from agentguard.policies.tool_policy import ToolPolicy
from agentguard.policies.rate_limit_policy import RateLimitPolicy
from agentguard.policies.content_policy import ContentPolicy

# Detectors
from agentguard.detectors.pii import RegexPIIDetector, PIIMatch

# Tracking
from agentguard.tracking.cost import CostTracker

# Logging & Audit
from agentguard.logging.audit import AuditLogger
from agentguard.logging.reader import AuditReader, RunTrace

__version__ = "0.1.0"

__all__ = [
    # Core
    "AgentGuard",
    "PolicyViolationError",
    # Events
    "AgentEvent",
    "EventType",
    "LLMCallEvent",
    "ToolCallEvent",
    "PolicyViolationEvent",
    "EscalationEvent",
    "PolicyAction",
    # Policies
    "Policy",
    "PolicyResult",
    "PolicyEngine",
    "CombinedPolicyResult",
    "PIIPolicy",
    "CostPolicy",
    "ToolPolicy",
    "RateLimitPolicy",
    "ContentPolicy",
    # Detectors
    "RegexPIIDetector",
    "PIIMatch",
    # Tracking
    "CostTracker",
    # Logging
    "AuditLogger",
    "AuditReader",
    "RunTrace",
]
