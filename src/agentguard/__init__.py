"""AgentGuard — Compliance, governance & observability layer for AI agents.

The governance layer that lets companies trust their AI agents
enough to actually deploy them.

Usage::

    from agentguard import AgentGuard, AuditReader

    guard = AgentGuard(policies=["pii", "content_filter", "cost_limit"])

    # OpenAI
    safe_openai = guard.wrap_openai(client)           # sync
    safe_openai = guard.wrap_openai_async(client)     # async

    # Anthropic
    safe_claude = guard.wrap_anthropic(client)        # sync
    safe_claude = guard.wrap_anthropic_async(client)  # async

    safe_tool = guard.wrap_tool(my_function)          # auto-detects sync/async
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
from agentguard.detectors.pii import PIIDetector, RegexPIIDetector, PIIMatch
from agentguard.detectors.content import (
    ContentDetector,
    ContentMatch,
    RegexContentDetector,
    ToxicityDetector,
    SemanticInjectionDetector,
    BiasDetector,
    HallucinationDetector,
    CompositeContentDetector,
)

# Tracking
from agentguard.tracking.cost import CostTracker

# Logging & Audit
from agentguard.logging.audit import AuditLogger
from agentguard.logging.reader import AuditReader, RunTrace
from agentguard.logging.backends.base import AuditBackend, AuditEntry, RetentionPolicy
from agentguard.logging.backends.local import LocalFileBackend

__version__ = "0.1.3"

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
    # PII Detectors
    "PIIDetector",
    "RegexPIIDetector",
    "PIIMatch",
    # Content Detectors
    "ContentDetector",
    "ContentMatch",
    "RegexContentDetector",
    "ToxicityDetector",
    "SemanticInjectionDetector",
    "BiasDetector",
    "HallucinationDetector",
    "CompositeContentDetector",
    # Tracking
    "CostTracker",
    # Logging & Audit
    "AuditLogger",
    "AuditReader",
    "RunTrace",
    "AuditBackend",
    "AuditEntry",
    "RetentionPolicy",
    "LocalFileBackend",
]

# ---------------------------------------------------------------------------
# Optional Integrations & Detectors (Imported lazily)
# ---------------------------------------------------------------------------

try:
    from agentguard.detectors.presidio import PresidioPIIDetector
    __all__.append("PresidioPIIDetector")
except ImportError:  # presidio packages not installed
    pass  # type: ignore[assignment]

try:
    from agentguard.integrations.anthropic import GuardedAnthropic, AsyncGuardedAnthropic
    __all__.extend(["GuardedAnthropic", "AsyncGuardedAnthropic"])
except ImportError:  # anthropic package not installed
    pass  # type: ignore[assignment]

try:
    from agentguard.logging.backends.s3 import S3Backend
    __all__.append("S3Backend")
except ImportError:
    pass

try:
    from agentguard.logging.backends.postgres import PostgresBackend
    __all__.append("PostgresBackend")
except ImportError:
    pass

try:
    from agentguard.logging.backends.siem import SIEMBackend
    __all__.append("SIEMBackend")
except ImportError:
    pass
