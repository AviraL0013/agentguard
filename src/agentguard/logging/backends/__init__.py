"""Audit backend package — pluggable storage backends for audit logging."""

from agentguard.logging.backends.base import (
    AuditBackend,
    AuditEntry,
    RetentionPolicy,
)
from agentguard.logging.backends.local import LocalFileBackend

__all__ = [
    "AuditBackend",
    "AuditEntry",
    "RetentionPolicy",
    "LocalFileBackend",
]

# Optional backends — imported lazily to avoid hard deps
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
