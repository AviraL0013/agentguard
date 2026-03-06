"""Base types for the audit backend system.

Defines the ``AuditBackend`` protocol, ``AuditEntry`` data class
(with SHA-256 hash chaining), and ``RetentionPolicy``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, runtime_checkable


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Retention Policy
# ---------------------------------------------------------------------------

@dataclass
class RetentionPolicy:
    """Configures automatic log retention / cleanup.

    Args:
        max_age_days: Delete entries older than this (0 = unlimited).
        max_entries: Keep at most this many entries (0 = unlimited).
        max_size_bytes: Maximum total size in bytes (0 = unlimited).
    """

    max_age_days: int = 0
    max_entries: int = 0
    max_size_bytes: int = 0


# ---------------------------------------------------------------------------
# Audit Entry (hash-chained)
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    """A single tamper-evident audit log entry.

    Each entry includes a SHA-256 hash of ``data + previous_hash``,
    creating a hash chain that makes tampering detectable.
    """

    sequence: int
    timestamp: str = field(default_factory=_utcnow_iso)
    data: str = ""            # JSON-serialized event data
    hash: str = ""            # SHA-256(data + previous_hash)
    previous_hash: str = ""   # hash of the previous entry
    encrypted: bool = False   # whether data is encrypted

    def compute_hash(self) -> str:
        """Compute SHA-256 hash for this entry."""
        payload = f"{self.sequence}:{self.timestamp}:{self.data}:{self.previous_hash}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def fill_hash(self) -> AuditEntry:
        """Compute and set the hash field. Returns self for chaining."""
        self.hash = self.compute_hash()
        return self

    def verify(self) -> bool:
        """Verify this entry's hash is correct."""
        return self.hash == self.compute_hash()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "seq": self.sequence,
            "ts": self.timestamp,
            "data": self.data,
            "hash": self.hash,
            "prev_hash": self.previous_hash,
            "encrypted": self.encrypted,
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AuditEntry:
        """Deserialize from a dictionary."""
        return cls(
            sequence=d.get("seq", 0),
            timestamp=d.get("ts", ""),
            data=d.get("data", ""),
            hash=d.get("hash", ""),
            previous_hash=d.get("prev_hash", ""),
            encrypted=d.get("encrypted", False),
        )

    @classmethod
    def from_json(cls, s: str) -> AuditEntry:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Backend Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class AuditBackend(Protocol):
    """Interface for audit log storage backends.

    Implement this to create custom storage (database, cloud, SIEM, etc.).
    """

    def write(self, entry: AuditEntry) -> None:
        """Write a single audit entry to storage."""
        ...

    def read(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        run_id: str | None = None,
        event_type: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> list[AuditEntry]:
        """Read entries with optional filtering."""
        ...

    def verify_chain(self) -> dict[str, Any]:
        """Verify the hash chain integrity.

        Returns a dict with:
        - ``valid``: bool — whether the chain is intact
        - ``total_entries``: int — total entries checked
        - ``first_invalid``: int | None — sequence number of first break
        - ``errors``: list[str] — descriptions of any issues
        """
        ...

    def close(self) -> None:
        """Flush and close the backend."""
        ...

    def flush(self) -> None:
        """Force flush any buffered data."""
        ...
