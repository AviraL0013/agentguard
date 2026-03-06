"""Thread-safe audit logger with pluggable backends and hash chaining.

The rewritten ``AuditLogger`` supports multiple backends (fan-out),
SHA-256 hash chaining for tamper evidence, and optional encryption.

Backward compatible — the default behavior is identical to the original
flat-file JSONL logger, but now with hash chaining built in.

Usage::

    # Simple (backward compatible)
    logger = AuditLogger("audit.jsonl")

    # With encryption
    logger = AuditLogger("audit.jsonl", encryption_key=Fernet.generate_key())

    # Multiple backends
    from agentguard.logging.backends import LocalFileBackend, S3Backend
    logger = AuditLogger(backends=[
        LocalFileBackend("audit.jsonl"),
        S3Backend("my-bucket", prefix="prod"),
    ])
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TextIO

from agentguard.core.events import AgentEvent
from agentguard.logging.backends.base import AuditBackend, AuditEntry
from agentguard.logging.backends.local import LocalFileBackend


class AuditLogger:
    """Thread-safe audit logger with pluggable backends.

    Features:
    - SHA-256 hash chaining across all entries
    - Multi-backend fan-out (write to multiple targets)
    - Optional Fernet encryption at rest
    - Backward compatible with the original API

    Args:
        path: Path for the default local file backend.
            Ignored if ``backends`` is provided.
        backends: List of ``AuditBackend`` instances for fan-out.
        encryption_key: Fernet key for encrypting data at rest
            (only applies to the default ``LocalFileBackend``).
        max_size_mb: Max file size before rotation (local backend only).
        rotate_daily: Daily file rotation (local backend only).
    """

    def __init__(
        self,
        path: str | Path = "agentguard_audit.jsonl",
        *,
        backends: list[AuditBackend] | None = None,
        encryption_key: str | None = None,
        max_size_mb: float = 100,
        rotate_daily: bool = False,
    ) -> None:
        if backends:
            self._backends: list[AuditBackend] = list(backends)
        else:
            self._backends = [
                LocalFileBackend(
                    path,
                    encryption_key=encryption_key,
                    max_size_mb=max_size_mb,
                    rotate_daily=rotate_daily,
                )
            ]

        self._lock = threading.Lock()
        self._last_hash: str = ""
        self._sequence: int = 0

        # Try to recover chain state from the first backend
        self._recover_chain_state()

    # -- public API --------------------------------------------------------

    def log(self, event: AgentEvent) -> None:
        """Write an event as a hash-chained entry to all backends."""
        data_json = event.model_dump_json()
        self._write_entry(data_json)

    def log_dict(self, data: dict[str, Any]) -> None:
        """Write a raw dict as a hash-chained entry (for custom events)."""
        if "timestamp" not in data:
            data["timestamp"] = datetime.now(timezone.utc).isoformat()
        data_json = json.dumps(data, default=str)
        self._write_entry(data_json)

    def verify(self) -> dict[str, Any]:
        """Verify hash chain integrity across all backends.

        Returns a dict per backend with verification results.
        """
        results = {}
        for i, backend in enumerate(self._backends):
            name = getattr(backend, "name", None) or type(backend).__name__
            key = f"{name}_{i}" if name in results else name
            results[key] = backend.verify_chain()
        return results

    def close(self) -> None:
        """Flush and close all backends."""
        with self._lock:
            for backend in self._backends:
                try:
                    backend.close()
                except Exception:
                    pass

    def flush(self) -> None:
        """Force flush all backends."""
        with self._lock:
            for backend in self._backends:
                try:
                    backend.flush()
                except Exception:
                    pass

    @property
    def backends(self) -> list[AuditBackend]:
        """List of active backends."""
        return list(self._backends)

    @property
    def path(self) -> Path:
        """Path of the first local file backend (backward compat)."""
        for backend in self._backends:
            if isinstance(backend, LocalFileBackend):
                return backend.path
        return Path("agentguard_audit.jsonl")

    # -- internals ---------------------------------------------------------

    def _write_entry(self, data_json: str) -> None:
        """Create a hash-chained entry and write to all backends."""
        with self._lock:
            entry = AuditEntry(
                sequence=self._sequence,
                data=data_json,
                previous_hash=self._last_hash,
            )
            entry.fill_hash()

            for backend in self._backends:
                backend.write(entry)

            self._last_hash = entry.hash
            self._sequence += 1

    def _recover_chain_state(self) -> None:
        """Try to recover chain state from backends."""
        # Check if the first backend is a LocalFileBackend
        for backend in self._backends:
            if isinstance(backend, LocalFileBackend):
                # The LocalFileBackend already recovers its own state
                self._last_hash = backend._last_hash
                self._sequence = backend._sequence
                break

    def __enter__(self) -> AuditLogger:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
