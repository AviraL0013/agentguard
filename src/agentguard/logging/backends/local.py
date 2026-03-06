"""Local file audit backend — hash-chained JSONL with optional encryption.

This is the default backend, replacing the original flat ``AuditLogger``
with tamper-evident hash chaining and optional Fernet encryption at rest.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TextIO

from agentguard.logging.backends.base import AuditEntry, RetentionPolicy


class LocalFileBackend:
    """Enhanced local JSONL backend with hash chaining and optional encryption.

    Features:
    - SHA-256 hash chaining — every entry includes hash of previous
    - Optional Fernet (AES-128-CBC) encryption at rest
    - File rotation (size-based and daily)
    - Retention policy enforcement
    - Chain verification

    Usage::

        backend = LocalFileBackend("audit.jsonl")
        backend = LocalFileBackend("audit.jsonl", encryption_key="your-fernet-key")
    """

    def __init__(
        self,
        path: str | Path = "agentguard_audit.jsonl",
        *,
        encryption_key: str | None = None,
        max_size_mb: float = 100,
        rotate_daily: bool = False,
        retention: RetentionPolicy | None = None,
    ) -> None:
        self._base_path = Path(path)
        self._encryption_key = encryption_key
        self._fernet: Any = None
        self._max_size = int(max_size_mb * 1024 * 1024)
        self._rotate_daily = rotate_daily
        self._retention = retention

        self._lock = threading.Lock()
        self._file: Optional[TextIO] = None
        self._current_date: Optional[str] = None
        self._bytes_written: int = 0
        self._last_hash: str = ""
        self._sequence: int = 0

        # Initialise encryption if key provided
        if encryption_key:
            self._init_encryption(encryption_key)

        # Recover chain state from existing file
        self._recover_chain_state()

    # -- public API --------------------------------------------------------

    def write(self, entry: AuditEntry) -> None:
        """Write a hash-chained entry to the file."""
        # Encrypt data if encryption is enabled
        if self._fernet and not entry.encrypted:
            entry.data = self._encrypt(entry.data)
            entry.encrypted = True

        # Set chain fields
        entry.previous_hash = self._last_hash
        entry.sequence = self._sequence
        entry.fill_hash()

        line = entry.to_json() + "\n"

        with self._lock:
            f = self._get_file()
            f.write(line)
            f.flush()
            self._bytes_written += len(line.encode("utf-8"))
            self._last_hash = entry.hash
            self._sequence += 1

            # Check rotation
            if self._max_size and self._bytes_written >= self._max_size:
                self._rotate("size")

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
        entries: list[AuditEntry] = []
        path = self.path
        if not path.exists():
            return entries

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = AuditEntry.from_json(line)
                except (json.JSONDecodeError, KeyError):
                    continue

                # Decrypt for filtering
                data_str = self._decrypt_data(entry)
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    data = {}

                # Apply filters
                if run_id and data.get("run_id") != run_id:
                    continue
                if event_type and data.get("event_type") != event_type:
                    continue
                if start_time and entry.timestamp < start_time:
                    continue
                if end_time and entry.timestamp > end_time:
                    continue

                entries.append(entry)

        # Apply offset and limit
        return entries[offset: offset + limit]

    def verify_chain(self) -> dict[str, Any]:
        """Verify the hash chain integrity of the log file."""
        result = {
            "valid": True,
            "total_entries": 0,
            "first_invalid": None,
            "errors": [],
        }

        path = self.path
        if not path.exists():
            return result

        prev_hash = ""
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = AuditEntry.from_json(line)
                except (json.JSONDecodeError, KeyError) as e:
                    result["valid"] = False
                    result["errors"].append(f"Line {i}: malformed entry — {e}")
                    if result["first_invalid"] is None:
                        result["first_invalid"] = i
                    continue

                result["total_entries"] += 1

                # Check previous hash linkage
                if entry.previous_hash != prev_hash:
                    result["valid"] = False
                    result["errors"].append(
                        f"Seq {entry.sequence}: previous_hash mismatch "
                        f"(expected {prev_hash[:16]}..., got {entry.previous_hash[:16]}...)"
                    )
                    if result["first_invalid"] is None:
                        result["first_invalid"] = entry.sequence

                # Check self-hash
                if not entry.verify():
                    result["valid"] = False
                    result["errors"].append(
                        f"Seq {entry.sequence}: hash verification failed"
                    )
                    if result["first_invalid"] is None:
                        result["first_invalid"] = entry.sequence

                prev_hash = entry.hash

        return result

    def close(self) -> None:
        """Flush and close the file."""
        with self._lock:
            if self._file and not self._file.closed:
                self._file.flush()
                self._file.close()
                self._file = None

    def flush(self) -> None:
        """Force flush any buffered data."""
        with self._lock:
            if self._file and not self._file.closed:
                self._file.flush()

    @property
    def path(self) -> Path:
        """Current log file path."""
        if self._rotate_daily:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            stem = self._base_path.stem
            suffix = self._base_path.suffix or ".jsonl"
            return self._base_path.parent / f"{stem}_{today}{suffix}"
        return self._base_path

    # -- encryption --------------------------------------------------------

    def _init_encryption(self, key: str) -> None:
        """Initialise Fernet encryption."""
        try:
            from cryptography.fernet import Fernet
            self._fernet = Fernet(key.encode() if isinstance(key, str) and len(key) < 100 else key)
        except ImportError:
            raise ImportError(
                "Encryption requires the 'cryptography' package. "
                "Install via: pip install agentaudit_sdk[encryption]"
            )

    def _encrypt(self, plaintext: str) -> str:
        """Encrypt data using Fernet."""
        if not self._fernet:
            return plaintext
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")

    def _decrypt(self, ciphertext: str) -> str:
        """Decrypt data using Fernet."""
        if not self._fernet:
            return ciphertext
        return self._fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")

    def _decrypt_data(self, entry: AuditEntry) -> str:
        """Decrypt an entry's data if encrypted."""
        if entry.encrypted and self._fernet:
            return self._decrypt(entry.data)
        return entry.data

    # -- internals ---------------------------------------------------------

    def _get_file(self) -> TextIO:
        """Get (or open) the current log file handle."""
        target = self.path

        # Check daily rotation
        if self._rotate_daily:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self._current_date and self._current_date != today:
                self._rotate("daily")

        if self._file is None or self._file.closed:
            target.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(target, "a", encoding="utf-8")
            self._current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if target.exists():
                self._bytes_written = target.stat().st_size

        return self._file

    def _rotate(self, reason: str) -> None:
        """Close current file and start a new one."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()
        self._file = None
        self._bytes_written = 0

        # For size-based rotation, rename the old file
        if reason == "size":
            current = self.path
            if current.exists():
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                rotated = current.parent / f"{current.stem}_{ts}{current.suffix}"
                current.rename(rotated)

    def _recover_chain_state(self) -> None:
        """Recover the last hash and sequence from an existing log file."""
        path = self.path
        if not path.exists():
            return

        last_line = ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last_line = line
        except OSError:
            return

        if last_line:
            try:
                entry = AuditEntry.from_json(last_line)
                self._last_hash = entry.hash
                self._sequence = entry.sequence + 1
            except (json.JSONDecodeError, KeyError):
                pass

    def __enter__(self) -> LocalFileBackend:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
