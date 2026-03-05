"""Thread-safe, append-only JSON-lines audit logger.

Each event is written as a single self-contained JSON line.
Supports file rotation by day or size.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TextIO

from agentguard.core.events import AgentEvent


class AuditLogger:
    """Append-only JSON-lines audit logger.

    Usage::

        logger = AuditLogger("./audit.jsonl")
        logger.log(event)
        logger.close()
    """

    def __init__(
        self,
        path: str | Path = "agentguard_audit.jsonl",
        *,
        max_size_mb: float = 100,
        rotate_daily: bool = False,
    ) -> None:
        self._base_path = Path(path)
        self._max_size = int(max_size_mb * 1024 * 1024)
        self._rotate_daily = rotate_daily
        self._lock = threading.Lock()
        self._file: Optional[TextIO] = None
        self._current_date: Optional[str] = None
        self._bytes_written: int = 0

    # -- public API --------------------------------------------------------

    def log(self, event: AgentEvent) -> None:
        """Write an event as a JSON line."""
        line = event.model_dump_json() + "\n"
        with self._lock:
            f = self._get_file()
            f.write(line)
            f.flush()
            self._bytes_written += len(line.encode("utf-8"))
            # Check rotation
            if self._max_size and self._bytes_written >= self._max_size:
                self._rotate("size")

    def log_dict(self, data: dict[str, Any]) -> None:
        """Write a raw dict as a JSON line (for custom events)."""
        if "timestamp" not in data:
            data["timestamp"] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(data, default=str) + "\n"
        with self._lock:
            f = self._get_file()
            f.write(line)
            f.flush()
            self._bytes_written += len(line.encode("utf-8"))

    def close(self) -> None:
        """Flush and close the current file."""
        with self._lock:
            if self._file and not self._file.closed:
                self._file.flush()
                self._file.close()
                self._file = None

    @property
    def path(self) -> Path:
        """Current log file path."""
        if self._rotate_daily:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            stem = self._base_path.stem
            suffix = self._base_path.suffix or ".jsonl"
            return self._base_path.parent / f"{stem}_{today}{suffix}"
        return self._base_path

    # -- internals ---------------------------------------------------------

    def _get_file(self) -> TextIO:
        """Get (or open) the current log file handle."""
        target = self.path

        # Check if we need to switch files (daily rotation)
        if self._rotate_daily:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self._current_date and self._current_date != today:
                self._rotate("daily")

        if self._file is None or self._file.closed:
            target.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(target, "a", encoding="utf-8")
            self._current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            # Approximate existing file size
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

    def __enter__(self) -> AuditLogger:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
