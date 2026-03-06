"""SIEM audit backend — Splunk HEC / Datadog / Generic Webhook integration.

Requires ``httpx`` for async HTTP delivery.
Install via: ``pip install agentaudit_sdk[siem]``

Usage::

    # Splunk HEC
    backend = SIEMBackend(
        provider="splunk",
        endpoint="https://splunk-hec.example.com:8088/services/collector/event",
        token="your-hec-token",
    )

    # Datadog
    backend = SIEMBackend(
        provider="datadog",
        endpoint="https://http-intake.logs.datadoghq.com/api/v2/logs",
        token="your-dd-api-key",
    )

    # Generic webhook
    backend = SIEMBackend(
        provider="webhook",
        endpoint="https://your-siem.example.com/api/logs",
        token="your-api-key",
    )
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from agentguard.logging.backends.base import AuditEntry

logger = logging.getLogger(__name__)


class SIEMBackend:
    """SIEM integration backend supporting Splunk HEC, Datadog, and generic webhooks.

    Features:
    - Provider-specific formatting (Splunk HEC, Datadog Logs API)
    - Buffered batch delivery
    - Retry with exponential backoff
    - Fallback to local file on delivery failure

    Args:
        provider: One of ``"splunk"``, ``"datadog"``, ``"webhook"``.
        endpoint: HTTP endpoint URL.
        token: Authentication token.
        buffer_size: Entries to buffer before flushing (default 50).
        max_retries: Maximum delivery retries (default 3).
        retry_delay: Base delay between retries in seconds (default 1.0).
        source: Event source identifier (Splunk HEC).
        source_type: Source type (Splunk HEC, default ``"_json"``).
        index: Splunk index name.
        service: Service name (Datadog).
        tags: Tags (Datadog).
        custom_headers: Additional HTTP headers.
        fallback_path: Local file path on delivery failure.
    """

    PROVIDERS = {"splunk", "datadog", "webhook"}

    def __init__(
        self,
        provider: str,
        endpoint: str,
        token: str,
        *,
        buffer_size: int = 50,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        source: str = "agentguard",
        source_type: str = "_json",
        index: str | None = None,
        service: str = "agentguard",
        tags: list[str] | None = None,
        custom_headers: dict[str, str] | None = None,
        fallback_path: str | None = None,
    ) -> None:
        if provider not in self.PROVIDERS:
            raise ValueError(
                f"Unknown SIEM provider: {provider!r}. "
                f"Supported: {sorted(self.PROVIDERS)}"
            )

        try:
            import httpx
            self._httpx = httpx
        except ImportError:
            raise ImportError(
                "SIEMBackend requires the 'httpx' package. "
                "Install via: pip install agentaudit_sdk[siem]"
            )

        self._provider = provider
        self._endpoint = endpoint
        self._token = token
        self._buffer_size = buffer_size
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        # Splunk-specific
        self._source = source
        self._source_type = source_type
        self._index = index

        # Datadog-specific
        self._service = service
        self._tags = tags or []

        # HTTP config
        self._headers = self._build_headers(custom_headers)
        self._fallback_path = fallback_path

        # Buffer
        self._lock = threading.Lock()
        self._buffer: list[AuditEntry] = []
        self._failed_count: int = 0

    # -- public API --------------------------------------------------------

    def write(self, entry: AuditEntry) -> None:
        """Buffer an entry.  Flushes at buffer_size."""
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self._buffer_size:
                self._flush_locked()

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
        """SIEM is write-only — read from your SIEM platform directly."""
        return []

    def verify_chain(self) -> dict[str, Any]:
        """Chain verification not supported for SIEM (write-only)."""
        return {
            "valid": True,
            "total_entries": 0,
            "first_invalid": None,
            "errors": ["Chain verification not available for SIEM backend"],
        }

    def close(self) -> None:
        """Flush remaining buffer and close."""
        self.flush()

    def flush(self) -> None:
        """Force flush the buffer."""
        with self._lock:
            self._flush_locked()

    # -- formatting --------------------------------------------------------

    def _format_splunk(self, entry: AuditEntry) -> dict[str, Any]:
        """Format as Splunk HEC event."""
        try:
            event_data = json.loads(entry.data)
        except json.JSONDecodeError:
            event_data = {"raw": entry.data}

        event = {
            "time": self._iso_to_epoch(entry.timestamp),
            "source": self._source,
            "sourcetype": self._source_type,
            "event": {
                **event_data,
                "_agentguard_seq": entry.sequence,
                "_agentguard_hash": entry.hash,
                "_agentguard_prev_hash": entry.previous_hash,
            },
        }
        if self._index:
            event["index"] = self._index
        return event

    def _format_datadog(self, entry: AuditEntry) -> dict[str, Any]:
        """Format as Datadog Logs API entry."""
        try:
            event_data = json.loads(entry.data)
        except json.JSONDecodeError:
            event_data = {"raw": entry.data}

        return {
            "ddsource": self._source,
            "ddtags": ",".join(self._tags) if self._tags else "",
            "hostname": "agentguard",
            "service": self._service,
            "message": json.dumps(event_data),
            "agentguard": {
                "sequence": entry.sequence,
                "hash": entry.hash,
                "previous_hash": entry.previous_hash,
                "timestamp": entry.timestamp,
            },
        }

    def _format_webhook(self, entry: AuditEntry) -> dict[str, Any]:
        """Format as generic webhook payload."""
        try:
            event_data = json.loads(entry.data)
        except json.JSONDecodeError:
            event_data = {"raw": entry.data}

        return {
            "source": "agentguard",
            "timestamp": entry.timestamp,
            "sequence": entry.sequence,
            "hash": entry.hash,
            "previous_hash": entry.previous_hash,
            "encrypted": entry.encrypted,
            "event": event_data,
        }

    # -- internals ---------------------------------------------------------

    def _build_headers(self, custom: dict[str, str] | None = None) -> dict[str, str]:
        """Build HTTP headers based on provider."""
        headers: dict[str, str] = {"Content-Type": "application/json"}

        if self._provider == "splunk":
            headers["Authorization"] = f"Splunk {self._token}"
        elif self._provider == "datadog":
            headers["DD-API-KEY"] = self._token
        elif self._provider == "webhook":
            headers["Authorization"] = f"Bearer {self._token}"

        if custom:
            headers.update(custom)
        return headers

    def _flush_locked(self) -> None:
        """Flush buffer with retry logic. Must be called with lock held."""
        if not self._buffer:
            return

        # Format entries based on provider
        formatter = {
            "splunk": self._format_splunk,
            "datadog": self._format_datadog,
            "webhook": self._format_webhook,
        }[self._provider]

        payloads = [formatter(entry) for entry in self._buffer]

        # Attempt delivery with retries
        success = False
        for attempt in range(self._max_retries):
            try:
                if self._provider == "splunk":
                    # Splunk HEC accepts newline-delimited JSON
                    body = "\n".join(json.dumps(p) for p in payloads)
                else:
                    body = json.dumps(payloads)

                response = self._httpx.post(
                    self._endpoint,
                    content=body,
                    headers=self._headers,
                    timeout=30.0,
                )

                if response.status_code in (200, 201, 202):
                    success = True
                    break
                else:
                    logger.warning(
                        "SIEM delivery attempt %d/%d failed: HTTP %d",
                        attempt + 1,
                        self._max_retries,
                        response.status_code,
                    )
            except Exception as e:
                logger.warning(
                    "SIEM delivery attempt %d/%d error: %s",
                    attempt + 1,
                    self._max_retries,
                    e,
                )

            # Exponential backoff
            if attempt < self._max_retries - 1:
                time.sleep(self._retry_delay * (2 ** attempt))

        if not success:
            self._failed_count += len(self._buffer)
            logger.error(
                "SIEM delivery failed after %d retries. %d entries lost.",
                self._max_retries,
                len(self._buffer),
            )
            # Fallback to local file
            if self._fallback_path:
                self._write_fallback(payloads)

        self._buffer.clear()

    def _write_fallback(self, payloads: list[dict[str, Any]]) -> None:
        """Write to local fallback file when SIEM delivery fails."""
        try:
            from pathlib import Path
            path = Path(self._fallback_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                for p in payloads:
                    f.write(json.dumps(p, default=str) + "\n")
            logger.info("Wrote %d entries to fallback file: %s", len(payloads), path)
        except Exception as e:
            logger.error("Failed to write fallback file: %s", e)

    @staticmethod
    def _iso_to_epoch(iso_str: str) -> float:
        """Convert ISO timestamp to epoch seconds for Splunk."""
        try:
            dt = datetime.fromisoformat(iso_str)
            return dt.timestamp()
        except (ValueError, TypeError):
            return time.time()

    def __enter__(self) -> SIEMBackend:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
