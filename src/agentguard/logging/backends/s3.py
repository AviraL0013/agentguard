"""AWS S3 audit backend — remote cloud storage for audit logs.

Requires ``boto3``.  Install via: ``pip install agentaudit_sdk[s3]``

Usage::

    backend = S3Backend(
        bucket="my-audit-bucket",
        prefix="agentguard/prod",
        region_name="us-east-1",
    )
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Optional

from agentguard.logging.backends.base import AuditEntry, RetentionPolicy


class S3Backend:
    """AWS S3 audit backend with buffered writes and hash chaining.

    Buffers entries in memory and flushes to S3 at a configurable
    interval or buffer size.  Objects are partitioned by date::

        s3://bucket/prefix/YYYY/MM/DD/audit_HHmmss_seq.jsonl

    Args:
        bucket: S3 bucket name.
        prefix: Object key prefix (e.g. ``"agentguard/prod"``).
        region_name: AWS region (optional, uses default if not set).
        aws_access_key_id: Explicit access key (optional).
        aws_secret_access_key: Explicit secret key (optional).
        buffer_size: Flush after this many entries (default 100).
        server_side_encryption: S3 SSE method (e.g. ``"AES256"``, ``"aws:kms"``).
        kms_key_id: KMS key ID if using ``aws:kms`` encryption.
        retention: Retention policy (applied on flush).
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "agentguard",
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        buffer_size: int = 100,
        server_side_encryption: str | None = None,
        kms_key_id: str | None = None,
        retention: RetentionPolicy | None = None,
    ) -> None:
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "S3Backend requires the 'boto3' package. "
                "Install via: pip install agentaudit_sdk[s3]"
            )

        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._buffer_size = buffer_size
        self._sse = server_side_encryption
        self._kms_key = kms_key_id
        self._retention = retention

        # Create S3 client
        kwargs: dict[str, Any] = {}
        if region_name:
            kwargs["region_name"] = region_name
        if aws_access_key_id:
            kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            kwargs["aws_secret_access_key"] = aws_secret_access_key
        self._s3 = boto3.client("s3", **kwargs)

        # Buffer
        self._lock = threading.Lock()
        self._buffer: list[AuditEntry] = []
        self._upload_count: int = 0

    # -- public API --------------------------------------------------------

    def write(self, entry: AuditEntry) -> None:
        """Buffer an entry.  Flushes automatically at buffer_size."""
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
        """Read entries from S3.

        Lists objects under the prefix and reads them.
        For production use, consider a database backend for querying.
        """
        entries: list[AuditEntry] = []
        paginator = self._s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".jsonl"):
                    continue
                try:
                    response = self._s3.get_object(Bucket=self._bucket, Key=key)
                    body = response["Body"].read().decode("utf-8")
                    for line in body.strip().split("\n"):
                        if not line.strip():
                            continue
                        try:
                            entry = AuditEntry.from_json(line)
                            data = json.loads(entry.data)

                            if run_id and data.get("run_id") != run_id:
                                continue
                            if event_type and data.get("event_type") != event_type:
                                continue
                            if start_time and entry.timestamp < start_time:
                                continue
                            if end_time and entry.timestamp > end_time:
                                continue

                            entries.append(entry)
                        except (json.JSONDecodeError, KeyError):
                            continue
                except Exception:
                    continue

                if len(entries) >= offset + limit:
                    break
            if len(entries) >= offset + limit:
                break

        return entries[offset: offset + limit]

    def verify_chain(self) -> dict[str, Any]:
        """Verify chain integrity across all uploaded objects."""
        result = {
            "valid": True,
            "total_entries": 0,
            "first_invalid": None,
            "errors": [],
        }

        all_entries: list[AuditEntry] = []
        paginator = self._s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".jsonl"):
                    continue
                try:
                    response = self._s3.get_object(Bucket=self._bucket, Key=key)
                    body = response["Body"].read().decode("utf-8")
                    for line in body.strip().split("\n"):
                        if line.strip():
                            try:
                                all_entries.append(AuditEntry.from_json(line))
                            except (json.JSONDecodeError, KeyError):
                                pass
                except Exception:
                    pass

        # Sort by sequence
        all_entries.sort(key=lambda e: e.sequence)

        prev_hash = ""
        for entry in all_entries:
            result["total_entries"] += 1
            if entry.previous_hash != prev_hash:
                result["valid"] = False
                result["errors"].append(
                    f"Seq {entry.sequence}: previous_hash mismatch"
                )
                if result["first_invalid"] is None:
                    result["first_invalid"] = entry.sequence
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
        """Flush remaining buffer and close."""
        self.flush()

    def flush(self) -> None:
        """Force flush the buffer to S3."""
        with self._lock:
            self._flush_locked()

    # -- internals ---------------------------------------------------------

    def _flush_locked(self) -> None:
        """Flush buffer to S3. Must be called with lock held."""
        if not self._buffer:
            return

        now = datetime.now(timezone.utc)
        key = (
            f"{self._prefix}/"
            f"{now.strftime('%Y/%m/%d')}/"
            f"audit_{now.strftime('%H%M%S')}_{self._upload_count}.jsonl"
        )

        content = "\n".join(entry.to_json() for entry in self._buffer) + "\n"

        put_kwargs: dict[str, Any] = {
            "Bucket": self._bucket,
            "Key": key,
            "Body": content.encode("utf-8"),
            "ContentType": "application/x-jsonlines",
        }

        if self._sse:
            put_kwargs["ServerSideEncryption"] = self._sse
        if self._kms_key:
            put_kwargs["SSEKMSKeyId"] = self._kms_key

        self._s3.put_object(**put_kwargs)

        self._buffer.clear()
        self._upload_count += 1

    def __enter__(self) -> S3Backend:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
