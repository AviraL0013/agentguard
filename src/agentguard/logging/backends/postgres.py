"""PostgreSQL audit backend — structured database storage via SQLAlchemy.

Requires ``sqlalchemy`` and ``psycopg2-binary``.
Install via: ``pip install agentaudit_sdk[postgres]``

Usage::

    backend = PostgresBackend(
        connection_url="postgresql://user:pass@localhost:5432/agentguard",
    )
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from agentguard.logging.backends.base import AuditEntry, RetentionPolicy


class PostgresBackend:
    """SQLAlchemy-based PostgreSQL audit backend.

    Stores audit entries in a structured ``audit_entries`` table
    with hash chaining maintained in the database.

    Args:
        connection_url: SQLAlchemy connection string.
        table_name: Name of the audit table (default "audit_entries").
        schema: Database schema (default None = public).
        retention: Retention policy.
        echo: SQLAlchemy echo mode (debug SQL queries).
    """

    def __init__(
        self,
        connection_url: str,
        *,
        table_name: str = "audit_entries",
        schema: str | None = None,
        retention: RetentionPolicy | None = None,
        echo: bool = False,
    ) -> None:
        try:
            from sqlalchemy import (
                Boolean,
                Column,
                DateTime,
                Integer,
                MetaData,
                String,
                Table,
                Text,
                create_engine,
            )
            from sqlalchemy.orm import Session
        except ImportError:
            raise ImportError(
                "PostgresBackend requires 'sqlalchemy' and 'psycopg2-binary'. "
                "Install via: pip install agentaudit_sdk[postgres]"
            )

        self._engine = create_engine(connection_url, echo=echo)
        self._Session = Session
        self._retention = retention

        # Define table
        metadata = MetaData(schema=schema)
        self._table = Table(
            table_name,
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("sequence", Integer, nullable=False, index=True, unique=True),
            Column("timestamp", String(64), nullable=False, index=True),
            Column("event_type", String(64), nullable=True, index=True),
            Column("run_id", String(64), nullable=True, index=True),
            Column("agent_id", String(128), nullable=True, index=True),
            Column("data_json", Text, nullable=False),
            Column("hash", String(64), nullable=False),
            Column("previous_hash", String(64), nullable=False),
            Column("encrypted", Boolean, default=False),
        )

        # Create table if not exists
        metadata.create_all(self._engine)

    # -- public API --------------------------------------------------------

    def write(self, entry: AuditEntry) -> None:
        """Insert an audit entry into the database."""
        # Parse event data for indexed columns
        try:
            data = json.loads(entry.data)
        except json.JSONDecodeError:
            data = {}

        with self._Session(self._engine) as session:
            session.execute(
                self._table.insert().values(
                    sequence=entry.sequence,
                    timestamp=entry.timestamp,
                    event_type=data.get("event_type", ""),
                    run_id=data.get("run_id", ""),
                    agent_id=data.get("agent_id", ""),
                    data_json=entry.data,
                    hash=entry.hash,
                    previous_hash=entry.previous_hash,
                    encrypted=entry.encrypted,
                )
            )
            session.commit()

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
        """Query entries with filtering via SQL."""
        from sqlalchemy import select

        query = select(self._table).order_by(self._table.c.sequence)

        if run_id:
            query = query.where(self._table.c.run_id == run_id)
        if event_type:
            query = query.where(self._table.c.event_type == event_type)
        if start_time:
            query = query.where(self._table.c.timestamp >= start_time)
        if end_time:
            query = query.where(self._table.c.timestamp <= end_time)

        query = query.offset(offset).limit(limit)

        entries: list[AuditEntry] = []
        with self._Session(self._engine) as session:
            for row in session.execute(query):
                entries.append(
                    AuditEntry(
                        sequence=row.sequence,
                        timestamp=row.timestamp,
                        data=row.data_json,
                        hash=row.hash,
                        previous_hash=row.previous_hash,
                        encrypted=row.encrypted,
                    )
                )

        return entries

    def verify_chain(self) -> dict[str, Any]:
        """Verify the hash chain integrity in the database."""
        from sqlalchemy import select, func

        result = {
            "valid": True,
            "total_entries": 0,
            "first_invalid": None,
            "errors": [],
        }

        with self._Session(self._engine) as session:
            # Get total count
            count_result = session.execute(
                select(func.count()).select_from(self._table)
            ).scalar()
            result["total_entries"] = count_result or 0

            if result["total_entries"] == 0:
                return result

            # Read all entries ordered by sequence
            query = select(self._table).order_by(self._table.c.sequence)
            prev_hash = ""

            for row in session.execute(query):
                entry = AuditEntry(
                    sequence=row.sequence,
                    timestamp=row.timestamp,
                    data=row.data_json,
                    hash=row.hash,
                    previous_hash=row.previous_hash,
                    encrypted=row.encrypted,
                )

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
        """Dispose of the SQLAlchemy engine."""
        self._engine.dispose()

    def flush(self) -> None:
        """No-op — PostgresBackend writes are immediate."""
        pass

    def __enter__(self) -> PostgresBackend:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
