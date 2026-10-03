"""
SQLite storage backend for Agent Black Box.

This module provides persistent local storage for TraceEvent objects.

Design goals:

- Implement the StorageBackend interface.
- Keep SQLite-specific logic isolated from the tracer.
- Automatically create the required database schema.
- Support event insertion, updates, retrieval, and deletion.
- Preserve structured event payloads as JSON.
- Use parameterized SQL queries.
- Maintain a small amount of schema-version metadata.
- Be safe for normal multi-threaded local application usage.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any

from pydantic import BaseModel

from agent_black_box.events import TraceEvent
from agent_black_box.storage.base import StorageBackend


DATABASE_SCHEMA_VERSION = 1


def _json_default(value: Any) -> Any:
    """
    Convert otherwise non-JSON-serializable Python values into
    JSON-compatible representations.

    Tracing should be observational and should not unnecessarily
    crash an agent simply because an input or output contains an
    object that the standard JSON encoder cannot serialize.
    """

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, set):
        return list(value)

    return repr(value)


def _serialize(value: Any) -> str | None:
    """
    Serialize a Python value into JSON.

    Returns None for None values.
    """

    if value is None:
        return None

    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _deserialize(value: str | None) -> Any:
    """
    Deserialize JSON stored in SQLite.

    Returns None for NULL database values.
    """

    if value is None:
        return None

    return json.loads(value)


class SQLiteStorage(StorageBackend):
    """
    Persistent SQLite implementation of StorageBackend.

    Example:

        storage = SQLiteStorage("data/traces.db")
        tracer = Tracer(storage=storage)

    The database and parent directories are created automatically.
    """

    def __init__(
        self,
        database_path: str | Path = "data/traces.db",
    ) -> None:
        """
        Initialize the SQLite storage backend.

        Args:
            database_path:
                Path to the SQLite database file.
        """

        self.database_path = Path(database_path)

        if self.database_path != Path(":memory:"):
            self.database_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._lock = RLock()

        self._connection = sqlite3.connect(
            str(self.database_path),
            check_same_thread=False,
        )

        self._connection.row_factory = sqlite3.Row

        self._configure_connection()
        self._initialize_schema()

    def _configure_connection(self) -> None:
        """
        Configure SQLite for reliable local application usage.
        """

        with self._connection:
            self._connection.execute(
                "PRAGMA foreign_keys = ON"
            )

            self._connection.execute(
                "PRAGMA busy_timeout = 5000"
            )

            self._connection.execute(
                "PRAGMA journal_mode = WAL"
            )

    def _initialize_schema(self) -> None:
        """
        Create the database schema if it does not already exist.
        """

        with self._lock:
            with self._connection:
                self._connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                    """
                )

                self._connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trace_events (
                        event_id TEXT PRIMARY KEY,
                        schema_version INTEGER NOT NULL,
                        run_id TEXT NOT NULL,
                        parent_event_id TEXT,
                        event_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        name TEXT,
                        timestamp TEXT NOT NULL,
                        duration_ms REAL,
                        input_data TEXT,
                        output_data TEXT,
                        provider TEXT,
                        model TEXT,
                        input_tokens INTEGER,
                        output_tokens INTEGER,
                        total_tokens INTEGER,
                        estimated_cost REAL,
                        error_type TEXT,
                        error_message TEXT,
                        metadata TEXT NOT NULL
                    )
                    """
                )

                self._connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_trace_events_run_id
                    ON trace_events(run_id)
                    """
                )

                self._connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_trace_events_timestamp
                    ON trace_events(timestamp)
                    """
                )

                self._connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_trace_events_parent_event_id
                    ON trace_events(parent_event_id)
                    """
                )

                self._connection.execute(
                    """
                    INSERT INTO schema_metadata(key, value)
                    VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (
                        "database_schema_version",
                        str(DATABASE_SCHEMA_VERSION),
                    ),
                )

    def save(self, event: TraceEvent) -> None:
        """
        Persist a new trace event.

        Raises:
            ValueError:
                If an event with the same event_id already exists.
        """

        with self._lock:
            try:
                with self._connection:
                    self._connection.execute(
                        """
                        INSERT INTO trace_events (
                            event_id,
                            schema_version,
                            run_id,
                            parent_event_id,
                            event_type,
                            status,
                            name,
                            timestamp,
                            duration_ms,
                            input_data,
                            output_data,
                            provider,
                            model,
                            input_tokens,
                            output_tokens,
                            total_tokens,
                            estimated_cost,
                            error_type,
                            error_message,
                            metadata
                        )
                        VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        """,
                        self._event_to_row(event),
                    )

            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"Event '{event.event_id}' already exists."
                ) from exc

    def update(self, event: TraceEvent) -> None:
        """
        Update an existing trace event.

        Raises:
            KeyError:
                If the event does not exist.
        """

        with self._lock:
            with self._connection:
                cursor = self._connection.execute(
                    """
                    UPDATE trace_events
                    SET
                        schema_version = ?,
                        run_id = ?,
                        parent_event_id = ?,
                        event_type = ?,
                        status = ?,
                        name = ?,
                        timestamp = ?,
                        duration_ms = ?,
                        input_data = ?,
                        output_data = ?,
                        provider = ?,
                        model = ?,
                        input_tokens = ?,
                        output_tokens = ?,
                        total_tokens = ?,
                        estimated_cost = ?,
                        error_type = ?,
                        error_message = ?,
                        metadata = ?
                    WHERE event_id = ?
                    """,
                    self._event_to_update_row(event),
                )

                if cursor.rowcount == 0:
                    raise KeyError(
                        f"Event '{event.event_id}' does not exist."
                    )

    def get(self, event_id: str) -> TraceEvent | None:
        """
        Retrieve one event by event ID.
        """

        with self._lock:
            cursor = self._connection.execute(
                """
                SELECT *
                FROM trace_events
                WHERE event_id = ?
                """,
                (event_id,),
            )

            row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_event(row)

    def get_by_run(self, run_id: str) -> list[TraceEvent]:
        """
        Retrieve all events belonging to a run.

        Events are returned in insertion order.
        """

        with self._lock:
            cursor = self._connection.execute(
                """
                SELECT *
                FROM trace_events
                WHERE run_id = ?
                ORDER BY rowid ASC
                """,
                (run_id,),
            )

            rows = cursor.fetchall()

        return [
            self._row_to_event(row)
            for row in rows
        ]

    def get_all(self) -> list[TraceEvent]:
        """
        Retrieve all stored events in insertion order.
        """

        with self._lock:
            cursor = self._connection.execute(
                """
                SELECT *
                FROM trace_events
                ORDER BY rowid ASC
                """
            )

            rows = cursor.fetchall()

        return [
            self._row_to_event(row)
            for row in rows
        ]

    def clear(self) -> None:
        """
        Delete all stored trace events.
        """

        with self._lock:
            with self._connection:
                self._connection.execute(
                    "DELETE FROM trace_events"
                )

    def close(self) -> None:
        """
        Close the SQLite connection.
        """

        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __enter__(self) -> "SQLiteStorage":
        """
        Support context-manager usage.
        """

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        """
        Close storage when leaving a context manager.
        """

        self.close()

    @staticmethod
    def _event_to_row(event: TraceEvent) -> tuple[Any, ...]:
        """
        Convert a TraceEvent into an INSERT-compatible row.
        """

        return (
            event.event_id,
            event.schema_version,
            event.run_id,
            event.parent_event_id,
            event.event_type,
            event.status,
            event.name,
            event.timestamp.isoformat(),
            event.duration_ms,
            _serialize(event.input_data),
            _serialize(event.output_data),
            event.provider,
            event.model,
            event.input_tokens,
            event.output_tokens,
            event.total_tokens,
            event.estimated_cost,
            event.error_type,
            event.error_message,
            _serialize(event.metadata) or "{}",
        )

    @staticmethod
    def _event_to_update_row(event: TraceEvent) -> tuple[Any, ...]:
        """
        Convert a TraceEvent into an UPDATE-compatible row.
        """

        return (
            event.schema_version,
            event.run_id,
            event.parent_event_id,
            event.event_type,
            event.status,
            event.name,
            event.timestamp.isoformat(),
            event.duration_ms,
            _serialize(event.input_data),
            _serialize(event.output_data),
            event.provider,
            event.model,
            event.input_tokens,
            event.output_tokens,
            event.total_tokens,
            event.estimated_cost,
            event.error_type,
            event.error_message,
            _serialize(event.metadata) or "{}",
            event.event_id,
        )

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> TraceEvent:
        """
        Convert a SQLite row back into a TraceEvent.
        """

        return TraceEvent(
            schema_version=row["schema_version"],
            event_id=row["event_id"],
            run_id=row["run_id"],
            parent_event_id=row["parent_event_id"],
            event_type=row["event_type"],
            status=row["status"],
            name=row["name"],
            timestamp=row["timestamp"],
            duration_ms=row["duration_ms"],
            input_data=_deserialize(row["input_data"]),
            output_data=_deserialize(row["output_data"]),
            provider=row["provider"],
            model=row["model"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            total_tokens=row["total_tokens"],
            estimated_cost=row["estimated_cost"],
            error_type=row["error_type"],
            error_message=row["error_message"],
            metadata=_deserialize(row["metadata"]) or {},
        )