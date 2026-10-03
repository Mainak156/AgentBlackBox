"""
JSONL storage backend for Agent Black Box.

This backend stores TraceEvent objects as newline-delimited JSON.

Design goals:

- Implement StorageBackend.
- Keep the format portable and human-readable.
- Use append-only writes.
- Support event updates by appending the latest event state.
- Rebuild the latest event index when the process starts.
- Preserve event ordering.
- Ensure in-memory state matches persisted JSON representation.
- Avoid making the tracer aware of JSONL.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any

from pydantic import BaseModel

from agent_black_box.events import TraceEvent
from agent_black_box.storage.base import StorageBackend


def _json_default(value: Any) -> Any:
    """
    Convert common non-JSON-serializable values into safe
    JSON-compatible representations.
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


def _serialize_event(event: TraceEvent) -> str:
    """
    Convert a TraceEvent into one JSONL record.
    """

    data = event.model_dump(mode="python")

    return json.dumps(
        data,
        default=_json_default,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _deserialize_event(line: str) -> TraceEvent:
    """
    Convert one JSONL record into a TraceEvent.
    """

    data = json.loads(line)

    return TraceEvent.model_validate(data)


class JSONLStorage(StorageBackend):
    """
    Persistent newline-delimited JSON storage backend.

    Events are written one per line.

    Updates append a new representation of the same event.
    The latest representation is considered authoritative.
    """

    def __init__(
        self,
        file_path: str | Path = "data/traces.jsonl",
    ) -> None:
        """
        Initialize JSONL storage.

        Args:
            file_path:
                Path to the JSONL file.
        """

        self.file_path = Path(file_path)

        self.file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = RLock()

        # Latest persisted representation of every event.
        self._events: dict[str, TraceEvent] = {}

        # Preserves original event insertion order.
        self._event_order: list[str] = []

        self._load()

    def _load(self) -> None:
        """
        Load the latest state of every event from the JSONL file.

        If an event appears multiple times, the latest occurrence
        becomes the authoritative state.
        """

        if not self.file_path.exists():
            return

        with self._lock:
            with self.file_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                for line_number, line in enumerate(
                    file,
                    start=1,
                ):
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        event = _deserialize_event(line)

                    except (
                        json.JSONDecodeError,
                        ValueError,
                    ) as exc:
                        raise ValueError(
                            "Invalid JSONL trace record at "
                            f"{self.file_path}:{line_number}"
                        ) from exc

                    if event.event_id not in self._events:
                        self._event_order.append(
                            event.event_id
                        )

                    self._events[event.event_id] = event

    def _append(
        self,
        event: TraceEvent,
    ) -> TraceEvent:
        """
        Persist an event and return the representation reconstructed
        from the persisted JSON.

        This ensures the in-memory cache behaves exactly like the
        durable JSONL representation.
        """

        record = _serialize_event(event)

        with self.file_path.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as file:
            file.write(record)
            file.write("\n")
            file.flush()

        return _deserialize_event(record)

    def save(self, event: TraceEvent) -> None:
        """
        Append a new event.

        Raises:
            ValueError:
                If the event already exists.
        """

        with self._lock:
            if event.event_id in self._events:
                raise ValueError(
                    f"Event '{event.event_id}' already exists."
                )

            persisted_event = self._append(event)

            self._events[event.event_id] = persisted_event
            self._event_order.append(event.event_id)

    def update(self, event: TraceEvent) -> None:
        """
        Append the latest state of an existing event.

        Raises:
            KeyError:
                If the event does not already exist.
        """

        with self._lock:
            if event.event_id not in self._events:
                raise KeyError(
                    f"Event '{event.event_id}' does not exist."
                )

            persisted_event = self._append(event)

            self._events[event.event_id] = persisted_event

    def get(
        self,
        event_id: str,
    ) -> TraceEvent | None:
        """
        Retrieve the latest persisted state of an event.
        """

        with self._lock:
            return self._events.get(event_id)

    def get_by_run(
        self,
        run_id: str,
    ) -> list[TraceEvent]:
        """
        Retrieve all latest event states belonging to a run.
        """

        with self._lock:
            return [
                self._events[event_id]
                for event_id in self._event_order
                if self._events[event_id].run_id == run_id
            ]

    def get_all(self) -> list[TraceEvent]:
        """
        Retrieve all latest event states in insertion order.
        """

        with self._lock:
            return [
                self._events[event_id]
                for event_id in self._event_order
            ]

    def clear(self) -> None:
        """
        Remove all events and truncate the JSONL file.
        """

        with self._lock:
            self._events.clear()
            self._event_order.clear()

            with self.file_path.open(
                "w",
                encoding="utf-8",
            ):
                pass

    def close(self) -> None:
        """
        Close the storage backend.

        JSONL does not maintain a persistent open file handle,
        so there is no external resource to release.
        """

        return None

    def __enter__(self) -> "JSONLStorage":
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