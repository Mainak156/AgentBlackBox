"""
In-memory storage backend for Agent Black Box.

This backend is primarily useful for:

- Unit tests
- Local development
- Short-lived agent executions
- Debugging

No data is persisted after the Python process exits.
"""

from __future__ import annotations

from agent_black_box.events import TraceEvent
from agent_black_box.storage.base import StorageBackend


class InMemoryStorage(StorageBackend):
    """
    Store trace events entirely in memory.

    Events are indexed by event_id while preserving insertion order.
    """

    def __init__(self) -> None:
        self._events: dict[str, TraceEvent] = {}

    def save(self, event: TraceEvent) -> None:
        """
        Store a new event.

        Raises:
            ValueError:
                If an event with the same event_id already exists.
        """

        if event.event_id in self._events:
            raise ValueError(
                f"Event '{event.event_id}' already exists."
            )

        self._events[event.event_id] = event

    def update(self, event: TraceEvent) -> None:
        """
        Update an existing event.

        Raises:
            KeyError:
                If the event does not already exist.
        """

        if event.event_id not in self._events:
            raise KeyError(
                f"Event '{event.event_id}' does not exist."
            )

        self._events[event.event_id] = event

    def get(self, event_id: str) -> TraceEvent | None:
        """
        Retrieve one event by ID.
        """

        return self._events.get(event_id)

    def get_by_run(self, run_id: str) -> list[TraceEvent]:
        """
        Retrieve all events belonging to a run.
        """

        return [
            event
            for event in self._events.values()
            if event.run_id == run_id
        ]

    def get_all(self) -> list[TraceEvent]:
        """
        Retrieve all stored events in insertion order.
        """

        return list(self._events.values())

    def clear(self) -> None:
        """
        Remove all stored events.
        """

        self._events.clear()

    def close(self) -> None:
        """
        Close the storage backend.

        In-memory storage has no external resources to release.
        """

        return None