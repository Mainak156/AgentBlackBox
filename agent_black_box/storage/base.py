"""
Storage interfaces for Agent Black Box.

This module defines the contract that every trace storage backend
must implement.

The tracer depends only on this interface and is therefore
independent of the underlying persistence technology.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent_black_box.events import TraceEvent


class StorageBackend(ABC):
    """
    Abstract interface for Agent Black Box trace storage.

    Implementations may store events in memory, SQLite, JSONL,
    PostgreSQL, or another persistence system.

    The tracer should never depend directly on a concrete storage
    implementation.
    """

    @abstractmethod
    def save(self, event: TraceEvent) -> None:
        """
        Persist a new trace event.

        Args:
            event:
                TraceEvent to persist.
        """

        raise NotImplementedError

    @abstractmethod
    def update(self, event: TraceEvent) -> None:
        """
        Update an existing trace event.

        Args:
            event:
                TraceEvent containing the updated state.
        """

        raise NotImplementedError

    @abstractmethod
    def get(self, event_id: str) -> TraceEvent | None:
        """
        Retrieve an event by its unique event ID.

        Args:
            event_id:
                Unique event identifier.

        Returns:
            TraceEvent | None:
                The matching event, or None if it does not exist.
        """

        raise NotImplementedError

    @abstractmethod
    def get_by_run(self, run_id: str) -> list[TraceEvent]:
        """
        Retrieve all events belonging to a specific run.

        Args:
            run_id:
                Agent execution identifier.

        Returns:
            list[TraceEvent]:
                Events belonging to the requested run.
        """

        raise NotImplementedError

    @abstractmethod
    def get_all(self) -> list[TraceEvent]:
        """
        Retrieve all stored trace events.

        Returns:
            list[TraceEvent]:
                All currently stored events.
        """

        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """
        Remove all stored events.
        """

        raise NotImplementedError

    def close(self) -> None:
        """
        Release storage resources.

        Backends that do not require cleanup may use this default
        implementation.

        SQLite and other resource-backed implementations can
        override this method.
        """

        return None