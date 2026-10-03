"""
Core tracing engine for Agent Black Box.

The Tracer is responsible for creating, updating, and retrieving
structured TraceEvent objects during an agent execution.

Storage and security are intentionally kept separate from the
core tracing API.
"""

from __future__ import annotations

from typing import Any

from agent_black_box.config import AgentBlackBoxConfig
from agent_black_box.context import (
    get_current_context,
    get_current_run_id,
)
from agent_black_box.events import (
    EventStatus,
    EventType,
    TraceEvent,
)
from agent_black_box.run import TraceRun
from agent_black_box.security.redaction import Redactor
from agent_black_box.storage.base import StorageBackend
from agent_black_box.storage.memory import InMemoryStorage


class Tracer:
    """
    Tracing engine with pluggable storage and redaction.

    By default:

    - Storage is in-memory.
    - Redaction is enabled.
    - Payloads are limited according to configuration.

    Persistence and security policies can be injected explicitly.
    """

    def __init__(
        self,
        *,
        storage: StorageBackend | None = None,
        config: AgentBlackBoxConfig | None = None,
        redactor: Redactor | None = None,
    ) -> None:
        """
        Initialize the tracer.
        """

        self.config = (
            config
            or AgentBlackBoxConfig()
        )

        self.storage = (
            storage
            or InMemoryStorage()
        )

        self.redactor = (
            redactor
            or Redactor(
                enabled=(
                    self.config.redact_sensitive_data
                ),
                max_payload_size=(
                    self.config.max_payload_size
                ),
            )
        )

    @property
    def events(self) -> list[TraceEvent]:
        """
        Return all recorded events.
        """

        return self.storage.get_all()

    # ------------------------------------------------------------------
    # Run Lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        *,
        name: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceRun:
        """
        Create a TraceRun context manager.

        Example:

            with tracer.start_run(
                name="support_agent",
            ) as run:
                ...

        Args:
            name:
                Human-readable run name.

            run_id:
                Optional externally supplied run ID.

            metadata:
                Optional run-level metadata.

        Returns:
            TraceRun:
                A context manager representing the execution.
        """

        return TraceRun(
            tracer=self,
            run_id=run_id,
            name=name,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Event Creation
    # ------------------------------------------------------------------

    def start_event(
        self,
        *,
        event_type: EventType,
        name: str | None = None,
        input_data: Any | None = None,
        metadata: dict[str, Any] | None = None,
        parent_event_id: str | None = None,
        **kwargs: Any,
    ) -> TraceEvent:
        """
        Create and record a new event with STARTED status.

        All user-controlled payloads pass through the redaction
        boundary before persistence.
        """

        run_id = get_current_run_id()

        if run_id is None:
            raise RuntimeError(
                "No active trace context. "
                "Use trace_context() or start_run() "
                "before creating events."
            )

        context = get_current_context()

        if (
            parent_event_id is None
            and context is not None
        ):
            parent_event_id = (
                context.current_event_id
            )

        safe_input = self.redactor.redact(
            input_data
        )

        safe_metadata = self.redactor.redact(
            metadata or {}
        )

        safe_kwargs = self.redactor.redact(
            kwargs
        )

        event = TraceEvent(
            run_id=run_id,
            event_type=event_type,
            status=EventStatus.STARTED,
            name=name,
            input_data=safe_input,
            metadata=safe_metadata,
            parent_event_id=parent_event_id,
            **safe_kwargs,
        )

        self.storage.save(event)

        if context is not None:
            context.push_event(
                event.event_id
            )

        return event

    # ------------------------------------------------------------------
    # Event Completion
    # ------------------------------------------------------------------

    def complete_event(
        self,
        event: TraceEvent,
        *,
        output_data: Any | None = None,
        duration_ms: float | None = None,
        **kwargs: Any,
    ) -> TraceEvent:
        """
        Mark an event as successfully completed.
        """

        event.status = EventStatus.SUCCESS

        event.output_data = (
            self.redactor.redact(
                output_data
            )
        )

        if duration_ms is not None:
            event.duration_ms = duration_ms

        safe_kwargs = self.redactor.redact(
            kwargs
        )

        for key, value in safe_kwargs.items():
            setattr(
                event,
                key,
                value,
            )

        self.storage.update(event)

        self._pop_event(event)

        return event

    # ------------------------------------------------------------------
    # Event Failure
    # ------------------------------------------------------------------

    def fail_event(
        self,
        event: TraceEvent,
        *,
        error: Exception | BaseException | None = None,
        duration_ms: float | None = None,
    ) -> TraceEvent:
        """
        Mark an event as failed.

        Exception details are sanitized before persistence.
        """

        event.status = EventStatus.FAILED

        if error is not None:
            event.error_type = type(
                error
            ).__name__

            event.error_message = (
                self.redactor.redact(
                    str(error)
                )
            )

        if duration_ms is not None:
            event.duration_ms = duration_ms

        self.storage.update(event)

        self._pop_event(event)

        return event

    # ------------------------------------------------------------------
    # Event Stack
    # ------------------------------------------------------------------

    def _pop_event(
        self,
        event: TraceEvent,
    ) -> None:
        """
        Remove an event from the active context stack.
        """

        context = get_current_context()

        if context is None:
            return

        if (
            context.current_event_id
            == event.event_id
        ):
            context.pop_event()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_events_for_run(
        self,
        run_id: str | None = None,
    ) -> list[TraceEvent]:
        """
        Return all events belonging to a run.
        """

        target_run_id = (
            run_id
            or get_current_run_id()
        )

        if target_run_id is None:
            return []

        return self.storage.get_by_run(
            target_run_id
        )

    # ------------------------------------------------------------------
    # Storage Management
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """
        Remove all stored events.
        """

        self.storage.clear()

    def close(self) -> None:
        """
        Close the configured storage backend.
        """

        self.storage.close()

    def __enter__(self) -> "Tracer":
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
        Close the tracer's storage backend.
        """

        self.close()