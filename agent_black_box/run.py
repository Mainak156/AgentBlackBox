"""
Run lifecycle management for Agent Black Box.

A TraceRun represents one complete agent execution.

It creates:

    RUN_START
        |
        | nested agent operations
        |
    RUN_END

The implementation supports nested runs by temporarily creating a
new tracing context and restoring the parent context when the nested
run completes.
"""

from __future__ import annotations

import time
from contextlib import AbstractContextManager
from typing import Any
from uuid import uuid4

from agent_black_box.context import (
    TraceContext,
    get_current_context,
    trace_context,
)
from agent_black_box.events import (
    EventStatus,
    EventType,
    TraceEvent,
)


class TraceRun(AbstractContextManager["TraceRun"]):
    """
    Represents one complete traced agent execution.

    TraceRun is normally created through:

        with tracer.start_run(
            name="customer_support_agent",
        ) as run:
            ...

    A run is considered failed when either:

    1. An exception escapes the run context.
    2. Any event belonging to the run finishes with FAILED status.

    The second rule is important for agent systems because a tool,
    function, or LLM operation may fail while the application catches
    the exception and continues execution.
    """

    def __init__(
        self,
        *,
        tracer: Any,
        run_id: str | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize a TraceRun.

        Args:
            tracer:
                Tracer responsible for recording events.

            run_id:
                Optional externally supplied run ID.

            name:
                Optional human-readable run name.

            metadata:
                Optional metadata associated with the run.
        """

        self.tracer = tracer

        self.run_id = (
            run_id
            or str(uuid4())
        )

        self.name = name
        self.metadata = metadata or {}

        self.start_event: TraceEvent | None = None
        self.end_event: TraceEvent | None = None

        self.status: EventStatus | None = None
        self.duration_ms: float | None = None

        self.exception: BaseException | None = None

        self._started_at: float | None = None
        self._context_manager: Any = None
        self._context: TraceContext | None = None
        self._parent_context: TraceContext | None = None
        self._parent_event_id: str | None = None

        self._entered = False
        self._closed = False

    # ------------------------------------------------------------------
    # Context Manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "TraceRun":
        """
        Start the run and create the RUN_START event.
        """

        if self._entered:
            raise RuntimeError(
                "TraceRun cannot be entered more than once."
            )

        self._entered = True

        self._parent_context = (
            get_current_context()
        )

        if self._parent_context is not None:
            self._parent_event_id = (
                self._parent_context.current_event_id
            )

        self._context_manager = trace_context(
            run_id=self.run_id,
            metadata=self.metadata,
            tracer=self.tracer,
        )

        self._context = (
            self._context_manager.__enter__()
        )

        self._started_at = time.perf_counter()

        self.start_event = self.tracer.start_event(
            event_type=EventType.RUN_START,
            name=self.name,
            metadata=self.metadata,
            parent_event_id=self._parent_event_id,
        )

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> bool:
        """
        Finish the run.

        A run is FAILED when:

        - an application exception escapes the context, or
        - any event in this run has FAILED status.

        Otherwise the run is SUCCESS.

        Application exceptions are never swallowed.
        """

        if self._closed:
            return False

        self.exception = exc_value

        if self._started_at is not None:
            self.duration_ms = (
                time.perf_counter()
                - self._started_at
            ) * 1000

        if self.exception is not None:
            self.status = EventStatus.FAILED

        elif self._has_failed_event():
            self.status = EventStatus.FAILED

        else:
            self.status = EventStatus.SUCCESS

        try:
            self._record_end_event()

        finally:
            self._restore_parent_context()

            self._closed = True

        # Returning False means the original application exception
        # propagates normally.
        return False

    # ------------------------------------------------------------------
    # Failure Detection
    # ------------------------------------------------------------------

    def _has_failed_event(self) -> bool:
        """
        Determine whether any event in this run failed.

        This allows the run to become FAILED even when an individual
        tool/function/LLM failure was caught and handled by the
        application before the run exited.
        """

        events = self.tracer.get_events_for_run(
            self.run_id
        )

        return any(
            event.status == EventStatus.FAILED
            for event in events
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _record_end_event(self) -> None:
        """
        Record the RUN_END event using the final run status.
        """

        if self.status is None:
            return

        self.end_event = self.tracer.start_event(
            event_type=EventType.RUN_END,
            name=self.name,
            parent_event_id=(
                self._current_run_start_id()
            ),
        )

        if self.status == EventStatus.FAILED:
            self.tracer.fail_event(
                self.end_event,
                error=self.exception,
                duration_ms=self.duration_ms,
            )

        else:
            self.tracer.complete_event(
                self.end_event,
                output_data={
                    "status": self.status.value,
                },
                duration_ms=self.duration_ms,
            )

    def _current_run_start_id(self) -> str | None:
        """
        Return the active RUN_START event ID.

        The RUN_START event remains at the bottom of this run's
        event stack while child events execute.
        """

        if self._context is None:
            return None

        return self._context.current_event_id

    def _restore_parent_context(self) -> None:
        """
        Remove the RUN_START event from this run's stack and restore
        the parent tracing context.
        """

        if self._context is not None:
            if (
                self._context.current_event_id
                == (
                    self.start_event.event_id
                    if self.start_event is not None
                    else None
                )
            ):
                self._context.pop_event()

        if self._context_manager is not None:
            self._context_manager.__exit__(
                None,
                None,
                None,
            )

    # ------------------------------------------------------------------
    # Public Helpers
    # ------------------------------------------------------------------

    @property
    def succeeded(self) -> bool:
        """
        Return True when the run completed successfully.
        """

        return self.status == EventStatus.SUCCESS

    @property
    def failed(self) -> bool:
        """
        Return True when the run failed.
        """

        return self.status == EventStatus.FAILED

    @property
    def completed(self) -> bool:
        """
        Return True when the lifecycle has ended.
        """

        return self._closed