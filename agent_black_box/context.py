"""
Execution context management for Agent Black Box.

This module provides a lightweight execution context that allows
trace information, especially the run_id and active tracer, to
propagate through nested function calls without requiring every
function to explicitly receive them.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator
from uuid import uuid4


# ---------------------------------------------------------------------------
# Context Variables
# ---------------------------------------------------------------------------

_current_context: ContextVar["TraceContext | None"] = ContextVar(
    "agent_black_box_context",
    default=None,
)


# ---------------------------------------------------------------------------
# Trace Context
# ---------------------------------------------------------------------------

@dataclass
class TraceContext:
    """
    Represents the execution context of one complete agent run.

    The context stores run-level information and provides access to
    the Tracer associated with the current execution.

    The tracer is intentionally typed as Any to avoid a circular import
    between context.py and tracer.py.
    """

    # Unique identifier for the complete execution.
    run_id: str = field(
        default_factory=lambda: str(uuid4())
    )

    # UTC timestamp at which the run started.
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Optional metadata associated with the complete run.
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # Optional parent event for the overall context.
    parent_event_id: str | None = None

    # Tracer responsible for recording events in this run.
    #
    # Typed as Any deliberately to avoid:
    #
    # context.py → tracer.py → context.py
    #
    # circular imports.
    tracer: Any = None

    # Stack used to track nested events.
    _event_stack: list[str] = field(
        default_factory=list
    )

    # -----------------------------------------------------------------------
    # Event Stack
    # -----------------------------------------------------------------------

    @property
    def current_event_id(self) -> str | None:
        """
        Return the ID of the currently active event.

        Returns:
            str | None:
                The most recently pushed event ID, or None if
                no event is currently active.
        """

        if not self._event_stack:
            return None

        return self._event_stack[-1]

    def push_event(self, event_id: str) -> None:
        """
        Push an event onto the active event stack.

        Args:
            event_id:
                Unique identifier of the event being activated.
        """

        self._event_stack.append(event_id)

    def pop_event(self) -> str | None:
        """
        Remove and return the most recently active event.

        Returns:
            str | None:
                The removed event ID, or None if the stack is empty.
        """

        if not self._event_stack:
            return None

        return self._event_stack.pop()


# ---------------------------------------------------------------------------
# Context Access
# ---------------------------------------------------------------------------

def get_current_context() -> TraceContext | None:
    """
    Return the currently active TraceContext.

    Returns:
        TraceContext | None:
            The active context, or None when no trace is active.
    """

    return _current_context.get()


def get_current_run_id() -> str | None:
    """
    Return the run ID of the currently active trace.

    Returns:
        str | None:
            The current run ID, or None when no trace is active.
    """

    context = get_current_context()

    if context is None:
        return None

    return context.run_id


def get_current_tracer() -> Any:
    """
    Return the tracer associated with the current execution context.

    Returns:
        Any:
            The active Tracer instance, or None if no tracer is attached.
    """

    context = get_current_context()

    if context is None:
        return None

    return context.tracer


# ---------------------------------------------------------------------------
# Context Manager
# ---------------------------------------------------------------------------

@contextmanager
def trace_context(
    *,
    run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    tracer: Any = None,
) -> Iterator[TraceContext]:
    """
    Create and activate a tracing context.

    Args:
        run_id:
            Optional externally supplied run ID.

            If omitted, a UUID is automatically generated.

        metadata:
            Optional metadata associated with the complete run.

        tracer:
            Optional Tracer instance responsible for recording events
            during this execution.

    Yields:
        TraceContext:
            The active tracing context.

    Example:
        tracer = Tracer()

        with trace_context(
            run_id="demo-001",
            tracer=tracer,
        ):
            ...
    """

    context = TraceContext(
        run_id=run_id or str(uuid4()),
        metadata=metadata or {},
        tracer=tracer,
    )

    token = _current_context.set(context)

    try:
        yield context

    finally:
        _current_context.reset(token)