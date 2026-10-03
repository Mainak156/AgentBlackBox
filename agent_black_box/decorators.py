"""
Tracing decorators for Agent Black Box.

This module provides the @trace decorator for automatically
recording function execution as TraceEvents.
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, TypeVar, cast

from agent_black_box.context import (
    get_current_run_id,
    get_current_tracer,
)
from agent_black_box.events import EventType


F = TypeVar("F", bound=Callable[..., Any])


def trace(
    func: F | None = None,
    *,
    name: str | None = None,
    event_type: EventType = EventType.FUNCTION_CALL,
) -> F | Callable[[F], F]:
    """
    Automatically trace a function execution.

    The decorator records:

    - Function name
    - Input arguments
    - Output value
    - Execution duration
    - Success/failure status
    - Exception details
    - Parent/child event relationships

    The decorator supports both forms:

        @trace
        def my_function():
            ...

    and:

        @trace(name="custom_name")
        def my_function():
            ...

    If no active tracing context or tracer exists, the function
    executes normally without generating a trace event.
    """

    def decorator(target: F) -> F:
        """Apply tracing behaviour to the target function."""

        @wraps(target)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Execute the target function with tracing."""

            # --------------------------------------------------------------
            # Retrieve active tracing infrastructure
            # --------------------------------------------------------------

            run_id = get_current_run_id()
            tracer = get_current_tracer()

            # --------------------------------------------------------------
            # Graceful fallback
            # --------------------------------------------------------------
            #
            # The observability system must never become a dependency
            # that causes the customer's application to fail.
            #
            # If tracing is not active, simply execute the function.
            # --------------------------------------------------------------

            if run_id is None or tracer is None:
                return target(*args, **kwargs)

            # --------------------------------------------------------------
            # Prepare event information
            # --------------------------------------------------------------

            event_name = name or target.__name__

            input_data = {
                "args": args,
                "kwargs": kwargs,
            }

            # --------------------------------------------------------------
            # Start timing
            # --------------------------------------------------------------

            start_time = time.perf_counter()

            # --------------------------------------------------------------
            # Create trace event
            # --------------------------------------------------------------

            event = tracer.start_event(
                event_type=event_type,
                name=event_name,
                input_data=input_data,
            )

            # --------------------------------------------------------------
            # Execute function
            # --------------------------------------------------------------

            try:
                result = target(*args, **kwargs)

            except Exception as exc:
                # ----------------------------------------------------------
                # Failure handling
                # ----------------------------------------------------------

                duration_ms = (
                    time.perf_counter() - start_time
                ) * 1000

                tracer.fail_event(
                    event,
                    error=exc,
                    duration_ms=duration_ms,
                )

                # Never swallow the customer's exception.
                raise

            else:
                # ----------------------------------------------------------
                # Success handling
                # ----------------------------------------------------------

                duration_ms = (
                    time.perf_counter() - start_time
                ) * 1000

                tracer.complete_event(
                    event,
                    output_data=result,
                    duration_ms=duration_ms,
                )

                return result

        return cast(F, wrapper)

    # ----------------------------------------------------------------------
    # Support both @trace and @trace(...)
    # ----------------------------------------------------------------------

    if func is not None:
        return decorator(func)

    return decorator