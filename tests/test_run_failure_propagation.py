"""
Regression tests for run-level failure propagation.
"""

from agent_black_box import (
    EventType,
    InMemoryStorage,
    Tracer,
)


def test_failed_child_event_marks_run_as_failed():
    """
    A failed child event must make the overall run FAILED even when
    no exception escapes the run context.
    """

    tracer = Tracer(
        storage=InMemoryStorage()
    )

    with tracer.start_run(
        run_id="run-failure-propagation-test",
        name="failure-propagation-agent",
    ):
        event = tracer.start_event(
            event_type=EventType.TOOL_CALL,
            name="failing_tool",
            input_data={
                "value": 42,
            },
        )

        tracer.fail_event(
            event,
            error=RuntimeError(
                "Synthetic tool failure"
            ),
            duration_ms=12.5,
        )

    events = tracer.get_events_for_run(
        "run-failure-propagation-test"
    )

    assert len(events) == 3

    tool_event = next(
        event
        for event in events
        if event.event_type == "tool_call"
    )

    run_end = next(
        event
        for event in events
        if event.event_type == "run_end"
    )

    assert tool_event.status == "failed"
    assert tool_event.error_type == "RuntimeError"
    assert (
        tool_event.error_message
        == "Synthetic tool failure"
    )

    assert run_end.status == "failed"


def test_successful_run_remains_successful():
    """
    A run containing only successful events must remain SUCCESS.
    """

    tracer = Tracer(
        storage=InMemoryStorage()
    )

    with tracer.start_run(
        run_id="successful-run-test",
        name="successful-agent",
    ):
        event = tracer.start_event(
            event_type=EventType.TOOL_CALL,
            name="successful_tool",
        )

        tracer.complete_event(
            event,
            output_data={
                "result": "ok",
            },
            duration_ms=5.0,
        )

    events = tracer.get_events_for_run(
        "successful-run-test"
    )

    run_end = next(
        event
        for event in events
        if event.event_type == "run_end"
    )

    assert run_end.status == "success"