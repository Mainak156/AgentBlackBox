"""
Tests for Agent Black Box run lifecycle management.
"""

from agent_black_box.context import (
    get_current_context,
    get_current_run_id,
)
from agent_black_box.events import (
    EventStatus,
    EventType,
)
from agent_black_box.tracer import Tracer


def test_start_run_creates_run_id():
    tracer = Tracer()

    with tracer.start_run(
        name="test-agent",
    ) as run:
        assert run.run_id is not None
        assert len(run.run_id) > 0


def test_start_run_uses_supplied_run_id():
    tracer = Tracer()

    with tracer.start_run(
        name="test-agent",
        run_id="run-123",
    ) as run:
        assert run.run_id == "run-123"


def test_run_records_start_and_end_events():
    tracer = Tracer()

    with tracer.start_run(
        name="test-agent",
        run_id="run-lifecycle",
    ) as run:
        assert run.start_event is not None

    events = tracer.get_events_for_run(
        "run-lifecycle"
    )

    assert len(events) == 2

    assert (
        events[0].event_type
        == EventType.RUN_START
    )

    assert (
        events[1].event_type
        == EventType.RUN_END
    )


def test_run_start_event_has_started_status():
    tracer = Tracer()

    with tracer.start_run(
        run_id="run-start-status",
    ):
        pass

    event = tracer.get_events_for_run(
        "run-start-status"
    )[0]

    assert (
        event.status
        == EventStatus.STARTED
    )


def test_successful_run_has_successful_end():
    tracer = Tracer()

    with tracer.start_run(
        run_id="successful-run",
    ) as run:
        pass

    assert run.succeeded is True
    assert run.failed is False
    assert run.completed is True

    events = tracer.get_events_for_run(
        "successful-run"
    )

    end_event = events[-1]

    assert (
        end_event.event_type
        == EventType.RUN_END
    )

    assert (
        end_event.status
        == EventStatus.SUCCESS
    )


def test_successful_run_records_duration():
    tracer = Tracer()

    with tracer.start_run(
        run_id="duration-run",
    ) as run:
        pass

    assert run.duration_ms is not None
    assert run.duration_ms >= 0

    end_event = tracer.get_events_for_run(
        "duration-run"
    )[-1]

    assert end_event.duration_ms is not None
    assert end_event.duration_ms >= 0


def test_successful_run_records_output_status():
    tracer = Tracer()

    with tracer.start_run(
        run_id="output-run",
    ):
        pass

    end_event = tracer.get_events_for_run(
        "output-run"
    )[-1]

    assert end_event.output_data == {
        "status": "success",
    }


def test_failed_run_records_failure():
    tracer = Tracer()

    try:
        with tracer.start_run(
            run_id="failed-run",
        ):
            raise ValueError(
                "Something went wrong"
            )
    except ValueError:
        pass

    events = tracer.get_events_for_run(
        "failed-run"
    )

    assert len(events) == 2

    end_event = events[-1]

    assert (
        end_event.event_type
        == EventType.RUN_END
    )

    assert (
        end_event.status
        == EventStatus.FAILED
    )

    assert (
        end_event.error_type
        == "ValueError"
    )

    assert (
        end_event.error_message
        == "Something went wrong"
    )


def test_failed_run_does_not_swallow_exception():
    tracer = Tracer()

    try:
        with tracer.start_run(
            run_id="exception-propagation",
        ):
            raise RuntimeError(
                "Application failure"
            )
    except RuntimeError as exc:
        assert str(exc) == (
            "Application failure"
        )
    else:
        raise AssertionError(
            "Application exception was swallowed."
        )


def test_failed_run_duration_is_recorded():
    tracer = Tracer()

    try:
        with tracer.start_run(
            run_id="failed-duration",
        ):
            raise RuntimeError(
                "failure"
            )
    except RuntimeError:
        pass

    end_event = tracer.get_events_for_run(
        "failed-duration"
    )[-1]

    assert end_event.duration_ms is not None
    assert end_event.duration_ms >= 0


def test_run_metadata_is_recorded():
    tracer = Tracer()

    with tracer.start_run(
        run_id="metadata-run",
        name="support-agent",
        metadata={
            "environment": "testing",
            "version": "1.0",
        },
    ):
        pass

    start_event = tracer.get_events_for_run(
        "metadata-run"
    )[0]

    assert start_event.name == "support-agent"

    assert start_event.metadata == {
        "environment": "testing",
        "version": "1.0",
    }


def test_run_context_is_available_inside_run():
    tracer = Tracer()

    with tracer.start_run(
        run_id="context-run",
    ):
        assert (
            get_current_run_id()
            == "context-run"
        )

        context = get_current_context()

        assert context is not None
        assert context.run_id == "context-run"


def test_run_context_is_restored_after_run():
    tracer = Tracer()

    assert get_current_context() is None

    with tracer.start_run(
        run_id="restore-run",
    ):
        assert (
            get_current_run_id()
            == "restore-run"
        )

    assert get_current_context() is None
    assert get_current_run_id() is None


def test_nested_events_are_children_of_run_start():
    tracer = Tracer()

    with tracer.start_run(
        run_id="nested-run",
    ) as run:
        event = tracer.start_event(
            event_type=EventType.FUNCTION_CALL,
            name="process",
        )

        tracer.complete_event(
            event,
            output_data={
                "result": "ok",
            },
        )

    events = tracer.get_events_for_run(
        "nested-run"
    )

    assert len(events) == 3

    start_event = events[0]
    function_event = events[1]
    end_event = events[-1]

    assert (
        start_event.event_type
        == EventType.RUN_START
    )

    assert (
        function_event.parent_event_id
        == start_event.event_id
    )

    assert (
        end_event.parent_event_id
        == start_event.event_id
    )

    assert run.start_event is not None
    assert run.end_event is not None


def test_nested_traced_function_is_inside_run():
    tracer = Tracer()

    from agent_black_box.decorators import trace

    @trace
    def process(value: int) -> int:
        return value * 2

    with tracer.start_run(
        run_id="decorated-run",
    ):
        result = process(21)

    assert result == 42

    events = tracer.get_events_for_run(
        "decorated-run"
    )

    assert len(events) == 3

    assert (
        events[0].event_type
        == EventType.RUN_START
    )

    assert (
        events[1].event_type
        == EventType.FUNCTION_CALL
    )

    assert (
        events[2].event_type
        == EventType.RUN_END
    )

    assert (
        events[1].parent_event_id
        == events[0].event_id
    )


def test_nested_run_gets_separate_run_id():
    tracer = Tracer()

    with tracer.start_run(
        run_id="outer-run",
    ) as outer:

        with tracer.start_run(
            run_id="inner-run",
        ) as inner:
            assert (
                get_current_run_id()
                == "inner-run"
            )

        assert (
            get_current_run_id()
            == "outer-run"
        )

    assert outer.run_id == "outer-run"
    assert inner.run_id == "inner-run"


def test_nested_run_is_parented_to_outer_event():
    tracer = Tracer()

    with tracer.start_run(
        run_id="outer-run",
    ):

        function_event = tracer.start_event(
            event_type=EventType.FUNCTION_CALL,
            name="outer_function",
        )

        with tracer.start_run(
            run_id="inner-run",
        ):
            inner_event = tracer.start_event(
                event_type=EventType.FUNCTION_CALL,
                name="inner_function",
            )

            tracer.complete_event(
                inner_event,
            )

        tracer.complete_event(
            function_event,
        )

    outer_events = tracer.get_events_for_run(
        "outer-run"
    )

    inner_events = tracer.get_events_for_run(
        "inner-run"
    )

    assert len(outer_events) == 3
    assert len(inner_events) == 3

    outer_function = outer_events[1]
    inner_start = inner_events[0]
    inner_function = inner_events[1]

    assert (
        inner_start.parent_event_id
        == outer_function.event_id
    )

    assert (
        inner_function.parent_event_id
        == inner_start.event_id
    )


def test_run_name_is_preserved():
    tracer = Tracer()

    with tracer.start_run(
        name="customer-support-agent",
        run_id="named-run",
    ):
        pass

    events = tracer.get_events_for_run(
        "named-run"
    )

    assert (
        events[0].name
        == "customer-support-agent"
    )

    assert (
        events[-1].name
        == "customer-support-agent"
    )


def test_run_generates_unique_ids():
    tracer = Tracer()

    with tracer.start_run() as first:
        pass

    with tracer.start_run() as second:
        pass

    assert first.run_id != second.run_id


def test_run_is_completed_after_success():
    tracer = Tracer()

    with tracer.start_run() as run:
        assert run.completed is False

    assert run.completed is True


def test_run_is_completed_after_failure():
    tracer = Tracer()

    try:
        with tracer.start_run(
            run_id="completed-failure",
        ) as run:
            raise RuntimeError(
                "failure"
            )
    except RuntimeError:
        pass

    assert run.completed is True
    assert run.failed is True


def test_run_end_is_available_after_completion():
    tracer = Tracer()

    with tracer.start_run() as run:
        assert run.end_event is None

    assert run.end_event is not None


def test_run_start_and_end_share_run_id():
    tracer = Tracer()

    with tracer.start_run(
        run_id="shared-run-id",
    ):
        pass

    events = tracer.get_events_for_run(
        "shared-run-id"
    )

    assert (
        events[0].run_id
        == events[1].run_id
        == "shared-run-id"
    )