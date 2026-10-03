from datetime import datetime

from agent_black_box.events import (
    EventStatus,
    EventType,
    TraceEvent,
)

from agent_black_box.context import (
    TraceContext,
    get_current_context,
    get_current_run_id,
    trace_context,
)

from agent_black_box.tracer import Tracer

from agent_black_box.decorators import trace


def test_trace_event_creation():
    event = TraceEvent(
        run_id="test-run-001",
        event_type=EventType.TOOL_CALL,
        status=EventStatus.SUCCESS,
        name="calculator",
        input_data={"a": 10, "b": 20},
        output_data={"result": 30},
        duration_ms=12.5,
    )

    assert event.run_id == "test-run-001"
    assert event.event_type == "tool_call"
    assert event.status == "success"
    assert event.name == "calculator"
    assert event.input_data["a"] == 10
    assert event.output_data["result"] == 30
    assert event.duration_ms == 12.5
    assert event.schema_version == 1


def test_event_id_is_generated():
    event = TraceEvent(
        run_id="test-run-001",
        event_type=EventType.RUN_START,
    )

    assert event.event_id is not None
    assert len(event.event_id) > 0


def test_timestamp_is_generated():
    event = TraceEvent(
        run_id="test-run-001",
        event_type=EventType.RUN_START,
    )

    assert isinstance(event.timestamp, datetime)


def test_default_status():
    event = TraceEvent(
        run_id="test-run-001",
        event_type=EventType.RUN_START,
    )

    assert event.status == "started"


def test_metadata_defaults_to_empty_dict():
    event = TraceEvent(
        run_id="test-run-001",
        event_type=EventType.RUN_START,
    )

    assert event.metadata == {}


def test_trace_context_creates_run_id():
    with trace_context() as context:
        assert context.run_id is not None
        assert len(context.run_id) > 0


def test_current_run_id_is_available_inside_context():
    with trace_context(run_id="run-123"):

        assert get_current_run_id() == "run-123"


def test_context_is_cleared_after_exit():
    with trace_context(run_id="run-123"):

        assert get_current_run_id() == "run-123"

    assert get_current_run_id() is None


def test_context_metadata():
    with trace_context(
        run_id="run-456",
        metadata={"environment": "test"},
    ) as context:

        assert context.metadata["environment"] == "test"


def test_event_stack():
    context = TraceContext(run_id="run-789")

    assert context.current_event_id is None

    context.push_event("event-1")

    assert context.current_event_id == "event-1"

    context.push_event("event-2")

    assert context.current_event_id == "event-2"

    popped = context.pop_event()

    assert popped == "event-2"
    assert context.current_event_id == "event-1"


def test_tracer_creates_event():
    tracer = Tracer()

    with trace_context(run_id="run-001"):

        event = tracer.start_event(
            event_type=EventType.TOOL_CALL,
            name="calculator",
            input_data={"a": 10, "b": 20},
        )

    assert event.run_id == "run-001"
    assert event.event_type == "tool_call"
    assert event.status == "started"
    assert event.name == "calculator"


def test_tracer_completes_event():
    tracer = Tracer()

    with trace_context(run_id="run-002"):

        event = tracer.start_event(
            event_type=EventType.TOOL_CALL,
            name="calculator",
        )

        tracer.complete_event(
            event,
            output_data={"result": 30},
            duration_ms=12.5,
        )

    assert event.status == "success"
    assert event.output_data == {"result": 30}
    assert event.duration_ms == 12.5


def test_tracer_fails_event():
    tracer = Tracer()

    with trace_context(run_id="run-003"):

        event = tracer.start_event(
            event_type=EventType.TOOL_CALL,
            name="calculator",
        )

        error = ValueError("Invalid calculation")

        tracer.fail_event(
            event,
            error=error,
            duration_ms=5.2,
        )

    assert event.status == "failed"
    assert event.error_type == "ValueError"
    assert event.error_message == "Invalid calculation"
    assert event.duration_ms == 5.2


def test_tracer_requires_context():
    tracer = Tracer()

    try:
        tracer.start_event(
            event_type=EventType.TOOL_CALL,
            name="calculator",
        )

        assert False, "Expected RuntimeError"

    except RuntimeError as exc:
        assert "No active trace context" in str(exc)


def test_tracer_filters_events_by_run():
    tracer = Tracer()

    with trace_context(run_id="run-A"):

        tracer.start_event(
            event_type=EventType.TOOL_CALL,
            name="tool-A",
        )

    with trace_context(run_id="run-B"):

        tracer.start_event(
            event_type=EventType.TOOL_CALL,
            name="tool-B",
        )

    events_a = tracer.get_events_for_run("run-A")
    events_b = tracer.get_events_for_run("run-B")

    assert len(events_a) == 1
    assert len(events_b) == 1

    assert events_a[0].name == "tool-A"
    assert events_b[0].name == "tool-B"


def test_nested_events():
    tracer = Tracer()

    with trace_context(run_id="nested-run"):

        parent = tracer.start_event(
            event_type=EventType.FUNCTION_CALL,
            name="agent",
        )

        child = tracer.start_event(
            event_type=EventType.LLM_CALL,
            name="llm",
        )

        assert child.parent_event_id == parent.event_id

        tracer.complete_event(
            child,
            output_data="response",
        )

        tracer.complete_event(
            parent,
            output_data="final",
        )


def test_trace_decorator_records_successful_function():
    tracer = Tracer()

    @trace
    def add(a: int, b: int) -> int:
        return a + b

    with trace_context(
        run_id="decorator-run-001",
        tracer=tracer,
    ):
        result = add(10, 20)

    assert result == 30

    events = tracer.get_events_for_run("decorator-run-001")

    assert len(events) == 1

    event = events[0]

    assert event.name == "add"
    assert event.event_type == "function_call"
    assert event.status == "success"
    assert event.input_data["args"] == (10, 20)
    assert event.output_data == 30
    assert event.duration_ms is not None
    assert event.duration_ms >= 0


def test_trace_decorator_records_failure():
    tracer = Tracer()

    @trace
    def failing_function() -> None:
        raise ValueError("Something went wrong")

    with trace_context(
        run_id="decorator-run-002",
        tracer=tracer,
    ):

        try:
            failing_function()
            assert False, "Expected ValueError"

        except ValueError as exc:
            assert str(exc) == "Something went wrong"

    events = tracer.get_events_for_run("decorator-run-002")

    assert len(events) == 1

    event = events[0]

    assert event.name == "failing_function"
    assert event.status == "failed"
    assert event.error_type == "ValueError"
    assert event.error_message == "Something went wrong"
    assert event.duration_ms is not None
    assert event.duration_ms >= 0


def test_trace_decorator_custom_name():
    tracer = Tracer()

    @trace(name="custom-operation")
    def calculate() -> int:
        return 42

    with trace_context(
        run_id="decorator-run-003",
        tracer=tracer,
    ):
        result = calculate()

    assert result == 42

    events = tracer.get_events_for_run("decorator-run-003")

    assert len(events) == 1
    assert events[0].name == "custom-operation"


def test_trace_decorator_preserves_function_metadata():
    tracer = Tracer()

    @trace
    def documented_function() -> str:
        """This is a test function."""

        return "hello"

    with trace_context(
        run_id="decorator-run-004",
        tracer=tracer,
    ):
        result = documented_function()

    assert result == "hello"
    assert documented_function.__name__ == "documented_function"
    assert documented_function.__doc__ == "This is a test function."


def test_trace_without_context_runs_normally():
    @trace
    def normal_function(value: int) -> int:
        return value * 2

    result = normal_function(10)

    assert result == 20


def test_trace_with_custom_event_type():
    tracer = Tracer()

    @trace(event_type=EventType.TOOL_CALL)
    def search_tool(query: str) -> str:
        return f"Results for {query}"

    with trace_context(
        run_id="decorator-run-005",
        tracer=tracer,
    ):
        result = search_tool("AI agents")

    assert result == "Results for AI agents"

    events = tracer.get_events_for_run("decorator-run-005")

    assert len(events) == 1
    assert events[0].event_type == "tool_call"


def test_nested_traced_functions_share_same_tracer():
    tracer = Tracer()

    @trace
    def inner_function(value: int) -> int:
        return value * 2

    @trace
    def outer_function(value: int) -> int:
        return inner_function(value)

    with trace_context(
        run_id="decorator-run-006",
        tracer=tracer,
    ):
        result = outer_function(10)

    assert result == 20

    events = tracer.get_events_for_run("decorator-run-006")

    assert len(events) == 2

    outer_event = events[0]
    inner_event = events[1]

    assert outer_event.name == "outer_function"
    assert inner_event.name == "inner_function"

    assert inner_event.parent_event_id == outer_event.event_id