from agent_black_box import (
    EventStatus,
    EventType,
    InMemoryStorage,
    JSONLStorage,
    SQLiteStorage,
    TraceEvent,
    Tracer,
    get_current_context,
    get_current_run_id,
    get_current_tracer,
    trace,
    trace_context,
)


def test_public_api_exports_core_objects():
    assert Tracer is not None
    assert TraceEvent is not None
    assert EventType is not None
    assert EventStatus is not None
    assert trace is not None


def test_public_api_exports_storage_backends():
    assert InMemoryStorage is not None
    assert SQLiteStorage is not None
    assert JSONLStorage is not None


def test_public_api_exports_context_helpers():
    assert trace_context is not None
    assert get_current_context is not None
    assert get_current_run_id is not None
    assert get_current_tracer is not None


def test_public_api_tracing_works():
    tracer = Tracer(storage=InMemoryStorage())

    with tracer.start_run(
        run_id="public-api-test",
        name="public-api-agent",
    ):
        @trace
        def hello(name: str) -> str:
            return f"Hello {name}"

        result = hello("Agent Black Box")

    assert result == "Hello Agent Black Box"

    events = tracer.get_events_for_run("public-api-test")

    assert len(events) == 3
    assert events[0].event_type == "run_start"
    assert events[1].event_type == "function_call"
    assert events[2].event_type == "run_end"