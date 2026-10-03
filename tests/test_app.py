"""
Tests for the Agent Black Box Streamlit replay dashboard.
"""

from agent_black_box import EventType, InMemoryStorage, Tracer

from dashboard.app import (
    calculate_run_duration,
    calculate_total_cost,
    calculate_total_tokens,
    event_depth,
    event_icon,
    event_to_json,
    format_cost,
    format_duration,
    get_failed_events,
    get_root_cause_event,
    get_run_end,
    get_run_events,
    get_run_ids,
    get_run_name,
    get_run_start,
    get_run_status,
)


def create_sample_trace():
    """Create a deterministic successful trace for dashboard tests."""
    storage = InMemoryStorage()
    tracer = Tracer(storage=storage)

    with tracer.start_run(
        run_id="dashboard-test-run",
        name="dashboard-agent",
        metadata={
            "environment": "testing",
        },
    ):
        event = tracer.start_event(
            event_type=EventType.FUNCTION_CALL,
            name="sample_function",
            input_data={
                "value": 10,
            },
        )

        tracer.complete_event(
            event,
            output_data={
                "result": 20,
            },
            duration_ms=12.5,
        )

    return tracer.get_events_for_run(
        "dashboard-test-run"
    )


def create_failed_trace():
    """Create a deterministic failed trace for root-cause tests."""
    storage = InMemoryStorage()
    tracer = Tracer(storage=storage)

    try:
        with tracer.start_run(
            run_id="dashboard-failed-run",
            name="failed-dashboard-agent",
            metadata={
                "environment": "testing",
            },
        ):
            successful_event = tracer.start_event(
                event_type=EventType.FUNCTION_CALL,
                name="classify_customer_issue",
                input_data={
                    "ticket": "delivery_issue",
                },
            )

            tracer.complete_event(
                successful_event,
                output_data={
                    "category": "delivery_issue",
                },
                duration_ms=3.4,
            )

            failed_event = tracer.start_event(
                event_type=EventType.TOOL_CALL,
                name="create_support_ticket",
                input_data={
                    "category": "delivery_issue",
                },
            )

            try:
                raise RuntimeError(
                    "Support ticket service returned HTTP 503"
                )
            except RuntimeError as exc:
                tracer.fail_event(
                    failed_event,
                    error=exc,
                    duration_ms=84.7,
                )
                raise

    except RuntimeError:
        pass

    return tracer.get_events_for_run(
        "dashboard-failed-run"
    )


def test_get_run_ids_preserves_order():
    events = create_sample_trace()

    duplicate_run_event = events[0].model_copy(
        update={
            "event_id": "duplicate-run-event",
        }
    )

    result = get_run_ids(
        events + [duplicate_run_event]
    )

    assert result == [
        "dashboard-test-run",
    ]


def test_get_run_events_filters_correct_run():
    events = create_sample_trace()

    result = get_run_events(
        events,
        "dashboard-test-run",
    )

    assert len(result) == 3

    assert get_run_events(
        events,
        "missing-run",
    ) == []


def test_get_run_start():
    events = create_sample_trace()

    run_start = get_run_start(events)

    assert run_start is not None
    assert run_start.event_type == "run_start"


def test_get_run_end():
    events = create_sample_trace()

    run_end = get_run_end(events)

    assert run_end is not None
    assert run_end.event_type == "run_end"
    assert run_end.status == "success"


def test_get_run_name():
    events = create_sample_trace()

    assert get_run_name(events) == "dashboard-agent"


def test_get_run_status():
    events = create_sample_trace()

    assert get_run_status(events) == "success"


def test_get_failed_events_returns_only_operational_failures():
    events = create_failed_trace()

    failed_events = get_failed_events(events)

    assert len(failed_events) == 1
    assert failed_events[0].name == "create_support_ticket"
    assert failed_events[0].event_type == "tool_call"
    assert failed_events[0].status == "failed"


def test_get_failed_events_excludes_run_end():
    events = create_failed_trace()

    failed_events = get_failed_events(events)

    assert all(
        event.event_type != "run_end"
        for event in failed_events
    )


def test_get_root_cause_event_returns_first_failed_operation():
    events = create_failed_trace()

    root_cause = get_root_cause_event(events)

    assert root_cause is not None
    assert root_cause.name == "create_support_ticket"
    assert root_cause.event_type == "tool_call"
    assert root_cause.error_type == "RuntimeError"
    assert (
        root_cause.error_message
        == "Support ticket service returned HTTP 503"
    )


def test_get_root_cause_event_returns_none_for_successful_run():
    events = create_sample_trace()

    assert get_root_cause_event(events) is None


def test_calculate_run_duration():
    events = create_sample_trace()

    duration = calculate_run_duration(events)

    assert duration is not None
    assert duration >= 0


def test_calculate_total_tokens():
    events = create_sample_trace()

    assert calculate_total_tokens(events) == 0


def test_calculate_total_cost():
    events = create_sample_trace()

    assert calculate_total_cost(events) == 0


def test_format_duration():
    assert format_duration(None) == "—"
    assert format_duration(12.5) == "12.50 ms"
    assert format_duration(1500) == "1.50 s"


def test_format_cost():
    assert format_cost(0) == "$0.00"
    assert format_cost(0.123456) == "$0.123456"


def test_event_icons():
    assert event_icon(
        "run_start",
        "started",
    ) == "▶️"

    assert event_icon(
        "run_end",
        "success",
    ) == "⏹️"

    assert event_icon(
        "llm_call",
        "started",
    ) == "🤖"

    assert event_icon(
        "tool_call",
        "started",
    ) == "🔧"

    assert event_icon(
        "function_call",
        "success",
    ) == "⚙️"

    assert event_icon(
        "unknown",
        "failed",
    ) == "🔴"


def test_event_depth():
    events = create_sample_trace()

    run_start = events[0]
    function_event = events[1]

    assert event_depth(
        run_start,
        events,
    ) == 0

    assert event_depth(
        function_event,
        events,
    ) == 1


def test_event_to_json():
    events = create_sample_trace()

    serialized = event_to_json(
        events[1]
    )

    assert (
        '"event_type": "function_call"'
        in serialized
    )

    assert (
        '"name": "sample_function"'
        in serialized
    )


def test_dashboard_run_contains_expected_events():
    events = create_sample_trace()

    event_types = [
        event.event_type
        for event in events
    ]

    assert event_types == [
        "run_start",
        "function_call",
        "run_end",
    ]