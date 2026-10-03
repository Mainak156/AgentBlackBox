"""
Tests for Agent Black Box storage backends.
"""

from pathlib import Path

import pytest

from agent_black_box.context import trace_context
from agent_black_box.events import (
    EventStatus,
    EventType,
    TraceEvent,
)
from agent_black_box.storage.base import StorageBackend
from agent_black_box.storage.jsonl import JSONLStorage
from agent_black_box.storage.memory import InMemoryStorage
from agent_black_box.storage.sqlite import SQLiteStorage
from agent_black_box.tracer import Tracer


def create_event(
    *,
    run_id: str = "run-001",
    name: str = "test_function",
) -> TraceEvent:
    """
    Create a test TraceEvent.
    """

    return TraceEvent(
        run_id=run_id,
        event_type=EventType.FUNCTION_CALL,
        status=EventStatus.STARTED,
        name=name,
    )


# ============================================================================
# InMemoryStorage Tests
# ============================================================================


def test_memory_storage_implements_storage_backend():
    storage = InMemoryStorage()

    assert isinstance(storage, StorageBackend)


def test_memory_storage_saves_event():
    storage = InMemoryStorage()
    event = create_event()

    storage.save(event)

    assert storage.get(event.event_id) == event


def test_memory_storage_get_missing_event():
    storage = InMemoryStorage()

    result = storage.get("does-not-exist")

    assert result is None


def test_memory_storage_updates_event():
    storage = InMemoryStorage()
    event = create_event()

    storage.save(event)

    event.status = EventStatus.SUCCESS

    storage.update(event)

    stored_event = storage.get(event.event_id)

    assert stored_event is not None
    assert stored_event.status == EventStatus.SUCCESS


def test_memory_storage_rejects_duplicate_event():
    storage = InMemoryStorage()
    event = create_event()

    storage.save(event)

    with pytest.raises(ValueError):
        storage.save(event)


def test_memory_storage_rejects_update_of_missing_event():
    storage = InMemoryStorage()
    event = create_event()

    with pytest.raises(KeyError):
        storage.update(event)


def test_memory_storage_filters_by_run():
    storage = InMemoryStorage()

    event_1 = create_event(run_id="run-001")
    event_2 = create_event(run_id="run-001")
    event_3 = create_event(run_id="run-002")

    storage.save(event_1)
    storage.save(event_2)
    storage.save(event_3)

    events = storage.get_by_run("run-001")

    assert len(events) == 2
    assert event_1 in events
    assert event_2 in events
    assert event_3 not in events


def test_memory_storage_returns_all_events():
    storage = InMemoryStorage()

    event_1 = create_event()
    event_2 = create_event()

    storage.save(event_1)
    storage.save(event_2)

    events = storage.get_all()

    assert len(events) == 2
    assert events[0] == event_1
    assert events[1] == event_2


def test_memory_storage_clear():
    storage = InMemoryStorage()

    storage.save(create_event())
    storage.save(create_event())

    storage.clear()

    assert storage.get_all() == []


def test_tracer_can_use_memory_storage():
    storage = InMemoryStorage()
    tracer = Tracer(storage=storage)

    with trace_context(tracer=tracer):
        event = tracer.start_event(
            event_type=EventType.FUNCTION_CALL,
            name="demo",
        )

        tracer.complete_event(
            event,
            output_data={"result": "success"},
            duration_ms=12.5,
        )

    stored_event = storage.get(event.event_id)

    assert stored_event is not None
    assert stored_event.status == EventStatus.SUCCESS
    assert stored_event.output_data == {"result": "success"}
    assert stored_event.duration_ms == 12.5


# ============================================================================
# SQLiteStorage Tests
# ============================================================================


def test_sqlite_storage_implements_storage_backend(tmp_path: Path):
    database_path = tmp_path / "test.db"

    storage = SQLiteStorage(database_path)

    try:
        assert isinstance(storage, StorageBackend)
    finally:
        storage.close()


def test_sqlite_storage_creates_database(tmp_path: Path):
    database_path = tmp_path / "nested" / "traces.db"

    storage = SQLiteStorage(database_path)

    try:
        assert database_path.exists()
    finally:
        storage.close()


def test_sqlite_storage_saves_and_gets_event(tmp_path: Path):
    database_path = tmp_path / "traces.db"

    storage = SQLiteStorage(database_path)
    event = create_event()

    try:
        storage.save(event)

        stored_event = storage.get(event.event_id)

        assert stored_event == event
    finally:
        storage.close()


def test_sqlite_storage_get_missing_event(tmp_path: Path):
    database_path = tmp_path / "traces.db"

    storage = SQLiteStorage(database_path)

    try:
        assert storage.get("does-not-exist") is None
    finally:
        storage.close()


def test_sqlite_storage_updates_event(tmp_path: Path):
    database_path = tmp_path / "traces.db"

    storage = SQLiteStorage(database_path)
    event = create_event()

    try:
        storage.save(event)

        event.status = EventStatus.SUCCESS
        event.output_data = {
            "answer": 42,
        }
        event.duration_ms = 25.5

        storage.update(event)

        stored_event = storage.get(event.event_id)

        assert stored_event is not None
        assert stored_event.status == EventStatus.SUCCESS
        assert stored_event.output_data == {
            "answer": 42,
        }
        assert stored_event.duration_ms == 25.5
    finally:
        storage.close()


def test_sqlite_storage_rejects_duplicate_event(tmp_path: Path):
    database_path = tmp_path / "traces.db"

    storage = SQLiteStorage(database_path)
    event = create_event()

    try:
        storage.save(event)

        with pytest.raises(ValueError):
            storage.save(event)
    finally:
        storage.close()


def test_sqlite_storage_rejects_missing_update(tmp_path: Path):
    database_path = tmp_path / "traces.db"

    storage = SQLiteStorage(database_path)

    try:
        with pytest.raises(KeyError):
            storage.update(create_event())
    finally:
        storage.close()


def test_sqlite_storage_filters_by_run(tmp_path: Path):
    database_path = tmp_path / "traces.db"

    storage = SQLiteStorage(database_path)

    event_1 = create_event(run_id="run-001")
    event_2 = create_event(run_id="run-001")
    event_3 = create_event(run_id="run-002")

    try:
        storage.save(event_1)
        storage.save(event_2)
        storage.save(event_3)

        events = storage.get_by_run("run-001")

        assert len(events) == 2
        assert events[0] == event_1
        assert events[1] == event_2
        assert event_3 not in events
    finally:
        storage.close()


def test_sqlite_storage_returns_all_events_in_order(tmp_path: Path):
    database_path = tmp_path / "traces.db"

    storage = SQLiteStorage(database_path)

    event_1 = create_event(name="first")
    event_2 = create_event(name="second")
    event_3 = create_event(name="third")

    try:
        storage.save(event_1)
        storage.save(event_2)
        storage.save(event_3)

        events = storage.get_all()

        assert events == [
            event_1,
            event_2,
            event_3,
        ]
    finally:
        storage.close()


def test_sqlite_storage_clear(tmp_path: Path):
    database_path = tmp_path / "traces.db"

    storage = SQLiteStorage(database_path)

    try:
        storage.save(create_event())
        storage.save(create_event())

        storage.clear()

        assert storage.get_all() == []
    finally:
        storage.close()


def test_sqlite_storage_preserves_structured_data(tmp_path: Path):
    database_path = tmp_path / "traces.db"

    storage = SQLiteStorage(database_path)

    event = TraceEvent(
        run_id="run-structured",
        event_type=EventType.LLM_CALL,
        status=EventStatus.SUCCESS,
        name="chat_completion",
        input_data={
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
            "temperature": 0.2,
        },
        output_data={
            "content": "Hello! How can I help?",
            "finish_reason": "stop",
        },
        provider="groq",
        model="example-model",
        input_tokens=15,
        output_tokens=8,
        total_tokens=23,
        estimated_cost=0.000123,
        metadata={
            "environment": "test",
            "tags": [
                "llm",
                "demo",
            ],
        },
    )

    try:
        storage.save(event)

        stored_event = storage.get(event.event_id)

        assert stored_event is not None
        assert stored_event.input_data == event.input_data
        assert stored_event.output_data == event.output_data
        assert stored_event.provider == "groq"
        assert stored_event.model == "example-model"
        assert stored_event.input_tokens == 15
        assert stored_event.output_tokens == 8
        assert stored_event.total_tokens == 23
        assert stored_event.estimated_cost == 0.000123
        assert stored_event.metadata == event.metadata
    finally:
        storage.close()


def test_sqlite_storage_handles_non_json_objects(tmp_path: Path):
    database_path = tmp_path / "traces.db"

    storage = SQLiteStorage(database_path)

    event = TraceEvent(
        run_id="run-object",
        event_type=EventType.FUNCTION_CALL,
        input_data={
            "object": object(),
            "set_data": {1, 2, 3},
        },
    )

    try:
        storage.save(event)

        stored_event = storage.get(event.event_id)

        assert stored_event is not None
        assert isinstance(
            stored_event.input_data["object"],
            str,
        )
        assert sorted(
            stored_event.input_data["set_data"]
        ) == [1, 2, 3]
    finally:
        storage.close()


def test_sqlite_storage_persists_across_instances(tmp_path: Path):
    database_path = tmp_path / "traces.db"

    event = create_event(
        run_id="persistent-run",
        name="persistent_function",
    )

    storage_1 = SQLiteStorage(database_path)

    try:
        storage_1.save(event)
    finally:
        storage_1.close()

    storage_2 = SQLiteStorage(database_path)

    try:
        stored_event = storage_2.get(event.event_id)

        assert stored_event == event
    finally:
        storage_2.close()


def test_tracer_can_use_sqlite_storage(tmp_path: Path):
    database_path = tmp_path / "traces.db"

    storage = SQLiteStorage(database_path)
    tracer = Tracer(storage=storage)

    try:
        with trace_context(
            run_id="sqlite-run",
            tracer=tracer,
        ):
            event = tracer.start_event(
                event_type=EventType.FUNCTION_CALL,
                name="sqlite_demo",
                input_data={
                    "value": 10,
                },
            )

            tracer.complete_event(
                event,
                output_data={
                    "value": 20,
                },
                duration_ms=10.5,
            )

        stored_event = storage.get(event.event_id)

        assert stored_event is not None
        assert stored_event.run_id == "sqlite-run"
        assert stored_event.status == EventStatus.SUCCESS
        assert stored_event.input_data == {
            "value": 10,
        }
        assert stored_event.output_data == {
            "value": 20,
        }
    finally:
        storage.close()


def test_sqlite_storage_context_manager(tmp_path: Path):
    database_path = tmp_path / "traces.db"

    with SQLiteStorage(database_path) as storage:
        event = create_event()

        storage.save(event)

        assert storage.get(event.event_id) == event


# ============================================================================
# JSONLStorage Tests
# ============================================================================


def test_jsonl_storage_implements_storage_backend(tmp_path: Path):
    file_path = tmp_path / "traces.jsonl"

    storage = JSONLStorage(file_path)

    assert isinstance(storage, StorageBackend)


def test_jsonl_storage_creates_file_on_first_save(tmp_path: Path):
    file_path = tmp_path / "nested" / "traces.jsonl"

    storage = JSONLStorage(file_path)
    event = create_event()

    assert not file_path.exists()

    storage.save(event)

    assert file_path.exists()


def test_jsonl_storage_saves_and_gets_event(tmp_path: Path):
    file_path = tmp_path / "traces.jsonl"

    storage = JSONLStorage(file_path)
    event = create_event()

    storage.save(event)

    stored_event = storage.get(event.event_id)

    assert stored_event == event


def test_jsonl_storage_get_missing_event(tmp_path: Path):
    file_path = tmp_path / "traces.jsonl"

    storage = JSONLStorage(file_path)

    assert storage.get("does-not-exist") is None


def test_jsonl_storage_updates_event(tmp_path: Path):
    file_path = tmp_path / "traces.jsonl"

    storage = JSONLStorage(file_path)
    event = create_event()

    storage.save(event)

    event.status = EventStatus.SUCCESS
    event.output_data = {
        "answer": 42,
    }
    event.duration_ms = 17.5

    storage.update(event)

    stored_event = storage.get(event.event_id)

    assert stored_event is not None
    assert stored_event.status == EventStatus.SUCCESS
    assert stored_event.output_data == {
        "answer": 42,
    }
    assert stored_event.duration_ms == 17.5


def test_jsonl_storage_rejects_duplicate_event(tmp_path: Path):
    file_path = tmp_path / "traces.jsonl"

    storage = JSONLStorage(file_path)
    event = create_event()

    storage.save(event)

    with pytest.raises(ValueError):
        storage.save(event)


def test_jsonl_storage_rejects_missing_update(tmp_path: Path):
    file_path = tmp_path / "traces.jsonl"

    storage = JSONLStorage(file_path)

    with pytest.raises(KeyError):
        storage.update(create_event())


def test_jsonl_storage_filters_by_run(tmp_path: Path):
    file_path = tmp_path / "traces.jsonl"

    storage = JSONLStorage(file_path)

    event_1 = create_event(run_id="run-001")
    event_2 = create_event(run_id="run-001")
    event_3 = create_event(run_id="run-002")

    storage.save(event_1)
    storage.save(event_2)
    storage.save(event_3)

    events = storage.get_by_run("run-001")

    assert len(events) == 2
    assert events[0] == event_1
    assert events[1] == event_2
    assert event_3 not in events


def test_jsonl_storage_returns_events_in_order(tmp_path: Path):
    file_path = tmp_path / "traces.jsonl"

    storage = JSONLStorage(file_path)

    event_1 = create_event(name="first")
    event_2 = create_event(name="second")
    event_3 = create_event(name="third")

    storage.save(event_1)
    storage.save(event_2)
    storage.save(event_3)

    events = storage.get_all()

    assert events == [
        event_1,
        event_2,
        event_3,
    ]


def test_jsonl_storage_update_does_not_change_event_order(
    tmp_path: Path,
):
    file_path = tmp_path / "traces.jsonl"

    storage = JSONLStorage(file_path)

    event_1 = create_event(name="first")
    event_2 = create_event(name="second")

    storage.save(event_1)
    storage.save(event_2)

    event_1.status = EventStatus.SUCCESS

    storage.update(event_1)

    events = storage.get_all()

    assert events[0].event_id == event_1.event_id
    assert events[0].status == EventStatus.SUCCESS
    assert events[1].event_id == event_2.event_id


def test_jsonl_storage_clear(tmp_path: Path):
    file_path = tmp_path / "traces.jsonl"

    storage = JSONLStorage(file_path)

    storage.save(create_event())
    storage.save(create_event())

    storage.clear()

    assert storage.get_all() == []
    assert file_path.exists()
    assert file_path.read_text(encoding="utf-8") == ""


def test_jsonl_storage_preserves_structured_data(
    tmp_path: Path,
):
    file_path = tmp_path / "traces.jsonl"

    storage = JSONLStorage(file_path)

    event = TraceEvent(
        run_id="run-structured",
        event_type=EventType.LLM_CALL,
        status=EventStatus.SUCCESS,
        name="chat_completion",
        input_data={
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
            "temperature": 0.2,
        },
        output_data={
            "content": "Hello! How can I help?",
            "finish_reason": "stop",
        },
        provider="groq",
        model="example-model",
        input_tokens=15,
        output_tokens=8,
        total_tokens=23,
        estimated_cost=0.000123,
        metadata={
            "environment": "test",
            "tags": [
                "llm",
                "demo",
            ],
        },
    )

    storage.save(event)

    stored_event = storage.get(event.event_id)

    assert stored_event is not None
    assert stored_event.input_data == event.input_data
    assert stored_event.output_data == event.output_data
    assert stored_event.provider == "groq"
    assert stored_event.model == "example-model"
    assert stored_event.input_tokens == 15
    assert stored_event.output_tokens == 8
    assert stored_event.total_tokens == 23
    assert stored_event.estimated_cost == 0.000123
    assert stored_event.metadata == event.metadata


def test_jsonl_storage_handles_non_json_objects(
    tmp_path: Path,
):
    file_path = tmp_path / "traces.jsonl"

    storage = JSONLStorage(file_path)

    event = TraceEvent(
        run_id="run-object",
        event_type=EventType.FUNCTION_CALL,
        input_data={
            "object": object(),
            "set_data": {1, 2, 3},
        },
    )

    storage.save(event)

    stored_event = storage.get(event.event_id)

    assert stored_event is not None
    assert isinstance(
        stored_event.input_data["object"],
        str,
    )
    assert sorted(
        stored_event.input_data["set_data"]
    ) == [1, 2, 3]


def test_jsonl_storage_persists_across_instances(
    tmp_path: Path,
):
    file_path = tmp_path / "traces.jsonl"

    event = create_event(
        run_id="persistent-run",
        name="persistent_function",
    )

    storage_1 = JSONLStorage(file_path)
    storage_1.save(event)
    storage_1.close()

    storage_2 = JSONLStorage(file_path)

    try:
        stored_event = storage_2.get(event.event_id)

        assert stored_event == event
    finally:
        storage_2.close()


def test_jsonl_storage_rebuilds_latest_updated_state(
    tmp_path: Path,
):
    file_path = tmp_path / "traces.jsonl"

    event = create_event()

    storage_1 = JSONLStorage(file_path)

    storage_1.save(event)

    event.status = EventStatus.SUCCESS
    event.output_data = {
        "result": "completed",
    }

    storage_1.update(event)
    storage_1.close()

    storage_2 = JSONLStorage(file_path)

    try:
        stored_event = storage_2.get(event.event_id)

        assert stored_event is not None
        assert stored_event.status == EventStatus.SUCCESS
        assert stored_event.output_data == {
            "result": "completed",
        }
    finally:
        storage_2.close()


def test_jsonl_storage_update_appends_record(
    tmp_path: Path,
):
    file_path = tmp_path / "traces.jsonl"

    storage = JSONLStorage(file_path)
    event = create_event()

    storage.save(event)

    first_line_count = len(
        file_path.read_text(
            encoding="utf-8",
        ).splitlines()
    )

    event.status = EventStatus.SUCCESS

    storage.update(event)

    second_line_count = len(
        file_path.read_text(
            encoding="utf-8",
        ).splitlines()
    )

    assert first_line_count == 1
    assert second_line_count == 2


def test_tracer_can_use_jsonl_storage(tmp_path: Path):
    file_path = tmp_path / "traces.jsonl"

    storage = JSONLStorage(file_path)
    tracer = Tracer(storage=storage)

    with trace_context(
        run_id="jsonl-run",
        tracer=tracer,
    ):
        event = tracer.start_event(
            event_type=EventType.FUNCTION_CALL,
            name="jsonl_demo",
            input_data={
                "value": 10,
            },
        )

        tracer.complete_event(
            event,
            output_data={
                "value": 20,
            },
            duration_ms=11.5,
        )

    stored_event = storage.get(event.event_id)

    assert stored_event is not None
    assert stored_event.run_id == "jsonl-run"
    assert stored_event.status == EventStatus.SUCCESS
    assert stored_event.input_data == {
        "value": 10,
    }
    assert stored_event.output_data == {
        "value": 20,
    }


def test_jsonl_storage_context_manager(tmp_path: Path):
    file_path = tmp_path / "traces.jsonl"

    with JSONLStorage(file_path) as storage:
        event = create_event()

        storage.save(event)

        assert storage.get(event.event_id) == event