"""
Tests for the Agent Black Box CLI.
"""

from pathlib import Path

from agent_black_box.cli import main
from agent_black_box.events import EventType
from agent_black_box.storage.sqlite import SQLiteStorage
from agent_black_box.tracer import Tracer


def _create_trace(
    database: Path,
) -> str:
    """Create a persisted test trace."""

    storage = SQLiteStorage(
        database_path=database
    )

    tracer = Tracer(
        storage=storage
    )

    with tracer.start_run(
        run_id="cli-test-run",
        name="cli-demo-agent",
        metadata={
            "environment": "testing",
        },
    ):
        event = tracer.start_event(
            event_type=EventType.FUNCTION_CALL,
            name="demo_function",
            input_data={
                "value": 10,
            },
        )

        tracer.complete_event(
            event,
            output_data={
                "result": 20,
            },
            duration_ms=1.5,
        )

    tracer.close()

    return "cli-test-run"


def test_version_command(
    capsys,
):
    exit_code = main(
        ["version"]
    )

    captured = capsys.readouterr()

    assert exit_code == 0

    assert (
        "Agent Black Box"
        in captured.out
    )


def test_help_command(
    capsys,
):
    try:
        main(
            ["--help"]
        )
    except SystemExit as exc:
        assert exc.code == 0

    captured = capsys.readouterr()

    assert (
        "observability"
        in captured.out.lower()
    )


def test_config_command(
    tmp_path,
    capsys,
):
    database = (
        tmp_path
        / "traces.sqlite3"
    )

    exit_code = main(
        [
            "--database",
            str(database),
            "config",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0

    assert (
        str(database)
        in captured.out
    )


def test_runs_command_with_empty_database(
    tmp_path,
    capsys,
):
    database = (
        tmp_path
        / "traces.sqlite3"
    )

    exit_code = main(
        [
            "--database",
            str(database),
            "runs",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0

    assert (
        "No traced runs found."
        in captured.out
    )


def test_runs_command_lists_run(
    tmp_path,
    capsys,
):
    database = (
        tmp_path
        / "traces.sqlite3"
    )

    _create_trace(
        database
    )

    exit_code = main(
        [
            "--database",
            str(database),
            "runs",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0

    assert (
        "cli-test-run"
        in captured.out
    )

    assert (
        "cli-demo-agent"
        in captured.out
    )

    assert (
        "success"
        in captured.out
    )


def test_runs_command_respects_limit(
    tmp_path,
    capsys,
):
    database = (
        tmp_path
        / "traces.sqlite3"
    )

    storage = SQLiteStorage(
        database_path=database
    )

    tracer = Tracer(
        storage=storage
    )

    for index in range(3):
        with tracer.start_run(
            run_id=f"run-{index}",
            name=f"agent-{index}",
        ):
            pass

    tracer.close()

    exit_code = main(
        [
            "--database",
            str(database),
            "runs",
            "--limit",
            "2",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0

    assert (
        "run-0"
        in captured.out
    )

    assert (
        "run-1"
        in captured.out
    )

    assert (
        "run-2"
        not in captured.out
    )


def test_events_command_lists_run_events(
    tmp_path,
    capsys,
):
    database = (
        tmp_path
        / "traces.sqlite3"
    )

    run_id = _create_trace(
        database
    )

    exit_code = main(
        [
            "--database",
            str(database),
            "events",
            run_id,
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0

    assert (
        "run_start"
        in captured.out
    )

    assert (
        "function_call"
        in captured.out
    )

    assert (
        "run_end"
        in captured.out
    )


def test_events_command_missing_run(
    tmp_path,
    capsys,
):
    database = (
        tmp_path
        / "traces.sqlite3"
    )

    exit_code = main(
        [
            "--database",
            str(database),
            "events",
            "missing-run",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1

    assert (
        "No events found"
        in captured.out
    )


def test_inspect_command_returns_json(
    tmp_path,
    capsys,
):
    database = (
        tmp_path
        / "traces.sqlite3"
    )

    run_id = _create_trace(
        database
    )

    exit_code = main(
        [
            "--database",
            str(database),
            "inspect",
            run_id,
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0

    assert (
        '"run_id": "cli-test-run"'
        in captured.out
    )

    assert (
        '"event_count": 3'
        in captured.out
    )

    assert (
        '"demo_function"'
        in captured.out
    )


def test_inspect_command_missing_run(
    tmp_path,
    capsys,
):
    database = (
        tmp_path
        / "traces.sqlite3"
    )

    exit_code = main(
        [
            "--database",
            str(database),
            "inspect",
            "missing-run",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1

    assert (
        "No events found"
        in captured.err
    )