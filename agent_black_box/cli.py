"""
Command-line interface for Agent Black Box.

The CLI provides a small operational interface for inspecting
persisted agent traces without coupling the CLI to the tracing
implementation.

Supported commands:

    blackbox version
    blackbox config
    blackbox runs
    blackbox events <run_id>
    blackbox inspect <run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

from agent_black_box.storage.sqlite import SQLiteStorage


VERSION = "0.1.0"
APPLICATION_NAME = "Agent Black Box"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """
    Build the Agent Black Box CLI argument parser.

    Returns:
        argparse.ArgumentParser:
            Configured command-line parser.
    """

    parser = argparse.ArgumentParser(
        prog="blackbox",
        description=(
            "Agent Black Box observability CLI for inspecting "
            "AI-agent execution traces."
        ),
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/traces.db"),
        help=(
            "Path to the SQLite trace database. "
            "Default: data/traces.db"
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ------------------------------------------------------------------
    # version
    # ------------------------------------------------------------------

    subparsers.add_parser(
        "version",
        help="Show the Agent Black Box version.",
    )

    # ------------------------------------------------------------------
    # config
    # ------------------------------------------------------------------

    subparsers.add_parser(
        "config",
        help="Show the active CLI configuration.",
    )

    # ------------------------------------------------------------------
    # runs
    # ------------------------------------------------------------------

    runs_parser = subparsers.add_parser(
        "runs",
        help="List traced agent runs.",
    )

    runs_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of runs to display.",
    )

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------

    events_parser = subparsers.add_parser(
        "events",
        help="List events belonging to a run.",
    )

    events_parser.add_argument(
        "run_id",
        help="Run identifier.",
    )

    # ------------------------------------------------------------------
    # inspect
    # ------------------------------------------------------------------

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Return a JSON summary of a run.",
    )

    inspect_parser.add_argument(
        "run_id",
        help="Run identifier.",
    )

    return parser


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _open_storage(
    database: Path,
) -> SQLiteStorage:
    """
    Open the SQLite trace store.

    The storage API uses ``database_path``.
    """

    return SQLiteStorage(
        database_path=database
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _command_version() -> int:
    """Print the application version."""

    print(
        f"{APPLICATION_NAME} v{VERSION}"
    )

    return 0


def _command_config(
    database: Path,
) -> int:
    """
    Print the active CLI configuration.

    Human-readable output is intentional so Windows paths are shown
    without JSON escaping.
    """

    print(
        f"database: {database}"
    )

    print(
        f"database_exists: {database.exists()}"
    )

    return 0


def _command_runs(
    database: Path,
    limit: int,
) -> int:
    """
    List persisted agent runs.

    Run information is reconstructed from the event stream because
    the current storage contract stores TraceEvent objects rather
    than a separate run table.
    """

    if limit <= 0:
        print(
            "No traced runs found."
        )

        return 0

    storage = _open_storage(
        database
    )

    try:
        events = storage.get_all()
    finally:
        storage.close()

    if not events:
        print(
            "No traced runs found."
        )

        return 0

    runs: OrderedDict[str, dict] = OrderedDict()

    for event in events:
        if event.run_id not in runs:
            runs[event.run_id] = {
                "run_id": event.run_id,
                "name": None,
                "status": None,
                "started_at": None,
                "event_count": 0,
            }

        run = runs[event.run_id]

        run["event_count"] += 1

        if event.event_type == "run_start":
            run["name"] = event.name
            run["started_at"] = event.timestamp.isoformat()

        elif event.event_type == "run_end":
            run["status"] = event.status

    selected_runs = list(
        runs.values()
    )[:limit]

    for run in selected_runs:
        print(
            f"{run['run_id']} | "
            f"{run['name'] or '-'} | "
            f"{run['status'] or '-'} | "
            f"{run['event_count']} events"
        )

    return 0


def _command_events(
    database: Path,
    run_id: str,
) -> int:
    """Print all events belonging to a run."""

    storage = _open_storage(
        database
    )

    try:
        events = storage.get_by_run(
            run_id
        )
    finally:
        storage.close()

    if not events:
        print(
            f"No events found for run '{run_id}'."
        )

        return 1

    for event in events:
        print(
            f"{event.timestamp.isoformat()} | "
            f"{event.event_type} | "
            f"{event.status} | "
            f"{event.name or '-'}"
        )

    return 0


def _command_inspect(
    database: Path,
    run_id: str,
) -> int:
    """
    Print a structured JSON summary of a run.
    """

    storage = _open_storage(
        database
    )

    try:
        events = storage.get_by_run(
            run_id
        )
    finally:
        storage.close()

    if not events:
        print(
            f"No events found for run '{run_id}'.",
            file=sys.stderr,
        )

        return 1

    run_start = next(
        (
            event
            for event in events
            if event.event_type == "run_start"
        ),
        None,
    )

    run_end = next(
        (
            event
            for event in reversed(events)
            if event.event_type == "run_end"
        ),
        None,
    )

    function_names = [
        event.name
        for event in events
        if (
            event.event_type == "function_call"
            and event.name
        )
    ]

    result = {
        "run_id": run_id,
        "name": (
            run_start.name
            if run_start is not None
            else None
        ),
        "status": (
            run_end.status
            if run_end is not None
            else None
        ),
        "event_count": len(events),
        "events": [
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "status": event.status,
                "name": event.name,
                "timestamp": event.timestamp.isoformat(),
                "duration_ms": event.duration_ms,
            }
            for event in events
        ],
        "function_names": function_names,
    }

    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )

    return 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(
    argv: Sequence[str] | None = None,
) -> int:
    """
    Execute the Agent Black Box CLI.

    Args:
        argv:
            Optional argument sequence. If omitted, argparse reads
            from sys.argv.

    Returns:
        int:
            Process exit code.
    """

    parser = build_parser()

    try:
        args = parser.parse_args(
            argv
        )

        if args.command == "version":
            return _command_version()

        if args.command == "config":
            return _command_config(
                args.database
            )

        if args.command == "runs":
            return _command_runs(
                args.database,
                args.limit,
            )

        if args.command == "events":
            return _command_events(
                args.database,
                args.run_id,
            )

        if args.command == "inspect":
            return _command_inspect(
                args.database,
                args.run_id,
            )

        parser.error(
            "No command specified."
        )

    except KeyboardInterrupt:
        print(
            "Interrupted.",
            file=sys.stderr,
        )

        return 130

    except Exception as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )