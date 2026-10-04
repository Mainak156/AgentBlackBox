"""
Agent Black Box - Streamlit Replay Viewer.

Provides a visual interface for inspecting persisted agent traces.

Run from the project root:

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Iterator

import streamlit as st

from agent_black_box.storage.sqlite import SQLiteStorage


DEFAULT_DATABASE = Path("data/traces.db")
APPLICATION_VERSION = "0.1.0"

LOGO_RESOURCE = files("agent_black_box").joinpath(
    "assets",
    "abb-logo-symbol.svg",
)


@contextmanager
def get_logo_path() -> Iterator[Path | None]:
    """
    Resolve the packaged Agent Black Box logo to a filesystem path.

    The logo is shipped as package data, so the dashboard does not depend
    on the user's local project directory structure.

    Yields:
        A temporary filesystem path to the logo, or None if unavailable.
    """
    try:
        with as_file(LOGO_RESOURCE) as logo_path:
            yield logo_path
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        yield None


def load_events(database_path: Path) -> list[Any]:
    """Load all persisted trace events from SQLite."""
    storage = SQLiteStorage(database_path=database_path)

    try:
        return storage.get_all()
    finally:
        storage.close()


def get_run_ids(events: list[Any]) -> list[str]:
    """Return unique run IDs in persisted order."""
    run_ids: list[str] = []
    seen: set[str] = set()

    for event in events:
        if event.run_id not in seen:
            seen.add(event.run_id)
            run_ids.append(event.run_id)

    return run_ids


def get_run_events(events: list[Any], run_id: str) -> list[Any]:
    """Return events belonging to one run."""
    return [
        event
        for event in events
        if event.run_id == run_id
    ]


def get_run_start(events: list[Any]) -> Any | None:
    """Return the run_start event if present."""
    return next(
        (
            event
            for event in events
            if event.event_type == "run_start"
        ),
        None,
    )


def get_run_end(events: list[Any]) -> Any | None:
    """Return the final run_end event if present."""
    for event in reversed(events):
        if event.event_type == "run_end":
            return event

    return None


def get_failed_events(events: list[Any]) -> list[Any]:
    """
    Return failed operational events in persisted order.

    RUN_END is deliberately excluded because it reports the final
    run outcome rather than identifying the operation that failed.
    """
    return [
        event
        for event in events
        if str(event.status) == "failed"
        and event.event_type != "run_end"
    ]


def get_root_cause_event(events: list[Any]) -> Any | None:
    """
    Return the first observable failed operational event.

    The first failed event is used as the root-cause candidate because
    persisted events are ordered chronologically. RUN_END is excluded
    because it is a lifecycle summary event.

    If no operational event is marked failed, an explicit ERROR event
    is used as a fallback. If neither exists, no root cause is shown.
    """
    failed_events = get_failed_events(events)

    if failed_events:
        return failed_events[0]

    return next(
        (
            event
            for event in events
            if event.event_type == "error"
        ),
        None,
    )


def calculate_run_duration(events: list[Any]) -> float | None:
    """Calculate run duration in milliseconds."""
    run_end = get_run_end(events)

    if run_end is not None and run_end.duration_ms is not None:
        return run_end.duration_ms

    run_start = get_run_start(events)

    if run_start is None or run_end is None:
        return None

    duration = (
        run_end.timestamp - run_start.timestamp
    ).total_seconds() * 1000

    return max(duration, 0.0)


def calculate_total_tokens(events: list[Any]) -> int:
    """Calculate total recorded token usage."""
    return sum(
        event.total_tokens or 0
        for event in events
    )


def calculate_total_cost(events: list[Any]) -> float:
    """Calculate total recorded estimated cost."""
    return sum(
        event.estimated_cost or 0.0
        for event in events
    )


def get_run_name(events: list[Any]) -> str:
    """Return the run name or a fallback."""
    run_start = get_run_start(events)

    if run_start is not None and run_start.name:
        return run_start.name

    return "Unnamed run"


def get_run_status(events: list[Any]) -> str:
    """Return the final run status."""
    run_end = get_run_end(events)

    if run_end is not None:
        return str(run_end.status)

    if events:
        return str(events[-1].status)

    return "unknown"


def format_duration(duration_ms: float | None) -> str:
    """Format milliseconds for display."""
    if duration_ms is None:
        return "—"

    if duration_ms < 1000:
        return f"{duration_ms:.2f} ms"

    return f"{duration_ms / 1000:.2f} s"


def format_cost(cost: float) -> str:
    """Format estimated cost."""
    if cost == 0:
        return "$0.00"

    return f"${cost:.6f}"


def event_icon(event_type: str, status: str) -> str:
    """Return a visual icon for an event."""
    if status == "failed":
        return "🔴"

    if event_type == "run_start":
        return "▶️"

    if event_type == "run_end":
        return "⏹️"

    if event_type == "llm_call":
        return "🤖"

    if event_type == "llm_response":
        return "💬"

    if event_type == "tool_call":
        return "🔧"

    if event_type == "tool_result":
        return "📦"

    if event_type == "function_call":
        return "⚙️"

    if event_type == "function_return":
        return "↩️"

    if event_type == "error":
        return "❌"

    return "●"


def event_depth(event: Any, events: list[Any]) -> int:
    """Calculate visual nesting depth from parent_event_id."""
    event_by_id = {
        item.event_id: item
        for item in events
    }

    depth = 0
    current_parent = event.parent_event_id
    visited: set[str] = set()

    while current_parent:
        if current_parent in visited:
            break

        visited.add(current_parent)

        parent = event_by_id.get(current_parent)

        if parent is None:
            break

        depth += 1
        current_parent = parent.parent_event_id

    return min(depth, 10)


def event_to_json(event: Any) -> str:
    """Serialize an event as formatted JSON."""
    payload = event.model_dump(mode="json")

    return json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


def render_header(logo_path: Path | None) -> None:
    """Render the branded Agent Black Box dashboard header."""
    logo_column, title_column = st.columns(
        [0.12, 0.88],
        vertical_alignment="center",
    )

    with logo_column:
        if logo_path is not None:
            st.image(
                str(logo_path),
                width=72,
            )

    with title_column:
        st.title("Agent Black Box")
        st.caption(
            "AI-agent observability and trace replay"
            f" · v{APPLICATION_VERSION}"
        )


def render_sidebar(logo_path: Path | None) -> Path:
    """Render sidebar controls."""
    if logo_path is not None:
        st.sidebar.image(
            str(logo_path),
            width=56,
        )

    st.sidebar.markdown(
        "### Agent Black Box"
    )

    st.sidebar.caption(
        "AI-agent observability and trace replay"
    )

    st.sidebar.divider()

    st.sidebar.header("Trace Source")

    database_text = st.sidebar.text_input(
        "SQLite database",
        value=str(DEFAULT_DATABASE),
        help="Path to the SQLite trace database.",
    )

    database_path = Path(database_text)

    st.sidebar.divider()

    st.sidebar.markdown(
        """
**Supported trace data**

- Run lifecycle
- Function calls
- Tool calls
- LLM calls
- Errors
- Timing
- Tokens
- Estimated cost
- Metadata
"""
    )

    return database_path


def render_run_summary(events: list[Any]) -> None:
    """Render summary metrics for a run."""
    status = get_run_status(events)
    duration = calculate_run_duration(events)
    tokens = calculate_total_tokens(events)
    cost = calculate_total_cost(events)

    (
        status_column,
        event_column,
        duration_column,
        token_column,
        cost_column,
    ) = st.columns(5)

    status_column.metric(
        "Status",
        status.upper(),
    )

    event_column.metric(
        "Events",
        len(events),
    )

    duration_column.metric(
        "Duration",
        format_duration(duration),
    )

    token_column.metric(
        "Tokens",
        f"{tokens:,}" if tokens else "—",
    )

    cost_column.metric(
        "Est. Cost",
        format_cost(cost),
    )


def render_run_metadata(events: list[Any]) -> None:
    """Render metadata associated with a run."""
    run_start = get_run_start(events)

    if run_start is None:
        return

    metadata = run_start.metadata or {}

    if not metadata:
        return

    with st.expander(
        "Run Metadata",
        expanded=False,
    ):
        st.json(metadata)


def render_root_cause(events: list[Any]) -> None:
    """
    Render a concise root-cause panel for failed runs.

    The panel is intentionally based on the persisted trace rather
    than trying to infer a cause from free-form model reasoning.
    """
    if get_run_status(events) != "failed":
        return

    root_cause = get_root_cause_event(events)

    st.subheader("Root Cause")

    if root_cause is None:
        st.warning(
            "The run failed, but no failed operational event "
            "was recorded."
        )
        return

    name = root_cause.name or "Unnamed event"
    event_type = str(root_cause.event_type)
    status = str(root_cause.status)

    st.error(
        f"🔴 {name} · `{event_type}` · `{status}`"
    )

    detail_columns = st.columns(3)

    detail_columns[0].metric(
        "Failed Event",
        name,
    )

    detail_columns[1].metric(
        "Event Type",
        event_type,
    )

    detail_columns[2].metric(
        "Duration",
        format_duration(root_cause.duration_ms),
    )

    if root_cause.error_type or root_cause.error_message:
        error_parts: list[str] = []

        if root_cause.error_type:
            error_parts.append(
                f"**Type:** `{root_cause.error_type}`"
            )

        if root_cause.error_message:
            error_parts.append(
                f"**Message:** {root_cause.error_message}"
            )

        st.markdown("\n\n".join(error_parts))

    st.caption(
        "Root cause candidate = first failed operational event "
        "recorded in this run."
    )


def render_timeline(events: list[Any]) -> int | None:
    """Render the execution timeline."""
    st.subheader("Execution Timeline")

    if not events:
        st.info("This run contains no events.")
        return None

    selected_index: int | None = None

    for index, event in enumerate(events):
        icon = event_icon(
            str(event.event_type),
            str(event.status),
        )

        depth = event_depth(event, events)
        indentation = "&nbsp;" * (depth * 4)

        name = event.name or "Unnamed event"
        event_type = str(event.event_type)
        status = str(event.status)
        duration = format_duration(event.duration_ms)

        label = (
            f"{indentation}{icon} **{name}**  \n"
            f"{indentation}`{event_type}` · "
            f"`{status}` · `{duration}`"
        )

        if st.button(
            label,
            key=f"event_{event.event_id}",
            use_container_width=True,
        ):
            selected_index = index

    return selected_index


def render_event_details(event: Any) -> None:
    """Render details for a selected event."""
    st.subheader("Event Details")

    status = str(event.status)

    if status == "failed":
        st.error(
            f"{event.name or 'Event'} failed"
        )

    columns = st.columns(4)

    columns[0].metric(
        "Type",
        str(event.event_type),
    )

    columns[1].metric(
        "Status",
        status,
    )

    columns[2].metric(
        "Duration",
        format_duration(event.duration_ms),
    )

    columns[3].metric(
        "Tokens",
        (
            f"{event.total_tokens:,}"
            if event.total_tokens is not None
            else "—"
        ),
    )

    if event.provider or event.model:
        provider_column, model_column = st.columns(2)

        provider_column.text_input(
            "Provider",
            value=event.provider or "—",
            disabled=True,
        )

        model_column.text_input(
            "Model",
            value=event.model or "—",
            disabled=True,
        )

    input_column, output_column = st.columns(2)

    with input_column:
        st.markdown("#### Input")

        if event.input_data is None:
            st.code(
                "No input recorded.",
                language="text",
            )
        else:
            st.json(event.input_data)

    with output_column:
        st.markdown("#### Output")

        if event.output_data is None:
            st.code(
                "No output recorded.",
                language="text",
            )
        else:
            st.json(event.output_data)

    if event.error_type or event.error_message:
        st.markdown("#### Error")

        error_parts: list[str] = []

        if event.error_type:
            error_parts.append(
                f"Type: {event.error_type}"
            )

        if event.error_message:
            error_parts.append(
                f"Message: {event.error_message}"
            )

        st.error(
            "\n\n".join(error_parts)
        )

    if event.metadata:
        with st.expander(
            "Event Metadata",
            expanded=False,
        ):
            st.json(event.metadata)

    with st.expander(
        "Raw Event JSON",
        expanded=False,
    ):
        st.code(
            event_to_json(event),
            language="json",
        )


def render_empty_state(database_path: Path) -> None:
    """Render an empty database state."""
    st.info(
        "No trace events were found in the selected database."
    )

    st.markdown(
        "### Get started\n\n"
        "Run the example from the project root:\n\n"
        "`python examples/basic_trace.py`\n\n"
        "Then change the database path in the sidebar to:\n\n"
        "`data/example_traces.db`"
    )

    if not database_path.exists():
        st.caption(
            f"Database does not exist yet: `{database_path}`"
        )


def main() -> None:
    """Run the Streamlit application."""
    with get_logo_path() as logo_path:
        st.set_page_config(
            page_title="Agent Black Box",
            page_icon=(
                str(logo_path)
                if logo_path is not None
                else "🔍"
            ),
            layout="wide",
            initial_sidebar_state="expanded",
        )

        render_header(logo_path)

        database_path = render_sidebar(logo_path)

        try:
            events = load_events(database_path)
        except Exception as exc:
            st.error(
                f"Unable to load trace database: {exc}"
            )
            return

        if not events:
            render_empty_state(database_path)
            return

        run_ids = get_run_ids(events)

        if not run_ids:
            render_empty_state(database_path)
            return

        st.sidebar.success(
            f"{len(run_ids)} traced run"
            f"{'s' if len(run_ids) != 1 else ''} found"
        )

        selected_run_id = st.sidebar.selectbox(
            "Select Run",
            options=run_ids,
            format_func=lambda run_id: (
                f"{get_run_name(get_run_events(events, run_id))} "
                f"({run_id})"
            ),
        )

        selected_events = get_run_events(
            events,
            selected_run_id,
        )

        run_name = get_run_name(selected_events)

        st.header(run_name)

        st.caption(
            f"Run ID: `{selected_run_id}`"
        )

        render_run_summary(selected_events)
        render_run_metadata(selected_events)
        render_root_cause(selected_events)

        st.divider()

        timeline_column, details_column = st.columns(
            [0.9, 1.5],
            gap="large",
        )

        with timeline_column:
            selected_index = render_timeline(
                selected_events
            )

        with details_column:
            if selected_index is None:
                root_cause = get_root_cause_event(
                    selected_events
                )

                if root_cause is not None:
                    render_event_details(root_cause)
                elif selected_events:
                    render_event_details(
                        selected_events[-1]
                    )
            else:
                render_event_details(
                    selected_events[selected_index]
                )


if __name__ == "__main__":
    main()