# Agent Black Box

A production-oriented observability and tracing toolkit for AI agents, LLM applications, tool calls, and function execution.

This repository provides a small, provider-agnostic tracing layer that records execution events, preserves nested run context, stores traces in local backends, and exposes a simple CLI and Streamlit dashboard for inspection.

## Project goal

The core idea is to capture structured execution traces for an agent or application without forcing the application to depend on a specific LLM provider or storage implementation.

```text
Application / AI Agent
        ↓
      Tracer
        ↓
    TraceEvent
        ↓
 StorageBackend
   ┌────┼─────┐
a memory SQLite JSONL
```

The project is intentionally organized around a few key concerns:

- event schema
- execution context
- tracing
- run lifecycle
- decorators
- storage
- security and redaction
- CLI inspection
- Streamlit dashboard

## Architecture at a glance

The code in this repository is built around these runtime components:

- `Tracer`: the main tracing engine
- `TraceRun`: run lifecycle manager for a complete agent execution
- `TraceEvent`: Pydantic-based event schema
- `TraceContext`: context-local state for nested execution
- `StorageBackend`: abstract storage contract
- `Redactor`: payload sanitization and size enforcement

Supported event types include:

- `run_start` / `run_end`
- `function_call` / `function_return`
- `tool_call` / `tool_result`
- `llm_call` / `llm_response`
- `error`

## Current implementation status

This repository is a focused, portfolio-oriented implementation of observability infrastructure for agent execution. It includes the core tracing system, storage backends, security redaction, a small CLI, and a local dashboard. It is not a full production deployment stack or an enterprise monitoring platform.

## Installation

This project targets Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

The package metadata declares the following dependencies:

- `pydantic`
- `python-dotenv`
- `streamlit`
- `groq`
- `httpx`
- `rich`
- `pytest` as the dev/test dependency

## Quick start

```python
from pathlib import Path

from agent_black_box import Tracer, trace, SQLiteStorage

storage = SQLiteStorage(database_path=Path("data/traces.db"))
tracer = Tracer(storage=storage)

@trace
def fetch_customer(customer_id: int) -> dict:
    return {"customer_id": customer_id, "name": "Demo Customer"}

with tracer.start_run(name="customer-agent") as run:
    result = fetch_customer(101)
    print(result)

print(tracer.get_events_for_run(run.run_id))
tracer.close()
```

The run lifecycle creates a `RUN_START` event, records nested events while the context is active, and emits a `RUN_END` event when the context exits. Exceptions raised inside the run propagate normally.

## Storage backends

The storage layer is intentionally separated from the tracing code so the tracer does not depend on a specific backend.

Implemented backends in this repo:

### In-memory storage

`InMemoryStorage`

- useful for local development and tests
- keeps trace events in memory only
- no persisted data after process exit

### SQLite storage

`SQLiteStorage`

- writes trace events to a local SQLite database
- creates the database and schema automatically
- stores event payloads as JSON
- includes indexes for run ID, timestamp, and parent-event lookups

### JSONL storage

`JSONLStorage`

- writes newline-delimited JSON records
- appends updates to the same event ID when modified
- rebuilds the latest event state from the file on load
- preserves insertion ordering while resolving latest state

## Event schema

Trace events are Pydantic models with a schema version and a consistent set of fields. The repository uses a versioned event contract with fields such as:

- `schema_version`
- `event_id`
- `run_id`
- `parent_event_id`
- `event_type`
- `status`
- `name`
- `timestamp`
- `duration_ms`
- `input_data`
- `output_data`
- `provider`
- `model`
- `input_tokens`
- `output_tokens`
- `total_tokens`
- `estimated_cost`
- `error_type`
- `error_message`
- `metadata`

## Tracing APIs

The library includes a decorator-based execution tracer:

```python
from agent_black_box import Tracer, trace

tracer = Tracer()

@trace
def do_work(value: int) -> int:
    return value * 2

with tracer.start_run(name="demo-run"):
    result = do_work(21)
    print(result)
```

The decorator records function arguments, output, timing, status, and exceptions if they occur. If no active tracing context is present, the function still executes normally without producing a trace event.

You can also create custom events directly:

```python
from agent_black_box.events import EventType

with tracer.start_run(name="tool-call-demo"):
    event = tracer.start_event(
        event_type=EventType.TOOL_CALL,
        name="pricing_calculation",
        input_data={"amount": 1000},
    )
    tracer.complete_event(
        event,
        output_data={"final_amount": 900},
        duration_ms=123.4,
    )
```

## Nested context and run lifecycle

The `TraceContext` system stores the active run and event stack in a `ContextVar`, so nested execution can propagate a current run ID and active tracer without every function needing explicit parameters.

A `TraceRun`:

1. creates a `run_start` event
2. establishes the active context for the run
3. tracks nested operations
4. records `run_end`
5. restores the parent tracing context when complete

If the application raises an exception inside the run, that exception is allowed to propagate; the run is marked as failed and the exception details are recorded on the end event.

## Security and redaction

The project includes a redaction boundary before event persistence. The redactor is designed to avoid storing sensitive values in trace payloads and to cap oversized data.

Default sensitive keys include common credential names such as:

- `password`
- `token`
- `api_key`
- `authorization`
- `client_secret`
- `private_key`
- `secret`
- `session_token`
- `access_token`

The redactor also recognizes common embedded secret patterns, including:

- bearer token strings
- `api_key=...` and `token=...` style values
- Groq-style keys such as `gsk_...`

Configuration behavior:

- redaction is enabled by default
- payload size limiting is enabled by default
- oversized values are truncated rather than stored in full
- the original object is never mutated

Default values are defined in `AgentBlackBoxConfig`:

- `redact_sensitive_data = True`
- `max_payload_size = 100_000`

This is a practical safety boundary for trace storage, but it is not presented as a full compliance or security guarantee for all environments.

## Configuration

Configuration is managed by `AgentBlackBoxConfig` and validated with Pydantic.

Supported configuration values include:

- `environment`: `development`, `testing`, `staging`, `production`
- `tracing_enabled`
- `storage_backend`: `memory`, `sqlite`, `jsonl`
- `sqlite_path`
- `jsonl_path`
- `redact_sensitive_data`
- `max_payload_size`
- `llm_provider`
- `llm_model`

Example environment variables (see `.env.example`):

```bash
ABB_ENVIRONMENT=development
ABB_TRACING_ENABLED=true
ABB_STORAGE_BACKEND=memory
ABB_SQLITE_PATH=data/traces.db
ABB_JSONL_PATH=data/traces.jsonl
ABB_REDACT_SENSITIVE_DATA=true
ABB_MAX_PAYLOAD_SIZE=100000
ABB_LLM_PROVIDER=groq
ABB_LLM_MODEL=
GROQ_API_KEY=
```

## LLM integration

The repository contains a provider-neutral LLM abstraction layer and a concrete Groq provider.

The included types are:

- `LLMMessage`
- `LLMResponse`
- `LLMProvider`
- `LLMClient`
- `GroqProvider`

The LLM client records provider calls as trace events when tracing is active and a tracer is available. The integration is intentionally abstracted behind a simple provider contract instead of hard-coding a single vendor.

## CLI

The package exposes a CLI entry point:

```bash
blackbox
```

Supported commands:

```bash
blackbox version
blackbox config
blackbox runs
blackbox events <run_id>
blackbox inspect <run_id>
```

The CLI accepts a database path override:

```bash
blackbox --database data/traces.db runs
```

Examples of CLI behavior:

- `blackbox version`: prints the library version
- `blackbox config`: prints the configured database path and existence status
- `blackbox runs`: lists traced runs with their ID, name, status, and number of events
- `blackbox events <run_id>`: prints stored events for one run
- `blackbox inspect <run_id>`: prints a JSON summary for one run

## Dashboard

The repository includes a local Streamlit application for replaying and inspecting persisted traces:

```bash
streamlit run dashboard/app.py
```

The dashboard reads from SQLite storage and presents recent run data, failure summaries, run status, token totals, cost totals, and event details.

## Example scripts

The repository includes example scripts in `examples/`:

- `examples/basic_trace.py`
- `examples/generate_demo_traces.py`

These demonstrate SQLite-backed tracing, example run lifecycle management, and event generation.

## Testing

The project uses `pytest`.

Typical test coverage includes:

- tracing lifecycle behavior
- nested run and context restoration
- redaction and size enforcement
- storage backends
- CLI behavior
- public API checks
- LLM integration behavior

## Repository layout

```text
.
├── agent_black_box/
│   ├── __init__.py
│   ├── tracer.py
│   ├── decorators.py
│   ├── context.py
│   ├── events.py
│   ├── run.py
│   ├── config.py
│   ├── cli.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── memory.py
│   │   ├── sqlite.py
│   │   └── jsonl.py
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── llm.py
│   │   └── providers/
│   │       └── groq.py
│   └── security/
│       ├── __init__.py
│       └── redaction.py
├── dashboard/
│   └── app.py
├── examples/
├── tests/
├── benchmarks/
├── docs/
├── .env.example
├── pyproject.toml
├── requirements.txt
├── README.md
└── .gitignore
```

## Notes

- This project is structured as a reusable tracing foundation for AI-agent and LLM observability, not a full end-to-end monitoring platform.
- Storage and security concerns are intentionally separated from the tracing API.
- The current implementation focuses on local persistence and inspection workflows rather than remote telemetry or cloud deployment features.
