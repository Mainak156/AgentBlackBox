"""
Agent Black Box.

A production-oriented observability and tracing toolkit
for AI agents, LLM applications, tools, and functions.
"""

from agent_black_box.context import (
    TraceContext,
    get_current_context,
    get_current_run_id,
    get_current_tracer,
    trace_context,
)
from agent_black_box.decorators import trace
from agent_black_box.events import (
    EventStatus,
    EventType,
    TraceEvent,
)
from agent_black_box.tracer import Tracer

from agent_black_box.storage.memory import InMemoryStorage
from agent_black_box.storage.sqlite import SQLiteStorage
from agent_black_box.storage.jsonl import JSONLStorage


__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Tracer",
    "trace",
    "TraceEvent",
    "EventType",
    "EventStatus",
    "TraceContext",
    "trace_context",
    "get_current_context",
    "get_current_run_id",
    "get_current_tracer",
    "InMemoryStorage",
    "SQLiteStorage",
    "JSONLStorage",
]