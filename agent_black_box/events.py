"""
Core event schemas for Agent Black Box.

This module defines the structured data contract used by the
tracing system. Every observable agent action is represented
as a TraceEvent.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Schema Version
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Event Types
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    """Types of events that can occur during an agent run."""

    RUN_START = "run_start"
    RUN_END = "run_end"

    FUNCTION_CALL = "function_call"
    FUNCTION_RETURN = "function_return"

    LLM_CALL = "llm_call"
    LLM_RESPONSE = "llm_response"

    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    ERROR = "error"


# ---------------------------------------------------------------------------
# Event Status
# ---------------------------------------------------------------------------

class EventStatus(str, Enum):
    """Execution status of a trace event."""

    STARTED = "started"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Trace Event
# ---------------------------------------------------------------------------

class TraceEvent(BaseModel):
    """
    Represents one observable event within an agent execution.

    A TraceEvent is intentionally provider-agnostic. It can represent
    events from different LLM providers, tools, functions, or agent
    frameworks.
    """

    model_config = ConfigDict(
        use_enum_values=True,
        extra="forbid",
        validate_assignment=True,
    )

    # -----------------------------------------------------------------------
    # Identity
    # -----------------------------------------------------------------------

    schema_version: int = Field(
        default=SCHEMA_VERSION,
        description="Version of the TraceEvent schema.",
    )

    event_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this event.",
    )

    run_id: str = Field(
        ...,
        description="Identifier of the complete agent execution.",
    )

    parent_event_id: str | None = Field(
        default=None,
        description="Parent event for nested execution.",
    )

    # -----------------------------------------------------------------------
    # Event Information
    # -----------------------------------------------------------------------

    event_type: EventType = Field(
        ...,
        description="Type of event being recorded.",
    )

    status: EventStatus = Field(
        default=EventStatus.STARTED,
        description="Current status of the event.",
    )

    name: str | None = Field(
        default=None,
        description="Human-readable name of the function, tool, or operation.",
    )

    # -----------------------------------------------------------------------
    # Timing
    # -----------------------------------------------------------------------

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the event was created.",
    )

    duration_ms: float | None = Field(
        default=None,
        ge=0,
        description="Execution duration in milliseconds.",
    )

    # -----------------------------------------------------------------------
    # Input / Output
    # -----------------------------------------------------------------------

    input_data: Any | None = Field(
        default=None,
        description="Input supplied to the operation.",
    )

    output_data: Any | None = Field(
        default=None,
        description="Output produced by the operation.",
    )

    # -----------------------------------------------------------------------
    # LLM Metadata
    # -----------------------------------------------------------------------

    provider: str | None = Field(
        default=None,
        description="LLM or service provider.",
    )

    model: str | None = Field(
        default=None,
        description="Model identifier.",
    )

    input_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Number of input tokens.",
    )

    output_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Number of output tokens.",
    )

    total_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Total token usage.",
    )

    estimated_cost: float | None = Field(
        default=None,
        ge=0,
        description="Estimated monetary cost of the operation.",
    )

    # -----------------------------------------------------------------------
    # Error Information
    # -----------------------------------------------------------------------

    error_type: str | None = Field(
        default=None,
        description="Exception class/type when an operation fails.",
    )

    error_message: str | None = Field(
        default=None,
        description="Human-readable error message.",
    )

    # -----------------------------------------------------------------------
    # Additional Metadata
    # -----------------------------------------------------------------------

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured metadata.",
    )