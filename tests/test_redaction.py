"""
Tests for the Agent Black Box security and redaction layer.
"""

from dataclasses import dataclass

import pytest

from agent_black_box.config import AgentBlackBoxConfig
from agent_black_box.context import trace_context
from agent_black_box.events import (
    EventStatus,
    EventType,
)
from agent_black_box.security.redaction import (
    REDACTED_VALUE,
    TRUNCATED_VALUE,
    Redactor,
)
from agent_black_box.tracer import Tracer


# ============================================================================
# Basic Redaction
# ============================================================================


def test_redacts_password():
    redactor = Redactor()

    value = {
        "username": "mainak",
        "password": "secret123",
    }

    result = redactor.redact(value)

    assert result["username"] == "mainak"
    assert result["password"] == REDACTED_VALUE


def test_redacts_api_key():
    redactor = Redactor()

    result = redactor.redact(
        {
            "api_key": "gsk_super_secret",
        }
    )

    assert result["api_key"] == REDACTED_VALUE


@pytest.mark.parametrize(
    "key",
    [
        "password",
        "passwd",
        "api_key",
        "apikey",
        "access_token",
        "auth_token",
        "authorization",
        "client_secret",
        "private_key",
        "secret",
        "secret_key",
        "session_token",
        "token",
    ],
)
def test_default_sensitive_keys(
    key,
):
    redactor = Redactor()

    result = redactor.redact(
        {
            key: "sensitive-value",
        }
    )

    assert result[key] == REDACTED_VALUE


def test_sensitive_keys_are_case_insensitive():
    redactor = Redactor()

    result = redactor.redact(
        {
            "PASSWORD": "secret",
            "Api_Key": "secret",
        }
    )

    assert result["PASSWORD"] == REDACTED_VALUE
    assert result["Api_Key"] == REDACTED_VALUE


# ============================================================================
# Nested Data
# ============================================================================


def test_redacts_nested_dictionary():
    redactor = Redactor()

    value = {
        "user": {
            "name": "Mainak",
            "credentials": {
                "password": "secret",
                "token": "abc",
            },
        }
    }

    result = redactor.redact(value)

    assert result["user"]["name"] == "Mainak"
    assert (
        result["user"]["credentials"]["password"]
        == REDACTED_VALUE
    )
    assert (
        result["user"]["credentials"]["token"]
        == REDACTED_VALUE
    )


def test_redacts_sensitive_values_inside_list():
    redactor = Redactor()

    value = [
        {
            "token": "secret-token",
        },
        {
            "name": "safe",
        },
    ]

    result = redactor.redact(value)

    assert result[0]["token"] == REDACTED_VALUE
    assert result[1]["name"] == "safe"


def test_redacts_tuple_values():
    redactor = Redactor()

    result = redactor.redact(
        (
            {
                "password": "secret",
            },
            "safe",
        )
    )

    assert result[0]["password"] == REDACTED_VALUE
    assert result[1] == "safe"


def test_redacts_set_values():
    redactor = Redactor()

    result = redactor.redact(
        {
            "safe",
            "another-safe",
        }
    )

    assert result == {
        "safe",
        "another-safe",
    }


# ============================================================================
# String Pattern Detection
# ============================================================================


def test_redacts_bearer_token():
    redactor = Redactor()

    result = redactor.redact(
        "Authorization: Bearer abc123secret"
    )

    assert "abc123secret" not in result
    assert "Bearer [REDACTED]" in result


def test_redacts_inline_api_key():
    redactor = Redactor()

    result = redactor.redact(
        "api_key=gsk_super_secret_value"
    )

    assert "gsk_super_secret_value" not in result
    assert "[REDACTED]" in result


def test_redacts_groq_style_key():
    redactor = Redactor()

    result = redactor.redact(
        "gsk_12345678901234567890"
    )

    assert result == REDACTED_VALUE


def test_preserves_normal_strings():
    redactor = Redactor()

    value = "This is a completely normal message."

    assert redactor.redact(value) == value


# ============================================================================
# Custom Configuration
# ============================================================================


def test_custom_sensitive_key():
    redactor = Redactor(
        sensitive_keys={
            "employee_id",
        }
    )

    result = redactor.redact(
        {
            "employee_id": "12345",
            "username": "mainak",
        }
    )

    assert result["employee_id"] == REDACTED_VALUE
    assert result["username"] == "mainak"


def test_disabled_redaction_returns_original_object():
    redactor = Redactor(
        enabled=False,
    )

    value = {
        "password": "secret",
    }

    result = redactor.redact(value)

    assert result is value
    assert result["password"] == "secret"


# ============================================================================
# Immutability
# ============================================================================


def test_redaction_does_not_mutate_original():
    redactor = Redactor()

    original = {
        "password": "secret",
        "nested": {
            "token": "secret-token",
        },
    }

    result = redactor.redact(original)

    assert original["password"] == "secret"
    assert (
        original["nested"]["token"]
        == "secret-token"
    )

    assert result["password"] == REDACTED_VALUE
    assert (
        result["nested"]["token"]
        == REDACTED_VALUE
    )


# ============================================================================
# Payload Limits
# ============================================================================


def test_large_payload_is_truncated():
    redactor = Redactor(
        max_payload_size=100,
    )

    value = {
        "message": "x" * 1000,
    }

    result = redactor.redact(value)

    assert result["_redacted"] is True
    assert (
        result["_reason"]
        == "payload_too_large"
    )
    assert result["value"] == TRUNCATED_VALUE
    assert (
        result["_original_size_bytes"]
        > 100
    )


def test_small_payload_is_preserved():
    redactor = Redactor(
        max_payload_size=1000,
    )

    value = {
        "message": "hello",
    }

    result = redactor.redact(value)

    assert result == value


def test_invalid_payload_limit_is_rejected():
    with pytest.raises(ValueError):
        Redactor(
            max_payload_size=0,
        )


# ============================================================================
# Tracer Integration
# ============================================================================


def test_tracer_redacts_input_data():
    tracer = Tracer()

    with trace_context(
        run_id="redaction-input",
        tracer=tracer,
    ):
        event = tracer.start_event(
            event_type=EventType.FUNCTION_CALL,
            name="login",
            input_data={
                "username": "mainak",
                "password": "secret",
            },
        )

        tracer.complete_event(
            event,
            output_data={
                "status": "success",
            },
        )

    stored = tracer.get_events_for_run(
        "redaction-input"
    )[0]

    assert (
        stored.input_data["password"]
        == REDACTED_VALUE
    )
    assert (
        stored.input_data["username"]
        == "mainak"
    )


def test_tracer_redacts_output_data():
    tracer = Tracer()

    with trace_context(
        run_id="redaction-output",
        tracer=tracer,
    ):
        event = tracer.start_event(
            event_type=EventType.FUNCTION_CALL,
            name="login",
        )

        tracer.complete_event(
            event,
            output_data={
                "access_token": "secret-token",
                "status": "success",
            },
        )

    stored = tracer.get_events_for_run(
        "redaction-output"
    )[0]

    assert (
        stored.output_data["access_token"]
        == REDACTED_VALUE
    )
    assert (
        stored.output_data["status"]
        == "success"
    )


def test_tracer_redacts_error_message():
    tracer = Tracer()

    with trace_context(
        run_id="redaction-error",
        tracer=tracer,
    ):
        event = tracer.start_event(
            event_type=EventType.FUNCTION_CALL,
            name="request",
        )

        error = RuntimeError(
            "Request failed api_key=super_secret_value"
        )

        tracer.fail_event(
            event,
            error=error,
        )

    stored = tracer.get_events_for_run(
        "redaction-error"
    )[0]

    assert (
        "super_secret_value"
        not in stored.error_message
    )

    assert (
        "[REDACTED]"
        in stored.error_message
    )

    assert stored.status == EventStatus.FAILED


def test_tracer_redacts_metadata():
    tracer = Tracer()

    with trace_context(
        run_id="redaction-metadata",
        tracer=tracer,
    ):
        event = tracer.start_event(
            event_type=EventType.FUNCTION_CALL,
            name="request",
            metadata={
                "service": "payments",
                "authorization": "Bearer secret-token",
            },
        )

        tracer.complete_event(event)

    stored = tracer.get_events_for_run(
        "redaction-metadata"
    )[0]

    assert (
        stored.metadata["authorization"]
        == REDACTED_VALUE
    )
    assert (
        stored.metadata["service"]
        == "payments"
    )


def test_tracer_can_disable_redaction():
    config = AgentBlackBoxConfig(
        redact_sensitive_data=False,
    )

    tracer = Tracer(
        config=config,
    )

    with trace_context(
        run_id="redaction-disabled",
        tracer=tracer,
    ):
        event = tracer.start_event(
            event_type=EventType.FUNCTION_CALL,
            name="debug",
            input_data={
                "password": "visible",
            },
        )

        tracer.complete_event(event)

    stored = tracer.get_events_for_run(
        "redaction-disabled"
    )[0]

    assert (
        stored.input_data["password"]
        == "visible"
    )


# ============================================================================
# Dataclass Support
# ============================================================================


@dataclass
class ExamplePayload:
    username: str
    password: str


def test_dataclass_is_redacted():
    redactor = Redactor()

    result = redactor.redact(
        ExamplePayload(
            username="mainak",
            password="secret",
        )
    )

    assert result["username"] == "mainak"
    assert result["password"] == REDACTED_VALUE


# ============================================================================
# Bytes Support
# ============================================================================


def test_bytes_are_safely_converted():
    redactor = Redactor()

    result = redactor.redact(
        b"safe-content"
    )

    assert result == "safe-content"