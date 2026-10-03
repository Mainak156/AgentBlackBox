"""
Tests for Agent Black Box configuration.
"""

from pathlib import Path

import pytest

from agent_black_box.config import (
    AgentBlackBoxConfig,
    get_config,
)


def test_default_configuration():
    config = AgentBlackBoxConfig()

    assert config.environment == "development"
    assert config.tracing_enabled is True
    assert config.storage_backend == "memory"
    assert config.sqlite_path == Path(
        "data/traces.db"
    )
    assert config.jsonl_path == Path(
        "data/traces.jsonl"
    )
    assert config.redact_sensitive_data is True
    assert config.max_payload_size == 100_000
    assert config.llm_provider == "groq"
    assert config.llm_model is None


def test_custom_configuration():
    config = AgentBlackBoxConfig(
        environment="production",
        tracing_enabled=False,
        storage_backend="sqlite",
        sqlite_path=Path("production/traces.db"),
        jsonl_path=Path("production/traces.jsonl"),
        redact_sensitive_data=False,
        max_payload_size=50_000,
        llm_provider="openai",
        llm_model="example-model",
    )

    assert config.environment == "production"
    assert config.tracing_enabled is False
    assert config.storage_backend == "sqlite"
    assert config.sqlite_path == Path(
        "production/traces.db"
    )
    assert config.jsonl_path == Path(
        "production/traces.jsonl"
    )
    assert config.redact_sensitive_data is False
    assert config.max_payload_size == 50_000
    assert config.llm_provider == "openai"
    assert config.llm_model == "example-model"


@pytest.mark.parametrize(
    "environment",
    [
        "development",
        "testing",
        "staging",
        "production",
    ],
)
def test_valid_environment(environment):
    config = AgentBlackBoxConfig(
        environment=environment,
    )

    assert config.environment == environment


def test_environment_is_normalized():
    config = AgentBlackBoxConfig(
        environment="  PRODUCTION  ",
    )

    assert config.environment == "production"


def test_invalid_environment():
    with pytest.raises(ValueError):
        AgentBlackBoxConfig(
            environment="invalid",
        )


@pytest.mark.parametrize(
    "storage_backend",
    [
        "memory",
        "sqlite",
        "jsonl",
    ],
)
def test_valid_storage_backend(storage_backend):
    config = AgentBlackBoxConfig(
        storage_backend=storage_backend,
    )

    assert config.storage_backend == storage_backend


def test_storage_backend_is_normalized():
    config = AgentBlackBoxConfig(
        storage_backend="  SQLITE  ",
    )

    assert config.storage_backend == "sqlite"


def test_invalid_storage_backend():
    with pytest.raises(ValueError):
        AgentBlackBoxConfig(
            storage_backend="postgres",
        )


def test_empty_llm_provider_is_rejected():
    with pytest.raises(ValueError):
        AgentBlackBoxConfig(
            llm_provider="   ",
        )


def test_negative_payload_size_is_rejected():
    with pytest.raises(ValueError):
        AgentBlackBoxConfig(
            max_payload_size=-1,
        )


def test_zero_payload_size_is_rejected():
    with pytest.raises(ValueError):
        AgentBlackBoxConfig(
            max_payload_size=0,
        )


def test_environment_configuration(
    monkeypatch,
):
    monkeypatch.setenv(
        "ABB_ENVIRONMENT",
        "production",
    )
    monkeypatch.setenv(
        "ABB_TRACING_ENABLED",
        "false",
    )
    monkeypatch.setenv(
        "ABB_STORAGE_BACKEND",
        "sqlite",
    )
    monkeypatch.setenv(
        "ABB_SQLITE_PATH",
        "custom/traces.db",
    )
    monkeypatch.setenv(
        "ABB_JSONL_PATH",
        "custom/traces.jsonl",
    )
    monkeypatch.setenv(
        "ABB_REDACT_SENSITIVE_DATA",
        "false",
    )
    monkeypatch.setenv(
        "ABB_MAX_PAYLOAD_SIZE",
        "50000",
    )
    monkeypatch.setenv(
        "ABB_LLM_PROVIDER",
        "openai",
    )
    monkeypatch.setenv(
        "ABB_LLM_MODEL",
        "test-model",
    )

    config = AgentBlackBoxConfig.from_environment()

    assert config.environment == "production"
    assert config.tracing_enabled is False
    assert config.storage_backend == "sqlite"
    assert config.sqlite_path == Path(
        "custom/traces.db"
    )
    assert config.jsonl_path == Path(
        "custom/traces.jsonl"
    )
    assert config.redact_sensitive_data is False
    assert config.max_payload_size == 50_000
    assert config.llm_provider == "openai"
    assert config.llm_model == "test-model"


def test_get_config_reads_environment(
    monkeypatch,
):
    monkeypatch.setenv(
        "ABB_STORAGE_BACKEND",
        "jsonl",
    )

    config = get_config()

    assert config.storage_backend == "jsonl"


@pytest.mark.parametrize(
    "value",
    [
        "true",
        "1",
        "yes",
        "on",
    ],
)
def test_environment_true_values(
    monkeypatch,
    value,
):
    monkeypatch.setenv(
        "ABB_TRACING_ENABLED",
        value,
    )

    config = AgentBlackBoxConfig.from_environment()

    assert config.tracing_enabled is True


@pytest.mark.parametrize(
    "value",
    [
        "false",
        "0",
        "no",
        "off",
    ],
)
def test_environment_false_values(
    monkeypatch,
    value,
):
    monkeypatch.setenv(
        "ABB_TRACING_ENABLED",
        value,
    )

    config = AgentBlackBoxConfig.from_environment()

    assert config.tracing_enabled is False


def test_invalid_environment_boolean(
    monkeypatch,
):
    monkeypatch.setenv(
        "ABB_TRACING_ENABLED",
        "maybe",
    )

    with pytest.raises(ValueError):
        AgentBlackBoxConfig.from_environment()