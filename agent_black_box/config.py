"""
Configuration management for Agent Black Box.

This module provides a single validated configuration contract for
the tracing system.

Configuration can be supplied explicitly in Python or loaded from
environment variables.

The rest of the application should consume AgentBlackBoxConfig
rather than reading environment variables directly.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentBlackBoxConfig(BaseModel):
    """
    Application configuration for Agent Black Box.

    The configuration is intentionally small at this stage.
    Additional settings should be introduced only when a concrete
    subsystem requires them.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------

    environment: str = Field(
        default="development",
        description="Application environment.",
    )

    tracing_enabled: bool = Field(
        default=True,
        description="Whether tracing is enabled.",
    )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    storage_backend: str = Field(
        default="memory",
        description="Active trace storage backend.",
    )

    sqlite_path: Path = Field(
        default=Path("data/traces.db"),
        description="SQLite database path.",
    )

    jsonl_path: Path = Field(
        default=Path("data/traces.jsonl"),
        description="JSONL trace file path.",
    )

    # ------------------------------------------------------------------
    # Security / Observability
    # ------------------------------------------------------------------

    redact_sensitive_data: bool = Field(
        default=True,
        description=(
            "Whether sensitive values should be redacted before "
            "being persisted."
        ),
    )

    max_payload_size: int = Field(
        default=100_000,
        ge=1,
        description=(
            "Maximum serialized payload size in bytes before "
            "truncation."
        ),
    )

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    llm_provider: str = Field(
        default="groq",
        description="Default LLM provider.",
    )

    llm_model: str | None = Field(
        default=None,
        description="Default LLM model.",
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @field_validator("environment")
    @classmethod
    def validate_environment(
        cls,
        value: str,
    ) -> str:
        """
        Validate the application environment.
        """

        normalized = value.strip().lower()

        allowed = {
            "development",
            "testing",
            "staging",
            "production",
        }

        if normalized not in allowed:
            raise ValueError(
                "environment must be one of: "
                "development, testing, staging, production"
            )

        return normalized

    @field_validator("storage_backend")
    @classmethod
    def validate_storage_backend(
        cls,
        value: str,
    ) -> str:
        """
        Validate the configured storage backend.
        """

        normalized = value.strip().lower()

        allowed = {
            "memory",
            "sqlite",
            "jsonl",
        }

        if normalized not in allowed:
            raise ValueError(
                "storage_backend must be one of: "
                "memory, sqlite, jsonl"
            )

        return normalized

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(
        cls,
        value: str,
    ) -> str:
        """
        Normalize the configured LLM provider.
        """

        normalized = value.strip().lower()

        if not normalized:
            raise ValueError(
                "llm_provider cannot be empty."
            )

        return normalized

    # ------------------------------------------------------------------
    # Environment Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_environment(
        cls,
    ) -> "AgentBlackBoxConfig":
        """
        Build configuration from environment variables.

        Supported variables:

        ABB_ENVIRONMENT
        ABB_TRACING_ENABLED
        ABB_STORAGE_BACKEND
        ABB_SQLITE_PATH
        ABB_JSONL_PATH
        ABB_REDACT_SENSITIVE_DATA
        ABB_MAX_PAYLOAD_SIZE
        ABB_LLM_PROVIDER
        ABB_LLM_MODEL
        """

        return cls(
            environment=os.getenv(
                "ABB_ENVIRONMENT",
                "development",
            ),
            tracing_enabled=_parse_bool(
                os.getenv(
                    "ABB_TRACING_ENABLED",
                    "true",
                )
            ),
            storage_backend=os.getenv(
                "ABB_STORAGE_BACKEND",
                "memory",
            ),
            sqlite_path=os.getenv(
                "ABB_SQLITE_PATH",
                "data/traces.db",
            ),
            jsonl_path=os.getenv(
                "ABB_JSONL_PATH",
                "data/traces.jsonl",
            ),
            redact_sensitive_data=_parse_bool(
                os.getenv(
                    "ABB_REDACT_SENSITIVE_DATA",
                    "true",
                )
            ),
            max_payload_size=int(
                os.getenv(
                    "ABB_MAX_PAYLOAD_SIZE",
                    "100000",
                )
            ),
            llm_provider=os.getenv(
                "ABB_LLM_PROVIDER",
                "groq",
            ),
            llm_model=os.getenv(
                "ABB_LLM_MODEL",
            ),
        )


def _parse_bool(value: str) -> bool:
    """
    Parse a boolean environment variable.

    Accepted true values:

        true
        1
        yes
        on

    Accepted false values:

        false
        0
        no
        off
    """

    normalized = value.strip().lower()

    if normalized in {
        "true",
        "1",
        "yes",
        "on",
    }:
        return True

    if normalized in {
        "false",
        "0",
        "no",
        "off",
    }:
        return False

    raise ValueError(
        f"Invalid boolean value: {value!r}"
    )


def get_config() -> AgentBlackBoxConfig:
    """
    Return the application configuration loaded from the environment.
    """

    return AgentBlackBoxConfig.from_environment()