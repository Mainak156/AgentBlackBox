"""
Security and payload redaction for Agent Black Box.

The redaction layer creates a safety boundary between application
data and trace persistence.

It supports:

- Sensitive dictionary keys
- Nested dictionaries
- Lists, tuples, and sets
- Common credential/secret patterns
- Authorization headers
- Oversized payload protection
- Pydantic models
- Dataclasses
- Non-JSON-compatible objects
- Configurable redaction enablement

The redactor never mutates the original input object.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel


REDACTED_VALUE = "[REDACTED]"
TRUNCATED_VALUE = "[TRUNCATED]"


# These are fields whose values themselves should be considered
# sensitive. Container/grouping keys such as "credentials" are
# intentionally excluded so that their nested secret fields can
# still be inspected and individually redacted.
DEFAULT_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "accesstoken",
        "auth_token",
        "authtoken",
        "authorization",
        "bearer_token",
        "client_secret",
        "password",
        "passwd",
        "private_key",
        "secret",
        "secret_key",
        "session_token",
        "token",
    }
)


# Common credential formats embedded inside strings.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"
    ),
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|"
        r"auth[_-]?token|client[_-]?secret|"
        r"secret[_-]?key|password)\s*[:=]\s*"
        r"[^\s,;]+"
    ),
    re.compile(
        r"\bsk-[A-Za-z0-9_-]{12,}\b"
    ),
    re.compile(
        r"\bgsk_[A-Za-z0-9_-]{12,}\b"
    ),
)


class Redactor:
    """
    Recursively sanitize values before they are recorded.

    Args:
        enabled:
            Enable or disable redaction.

        max_payload_size:
            Maximum serialized payload size in bytes.

        sensitive_keys:
            Optional custom set of keys that should always be
            redacted.

    Example:

        redactor = Redactor()

        safe = redactor.redact(
            {
                "username": "mainak",
                "password": "secret",
            }
        )
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_payload_size: int = 100_000,
        sensitive_keys: set[str]
        | frozenset[str]
        | None = None,
    ) -> None:
        if max_payload_size < 1:
            raise ValueError(
                "max_payload_size must be greater than zero."
            )

        self.enabled = enabled
        self.max_payload_size = max_payload_size

        self.sensitive_keys = {
            key.strip().lower()
            for key in (
                sensitive_keys
                if sensitive_keys is not None
                else DEFAULT_SENSITIVE_KEYS
            )
        }

    def redact(self, value: Any) -> Any:
        """
        Return a sanitized copy of a value.

        The original object is never modified.
        """

        if not self.enabled:
            return value

        sanitized = self._redact_value(value)

        return self._enforce_size_limit(
            sanitized
        )

    def _redact_value(
        self,
        value: Any,
        *,
        key_context: str | None = None,
    ) -> Any:
        """
        Recursively sanitize a value.
        """

        # Sensitive keys represent secret-bearing values.
        #
        # This check intentionally happens before recursive traversal.
        # Therefore a field such as:
        #
        #     {"password": "secret"}
        #
        # becomes:
        #
        #     {"password": "[REDACTED]"}
        #
        # Container keys such as "credentials" are deliberately not
        # included in DEFAULT_SENSITIVE_KEYS, allowing their children
        # to be inspected recursively.

        if key_context is not None:
            normalized_key = (
                key_context.strip().lower()
            )

            if (
                normalized_key
                in self.sensitive_keys
            ):
                return REDACTED_VALUE

        if isinstance(value, str):
            return self._redact_string(value)

        if value is None or isinstance(
            value,
            (bool, int, float),
        ):
            return value

        if isinstance(value, Mapping):
            return {
                key: self._redact_value(
                    item,
                    key_context=str(key),
                )
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                self._redact_value(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return tuple(
                self._redact_value(item)
                for item in value
            )

        if isinstance(value, set):
            return {
                self._redact_value(item)
                for item in value
            }

        if isinstance(value, BaseModel):
            return self._redact_value(
                value.model_dump(
                    mode="python"
                )
            )

        if is_dataclass(value):
            return self._redact_value(
                asdict(value)
            )

        if isinstance(value, Enum):
            return self._redact_value(
                value.value
            )

        if isinstance(value, bytes):
            return self._redact_string(
                value.decode(
                    "utf-8",
                    errors="replace",
                )
            )

        # Arbitrary objects are converted to a safe string
        # representation rather than being retained directly.
        return self._redact_string(
            repr(value)
        )

    def _redact_string(
        self,
        value: str,
    ) -> str:
        """
        Redact known credential patterns embedded in strings.
        """

        sanitized = value

        for pattern in _SECRET_PATTERNS:
            sanitized = pattern.sub(
                self._replace_secret_match,
                sanitized,
            )

        return sanitized

    @staticmethod
    def _replace_secret_match(
        match: re.Match[str],
    ) -> str:
        """
        Replace a detected secret pattern.
        """

        text = match.group(0)

        if text.lower().startswith(
            "bearer "
        ):
            return (
                "Bearer "
                + REDACTED_VALUE
            )

        if "=" in text:
            prefix, _ = text.split(
                "=",
                1,
            )
            return (
                f"{prefix}={REDACTED_VALUE}"
            )

        if ":" in text:
            prefix, _ = text.split(
                ":",
                1,
            )
            return (
                f"{prefix}:{REDACTED_VALUE}"
            )

        return REDACTED_VALUE

    def _enforce_size_limit(
        self,
        value: Any,
    ) -> Any:
        """
        Ensure the final serialized representation does not exceed
        max_payload_size.

        Oversized structures are replaced with a structured
        truncation marker instead of arbitrarily cutting JSON.
        """

        try:
            serialized = json.dumps(
                value,
                default=self._json_default,
                ensure_ascii=False,
            )
        except (
            TypeError,
            ValueError,
        ):
            serialized = repr(value)

        size = len(
            serialized.encode("utf-8")
        )

        if size <= self.max_payload_size:
            return value

        return {
            "_redacted": True,
            "_reason": "payload_too_large",
            "_original_size_bytes": size,
            "value": TRUNCATED_VALUE,
        }

    @staticmethod
    def _json_default(
        value: Any,
    ) -> Any:
        """
        Convert values into JSON-compatible representations
        for size calculation.
        """

        if isinstance(value, bytes):
            return value.decode(
                "utf-8",
                errors="replace",
            )

        if isinstance(value, Enum):
            return value.value

        if isinstance(value, BaseModel):
            return value.model_dump(
                mode="json"
            )

        return repr(value)