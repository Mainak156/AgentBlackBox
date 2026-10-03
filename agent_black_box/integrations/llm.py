"""
Provider-agnostic LLM integration for Agent Black Box.

This module defines:

- LLMMessage
- LLMResponse
- LLMProvider
- LLMClient

The LLMClient is responsible for integrating LLM execution with
the Agent Black Box tracing system.

Concrete providers such as Groq live in the providers package.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from agent_black_box.config import AgentBlackBoxConfig
from agent_black_box.context import get_current_tracer
from agent_black_box.events import EventType
from agent_black_box.tracer import Tracer


# ============================================================================
# Data Contracts
# ============================================================================


class LLMMessage(BaseModel):
    """
    Provider-neutral representation of an LLM message.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    role: str = Field(
        ...,
        min_length=1,
    )

    content: str


class LLMResponse(BaseModel):
    """
    Provider-neutral representation of an LLM response.

    Concrete providers convert their native response format into
    this structure.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    content: str

    provider: str

    model: str

    input_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    output_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    total_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    estimated_cost: float | None = Field(
        default=None,
        ge=0,
    )

    finish_reason: str | None = None

    raw_response: Any | None = None


# ============================================================================
# Provider Contract
# ============================================================================


class LLMProvider(ABC):
    """
    Abstract contract for an LLM provider.

    Implementations should perform the provider-specific API call
    and return an LLMResponse.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Return the canonical provider name.
        """

        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        *,
        messages: Sequence[LLMMessage],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate an LLM response.

        Args:
            messages:
                Provider-neutral messages.

            model:
                Provider model identifier.

            kwargs:
                Provider-specific generation parameters.

        Returns:
            LLMResponse:
                Normalized provider response.
        """

        raise NotImplementedError


# ============================================================================
# LLM Client
# ============================================================================


class LLMClient:
    """
    Provider-agnostic LLM client with optional tracing.

    The client does not know how a provider communicates with its
    backend. It only knows the LLMProvider contract.

    Example:

        provider = GroqProvider(api_key="...")
        client = LLMClient(
            provider=provider,
            model="some-model",
        )

        response = client.generate(
            messages=[
                LLMMessage(
                    role="user",
                    content="Hello!",
                )
            ]
        )
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        tracer: Tracer | None = None,
        config: AgentBlackBoxConfig | None = None,
        cost_per_1k_input_tokens: float | None = None,
        cost_per_1k_output_tokens: float | None = None,
    ) -> None:
        """
        Initialize the LLM client.

        Args:
            provider:
                Concrete LLM provider implementation.

            model:
                Model identifier.

            tracer:
                Optional explicit tracer.

                If omitted, the currently active tracer from the
                execution context is used.

            config:
                Optional Agent Black Box configuration.

            cost_per_1k_input_tokens:
                Optional input-token pricing.

            cost_per_1k_output_tokens:
                Optional output-token pricing.
        """

        if not model.strip():
            raise ValueError(
                "model cannot be empty."
            )

        self.provider = provider
        self.model = model
        self.tracer = tracer
        self.config = config or AgentBlackBoxConfig()

        self.cost_per_1k_input_tokens = (
            cost_per_1k_input_tokens
        )

        self.cost_per_1k_output_tokens = (
            cost_per_1k_output_tokens
        )

    def generate(
        self,
        *,
        messages: Sequence[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate an LLM response and record the operation.

        If tracing is disabled or no active tracer exists, the
        provider is called normally.

        Exceptions from the provider are re-raised after being
        recorded in the trace.
        """

        normalized_messages = [
            message
            if isinstance(message, LLMMessage)
            else LLMMessage.model_validate(message)
            for message in messages
        ]

        tracer = self._get_tracer()

        if not self.config.tracing_enabled or tracer is None:
            return self.provider.generate(
                messages=normalized_messages,
                model=self.model,
                **kwargs,
            )

        event = tracer.start_event(
            event_type=EventType.LLM_CALL,
            name="llm.generate",
            input_data={
                "messages": [
                    message.model_dump()
                    for message in normalized_messages
                ],
                "parameters": kwargs,
            },
            provider=self.provider.provider_name,
            model=self.model,
        )

        start_time = time.perf_counter()

        try:
            response = self.provider.generate(
                messages=normalized_messages,
                model=self.model,
                **kwargs,
            )

        except Exception as exc:
            duration_ms = (
                time.perf_counter() - start_time
            ) * 1000

            tracer.fail_event(
                event,
                error=exc,
                duration_ms=duration_ms,
            )

            raise

        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        estimated_cost = (
            response.estimated_cost
            if response.estimated_cost is not None
            else self._estimate_cost(response)
        )

        tracer.complete_event(
            event,
            output_data={
                "content": response.content,
                "finish_reason": response.finish_reason,
            },
            duration_ms=duration_ms,
            provider=response.provider,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
            estimated_cost=estimated_cost,
        )

        return response

    def _get_tracer(self) -> Tracer | None:
        """
        Resolve the tracer explicitly supplied to the client or
        inherited from the current execution context.
        """

        if self.tracer is not None:
            return self.tracer

        return get_current_tracer()

    def _estimate_cost(
        self,
        response: LLMResponse,
    ) -> float | None:
        """
        Estimate response cost when pricing information is supplied.

        If either input or output pricing is unavailable, None is
        returned rather than inventing a cost.
        """

        if (
            self.cost_per_1k_input_tokens is None
            and self.cost_per_1k_output_tokens is None
        ):
            return None

        input_cost = 0.0
        output_cost = 0.0

        if (
            self.cost_per_1k_input_tokens is not None
            and response.input_tokens is not None
        ):
            input_cost = (
                response.input_tokens / 1000
            ) * self.cost_per_1k_input_tokens

        if (
            self.cost_per_1k_output_tokens is not None
            and response.output_tokens is not None
        ):
            output_cost = (
                response.output_tokens / 1000
            ) * self.cost_per_1k_output_tokens

        return input_cost + output_cost