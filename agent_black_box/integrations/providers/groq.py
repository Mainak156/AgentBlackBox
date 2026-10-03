"""
Groq LLM provider for Agent Black Box.

This module contains only Groq-specific API logic.

The rest of Agent Black Box communicates through the provider-neutral
LLMProvider and LLMResponse contracts.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from groq import Groq

from agent_black_box.integrations.llm import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
)


class GroqProvider(LLMProvider):
    """
    Groq implementation of the LLMProvider contract.

    The API key can be supplied explicitly or loaded from
    GROQ_API_KEY.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Groq | None = None,
    ) -> None:
        """
        Initialize the Groq provider.

        Args:
            api_key:
                Groq API key.

                If omitted, GROQ_API_KEY is read from the environment.

            client:
                Optional preconfigured Groq client.

                This is particularly useful for testing.
        """

        if client is not None:
            self._client = client
            return

        resolved_api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
        )

        if not resolved_api_key:
            raise ValueError(
                "Groq API key is required. "
                "Provide api_key or set GROQ_API_KEY."
            )

        self._client = Groq(
            api_key=resolved_api_key,
        )

    @property
    def provider_name(self) -> str:
        """
        Return the canonical provider name.
        """

        return "groq"

    def generate(
        self,
        *,
        messages: Sequence[LLMMessage],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response using the Groq Chat Completions API.
        """

        provider_messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]

        response = self._client.chat.completions.create(
            model=model,
            messages=provider_messages,
            **kwargs,
        )

        choice = response.choices[0]

        usage = response.usage

        input_tokens = (
            usage.prompt_tokens
            if usage is not None
            else None
        )

        output_tokens = (
            usage.completion_tokens
            if usage is not None
            else None
        )

        total_tokens = (
            usage.total_tokens
            if usage is not None
            else None
        )

        finish_reason = choice.finish_reason

        return LLMResponse(
            content=choice.message.content or "",
            provider=self.provider_name,
            model=response.model or model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            raw_response=response,
        )