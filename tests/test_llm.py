"""
Tests for the provider-agnostic LLM integration.
"""

from pathlib import Path
from typing import Any

import pytest

from agent_black_box.config import AgentBlackBoxConfig
from agent_black_box.context import trace_context
from agent_black_box.events import EventStatus, EventType
from agent_black_box.integrations.llm import (
    LLMClient,
    LLMMessage,
    LLMProvider,
    LLMResponse,
)
from agent_black_box.storage.sqlite import SQLiteStorage
from agent_black_box.tracer import Tracer


class FakeLLMProvider(LLMProvider):
    """
    Deterministic fake provider used for unit tests.

    No external network calls are made.
    """

    def __init__(
        self,
        *,
        content: str = "Hello from fake provider.",
        input_tokens: int = 10,
        output_tokens: int = 5,
        should_fail: bool = False,
    ) -> None:
        self.content = content
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.should_fail = should_fail

        self.calls: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    def generate(
        self,
        *,
        messages,
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "model": model,
                "kwargs": kwargs,
            }
        )

        if self.should_fail:
            raise RuntimeError(
                "Simulated provider failure"
            )

        return LLMResponse(
            content=self.content,
            provider=self.provider_name,
            model=model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=(
                self.input_tokens
                + self.output_tokens
            ),
            finish_reason="stop",
        )


# ============================================================================
# Data Contract Tests
# ============================================================================


def test_llm_message():
    message = LLMMessage(
        role="user",
        content="Hello",
    )

    assert message.role == "user"
    assert message.content == "Hello"


def test_llm_response():
    response = LLMResponse(
        content="Hello",
        provider="fake",
        model="fake-model",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        finish_reason="stop",
    )

    assert response.content == "Hello"
    assert response.provider == "fake"
    assert response.model == "fake-model"
    assert response.input_tokens == 10
    assert response.output_tokens == 5
    assert response.total_tokens == 15
    assert response.finish_reason == "stop"


# ============================================================================
# Provider Contract Tests
# ============================================================================


def test_fake_provider_implements_contract():
    provider = FakeLLMProvider()

    assert isinstance(provider, LLMProvider)
    assert provider.provider_name == "fake"


def test_fake_provider_generates_response():
    provider = FakeLLMProvider()

    response = provider.generate(
        messages=[
            LLMMessage(
                role="user",
                content="Hello",
            )
        ],
        model="fake-model",
    )

    assert response.content == (
        "Hello from fake provider."
    )
    assert response.provider == "fake"
    assert response.model == "fake-model"


# ============================================================================
# LLMClient Tests
# ============================================================================


def test_llm_client_requires_model():
    provider = FakeLLMProvider()

    with pytest.raises(ValueError):
        LLMClient(
            provider=provider,
            model="   ",
        )


def test_llm_client_without_tracing():
    provider = FakeLLMProvider()

    config = AgentBlackBoxConfig(
        tracing_enabled=False,
    )

    client = LLMClient(
        provider=provider,
        model="fake-model",
        config=config,
    )

    response = client.generate(
        messages=[
            LLMMessage(
                role="user",
                content="Hello",
            )
        ]
    )

    assert response.content == (
        "Hello from fake provider."
    )

    assert len(provider.calls) == 1


def test_llm_client_traces_successful_call():
    provider = FakeLLMProvider()

    tracer = Tracer()

    client = LLMClient(
        provider=provider,
        model="fake-model",
        tracer=tracer,
    )

    with trace_context(
        run_id="llm-run",
        tracer=tracer,
    ):
        response = client.generate(
            messages=[
                LLMMessage(
                    role="user",
                    content="Hello",
                )
            ]
        )

    assert response.content == (
        "Hello from fake provider."
    )

    events = tracer.get_events_for_run(
        "llm-run"
    )

    assert len(events) == 1

    event = events[0]

    assert event.event_type == EventType.LLM_CALL
    assert event.status == EventStatus.SUCCESS
    assert event.provider == "fake"
    assert event.model == "fake-model"

    assert event.input_tokens == 10
    assert event.output_tokens == 5
    assert event.total_tokens == 15

    assert event.duration_ms is not None
    assert event.duration_ms >= 0

    assert event.input_data["messages"][0] == {
        "role": "user",
        "content": "Hello",
    }

    assert event.output_data == {
        "content": "Hello from fake provider.",
        "finish_reason": "stop",
    }


def test_llm_client_traces_failure():
    provider = FakeLLMProvider(
        should_fail=True,
    )

    tracer = Tracer()

    client = LLMClient(
        provider=provider,
        model="fake-model",
        tracer=tracer,
    )

    with trace_context(
        run_id="failed-llm-run",
        tracer=tracer,
    ):
        with pytest.raises(RuntimeError):
            client.generate(
                messages=[
                    LLMMessage(
                        role="user",
                        content="Hello",
                    )
                ]
            )

    events = tracer.get_events_for_run(
        "failed-llm-run"
    )

    assert len(events) == 1

    event = events[0]

    assert event.event_type == EventType.LLM_CALL
    assert event.status == EventStatus.FAILED
    assert event.error_type == "RuntimeError"
    assert event.error_message == (
        "Simulated provider failure"
    )
    assert event.duration_ms is not None


def test_llm_client_uses_context_tracer():
    provider = FakeLLMProvider()

    tracer = Tracer()

    client = LLMClient(
        provider=provider,
        model="fake-model",
    )

    with trace_context(
        run_id="context-llm-run",
        tracer=tracer,
    ):
        client.generate(
            messages=[
                LLMMessage(
                    role="user",
                    content="Hello",
                )
            ]
        )

    events = tracer.get_events_for_run(
        "context-llm-run"
    )

    assert len(events) == 1


def test_llm_client_uses_explicit_tracer_over_context():
    provider = FakeLLMProvider()

    context_tracer = Tracer()
    explicit_tracer = Tracer()

    client = LLMClient(
        provider=provider,
        model="fake-model",
        tracer=explicit_tracer,
    )

    with trace_context(
        run_id="context-run",
        tracer=context_tracer,
    ):
        client.generate(
            messages=[
                LLMMessage(
                    role="user",
                    content="Hello",
                )
            ]
        )

    assert (
        len(
            context_tracer.get_events_for_run(
                "context-run"
            )
        )
        == 0
    )

    assert len(explicit_tracer.events) == 1


def test_llm_client_estimates_cost():
    provider = FakeLLMProvider(
        input_tokens=1000,
        output_tokens=500,
    )

    tracer = Tracer()

    client = LLMClient(
        provider=provider,
        model="fake-model",
        tracer=tracer,
        cost_per_1k_input_tokens=0.01,
        cost_per_1k_output_tokens=0.02,
    )

    with trace_context(
        run_id="cost-run",
        tracer=tracer,
    ):
        client.generate(
            messages=[
                LLMMessage(
                    role="user",
                    content="Calculate cost",
                )
            ]
        )

    event = tracer.get_events_for_run(
        "cost-run"
    )[0]

    assert event.estimated_cost == pytest.approx(
        0.02
    )


def test_provider_supplied_cost_is_preserved():
    class CostProvider(FakeLLMProvider):
        def generate(
            self,
            *,
            messages,
            model: str,
            **kwargs: Any,
        ) -> LLMResponse:
            return LLMResponse(
                content="priced",
                provider="cost-provider",
                model=model,
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                estimated_cost=0.123,
            )

    provider = CostProvider()
    tracer = Tracer()

    client = LLMClient(
        provider=provider,
        model="priced-model",
        tracer=tracer,
        cost_per_1k_input_tokens=999,
        cost_per_1k_output_tokens=999,
    )

    with trace_context(
        run_id="provider-cost-run",
        tracer=tracer,
    ):
        client.generate(
            messages=[
                LLMMessage(
                    role="user",
                    content="Hello",
                )
            ]
        )

    event = tracer.get_events_for_run(
        "provider-cost-run"
    )[0]

    assert event.estimated_cost == 0.123


def test_llm_client_accepts_message_dicts():
    provider = FakeLLMProvider()

    tracer = Tracer()

    client = LLMClient(
        provider=provider,
        model="fake-model",
        tracer=tracer,
    )

    with trace_context(
        run_id="dict-message-run",
        tracer=tracer,
    ):
        response = client.generate(
            messages=[
                {
                    "role": "user",
                    "content": "Hello",
                }
            ]
        )

    assert response.content == (
        "Hello from fake provider."
    )


def test_llm_client_supports_provider_kwargs():
    provider = FakeLLMProvider()

    client = LLMClient(
        provider=provider,
        model="fake-model",
    )

    client.generate(
        messages=[
            LLMMessage(
                role="user",
                content="Hello",
            )
        ],
        temperature=0.2,
        max_tokens=100,
    )

    assert len(provider.calls) == 1

    assert provider.calls[0]["kwargs"] == {
        "temperature": 0.2,
        "max_tokens": 100,
    }


def test_llm_trace_persists_to_sqlite(
    tmp_path: Path,
):
    database_path = tmp_path / "llm.db"

    storage = SQLiteStorage(
        database_path
    )

    tracer = Tracer(
        storage=storage
    )

    provider = FakeLLMProvider()

    client = LLMClient(
        provider=provider,
        model="fake-model",
        tracer=tracer,
    )

    try:
        with trace_context(
            run_id="persistent-llm-run",
            tracer=tracer,
        ):
            client.generate(
                messages=[
                    LLMMessage(
                        role="user",
                        content="Persist me",
                    )
                ]
            )

        stored_events = (
            storage.get_by_run(
                "persistent-llm-run"
            )
        )

        assert len(stored_events) == 1
        assert (
            stored_events[0].event_type
            == EventType.LLM_CALL
        )
        assert (
            stored_events[0].status
            == EventStatus.SUCCESS
        )
    finally:
        storage.close()