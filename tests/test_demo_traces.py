"""
Tests for the Agent Black Box demonstration trace scenarios.
"""

from agent_black_box import (
    EventType,
    InMemoryStorage,
    Tracer,
)


def create_demo_tracer() -> Tracer:
    """Create an in-memory tracer for demo scenario tests."""
    return Tracer(
        storage=InMemoryStorage()
    )


def test_pricing_scenario():
    tracer = create_demo_tracer()

    with tracer.start_run(
        run_id="demo-pricing-test",
        name="customer-pricing-agent",
    ):
        customer_event = tracer.start_event(
            event_type=EventType.FUNCTION_CALL,
            name="retrieve_customer",
            input_data={
                "customer_id": 101,
            },
        )

        tracer.complete_event(
            customer_event,
            output_data={
                "customer_id": 101,
                "plan": "professional",
            },
            duration_ms=2.1,
        )

        pricing_event = tracer.start_event(
            event_type=EventType.TOOL_CALL,
            name="pricing_calculation",
            input_data={
                "amount": 1000.0,
                "discount_percentage": 10.0,
            },
        )

        tracer.complete_event(
            pricing_event,
            output_data={
                "final_amount": 900.0,
            },
            duration_ms=1.3,
        )

    events = tracer.get_events_for_run(
        "demo-pricing-test"
    )

    assert len(events) == 4
    assert events[0].event_type == "run_start"
    assert events[1].event_type == "function_call"
    assert events[2].event_type == "tool_call"
    assert events[3].event_type == "run_end"

    assert events[3].status == "success"


def test_support_failure_scenario():
    tracer = create_demo_tracer()

    with tracer.start_run(
        run_id="demo-support-test",
        name="customer-support-agent",
    ):
        classify_event = tracer.start_event(
            event_type=EventType.FUNCTION_CALL,
            name="classify_customer_issue",
            input_data={
                "message": "My order has not arrived yet.",
            },
        )

        tracer.complete_event(
            classify_event,
            output_data={
                "category": "delivery_issue",
                "priority": "high",
            },
            duration_ms=3.4,
        )

        ticket_event = tracer.start_event(
            event_type=EventType.TOOL_CALL,
            name="create_support_ticket",
            input_data={
                "category": "delivery_issue",
                "priority": "high",
            },
        )

        tracer.fail_event(
            ticket_event,
            error=RuntimeError(
                "Support ticket service returned HTTP 503"
            ),
            duration_ms=84.7,
        )

    events = tracer.get_events_for_run(
        "demo-support-test"
    )

    assert len(events) == 4

    tool_failure = next(
        event
        for event in events
        if (
            event.event_type == "tool_call"
            and event.name == "create_support_ticket"
        )
    )

    assert tool_failure.status == "failed"
    assert tool_failure.error_type == "RuntimeError"
    assert (
        tool_failure.error_message
        == "Support ticket service returned HTTP 503"
    )

    run_end = next(
        event
        for event in events
        if event.event_type == "run_end"
    )

    assert run_end.status == "failed"

    assert events[-1].event_type == "run_end"
    assert events[-1].status == "failed"


def test_research_llm_scenario():
    tracer = create_demo_tracer()

    with tracer.start_run(
        run_id="demo-research-test",
        name="research-agent",
    ):
        search_event = tracer.start_event(
            event_type=EventType.TOOL_CALL,
            name="web_search",
            input_data={
                "query": "AI agent observability",
                "max_results": 5,
            },
        )

        tracer.complete_event(
            search_event,
            output_data={
                "results_found": 5,
                "source": "synthetic-demo-search",
            },
            duration_ms=118.4,
        )

        llm_event = tracer.start_event(
            event_type=EventType.LLM_CALL,
            name="research_synthesis",
            input_data={
                "prompt": (
                    "Summarize the retrieved information "
                    "about AI agent observability."
                ),
            },
            provider="groq",
            model="demo-model",
        )

        tracer.complete_event(
            llm_event,
            output_data={
                "response": (
                    "AI-agent observability captures execution "
                    "steps, model interactions, tool calls, "
                    "latency, failures, and usage information."
                ),
            },
            duration_ms=742.6,
            input_tokens=184,
            output_tokens=96,
            total_tokens=280,
            estimated_cost=0.00042,
        )

    events = tracer.get_events_for_run(
        "demo-research-test"
    )

    llm_events = [
        event
        for event in events
        if event.event_type == "llm_call"
    ]

    assert len(llm_events) == 1

    llm_event = llm_events[0]

    assert llm_event.provider == "groq"
    assert llm_event.model == "demo-model"
    assert llm_event.input_tokens == 184
    assert llm_event.output_tokens == 96
    assert llm_event.total_tokens == 280
    assert llm_event.estimated_cost == 0.00042


def test_payment_scenario():
    tracer = create_demo_tracer()

    with tracer.start_run(
        run_id="demo-payment-test",
        name="payment-agent",
    ):
        validation_event = tracer.start_event(
            event_type=EventType.FUNCTION_CALL,
            name="validate_payment_request",
            input_data={
                "currency": "USD",
                "amount": 249.99,
            },
        )

        tracer.complete_event(
            validation_event,
            output_data={
                "valid": True,
                "currency": "USD",
                "amount": 249.99,
            },
            duration_ms=1.8,
        )

        payment_event = tracer.start_event(
            event_type=EventType.TOOL_CALL,
            name="process_payment",
            input_data={
                "amount": 249.99,
                "currency": "USD",
                "payment_method": "demo_card",
            },
        )

        tracer.complete_event(
            payment_event,
            output_data={
                "status": "approved",
                "transaction_id": "demo-txn-test",
            },
            duration_ms=156.2,
        )

    events = tracer.get_events_for_run(
        "demo-payment-test"
    )

    payment_events = [
        event
        for event in events
        if event.name == "process_payment"
    ]

    assert len(payment_events) == 1
    assert payment_events[0].status == "success"
    assert payment_events[0].output_data["status"] == "approved"