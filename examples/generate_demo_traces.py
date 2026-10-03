"""
Generate realistic demonstration traces for Agent Black Box.

This script creates a synthetic SQLite trace database containing
multiple agent runs:

1. Customer pricing agent       - successful
2. Customer support agent       - failed tool call
3. Research agent              - successful LLM workflow
4. Payment agent               - successful tool workflow

The data is synthetic and contains no real personal information.

Run from the project root:

    python examples/generate_demo_traces.py
"""

from __future__ import annotations

import time
from pathlib import Path

from agent_black_box import EventType, SQLiteStorage, Tracer


DATABASE_PATH = Path("data/demo_traces.db")


def create_pricing_run(tracer: Tracer) -> None:
    """Create a successful customer pricing agent run."""

    with tracer.start_run(
        run_id="demo-pricing-001",
        name="customer-pricing-agent",
        metadata={
            "environment": "demo",
            "agent_version": "1.0.0",
            "scenario": "successful pricing workflow",
        },
    ):
        customer_event = tracer.start_event(
            event_type=EventType.FUNCTION_CALL,
            name="retrieve_customer",
            input_data={
                "customer_id": 101,
            },
        )

        time.sleep(0.002)

        tracer.complete_event(
            customer_event,
            output_data={
                "customer_id": 101,
                "name": "Demo Customer",
                "plan": "professional",
            },
            duration_ms=2.1,
        )

        pricing_event = tracer.start_event(
            event_type=EventType.TOOL_CALL,
            name="pricing_calculation",
            input_data={
                "customer_id": 101,
                "amount": 1000.0,
                "discount_percentage": 10.0,
            },
        )

        time.sleep(0.001)

        tracer.complete_event(
            pricing_event,
            output_data={
                "customer_id": 101,
                "original_amount": 1000.0,
                "discount_amount": 100.0,
                "final_amount": 900.0,
            },
            duration_ms=1.3,
        )


def create_support_failure_run(tracer: Tracer) -> None:
    """Create a run containing a failed support tool call."""

    with tracer.start_run(
        run_id="demo-support-002",
        name="customer-support-agent",
        metadata={
            "environment": "demo",
            "agent_version": "1.0.0",
            "scenario": "tool failure",
        },
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

        error = RuntimeError(
            "Support ticket service returned HTTP 503"
        )

        tracer.fail_event(
            ticket_event,
            error=error,
            duration_ms=84.7,
        )


def create_research_run(tracer: Tracer) -> None:
    """Create a successful LLM-powered research workflow."""

    with tracer.start_run(
        run_id="demo-research-003",
        name="research-agent",
        metadata={
            "environment": "demo",
            "agent_version": "1.0.0",
            "scenario": "LLM research workflow",
        },
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


def create_payment_run(tracer: Tracer) -> None:
    """Create a successful payment workflow."""

    with tracer.start_run(
        run_id="demo-payment-004",
        name="payment-agent",
        metadata={
            "environment": "demo",
            "agent_version": "1.0.0",
            "scenario": "successful payment workflow",
        },
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
                "transaction_id": "demo-txn-004",
            },
            duration_ms=156.2,
        )


def main() -> None:
    """Generate the demonstration trace database."""

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()

    storage = SQLiteStorage(
        database_path=DATABASE_PATH,
    )

    tracer = Tracer(
        storage=storage,
    )

    try:
        create_pricing_run(tracer)
        create_support_failure_run(tracer)
        create_research_run(tracer)
        create_payment_run(tracer)
    finally:
        tracer.close()

    print("Demo trace database created successfully.")
    print(f"Database: {DATABASE_PATH}")
    print()
    print("Runs:")
    print("  1. customer-pricing-agent     SUCCESS")
    print("  2. customer-support-agent     FAILED")
    print("  3. research-agent             SUCCESS")
    print("  4. payment-agent              SUCCESS")


if __name__ == "__main__":
    main()