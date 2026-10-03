"""
Basic Agent Black Box example.

Demonstrates:

1. SQLite-backed tracing.
2. Agent run lifecycle.
3. Function tracing.
4. Tool tracing.
5. Inputs and outputs.
6. Execution duration.
7. Reading persisted trace events.
"""

from pathlib import Path
import time

from agent_black_box import (
    EventType,
    SQLiteStorage,
    Tracer,
    trace,
)


DATABASE_PATH = Path("data/example_traces.db")


@trace
def retrieve_customer(customer_id: int) -> dict:
    """Simulate retrieving customer information."""
    return {
        "customer_id": customer_id,
        "name": "Demo Customer",
        "plan": "professional",
    }


@trace
def calculate_discount(amount: float, percentage: float) -> float:
    """Calculate a discounted amount."""
    return amount * (1 - percentage / 100)


def main() -> None:
    storage = SQLiteStorage(database_path=DATABASE_PATH)
    tracer = Tracer(storage=storage)

    try:
        with tracer.start_run(
            run_id="example-agent-run",
            name="customer-pricing-agent",
            metadata={
                "environment": "example",
                "application": "agent-black-box-demo",
            },
        ):
            customer = retrieve_customer(101)

            amount = 1000.0
            discount = 10.0

            tool_event = tracer.start_event(
                event_type=EventType.TOOL_CALL,
                name="pricing_calculation",
                input_data={
                    "customer_id": customer["customer_id"],
                    "amount": amount,
                    "discount": discount,
                },
            )

            start_time = time.perf_counter()

            final_amount = calculate_discount(
                amount=amount,
                percentage=discount,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            tracer.complete_event(
                tool_event,
                output_data={
                    "customer_id": customer["customer_id"],
                    "final_amount": final_amount,
                },
                duration_ms=duration_ms,
            )

            print(f"Customer: {customer['name']}")
            print(f"Final amount: {final_amount:.2f}")

        events = tracer.get_events_for_run("example-agent-run")

        print("\nTrace")
        print("-" * 70)

        for event in events:
            print(
                f"{event.event_type:<20} "
                f"{event.status:<10} "
                f"{event.name or '-'}"
            )

        print("-" * 70)
        print(f"Total events: {len(events)}")
        print(f"Database: {DATABASE_PATH}")

    finally:
        tracer.close()


if __name__ == "__main__":
    main()