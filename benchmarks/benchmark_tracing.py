"""
Performance benchmark for Agent Black Box.

The benchmark compares the same deterministic workload under:

1. No tracing.
2. In-memory tracing.
3. SQLite tracing.

The benchmark reports:

- median runtime
- absolute overhead
- relative overhead

The workload is intentionally large enough that extremely small
microsecond-level baselines do not distort the percentage.

The SQLite connection is reused across iterations so the benchmark
measures trace persistence rather than repeated database startup.

Run from the project root:

    python benchmarks/benchmark_tracing.py
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from agent_black_box.events import EventType
from agent_black_box.storage.memory import InMemoryStorage
from agent_black_box.storage.sqlite import SQLiteStorage
from agent_black_box.tracer import Tracer


ITERATIONS = 100
WARMUP_ITERATIONS = 10

WORKLOAD_SIZE = 50_000
EVENT_COUNT = 5


def simulated_agent_work() -> int:
    """
    Execute deterministic CPU work representative of an agent step.

    The workload intentionally avoids network and LLM calls so the
    benchmark isolates tracing overhead.
    """

    total = 0

    for value in range(WORKLOAD_SIZE):
        total += (value * value) % 997

    return total


def run_baseline() -> int:
    """
    Execute the workload without tracing.
    """

    return simulated_agent_work()


def run_memory_tracing(
    tracer: Tracer,
) -> int:
    """
    Execute the workload using an already-created in-memory tracer.
    """

    with tracer.start_run(
        name="benchmark-agent",
    ):
        result = 0

        for index in range(EVENT_COUNT):
            event = tracer.start_event(
                event_type=EventType.FUNCTION_CALL,
                name=f"benchmark_step_{index + 1}",
                input_data={
                    "step": index + 1,
                },
            )

            result = simulated_agent_work()

            tracer.complete_event(
                event,
                output_data={
                    "result": result,
                },
            )

        return result


def run_sqlite_tracing(
    tracer: Tracer,
) -> int:
    """
    Execute the workload using an already-created SQLite tracer.
    """

    with tracer.start_run(
        name="benchmark-agent",
    ):
        result = 0

        for index in range(EVENT_COUNT):
            event = tracer.start_event(
                event_type=EventType.FUNCTION_CALL,
                name=f"benchmark_step_{index + 1}",
                input_data={
                    "step": index + 1,
                },
            )

            result = simulated_agent_work()

            tracer.complete_event(
                event,
                output_data={
                    "result": result,
                },
            )

        return result


def measure(
    function: Callable[[], int],
    iterations: int,
) -> list[float]:
    """
    Measure repeated execution time in milliseconds.
    """

    timings: list[float] = []

    for _ in range(iterations):
        started = time.perf_counter()

        function()

        elapsed = (
            time.perf_counter() - started
        ) * 1000

        timings.append(elapsed)

    return timings


def median_ms(
    timings: list[float],
) -> float:
    """
    Return the median execution time.
    """

    return statistics.median(timings)


def calculate_absolute_overhead(
    baseline_ms: float,
    traced_ms: float,
) -> float:
    """
    Calculate absolute tracing overhead in milliseconds.
    """

    return traced_ms - baseline_ms


def calculate_relative_overhead(
    baseline_ms: float,
    traced_ms: float,
) -> float:
    """
    Calculate relative tracing overhead as a percentage.
    """

    if baseline_ms <= 0:
        return 0.0

    return (
        (traced_ms - baseline_ms)
        / baseline_ms
    ) * 100


def print_result(
    name: str,
    timings: list[float],
    baseline_ms: float,
) -> None:
    """
    Print one benchmark result.
    """

    median = median_ms(timings)

    absolute_overhead = (
        calculate_absolute_overhead(
            baseline_ms,
            median,
        )
    )

    relative_overhead = (
        calculate_relative_overhead(
            baseline_ms,
            median,
        )
    )

    print(
        f"{name:<22}"
        f"{median:>12.4f} ms"
        f"{absolute_overhead:>14.4f} ms"
        f"{relative_overhead:>14.2f}%"
    )


def main() -> None:
    """
    Run the complete tracing benchmark.
    """

    print("=" * 82)
    print("Agent Black Box - Tracing Performance Benchmark")
    print("=" * 82)
    print()

    print(
        f"Workload iterations : {WORKLOAD_SIZE:,}"
    )
    print(
        f"Trace events/run    : {EVENT_COUNT}"
    )
    print(
        f"Warmup iterations   : {WARMUP_ITERATIONS}"
    )
    print(
        f"Measured iterations : {ITERATIONS}"
    )
    print()

    # --------------------------------------------------------------
    # Warmup
    # --------------------------------------------------------------

    for _ in range(WARMUP_ITERATIONS):
        run_baseline()

    # --------------------------------------------------------------
    # Baseline
    # --------------------------------------------------------------

    baseline_timings = measure(
        run_baseline,
        ITERATIONS,
    )

    baseline_median = median_ms(
        baseline_timings
    )

    # --------------------------------------------------------------
    # In-memory tracing
    # --------------------------------------------------------------

    memory_storage = InMemoryStorage()
    memory_tracer = Tracer(
        storage=memory_storage,
    )

    try:
        memory_timings = measure(
            lambda: run_memory_tracing(
                memory_tracer
            ),
            ITERATIONS,
        )
    finally:
        memory_tracer.close()

    # --------------------------------------------------------------
    # SQLite tracing
    # --------------------------------------------------------------

    with TemporaryDirectory(
        prefix="agent_black_box_benchmark_"
    ) as temporary_directory:

        database_path = (
            Path(temporary_directory)
            / "benchmark.db"
        )

        sqlite_storage = SQLiteStorage(
            database_path=database_path,
        )

        sqlite_tracer = Tracer(
            storage=sqlite_storage,
        )

        try:
            sqlite_timings = measure(
                lambda: run_sqlite_tracing(
                    sqlite_tracer
                ),
                ITERATIONS,
            )
        finally:
            sqlite_tracer.close()

    # --------------------------------------------------------------
    # Results
    # --------------------------------------------------------------

    print(
        f"{'Implementation':<22}"
        f"{'Median':>12}"
        f"{'Abs. Overhead':>16}"
        f"{'Rel. Overhead':>16}"
    )

    print("-" * 82)

    print(
        f"{'Baseline':<22}"
        f"{baseline_median:>12.4f} ms"
        f"{'0.0000 ms':>16}"
        f"{'0.00%':>16}"
    )

    print_result(
        "In-memory tracing",
        memory_timings,
        baseline_median,
    )

    print_result(
        "SQLite tracing",
        sqlite_timings,
        baseline_median,
    )

    print()
    print("=" * 82)
    print("Benchmark completed successfully.")
    print("=" * 82)
    print()
    print(
        "Interpretation:"
    )
    print(
        "Absolute overhead is more informative than percentage "
        "when the baseline workload is extremely small."
    )


if __name__ == "__main__":
    main()