#!/usr/bin/env python3
"""Summarize recorded vs derived runtime stats for benchmark runs."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

from validate_guardrails import (
    COHORTS,
    derive_runtime_seconds,
    get_reported_duration_seconds,
    iter_run_dirs,
    parse_run_identity,
)


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        raise ValueError("percentile() requires at least one value")
    if len(sorted_values) == 1:
        return sorted_values[0]

    pos = (len(sorted_values) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    q1 = percentile(ordered, 0.25)
    q3 = percentile(ordered, 0.75)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }


def format_stats(label: str, values: list[float]) -> str:
    if not values:
        return f"{label}: no data"
    stats = summarize(values)
    return (
        f"{label}: n={stats['count']}, min={stats['min']:.2f}s, max={stats['max']:.2f}s, "
        f"mean={stats['mean']:.2f}s, median={stats['median']:.2f}s, "
        f"IQR={stats['iqr']:.2f}s (Q1={stats['q1']:.2f}s, Q3={stats['q3']:.2f}s)"
    )


def collect_runtimes(approach: str, cohort_filter: str | None) -> tuple[list[float], list[float]]:
    reported: list[float] = []
    derived: list[float] = []

    for cohort, cfg in COHORTS.items():
        if cohort_filter and cohort != cohort_filter:
            continue

        results_dir = cfg["results_dir"]
        if not results_dir.exists():
            continue

        for run_dir in iter_run_dirs(results_dir):
            ident = parse_run_identity(run_dir, results_dir)
            if ident["approach"] != approach:
                continue

            reported_duration = get_reported_duration_seconds(run_dir)
            if reported_duration is not None:
                reported.append(reported_duration)

            derived_duration = derive_runtime_seconds(run_dir, approach)
            if derived_duration is not None:
                derived.append(derived_duration)

    return reported, derived


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach", default="codex", help="Runner approach to summarize, e.g. codex or opencode")
    parser.add_argument("--cohort", default=None, help="Optional cohort name from validate_guardrails.COHORTS")
    args = parser.parse_args()

    reported, derived = collect_runtimes(args.approach, args.cohort)

    print(f"Approach: {args.approach}")
    if args.cohort:
        print(f"Cohort: {args.cohort}")
    print(format_stats("Recorded runtime", reported))
    print(format_stats("Derived runtime", derived))


if __name__ == "__main__":
    main()
