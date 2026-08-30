"""Measure what daily peeking costs, using the shipped monitor code.

Simulates whole experiments as look sequences and applies both stopping rules
from :mod:`abtestlab.monitor.policy` to each one:

* under a true null, how often does each rule declare a winner at least once?
  (the always-valid rule is supposed to stay at or below alpha; the naive rule
  is not.)
* under a true lift, how often does each rule fire, at which look, and how
  often does the anytime-valid p-value cross alpha and then drift back above it
  by the final look -- the case that makes "report the first crossing" and
  "report the current status" give different answers.

Usage::

    uv run python scripts/peeking_calibration.py
    uv run python scripts/peeking_calibration.py --trials 5000 --looks 14
"""

from __future__ import annotations

import argparse
import statistics

import numpy as np

from abtestlab.monitor.looks import Look
from abtestlab.monitor.policy import decide_always_valid, decide_naive_fixed
from abtestlab.monitor.sequential import evaluate_looks


def simulate_looks(
    rng: np.random.Generator,
    *,
    n_looks: int,
    per_look: int,
    baseline: float,
    lift: float,
) -> tuple[Look, ...]:
    """Draw one experiment as a sequence of cumulative looks.

    Args:
        rng: Seeded generator.
        n_looks: Number of checkpoints.
        per_look: New units per arm at each checkpoint.
        baseline: Control conversion rate.
        lift: Relative lift applied to the treatment rate.

    Returns:
        Cumulative looks for a single simulated experiment.
    """
    rate_treatment = min(baseline * (1.0 + lift), 0.9999)
    inc_control = rng.binomial(per_look, baseline, n_looks)
    inc_treatment = rng.binomial(per_look, rate_treatment, n_looks)
    cum_control = np.cumsum(inc_control)
    cum_treatment = np.cumsum(inc_treatment)
    return tuple(
        Look(
            index=i,
            label=f"day-{i + 1}",
            conversions_control=int(cum_control[i]),
            n_control=per_look * (i + 1),
            conversions_treatment=int(cum_treatment[i]),
            n_treatment=per_look * (i + 1),
        )
        for i in range(n_looks)
    )


def run_scenario(
    *,
    label: str,
    lift: float,
    trials: int,
    n_looks: int,
    per_look: int,
    baseline: float,
    alpha: float,
    seed: int,
) -> dict[str, float]:
    """Run one scenario and summarise both stopping rules over it.

    Args:
        label: Scenario name for the printed table.
        lift: True relative lift (0.0 for the null).
        trials: Number of simulated experiments.
        n_looks: Checkpoints per experiment.
        per_look: New units per arm per checkpoint.
        baseline: Control conversion rate.
        alpha: Significance level.
        seed: Base seed.

    Returns:
        Summary statistics for the scenario.
    """
    rng = np.random.default_rng(seed)
    naive_fires = 0
    safe_fires = 0
    final_look_only = 0
    recovered = 0
    stop_positions: list[int] = []

    for _ in range(trials):
        looks = simulate_looks(
            rng, n_looks=n_looks, per_look=per_look, baseline=baseline, lift=lift
        )
        evaluations = evaluate_looks(looks, alpha=alpha)
        safe = decide_always_valid(evaluations, alpha=alpha)
        naive = decide_naive_fixed(evaluations, alpha=alpha)
        naive_fires += int(naive.stopped)
        safe_fires += int(safe.stopped)
        final_look_only += int(evaluations[-1].crosses_always_valid)
        if safe.stopped:
            stop_positions.append(safe.looks_taken)
            recovered += int(safe.recovered_after_crossing)

    return {
        "label": label,
        "trials": float(trials),
        "naive_rate": naive_fires / trials,
        "safe_rate": safe_fires / trials,
        "final_look_rate": final_look_only / trials,
        "recovered_share": (recovered / safe_fires) if safe_fires else 0.0,
        "median_stop_look": float(statistics.median(stop_positions)) if stop_positions else 0.0,
    }


def main() -> None:
    """Parse arguments, run both scenarios, and print the comparison table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--looks", type=int, default=12)
    parser.add_argument("--per-look", type=int, default=1500)
    parser.add_argument("--baseline", type=float, default=0.05)
    parser.add_argument("--lift", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20240)
    args = parser.parse_args()

    common = {
        "trials": args.trials,
        "n_looks": args.looks,
        "per_look": args.per_look,
        "baseline": args.baseline,
        "alpha": args.alpha,
    }
    scenarios = [
        run_scenario(label="null (no effect)", lift=0.0, seed=args.seed, **common),
        run_scenario(
            label=f"true lift {args.lift:+.0%}", lift=args.lift, seed=args.seed + 1, **common
        ),
    ]

    print(
        f"{args.trials} simulated experiments per scenario, {args.looks} looks each, "
        f"{args.per_look} new units per arm per look, baseline {args.baseline:.1%}, "
        f"alpha {args.alpha:g}"
    )
    print()
    header = f"{'scenario':<20}{'naive peek':>12}{'always-valid':>14}{'final look':>12}{'median stop':>13}"
    print(header)
    print("-" * len(header))
    for row in scenarios:
        print(
            f"{row['label']:<20}"
            f"{row['naive_rate']:>11.1%}"
            f"{row['safe_rate']:>14.1%}"
            f"{row['final_look_rate']:>12.1%}"
            f"{row['median_stop_look']:>13.0f}"
        )
    print()
    print(
        "naive peek   = stopped at the first look where the fixed-horizon p < alpha\n"
        "always-valid = stopped at the first look where the mSPRT p < alpha\n"
        "final look   = mSPRT p < alpha when read only once, at the last look"
    )
    for row in scenarios:
        print(
            f"{row['label']}: of the runs the always-valid rule stopped, "
            f"{row['recovered_share']:.1%} were back above alpha by the final look."
        )


if __name__ == "__main__":
    main()
