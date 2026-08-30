"""Regenerate the simulated observation logs under ``data/``.

The logs are simulated, not real traffic. They exist so ``abtestlab monitor``
and the Streamlit workbench have something concrete to replay, and so the two
behaviours worth demonstrating -- an early crossing of the always-valid
boundary, and a refusal on a broken assignment -- are reproducible.

Usage::

    uv run python scripts/make_example_logs.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
COLUMNS = ["label", "conversions_control", "n_control", "conversions_treatment", "n_treatment"]


def build(
    *,
    seed: int,
    n_looks: int,
    per_look_control: int,
    per_look_treatment: int,
    baseline: float,
    lift: float,
) -> list[dict[str, object]]:
    """Simulate one cumulative observation log.

    Args:
        seed: Generator seed.
        n_looks: Number of daily checkpoints.
        per_look_control: New control units per day.
        per_look_treatment: New treatment units per day.
        baseline: Control conversion rate.
        lift: Relative lift applied to the treatment rate.

    Returns:
        Cumulative rows ready to write as CSV.
    """
    rng = np.random.default_rng(seed)
    rate_treatment = baseline * (1.0 + lift)
    cum_c = np.cumsum(rng.binomial(per_look_control, baseline, n_looks))
    cum_t = np.cumsum(rng.binomial(per_look_treatment, rate_treatment, n_looks))
    return [
        {
            "label": f"day-{i + 1:02d}",
            "conversions_control": int(cum_c[i]),
            "n_control": per_look_control * (i + 1),
            "conversions_treatment": int(cum_t[i]),
            "n_treatment": per_look_treatment * (i + 1),
        }
        for i in range(n_looks)
    ]


def write(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to ``path`` as CSV with a fixed column order.

    Args:
        path: Destination file.
        rows: Rows to write.
    """
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} looks)")


def main() -> None:
    """Regenerate both example logs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write(
        DATA_DIR / "example_looks.csv",
        build(
            seed=7,
            n_looks=14,
            per_look_control=4000,
            per_look_treatment=4000,
            baseline=0.05,
            lift=0.12,
        ),
    )
    # Treatment quietly loses ~4% of its traffic to a broken redirect.
    write(
        DATA_DIR / "example_looks_srm.csv",
        build(
            seed=11,
            n_looks=14,
            per_look_control=4000,
            per_look_treatment=3840,
            baseline=0.05,
            lift=0.12,
        ),
    )


if __name__ == "__main__":
    main()
