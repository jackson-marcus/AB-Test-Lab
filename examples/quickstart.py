"""Runnable library example used by the README.

Run from the repo root after ``uv sync``::

    uv run python examples/quickstart.py
"""

from __future__ import annotations

from abtestlab.functional import (
    ExperimentArm,
    apply_cuped,
    compute_sample_size,
    evaluate_bayesian_beta_binomial,
    evaluate_msprt,
)


def main() -> None:
    """Print sample-size, mSPRT, Bayesian, and CUPED results for a toy experiment."""
    sample_n = compute_sample_size(p_baseline=0.08, mde_relative=0.10, alpha=0.05, power=0.80)
    print(f"Required per-arm sample size: {sample_n}")

    control = ExperimentArm(name="Control_V1", conversions=820, sample_size=10000)
    treatment = ExperimentArm(name="Treatment_V2", conversions=960, sample_size=10000)

    msprt_res = evaluate_msprt(control, treatment, mixing_variance_theta=0.05, alpha=0.05)
    print(
        f"mSPRT stopping decision: {msprt_res.should_stop_early} "
        f"(likelihood ratio: {msprt_res.likelihood_ratio:.2f})"
    )

    bayes_res = evaluate_bayesian_beta_binomial(control, treatment)
    print(f"P(treatment > control): {bayes_res.p_treatment_beats_control:.1%}")

    cuped_res = apply_cuped(y_post=[12.0, 15.0, 18.0, 22.0], x_pre=[10.0, 14.0, 17.0, 20.0])
    print(f"CUPED variance reduced by: {cuped_res.variance_reduction_pct:.1f}%")


if __name__ == "__main__":
    main()
