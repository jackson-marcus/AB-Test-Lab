"""HTTP routes for power, fixed-horizon, sequential, CUPED demo, and Bayes."""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from abtestlab.monitor import DEFAULT_SRM_THRESHOLD, LookSequenceError, run_monitor
from abtestlab.stats.core import (
    BayesResult,
    CupedTestResult,
    ZTestResult,
    bayes_beta_bernoulli,
    cuped_test,
    msprt,
    sample_size_two_proportions,
    z_test_two_proportions,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class Counts(BaseModel):
    """Binomial counts for control and treatment arms."""

    conversions_control: int = Field(ge=0)
    n_control: int = Field(gt=0)
    conversions_treatment: int = Field(ge=0)
    n_treatment: int = Field(gt=0)

    @model_validator(mode="after")
    def conversions_fit_sample_size(self) -> Counts:
        """Reject conversions that exceed their arm's sample size.

        Returns:
            This model if valid.

        Raises:
            ValueError: If either arm has more conversions than units.
        """
        if self.conversions_control > self.n_control:
            raise ValueError("conversions_control cannot exceed n_control")
        if self.conversions_treatment > self.n_treatment:
            raise ValueError("conversions_treatment cannot exceed n_treatment")
        return self


class PowerRequest(BaseModel):
    """Inputs for a two-proportion sample-size calculation."""

    baseline_rate: float = Field(gt=0, lt=1)
    mde_relative: float = Field(gt=0, le=2)
    alpha: float = Field(default=0.05, gt=0, lt=0.5)
    power: float = Field(default=0.8, gt=0.5, lt=1)


class CupedDemoRequest(BaseModel):
    """Simulation parameters for the CUPED demonstration endpoint."""

    n_per_arm: int = Field(default=2000, ge=100, le=100_000)
    true_lift: float = Field(default=0.03, ge=0, le=1)
    covariate_correlation: float = Field(default=0.6, ge=0, le=0.95)
    seed: int = 0


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe.

    Returns:
        ``{"status": "ok"}`` when the process can serve requests.
    """
    return {"status": "ok"}


@router.post("/power")
def power(request: PowerRequest) -> dict[str, int]:
    """Compute per-arm sample size for a relative MDE.

    Args:
        request: Baseline rate, MDE, alpha, and power.

    Returns:
        Per-arm n and total (2n).
    """
    logger.info(
        "power baseline=%.4f mde=%.4f alpha=%.3f power=%.3f",
        request.baseline_rate,
        request.mde_relative,
        request.alpha,
        request.power,
    )
    try:
        n = sample_size_two_proportions(
            request.baseline_rate, request.mde_relative, request.alpha, request.power
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"per_arm_sample_size": n, "total": 2 * n}


@router.post("/analyze")
def analyze(counts: Counts) -> ZTestResult:
    """Fixed-horizon two-proportion z-test.

    Args:
        counts: Arm conversion counts.

    Returns:
        Z-test payload from :func:`abtestlab.stats.core.z_test_two_proportions`.
    """
    logger.info(
        "analyze n_c=%s n_t=%s",
        counts.n_control,
        counts.n_treatment,
    )
    return z_test_two_proportions(
        counts.conversions_control,
        counts.n_control,
        counts.conversions_treatment,
        counts.n_treatment,
    )


@router.post("/sequential")
def sequential(counts: Counts) -> dict[str, float | bool]:
    """Config-backed mSPRT (API formula, not the functional library).

    Args:
        counts: Arm conversion counts.

    Returns:
        Mixture SPRT payload from :func:`abtestlab.stats.core.msprt`.
    """
    logger.info("sequential n_c=%s n_t=%s", counts.n_control, counts.n_treatment)
    return msprt(
        counts.conversions_control,
        counts.n_control,
        counts.conversions_treatment,
        counts.n_treatment,
    )


@router.post("/bayes")
def bayes(counts: Counts) -> BayesResult:
    """Beta-Bernoulli probability-to-beat-control and expected loss.

    Args:
        counts: Arm conversion counts.

    Returns:
        Bayesian payload from :func:`abtestlab.stats.core.bayes_beta_bernoulli`.
    """
    logger.info("bayes n_c=%s n_t=%s", counts.n_control, counts.n_treatment)
    return bayes_beta_bernoulli(
        counts.conversions_control,
        counts.n_control,
        counts.conversions_treatment,
        counts.n_treatment,
    )


@router.post("/cuped-demo")
def cuped_demo(request: CupedDemoRequest) -> CupedTestResult:
    """Simulate a correlated pre-period covariate and report CUPED vs raw tests.

    Args:
        request: Simulation size, true lift, correlation, and seed.

    Returns:
        CUPED comparison from :func:`abtestlab.stats.core.cuped_test`.
    """
    logger.info(
        "cuped-demo n=%s lift=%.4f rho=%.3f",
        request.n_per_arm,
        request.true_lift,
        request.covariate_correlation,
    )
    rng = np.random.default_rng(request.seed)
    n = request.n_per_arm
    rho = request.covariate_correlation
    pre_a = rng.gamma(2, 10, n)
    pre_b = rng.gamma(2, 10, n)
    noise_scale = np.sqrt(max(1 - rho**2, 1e-6))
    metric_a = rho * pre_a + noise_scale * rng.gamma(2, 10, n)
    metric_b = (rho * pre_b + noise_scale * rng.gamma(2, 10, n)) * (1 + request.true_lift)
    return cuped_test(metric_a, pre_a, metric_b, pre_b)


class LookRow(BaseModel):
    """One checkpoint in an experiment observation log."""

    label: str | None = None
    conversions_control: int = Field(ge=0)
    n_control: int = Field(ge=0)
    conversions_treatment: int = Field(ge=0)
    n_treatment: int = Field(ge=0)


class MonitorRequest(BaseModel):
    """An ordered observation log plus the monitoring policy to apply."""

    looks: list[LookRow] = Field(min_length=1)
    cumulative: bool = True
    alpha: float = Field(default=0.05, gt=0, lt=0.5)
    expected_share_control: float = Field(default=0.5, gt=0, lt=1)
    srm_threshold: float = Field(default=DEFAULT_SRM_THRESHOLD, gt=0, lt=1)


@router.post("/monitor")
def monitor(request: MonitorRequest) -> dict[str, object]:
    """Replay an experiment look by look and return a peek-safe recommendation.

    Runs the sample-ratio-mismatch gate on exposure counts first; if assignment
    is broken the response carries ``status="refused_srm"`` and no effect
    estimate. Otherwise every look is scored with both the always-valid mSPRT
    p-value and the fixed-horizon p-value, and both stopping rules are reported
    so the cost of naive peeking is visible on the caller's own data.

    Args:
        request: Observation log, alpha, and SRM policy.

    Returns:
        Serialized :class:`abtestlab.monitor.MonitorReport`.

    Raises:
        HTTPException: 422 if the log is not a readable ordered experiment.
    """
    logger.info("monitor looks=%s cumulative=%s", len(request.looks), request.cumulative)
    rows = [row.model_dump(exclude_none=True) for row in request.looks]
    try:
        report = run_monitor(
            rows,
            cumulative=request.cumulative,
            alpha=request.alpha,
            expected_share_control=request.expected_share_control,
            srm_threshold=request.srm_threshold,
        )
    except LookSequenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.info("monitor status=%s", report.status)
    return report.as_dict()
