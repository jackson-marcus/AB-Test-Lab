"""API routes: /power, /analyze, /sequential, /cuped-demo, /bayes, /health."""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from abtestlab.stats.core import (
    bayes_beta_bernoulli,
    cuped_test,
    msprt,
    sample_size_two_proportions,
    z_test_two_proportions,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class Counts(BaseModel):
    conversions_control: int = Field(ge=0)
    n_control: int = Field(gt=0)
    conversions_treatment: int = Field(ge=0)
    n_treatment: int = Field(gt=0)


class PowerRequest(BaseModel):
    baseline_rate: float = Field(gt=0, lt=1)
    mde_relative: float = Field(gt=0, le=2)
    alpha: float = Field(default=0.05, gt=0, lt=0.5)
    power: float = Field(default=0.8, gt=0.5, lt=1)


class CupedDemoRequest(BaseModel):
    n_per_arm: int = Field(default=2000, ge=100, le=100_000)
    true_lift: float = Field(default=0.03, ge=0, le=1)
    covariate_correlation: float = Field(default=0.6, ge=0, le=0.95)
    seed: int = 0


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/power")
def power(request: PowerRequest) -> dict:
    n = sample_size_two_proportions(
        request.baseline_rate, request.mde_relative, request.alpha, request.power
    )
    return {"per_arm_sample_size": n, "total": 2 * n}


@router.post("/analyze")
def analyze(counts: Counts) -> dict:
    return z_test_two_proportions(
        counts.conversions_control,
        counts.n_control,
        counts.conversions_treatment,
        counts.n_treatment,
    )


@router.post("/sequential")
def sequential(counts: Counts) -> dict:
    return msprt(
        counts.conversions_control,
        counts.n_control,
        counts.conversions_treatment,
        counts.n_treatment,
    )


@router.post("/bayes")
def bayes(counts: Counts) -> dict:
    return bayes_beta_bernoulli(
        counts.conversions_control,
        counts.n_control,
        counts.conversions_treatment,
        counts.n_treatment,
    )


@router.post("/cuped-demo")
def cuped_demo(request: CupedDemoRequest) -> dict:
    """Simulate a revenue-style metric with a correlated pre-period covariate
    and show CUPED's variance reduction on the same data."""
    rng = np.random.default_rng(request.seed)
    n = request.n_per_arm
    rho = request.covariate_correlation
    pre_a = rng.gamma(2, 10, n)
    pre_b = rng.gamma(2, 10, n)
    noise_scale = np.sqrt(max(1 - rho**2, 1e-6))
    metric_a = rho * pre_a + noise_scale * rng.gamma(2, 10, n)
    metric_b = (rho * pre_b + noise_scale * rng.gamma(2, 10, n)) * (1 + request.true_lift)
    return cuped_test(metric_a, pre_a, metric_b, pre_b)
