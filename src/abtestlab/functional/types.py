"""Pure Functional Library - Immutable Statistics Types.

Defines immutable input and output value objects for side-effect-free
experimentation mathematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentArm:
    """Immutable record of an experiment variant arm."""

    name: str
    conversions: int
    sample_size: int

    def __post_init__(self) -> None:
        if self.sample_size <= 0:
            raise ValueError(f"Sample size must be positive, got {self.sample_size}")
        if self.conversions < 0:
            raise ValueError(f"Conversions cannot be negative, got {self.conversions}")
        if self.conversions > self.sample_size:
            raise ValueError(
                f"Conversions ({self.conversions}) cannot exceed sample size ({self.sample_size})"
            )

    @property
    def conversion_rate(self) -> float:
        return self.conversions / self.sample_size


@dataclass(frozen=True)
class FixedHorizonResult:
    """Result of a two-proportion Z-test."""

    p_control: float
    p_treatment: float
    absolute_difference: float
    relative_lift: float
    z_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    is_significant: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "p_control": round(self.p_control, 5),
            "p_treatment": round(self.p_treatment, 5),
            "abs_diff": round(self.absolute_difference, 5),
            "relative_lift": round(self.relative_lift, 5),
            "z_stat": round(self.z_statistic, 4),
            "p_value": round(self.p_value, 6),
            "ci_95": [round(self.ci_lower, 5), round(self.ci_upper, 5)],
            "is_significant": self.is_significant,
        }


@dataclass(frozen=True)
class MSPRTResult:
    """Result of an always-valid mixture sequential probability ratio test."""

    likelihood_ratio: float
    stopping_threshold: float
    should_stop_early: bool
    nominal_alpha: float
    estimated_p_value_bound: float


@dataclass(frozen=True)
class BayesianBetaBinomialResult:
    """Result of conjugate Beta-Bernoulli Bayesian posterior analysis."""

    p_treatment_beats_control: float
    expected_loss_control_if_ship: float
    expected_loss_treatment_if_ship: float
    posterior_a_mean: float
    posterior_b_mean: float


@dataclass(frozen=True)
class CupedResult:
    """Result of CUPED pre-experiment covariate variance reduction."""

    theta_optimal: float
    raw_variance: float
    adjusted_variance: float
    variance_reduction_pct: float
    adjusted_values: tuple[float, ...]
