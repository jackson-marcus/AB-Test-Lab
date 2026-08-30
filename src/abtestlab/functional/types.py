"""Pure Functional Library — immutable statistics types.

Input and output value objects for side-effect-free experimentation math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentArm:
    """Immutable record of one experiment variant.

    Attributes:
        name: Human-readable arm label.
        conversions: Count of successes; must satisfy ``0 <= conversions <= sample_size``.
        sample_size: Number of units assigned to the arm; must be positive.

    Raises:
        ValueError: If sample size is not positive or conversions are out of range.
    """

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
        """Empirical conversion rate ``conversions / sample_size``."""
        return self.conversions / self.sample_size


@dataclass(frozen=True)
class FixedHorizonResult:
    """Result of a two-proportion Z-test.

    Attributes:
        p_control: Control conversion rate.
        p_treatment: Treatment conversion rate.
        absolute_difference: Treatment minus control.
        relative_lift: Relative change vs control.
        z_statistic: Pooled two-proportion z.
        p_value: Two-sided normal p-value.
        ci_lower: Lower bound of the difference CI.
        ci_upper: Upper bound of the difference CI.
        is_significant: Whether ``p_value < alpha`` used at evaluation time.
    """

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
        """Return a JSON-friendly rounded mapping of this result.

        Returns:
            Dict with abbreviated keys used by notebooks and demos.
        """
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
    """Result of the library mSPRT (Gaussian-mixture likelihood ratio).

    Attributes:
        likelihood_ratio: Mixture likelihood ratio ``Lambda_t``.
        stopping_threshold: ``1 / alpha``.
        should_stop_early: Whether ``Lambda_t`` meets the threshold.
        nominal_alpha: Alpha used to form the threshold.
        estimated_p_value_bound: ``min(1, 1 / Lambda_t)``.
    """

    likelihood_ratio: float
    stopping_threshold: float
    should_stop_early: bool
    nominal_alpha: float
    estimated_p_value_bound: float


@dataclass(frozen=True)
class BayesianBetaBinomialResult:
    """Result of conjugate Beta-Bernoulli Monte Carlo analysis.

    Attributes:
        p_treatment_beats_control: Share of draws with treatment rate higher.
        expected_loss_control_if_ship: E[max(treatment - control, 0)].
        expected_loss_treatment_if_ship: E[max(control - treatment, 0)].
        posterior_a_mean: Control posterior mean.
        posterior_b_mean: Treatment posterior mean.
    """

    p_treatment_beats_control: float
    expected_loss_control_if_ship: float
    expected_loss_treatment_if_ship: float
    posterior_a_mean: float
    posterior_b_mean: float


@dataclass(frozen=True)
class CupedResult:
    """Result of CUPED pre-experiment covariate variance reduction.

    Attributes:
        theta_optimal: Fitted CUPED coefficient.
        raw_variance: Sample variance of Y (ddof=1).
        adjusted_variance: Sample variance of the CUPED residual.
        variance_reduction_pct: Percent reduction, floored at 0.
        adjusted_values: CUPED-adjusted Y values.
    """

    theta_optimal: float
    raw_variance: float
    adjusted_variance: float
    variance_reduction_pct: float
    adjusted_values: tuple[float, ...]
