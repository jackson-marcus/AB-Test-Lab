"""Pure functional statistics for online experimentation.

Functions here perform no I/O and do not read process config. For the
YAML-backed API implementations see :mod:`abtestlab.stats.core`.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from abtestlab.functional.types import (
    BayesianBetaBinomialResult,
    CupedResult,
    ExperimentArm,
    FixedHorizonResult,
    MSPRTResult,
)


def compute_sample_size(
    p_baseline: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Calculate required per-arm sample size for a two-proportion test.

    The alternative rate is ``p_baseline * (1 + mde_relative)``, capped at
    0.9999 so the formula stays defined for large relative MDEs.

    Args:
        p_baseline: Control conversion rate in ``(0, 1)``.
        mde_relative: Minimum detectable relative lift; must be ``> 0``.
        alpha: Two-sided Type I error.
        power: Target power in ``(0, 1)``.

    Returns:
        Ceiling of the standard two-proportion n per arm.

    Raises:
        ValueError: If ``p_baseline`` is not in ``(0, 1)`` or ``mde_relative``
            is not strictly positive.
    """
    if not (0.0 < p_baseline < 1.0):
        raise ValueError(f"Baseline proportion must be in (0, 1), got {p_baseline}")
    if mde_relative <= 0.0:
        raise ValueError(f"MDE relative must be strictly positive, got {mde_relative}")

    p2 = p_baseline * (1.0 + mde_relative)
    p2 = min(p2, 0.9999)

    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)
    p_pool = (p_baseline + p2) / 2.0

    numerator = (
        z_alpha * math.sqrt(2 * p_pool * (1 - p_pool))
        + z_beta * math.sqrt(p_baseline * (1 - p_baseline) + p2 * (1 - p2))
    ) ** 2
    denominator = (p2 - p_baseline) ** 2

    return math.ceil(numerator / denominator)


def evaluate_fixed_horizon(
    control: ExperimentArm,
    treatment: ExperimentArm,
    alpha: float = 0.05,
) -> FixedHorizonResult:
    """Two-sided z-test of two independent binomial proportions.

    Args:
        control: Control arm counts.
        treatment: Treatment arm counts.
        alpha: Significance level used for the CI and ``is_significant``.

    Returns:
        Immutable :class:`FixedHorizonResult`.
    """
    p_a = control.conversion_rate
    p_b = treatment.conversion_rate
    n_a = control.sample_size
    n_b = treatment.sample_size

    p_pool = (control.conversions + treatment.conversions) / (n_a + n_b)
    se_pool = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))

    z = (p_b - p_a) / max(se_pool, 1e-12)
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

    lift_rel = (p_b - p_a) / max(p_a, 1e-12)
    se_diff = math.sqrt(p_a * (1.0 - p_a) / n_a + p_b * (1.0 - p_b) / n_b)

    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci_lo = float((p_b - p_a) - z_crit * se_diff)
    ci_hi = float((p_b - p_a) + z_crit * se_diff)

    return FixedHorizonResult(
        p_control=p_a,
        p_treatment=p_b,
        absolute_difference=p_b - p_a,
        relative_lift=lift_rel,
        z_statistic=z,
        p_value=p_val,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        is_significant=p_val < alpha,
    )


def evaluate_msprt(
    control: ExperimentArm,
    treatment: ExperimentArm,
    mixing_variance_theta: float = 0.05,
    alpha: float = 0.05,
) -> MSPRTResult:
    """Mixture sequential probability ratio test (library parameterization).

    Uses effective sample size ``n_eff = n_A n_B / (n_A + n_B)`` and mixture
    variance ``mixing_variance_theta``. This is **not** the same formula as
    :func:`abtestlab.stats.core.msprt`.

    Args:
        control: Control arm counts.
        treatment: Treatment arm counts.
        mixing_variance_theta: Gaussian mixture variance (tau^2 analogue).
        alpha: Type I bound used as stopping threshold ``1 / alpha``.

    Returns:
        Immutable :class:`MSPRTResult`.

    Raises:
        ValueError: If ``mixing_variance_theta`` or ``alpha`` is not positive.
    """
    if mixing_variance_theta <= 0.0:
        raise ValueError(f"mixing_variance_theta must be positive, got {mixing_variance_theta}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n_a, n_b = control.sample_size, treatment.sample_size
    p_a, p_b = control.conversion_rate, treatment.conversion_rate

    n_eff = (n_a * n_b) / max(n_a + n_b, 1)
    p_bar = (control.conversions + treatment.conversions) / max(n_a + n_b, 1)
    sigma2 = max(p_bar * (1.0 - p_bar), 1e-6)

    v = n_eff / sigma2
    lambda_lr = math.sqrt(1.0 / (1.0 + v * mixing_variance_theta)) * math.exp(
        (v**2 * mixing_variance_theta * (p_b - p_a) ** 2)
        / (2.0 * (1.0 + v * mixing_variance_theta))
    )

    stopping_bound = 1.0 / alpha
    should_stop = lambda_lr >= stopping_bound
    p_val_bound = min(1.0, 1.0 / max(lambda_lr, 1e-12))

    return MSPRTResult(
        likelihood_ratio=float(lambda_lr),
        stopping_threshold=stopping_bound,
        should_stop_early=should_stop,
        nominal_alpha=alpha,
        estimated_p_value_bound=float(p_val_bound),
    )


def evaluate_bayesian_beta_binomial(
    control: ExperimentArm,
    treatment: ExperimentArm,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    mc_samples: int = 20000,
    seed: int = 42,
) -> BayesianBetaBinomialResult:
    """Monte Carlo comparison of two conjugate Beta posteriors.

    Sampling uses a local ``numpy`` generator; the same ``seed`` is
    deterministic. This is the only source of randomness in the functional
    package.

    Args:
        control: Control arm counts.
        treatment: Treatment arm counts.
        alpha_prior: Beta prior alpha (successes).
        beta_prior: Beta prior beta (failures).
        mc_samples: Number of posterior draws.
        seed: Generator seed.

    Returns:
        Immutable :class:`BayesianBetaBinomialResult`.

    Raises:
        ValueError: If priors are not positive or ``mc_samples`` is not positive.
    """
    if alpha_prior <= 0.0 or beta_prior <= 0.0:
        raise ValueError("Beta priors must be strictly positive")
    if mc_samples <= 0:
        raise ValueError(f"mc_samples must be positive, got {mc_samples}")

    a_post_a = alpha_prior + control.conversions
    b_post_a = beta_prior + control.sample_size - control.conversions

    a_post_b = alpha_prior + treatment.conversions
    b_post_b = beta_prior + treatment.sample_size - treatment.conversions

    rng = np.random.default_rng(seed)
    samples_a = rng.beta(a_post_a, b_post_a, size=mc_samples)
    samples_b = rng.beta(a_post_b, b_post_b, size=mc_samples)

    p_b_beats_a = float(np.mean(samples_b > samples_a))
    loss_control = float(np.mean(np.maximum(samples_b - samples_a, 0.0)))
    loss_treatment = float(np.mean(np.maximum(samples_a - samples_b, 0.0)))

    return BayesianBetaBinomialResult(
        p_treatment_beats_control=p_b_beats_a,
        expected_loss_control_if_ship=loss_control,
        expected_loss_treatment_if_ship=loss_treatment,
        posterior_a_mean=float(a_post_a / (a_post_a + b_post_a)),
        posterior_b_mean=float(a_post_b / (a_post_b + b_post_b)),
    )


def apply_cuped(
    y_post: tuple[float, ...] | list[float],
    x_pre: tuple[float, ...] | list[float],
) -> CupedResult:
    """CUPED: subtract the optimal linear adjustment using a pre-period covariate.

    Sample variances use ``ddof=1``. The reported reduction is floored at 0.

    Args:
        y_post: Post-experiment metric.
        x_pre: Pre-experiment covariate, same length as ``y_post``.

    Returns:
        Immutable :class:`CupedResult`.

    Raises:
        ValueError: If lengths differ or there are fewer than two observations.
    """
    y = np.asarray(y_post, dtype=np.float64)
    x = np.asarray(x_pre, dtype=np.float64)

    if len(y) != len(x):
        raise ValueError(f"Lengths must match: len(y)={len(y)} != len(x)={len(x)}")
    if len(y) < 2:
        raise ValueError("At least 2 observations required for CUPED variance reduction")

    cov_matrix = np.cov(x, y)
    var_x = float(cov_matrix[0, 0])
    cov_xy = float(cov_matrix[0, 1])
    var_y = float(cov_matrix[1, 1])

    theta = cov_xy / max(var_x, 1e-12)
    x_mean = float(np.mean(x))
    y_cuped = y - theta * (x - x_mean)

    var_cuped = float(np.var(y_cuped, ddof=1))
    var_reduction = max(0.0, 1.0 - (var_cuped / max(var_y, 1e-12)))

    return CupedResult(
        theta_optimal=float(theta),
        raw_variance=var_y,
        adjusted_variance=var_cuped,
        variance_reduction_pct=float(var_reduction * 100.0),
        adjusted_values=tuple(float(v) for v in y_cuped),
    )
