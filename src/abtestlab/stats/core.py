"""Config-backed experimentation statistics used by the FastAPI layer.

This module is intentionally separate from :mod:`abtestlab.functional`. The
typed library is side-effect-free; these helpers read :func:`abtestlab.settings.get_config`
for alpha, mSPRT ``tau_squared``, and Bayesian priors. The two mSPRT
implementations do **not** use the same parameterization — do not treat their
outputs as interchangeable.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from abtestlab.settings import get_config


class ZTestResult(TypedDict):
    """JSON-serializable two-proportion z-test payload."""

    p_control: float
    p_treatment: float
    relative_lift: float
    z: float
    p_value: float
    ci_diff_95: list[float]
    significant: bool


class CupedTestResult(TypedDict):
    """JSON-serializable two-arm CUPED comparison payload."""

    raw_p_value: float
    cuped_p_value: float
    variance_reduction: float
    mean_diff: float


class BayesResult(TypedDict):
    """JSON-serializable Beta-Bernoulli decision payload."""

    prob_treatment_beats_control: float
    expected_loss_if_ship: float
    lift_p5: float
    lift_p50: float
    lift_p95: float


def _validate_counts(conv_a: int, n_a: int, conv_b: int, n_b: int) -> None:
    """Reject non-positive sample sizes or conversion counts outside ``[0, n]``.

    Args:
        conv_a: Control conversions.
        n_a: Control sample size.
        conv_b: Treatment conversions.
        n_b: Treatment sample size.

    Raises:
        ValueError: If any count is inconsistent.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError(f"Sample sizes must be positive, got n_a={n_a}, n_b={n_b}")
    if conv_a < 0 or conv_b < 0:
        raise ValueError(f"Conversions cannot be negative, got conv_a={conv_a}, conv_b={conv_b}")
    if conv_a > n_a or conv_b > n_b:
        raise ValueError(
            f"Conversions cannot exceed sample size "
            f"(control {conv_a}/{n_a}, treatment {conv_b}/{n_b})"
        )


def sample_size_two_proportions(
    p_control: float, mde_rel: float, alpha: float | None = None, power: float | None = None
) -> int:
    """Per-arm n for detecting a relative lift ``mde_rel`` on baseline ``p_control``.

    Args:
        p_control: Baseline conversion rate in ``(0, 1)``.
        mde_rel: Minimum detectable relative lift (strictly positive).
        alpha: Two-sided Type I error. Defaults to ``defaults.alpha`` in YAML.
        power: Target power. Defaults to ``defaults.power`` in YAML.

    Returns:
        Ceiling of the textbook two-proportion sample-size formula (per arm).

    Raises:
        ValueError: If ``p_control`` is not in ``(0, 1)``, ``mde_rel`` is not
            strictly positive, or the implied alternative rate
            ``p_control * (1 + mde_rel)`` reaches 1.0 (the variance term
            ``p2 * (1 - p2)`` goes negative there and the formula stops meaning
            anything).
    """
    if not (0.0 < p_control < 1.0):
        raise ValueError(f"Baseline proportion must be in (0, 1), got {p_control}")
    if mde_rel <= 0.0:
        raise ValueError(f"MDE relative must be strictly positive, got {mde_rel}")
    if p_control * (1.0 + mde_rel) >= 1.0:
        raise ValueError(
            f"Baseline {p_control} with relative MDE {mde_rel} implies an alternative rate "
            f"of {p_control * (1.0 + mde_rel):.4f}, which is not a probability"
        )
    cfg = get_config()["defaults"]
    alpha = cfg["alpha"] if alpha is None else alpha
    power = cfg["power"] if power is None else power
    p2 = p_control * (1 + mde_rel)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_bar = (p_control + p2) / 2
    numerator = (
        z_a * np.sqrt(2 * p_bar * (1 - p_bar))
        + z_b * np.sqrt(p_control * (1 - p_control) + p2 * (1 - p2))
    ) ** 2
    return int(np.ceil(numerator / (p2 - p_control) ** 2))


def z_test_two_proportions(conv_a: int, n_a: int, conv_b: int, n_b: int) -> ZTestResult:
    """Two-sided two-proportion z-test with a Wald CI using z = 1.96.

    Significance uses YAML ``defaults.alpha``. The CI remains a conventional
    95% interval (z = 1.96), which may disagree with ``alpha`` if that default
    is changed.

    Args:
        conv_a: Control conversions.
        n_a: Control sample size.
        conv_b: Treatment conversions.
        n_b: Treatment sample size.

    Returns:
        Rounded fields suitable for JSON responses.

    Raises:
        ValueError: If counts are invalid.
    """
    _validate_counts(conv_a, n_a, conv_b, n_b)
    p_a, p_b = conv_a / n_a, conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / max(se, 1e-12)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    lift = (p_b - p_a) / max(p_a, 1e-12)
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    return {
        "p_control": round(p_a, 5),
        "p_treatment": round(p_b, 5),
        "relative_lift": round(lift, 5),
        "z": round(float(z), 4),
        "p_value": round(float(p_value), 6),
        "ci_diff_95": [
            round(float(p_b - p_a - 1.96 * se_diff), 5),
            round(float(p_b - p_a + 1.96 * se_diff), 5),
        ],
        "significant": bool(p_value < get_config()["defaults"]["alpha"]),
    }


def msprt(conv_a: int, n_a: int, conv_b: int, n_b: int) -> dict[str, float | bool]:
    """Always-valid mixture SPRT for the difference in proportions (API formula).

    Uses ``sequential.tau_squared`` from YAML. This is **not** the same
    parameterization as :func:`abtestlab.functional.stats.evaluate_msprt`.

    Args:
        conv_a: Control conversions.
        n_a: Control sample size.
        conv_b: Treatment conversions.
        n_b: Treatment sample size.

    Returns:
        ``lambda``, always-valid p-value bound, and whether that bound is below
        YAML ``defaults.alpha``.

    Raises:
        ValueError: If counts are invalid.
    """
    _validate_counts(conv_a, n_a, conv_b, n_b)
    tau2 = get_config()["sequential"]["tau_squared"]
    p_a, p_b = conv_a / n_a, conv_b / n_b
    theta = p_b - p_a
    pooled = (conv_a + conv_b) / (n_a + n_b)
    var = pooled * (1 - pooled) * (1 / n_a + 1 / n_b)
    if var <= 0:
        return {"lambda": 1.0, "always_valid_p": 1.0, "can_stop": False}
    lam = float(np.sqrt(var / (var + tau2)) * np.exp(theta**2 * tau2 / (2 * var * (var + tau2))))
    p_av = min(1.0, 1.0 / lam)
    return {
        "lambda": round(lam, 4),
        "always_valid_p": round(p_av, 6),
        "can_stop": bool(p_av < get_config()["defaults"]["alpha"]),
    }


def cuped_adjust(
    metric: NDArray[np.floating], covariate: NDArray[np.floating]
) -> tuple[NDArray[np.floating], float]:
    """Adjust a metric with a pre-period covariate (CUPED).

    Covariance and variance both use ``ddof=1`` so theta and the reported
    reduction share the same second-moment convention.

    Args:
        metric: Post-period observations.
        covariate: Pre-period observations, same length as ``metric``.

    Returns:
        Tuple of ``(adjusted_metric, variance_reduction_fraction)``.

    Raises:
        ValueError: If lengths differ or there are fewer than two points.
    """
    if metric.shape != covariate.shape:
        raise ValueError(
            f"metric and covariate must have the same shape, got {metric.shape} vs {covariate.shape}"
        )
    if metric.size < 2:
        raise ValueError("At least 2 observations required for CUPED")
    var_x = float(np.var(covariate, ddof=1))
    theta = float(np.cov(metric, covariate)[0, 1] / max(var_x, 1e-12))
    adjusted = metric - theta * (covariate - covariate.mean())
    reduction = 1 - float(np.var(adjusted, ddof=1) / max(np.var(metric, ddof=1), 1e-12))
    return adjusted, reduction


def cuped_test(
    metric_a: NDArray[np.floating],
    cov_a: NDArray[np.floating],
    metric_b: NDArray[np.floating],
    cov_b: NDArray[np.floating],
) -> CupedTestResult:
    """Welch t-tests on raw vs CUPED-adjusted metrics for two arms.

    Theta is fit on the pooled (metric, covariate) pairs. Variances use
    ``ddof=1``.

    Args:
        metric_a: Control post-period metric.
        cov_a: Control pre-period covariate.
        metric_b: Treatment post-period metric.
        cov_b: Treatment pre-period covariate.

    Returns:
        Raw and CUPED p-values, pooled variance reduction, and raw mean difference.
    """
    metric = np.concatenate([metric_a, metric_b])
    covariate = np.concatenate([cov_a, cov_b])
    var_x = float(np.var(covariate, ddof=1))
    theta = float(np.cov(metric, covariate)[0, 1] / max(var_x, 1e-12))
    adj_a = metric_a - theta * (cov_a - covariate.mean())
    adj_b = metric_b - theta * (cov_b - covariate.mean())
    t_raw = stats.ttest_ind(metric_b, metric_a, equal_var=False)
    t_adj = stats.ttest_ind(adj_b, adj_a, equal_var=False)
    var_reduction = 1 - (np.var(adj_a, ddof=1) + np.var(adj_b, ddof=1)) / max(
        np.var(metric_a, ddof=1) + np.var(metric_b, ddof=1), 1e-12
    )
    return {
        "raw_p_value": round(float(t_raw.pvalue), 6),
        "cuped_p_value": round(float(t_adj.pvalue), 6),
        "variance_reduction": round(float(var_reduction), 4),
        "mean_diff": round(float(metric_b.mean() - metric_a.mean()), 5),
    }


def bayes_beta_bernoulli(
    conv_a: int, n_a: int, conv_b: int, n_b: int, seed: int = 0
) -> BayesResult:
    """Monte Carlo P(treatment > control) under conjugate Beta posteriors.

    Priors and draw count come from YAML ``bayes``.

    Args:
        conv_a: Control conversions.
        n_a: Control sample size.
        conv_b: Treatment conversions.
        n_b: Treatment sample size.
        seed: Generator seed for reproducibility.

    Returns:
        Probability to beat control, expected loss if shipping treatment, and
        posterior lift quantiles.

    Raises:
        ValueError: If counts are invalid.
    """
    _validate_counts(conv_a, n_a, conv_b, n_b)
    cfg = get_config()["bayes"]
    rng = np.random.default_rng(seed)
    post_a = rng.beta(cfg["prior_alpha"] + conv_a, cfg["prior_beta"] + n_a - conv_a, cfg["n_draws"])
    post_b = rng.beta(cfg["prior_alpha"] + conv_b, cfg["prior_beta"] + n_b - conv_b, cfg["n_draws"])
    prob_beat = float((post_b > post_a).mean())
    loss_b = float(np.maximum(post_a - post_b, 0).mean())
    lift = (post_b - post_a) / np.maximum(post_a, 1e-9)
    return {
        "prob_treatment_beats_control": round(prob_beat, 4),
        "expected_loss_if_ship": round(loss_b, 6),
        "lift_p5": round(float(np.quantile(lift, 0.05)), 5),
        "lift_p50": round(float(np.quantile(lift, 0.5)), 5),
        "lift_p95": round(float(np.quantile(lift, 0.95)), 5),
    }
