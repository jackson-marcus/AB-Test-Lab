"""Experimentation statistics, hand-rolled and unit-tested against theory.

- Power / sample-size for two-proportion tests
- Fixed-horizon z-test
- Always-valid mSPRT sequential test (mixture likelihood ratio)
- CUPED variance reduction with a pre-period covariate
- Beta-Bernoulli Bayesian analysis (P(beat control), expected loss)
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from abtestlab.settings import get_config

# ---------- power / sample size ----------


def sample_size_two_proportions(
    p_control: float, mde_rel: float, alpha: float | None = None, power: float | None = None
) -> int:
    """Per-arm n for detecting a relative lift `mde_rel` on baseline p_control."""
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


# ---------- fixed horizon ----------


def z_test_two_proportions(conv_a: int, n_a: int, conv_b: int, n_b: int) -> dict:
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


# ---------- sequential (mSPRT) ----------


def msprt(conv_a: int, n_a: int, conv_b: int, n_b: int) -> dict:
    """Always-valid mixture SPRT for the difference in proportions.

    Lambda_n = sqrt(V/(V+tau2)) * exp(tau2 * theta_hat^2 * n_eff / (2V(V+tau2)/n_eff))
    Using the normal-approximation form on the difference estimate.
    """
    tau2 = get_config()["sequential"]["tau_squared"]
    p_a, p_b = conv_a / max(n_a, 1), conv_b / max(n_b, 1)
    theta = p_b - p_a
    pooled = (conv_a + conv_b) / max(n_a + n_b, 1)
    var = pooled * (1 - pooled) * (1 / max(n_a, 1) + 1 / max(n_b, 1))
    if var <= 0:
        return {"lambda": 1.0, "always_valid_p": 1.0, "can_stop": False}
    lam = float(np.sqrt(var / (var + tau2)) * np.exp(theta**2 * tau2 / (2 * var * (var + tau2))))
    p_av = min(1.0, 1.0 / lam)
    return {
        "lambda": round(lam, 4),
        "always_valid_p": round(p_av, 6),
        "can_stop": bool(p_av < get_config()["defaults"]["alpha"]),
    }


# ---------- CUPED ----------


def cuped_adjust(metric: np.ndarray, covariate: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (adjusted metric, variance reduction fraction)."""
    theta = float(np.cov(metric, covariate)[0, 1] / max(np.var(covariate), 1e-12))
    adjusted = metric - theta * (covariate - covariate.mean())
    reduction = 1 - float(np.var(adjusted) / max(np.var(metric), 1e-12))
    return adjusted, reduction


def cuped_test(
    metric_a: np.ndarray, cov_a: np.ndarray, metric_b: np.ndarray, cov_b: np.ndarray
) -> dict:
    metric = np.concatenate([metric_a, metric_b])
    covariate = np.concatenate([cov_a, cov_b])
    theta = float(np.cov(metric, covariate)[0, 1] / max(np.var(covariate), 1e-12))
    adj_a = metric_a - theta * (cov_a - covariate.mean())
    adj_b = metric_b - theta * (cov_b - covariate.mean())
    t_raw = stats.ttest_ind(metric_b, metric_a, equal_var=False)
    t_adj = stats.ttest_ind(adj_b, adj_a, equal_var=False)
    var_reduction = 1 - (np.var(adj_a) + np.var(adj_b)) / max(
        np.var(metric_a) + np.var(metric_b), 1e-12
    )
    return {
        "raw_p_value": round(float(t_raw.pvalue), 6),
        "cuped_p_value": round(float(t_adj.pvalue), 6),
        "variance_reduction": round(float(var_reduction), 4),
        "mean_diff": round(float(metric_b.mean() - metric_a.mean()), 5),
    }


# ---------- Bayesian ----------


def bayes_beta_bernoulli(conv_a: int, n_a: int, conv_b: int, n_b: int, seed: int = 0) -> dict:
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
