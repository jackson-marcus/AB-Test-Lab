"""Experimentation stats validated against theory and simulation."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from abtestlab.api.main import create_app
from abtestlab.stats.core import (
    bayes_beta_bernoulli,
    cuped_adjust,
    cuped_test,
    msprt,
    sample_size_two_proportions,
    z_test_two_proportions,
)


def test_sample_size_matches_textbook():
    # 5% baseline, 10% relative MDE, alpha .05, power .8 -> ~31k/arm (textbook ballpark).
    n = sample_size_two_proportions(0.05, 0.10)
    assert 28_000 <= n <= 34_000, f"got {n}"


def test_sample_size_shrinks_with_bigger_effect():
    assert sample_size_two_proportions(0.05, 0.3) < sample_size_two_proportions(0.05, 0.1)


def test_z_test_null_calibration():
    """Under the null, ~5% of simulated tests should be significant."""
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(400):
        a = rng.binomial(5000, 0.05)
        b = rng.binomial(5000, 0.05)
        if z_test_two_proportions(a, 5000, b, 5000)["significant"]:
            hits += 1
    rate = hits / 400
    assert 0.02 <= rate <= 0.09, f"false-positive rate {rate:.3f}"


def test_z_test_detects_real_effect():
    result = z_test_two_proportions(500, 10_000, 600, 10_000)
    assert result["significant"]
    assert result["relative_lift"] > 0.15


def test_msprt_null_rarely_stops():
    rng = np.random.default_rng(1)
    stops = 0
    for _ in range(200):
        a = rng.binomial(8000, 0.05)
        b = rng.binomial(8000, 0.05)
        if msprt(a, 8000, b, 8000)["can_stop"]:
            stops += 1
    assert stops / 200 <= 0.07, f"sequential false-stop rate {stops / 200:.3f}"


def test_msprt_stops_on_strong_effect():
    assert msprt(400, 8000, 560, 8000)["can_stop"]


def test_cuped_reduces_variance_proportionally():
    rng = np.random.default_rng(2)
    covariate = rng.normal(0, 1, 5000)
    metric = 0.7 * covariate + rng.normal(0, np.sqrt(1 - 0.49), 5000)
    _, reduction = cuped_adjust(metric, covariate)
    assert abs(reduction - 0.49) < 0.05  # rho^2 in theory


def test_cuped_test_tightens_pvalue():
    rng = np.random.default_rng(3)
    n = 1500
    pre_a, pre_b = rng.gamma(2, 10, n), rng.gamma(2, 10, n)
    metric_a = 0.7 * pre_a + 0.3 * rng.gamma(2, 10, n)
    metric_b = (0.7 * pre_b + 0.3 * rng.gamma(2, 10, n)) * 1.02
    result = cuped_test(metric_a, pre_a, metric_b, pre_b)
    assert result["variance_reduction"] > 0.3
    assert result["cuped_p_value"] <= result["raw_p_value"] + 1e-9


def test_bayes_confident_on_clear_winner():
    result = bayes_beta_bernoulli(500, 10_000, 650, 10_000)
    assert result["prob_treatment_beats_control"] > 0.99
    assert result["lift_p5"] > 0


def test_bayes_uncertain_on_null():
    result = bayes_beta_bernoulli(500, 10_000, 505, 10_000)
    assert 0.3 < result["prob_treatment_beats_control"] < 0.7


def test_api_roundtrip():
    client = TestClient(create_app())
    assert client.get("/health").json() == {"status": "ok"}
    payload = {
        "conversions_control": 500,
        "n_control": 10_000,
        "conversions_treatment": 600,
        "n_treatment": 10_000,
    }
    assert client.post("/analyze", json=payload).json()["significant"]
    assert client.post("/sequential", json=payload).status_code == 200
    assert client.post("/bayes", json=payload).status_code == 200
    r = client.post("/power", json={"baseline_rate": 0.05, "mde_relative": 0.1})
    assert r.json()["per_arm_sample_size"] > 10_000
    demo = client.post("/cuped-demo", json={"covariate_correlation": 0.7}).json()
    assert demo["variance_reduction"] > 0.2


def test_sample_size_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="Baseline"):
        sample_size_two_proportions(0.0, 0.1)
    with pytest.raises(ValueError, match="MDE"):
        sample_size_two_proportions(0.05, 0.0)


def test_core_rejects_conversions_exceeding_n():
    with pytest.raises(ValueError, match="cannot exceed"):
        z_test_two_proportions(11, 10, 1, 10)
    with pytest.raises(ValueError, match="cannot exceed"):
        msprt(1, 10, 11, 10)


def test_cuped_adjust_uses_consistent_ddof():
    """Regression: np.cov uses ddof=1; variance must use the same convention."""
    rng = np.random.default_rng(2)
    covariate = rng.normal(0, 1, 5000)
    metric = 0.7 * covariate + rng.normal(0, np.sqrt(1 - 0.49), 5000)
    _, reduction = cuped_adjust(metric, covariate)
    var_x = float(np.var(covariate, ddof=1))
    theta = float(np.cov(metric, covariate)[0, 1] / var_x)
    adjusted = metric - theta * (covariate - covariate.mean())
    expected = 1 - float(np.var(adjusted, ddof=1) / np.var(metric, ddof=1))
    assert abs(reduction - expected) < 1e-12
    assert 0.0 <= reduction <= 1.0


def test_msprt_library_and_api_formulas_are_not_interchangeable():
    """The two mSPRT implementations use different parameterizations by design."""
    from abtestlab.functional import ExperimentArm, evaluate_msprt

    library = evaluate_msprt(
        ExperimentArm("c", 400, 8000),
        ExperimentArm("t", 560, 8000),
    )
    api = msprt(400, 8000, 560, 8000)
    assert library.likelihood_ratio != pytest.approx(float(api["lambda"]))


def test_api_rejects_invalid_counts():
    client = TestClient(create_app())
    r = client.post(
        "/analyze",
        json={
            "conversions_control": 11,
            "n_control": 10,
            "conversions_treatment": 1,
            "n_treatment": 10,
        },
    )
    assert r.status_code == 422


def test_sample_size_rejects_an_alternative_rate_that_is_not_a_probability():
    """Regression: p_control * (1 + mde) >= 1 made p2 * (1 - p2) negative.

    That produced a silent nonsense answer for (0.6, 1.0) -- two units per arm --
    and a NaN that blew up as an unhandled 500 for (0.5, 1.5).
    """
    with pytest.raises(ValueError, match="not a probability"):
        sample_size_two_proportions(0.6, 1.0)
    with pytest.raises(ValueError, match="not a probability"):
        sample_size_two_proportions(0.5, 1.5)
    # 0.6 -> 0.9 is still a legal (enormous) alternative, so it must still compute,
    # and shrink as the effect grows.
    assert sample_size_two_proportions(0.6, 0.5) == 32
    assert sample_size_two_proportions(0.6, 0.5) > sample_size_two_proportions(0.6, 0.6)


def test_power_endpoint_reports_impossible_alternatives_instead_of_500():
    client = TestClient(create_app())
    response = client.post("/power", json={"baseline_rate": 0.6, "mde_relative": 1.0})
    assert response.status_code == 422
    assert "not a probability" in response.json()["detail"]
