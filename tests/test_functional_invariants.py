"""Property-based invariant tests for the Pure Functional Statistics Library in ABTestLab."""

from __future__ import annotations

import numpy as np
import pytest

from abtestlab.functional import (
    ExperimentArm,
    apply_cuped,
    compute_sample_size,
    evaluate_bayesian_beta_binomial,
    evaluate_fixed_horizon,
    evaluate_msprt,
)


def test_experiment_arm_invariants():
    # Valid
    arm = ExperimentArm(name="ctrl", conversions=50, sample_size=1000)
    assert arm.conversion_rate == 0.05

    # Negative conversions rejected
    with pytest.raises(ValueError, match="cannot be negative"):
        ExperimentArm(name="bad", conversions=-1, sample_size=100)

    # Conversions > sample size rejected
    with pytest.raises(ValueError, match="cannot exceed sample size"):
        ExperimentArm(name="bad", conversions=150, sample_size=100)


def test_sample_size_monotonicity():
    # Smaller MDE requires strictly larger sample size
    n_small_mde = compute_sample_size(p_baseline=0.10, mde_relative=0.05, alpha=0.05, power=0.80)
    n_large_mde = compute_sample_size(p_baseline=0.10, mde_relative=0.20, alpha=0.05, power=0.80)
    assert n_small_mde > n_large_mde


def test_fixed_horizon_symmetry_and_significance():
    ctrl = ExperimentArm(name="ctrl", conversions=100, sample_size=1000)  # 10%
    treat = ExperimentArm(name="treat", conversions=150, sample_size=1000)  # 15%

    res = evaluate_fixed_horizon(ctrl, treat, alpha=0.05)
    assert res.is_significant is True
    assert res.z_statistic > 0
    assert 0.0 <= res.p_value <= 1.0
    assert res.ci_lower < res.absolute_difference < res.ci_upper


def test_msprt_sequential_boundary():
    ctrl = ExperimentArm(name="ctrl", conversions=100, sample_size=1000)
    treat_massive = ExperimentArm(name="treat", conversions=300, sample_size=1000)

    res = evaluate_msprt(ctrl, treat_massive, mixing_variance_theta=0.05, alpha=0.05)
    assert res.should_stop_early is True
    assert res.likelihood_ratio >= res.stopping_threshold


def test_cuped_variance_reduction_invariant():
    np.random.seed(42)
    # Generate correlated pre and post variables
    x_pre = np.random.normal(50, 10, 500)
    noise = np.random.normal(0, 5, 500)
    y_post = 0.8 * x_pre + noise

    res = apply_cuped(y_post=tuple(y_post), x_pre=tuple(x_pre))
    # Fundamental CUPED Invariant: Adjusted Variance <= Raw Variance
    assert res.adjusted_variance <= res.raw_variance
    assert res.variance_reduction_pct > 30.0


def test_bayesian_beta_binomial_monotonicity():
    ctrl = ExperimentArm(name="ctrl", conversions=100, sample_size=1000)
    treat_weak = ExperimentArm(name="treat_weak", conversions=105, sample_size=1000)
    treat_strong = ExperimentArm(name="treat_strong", conversions=180, sample_size=1000)

    res_weak = evaluate_bayesian_beta_binomial(ctrl, treat_weak)
    res_strong = evaluate_bayesian_beta_binomial(ctrl, treat_strong)

    # Monotonicity Invariant
    assert res_strong.p_treatment_beats_control > res_weak.p_treatment_beats_control
    assert 0.0 <= res_strong.p_treatment_beats_control <= 1.0
    assert res_strong.expected_loss_control_if_ship > res_weak.expected_loss_control_if_ship
