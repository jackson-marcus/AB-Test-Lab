"""Pure Functional Statistics Package for ABTestLab.

Side-effect-free, algebraic experimentation statistics and property invariants.
"""

from abtestlab.functional.stats import (
    apply_cuped,
    compute_sample_size,
    evaluate_bayesian_beta_binomial,
    evaluate_fixed_horizon,
    evaluate_msprt,
)
from abtestlab.functional.types import (
    BayesianBetaBinomialResult,
    CupedResult,
    ExperimentArm,
    FixedHorizonResult,
    MSPRTResult,
)

__all__ = [
    "BayesianBetaBinomialResult",
    "CupedResult",
    "ExperimentArm",
    "FixedHorizonResult",
    "MSPRTResult",
    "apply_cuped",
    "compute_sample_size",
    "evaluate_bayesian_beta_binomial",
    "evaluate_fixed_horizon",
    "evaluate_msprt",
]
