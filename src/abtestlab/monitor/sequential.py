"""Evaluate every look in an experiment, both the safe way and the naive way.

Each look is scored twice: with the always-valid mSPRT p-value (which may be
compared against alpha as often as you like) and with the fixed-horizon z-test
p-value (which may not). Carrying both is the point -- the gap between them is
what a peeking analyst is actually paying, and the stopping policy in
:mod:`abtestlab.monitor.policy` reports it per experiment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from abtestlab.monitor.looks import Look
from abtestlab.stats.core import msprt, z_test_two_proportions


@dataclass(frozen=True)
class LookEvaluation:
    """Both tests applied to the cumulative counts at one look.

    Attributes:
        look: The cumulative counts this evaluation describes.
        always_valid_p: mSPRT anytime-valid p-value.
        likelihood_ratio: mSPRT mixture likelihood ratio at this look.
        fixed_horizon_p: Two-proportion z-test p-value, valid only at a
            pre-planned sample size.
        relative_lift: Cumulative treatment lift over control.
        crosses_always_valid: Whether the anytime-valid p-value is below alpha.
        crosses_fixed: Whether the fixed-horizon p-value is below alpha.
    """

    look: Look
    always_valid_p: float
    likelihood_ratio: float
    fixed_horizon_p: float
    relative_lift: float
    crosses_always_valid: bool
    crosses_fixed: bool


def evaluate_look(look: Look, *, alpha: float) -> LookEvaluation:
    """Score one look with both the sequential and the fixed-horizon test.

    Args:
        look: Cumulative counts at this checkpoint.
        alpha: Significance level both crossings are judged against.

    Returns:
        The populated :class:`LookEvaluation`.

    Raises:
        ValueError: If ``alpha`` is not strictly inside ``(0, 1)``.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    counts = (
        look.conversions_control,
        look.n_control,
        look.conversions_treatment,
        look.n_treatment,
    )
    sequential = msprt(*counts)
    fixed = z_test_two_proportions(*counts)
    always_valid_p = float(sequential["always_valid_p"])
    fixed_p = float(fixed["p_value"])

    return LookEvaluation(
        look=look,
        always_valid_p=always_valid_p,
        likelihood_ratio=float(sequential["lambda"]),
        fixed_horizon_p=fixed_p,
        relative_lift=float(fixed["relative_lift"]),
        crosses_always_valid=always_valid_p < alpha,
        crosses_fixed=fixed_p < alpha,
    )


def evaluate_looks(looks: Sequence[Look], *, alpha: float) -> tuple[LookEvaluation, ...]:
    """Score an ordered look sequence.

    Args:
        looks: Cumulative looks in chronological order.
        alpha: Significance level both crossings are judged against.

    Returns:
        One :class:`LookEvaluation` per look, in the same order.
    """
    return tuple(evaluate_look(look, alpha=alpha) for look in looks)
