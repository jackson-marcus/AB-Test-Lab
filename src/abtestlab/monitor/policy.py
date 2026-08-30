"""Stopping policies over an evaluated look sequence.

The anytime-valid guarantee is a statement about a *stopping rule*: stop the
first time the mSPRT p-value drops below alpha and your Type-I error is bounded
by alpha no matter how many looks you took. It is emphatically not a statement
about the p-value at the last look. Lambda_t is a martingale, so it can cross
and then drift back up -- a monitor that only reports "the current p-value is
0.09" throws away results the rule already entitled you to call.

That is why :class:`StopDecision` records the *first* crossing and flags
``recovered_after_crossing`` when the final look no longer looks significant.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from abtestlab.monitor.sequential import LookEvaluation

ALWAYS_VALID = "always_valid"
NAIVE_FIXED = "naive_fixed"


@dataclass(frozen=True)
class StopDecision:
    """When a stopping rule would have called an experiment.

    Attributes:
        rule: ``"always_valid"`` (mSPRT, peek-safe) or ``"naive_fixed"``
            (fixed-horizon z-test read at every look, which is not).
        stopped: Whether the rule fired anywhere in the sequence.
        stop_index: Zero-based index of the first crossing, or ``None``.
        stop_label: Label of the first crossing, or ``None``.
        looks_taken: Looks the rule consumed before firing (all of them if it
            never fired).
        looks_saved: Looks after the crossing that would not have been needed.
        p_at_stop: Rule's p-value at the crossing, or ``None``.
        p_at_final_look: Rule's p-value at the last look in the log.
        recovered_after_crossing: The rule fired earlier but the final look is
            back above alpha. Only meaningful for the always-valid rule, where
            the earlier crossing is still a legitimate stop.
    """

    rule: str
    stopped: bool
    stop_index: int | None
    stop_label: str | None
    looks_taken: int
    looks_saved: int
    p_at_stop: float | None
    p_at_final_look: float
    recovered_after_crossing: bool


def _decide(
    evaluations: Sequence[LookEvaluation],
    *,
    rule: str,
    alpha: float,
) -> StopDecision:
    """Find the first crossing under one rule.

    Args:
        evaluations: Scored looks in chronological order.
        rule: ``ALWAYS_VALID`` or ``NAIVE_FIXED``.
        alpha: Significance level.

    Returns:
        The resulting :class:`StopDecision`.

    Raises:
        ValueError: If ``evaluations`` is empty or ``rule`` is unknown.
    """
    if not evaluations:
        raise ValueError("cannot decide on an empty look sequence")
    if rule == ALWAYS_VALID:
        p_values = [item.always_valid_p for item in evaluations]
    elif rule == NAIVE_FIXED:
        p_values = [item.fixed_horizon_p for item in evaluations]
    else:
        raise ValueError(f"unknown stopping rule {rule!r}")

    total = len(evaluations)
    final_p = p_values[-1]
    for position, p_value in enumerate(p_values):
        if p_value < alpha:
            return StopDecision(
                rule=rule,
                stopped=True,
                stop_index=position,
                stop_label=evaluations[position].look.label,
                looks_taken=position + 1,
                looks_saved=total - position - 1,
                p_at_stop=p_value,
                p_at_final_look=final_p,
                recovered_after_crossing=final_p >= alpha,
            )

    return StopDecision(
        rule=rule,
        stopped=False,
        stop_index=None,
        stop_label=None,
        looks_taken=total,
        looks_saved=0,
        p_at_stop=None,
        p_at_final_look=final_p,
        recovered_after_crossing=False,
    )


def decide_always_valid(
    evaluations: Sequence[LookEvaluation], *, alpha: float = 0.05
) -> StopDecision:
    """Apply the peek-safe mSPRT stopping rule.

    Args:
        evaluations: Scored looks in chronological order.
        alpha: Significance level.

    Returns:
        The first-crossing decision under the anytime-valid p-value.
    """
    return _decide(evaluations, rule=ALWAYS_VALID, alpha=alpha)


def decide_naive_fixed(
    evaluations: Sequence[LookEvaluation], *, alpha: float = 0.05
) -> StopDecision:
    """Apply the rule an unguarded dashboard invites: stop at the first ``p < alpha``.

    This is reported alongside the safe rule so a user can see, for their own
    experiment, where the two disagree. It is not a rule this package endorses.

    Args:
        evaluations: Scored looks in chronological order.
        alpha: Significance level.

    Returns:
        The first-crossing decision under the fixed-horizon p-value.
    """
    return _decide(evaluations, rule=NAIVE_FIXED, alpha=alpha)
