"""Sample-ratio-mismatch gate.

Every statistic in this repo compares two arms on the assumption that units were
split between them by a fair coin. When that assumption breaks -- a bot filter
that only runs on one variant, a redirect that drops traffic, a bad bucketing
hash -- the arms are no longer exchangeable and the lift estimate is measuring
the leak rather than the feature. The standard defence is a chi-square goodness
-of-fit test on the *exposure* counts alone, run before anyone looks at the
conversion numbers.

The threshold is deliberately much stricter than the analysis alpha: an SRM
check runs on every experiment on every look, so at 0.05 a healthy programme
would be drowning in false alarms.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy import stats

from abtestlab.monitor.looks import Look

DEFAULT_SRM_THRESHOLD = 0.001


@dataclass(frozen=True)
class SrmVerdict:
    """Outcome of the sample-ratio-mismatch check at one look.

    Attributes:
        expected_share_control: Share of units the design intended for control.
        observed_share_control: Share of units control actually received.
        chi_square: One-degree-of-freedom goodness-of-fit statistic.
        p_value: Probability of a split this lopsided under a fair assignment.
        threshold: p-value below which assignment is declared broken.
        healthy: ``True`` when the split is consistent with the design.
    """

    expected_share_control: float
    observed_share_control: float
    chi_square: float
    p_value: float
    threshold: float
    healthy: bool


def check_srm(
    look: Look,
    *,
    expected_share_control: float = 0.5,
    threshold: float = DEFAULT_SRM_THRESHOLD,
) -> SrmVerdict:
    """Test whether the exposure split at ``look`` matches the intended design.

    Args:
        look: Cumulative look to test. Only exposure counts are used.
        expected_share_control: Intended share of units for control, in ``(0, 1)``.
        threshold: p-value below which the split is declared broken.

    Returns:
        An :class:`SrmVerdict` for this look.

    Raises:
        ValueError: If ``expected_share_control`` is not strictly inside ``(0, 1)``
            or ``threshold`` is not strictly inside ``(0, 1)``.
    """
    if not (0.0 < expected_share_control < 1.0):
        raise ValueError(f"expected_share_control must be in (0, 1), got {expected_share_control}")
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    total = look.total_units
    expected_control = total * expected_share_control
    expected_treatment = total * (1.0 - expected_share_control)
    chi_square = (look.n_control - expected_control) ** 2 / expected_control + (
        look.n_treatment - expected_treatment
    ) ** 2 / expected_treatment
    p_value = float(stats.chi2.sf(chi_square, df=1))

    return SrmVerdict(
        expected_share_control=expected_share_control,
        observed_share_control=look.n_control / total,
        chi_square=float(chi_square),
        p_value=p_value,
        threshold=threshold,
        healthy=p_value >= threshold,
    )
