"""Replay an experiment's observation log look by look and decide what to do.

Order matters and is not cosmetic. The SRM gate runs on exposure counts *before*
any conversion number is read, and a failed gate short-circuits the run: no
lift, no p-value, no recommendation. Reporting an effect from a broken
assignment is worse than reporting nothing, because the number looks usable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from abtestlab.monitor.guards import DEFAULT_SRM_THRESHOLD, SrmVerdict, check_srm
from abtestlab.monitor.looks import Look, build_looks
from abtestlab.monitor.policy import StopDecision, decide_always_valid, decide_naive_fixed
from abtestlab.monitor.sequential import LookEvaluation, evaluate_looks
from abtestlab.stats.core import bayes_beta_bernoulli

SHIP = "ship"
ROLL_BACK = "roll_back"
KEEP_RUNNING = "keep_running"
REFUSED_SRM = "refused_srm"


@dataclass(frozen=True)
class MonitorReport:
    """Result of replaying one experiment log.

    Attributes:
        status: ``"ship"``, ``"roll_back"``, ``"keep_running"``, or
            ``"refused_srm"``.
        reason: One sentence explaining the status.
        srm: Verdict of the assignment-health gate on the final look.
        looks: Per-look evaluations, empty when the SRM gate refused the run.
        always_valid: Decision under the peek-safe rule, or ``None`` if refused.
        naive_fixed: Decision under the naive dashboard rule, or ``None``.
        decision_look: Label of the look the recommendation is based on.
        relative_lift: Cumulative lift at the decision look.
        prob_treatment_beats_control: Posterior probability at the decision look.
        expected_loss_if_ship: Posterior expected loss at the decision look.
    """

    status: str
    reason: str
    srm: SrmVerdict
    looks: tuple[LookEvaluation, ...]
    always_valid: StopDecision | None
    naive_fixed: StopDecision | None
    decision_look: str | None
    relative_lift: float | None
    prob_treatment_beats_control: float | None
    expected_loss_if_ship: float | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view for the API and CLI.

        Returns:
            Nested plain-Python mapping of the whole report.
        """
        return {
            "status": self.status,
            "reason": self.reason,
            "srm": asdict(self.srm),
            "looks": [
                {
                    "index": item.look.index,
                    "label": item.look.label,
                    "n_control": item.look.n_control,
                    "n_treatment": item.look.n_treatment,
                    "rate_control": round(item.look.rate_control, 5),
                    "rate_treatment": round(item.look.rate_treatment, 5),
                    "relative_lift": round(item.relative_lift, 5),
                    "always_valid_p": round(item.always_valid_p, 6),
                    "fixed_horizon_p": round(item.fixed_horizon_p, 6),
                    "crosses_always_valid": item.crosses_always_valid,
                    "crosses_fixed": item.crosses_fixed,
                }
                for item in self.looks
            ],
            "always_valid": None if self.always_valid is None else asdict(self.always_valid),
            "naive_fixed": None if self.naive_fixed is None else asdict(self.naive_fixed),
            "decision_look": self.decision_look,
            "relative_lift": self.relative_lift,
            "prob_treatment_beats_control": self.prob_treatment_beats_control,
            "expected_loss_if_ship": self.expected_loss_if_ship,
        }


def _refuse(srm: SrmVerdict, final: Look) -> MonitorReport:
    """Build the report used when the assignment gate fails.

    Args:
        srm: The failing verdict.
        final: Final look, used to quote the observed split.

    Returns:
        A refusal report with no effect estimate attached.
    """
    reason = (
        f"sample ratio mismatch: control took {srm.observed_share_control:.4%} of "
        f"{final.total_units:,} units against an intended {srm.expected_share_control:.2%} "
        f"(chi-square p={srm.p_value:.2e} < {srm.threshold:g}). "
        "Assignment is broken, so no lift is reported."
    )
    return MonitorReport(
        status=REFUSED_SRM,
        reason=reason,
        srm=srm,
        looks=(),
        always_valid=None,
        naive_fixed=None,
        decision_look=None,
        relative_lift=None,
        prob_treatment_beats_control=None,
        expected_loss_if_ship=None,
    )


def run_monitor(
    rows: Sequence[Mapping[str, Any]],
    *,
    cumulative: bool = True,
    alpha: float = 0.05,
    expected_share_control: float = 0.5,
    srm_threshold: float = DEFAULT_SRM_THRESHOLD,
) -> MonitorReport:
    """Replay an observation log and return a peek-safe recommendation.

    Args:
        rows: Chronological log rows (see :func:`abtestlab.monitor.looks.build_looks`).
        cumulative: Whether rows carry running totals or per-period increments.
        alpha: Significance level for both stopping rules.
        expected_share_control: Intended control share for the SRM gate.
        srm_threshold: SRM p-value below which the run is refused.

    Returns:
        The :class:`MonitorReport` for this log.

    Raises:
        LookSequenceError: If the log cannot be read as an ordered experiment.
        ValueError: If ``alpha``, ``expected_share_control``, or ``srm_threshold``
            is outside ``(0, 1)``.
    """
    looks = build_looks(rows, cumulative=cumulative)
    srm = check_srm(
        looks[-1],
        expected_share_control=expected_share_control,
        threshold=srm_threshold,
    )
    if not srm.healthy:
        return _refuse(srm, looks[-1])

    evaluations = evaluate_looks(looks, alpha=alpha)
    always_valid = decide_always_valid(evaluations, alpha=alpha)
    naive_fixed = decide_naive_fixed(evaluations, alpha=alpha)

    decision_index = always_valid.stop_index if always_valid.stopped else len(evaluations) - 1
    decided = evaluations[decision_index]
    posterior = bayes_beta_bernoulli(
        decided.look.conversions_control,
        decided.look.n_control,
        decided.look.conversions_treatment,
        decided.look.n_treatment,
    )

    if not always_valid.stopped:
        status = KEEP_RUNNING
        reason = (
            f"after {len(evaluations)} looks the always-valid p-value is "
            f"{always_valid.p_at_final_look:.4f}, still above alpha={alpha:g}; keep collecting."
        )
    elif decided.relative_lift >= 0.0:
        status = SHIP
        reason = (
            f"the always-valid p-value crossed alpha={alpha:g} at look "
            f"{always_valid.stop_label!r} with a {decided.relative_lift:+.2%} lift; "
            f"{always_valid.looks_saved} later look(s) were not needed."
        )
    else:
        status = ROLL_BACK
        reason = (
            f"the always-valid p-value crossed alpha={alpha:g} at look "
            f"{always_valid.stop_label!r} on a {decided.relative_lift:+.2%} lift; "
            "the treatment is significantly worse."
        )

    if always_valid.recovered_after_crossing:
        reason += (
            f" Note: the anytime-valid p-value drifted back to "
            f"{always_valid.p_at_final_look:.4f} by the final look. The crossing still "
            "stands: the rule is to stop at the first crossing, not to read the last one."
        )

    return MonitorReport(
        status=status,
        reason=reason,
        srm=srm,
        looks=evaluations,
        always_valid=always_valid,
        naive_fixed=naive_fixed,
        decision_look=decided.look.label,
        relative_lift=round(decided.relative_lift, 5),
        prob_treatment_beats_control=posterior["prob_treatment_beats_control"],
        expected_loss_if_ship=posterior["expected_loss_if_ship"],
    )
