"""Behaviour of the look-by-look experiment monitor.

These tests are about decisions, not about objects existing: what the SRM gate
refuses, which look a stopping rule fires on, and how much error the naive
peeking rule actually accumulates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from abtestlab.api.main import create_app
from abtestlab.monitor import (
    Look,
    LookSequenceError,
    build_looks,
    check_srm,
    decide_always_valid,
    decide_naive_fixed,
    evaluate_looks,
    run_monitor,
)

EXAMPLE_LOG = Path(__file__).resolve().parents[1] / "data" / "example_looks.csv"
BROKEN_LOG = Path(__file__).resolve().parents[1] / "data" / "example_looks_srm.csv"


def read_csv_log(path: Path) -> list[dict[str, str]]:
    """Read a committed example log without pulling in pandas."""
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def synthetic_log(
    *,
    n_looks: int,
    per_look: int,
    baseline: float,
    lift: float,
    seed: int,
    treatment_per_look: int | None = None,
) -> tuple[Look, ...]:
    """Simulate one experiment as cumulative looks."""
    rng = np.random.default_rng(seed)
    n_t = per_look if treatment_per_look is None else treatment_per_look
    cum_c = np.cumsum(rng.binomial(per_look, baseline, n_looks))
    cum_t = np.cumsum(rng.binomial(n_t, baseline * (1 + lift), n_looks))
    return tuple(
        Look(
            index=i,
            label=f"day-{i + 1}",
            conversions_control=int(cum_c[i]),
            n_control=per_look * (i + 1),
            conversions_treatment=int(cum_t[i]),
            n_treatment=n_t * (i + 1),
        )
        for i in range(n_looks)
    )


def test_increments_accumulate_into_running_totals():
    rows = [
        {
            "conversions_control": 10,
            "n_control": 200,
            "conversions_treatment": 12,
            "n_treatment": 200,
        },
        {
            "conversions_control": 11,
            "n_control": 200,
            "conversions_treatment": 15,
            "n_treatment": 200,
        },
        {
            "conversions_control": 9,
            "n_control": 200,
            "conversions_treatment": 14,
            "n_treatment": 200,
        },
    ]
    looks = build_looks(rows, cumulative=False)
    assert [look.n_control for look in looks] == [200, 400, 600]
    assert [look.conversions_treatment for look in looks] == [12, 27, 41]
    assert looks[-1].rate_control == pytest.approx(30 / 600)


def test_cumulative_log_that_moves_backwards_is_rejected():
    """A re-run export that drops rows must not be silently averaged over."""
    rows = [
        {
            "label": "d1",
            "conversions_control": 50,
            "n_control": 1000,
            "conversions_treatment": 60,
            "n_treatment": 1000,
        },
        {
            "label": "d2",
            "conversions_control": 48,
            "n_control": 1900,
            "conversions_treatment": 130,
            "n_treatment": 2000,
        },
    ]
    with pytest.raises(LookSequenceError) as excinfo:
        build_looks(rows, cumulative=True)
    message = str(excinfo.value)
    assert "conversions_control" in message and "'d2'" in message and "50 -> 48" in message


def test_conversions_above_exposures_are_rejected():
    rows = [
        {
            "label": "d1",
            "conversions_control": 5,
            "n_control": 100,
            "conversions_treatment": 101,
            "n_treatment": 100,
        }
    ]
    with pytest.raises(LookSequenceError, match="101 treatment conversions"):
        build_looks(rows)


def test_empty_and_unexposed_logs_are_rejected():
    with pytest.raises(LookSequenceError, match="empty"):
        build_looks([])
    with pytest.raises(LookSequenceError, match="unexposed arm"):
        build_looks(
            [
                {
                    "conversions_control": 0,
                    "n_control": 0,
                    "conversions_treatment": 0,
                    "n_treatment": 10,
                }
            ]
        )


def test_srm_gate_catches_a_small_persistent_traffic_leak():
    """A 4% shortfall in one arm is invisible by eye and fatal to the estimate."""
    leaking = Look(0, "d14", 2800, 56_000, 3100, 53_760)
    verdict = check_srm(leaking)
    assert not verdict.healthy
    assert verdict.p_value < 1e-9
    assert verdict.observed_share_control > 0.5

    fair = Look(0, "d14", 2800, 56_000, 3100, 56_040)
    assert check_srm(fair).healthy


def test_srm_threshold_is_strict_enough_to_run_on_every_look():
    """At alpha=0.05 a healthy programme would drown in SRM false alarms."""
    rng = np.random.default_rng(5)
    total = 400
    strict_alarms = 0
    loose_alarms = 0
    for _ in range(total):
        n_control = int(rng.binomial(100_000, 0.5))
        look = Look(0, "d1", 5000, n_control, 5000, 100_000 - n_control)
        strict_alarms += int(not check_srm(look, threshold=0.001).healthy)
        loose_alarms += int(not check_srm(look, threshold=0.05).healthy)
    assert strict_alarms / total <= 0.01, f"strict gate fired {strict_alarms}/{total}"
    assert loose_alarms > strict_alarms, "the loose threshold should be visibly noisier"


def test_broken_assignment_refuses_to_report_a_lift():
    """Refusal means no number at all -- a confounded lift that looks usable is worse."""
    report = run_monitor(read_csv_log(BROKEN_LOG))
    assert report.status == "refused_srm"
    assert report.relative_lift is None
    assert report.prob_treatment_beats_control is None
    assert report.always_valid is None
    assert report.looks == ()
    assert "sample ratio mismatch" in report.reason


def test_healthy_log_stops_at_the_first_always_valid_crossing():
    report = run_monitor(read_csv_log(EXAMPLE_LOG))
    assert report.status == "ship"
    assert report.always_valid.stopped
    assert report.always_valid.stop_label == "day-03"
    assert report.always_valid.looks_saved == 11
    assert report.decision_look == "day-03"
    assert report.relative_lift > 0.15


def test_always_valid_p_value_is_not_monotone():
    """The committed example crosses at day-03 and drifts back above alpha at day-04.

    This is the whole reason the monitor reports the first crossing rather than
    the current status: a dashboard that only shows "today's p-value" would have
    lost this result on day 4 and re-found it on day 5.
    """
    looks = build_looks(read_csv_log(EXAMPLE_LOG))
    evaluations = evaluate_looks(looks, alpha=0.05)
    p_values = [item.always_valid_p for item in evaluations]
    crossings = [i for i, p in enumerate(p_values) if p < 0.05]
    assert crossings, "example log is supposed to cross"
    first = crossings[0]
    rebounds = [i for i in range(first + 1, len(p_values)) if p_values[i] >= 0.05]
    assert rebounds, "example log no longer demonstrates the rebound; regenerate it"
    assert p_values[first] < 0.05 <= p_values[rebounds[0]]


def test_decision_reports_the_first_crossing_even_after_a_rebound():
    looks = build_looks(read_csv_log(EXAMPLE_LOG))
    evaluations = evaluate_looks(looks, alpha=0.05)
    truncated = evaluations[:4]  # day-03 crosses, day-04 rebounds
    decision = decide_always_valid(truncated, alpha=0.05)
    assert decision.stopped
    assert decision.stop_index == 2
    assert decision.recovered_after_crossing
    assert decision.p_at_final_look >= 0.05
    assert decision.p_at_stop < 0.05


def test_anytime_valid_p_is_never_more_permissive_than_the_fixed_horizon_p():
    """The mixture pays a penalty for the right to peek; it must never be a discount."""
    rng = np.random.default_rng(9)
    for seed in rng.integers(0, 10_000, 40):
        looks = synthetic_log(n_looks=6, per_look=2500, baseline=0.06, lift=0.15, seed=int(seed))
        for item in evaluate_looks(looks, alpha=0.05):
            assert item.always_valid_p >= item.fixed_horizon_p - 1e-9, (
                f"always-valid p {item.always_valid_p} beat fixed p {item.fixed_horizon_p} "
                f"at {item.look.label}"
            )


def test_naive_peeking_inflates_the_false_positive_rate():
    """Under a true null, reading the fixed-horizon p at every look blows past alpha."""
    trials, alpha = 300, 0.05
    naive_fires = 0
    safe_fires = 0
    for seed in range(trials):
        looks = synthetic_log(n_looks=8, per_look=1500, baseline=0.05, lift=0.0, seed=seed)
        evaluations = evaluate_looks(looks, alpha=alpha)
        naive_fires += int(decide_naive_fixed(evaluations, alpha=alpha).stopped)
        safe_fires += int(decide_always_valid(evaluations, alpha=alpha).stopped)
    naive_rate, safe_rate = naive_fires / trials, safe_fires / trials
    assert naive_rate > 2 * alpha, f"naive peeking only reached {naive_rate:.3f}"
    assert safe_rate <= alpha, f"always-valid rule exceeded its own bound at {safe_rate:.3f}"
    assert naive_rate > safe_rate


def test_stopping_rules_reject_an_empty_sequence():
    with pytest.raises(ValueError, match="empty look sequence"):
        decide_always_valid([])


def test_monitor_endpoint_reports_both_rules():
    client = TestClient(create_app())
    body = client.post(
        "/monitor",
        json={"looks": read_csv_log(EXAMPLE_LOG), "cumulative": True, "alpha": 0.05},
    ).json()
    assert body["status"] == "ship"
    assert body["always_valid"]["stop_label"] == "day-03"
    assert body["naive_fixed"]["stop_label"] == "day-03"
    assert len(body["looks"]) == 14
    assert body["looks"][3]["always_valid_p"] >= 0.05 > body["looks"][2]["always_valid_p"]


def test_monitor_endpoint_refuses_a_broken_split():
    client = TestClient(create_app())
    body = client.post("/monitor", json={"looks": read_csv_log(BROKEN_LOG)}).json()
    assert body["status"] == "refused_srm"
    assert body["relative_lift"] is None
    assert body["looks"] == []
    assert body["srm"]["p_value"] < 0.001


def test_monitor_endpoint_rejects_a_log_that_moves_backwards():
    client = TestClient(create_app())
    response = client.post(
        "/monitor",
        json={
            "looks": [
                {
                    "label": "d1",
                    "conversions_control": 50,
                    "n_control": 1000,
                    "conversions_treatment": 60,
                    "n_treatment": 1000,
                },
                {
                    "label": "d2",
                    "conversions_control": 50,
                    "n_control": 900,
                    "conversions_treatment": 130,
                    "n_treatment": 2000,
                },
            ]
        },
    )
    assert response.status_code == 422
    assert "went backwards" in response.json()["detail"]


def test_monitor_endpoint_keeps_running_on_a_flat_experiment():
    client = TestClient(create_app())
    looks = [
        {
            "label": f"day-{i}",
            "conversions_control": 100 * i,
            "n_control": 2000 * i,
            "conversions_treatment": 101 * i,
            "n_treatment": 2000 * i,
        }
        for i in range(1, 6)
    ]
    body = client.post("/monitor", json={"looks": looks}).json()
    assert body["status"] == "keep_running"
    assert body["always_valid"]["stopped"] is False
    assert body["decision_look"] == "day-5"


def test_cli_srm_exit_code_gates_a_broken_export():
    from typer.testing import CliRunner

    from abtestlab.cli import app

    runner = CliRunner()
    assert runner.invoke(app, ["srm", str(EXAMPLE_LOG)]).exit_code == 0
    assert runner.invoke(app, ["srm", str(BROKEN_LOG)]).exit_code == 1


def test_cli_monitor_prints_the_crossing_look():
    from typer.testing import CliRunner

    from abtestlab.cli import app

    result = CliRunner().invoke(app, ["monitor", str(EXAMPLE_LOG)])
    assert result.exit_code == 0
    assert "day-03" in result.stdout
    assert "SHIP" in result.stdout
