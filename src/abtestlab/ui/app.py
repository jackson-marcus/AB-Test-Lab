"""Streamlit demo: plan sample size, analyze counts, and run a CUPED simulation.

Talks to the FastAPI process at ``ABTESTLAB_API_URL`` (default
``http://localhost:8240``).
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("ABTESTLAB_API_URL", "http://localhost:8240")

st.set_page_config(page_title="abtestlab", page_icon="🧪", layout="wide")
st.title("🧪 abtestlab")
st.caption("Power planning, fixed + always-valid sequential tests, CUPED, Bayesian decisions")


def _ok() -> bool:
    """Return True if the API health endpoint responds with HTTP 200."""
    try:
        return httpx.get(f"{API_URL}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


def _post_json(path: str, payload: dict, timeout: float) -> dict:
    """POST JSON and return the decoded body.

    Args:
        path: URL path beginning with ``/``.
        payload: JSON-serializable body.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON object.

    Raises:
        httpx.HTTPError: On transport failure or non-success status.
    """
    response = httpx.post(f"{API_URL}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


if not _ok():
    st.error(f"API not reachable at {API_URL}. Start it with `make api`.")
    st.stop()

tab_plan, tab_monitor, tab_analyze, tab_cuped = st.tabs(
    ["Plan (power)", "Monitor a running test", "Analyze", "CUPED demo"]
)

with tab_plan:
    c1, c2 = st.columns(2)
    baseline = c1.slider("Baseline conversion", 0.005, 0.5, 0.05, 0.005)
    mde = c2.slider("Minimum detectable relative lift", 0.01, 0.5, 0.10, 0.01)
    try:
        body = _post_json(
            "/power",
            {"baseline_rate": baseline, "mde_relative": mde},
            timeout=30,
        )
        st.metric("Sample size per arm", f"{body['per_arm_sample_size']:,}")
        st.caption(f"Total {body['total']:,} at alpha=0.05, power=0.8")
    except httpx.HTTPError as exc:
        st.error(f"Power request failed: {exc}")

with tab_analyze:
    c1, c2, c3, c4 = st.columns(4)
    ca = c1.number_input("Control conversions", 0, 10_000_000, 500)
    na = c2.number_input("Control n", 1, 10_000_000, 10_000)
    cb = c3.number_input("Treatment conversions", 0, 10_000_000, 565)
    nb = c4.number_input("Treatment n", 1, 10_000_000, 10_000)
    payload = {
        "conversions_control": int(ca),
        "n_control": int(na),
        "conversions_treatment": int(cb),
        "n_treatment": int(nb),
    }
    if st.button("Analyze", type="primary"):
        try:
            fixed = _post_json("/analyze", payload, timeout=30)
            seq = _post_json("/sequential", payload, timeout=30)
            bayes = _post_json("/bayes", payload, timeout=60)
        except httpx.HTTPError as exc:
            st.error(f"Analyze request failed: {exc}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Relative lift",
                f"{fixed['relative_lift']:+.2%}",
                delta="significant" if fixed["significant"] else "not significant",
            )
            c2.metric("Fixed-horizon p", f"{fixed['p_value']:.4f}")
            c3.metric(
                "Always-valid p (mSPRT)",
                f"{seq['always_valid_p']:.4f}",
                delta="can stop" if seq["can_stop"] else "keep collecting",
            )
            st.markdown(
                f"**Bayesian:** P(beat control) = **{bayes['prob_treatment_beats_control']:.1%}**, "
                f"expected loss if shipped = {bayes['expected_loss_if_ship']:.5f}, "
                f"lift 90% CI [{bayes['lift_p5']:+.2%}, {bayes['lift_p95']:+.2%}]"
            )
            st.caption(
                "The mSPRT p-value is valid under continuous peeking; the fixed-horizon "
                "p-value is only valid at the planned sample size. Sequential results use "
                "the API formula in stats.core, not the functional library mSPRT."
            )

with tab_cuped:
    c1, c2 = st.columns(2)
    rho = c1.slider("Pre-period correlation", 0.0, 0.95, 0.6, 0.05)
    lift = c2.slider("True lift", 0.0, 0.10, 0.03, 0.005)
    try:
        body = _post_json(
            "/cuped-demo",
            {"n_per_arm": 2000, "true_lift": lift, "covariate_correlation": rho},
            timeout=30,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw p-value", f"{body['raw_p_value']:.4f}")
        c2.metric("CUPED p-value", f"{body['cuped_p_value']:.4f}")
        c3.metric("Variance reduction", f"{body['variance_reduction']:.0%}")
        st.caption("Same data, same effect — CUPED just removes pre-period noise.")
    except httpx.HTTPError as exc:
        st.error(f"CUPED demo request failed: {exc}")

with tab_monitor:
    st.markdown(
        "Paste or edit a chronological log of **cumulative** per-arm counts. Each row is one "
        "look at the dashboard. The monitor checks assignment health first, then replays the "
        "log under the peek-safe rule and under the rule an unguarded dashboard invites."
    )
    default_log = Path("data/example_looks.csv")
    if "monitor_log" not in st.session_state:
        st.session_state["monitor_log"] = (
            pd.read_csv(default_log)
            if default_log.is_file()
            else pd.DataFrame(
                {
                    "label": [f"day-{i:02d}" for i in range(1, 5)],
                    "conversions_control": [200, 405, 600, 812],
                    "n_control": [4000, 8000, 12000, 16000],
                    "conversions_treatment": [214, 441, 668, 902],
                    "n_treatment": [4000, 8000, 12000, 16000],
                }
            )
        )
    edited = st.data_editor(
        st.session_state["monitor_log"], num_rows="dynamic", width="stretch", key="monitor_editor"
    )
    c1, c2 = st.columns(2)
    monitor_alpha = c1.slider("Alpha", 0.01, 0.20, 0.05, 0.01, key="monitor_alpha")
    control_share = c2.slider("Intended control share", 0.05, 0.95, 0.5, 0.05)

    if st.button("Replay log", type="primary"):
        payload = {
            "looks": edited.fillna(0).to_dict(orient="records"),
            "cumulative": True,
            "alpha": monitor_alpha,
            "expected_share_control": control_share,
        }
        try:
            report = _post_json("/monitor", payload, timeout=60)
        except httpx.HTTPError as exc:
            st.error(f"Monitor request failed: {exc}")
        else:
            srm = report["srm"]
            if report["status"] == "refused_srm":
                st.error(f"**Refused.** {report['reason']}")
            else:
                st.success(f"**{report['status'].replace('_', ' ').upper()}** — {report['reason']}")
                st.caption(
                    f"SRM gate passed: control took {srm['observed_share_control']:.4%} of units "
                    f"(chi-square p = {srm['p_value']:.3g})."
                )
                looks = pd.DataFrame(report["looks"])
                st.line_chart(
                    looks.set_index("label")[["always_valid_p", "fixed_horizon_p"]],
                    height=260,
                )
                st.caption(
                    f"Horizontal reference: alpha = {monitor_alpha:g}. The always-valid curve is "
                    "not monotone — it can cross and drift back."
                )
                safe, naive = report["always_valid"], report["naive_fixed"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Always-valid rule fires", safe["stop_label"] or "never")
                c2.metric("Naive fixed-horizon fires", naive["stop_label"] or "never")
                c3.metric(
                    "P(beat control) at decision",
                    f"{report['prob_treatment_beats_control']:.1%}",
                )
                st.dataframe(looks, width="stretch", hide_index=True)
