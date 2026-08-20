"""Streamlit demo: plan, analyze, monitor sequentially, decide Bayesian."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("ABTESTLAB_API_URL", "http://localhost:8240")

st.set_page_config(page_title="abtestlab", page_icon="🧪", layout="wide")
st.title("🧪 abtestlab")
st.caption("Power planning, fixed + always-valid sequential tests, CUPED, Bayesian decisions")


def _ok() -> bool:
    try:
        return httpx.get(f"{API_URL}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


if not _ok():
    st.error(f"API not reachable at {API_URL}. Start it with `make api`.")
    st.stop()

tab_plan, tab_analyze, tab_cuped = st.tabs(["Plan (power)", "Analyze", "CUPED demo"])

with tab_plan:
    c1, c2 = st.columns(2)
    baseline = c1.slider("Baseline conversion", 0.005, 0.5, 0.05, 0.005)
    mde = c2.slider("Minimum detectable relative lift", 0.01, 0.5, 0.10, 0.01)
    r = httpx.post(
        f"{API_URL}/power", json={"baseline_rate": baseline, "mde_relative": mde}, timeout=30
    )
    if r.status_code == 200:
        body = r.json()
        st.metric("Sample size per arm", f"{body['per_arm_sample_size']:,}")
        st.caption(f"Total {body['total']:,} at alpha=0.05, power=0.8")

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
        fixed = httpx.post(f"{API_URL}/analyze", json=payload, timeout=30).json()
        seq = httpx.post(f"{API_URL}/sequential", json=payload, timeout=30).json()
        bayes = httpx.post(f"{API_URL}/bayes", json=payload, timeout=60).json()
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
            "p-value is only valid at the planned sample size."
        )

with tab_cuped:
    c1, c2 = st.columns(2)
    rho = c1.slider("Pre-period correlation", 0.0, 0.95, 0.6, 0.05)
    lift = c2.slider("True lift", 0.0, 0.10, 0.03, 0.005)
    r = httpx.post(
        f"{API_URL}/cuped-demo",
        json={"n_per_arm": 2000, "true_lift": lift, "covariate_correlation": rho},
        timeout=30,
    )
    if r.status_code == 200:
        body = r.json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw p-value", f"{body['raw_p_value']:.4f}")
        c2.metric("CUPED p-value", f"{body['cuped_p_value']:.4f}")
        c3.metric("Variance reduction", f"{body['variance_reduction']:.0%}")
        st.caption("Same data, same effect — CUPED just removes pre-period noise.")
