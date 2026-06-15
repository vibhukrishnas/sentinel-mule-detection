"""
SENTINEL — live demo dashboard for PS2 mule-account detection.
Run:  streamlit run app.py

Pick a real account (or paste raw feature JSON) -> instant calibrated risk score,
plain-English reasons, an analyst-ready investigation report, and a precision/recall
dial the risk officer controls. This is the "feel it work" surface for judges.
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import pandas as pd
import streamlit as st
from sentinel import SentinelEngine, ART, ACTION

st.set_page_config(page_title="SENTINEL · Mule Account Risk Engine", layout="wide")


@st.cache_resource
def load_engine():
    return SentinelEngine()


@st.cache_data
def load_demo():
    X = pd.read_parquet(ART / "demo_accounts.parquet")
    y = pd.read_parquet(ART / "demo_targets.parquet")["target"]
    return X, y


@st.cache_data
def load_metrics():
    p = ART / "holdout_metrics.json"
    m = json.loads(p.read_text()) if p.exists() else {}
    he = ART / "honest_eval.json"
    if he.exists():
        h = json.loads(he.read_text())
        win = h["leaderboard"][0]                  # LightGBM, sorted by PR-AUC
        m["cv_pr_auc"], m["cv_roc_auc"], m["cv_brier"] = win["pr_auc"], win["roc_auc"], win["brier"]
        m["latency_p95"] = h["latency_ms"]["shap_ms_p95"]
        m["pct_mules_ge70"] = h["pct_mules_ge70"]
    return m


eng = load_engine()
demoX, demoY = load_demo()
metrics = load_metrics()

st.title("🛡️ SENTINEL — Suspicious / Mule Account Risk Engine")
st.caption("BOI Hackathon · PS2 · Calibrated risk scoring with explainable, "
           "investigation-ready alerts. F3912 (leakage) excluded — these are honest numbers.")

# ---- model scorecard ----
if metrics:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CV PR-AUC (5×2)", f"{metrics.get('cv_pr_auc', 0):.3f}",
              help=f"Leak-removed, repeated CV. Random baseline = {metrics.get('prevalence', 0):.4f} (~100× lift)")
    c2.metric("CV ROC-AUC", f"{metrics.get('cv_roc_auc', 0):.3f}")
    c3.metric("CV Brier (calibration)", f"{metrics.get('cv_brier', 0):.4f}",
              help="lower = better-calibrated probabilities (measured on CV, not the noisy holdout)")
    c4.metric("Score+explain latency", f"{metrics.get('latency_p95', 0):.0f} ms",
              help="p95, predict_proba + SHAP — real-time ready")

st.divider()
left, right = st.columns([1, 1.4])

with left:
    st.subheader("Select an account")
    labels = {i: f"Account #{i}  ({'MULE' if demoY.loc[i]==1 else 'legit'})"
              for i in demoX.index}
    pick = st.selectbox("Demo accounts (5 real mules + 5 legit):",
                        options=list(demoX.index), format_func=lambda i: labels[i])
    threshold = st.slider("Alert threshold (risk officer's dial)", 0.05, 0.99, 0.50, 0.01,
                          help="Lower = catch more mules but more false alarms.")

    account = demoX.loc[pick]
    sc = eng.score(account)                 # fast: no SHAP, always renders
    band_color = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}[sc["band"]]
    st.markdown(f"### {band_color} Risk Score: **{sc['risk_score']}/100** ({sc['band']})")
    st.progress(min(sc["risk_score"], 100) / 100)
    st.write(f"Calibrated probability of mule activity: **{sc['probability']:.1%}**")
    st.write(f"Ground-truth label: **{'MULE' if demoY.loc[pick]==1 else 'LEGITIMATE'}**")
    if sc["probability"] >= threshold:
        st.error(f"🚨 ALERT raised — {ACTION[sc['band']]}")
    else:
        st.success("No alert at current threshold.")

with right:
    st.subheader("Why — top risk drivers (SHAP)")
    # SHAP is lazy-loaded on click so the dashboard renders instantly on first
    # load (and on memory-constrained cloud hosts) — also a live latency demo.
    if st.button("🔍 Explain this account (SHAP) + investigation report", type="primary"):
        try:
            drivers = eng.explain(account, top_k=6)
            dd = pd.DataFrame([{
                "Factor": d["label"], "Value": d["value_readable"],
                "Effect": ("▲ raises" if d["shap"] > 0 else "▼ lowers"),
                "Impact": round(abs(d["shap"]), 3), "Context": d["context"],
            } for d in drivers])
            st.dataframe(dd, hide_index=True, use_container_width=True)
            st.subheader("📄 Auto-generated investigation report")
            st.code(eng.report(account, account_id=pick), language="text")
        except Exception as e:                # never let explainability hang the app
            st.warning(f"SHAP explanation unavailable in this environment ({type(e).__name__}). "
                       "Risk score and alert above are unaffected.")
    else:
        st.caption("Click to compute the per-account SHAP attribution and the "
                   "analyst-ready report (≈35 ms once the explainer warms up).")

if metrics.get("threshold_table"):
    st.divider()
    st.subheader("Precision / recall trade-off (holdout)")
    st.dataframe(pd.DataFrame(metrics["threshold_table"]), hide_index=True,
                 use_container_width=True)

# ---- ranked watchlist (the product output) ----
_wl = ART.parent / "outputs" / "top_suspicious_accounts.csv"
if _wl.exists():
    st.divider()
    st.subheader("🚨 Watchlist — top suspicious accounts (out-of-fold)")
    wl = pd.read_csv(_wl)
    hits = int((wl["actual_label"] == 1).sum())
    st.caption(f"Of the top {len(wl)} flagged accounts, {hits} are confirmed mules "
               f"(precision@{len(wl)} = {hits/len(wl):.0%}) — in a 0.89%-mule population.")
    st.dataframe(wl[["account_id", "risk_score", "band", "top_reasons"]].head(20),
                 hide_index=True, use_container_width=True)

# ---- model performance gallery ----
_fig = ART.parent / "figures"
if _fig.exists():
    st.divider()
    st.subheader("📊 Model performance (out-of-fold)")
    figs = ["03_score_distribution.png", "01_pr_curve.png", "04_confusion_matrix.png",
            "05_calibration.png", "08_leakage_sensitivity.png", "09_cost_curve.png"]
    cols = st.columns(3)
    for i, f in enumerate(figs):
        p = _fig / f
        if p.exists():
            cols[i % 3].image(str(p), use_container_width=True)
