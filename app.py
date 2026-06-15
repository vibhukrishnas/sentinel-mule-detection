"""
SENTINEL — live demo dashboard for PS2 mule-account detection.
Run:  streamlit run app.py

Pick a real account -> instant calibrated risk score, plain-English SHAP reasons, an
analyst-ready investigation report, and a precision/recall dial the risk officer controls.
Data source is dynamic: upload a CSV (raw DataSet.csv format or a cleaned export) and the
WHOLE dashboard re-scores on it; otherwise it runs on a committed sample, or the built-in
demo. The full bank dataset is NOT shipped in this public repo by design.
"""
import sys, json, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import joblib
import pandas as pd
import streamlit as st
from sentinel import SentinelEngine, ART, ACTION, band_for
from preprocess import prepare_frame

st.set_page_config(page_title="SENTINEL · Mule Account Risk Engine", layout="wide")


@st.cache_resource
def load_engine():
    return SentinelEngine()


@st.cache_data
def load_cols_maps():
    cols = list(joblib.load(ART / "population_stats.joblib")["columns"])
    cat_maps = json.loads((ART / "categorical_maps.json").read_text())
    return cols, cat_maps


@st.cache_data
def read_default(cols, cat_maps):
    """Out-of-box data: committed sample if present (raw CSV), else the 10-account demo."""
    samp = ROOT / "samples" / "sample_accounts.csv"
    if samp.exists():
        df = pd.read_csv(samp, index_col=0, low_memory=False)
        X, y = prepare_frame(df, cols, cat_maps)
        return X, y, f"committed sample — {len(X):,} accounts (all known mules + a legit sample)"
    dX = pd.read_parquet(ART / "demo_accounts.parquet").reindex(columns=cols).astype("float32")
    dY = pd.read_parquet(ART / "demo_targets.parquet")["target"]
    return dX, dY, "built-in demo — 10 accounts"


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
cols, cat_maps = load_cols_maps()
metrics = load_metrics()

# ---- dynamic data source (sidebar): upload > sample > demo ----
st.sidebar.header("📥 Data source")
st.sidebar.caption("Upload the provided **DataSet.csv** (raw) or a cleaned export. The full "
                   "dataset is not shipped in this public repo; the demo runs on a small "
                   "committed sample. Uploaded data stays in this session — it is not stored.")
up = st.sidebar.file_uploader("Upload account CSV", type=["csv"])
if up is not None:
    key = (up.name, up.size)
    if st.session_state.get("src_key") != key:
        try:
            df = pd.read_csv(up, index_col=0, low_memory=False)
            X, y = prepare_frame(df, cols, cat_maps)
            st.session_state.src = (X, y, f"📤 {up.name} — {len(X):,} accounts")
            st.session_state.src_key = key
            st.session_state.scores = None
        except Exception as e:
            st.sidebar.error(f"Couldn't read that CSV — {type(e).__name__}: {e}")

if "src" not in st.session_state:
    st.session_state.src = read_default(cols, cat_maps)
    st.session_state.scores = None

X_all, y_all, src_label = st.session_state.src
st.sidebar.success(f"Active: {src_label}")


def gt(i):
    """Ground-truth label if the data carries one, else None."""
    if y_all is None:
        return None
    return "MULE" if int(y_all.loc[i]) == 1 else "legit"


# score the whole active dataset once (deployed model); recompute only on source change
if st.session_state.get("scores") is None:
    with st.spinner(f"Scoring {len(X_all):,} accounts…"):
        proba = eng.model.predict_proba(X_all.astype("float32"))[:, 1]
        s = (proba * 100).round().astype(int)
        st.session_state.scores = pd.DataFrame(
            {"risk_score": s, "probability": proba, "band": [band_for(v) for v in s]},
            index=X_all.index)
allscores = st.session_state.scores

st.title("🛡️ SENTINEL — Suspicious / Mule Account Risk Engine")
st.caption("BOI Hackathon · PS2 · Calibrated risk scoring with explainable, "
           "investigation-ready alerts. F3912 (leakage) excluded — these are honest numbers.")

# ---- model scorecard (validated model — fixed, not per-account) ----
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
    order = list(allscores.sort_values("risk_score", ascending=False).index)

    def acct_label(i):
        r = allscores.loc[i]
        g = gt(i)
        tag = f" · {g}" if g else ""
        return f"#{i} · {int(r['risk_score'])}/100 {r['band']}{tag}"

    only_flagged = st.checkbox(
        f"Show only flagged accounts (band ≥ HIGH)  ·  loaded = {len(order):,}",
        value=False, help="Tick to focus the picker on the high-risk shortlist.")
    if only_flagged:
        order = [i for i in order if allscores.loc[i, "band"] in ("HIGH", "CRITICAL")]
    pick = st.selectbox(f"Account ({len(order):,} shown · sorted by risk, most critical first):",
                        options=order, format_func=acct_label)
    st.caption("Scores here are the **deployed model** on the loaded data. If you load the "
               "full training population they separate cleanly (the model has seen it); the "
               "*honest* generalization number is the out-of-fold CV in the scorecard above.")
    threshold = st.slider("Alert threshold (risk officer's dial)", 0.05, 0.99, 0.50, 0.01,
                          help="Lower = catch more mules but more false alarms.")

    account = X_all.loc[pick]
    sc = eng.score(account)                 # fast: no SHAP, always renders
    band_color = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}[sc["band"]]
    st.markdown(f"### {band_color} Risk Score: **{sc['risk_score']}/100** ({sc['band']})")
    st.progress(min(sc["risk_score"], 100) / 100)
    st.write(f"Calibrated probability of mule activity: **{sc['probability']:.1%}**")
    g = gt(pick)
    st.write(f"Ground-truth label: **{('MULE' if g=='MULE' else 'LEGITIMATE') if g else 'unknown (no label in data)'}**")
    raised = sc["probability"] >= threshold
    if raised:
        st.error(f"🚨 ALERT raised — {ACTION[sc['band']]}")
    else:
        st.success("No alert at current threshold.")

    # ---- rolling audit trail: log each distinct review (traceability) ----
    if "audit" not in st.session_state:
        st.session_state.audit = []
        st.session_state.last_sig = None
    sig = (pick, round(threshold, 2))          # new account or new threshold = new review
    if sig != st.session_state.last_sig:
        st.session_state.last_sig = sig
        st.session_state.audit.append({
            "time (UTC)": datetime.datetime.utcnow().strftime("%H:%M:%S"),
            "account": int(pick),
            "score": sc["risk_score"],
            "band": sc["band"],
            "probability": round(sc["probability"], 4),
            "threshold": round(threshold, 2),
            "decision": "ALERT" if raised else "clear",
            "ground_truth": g or "unknown",
        })

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

# ---- rolling investigation log (dynamic, per-session, exportable) ----
st.divider()
st.subheader("🧾 Investigation activity — this session (live audit trail)")
audit = st.session_state.get("audit", [])
if audit:
    log_df = pd.DataFrame(audit)[::-1].reset_index(drop=True)   # newest first
    a, b, c = st.columns(3)
    a.metric("Reviews logged", len(audit))
    b.metric("Alerts raised", int((log_df["decision"] == "ALERT").sum()))
    caught = int(((log_df["decision"] == "ALERT") & (log_df["ground_truth"] == "MULE")).sum())
    c.metric("Mules caught in trail", caught)
    st.dataframe(log_df, hide_index=True, use_container_width=True)
    d1, d2 = st.columns([1, 4])
    d1.download_button("⬇ Export audit log (CSV)",
                       log_df.to_csv(index=False).encode(),
                       file_name="sentinel_audit_log.csv", mime="text/csv")
    if d2.button("Clear log"):
        st.session_state.audit = []
        st.session_state.last_sig = None
        st.rerun()
    st.caption("Every account you review (and each threshold change) is timestamped and "
               "appended here — an analyst-ready, exportable audit trail.")
else:
    st.caption("Select accounts and adjust the threshold above — each review is logged here "
               "with a timestamp, the decision, and ground truth, then exportable as CSV.")

# ---- ranked watchlist (DYNAMIC: recomputed from the loaded data) ----
st.divider()
st.subheader("🚨 Watchlist — highest-risk accounts (current dataset)")
topn = allscores.sort_values("risk_score", ascending=False).head(20).copy()
topn.insert(0, "account_id", topn.index)
if y_all is not None:
    topn["ground_truth"] = [gt(i) for i in topn.index]
    k = min(50, len(allscores))
    topk_idx = allscores.sort_values("risk_score", ascending=False).head(k).index
    hits = int((y_all.loc[topk_idx] == 1).sum())
    st.caption(f"Top {k} highest-risk accounts contain {hits} confirmed mules "
               f"(precision@{k} = {hits/k:.0%}). Every account is selectable in the picker — "
               "drill into any of them live. (Headline validation: precision@50 = 100% "
               "out-of-fold on the full population — see the performance section below.)")
else:
    st.caption("Ranked by deployed-model risk. Upload data with the target column to see "
               "precision@k against ground truth. Every account is selectable in the picker above.")
st.dataframe(topn[["account_id", "risk_score", "band", "probability"]
                  + (["ground_truth"] if y_all is not None else [])],
             hide_index=True, use_container_width=True)

# ---- model performance gallery (validated model — fixed by design) ----
_fig = ROOT / "figures"
if _fig.exists():
    st.divider()
    st.subheader("📊 Model performance (out-of-fold, validated — fixed, not per-account)")
    figs = ["03_score_distribution.png", "01_pr_curve.png", "04_confusion_matrix.png",
            "05_calibration.png", "08_leakage_sensitivity.png", "09_cost_curve.png"]
    gcols = st.columns(3)
    for i, f in enumerate(figs):
        p = _fig / f
        if p.exists():
            gcols[i % 3].image(str(p), use_container_width=True)
