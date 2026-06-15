"""
SENTINEL — live demo dashboard for PS2 mule-account detection.
Run:  streamlit run app.py

Four tabs: Investigate (per-account scoring + SHAP + downloadable report), Analytics
(dataset + session analytics, live threshold impact), Activity log (audit trail), and
Model & validation (the fixed, validated metrics). Data source is dynamic — upload a CSV
(raw DataSet.csv or a cleaned export) and the WHOLE dashboard re-scores on it. The full
bank dataset is NOT shipped in this public repo by design.
"""
import sys, json, datetime, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sentinel import SentinelEngine, ART, ACTION, band_for
from preprocess import prepare_frame

st.set_page_config(page_title="SENTINEL · Mule Account Risk Engine", layout="wide")

# illustrative, configurable INR assumptions (mirror src/insights.py)
MULE_LOSS, REVIEW_COST, FP_HARM = 250_000, 400, 5_000
BAND_EMOJI = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}


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


@st.cache_data
def load_rings():
    p = ART / "mule_network.json"
    return json.loads(p.read_text()) if p.exists() else None


eng = load_engine()
cols, cat_maps = load_cols_maps()
metrics = load_metrics()
rings = load_rings()


def ring_of(i):
    """Return the candidate-ring dict this account belongs to, or None."""
    if not rings:
        return None
    for r in rings["rings"]:
        if int(i) in r["members"]:
            return r
    return None


def verdict_line(sc):
    """One-line, plain-English summary of the model's call."""
    p = sc["probability"]
    if sc["band"] in ("CRITICAL", "HIGH"):
        conf = "High" if p >= 0.9 else "Elevated"
        return (f"🧭 **Verdict:** {conf} confidence ({p:.0%}) this account **behaves like a mule** "
                f"— recommend *{ 'escalate to L2' if sc['band']=='CRITICAL' else 'same-day analyst review'}*.")
    if sc["band"] == "MEDIUM":
        return f"🧭 **Verdict:** Mixed signals ({p:.0%}) — add to enhanced monitoring, not an immediate alert."
    return f"🧭 **Verdict:** Low risk ({p:.0%}) — behaves like a legitimate account; routine monitoring."


def audit_actions():
    """Derive live investigation state from the session activity log:
    set of reviewed account ids, and {account: latest explicit analyst action}."""
    reviewed, actions = set(), {}
    for row in st.session_state.get("audit", []):
        reviewed.add(row["account"])
        if row.get("action") in ("ESCALATED", "CLEARED", "MONITORED"):
            actions[row["account"]] = row["action"]
    return reviewed, actions


def audit_row(i, sc_row, g, ring_id, action):
    """Build one audit-trail record (used for single reviews and batch ring escalation)."""
    return {
        "time (UTC)": datetime.datetime.utcnow().strftime("%H:%M:%S"),
        "account": int(i), "score": int(sc_row["risk_score"]), "band": sc_row["band"],
        "probability": round(float(sc_row["probability"]), 4), "threshold": round(threshold, 2),
        "decision": "ALERT" if sc_row["probability"] >= threshold else "clear",
        "ground_truth": g or "unknown", "ring": ring_id, "action": action,
    }


def build_shift_report():
    """A close-of-shift summary of this session — the analyst hand-off artifact."""
    audit = st.session_state.get("audit", [])
    reviewed, actions = audit_actions()
    esc = sorted(a for a, v in actions.items() if v == "ESCALATED")
    clr = [a for a, v in actions.items() if v == "CLEARED"]
    mon = [a for a, v in actions.items() if v == "MONITORED"]
    L = ["SENTINEL — Analyst shift report", "=" * 44,
         f"Generated (UTC): {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
         f"Data source    : {src_label}", "",
         f"Reviews logged    : {len(audit)}",
         f"Distinct accounts : {len({r['account'] for r in audit})}",
         f"Escalated to L2   : {len(esc)}",
         f"Cleared           : {len(clr)}",
         f"Under monitoring  : {len(mon)}"]
    if rings:
        contained, exposure = 0, 0.0
        ring_lines = []
        for r in rings["rings"]:
            e = sum(1 for m in r["members"] if actions.get(m) == "ESCALATED")
            if e:
                frac = e / r["size"]
                exposure += r["exposure_rupees"] * frac
                contained += (e == r["size"])
                ring_lines.append(f"  - Ring #{r['ring_id']}: {e}/{r['size']} escalated "
                                  f"(~₹{int(r['exposure_rupees']*frac):,} of ₹{r['exposure_rupees']:,})")
        L += ["", f"Rings contained             : {contained}/{rings['n_candidate_rings']}",
              f"Potential exposure addressed : ₹{int(exposure):,}"]
        if ring_lines:
            L += ["", "Ring actions:"] + ring_lines
    if esc:
        L += ["", f"Escalated accounts ({len(esc)}): " + ", ".join(f"#{a}" for a in esc[:60])]
    return "\n".join(L)

# ============================ SIDEBAR: data + global dial ============================
st.sidebar.header("📥 Data source")
st.sidebar.caption("Upload the provided **DataSet.csv** (raw) or a cleaned export. The full "
                   "dataset isn't shipped in this public repo; the demo runs on a small "
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

# score the whole active dataset once (deployed model); recompute only on source change
if st.session_state.get("scores") is None:
    with st.spinner(f"Scoring {len(X_all):,} accounts…"):
        proba = eng.model.predict_proba(X_all.astype("float32"))[:, 1]
        s = (proba * 100).round().astype(int)
        st.session_state.scores = pd.DataFrame(
            {"risk_score": s, "probability": proba, "band": [band_for(v) for v in s]},
            index=X_all.index)
allscores = st.session_state.scores
has_labels = y_all is not None
ym = (y_all.reindex(allscores.index).fillna(0).astype(int) if has_labels else None)


def gt(i):
    return None if not has_labels else ("MULE" if int(y_all.loc[i]) == 1 else "legit")


# global risk-officer dial — drives BOTH the per-account alert and the analytics
st.sidebar.divider()
st.sidebar.header("🎚️ Alert threshold")
threshold = st.sidebar.slider(
    "Flag an account when its mule-probability ≥", 0.01, 0.99, 0.50, 0.01,
    help="The risk officer's dial. Lower = catch more mules but raise more false alarms. "
         "Watch the live impact below and on the Analytics tab.")
_flagged = allscores["probability"] >= threshold
_alerts = int(_flagged.sum())
st.sidebar.metric("Accounts flagged at this threshold", f"{_alerts:,} / {len(allscores):,}")
if has_labels:
    _tp = int((_flagged & (ym == 1)).sum())
    _fn = int((~_flagged & (ym == 1)).sum())
    _fp = int((_flagged & (ym == 0)).sum())
    _recall = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
    _prec = _tp / _alerts if _alerts else 0.0
    cA, cB = st.sidebar.columns(2)
    cA.metric("Mules caught", f"{_tp}/{_tp+_fn}", help=f"recall {_recall:.0%}")
    cB.metric("False alarms", f"{_fp}", help=f"precision {_prec:.0%}")

# ================================== HEADER + SCORECARD ==================================
st.title("🛡️ SENTINEL — Suspicious / Mule Account Risk Engine")
st.caption("BOI Hackathon · PS2 · Calibrated risk scoring with explainable, "
           "investigation-ready alerts. F3912 (leakage) excluded — these are honest numbers.")
with st.expander("👋 Judges — try this in 30 seconds"):
    st.markdown(
        "1. **Investigate** → click **🎲 Random account**, then **🔍 Check risk score** "
        "(the model scores on demand — score, band, and ground-truth are revealed).\n"
        "2. Click **🧠 Explain** for the SHAP reasons + a downloadable investigation report.\n"
        "3. If the account is in a **candidate mule-ring**, hit **🚩 Escalate entire ring** — "
        "then open **📊 Analytics** and watch that ring flip to **🟢 Contained**.\n"
        "4. Drag the sidebar **🎚️ alert threshold** — recall / false-alarms / ₹ impact move live.\n"
        "5. **🧾 Activity log** → download the **shift report** of everything you did.")
if metrics:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CV PR-AUC (5×2)", f"{metrics.get('cv_pr_auc', 0):.3f}",
              help=f"Leak-removed, repeated CV. Random baseline = {metrics.get('prevalence', 0):.4f} (~100× lift)")
    c2.metric("CV ROC-AUC", f"{metrics.get('cv_roc_auc', 0):.3f}")
    c3.metric("CV Brier (calibration)", f"{metrics.get('cv_brier', 0):.4f}",
              help="lower = better-calibrated probabilities (measured on CV, not the noisy holdout)")
    c4.metric("Score+explain latency", f"{metrics.get('latency_p95', 0):.0f} ms",
              help="p95, predict_proba + SHAP — real-time ready")

tab_inv, tab_an, tab_log, tab_model = st.tabs(
    ["🔍 Investigate", "📊 Analytics", "🧾 Activity log", "📈 Model & validation"])

# ===================================== INVESTIGATE =====================================
with tab_inv:
    left, right = st.columns([1, 1.4])
    with left:
        st.subheader("Select an account")
        # neutral picker (ID only) — the score is REVEALED after you pick, not spoiled in the list
        order = sorted(allscores.index.tolist())
        if "pick_id" not in st.session_state or st.session_state.pick_id not in order:
            st.session_state.pick_id = order[0]
        if st.button("🎲 Random account", help="Jump to a random account — could be a mule or legit."):
            st.session_state.pick_id = random.choice(order)
        pick = st.selectbox(f"Account ({len(order):,} loaded · pick one, then see what the model says):",
                            options=order, index=order.index(st.session_state.pick_id),
                            format_func=lambda i: f"Account #{i}")
        st.session_state.pick_id = pick

        account = X_all.loc[pick]
        # genuine flow: nothing is shown until the analyst runs the model on this account
        if st.button("🔍 Check risk score", type="primary", key="check_score"):
            st.session_state.checked = pick
        revealed = st.session_state.get("checked") == pick
        sc = None
        if not revealed:
            st.info("Pick an account, then click **Check risk score** to run the model on it.")
        else:
            with st.spinner("Scoring…"):
                sc = eng.score(account)
            st.markdown(f"### {BAND_EMOJI[sc['band']]} Risk Score: **{sc['risk_score']}/100** ({sc['band']})")
            st.progress(min(sc["risk_score"], 100) / 100)
            st.write(f"Calibrated probability of mule activity: **{sc['probability']:.1%}**")
            g = gt(pick)
            if g:
                ok = (g == "MULE") == (sc["probability"] >= threshold)
                st.write(f"Ground-truth label: **{'MULE' if g=='MULE' else 'LEGITIMATE'}**  "
                         f"{'✅ model agrees' if ok else '⚠️ model disagrees at current threshold'}")
            else:
                st.write("Ground-truth label: **unknown (no label in uploaded data)**")
            raised = sc["probability"] >= threshold
            if raised:
                st.error(f"🚨 ALERT at threshold {threshold:.2f} — {ACTION[sc['band']]}")
            else:
                st.success(f"No alert at threshold {threshold:.2f} (probability below the dial).")

            st.markdown(verdict_line(sc))

            if "audit" not in st.session_state:
                st.session_state.audit, st.session_state.last_sig = [], None
            reviewed, actions = audit_actions()
            r = ring_of(pick)
            ring_id = f"#{r['ring_id']}" if r else "—"

            # mule-ring callout (the differentiator) — candidate, with LIVE progress from the log
            if r:
                members = r["members"]
                revd = sum(1 for m in members if m in reviewed)
                esc = sum(1 for m in members if actions.get(m) == "ESCALATED")
                st.warning(f"🕸️ **Candidate Ring #{r['ring_id']}** — clusters with **{r['size']} "
                           f"near-identical accounts** (~₹{r['exposure_rupees']:,} potential exposure). "
                           "Investigate as a batch. *Candidate grouping; confirmation needs bank link data.*")
                st.progress(revd / r["size"],
                            text=f"Ring progress this session: {revd}/{r['size']} reviewed · {esc} escalated")
                esc_peers = [m for m in members if m != pick and actions.get(m) == "ESCALATED"]
                if esc_peers:
                    shown = ", ".join(f"#{m}" for m in esc_peers[:5])
                    st.info(f"🔗 Shares Ring #{r['ring_id']} with **{len(esc_peers)}** account(s) you already "
                            f"escalated this session: {shown}{'…' if len(esc_peers) > 5 else ''}.")
                if st.button(f"🚩 Escalate entire Ring #{r['ring_id']} ({r['size']} accounts)", key="esc_ring"):
                    added = 0
                    for m in members:
                        if actions.get(m) != "ESCALATED":
                            st.session_state.audit.append(
                                audit_row(m, allscores.loc[m], gt(m), f"#{r['ring_id']}", "ESCALATED"))
                            added += 1
                    st.success(f"Escalated {added} member(s) of Ring #{r['ring_id']} — see the activity log & rings.")
                    st.rerun()

            # rolling audit trail: log each distinct review (account or threshold change)
            sig = (pick, round(threshold, 2))
            if sig != st.session_state.last_sig:
                st.session_state.last_sig = sig
                st.session_state.audit.append(audit_row(pick, sc, g, ring_id, "reviewed"))

            # ---- analyst case decision (accountability) ----
            st.markdown("**Analyst decision** (logged to the audit trail):")
            d1, d2, d3 = st.columns(3)
            decided = (d1.button("✅ Clear", key="act_clear") and "CLEARED") or \
                      (d2.button("👁 Monitor", key="act_mon") and "MONITORED") or \
                      (d3.button("🚩 Escalate to L2", key="act_esc") and "ESCALATED")
            if decided:
                st.session_state.audit.append(audit_row(pick, sc, g, ring_id, decided))
                st.success(f"Logged **{decided}** for account #{pick}.")
                st.rerun()

    with right:
        st.subheader("Why — top risk drivers (SHAP) + investigation report")
        if "expl" not in st.session_state:
            st.session_state.expl = {}
        if not revealed:
            st.caption("Check the risk score first (left) — then explain what drove it.")
        else:
            if st.button("🧠 Explain this account + build report", type="primary"):
                try:
                    drivers = eng.explain(account, top_k=6)
                    report = eng.report(account, account_id=pick)
                    st.session_state.expl[pick] = {"drivers": drivers, "report": report}
                except Exception as e:
                    st.session_state.expl[pick] = {"error": type(e).__name__}
            data = st.session_state.expl.get(pick)
            if not data:
                st.caption("Click to compute the per-account SHAP attribution and the "
                           "analyst-ready report (≈35 ms once the explainer warms up).")
            elif "error" in data:
                st.warning(f"SHAP unavailable here ({data['error']}). Score and alert are unaffected.")
            else:
                dd = pd.DataFrame([{
                    "Factor": d["label"], "Value": d["value_readable"],
                    "Effect": ("▲ raises" if d["shap"] > 0 else "▼ lowers"),
                    "Impact": round(abs(d["shap"]), 3), "Context": d["context"],
                } for d in data["drivers"]])
                st.dataframe(dd, hide_index=True, use_container_width=True)
                st.subheader("📄 Investigation report")
                st.code(data["report"], language="text")
                st.download_button("⬇ Download investigation report (.txt)", data["report"],
                                   file_name=f"SENTINEL_investigation_account_{pick}.txt",
                                   mime="text/plain", key=f"dl_{pick}")

# ====================================== ANALYTICS ======================================
with tab_an:
    st.subheader("Population analytics — current dataset")
    n = len(allscores)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accounts loaded", f"{n:,}")
    if has_labels:
        m2.metric("Known mules", f"{int((ym==1).sum()):,}")
        m3.metric("Known legit", f"{int((ym==0).sum()):,}")
    m4.metric("Flagged (band ≥ HIGH)", f"{int(allscores['band'].isin(['HIGH','CRITICAL']).sum()):,}")

    cL, cR = st.columns(2)
    with cL:
        st.markdown("**Risk-score distribution**")
        buckets = (allscores["risk_score"] // 10 * 10).astype(int)
        if has_labels:
            dist = pd.DataFrame({"score_bucket": buckets,
                                 "class": np.where(ym.values == 1, "mule", "legit")})
            chart = dist.groupby(["score_bucket", "class"]).size().unstack(fill_value=0)
        else:
            chart = buckets.value_counts().sort_index().to_frame("accounts")
        st.bar_chart(chart)
        st.caption("Legit pile up near 0; mules near 100 — the separation the model exploits.")
    with cR:
        st.markdown("**Severity-band breakdown**")
        if has_labels:
            bdf = pd.DataFrame({"band": allscores["band"],
                                "class": np.where(ym.values == 1, "mule", "legit")})
            bchart = (bdf.groupby(["band", "class"]).size().unstack(fill_value=0)
                      .reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0))
        else:
            bchart = (allscores["band"].value_counts()
                      .reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0).to_frame("accounts"))
        st.bar_chart(bchart)

    st.divider()
    st.subheader(f"🎚️ Live impact at threshold = {threshold:.2f}")
    st.caption("Drag the **Alert threshold** in the sidebar — every number below moves with it.")
    if has_labels:
        flagged = allscores["probability"] >= threshold
        tp = int((flagged & (ym == 1)).sum()); fp = int((flagged & (ym == 0)).sum())
        fn = int((~flagged & (ym == 1)).sum()); tn = int((~flagged & (ym == 0)).sum())
        alerts = tp + fp
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / alerts if alerts else 0.0
        net = tp * MULE_LOSS - alerts * REVIEW_COST - fp * FP_HARM
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Alerts raised", f"{alerts:,}")
        k2.metric("Mules caught", f"{tp}/{tp+fn}", help=f"recall {recall:.0%}")
        k3.metric("Precision", f"{prec:.0%}")
        k4.metric("False alarms", f"{fp:,}")
        k5.metric("Net ₹ impact", f"₹{net/1e5:.2f} L", help="illustrative: mule loss ₹2.5L, "
                  "review ₹400, false-freeze ₹5,000 — all configurable")
        cm = pd.DataFrame([[tp, fn], [fp, tn]],
                          index=["Actual MULE", "Actual legit"],
                          columns=["Flagged", "Not flagged"])
        st.markdown("**Confusion matrix at this threshold**")
        st.dataframe(cm, use_container_width=False)
    else:
        alerts = int((allscores["probability"] >= threshold).sum())
        st.metric("Alerts raised", f"{alerts:,} / {len(allscores):,}")
        st.caption("Upload data with the target column (F3924) to see recall / precision / ₹ impact.")

    st.divider()
    st.subheader("🚨 Watchlist — highest-risk accounts (current dataset)")
    topn = allscores.sort_values("risk_score", ascending=False).head(20).copy()
    topn.insert(0, "account_id", topn.index)
    topn["ring"] = [f"#{r['ring_id']}" if (r := ring_of(i)) else "—" for i in topn.index]
    show_cols = ["account_id", "risk_score", "band", "probability", "ring"]
    if has_labels:
        topn["ground_truth"] = [gt(i) for i in topn.index]; show_cols.append("ground_truth")
    st.dataframe(topn[show_cols], hide_index=True, use_container_width=True)
    # export the full current alert list (everything flagged at the threshold) for the fraud team
    alert_tbl = allscores[allscores["probability"] >= threshold].sort_values(
        "risk_score", ascending=False).copy()
    alert_tbl.insert(0, "account_id", alert_tbl.index)
    alert_tbl["ring"] = [f"#{r['ring_id']}" if (r := ring_of(i)) else "—" for i in alert_tbl.index]
    if has_labels:
        alert_tbl["ground_truth"] = [gt(i) for i in alert_tbl.index]
    st.download_button(f"⬇ Export alert list at threshold {threshold:.2f} ({len(alert_tbl):,} accounts, CSV)",
                       alert_tbl.to_csv(index=False).encode(),
                       file_name=f"sentinel_alerts_thr{threshold:.2f}.csv", mime="text/csv")

    # ---- mule-ring detection (the differentiator) — progress is LIVE from the activity log ----
    if rings:
        st.divider()
        st.subheader("🕸️ Mule-ring detection (candidate behavioral rings)")
        v = rings.get("validation", {})
        reviewed, actions = audit_actions()
        ring_rows, total_esc = [], 0
        for r in rings["rings"]:
            mem = r["members"]
            revd = sum(1 for m in mem if m in reviewed)
            esc = sum(1 for m in mem if actions.get(m) == "ESCALATED")
            total_esc += esc
            status = ("🟢 Contained" if esc == r["size"]
                      else "🟡 In progress" if revd else "⚪ Open")
            ring_rows.append({
                "ring": f"#{r['ring_id']}", "accounts": r["size"],
                "reviewed": f"{revd}/{r['size']}", "escalated": esc, "status": status,
                "lead_account": r["rep_account"], "potential_exposure": f"₹{r['exposure_rupees']:,}",
            })
        contained = sum(1 for row in ring_rows if row["status"].startswith("🟢"))
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Candidate rings", rings["n_candidate_rings"])
        r2.metric("Mules grouped", f"{rings['mules_in_rings']}/{rings['n_mules']}")
        r3.metric("Ring members escalated", total_esc, help="live — from your activity this session")
        r4.metric("Rings contained", f"{contained}/{rings['n_candidate_rings']}",
                  help="all members escalated this session")
        st.dataframe(pd.DataFrame(ring_rows), hide_index=True, use_container_width=True)
        st.caption(f"Status columns are **live** from your activity log — escalate a ring's members "
                   f"(Investigate tab) and it moves Open → In progress → Contained. Candidate Ring #1 is "
                   f"**~{v.get('ring1_intra_sim',0)/max(v.get('legit_subset_sim_mean',1e-9),1e-9):.0f}× tighter** "
                   f"than a random legit group ({v.get('ring1_intra_sim',0):.2f} vs {v.get('legit_subset_sim_mean',0):.2f}), "
                   f"stable under subsampling (Jaccard {v.get('ring1_subsample_stability_jaccard',0):.2f}) — a validated "
                   "proxy, **not** confirmed rings (needs bank link/device data, Phase-2).")
        fig = ROOT / "figures" / "11_mule_network.png"
        if fig.exists():
            st.image(str(fig), use_container_width=True,
                     caption="Behavioral-similarity graph of the 81 mules — candidate rings.")

    st.divider()
    st.subheader("📋 This session's activity")
    audit = st.session_state.get("audit", [])
    if audit:
        adf = pd.DataFrame(audit)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Reviews", len(adf))
        s2.metric("Distinct accounts", adf["account"].nunique())
        s3.metric("Alerts raised", int((adf["decision"] == "ALERT").sum()))
        s4.metric("Mules caught", int(((adf["decision"] == "ALERT") & (adf["ground_truth"] == "MULE")).sum()))
        st.markdown("**Risk score of accounts you reviewed (in order)**")
        st.line_chart(adf.reset_index()[["score"]])
    else:
        st.caption("No activity yet — investigate a few accounts and this fills in live.")

# ===================================== ACTIVITY LOG =====================================
with tab_log:
    st.subheader("🧾 Investigation activity — this session (live audit trail)")
    audit = st.session_state.get("audit", [])
    if audit:
        log_df = pd.DataFrame(audit)[::-1].reset_index(drop=True)   # newest first
        a, b, c = st.columns(3)
        a.metric("Reviews logged", len(audit))
        b.metric("Alerts raised", int((log_df["decision"] == "ALERT").sum()))
        c.metric("Mules caught in trail",
                 int(((log_df["decision"] == "ALERT") & (log_df["ground_truth"] == "MULE")).sum()))
        st.dataframe(log_df, hide_index=True, use_container_width=True)
        d1, d2, d3 = st.columns([1.4, 1.4, 3])
        d1.download_button("⬇ Export audit log (CSV)", log_df.to_csv(index=False).encode(),
                           file_name="sentinel_audit_log.csv", mime="text/csv")
        d2.download_button("📄 Download shift report", build_shift_report(),
                           file_name="sentinel_shift_report.txt", mime="text/plain",
                           help="A close-of-shift summary: reviews, escalations, rings contained, ₹ exposure addressed.")
        if d3.button("Clear log"):
            st.session_state.audit, st.session_state.last_sig = [], None
            st.rerun()
        with st.expander("📄 Preview shift report"):
            st.code(build_shift_report(), language="text")
        st.caption("Every account you review (and each threshold change) is timestamped and "
                   "appended here — an analyst-ready, exportable audit trail.")
    else:
        st.caption("Select accounts and adjust the threshold — each review is logged here "
                   "with a timestamp, the decision, and ground truth, then exportable as CSV.")

# =================================== MODEL & VALIDATION ===================================
with tab_model:
    st.caption("These are the **validated, fixed** properties of the model — they do not change "
               "per account or per threshold. The honest generalization numbers, not in-sample.")
    if metrics.get("threshold_table"):
        st.subheader("Precision / recall trade-off (held-out)")
        st.dataframe(pd.DataFrame(metrics["threshold_table"]), hide_index=True,
                     use_container_width=True)
    _fig = ROOT / "figures"
    if _fig.exists():
        st.subheader("📊 Model performance (out-of-fold)")
        figs = ["03_score_distribution.png", "01_pr_curve.png", "04_confusion_matrix.png",
                "05_calibration.png", "08_leakage_sensitivity.png", "09_cost_curve.png"]
        gcols = st.columns(3)
        for i, f in enumerate(figs):
            p = _fig / f
            if p.exists():
                gcols[i % 3].image(str(p), use_container_width=True)
