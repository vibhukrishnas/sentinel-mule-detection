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
from txn_ingest import (score_transactions, fuse_alerts, sample_transactions,
                        cross_channel_view, regulatory_connector, stream_score,
                        sample_regulatory_feed, sample_cross_channel)
try:
    import casestore                      # durable SQLite case store (best-effort)
    _PERSIST = True
except Exception:
    _PERSIST = False
try:
    import plotly.graph_objects as go     # interactive network / money-flow map
    _PLOTLY = True
except Exception:
    _PLOTLY = False
import math

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


@st.cache_data
def load_uncertainty():
    p = ART / "uncertainty.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_json(name):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_resource
def load_anomaly():
    p = ART / "anomaly_if.joblib"
    return joblib.load(p) if p.exists() else None


eng = load_engine()
cols, cat_maps = load_cols_maps()
metrics = load_metrics()
rings = load_rings()
unc = load_uncertainty()
anom_meta = load_json("anomaly_meta.json")
net_edges = load_json("mule_network_edges.json")
amlsim_flow = load_json("amlsim_flow.json")
_anom = load_anomaly()


def anomaly_scores(Xframe):
    """Unsupervised IsolationForest anomaly score (0..1) for each row, aligned to the
    model's columns. Returns None if the detector artifact is unavailable."""
    if _anom is None:
        return None
    try:
        Xa = Xframe.reindex(columns=_anom["columns"]).astype("float32")
        raw = -_anom["model"].score_samples(_anom["imputer"].transform(Xa))
        lo, hi = anom_meta["raw_lo"], anom_meta["raw_hi"]
        return pd.Series(np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1), index=Xframe.index)
    except Exception:
        return None


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
        "time (UTC)": datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S"),
        "account": int(i), "score": int(sc_row["risk_score"]), "band": sc_row["band"],
        "probability": round(float(sc_row["probability"]), 4), "threshold": round(threshold, 2),
        "decision": "ALERT" if sc_row["probability"] >= threshold else "clear",
        "ground_truth": g or "unknown", "ring": ring_id, "action": action,
    }


def log_audit(row):
    """Append to the in-session log AND best-effort persist to the durable case store."""
    st.session_state.audit.append(row)
    if _PERSIST:
        try:
            casestore.log_event(row, st.session_state.get("session_id", "?"))
        except Exception:
            pass


def build_shift_report():
    """A close-of-shift summary of this session — the analyst hand-off artifact."""
    audit = st.session_state.get("audit", [])
    reviewed, actions = audit_actions()
    esc = sorted(a for a, v in actions.items() if v == "ESCALATED")
    clr = [a for a, v in actions.items() if v == "CLEARED"]
    mon = [a for a, v in actions.items() if v == "MONITORED"]
    L = ["SENTINEL — Analyst shift report", "=" * 44,
         f"Generated (UTC): {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
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

def _circular_pos(items, cx=0.0, cy=0.0, r=1.0):
    n = max(len(items), 1)
    return {it: (cx + r * math.cos(2 * math.pi * i / n + math.pi / 2),
                 cy + r * math.sin(2 * math.pi * i / n + math.pi / 2)) for i, it in enumerate(items)}


def boi_ring_figure(ring, edges_all, risk_map, selected, escalated):
    """Interactive plot of one BOI candidate ring: nodes=accounts, edges=behavioral
    similarity. Honest — this is a similarity proxy, NOT money flow (BOI has no edges)."""
    members = list(ring["members"]); mset = set(members)
    pos = _circular_pos(members)
    redges = [e for e in (edges_all or []) if e["source"] in mset and e["target"] in mset]
    ex, ey = [], []
    for e in redges:
        x0, y0 = pos[e["source"]]; x1, y1 = pos[e["target"]]
        ex += [x0, x1, None]; ey += [y0, y1, None]
    et = go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1, color="#c9c9c9"), hoverinfo="none")
    nx_, ny_, col, sz, hov, lab = [], [], [], [], [], []
    for m in members:
        x, y = pos[m]; nx_.append(x); ny_.append(y); rsk = risk_map.get(m, 0)
        col.append("#7b1fa2" if m == selected else ("#d11f2d" if m in escalated else "#1f3b6e"))
        sz.append(34 if m == selected else 14 + rsk / 8)
        hov.append(f"Account #{m}<br>risk {rsk}/100" + ("<br>(selected)" if m == selected
                   else "<br>(escalated)" if m in escalated else ""))
        lab.append(f"#{m}")
    nt = go.Scatter(x=nx_, y=ny_, mode="markers+text", text=lab, textposition="top center",
                    textfont=dict(size=9), marker=dict(size=sz, color=col, line=dict(width=1.2, color="white")),
                    hovertext=hov, hoverinfo="text")
    fig = go.Figure([et, nt])
    fig.update_layout(showlegend=False, height=460, margin=dict(l=8, r=8, t=8, b=8),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig


def amlsim_flow_figure(typ):
    """Interactive REAL money-flow map for one AMLSim typology: directed arrows = money
    movement, labelled with ₹ amounts. Demonstrates SENTINEL on transaction-link data."""
    accts = typ["accounts"]; edges = typ["edges"]
    # fan-in/out: put the hub (most-connected account) in the centre; cycle: ring layout
    deg = {a: 0 for a in accts}
    for e in edges:
        deg[e["source"]] = deg.get(e["source"], 0) + 1; deg[e["target"]] = deg.get(e["target"], 0) + 1
    if typ["type"] in ("fan_in", "fan_out"):
        hub = max(deg, key=deg.get); rim = [a for a in accts if a != hub]
        pos = _circular_pos(rim, r=1.0); pos[hub] = (0.0, 0.0)
    else:
        pos = _circular_pos(accts, r=1.0)
    annos = []
    for e in edges:
        x0, y0 = pos[e["source"]]; x1, y1 = pos[e["target"]]
        annos.append(dict(ax=x0, ay=y0, x=x1, y=y1, xref="x", yref="y", axref="x", ayref="y",
                          showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=2, arrowcolor="#d11f2d"))
        annos.append(dict(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=f"₹{e['amount']:,.0f}",
                          showarrow=False, font=dict(size=9, color="#444"), bgcolor="rgba(255,255,255,0.7)"))
    nx_, ny_, lab = [], [], []
    for a in accts:
        x, y = pos[a]; nx_.append(x); ny_.append(y); lab.append(f"#{a}")
    nt = go.Scatter(x=nx_, y=ny_, mode="markers+text", text=lab, textposition="bottom center",
                    marker=dict(size=26, color="#1f3b6e", line=dict(width=1.5, color="white")),
                    hovertext=[f"Account #{a}" for a in accts], hoverinfo="text")
    fig = go.Figure([nt]); fig.update_layout(annotations=annos, showlegend=False, height=460,
        margin=dict(l=8, r=8, t=8, b=8), xaxis=dict(visible=False, range=[-1.4, 1.4]),
        yaxis=dict(visible=False, range=[-1.4, 1.4]), plot_bgcolor="white", paper_bgcolor="white")
    return fig


# ============================ SIDEBAR: data + global dial ============================
st.sidebar.header("📥 Data source")
st.sidebar.caption("Upload the provided **DataSet.csv** (raw) or a cleaned export. The full "
                   "dataset isn't shipped in this public repo; the demo runs on a small "
                   "committed sample. Uploaded data stays in this session — it is not stored.")
MAX_UPLOAD_MB, MAX_ROWS = 150, 20_000
up = st.sidebar.file_uploader(f"Upload account CSV (≤ {MAX_UPLOAD_MB} MB)", type=["csv"])
if up is not None:
    key = (up.name, up.size)
    if up.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.sidebar.error(f"That file is {up.size/1e6:.0f} MB — please keep it under "
                         f"{MAX_UPLOAD_MB} MB for this shared demo host.")
    elif st.session_state.get("src_key") != key:
        try:
            # nrows cap bounds memory regardless of file size (second guard vs OOM)
            df = pd.read_csv(up, index_col=0, low_memory=False, nrows=MAX_ROWS)
            capped = f" (capped to first {MAX_ROWS:,} rows)" if len(df) >= MAX_ROWS else ""
            X, y = prepare_frame(df, cols, cat_maps)
            st.session_state.src = (X, y, f"📤 {up.name} — {len(X):,} accounts{capped}")
            st.session_state.src_key = key
            st.session_state.scores = None
        except Exception as e:
            st.sidebar.error(f"Couldn't read that CSV — {type(e).__name__}: {e}")

if "src" not in st.session_state:
    st.session_state.src = read_default(cols, cat_maps)
    st.session_state.scores = None

X_all, y_all, src_label = st.session_state.src
st.sidebar.success(f"Active: {src_label}")


def tier_of(p):
    """Confidence tier from the abstention thresholds — computed app-side so it never
    depends on a cached engine instance (avoids stale @st.cache_resource crashes)."""
    lo = unc["t_lo"] if unc else 0.10
    hi = unc["t_hi"] if unc else 0.90
    return "CONFIDENT-MULE" if p >= hi else "CONFIDENT-LEGIT" if p <= lo else "UNCERTAIN"

# score the whole active dataset once (deployed model); recompute only on source change
if st.session_state.get("scores") is None:
    with st.spinner(f"Scoring {len(X_all):,} accounts…"):
        proba = eng.model.predict_proba(X_all.astype("float32"))[:, 1]
        s = (proba * 100).round().astype(int)
        df_sc = pd.DataFrame(
            {"risk_score": s, "probability": proba, "band": [band_for(v) for v in s]},
            index=X_all.index)
        an = anomaly_scores(X_all)               # unsupervised second opinion (0..1)
        if an is not None:
            df_sc["anomaly"] = an.reindex(df_sc.index).values
        st.session_state.scores = df_sc
allscores = st.session_state.scores
if "audit" not in st.session_state:
    st.session_state.audit = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = uuid.uuid4().hex[:8]
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

# editable cost assumptions — don't trust ours, plug in YOUR numbers (the ₹ impact recomputes)
st.sidebar.divider()
with st.sidebar.expander("💰 Cost assumptions (edit me)"):
    st.caption("The ₹ impact on the Analytics tab uses these — set your own.")
    mule_loss = st.number_input("Avg loss per missed mule (₹)", 0, 10_000_000, MULE_LOSS, 50_000)
    review_cost = st.number_input("Analyst review cost / alert (₹)", 0, 100_000, REVIEW_COST, 100)
    fp_harm = st.number_input("Cost of freezing a legit customer (₹)", 0, 1_000_000, FP_HARM, 1_000)

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
        "5. **🧾 Activity log** → download the **shift report** of everything you did.\n\n"
        "_First load after the app has been idle can take ~20s to wake (free host) — that's "
        "the hosting tier, not the model, which scores in ~33 ms._")
# Lead with VALUE, not metrics. The validated PR-AUC / leakage story lives on the Model tab (last).
_rd = load_json("boi/realtime_decisioning.json") or {}
_dec = _rd.get("decisioning") or {}
_grad = _dec.get("sentinel_graduated") or {}
_rt = _rd.get("realtime") or {}
v1, v2, v3, v4 = st.columns(4)
v1.metric("🚦 Auto-decided", f"{unc['coverage_auto_decided']*100:.0f}%" if unc else "—",
          help="The model auto-clears/flags the confident cases; only the ambiguous rest reaches a human.")
v2.metric("🛟 Wrongful freezes", _grad.get("wrongful_freezes", "—"),
          help="Innocent customers wrongly frozen under the graduated policy — protected.")
v3.metric("💰 Loss prevented", f"₹{_dec.get('expected_cost_reduction_rupees', 0)/1e7:.2f} Cr" if _dec else "—",
          help="Expected cost reduction vs a naïve freeze-at-0.5 policy.")
v4.metric("⚡ Throughput", f"{_rt.get('batch_throughput_accts_per_sec', '—')}/s" if _rt else "—",
          help="Scores on commodity CPU — no GPU needed.")
st.caption("Validated performance (PR-AUC, bootstrap CI, the leakage story) is on the **📈 Model & validation** tab.")

tab_inv, tab_net, tab_feeds, tab_alerts, tab_an, tab_log, tab_model, tab_copilot = st.tabs(
    ["🔍 Investigate", "🕸️ Account Network", "🔌 Feeds & Transactions", "🚨 Alert Management",
     "📊 Analytics", "🧾 Activity log", "📈 Model & validation", "🤖 AI Copilot"])

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
            st.session_state.pending_log = pick      # log exactly one review per check
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
            # abstention: decline to auto-decide the ambiguous middle (see src/uncertainty.py)
            tier = tier_of(sc["probability"])
            if tier == "UNCERTAIN":
                st.info("🤔 **UNCERTAIN — route to analyst.** The model declines to auto-decide "
                        "this account (probability in the ambiguous band) rather than make an "
                        "overconfident call. This is where the hardest mules hide.")
            else:
                st.caption(f"Model confidence tier: **{tier.replace('-', ' ').title()}** "
                           "(auto-decided; outside the abstention band).")

            # Model Trust Score (how much to trust THIS call) + Data-quality auditor
            n_feat = len(eng.columns)
            n_blank = int(pd.isna(account.reindex(eng.columns)).sum())
            completeness = 1 - n_blank / max(n_feat, 1)
            decisiveness = abs(sc["probability"] - 0.5) * 2          # 0 ambiguous … 1 decisive
            trust = int(round(100 * (0.6 * decisiveness + 0.4 * completeness)))
            tc1, tc2 = st.columns(2)
            tc1.metric("🔒 Model trust (this call)", f"{trust}%",
                       help="Blends decision-confidence with data completeness — how much weight to put on THIS prediction.")
            tc2.metric("📋 Data completeness", f"{completeness:.0%}",
                       help=f"{n_feat - n_blank:,}/{n_feat:,} features populated · {n_blank:,} blank")
            if completeness < 0.5:
                st.caption("⚠️ **Data-quality notice:** many features are blank for this account — "
                           "prediction is lower-trust; consider a step-up verification before acting.")

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
                            log_audit(audit_row(m, allscores.loc[m], gt(m), f"#{r['ring_id']}", "ESCALATED"))
                            added += 1
                    st.success(f"Escalated {added} member(s) of Ring #{r['ring_id']} — see the activity log & rings.")
                    st.rerun()

            # log exactly one "review" per Check click (threshold nudges don't inflate it)
            if st.session_state.get("pending_log") == pick:
                st.session_state.pending_log = None
                log_audit(audit_row(pick, sc, g, ring_id, "reviewed"))

            # ---- analyst case decision (accountability) ----
            st.markdown("**Analyst decision** (logged to the session log):")
            d1, d2, d3 = st.columns(3)
            decided = (d1.button("✅ Clear", key="act_clear") and "CLEARED") or \
                      (d2.button("👁 Monitor", key="act_mon") and "MONITORED") or \
                      (d3.button("🚩 Escalate to L2", key="act_esc") and "ESCALATED")
            if decided:
                log_audit(audit_row(pick, sc, g, ring_id, decided))
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

# ================================ NETWORK & MONEY-FLOW ================================
with tab_net:
    st.subheader("🗺️ Account network")
    st.caption("The network behind an account — the validated candidate rings among known mules. "
               "BOI is an account *snapshot* with no transaction edges, so this is a behavioural-"
               "**similarity** graph: a data-grounded *proxy* for the link data a bank holds. "
               "Confirmation needs bank link/device data (Phase-2).")
    # ----------------------------- BOI candidate-ring network -----------------------------
    st.subheader("🕸️ Candidate rings (behavioral-similarity graph)")
    st.caption("**Honest framing:** the BOI dataset is an account *snapshot* with **no transaction "
               "edges**, so this is a behavioral-**similarity** graph (accounts that look near-identical "
               "on leak-removed features) — a data-grounded *proxy* for the link data a bank holds, "
               "**not** money flow. Confirmation needs bank link/device data (Phase-2).")
    if rings and _PLOTLY and net_edges:
        rmap = {n["id"]: n["risk"] for n in net_edges["nodes"]}
        ridx = {f"Ring #{r['ring_id']} · {r['size']} accounts · ~₹{r['exposure_rupees']:,}": r for r in rings["rings"]}
        sel_pick = st.session_state.get("pick_id")
        default_key = next((k for k, r in ridx.items() if sel_pick in r["members"]), list(ridx)[0])
        choice = st.selectbox("Show ring", list(ridx), index=list(ridx).index(default_key))
        ring = ridx[choice]
        _, actions = audit_actions()
        escalated = {m for m in ring["members"] if actions.get(m) == "ESCALATED"}
        st.plotly_chart(boi_ring_figure(ring, net_edges["edges"], rmap,
                        sel_pick if sel_pick in ring["members"] else None, escalated),
                        use_container_width=True)
        st.caption("🟣 selected · 🔴 escalated this session · 🔵 other ring members · node size ∝ risk. "
                   "Escalate the ring from the **Investigate** tab and members turn red here.")
    elif not _PLOTLY:
        st.info("Interactive graph needs `plotly` (in requirements.txt). Static ring figure below.")
        fig = ROOT / "figures" / "11_mule_network.png"
        if fig.exists():
            st.image(str(fig), use_container_width=True)

    # ----------------------------- anomaly detection (honest) -----------------------------
    if anom_meta:
        st.divider()
        st.subheader("🧪 Anomaly detection — tested honestly (Isolation Forest + LOF)")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Supervised PR-AUC", f"{anom_meta['supervised_pr_auc']:.3f}")
        a2.metric("Isolation Forest", f"{anom_meta['anomaly_pr_auc']:.3f}", help="Unsupervised — alone")
        a3.metric("Local Outlier Factor", f"{anom_meta.get('lof_pr_auc', 0):.3f}", help="2nd unsupervised detector")
        a4.metric("Naïve 60/20/20 hybrid", f"{anom_meta['hybrid_pr_auc']:.3f}",
                  delta=f"{anom_meta['hybrid_pr_auc']-anom_meta['supervised_pr_auc']:.2f}", delta_color="inverse",
                  help="Blending anomaly IN measurably HURTS the score")
        st.warning("**Honest finding (PS2 asks for anomaly detection — so we built it *and measured it*):** "
                   f"**two** independent unsupervised detectors — IsolationForest ({anom_meta['anomaly_pr_auc']:.3f}) "
                   f"and LOF ({anom_meta.get('lof_pr_auc',0):.3f}) — both score **near random** vs the supervised "
                   f"{anom_meta['supervised_pr_auc']:.3f}. Mules here are **not statistical outliers** (the "
                   "'invisible' ones sit at the *median* anomaly rank), so a fixed 60/20/20 hybrid actually "
                   f"**drops** the score to {anom_meta['hybrid_pr_auc']:.3f}. We therefore surface anomaly as a "
                   "**second opinion** in the queue but do **not** fold it into the deployed score. Shipping a "
                   "blend we've proven makes us worse would be the exact dishonesty our leakage auditor exists "
                   "to catch — so we report the verdict instead. *That* is the rigor.")

# ============================ FEEDS & TRANSACTIONS ============================
with tab_feeds:
    st.subheader("🔌 Feeds & Transactions — real-time ingest · cross-channel · regulatory")
    st.caption("Ingests financial transactions, fraud/TMS/govt alert feeds, and regulatory cyber-fraud tickets. "
               "Scores suspicious transactions in real time, correlates accounts across payment rails (UPI/IMPS/card/NEFT), "
               "and corroborates every external alert against the deployed model risk.")

    # auto-load sample data on first visit so the tab is never blank
    if "txn_df" not in st.session_state:
        st.session_state.txn_df = sample_transactions()
    if "cc_df" not in st.session_state:
        st.session_state.cc_df = sample_cross_channel()
    if "reg_df" not in st.session_state:
        st.session_state.reg_df = sample_regulatory_feed()
    if "alert_df" not in st.session_state:
        ids = allscores.sort_values("risk_score", ascending=False).index[:4].tolist()
        st.session_state.alert_df = pd.DataFrame({
            "account": ids,
            "source": ["TMS", "FraudMon", "TMS", "FraudMon"],
            "severity": ["HIGH", "HIGH", "MEDIUM", "LOW"],
        })

    sub_txn, sub_cc, sub_reg, sub_alert = st.tabs([
        "💳 Transaction feed", "🔗 Cross-channel", "🏛️ Regulatory feed", "📨 Alert / TMS feed"])

    # ── 1. TRANSACTION FEED ──────────────────────────────────────────────────
    with sub_txn:
        c1, c2 = st.columns([1, 1])
        up_txn = c1.file_uploader("Upload a transaction CSV", type=["csv"], key="txn_up",
                                  help="Any bank export with an amount column.")
        if c2.button("↺ Reset to synthetic sample", key="txn_sample"):
            st.session_state.txn_df = sample_transactions()
        if up_txn is not None:
            try:
                st.session_state.txn_df = pd.read_csv(up_txn, low_memory=False)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
        try:
            res = score_transactions(st.session_state.txn_df)
            out, rollup = res["transactions"], res["rollup"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Transactions ingested", f"{len(out):,}")
            m2.metric("🚩 Suspicious flagged", f"{res['n_flagged']:,}")
            m3.metric("Accounts implicated", f"{int((rollup['suspicious_txns'] > 0).sum()):,}" if len(rollup) else "—")
            m4.metric("Flag rate", f"{res['n_flagged']/max(len(out),1)*100:.0f}%")
            st.markdown("**Flagged transactions — scored in real time, reason-tagged:**")
            show = out[out["flag"]].sort_values("suspicion_score", ascending=False)
            st.dataframe(show[[res["amount_col"], "suspicion_score", "reasons"] +
                              ([res["orig_col"]] if res["orig_col"] else [])].head(200),
                         use_container_width=True, hide_index=True)
            if len(rollup):
                st.markdown("**Account-level circulation view — where to act to stop the money moving:**")
                st.dataframe(rollup.head(50), use_container_width=True, hide_index=True)
            st.markdown("**⚡ Real-time stream — each transaction scored + action assigned as it arrives:**")
            stream_rows = [ev for ev in stream_score(out.head(20))]
            if stream_rows:
                st.dataframe(pd.DataFrame(stream_rows), use_container_width=True, hide_index=True)
            st.caption("Typologies: high-value spikes · structuring (smurfing) · account-drain · "
                       "pass-through · cash-out/transfer · transfer→cash-out layering · high velocity · cross-channel layering")
        except ValueError as e:
            st.warning(str(e))

    # ── 2. CROSS-CHANNEL CORRELATION ─────────────────────────────────────────
    with sub_cc:
        st.caption("Merges transactions across **UPI / IMPS / card / NEFT** rails. "
                   "Accounts active on ≥2 rails = cross-channel layering signal that single-rail TMS cannot see.")
        c1, c2 = st.columns([1, 1])
        up_cc = c1.file_uploader("Upload multi-rail transaction CSV", type=["csv"], key="cc_up",
                                 help="Needs a channel/rail column (channel / rail / mode / payment_channel).")
        if c2.button("↺ Reset to synthetic feed", key="cc_sample"):
            st.session_state.cc_df = sample_cross_channel()
        if up_cc is not None:
            try:
                st.session_state.cc_df = pd.read_csv(up_cc, low_memory=False)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
        try:
            cc = cross_channel_view(st.session_state.cc_df)
            m1, m2, m3 = st.columns(3)
            m1.metric("Payment rails detected", ", ".join(cc["channels_seen"]))
            m2.metric("Accounts analysed", cc["n_accounts"])
            m3.metric("🔗 Cross-channel (≥2 rails)", cc["n_cross_channel"],
                      help="These accounts move money across multiple payment rails — classic layering.")
            st.markdown("**Cross-channel account view** (multi-rail accounts first):")
            st.dataframe(cc["accounts"].head(50), use_container_width=True, hide_index=True)
            st.caption(f"Rails: {', '.join(cc['channels_seen'])}. "
                       "Multi-rail accounts are highlighted — corroborate with the model score for escalation.")
        except ValueError as e:
            st.warning(str(e))

    # ── 3. REGULATORY FEED (I4C / NCRP / RBI-style) ─────────────────────────
    with sub_reg:
        st.caption("Ingests **govt cyber-fraud tickets** (I4C / NCRP / RBI format), normalises through "
                   "the regulatory connector, then corroborates against the deployed model. "
                   "An account flagged by both a ticket and the model → immediate escalation.")
        c1, c2 = st.columns([1, 1])
        up_reg = c1.file_uploader("Upload regulatory ticket CSV", type=["csv"], key="reg_up",
                                  help="I4C/NCRP-style: needs a beneficiary account column.")
        if c2.button("↺ Reset to synthetic I4C/NCRP tickets", key="reg_sample"):
            st.session_state.reg_df = sample_regulatory_feed()
        if up_reg is not None:
            try:
                st.session_state.reg_df = pd.read_csv(up_reg, low_memory=False)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
        try:
            st.markdown("**Raw regulatory tickets (I4C/NCRP/RBI format):**")
            st.dataframe(st.session_state.reg_df, use_container_width=True, hide_index=True)
            normalised = regulatory_connector(st.session_state.reg_df)
            fr = fuse_alerts(normalised, allscores["risk_score"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Regulatory tickets ingested", fr["n_alerts"])
            m2.metric("✅ Corroborated by model", fr["n_corroborated"])
            m3.metric("Source", "Govt-I4C")
            st.markdown("**Corroboration — ticket × deployed model risk:**")
            st.dataframe(fr["alerts"], use_container_width=True, hide_index=True)
            st.caption("Ticket normalised → model risk lookup → **CORROBORATED** (risk ≥70) / "
                       "*model also elevated* (≥40) / review. Two independent signals = defensible escalation.")
        except ValueError as e:
            st.warning(str(e))

    # ── 4. FRAUD-MON / TMS ALERT FEED ────────────────────────────────────────
    with sub_alert:
        st.caption("Ingest external **TMS / fraud-monitoring** alert tickets and corroborate against model risk.")
        c1, c2 = st.columns([1, 1])
        up_al = c1.file_uploader("Upload alert-ticket CSV", type=["csv"], key="al_up",
                                 help="Needs account column; optional source & severity.")
        if c2.button("↺ Reset to synthetic TMS alerts", key="al_sample"):
            ids = allscores.sort_values("risk_score", ascending=False).index[:4].tolist()
            st.session_state.alert_df = pd.DataFrame({
                "account": ids, "source": ["TMS", "FraudMon", "TMS", "FraudMon"],
                "severity": ["HIGH", "HIGH", "MEDIUM", "LOW"],
            })
        if up_al is not None:
            try:
                st.session_state.alert_df = pd.read_csv(up_al, low_memory=False)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
        try:
            fr = fuse_alerts(st.session_state.alert_df, allscores["risk_score"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Alerts ingested", fr["n_alerts"])
            m2.metric("✅ Corroborated", fr["n_corroborated"])
            m3.metric("Sources", len(fr["by_source"]))
            st.dataframe(fr["alerts"], use_container_width=True, hide_index=True)
        except ValueError as e:
                st.warning(f"Error parsing alert feed: {e}")
                if "needs an account" in str(e).lower():
                    st.info("💡 **Did you mean to upload the account snapshot (`DataSet.csv`)?** The main dataset goes in the **sidebar on the left ('Data source')**. This uploader is specifically for external alert tickets.")
        else:
            st.info("Upload an alert-ticket feed or load the synthetic TMS sample.")

# =================================== ALERT MANAGEMENT ===================================
with tab_alerts:
    st.subheader("🚨 Alert management — accountable case workflow")
    st.caption("This is the **post-detection workflow** — what a fraud desk actually operates after the model "
               "flags an account. Every account at/above the **alert threshold** (sidebar) becomes a case with a "
               "**status** and an **owner**, turning a score into accountable action. Works on the committed "
               "sample **or any uploaded dataset**.")
    STATUSES = ["NEW", "INVESTIGATING", "ESCALATED", "CLEARED"]
    stt = st.session_state.setdefault("alert_status", {})
    own = st.session_state.setdefault("alert_owner", {})
    logged = st.session_state.setdefault("alert_logged", {})
    alerts = allscores[allscores["probability"] >= threshold].sort_values("risk_score", ascending=False)

    # sync analyst edits (widgets write to session_state) BEFORE counting/logging -> live metrics
    for i in alerts.index:
        sk, ok = f"ast_{int(i)}", f"own_{int(i)}"
        if sk in st.session_state:
            stt[int(i)] = st.session_state[sk]
        if ok in st.session_state:
            own[int(i)] = st.session_state[ok]
    # log every NEW status change or owner assignment to the durable audit trail (real accountability)
    for i in alerts.index:
        ii = int(i); cs = stt.get(ii, "NEW"); ow = own.get(ii, "")
        if (cs != "NEW" or ow) and logged.get(ii) != (cs, ow):
            rg = ring_of(i)
            act = cs + (f" · owner {ow}" if ow else "")
            log_audit(audit_row(i, allscores.loc[i], gt(i), f"#{rg['ring_id']}" if rg else "—", act))
            logged[ii] = (cs, ow)

    def _ast(i):
        return stt.get(int(i), "NEW")
    counts = {s: 0 for s in STATUSES}
    for i in alerts.index:
        counts[_ast(i)] += 1
    assigned = sum(1 for i in alerts.index if own.get(int(i)))
    a1, a2, a3, a4 = st.columns(4)
    a1.metric(f"🚨 Open alerts @ {threshold:.2f}", f"{len(alerts):,}")
    a2.metric("🟡 New / untouched", counts["NEW"])
    a3.metric("🔵 Investigating + 🔴 Escalated", counts["INVESTIGATING"] + counts["ESCALATED"])
    a4.metric("✅ Cleared", counts["CLEARED"])
    handled = counts["INVESTIGATING"] + counts["ESCALATED"] + counts["CLEARED"]
    st.caption(f"Handled **{handled}/{len(alerts)}** · **{assigned}** assigned to an owner this session · "
               "SLA: Critical (≥90) same-day, High (70–89) 48h.")
    st.info("**Where ‘assign’ enacts:** setting an owner + status writes a timestamped, **named** record to the "
            "**🧾 Activity log** and the **durable SQLite case store** — e.g. `ESCALATED · owner priya`. So every "
            "action taken on a customer account is attributable to a specific analyst — the audit accountability a "
            "bank (and an RBI / SAR review) requires. Change a status below, then open the **Activity log** tab.")

    if not len(alerts):
        st.info("No alerts at this threshold — lower the sidebar dial to surface more.")
    else:
        hdr = st.columns([0.8, 1, 1, 1.4, 1.6, 1.4])
        for c, t in zip(hdr, ["Account", "Risk", "Truth", "Owner", "Status", "Logged as"]):
            c.markdown(f"**{t}**")
        for i in alerts.head(25).index:
            r = allscores.loc[i]
            c = st.columns([0.8, 1, 1, 1.4, 1.6, 1.4])
            c[0].markdown(f"#{int(i)}")
            c[1].markdown(f"{int(r['risk_score'])} · {r['band']}")
            c[2].markdown(gt(i) or "—")
            c[3].text_input("owner", key=f"own_{int(i)}", label_visibility="collapsed", placeholder="assign…")
            c[4].selectbox("status", STATUSES, index=STATUSES.index(_ast(i)), key=f"ast_{int(i)}",
                           label_visibility="collapsed")
            ow = own.get(int(i), "")
            c[5].markdown(f"_{_ast(i).title()}{(' · ' + ow) if ow else ''}_")
        if len(alerts) > 25:
            st.caption(f"Showing top 25 of {len(alerts):,} alerts by risk.")

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
        net = tp * mule_loss - alerts * review_cost - fp * fp_harm
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Alerts raised", f"{alerts:,}")
        k2.metric("Mules caught", f"{tp}/{tp+fn}", help=f"recall {recall:.0%}")
        k3.metric("Precision", f"{prec:.0%}")
        k4.metric("False alarms", f"{fp:,}")
        k5.metric("Net ₹ impact", f"₹{net/1e5:.2f} L",
                  help=f"YOUR assumptions (sidebar): mule loss ₹{mule_loss:,}, review ₹{review_cost:,}, "
                       f"false-freeze ₹{fp_harm:,}. Edit them — this recomputes live.")
        cm = pd.DataFrame([[tp, fn], [fp, tn]],
                          index=["Actual MULE", "Actual legit"],
                          columns=["Flagged", "Not flagged"])
        st.markdown("**Confusion matrix at this threshold**")
        st.dataframe(cm, use_container_width=False)
    else:
        alerts = int((allscores["probability"] >= threshold).sum())
        st.metric("Alerts raised", f"{alerts:,} / {len(allscores):,}")
        st.caption("Upload data with the target column (F3924) to see recall / precision / ₹ impact.")

    if unc:
        st.divider()
        st.subheader("🤔 Abstention policy — decline, don't guess (validated out-of-fold)")
        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Auto-decided", f"{unc['coverage_auto_decided']:.1%}")
        u2.metric("Auto-zone error", f"{unc['auto_zone_error_rate']:.2%}")
        u3.metric("Routed to analyst", f"{unc['review_rate']:.1%}",
                  help=f"{unc['n_uncertain']} of {unc['n']} accounts land in the ambiguous band")
        u4.metric("Hard mules caught by review", unc["mules_routed_to_review"],
                  help="real mules that score ambiguously — sent to a human instead of mis-cleared")
        st.caption(f"On out-of-fold data the model **auto-decides {unc['coverage_auto_decided']:.0%}** of "
                   f"accounts at **{unc['auto_zone_error_rate']:.2%}** error and **routes the ambiguous "
                   f"{unc['review_rate']:.0%}** (probability {unc['t_lo']:.2f}–{unc['t_hi']:.2f}) to an analyst "
                   f"— rather than make overconfident calls on the hard tail. Confident-mule zone precision "
                   f"{unc['confident_mule_precision']:.0%}; confident-legit NPV {unc['confident_legit_npv']:.1%}.")

    st.divider()
    st.subheader("📉 Alert-fatigue reduction & analyst-capacity optimizer")
    st.caption("Banks don't fail at *detecting* — they drown in alerts. This sizes the workload SENTINEL "
               "actually puts on a desk, and finds the threshold that fits your team.")
    t_lo_v = unc["t_lo"] if unc else 0.10
    t_hi_v = unc["t_hi"] if unc else 0.90
    N = len(allscores); pcol = allscores["probability"]
    auto_clear = int((pcol <= t_lo_v).sum())
    priority = int((pcol > t_lo_v).sum())          # everything not confidently-legit needs a human
    af1, af2, af3 = st.columns(3)
    af1.metric("Without prioritization", f"{N:,}", help="Every account a human would have to triage")
    af2.metric("With SENTINEL (priority queue)", f"{priority:,}",
               delta=f"-{(1-priority/max(N,1))*100:.0f}% workload", delta_color="inverse",
               help=f"{auto_clear:,} confidently-legit accounts auto-cleared; only the rest reach a human")
    af3.metric("Auto-cleared (no human)", f"{auto_clear:,}", help=f"p ≤ {t_lo_v:.2f} — confident-legit")
    cap1, cap2 = st.columns(2)
    with cap1:
        n_analysts = st.number_input("Analysts on the desk", 1, 500, 20, 1, key="cap_analysts")
        mins = st.number_input("Minutes per review", 1, 120, 15, 1, key="cap_mins")
    daily_cap = int(n_analysts * 8 * 60 / mins)    # 8-hour shift
    # recommend the threshold whose alert volume fits capacity
    rec_thr, rec_alerts = None, None
    for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
        a = int((pcol >= t).sum())
        if a <= daily_cap:
            rec_thr, rec_alerts = float(t), a; break
    with cap2:
        st.metric("Daily review capacity", f"{daily_cap:,} cases/day",
                  help=f"{n_analysts} analysts × 8h ÷ {mins} min")
        if rec_thr is not None:
            need = int(np.ceil(rec_alerts * mins / (8 * 60)))
            st.metric("Recommended threshold", f"{rec_thr:.2f}",
                      help=f"Fits your capacity: {rec_alerts:,} alerts/day ≈ {need} analyst-days")
            st.caption(f"At threshold **{rec_thr:.2f}**, SENTINEL raises **{rec_alerts:,} alerts** — workable "
                       f"by ~**{need}** of your {n_analysts} analysts, leaving headroom for investigations.")
        else:
            st.caption("Even at the strictest threshold the alert volume exceeds capacity — add analysts "
                       "or raise the auto-clear bar.")

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
        st.markdown("**What the colours mean:** each colour is a **distinct candidate ring** — "
                    "Ring #1 🔴 red, #2 🔵 blue, #3 🟢 green, #4 🟣 purple, #5 🟠 orange; "
                    "**grey** nodes are the 14 mules not grouped into any ring. Edges connect "
                    "behaviourally near-identical accounts.")
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
    st.subheader("🧾 Investigation activity — session log (live, in-memory)")
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
        if d3.button("Clear session log"):
            st.session_state.audit = []
            st.rerun()
        with st.expander("📄 Preview shift report"):
            st.code(build_shift_report(), language="text")
        st.caption("Each review/decision is timestamped and appended here — an exportable "
                   "**session log**, also persisted to a durable case store (below).")
    # durable case store — survives refresh / other sessions (this deployment)
    if _PERSIST:
        try:
            persisted = casestore.load_all()
            with st.expander(f"🗄️ Durable case store — {len(persisted):,} persisted events "
                             "(across sessions on this deployment)"):
                st.caption(f"SQLite at `{casestore.DB_PATH}` — durable on a real/local deployment. "
                           "On Streamlit Community Cloud the filesystem is ephemeral, so this survives "
                           "refreshes and other sessions but resets on app reboot; production would point "
                           "`SENTINEL_DB` at a managed SQL store.")
                if len(persisted):
                    st.dataframe(persisted, hide_index=True, use_container_width=True)
                    st.download_button("⬇ Export full case store (CSV)",
                                       persisted.to_csv(index=False).encode(),
                                       file_name="sentinel_case_store.csv", mime="text/csv")
        except Exception as e:
            st.caption(f"(Case store unavailable here: {type(e).__name__})")
    else:
        st.caption("Select an account and click **Check risk score** — each review is logged "
                   "here with a timestamp, the decision, and ground truth, exportable as CSV.")

# =================================== MODEL & VALIDATION ===================================
with tab_model:
    st.caption("These are the **validated, fixed** properties of the model — they do not change "
               "per account or per threshold. The honest generalization numbers, not in-sample.")

    # ---- THE differentiator: leakage audit before/after (0.998 -> leak -> 0.885) ----
    sens = load_json("leak_sensitivity.json")
    intg = load_json("integrity_audit.json")
    st.subheader("🎭 The leakage story — why our number is *lower*, and *true*")
    naive_pr = next((r["pr_auc"] for r in sens if r.get("leak_thr", 0) > 1), 0.998) if sens else 0.998
    honest_pr = metrics.get("cv_pr_auc", 0.885)
    # bucket-leaks REMOVED by the deployed pipeline (conservative 0.10 fraud-rate threshold)
    n_block = None
    if sens:
        cand = [r for r in sens if abs(r.get("leak_thr", 9) - 0.10) < 1e-6]
        n_block = cand[0]["n_leaks"] if cand else None
    if n_block is None and intg:
        n_block = len(intg.get("auto_block_recommended", []))
    s1, s2, s3 = st.columns(3)
    s1.metric("① Naïve model (all features)", f"{naive_pr:.3f}", help="What most teams will proudly report")
    s2.metric("② Auditor flags leakage", "F3912 + bucket leaks",
              help=f"{n_block} CRITICAL/HIGH non-bank features auto-blocked" if n_block else "label-proxy + month-stamp leaks")
    s3.metric("③ Honest model (leaks removed)", f"{honest_pr:.3f}",
              delta=f"{honest_pr-naive_pr:.3f}", delta_color="off",
              help="The defensible number — ~100× the random baseline")
    st.error(f"**A naïve model scores PR-AUC {naive_pr:.3f} — and it's a lie.** It reads `F3912`, a "
             "post-hoc fraud flag aligned ~96% with the label (the model is reading the answer). Our "
             "**Data Integrity Auditor** catches it" + (f" plus ~{n_block} bucket/range leaks" if n_block else "") +
             f", we remove them, and report **{honest_pr:.3f}** — the number that survives in production. "
             "**Every team reporting ~0.99 on this dataset is reporting the leak.** We're the team that found it.")
    if sens:
        sdf = pd.DataFrame(sens)
        sdf["stage"] = ["no removal (LEAK)" if r > 1 else f"≥{r:.0%} fraud-bucket removed"
                        for r in sdf["leak_thr"]]
        st.markdown("**Leakage-sensitivity sweep** — PR-AUC as leaks are progressively removed; it "
                    "**plateaus at ~0.86**, proving the remaining signal is genuine behaviour, not leakage:")
        st.line_chart(sdf.set_index("stage")["pr_auc"])
    st.divider()

    boot = load_json("bootstrap_ci.json")
    infold = load_json("infold_leak_check.json")
    robust = load_json("robustness.json")
    if boot or infold:
        st.subheader("🔬 Rigor — leakage-proof, honestly bounded")
        rc = st.columns(3)
        if boot:
            ci = boot["bootstrap_pr_auc_ci95"]
            rc[0].metric("OOF PR-AUC (non-overlapping 10-fold)", f"{boot['oof_pr_auc']:.3f}")
            rc[1].metric("Bootstrap 95% CI", f"{ci[0]:.3f}–{ci[1]:.3f}",
                         help="Assumption-light CI by resampling accounts (B=2000) — fixes the overlapping-fold caveat")
        if infold:
            rc[2].metric("In-fold leak-check PR-AUC", f"{infold['pr_auc_mean']:.3f} ± {infold['pr_auc_std']:.3f}",
                         help="Blocklist re-detected inside each fold (never sees val labels) — vs 0.885 headline")
        st.caption("The headline (0.885) is reproduced by a non-overlapping-fold OOF (0.902, CI 0.84–0.95) "
                   "and survives in-fold leak detection (0.907) — so it is neither overlap-optimistic nor "
                   "blocklist-inflated.")

    if robust:
        st.subheader("🛡️ Robustness & sensitivity (stated honestly)")
        nz = robust.get("noise_injection_continuous", {})
        dp = robust.get("feature_dropout", {})
        base = robust.get("baseline_oof_pr_auc", 0)
        if dp:
            st.markdown("**Missing data (feature dropout)** — the realistic failure mode. "
                        "Model consumes NaN natively and degrades *gracefully*:")
            st.line_chart(pd.DataFrame({"PR-AUC": {f"{int(float(k)*100)}%": v for k, v in dp.items()}}))
        cc = st.columns(2)
        if dp:
            cc[0].metric("PR-AUC · 25% features missing", f"{dp.get('0.25', 0):.3f}",
                         help=f"vs {base:.3f} baseline — graceful")
        if nz:
            cc[1].metric("PR-AUC · 0.1σ noise on all continuous", f"{nz.get('0.1', 0):.3f}",
                         delta=f"{nz.get('0.1',0)-base:.2f}", delta_color="inverse",
                         help="Simultaneous Gaussian noise on all 1,922 continuous features")
        st.caption("**Honest finding:** robust to *missing* data (graceful), but **sensitive to feature "
                   "*noise*** — simultaneous 0.1σ perturbation of all continuous features drops PR-AUC "
                   f"to {nz.get('0.1',0):.2f}. The model leans on precise values (expected with 81 positives "
                   "and concentrated signal). Mitigations already in SENTINEL: the conservative **floor "
                   "(~0.81)** and the **abstention layer** that routes ambiguous accounts to analysts.")
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

# ======================================= AI COPILOT =======================================
with tab_copilot:
    st.subheader("🤖 AI Copilot — ask how SENTINEL works")
    st.caption("Answers grounded in the live numbers. Runs in-app (no external calls); can be wired "
               "to an LLM endpoint for open-domain Q&A.")
    _pra = metrics.get("cv_pr_auc", 0.885); _roc = metrics.get("cv_roc_auc", 0.98)
    _naive = next((r["pr_auc"] for r in (load_json("leak_sensitivity.json") or []) if r.get("leak_thr", 0) > 1), 0.998)
    _nleak = next((r["n_leaks"] for r in (load_json("leak_sensitivity.json") or []) if abs(r.get("leak_thr", 9) - 0.10) < 1e-6), 582)
    _cov = (unc or {}).get("coverage_auto_decided", 0.99)
    _wf = _grad.get("wrongful_freezes", 0); _ms = _grad.get("mules_surfaced_by_stepup", "several")
    _tp = _rt.get("batch_throughput_accts_per_sec", "~2,300")
    _anp = (anom_meta or {}).get("anomaly_pr_auc", 0.01)
    KB = [
        (["how", "work", "what is", "overview", "about", "pipeline"],
         "SENTINEL scores each account 0–100 for mule-risk with a calibrated LightGBM model, explains "
         "every alert with SHAP, groups look-alike accounts into candidate rings, and recommends a "
         "graduated action (clear → step-up verify → freeze). It abstains on ambiguous cases and routes "
         "them to a human."),
        (["leak", "f3912", "0.99", "0.998", "honest", "why low", "lie"],
         f"The raw data has target leakage: F3912 is a post-hoc fraud flag ~96% aligned with the label, so "
         f"a naïve model scores PR-AUC {_naive:.3f} by reading the answer. Our Data Integrity Auditor removes "
         f"it plus {_nleak} bucket leaks, giving a defensible PR-AUC {_pra:.3f} that holds in production."),
        (["pr-auc", "prauc", "accuracy", "metric", "performance", "roc", "how good"],
         f"PR-AUC {_pra:.3f}, ROC-AUC {_roc:.3f}. We lead with PR-AUC (not accuracy) because at ~1% fraud, "
         "'predict all clean' is 99% accurate yet useless."),
        (["ring", "network", "cluster", "graph", "colour", "color", "legend"],
         "We group mules via a behavioural-similarity graph. Node colour = risk band: 🔴 ≥90, 🟠 70–89, "
         "🔵 40–69, 🟢 <40; size ∝ risk; 🟣 = selected/escalated. It's a proxy for bank link data (Phase-2)."),
        (["abstain", "uncertain", "confidence", "route", "review", "fatigue"],
         f"Instead of forcing every call, SENTINEL auto-decides {_cov*100:.0f}% of accounts and routes only "
         "the ambiguous band to a human — crushing alert fatigue instead of guessing on the hard tail."),
        (["freeze", "customer", "harm", "decision", "action", "step-up", "stepup", "contain"],
         f"Graduated response: clear → step-up verify → freeze. We freeze only high-confidence mules and "
         f"step-up-verify the ambiguous middle (a mule fails KYC re-check, a real customer passes). Result on "
         f"the sample: {_wf} wrongful freezes, {_ms} hard mules surfaced — protecting customers and analysts."),
        (["latency", "fast", "real-time", "realtime", "speed", "gpu", "ms"],
         f"Real-time on commodity CPU — {_tp} accounts/sec batch. GPUs give no benefit on tabular boosting, "
         "so there's no GPU dependency: easier for a bank to deploy on existing infra."),
        (["anomaly", "isolation forest", "outlier", "lof", "unsupervised"],
         f"We tested unsupervised anomaly detection (Isolation Forest + LOF). Mules here are NOT outliers — "
         f"both score near random (~{_anp:.3f}), and a naïve hybrid would hurt the score. So this is a "
         "supervised problem; we report that honestly rather than ship a metric that makes us worse."),
        (["upload", "csv", "new data", "dataset", "my data"],
         "Use the sidebar uploader. The whole dashboard re-scores on your CSV — Investigate, Network, Alert "
         "Management and Analytics all update. Uploaded data is processed in-session, not stored."),
        (["alert", "threshold", "queue", "manage", "accountability", "sla"],
         "Alert Management turns every account above the alert threshold into a case with a status "
         "(New → Investigating → Escalated → Cleared) and an owner. Works on the sample or any uploaded data."),
        (["mule", "mule account"],
         "A mule account receives and moves illicit funds — often a real customer's account that's been "
         "recruited or compromised. Catching mules early breaks the laundering chain before cash-out."),
        (["deploy", "production", "bank", "integrate", "scale"],
         "Deployment is API-first: a FastAPI service hosts the model (/score, /report, /network) and the UI. "
         "It runs on CPU, slotting into existing bank infra; auth, rate-limiting and a durable case store are built in."),
    ]
    SUGG = ["How does SENTINEL work?", "Why isn't your PR-AUC 0.99?", "What do the ring colours mean?",
            "How do you protect innocent customers?", "Is it real-time?", "Did you test anomaly detection?"]

    def _answer(q):
        t = q.lower(); best, sc = None, 0
        for kws, ans in KB:
            m = sum(1 for k in kws if k in t)
            if m > sc:
                sc, best = m, ans
        return best if best and sc else ("I can answer questions about how SENTINEL works — the model, the "
            "leakage story, rings, abstention, customer-fair decisions, real-time performance, anomaly "
            "detection, uploading data, and deployment. Try one of the suggested questions.")

    if "copilot_log" not in st.session_state:
        st.session_state.copilot_log = []
    st.markdown("**Suggested:**")
    scols = st.columns(3)
    for j, sug in enumerate(SUGG):
        if scols[j % 3].button(sug, key=f"sug_{j}"):
            st.session_state.copilot_log.append((sug, _answer(sug)))
    q = st.chat_input("Ask the copilot…") if hasattr(st, "chat_input") else None
    if q is None:
        q = st.text_input("Ask the copilot…", key="copilot_q")
        if st.button("Ask", key="copilot_ask") and q:
            st.session_state.copilot_log.append((q, _answer(q)))
    elif q:
        st.session_state.copilot_log.append((q, _answer(q)))
    for qq, aa in reversed(st.session_state.copilot_log[-12:]):
        st.markdown(f"**🧑 {qq}**")
        st.info(f"🤖 {aa}")
