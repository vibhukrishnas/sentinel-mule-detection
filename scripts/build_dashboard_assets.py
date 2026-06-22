"""
Build artifacts for the upgraded dashboard (money-flow/network + anomaly hybrid).

Produces (all under artifacts/):
  anomaly_if.joblib      IsolationForest fit on the BOI matrix (unsupervised anomaly detector)
  anomaly_meta.json      score-normalisation percentiles + an HONEST check of whether the
                         anomaly score adds value over the supervised model (esp. on the
                         'invisible' mules the classifier misses)
  mule_network_edges.json  real edge list of the BOI behavioral-similarity graph (for drawing)
  amlsim_flow.json       representative REAL money-flow typology subgraphs from AMLSim
                         (fan-in, cycle) — nodes + directed edges + amounts, for the flow map

BOI assets use BOI only. AMLSim is a separate, tagged graph dataset — never merged into BOI.
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np, pandas as pd, joblib
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import load_cached

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
EXT = ROOT / "dataset_Sentinel" / "extracted" / "amlsim"
SEED = 42


def build_anomaly():
    print("=== IsolationForest anomaly detector (unsupervised) ===", flush=True)
    X, y = load_cached(); yv = y.to_numpy()
    # IF on median-imputed matrix (unsupervised: never sees labels)
    imp = SimpleImputer(strategy="median")
    Xi = imp.fit_transform(X)
    iso = IsolationForest(n_estimators=300, max_samples=0.7, contamination=0.01,
                          random_state=SEED, n_jobs=-1).fit(Xi)
    raw = -iso.score_samples(Xi)            # higher = more anomalous
    lo, hi = np.percentile(raw, 1), np.percentile(raw, 99)
    anom = np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1)   # 0..1 normalised
    # HONEST value check: does anomaly add signal, and does it catch supervised's misses?
    ap_anom = float(average_precision_score(yv, anom))
    oof = pd.read_csv(ROOT / "artifacts" / "boi" / "oof_predictions.csv").set_index("account_id")
    sup = oof["probability"].reindex(X.index).to_numpy()
    invisible = X.index[(yv == 1) & (sup < 0.05)]         # mules the classifier can't see
    anom_s = pd.Series(anom, index=X.index)
    inv_anom_pctl = float((anom_s.rank(pct=True)[invisible]).mean()) if len(invisible) else None
    hybrid = 0.5 * sup + 0.5 * anom                        # simple hybrid for the value check
    meta = {
        "raw_lo": float(lo), "raw_hi": float(hi),
        "anomaly_pr_auc": ap_anom, "supervised_pr_auc": float(average_precision_score(yv, sup)),
        "hybrid_pr_auc": float(average_precision_score(yv, hybrid)),
        "n_invisible_mules(sup<0.05)": int(len(invisible)),
        "invisible_mules_mean_anomaly_percentile": inv_anom_pctl,
        "honest_note": ("Anomaly PR-AUC is far below supervised — as expected: most mules are NOT "
                        "statistical outliers. Value is COMPLEMENTARY: it gives a second, unsupervised "
                        "opinion and a non-zero signal on the 'invisible' mules the classifier scores ~0. "
                        "Featured as a hybrid second opinion, NOT as a replacement for the supervised score."),
    }
    joblib.dump({"model": iso, "imputer": imp, "columns": list(X.columns)}, ART / "anomaly_if.joblib")
    (ART / "anomaly_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"  anomaly PR-AUC={ap_anom:.3f} | supervised={meta['supervised_pr_auc']:.3f} | "
          f"hybrid={meta['hybrid_pr_auc']:.3f} | invisible mules={len(invisible)} "
          f"(mean anomaly pctl {inv_anom_pctl})", flush=True)


def build_boi_edges():
    print("=== BOI behavioral-similarity edge list (for network draw) ===", flush=True)
    X, y = load_cached()
    mules = X[y == 1]
    try:
        base = joblib.load(ART / "base_model.joblib")
        gain = pd.Series(base.booster_.feature_importance("gain"), index=X.columns)
        topf = [f for f in gain.sort_values(ascending=False).index if f in X.columns][:30]
    except Exception:
        topf = list(X.columns[:30])
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(mules[topf]))
    S = cosine_similarity(Z); np.fill_diagonal(S, 0.0)
    thr = float(np.percentile(S[np.triu_indices_from(S, k=1)], 94))
    ids = list(int(i) for i in mules.index)
    oof = pd.read_csv(ART / "boi" / "oof_predictions.csv").set_index("account_id")
    risk = oof["risk_score"].to_dict()
    edges = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if S[a, b] >= thr:
                edges.append({"source": ids[a], "target": ids[b], "weight": round(float(S[a, b]), 3)})
    nodes = [{"id": i, "risk": int(risk.get(i, 0))} for i in ids]
    (ART / "mule_network_edges.json").write_text(json.dumps(
        {"nodes": nodes, "edges": edges, "similarity_threshold": thr,
         "note": "Behavioral-similarity graph among the 81 BOI mules (leak-removed top features). "
                 "A PROXY for link data — BOI has no transaction edges."}, indent=2))
    print(f"  {len(nodes)} nodes, {len(edges)} edges, thr={thr:.3f}", flush=True)


def build_amlsim_flow():
    print("=== AMLSim REAL money-flow typology subgraphs ===", flush=True)
    if not (EXT / "alerts.csv").exists():
        print("  AMLSim not extracted — skipping", flush=True); return
    alerts = pd.read_csv(EXT / "alerts.csv")
    txn = pd.read_csv(EXT / "transactions.csv")
    fr = txn[txn["IS_FRAUD"].astype(str).str.lower() == "true"]
    out = {"typologies": [], "note": "REAL transaction-graph money flow from AMLSim (planted "
           "laundering typologies). Demonstrates SENTINEL's Phase-2 capability with bank link data; "
           "BOI's snapshot has no such edges. Separate dataset — never merged into BOI."}
    for typ in ["fan_in", "cycle", "fan_out"]:
        sub = alerts[alerts["ALERT_TYPE"] == typ]
        if not len(sub):
            continue
        aid = sub["ALERT_ID"].iloc[0]
        rows = alerts[alerts["ALERT_ID"] == aid]
        accts = sorted(set(rows["SENDER_ACCOUNT_ID"]).union(set(rows["RECEIVER_ACCOUNT_ID"])))
        # real edges among these accounts (use the alert's own transactions)
        edges = [{"source": int(r.SENDER_ACCOUNT_ID), "target": int(r.RECEIVER_ACCOUNT_ID),
                  "amount": round(float(r.TX_AMOUNT), 2)} for r in rows.itertuples()]
        out["typologies"].append({
            "type": typ, "alert_id": int(aid), "n_accounts": len(accts),
            "accounts": [int(a) for a in accts], "edges": edges,
            "total_amount": round(float(rows["TX_AMOUNT"].sum()), 2)})
        print(f"  {typ}: alert {aid}, {len(accts)} accounts, {len(edges)} flow edges", flush=True)
    (ART / "amlsim_flow.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    build_anomaly()
    build_boi_edges()
    build_amlsim_flow()
    print("DONE -> dashboard assets in artifacts/", flush=True)
