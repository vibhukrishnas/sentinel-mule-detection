"""
PHASE 2 — EXTERNAL GENERALISATION on BAF (Bank Account Fraud, NeurIPS 2022 'Base').

Runs the SAME methodology as BOI (target-free hygiene -> Data Integrity Auditor ->
leakage-safe CV -> calibration -> recall operating point) on a DIFFERENT, realistic,
leak-free-by-design dataset, using BAF's intended TEMPORAL split (train months 0-5,
test months 6-7). This is evidence the METHOD generalises — NOT the BOI number.

INTEGRITY: BAF rows are never merged with BOI. Every metric tagged dataset="BAF".
Output: artifacts/external/baf/metrics.json
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from ext_lib import leakage_safe_prep, integrity_audit, cv_pr_auc, recall_operating_point

ROOT = Path(__file__).resolve().parent.parent
BAF = ROOT / "dataset_Sentinel" / "extracted" / "baf" / "Base.csv"
OUT = ROOT / "artifacts" / "external" / "baf"; OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def mk_lgbm(spw):
    return lgb.LGBMClassifier(n_estimators=500, learning_rate=0.04, num_leaves=48, subsample=0.85,
        colsample_bytree=0.7, reg_lambda=4.0, min_child_samples=50, scale_pos_weight=spw,
        n_jobs=-1, random_state=SEED, verbose=-1)


def main():
    t0 = time.time()
    print("=== PHASE 2 — BAF external generalisation ===", flush=True)
    df = pd.read_csv(BAF)
    print(f"BAF rows={len(df)} cols={df.shape[1]} fraud_rate={df['fraud_bool'].mean()*100:.2f}%", flush=True)
    X, y, cats = leakage_safe_prep(df, target="fraud_bool")
    months = df["month"].to_numpy()

    # (1) auditor on BAF (should find NO critical leaks — it's leak-free by design)
    audit, base = integrity_audit(X, y)
    crit = audit[audit.severity.isin(["CRITICAL", "HIGH"])] if len(audit) else pd.DataFrame()
    print(f"auditor: {len(crit)} CRITICAL/HIGH flags on BAF (expected ~0; leak-free by design)", flush=True)
    if len(crit):
        print(crit.head(8).to_string(index=False), flush=True)

    # (2) leakage-safe 5-fold CV PR-AUC on the full data (central estimate)
    cv = cv_pr_auc(mk_lgbm, X, y, n_splits=5, seed=SEED, calibrate=False)
    print(f"5-fold CV: PR-AUC={cv['pr_auc']:.3f} ROC={cv['roc_auc']:.3f} Brier={cv['brier']:.4f}", flush=True)

    # (3) BAF intended TEMPORAL split: train months 0-5, test 6-7 (used exactly once)
    tr = months <= 5; te = months >= 6
    spw = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
    cal = CalibratedClassifierCV(mk_lgbm(spw), method="sigmoid", cv=3).fit(X[tr], y[tr])
    pte = cal.predict_proba(X[te])[:, 1]; yte = y[te].to_numpy()
    hold = {"split": "temporal (train months<=5, test months>=6)", "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "test_fraud_rate": float(yte.mean()),
            "pr_auc": float(average_precision_score(yte, pte)), "roc_auc": float(roc_auc_score(yte, pte)),
            "brier": float(brier_score_loss(yte, pte))}
    op, _ = recall_operating_point(pte, yte, 0.97)
    print(f"temporal holdout: PR-AUC={hold['pr_auc']:.3f} ROC={hold['roc_auc']:.3f} | recall>=0.97 -> {op}", flush=True)

    metrics = {"dataset": "BAF (external)", "role": "EXTERNAL GENERALISATION — NOT the BOI number",
               "source_file": "dataset_Sentinel/extracted/baf/Base.csv", "n_rows": int(len(df)),
               "prevalence": float(base), "n_features": int(X.shape[1]),
               "auditor_critical_high": int(len(crit)),
               "auditor_top_flags": (crit.head(8).to_dict("records") if len(crit) else []),
               "cv_5fold": {k: cv[k] for k in ("pr_auc", "roc_auc", "brier", "prevalence")},
               "temporal_holdout": hold, "recall_operating_point": op,
               "runtime_s": round(time.time() - t0, 1),
               "note": "Same pipeline as BOI; different dataset. PR-AUC primary. Never merged with BOI."}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nPHASE 2 DONE in {metrics['runtime_s']}s -> {OUT/'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
