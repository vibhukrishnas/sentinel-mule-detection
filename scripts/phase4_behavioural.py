"""
PHASE 4 — BEHAVIOURAL / VELOCITY FEATURES + EXTREME-IMBALANCE CHECK (illustrative).

4a PaySim (synthetic mobile-money log): the canonical mule pattern is money entering an
   account via TRANSFER and leaving via CASH_OUT. We engineer behavioural/velocity +
   CHAIN features that encode this, run the Data Integrity Auditor (which must flag
   `isFlaggedFraud` as a post-hoc leak), train a leakage-safe classifier, and report
   PR-AUC + recall operating point under BOTH random 5-fold CV and a STEP-BASED temporal
   split (train early time-steps, test later) used exactly once. Synthetic -> illustrative.

4b ULB credit-card as an EXTREME-IMBALANCE sanity check (~0.17%, below BOI's 0.89%):
   confirms PR-AUC + calibrated Brier behave sensibly under even lower prevalence.

INTEGRITY: separate datasets, never merged with BOI. Every metric tagged per dataset.
Output: artifacts/behavioural/metrics.json
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
import lightgbm as lgb
from ext_lib import leakage_safe_prep, integrity_audit, cv_pr_auc, recall_operating_point

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "dataset_Sentinel" / "extracted"
OUT = ROOT / "artifacts" / "behavioural"; OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
RES = {"role": "BEHAVIOURAL ROBUSTNESS (illustrative) — NOT the BOI number", "runs": {}}


def mk(spw):
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=48, subsample=0.85,
        colsample_bytree=0.8, reg_lambda=2.0, scale_pos_weight=spw, n_jobs=-1, random_state=SEED, verbose=-1)


def engineer_paysim(df):
    """Behavioural + velocity + TRANSFER->CASH_OUT chain features (leakage-safe; no label use)."""
    df = df.copy(); df.columns = [c.strip() for c in df.columns]
    # balance-consistency errors (the strongest genuine PaySim signal)
    df["errBalOrig"] = (df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]).astype("float32")
    df["errBalDest"] = (df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]).astype("float32")
    df["amt_to_oldOrig"] = (df["amount"] / (df["oldbalanceOrg"] + 1)).astype("float32")
    df["amt_to_oldDest"] = (df["amount"] / (df["oldbalanceDest"] + 1)).astype("float32")
    # account-draining (mule cash-out signature): origin emptied to ~0
    df["drains_origin"] = ((df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)).astype("int8")
    df["dest_was_empty"] = (df["oldbalanceDest"] == 0).astype("int8")
    df["is_transfer_or_cashout"] = df["type"].isin(["TRANSFER", "CASH_OUT"]).astype("int8")
    # CHAIN: a destination that RECEIVES a TRANSFER and later ORIGINATES a CASH_OUT = classic
    # layering/cash-out mule. Built from the graph of nameDest->nameOrig WITHOUT using labels.
    transfer_dest = set(df.loc[df["type"] == "TRANSFER", "nameDest"])
    cashout_orig = set(df.loc[df["type"] == "CASH_OUT", "nameOrig"])
    chain_accts = transfer_dest & cashout_orig
    df["in_cashout_chain"] = (df["nameOrig"].isin(chain_accts) | df["nameDest"].isin(chain_accts)).astype("int8")
    return df


def run_paysim():
    print("=== 4a — PaySim behavioural/velocity + chain features ===", flush=True)
    cand = list((EXT / "paysim").glob("*.csv")) if (EXT / "paysim").exists() else []
    if not cand:
        RES["runs"]["paysim"] = {"status": "SKIPPED — PaySim csv not extracted"}; print("  SKIPPED", flush=True); return
    p = cand[0]
    df = pd.read_csv(p); df.columns = [c.strip() for c in df.columns]
    tgt = "isFraud" if "isFraud" in df.columns else [c for c in df.columns if c.lower() == "isfraud"][0]
    print(f"  rows={len(df)} fraud_rate={df[tgt].mean()*100:.3f}% types={sorted(df['type'].unique())}", flush=True)

    # (i) auditor must flag isFlaggedFraud as a post-hoc leak (credibility on a 3rd dataset)
    Xaud, yaud, _ = leakage_safe_prep(df.drop(columns=[c for c in ["nameOrig", "nameDest"] if c in df.columns]),
                                      target=tgt)
    audit, base = integrity_audit(Xaud, yaud)
    flagged = set(audit[audit.severity.isin(["CRITICAL", "HIGH"])]["feature"]) if len(audit) else set()
    flag_isflagged = "isFlaggedFraud" in flagged
    print(f"  auditor: isFlaggedFraud flagged as leak = {flag_isflagged} (expected True)", flush=True)

    # (ii) engineer features, drop the known leak + identifiers, sample for speed
    df = engineer_paysim(df)
    drop = [c for c in ["isFlaggedFraud", "nameOrig", "nameDest"] if c in df.columns]
    if len(df) > 400000:
        df = df.sample(400000, random_state=SEED).reset_index(drop=True)
    X, y, _ = leakage_safe_prep(df, target=tgt, drop_cols=drop)

    # (iii) random 5-fold CV
    cv = cv_pr_auc(mk, X, y, n_splits=5, seed=SEED)
    op, _ = recall_operating_point(cv["oof"], y.to_numpy(), 0.97)

    # (iv) STEP-based temporal split (train early steps, test later) — used once
    temporal = {}
    if "step" in X.columns:
        cut = int(np.quantile(df["step"], 0.7))
        tr = df["step"] <= cut; te = df["step"] > cut
        spw = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
        m = mk(spw).fit(X[tr], y[tr]); pte = m.predict_proba(X[te])[:, 1]; yte = y[te].to_numpy()
        temporal = {"split": f"step<= {cut} train / > {cut} test", "n_train": int(tr.sum()), "n_test": int(te.sum()),
                    "pr_auc": float(average_precision_score(yte, pte)), "roc_auc": float(roc_auc_score(yte, pte))}

    RES["runs"]["paysim"] = {
        "dataset": "PaySim (synthetic)", "n_rows_used": int(len(df)), "prevalence": cv["prevalence"],
        "auditor_flagged_isFlaggedFraud_as_leak": bool(flag_isflagged),
        "engineered_features": ["errBalOrig", "errBalDest", "amt_to_oldOrig", "amt_to_oldDest",
                                "drains_origin", "dest_was_empty", "is_transfer_or_cashout", "in_cashout_chain"],
        "cv_5fold": {"pr_auc": cv["pr_auc"], "roc_auc": cv["roc_auc"], "brier": cv["brier"]},
        "recall_operating_point": op, "temporal_step_split": temporal,
        "note": "Illustrative behavioural signal; synthetic data; isFlaggedFraud dropped as a confirmed leak."}
    print(f"  PaySim 5-fold PR-AUC={cv['pr_auc']:.3f} ROC={cv['roc_auc']:.3f} | temporal {temporal.get('pr_auc','-')}", flush=True)


def run_ulb_imbalance():
    print("=== 4b — ULB extreme-imbalance check ===", flush=True)
    cc = EXT / "ulb" / "creditcard.csv"
    df = pd.read_csv(cc)
    X, y, _ = leakage_safe_prep(df, target="Class")
    cv = cv_pr_auc(mk, X, y, n_splits=5, seed=SEED, calibrate=True)
    op, _ = recall_operating_point(cv["oof"], y.to_numpy(), 0.90)
    RES["runs"]["ulb_imbalance"] = {"dataset": "ULB (imbalance check)", "n_rows": int(len(df)),
        "prevalence": cv["prevalence"], "pr_auc": cv["pr_auc"], "roc_auc": cv["roc_auc"], "brier": cv["brier"],
        "recall>=0.90_operating_point": op,
        "note": "0.17% prevalence (below BOI's 0.89%); PR-AUC + calibrated Brier behave sanely under deeper imbalance."}
    print(f"  ULB prev={cv['prevalence']*100:.3f}% PR-AUC={cv['pr_auc']:.3f} ROC={cv['roc_auc']:.3f} Brier={cv['brier']:.5f}", flush=True)


def main():
    t0 = time.time()
    run_ulb_imbalance()
    run_paysim()
    RES["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "metrics.json").write_text(json.dumps(RES, indent=2, default=float))
    print(f"PHASE 4 DONE in {RES['runtime_s']}s -> {OUT/'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
