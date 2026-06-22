"""
PHASE 3 — DATA INTEGRITY AUDITOR CREDIBILITY (it's a reusable tool, not tuned to F3912).

3a IEEE-CIS: run the 4-signature auditor over the transaction(+identity) features and
   show what it flags as leakage-prone / near-label, with the evidence (univariate AUC,
   bucket purity).
3b ULB credit-card POSITIVE CONTROL: inject ONE synthetic near-label column (noisy copy
   of the target) into creditcard.csv, run the auditor, and confirm it catches the
   injected leak while leaving the genuine PCA features V1..V28 untouched.

INTEGRITY: separate datasets, never merged with BOI. Metrics tagged per dataset.
Output: artifacts/auditor/metrics.json
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np, pandas as pd
from ext_lib import leakage_safe_prep, integrity_audit

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "dataset_Sentinel" / "extracted"
OUT = ROOT / "artifacts" / "auditor"; OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
RES = {"role": "AUDITOR CREDIBILITY — reusable rigour tool, not hand-tuned to BOI", "runs": {}}


def run_ieee():
    print("=== 3a — IEEE-CIS auditor run ===", flush=True)
    txn = EXT / "ieee" / "train_transaction.csv"
    if not txn.exists():
        RES["runs"]["ieee_cis"] = {"status": "SKIPPED — train_transaction.csv not extracted"}
        print("  SKIPPED (not extracted)", flush=True); return
    df = pd.read_csv(txn)  # ~590k rows; identity is optional and sparse
    # downsample columns? keep all; auditor is univariate. Sample rows for speed if huge.
    if len(df) > 200000:
        df = df.sample(200000, random_state=SEED)
    X, y, cats = leakage_safe_prep(df, target="isFraud", drop_cols=["TransactionID"])
    audit, base = integrity_audit(X, y)
    flagged = audit[audit.severity.isin(["CRITICAL", "HIGH"])] if len(audit) else pd.DataFrame()
    RES["runs"]["ieee_cis"] = {"dataset": "IEEE-CIS", "n_rows_sampled": int(len(df)), "prevalence": float(base),
                               "n_features": int(X.shape[1]), "n_critical_high": int(len(flagged)),
                               "top_flags": (flagged.head(15).to_dict("records") if len(flagged) else []),
                               "proves": "auditor surfaces high-AUC / near-label fields on a real fraud dataset."}
    print(f"  IEEE: {len(flagged)} CRITICAL/HIGH flagged of {X.shape[1]} features", flush=True)


def run_ulb_control():
    print("=== 3b — ULB injected-leak positive control ===", flush=True)
    cc = EXT / "ulb" / "creditcard.csv"
    df = pd.read_csv(cc)
    y = df["Class"].astype(int)
    rng = np.random.RandomState(SEED)
    # inject ONE near-label leak: noisy copy of target (flip ~3% of labels)
    leak = y.to_numpy().astype(float).copy()
    flip = rng.rand(len(leak)) < 0.03
    leak[flip] = 1 - leak[flip]
    df_inj = df.copy(); df_inj["LEAK_injected"] = leak
    X, yy, _ = leakage_safe_prep(df_inj, target="Class")
    audit, base = integrity_audit(X, yy)
    flagged = audit[audit.severity.isin(["CRITICAL", "HIGH"])] if len(audit) else pd.DataFrame()
    caught = "LEAK_injected" in set(flagged["feature"]) if len(flagged) else False
    pca_flagged = [f for f in (flagged["feature"].tolist() if len(flagged) else []) if f.startswith("V")]
    RES["runs"]["ulb_injected_control"] = {
        "dataset": "ULB credit-card (injected control)", "n_rows": int(len(df)), "prevalence": float(base),
        "injected_leak_caught": bool(caught),
        "injected_leak_severity": (flagged[flagged.feature == "LEAK_injected"].iloc[0]["severity"]
                                   if caught else None),
        "genuine_PCA_features_flagged": pca_flagged,
        "n_critical_high": int(len(flagged)),
        "proves": ("auditor CATCHES the planted near-label leak and does NOT false-flag the genuine "
                   "PCA features V1..V28 — a clean positive control."),
        "PASS": bool(caught and len(pca_flagged) == 0)}
    print(f"  ULB control: injected leak caught={caught} | PCA false-flags={len(pca_flagged)} | "
          f"PASS={RES['runs']['ulb_injected_control']['PASS']}", flush=True)


def main():
    t0 = time.time()
    run_ulb_control()   # fast, always available
    run_ieee()          # needs extraction; skips cleanly if absent
    RES["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "metrics.json").write_text(json.dumps(RES, indent=2, default=float))
    print(f"PHASE 3 DONE in {RES['runtime_s']}s -> {OUT/'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
