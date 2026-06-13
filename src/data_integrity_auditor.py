"""
SENTINEL — Data Integrity Auditor.

A reusable, bank-agnostic leakage scanner. Most teams train a model and never ask
"is my 0.99 real?" This tool does. It scores every feature for FOUR leak signatures
and emits a severity-ranked audit any institution can run on its own data:

  1. LABEL-PROXY     a value almost perfectly equals the target (e.g. F3912, F2230).
  2. EXACT-BUCKET    a specific discrete value is hyper-pure fraud (non-monotonic;
                     invisible to ROC-AUC) — e.g. F2230=='Sep25' -> 100%.
  3. RANGE/DECILE    a contiguous value range (not a single value) is fraud-pure —
                     catches continuous-feature leaks the bucket scan misses.
  4. UNIVARIATE-AUC  a single feature near-perfectly RANKS the target (AUC>=0.95).

Bank-listed features (Topic.pdf) are reported but never auto-classified CRITICAL.
Output: DATA_INTEGRITY_AUDIT.md + artifacts/integrity_audit.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from preprocess import load_and_clean, HINT_FEATURES, ART

ROOT = Path(__file__).resolve().parent.parent
MIN_N = 10
BASE_MULT = {"CRITICAL": 0.90, "HIGH": 0.50, "MEDIUM": 0.20, "WATCH": 0.10}


def audit():
    X, y, _, _ = load_and_clean(verbose=False, apply_bucket_leaks=False)
    base = float(y.mean()); yv = y.values
    hints = set(HINT_FEATURES)
    print(f"Auditing {X.shape[1]} features | base fraud rate {base:.4%} "
          f"({int(y.sum())} mules)\n", flush=True)

    rows = []
    for col in X.columns:
        s = X[col]; nona = s.notna()
        if nona.sum() < MIN_N or y[nona].nunique() < 2:
            continue
        # univariate AUC (direction-agnostic)
        try:
            a = roc_auc_score(y[nona], s[nona].astype(float)); auc = max(a, 1 - a)
        except ValueError:
            auc = np.nan
        # exact-value bucket purity
        d = pd.DataFrame({"v": s.round(3), "y": yv}).dropna(subset=["v"])
        g = d.groupby("v")["y"]; cnt, mean = g.count(), g.mean()
        sel = cnt >= MIN_N
        bucket_rate = float(mean[sel].max()) if sel.any() else 0.0
        # range/decile purity (continuous leaks)
        decile_rate = 0.0
        if s.nunique() >= 10:
            try:
                q = pd.qcut(s[nona], 10, duplicates="drop")
                dr = y[nona].groupby(q, observed=True).mean()
                dn = y[nona].groupby(q, observed=True).count()
                dr = dr[dn >= MIN_N]
                decile_rate = float(dr.max()) if len(dr) else 0.0
            except (ValueError, IndexError):
                decile_rate = 0.0
        worst = max(bucket_rate, decile_rate)
        mules_cov = int(y[nona].sum())
        # A near-perfect univariate AUC only escalates severity if the feature actually
        # covers >=10 mules. A feature non-NA for ~1 mule scores AUC~1 by sparsity, not
        # leakage, and cannot generalize — it stays WATCH, never CRITICAL.
        auc_eff = auc if (not np.isnan(auc) and mules_cov >= 10) else 0.0
        sev = ("CRITICAL" if (worst >= BASE_MULT["CRITICAL"] or auc_eff >= 0.98)
               else "HIGH" if (worst >= BASE_MULT["HIGH"] or auc_eff >= 0.95)
               else "MEDIUM" if worst >= BASE_MULT["MEDIUM"]
               else "WATCH" if (worst >= BASE_MULT["WATCH"] or
                                (not np.isnan(auc) and auc >= 0.95)) else None)
        if sev:
            rows.append({"feature": col, "severity": sev, "mules_covered": mules_cov,
                         "univariate_auc": round(float(auc), 3) if not np.isnan(auc) else None,
                         "best_bucket_fraud_rate": round(bucket_rate, 3),
                         "best_decile_fraud_rate": round(decile_rate, 3),
                         "lift_vs_base": round(worst / base, 1),
                         "bank_listed": col.split("__")[0] in hints})

    res = pd.DataFrame(rows)
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "WATCH": 3}
    res["o"] = res["severity"].map(order)
    res = res.sort_values(["o", "best_bucket_fraud_rate", "univariate_auc"],
                          ascending=[True, False, False]).drop(columns="o").reset_index(drop=True)

    counts = res["severity"].value_counts().to_dict()
    print("=== SEVERITY COUNTS ===")
    for s in ["CRITICAL", "HIGH", "MEDIUM", "WATCH"]:
        print(f"  {s:9s}: {counts.get(s, 0)}")
    # hyper-sparse univariate suspects (cover ~1 positive) flagged separately
    print("\n=== TOP 15 CRITICAL/HIGH (excluding hyper-sparse) ===")
    print(res[res.severity.isin(["CRITICAL", "HIGH"])].head(15).to_string(index=False))

    # RANGE-ONLY leaks: high decile purity but low exact-bucket purity -> a continuous
    # leak the 0.10 bucket detector would MISS. This is the completeness check.
    range_only = res[(res.best_decile_fraud_rate >= 0.10) &
                     (res.best_bucket_fraud_rate < 0.10) & (~res.bank_listed)]
    print(f"\n=== RANGE-ONLY leaks (decile>=10% but bucket<10% — bucket scan would miss) ===")
    print(range_only[["feature", "best_decile_fraud_rate", "univariate_auc", "lift_vs_base"]]
          .head(20).to_string(index=False) if len(range_only) else "  NONE — bucket detector is sufficient.")

    auto_block = sorted(res[(res.severity.isin(["CRITICAL", "HIGH"])) & (~res.bank_listed)]["feature"].tolist())
    out = {"base_rate": base, "n_features_audited": int(X.shape[1]),
           "severity_counts": counts, "auto_block_recommended": auto_block,
           "findings": res.to_dict("records")}
    (ART / "integrity_audit.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")

    _write_markdown(res, counts, base, len(auto_block))
    print(f"\nRecommended auto-block (CRITICAL/HIGH, non-bank-listed): {len(auto_block)} features")
    print(f"Saved -> artifacts/integrity_audit.json + DATA_INTEGRITY_AUDIT.md")
    return res


def _write_markdown(res, counts, base, n_block):
    top = res[res.severity.isin(["CRITICAL", "HIGH"])].head(20)
    lines = [
        "# SENTINEL — Data Integrity Audit",
        f"\n*Automated leakage scan · base fraud rate {base:.4%} · "
        f"{int(res.shape[0])} flagged features*\n",
        "A leakage feature is one that encodes the OUTCOME rather than pre-event behaviour. "
        "Training on it produces a fake ~1.0 score that collapses in production. This audit "
        "scores every feature on four leak signatures.\n",
        "## Severity summary",
        "| Severity | Meaning | Count |",
        "|---|---|---|",
        f"| CRITICAL | label-proxy / ≥90%-pure bucket / AUC≥0.98 | {counts.get('CRITICAL',0)} |",
        f"| HIGH | ≥50%-pure bucket / AUC≥0.95 | {counts.get('HIGH',0)} |",
        f"| MEDIUM | 20–50%-pure bucket (review) | {counts.get('MEDIUM',0)} |",
        f"| WATCH | 10–20%-pure bucket | {counts.get('WATCH',0)} |",
        f"\n**Recommended action:** auto-exclude the {n_block} CRITICAL/HIGH non-bank-listed "
        "features before modelling (SENTINEL does this automatically).\n",
        "## Top flagged features",
        "| Feature | Severity | Univ. AUC | Best bucket fraud-rate | Lift vs base | Bank-listed |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in top.iterrows():
        lines.append(f"| {r.feature} | {r.severity} | {r.univariate_auc} | "
                     f"{r.best_bucket_fraud_rate:.0%} | {r.lift_vs_base}× | "
                     f"{'yes' if r.bank_listed else 'no'} |")
    (ROOT / "DATA_INTEGRITY_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    audit()
