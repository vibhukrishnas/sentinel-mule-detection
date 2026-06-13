"""
SENTINEL — Leakage / too-good-feature audit.

A 0.999 ROC-AUC on 81 anonymized positives is a claim that must be EARNED, not
trusted. This script hunts the two ways such a number is usually a lie:

  1) A single feature that near-perfectly separates the classes (univariate ROC-AUC
     ~1.0) -> likely a post-event / leaked field (e.g. "account_frozen_flag").
  2) A missingness pattern that is itself the label (an __ismissing flag with
     near-perfect AUC -> a field that only exists for one class).

We report the worst offenders so a human can decide: real signal or leakage.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from preprocess import load_cached, ART, HINT_FEATURES


def univariate_auc(x: pd.Series, y: pd.Series):
    m = x.notna()
    if m.sum() < 30 or y[m].nunique() < 2:
        return np.nan, np.nan
    xv, yv = x[m].values.astype(float), y[m].values
    try:
        auc = roc_auc_score(yv, xv)
    except ValueError:
        return np.nan, np.nan
    # direction-agnostic separation strength, plus coverage of positives
    pos_cov = float(y[m].sum()) / float(y.sum())
    return max(auc, 1 - auc), pos_cov


def main():
    X, y = load_cached()
    print(f"Auditing {X.shape[1]} features against target ({int(y.sum())} positives)\n")

    recs = []
    for c in X.columns:
        auc, cov = univariate_auc(X[c], y)
        if not np.isnan(auc):
            recs.append((c, auc, cov))
    df = pd.DataFrame(recs, columns=["feature", "auc", "pos_coverage"]).sort_values(
        "auc", ascending=False).reset_index(drop=True)

    print("=== TOP 25 most-separating single features (suspect if AUC ~ 1.0) ===")
    print(df.head(25).to_string(index=False))

    suspect = df[df["auc"] >= 0.98]
    near = df[(df["auc"] >= 0.90) & (df["auc"] < 0.98)]
    print(f"\nFeatures with univariate AUC >= 0.98 (LEAKAGE SUSPECTS): {len(suspect)}")
    print(f"Features with univariate AUC 0.90-0.98 (strong, watch):   {len(near)}")

    miss_suspect = suspect[suspect["feature"].str.endswith("__ismissing")]
    print(f"  ...of suspects, missingness-pattern flags: {len(miss_suspect)}")

    print("\n=== Bank-hinted features' univariate AUC (sanity) ===")
    hint_rows = df[df["feature"].isin(HINT_FEATURES)]
    print(hint_rows.to_string(index=False))

    out = {
        "n_suspect_ge_098": int(len(suspect)),
        "n_strong_090_098": int(len(near)),
        "suspects": suspect.head(40).to_dict(orient="records"),
        "top25": df.head(25).to_dict(orient="records"),
        "hint_feature_auc": hint_rows.to_dict(orient="records"),
    }
    (ART / "leakage_audit.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved -> {ART/'leakage_audit.json'}")


if __name__ == "__main__":
    main()
