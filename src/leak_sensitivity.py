"""
Leakage sensitivity sweep — the honest-performance curve.

Load the data ONCE (no bucket-leak removal), then progressively strip features whose
purest value-bucket exceeds a falling fraud-rate threshold, and measure LightGBM CV
PR-AUC at each level. Where the curve FLATTENS is the genuine behavioral signal;
the steep early drop is leakage being removed.
"""
from __future__ import annotations
import json, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_validate
from preprocess import load_and_clean, detect_bucket_leaks, ART
warnings.filterwarnings("ignore")

X_full, y, _, _ = load_and_clean(verbose=False, apply_bucket_leaks=False)
pw = float((y == 0).sum() / (y == 1).sum())
cv = StratifiedKFold(3, shuffle=True, random_state=42)   # 3-fold for speed in the sweep
print(f"Full matrix (no bucket-leak removal): {X_full.shape}, pos={int(y.sum())}\n")

def cv_pr(X):
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31,
        subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0, min_child_samples=20,
        scale_pos_weight=pw, n_jobs=-1, random_state=42, verbose=-1)
    r = cross_validate(m, X, y, cv=cv, scoring={"pr": "average_precision",
                       "roc": "roc_auc"}, n_jobs=1)
    return r["test_pr"].mean(), r["test_pr"].std(), r["test_roc"].mean()

rows = []
print(f"{'leak_thr':>9}{'n_leaks':>9}{'n_feats':>9}{'PR-AUC':>9}{'±std':>7}{'ROC':>7}")
for thr in [1.01, 0.30, 0.20, 0.10, 0.05, 0.03, 0.02]:
    leaks = [] if thr > 1 else detect_bucket_leaks(X_full, y, rate=thr)
    Xs = X_full.drop(columns=leaks)
    pr, sd, roc = cv_pr(Xs)
    rows.append({"leak_thr": thr, "n_leaks": len(leaks), "n_feats": Xs.shape[1],
                 "pr_auc": pr, "pr_std": sd, "roc_auc": roc})
    label = "none" if thr > 1 else f"{thr:.2f}"
    print(f"{label:>9}{len(leaks):>9}{Xs.shape[1]:>9}{pr:>9.3f}{sd:>7.3f}{roc:>7.3f}", flush=True)

(ART / "leak_sensitivity.json").write_text(json.dumps(rows, indent=2, default=float))
print(f"\nbaseline PR-AUC (prevalence) = {y.mean():.4f}")
print("Where PR-AUC flattens across thresholds = genuine signal; steep drop = leakage.")
