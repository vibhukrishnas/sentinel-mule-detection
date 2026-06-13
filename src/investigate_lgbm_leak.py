"""
Why is LightGBM at 0.997 PR-AUC when RandomForest is 0.809 on the SAME data?
That gap usually means a second leak that trees exploit but linear/forest models
don't — often NON-MONOTONIC (a value that == fraud), which univariate ROC-AUC misses.

This fits LightGBM once, ranks features by gain, and for each top feature checks for
leakage the way we caught F3912: coverage, value distribution by class, and the
PUREST single value/bucket (does some value almost always mean 'mule'?).
"""
from __future__ import annotations
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from preprocess import load_cached

X, y = load_cached()
pw = float((y == 0).sum() / (y == 1).sum())
m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
                       subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
                       min_child_samples=20, scale_pos_weight=pw, n_jobs=-1,
                       random_state=42, verbose=-1)
m.fit(X, y)
imp = pd.Series(m.booster_.feature_importance(importance_type="gain"),
                index=X.columns).sort_values(ascending=False)

print("=== Top 20 LightGBM features by gain — leakage interrogation ===\n")
print(f"{'feature':16s}{'gain%':>7s}{'NA%':>7s}{'nuniq':>7s}{'AUC':>6s}"
      f"{'meanLegit':>11s}{'meanMule':>10s}{'purestVal(fraud-rate, n)':>30s}")
total_gain = imp.sum()
for f in imp.head(20).index:
    s = X[f]
    nona = s.notna()
    auc = np.nan
    if nona.sum() > 30 and y[nona].nunique() == 2:
        a = roc_auc_score(y[nona], s[nona].astype(float)); auc = max(a, 1 - a)
    ml = s[y == 0].mean(); mm = s[y == 1].mean()
    # purest value/bucket: among values present in >=10 rows, which has highest fraud rate?
    purest = ""
    vc = s.round(3).value_counts()
    cand = vc[vc >= 10].index
    best_rate, best_v, best_n = 0, None, 0
    for v in cand:
        mask = s.round(3) == v
        rate = y[mask].mean()
        if rate > best_rate:
            best_rate, best_v, best_n = rate, v, int(mask.sum())
    if best_v is not None:
        purest = f"{best_v}: {best_rate:.0%} fraud (n={best_n})"
    print(f"{f:16s}{100*imp[f]/total_gain:>6.1f}%{s.isna().mean():>7.2f}"
          f"{s.nunique():>7d}{auc:>6.2f}{ml:>11.3f}{mm:>10.3f}{purest:>30s}")

print("\nIf a top feature shows a value with ~100% fraud rate at meaningful n, or "
      "covers all positives with AUC~1, it is leakage and must be blocklisted.")
