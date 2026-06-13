"""Interrogate F3912 (AUC 0.987, full coverage) + the tail feature block for outcome
leakage. Uses the light cached parquet (the raw CSV OOMs this box)."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from preprocess import load_cached

X, y = load_cached()

print("=== F3912 forensics ===")
s = X["F3912"]
print("dtype:", s.dtype, "| NA%:", round(s.isna().mean(), 4), "| nunique:", s.nunique())
print("value_counts (top 12):")
print(s.value_counts(dropna=False).head(12))
print("\nCrosstab F3912 (rounded) vs target:")
ct = pd.crosstab(s.round(3), y, dropna=False)
print(ct.head(25))
print("\nBy class describe:")
print(pd.concat([s[y==0].describe().rename('legit'),
                 s[y==1].describe().rename('mule')], axis=1))

# Does a single threshold near-perfectly classify? find best split
order = s.dropna().sort_values().unique()
print(f"\nF3912 range: [{order.min():.3f}, {order.max():.3f}], {len(order)} unique values")
print("class-1 (mule) F3912 values (all 81):")
print(sorted(s[y==1].dropna().round(3).tolist()))

print("\n=== Tail block F3900..F3923 leakage scan (cached) ===")
for i in range(3900, 3924):
    col = f"F{i}"
    if col not in X.columns:
        print(f"{col}: (dropped as constant/near-empty in preprocessing)")
        continue
    x = X[col]; m = x.notna()
    if m.sum() < 30 or y[m].nunique() < 2:
        print(f"{col}: NA%={x.isna().mean():.3f} nunique={x.nunique()} (skip)")
        continue
    auc = roc_auc_score(y[m], x[m].astype(float)); auc = max(auc, 1-auc)
    print(f"{col}: NA%={x.isna().mean():.3f} nunique={x.nunique()} "
          f"covers_pos={int(y[m].sum())}/81 AUC={auc:.3f}")
