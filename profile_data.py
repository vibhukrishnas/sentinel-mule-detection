"""Quick, brutal profiling of DataSet.csv for PS2 (mule account classification)."""
import pandas as pd
import numpy as np

print("Loading...", flush=True)
df = pd.read_csv("DataSet.csv", index_col=0, low_memory=False)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns[0]} ... {df.columns[-1]}")

TARGET = "F3924"
print("\n=== TARGET (F3924) ===")
print("dtype:", df[TARGET].dtype)
print(df[TARGET].value_counts(dropna=False))
vc = df[TARGET].value_counts(dropna=False, normalize=True)
print("\nProportions:")
print(vc)

print("\n=== OVERALL SPARSITY ===")
total_cells = df.shape[0] * df.shape[1]
na_cells = df.isna().sum().sum()
print(f"NA fraction overall: {na_cells/total_cells:.4f}")

# per-column NA fraction distribution
na_frac = df.isna().mean()
print("\nPer-column NA fraction quantiles:")
print(na_frac.quantile([0, .1, .25, .5, .75, .9, .99, 1.0]))
print(f"Columns >99% NA: {(na_frac > 0.99).sum()}")
print(f"Columns >90% NA: {(na_frac > 0.90).sum()}")
print(f"Columns >50% NA: {(na_frac > 0.50).sum()}")
print(f"Columns 0% NA (fully populated): {(na_frac == 0).sum()}")

print("\n=== COLUMN DTYPES ===")
print(df.dtypes.value_counts())

# constant / near-constant
nunique = df.nunique(dropna=True)
print(f"\nConstant columns (nunique<=1): {(nunique <= 1).sum()}")
print(f"Near-constant (nunique==2): {(nunique == 2).sum()}")
print("nunique quantiles:")
print(nunique.quantile([0, .25, .5, .75, .9, .99, 1.0]))

print("\n=== HINT FEATURES (from Topic.pdf) ===")
hints = ["F115","F321","F527","F531","F670","F1692","F2082","F2122",
         "F2582","F2678","F2737","F2956","F3043","F3836","F3887","F3889","F3891","F3894"]
for h in hints:
    if h in df.columns:
        s = df[h]
        print(f"{h}: dtype={s.dtype}, NA%={s.isna().mean():.3f}, nunique={s.nunique()}, "
              f"sample={s.dropna().unique()[:5]}")
    else:
        print(f"{h}: MISSING from dataset")

# correlation of hint features with target if numeric
print("\n=== HINT FEATURE vs TARGET (numeric corr / mean-by-class) ===")
y = df[TARGET]
for h in hints:
    if h in df.columns and pd.api.types.is_numeric_dtype(df[h]):
        try:
            grp = df.groupby(TARGET)[h].mean()
            print(f"{h}: mean by class -> {dict(grp.round(3))}")
        except Exception as e:
            print(f"{h}: {e}")
