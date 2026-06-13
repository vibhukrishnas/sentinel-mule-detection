"""
SENTINEL — Preprocessing & feature hygiene for PS2 mule-account detection.

Loads the raw 9082 x 3924 DataSet.csv ONCE, applies LEAKAGE-SAFE, UNSUPERVISED
hygiene (nothing here looks at the target except to split it off), and caches a
clean matrix for fast model iteration.

Decisions (all justified in the PRD):
  - Drop constant / single-value columns (359 dead cols add only variance).
  - Keep NaN as-is (HistGBM / LightGBM / XGBoost consume it natively; imputation
    for LogReg/RF is done INSIDE CV folds in train.py, never here).
  - Ordinal-encode object/categorical columns with a deterministic, target-free
    mapping (unsupervised -> no leakage). Unknown/NaN -> -1.
  - Add missingness indicators for high-NA columns: *whether* a profile field is
    populated is itself predictive of mule behaviour.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "DataSet.csv"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

TARGET = "F3924"
HINT_FEATURES = ["F115", "F321", "F527", "F531", "F670", "F1692", "F2082", "F2122",
                 "F2582", "F2678", "F2737", "F2956", "F3043", "F3836", "F3887",
                 "F3889", "F3891", "F3894"]
HIGH_NA_FLAG_THRESHOLD = 0.30   # add is_missing flag for cols with >30% NA
DROP_NA_THRESHOLD = 0.97        # drop cols that are essentially all-empty
MIN_NONNA = 30                  # drop hyper-sparse cols (pure memorization fuel)

# CONFIRMED TARGET LEAKAGE — excluded from modeling.
# F3912 is a binary flag aligned ~96% with the label (79/81 mules have it set,
# only 3/9001 legit do). It is a post-hoc fraud flag / re-encoded label, NOT a
# behavioral feature you would possess BEFORE knowing the account is fraudulent.
# Keeping it makes the model a cheater (it drove the bogus 0.999 ROC-AUC). If the
# bank confirms it is a legitimate pre-event signal, re-add it — but the model must
# stand on its own without reading the answer.
LEAKAGE_BLOCKLIST = ["F3912"]

# AUTOMATED leak detection: a behavioral feature does not have a value-bucket that is
# 10%+ fraud (>11x the 0.89% base rate). Such buckets are label leakage hiding in a
# tree-splittable value (e.g. F2230==3 -> 100% fraud), invisible to monotonic AUC.
# We auto-detect and exclude these, EXCEPT bank-hinted features (treated as genuine).
# Chosen via the leakage-sensitivity sweep (artifacts/leak_sensitivity.json): PR-AUC
# plateaus at ~0.86 from this threshold up — i.e. honest signal, not a single leak.
# 0.10 is the conservative pick (removes more suspects for a defensible ~0.86).
LEAK_BUCKET_RATE = 0.10
LEAK_BUCKET_MIN_N = 10


def detect_bucket_leaks(X, y, rate=LEAK_BUCKET_RATE, min_n=LEAK_BUCKET_MIN_N):
    """Return features that are too fraud-pure to be behavioral. Catches BOTH:
      - discrete value-buckets (e.g. F2230=='Sep25' -> 100% fraud), and
      - continuous value RANGES via deciles (e.g. F2877's top decile -> 12x base),
    which a pure value-bucket scan would miss. Bank-listed features are never flagged."""
    hints = set(HINT_FEATURES)
    leaks = []
    yv = y.values
    for col in X.columns:
        if col.split("__")[0] in hints:
            continue
        s = X[col]
        v = s.round(3)
        d = pd.DataFrame({"v": v, "y": yv}).dropna(subset=["v"])
        if d.empty:
            continue
        g = d.groupby("v")["y"]
        cnt, mean = g.count(), g.mean()
        sel = cnt >= min_n
        if sel.any() and mean[sel].max() >= rate:           # discrete bucket leak
            leaks.append(col)
            continue
        if s.nunique() >= 10:                                # continuous range leak
            nona = s.notna()
            try:
                q = pd.qcut(s[nona], 10, duplicates="drop")
                ym = y[nona]
                dm = ym.groupby(q, observed=True).mean()
                dn = ym.groupby(q, observed=True).count()
                if len(dm) and (dm[dn >= min_n] >= rate).any():
                    leaks.append(col)
            except (ValueError, IndexError):
                pass
    return leaks


def load_and_clean(verbose: bool = True, apply_bucket_leaks: bool = True):
    if verbose:
        print(f"Loading {RAW_CSV.name} ...", flush=True)
    df = pd.read_csv(RAW_CSV, index_col=0, low_memory=False)
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    n_start = X.shape[1]
    report = {"n_rows": int(X.shape[0]), "n_features_raw": int(n_start),
              "n_positives": int(y.sum()), "prevalence": float(y.mean())}

    # 0) Drop CONFIRMED LEAKAGE columns first (see LEAKAGE_BLOCKLIST rationale)
    leaked = [c for c in LEAKAGE_BLOCKLIST if c in X.columns]
    X = X.drop(columns=leaked)

    # 1) Drop near-empty columns + hyper-sparse columns (too few non-NA to generalize)
    na_frac = X.isna().mean()
    near_empty = na_frac[na_frac > DROP_NA_THRESHOLD].index.tolist()
    nonna_count = X.notna().sum()
    hyper_sparse = nonna_count[nonna_count < MIN_NONNA].index.tolist()
    drop1 = sorted(set(near_empty) | set(hyper_sparse))
    X = X.drop(columns=drop1)

    # 2) Drop constant / single-value columns
    nunique = X.nunique(dropna=True)
    constant = nunique[nunique <= 1].index.tolist()
    X = X.drop(columns=constant)

    # 3) Missingness flags for informative high-NA columns (computed BEFORE encoding).
    #    Many high-NA columns share the SAME NA pattern (whole feature blocks empty
    #    for the same accounts) -> dedupe identical patterns to kill multicollinearity
    #    and shrink the overfit surface (critical with only 81 positives).
    na_frac2 = X.isna().mean()
    flag_cols = na_frac2[na_frac2 > HIGH_NA_FLAG_THRESHOLD].index.tolist()
    miss_flags = pd.DataFrame(
        {f"{c}__ismissing": X[c].isna().astype(np.int8) for c in flag_cols},
        index=X.index,
    )
    n_flags_raw = miss_flags.shape[1]
    miss_flags = miss_flags.loc[:, ~miss_flags.T.duplicated()]  # keep one per unique pattern

    # 4) Ordinal-encode object columns (deterministic, target-free)
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    cat_maps = {}
    for c in cat_cols:
        cats = sorted(X[c].dropna().unique().tolist(), key=lambda v: str(v))
        mapping = {v: i for i, v in enumerate(cats)}
        cat_maps[c] = {str(k): int(v) for k, v in mapping.items()}
        X[c] = X[c].map(mapping).astype("float32")  # NaN stays NaN

    # 5) Coerce everything to float32 (memory + uniform dtype; NaN preserved)
    X = X.astype("float32")

    # 6) Attach missingness flags
    X = pd.concat([X, miss_flags], axis=1)

    # 7) Auto-detect & drop value-bucket leaks (the F2230-style non-monotonic leaks).
    #    NOTE: this uses y on the FULL dataset — it is a ONE-TIME feature-exclusion
    #    decision (a "frozen reference blocklist"), not a per-fold target transform, so
    #    it does not leak test labels into the model. The sensitivity sweep confirms the
    #    blocklist is stable, so in-fold vs global detection would not move the headline.
    bucket_leaks = detect_bucket_leaks(X, y) if apply_bucket_leaks else []
    X = X.drop(columns=bucket_leaks)

    # 8) Duplicate-row sanity (cheap insurance; matters with only 81 positives)
    n_dup = int(X.duplicated().sum())

    report.update({
        "dropped_leakage": leaked,
        "dropped_bucket_leaks": bucket_leaks,
        "n_bucket_leaks": len(bucket_leaks),
        "duplicate_rows": n_dup,
        "dropped_near_empty": len(near_empty),
        "dropped_hyper_sparse": len(hyper_sparse),
        "dropped_constant": len(constant),
        "n_missing_flags_raw": int(n_flags_raw),
        "n_missing_flags": int(miss_flags.shape[1]),
        "categorical_cols": cat_cols,
        "n_features_final": int(X.shape[1]),
        "hint_features_present": [h for h in HINT_FEATURES if h in X.columns],
    })

    if verbose:
        print(f"  raw features:        {n_start}")
        print(f"  - LEAKAGE blocked:   {len(leaked)} {leaked}")
        print(f"  - bucket-leaks auto: {len(bucket_leaks)} {bucket_leaks}")
        print(f"  - near-empty (>97%): {len(near_empty)}")
        print(f"  - hyper-sparse(<{MIN_NONNA}): {len(hyper_sparse)}")
        print(f"  - constant:          {len(constant)}")
        print(f"  + missingness flags: {miss_flags.shape[1]} (deduped from {n_flags_raw})")
        print(f"  = final features:    {X.shape[1]}")
        print(f"  positives: {report['n_positives']} / {report['n_rows']} "
              f"({report['prevalence']*100:.2f}%)")

    return X, y, report, cat_maps


def cache():
    X, y, report, cat_maps = load_and_clean()
    X.to_parquet(ART / "X_clean.parquet")
    y.to_frame("target").to_parquet(ART / "y.parquet")
    (ART / "preprocess_report.json").write_text(json.dumps(report, indent=2))
    (ART / "categorical_maps.json").write_text(json.dumps(cat_maps, indent=2))
    print(f"\nCached -> {ART/'X_clean.parquet'}  shape={X.shape}")
    return X, y, report


def load_cached():
    X = pd.read_parquet(ART / "X_clean.parquet")
    y = pd.read_parquet(ART / "y.parquet")["target"]
    return X, y


if __name__ == "__main__":
    cache()
