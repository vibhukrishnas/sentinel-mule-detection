"""
Robustness probe — does the model degrade GRACEFULLY under noise & missing data?

A model trained on 81 positives could be brittle. We measure honest OOF PR-AUC as we
(a) inject Gaussian noise scaled to each feature's std, and (b) randomly drop features
to NaN (LightGBM consumes NaN natively). Graceful, monotone degradation = robust signal,
not memorized spikes.

Run: python src/robustness.py   ->  artifacts/robustness.json   (needs artifacts/X_clean.parquet)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

from preprocess import load_cached
from model_config import make_lgbm

ART = Path(__file__).resolve().parent.parent / "artifacts"
RNG = np.random.RandomState(42)
NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0]      # x feature std (CONTINUOUS features only)
DROP_LEVELS = [0.0, 0.1, 0.25, 0.5]            # fraction of features blanked


def continuous_cols(X):
    """Only genuinely continuous features — additive Gaussian noise is meaningless on
    binary missingness flags and ordinal-encoded categoricals, so we exclude them."""
    cat = set(json.loads((ART / "categorical_maps.json").read_text()).keys())
    out = []
    for c in X.columns:
        if c.endswith("__ismissing") or c in cat:
            continue
        if X[c].nunique(dropna=True) > 20:     # discrete/low-card -> skip
            out.append(c)
    return out


def oof_models(X, y):
    """Fit one model per fold; return list of (model, val_idx)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    yv = y.to_numpy()
    for tr, va in skf.split(X, y):
        pw = float((yv[tr] == 0).sum() / (yv[tr] == 1).sum())
        m = make_lgbm(pw)
        m.fit(X.iloc[tr], y.iloc[tr])
        folds.append((m, va))
    return folds


def scored(folds, X, y, perturb):
    """Pooled OOF PR-AUC after applying `perturb` to each fold's validation block."""
    yv = y.to_numpy()
    oof = np.zeros(len(y))
    for m, va in folds:
        Xv = X.iloc[va].astype("float32").copy()    # float copy -> NaN assignment is clean
        oof[va] = m.predict_proba(perturb(Xv))[:, 1]
    return float(average_precision_score(yv, oof))


def main():
    X, y = load_cached()
    folds = oof_models(X, y)
    cont = continuous_cols(X)
    cont_idx = [X.columns.get_loc(c) for c in cont]
    cont_std = X[cont].std().fillna(0.0).to_numpy()
    n_cols = X.shape[1]
    print(f"continuous features perturbed by noise: {len(cont)} of {n_cols}", flush=True)

    noise = {}
    for lv in NOISE_LEVELS:
        def perturb(Xv, lv=lv):
            if lv == 0:
                return Xv
            arr = Xv.to_numpy(copy=True)
            sub = arr[:, cont_idx]
            mask = ~np.isnan(sub)
            noise_mat = RNG.normal(0, 1, sub.shape) * (cont_std * lv)
            sub[mask] += noise_mat[mask]
            arr[:, cont_idx] = sub
            return pd.DataFrame(arr, index=Xv.index, columns=Xv.columns)
        noise[lv] = round(scored(folds, X, y, perturb), 4)
        print(f"  noise x{lv} (continuous): PR-AUC={noise[lv]}", flush=True)

    drop = {}
    for lv in DROP_LEVELS:
        def perturb(Xv, lv=lv):
            if lv == 0:
                return Xv
            k = int(n_cols * lv)
            blank = RNG.choice(n_cols, k, replace=False)
            Xv.iloc[:, blank] = np.nan
            return Xv
        drop[lv] = round(scored(folds, X, y, perturb), 4)
        print(f"  drop {int(lv*100)}%: PR-AUC={drop[lv]}", flush=True)

    base = noise[0.0]
    out = {
        "baseline_oof_pr_auc": base,
        "n_continuous_perturbed": len(cont), "n_features": int(n_cols),
        "noise_injection_continuous": {str(k): v for k, v in noise.items()},
        "feature_dropout": {str(k): v for k, v in drop.items()},
        "retained_frac_noise_x0.25": round(noise[0.25] / base, 3),
        "retained_frac_dropout_25pct": round(drop[0.25] / base, 3),
        "note": ("5-fold OOF; perturbations on validation only. Gaussian noise applied ONLY to "
                 "continuous features (binary flags + ordinal categoricals excluded, where additive "
                 "noise is meaningless). Graceful degradation = robust signal."),
    }
    (ART / "robustness.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
