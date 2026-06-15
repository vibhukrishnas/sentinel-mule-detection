"""
Non-overlapping OOF + bootstrap CI — fixes the "the printed CI is a lower bound" caveat.

The 5x2 repeated folds share ~80% of rows, so the t-CI understates uncertainty. Here we
take a SINGLE non-overlapping 10-fold split (each account predicted exactly once by a model
that never saw it), then bootstrap the PR-AUC by resampling accounts with replacement. The
percentile interval is an honest, assumption-light CI.

Run: python src/bootstrap_ci.py   ->  artifacts/bootstrap_ci.json   (needs artifacts/X_clean.parquet)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score

from preprocess import load_cached
from model_config import make_lgbm

ART = Path(__file__).resolve().parent.parent / "artifacts"
N_BOOT = 2000
RNG = np.random.RandomState(42)


def main():
    X, y = load_cached()
    yv = y.to_numpy()
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        pw = float((yv[tr] == 0).sum() / (yv[tr] == 1).sum())
        m = make_lgbm(pw)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        print(f"  fold {k:2d} done", flush=True)

    pr = average_precision_score(yv, oof)
    roc = roc_auc_score(yv, oof)
    # bootstrap over accounts (resample rows with replacement)
    boots = []
    n = len(yv)
    for _ in range(N_BOOT):
        idx = RNG.randint(0, n, n)
        if yv[idx].sum() < 2:                 # need both classes
            continue
        boots.append(average_precision_score(yv[idx], oof[idx]))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    out = {
        "method": "single non-overlapping 10-fold OOF, bootstrap CI over accounts (B=2000)",
        "oof_pr_auc": round(float(pr), 4), "oof_roc_auc": round(float(roc), 4),
        "bootstrap_pr_auc_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "bootstrap_pr_auc_mean": round(float(boots.mean()), 4),
        "headline_5x2_pr_auc": 0.885,
    }
    (ART / "bootstrap_ci.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nOOF PR-AUC {pr:.3f}  |  bootstrap 95% CI [{lo:.3f}, {hi:.3f}]  (headline 0.885)")


if __name__ == "__main__":
    main()
