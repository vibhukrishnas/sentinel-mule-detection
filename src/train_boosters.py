"""
Lean booster comparison: LightGBM + XGBoost (the real deployment contenders),
appended to the existing leaderboard. sklearn HistGBM was dropped from the
tournament on purpose: it is pathologically slow on 3,549-wide data and is
redundant with these two NaN-native boosters.
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from preprocess import load_cached, ART

warnings.filterwarnings("ignore")
N_SPLITS, N_REPEATS, SEED = 5, 2, 42
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}


def main():
    X, y = load_cached()
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    print(f"Data {X.shape}, pos={int(y.sum())}, spw={pos_weight:.1f}\n", flush=True)

    import lightgbm as lgb, xgboost as xgb
    models = {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8,
            colsample_bytree=0.6, reg_lambda=1.0, min_child_samples=20,
            scale_pos_weight=pos_weight, n_jobs=-1, random_state=SEED, verbose=-1),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.6, reg_lambda=1.0, min_child_weight=5,
            scale_pos_weight=pos_weight, tree_method="hist", eval_metric="aucpr",
            n_jobs=-1, random_state=SEED),
    }

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    existing = json.loads((ART / "tournament_results.json").read_text())
    rows = list(existing)
    for name, model in models.items():
        t0 = time.time()
        res = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=1,
                             return_train_score=True, error_score="raise")
        rows.append({
            "model": name,
            "pr_auc": float(res["test_pr_auc"].mean()),
            "pr_auc_std": float(res["test_pr_auc"].std()),
            "roc_auc": float(res["test_roc_auc"].mean()),
            "roc_auc_std": float(res["test_roc_auc"].std()),
            "train_pr_auc": float(res["train_pr_auc"].mean()),
            "fit_time_s": (time.time() - t0) / (N_SPLITS * N_REPEATS),
        })
        pd.DataFrame(rows).to_json(ART / "tournament_results.json", orient="records", indent=2)
        r = rows[-1]
        print(f"{name:12s} | PR-AUC {r['pr_auc']:.3f}±{r['pr_auc_std']:.3f} "
              f"| ROC-AUC {r['roc_auc']:.3f} | train PR-AUC {r['train_pr_auc']:.3f} "
              f"| {r['fit_time_s']:.1f}s/fit", flush=True)

    df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    print("\n=== FINAL LEADERBOARD (by PR-AUC, F3912 leakage removed) ===")
    print(df[["model", "pr_auc", "pr_auc_std", "roc_auc", "train_pr_auc"]].to_string(index=False))
    base = float(y.mean())
    w = df.iloc[0]
    print(f"\nRandom baseline PR-AUC = {base:.4f}")
    print(f"Winner: {w['model']}  ({w['pr_auc']/base:.0f}x lift, "
          f"overfit gap {w['train_pr_auc']-w['pr_auc']:.3f})")
    df.to_json(ART / "tournament_results.json", orient="records", indent=2)


if __name__ == "__main__":
    main()
