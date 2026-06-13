"""
SENTINEL — Model tournament with a LEAKAGE-SAFE, repeated-stratified-CV harness.

Why this file is built the way it is (the discipline that wins):
  - Metric = PR-AUC (average precision) PRIMARY + ROC-AUC. Accuracy is banned
    (predicting "all clean" scores 99.1%). With 0.89% prevalence, PR-AUC is the
    honest ranking metric.
  - RepeatedStratifiedKFold (5 folds x N repeats): one split with ~16 positives in
    the test fold is statistically a coin-flip; we report mean +/- std across many.
  - ALL preprocessing that learns from data (imputation, scaling) lives INSIDE the
    pipeline, so it is re-fit on each train fold only -> zero leakage to test folds.
  - Tree boosters (HistGBM / LightGBM / XGBoost) consume NaN natively; linear/forest
    baselines get median-imputation so we can prove the boosters earn their keep.
  - Imbalance handled by class weights / scale_pos_weight, NOT blind oversampling.
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from preprocess import load_cached, ART

warnings.filterwarnings("ignore")

N_SPLITS = 5
N_REPEATS = 2          # 5x2 = 10 fits/model for the report; bump for the final model
SEED = 42

SCORING = {
    "pr_auc": "average_precision",   # PRIMARY
    "roc_auc": "roc_auc",
}


def build_models(pos_weight: float):
    """Model registry. Boosters see raw NaN; linear/forest get imputation."""
    impute = ("impute", SimpleImputer(strategy="median"))
    models = {}

    # --- Baselines (must be beaten to justify the boosters) ---
    models["LogReg (L2, balanced)"] = Pipeline([
        impute,
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced",
                                   solver="lbfgs", n_jobs=-1)),
    ])
    models["RandomForest (balanced)"] = Pipeline([
        impute,
        ("clf", RandomForestClassifier(n_estimators=400, max_depth=None,
                                       min_samples_leaf=2, max_features="sqrt",
                                       class_weight="balanced_subsample",
                                       n_jobs=-1, random_state=SEED)),
    ])

    # --- Gradient-boosted tree family (NaN-native, the real contenders) ---
    models["HistGBM (balanced)"] = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, class_weight="balanced",
        random_state=SEED)

    import lightgbm as lgb
    models["LightGBM"] = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8,
        colsample_bytree=0.6, reg_lambda=1.0, min_child_samples=20,
        scale_pos_weight=pos_weight, n_jobs=-1, random_state=SEED, verbose=-1)

    import xgboost as xgb
    models["XGBoost"] = xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8,
        colsample_bytree=0.6, reg_lambda=1.0, min_child_weight=5,
        scale_pos_weight=pos_weight, tree_method="hist", eval_metric="aucpr",
        n_jobs=-1, random_state=SEED)

    return models


def run_tournament():
    X, y = load_cached()
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    print(f"Data: {X.shape}, positives={int(y.sum())} ({y.mean()*100:.2f}%), "
          f"scale_pos_weight={pos_weight:.1f}\n")

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    models = build_models(pos_weight)

    rows = []
    for name, model in models.items():
        t0 = time.time()
        try:
            res = cross_validate(model, X, y, cv=cv, scoring=SCORING,
                                 n_jobs=1, return_train_score=True, error_score="raise")
        except Exception as e:  # one model OOM/error must not kill the tournament
            print(f"{name:28s} | FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        dt = time.time() - t0
        row = {
            "model": name,
            "pr_auc": res["test_pr_auc"].mean(),
            "pr_auc_std": res["test_pr_auc"].std(),
            "roc_auc": res["test_roc_auc"].mean(),
            "roc_auc_std": res["test_roc_auc"].std(),
            "train_pr_auc": res["train_pr_auc"].mean(),  # overfit gap check
            "fit_time_s": dt / (N_SPLITS * N_REPEATS),
        }
        rows.append(row)
        # incremental save so a later crash can't erase earlier results
        pd.DataFrame(rows).to_json(ART / "tournament_results.json", orient="records", indent=2)
        print(f"{name:28s} | PR-AUC {row['pr_auc']:.3f}±{row['pr_auc_std']:.3f} "
              f"| ROC-AUC {row['roc_auc']:.3f}±{row['roc_auc_std']:.3f} "
              f"| train PR-AUC {row['train_pr_auc']:.3f} "
              f"| {row['fit_time_s']:.1f}s/fit", flush=True)

    res_df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    print("\n=== LEADERBOARD (by PR-AUC) ===")
    print(res_df[["model", "pr_auc", "pr_auc_std", "roc_auc", "train_pr_auc"]]
          .to_string(index=False))

    baseline_pr = 1.0 * y.mean()  # PR-AUC of a random classifier == prevalence
    print(f"\nRandom-baseline PR-AUC (prevalence) = {baseline_pr:.4f}")
    winner = res_df.iloc[0]
    lift = winner["pr_auc"] / baseline_pr
    print(f"Winner: {winner['model']}  ->  {lift:.0f}x better than random.")
    print(f"Overfit gap (train-test PR-AUC) for winner: "
          f"{winner['train_pr_auc'] - winner['pr_auc']:.3f}")

    res_df.to_json(ART / "tournament_results.json", orient="records", indent=2)
    print(f"\nSaved -> {ART/'tournament_results.json'}")
    return res_df


if __name__ == "__main__":
    run_tournament()
