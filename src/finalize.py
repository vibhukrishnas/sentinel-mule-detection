"""
SENTINEL — Train the chosen model on a proper split, CALIBRATE it, and persist
everything the real-time engine and the demo need.

Honest evaluation protocol:
  - Stratified 80/20 train/holdout. The holdout is touched exactly once, at the end.
  - Calibration (Platt/sigmoid, robust with few positives) fit via internal CV on the
    TRAIN portion only -> the 0-100 risk score means what it says.
  - We report PR-AUC, ROC-AUC, Brier (calibration), and a full threshold table so the
    risk officer can pick the precision/recall trade-off explicitly.
  - We persist population + class-conditional stats so alerts can say "this value is
    in the Nth percentile / typical of mules", grounded in data, not invented.
"""
from __future__ import annotations
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             precision_recall_curve)

from preprocess import load_cached, ART

SEED = 42
MODEL_NAME = "LightGBM"   # set to the tournament winner; overridable via build_final_model


def build_final_model(pos_weight: float, name: str = MODEL_NAME):
    if name == "LightGBM":
        from model_config import make_lgbm
        return make_lgbm(pos_weight)   # tuned config (src/model_config.py)
    if name == "XGBoost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.6, reg_lambda=1.0, min_child_weight=5,
            scale_pos_weight=pos_weight, tree_method="hist", eval_metric="aucpr",
            n_jobs=-1, random_state=SEED)
    if name == "HistGBM":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, class_weight="balanced",
            random_state=SEED)
    raise ValueError(name)


def threshold_table(y_true, p, grid=(0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)):
    rows = []
    n_pos = int(y_true.sum())
    for t in grid:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        alerts = tp + fp
        prec = tp / alerts if alerts else 0.0
        rec = tp / n_pos if n_pos else 0.0
        rows.append({"threshold": t, "alerts": alerts, "mules_caught": tp,
                     "false_alarms": fp, "precision": round(prec, 3),
                     "recall": round(rec, 3)})
    return rows


def main(name: str = MODEL_NAME):
    X, y = load_cached()
    pos_weight = float((y == 0).sum() / (y == 1).sum())

    X_tr, X_ho, y_tr, y_ho = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    print(f"Train {X_tr.shape} ({int(y_tr.sum())} pos) | "
          f"Holdout {X_ho.shape} ({int(y_ho.sum())} pos)")

    base = build_final_model(pos_weight, name)
    print(f"Calibrating {name} (sigmoid, 5-fold internal CV on train)...", flush=True)
    t0 = time.time()
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    clf.fit(X_tr, y_tr)
    print(f"  fit in {time.time()-t0:.1f}s")

    p_ho = clf.predict_proba(X_ho)[:, 1]
    metrics = {
        "model": name,
        "holdout_pr_auc": float(average_precision_score(y_ho, p_ho)),
        "holdout_roc_auc": float(roc_auc_score(y_ho, p_ho)),
        "holdout_brier": float(brier_score_loss(y_ho, p_ho)),
        "prevalence": float(y.mean()),
        "holdout_positives": int(y_ho.sum()),
    }
    # recall at precision >= 0.5
    prec, rec, thr = precision_recall_curve(y_ho, p_ho)
    ok = prec[:-1] >= 0.5
    metrics["recall_at_prec50"] = float(rec[:-1][ok].max()) if ok.any() else 0.0
    metrics["threshold_table"] = threshold_table(y_ho.values, p_ho)

    print("\n=== HOLDOUT METRICS ===")
    print(f"PR-AUC  : {metrics['holdout_pr_auc']:.3f}  "
          f"(random baseline = {metrics['prevalence']:.4f})")
    print(f"ROC-AUC : {metrics['holdout_roc_auc']:.3f}")
    print(f"Brier   : {metrics['holdout_brier']:.4f}  (lower = better calibrated)")
    print(f"Recall @ precision>=0.5 : {metrics['recall_at_prec50']:.3f}")
    print("\nThreshold table (risk officer's dial):")
    print(pd.DataFrame(metrics["threshold_table"]).to_string(index=False))

    # Refit calibrated model on ALL data for deployment
    print("\nRefitting on full dataset for deployment artifact...", flush=True)
    final = CalibratedClassifierCV(build_final_model(pos_weight, name),
                                   method="sigmoid", cv=5)
    final.fit(X, y)
    # Standalone base model on ALL data -> exact, fast SHAP attribution
    base_full = build_final_model(pos_weight, name)
    base_full.fit(X, y)
    joblib.dump(base_full, ART / "base_model.joblib")

    # Population + class-conditional stats for data-grounded narratives
    stats = {
        "columns": list(X.columns),
        "median": X.median(numeric_only=True).to_dict(),
        "q05": X.quantile(0.05).to_dict(),
        "q95": X.quantile(0.95).to_dict(),
        "mean_legit": X[y == 0].mean(numeric_only=True).to_dict(),
        "mean_mule": X[y == 1].mean(numeric_only=True).to_dict(),
        "std": X.std(numeric_only=True).replace(0, np.nan).to_dict(),
    }

    joblib.dump(final, ART / "sentinel_model.joblib")
    (ART / "holdout_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    joblib.dump(stats, ART / "population_stats.joblib")
    # a few real mule + legit example rows for the demo
    demo = pd.concat([X[y == 1].head(5), X[y == 0].head(5)])
    demo_y = pd.concat([y[y == 1].head(5), y[y == 0].head(5)])
    demo.to_parquet(ART / "demo_accounts.parquet")
    demo_y.to_frame("target").to_parquet(ART / "demo_targets.parquet")
    print(f"\nSaved deployment artifacts -> {ART}")
    return metrics


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else MODEL_NAME)
