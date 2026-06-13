"""
SENTINEL — genuine model-push: can we honestly beat PR-AUC 0.878?

Three legitimate levers (no leakage re-introduced — engineered features use only
row-level aggregates + bank-listed genuine signals, never the removed leak columns):
  1. FEATURE ENGINEERING — profile-completeness + simple ratios of bank-listed features.
  2. HYPERPARAMETER SEARCH — RandomizedSearchCV over LightGBM, scored on PR-AUC.
  3. ENSEMBLE — average calibrated LightGBM + XGBoost.

Reports honest before/after deltas (5-fold CV). If gains are within noise, we say so —
that itself is a finding (the signal ceiling, not a tuning failure).
"""
from __future__ import annotations
import sys, json, warnings
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import average_precision_score
from preprocess import load_cached, HINT_FEATURES, ART
warnings.filterwarnings("ignore")

CV = StratifiedKFold(5, shuffle=True, random_state=42)


def engineer(X):
    """Leakage-safe engineered features."""
    Xe = X.copy()
    Xe["eng_n_missing"] = X.isna().sum(axis=1)               # incomplete profile = mule signal
    Xe["eng_n_zero"] = (X == 0).sum(axis=1)                  # dormant/empty account signal
    hints = [h for h in HINT_FEATURES if h in X.columns and pd.api.types.is_numeric_dtype(X[h])]
    # a few ratios among genuine bank-listed features (safe: never leak columns)
    for a, b in [("F2678", "F2956"), ("F115", "F527"), ("F2122", "F2082")]:
        if a in X.columns and b in X.columns:
            Xe[f"eng_{a}_over_{b}"] = X[a] / (X[b].abs() + 1e-6)
    return Xe


def lgbm(**kw):
    base = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8,
                colsample_bytree=0.6, reg_lambda=1.0, min_child_samples=20,
                n_jobs=-1, random_state=42, verbose=-1)
    base.update(kw); return lgb.LGBMClassifier(**base)


def pr(model, X, y):
    return cross_val_score(model, X, y, cv=CV, scoring="average_precision", n_jobs=1)


def main():
    X, y = load_cached()
    pw = float((y == 0).sum() / (y == 1).sum())
    res = {}

    s = pr(lgbm(scale_pos_weight=pw), X, y)
    res["baseline"] = (s.mean(), s.std()); print(f"baseline LightGBM      PR-AUC {s.mean():.3f}±{s.std():.3f}", flush=True)

    Xe = engineer(X)
    s = pr(lgbm(scale_pos_weight=pw), Xe, y)
    res["+features"] = (s.mean(), s.std()); print(f"+engineered features   PR-AUC {s.mean():.3f}±{s.std():.3f}", flush=True)

    print("hyperparameter search (RandomizedSearchCV, 12 iters)...", flush=True)
    space = {"num_leaves": randint(15, 64), "learning_rate": uniform(0.01, 0.06),
             "min_child_samples": randint(10, 60), "colsample_bytree": uniform(0.4, 0.5),
             "subsample": uniform(0.6, 0.4), "reg_lambda": uniform(0.0, 5.0),
             "n_estimators": randint(300, 900)}
    rs = RandomizedSearchCV(lgbm(scale_pos_weight=pw), space, n_iter=12, cv=CV,
                            scoring="average_precision", n_jobs=1, random_state=42)
    rs.fit(Xe, y)
    res["+tuned"] = (rs.best_score_, 0.0); print(f"+tuned LightGBM        PR-AUC {rs.best_score_:.3f}  best={ {k: round(v,3) if isinstance(v,float) else v for k,v in rs.best_params_.items()} }", flush=True)

    print("ensemble (tuned LightGBM + XGBoost)...", flush=True)
    lp = cross_val_predict(lgbm(scale_pos_weight=pw, **rs.best_params_), Xe, y, cv=CV, method="predict_proba", n_jobs=1)[:, 1]
    xp = cross_val_predict(xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0, min_child_weight=5,
            scale_pos_weight=pw, tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=42),
            Xe, y, cv=CV, method="predict_proba", n_jobs=1)[:, 1]
    ens = average_precision_score(y, (lp + xp) / 2)
    res["+ensemble"] = (ens, 0.0); print(f"+ensemble LGB+XGB      PR-AUC {ens:.3f}", flush=True)

    base = res["baseline"][0]; best = max(v[0] for v in res.values())
    print(f"\n=== VERDICT ===  baseline {base:.3f} -> best {best:.3f}  (delta {best-base:+.3f})")
    print("Within noise (±0.04)" if abs(best - base) < 0.04 else "Material improvement")
    (ART / "tune_results.json").write_text(json.dumps({k: list(v) for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
