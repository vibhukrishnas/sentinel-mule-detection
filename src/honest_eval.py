"""
Consolidated HONEST evaluation — regenerates every number on the CURRENT leak-removed
matrix so all artifacts tell one story. Fixes from the adversarial review:
  - P0: overwrite the stale tournament_results.json (it held pre-leak-removal 0.99s).
  - P1: report per-fold PR-AUC + a t-based CI, and flag that 5x2 folds overlap so the
        printed CI is a LOWER BOUND on uncertainty.
  - P1: add CV Brier (calibration) instead of trusting the 16-positive holdout.
  - P1: derive risk-score BANDS from pooled out-of-fold probabilities of the 81 real
        mules (uses all positives, leakage-free) rather than the lucky holdout.
  - P3: benchmark the real-time scoring path (predict_proba + SHAP) — measured, not claimed.
"""
from __future__ import annotations
import json, time, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from preprocess import load_cached, ART
from model_config import make_lgbm
warnings.filterwarnings("ignore")

X, y = load_cached()
pw = float((y == 0).sum() / (y == 1).sum())
print(f"Honest eval on leak-removed matrix {X.shape}, pos={int(y.sum())}\n", flush=True)

def lgbm():
    return make_lgbm(pw)   # tuned config (src/model_config.py)

models = {
    "LogReg (L2, balanced)": Pipeline([("i", SimpleImputer(strategy="median")),
        ("s", StandardScaler()), ("c", LogisticRegression(max_iter=2000, C=0.1,
        class_weight="balanced"))]),
    "RandomForest (balanced)": Pipeline([("i", SimpleImputer(strategy="median")),
        ("c", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1,
        random_state=42))]),
    "LightGBM": lgbm(),
}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scoring = {"pr": "average_precision", "roc": "roc_auc", "brier": "neg_brier_score"}
tcrit = float(stats.t.ppf(0.975, df=9))  # 10 folds -> df=9 -> ~2.262

board = []
for name, m in models.items():
    t0 = time.time()
    r = cross_validate(m, X, y, cv=cv, scoring=scoring, return_train_score=False, n_jobs=1)
    folds = r["test_pr"]
    hw = tcrit * folds.std() / np.sqrt(len(folds))      # t-CI half-width (optimistic: folds overlap)
    board.append({
        "model": name,
        "pr_auc": float(folds.mean()), "pr_std": float(folds.std()),
        "pr_folds": [round(float(v), 3) for v in folds],
        "ci_lo_optimistic": float(folds.mean() - hw),
        "ci_hi_optimistic": float(folds.mean() + hw),
        "roc_auc": float(r["test_roc"].mean()),
        "brier": float(-r["test_brier"].mean()),
        "fit_s": (time.time() - t0) / len(folds),
    })
    b = board[-1]
    print(f"{name:26s} PR-AUC {b['pr_auc']:.3f}±{b['pr_std']:.3f} "
          f"(t-CI {b['ci_lo_optimistic']:.3f}-{b['ci_hi_optimistic']:.3f}, optimistic) "
          f"ROC {b['roc_auc']:.3f} Brier {b['brier']:.4f}", flush=True)

board.sort(key=lambda d: -d["pr_auc"])
# overwrite the stale leaderboard so all artifacts agree
(ART / "tournament_results.json").write_text(json.dumps(board, indent=2))

# ---- OOF-derived risk-score bands (uses all 81 mules, leakage-free) ----
print("\nDeriving risk-score bands from out-of-fold probabilities of real mules...", flush=True)
oof = cross_val_predict(lgbm(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                        method="predict_proba", n_jobs=1)[:, 1]
mule_scores = np.round(100 * oof[y.values == 1]).astype(int)
qs = {q: int(np.percentile(mule_scores, q)) for q in [10, 25, 50, 75, 90]}
print(f"Real-mule OOF score percentiles: {qs}")
print(f"  -> {int((mule_scores>=70).mean()*100)}% of real mules score >=70 (HIGH+)")

# ---- latency benchmark (measured, not claimed) ----
print("\nBenchmarking real-time scoring path...", flush=True)
import shap
base = lgbm().fit(X, y)
ex = shap.TreeExplainer(base)
row = X.iloc[[0]]
# warmup
base.predict_proba(row); ex.shap_values(row)
t = []
for _ in range(100):
    s = time.perf_counter(); base.predict_proba(row); t.append((time.perf_counter()-s)*1000)
ts = []
for _ in range(30):
    s = time.perf_counter(); ex.shap_values(row); ts.append((time.perf_counter()-s)*1000)
lat = {"predict_ms_p50": float(np.percentile(t, 50)), "predict_ms_p95": float(np.percentile(t, 95)),
       "shap_ms_p50": float(np.percentile(ts, 50)), "shap_ms_p95": float(np.percentile(ts, 95))}
print(f"  predict_proba: p50 {lat['predict_ms_p50']:.1f}ms p95 {lat['predict_ms_p95']:.1f}ms")
print(f"  + SHAP explain: p50 {lat['shap_ms_p50']:.1f}ms p95 {lat['shap_ms_p95']:.1f}ms")

json.dump({"leaderboard": board, "mule_oof_score_pctl": qs,
           "pct_mules_ge70": float((mule_scores >= 70).mean()), "latency_ms": lat,
           "t_crit_df9": tcrit, "note": "5x2 CV folds overlap ~80%; printed CI is a lower bound"},
          open(ART / "honest_eval.json", "w"), indent=2)
print(f"\nSaved -> {ART/'honest_eval.json'} (and overwrote stale tournament_results.json)")
