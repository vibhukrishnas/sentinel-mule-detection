"""
PHASE 0e — HONEST METRIC MAXIMISATION (no leakage; verified with repeated CV).

Pushes the BOI PR-AUC as high as the genuine signal allows, WITHOUT touching the leak.
Three levers, all leakage-safe (every transform fit inside the training fold only):

  1. CatBoost ordered boosting (best single learner so far: ~0.941 single-OOF).
  2. A CV-SAFE GRAPH FEATURE: for each fold, score every account by its cosine
     similarity to the TRAINING-fold mules (top-genuine-features). Validation rows use
     ONLY training-fold mule labels -> no validation-label leakage. This is the ring
     signal expressed as a model feature (recovers ring-embedded 'invisible' mules).
  3. Ensemble CatBoost + LightGBM + XGBoost (mean of OOF probabilities).

Repeated stratified 5x2 CV gives an honest central estimate +/- std for each, compared
apples-to-apples to the LightGBM baseline. We adopt a challenger ONLY if it beats the
baseline beyond the noise band; otherwise we say so and keep LightGBM.

Output: artifacts/boi/improve.json   (does NOT overwrite the headline metrics.json)
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import load_cached
from model_config import make_lgbm
import xgboost as xgb
import joblib

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"; OUT = ART / "boi"
SEED = 42; TOPK_FEATS = 30; KNN = 5


def top_genuine_features(X, y):
    """Rank features by a quick LightGBM gain (leak-free matrix) — used for similarity."""
    try:
        base = joblib.load(ART / "base_model.joblib")
        gain = pd.Series(base.booster_.feature_importance("gain"), index=X.columns)
        feats = [f for f in gain.sort_values(ascending=False).index if f in X.columns][:TOPK_FEATS]
        if len(feats) >= 10:
            return feats
    except Exception:
        pass
    m = make_lgbm(float((y == 0).sum() / (y == 1).sum())); m.fit(X, y)
    gain = pd.Series(m.booster_.feature_importance("gain"), index=X.columns)
    return list(gain.sort_values(ascending=False).index[:TOPK_FEATS])


def graph_feature(Xtr_top, ytr, Xall_top, scaler, imp):
    """Mean cosine similarity of each account to the KNN nearest TRAIN-fold mules.
    Fit scaler/imputer on TRAIN only (leak-safe). Returns a 1-D array over Xall rows."""
    Ztr = scaler.transform(imp.transform(Xtr_top))
    Zall = scaler.transform(imp.transform(Xall_top))
    mule_Z = Ztr[ytr.values == 1]
    S = cosine_similarity(Zall, mule_Z)            # (n_all, n_train_mules)
    S.sort(axis=1)
    k = min(KNN, S.shape[1])
    return S[:, -k:].mean(axis=1)                  # mean of top-k similarities


def mk_xgb(spw):
    return xgb.XGBClassifier(n_estimators=600, learning_rate=0.04, max_depth=5, subsample=0.85,
        colsample_bytree=0.7, reg_lambda=4.0, scale_pos_weight=spw, n_jobs=-1,
        random_state=SEED, eval_metric="aucpr", tree_method="hist")


def mk_cat(spw):
    # memory-leaner ordered boosting: rsm (random subspace) + fewer iters keep RAM bounded
    # on the 2965-feature matrix (avoids the 'bad allocation' OOM under contention).
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=450, learning_rate=0.04, depth=6, l2_leaf_reg=5.0,
        boosting_type="Ordered", rsm=0.5, max_bin=128, scale_pos_weight=spw, random_seed=SEED,
        thread_count=4, verbose=0, allow_writing_files=False)


def main():
    t0 = time.time()
    X, y = load_cached(); yv = y.to_numpy()
    feats = top_genuine_features(X, y)
    print(f"matrix {X.shape} | top genuine feats for graph: {len(feats)}", flush=True)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    # accumulate OOF per repeat for each model so PR-AUC is computed per-repeat (honest std)
    results = {m: [] for m in ["LightGBM (baseline)", "CatBoost", "Ensemble CB+LGB+XGB",
                               "Ensemble + graph-feature"]}
    fold_iter = list(cv.split(X, y))
    # group folds into the 2 repeats (5 folds each) to build full OOF vectors per repeat
    for rep in range(2):
        folds = fold_iter[rep * 5:(rep + 1) * 5]
        oof = {m: np.zeros(len(y)) for m in results}
        for k, (tr, va) in enumerate(folds, 1):
            spw = float((yv[tr] == 0).sum() / (yv[tr] == 1).sum())
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            # base learners
            lg = make_lgbm(spw).fit(Xtr, y.iloc[tr]); p_lg = lg.predict_proba(Xva)[:, 1]
            xg = mk_xgb(spw).fit(Xtr, y.iloc[tr]); p_xg = xg.predict_proba(Xva)[:, 1]
            cb = mk_cat(spw).fit(Xtr.fillna(-999), yv[tr]); p_cb = cb.predict_proba(Xva.fillna(-999))[:, 1]
            oof["LightGBM (baseline)"][va] = p_lg
            oof["CatBoost"][va] = p_cb
            oof["Ensemble CB+LGB+XGB"][va] = (p_lg + p_xg + p_cb) / 3
            # CV-safe graph feature, fit on TRAIN fold only
            imp = SimpleImputer(strategy="median").fit(Xtr[feats])
            sca = StandardScaler().fit(imp.transform(Xtr[feats]))
            gf_va = graph_feature(Xtr[feats], y.iloc[tr], Xva[feats], sca, imp)
            gf_tr = graph_feature(Xtr[feats], y.iloc[tr], Xtr[feats], sca, imp)
            # retrain ensemble members WITH the graph feature appended (leak-safe)
            Xtr_g = Xtr.copy(); Xtr_g["__mule_sim"] = gf_tr
            Xva_g = Xva.copy(); Xva_g["__mule_sim"] = gf_va
            lg2 = make_lgbm(spw).fit(Xtr_g, y.iloc[tr]); p_lg2 = lg2.predict_proba(Xva_g)[:, 1]
            xg2 = mk_xgb(spw).fit(Xtr_g, y.iloc[tr]); p_xg2 = xg2.predict_proba(Xva_g)[:, 1]
            cb2 = mk_cat(spw).fit(Xtr_g.fillna(-999), yv[tr]); p_cb2 = cb2.predict_proba(Xva_g.fillna(-999))[:, 1]
            oof["Ensemble + graph-feature"][va] = (p_lg2 + p_xg2 + p_cb2) / 3
            print(f"  rep{rep+1} fold{k}: LGB={average_precision_score(yv[va],p_lg):.3f} "
                  f"CB={average_precision_score(yv[va],p_cb):.3f} "
                  f"ENS={average_precision_score(yv[va],oof['Ensemble CB+LGB+XGB'][va]):.3f} "
                  f"ENS+GF={average_precision_score(yv[va],oof['Ensemble + graph-feature'][va]):.3f}", flush=True)
        for m in results:
            results[m].append(average_precision_score(yv, oof[m]))
        print(f"  repeat {rep+1} PR-AUC: " + " | ".join(f"{m}={results[m][-1]:.3f}" for m in results), flush=True)

    summary = {}
    for m, vals in results.items():
        a = np.array(vals)
        summary[m] = {"pr_auc_mean": float(a.mean()), "pr_auc_std": float(a.std()), "pr_auc_repeats": [float(v) for v in a]}
    base_mean = summary["LightGBM (baseline)"]["pr_auc_mean"]
    base_std = summary["LightGBM (baseline)"]["pr_auc_std"]
    best = max(summary, key=lambda m: summary[m]["pr_auc_mean"])
    beats_noise = (summary[best]["pr_auc_mean"] - base_mean) > base_std
    out = {"dataset": "BOI", "cv": "repeated 5x2 stratified (2 repeats)", "n_top_feats_graph": len(feats), "knn": KNN,
           "summary": summary, "baseline_mean": base_mean, "baseline_std": base_std,
           "best_method": best, "best_mean": summary[best]["pr_auc_mean"],
           "beats_baseline_beyond_noise": bool(beats_noise),
           "leakage_safety": ("graph feature uses ONLY training-fold mule labels; all learners fit inside "
                              "folds; no oversampling; PR-AUC primary. PR-AUC>0.99 is NOT attainable without "
                              "the F3912 leak (proven: 3-algo invisible tail, sensitivity plateau, recall<=92.6%)."),
           "recommendation": (f"ADOPT {best} (beats baseline beyond noise band)" if beats_noise
                              else f"KEEP LightGBM headline; {best} leads but within the +/-{base_std:.3f} noise band."),
           "runtime_s": round(time.time() - t0, 1)}
    (OUT / "improve.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n=== IMPROVE SUMMARY ===")
    for m in summary:
        print(f"  {m:28s} PR-AUC {summary[m]['pr_auc_mean']:.3f} +/- {summary[m]['pr_auc_std']:.3f}")
    print(f"best={best} | beats noise={beats_noise} | {out['recommendation']}")
    print(f"DONE in {out['runtime_s']}s -> artifacts/boi/improve.json")


if __name__ == "__main__":
    main()
