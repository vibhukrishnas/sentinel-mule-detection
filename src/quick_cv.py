"""Fast single-model 5-fold CV PR-AUC/ROC-AUC — used to iterate on leak removal."""
import sys, warnings, numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from preprocess import load_cached
warnings.filterwarnings("ignore")

name = sys.argv[1] if len(sys.argv) > 1 else "LightGBM"
X, y = load_cached()
pw = float((y == 0).sum() / (y == 1).sum())
print(f"{name}: data {X.shape}, pos={int(y.sum())}", flush=True)

if name == "LightGBM":
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0, min_child_samples=20,
        scale_pos_weight=pw, n_jobs=-1, random_state=42, verbose=-1)
elif name == "XGBoost":
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0, min_child_weight=5,
        scale_pos_weight=pw, tree_method="hist", eval_metric="aucpr", n_jobs=-1,
        random_state=42)

cv = StratifiedKFold(5, shuffle=True, random_state=42)
r = cross_validate(m, X, y, cv=cv, scoring={"pr": "average_precision", "roc": "roc_auc"},
                   return_train_score=True, n_jobs=1)
print(f"PR-AUC {r['test_pr'].mean():.3f}±{r['test_pr'].std():.3f} | "
      f"ROC-AUC {r['test_roc'].mean():.3f} | train PR-AUC {r['train_pr'].mean():.3f} | "
      f"baseline {y.mean():.4f}")
