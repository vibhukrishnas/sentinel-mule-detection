"""Single source of truth for the deployed model config.

Tuned via RandomizedSearchCV (src/tune.py): lifts CV PR-AUC 0.888 -> 0.899 with a
better-regularized profile (larger min_child_samples + reg_lambda, more trees). The
+0.013 gain is within the ±0.04 CV noise band — we adopt it for the sounder
regularization, not to chase a decimal. (Ensemble LGB+XGB reaches 0.901 but a single
model keeps SHAP exact and deployment simple.)"""
import lightgbm as lgb

LGBM_PARAMS = dict(
    n_estimators=776, learning_rate=0.036, num_leaves=32, subsample=0.976,
    colsample_bytree=0.778, reg_lambda=3.876, min_child_samples=51,
    n_jobs=-1, random_state=42, verbose=-1,
)


def make_lgbm(scale_pos_weight, **override):
    p = dict(LGBM_PARAMS, scale_pos_weight=scale_pos_weight)
    p.update(override)
    return lgb.LGBMClassifier(**p)
