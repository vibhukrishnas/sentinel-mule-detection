"""
Definitive honest measurement + leak-paranoid ablation floor.

(A) HEADLINE: RepeatedStratifiedKFold(5x2) LightGBM on the leak-removed matrix
    (F3912 + bucket-leaks gone) -> PR-AUC mean +/- std with a confidence interval.
(B) CONSERVATIVE FLOOR: same, but ALSO drop the near-label tail block F3895..F3923
    (keeping the genuine bank-hint features F3887/F3889/F3891/F3894, all <= F3894).
    LightGBM leans hard on F3898/F3908/F3914 there; if those are subtle outcome leaks,
    this floor is the number that still holds. Truth lives between (B) and (A).
"""
from __future__ import annotations
import re, json, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from preprocess import load_cached, ART
warnings.filterwarnings("ignore")

X, y = load_cached()
pw = float((y == 0).sum() / (y == 1).sum())
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

def model():
    return lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0, min_child_samples=20,
        scale_pos_weight=pw, n_jobs=-1, random_state=42, verbose=-1)

def measure(Xm, tag):
    r = cross_validate(model(), Xm, y, cv=cv,
                       scoring={"pr": "average_precision", "roc": "roc_auc"}, n_jobs=1)
    pr, sd, roc = r["test_pr"].mean(), r["test_pr"].std(), r["test_roc"].mean()
    ci = 1.96 * sd / np.sqrt(len(r["test_pr"]))
    print(f"{tag:22s} feats={Xm.shape[1]:5d} | PR-AUC {pr:.3f}±{sd:.3f} "
          f"(95% CI {pr-ci:.3f}-{pr+ci:.3f}) | ROC-AUC {roc:.3f}", flush=True)
    return {"tag": tag, "n_feats": int(Xm.shape[1]), "pr_auc": float(pr),
            "pr_std": float(sd), "ci_lo": float(pr-ci), "ci_hi": float(pr+ci),
            "roc_auc": float(roc)}

# near-label tail block to ablate: F3895..F3923 (genuine hints are all <= F3894)
def in_tail(col):
    m = re.fullmatch(r"F(\d+)", col.split("__")[0])
    return bool(m) and 3895 <= int(m.group(1)) <= 3923

tail = [c for c in X.columns if in_tail(c)]
print(f"Near-label tail features present (ablated in floor): {tail}\n")

out = []
out.append(measure(X, "HEADLINE (all)"))
out.append(measure(X.drop(columns=tail), "FLOOR (no tail block)"))
print(f"\nbaseline PR-AUC = {y.mean():.4f}")
(ART / "definitive_cv.json").write_text(json.dumps(out, indent=2))
print("Honest range: FLOOR <= true behavioral PR-AUC <= HEADLINE")
