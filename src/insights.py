"""
SENTINEL — Business-intelligence layer (one out-of-fold pass, three deliverables):

  (A) RUPEE-COST DECISION CURVE — translates the threshold into money. Banks act on
      ₹ saved, not PR-AUC. Finds the operating point that maximises net savings.
  (B) ERROR ANALYSIS — who are the mules we MISS at the recommended threshold, and
      what makes them hard? (worth more than +0.01 PR-AUC.)
  (C) MULE TYPOLOGY — clusters the 81 mules into behavioural archetypes the bank can
      act on, instead of one undifferentiated "fraud" bucket.

All on out-of-fold predictions (leakage-free, uses all 81 positives).
Cost assumptions are explicit and configurable — change them to your bank's reality.
"""
from __future__ import annotations
import sys, json
try:                                   # Windows console defaults to cp1252; ₹ needs UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from preprocess import load_cached, ART

# --- EXPLICIT cost assumptions (₹). Replace with the bank's real figures. ---
AVG_MULE_LOSS = 250_000      # avg ₹ laundered through a mule before it is stopped
ANALYST_REVIEW_COST = 400    # fully-loaded cost to investigate one alert
FP_HARM_COST = 5_000         # friction/churn/reputational cost of flagging a LEGIT customer
CCY = "₹"


def main():
    X, y = load_cached()
    pw = float((y == 0).sum() / (y == 1).sum())
    from model_config import make_lgbm
    model = make_lgbm(pw)   # tuned config (src/model_config.py)
    print("Computing out-of-fold probabilities (5-fold)...", flush=True)
    p = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          method="predict_proba", n_jobs=1)[:, 1]
    yv = y.values
    n_pos = int(yv.sum())

    # ---------- (A) rupee-cost decision curve ----------
    print(f"\n=== (A) {CCY} COST CURVE (mule loss={CCY}{AVG_MULE_LOSS:,}, "
          f"review={CCY}{ANALYST_REVIEW_COST}, false-positive harm={CCY}{FP_HARM_COST:,}) ===")
    rows = []
    # grid extended below 0.05 so the optimum can't be an unverified grid-edge artifact
    grid = np.round(np.concatenate([[0.01, 0.02, 0.03], np.arange(0.05, 1.00, 0.05)]), 2)
    for t in grid:
        pred = p >= t
        tp = int((pred & (yv == 1)).sum()); fp = int((pred & (yv == 0)).sum())
        alerts = tp + fp
        # net vs do-nothing: mule losses prevented MINUS review cost MINUS harm of
        # freezing legitimate customers (the cost high-recall thresholds incur)
        net = tp * AVG_MULE_LOSS - alerts * ANALYST_REVIEW_COST - fp * FP_HARM_COST
        rows.append({"threshold": float(t), "alerts": alerts, "mules_caught": tp,
                     "recall": round(tp / n_pos, 3),
                     "precision": round(tp / alerts, 3) if alerts else None,
                     "net_savings": int(net)})
    curve = pd.DataFrame(rows)
    best = curve.loc[curve["net_savings"].idxmax()]
    print(curve.to_string(index=False))
    print(f"\nOPTIMAL operating point: threshold={best.threshold} -> "
          f"catch {int(best.mules_caught)}/{n_pos} mules ({best.recall:.0%}), "
          f"{int(best.alerts)} alerts, net {CCY}{int(best.net_savings):,}/scored-population.")

    # ---------- (B) error analysis on missed mules ----------
    t = float(best.threshold)
    missed = (yv == 1) & (p < t)
    caught = (yv == 1) & (p >= t)
    base = model.fit(X, y)
    gain = pd.Series(base.booster_.feature_importance("gain"), index=X.columns)
    topf = gain.sort_values(ascending=False).head(15).index
    diff = pd.DataFrame({
        "missed_mean": X.loc[missed, topf].mean(),
        "caught_mean": X.loc[caught, topf].mean(),
        "legit_mean": X.loc[yv == 0, topf].mean(),
    })
    diff["missed_vs_caught_gap"] = (diff["missed_mean"] - diff["caught_mean"]).abs()
    diff = diff.sort_values("missed_vs_caught_gap", ascending=False)
    print(f"\n=== (B) ERROR ANALYSIS: {int(missed.sum())} mules MISSED at threshold {t} ===")
    print(diff.head(8).round(3).to_string())
    print("Insight: missed mules look 'legit' on the top drivers above — the hard tail.")

    # ---------- (C) mule typology ----------
    print(f"\n=== (C) MULE TYPOLOGY (clustering {n_pos} mules) ===")
    Xm = X.loc[yv == 1, topf].fillna(X.loc[yv == 1, topf].median())  # mule-only median (no pop. leak)
    Z = StandardScaler().fit_transform(Xm)
    k = 3
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
    Xm = Xm.copy(); Xm["cluster"] = km.labels_; Xm["risk"] = (100 * p[yv == 1]).round()
    archetypes = []
    for c in range(k):
        grp = Xm[Xm.cluster == c]
        # the 3 features where this cluster deviates most from the other mules
        gmean = Xm[topf].mean()
        dev = ((grp[topf].mean() - gmean) / (Xm[topf].std() + 1e-9)).abs().sort_values(ascending=False)
        archetypes.append({"cluster": c, "size": int(len(grp)),
                           "avg_risk_score": float(grp["risk"].mean()),
                           "distinguishing_features": dev.head(3).index.tolist()})
        print(f"  Archetype {c}: {len(grp)} mules, avg risk {grp['risk'].mean():.0f}/100, "
              f"driven by {dev.head(3).index.tolist()}")

    out = {"cost_assumptions": {"avg_mule_loss": AVG_MULE_LOSS, "analyst_review_cost": ANALYST_REVIEW_COST,
                                "fp_harm_cost": FP_HARM_COST},
           "cost_curve": rows, "optimal_threshold": t,
           "optimal_net_savings": int(best.net_savings), "optimal_recall": float(best.recall),
           "missed_at_optimal": int(missed.sum()),
           "error_analysis_top": diff.head(8).round(4).reset_index().rename(columns={"index": "feature"}).to_dict("records"),
           "typology": archetypes}
    (ART / "insights.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved -> artifacts/insights.json")


if __name__ == "__main__":
    main()
