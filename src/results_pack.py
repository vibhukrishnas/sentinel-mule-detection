"""
SENTINEL — Visual results pack + the actual product output.

Generates the things a judge/bank actually wants to SEE and USE:
  figures/  — PR curve, ROC, score distribution, confusion matrix, calibration,
              feature importance, SHAP beeswarm, leakage-sensitivity, ₹-cost curve.
  outputs/predictions.csv          — every account: OOF risk score, band, label, prob.
  outputs/top_suspicious_accounts.csv — the ranked watchlist + plain-English reasons.

All on leakage-free out-of-fold predictions, so the visuals match the reported 0.878.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (precision_recall_curve, roc_curve, average_precision_score,
                             roc_auc_score, confusion_matrix)
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocess import load_cached, ART
from model_config import make_lgbm
from sentinel import SentinelEngine
from feature_meanings import human_label

FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
MULE, LEGIT = "#d62728", "#2ca02c"


def main():
    X, y = load_cached(); yv = y.values
    pw = float((y == 0).sum() / (y == 1).sum())
    model = make_lgbm(pw)   # tuned config (src/model_config.py)
    print("Computing out-of-fold predictions (5-fold)...", flush=True)
    p = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          method="predict_proba", n_jobs=1)[:, 1]
    ap = average_precision_score(yv, p); auc = roc_auc_score(yv, p)
    score = np.round(100 * p).astype(int)
    print(f"OOF PR-AUC={ap:.3f}  ROC-AUC={auc:.3f}", flush=True)

    # ---------- 1. PR curve ----------
    pr, rc, _ = precision_recall_curve(yv, p)
    plt.figure(figsize=(6, 5)); plt.plot(rc, pr, color=MULE, lw=2, label=f"SENTINEL (AP={ap:.3f})")
    plt.axhline(yv.mean(), ls="--", c="gray", label=f"random ({yv.mean():.4f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall (out-of-fold)")
    plt.legend(); plt.tight_layout(); plt.savefig(FIG / "01_pr_curve.png"); plt.close()

    # ---------- 2. ROC ----------
    fpr, tpr, _ = roc_curve(yv, p)
    plt.figure(figsize=(6, 5)); plt.plot(fpr, tpr, color=MULE, lw=2, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], ls="--", c="gray"); plt.xlabel("False positive rate")
    plt.ylabel("True positive rate"); plt.title("ROC (out-of-fold)"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG / "02_roc_curve.png"); plt.close()

    # ---------- 3. Score distribution by class ----------
    plt.figure(figsize=(7, 5))
    plt.hist(score[yv == 0], bins=40, color=LEGIT, alpha=0.6, density=True, label="legitimate")
    plt.hist(score[yv == 1], bins=40, color=MULE, alpha=0.7, density=True, label="mule")
    plt.axvline(70, ls="--", c="black", label="HIGH threshold (70)")
    plt.xlabel("Risk score (0–100)"); plt.ylabel("density")
    plt.title("Risk-score distribution: mules pushed to the right"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG / "03_score_distribution.png"); plt.close()

    # ---------- 4. Confusion matrix @ recommended threshold ----------
    thr = 0.05
    cm = confusion_matrix(yv, (p >= thr).astype(int))
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap="Reds"); plt.colorbar()
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, f"{v:,}", ha="center", va="center",
                 color="white" if v > cm.max() / 2 else "black", fontsize=13, fontweight="bold")
    plt.xticks([0, 1], ["pred legit", "pred mule"]); plt.yticks([0, 1], ["actual legit", "actual mule"])
    tp, fn = cm[1, 1], cm[1, 0]
    plt.title(f"Confusion @ score≥{int(thr*100)}\nrecall={tp/(tp+fn):.0%}, mules caught {tp}/{tp+fn}")
    plt.tight_layout(); plt.savefig(FIG / "04_confusion_matrix.png"); plt.close()

    # ---------- 5. Calibration ----------
    frac, mean_pred = calibration_curve(yv, p, n_bins=8, strategy="quantile")
    plt.figure(figsize=(6, 5)); plt.plot(mean_pred, frac, "o-", color=MULE, label="SENTINEL")
    plt.plot([0, 1], [0, 1], ls="--", c="gray", label="perfect"); plt.xlabel("predicted probability")
    plt.ylabel("observed fraud fraction"); plt.title("Calibration (out-of-fold)"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG / "05_calibration.png"); plt.close()

    # ---------- 6. Feature importance ----------
    base = model.fit(X, y)
    gain = pd.Series(base.booster_.feature_importance("gain"), index=X.columns).sort_values()[-20:]
    plt.figure(figsize=(7, 7)); plt.barh([human_label(c)[:42] for c in gain.index], gain.values, color=MULE)
    plt.xlabel("gain"); plt.title("Top-20 features by importance"); plt.tight_layout()
    plt.savefig(FIG / "06_feature_importance.png"); plt.close()

    # ---------- 7. SHAP beeswarm (sample) ----------
    try:
        import shap
        idx = np.random.RandomState(42).choice(len(X), min(800, len(X)), replace=False)
        ex = shap.TreeExplainer(base); sv = ex.shap_values(X.iloc[idx])
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if np.ndim(sv) == 3 else sv)
        shap.summary_plot(sv, X.iloc[idx], max_display=15, show=False)
        plt.title("SHAP — what drives the risk score"); plt.tight_layout()
        plt.savefig(FIG / "07_shap_summary.png", bbox_inches="tight"); plt.close()
    except Exception as e:
        print(f"(shap plot skipped: {e})")

    # ---------- 8. Leakage sensitivity ----------
    ls = ART / "leak_sensitivity.json"
    if ls.exists():
        d = pd.DataFrame(json.loads(ls.read_text()))
        plt.figure(figsize=(7, 5)); plt.plot(range(len(d)), d["pr_auc"], "o-", color=MULE)
        plt.xticks(range(len(d)), [f"{int(n)}" for n in d["n_leaks"]], rotation=45)
        plt.axhline(0.0089, ls="--", c="gray", label="random baseline")
        plt.xlabel("leak features removed"); plt.ylabel("CV PR-AUC")
        plt.title("Leakage-sensitivity: 0.998→plateau ~0.86 as leaks removed"); plt.legend()
        plt.tight_layout(); plt.savefig(FIG / "08_leakage_sensitivity.png"); plt.close()

    # ---------- 9. ₹-cost curve ----------
    ins = ART / "insights.json"
    if ins.exists():
        c = pd.DataFrame(json.loads(ins.read_text())["cost_curve"])
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax1.plot(c["threshold"], c["net_savings"] / 1e7, "o-", color=MULE, label="net savings (₹ cr)")
        ax1.set_xlabel("alert threshold"); ax1.set_ylabel("net savings (₹ crore)", color=MULE)
        ax2 = ax1.twinx(); ax2.plot(c["threshold"], c["recall"], "s--", color="#1f77b4", label="recall")
        ax2.set_ylabel("recall", color="#1f77b4"); plt.title("₹-cost vs recall by threshold")
        fig.tight_layout(); plt.savefig(FIG / "09_cost_curve.png"); plt.close()

    # ---------- predictions + watchlist ----------
    band = pd.cut(score, [-1, 39, 69, 89, 100], labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    preds = pd.DataFrame({"account_id": X.index, "risk_score": score,
                          "band": band, "probability": p.round(4), "actual_label": yv})
    preds.sort_values("risk_score", ascending=False).to_csv(OUT / "predictions.csv", index=False)

    # watchlist: top 50 with plain-English reasons (SHAP via the deployed engine)
    eng = SentinelEngine()
    top = preds.sort_values("risk_score", ascending=False).head(50).copy()
    reasons = []
    for aid in top["account_id"]:
        drv = [d for d in eng.explain(X.loc[aid], top_k=3) if d["shap"] > 0]
        reasons.append(" | ".join(f"{d['label']}={d['value_readable']}" for d in drv[:3]))
    top["top_reasons"] = reasons
    top.to_csv(OUT / "top_suspicious_accounts.csv", index=False)

    print("\n=== TOP 15 SUSPICIOUS ACCOUNTS (the watchlist) ===")
    print(top[["account_id", "risk_score", "band", "actual_label"]].head(15).to_string(index=False))
    hit = (top["actual_label"] == 1).sum()
    print(f"\nOf the top-50 flagged, {hit} are real mules "
          f"(precision@50 = {hit/50:.0%}; only 0.89% of accounts are mules).")
    print(f"\nGenerated {len(list(FIG.glob('*.png')))} figures -> figures/")
    print(f"Wrote outputs/predictions.csv ({len(preds):,} rows) + top_suspicious_accounts.csv")


if __name__ == "__main__":
    main()
