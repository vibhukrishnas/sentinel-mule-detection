# 🛡️ SENTINEL — Suspicious / Mule Account Risk Engine
**BOI Hackathon · Problem Statement 2 · AI/ML Classification of Suspicious Mule Accounts**

SENTINEL turns 3,924 raw, anonymized, half-empty account features into a **single
calibrated risk score (0–100)** per account — in milliseconds — with a **plain-English
"why" attached to every alert** and an **auto-generated, analyst-ready investigation
report**. Built fundamentals-first: ruthless feature hygiene, leakage-proof validation,
the right metric (PR-AUC, not accuracy), calibrated probabilities, and explainability.

> **Honest headline: PR-AUC 0.81–0.89, ROC-AUC ~0.98** (repeated CV; ~91–99× the 0.0089
> random baseline). The range is a leak-paranoid floor (~0.81, near-label block F3895–F3923
> removed) to best-estimate (tuned 0.885 ± 0.055); the exact number depends on the bank confirming
> whether F3895–F3923 are pre- or post-event features. We lead with the range, not a single
> flattering point.
>
> **Why a range and not 1.0?** This dataset is **riddled with target leakage**. A naive
> XGBoost scores **PR-AUC 1.000** — impossible for real fraud detection; it's reading the
> answer. We built **rigorous, end-to-end leakage detection** that found and removed `F3912`
> (a 96%-aligned label proxy) **plus ~584 non-monotonic "bucket/range leaks"** (e.g. `F2230`,
> a month-stamp that is 100% fraud for 3 of 4 values — invisible to monotonic AUC). Naive
> competitors ship a 0.99 they can't explain when a judge asks "why so perfect?" — we hand
> over the audit and the defensible number.

---

## The data, honestly
| | |
|---|---|
| Accounts | 9,082 |
| Features | 3,924 anonymized (`F1`..`F3923`), target `F3924` |
| **Mules (positives)** | **81 (0.89%)** — severe imbalance, tiny positive set |
| Missing cells | 27.6% |
| Columns dropped | 281 constant + 267 near-empty + 135 hyper-sparse + **1 label-proxy (F3912) + ~584 bucket/range leaks** |
| Shape used for modeling | 9,082 × 2,965 (incl. 175 deduped missingness flags) |

## Pipeline
```
DataSet.csv
  └─ src/preprocess.py             hygiene, encode, missingness flags, auto leak removal  → artifacts/X_clean.parquet
  └─ src/data_integrity_auditor.py 4-signature leakage audit → DATA_INTEGRITY_AUDIT.md  (the differentiator)
  └─ src/honest_eval.py            leak-removed leaderboard + CV Brier + t-CI + OOF bands + latency
  └─ src/finalize.py               train + calibrate winner, holdout, threshold table, deployment artifacts
  └─ src/insights.py               ₹-cost decision curve + error analysis + mule typology
  └─ src/sentinel.py               real-time scoring + SHAP explainability + alerts + investigation reports
  └─ src/api.py                    FastAPI /score + /report service
  └─ app.py                        Streamlit demo dashboard
```

## How to run
```bash
pip install -r requirements.txt
python src/preprocess.py              # cache clean, leak-removed matrix
python src/data_integrity_auditor.py  # leakage audit -> DATA_INTEGRITY_AUDIT.md
python src/honest_eval.py             # honest leaderboard + bands + latency
python src/finalize.py LightGBM       # train+calibrate winner, save artifacts
python src/insights.py                # ₹-cost curve, error analysis, mule typology
python -m pytest tests/ -q            # smoke tests (leakage blocked, sane metrics)
streamlit run app.py                  # live demo dashboard
uvicorn src.api:app                   # real-time scoring API
```

## Why this wins (and how a stronger team would attack us)
- **Right metric:** PR-AUC at 0.89% prevalence; accuracy is explicitly banned (99.1% by predicting "all clean").
- **Leakage-proof:** all learned preprocessing lives inside CV folds; `F3912` caught and removed.
- **Honest validation:** Repeated Stratified K-Fold + a held-out set touched once; metrics reported with spread, not a single lucky split.
- **Calibrated:** the 0–100 score means what it says (Brier-checked).
- **Explainable + actionable:** every alert carries SHAP-driven reasons and a recommended action — the "feel it" factor.
- **Stated limits:** snapshot data (not a transaction stream); no graph/network features provided. We don't pretend otherwise.

## Results (measured, leak-removed, repeated 5×2 CV)
| Model | PR-AUC | ROC-AUC | CV Brier |
|---|---|---|---|
| **LightGBM (tuned, deployed)** | **0.885 ± 0.055** | 0.979 | 0.0022 |
| RandomForest | 0.771 ± 0.065 | 0.977 | 0.0043 |
| LogReg (L2) | 0.404 ± 0.059 | 0.936 | 0.0103 |

- **🎯 precision@50 = 100%** — the model's 50 highest-risk accounts are *all 50 real mules*, zero false positives (in a population that's only 0.89% mules). The actual product: `outputs/top_suspicious_accounts.csv`.
- **Honest range (FLOOR–HEADLINE): PR-AUC 0.81–0.89, ROC-AUC ~0.98** — ~91–99× the 0.0089 random baseline. Even deleting the entire near-label block, the floor holds at ~0.81.
- **Business impact:** ≈ **₹1.7 crore saved** per 9,082-account population at **85–89% mules caught** (net savings flat across thresholds → analyst-capacity-bound, `src/insights.py`).
- **Calibration:** CV Brier 0.0022. **Latency:** score + SHAP ≈ 35 ms (p95). **Recall:** ≈72% of real mules (58/81) score ≥70/100 (out-of-fold).
- The honest range is wider than a tight ±0.04 (5×2 CV folds overlap) — see `RESULTS.md` §7.5 for every caveat we surfaced via our own adversarial review.

## Visual results (`figures/`) + product output (`outputs/`)
Run `python src/results_pack.py`. Generates (all on leakage-free out-of-fold predictions):
`01_pr_curve` · `02_roc_curve` · `03_score_distribution` (mules vs legit) · `04_confusion_matrix` · `05_calibration` · `06_feature_importance` · `07_shap_summary` · `08_leakage_sensitivity` · `09_cost_curve`.
And the actual deliverables a bank uses:
- **`outputs/predictions.csv`** — all 9,082 accounts scored (risk score, band, probability).
- **`outputs/top_suspicious_accounts.csv`** — the ranked **watchlist** with plain-English reasons per account.

> Full leaderboard, sensitivity sweep, threshold dial, caveats, and judge-attack/defense table: **`RESULTS.md`**.
