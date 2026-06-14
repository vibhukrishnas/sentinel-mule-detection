# Model Card — SENTINEL Mule-Account Classifier (PS2)

## Overview
- **Task:** binary classification of bank accounts as suspicious/mule (1) vs legitimate (0).
- **Model:** calibrated LightGBM (gradient-boosted trees), sigmoid calibration via 5-fold internal CV.
- **Output:** calibrated probability → 0–100 risk score → severity band (LOW/MEDIUM/HIGH/CRITICAL) + SHAP-based reasons.
- **Version:** 1.0 · 2026-06-10 · seed 42 throughout.

## Training data
- 9,082 accounts × 3,924 anonymized features (`F1`..`F3923`), target `F3924`.
- **81 positives (0.89%)** — severe class imbalance; handled via `scale_pos_weight`, never SMOTE.
- 27.6% missing cells; LightGBM consumes NaN natively.
- After hygiene + leakage removal: **2,9xx features** (constants, near-empty, hyper-sparse, and leak features dropped).

## Leakage handling (critical)
The raw data contains target leakage. Excluded automatically by `src/preprocess.py` +
audited by `src/data_integrity_auditor.py`:
- **`F3912`** — binary flag ~96% aligned with the label (post-hoc fraud flag).
- **`F2230` & ~580 others** — value-bucket / range leaks (e.g. `F2230` is a month-stamp:
  `Oct25`=100% legit, `Sep/Nov/Dec25`=100% fraud, capturing all 81 mules).
Bank-listed features (Topic.pdf) are never auto-removed.

## Performance (honest, leak-removed, repeated 5×2 stratified CV)
| Metric | Value |
|---|---|
| PR-AUC (headline, tuned LightGBM) | **0.885 ± 0.055** |
| PR-AUC (leak-paranoid floor, near-label block removed) | ≈0.81 |
| ROC-AUC | 0.979 |
| CV Brier (calibration) | 0.0022 |
| Score + SHAP latency | ~44 ms p95 |
| Recall / net savings (capacity-bound) | 85–89% mules · ≈₹1.7 cr per 9,082 accounts |
- Random baseline (prevalence) PR-AUC = 0.0089 → ~100× lift.
- Accuracy is **not** reported (useless at 0.89% prevalence).

## Intended use
- Decision-support: rank accounts for analyst review; generate explainable alerts.
- **Not** an autonomous block/freeze authority — a human analyst acts on the alert.

## Limitations
- Account-level **snapshot**, not a transaction stream → on-demand scoring, not stream processing.
- Anonymized features → semantic labels in narratives are **inferred**, not bank-confirmed.
- 81 positives → estimates have real variance; CI is a lower bound (overlapping CV folds).
- Near-label block (F3895–F3923) pre/post-event status needs bank confirmation.
- ~33% of mules are "hard" (low scores) — recall ceiling at high thresholds.

## Ethical / operational
- False positives freeze legitimate customers → use the ₹-cost threshold (`src/insights.py`) to balance harm.
- Re-audit for leakage and re-validate before any production deployment; monitor for drift.
