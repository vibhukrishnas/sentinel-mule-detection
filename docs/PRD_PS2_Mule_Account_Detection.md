# PRD — PS2: AI/ML Classification of Suspicious Mule Accounts
**BOI Hackathon · Bank of India · v1.0 · 2026-06-09**

> Product codename: **SENTINEL** — a real-time mule-account risk engine with explainable, investigation-ready alerts.

---

## 0. TL;DR (for judges, in 30 seconds)
We turn 3,924 raw, anonymized, half-empty account features into a **single calibrated risk score (0–100)** per account, in **<50 ms**, with a **plain-English "why" attached to every alert** and an **auto-generated investigator report**. We do not chase a fancy deep model on 81 fraud cases — that would overfit and lose. We win on **fundamentals executed perfectly**: ruthless feature hygiene, leakage-proof validation, the right metric (PR-AUC, not accuracy), calibrated probabilities, and explainability a fraud analyst can act on *today*.

---

## 1. Problem & Context
Banks lose money and trust to **mule accounts** — accounts that receive, move, and launder fraudulent funds. Rule-based monitoring is reactive and brittle; it misses evolving patterns and drowns analysts in false positives.

PS2 asks us to: **(a)** engineer features from the provided dataset, **(b)** build a classifier that separates mule/suspicious accounts from legitimate ones, and **(c)** deliver anomaly detection, predictive risk scoring, and intelligent alerting.

## 2. The Brutal Data Reality (constraints that dictate the design)
*Verified by direct profiling of `DataSet.csv`, not assumed.*

| Fact | Value | Consequence |
|---|---|---|
| Rows (accounts) | 9,082 | Small. Statistical fragility is real. |
| Features | 3,924 (`F1`..`F3923`), target `F3924` | High-dim, anonymized. Feature selection mandatory. |
| **Positives (mule)** | **81 (0.89%)** | **The dominant constraint.** Severe imbalance, tiny positive set → overfitting & metric-instability are the #1 risk. |
| NA cells | 27.6% overall | Need NaN-native models + disciplined imputation. |
| Dead columns | 359 constant + 527 near-constant | Drop them; they only add variance/noise. |
| >50% empty columns | 1,138 | Flag missingness as signal, then prune. |
| Categoricals | 8 (e.g. `F3891`=occupation, `F3889`=tenure code, `F3894`≈age) | Encode carefully (no target leakage). |
| Data shape | **Account snapshot, not a transaction stream** | "Real-time" = on-demand low-latency scoring, **not** stream processing. We say so honestly. |

**Known signal exists** (bank-hinted features): `F2678` (class-mean 351 vs 11), `F3836`, `F2956` (133 vs 58), `F670` (2.6× more common in mules). The problem is winnable — but only with discipline.

## 3. Goals & Non-Goals
**Goals**
1. A reproducible, leakage-proof classifier with **honest cross-validated metrics**.
2. **Calibrated** risk probability → 0–100 score with severity bands (Low/Medium/High/Critical).
3. **<50 ms** single-account inference (real-time-ready).
4. **Every alert explained**: top contributing factors in plain English (SHAP-driven).
5. **Auto-generated investigation report** per flagged account (the "wow", and directly maps to the PS theme of "intelligent threat summarization").
6. A **demo app** an analyst/judge can poke live.

**Non-Goals (stated to avoid overclaiming)**
- No transaction-stream / Kafka pipeline — the data doesn't support it; we'd be faking it.
- No graph/mule-network detection — no link data provided. (Listed as a roadmap item, not a claim.)
- No deep neural net — 81 positives makes DL a liability, not an asset.

## 4. Success Metrics (the RIGHT ones)
Accuracy is **banned** as a headline metric (99.1% by predicting "all clean").

| Metric | Why it matters | Target |
|---|---|---|
| **PR-AUC (Average Precision)** | Correct metric for 0.89% prevalence | Primary leaderboard metric; maximize |
| **ROC-AUC** | Threshold-independent ranking quality | ≥ 0.90 (stretch ≥ 0.95) |
| **Recall @ fixed precision** | "Of mules, how many caught while keeping analysts sane?" | Report recall @ precision=0.5 and the full PR curve |
| **Brier score / calibration** | Risk score must mean what it says | Calibrated (isotonic/sigmoid) |
| **Alert volume** | Analyst workload | Report mules-caught per N alerts |

All metrics reported with **stratified 5-fold CV + repeats** and spread (mean ± std), because with 81 positives a single split lies.

## 5. Users & Jobs-To-Be-Done
- **Fraud analyst** — "Tell me which accounts to investigate first, and *why*, so I act in minutes not hours."
- **Risk/compliance officer** — "Show me the score distribution and the cost trade-off at each threshold."
- **Engineer** — "Give me a clean API and a reproducible model artifact."

## 6. Solution Architecture
```
DataSet.csv ──▶ [1] Feature Hygiene ──▶ [2] Feature Engineering ──▶ [3] Model (HistGBM/LGBM, calibrated)
                  drop dead cols          missingness flags,            stratified CV, class weights
                  type coercion           categorical encoding,         ↓
                  NA strategy             interaction of hint feats   [4] Risk Score 0–100 + bands
                                                                         ↓
                                              [6] Explainability (SHAP) ◀─┤
                                                                         ↓
                                              [5] Alert + Auto Investigation Report (GenAI/template)
                                                                         ↓
                                              [7] Real-time API + Demo dashboard
```

## 7. Feature Engineering Strategy (fundamentals, mastered)
1. **Hygiene:** drop 359 constant + obviously dead cols; coerce types; keep NaN for NaN-native model.
2. **Missingness as signal:** for high-NA columns, add `is_missing` indicators — *whether* a field is populated is itself predictive for mules.
3. **Categoricals:** ordinal/target-encoding **inside CV folds only** (no leakage); `F3891` occupation, `F3889` tenure, etc.
4. **Hint-feature focus:** the 18 bank-known features get first-class treatment + simple interactions/ratios.
5. **Supervised selection inside CV:** model-based importance / mutual information to prune 3,924 → a defensible top-K, preventing overfit and speeding inference.

## 8. Modeling Strategy
- **Workhorse:** `HistGradientBoostingClassifier` (sklearn) — **handles NaN natively** (perfect for 27.6% NA), strong on tabular, no install needed. Plus **LightGBM/XGBoost** for the tournament.
- **Baselines for honesty:** regularized Logistic Regression + RandomForest (so we prove the gradient-boosted model actually earns its complexity).
- **Imbalance:** class weights / `scale_pos_weight`, threshold tuning on the PR curve. **SMOTE deliberately not used** — synthesising neighbours from 81 positives in ~3,000 dimensions manufactures noise, not signal (curse of dimensionality); class weighting is the correct lever here.
- **Calibration:** isotonic/Platt on held-out folds so the 0–100 score is trustworthy.
- **Validation:** Repeated Stratified K-Fold. One number is a lie; we report the distribution.

## 9. Real-Time Scoring, Alerting & Explainability
- **Score:** calibrated P(mule) → `round(100·p)`; bands: 0–39 Low, 40–69 Medium, 70–89 High, 90–100 Critical.
- **Latency:** model + SHAP on one row is single-digit ms; API target <50 ms.
- **Alert:** triggered above a configurable threshold; carries score, band, and top-5 contributing features.
- **Explainability:** SHAP values per prediction → human-readable reasons, mapping known features to meaning (occupation, tenure, age, balance-velocity proxies).

## 10. The Unique / "Feel It" Features (simple, clean, high-impact)
1. **Data Integrity Auditor (rigour differentiator).** An automated scanner that scores every feature on 4 leak signatures (label-proxy, exact-value bucket, continuous range, univariate-AUC) and emits `DATA_INTEGRITY_AUDIT.md`. The signatures are standard heuristics — the edge is that we *systematically applied* them and caught `F3912` + ~584 leaks that fake a perfect score, then report the defensible number instead of a 0.99 we can't explain.
2. **₹-cost decision engine.** Translates the threshold into money (`src/insights.py`, incl. false-positive harm): **85–89% of mules caught → ≈₹1.7 crore saved** per 9,082-account population. Net savings is flat across thresholds, so the operating point is analyst-capacity-bound — banks decide in rupees and staffing, not PR-AUC.
3. **Reason-coded alerts** — not just a score, but *"Flagged: account tenure < 31 days + occupation 'student' + abnormal F2678 vs peers + 6 blank profile fields."* Analysts act on what they understand.
4. **Auto investigation report** — one-click, template-driven NL summary per account (risk, evidence, recommended action, calibration caveats) — PS2's "intelligent alert generation", no LLM dependency.
5. **Mule typology + peer-anomaly framing** — clustering surfaces one dominant mule archetype (78/81) plus a small hard-to-detect tail (honest: limited sub-type diversity), and shows z-score deviation vs cohort, making anomaly detection visible.
6. **Live demo + real-time API** — Streamlit dashboard (`app.py`) and FastAPI `/score` (`src/api.py`), ~44 ms per account. The judge *feels* it work.

## 11. Risks & Mitigations (adversarial — how a stronger team beats us)
| Risk | Likelihood | Mitigation |
|---|---|---|
| **Overfitting on 81 positives** | High | Repeated stratified CV, regularization, feature pruning, report variance honestly |
| **Data leakage** (target-encoding, scaling outside CV, ID-like cols) | High | All preprocessing inside CV folds; adversarial leakage review; audit suspiciously perfect features |
| **Metric theater** (showing accuracy) | Certain to tempt | PR-AUC primary; accuracy explicitly de-emphasized |
| **Uncalibrated scores** | Medium | Calibration layer + Brier/reliability check |
| **Anonymized features = weak narrative** | Medium | Lean on the 18 known features for the human story |
| **Demo fails live** | Medium | Pre-baked sample accounts + cached artifacts; no live training in demo |

## 12. Deliverables & Milestones
- **M1 — Pipeline + baseline metrics** (this session): hygiene, CV harness, real numbers.
- **M2 — Model tournament + calibration**: pick winner on PR-AUC, calibrate.
- **M3 — Explainability + alert/report layer**: SHAP, reason codes, report generator.
- **M4 — Demo app + reproducibility**: scoring API/dashboard, saved artifacts, README.
- **M5 — Results writeup + judge-risk assessment**: honest metrics + competitive analysis.

## 13. Definition of Done
Reproducible (`python train.py` regenerates metrics), measurable (CV PR-AUC/ROC-AUC with spread), explainable (per-account reasons), demoable (live scoring), and **honest** (limitations stated, not hidden).
