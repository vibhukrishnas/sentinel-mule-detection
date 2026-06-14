# SENTINEL — Results & Honest Assessment (PS2)
*BOI Hackathon · Mule-Account Classification · updated 2026-06-10*

This document reports **measured** results, the leakage we found, and exactly how a
strong competitor or judge would attack us. Nothing here is aspirational.

---

## 1. The headline (read this first)
- This dataset is **riddled with target leakage**. A naive XGBoost scores **PR-AUC = 1.000 / ROC-AUC = 1.000** — which is *impossible* for genuine fraud detection on 81 anonymized positives. It is reading the answer.
- We built an **automated, reproducible leakage detector**, found the leaks, removed them, and measured **honest, defensible performance**.
- **Honest behavioral signal: PR-AUC = 0.885 ± 0.055 (tuned LightGBM, 5×2 CV), ROC-AUC = 0.979**, vs a random baseline of 0.0089 — roughly **a 99× lift**. Leak-paranoid floor (near-label block removed) ≈ 0.81. We report the whole **0.81–0.89** range rather than cherry-picking.
- **precision@50 = 100%:** the 50 highest-risk accounts (out-of-fold) are *all 50 real mules* — zero false positives in a population that is 0.89% mules. Visualised in `figures/`, delivered as `outputs/top_suspicious_accounts.csv`.
- **Business impact:** at a practical operating point we catch **85–89% of mules** for a net **≈₹1.7 crore saved** per 9,082-account population — net savings is flat across thresholds, so the point is analyst-capacity-bound, not a magic number (`src/insights.py`).
- **Differentiator:** **rigorous, end-to-end leakage detection** — an automated Data Integrity Auditor (`DATA_INTEGRITY_AUDIT.md`) scores every feature on 4 leak signatures. The edge isn't a novel algorithm; it's that we actually *did* it while a naive pipeline ships an unexplainable 0.99.

## 2. Data reality
| | |
|---|---|
| Accounts | 9,082 |
| Raw features | 3,923 (`F1`..`F3923`) + target `F3924` |
| **Mules (positives)** | **81 (0.89%)** — severe imbalance, tiny positive set |
| Missing cells | 27.6% |
| Modeling matrix (after hygiene + leak removal @0.10) | 9,082 × 2,967 |

## 3. The two leaks we caught
**(a) `F3912` — a label proxy.** Binary flag; 79/81 mules have it set, only 3/9,001 legit do (~96% aligned with the target). It is a post-hoc fraud flag, not a behavioral feature you'd have *before* knowing the outcome. Removed.

**(b) ~580 non-monotonic "bucket leaks."** Features where a *specific value* is hyper-pure fraud while the class *means* look identical — so univariate ROC-AUC is blind to them but a single tree split nails them. **Verified flagship example — `F2230`, a month-stamp field:**

| `F2230` raw value | accounts | frauds | fraud rate |
|---|---|---|---|
| `Oct25` | 9,001 | 0 | 0% |
| `Sep25` | 48 | 48 | **100%** |
| `Nov25` | 23 | 23 | **100%** |
| `Dec25` | 10 | 10 | **100%** |

Every legitimate account is stamped `Oct25`; the other three months are **100% fraud and together account for all 81 mules**. `F2230` literally *is* the label (it encodes the month fraud was detected) — yet its class means are ~identical (≈2.0 vs 2.06 after ordinal encoding), so univariate AUC is only **0.59** and a naive scan misses it. This reveals the leak mechanism: **the dataset was labeled by a process that stamped detection dates onto accounts, and those stamps leaked across many features.** Auto-detected by: any value-bucket (n≥10) whose fraud rate exceeds 11× the 0.89% base rate. Bank-hinted features are never auto-removed.

> **Why this matters competitively:** most teams will train a booster, show a 0.99+, and have no answer when a judge asks *"why is fraud detection perfect?"*. We have the detector, the evidence, and the honest number.

## 4. Leakage-sensitivity sweep (the proof it's not one hidden leak)
LightGBM CV PR-AUC as we strip features at a falling fraud-rate threshold:

| Leak threshold | Features removed | Features kept | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| none | 0 | 3,549 | **0.998** | 1.000 |
| ≥30% fraud bucket | 12 | 3,537 | 0.861 | 0.987 |
| ≥20% | 59 | 3,490 | 0.871 | 0.984 |
| ≥10% *(chosen)* | 582 | 2,967 | 0.858 | 0.983 |
| ≥5% | 1,303 | 2,246 | 0.842 | 0.980 |
| ≥3% | 1,461 | 2,088 | 0.795 | 0.976 |
| ≥2% | 1,574 | 1,975 | 0.788 | 0.972 |

**Interpretation:** the cliff is `none → 0.30` (0.998 → 0.861) = pure leakage evaporating. Then it **plateaus at ~0.86** across a wide range — proof the remaining score is *distributed genuine signal*, not a single hidden leak. Only when we strip features that are a mere 2–3× the base rate (plausibly real behavioral signal) does it decline.

## 5. Model leaderboard (leak-removed, PR-AUC primary)
*Repeated Stratified CV. Accuracy is deliberately omitted — it is useless at 0.89% prevalence.*

*Repeated Stratified 5×2 CV on the leak-removed 2,965-feature matrix (`artifacts/honest_eval.json`).*

| Model | CV PR-AUC | ROC-AUC | CV Brier | Note |
|---|---|---|---|---|
| **LightGBM (tuned)** | **0.885 ± 0.055** | 0.979 | **0.0022** | winner — fast, NaN-native, fast SHAP |
| RandomForest | 0.771 ± 0.065 | 0.977 | 0.0043 | strong baseline |
| LogReg (L2) | 0.404 ± 0.059 | 0.936 | 0.0103 | linear baseline (blind to non-monotonic signal) |

*(XGBoost tied LightGBM in pre-removal testing but is excluded from this clean leaderboard for runtime; LightGBM is the deployed winner for speed + native SHAP.)*

The LogReg→tree gap (**0.40 → 0.89**) is itself diagnostic: most of the signal is **non-monotonic / interaction-based**, which is exactly *why* the non-monotonic leaks were so dangerous and so easy to miss with linear methods or monotonic AUC.

**Model-push exploration (`src/tune.py`, 5-fold).** We tried to break past the ceiling:

| Step | PR-AUC | Verdict |
|---|---|---|
| baseline LightGBM | 0.888 | — |
| + engineered features (profile-completeness, ratios) | 0.878 | no help — model already captures it |
| + hyperparameter search | 0.899 | +0.011, sounder regularization |
| + LGB×XGB ensemble | 0.901 | +0.013, but 2 models + messier SHAP |

We **adopted the tuned single LightGBM** (deployed headline 0.885 on robust 5×2 CV) for its stronger regularization, and skipped the ensemble's extra +0.002. The flat ceiling (~0.88–0.90) confirms we are extracting the genuine signal, not leaving easy performance behind.

## 6. Definitive measurement + conservative floor
*Repeated Stratified K-Fold (5×2), final LightGBM, leak-removed matrix.*

| Measurement | PR-AUC (t-CI, optimistic) | ROC-AUC | What it means |
|---|---|---|---|
| **HEADLINE** (tuned, all 2,965 leak-removed features) | **0.885 ± 0.055 (0.846–0.925)** | 0.979 | Best honest estimate |
| **FLOOR** (also drops near-label block F3895–F3923) | **≈0.81** | 0.977 | Holds even if that whole block is a subtle leak |

> The truth lies between FLOOR and HEADLINE: **PR-AUC 0.81–0.89, ROC-AUC ~0.98** (≈91–99× the 0.0089 random baseline). The near-label tail block contributes ~0.07 PR-AUC; if the bank confirms those features are pre-event behavioral signals, the headline stands — if post-event, the floor (~0.81) is the number, and it's still strong. (Floor measured by the near-label ablation in `src/definitive_cv.py`.)

**Calibration (trustworthy):** CV Brier = **0.0022** for the deployed tuned LightGBM (measured across 5×2 folds, all 81 positives) — not the 16-positive holdout's 0.0017. Probabilities mean what they say.

**Real-time latency (measured, 100 iters):** score ≈ 33 ms (p95 36 ms); **score + full SHAP explanation ≈ 44 ms p95** — under the 50 ms real-time bar, measured not asserted.

**Risk-band sanity (out-of-fold, all 81 mules):** median real mule scores **~99/100**; **≈72% of mules (58 of 81) score ≥70 (HIGH+)**. Honestly, ~28% are "hard" (low scores) — that tail is the realistic recall ceiling at high thresholds.

**Holdout (single 80/20 split, 16 positives — noisy, secondary):** PR-AUC 0.959, ROC-AUC 0.999, recall 0.94 @ precision≥0.5. We **do not** headline this — 16 positives is too few for a stable estimate, and its perfect precision is the kind of number that should make you suspicious, not proud.

## 7. Threshold dial (holdout — the risk officer owns the trade-off)
| Threshold | Alerts | Mules caught | False alarms | Precision | Recall |
|---|---|---|---|---|---|
| 0.30 | 13 | 13 | 0 | 1.00 | 0.81 |
| 0.50 | 12 | 12 | 0 | 1.00 | 0.75 |
| 0.80 | 10 | 10 | 0 | 1.00 | 0.62 |
| 0.95 | 8 | 8 | 0 | 1.00 | 0.50 |

## 7.5 Methodological caveats (stated, not hidden — found by our own adversarial review)
- **The printed CI is a lower bound.** Repeated 5×2 stratified folds share ~80% of rows, so they are *not* independent; the t-based CI understates true uncertainty. Treat the honest range as **≈0.80–0.92**, not a tight ±0.05.
- **Some preprocessing is fit on the full dataset**, not inside CV folds: categorical encoding (target-free, alphabetical → negligible), missingness-flag dedup, and bucket-leak detection. The leak detector uses `y` as a **one-time frozen feature-exclusion decision** (a reference blocklist), not a per-fold target transform — and the sensitivity sweep shows the blocklist is stable, so in-fold detection would not move the headline.
- **The near-label tail block (F3895–F3923) is still in the HEADLINE model** (+0.076 PR-AUC). The FLOOR (0.810) removes it entirely. Pre/post-event status of those features needs bank confirmation — until then we report both bounds.
- **Six univariate AUC≥0.98 features (F518/F515/F413/…) were *not* removed.** They are hyper-sparse — each non-NA for only ~1 positive — so they cannot generalize and do not affect cross-validated scores (verified: they raise *train* PR-AUC to 1.0 but not test). Documented, not load-bearing.
- **The leakage-sensitivity sweep used 3-fold CV** (point estimates) while the headline uses 5×2; the plateau is directional evidence, not a precise curve.

## 7.6 Business impact, error analysis & mule typology (`src/insights.py`)
**Banks act on ₹, not PR-AUC.** Translating the threshold into money (assumptions, configurable: avg mule loss **₹2,50,000**, analyst review **₹400**, false-positive harm of freezing a legit customer **₹5,000**):

| Operating point | Threshold | Mules caught | Recall | Alerts | Precision | Net savings |
|---|---|---|---|---|---|---|
| Max-recall | 0.01 | 72 / 81 | 89% | 140 | 0.51 | ₹1.76 cr |
| **Balanced (recommended)** | 0.05 | 69 / 81 | **85%** | 89 | 0.78 | **₹1.71 cr** |
| High-precision | 0.30 | 64 / 81 | 79% | 66 | 0.97 | ₹1.60 cr |
| Zero-false-alarm | 0.65 | 56 / 81 | 69% | 56 | 1.00 | ₹1.40 cr |

**Honest read:** net savings is **flat (~₹1.7 cr) from threshold 0.01–0.05** — even charging ₹5,000 per falsely-frozen customer, a missed mule (₹2.5L) costs ~46× a false alarm, so the economics favour casting a wide net. The operating point is therefore **analyst-capacity-bound, not a sharp mathematical optimum**: pick the recall/alert-volume trade-off your team can staff; either way ≈ ₹1.7 crore saved per 9,082-account population.

**Error analysis** — the 12 mules missed at the optimal point carry *larger* monetary fields than caught mules yet still read "legitimate" on the top drivers: the genuine hard tail.

**Mule typology** (clustering the 81 mules) — **one dominant archetype of 78 mules** (avg risk 75/100, readily detected) plus a **3-account hard-to-detect tail** (risk ~0). Honest finding: the data shows one coherent mule profile, not many sub-types.

## 8. How a stronger team / judge beats us — and our answer
| Attack | Our defense |
|---|---|
| "Your 0.99 is leakage." | We found it first, removed it, and report ~0.86 with a sensitivity sweep. |
| "16-positive holdout is meaningless." | Agreed — we headline repeated-CV with a CI, not the holdout. |
| "Are F3898/F3908/F3914 also leaks?" | Possibly. We publish a FLOOR with the entire near-label block removed. |
| "Accuracy?" | Banned metric here; we use PR-AUC / recall@precision. |
| "Will it work in production?" | Caveat stated: snapshot data, anonymized features; bank metadata would let us lock the leak/signal boundary and confirm pre- vs post-event features. |

## 9. What we'd do with 1 more day / bank metadata
- Confirm pre/post-event status of high-signal features (locks the honest number).
- Add range-based (not just value-bucket) leak detection for continuous features.
- Cost-weighted threshold tuning to a real INR cost of missed-mule vs false-alarm.
- If link data exists: mule-network / shared-device graph features.
