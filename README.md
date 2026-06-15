# 🛡️ SENTINEL — Suspicious / Mule Account Risk Engine

**CyberShield Hackathon 2026 · Bank of India × IIT Hyderabad · Problem Statement 2 (AI/ML Classification of Suspicious Mule Accounts)**

SENTINEL is the **leakage-proof detection & containment core** of a mule-account engine. It turns 3,924 raw, anonymized, half-empty account features into a **calibrated 0–100 risk score** with **on-demand (real-time-grade, ~35 ms) scoring**, attaches a **plain-English reason to every alert**, recommends a **containment action** (monitor / hold / escalate), and **detects & removes the data leakage that fakes a perfect score** — so we report a number we can defend.

*On "real-time": scoring is **on-demand at ~33 ms p95** (real-time-grade latency) over an account **snapshot** — not a transaction stream. We say "on-demand" consistently; stream processing is Phase-2.*

> **Honest headline: PR-AUC 0.81–0.89, ROC-AUC ~0.98** (repeated 5×2 CV; ~91–99× the 0.0089 random baseline) — best estimate **0.885 ± 0.055**, leak-paranoid floor ~0.81.
>
> **Why a range, not 1.0?** This dataset is **riddled with target leakage**. A naive XGBoost scores **PR-AUC 1.000** — impossible for real fraud detection; it's reading the answer. Our automated **Data Integrity Auditor** found and removed `F3912` (a 96%-aligned label proxy) **plus ~584 non-monotonic bucket/range leaks** (e.g. `F2230`, a month-stamp that is 100% fraud for 3 of 4 values). We report the defensible number, with the audit to prove it.
>
> **Is the honest number itself leakage-inflated?** No — proven. Re-detecting the leak blocklist **strictly inside each CV training fold** (never touching validation labels) gives **PR-AUC 0.907 ± 0.037**, statistically indistinguishable from the 0.885 headline (`src/infold_leak_check.py`). The blocklist being fit on full-data labels does **not** inflate the result.

---

## The data (honestly)

| | |
|---|---|
| Accounts | 9,082 |
| Features | 3,924 anonymized (`F1`..`F3923`), target `F3924` |
| **Mules (positives)** | **81 (0.89%)** — severe imbalance |
| Missing cells | 27.6% |
| Columns dropped | 281 constant + 267 near-empty + 135 hyper-sparse + **1 label-proxy + ~584 bucket/range leaks** |
| Modeling matrix | 9,082 × 2,965 (incl. 175 deduped missingness flags) |

## Results (measured, leak-removed, repeated 5×2 CV)

| Model | PR-AUC | ROC-AUC | CV Brier |
|---|---|---|---|
| **LightGBM (tuned, deployed)** | **0.885 ± 0.055** | 0.979 | 0.0022 |
| RandomForest | 0.771 ± 0.065 | 0.977 | 0.0043 |
| LogReg (L2) | 0.404 ± 0.059 | 0.936 | 0.0103 |

- **Top-50 watchlist caught all 50 true mules** (out-of-fold validation) — the 50 highest-risk *ranked* accounts were all real mules, zero false positives in a 0.89%-mule population. A validation result on the provided data, not a universal guarantee. → `outputs/top_suspicious_accounts.csv`
- **Mule-ring prototype:** a behavioral-similarity graph groups **67 of 81 mules into 5 candidate rings** (largest = 50 near-identical accounts). Candidate Ring #1 is **~30× tighter than a random legit group** (similarity 0.43 vs 0.01) and **stable under feature subsampling** (Jaccard 0.82) — a real, validated proxy, not a coincidence. Case study: demo account **#9003 is a central node in candidate Ring #1** (~₹1.25 cr *potential* exposure, configurable assumption) → investigate the ring as a batch. Confirmation needs bank link data (Phase-2); `src/mule_network.py`.
- **Business impact:** ≈ **₹1.7 crore saved** per 9,082-account population at **85–89% mules caught** (analyst-capacity-bound).
- **Calibration:** CV Brier 0.0022. **Latency:** score + SHAP ≈ 35 ms (p95). **Recall:** ≈72% of mules (58/81) score ≥70/100.

Full leaderboard, sensitivity sweep, threshold dial, and every caveat: **[`docs/RESULTS.md`](docs/RESULTS.md)**. Model card: **[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)**.

## Repository structure

```
.
├── app.py                          Streamlit demo dashboard (deploy entry point)
├── requirements.txt · packages.txt Pinned deps (+ libgomp1 for LightGBM)
├── docs/                           submission PDF + all write-ups
│   ├── SOLUTION_APPROACH_PS2.pdf   hackathon submission deck
│   ├── RESULTS.md · MODEL_CARD.md · PRD_PS2_Mule_Account_Detection.md
│   └── DATA_INTEGRITY_AUDIT.md · DEPLOY.md
├── scripts/                        deliverable generators
│   ├── make_solution_pdf.py        builds docs/SOLUTION_APPROACH_PS2.pdf
│   └── make_colab_notebook.py      builds colab/ notebook
├── src/
│   ├── preprocess.py               hygiene, encode, missingness flags, auto leak removal
│   ├── data_integrity_auditor.py   4-signature leakage audit (the differentiator)
│   ├── model_config.py             single source of truth for the tuned model
│   ├── honest_eval.py              leak-removed leaderboard + CV Brier + bands + latency
│   ├── finalize.py                 train + calibrate winner → deployment artifacts
│   ├── tune.py                     hyperparameter search + ensemble (model-push)
│   ├── leak_sensitivity.py         leakage-sensitivity sweep
│   ├── definitive_cv.py            headline CV + near-label ablation floor
│   ├── insights.py                 ₹-cost curve + error analysis + mule typology
│   ├── results_pack.py             figures + predictions.csv + watchlist
│   ├── sentinel.py                 real-time scoring + SHAP + investigation reports
│   ├── feature_meanings.py         honest, hedged feature semantics
│   ├── api.py                      FastAPI /score + /report service
│   ├── mule_network.py             mule-RING prototype (behavioral-similarity graph)
│   ├── arch_diagram.py · leak_story_fig.py   deck visuals
│   └── demo_shots.py               live product screenshots (for the deck)
├── artifacts/                      trained model + stats + metrics (deployable, ~13 MB)
├── figures/                        generated plots + product screenshots
├── outputs/                        predictions.csv + top_suspicious_accounts.csv
├── tests/                          pytest smoke tests (leakage blocked, sane metrics)
└── colab/                          self-contained Colab notebook
```

## Quickstart

```bash
pip install -r requirements.txt
python src/preprocess.py              # cache clean, leak-removed matrix  (needs DataSet.csv)
python src/data_integrity_auditor.py  # leakage audit  → docs/DATA_INTEGRITY_AUDIT.md
python src/honest_eval.py             # honest leaderboard + bands + latency
python src/finalize.py LightGBM       # train + calibrate winner → artifacts/
python src/insights.py                # ₹-cost curve, error analysis, typology
python src/results_pack.py            # figures/ + outputs/ (predictions + watchlist)
python src/mule_network.py            # mule-ring prototype → figures/11_mule_network.png
python -m pytest tests/ -q            # smoke tests
streamlit run app.py                  # live demo dashboard
uvicorn src.api:app                   # real-time scoring API
```
> Place the provided `DataSet.csv` in the repo root before running the pipeline (it is git-ignored). The Streamlit demo runs on the committed `artifacts/` alone — no raw data needed.

### Data handling (why the full dataset isn't in this repo)
The bank-provided dataset is **not redistributed** in this public repo (the hackathon grants no redistribution right — it's "data provided in this portal" for building the solution). So:
- The demo ships a small **`samples/sample_accounts.csv`** (all known mules + a legit sample) so every flagged/watchlist account is verifiable out-of-box.
- The dashboard has a **CSV uploader** — drop in the raw `DataSet.csv` (or a cleaned export) and the *entire* dashboard re-scores on it live (`src/preprocess.py::prepare_frame` re-applies the exact, saved, target-free cleaning). Uploaded data stays in-session; it is not stored.
- The full 9,082-account matrix (`artifacts/all_accounts.parquet`) is **git-ignored** — generate it locally and upload it in the app to demo the whole population without publishing it.

## Deploy

The demo deploys free on **Streamlit Community Cloud** in ~10 min — see **[`docs/DEPLOY.md`](docs/DEPLOY.md)**. Live demo: `[demo URL]`.

## Honest limitations (stated, not hidden)

- Account-level **snapshot**, not a transaction stream → "real-time" = on-demand scoring, not stream processing.
- Anonymized features → narrative labels are **inferred**, not bank-confirmed.
- 81 positives → estimates carry real variance; we report a range + confidence interval, not a single point.
- Decision-support for analysts, **not** an autonomous freeze authority.

## Author

**Team Probe Rockerz** (individual participation) — Vibhu Krishna S, SRM Easwari Engineering College.
