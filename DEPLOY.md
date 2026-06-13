# Deploying the SENTINEL demo (free, ~10 minutes)

The Streamlit dashboard (`app.py`) needs only the small artifacts in `artifacts/`,
`outputs/`, `figures/` — **not** the 112 MB `DataSet.csv` or 18 MB `X_clean.parquet`
(both git-ignored). Total deploy footprint ≈ 13 MB.

---

## Option A — Streamlit Community Cloud (recommended, free, persistent URL)

**1. Push this project to a PUBLIC GitHub repo.** A local git repo is already
initialised and committed for you. Create an empty repo on github.com (e.g.
`sentinel-mule-detection`), then:

```bash
git remote add origin https://github.com/<your-username>/sentinel-mule-detection.git
git branch -M main
git push -u origin main
```

**2. Deploy.** Go to https://share.streamlit.io → sign in with GitHub → **New app** →
   - Repository: `<your-username>/sentinel-mule-detection`
   - Branch: `main`
   - Main file path: `app.py`
   - Click **Deploy**. First build takes ~3–5 min (installs `requirements.txt` +
     `libgomp1` from `packages.txt` for LightGBM).

**3. Get your URL** — `https://<something>.streamlit.app`. Paste it into the cover +
   §8 of `SOLUTION_APPROACH_PS2.pdf` (the `[demo URL]` placeholders) and re-export.

---

## Option B — Hugging Face Spaces (alternative, free)
1. Create a Space → SDK = **Streamlit**.
2. Upload `app.py`, `requirements.txt`, `packages.txt`, and the `src/`, `artifacts/`,
   `outputs/`, `figures/` folders (or `git push` to the Space repo).
3. It auto-builds and serves a public URL.

---

## Option C — Run locally (for an offline demo on your laptop)
```bash
pip install -r requirements.txt
streamlit run app.py     # opens http://localhost:8501
```

---

## Notes / gotchas
- **Versions are pinned** in `requirements.txt` so the saved model unpickles correctly
  (scikit-learn 1.7.1, lightgbm 4.6.0) — do not loosen these or the model load may fail.
- The FastAPI scoring service (`src/api.py`) deploys separately on Render/Railway
  (`uvicorn src.api:app`); for the hackathon, the Streamlit app is the demo to show.
- If a cloud build OOMs (rare), it's the SHAP explainer — it lazy-loads on first
  "explain" click, so the dashboard still starts.
