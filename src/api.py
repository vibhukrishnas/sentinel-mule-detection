"""
SENTINEL — FastAPI on-demand scoring service.  ⚠️ INTERNAL / DEV ONLY.

  POST /score    {features: {...}}  -> risk score, band, probability, top drivers, action
  POST /report   {features: {...}}  -> plain-text investigation report
  GET  /health                      -> liveness + model metadata

Run:  uvicorn src.api:app --reload    (or: python -m uvicorn src.api:app)
Accepts a PARTIAL feature dict — unknown/missing features are treated as blank (the
boosted model + SHAP handle NaN natively). Single-account latency is ~40 ms (see RESULTS).

NOT a public endpoint: there is no auth or rate limiting, and /score runs SHAP per call
(CPU-amplification vector). Gate behind auth + a rate limiter before any real deployment.
A basic per-request feature-count cap (MAX_FEATURES) is enforced as a first guard.
"""
from __future__ import annotations
from pathlib import Path
import os, sys, time
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import io, json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Header, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentinel import SentinelEngine, ART, band_for
from preprocess import prepare_frame

MAX_ROWS = 20000                # cap uploaded rows (memory guard for the shared demo)

MAX_FEATURES = 5000             # reject absurd payloads (the model uses ~2,965 features)
API_KEY = os.getenv("SENTINEL_API_KEY")   # if set, callers must send X-API-Key: <key>
RATE_LIMIT, RATE_WINDOW = 60, 60          # max 60 requests / 60 s per client IP
app = FastAPI(title="SENTINEL — Mule Account Risk Engine", version="1.1")
# CORS so the React dashboard (different origin/port) can call this API in dev + deploy.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
ROOT = ART.parent
_engine: SentinelEngine | None = None
_hits: dict[str, list[float]] = defaultdict(list)
_data: dict | None = None        # cached sample accounts + scores for the dashboard


def _load_json(name: str):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else None


def _guard(features: dict):
    if not isinstance(features, dict) or len(features) > MAX_FEATURES:
        raise HTTPException(status_code=413, detail=f"features must be a dict of ≤ {MAX_FEATURES} keys")


def _authz(request: Request, x_api_key: str | None):
    """Per-IP rate limit + optional API-key auth (enabled when SENTINEL_API_KEY is set)."""
    now = time.time()
    ip = request.client.host if request.client else "?"
    _hits[ip] = [t for t in _hits[ip] if now - t < RATE_WINDOW]
    if len(_hits[ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="rate limit exceeded; slow down")
    _hits[ip].append(now)
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def engine() -> SentinelEngine:
    global _engine
    if _engine is None:
        _engine = SentinelEngine()
    return _engine


class Account(BaseModel):
    account_id: str | int = "?"
    features: dict  # {"F115": 0.5, "F3891": 2, ...} — partial is fine


@app.get("/health")
def health():
    eng = engine()
    return {"status": "ok", "n_features": len(eng.columns), "model": "LightGBM (calibrated)"}


@app.post("/score")
def score(acct: Account, request: Request, x_api_key: str | None = Header(None)):
    _authz(request, x_api_key)
    _guard(acct.features)
    eng = engine()
    sc = eng.score(acct.features)
    alert = eng.alert(acct.features, threshold=0.5, account_id=acct.account_id)
    return {"account_id": acct.account_id, **sc,
            "confidence_tier": eng.confidence_tier(sc["probability"]),
            "alert": alert is not None,
            "recommended_action": (alert or {}).get("recommended_action"),
            "top_drivers": eng.explain(acct.features, top_k=5)}


@app.post("/report")
def report(acct: Account, request: Request, x_api_key: str | None = Header(None)):
    _authz(request, x_api_key)
    _guard(acct.features)
    return {"account_id": acct.account_id,
            "report": engine().report(acct.features, account_id=acct.account_id)}


# ============================ DASHBOARD READ ENDPOINTS ============================
# Serve the React dashboard: a real model-scored sample population + the validated
# artifacts (leakage story, abstention, decisioning, benchmark, rings, money-flow).

def _score_population(X: pd.DataFrame, y, src: str) -> dict:
    """Score a whole population with the deployed model -> rows + cached frame for the dashboard."""
    global _data
    eng = engine()
    proba = eng.model.predict_proba(X.astype("float32"))[:, 1]
    rows = []
    for i, p in zip(X.index, proba):
        s = int(round(p * 100))
        gt = None
        if y is not None:
            try: gt = "MULE" if int(y.loc[i]) == 1 else "legit"
            except Exception: gt = None
        rows.append({"account_id": int(i), "risk_score": s, "probability": round(float(p), 4),
                     "band": band_for(s), "confidence_tier": eng.confidence_tier(float(p)), "ground_truth": gt})
    _data = {"X": X, "rows": rows, "src": src}
    return _data


def _dashboard_data() -> dict:
    """Active population. Defaults to the committed sample (all 81 mules + legit sample)."""
    global _data
    if _data is not None:
        return _data
    cols = engine().columns
    cat_maps = json.loads((ART / "categorical_maps.json").read_text())
    samp = ROOT / "samples" / "sample_accounts.csv"
    if samp.exists():
        df = pd.read_csv(samp, index_col=0, low_memory=False)
        X, y = prepare_frame(df, cols, cat_maps); src = f"committed sample — {len(X):,} accounts"
    else:
        X = pd.read_parquet(ART / "demo_accounts.parquet").reindex(columns=cols).astype("float32")
        y = pd.read_parquet(ART / "demo_targets.parquet")["target"]; src = "built-in demo — 10 accounts"
    return _score_population(X, y, src)


def _top_features(k=30):
    """Genuine top features for similarity (from the ring artifact, else model gain)."""
    mn = _load_json("mule_network.json") or {}
    feats = [f for f in mn.get("top_features_used", []) if f in engine().columns]
    return feats[:k] if len(feats) >= 8 else list(engine().columns[:k])


@app.get("/summary")
def summary():
    """Headline metrics + the leakage story + abstention + decisioning + benchmark + anomaly."""
    he = _load_json("honest_eval.json") or {}
    win = (he.get("leaderboard") or [{}])[0]
    sens = _load_json("leak_sensitivity.json") or []
    naive = next((r["pr_auc"] for r in sens if r.get("leak_thr", 0) > 1), 0.998)
    honest = win.get("pr_auc", 0.885)
    n_leaks = next((r["n_leaks"] for r in sens if abs(r.get("leak_thr", 9) - 0.10) < 1e-6), None)
    boot = _load_json("bootstrap_ci.json") or {}
    infold = _load_json("infold_leak_check.json") or {}
    unc = _load_json("uncertainty.json") or {}
    rd = _load_json("boi/realtime_decisioning.json") or {}
    anom = _load_json("anomaly_meta.json") or {}
    return {
        "headline": {"pr_auc": honest, "pr_std": win.get("pr_std"), "roc_auc": win.get("roc_auc"),
                     "brier": win.get("brier"), "naive_with_leak": naive,
                     "bootstrap_ci": boot.get("bootstrap_pr_auc_ci95"), "oof_pr_auc": boot.get("oof_pr_auc"),
                     "infold_pr_auc": infold.get("pr_auc_mean"), "bucket_leaks_removed": n_leaks},
        "leakage_sweep": sens,
        "abstention": unc, "decisioning": rd.get("decisioning"), "realtime": rd.get("realtime"),
        "anomaly": anom, "source": _dashboard_data()["src"],
    }


@app.get("/accounts")
def accounts():
    """The model-scored sample population for the queue / picker / watchlist."""
    return {"accounts": _dashboard_data()["rows"], "source": _dashboard_data()["src"]}


@app.get("/account/{account_id}")
def account(account_id: int):
    """Live model score + SHAP drivers + investigation report + data-quality for one account."""
    d = _dashboard_data(); X = d["X"]
    if account_id not in X.index:
        raise HTTPException(status_code=404, detail=f"account {account_id} not in sample population")
    eng = engine(); row = X.loc[account_id]
    sc = eng.score(row); drivers = eng.explain(row, top_k=6)
    n_blank = int(pd.isna(row.reindex(eng.columns)).sum()); n_feat = len(eng.columns)
    completeness = 1 - n_blank / max(n_feat, 1)
    trust = int(round(100 * (0.6 * abs(sc["probability"] - 0.5) * 2 + 0.4 * completeness)))
    meta = next((r for r in d["rows"] if r["account_id"] == account_id), {})
    return {"account_id": account_id, **sc, "confidence_tier": eng.confidence_tier(sc["probability"]),
            "ground_truth": meta.get("ground_truth"), "top_drivers": drivers,
            "report": eng.report(row, account_id=account_id),
            "data_quality": {"completeness": round(completeness, 3), "n_blank": n_blank,
                             "n_features": n_feat, "model_trust": trust}}


@app.get("/rings")
def rings():
    """BOI candidate-ring network (behavioral-similarity proxy) + ring metadata."""
    return {"network": _load_json("mule_network_edges.json"), "rings": _load_json("mule_network.json")}


@app.get("/flow")
def flow():
    """Real AMLSim money-flow typologies (fan-in, cycle) — Phase-2 capability demo."""
    return _load_json("amlsim_flow.json") or {"typologies": []}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a CSV (raw DataSet.csv-style or cleaned export). The WHOLE dashboard re-scores
    on it — accounts, queue, analytics, and the account-network map all reflect the upload."""
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw), index_col=0, low_memory=False, nrows=MAX_ROWS)
        cols = engine().columns
        cat_maps = json.loads((ART / "categorical_maps.json").read_text())
        X, y = prepare_frame(df, cols, cat_maps)
        capped = f" (capped to {MAX_ROWS:,} rows)" if len(df) >= MAX_ROWS else ""
        d = _score_population(X, y, f"📤 {file.filename} — {len(X):,} accounts{capped}")
        n_mule = sum(1 for r in d["rows"] if r["ground_truth"] == "MULE")
        return {"ok": True, "source": d["src"], "n_accounts": len(d["rows"]),
                "n_labelled_mules": n_mule, "has_labels": any(r["ground_truth"] for r in d["rows"])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Couldn't read CSV — {type(e).__name__}: {e}")


@app.post("/reset")
def reset():
    """Drop any uploaded data, return to the committed sample."""
    global _data
    _data = None
    return {"ok": True, "source": _dashboard_data()["src"]}


@app.get("/network/{account_id}")
def network(account_id: int, k: int = 12):
    """Account-network map for ANY account in the ACTIVE population (sample or uploaded):
    its top behaviorally-similar accounts (proxy for link data, since BOI has no transaction
    edges). Returns the ego-graph: centre = the account, neighbours = look-alikes, edges
    weighted by similarity, arrows funnel look-alikes toward the higher-risk node."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity
    d = _dashboard_data(); X = d["X"]
    if account_id not in X.index:
        raise HTTPException(status_code=404, detail=f"account {account_id} not in active population")
    feats = _top_features()
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X[feats]))
    ids = list(X.index); ci = ids.index(account_id)
    sims = cosine_similarity(Z[ci:ci + 1], Z)[0]
    order = np.argsort(-sims)
    neigh = [j for j in order if ids[j] != account_id][:k]
    risk = {r["account_id"]: r for r in d["rows"]}
    nodes = [{"id": int(account_id), "risk": risk.get(account_id, {}).get("risk_score", 0),
              "band": risk.get(account_id, {}).get("band", "LOW"), "center": True}]
    edges = []
    for j in neigh:
        nid = int(ids[j]); rr = risk.get(nid, {})
        nodes.append({"id": nid, "risk": rr.get("risk_score", 0), "band": rr.get("band", "LOW"), "center": False})
        hi, lo = (account_id, nid) if risk.get(account_id, {}).get("risk_score", 0) >= rr.get("risk_score", 0) else (nid, account_id)
        edges.append({"source": int(lo), "target": int(hi), "weight": round(float(sims[j]), 3)})
    return {"account_id": account_id, "nodes": nodes, "edges": edges, "features_used": len(feats),
            "note": "Behavioral-similarity ego-network (proxy for bank link data; BOI is a snapshot)."}


# ---- serve the built React UI (single service): API routes above take precedence ----
# Build it first:  cd frontend && npm install && npm run build  -> frontend/dist
_DIST = ROOT / "frontend" / "dist"
if _DIST.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="ui")

