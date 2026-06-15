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
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentinel import SentinelEngine

MAX_FEATURES = 5000   # reject absurd payloads (the model uses ~2,965 features)
app = FastAPI(title="SENTINEL — Mule Account Risk Engine (internal/dev)", version="1.0")
_engine: SentinelEngine | None = None


def _guard(features: dict):
    if not isinstance(features, dict) or len(features) > MAX_FEATURES:
        raise HTTPException(status_code=413, detail=f"features must be a dict of ≤ {MAX_FEATURES} keys")


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
def score(acct: Account):
    _guard(acct.features)
    eng = engine()
    sc = eng.score(acct.features)
    alert = eng.alert(acct.features, threshold=0.5, account_id=acct.account_id)
    return {"account_id": acct.account_id, **sc,
            "alert": alert is not None,
            "recommended_action": (alert or {}).get("recommended_action"),
            "top_drivers": eng.explain(acct.features, top_k=5)}


@app.post("/report")
def report(acct: Account):
    _guard(acct.features)
    return {"account_id": acct.account_id,
            "report": engine().report(acct.features, account_id=acct.account_id)}
