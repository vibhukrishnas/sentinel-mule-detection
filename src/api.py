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
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from sentinel import SentinelEngine

MAX_FEATURES = 5000             # reject absurd payloads (the model uses ~2,965 features)
API_KEY = os.getenv("SENTINEL_API_KEY")   # if set, callers must send X-API-Key: <key>
RATE_LIMIT, RATE_WINDOW = 60, 60          # max 60 requests / 60 s per client IP
app = FastAPI(title="SENTINEL — Mule Account Risk Engine (internal/dev)", version="1.0")
_engine: SentinelEngine | None = None
_hits: dict[str, list[float]] = defaultdict(list)


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
