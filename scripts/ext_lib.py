"""
Shared library for EXTERNAL-dataset experiments (Phases 2-4) and the auditor-credibility
runs (Phase 3). NEW code — never touches BOI training/eval, never merges rows across
datasets. Every caller tags its metrics with the dataset name.

Provides:
  - leakage_safe_prep(df, target): unsupervised hygiene (target-free ordinal encode,
    drop constants, keep NaN) — the SAME philosophy as src/preprocess.py, generalised.
  - integrity_audit(X, y): the 4-signature Data Integrity Auditor (label-proxy,
    exact-bucket, range/decile, univariate-AUC) as a reusable (X,y)->DataFrame tool,
    matching src/data_integrity_auditor.py's logic but dataset-agnostic.
  - cv_pr_auc(model_fn, X, y, groups=None): leakage-safe CV — all fitting inside folds.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

MIN_N = 10
BASE_MULT = {"CRITICAL": 0.90, "HIGH": 0.50, "MEDIUM": 0.20, "WATCH": 0.10}


def leakage_safe_prep(df: pd.DataFrame, target: str, drop_cols=None):
    """Target-free hygiene. Returns (X float32 with NaN preserved, y int, cat_cols)."""
    df = df.copy()
    for c in (drop_cols or []):
        if c in df.columns:
            df = df.drop(columns=c)
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    # drop constant columns
    nun = X.nunique(dropna=True)
    X = X.drop(columns=nun[nun <= 1].index.tolist())
    # ordinal-encode object cols (deterministic, target-free; NaN stays NaN)
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        cats = sorted(X[c].dropna().unique().tolist(), key=lambda v: str(v))
        X[c] = X[c].map({v: i for i, v in enumerate(cats)}).astype("float32")
    return X.astype("float32"), y, cat_cols


def integrity_audit(X: pd.DataFrame, y: pd.Series, hints=None):
    """4-signature leakage scan. Returns severity-ranked DataFrame. dataset-agnostic."""
    hints = set(hints or [])
    base = float(y.mean()); yv = y.values
    rows = []
    for col in X.columns:
        s = X[col]; nona = s.notna()
        if nona.sum() < MIN_N or y[nona].nunique() < 2:
            continue
        try:
            a = roc_auc_score(y[nona], s[nona].astype(float)); auc = max(a, 1 - a)
        except ValueError:
            auc = np.nan
        d = pd.DataFrame({"v": s.round(3), "y": yv}).dropna(subset=["v"])
        g = d.groupby("v")["y"]; cnt, mean = g.count(), g.mean(); sel = cnt >= MIN_N
        bucket_rate = float(mean[sel].max()) if sel.any() else 0.0
        decile_rate = 0.0
        if s.nunique() >= 10:
            try:
                q = pd.qcut(s[nona], 10, duplicates="drop")
                dr = y[nona].groupby(q, observed=True).mean()
                dn = y[nona].groupby(q, observed=True).count(); dr = dr[dn >= MIN_N]
                decile_rate = float(dr.max()) if len(dr) else 0.0
            except (ValueError, IndexError):
                decile_rate = 0.0
        worst = max(bucket_rate, decile_rate); mules_cov = int(y[nona].sum())
        auc_eff = auc if (not np.isnan(auc) and mules_cov >= 10) else 0.0
        sev = ("CRITICAL" if (worst >= BASE_MULT["CRITICAL"] or auc_eff >= 0.98)
               else "HIGH" if (worst >= BASE_MULT["HIGH"] or auc_eff >= 0.95)
               else "MEDIUM" if worst >= BASE_MULT["MEDIUM"]
               else "WATCH" if (worst >= BASE_MULT["WATCH"] or (not np.isnan(auc) and auc >= 0.95)) else None)
        if sev:
            rows.append({"feature": col, "severity": sev, "positives_covered": mules_cov,
                         "univariate_auc": round(float(auc), 3) if not np.isnan(auc) else None,
                         "best_bucket_rate": round(bucket_rate, 3),
                         "best_decile_rate": round(decile_rate, 3),
                         "lift_vs_base": round(worst / base, 1) if base else None,
                         "hint_listed": col in hints})
    res = pd.DataFrame(rows)
    if len(res):
        order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "WATCH": 3}
        res["o"] = res["severity"].map(order)
        res = res.sort_values(["o", "best_bucket_rate", "univariate_auc"],
                              ascending=[True, False, False]).drop(columns="o").reset_index(drop=True)
    return res, base


def cv_pr_auc(model_fn, X, y, n_splits=5, seed=42, calibrate=False):
    """Leakage-safe stratified OOF. model_fn(spw)->estimator. Returns dict of metrics."""
    from sklearn.calibration import CalibratedClassifierCV
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    yv = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    oof = np.zeros(len(yv))
    for tr, va in skf.split(X, yv):
        spw = float((yv[tr] == 0).sum() / max((yv[tr] == 1).sum(), 1))
        m = model_fn(spw)
        if calibrate:
            m = CalibratedClassifierCV(m, method="sigmoid", cv=3)
        Xtr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
        Xva = X.iloc[va] if hasattr(X, "iloc") else X[va]
        m.fit(Xtr, yv[tr]); oof[va] = m.predict_proba(Xva)[:, 1]
    return {"oof": oof, "pr_auc": float(average_precision_score(yv, oof)),
            "roc_auc": float(roc_auc_score(yv, oof)),
            "brier": float(brier_score_loss(yv, oof)), "prevalence": float(yv.mean())}


def recall_operating_point(oof, yv, target_recall=0.97):
    """Lowest threshold achieving recall>=target; report precision cost beside it."""
    yv = np.asarray(yv); n_pos = int(yv.sum())
    rows = []
    for t in np.round(np.arange(0.005, 1.0, 0.005), 3):
        pred = oof >= t; tp = int((pred & (yv == 1)).sum()); fp = int((pred & (yv == 0)).sum()); al = tp + fp
        rows.append({"threshold": float(t), "recall": tp / n_pos, "precision": (tp / al) if al else 0.0,
                     "false_alarms": fp, "alerts": al, "tp": tp})
    tdf = pd.DataFrame(rows)
    feas = tdf[tdf.recall >= target_recall]
    if len(feas):
        r = feas.sort_values("threshold", ascending=False).iloc[0]
        sel = {"target_recall": target_recall, "achieved_recall": float(r.recall), "threshold": float(r.threshold),
               "precision": float(r.precision), "false_alarms": int(r.false_alarms), "alerts": int(r.alerts)}
    else:
        sel = {"target_recall": target_recall, "achievable": False, "max_recall": float(tdf.recall.max())}
    return sel, tdf
