"""
SENTINEL — real-time scoring + explainability + alerting + investigation reports.

This is the production-facing engine. Given one account's features it returns, in a
few milliseconds:
  - a calibrated probability and a 0-100 risk score with a severity band,
  - the top contributing factors (SHAP), each rendered in plain English and grounded
    in the account's value + its population percentile (no invented semantics),
  - an alert object when the score crosses a configurable threshold,
  - a natural-language investigation report an analyst can act on immediately.

Why two models? The CALIBRATED model gives a trustworthy probability; a STANDALONE
base tree model (same family, fit on all data) drives SHAP attribution. Their
rankings agree (calibration is monotonic), and TreeExplainer is exact + fast on it.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from feature_meanings import human_label, decode_value, base_feature, BANK_HINTED

ART = Path(__file__).resolve().parent.parent / "artifacts"

# Band cutoffs validated against out-of-fold scores of all 81 real mules
# (honest_eval.json): median real mule scores 98/100, and ~69% (56/81) score >=70 — so
# CRITICAL>=90 captures the dense cluster and HIGH>=70 the bulk. Derived from CV,
# NOT from the noisy 16-positive holdout.
BANDS = [(90, "CRITICAL"), (70, "HIGH"), (40, "MEDIUM"), (0, "LOW")]
ACTION = {
    "CRITICAL": "Freeze outbound transfers immediately; escalate to L2 fraud unit; "
                "file internal SAR review.",
    "HIGH": "Hold high-value outbound transactions; assign to analyst for same-day review.",
    "MEDIUM": "Add to enhanced-monitoring watchlist; review on next cycle.",
    "LOW": "No action; routine monitoring.",
}


def band_for(score: int) -> str:
    for thr, name in BANDS:
        if score >= thr:
            return name
    return "LOW"


class SentinelEngine:
    def __init__(self, art: Path = ART):
        self.model = joblib.load(art / "sentinel_model.joblib")      # calibrated
        self.base = joblib.load(art / "base_model.joblib")           # for SHAP
        self.stats = joblib.load(art / "population_stats.joblib")
        self.columns = self.stats["columns"]
        # guard: model and narrative must agree on feature space / order
        assert getattr(self.base, "n_features_in_", len(self.columns)) == len(self.columns), \
            "base model feature count != population_stats columns"
        self._explainer = None
        # abstention thresholds (calibrated on OOF predictions; see src/uncertainty.py)
        import json
        up = art / "uncertainty.json"
        u = json.loads(up.read_text()) if up.exists() else {"t_lo": 0.10, "t_hi": 0.90}
        self.t_lo, self.t_hi = float(u["t_lo"]), float(u["t_hi"])

    def confidence_tier(self, probability: float) -> str:
        """Abstain on the ambiguous middle instead of making an overconfident call."""
        if probability >= self.t_hi:
            return "CONFIDENT-MULE"
        if probability <= self.t_lo:
            return "CONFIDENT-LEGIT"
        return "UNCERTAIN"

    # ---- input alignment ------------------------------------------------
    def _align(self, account) -> pd.DataFrame:
        if isinstance(account, pd.Series):
            account = account.to_dict()
        if isinstance(account, dict):
            row = {c: account.get(c, np.nan) for c in self.columns}
            X = pd.DataFrame([row], columns=self.columns)
        elif isinstance(account, pd.DataFrame):
            X = account.reindex(columns=self.columns)
        else:
            raise TypeError("account must be dict / Series / DataFrame")
        return X.astype("float32")

    # ---- scoring --------------------------------------------------------
    def score(self, account) -> dict:
        X = self._align(account)
        p = float(self.model.predict_proba(X)[:, 1][0])
        s = int(round(100 * p))
        return {"probability": p, "risk_score": s, "band": band_for(s)}

    # ---- explainability -------------------------------------------------
    @property
    def explainer(self):
        if self._explainer is None:
            import shap
            self._explainer = shap.TreeExplainer(self.base)
        return self._explainer

    def _percentile_note(self, col: str, value) -> str:
        """Where does this value sit vs the population, and toward which class?"""
        if col.endswith("__ismissing") or pd.isna(value):
            return ""
        q05 = self.stats["q05"].get(col); q95 = self.stats["q95"].get(col)
        med = self.stats["median"].get(col)
        ml = self.stats["mean_legit"].get(col); mm = self.stats["mean_mule"].get(col)
        parts = []
        try:
            v = float(value)
            if q05 is not None and v <= q05:
                parts.append("in the bottom 5% of accounts")
            elif q95 is not None and v >= q95:
                parts.append("in the top 5% of accounts")
            elif med is not None:
                parts.append("above the median" if v > med else "below the median")
            if ml is not None and mm is not None and not (np.isnan(ml) or np.isnan(mm)):
                # which class mean is this value closer to?
                closer = "mule" if abs(v - mm) < abs(v - ml) else "legitimate"
                parts.append(f"closer to the typical {closer} profile")
        except (TypeError, ValueError):
            pass
        return "; ".join(parts)

    def explain(self, account, top_k: int = 6) -> list[dict]:
        X = self._align(account)
        sv = self.explainer.shap_values(X)
        if isinstance(sv, list):           # old SHAP API: one array per class
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:                   # new API: (n_samples, n_features, n_classes)
            sv = sv[..., 1]
        sv = sv.reshape(-1)                # single row -> flat vector
        assert sv.shape[0] == len(self.columns), \
            f"SHAP shape {sv.shape} != {len(self.columns)} features"
        order = np.argsort(-np.abs(sv))[:top_k]
        out = []
        for i in order:
            col = self.columns[i]
            raw = X.iloc[0, i]
            val = None if pd.isna(raw) else (int(raw) if float(raw).is_integer() else float(raw))
            out.append({
                "feature": col,
                "label": human_label(col),
                "value": "BLANK" if pd.isna(raw) else val,
                "value_readable": decode_value(col, "BLANK" if pd.isna(raw) else val),
                "shap": float(sv[i]),
                "direction": "raises risk" if sv[i] > 0 else "lowers risk",
                "bank_known": base_feature(col) in BANK_HINTED,
                "context": self._percentile_note(col, raw),
            })
        return out

    # ---- alert ----------------------------------------------------------
    def alert(self, account, threshold: float = 0.5, account_id="?") -> dict | None:
        sc = self.score(account)
        if sc["probability"] < threshold:
            return None
        drivers = [d for d in self.explain(account) if d["shap"] > 0][:5]
        return {
            "account_id": account_id,
            "risk_score": sc["risk_score"],
            "band": sc["band"],
            "probability": round(sc["probability"], 4),
            "recommended_action": ACTION[sc["band"]],
            "top_drivers": drivers,
        }

    # ---- natural-language investigation report --------------------------
    def report(self, account, account_id="?") -> str:
        sc = self.score(account)
        drivers = self.explain(account, top_k=6)
        up = [d for d in drivers if d["shap"] > 0][:5]
        down = [d for d in drivers if d["shap"] < 0][:2]
        X = self._align(account)
        n_blank = int(X.iloc[0].isna().sum())

        L = []
        L.append(f"INVESTIGATION REPORT — Account #{account_id}")
        L.append("=" * 56)
        L.append(f"Risk Score : {sc['risk_score']}/100  ({sc['band']})")
        L.append(f"Calibrated probability of mule activity : {sc['probability']:.1%}")
        L.append(f"Recommended action : {ACTION[sc['band']]}")
        L.append("")
        L.append("WHY THIS ACCOUNT WAS FLAGGED (top risk drivers):")
        if not up:
            L.append("  - No positive risk drivers; score is low.")
        for i, d in enumerate(up, 1):
            badge = " [bank-listed feature]" if d["bank_known"] else ""
            ctx = f" — {d['context']}" if d["context"] else ""
            L.append(f"  {i}. {d['label']}{badge}: {d['value_readable']}{ctx}")
        if down:
            L.append("")
            L.append("MITIGATING FACTORS (these lowered the score):")
            for d in down:
                L.append(f"  - {d['label']}: {d['value_readable']}")
        L.append("")
        L.append("CONFIDENCE & CAVEATS:")
        L.append(f"  - Score derived from {len(self.columns) - n_blank} populated "
                 f"features; {n_blank} were blank for this account.")
        L.append("  - Snapshot-based; not a substitute for transaction-level review.")
        L.append("  - F3912 (post-hoc fraud flag) was excluded as leakage — this score "
                 "reflects behavioral signal only.")
        return "\n".join(L)


if __name__ == "__main__":
    eng = SentinelEngine()
    demo = pd.read_parquet(ART / "demo_accounts.parquet")
    demo_y = pd.read_parquet(ART / "demo_targets.parquet")["target"]
    for idx in demo.index[:3]:
        print(eng.report(demo.loc[idx], account_id=idx))
        print(f"(actual label: {demo_y.loc[idx]})\n")
