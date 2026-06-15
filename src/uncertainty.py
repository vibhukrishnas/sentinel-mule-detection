"""
Abstention / uncertainty policy — the honest fix for "overconfident wrong calls".

Instead of forcing every account into mule/legit, SENTINEL carves a middle
**"uncertain → route to analyst"** band. Thresholds are calibrated on the
leakage-free OUT-OF-FOLD predictions (outputs/predictions.csv), so the policy's
coverage/error numbers are honest generalization estimates, not in-sample.

Policy (class-conditional, conformal-flavored):
  - CONFIDENT-MULE  : p >= t_hi, chosen so OOF precision in this zone >= TARGET_PREC
  - CONFIDENT-LEGIT : p <= t_lo, chosen so OOF NPV in this zone       >= TARGET_NPV
  - UNCERTAIN       : t_lo < p < t_hi  -> human review (the model declines to auto-decide)

Run: python src/uncertainty.py   ->  artifacts/uncertainty.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
TARGET_PREC = 0.95     # in the confident-MULE zone, >=95% really are mules
TARGET_NPV = 0.999     # in the confident-LEGIT zone, >=99.9% really are legit


def main():
    df = pd.read_csv(ROOT / "outputs" / "predictions.csv")
    p = df["probability"].to_numpy()
    y = df["actual_label"].to_numpy()

    # t_hi: lowest prob whose upper tail is >=TARGET_PREC mules
    grid = np.unique(np.round(p, 4))
    t_hi = 1.0
    for t in grid:
        sel = p >= t
        if sel.sum() and y[sel].mean() >= TARGET_PREC:
            t_hi = float(t)
            break
    # t_lo: highest prob whose lower tail is >=TARGET_NPV legit
    t_lo = 0.0
    for t in grid[::-1]:
        sel = p <= t
        if sel.sum() and (1 - y[sel]).mean() >= TARGET_NPV:
            t_lo = float(t)
            break
    if t_lo >= t_hi:                       # degenerate guard
        t_lo, t_hi = 0.10, 0.90

    conf_mule = p >= t_hi
    conf_legit = p <= t_lo
    uncertain = ~conf_mule & ~conf_legit
    auto = conf_mule | conf_legit
    # errors inside the auto-decided zones
    err = int((conf_mule & (y == 0)).sum() + (conf_legit & (y == 1)).sum())
    mules_in_review = int((uncertain & (y == 1)).sum())

    out = {
        "t_lo": round(t_lo, 4), "t_hi": round(t_hi, 4),
        "target_precision_mule_zone": TARGET_PREC, "target_npv_legit_zone": TARGET_NPV,
        "n": int(len(p)),
        "coverage_auto_decided": round(float(auto.mean()), 4),
        "review_rate": round(float(uncertain.mean()), 4),
        "n_uncertain": int(uncertain.sum()),
        "auto_zone_errors": err,
        "auto_zone_error_rate": round(err / max(int(auto.sum()), 1), 5),
        "confident_mule_precision": round(float(y[conf_mule].mean()) if conf_mule.any() else 0.0, 4),
        "confident_legit_npv": round(float((1 - y[conf_legit]).mean()) if conf_legit.any() else 0.0, 4),
        "mules_routed_to_review": mules_in_review,
        "basis": "leakage-free out-of-fold predictions (outputs/predictions.csv)",
    }
    (ART / "uncertainty.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\n-> auto-decide {out['coverage_auto_decided']:.0%} at {out['auto_zone_error_rate']:.2%} error; "
          f"route {out['review_rate']:.0%} ({out['n_uncertain']}) to analysts.")


if __name__ == "__main__":
    main()
