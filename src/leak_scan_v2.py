"""
Comprehensive NON-MONOTONIC leak scan (the kind univariate ROC-AUC misses).

For every feature we bucket its values and ask: does ANY value-bucket (with enough
support) have a fraud rate wildly above the 0.89% base rate? A bucket that is, say,
100% fraud over 48 accounts is not a behavioral signal — it is the label leaking
through a tree-splittable value (this is exactly how F2230==3 hid).

Outputs a blocklist of leak suspects and how many of the 81 mules they 'explain',
so we can remove them and finally measure HONEST performance.
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from preprocess import load_cached, ART, HINT_FEATURES

MIN_N = 10          # bucket must cover >=10 accounts to count
LEAK_RATE = 0.30    # >=30% fraud in a bucket (~34x base rate) => leak suspect
PURE_RATE = 0.90    # buckets this pure essentially encode the label

X, y = load_cached()
base = y.mean()
print(f"Base fraud rate = {base:.4%}. Scanning {X.shape[1]} features "
      f"(bucket n>={MIN_N}, leak if fraud-rate>={LEAK_RATE:.0%})\n")

records = []
mules_in_pure = set()
for col in X.columns:
    s = X[col].round(3)
    df = pd.DataFrame({"v": s, "y": y.values})
    g = df.dropna(subset=["v"]).groupby("v")["y"]
    cnt = g.count(); rate = g.mean()
    keep = cnt >= MIN_N
    if not keep.any():
        continue
    rate_k = rate[keep]; cnt_k = cnt[keep]
    best_rate = rate_k.max()
    if best_rate >= LEAK_RATE:
        best_v = rate_k.idxmax()
        n = int(cnt_k[best_v]); mules_here = int(round(best_rate * n))
        records.append({"feature": col, "best_value": float(best_v),
                        "fraud_rate": float(best_rate), "n": n,
                        "mules_in_bucket": mules_here,
                        "bank_hint": col.split("__")[0] in HINT_FEATURES})
        if best_rate >= PURE_RATE:
            pure_mask = (s == best_v)
            mules_in_pure |= set(y.index[pure_mask & (y == 1)])

res = pd.DataFrame(records).sort_values("fraud_rate", ascending=False).reset_index(drop=True)
print(f"=== LEAK SUSPECTS: {len(res)} features with a >={LEAK_RATE:.0%}-fraud bucket ===")
print(res.head(40).to_string(index=False))

pure = res[res["fraud_rate"] >= PURE_RATE]
print(f"\nFeatures with a >={PURE_RATE:.0%}-pure fraud bucket: {len(pure)}")
print(f"Distinct mules captured by pure buckets alone: {len(mules_in_pure)} / {int(y.sum())}")
print("\nBank-hinted features among suspects (keep these — likely genuine strong signal):")
print(res[res["bank_hint"]].to_string(index=False) if res["bank_hint"].any() else "  none")

# Proposed blocklist = pure-bucket leakers that are NOT bank-hinted
blocklist = sorted(pure[~pure["bank_hint"]]["feature"].tolist())
(ART / "leak_scan_v2.json").write_text(json.dumps(
    {"leak_rate": LEAK_RATE, "pure_rate": PURE_RATE,
     "suspects": res.to_dict("records"),
     "proposed_blocklist": blocklist,
     "mules_in_pure_buckets": len(mules_in_pure)}, indent=2, default=float))
print(f"\nProposed blocklist ({len(blocklist)} features) -> artifacts/leak_scan_v2.json")
print(blocklist)
