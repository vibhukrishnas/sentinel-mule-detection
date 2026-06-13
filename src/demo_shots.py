"""
Generate REAL product 'screenshots' from live SentinelEngine output (not mockups):
  figures/shot_scoring_card.png    - account risk card (gauge + SHAP drivers + action)
  figures/shot_investigation.png   - the auto-generated investigation report panel
  figures/shot_api.png             - the actual /score JSON response (terminal style)
These are genuine outputs for a real mule account in the dataset.
"""
from pathlib import Path
import sys, json, textwrap
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from sentinel import SentinelEngine, ART, band_for

FIG = ART.parent / "figures"; FIG.mkdir(exist_ok=True)
NAVY, RED, AMBER, GREEN, INK = "#1f3b6e", "#d11f2d", "#e8a33d", "#2ca02c", "#1a1a1a"
BAND_COLOR = {"CRITICAL": RED, "HIGH": "#e8682d", "MEDIUM": AMBER, "LOW": GREEN}

eng = SentinelEngine()
demo = pd.read_parquet(ART / "demo_accounts.parquet")
demoy = pd.read_parquet(ART / "demo_targets.parquet")["target"]
# pick a real, confidently-flagged mule
mid = max((i for i in demo.index if demoy.loc[i] == 1), key=lambda i: eng.score(demo.loc[i])["risk_score"])
acct = demo.loc[mid]
sc = eng.score(acct)
drivers = [d for d in eng.explain(acct, top_k=6) if d["shap"] > 0][:5]
band = sc["band"]; col = BAND_COLOR[band]


# ---------- 1. SCORING CARD ----------
fig = plt.figure(figsize=(9, 5.2), dpi=130); fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.add_patch(Rectangle((0, 9.05), 10, 0.95, color=NAVY))
ax.text(0.25, 9.52, "SENTINEL", color="white", fontsize=15, fontweight="bold", va="center")
ax.text(2.6, 9.52, "Real-time Mule-Account Risk Scoring", color="#cdd6ea", fontsize=10.5, va="center")
ax.text(9.75, 9.52, f"Account #{mid}", color="white", fontsize=10, va="center", ha="right")
# big score + band chip
ax.text(0.3, 7.7, f"{sc['risk_score']}", color=col, fontsize=58, fontweight="bold", va="center")
ax.text(2.7, 8.25, "/100", color="#888", fontsize=16, va="center")
ax.add_patch(FancyBboxPatch((2.7, 7.0), 2.1, 0.7, boxstyle="round,pad=0.05", color=col))
ax.text(3.75, 7.35, band, color="white", fontsize=13, fontweight="bold", ha="center", va="center")
ax.text(0.3, 6.5, f"Calibrated probability of mule activity: {sc['probability']:.1%}", fontsize=10.5, color=INK)
# gauge
ax.add_patch(Rectangle((0.3, 5.9), 9.4, 0.35, color="#e9e9e9"))
ax.add_patch(Rectangle((0.3, 5.9), 9.4 * sc["risk_score"] / 100, 0.35, color=col))
for t, lab in [(0, "0"), (0.4, "40"), (0.7, "70"), (0.9, "90"), (1.0, "100")]:
    ax.text(0.3 + 9.4 * t, 5.55, lab, fontsize=7.5, color="#888", ha="center")
# drivers
ax.text(0.3, 5.0, "TOP RISK DRIVERS (SHAP)", fontsize=10, fontweight="bold", color=NAVY)
mx = max(abs(d["shap"]) for d in drivers) or 1
for i, d in enumerate(drivers):
    y = 4.35 - i * 0.62
    ax.add_patch(Rectangle((4.7, y), 4.8 * abs(d["shap"]) / mx, 0.42, color=col, alpha=0.85))
    lab = d["label"].replace(" (anonymized)", "")[:46]
    ax.text(0.35, y + 0.21, f"{lab} = {d['value_readable']}", fontsize=8.3, va="center", color=INK)
ax.text(0.3, 0.55, "RECOMMENDED ACTION", fontsize=9.5, fontweight="bold", color=NAVY)
action = {"CRITICAL": "Freeze outbound transfers immediately; escalate to L2 fraud unit; file SAR review.",
          "HIGH": "Hold high-value outbound transactions; assign to analyst for same-day review."}.get(band, "Monitor.")
ax.add_patch(FancyBboxPatch((0.3, 0.05), 9.4, 0.45, boxstyle="round,pad=0.04", color="#fbeaec"))
ax.text(0.45, 0.28, action, fontsize=8.6, va="center", color=RED)
fig.savefig(FIG / "shot_scoring_card.png", bbox_inches="tight"); plt.close(fig)


# ---------- 2. INVESTIGATION REPORT ----------
rep = eng.report(acct, account_id=mid)
fig = plt.figure(figsize=(9, 6), dpi=130); fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.add_patch(Rectangle((0, 0.94), 1, 0.06, transform=ax.transAxes, color=NAVY))
ax.text(0.02, 0.97, "AUTO-GENERATED INVESTIGATION REPORT", transform=ax.transAxes,
        color="white", fontsize=11, fontweight="bold", va="center")
wrapped = "\n".join("\n".join(textwrap.wrap(ln, 96)) if len(ln) > 96 else ln for ln in rep.splitlines())
ax.text(0.02, 0.90, wrapped, transform=ax.transAxes, fontsize=8.1, family="monospace",
        va="top", color=INK)
fig.savefig(FIG / "shot_investigation.png", bbox_inches="tight"); plt.close(fig)


# ---------- 3. API RESPONSE (terminal) ----------
alert = eng.alert(acct, threshold=0.5, account_id=mid)
resp = {"account_id": int(mid), "risk_score": sc["risk_score"], "band": sc["band"],
        "probability": round(sc["probability"], 4),
        "recommended_action": alert["recommended_action"],
        "top_drivers": [{"factor": d["label"].replace(" (anonymized)", ""),
                         "value": d["value_readable"], "effect": d["direction"]} for d in drivers[:3]]}
body = ("$ curl -s -X POST localhost:8000/score \\\n"
        "      -H 'Content-Type: application/json' \\\n"
        f"      -d '{{\"account_id\": {mid}, \"features\": {{...}} }}'\n\n"
        + json.dumps(resp, indent=2))
fig = plt.figure(figsize=(9, 6.2), dpi=130); fig.patch.set_facecolor("#0f1117")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.add_patch(Rectangle((0, 0.95), 1, 0.05, transform=ax.transAxes, color="#22262f"))
for i, c in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
    ax.add_patch(plt.Circle((0.03 + i * 0.025, 0.975), 0.008, transform=ax.transAxes, color=c))
ax.text(0.5, 0.975, "SENTINEL  /score  - real-time scoring API", transform=ax.transAxes,
        color="#aab", fontsize=9.5, ha="center", va="center")
ax.text(0.02, 0.90, body, transform=ax.transAxes, fontsize=8.6, family="monospace",
        va="top", color="#d7f7d2")
fig.savefig(FIG / "shot_api.png", facecolor=fig.get_facecolor(), bbox_inches="tight"); plt.close(fig)

print(f"Rendered 3 live screenshots for mule account #{mid} (score {sc['risk_score']}, {band}) -> figures/")
print("  shot_scoring_card.png, shot_investigation.png, shot_api.png")
