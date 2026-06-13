"""Render figures/arch_diagram.png - SENTINEL target architecture with an HONEST
built-vs-roadmap distinction (solid navy = built & validated on the provided data;
dashed grey = Phase-2 live feeds + graph mule-ring intelligence)."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

FIG = Path(__file__).resolve().parent.parent / "figures"; FIG.mkdir(exist_ok=True)
NAVY, RED, GREY, INK = "#1f3b6e", "#d11f2d", "#8a8a8a", "#1a1a1a"

fig, ax = plt.subplots(figsize=(11.5, 6.6), dpi=130); ax.axis("off")
ax.set_xlim(0, 100); ax.set_ylim(0, 74)


def box(x, y, w, h, title, sub, built=True):
    ec = NAVY if built else GREY
    fc = "#eef1f7" if built else "#fbfbfb"
    ls = "-" if built else (0, (4, 2))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.8",
                                fc=fc, ec=ec, lw=1.5, linestyle=ls))
    ax.text(x + w / 2, y + h - 3.6, title, ha="center", va="top", fontsize=8.6,
            fontweight="bold", color=NAVY if built else GREY)
    ax.text(x + w / 2, y + h - 7.6, sub, ha="center", va="top", fontsize=6.7, color=GREY)


def arr(x1, y1, x2, y2, built=True):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 lw=1.4, color=RED if built else GREY,
                                 linestyle="-" if built else (0, (3, 2))))


def band(y, label):
    ax.text(-0.5, y, label, ha="left", va="center", fontsize=7.5, fontweight="bold",
            color=RED, rotation=90)


ax.text(50, 72.5, "SENTINEL - real-time mule-account containment engine", ha="center",
        fontsize=12.5, fontweight="bold", color=NAVY)

# Row 1 - INGESTION (multi-feed)
band(60, "INGEST")
box(8, 53, 21, 14, "Account features", "provided dataset\n(9,082 x 3,924)", True)
box(31, 53, 21, 14, "Transaction stream", "live txns / velocity", False)
box(54, 53, 21, 14, "Alert feeds", "fraud + txn-monitoring\nsystem alerts", False)
box(77, 53, 21, 14, "Govt / regulatory", "cyber-fraud tickets,\nRBI / FIU feeds", False)

# Row 2 - DETECTION CORE
band(37, "DETECT")
box(8, 30, 26, 15, "Hygiene + DATA INTEGRITY\nAUDITOR", "drop dead cols; block\n~585 leak features", True)
box(38, 30, 26, 15, "Calibrated LightGBM", "rules + anomaly + supervised\nscoring (NaN-native)", True)
box(68, 30, 30, 15, "Graph mule-ring engine", "shared device/IP/beneficiary,\ncircular fund flows", False)

# Row 3 - DECISION / CONTAINMENT / SERVE
band(13, "ACT")
box(8, 6, 26, 15, "Risk score 0-100\n+ SHAP explain", "calibrated prob +\nplain-English reasons", True)
box(38, 6, 26, 15, "CONTAINMENT action", "monitor / soft-hold /\nhard-hold / escalate", True)
box(68, 6, 30, 15, "Dashboard + API + report", "FastAPI ~35ms, Streamlit,\ninvestigation report", True)

# arrows ingest -> detect
for x in (18.5, 41.5, 64.5, 87.5):
    arr(x, 53, 41 if x < 50 else 83, 45, built=(x < 29))
# detect -> act
arr(21, 30, 21, 21); arr(51, 30, 51, 21); arr(83, 30, 83, 21, built=False)
# core score -> containment -> serve
arr(34, 13.5, 38, 13.5); arr(64, 13.5, 68, 13.5)

# legend
lx, ly = 50, 0.5
ax.add_patch(FancyBboxPatch((lx - 30, ly), 7, 3, boxstyle="round,pad=0.2", fc="#eef1f7", ec=NAVY, lw=1.4))
ax.text(lx - 22, ly + 1.5, "Built & validated on the provided data", va="center", fontsize=7, color=INK)
ax.add_patch(FancyBboxPatch((lx + 14, ly), 7, 3, boxstyle="round,pad=0.2", fc="#fbfbfb", ec=GREY, lw=1.4, linestyle=(0, (4, 2))))
ax.text(lx + 22, ly + 1.5, "Phase-2: live feeds + graph mule-rings", va="center", fontsize=7, color=GREY)

fig.savefig(FIG / "arch_diagram.png", bbox_inches="tight"); plt.close(fig)
print("wrote figures/arch_diagram.png (target architecture, built vs roadmap)")
