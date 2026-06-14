"""Render figures/10_leakage_before_after.png - the leakage story in ONE picture:
fake 0.998 (with leakage) -> auditor removes leaks -> defensible 0.81-0.89, plus the
concrete F2230 example. Static narrative figure (no data/model needed)."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = Path(__file__).resolve().parent.parent / "figures"; FIG.mkdir(exist_ok=True)
NAVY, RED, GREEN, GREY, INK = "#1f3b6e", "#d11f2d", "#2ca02c", "#5a5a5a", "#1a1a1a"

fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=130); ax.axis("off")
ax.set_xlim(0, 100); ax.set_ylim(0, 50)
ax.text(50, 47.5, "The leakage story in one picture", ha="center", fontsize=14,
        fontweight="bold", color=NAVY)

# BEFORE box (red)
ax.add_patch(FancyBboxPatch((2, 22), 30, 20, boxstyle="round,pad=0.6,rounding_size=2",
                            fc="#fbeaec", ec=RED, lw=1.8))
ax.text(17, 39, "WITH all features", ha="center", fontsize=10.5, fontweight="bold", color=RED)
ax.text(17, 32.5, "PR-AUC 0.998", ha="center", fontsize=17, fontweight="bold", color=RED)
ax.text(17, 28, "ROC-AUC 1.000", ha="center", fontsize=10, color=INK)
ax.text(17, 24.2, "FAKE - the model reads the answer", ha="center", fontsize=8.2, style="italic", color=RED)

# arrow + auditor
ax.add_patch(FancyArrowPatch((33, 32), (67, 32), arrowstyle="-|>", mutation_scale=22, lw=2.2, color=NAVY))
ax.text(50, 36.5, "Data Integrity Auditor", ha="center", fontsize=10, fontweight="bold", color=NAVY)
ax.text(50, 33.3, "removes F3912 + ~585 leak features", ha="center", fontsize=8.4, color=GREY)
ax.text(50, 27.5, "(auto-detected; confirmed by the\nleakage-sensitivity sweep)", ha="center",
        fontsize=7.6, style="italic", color=GREY)

# AFTER box (green)
ax.add_patch(FancyBboxPatch((68, 22), 30, 20, boxstyle="round,pad=0.6,rounding_size=2",
                            fc="#eaf6ea", ec=GREEN, lw=1.8))
ax.text(83, 39, "LEAKAGE-REMOVED (honest)", ha="center", fontsize=9.8, fontweight="bold", color=GREEN)
ax.text(83, 32.5, "PR-AUC 0.81-0.89", ha="center", fontsize=15, fontweight="bold", color=GREEN)
ax.text(83, 28, "ROC-AUC ~0.98", ha="center", fontsize=10, color=INK)
ax.text(83, 24.2, "DEFENSIBLE - what we report", ha="center", fontsize=8.2, style="italic", color=GREEN)

# concrete example callout
ax.add_patch(FancyBboxPatch((2, 2), 96, 15, boxstyle="round,pad=0.5,rounding_size=2",
                            fc="#f5f6fa", ec=NAVY, lw=1.3))
ax.text(50, 14, "Concrete leak we caught - F2230 (a month-stamp field)", ha="center",
        fontsize=9.5, fontweight="bold", color=NAVY)
ax.text(50, 9.2, "'Oct25' = 100% legitimate    |    'Sep25' / 'Nov25' / 'Dec25' = 100% fraud "
        "(captures ALL 81 mules)", ha="center", fontsize=8.6, color=INK)
ax.text(50, 4.8, "Class means look identical (AUC = 0.59 misses it) - but a single tree split nails it. "
        "Our value-bucket scan catches exactly this.", ha="center", fontsize=8.0, style="italic", color=GREY)

fig.savefig(FIG / "10_leakage_before_after.png", bbox_inches="tight"); plt.close(fig)
print("wrote figures/10_leakage_before_after.png")
