"""E5 figures: partial-rho depth profiles (4 families) + strata medians at
matched rate."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["qwen2.5-1.5b", "gemma-2-2b", "pythia-1.4b", "tinyllama-1.1b"]
MC = {"qwen2.5-1.5b": "#2a78d6", "gemma-2-2b": "#eda100",
      "pythia-1.4b": "#4a3aa7", "tinyllama-1.1b": "#1baf7a"}
MLAB = {m: m.split("-")[0].replace("qwen2.5", "Qwen").replace("gemma", "Gemma-2")
        .replace("pythia", "Pythia").replace("tinyllama", "TinyLlama") for m in MODELS}
INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
STRATA = ["opponent", "duplicate", "uncoupled-crowded", "isolated"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
})

S = {m: json.load(open(ROOT / "results" / m / "e5_stats.json")) for m in MODELS}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
ax = axes[0]
for m in MODELS:
    r = S[m]["partial_rho_by_layer"]["top10"]
    ax.plot(range(len(r)), r, "-", lw=2, color=MC[m], label=MLAB[m])
ax.axhline(0, color=INK, lw=1)
ax.axhspan(-0.10, 0.10, color=GRID, alpha=0.6)
ax.annotate("pre-registered null band", (0.02, 0.04), xycoords="axes fraction",
            fontsize=8.5, color=MUTED)
ax.axhline(0.15, color="#e34948", lw=1.2, ls="--")
ax.annotate("O1 threshold", (0.02, 0.9), xycoords="axes fraction", fontsize=8.5,
            color="#e34948")
ax.set_xlabel("layer"); ax.set_ylabel("partial Spearman ρ | log rate")
ax.set_title("Regime mixing vs local crowding (top-10 |cos|), rate-partialed")
ax.set_ylim(-0.35, 0.35)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_axisbelow(True)

ax = axes[1]
w = 0.19
x = np.arange(len(STRATA))
for i, m in enumerate(MODELS):
    sm = S[m]["strata"]["rate_matched_median"]
    vals = [sm.get(s, np.nan) for s in STRATA]
    ax.bar(x + (i - 1.5) * w, vals, w, color=MC[m], label=MLAB[m])
ax.set_xticks(x)
ax.set_xticklabels(["opponent\ncouple", "duplicate\ncouple", "uncoupled\ncrowded", "isolated"],
                   fontsize=9)
ax.set_ylabel("median regime entropy (bits), rate-matched")
ax.set_title("Mixing by geometric neighborhood type")
ax.set_ylim(1.4, 2.32)
ax.legend(frameon=False, fontsize=8)
ax.set_axisbelow(True)

fig.suptitle("E5 — does polysemanticity co-locate with crowding?", y=1.04,
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(ROOT / "results" / "e5_summary.png", dpi=160, bbox_inches="tight")
print("wrote figure")
