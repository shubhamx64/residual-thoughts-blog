"""Paper figures: E4 two-family protection hierarchy (the flip) + drift canary.
Numbers computed from e4-continual/results logs (see make_figs comments)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "e4-continual" / "results"
FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(exist_ok=True)

INK, MUTED, GRID = "#16202b", "#5c6b7a", "#d7dde2"
QWEN, TINY = "#2a78d6", "#1baf7a"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "figure.dpi": 150,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
})

ARMS = ["baseline", "random", "weights", "footprint", "join", "fisher"]


def load_deg(suffix):
    out = {}
    for a in ARMS:
        p = RES / f"log_B_{a}{suffix}.jsonl"
        if not p.exists():
            continue
        L = [json.loads(x) for x in open(p)]
        out[a] = {"deg": 100 * (L[-1]["ppl_math"] / L[0]["ppl_math"] - 1),
                  "log": L}
    b, f = out["baseline"]["deg"], out["fisher"]["deg"]
    for a in out:
        out[a]["pctF"] = 100 * (b - out[a]["deg"]) / (b - f)
    return out


tiny = load_deg("")
qwen = load_deg("_qwen")

# ---- Figure 1: the flip -- % of Fisher recovered, cheap arms, both families ----
cheap = ["weights", "footprint", "join", "random"]
labels = ["weights\n(geometry,\nzero-data)", "footprint\n(usage)", "join\n(geom x usage)", "random"]
x = np.arange(len(cheap))
w = 0.38
fig, ax = plt.subplots(figsize=(6.6, 3.4))
tv = [tiny[a]["pctF"] for a in cheap]
qv = [qwen[a]["pctF"] for a in cheap]
b1 = ax.bar(x - w/2, tv, w, color=TINY, label="TinyLlama-1.1B", zorder=3)
b2 = ax.bar(x + w/2, qv, w, color=QWEN, label="Qwen2.5-1.5B", zorder=3)
for bars in (b1, b2):
    for r in bars:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2, h + (2 if h >= 0 else -2),
                f"{h:.0f}", ha="center", va="bottom" if h >= 0 else "top",
                fontsize=8, color=INK)
ax.axhline(0, color=MUTED, lw=0.8)
ax.axhline(100, color="#b0392b", lw=1, ls="--", zorder=2)
ax.text(len(cheap)-0.5, 101, "Fisher (backward, full data) = 100%", ha="right",
        va="bottom", fontsize=8, color="#b0392b")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("% of Fisher's protection recovered")
ax.set_title("Which cheap protector wins is family-dependent: geometry vs usage flips",
             fontsize=10.5, fontweight="bold", pad=8)
ax.set_ylim(-25, 118)
ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2)
ax.set_axisbelow(True)
fig.text(0.5, -0.06,
         "TinyLlama: usage arms (footprint/join) beat pure geometry (weights).  "
         "Qwen: pure geometry beats usage.  Both: random ~ baseline ~ 0.",
         ha="center", fontsize=8, color=MUTED)
fig.savefig(FIGS / "fig_e4_flip.pdf")
fig.savefig(FIGS / "fig_e4_flip.png", dpi=150)
plt.close(fig)

# ---- Figure 2: drift canary, both families, pooled rho ----
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for fam, data, col in (("TinyLlama-1.1B", tiny, TINY), ("Qwen2.5-1.5B", qwen, QWEN)):
    dd, rr = [], []
    for a, d in data.items():
        L = d["log"]; p0 = L[0]["ppl_math"]
        for r in L[1:]:
            if r.get("fp_drift") is not None:
                dd.append(r["fp_drift"]); rr.append(100 * (r["ppl_math"]/p0 - 1))
    rho, _ = stats.spearmanr(dd, rr)
    ax.scatter(dd, rr, s=26, color=col, alpha=0.75, edgecolor="white", lw=0.5,
               label=f"{fam}  (rho={rho:.3f})", zorder=3)
ax.set_xlabel("footprint drift (1 - cos to after-A ref)")
ax.set_ylabel("retention degradation (%)")
ax.set_title("Drift canary: step-100+ drift tracks final retention loss",
             fontsize=10.5, fontweight="bold", pad=8)
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
ax.set_axisbelow(True)
fig.savefig(FIGS / "fig_e4_drift.pdf")
fig.savefig(FIGS / "fig_e4_drift.png", dpi=150)
plt.close(fig)

print("wrote", *(p.name for p in sorted(FIGS.glob("*.pdf"))))
