"""E-Q figure: salient-protection curves per map, both models."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
MAP_COLOR = {"random": "#898781", "reader": "#eda100",
             "footprint": "#1baf7a", "fisher": "#4a3aa7"}
MODELS = ["tinyllama-1.1b", "qwen2.5-1.5b"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
for ax, m in zip(axes, MODELS):
    with open(ROOT / "results" / m / "eq2_results_low4.json") as f:
        R = json.load(f)
    with open(ROOT / "results" / m / "eq_results.json") as f:
        R0 = json.load(f)
    # math+code geometric mean (prose is noisy on TinyLlama)
    def score(r):
        return float(np.sqrt(r["math"] * r["code"]))
    xs = [0, 1, 5, 10]
    for name in ("random", "reader", "footprint", "fisher"):
        ys = [score(R["uniform4"])] + [score(R[f"{name}_p{k}"]) for k in (1, 5, 10)]
        ax.plot(xs, ys, "o-", lw=2, ms=5, color=MAP_COLOR[name], label=name)
    bf = score(R0["bf16"])
    ax.axhline(bf, color=INK, lw=1.2, ls="--")
    ax.annotate("bf16", (10, bf), textcoords="offset points", xytext=(-2, 4),
                ha="right", fontsize=8, color=INK)
    ax.set_title(m)
    ax.set_xlabel("% neurons protected at 8-bit (rest 4-bit)")
    ax.set_ylabel("held-out ppl (geo-mean math+code)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_axisbelow(True)
fig.suptitle("E-Q — salient protection: which map finds the neurons that matter?",
             y=1.04, fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(ROOT / "results" / "eq_protection_curves.png", dpi=160, bbox_inches="tight")
print("wrote figure")
