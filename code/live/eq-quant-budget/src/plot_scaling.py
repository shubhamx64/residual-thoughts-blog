"""E-Q3 figure: 3-bit salience effects across scale (WikiText-2, ratio vs bf16)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
MODELS = ["qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"]
LABELS = ["1.5B", "3B", "7B"]
ARMS = [("gptq3", "GPTQ w3 uniform", "#e34948"),
        ("gptq_fp3", "+ footprint 1% (count)", "#1baf7a"),
        ("gptq_hdiag3", "+ H-diag 1% (energy)", "#2a78d6")]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
})

R = {m: json.load(open(ROOT / "results" / m / "sota_results.json")) for m in MODELS}

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
ax = axes[0]
x = np.arange(len(MODELS))
for key, label, color in ARMS:
    ys = [R[m][key]["wikitext2"] / R[m]["bf16"]["wikitext2"] for m in MODELS]
    ax.plot(x, ys, "o-", lw=2, ms=6, color=color, label=label)
ax.axhline(1.0, color=INK, lw=1.2, ls="--")
ax.annotate("bf16", (x[-1], 1.0), textcoords="offset points", xytext=(-2, 5),
            ha="right", fontsize=8, color=INK)
ax.set_xticks(x); ax.set_xticklabels(LABELS)
ax.set_xlabel("model size"); ax.set_ylabel("WikiText-2 ppl ratio vs bf16")
ax.set_title("3-bit damage and what salience protection recovers")
ax.legend(frameon=False, fontsize=8); ax.set_axisbelow(True)

ax = axes[1]
gaps = []
for m in MODELS:
    fp = R[m]["gptq_fp3"]["wikitext2"]
    hd = R[m]["gptq_hdiag3"]["wikitext2"]
    un = R[m]["gptq3"]["wikitext2"]
    bf = R[m]["bf16"]["wikitext2"]
    gaps.append(((un - fp) / (un - bf + 1e-9), (un - hd) / (un - bf + 1e-9)))
gaps = np.array(gaps)
w = 0.35
ax.bar(x - w / 2, 100 * gaps[:, 0], w, color="#1baf7a", label="footprint (count)")
ax.bar(x + w / 2, 100 * gaps[:, 1], w, color="#2a78d6", label="H-diag (energy)")
ax.set_xticks(x); ax.set_xticklabels(LABELS)
ax.set_xlabel("model size"); ax.set_ylabel("% of 3-bit damage recovered")
ax.set_title("Count vs energy salience: the gap closes with scale")
ax.legend(frameon=False, fontsize=8); ax.set_axisbelow(True)

fig.suptitle("E-Q3 — salience-guided 3-bit quantization across scale (Qwen2.5)",
             y=1.04, fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(ROOT / "results" / "eq3_scaling.png", dpi=160, bbox_inches="tight")
print("wrote", ROOT / "results" / "eq3_scaling.png")
