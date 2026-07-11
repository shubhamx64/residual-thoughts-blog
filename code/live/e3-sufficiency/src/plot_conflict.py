"""Figure: distinctive-substrate regime-conflict matrices, 4 models."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common_e3 import result_dir, ROOT
from common import CLASSES

INK, MUTED, SURF = "#0b0b0b", "#898781", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f4f7fb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
MODELS = ["qwen2.5-1.5b", "gemma-2-2b", "pythia-1.4b", "tinyllama-1.1b"]
LBL = [c.replace("_prose", "-pr") for c in CLASSES]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
})

fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, m in zip(axes, MODELS):
    with open(result_dir(m) / "conflict_matrix_distinctive.json") as f:
        C = np.array(json.load(f)["normalized_conflict_layer_avg"])
    Cm = C.copy()
    np.fill_diagonal(Cm, np.nan)  # off-diagonals carry the story
    im = ax.imshow(Cm, cmap=SEQ, vmin=0.15, vmax=0.6)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(LBL, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(LBL if m == MODELS[0] else [], fontsize=8)
    ax.set_title(m, fontsize=10)
    for a in range(5):
        for b in range(5):
            if a != b:
                ax.text(b, a, f"{C[a, b]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if C[a, b] > 0.45 else INK)
fig.colorbar(im, ax=axes, shrink=0.8, label="normalized conflict (distinctive substrate)")
fig.suptitle("Predicted regime-conflict matrices — packing table × class footprints "
             "(weights + one cheap capture)", y=1.06, fontsize=12, fontweight="bold")
fig.savefig(ROOT / "results" / "conflict_matrices.png", dpi=160, bbox_inches="tight")
print("wrote", ROOT / "results" / "conflict_matrices.png")
