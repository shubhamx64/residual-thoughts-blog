"""Generate theme-matched figures for the Paper 2 mechanism blog post.

Values are extracted from the frozen e5-mechanism result artifacts
(pairset/rows npz for the matched contrast, outcome_half_evals.json plus the
phase-B logs for the selector race, isolation_v2.json for the splice bars).
We render an opaque light and dark asset for each chart because a theme toggle
cannot recolor text baked into a raster image.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "posts" / "why-crowding-protects" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "baseline": ("#6E7480", "#C7CBD3"),
    "weights": ("#2563D8", "#75A7FF"),
    "fisher": ("#C33D38", "#FF8C85"),
    "curvK2": ("#0B7A6B", "#4FD8C2"),
    "random": ("#9AA1AD", "#939AA7"),
}

# Per-layer median coupling ratio, crowded vs matched orthogonal pair-pairs
# (unit normalization, 202 contrasts; e5-mechanism/results/pairset_*.npz +
# rows_*.npz via the analyze_m1 matched-contrast block).
LAYER_RATIOS = {
    2: 4.55, 4: 13.94, 5: 46.28, 6: 9.06, 7: 10.22, 8: 2.34, 9: 16.25,
    10: 4.43, 11: 7.31, 12: 2.02, 13: 3.19, 14: 6.11, 15: 10.48, 16: 4.36,
    17: 3.66, 20: 21.81,
}
MEDIAN_RATIO = 6.12

# Final outcome-half NLL degradation per data seed (after-A NLL 1.4769);
# outcome_half_evals.json + e4-continual/results/log_B_*.jsonl.
SEED_DEG = {
    "baseline": {0: 1.2172, 1: 1.2581, 2: 1.2566, 3: 1.2218, 4: 1.2042},
    "weights": {0: 1.1077, 1: 1.0829, 2: 1.0937, 3: 1.0706, 4: 1.0899},
    "fisher": {0: 0.8741, 1: 0.8665, 2: 0.8534, 3: 0.8505, 4: 0.8668,
               5: 0.8748, 6: 0.8813},
    "curvK2": {1: 0.6833, 2: 0.6865, 3: 0.6699, 4: 0.6909, 5: 0.6674,
               6: 0.6974},
}
ARM_LABELS = {
    "baseline": "no protection",
    "weights": "crowding\n(zero data)",
    "fisher": "Fisher\n(task gradients)",
    "curvK2": "measured coupling\n(K2)",
}

# Isolation v2 splice recovery under audited balance; isolation_v2.json.
ISO = {"high": 0.768, "rand": 0.323, "low": 0.184}


def set_theme(theme: str):
    dark = theme == "dark"
    bg = "#101218" if dark else "#FBFAF6"
    panel = "#151923" if dark else "#FFFEFA"
    ink = "#F3EFE7" if dark else "#1C1F2E"
    muted = "#B7B4AE" if dark else "#66645D"
    grid = "#343A48" if dark else "#DDD9CE"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": bg,
            "savefig.facecolor": bg,
            "axes.facecolor": panel,
            "axes.edgecolor": muted,
            "axes.labelcolor": ink,
            "axes.titlecolor": ink,
            "xtick.color": muted,
            "ytick.color": muted,
            "text.color": ink,
            "axes.grid": True,
            "grid.color": grid,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
        }
    )
    return dark, ink, muted


def contrast(theme: str):
    dark, ink, muted = set_theme(theme)
    teal = COLORS["curvK2"][1 if dark else 0]
    red = COLORS["fisher"][1 if dark else 0]
    layers = sorted(LAYER_RATIOS)
    vals = [LAYER_RATIOS[l] for l in layers]
    x = np.arange(len(layers))

    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ax.bar(x, vals, width=0.62, color=teal, zorder=3)
    ax.axhline(1.0, color=muted, lw=1.1)
    ax.axhline(MEDIAN_RATIO, color=red, lw=1.3, ls="--")
    ax.text(9.5, MEDIAN_RATIO * 1.12, f"median {MEDIAN_RATIO:.2f}x",
            ha="center", fontsize=9, color=red,
            bbox={"facecolor": plt.rcParams["axes.facecolor"],
                  "edgecolor": "none", "pad": 1.2})
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["1x", "2x", "5x", "10x", "20x", "50x"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel("crowded / orthogonal coupling")
    ax.set_title(
        "Crowded pairs carry more curvature coupling in every measured layer",
        fontsize=12.5, fontweight="bold", pad=10,
    )
    ax.set_axisbelow(True)
    fig.text(
        0.5, -0.055,
        "202 caliper-matched pair contrasts (matched on firing rate, gradient "
        "magnitude, weight norm, update norm, co-firing);\n16/16 layers above "
        "parity, one-sided sign p = 1.5e-05, per-layer minimum 2.02",
        ha="center", fontsize=8.3, color=muted,
    )
    fig.savefig(OUT / f"fig_contrast_{theme}.png", dpi=180)
    plt.close(fig)


def selector(theme: str):
    dark, ink, muted = set_theme(theme)
    arms = ["baseline", "weights", "fisher", "curvK2"]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    rng = np.random.default_rng(5)
    for i, arm in enumerate(arms):
        color = COLORS[arm][1 if dark else 0]
        vals = np.array(list(SEED_DEG[arm].values()))
        jx = i + rng.uniform(-0.09, 0.09, len(vals))
        ax.scatter(jx, vals, s=42, color=color, zorder=3,
                   edgecolors=ink, linewidths=0.4)
        ax.hlines(vals.mean(), i - 0.24, i + 0.24, color=color, lw=2.6,
                  zorder=4)
        ax.text(i + 0.28, vals.mean(), f"{vals.mean():+.3f}",
                va="center", fontsize=9.5, color=ink)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ARM_LABELS[a] for a in arms], fontsize=9.5)
    ax.set_ylabel("held-out math NLL degradation\n(outcome half, final)  ↓ better")
    ax.set_title(
        "Measuring the coupling beats every cheaper signal it explains",
        fontsize=12.5, fontweight="bold", pad=10,
    )
    ax.set_axisbelow(True)
    fig.text(
        0.5, -0.03,
        "Each dot is one data seed (freeze top-20%/layer during new-task "
        "training). K2 vs Fisher: 6/6 seeds, exact one-sided p = 0.0156.",
        ha="center", fontsize=8.3, color=muted,
    )
    fig.savefig(OUT / f"fig_selector_{theme}.png", dpi=180)
    plt.close(fig)


def isolation(theme: str):
    dark, ink, muted = set_theme(theme)
    labels = ["high off-diagonal\ncoupling", "size-matched\nrandom",
              "matched low\ncoupling"]
    keys = ["high", "rand", "low"]
    colors = [COLORS["curvK2"][1 if dark else 0],
              COLORS["random"][1 if dark else 0],
              COLORS["baseline"][1 if dark else 0]]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(3)
    vals = [ISO[k] for k in keys]
    ax.bar(x, vals, width=0.56, color=colors, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.018, f"+{v:.3f}", ha="center", fontsize=10.5,
                color=ink, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 0.88)
    ax.set_ylabel("fraction of damage undone\nby restoring the set  ↑ better")
    ax.set_title(
        "The value is off-diagonal specifically",
        fontsize=12.5, fontweight="bold", pad=10,
    )
    ax.set_axisbelow(True)
    fig.text(
        0.5, -0.115,
        "28,443 neurons per set, matched on crowding, self-curvature, "
        "first-order alignment, Fisher, and movement;\naudited on exact "
        "quantities (self-curvature ratio 0.842, slightly adverse to the "
        "high set).",
        ha="center", fontsize=8.3, color=muted,
    )
    fig.savefig(OUT / f"fig_isolation_{theme}.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    for selected_theme in ("light", "dark"):
        contrast(selected_theme)
        selector(selected_theme)
        isolation(selected_theme)
    for path in sorted(OUT.glob("fig_*.png")):
        print("wrote", path.relative_to(ROOT))
