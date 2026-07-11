"""Report figures from metrics/clustering outputs. Static PNG, light surface."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common import CLASSES, result_dir

INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
# hue = domain (blue math / aqua code / yellow prose); light step + triangle = prose-surface form
CLASS_COLOR = {"math": "#256abf", "math_prose": "#86b6ef",
               "code": "#0e8a5f", "code_prose": "#8bd9bb", "prose": "#eda100"}
CLASS_MARK = {"math": "o", "math_prose": "^", "code": "o", "code_prose": "^", "prose": "s"}
DIV = LinearSegmentedColormap.from_list(
    "bwr_ref", ["#0d366b", "#3987e5", "#f0efec", "#e66767", "#8f1f1f"])

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
})


def nice(ax):
    ax.set_axisbelow(True)


def fig_depth_profile(M, out):
    pl = M["per_layer"]
    L = np.arange(M["n_layers"])
    within = np.array([p["within_ccos"] for p in pl])
    across = np.array([p["across_ccos"] for p in pl])
    margin = within - across
    noise_std = np.array([p["noise_std"] for p in pl])
    j256 = np.array([p["margin_j256"] for p in pl])
    acc = np.array(M["classification"]["per_layer_acc5"])

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    ax = axes[0]
    ax.plot(L, within, color="#256abf", lw=2, label="within regime")
    ax.plot(L, across, color="#e34948", lw=2, label="across regimes")
    ax.set_title("Half-footprint similarity by depth (centered cosine)")
    ax.set_xlabel("layer"); ax.set_ylabel("centered cosine")
    ax.legend(frameon=False, fontsize=9); nice(ax)

    ax = axes[1]
    ax.plot(L, margin, color=INK, lw=2, label="margin (ccos)")
    ax.fill_between(L, -3 * noise_std, 3 * noise_std, color=GRID, label="±3σ noise floor")
    ax.plot(L, j256, color="#0e8a5f", lw=2, label="margin (Jaccard top-256)")
    sh_x = [int(k) for k in M["shuffle"]]
    sh = np.array([M["shuffle"][str(k)] for k in sh_x])
    ax.errorbar(sh_x, sh[:, 0], yerr=2 * sh[:, 1], fmt="o", ms=5, color=MUTED,
                capsize=3, label="shuffled labels")
    ax.set_title("Within−across margin vs controls")
    ax.set_xlabel("layer"); ax.legend(frameon=False, fontsize=8); nice(ax)

    ax = axes[2]
    ax.plot(L, 100 * acc, color="#256abf", lw=2)
    ax.axhline(20, color=MUTED, lw=1, ls="--")
    ax.annotate("chance (5-class)", (L[-1], 20), textcoords="offset points",
                xytext=(-4, 5), ha="right", fontsize=8, color=MUTED)
    ax.set_title("Per-sequence fingerprint decodability")
    ax.set_xlabel("layer (single-layer classifier)"); ax.set_ylabel("accuracy %")
    ax.set_ylim(0, 104); nice(ax)

    fig.suptitle(f"{M['model']} — footprint stability & separability", y=1.03,
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "depth_profile.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_heatmap(M, out):
    keys = M["heatmap_keys"]
    layers = sorted(M["heatmaps"], key=int)
    fig, axes = plt.subplots(1, len(layers), figsize=(3.1 * len(layers), 3.6))
    for ax, l in zip(np.atleast_1d(axes), layers):
        H = np.array(M["heatmaps"][l])
        im = ax.imshow(H, cmap=DIV, vmin=-1, vmax=1)
        ax.set_xticks(range(len(keys))); ax.set_yticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=90, fontsize=7)
        ax.set_yticklabels(keys if l == layers[0] else [], fontsize=7)
        ax.set_title(f"layer {l}", fontsize=10)
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="centered cosine")
    fig.suptitle(f"{M['model']} — half-footprint similarity (5 classes × 2 halves)",
                 y=1.02, fontsize=12, fontweight="bold")
    fig.savefig(out / "similarity_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_placement(M, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (contrast, sib_color) in zip(axes, [("math_prose", "#256abf"),
                                                ("code_prose", "#0e8a5f")]):
        p = M["contrast_placement"][contrast]
        L = np.arange(len(p["per_layer_to_prose"]))
        ax.plot(L, p["per_layer_to_sibling"], color=sib_color, lw=2,
                label=f"to {p['sibling']} (computation sibling)")
        ax.plot(L, p["per_layer_to_prose"], color="#eda100", lw=2, label="to prose (surface sibling)")
        ax.axhline(p["token_to_sibling"], color=sib_color, lw=1.2, ls="--")
        ax.axhline(p["token_to_prose"], color="#eda100", lw=1.2, ls="--")
        ax.annotate("dashed = token-unigram baseline", (0.02, 0.02),
                    xycoords="axes fraction", fontsize=8, color=MUTED)
        ax.set_title(f"{contrast}: which neighbor?")
        ax.set_xlabel("layer"); nice(ax)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_ylabel("centered cosine to neighbor")
    fig.suptitle(f"{M['model']} — contrast-class placement: computation vs surface",
                 y=1.04, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "contrast_placement.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_scatter(model, out):
    z = np.load(result_dir(model) / "cluster_coords_all5.npz", allow_pickle=True)
    coords, labels = z["coords"], z["labels"]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for k, cls in enumerate(CLASSES):
        pts = coords[labels == k]
        ax.scatter(pts[:, 0], pts[:, 1], s=22, c=CLASS_COLOR[cls],
                   marker=CLASS_MARK[cls], label=cls, edgecolors="white", linewidths=0.4)
    ax.set_title(f"{model} — per-sequence footprints (SVD of firing vectors, first 2 comps)")
    ax.set_xlabel("component 1"); ax.set_ylabel("component 2")
    ax.legend(frameon=False, fontsize=9)
    nice(ax)
    fig.tight_layout()
    fig.savefig(out / "footprint_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantile", type=float, default=99.0)
    args = ap.parse_args()
    rd = result_dir(args.model)
    with open(rd / f"metrics_q{args.quantile}.json") as f:
        M = json.load(f)
    out = rd / "figures"
    fig_depth_profile(M, out)
    fig_heatmap(M, out)
    fig_placement(M, out)
    fig_scatter(args.model, out)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
