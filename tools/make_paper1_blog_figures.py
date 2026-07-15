"""Generate theme-matched figures for the Paper 1 blog post.

The values below are the five-seed means and 95% bootstrap intervals reported in
the current Paper 1 manuscript.  We render an opaque light and dark asset for
each chart because a theme toggle cannot recolor text baked into a raster image.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "posts" / "how-far-can-you-read-a-model-from-its-weights" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = ["baseline", "random", "weights", "footprint", "join", "fisher"]
COLORS = {
    "baseline": ("#6E7480", "#C7CBD3"),
    "random": ("#9AA1AD", "#939AA7"),
    "weights": ("#2563D8", "#75A7FF"),
    "footprint": ("#087F6E", "#55D6BE"),
    "join": ("#7556B8", "#BEA4FF"),
    "fisher": ("#C33D38", "#FF8C85"),
}
MARKERS = {
    "baseline": "X",
    "random": "o",
    "weights": "s",
    "footprint": "D",
    "join": "P",
    "fisher": "*",
}

DATA = {
    "TinyLlama: clean step-100 window": {
        "baseline": (25.4, 2.1, 2.2, 96.3, 4.5, 4.0),
        "random": (33.9, 1.8, 1.5, 77.3, 3.2, 3.3),
        "weights": (38.4, 1.2, 1.1, 57.4, 1.3, 1.2),
        "footprint": (39.4, 1.0, 1.0, 51.1, 1.0, 0.9),
        "join": (39.5, 0.8, 0.9, 51.5, 0.8, 0.6),
        "fisher": (39.5, 0.8, 0.7, 42.7, 0.5, 0.5),
    },
    "Qwen: validation-gated endpoint": {
        "baseline": (3.510, 0.022, 0.022, 0.320, 0.021, 0.021),
        "random": (2.915, 0.037, 0.026, 0.228, 0.015, 0.022),
        "weights": (2.226, 0.007, 0.008, 0.147, 0.011, 0.012),
        "footprint": (2.562, 0.017, 0.016, 0.212, 0.023, 0.021),
        "join": (2.556, 0.007, 0.007, 0.217, 0.017, 0.013),
        "fisher": (2.118, 0.023, 0.021, 0.095, 0.033, 0.026),
    },
}


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
    return dark, bg, panel, ink, muted, grid


def tradeoff(theme: str):
    dark, _bg, _panel, ink, _muted, _grid = set_theme(theme)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.35))
    fig.suptitle(
        "Retention gains have to be read beside new-task learning",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    limits = [((22.2, 41.3), (39, 103)), ((2.02, 3.61), (0.05, 0.355))]
    label_offsets = {
        0: {
            "baseline": (0.5, 2.0),
            "random": (0.45, 2.4),
            "weights": (0.45, 2.0),
            "footprint": (0.45, 3.1),
            "join": (-1.7, -4.5),
            "fisher": (0.45, -4.2),
        },
        1: {
            "baseline": (0.045, 0.012),
            "random": (0.045, 0.012),
            "weights": (0.045, 0.010),
            "footprint": (0.06, -0.017),
            "join": (0.055, 0.017),
            "fisher": (0.055, 0.008),
        },
    }

    for idx, (ax, (title, vals)) in enumerate(zip(axes, DATA.items())):
        for arm in ARMS:
            x, xlo, xhi, y, ylo, yhi = vals[arm]
            color = COLORS[arm][1 if dark else 0]
            ax.errorbar(
                x,
                y,
                xerr=np.array([[xlo], [xhi]]),
                yerr=np.array([[ylo], [yhi]]),
                fmt=MARKERS[arm],
                ms=9 if arm != "fisher" else 12,
                color=color,
                mec=ink if arm in {"baseline", "random"} else color,
                mew=0.7,
                ecolor=color,
                elinewidth=1.2,
                capsize=3,
                zorder=3,
            )
            dx, dy = label_offsets[idx][arm]
            ax.text(x + dx, y + dy, arm, fontsize=9.5, color=ink)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xlim(*limits[idx][0])
        ax.set_ylim(*limits[idx][1])
        ax.set_xlabel("held-out code improvement (%)  → better")
        if idx == 0:
            ax.set_ylabel("held-out math degradation (%)  ↓ better")
        ax.set_axisbelow(True)
        ax.annotate(
            "preferred direction",
            xy=(0.96, 0.08),
            xytext=(0.72, 0.19),
            xycoords="axes fraction",
            textcoords="axes fraction",
            fontsize=8.5,
            color=COLORS["footprint"][1 if dark else 0],
            arrowprops={"arrowstyle": "->", "color": COLORS["footprint"][1 if dark else 0]},
        )
    fig.subplots_adjust(wspace=0.29, top=0.82, bottom=0.16)
    fig.savefig(OUT / f"fig_tradeoff_{theme}.png", dpi=180)
    plt.close(fig)


def scale_alignment(theme: str):
    dark, _bg, _panel, ink, muted, _grid = set_theme(theme)
    models = ["1.5B", "3B", "7B", "14B"]
    rho = np.array([0.12, 0.10, 0.17, 0.13])
    jac = np.array([0.20, 0.17, 0.18, 0.17])
    x = np.arange(len(models))
    blue = COLORS["weights"][1 if dark else 0]
    violet = COLORS["join"][1 if dark else 0]
    red = COLORS["fisher"][1 if dark else 0]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.45))
    fig.suptitle(
        "The zero-data proxy does not tighten across the tested Qwen sizes",
        fontsize=13.5,
        fontweight="bold",
        y=1.03,
    )
    axes[0].plot(x, rho, marker="o", ms=7, lw=2, color=blue, zorder=3)
    axes[0].axhline(0, color=muted, lw=0.8)
    axes[0].set_ylim(0, 0.23)
    axes[0].set_title("Crowding vs Fisher rank alignment", fontsize=10.5, fontweight="bold")
    axes[0].set_ylabel("median partial Spearman ρ")
    for xi, value in zip(x, rho):
        axes[0].text(xi, value + 0.012, f"{value:.2f}", ha="center", fontsize=8.5, color=ink)

    axes[1].plot(x, jac, marker="s", ms=7, lw=2, color=violet, zorder=3)
    axes[1].axhline(0.11, color=red, lw=1.2, ls="--")
    axes[1].text(2.95, 0.114, "chance floor 0.11", ha="right", va="bottom", fontsize=8, color=red)
    axes[1].set_ylim(0.08, 0.24)
    axes[1].set_title("Top-20% salient-set overlap", fontsize=10.5, fontweight="bold")
    axes[1].set_ylabel("Jaccard")
    for xi, value in zip(x, jac):
        axes[1].text(xi, value + 0.010, f"{value:.2f}", ha="center", fontsize=8.5, color=ink)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_xlabel("Qwen2.5 checkpoint size")
        ax.set_axisbelow(True)
    fig.subplots_adjust(wspace=0.34, top=0.78, bottom=0.2)
    fig.savefig(OUT / f"fig_scale_{theme}.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    for selected_theme in ("light", "dark"):
        tradeoff(selected_theme)
        scale_alignment(selected_theme)
    for path in sorted(OUT.glob("fig_*.png")):
        print("wrote", path.relative_to(ROOT))
