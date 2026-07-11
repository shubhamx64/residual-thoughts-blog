"""E2 figures: packing depth profiles, coherence vs Welch, gain maps, and the
E1 bridge (does weight-space packing predict footprint separation depth?)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
E1_RESULTS = ROOT.parent / "e1-footprint-stability" / "results"

INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
C = {"mlp_write": "#2a78d6", "mlp_read": "#1baf7a",
     "attn_write": "#eda100", "attn_read_q": "#4a3aa7"}
SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
})


def fig_packing(M, out):
    L = np.arange(M["n_layers"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    for key, label in [("mlp_write", "MLP write (down cols)"),
                       ("mlp_read", "MLP read (gate rows)"),
                       ("attn_write", "attn write (o cols)"),
                       ("attn_read_q", "attn read (q rows)")]:
        ax.plot(L, [p[key]["fp_ratio"] for p in M["per_layer"]], color=C[key], lw=2, label=label)
    ax.axhline(M["token_dict"]["fp_ratio"], color=MUTED, lw=1.4, ls="--")
    ax.annotate("token dictionary", (L[-1], M["token_dict"]["fp_ratio"]),
                textcoords="offset points", xytext=(-4, 5), ha="right", fontsize=8, color=MUTED)
    ax.set_title("Packing efficiency by depth (frame-potential ratio)")
    ax.set_xlabel("layer"); ax.set_ylabel("FP_min / FP  (1 = tight frame)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8); ax.set_axisbelow(True)

    ax = axes[1]
    q99 = [p["mlp_write"]["q99"] for p in M["per_layer"]]
    coh = [p["mlp_write"]["coherence_max"] for p in M["per_layer"]]
    welch = [p["mlp_write"]["welch_bound"] for p in M["per_layer"]]
    ax.plot(L, coh, color="#e34948", lw=2, label="max coherence")
    ax.plot(L, q99, color="#2a78d6", lw=2, label="|cos| q99")
    ax.plot(L, welch, color=INK, lw=1.4, ls="--", label="Welch bound")
    ax.set_yscale("log")
    ax.set_title("MLP write dictionary: overlap vs the packing floor")
    ax.set_xlabel("layer"); ax.set_ylabel("|cos| (log)")
    ax.legend(frameon=False, fontsize=8); ax.set_axisbelow(True)
    fig.suptitle(f"{M['model']} — weight-space packing", y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "packing_depth.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_gain(M, out):
    pl = M["per_layer"]
    L = np.arange(M["n_layers"])
    qk = np.array([p["gain"]["qk"] for p in pl])
    ov = np.array([p["gain"]["ov"] for p in pl])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, mat, name in ((axes[0], qk, "σ_max(QK)/√d_h — routing sharpness"),
                          (axes[1], ov, "σ_max(OV) — value-path gain")):
        im = ax.imshow(mat.T, aspect="auto", cmap=SEQ, origin="lower")
        ax.set_xlabel("layer"); ax.set_ylabel("head")
        ax.set_title(name, fontsize=10)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.85)
    ax = axes[2]
    g = np.cumprod([(1 + p["gain"]["g_attn"]) * (1 + p["gain"]["g_mlp"]) for p in pl])
    ax.plot(L, g, color="#2a78d6", lw=2)
    ax.set_yscale("log")
    ax.set_title("Cumulative gain bound Π(1+g) — heuristic upper map", fontsize=10)
    ax.set_xlabel("layer"); ax.set_axisbelow(True)
    fig.suptitle(f"{M['model']} — gain map (weights only)", y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "gain_map.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_bridge(M, out):
    """Per-layer weight-space packing vs E1 footprint separation."""
    e1_path = E1_RESULTS / M["model"] / "metrics_q99.0.json"
    if not e1_path.exists():
        return None
    with open(e1_path) as f:
        e1 = json.load(f)
    j256 = np.array([p["margin_j256"] for p in e1["per_layer"]])
    fp = np.array([p["mlp_write"]["fp_ratio"] for p in M["per_layer"]])
    n = min(len(j256), len(fp))
    j256, fp = j256[:n], fp[:n]
    r = float(np.corrcoef(fp, j256)[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sc = ax.scatter(fp, j256, c=np.arange(n), cmap=SEQ, s=42, edgecolors="white", linewidths=0.5)
    fig.colorbar(sc, ax=ax, label="layer")
    ax.set_xlabel("packing efficiency (MLP write, FP ratio)")
    ax.set_ylabel("E1 footprint separation (Jaccard-256 margin)")
    ax.set_title(f"{M['model']} — packing vs footprint separation (r = {r:+.2f})")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "e1_bridge.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    rd = ROOT / "results" / args.model
    with open(rd / "e2_metrics.json") as f:
        M = json.load(f)
    out = rd / "figures"
    out.mkdir(exist_ok=True)
    fig_packing(M, out)
    fig_gain(M, out)
    r = fig_bridge(M, out)
    print(f"figures -> {out}" + (f" | bridge r = {r:+.3f}" if r is not None else ""))


if __name__ == "__main__":
    main()
