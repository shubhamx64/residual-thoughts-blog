"""motif_legend_figure.py

Produce a compact legend figure for the program motifs used in the draft:

  REINFORCE:   i → i → i
  SHIFT:       i → i → k
  CROSS_COPY:  i → j → j
  RELAY:       i → j → i
  TRANSFORM:   i → j → k
  SUPPRESS:    i → j → −k

Outputs a single PNG (by default: fig03_motif_legend.png).
"""

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt


def _draw_triplet(ax, title: str, labels, edges, suppress_k: bool = False):
    """Draw a 3-node motif.

    labels: tuple of node labels (i_label, j_label, k_label)
    edges: list of (src_idx, dst_idx) with nodes indexed 0,1,2
    """
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.axis("off")

    # fixed triangle layout
    pts = [(0.1, 0.5), (0.55, 0.9), (1.0, 0.5)]

    # nodes
    for idx, (x, y) in enumerate(pts):
        ax.scatter([x], [y], s=700)
        txt = labels[idx]
        if suppress_k and idx == 2:
            txt = "−" + txt
        ax.text(x, y, txt, ha="center", va="center", fontsize=12, color="white")

    # arrows
    for (s, t) in edges:
        x0, y0 = pts[s]
        x1, y1 = pts[t]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=2.2),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig03_motif_legend.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    axes = axes.flatten()

    # REINFORCE: i -> i -> i (we depict as self-loop-ish by i->j and j->k with all labels i)
    _draw_triplet(axes[0], "REINFORCE", ("i", "i", "i"), edges=[(0, 1), (1, 2)])
    _draw_triplet(axes[1], "SHIFT", ("i", "i", "k"), edges=[(0, 1), (1, 2)])
    _draw_triplet(axes[2], "CROSS_COPY", ("i", "j", "j"), edges=[(0, 1), (1, 2)])
    _draw_triplet(axes[3], "RELAY", ("i", "j", "i"), edges=[(0, 1), (1, 2)])
    _draw_triplet(axes[4], "TRANSFORM", ("i", "j", "k"), edges=[(0, 1), (1, 2)])
    _draw_triplet(axes[5], "SUPPRESS", ("i", "j", "k"), edges=[(0, 1), (1, 2)], suppress_k=True)

    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    plt.close()
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
