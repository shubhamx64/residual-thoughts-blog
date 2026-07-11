"""Generate the three comparison figures for the errata/follow-up blog post.

Spearman numbers are taken verbatim from errata_full_depth.md (the corrected
2026-06-10 run); the diagonal-softmax-mass panel is computed directly from
analysis_outputs/analysis_20260610_233407.json (verified L10 = 0.124).

Outputs land directly in the new post's figs/ directory.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(
    HERE, "..", "residual-thoughts-blog", "posts",
    "correcting-the-weight-space-map", "figs"))
os.makedirs(OUT, exist_ok=True)

# --- per-layer mean Spearman, verbatim from errata_full_depth.md -------------
layers = list(range(26))
pre = [None, 0.051, 0.039, 0.064, -0.032, 0.143, 0.101, -0.009, 0.145, -0.156,
       -0.193, 0.175, -0.184, -0.248, 0.123, -0.055, -0.113, 0.113, 0.006,
       0.020, -0.000, 0.077, 0.030, -0.007, -0.064, 0.244]
sae = [0.286, 0.062, 0.073, 0.082, 0.115, 0.289, 0.254, 0.141, 0.274, 0.255,
       0.164, 0.352, 0.311, 0.022, 0.362, 0.186, 0.238, 0.329, 0.314, 0.178,
       0.202, 0.253, 0.316, 0.340, 0.334, 0.220]
tok = [0.109, 0.093, -0.007, 0.130, 0.074, 0.176, 0.227, 0.318, 0.249, 0.375,
       0.351, 0.401, 0.329, 0.377, 0.336, 0.261, 0.379, 0.363, 0.352, 0.277,
       0.294, 0.217, 0.179, 0.298, 0.327, 0.288]

INK = "#1a1a1a"
RED = "#c0392b"
BLUE = "#2c6fbb"
GREEN = "#2e8b57"
GREY = "#9aa0a6"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "figure.dpi": 140,
})

# === Figure 1: pre-fix vs fixed per-layer Spearman ==========================
fig, ax = plt.subplots(figsize=(8.4, 4.4))
ax.axhline(0, color="#888888", lw=1.0, zorder=1)
pre_x = [L for L, v in zip(layers, pre) if v is not None]
pre_y = [v for v in pre if v is not None]
ax.plot(pre_x, pre_y, "o-", color=GREY, lw=1.6, ms=4,
        label="pre-fix (buggy pipeline)", zorder=2)
ax.plot(layers, sae, "o-", color=BLUE, lw=2.0, ms=4.5,
        label="fixed, SAE basis", zorder=3)
ax.fill_between(pre_x, pre_y, 0, where=[y < 0 for y in pre_y],
                color=RED, alpha=0.10, zorder=1)
ax.set_xlabel("layer")
ax.set_ylabel(r"mean Spearman $\rho$  (predicted vs real attention)")
ax.set_title("The late-layer 'anti-predictive regime' was an artifact",
             fontsize=12, color=INK)
ax.set_xticks(range(0, 26, 2))
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.text(0.99, 0.03,
        "grand mean: pre-fix +0.011 (noise)  →  fixed +0.229",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8.5, color="#555555")
ax.grid(axis="y", color="#eeeeee", lw=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_prefix_vs_fixed_spearman.png"),
            bbox_inches="tight")
plt.close(fig)

# === Figure 2: dual-basis SAE vs token, per layer ===========================
fig, ax = plt.subplots(figsize=(8.4, 4.4))
x = np.arange(26)
w = 0.4
ax.bar(x - w / 2, sae, w, color=BLUE, label="Gemma Scope SAE basis")
ax.bar(x + w / 2, tok, w, color=GREEN, label="token-embedding basis (no SAE)")
ax.axhline(0, color="#888888", lw=0.8)
# mark L13, where the bases disagree sharply
ax.annotate("L13: SAE checkpoint\nfails, free basis fine",
            xy=(13 + w / 2, tok[13]), xytext=(15.5, 0.45),
            fontsize=8, color="#555555",
            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))
ax.set_xlabel("layer")
ax.set_ylabel(r"mean Spearman $\rho$")
ax.set_title("Dual basis: a dictionary-free basis is competitive (often better)",
             fontsize=12, color=INK)
ax.set_xticks(range(0, 26, 2))
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.text(0.99, 0.97,
        "grand mean: SAE 0.229  vs  token 0.260  (token wins 15/26)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8.5, color="#555555")
ax.grid(axis="y", color="#eeeeee", lw=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_dual_basis.png"), bbox_inches="tight")
plt.close(fig)

# === Figure 3: identity-routing (diagonal softmax mass) by layer ============
d = json.load(open(os.path.join(
    HERE, "analysis_outputs", "analysis_20260610_233407.json")))
lr = d["layer_results"]
diag = []
for L in range(26):
    heads = lr[str(L)]["routing_results"]
    diag.append(sum(h["metrics"]["diagonal_softmax_mass"]
                    for h in heads) / len(heads))
fig, ax = plt.subplots(figsize=(8.4, 4.0))
bars = ax.bar(range(26), diag, color="#b9c4d4")
peak = int(np.argmax(diag))
bars[peak].set_color(RED)
bars[6].set_color("#e0a800")
ax.set_xlabel("layer")
ax.set_ylabel("mean diagonal softmax mass")
ax.set_title("Identity/copy-routing peak moved from L6 to L10",
             fontsize=12, color=INK)
ax.set_xticks(range(0, 26, 2))
ax.annotate(f"L10 = {diag[10]:.3f}", xy=(10, diag[10]),
            xytext=(12.5, diag[10] + 0.005), fontsize=9, color=RED)
ax.annotate("L6 (old peak)", xy=(6, diag[6]),
            xytext=(1.5, diag[6] + 0.02), fontsize=8.5, color="#9a7b00",
            arrowprops=dict(arrowstyle="->", color="#c9a400", lw=0.8))
ax.grid(axis="y", color="#eeeeee", lw=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_diag_mass_by_layer.png"),
            bbox_inches="tight")
plt.close(fig)

print("wrote 3 figures to", OUT)
print("L10 diag mass =", round(diag[10], 4), " L6 =", round(diag[6], 4))
