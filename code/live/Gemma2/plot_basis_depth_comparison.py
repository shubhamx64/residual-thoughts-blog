"""
Depth-profile comparison: SAE basis vs token-embedding basis.

Parses two validation reports (generate_report format) and plots per-layer
routing-validation quality for both probe bases. Hypothesis under test:
token basis better early (residual ~ embeddings) and late (residual ~
logits / logit-lens regime), SAE basis better in the mid-stack bottleneck.

Usage:
    python plot_basis_depth_comparison.py --sae-report validation_report_postfix.txt \
        --token-report validation_report_token.txt [--out figs/basis_depth_comparison.png]
"""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_validation_report import parse_report


def per_layer_stats(heads):
    """Aggregate HeadStats per layer: mean/max overall Spearman, mean local/mid."""
    by_layer = defaultdict(list)
    for h in heads:
        by_layer[h.layer].append(h)

    stats = {}
    for layer, hs in sorted(by_layer.items()):
        spearmans = [h.spearman for h in hs if h.spearman is not None]
        locals_ = [h.local_s for h in hs if h.local_s is not None]
        mids = [h.mid_s for h in hs if h.mid_s is not None]
        stats[layer] = {
            "mean_s": float(np.mean(spearmans)) if spearmans else np.nan,
            "max_s": float(np.max(spearmans)) if spearmans else np.nan,
            "mean_local": float(np.mean(locals_)) if locals_ else np.nan,
            "mean_mid": float(np.mean(mids)) if mids else np.nan,
            "n_heads": len(hs),
        }
    return stats


def plot_panel(ax, sae_stats, token_stats, key, title):
    for stats, label, color in [(sae_stats, "SAE basis", "tab:blue"),
                                (token_stats, "Token basis", "tab:orange")]:
        layers = sorted(stats.keys())
        vals = [stats[l][key] for l in layers]
        ax.plot(layers, vals, marker="o", label=label, color=color)
    ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman ρ")
    ax.legend()
    ax.grid(alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sae-report", required=True)
    parser.add_argument("--token-report", required=True)
    parser.add_argument("--out", default="figs/basis_depth_comparison.png")
    args = parser.parse_args()

    sae_stats = per_layer_stats(parse_report(args.sae_report))
    token_stats = per_layer_stats(parse_report(args.token_report))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plot_panel(axes[0, 0], sae_stats, token_stats, "mean_s", "Mean overall Spearman (per layer)")
    plot_panel(axes[0, 1], sae_stats, token_stats, "max_s", "Max overall Spearman (per layer)")
    plot_panel(axes[1, 0], sae_stats, token_stats, "mean_local", "Mean local Spearman (0-4 tokens)")
    plot_panel(axes[1, 1], sae_stats, token_stats, "mean_mid", "Mean mid Spearman (16-32 tokens)")
    fig.suptitle("Routing validation by probe basis: SAE vs token-embedding", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")

    # Per-layer delta table
    all_layers = sorted(set(sae_stats) | set(token_stats))
    print(f"\n{'layer':>6} {'sae_mean_S':>11} {'token_mean_S':>13} {'delta':>8}   better")
    for l in all_layers:
        s = sae_stats.get(l, {}).get("mean_s", np.nan)
        t = token_stats.get(l, {}).get("mean_s", np.nan)
        delta = t - s
        better = "token" if delta > 0 else "sae"
        print(f"{l:>6} {s:>11.4f} {t:>13.4f} {delta:>+8.4f}   {better}")


if __name__ == "__main__":
    main()
