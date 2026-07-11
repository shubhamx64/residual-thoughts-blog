"""
Generate assets for the Gemma-2 weight-space blog post.

New outputs include:
1. Validation statistics & visualizations (auto-detect newest report).
2. OV_f cosine stability plots (highlights sign flips).
3. RoPE stability curves / heatmaps (unchanged).
4. Copy-dominance and program-distribution diagnostics.

Reads from:
- full_feature_test*.txt (preferred) or text_outputs/validation_long_prompts_all_layers.txt
- analysis_outputs/analysis_*.json
"""

import json
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Resolve directories
ROOT = Path(__file__).resolve().parent
TEXT_OUTPUTS = ROOT / "text_outputs"
ANALYSIS_OUTPUTS = ROOT / "analysis_outputs"
OUTPUT_DIR = ROOT / "figs" / "blog"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PART 1: Parse validation report and compute summary statistics
# ============================================================================

@dataclass
class HeadValidationStats:
    """Validation statistics for a single head."""
    layer: int
    head: int
    overall_pearson: float
    overall_spearman: float
    local_spearman: float  # 0-4 bin
    mid_spearman: float    # 16-32 bin
    long_spearman: float   # 128-256 bin
    far_spearman: float    # 256+ bin
    sign_stability: float
    n_pairs: int


@dataclass
class OVWriteStats:
    layer: int
    head: int
    cosine: float
    ci_low: float
    ci_high: float
    std: float
    positions: int


def resolve_validation_report() -> Tuple[Optional[Path], Optional[str]]:
    """Return the best available validation report path and a short label."""
    candidates = sorted(ROOT.glob("full_feature_test*.txt"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1], "full_feature"
    fallback = TEXT_OUTPUTS / "validation_long_prompts_all_layers.txt"
    if fallback.exists():
        return fallback, "legacy_validation"
    return None, None


def resolve_latest_analysis_json() -> Optional[Path]:
    """Pick the newest analysis_*.json file."""
    candidates = sorted(ANALYSIS_OUTPUTS.glob("analysis_*.json"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]
    return None


def parse_validation_report(report_path: Path) -> Tuple[List[HeadValidationStats], List[OVWriteStats]]:
    """Parse the validation report and extract per-head statistics + OV_f cosines."""
    with report_path.open('r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Split by head sections (QK validation blocks)
    head_sections = re.split(r'={60,}\n(Layer \d+, Head \d+)\n={60,}', content)

    results: List[HeadValidationStats] = []
    i = 1
    while i < len(head_sections):
        header = head_sections[i]
        section = head_sections[i + 1] if i + 1 < len(head_sections) else ""
        i += 2

        match = re.match(r'Layer (\d+), Head (\d+)', header)
        if not match:
            continue
        layer, head = int(match.group(1)), int(match.group(2))

        pearson_match = re.search(r'Overall Pearson r:\s+([-\d.]+)', section)
        spearman_match = re.search(r'Overall Spearman r:\s+([-\d.]+)', section)

        if not pearson_match or not spearman_match:
            continue

        overall_pearson = float(pearson_match.group(1))
        overall_spearman = float(spearman_match.group(1))

        pairs_match = re.search(r'Total pairs:\s+([\d,]+)', section)
        n_pairs = int(pairs_match.group(1).replace(',', '')) if pairs_match else 0

        local_match = re.search(r'Local\s+\(0-4\):\s+([-\d.]+)', section)
        mid_match = re.search(r'Mid\s+\(16-32\):\s+([-\d.]+)', section)
        long_match = re.search(r'Long\s+\(128-256\):\s+([-\d.]+)', section)
        far_match = re.search(r'256\+\s+\(256\+\):\s+([-\d.]+)', section)

        local_spearman = float(local_match.group(1)) if local_match else 0.0
        mid_spearman = float(mid_match.group(1)) if mid_match else 0.0
        long_spearman = float(long_match.group(1)) if long_match else 0.0
        far_spearman = float(far_match.group(1)) if far_match else 0.0

        sign_match = re.search(r'Sign stability:\s+([\d.]+)%', section)
        sign_stability = float(sign_match.group(1)) if sign_match else 0.0

        results.append(HeadValidationStats(
            layer=layer,
            head=head,
            overall_pearson=overall_pearson,
            overall_spearman=overall_spearman,
            local_spearman=local_spearman,
            mid_spearman=mid_spearman,
            long_spearman=long_spearman,
            far_spearman=far_spearman,
            sign_stability=sign_stability,
            n_pairs=n_pairs
        ))

    ov_entries: List[OVWriteStats] = []
    lines = content.splitlines()
    ov_header_re = re.compile(r'^Layer\s+(\d+),\s*Head\s+(\d+):\s*$')
    cos_re = re.compile(r'Cosine similarity:\s*([-\d.]+)\s+95% CI:\s*\[([-\d.]+),\s*([-\d.]+)\]')
    std_re = re.compile(r'Std deviation:\s*([-\d.]+)')
    pos_re = re.compile(r'Positions:\s*([\d,]+)')

    for idx, line in enumerate(lines):
        m = ov_header_re.match(line.strip())
        if not m:
            continue
        if not line.strip().endswith(':'):
            continue
        layer = int(m.group(1))
        head = int(m.group(2))
        if idx + 3 >= len(lines):
            continue
        cos_line = lines[idx + 1].strip()
        std_line = lines[idx + 2].strip()
        pos_line = lines[idx + 3].strip()
        cos_match = cos_re.search(cos_line)
        std_match = std_re.search(std_line)
        pos_match = pos_re.search(pos_line)
        if not (cos_match and std_match and pos_match):
            continue
        ov_entries.append(OVWriteStats(
            layer=layer,
            head=head,
            cosine=float(cos_match.group(1)),
            ci_low=float(cos_match.group(2)),
            ci_high=float(cos_match.group(3)),
            std=float(std_match.group(1)),
            positions=int(pos_match.group(1).replace(',', ''))
        ))

    return results, ov_entries


def compute_summary_statistics(stats: List[HeadValidationStats]) -> dict:
    """Compute aggregate statistics across all heads."""
    if not stats:
        return {}

    pearsons = [s.overall_pearson for s in stats]
    spearmans = [s.overall_spearman for s in stats]
    local_spearmans = [s.local_spearman for s in stats]
    sign_stabs = [s.sign_stability for s in stats]
    structured_heads = [s for s in stats if abs(s.local_spearman) > 0.1]
    structured_local = [s.local_spearman for s in structured_heads]
    high_stability_heads = [s for s in stats if s.sign_stability > 80]

    return {
        'n_heads_total': len(stats),
        'n_heads_structured': len(structured_heads),
        'n_heads_high_stability': len(high_stability_heads),
        'overall_pearson_mean': np.mean(pearsons),
        'overall_pearson_std': np.std(pearsons),
        'overall_pearson_median': np.median(pearsons),
        'overall_pearson_max': np.max(pearsons),
        'overall_spearman_mean': np.mean(spearmans),
        'overall_spearman_std': np.std(spearmans),
        'overall_spearman_median': np.median(spearmans),
        'overall_spearman_max': np.max(spearmans),
        'local_spearman_mean': np.mean(local_spearmans),
        'local_spearman_std': np.std(local_spearmans),
        'local_spearman_max': np.max(local_spearmans),
        'structured_local_mean': np.mean(structured_local) if structured_local else 0,
        'structured_local_max': np.max(structured_local) if structured_local else 0,
        'sign_stability_mean': np.mean(sign_stabs),
        'sign_stability_median': np.median(sign_stabs),
        'best_overall_spearman': max(stats, key=lambda s: s.overall_spearman),
        'best_local_spearman': max(stats, key=lambda s: s.local_spearman),
    }


def generate_validation_table(stats: List[HeadValidationStats], summary: dict) -> str:
    """Generate markdown table for blog post."""
    if not stats:
        return "## Validation Summary\n\n_No validation stats available._"

    lines = []
    lines.append("## Activation-Grounding Validation Summary")
    lines.append("")
    lines.append(f"**Total heads validated:** {summary['n_heads_total']}")
    lines.append(f"**Total pairs per head:** ~{stats[0].n_pairs:,}")
    lines.append("")

    lines.append("### Aggregate Statistics")
    lines.append("")
    lines.append("| Metric | Mean | Std | Max |")
    lines.append("|--------|------|-----|-----|")
    lines.append(f"| Overall Pearson r | {summary['overall_pearson_mean']:.3f} | {summary['overall_pearson_std']:.3f} | {summary['overall_pearson_max']:.3f} |")
    lines.append(f"| Overall Spearman ρ | {summary['overall_spearman_mean']:.3f} | {summary['overall_spearman_std']:.3f} | {summary['overall_spearman_max']:.3f} |")
    lines.append(f"| Local (0-4) Spearman ρ | {summary['local_spearman_mean']:.3f} | {summary['local_spearman_std']:.3f} | {summary['local_spearman_max']:.3f} |")
    lines.append("")

    lines.append(f"**Heads with |local ρ| > 0.1:** {summary['n_heads_structured']} / {summary['n_heads_total']}")
    lines.append(f"**Mean sign stability:** {summary['sign_stability_mean']:.1f}%")
    lines.append("")

    lines.append("### Top Validated Heads (by local ρ)")
    lines.append("")
    lines.append("| Layer | Head | Overall ρ | Local ρ | Sign Stability |")
    lines.append("|-------|------|-----------|---------|----------------|")

    top_heads = sorted(stats, key=lambda s: s.local_spearman, reverse=True)[:10]
    for h in top_heads:
        lines.append(f"| {h.layer} | {h.head} | {h.overall_spearman:.3f} | {h.local_spearman:.3f} | {h.sign_stability:.1f}% |")

    return "\n".join(lines)


# ============================================================================
# Plot helpers for validation + OV analysis
# ============================================================================

def group_by_layer(values: Iterable[Tuple[int, float]]) -> Dict[int, List[float]]:
    grouped: Dict[int, List[float]] = defaultdict(list)
    for layer, val in values:
        grouped[layer].append(val)
    return grouped


def plot_local_spearman_by_layer(stats: List[HeadValidationStats], output_path: Path) -> None:
    if not stats:
        return
    layer_to_vals = group_by_layer((s.layer, s.local_spearman) for s in stats)
    layers = sorted(layer_to_vals.keys())
    data = [layer_to_vals[layer] for layer in layers]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, tick_labels=[f"L{l}" for l in layers], showfliers=False)
    plt.scatter(np.arange(1, len(layers) + 1), [np.mean(vals) for vals in data], color='tomato', label="Layer mean")
    plt.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    plt.title("Local (0–4 token) Spearman ρ by Layer (16K validation)")
    plt.ylabel("Spearman ρ")
    plt.xlabel("Layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_threshold_counts(stats: List[HeadValidationStats], attr: str, thresholds: Tuple[float, ...], title: str, output_path: Path) -> None:
    if not stats:
        return
    layers = sorted({s.layer for s in stats})
    x = np.arange(len(layers))
    bottom = np.zeros(len(layers))
    plt.figure(figsize=(12, 5))
    colors = ["#c7e9c0", "#74c476", "#238b45"]

    for idx, thr in enumerate(thresholds):
        counts = []
        for layer in layers:
            vals = [getattr(s, attr) for s in stats if s.layer == layer]
            counts.append(sum(1 for v in vals if abs(v) >= thr))
        counts_arr = np.array(counts)
        plt.bar(x, counts_arr, bottom=bottom, color=colors[idx % len(colors)], label=f"|ρ| ≥ {thr:.2f}")
        bottom += counts_arr

    plt.xticks(x, [f"L{l}" for l in layers], rotation=90)
    plt.ylabel("# heads")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_long_distance_thresholds(stats: List[HeadValidationStats], output_path: Path) -> None:
    """Plot threshold counts for the 128-256 and 256+ bins side by side."""
    if not stats:
        return

    layers = sorted({s.layer for s in stats})
    thresholds = (0.10, 0.20)
    metric_specs = (
        ("long_spearman", "128-256 tokens"),
        ("far_spearman", "256+ tokens"),
    )
    colors = ("#74a9cf", "#2b8cbe")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = np.arange(len(layers))
    width = 0.34

    for ax, (attr, label) in zip(axes, metric_specs):
        counts_by_thr = []
        for thr in thresholds:
            counts = []
            for layer in layers:
                vals = [getattr(s, attr) for s in stats if s.layer == layer]
                counts.append(sum(1 for v in vals if abs(v) >= thr))
            counts_by_thr.append(counts)

        total_above_low = sum(abs(getattr(s, attr)) >= thresholds[0] for s in stats)
        for idx, thr in enumerate(thresholds):
            offset = (idx - 0.5) * width
            ax.bar(
                x + offset,
                counts_by_thr[idx],
                width=width,
                color=colors[idx],
                label=f"|rho| >= {thr:.2f}",
            )

        ax.set_ylabel("# heads")
        ax.set_title(f"{label}: {total_above_low}/200 heads with |rho| >= {thresholds[0]:.2f}")
        ax.grid(True, axis='y', alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xticks(x, [f"L{l}" for l in layers], rotation=90)
    axes[-1].set_xlabel("Layer")
    fig.suptitle("Long-distance routing weakens but persists beyond local context", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ov_cosine_by_layer(ov_stats: List[OVWriteStats], output_path: Path) -> None:
    if not ov_stats:
        return
    layer_to_vals = group_by_layer((s.layer, s.cosine) for s in ov_stats)
    layers = sorted(layer_to_vals.keys())
    medians, q1, q3 = [], [], []
    neg_share_by_layer = {}
    for layer in layers:
        vals = np.array(layer_to_vals[layer])
        medians.append(np.median(vals))
        q1.append(np.percentile(vals, 25))
        q3.append(np.percentile(vals, 75))
        neg_share_by_layer[layer] = float(np.mean(vals < 0))

    late_layers = [layer for layer in layers if layer >= 15]
    late_neg_share = [neg_share_by_layer[layer] for layer in late_layers]
    peak_layer = late_layers[int(np.argmax(late_neg_share))] if late_layers else None
    peak_share = max(late_neg_share) if late_neg_share else 0.0

    fig, (ax_main, ax_neg) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [3.2, 1.2]},
        constrained_layout=True,
    )

    for layer in layers:
        vals = np.array(layer_to_vals[layer])
        jitter = np.linspace(-0.22, 0.22, len(vals))
        colors = np.where(vals < 0, "#d7301f", "#3182bd")
        ax_main.scatter(
            np.full(len(vals), layer) + jitter,
            vals,
            c=colors,
            s=36,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )

    ax_main.plot(layers, medians, color='black', linewidth=2, label="Layer median", zorder=4)
    ax_main.fill_between(layers, q1, q3, color='skyblue', alpha=0.28, label="IQR", zorder=1)
    ax_main.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    ax_main.set_xlim(0.4, max(layers) + 0.6)
    ax_main.set_ylabel("Cosine (predicted vs actual write)")
    ax_main.set_title("OV_f validation: late negative outliers without a layer-wide median flip")
    ax_main.legend(loc="upper right")

    most_negative = sorted(ov_stats, key=lambda s: s.cosine)[:4]
    for entry in most_negative:
        ax_main.scatter(entry.layer, entry.cosine, s=82, facecolors='none', edgecolors='black', linewidths=1.2, zorder=5)
        ax_main.annotate(
            f"L{entry.layer}H{entry.head} ({entry.cosine:.2f})",
            xy=(entry.layer, entry.cosine),
            xytext=(entry.layer + 0.25, entry.cosine - 0.03),
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.8},
        )

    ax_neg.bar(late_layers, late_neg_share, color='#fb6a4a', width=0.7)
    ax_neg.axhline(0.0, color='gray', linewidth=1)
    if peak_layer is not None:
        ax_neg.set_title(f"Late-layer negative share peaks at L{peak_layer} ({peak_share:.0%} of heads)")
    else:
        ax_neg.set_title("Late-layer negative share")
    ax_neg.set_ylabel("Neg share")
    ax_neg.set_xlabel("Layer")
    ax_neg.set_xticks(late_layers)
    ax_neg.set_ylim(0, max(0.30, peak_share + 0.05))

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    plt.figure(figsize=(12, 3))
    plt.bar([f"L{l}" for l in late_layers], late_neg_share, color='#fb6a4a')
    plt.axhline(peak_share, color='gray', linestyle='--', linewidth=1)
    plt.ylabel("Fraction of heads w/ cosine < 0")
    plt.title("Late-layer negative OV heads by layer")
    plt.tight_layout()
    plt.savefig(output_path.with_name(output_path.stem + "_neg_share.png"), dpi=150)
    plt.close()


# ============================================================================
# PART 2: Generate RoPE stability curves
# ============================================================================

def load_rope_stability_data(json_path: Path) -> Dict[int, Dict[int, Dict[int, float]]]:
    """
    Load RoPE stability data from analysis JSON.

    Returns: {layer: {head: {delta: stability}}}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    result = {}
    layer_results = data.get('layer_results', {})

    for layer_key, layer_data in layer_results.items():
        layer_idx = int(layer_key)
        rope_stability = layer_data.get('rope_stability', [])

        if rope_stability:
            result[layer_idx] = {}
            for head_data in rope_stability:
                head_idx = head_data['query_head']
                stability_by_delta = head_data['stability_by_delta']
                result[layer_idx][head_idx] = {
                    int(k): v for k, v in stability_by_delta.items()
                }

    return result


def get_semantic_controllability(json_path: Path) -> Dict[Tuple[int, int], float]:
    """Get semantic controllability scores per (layer, head)."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    result = {}
    layer_results = data.get('layer_results', {})

    for layer_key, layer_data in layer_results.items():
        layer_idx = int(layer_key)
        rope_stability = layer_data.get('rope_stability', [])

        for head_data in rope_stability:
            head_idx = head_data['query_head']
            ctrl = head_data.get('semantic_controllability', 0)
            result[(layer_idx, head_idx)] = ctrl

    return result


def plot_rope_stability_curves(
    rope_data: Dict[int, Dict[int, Dict[int, float]]],
    heads_to_plot: List[Tuple[int, int]],
    title: str = "RoPE Stability vs. Relative Position",
    output_path: Optional[str] = None
):
    """
    Plot stability(Δ) curves for selected heads.

    Args:
        rope_data: {layer: {head: {delta: stability}}}
        heads_to_plot: List of (layer, head) tuples
        title: Plot title
        output_path: If provided, save to this path
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use log scale for x-axis
    deltas = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(heads_to_plot)))

    for (layer, head), color in zip(heads_to_plot, colors):
        if layer in rope_data and head in rope_data[layer]:
            stability = rope_data[layer][head]
            y_vals = [stability.get(d, np.nan) for d in deltas]

            ax.plot(deltas, y_vals, 'o-', color=color,
                   label=f'L{layer}H{head}', linewidth=2, markersize=6)

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Relative Position Δ', fontsize=12)
    ax.set_ylabel('Cosine Similarity to B₀', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% similarity')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close(fig)
    return fig


def plot_stability_heatmap(
    controllability: Dict[Tuple[int, int], float],
    layers: List[int],
    n_heads: int = 8,
    output_path: Optional[str] = None
):
    """
    Plot heatmap of semantic controllability across layers and heads.
    """
    matrix = np.zeros((len(layers), n_heads))
    for i, layer in enumerate(layers):
        for head in range(n_heads):
            matrix[i, head] = controllability.get((layer, head), 0)

    fig, ax = plt.subplots(figsize=(10, 12))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=0.8)

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f'H{h}' for h in range(n_heads)])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f'L{l}' for l in layers])

    ax.set_xlabel('Head', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('RoPE Stability (AUC across relative positions)', fontsize=14)

    plt.colorbar(im, ax=ax, label='RoPE Stability')

    for i, layer in enumerate(layers):
        if layer % 2 == 0:
            ax.axhline(y=i-0.5, color='blue', linewidth=0.5, alpha=0.3)
            ax.axhline(y=i+0.5, color='blue', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close(fig)
    return fig


# ============================================================================
# Copy dominance & program distribution helpers
# ============================================================================

def extract_copy_dominance(json_path: Path) -> List[Tuple[int, int, float]]:
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    rows: List[Tuple[int, int, float]] = []
    for layer_key, layer_data in data.get('layer_results', {}).items():
        layer = int(layer_key)
        for wr in layer_data.get('writing_results', []):
            metrics = wr.get('metrics', {})
            if 'copy_dominance' in metrics:
                rows.append((layer, wr.get('query_head', -1), float(metrics['copy_dominance'])))
    return rows


def extract_program_counts(json_path: Path) -> Dict[int, Counter]:
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    per_layer: Dict[int, Counter] = defaultdict(Counter)
    for layer_key, layer_data in data.get('layer_results', {}).items():
        layer = int(layer_key)
        for prog in layer_data.get('program_results', []):
            counts = prog.get('program_counts', {})
            for prog_type, count in counts.items():
                per_layer[layer][prog_type.upper()] += count
    return per_layer


def plot_copy_dominance(copy_rows: List[Tuple[int, int, float]], output_path: Path) -> None:
    if not copy_rows:
        return
    layer_to_vals = group_by_layer((layer, val) for layer, _, val in copy_rows)
    layers = sorted(layer_to_vals.keys())
    data = [layer_to_vals[layer] for layer in layers]

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axs[0].violinplot(data, showmeans=True, showmedians=False)
    axs[0].set_title("Copy Dominance Distribution per Layer")
    axs[0].set_ylabel("copy_dominance")
    axs[0].axhline(0.5, color='gray', linestyle='--', linewidth=1)

    medians = [np.median(vals) for vals in data]
    axs[1].plot(layers, medians, marker='o')
    axs[1].set_ylabel("Median copy_dominance")
    axs[1].set_xlabel("Layer")
    axs[1].set_xticks(layers)
    axs[1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_program_distribution(per_layer: Dict[int, Counter], output_path: Path, top_types: int = 6) -> None:
    if not per_layer:
        return
    layers = sorted(per_layer.keys())
    total_counts = Counter()
    for counter in per_layer.values():
        total_counts.update(counter)
    most_common = [t for t, _ in total_counts.most_common(top_types)]

    stacked_values = {t: [] for t in most_common}
    for layer in layers:
        counter = per_layer[layer]
        layer_total = sum(counter.values()) or 1
        for prog_type in most_common:
            stacked_values[prog_type].append(counter.get(prog_type, 0) / layer_total)

    x = np.arange(len(layers))
    bottom = np.zeros(len(layers))
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(most_common)))
    for idx, prog_type in enumerate(most_common):
        vals = stacked_values[prog_type]
        plt.bar(x, vals, bottom=bottom, color=colors[idx], label=prog_type.title())
        bottom += vals

    plt.xticks(x, [f"L{l}" for l in layers], rotation=90)
    plt.ylabel("Program share")
    plt.title("Weight-Space Program Distribution by Layer")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================================
# PART 5: Depth-trend figures (selectivity, RoPE, archetypes, redundancy)
# ============================================================================

def extract_selectivity_metrics(json_path: Path) -> List[dict]:
    """Extract per-head selectivity and diagonal mass metrics with baselines.

    Returns list of dicts with keys: layer, head, is_sliding_window,
    top1_mass, diag_mass, baseline_rw_top1, baseline_pk_diag.
    """
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for layer_key, layer_data in data.get('layer_results', {}).items():
        layer = int(layer_key)
        is_sw = layer_data.get('is_sliding_window', layer % 2 == 0)
        for rr in layer_data.get('routing_results', []):
            m = rr.get('metrics', {})
            brw = rr.get('baseline_random_weights', {})
            bpk = rr.get('baseline_permuted_k', {})
            rows.append({
                'layer': layer,
                'head': rr.get('query_head', -1),
                'is_sliding_window': is_sw,
                'top1_mass': m.get('top1_mass_mean', 0),
                'diag_mass': m.get('diagonal_softmax_mass', 0),
                'baseline_rw_top1': brw.get('top1_mass_mean', 1e-9),
                'baseline_pk_diag': bpk.get('diagonal_softmax_mass', 1e-9),
            })
    return rows


def plot_selectivity_by_type(rows: List[dict], output_path: Path) -> None:
    """Boxplot of Sel×U grouped by sliding-window vs global layers."""
    if not rows:
        return
    sw_vals = [r['top1_mass'] / max(r['baseline_rw_top1'], 1e-12) for r in rows if r['is_sliding_window']]
    gl_vals = [r['top1_mass'] / max(r['baseline_rw_top1'], 1e-12) for r in rows if not r['is_sliding_window']]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot([sw_vals, gl_vals], tick_labels=['Sliding-window', 'Global'],
                     showfliers=False, patch_artist=True)
    bp['boxes'][0].set_facecolor('#a6cee3')
    bp['boxes'][1].set_facecolor('#fb9a99')
    ax.set_ylabel('Selectivity × Uniform (Sel×U)')
    ax.set_title('Routing selectivity: sliding-window vs global layers')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_diagm_by_type(rows: List[dict], output_path: Path) -> None:
    """Boxplot of DiagM×U grouped by sliding-window vs global layers."""
    if not rows:
        return
    sw_vals = [r['diag_mass'] / max(r['baseline_pk_diag'], 1e-12) for r in rows if r['is_sliding_window']]
    gl_vals = [r['diag_mass'] / max(r['baseline_pk_diag'], 1e-12) for r in rows if not r['is_sliding_window']]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot([sw_vals, gl_vals], tick_labels=['Sliding-window', 'Global'],
                     showfliers=False, patch_artist=True)
    bp['boxes'][0].set_facecolor('#a6cee3')
    bp['boxes'][1].set_facecolor('#fb9a99')
    ax.set_ylabel('Diagonal Mass × Uniform (DiagM×U)')
    ax.set_title('Identity sensitivity: sliding-window vs global layers')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_sel_diagm_vs_layer(rows: List[dict], output_path: Path) -> None:
    """Two-panel plot of Sel×U and DiagM×U vs layer."""
    if not rows:
        return
    layer_sel = defaultdict(list)
    layer_diag = defaultdict(list)
    for r in rows:
        sel_xu = r['top1_mass'] / max(r['baseline_rw_top1'], 1e-12)
        diag_xu = r['diag_mass'] / max(r['baseline_pk_diag'], 1e-12)
        layer_sel[r['layer']].append(sel_xu)
        layer_diag[r['layer']].append(diag_xu)

    layers = sorted(layer_sel.keys())
    sel_means = [np.mean(layer_sel[l]) for l in layers]
    diag_means = [np.mean(layer_diag[l]) for l in layers]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Sel×U
    for l in layers:
        vals = layer_sel[l]
        jitter = np.linspace(-0.2, 0.2, len(vals))
        color = '#a6cee3' if (l % 2 == 0) else '#fb9a99'
        ax1.scatter(np.full(len(vals), l) + jitter, vals, c=color, s=30, alpha=0.7, edgecolors='white', linewidths=0.3)
    ax1.plot(layers, sel_means, 'k-o', markersize=5, linewidth=1.5, label='Layer mean')
    ax1.set_ylabel('Sel×U')
    ax1.set_title('Selectivity × Uniform baseline by layer')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    # DiagM×U
    for l in layers:
        vals = layer_diag[l]
        jitter = np.linspace(-0.2, 0.2, len(vals))
        color = '#a6cee3' if (l % 2 == 0) else '#fb9a99'
        ax2.scatter(np.full(len(vals), l) + jitter, vals, c=color, s=30, alpha=0.7, edgecolors='white', linewidths=0.3)
    ax2.plot(layers, diag_means, 'k-o', markersize=5, linewidth=1.5, label='Layer mean')
    ax2.set_ylabel('DiagM×U')
    ax2.set_xlabel('Layer')
    ax2.set_title('Diagonal mass × Uniform baseline by layer')
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#a6cee3', label='Sliding-window'),
                       Patch(facecolor='#fb9a99', label='Global')]
    ax1.legend(handles=legend_elements + [plt.Line2D([0], [0], color='k', marker='o', label='Layer mean')],
               loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_rope_vs_layer(controllability: Dict[Tuple[int, int], float], output_path: Path) -> None:
    """Mean semantic controllability per layer as a line plot."""
    if not controllability:
        return
    layer_vals = defaultdict(list)
    for (layer, head), ctrl in controllability.items():
        layer_vals[layer].append(ctrl)

    layers = sorted(layer_vals.keys())
    means = [np.mean(layer_vals[l]) for l in layers]
    stds = [np.std(layer_vals[l]) for l in layers]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.errorbar(layers, means, yerr=stds, fmt='o-', color='#2b8cbe', capsize=3, linewidth=1.5, markersize=6)

    # Color background by layer type
    for l in layers:
        color = '#e0f0ff' if (l % 2 == 0) else '#fff0e0'
        ax.axvspan(l - 0.5, l + 0.5, alpha=0.3, color=color)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Semantic Controllability (RoPE AUC)')
    ax.set_title('RoPE stability by layer (mean ± std across heads)')
    ax.set_xticks(layers)
    ax.set_xticklabels([f'L{l}' for l in layers], rotation=90)
    ax.grid(True, axis='y', alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e0f0ff', label='Sliding-window'),
                       Patch(facecolor='#fff0e0', label='Global')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def extract_archetype_counts(json_path: Path) -> Dict[int, Counter]:
    """Extract write archetype counts per layer."""
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    per_layer: Dict[int, Counter] = defaultdict(Counter)
    for layer_key, layer_data in data.get('layer_results', {}).items():
        layer = int(layer_key)
        for wr in layer_data.get('writing_results', []):
            archetype = wr.get('metrics', {}).get('archetype', 'DIFFUSE')
            per_layer[layer][archetype.upper()] += 1
    return per_layer


def plot_archetype_fraction(per_layer: Dict[int, Counter], output_path: Path) -> None:
    """Stacked bar chart of write archetype fractions by layer."""
    if not per_layer:
        return
    layers = sorted(per_layer.keys())
    all_types = Counter()
    for c in per_layer.values():
        all_types.update(c)
    # Focus on key archetypes
    focus_types = ['TRANSFORM', 'BROADCAST', 'COPY', 'SUPPRESS', 'DIFFUSE']
    focus_types = [t for t in focus_types if all_types.get(t, 0) > 0]

    x = np.arange(len(layers))
    bottom = np.zeros(len(layers))
    colors_map = {
        'TRANSFORM': '#e41a1c', 'BROADCAST': '#377eb8', 'COPY': '#4daf4a',
        'SUPPRESS': '#984ea3', 'DIFFUSE': '#cccccc'
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    for atype in focus_types:
        fracs = []
        for layer in layers:
            total = sum(per_layer[layer].values()) or 1
            fracs.append(per_layer[layer].get(atype, 0) / total)
        fracs_arr = np.array(fracs)
        ax.bar(x, fracs_arr, bottom=bottom, color=colors_map.get(atype, '#999999'), label=atype.capitalize())
        bottom += fracs_arr

    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers], rotation=90)
    ax.set_ylabel('Fraction of heads')
    ax.set_title('Write archetype distribution by layer')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def extract_redundancy(json_path: Path) -> Dict[int, float]:
    """Extract mean within-layer Jaccard redundancy per layer."""
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    result = {}
    for layer_key, layer_data in data.get('layer_results', {}).items():
        layer = int(layer_key)
        cr = layer_data.get('cross_head_redundancy', {})
        if isinstance(cr, dict) and cr:
            vals = [v for v in cr.values() if isinstance(v, (int, float))]
            result[layer] = np.mean(vals) if vals else 0.0
    return result


def plot_redundancy_vs_layer(redundancy: Dict[int, float], output_path: Path) -> None:
    """Line plot of mean within-layer Jaccard redundancy vs layer."""
    if not redundancy:
        return
    layers = sorted(redundancy.keys())
    vals = [redundancy[l] for l in layers]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(layers, vals, color=['#a6cee3' if l % 2 == 0 else '#fb9a99' for l in layers], width=0.7)
    ax.plot(layers, vals, 'ko-', markersize=5, linewidth=1)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Jaccard similarity (top-20 routing pairs)')
    ax.set_title('Within-layer head redundancy by layer')
    ax.set_xticks(layers)
    ax.set_xticklabels([f'L{l}' for l in layers], rotation=90)
    ax.grid(True, axis='y', alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#a6cee3', label='Sliding-window'),
                       Patch(facecolor='#fb9a99', label='Global')]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("GENERATING BLOG ASSETS")
    print("=" * 60)

    # --- PART 1: Validation Statistics ---
    print("\n[1/4] Parsing validation report...")
    report_path, report_label = resolve_validation_report()

    stats: List[HeadValidationStats] = []
    ov_entries: List[OVWriteStats] = []
    summary = {}

    if report_path:
        stats, ov_entries = parse_validation_report(report_path)
        summary = compute_summary_statistics(stats)
        print(f"  Parsed {len(stats)} heads from {report_path.name} ({report_label})")
        if stats:
            print(f"  Mean overall Spearman: {summary['overall_spearman_mean']:.3f}")
            print(f"  Max local Spearman: {summary['local_spearman_max']:.3f}")

            table = generate_validation_table(stats, summary)
            table_path = OUTPUT_DIR / "validation_summary.md"
            with table_path.open('w', encoding='utf-8') as f:
                f.write(table)
            print(f"  Saved summary: {table_path}")

            plot_local_spearman_by_layer(stats, OUTPUT_DIR / "fig01_local_spearman_box.png")
            plot_threshold_counts(
                stats,
                attr="local_spearman",
                thresholds=(0.1, 0.3, 0.5),
                title="Heads with |local ρ| above thresholds",
                output_path=OUTPUT_DIR / "fig02_local_thresholds.png"
            )
            plot_long_distance_thresholds(stats, OUTPUT_DIR / "fig03_long_thresholds.png")
        if ov_entries:
            plot_ov_cosine_by_layer(ov_entries, OUTPUT_DIR / "fig10_ov_signflip.png")
    else:
        print("  Warning: No validation report found. Skipping validation plots.")

    # --- PART 2: RoPE Stability Curves ---
    print("\n[2/4] Loading RoPE stability data...")
    json_path = resolve_latest_analysis_json()

    if json_path:
        rope_data = load_rope_stability_data(json_path)
        controllability = get_semantic_controllability(json_path)

        print(f"  Loaded RoPE data for {len(rope_data)} layers from {json_path.name}")

        sorted_by_ctrl = sorted(controllability.items(), key=lambda x: x[1], reverse=True)
        high_ctrl_heads = [kv[0] for kv in sorted_by_ctrl[:3]]
        low_ctrl_heads = [kv[0] for kv in sorted_by_ctrl[-3:]]

        showcase_heads = high_ctrl_heads[:2] + low_ctrl_heads[:2]
        plot_rope_stability_curves(
            rope_data,
            showcase_heads,
            title="RoPE Stability: High vs Low Heads",
            output_path=str(OUTPUT_DIR / "fig04_rope_stability_curves.png")
        )

        layers = sorted(rope_data.keys())
        plot_stability_heatmap(
            controllability,
            layers,
            output_path=str(OUTPUT_DIR / "fig04b_controllability_heatmap.png")
        )
    else:
        print("  Warning: No analysis JSON found. Skipping RoPE plots.")

    # --- PART 3: Copy dominance & program distribution ---
    print("\n[3/4] Generating copy/program figures...")
    if json_path:
        copy_rows = extract_copy_dominance(json_path)
        if copy_rows:
            plot_copy_dominance(copy_rows, OUTPUT_DIR / "fig05_copy_dominance.png")
            print(f"  Copy dominance entries: {len(copy_rows)}")
        else:
            print("  Warning: No copy_dominance entries in JSON.")

        per_layer_programs = extract_program_counts(json_path)
        if per_layer_programs:
            plot_program_distribution(per_layer_programs, OUTPUT_DIR / "fig06_program_distribution.png")
            print(f"  Program layers covered: {len(per_layer_programs)}")
        else:
            print("  Warning: No program counts found.")
    else:
        print("  Skipping copy/program plots (analysis JSON missing).")

    # --- PART 3b: Depth-trend figures ---
    print("\n[3b/4] Generating depth-trend figures...")
    if json_path:
        sel_rows = extract_selectivity_metrics(json_path)
        if sel_rows:
            plot_selectivity_by_type(sel_rows, OUTPUT_DIR / "fig07_selectivity_by_type.png")
            plot_diagm_by_type(sel_rows, OUTPUT_DIR / "fig08_diagm_by_type.png")
            plot_sel_diagm_vs_layer(sel_rows, OUTPUT_DIR / "fig09_sel_diagm_vs_layer.png")
            print(f"  Selectivity/DiagM figures: {len(sel_rows)} heads")
        else:
            print("  Warning: No selectivity metrics found.")

        if controllability:
            plot_rope_vs_layer(controllability, OUTPUT_DIR / "fig11_rope_vs_layer.png")
        else:
            print("  Warning: No controllability data for RoPE depth plot.")

        archetype_counts = extract_archetype_counts(json_path)
        if archetype_counts:
            plot_archetype_fraction(archetype_counts, OUTPUT_DIR / "fig12_archetype_fraction.png")
            print(f"  Archetype layers: {len(archetype_counts)}")
        else:
            print("  Warning: No archetype data.")

        redundancy = extract_redundancy(json_path)
        if redundancy:
            plot_redundancy_vs_layer(redundancy, OUTPUT_DIR / "fig13_redundancy_vs_layer.png")
            print(f"  Redundancy layers: {len(redundancy)}")
        else:
            print("  Warning: No redundancy data.")
    else:
        print("  Skipping depth-trend plots (analysis JSON missing).")

    # --- PART 4: Summary for blog ---
    print("\n[4/4] Generating blog-ready summary...")

    summary_lines = []
    summary_lines.append("# Blog Asset Summary")
    summary_lines.append("")

    if stats:
        best = summary['best_local_spearman']
        long_ge_01 = sum(abs(s.long_spearman) >= 0.10 for s in stats)
        far_ge_01 = sum(abs(s.far_spearman) >= 0.10 for s in stats)
        summary_lines.append("## Activation Grounding Headline Numbers")
        summary_lines.append("")
        summary_lines.append("Use these bullet points in the validation section:")
        summary_lines.append(f"- **{len(stats)} heads** validated across 25 layers (~{stats[0].n_pairs:,} token pairs per head).")
        summary_lines.append(f"- Mean overall Spearman ρ = **{summary['overall_spearman_mean']:.2f}** (std={summary['overall_spearman_std']:.2f}).")
        summary_lines.append(f"- Mean local (0–4) Spearman ρ = **{summary['local_spearman_mean']:.2f}** (max={summary['local_spearman_max']:.2f}).")
        summary_lines.append(f"- **{summary['n_heads_structured']} heads** show |local ρ| > 0.1.")
        summary_lines.append(f"- Best local correlation: **L{best.layer}H{best.head}** with ρ={best.local_spearman:.2f}.")
        summary_lines.append(f"- Long-range structure persists in a subset of heads: **{long_ge_01}/200** clear |ρ| >= 0.10 at 128-256 tokens and **{far_ge_01}/200** at 256+.")
        summary_lines.append("")
        if ov_entries:
            late_entries = [o for o in ov_entries if o.layer >= 15]
            late_layers = sorted({o.layer for o in late_entries})
            late_neg_share = {
                layer: np.mean([o.cosine < 0 for o in late_entries if o.layer == layer])
                for layer in late_layers
            }
            peak_layer = max(late_neg_share, key=late_neg_share.get) if late_neg_share else None
            peak_share = late_neg_share[peak_layer] if peak_layer is not None else 0.0
            most_negative = min(ov_entries, key=lambda o: o.cosine)
            summary_lines.append(
                f"- OV_f negatives are late-layer outliers, not a median layer-wide flip: peak negative share is **{peak_share*100:.0f}%** at **L{peak_layer}**, and the strongest outlier is **L{most_negative.layer}H{most_negative.head} = {most_negative.cosine:.2f}**."
            )
            summary_lines.append("")

    summary_path = OUTPUT_DIR / "blog_summary.md"
    with summary_path.open('w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))
    print(f"  Saved: {summary_path}")

    print("\n" + "=" * 60)
    print("DONE! Check ./figs/blog/ for outputs")
    print("=" * 60)


if __name__ == "__main__":
    main()
