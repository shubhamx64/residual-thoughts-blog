"""
RGB Composite Visualization for Attention Heads.

Creates visual "circuit microscopes" for each (layer, head) where:
- R (Red) = QK affinity (who routes to whom)
- G (Green) = OV∘Q (routed info writes into query direction)
- B (Blue) = OV∘K (routed info writes into key direction)

Co-colored patches reveal circuit motifs:
- Red blobs → strong routing with little rewriting
- Green streaks → query-anchored rewriting (self-reinforcement/cleanup)
- Blue streaks → key-anchored copying (copy/move/induction-ish)
- Cyan/Magenta/Yellow/White → combinations

Outputs POS and NEG images to preserve sign information.

Usage:
    python rgb_circuit_viz.py --layers 1 5 10 --heads 0 1 2 3 --outdir ./rgb_outputs
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local imports
from config import Gemma2Config, GEMMA2_CONFIG
from sae_utils import SAEManager, prepare_sae_features
from weight_extraction import WeightExtractor, extract_layer_weights, get_qkv_for_head
from qk_routing import apply_softcap


# =============================================================================
# Core Computation (returns torch tensors, stays on GPU)
# =============================================================================

def compute_feature_projections(
    decoder: torch.Tensor,  # [F, d] normalized SAE decoder
    W_Q: torch.Tensor,      # [head_dim, d]
    W_K: torch.Tensor,      # [head_dim, d]
    W_V: torch.Tensor,      # [head_dim, d]
    W_O: torch.Tensor,      # [d, head_dim]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project decoder directions into Q, K, and OV spaces.
    
    Returns:
        Q_f: [F, head_dim] features projected into Q space
        K_f: [F, head_dim] features projected into K space
        OV_f: [F, d] features after OV transformation
    """
    Q_f = decoder @ W_Q.T   # [F, head_dim]
    K_f = decoder @ W_K.T   # [F, head_dim]
    V_f = decoder @ W_V.T   # [F, head_dim]
    OV_f = V_f @ W_O.T      # [F, d]
    
    return Q_f, K_f, OV_f


@torch.no_grad()
def compute_rgb_matrices_torch(
    decoder: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    W_O: torch.Tensor,
    config: Gemma2Config = GEMMA2_CONFIG,
    symmetrize_qk: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the three alignment matrices as torch tensors (stays on GPU).
    
    Returns M_QK, M_OVQ, M_OVK each [F, F] float32.
    """
    Q_f, K_f, OV_f = compute_feature_projections(decoder, W_Q, W_K, W_V, W_O)
    
    # QK affinity: M_QK[i,j] = (d_i @ W_Q.T) · (d_j @ W_K.T) * scale
    M_QK = Q_f @ K_f.T
    M_QK = M_QK * config.attention_scale
    
    if config.apply_softcap and config.attn_logit_softcapping is not None:
        M_QK = apply_softcap(M_QK, config.attn_logit_softcapping)
    
    if symmetrize_qk:
        M_QK = 0.5 * (M_QK + M_QK.T)
    
    # OV∘Q: M_OVQ[i,j] = Q_f[i] · (OV_f[j] @ W_Q.T)
    OV_in_Q_space = OV_f @ W_Q.T  # [F, head_dim]
    M_OVQ = Q_f @ OV_in_Q_space.T  # [F, F]
    
    # OV∘K: M_OVK[i,j] = K_f[i] · (OV_f[j] @ W_K.T)
    OV_in_K_space = OV_f @ W_K.T  # [F, head_dim]
    M_OVK = K_f @ OV_in_K_space.T  # [F, F]
    
    return M_QK.float(), M_OVQ.float(), M_OVK.float()


# =============================================================================
# Normalization & RGB Creation (torch-native, fast)
# =============================================================================

@torch.no_grad()
def sample_quantiles(x: torch.Tensor, q_lo: float, q_hi: float, sample: int = 200_000):
    """Compute quantiles on a random sample (faster for large tensors)."""
    flat = x.flatten()
    if flat.numel() > sample:
        idx = torch.randint(0, flat.numel(), (sample,), device=flat.device)
        flat = flat[idx]
    flat = flat.float()
    lo = torch.quantile(flat, q_lo)
    hi = torch.quantile(flat, q_hi)
    return lo, hi


@torch.no_grad()
def norm01(M: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Normalize to [0,1] given precomputed lo/hi."""
    return ((M.float() - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


@torch.no_grad()
def make_rgb_float(
    M_QK: torch.Tensor,
    M_OVQ: torch.Tensor,
    M_OVK: torch.Tensor,
    percentile: int = 99,
    norm_mode: Literal["per_channel", "shared_per_head"] = "shared_per_head",
) -> torch.Tensor:
    """
    Create RGB float tensor [F, F, 3] in [0,1] from three matrices.
    
    norm_mode:
        - "per_channel": each channel gets its own lo/hi (your original)
        - "shared_per_head": one lo/hi from all three channels (preserves relative magnitude)
    """
    q_hi = percentile / 100.0
    q_lo = 1.0 - q_hi

    if norm_mode == "per_channel":
        loR, hiR = sample_quantiles(M_QK,  q_lo, q_hi)
        loG, hiG = sample_quantiles(M_OVQ, q_lo, q_hi)
        loB, hiB = sample_quantiles(M_OVK, q_lo, q_hi)
        R = norm01(M_QK,  loR, hiR)
        G = norm01(M_OVQ, loG, hiG)
        B = norm01(M_OVK, loB, hiB)

    elif norm_mode == "shared_per_head":
        cat = torch.cat([M_QK.flatten(), M_OVQ.flatten(), M_OVK.flatten()], dim=0)
        lo, hi = sample_quantiles(cat, q_lo, q_hi)
        R = norm01(M_QK,  lo, hi)
        G = norm01(M_OVQ, lo, hi)
        B = norm01(M_OVK, lo, hi)

    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}")

    return torch.stack([R, G, B], dim=-1)  # [F, F, 3]


@torch.no_grad()
def split_pos_neg(M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split matrix into positive and negative parts."""
    return M.clamp_min(0), (-M).clamp_min(0)


# =============================================================================
# Downsampling (area-based, not aliasing stride)
# =============================================================================

@torch.no_grad()
def downsample_rgb_area(rgb_float: torch.Tensor, out_size: int) -> torch.Tensor:
    """
    Downsample [F, F, 3] float RGB using area averaging (no aliasing).
    
    Returns [out_size, out_size, 3] float in [0,1].
    """
    x = rgb_float.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    x = F.interpolate(x, size=(out_size, out_size), mode="area")
    return x.squeeze(0).permute(1, 2, 0)  # [out_size, out_size, 3]


# =============================================================================
# Reordering (reveal block structure)
# =============================================================================

@torch.no_grad()
def order_by_degree(M: torch.Tensor) -> torch.Tensor:
    """
    Compute ordering of features by "degree" in the routing matrix.
    High-degree features (lots of routing activity) come first.
    """
    deg = M.abs().sum(dim=0) + M.abs().sum(dim=1)
    return torch.argsort(deg, descending=True)


@torch.no_grad()
def apply_reorder(M: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Reorder both axes of a matrix according to order."""
    return M[order][:, order]


# =============================================================================
# Top-K Pairs Extraction
# =============================================================================

@torch.no_grad()
def topk_pairs(
    M: torch.Tensor,
    feature_ids: np.ndarray,
    k: int = 50,
    absval: bool = True,
) -> List[Dict]:
    """
    Extract top-K pairs from matrix by absolute value or raw value.
    
    Returns list of dicts with i, j, feat_i, feat_j, score.
    """
    X = M.abs() if absval else M
    flat = X.flatten()
    vals, idx = torch.topk(flat, min(k, flat.numel()))
    n = M.shape[0]
    ii = (idx // n).cpu().numpy()
    jj = (idx %  n).cpu().numpy()
    
    out = []
    for v, i, j in zip(vals.cpu().numpy(), ii, jj):
        out.append({
            "i": int(i),
            "j": int(j),
            "feat_i": int(feature_ids[i]),
            "feat_j": int(feature_ids[j]),
            "score": float(v),
        })
    return out


# =============================================================================
# Per-Head Metrics (for ranking interesting heads)
# =============================================================================

@torch.no_grad()
def compute_head_metrics(
    M_QK: torch.Tensor,
    M_OVQ: torch.Tensor,
    M_OVK: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute cheap metrics for ranking heads by "interestingness".
    """
    n = M_QK.shape[0]
    
    # Diagonal mass ratio on QK (identity routing)
    diag_qk = torch.diag(M_QK).abs().sum()
    total_qk = M_QK.abs().sum()
    diag_ratio = (diag_qk / total_qk.clamp_min(1e-9)).item()
    
    # Row concentration (stripe score) - how concentrated are rows?
    row_max = M_QK.abs().max(dim=1).values.mean()
    row_mean = M_QK.abs().mean(dim=1).mean()
    stripe_score = (row_max / row_mean.clamp_min(1e-9)).item()
    
    # Correlations between channels (flattened)
    def corr(a, b):
        a_flat = a.flatten().float()
        b_flat = b.flatten().float()
        a_centered = a_flat - a_flat.mean()
        b_centered = b_flat - b_flat.mean()
        num = (a_centered * b_centered).sum()
        denom = (a_centered.pow(2).sum() * b_centered.pow(2).sum()).sqrt()
        return (num / denom.clamp_min(1e-9)).item()
    
    corr_rg = corr(M_QK, M_OVQ)
    corr_rb = corr(M_QK, M_OVK)
    corr_gb = corr(M_OVQ, M_OVK)
    
    # Magnitude stats
    qk_std = M_QK.std().item()
    ovq_std = M_OVQ.std().item()
    ovk_std = M_OVK.std().item()
    
    return {
        "diag_ratio": diag_ratio,
        "stripe_score": stripe_score,
        "corr_rg": corr_rg,
        "corr_rb": corr_rb,
        "corr_gb": corr_gb,
        "qk_std": qk_std,
        "ovq_std": ovq_std,
        "ovk_std": ovk_std,
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_head_rgb(
    rgb: np.ndarray,
    layer_idx: int,
    head_idx: int,
    outdir: str,
    suffix: str = "",
    show: bool = False,
    figsize: Tuple[int, int] = (10, 10),
) -> str:
    """Save individual head RGB plot."""
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.imshow(rgb, origin='lower', interpolation='nearest')
    
    ax.set_xlabel('Key Feature Index (j)', fontsize=12)
    ax.set_ylabel('Query Feature Index (i)', fontsize=12)
    title = f'Layer {layer_idx}, Head {head_idx}'
    if suffix:
        title += f' ({suffix.upper()})'
    ax.set_title(f'{title}\nR=QK routing | G=OV→Q | B=OV→K', fontsize=14)
    
    legend_text = (
        'Red: strong routing\n'
        'Green: query-aligned writing\n'
        'Blue: key-aligned writing\n'
        'White: all three active'
    )
    ax.text(1.02, 0.98, legend_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    os.makedirs(outdir, exist_ok=True)
    fname = f'L{layer_idx}_H{head_idx}_rgb'
    if suffix:
        fname += f'_{suffix}'
    path = os.path.join(outdir, f'{fname}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close()
    
    return path


def plot_layer_head_grid(
    rgb_images: Dict[Tuple[int, int], np.ndarray],
    layers: List[int],
    heads: List[int],
    outdir: str,
    suffix: str = "",
    show: bool = False,
    cell_size: float = 1.5,
) -> str:
    """Create layer×head grid of RGB thumbnails."""
    n_layers = len(layers)
    n_heads = len(heads)
    
    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(n_heads * cell_size, n_layers * cell_size),
        squeeze=False
    )
    
    for row, layer in enumerate(layers):
        for col, head in enumerate(heads):
            ax = axes[row, col]
            key = (layer, head)
            
            if key in rgb_images:
                ax.imshow(rgb_images[key], origin='lower', interpolation='nearest', aspect='auto')
            else:
                ax.set_facecolor('lightgray')
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            if row == 0:
                ax.set_title(f'H{head}', fontsize=8)
        
        if n_heads > 0:
            axes[row, 0].set_ylabel(f'L{layer}', fontsize=8, rotation=0, ha='right', va='center')
    
    title = 'RGB Circuit Visualization: Layer × Head Grid'
    if suffix:
        title += f' ({suffix.upper()})'
    fig.suptitle(f'{title}\nR=QK routing | G=OV→Q | B=OV→K',
                 fontsize=12, y=1.02)
    
    plt.tight_layout()
    
    os.makedirs(outdir, exist_ok=True)
    fname = 'layer_head_grid'
    if suffix:
        fname += f'_{suffix}'
    path = os.path.join(outdir, f'{fname}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close()
    
    return path


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RGBVizConfig:
    """Configuration for RGB visualization."""
    layers: List[int]
    heads: List[int]
    feature_subset: int = 4096
    outdir: str = "./rgb_outputs"
    show: bool = False
    save_individual: bool = True
    save_grid: bool = True
    percentile: int = 99
    norm_mode: Literal["per_channel", "shared_per_head"] = "shared_per_head"
    grid_thumbnail_size: int = 256
    reorder_by_degree: bool = True
    save_top_pairs: bool = True
    top_k_pairs: int = 50
    save_metrics_csv: bool = True
    device: str = "cuda"
    dtype: torch.dtype = torch.float16


# =============================================================================
# Main Pipeline
# =============================================================================

@torch.no_grad()
def run_rgb_visualization(
    model,
    config: Gemma2Config = GEMMA2_CONFIG,
    viz_config: RGBVizConfig = None,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Run full RGB visualization pipeline with all quality upgrades.
    
    Outputs:
    - Per-head POS and NEG RGB images (full resolution)
    - Per-head top pairs JSON files
    - Layer×head grid images (POS and NEG)
    - Head metrics CSV
    - Reordering mappings (if enabled)
    """
    if viz_config is None:
        viz_config = RGBVizConfig(
            layers=list(range(1, config.num_hidden_layers)),
            heads=list(range(config.num_attention_heads)),
        )
    
    os.makedirs(viz_config.outdir, exist_ok=True)
    
    # Initialize managers
    weight_extractor = WeightExtractor(model, config, fold_gamma=config.fold_rmsnorm_gamma)
    sae_manager = SAEManager(config, subset_size=viz_config.feature_subset)
    
    # Storage for grid (thumbnails only)
    rgb_thumbs_pos: Dict[Tuple[int, int], np.ndarray] = {}
    rgb_thumbs_neg: Dict[Tuple[int, int], np.ndarray] = {}
    
    # Metrics collection
    all_metrics = []
    
    print(f"Generating RGB visualizations for {len(viz_config.layers)} layers × {len(viz_config.heads)} heads")
    print(f"Feature subset size: {viz_config.feature_subset}")
    print(f"Norm mode: {viz_config.norm_mode}")
    print(f"Reorder by degree: {viz_config.reorder_by_degree}")
    print(f"Output directory: {viz_config.outdir}")
    
    for layer_idx in tqdm(viz_config.layers, desc="Layers"):
        # Get SAE features for this layer
        sae_layer = config.get_sae_layer_for_attn(layer_idx)
        sae_features = sae_manager.get_features(
            sae_layer,
            device=viz_config.device,
            dtype=viz_config.dtype,
            compute_gram=False,
        )
        decoder = sae_features.decoder_subset  # [F, d]
        feature_ids = sae_features.feature_indices.cpu().numpy()
        
        # Get layer weights
        layer_weights = weight_extractor.get_layer(
            layer_idx,
            device=viz_config.device,
            dtype=viz_config.dtype,
        )
        
        for head_idx in viz_config.heads:
            # Get per-head weights
            W_Q, W_K, W_V = get_qkv_for_head(layer_weights, head_idx, config)
            W_O = layer_weights.W_O[head_idx]
            
            # Compute matrices (torch, on GPU)
            M_QK, M_OVQ, M_OVK = compute_rgb_matrices_torch(
                decoder, W_Q, W_K, W_V, W_O, config
            )
            
            # Compute head metrics before reordering
            metrics = compute_head_metrics(M_QK, M_OVQ, M_OVK)
            metrics["layer"] = layer_idx
            metrics["head"] = head_idx
            all_metrics.append(metrics)
            
            # Save top pairs before reordering (use original feature IDs)
            if viz_config.save_top_pairs:
                pairs_dir = os.path.join(viz_config.outdir, "top_pairs")
                os.makedirs(pairs_dir, exist_ok=True)
                
                for name, mat in [("qk", M_QK), ("ovq", M_OVQ), ("ovk", M_OVK)]:
                    pairs = topk_pairs(mat, feature_ids, k=viz_config.top_k_pairs)
                    path = os.path.join(pairs_dir, f"L{layer_idx}_H{head_idx}_{name}.json")
                    with open(path, "w") as f:
                        json.dump(pairs, f, indent=2)
            
            # Reorder by degree if enabled
            if viz_config.reorder_by_degree:
                order = order_by_degree(M_QK)
                M_QK = apply_reorder(M_QK, order)
                M_OVQ = apply_reorder(M_OVQ, order)
                M_OVK = apply_reorder(M_OVK, order)
                feature_ids_ordered = feature_ids[order.cpu().numpy()]
                
                # Save order mapping
                order_dir = os.path.join(viz_config.outdir, "orderings")
                os.makedirs(order_dir, exist_ok=True)
                np.save(os.path.join(order_dir, f"L{layer_idx}_H{head_idx}_order.npy"), 
                        order.cpu().numpy())
            else:
                feature_ids_ordered = feature_ids
            
            # Split into positive and negative
            M_QK_pos, M_QK_neg = split_pos_neg(M_QK)
            M_OVQ_pos, M_OVQ_neg = split_pos_neg(M_OVQ)
            M_OVK_pos, M_OVK_neg = split_pos_neg(M_OVK)
            
            # Create RGB images (POS and NEG)
            rgb_pos = make_rgb_float(M_QK_pos, M_OVQ_pos, M_OVK_pos, 
                                     viz_config.percentile, viz_config.norm_mode)
            rgb_neg = make_rgb_float(M_QK_neg, M_OVQ_neg, M_OVK_neg, 
                                     viz_config.percentile, viz_config.norm_mode)
            
            # Create thumbnails for grid (area downsampling)
            thumb_pos = downsample_rgb_area(rgb_pos, viz_config.grid_thumbnail_size)
            thumb_neg = downsample_rgb_area(rgb_neg, viz_config.grid_thumbnail_size)
            
            # Convert to uint8 for storage
            thumb_pos_uint8 = (thumb_pos * 255).to(torch.uint8).cpu().numpy()
            thumb_neg_uint8 = (thumb_neg * 255).to(torch.uint8).cpu().numpy()
            
            rgb_thumbs_pos[(layer_idx, head_idx)] = thumb_pos_uint8
            rgb_thumbs_neg[(layer_idx, head_idx)] = thumb_neg_uint8
            
            # Save individual plots
            if viz_config.save_individual:
                rgb_pos_uint8 = (rgb_pos * 255).to(torch.uint8).cpu().numpy()
                rgb_neg_uint8 = (rgb_neg * 255).to(torch.uint8).cpu().numpy()
                
                plot_head_rgb(rgb_pos_uint8, layer_idx, head_idx,
                              outdir=viz_config.outdir, suffix="pos", show=viz_config.show)
                plot_head_rgb(rgb_neg_uint8, layer_idx, head_idx,
                              outdir=viz_config.outdir, suffix="neg", show=viz_config.show)
    
    # Save grid views
    if viz_config.save_grid:
        plot_layer_head_grid(rgb_thumbs_pos, layers=viz_config.layers, heads=viz_config.heads,
                             outdir=viz_config.outdir, suffix="pos", show=viz_config.show)
        plot_layer_head_grid(rgb_thumbs_neg, layers=viz_config.layers, heads=viz_config.heads,
                             outdir=viz_config.outdir, suffix="neg", show=viz_config.show)
        print(f"Saved grid views to {viz_config.outdir}/layer_head_grid_pos.png and _neg.png")
    
    # Save metrics CSV
    if viz_config.save_metrics_csv and all_metrics:
        import csv
        csv_path = os.path.join(viz_config.outdir, "head_metrics.csv")
        fieldnames = ["layer", "head"] + [k for k in all_metrics[0].keys() if k not in ("layer", "head")]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"Saved head metrics to {csv_path}")
    
    return rgb_thumbs_pos


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate RGB composite visualizations for attention heads."
    )
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Layer indices to visualize (default: all except 0)")
    parser.add_argument("--heads", nargs="+", type=int, default=None,
                        help="Head indices to visualize (default: all)")
    parser.add_argument("--feature-subset", type=int, default=4096,
                        help="Number of SAE features to use (default: 4096)")
    parser.add_argument("--outdir", type=str, default="./rgb_outputs",
                        help="Output directory (default: ./rgb_outputs)")
    parser.add_argument("--percentile", type=int, default=99,
                        help="Percentile for normalization (default: 99)")
    parser.add_argument("--norm-mode", choices=["per_channel", "shared_per_head"],
                        default="shared_per_head",
                        help="Normalization mode (default: shared_per_head)")
    parser.add_argument("--no-reorder", action="store_true",
                        help="Disable reordering by degree")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively")
    parser.add_argument("--no-individual", action="store_true",
                        help="Skip saving individual head plots")
    parser.add_argument("--no-grid", action="store_true",
                        help="Skip saving grid view")
    parser.add_argument("--no-pairs", action="store_true",
                        help="Skip saving top pairs JSON")
    parser.add_argument("--no-metrics", action="store_true",
                        help="Skip saving metrics CSV")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA2_CONFIG.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    
    # Set up configuration
    layers = args.layers if args.layers else list(range(1, GEMMA2_CONFIG.num_hidden_layers))
    heads = args.heads if args.heads else list(range(GEMMA2_CONFIG.num_attention_heads))
    
    viz_config = RGBVizConfig(
        layers=layers,
        heads=heads,
        feature_subset=args.feature_subset,
        outdir=args.outdir,
        show=args.show,
        save_individual=not args.no_individual,
        save_grid=not args.no_grid,
        percentile=args.percentile,
        norm_mode=args.norm_mode,
        reorder_by_degree=not args.no_reorder,
        save_top_pairs=not args.no_pairs,
        save_metrics_csv=not args.no_metrics,
        device=args.device,
        dtype=torch.float16,
    )
    
    # Run visualization
    rgb_images = run_rgb_visualization(model, GEMMA2_CONFIG, viz_config)
    
    print(f"\nGenerated {len(rgb_images)} RGB visualizations")
    print(f"Output saved to: {args.outdir}")


if __name__ == "__main__":
    main()
