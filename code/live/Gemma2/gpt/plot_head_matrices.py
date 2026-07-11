"""plot_head_matrices.py

Compute and plot per-head matrices for the blog exemplars:

  - Routing affinity matrix B (Q_f @ K_f^T with Gemma-2 scaling + softcap)
  - Write-to-feature matrix W2F (cosine similarity between write vectors and decoder directions)

This fills draft placeholders like:
  Fig 1 (a/b/c): routing exemplars (diagonal / off-diagonal / avoidance)
  Fig 2 (a/b/c): writing exemplars (TRANSFORM / BROADCAST / COPY)

It *does* reload model weights + SAE decoders, because the analysis JSON
does not store full matrices (too large).

Usage:
  python plot_head_matrices.py \
      --layer 10 --head 5 \
      --feature-subset 4096 \
      --outdir ./blog_assets/exemplars

Notes:
  - This uses the same RMS-normalized decoder directions as sae_utils.py.
  - Reordering (by routing degree) is enabled by default to reveal structure.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import GEMMA2_CONFIG, AnalysisConfig
from sae_utils import SAEManager, project_to_qk_space, project_writes_to_features
from weight_extraction import WeightExtractor, get_qkv_for_head
from qk_routing import compute_affinity_matrix, extract_top_pairs
from ov_writing import compute_write_vectors_fast


@torch.no_grad()
def order_by_degree(M: torch.Tensor) -> torch.Tensor:
    """Reorder indices by absolute "degree" to reveal blocks."""
    deg = M.abs().sum(dim=0) + M.abs().sum(dim=1)
    return torch.argsort(deg, descending=True)


@torch.no_grad()
def apply_reorder(M: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    return M[order][:, order]


def _quantile_clip(x: np.ndarray, q: float = 0.995) -> Tuple[float, float]:
    hi = float(np.quantile(x, q))
    lo = float(np.quantile(x, 1 - q))
    # symmetric for nicer diverging maps
    m = max(abs(lo), abs(hi))
    return -m, m


def _imshow_matrix(
    M: torch.Tensor,
    title: str,
    outpath: str,
    downsample: int = 512,
    quantile: float = 0.995,
    cmap: str = "coolwarm",
):
    """Save a downsampled heatmap."""
    M = M.detach().float().cpu()

    # Downsample with area averaging (anti-aliasing)
    if downsample and (M.shape[0] > downsample):
        x = M.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        x = F.interpolate(x, size=(downsample, downsample), mode="area")
        M = x.squeeze(0).squeeze(0)

    arr = M.numpy()
    vmin, vmax = _quantile_clip(arr.flatten(), q=quantile)

    plt.figure(figsize=(6.2, 5.4))
    plt.imshow(arr, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--head", type=int, required=True)
    ap.add_argument("--feature-subset", type=int, default=4096)
    ap.add_argument("--outdir", type=str, default="./blog_assets/exemplars")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--no-reorder", action="store_true")
    ap.add_argument("--downsample", type=int, default=512)
    ap.add_argument("--save-top-pairs", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Load model
    from transformers import AutoModelForCausalLM

    print(f"Loading model: {GEMMA2_CONFIG.model_id}")
    # HuggingFace uses `device_map` (accelerate) for placement.
    # For a single-GPU setup, `device_map="auto"` is the least surprising.
    # (Passing e.g. "cuda" directly is *not* a valid device_map.)
    device_map = "auto" if str(args.device).startswith("cuda") else {"": "cpu"}
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA2_CONFIG.model_id,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()

    # Managers
    cfg = GEMMA2_CONFIG
    sae_manager = SAEManager(cfg, subset_size=args.feature_subset)
    weight_extractor = WeightExtractor(model, cfg, fold_gamma=cfg.fold_rmsnorm_gamma)

    # SAE layer offset (matches attention input)
    sae_layer = cfg.get_sae_layer_for_attn(args.layer)
    sae = sae_manager.get_features(sae_layer, device=args.device, dtype=dtype, compute_gram=False)
    decoder = sae.decoder_subset  # [F, d_model]
    feat_ids = sae.feature_indices.detach().cpu().numpy()

    # Weights
    lw = weight_extractor.get_layer(args.layer, device=args.device, dtype=dtype)
    W_Q, W_K, W_V = get_qkv_for_head(lw, args.head, cfg)
    W_O = lw.W_O[args.head]

    # Routing matrix (B)
    Q_f, K_f = project_to_qk_space(decoder, W_Q, W_K)
    B = compute_affinity_matrix(Q_f, K_f, cfg)  # includes scale + softcap if enabled

    # Write matrix (W2F)
    write_vecs = compute_write_vectors_fast(decoder, W_V, W_O)
    W2F = project_writes_to_features(write_vecs, decoder)  # cosine similarities

    order = None
    if not args.no_reorder:
        order = order_by_degree(B)
        B = apply_reorder(B, order)
        W2F = apply_reorder(W2F, order)
        feat_ids = feat_ids[order.detach().cpu().numpy()]

    # Save heatmaps
    base = f"L{args.layer}_H{args.head}_F{args.feature_subset}"
    b_path = os.path.join(args.outdir, f"{base}_B_logits.png")
    w_path = os.path.join(args.outdir, f"{base}_W2F_cos.png")

    _imshow_matrix(B, title=f"Routing B (logits, softcapped) — {base}", outpath=b_path, downsample=args.downsample)
    _imshow_matrix(W2F, title=f"Write-to-feature W2F (cosine) — {base}", outpath=w_path, downsample=args.downsample)

    print(f"Wrote: {b_path}")
    print(f"Wrote: {w_path}")

    # Optional: top pairs JSON for captions
    if args.save_top_pairs:
        pairs = extract_top_pairs(B, k=50)
        out = {
            "layer": args.layer,
            "head": args.head,
            "feature_subset": args.feature_subset,
            "reordered": (order is not None),
            "feature_ids_in_order": feat_ids.tolist(),
            "top_positive_pairs": [(int(i), int(j), float(s)) for (i, j, s) in pairs.positive_pairs],
            "top_negative_pairs": [(int(i), int(j), float(s)) for (i, j, s) in pairs.negative_pairs],
        }
        p_path = os.path.join(args.outdir, f"{base}_top_pairs.json")
        with open(p_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote: {p_path}")


if __name__ == "__main__":
    main()
