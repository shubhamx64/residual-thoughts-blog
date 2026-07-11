# gemma3_full_connectome_v2.py
# Full weight-space connectome analysis for Gemma3 models
#
# Key features:
#  1) Global + forward-only summaries (and forward-weighted) for Q/K preference.
#  2) Vectorized neuron analysis.
#  3) Q/K read bases include pre-attn RMSNorm gamma (+ optional q_norm/k_norm).
#  4) WRITE gamma is evaluated in TWO built-in flavors (no runtime arg):
#       - write_gamma=none
#       - write_gamma=pre_attn
#     Each flavor gets its own output folder and plots.
#  5) Random-subspace baseline per (d_model, K).
#  6) NaN-safe plotting/metrics while keeping strict JSON (NaN->null) at serialization time.
#  7) Spectral profiling is OPTIONAL (default off) to avoid nuking runtime on 4B.
#
# Usage:
#   python gemma3_full_connectome_v2.py --model google/gemma-3-1b-pt --outdir out_1b --K 32,64,128
#   python gemma3_full_connectome_v2.py --model google/gemma-3-4b-pt --outdir out_4b --K 32,64,128 --do_spectral 0
#
# Optional:
#   --do_spectral 1 --spectral_layers 0,mid,last

import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from transformers import AutoModelForCausalLM

EPS = 1e-12

# =============================================================================
# JSON + numeric helpers
# =============================================================================

def to_float_array(x) -> np.ndarray:
    """
    Convert list possibly containing None -> float32 array with NaN.
    Also passes through numpy arrays untouched (cast to float32).
    """
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    return np.array([np.nan if v is None else float(v) for v in x], dtype=np.float32)

def nan_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Correlation computed only on indices where both are finite.
    Returns 0.0 if degenerate.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return 0.0
    aa = a[m]
    bb = b[m]
    if np.nanstd(aa) < 1e-8 or np.nanstd(bb) < 1e-8:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])

def sanitize_for_json(obj):
    """
    Strict JSON sanitization:
      - np.ndarray -> list with NaN -> None
      - floats NaN -> None
      - torch tensors -> same
    """
    if isinstance(obj, torch.Tensor):
        return sanitize_for_json(obj.detach().cpu().numpy())
    if isinstance(obj, np.ndarray):
        out = []
        for v in obj.tolist():
            if v is None:
                out.append(None)
            else:
                try:
                    fv = float(v)
                    out.append(None if np.isnan(fv) else fv)
                except Exception:
                    out.append(None)
        return out
    if isinstance(obj, (np.floating, float)):
        fv = float(obj)
        return None if np.isnan(fv) else fv
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj

# =============================================================================
# Linear Algebra Helpers
# =============================================================================

def _svd_left(M: torch.Tensor, k: int) -> torch.Tensor:
    """Economy SVD; returns U[:, :k] (left singular vectors)."""
    # Note: this is the expensive part by design (per your request).
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    k_eff = min(k, U.shape[1])
    return U[:, :k_eff].contiguous()

def _orth(U: torch.Tensor) -> torch.Tensor:
    """Orthonormalize columns via QR."""
    Q, _ = torch.linalg.qr(U, mode="reduced")
    return Q

def subspace_align(U: torch.Tensor, V: torch.Tensor, mode: str = "mean") -> float:
    """
    Principal-angle similarity between subspaces spanned by U and V.
    Returns mean/max/top8 of cosines of principal angles.
    """
    Uo = _orth(U)
    Vo = _orth(V)
    S = Uo.T @ Vo
    _, sv, _ = torch.linalg.svd(S, full_matrices=False)
    sv = torch.clamp(sv, 0.0, 1.0)

    if mode == "mean":
        return float(sv.mean().item())
    if mode == "top8":
        m = min(8, sv.numel())
        return float(sv[:m].mean().item())
    if mode == "max":
        return float(sv.max().item())
    raise ValueError("mode must be one of: mean, top8, max")

def concat_bases(bases: List[torch.Tensor], k: int) -> torch.Tensor:
    """Concatenate bases and compress back to rank-k via SVD."""
    if not bases:
        raise ValueError("No bases to concatenate")
    X = torch.cat(bases, dim=1)
    return _svd_left(X, k)

# =============================================================================
# Model Loading & Architecture Detection
# =============================================================================

def get_layers(model) -> torch.nn.ModuleList:
    """
    Return the text decoder ModuleList for Gemma3 models.
    Handles both text-only and multimodal variants.
    """
    candidate_paths = [
        "model.layers",
        "model.model.layers",
        "model.language_model.layers",
        "language_model.layers",
        "model.text_model.layers",
    ]

    def try_get(obj, path: str):
        cur = obj
        for part in path.split("."):
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
        return cur

    for p in candidate_paths:
        x = try_get(model, p)
        if isinstance(x, torch.nn.ModuleList) and len(x) > 0:
            if hasattr(x[0], "self_attn") and hasattr(x[0], "mlp"):
                return x

    best, best_len = None, -1
    for _, child in model.named_modules():
        if isinstance(child, torch.nn.ModuleList):
            try:
                L = len(child)
                if L > best_len and hasattr(child[0], "self_attn") and hasattr(child[0], "mlp"):
                    best, best_len = child, L
            except Exception:
                continue

    if best is None:
        raise AttributeError("Could not find Gemma3 decoder layers")
    return best

@dataclass
class ModelShapes:
    d_model: int
    d_ff: int
    head_dim: int
    num_q_heads: int
    num_kv_heads: int
    group_size: int
    n_layers: int

def _get_config_attr(model, name: str):
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    # Gemma multimodal sometimes has text_config
    tc = getattr(cfg, "text_config", None)
    if tc is not None and hasattr(tc, name):
        return getattr(tc, name)
    if hasattr(cfg, name):
        return getattr(cfg, name)
    return None

def infer_shapes(model, model_layers: torch.nn.ModuleList) -> ModelShapes:
    """
    Less brittle shape inference:
      - prefer config (num_attention_heads / num_key_value_heads)
      - then module attrs (num_heads / num_key_value_heads)
      - then q_norm weight length as head_dim
      - asserts on divisibility
    """
    layer0 = model_layers[0]
    attn = layer0.self_attn
    mlp = layer0.mlp

    W_Q = attn.q_proj.weight
    W_K = attn.k_proj.weight

    d_model = int(W_Q.shape[1])
    q_out = int(W_Q.shape[0])
    kv_out = int(W_K.shape[0])

    # Prefer config
    num_q_heads = _get_config_attr(model, "num_attention_heads")
    num_kv_heads = _get_config_attr(model, "num_key_value_heads")

    # Fallback module attrs
    if num_q_heads is None and hasattr(attn, "num_heads"):
        num_q_heads = int(getattr(attn, "num_heads"))
    if num_q_heads is None and hasattr(attn, "num_attention_heads"):
        num_q_heads = int(getattr(attn, "num_attention_heads"))

    if num_kv_heads is None and hasattr(attn, "num_key_value_heads"):
        num_kv_heads = int(getattr(attn, "num_key_value_heads"))
    if num_kv_heads is None and hasattr(attn, "num_kv_heads"):
        num_kv_heads = int(getattr(attn, "num_kv_heads"))

    # Infer head_dim
    head_dim = None

    if num_q_heads is not None:
        assert q_out % int(num_q_heads) == 0, f"q_out={q_out} not divisible by num_q_heads={num_q_heads}"
        head_dim = q_out // int(num_q_heads)

    if head_dim is None and hasattr(attn, "q_norm") and hasattr(attn.q_norm, "weight"):
        # q_norm.weight is [head_dim] for Gemma-style per-head norm
        head_dim = int(attn.q_norm.weight.numel())

    if head_dim is None:
        # last resort: try to deduce from o_proj input width and plausible head count
        o_in = int(attn.o_proj.weight.shape[1])  # usually num_q_heads * head_dim
        # Try a few common head dims (Gemma3 uses 256)
        for cand in [64, 80, 96, 128, 160, 192, 256]:
            if q_out % cand == 0 and o_in % cand == 0:
                head_dim = cand
                break

    if head_dim is None:
        raise ValueError("Could not infer head_dim reliably. Please inspect attn module shapes.")

    assert q_out % head_dim == 0, f"q_out={q_out} not divisible by head_dim={head_dim}"
    num_q_heads = q_out // head_dim if num_q_heads is None else int(num_q_heads)

    assert kv_out % head_dim == 0, f"kv_out={kv_out} not divisible by head_dim={head_dim}"
    num_kv_heads = kv_out // head_dim if num_kv_heads is None else int(num_kv_heads)

    group_size = num_q_heads // max(1, num_kv_heads)
    assert group_size >= 1, "group_size computed < 1"

    d_ff = int(mlp.gate_proj.weight.shape[0])

    return ModelShapes(
        d_model=d_model,
        d_ff=d_ff,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        group_size=group_size,
        n_layers=len(model_layers),
    )

# =============================================================================
# Norm handling (RMSNorm gammas)
# =============================================================================

def _get_norm_weight(obj, candidates: List[str]) -> Optional[torch.Tensor]:
    """
    Try common attribute names for RMSNorm/LayerNorm modules with a `.weight`.
    Returns the weight tensor or None.
    """
    for name in candidates:
        if hasattr(obj, name):
            mod = getattr(obj, name)
            if hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor):
                return mod.weight.detach()
    return None

def get_pre_attn_gamma(layer) -> Optional[torch.Tensor]:
    return _get_norm_weight(layer, [
        "input_layernorm",
        "pre_attention_layernorm",
        "attention_norm",
        "self_attn_layer_norm",
        "attn_norm",
    ])

def get_pre_mlp_gamma(layer) -> Optional[torch.Tensor]:
    """
    Gemma3: pre_feedforward_layernorm is applied to MLP input.
    """
    return _get_norm_weight(layer, [
        "pre_feedforward_layernorm",
        "post_attention_layernorm",
        "ffn_norm",
        "mlp_norm",
        "ff_norm",
    ])

def apply_input_gamma(W: torch.Tensor, gamma: Optional[torch.Tensor]) -> torch.Tensor:
    """
    If x' = diag(gamma) x, then W x' = (W diag(gamma)) x.
    So right-multiply by diag(gamma) => scale columns of W by gamma.
    W: [*, d_model], gamma: [d_model]
    """
    if gamma is None:
        return W
    return W * gamma.view(1, -1)

# =============================================================================
# Basis Extraction: Attention
# =============================================================================

def build_attn_write_basis(layer, head_idx: int, shapes: ModelShapes, k: int, write_gamma_mode: str) -> torch.Tensor:
    """
    Attention write subspace for a single head:
        M = W_O(h) @ W_V(kv(h))
    Two flavors:
      - write_gamma_mode="none": M
      - write_gamma_mode="pre_attn": M_eff = W_O @ (W_V diag(gamma_pre_attn))
        This can change TOP-K directions (even if full span doesn't), which is exactly why we compare both.
    """
    attn = layer.self_attn
    W_V = attn.v_proj.weight.detach()
    W_O = attn.o_proj.weight.detach()

    hd = shapes.head_dim
    h = head_idx

    kv_head = h // shapes.group_size if shapes.num_kv_heads > 1 else 0

    W_V_slice = W_V[kv_head * hd : (kv_head + 1) * hd, :]  # [hd, d_model]
    W_O_slice = W_O[:, h * hd : (h + 1) * hd]              # [d_model, hd]

    if write_gamma_mode == "pre_attn":
        pre_attn_gamma = get_pre_attn_gamma(layer)
        W_V_slice = apply_input_gamma(W_V_slice, pre_attn_gamma)
    elif write_gamma_mode == "none":
        pass
    else:
        raise ValueError("write_gamma_mode must be one of: none, pre_attn")

    M = W_O_slice @ W_V_slice  # [d_model, d_model]
    return _svd_left(M, k)

def build_attn_Q_basis(layer, head_idx: int, shapes: ModelShapes, k: int, use_qk_norm_gamma: bool = True) -> torch.Tensor:
    """
    Query read subspace for a single head: col(W_Q_eff^T).
    W_Q_eff includes pre-attn RMSNorm gamma and (optional) q_norm gamma.
    Shape: [d_model, k]
    """
    attn = layer.self_attn
    W_Q = attn.q_proj.weight.detach()

    hd = shapes.head_dim
    h = head_idx

    pre_attn_gamma = get_pre_attn_gamma(layer)

    W_Q_slice = W_Q[h * hd : (h + 1) * hd, :]                  # [hd, d_model]
    W_Q_slice = apply_input_gamma(W_Q_slice, pre_attn_gamma)

    if use_qk_norm_gamma and hasattr(attn, "q_norm"):
        gamma_h = attn.q_norm.weight.detach()                  # [hd]
        W_Q_slice = gamma_h.view(-1, 1) * W_Q_slice

    return _svd_left(W_Q_slice.T, k)                            # [d_model, k]

def build_attn_K_basis(layer, kv_head_idx: int, shapes: ModelShapes, k: int, use_qk_norm_gamma: bool = True) -> torch.Tensor:
    """
    Key read subspace for a single KV head: col(W_K_eff^T).
    W_K_eff includes pre-attn RMSNorm gamma and (optional) k_norm gamma.
    Shape: [d_model, k]
    """
    attn = layer.self_attn
    W_K = attn.k_proj.weight.detach()

    hd = shapes.head_dim
    kv = kv_head_idx

    pre_attn_gamma = get_pre_attn_gamma(layer)

    W_K_slice = W_K[kv * hd : (kv + 1) * hd, :]                # [hd, d_model]
    W_K_slice = apply_input_gamma(W_K_slice, pre_attn_gamma)

    if use_qk_norm_gamma and hasattr(attn, "k_norm"):
        gamma_h = attn.k_norm.weight.detach()                  # [hd]
        W_K_slice = gamma_h.view(-1, 1) * W_K_slice

    return _svd_left(W_K_slice.T, k)                            # [d_model, k]

# =============================================================================
# Basis Extraction: MLP
# =============================================================================

def build_mlp_write_basis(layer, k: int) -> torch.Tensor:
    """MLP write subspace: col(W_down)."""
    W_down = layer.mlp.down_proj.weight.detach()  # [d_model, d_ff]
    return _svd_left(W_down, k)

def build_mlp_read_basis(layer, k: int, mode: str = "combined") -> torch.Tensor:
    """
    MLP read subspace (row space proxy of gate/up), with pre-MLP RMSNorm gamma.
    Shape: [d_model, k]
    """
    mlp = layer.mlp
    W_gate = mlp.gate_proj.weight.detach()  # [d_ff, d_model]
    W_up   = mlp.up_proj.weight.detach()    # [d_ff, d_model]

    pre_mlp_gamma = get_pre_mlp_gamma(layer)

    W_gate = apply_input_gamma(W_gate, pre_mlp_gamma)
    W_up   = apply_input_gamma(W_up,   pre_mlp_gamma)

    if mode == "combined":
        combined = torch.cat([W_gate, W_up], dim=0)  # [2*d_ff, d_model]
        return _svd_left(combined.T, k)              # [d_model, k]
    if mode == "gate":
        return _svd_left(W_gate.T, k)
    if mode == "up":
        return _svd_left(W_up.T, k)
    raise ValueError(f"Unknown mode: {mode}")

# =============================================================================
# Neuron-Level Analysis (vectorized)
# =============================================================================

def analyze_mlp_neurons(layer, top_n: int = 100) -> dict:
    """
    Analyze MLP neurons as key-value pairs (vectorized).
    """
    mlp = layer.mlp
    W_up   = mlp.up_proj.weight.detach()      # [d_ff, d_model]
    W_gate = mlp.gate_proj.weight.detach()    # [d_ff, d_model]
    W_down = mlp.down_proj.weight.detach()    # [d_model, d_ff]
    W_down_T = W_down.T.contiguous()          # [d_ff, d_model]

    dots = (W_up * W_down_T).sum(dim=1)
    up_norm = W_up.norm(dim=1)
    down_norm = W_down_T.norm(dim=1)
    kv = (dots / (up_norm * down_norm + EPS)).cpu().numpy()

    dots2 = (W_gate * W_up).sum(dim=1)
    gate_norm = W_gate.norm(dim=1)
    gate_key = (dots2 / (gate_norm * up_norm + EPS)).cpu().numpy()

    key_norms = up_norm.cpu().numpy()
    val_norms = down_norm.cpu().numpy()

    idx = np.argsort(-np.abs(kv))[:top_n]
    top_vals = kv[idx].tolist()

    return {
        "kv_alignment_mean": float(np.mean(kv)),
        "kv_alignment_std": float(np.std(kv)),
        "kv_alignment_positive_frac": float((kv > 0).mean()),
        "gate_key_alignment_mean": float(np.mean(gate_key)),
        "top_kv_alignments": top_vals,
        "key_norm_mean": float(np.mean(key_norms)),
        "val_norm_mean": float(np.mean(val_norms)),
    }

# =============================================================================
# Per-Layer Basis Computation
# =============================================================================

@dataclass
class LayerBases:
    attn_write: torch.Tensor
    attn_Q: torch.Tensor
    attn_K: torch.Tensor
    mlp_write: torch.Tensor
    mlp_read: torch.Tensor

def compute_layer_bases(layer, shapes: ModelShapes, k: int, write_gamma_mode: str) -> LayerBases:
    attn_write_bases = [build_attn_write_basis(layer, h, shapes, k, write_gamma_mode=write_gamma_mode)
                        for h in range(shapes.num_q_heads)]
    attn_write = concat_bases(attn_write_bases, k)

    attn_Q_bases = [build_attn_Q_basis(layer, h, shapes, k) for h in range(shapes.num_q_heads)]
    attn_Q = concat_bases(attn_Q_bases, k)

    attn_K_bases = [build_attn_K_basis(layer, kv, shapes, k) for kv in range(shapes.num_kv_heads)]
    attn_K = concat_bases(attn_K_bases, k)

    mlp_write = build_mlp_write_basis(layer, k)
    mlp_read  = build_mlp_read_basis(layer, k, mode="combined")

    return LayerBases(
        attn_write=attn_write,
        attn_Q=attn_Q,
        attn_K=attn_K,
        mlp_write=mlp_write,
        mlp_read=mlp_read,
    )

# =============================================================================
# Cross-Component Alignment Matrices
# =============================================================================

@dataclass
class ConnectomeMatrices:
    attn_write_to_Q: np.ndarray
    attn_write_to_K: np.ndarray
    mlp_write_to_read: np.ndarray
    attn_write_to_mlp_read: np.ndarray
    mlp_write_to_Q: np.ndarray
    mlp_write_to_K: np.ndarray
    attn_mlp_write_overlap: np.ndarray

def compute_connectome(all_bases: List[LayerBases], n_layers: int) -> ConnectomeMatrices:
    attn_write_to_Q = np.zeros((n_layers, n_layers), dtype=np.float32)
    attn_write_to_K = np.zeros((n_layers, n_layers), dtype=np.float32)
    mlp_write_to_read = np.zeros((n_layers, n_layers), dtype=np.float32)
    attn_write_to_mlp_read = np.zeros((n_layers, n_layers), dtype=np.float32)
    mlp_write_to_Q = np.zeros((n_layers, n_layers), dtype=np.float32)
    mlp_write_to_K = np.zeros((n_layers, n_layers), dtype=np.float32)
    attn_mlp_write_overlap = np.zeros(n_layers, dtype=np.float32)

    for i in range(n_layers):
        src = all_bases[i]
        attn_mlp_write_overlap[i] = subspace_align(src.attn_write, src.mlp_write)

        for j in range(n_layers):
            tgt = all_bases[j]
            attn_write_to_Q[i, j] = subspace_align(src.attn_write, tgt.attn_Q)
            attn_write_to_K[i, j] = subspace_align(src.attn_write, tgt.attn_K)
            mlp_write_to_read[i, j] = subspace_align(src.mlp_write, tgt.mlp_read)

            attn_write_to_mlp_read[i, j] = subspace_align(src.attn_write, tgt.mlp_read)
            mlp_write_to_Q[i, j] = subspace_align(src.mlp_write, tgt.attn_Q)
            mlp_write_to_K[i, j] = subspace_align(src.mlp_write, tgt.attn_K)

    return ConnectomeMatrices(
        attn_write_to_Q=attn_write_to_Q,
        attn_write_to_K=attn_write_to_K,
        mlp_write_to_read=mlp_write_to_read,
        attn_write_to_mlp_read=attn_write_to_mlp_read,
        mlp_write_to_Q=mlp_write_to_Q,
        mlp_write_to_K=mlp_write_to_K,
        attn_mlp_write_overlap=attn_mlp_write_overlap,
    )

# =============================================================================
# Random baseline for subspace alignment
# =============================================================================

@torch.no_grad()
def random_subspace_baseline(d_model: int, K: int, trials: int = 200, seed: int = 0, device: str = "cpu") -> dict:
    """
    Estimate E[subspace_align] for two random K-dim subspaces in R^{d_model}.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    vals = []
    for _ in range(trials):
        A = torch.randn(d_model, K, generator=g, device=device)
        B = torch.randn(d_model, K, generator=g, device=device)
        QA = _orth(A)
        QB = _orth(B)
        vals.append(subspace_align(QA, QB, mode="mean"))
    vals = np.array(vals, dtype=np.float32)
    return {"mean": float(vals.mean()), "std": float(vals.std()), "trials": int(trials), "seed": int(seed)}

# =============================================================================
# Phase / "causal-ish" summaries
# =============================================================================

def masked_row_mean(M: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Per-row mean over entries where mask=True.
    Returns NaN for rows with no valid entries.
    """
    masked = np.where(mask, M, 0.0).astype(np.float64)
    counts = mask.sum(axis=1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = masked.sum(axis=1) / counts
    return result.astype(np.float32)

def forward_masks(n_layers: int, exclude_diag: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n_layers)
    I, J = np.meshgrid(idx, idx, indexing="ij")
    all_mask = (I != J) if exclude_diag else np.ones((n_layers, n_layers), dtype=bool)
    fwd_mask = (J > I)
    bwd_mask = (J < I)
    return all_mask, fwd_mask, bwd_mask

def forward_weighted_row_mean(M: np.ndarray, half_life: float = 4.0) -> np.ndarray:
    """
    Distance-weighted forward mean (j>i).
    Returns NaN for layers with no forward targets.
    """
    n = M.shape[0]
    idx = np.arange(n)
    I, J = np.meshgrid(idx, idx, indexing="ij")
    dist = (J - I - 1).astype(np.float32)

    w = np.zeros_like(M, dtype=np.float32)
    valid = dist >= 0
    lam = np.log(2.0) / max(half_life, 1e-6)
    w[valid] = np.exp(-lam * dist[valid])

    denom = w.sum(axis=1).astype(np.float64)
    num = (M * w).sum(axis=1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = num / denom
    return result.astype(np.float32)

def detect_phase_transitions(
    qk_diff: np.ndarray,
    smooth_window: int = 3,
    exclude_edges: int = 1,
    min_jump: float = 0.002,
    return_details: bool = False,
):
    """
    Detect sign changes in smoothed signal.
    Uses sharpness to filter jitter: keep transitions with |delta| >= min_jump.
    """
    qk_diff = np.asarray(qk_diff, dtype=np.float32)
    n = len(qk_diff)

    if exclude_edges > 0 and n > 2 * exclude_edges:
        work = qk_diff[exclude_edges:-exclude_edges]
        offset = exclude_edges
    else:
        work = qk_diff
        offset = 0

    if len(work) < smooth_window + 1:
        return {"layers": [], "sharpness": []} if return_details else []

    # Smoothing with NaNs: fill them to 0 for smoothing; edges are excluded anyway.
    work_s = np.nan_to_num(work, nan=0.0)
    smoothed = uniform_filter1d(work_s, size=smooth_window)

    signs = np.sign(smoothed)
    sign_changes = np.where(np.diff(signs) != 0)[0]

    layers = []
    sharpness = []
    for idx in sign_changes:
        delta = float(abs(smoothed[idx + 1] - smoothed[idx]))
        if delta >= min_jump:
            layers.append(int(idx + 1 + offset))
            sharpness.append(delta)

    if not return_details:
        return layers

    return {"layers": layers, "sharpness": sharpness, "min_jump": float(min_jump)}

def compute_phase_stats(
    matrices: ConnectomeMatrices,
    n_layers: int,
    half_life: float = 4.0,
    exclude_diag: bool = False,
    min_jump: float = 0.002,
) -> dict:
    """
    Returns arrays as numpy float32 (with NaNs). Serialization happens later.
    """
    all_mask, fwd_mask, bwd_mask = forward_masks(n_layers, exclude_diag=exclude_diag)

    attn_row_Q_all = masked_row_mean(matrices.attn_write_to_Q, all_mask)
    attn_row_K_all = masked_row_mean(matrices.attn_write_to_K, all_mask)
    attn_qk_diff_all = attn_row_Q_all - attn_row_K_all

    attn_row_Q_fwd = masked_row_mean(matrices.attn_write_to_Q, fwd_mask)
    attn_row_K_fwd = masked_row_mean(matrices.attn_write_to_K, fwd_mask)
    attn_qk_diff_fwd = attn_row_Q_fwd - attn_row_K_fwd

    attn_row_Q_fwd_w = forward_weighted_row_mean(matrices.attn_write_to_Q, half_life=half_life)
    attn_row_K_fwd_w = forward_weighted_row_mean(matrices.attn_write_to_K, half_life=half_life)
    attn_qk_diff_fwd_w = attn_row_Q_fwd_w - attn_row_K_fwd_w

    mlp_row_Q_all = masked_row_mean(matrices.mlp_write_to_Q, all_mask)
    mlp_row_K_all = masked_row_mean(matrices.mlp_write_to_K, all_mask)
    mlp_qk_diff_all = mlp_row_Q_all - mlp_row_K_all

    mlp_row_Q_fwd = masked_row_mean(matrices.mlp_write_to_Q, fwd_mask)
    mlp_row_K_fwd = masked_row_mean(matrices.mlp_write_to_K, fwd_mask)
    mlp_qk_diff_fwd = mlp_row_Q_fwd - mlp_row_K_fwd

    mlp_row_Q_fwd_w = forward_weighted_row_mean(matrices.mlp_write_to_Q, half_life=half_life)
    mlp_row_K_fwd_w = forward_weighted_row_mean(matrices.mlp_write_to_K, half_life=half_life)
    mlp_qk_diff_fwd_w = mlp_row_Q_fwd_w - mlp_row_K_fwd_w

    # Forward diffs have NaN at last layer; exclude_edges=1 is appropriate.
    attn_transitions_fwd = detect_phase_transitions(attn_qk_diff_fwd, exclude_edges=1, min_jump=min_jump)
    mlp_transitions_fwd  = detect_phase_transitions(mlp_qk_diff_fwd, exclude_edges=1, min_jump=min_jump)

    attn_transitions_all = detect_phase_transitions(attn_qk_diff_all, exclude_edges=0, min_jump=min_jump)
    mlp_transitions_all  = detect_phase_transitions(mlp_qk_diff_all, exclude_edges=0, min_jump=min_jump)

    return {
        "half_life": float(half_life),
        "exclude_diag": bool(exclude_diag),
        "min_jump": float(min_jump),

        "attn_row_Q_all": attn_row_Q_all,
        "attn_row_K_all": attn_row_K_all,
        "attn_qk_diff_all": attn_qk_diff_all,
        "attn_transitions_all": attn_transitions_all,

        "attn_row_Q_fwd": attn_row_Q_fwd,
        "attn_row_K_fwd": attn_row_K_fwd,
        "attn_qk_diff_fwd": attn_qk_diff_fwd,
        "attn_transitions_fwd": attn_transitions_fwd,

        "attn_row_Q_fwd_w": attn_row_Q_fwd_w,
        "attn_row_K_fwd_w": attn_row_K_fwd_w,
        "attn_qk_diff_fwd_w": attn_qk_diff_fwd_w,

        "mlp_row_Q_all": mlp_row_Q_all,
        "mlp_row_K_all": mlp_row_K_all,
        "mlp_qk_diff_all": mlp_qk_diff_all,
        "mlp_transitions_all": mlp_transitions_all,

        "mlp_row_Q_fwd": mlp_row_Q_fwd,
        "mlp_row_K_fwd": mlp_row_K_fwd,
        "mlp_qk_diff_fwd": mlp_qk_diff_fwd,
        "mlp_transitions_fwd": mlp_transitions_fwd,

        "mlp_row_Q_fwd_w": mlp_row_Q_fwd_w,
        "mlp_row_K_fwd_w": mlp_row_K_fwd_w,
        "mlp_qk_diff_fwd_w": mlp_qk_diff_fwd_w,

        "attn_mlp_write_overlap": matrices.attn_mlp_write_overlap,
    }

# =============================================================================
# Plotting
# =============================================================================

def analyze_spectral_profile(layer, shapes: ModelShapes, outdir: Path, layer_idx: int):
    """
    Spectral profiling (expensive). Intended for a FEW layers, not all layers.
    """
    attn = layer.self_attn
    mlp = layer.mlp

    W_Q = attn.q_proj.weight.detach()
    W_K = attn.k_proj.weight.detach()
    W_V = attn.v_proj.weight.detach()
    W_O = attn.o_proj.weight.detach()
    W_down = mlp.down_proj.weight.detach()
    W_up = mlp.up_proj.weight.detach()

    hd = shapes.head_dim
    M_OV = W_O[:, :hd] @ W_V[:hd, :]

    matrices = {
        "OV_head0": M_OV,
        "Q_proj": W_Q,
        "K_proj": W_K,
        "MLP_down": W_down,
        "MLP_up": W_up.T,
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    results = {}
    for idx, (name, M) in enumerate(matrices.items()):
        if idx >= len(axes):
            break

        # full SVD is still expensive; ok for a handful of layers
        S = torch.linalg.svdvals(M.float())
        S_np = S.cpu().numpy()

        var = S_np ** 2
        cum_var = np.cumsum(var) / max(var.sum(), 1e-12)

        ax = axes[idx]
        ax.plot(cum_var, linewidth=2)
        ax.axhline(0.9, linestyle="--", alpha=0.5)
        ax.axhline(0.95, linestyle="--", alpha=0.5)
        ax.set_xlabel("Components")
        ax.set_ylabel("Cumulative Variance")
        ax.set_title(f"{name}", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        for thresh in [0.5, 0.75, 0.9, 0.95, 0.99]:
            if np.any(cum_var >= thresh):
                k = int(np.argmax(cum_var >= thresh) + 1)
                results[f"{name}_k{int(thresh*100)}"] = k

    for idx in range(len(matrices), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"Spectral Analysis - Layer {layer_idx}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outdir / f"spectral_L{layer_idx:02d}.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results

def plot_connectome_overview(matrices: ConnectomeMatrices, K: int, outdir: Path, flavor_tag: str):
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))

    def qrange(M):
        lo = float(np.quantile(M, 0.05))
        hi = float(np.quantile(M, 0.95))
        return lo, hi

    vmin_q, vmax_q = qrange(matrices.attn_write_to_Q)
    vmin_k, vmax_k = qrange(matrices.attn_write_to_K)
    vmin_mq, vmax_mq = qrange(matrices.mlp_write_to_Q)
    vmin_mk, vmax_mk = qrange(matrices.mlp_write_to_K)
    vmin_am, vmax_am = qrange(matrices.attn_write_to_mlp_read)
    vmin_mm, vmax_mm = qrange(matrices.mlp_write_to_read)

    matrix_specs = [
        (matrices.attn_write_to_Q, "Attn Write → Attn Q", axes[0, 0], vmin_q, vmax_q, "viridis"),
        (matrices.attn_write_to_K, "Attn Write → Attn K", axes[0, 1], vmin_k, vmax_k, "viridis"),
        (matrices.attn_write_to_Q - matrices.attn_write_to_K, "Attn: Q-K Diff", axes[0, 2], -0.08, 0.08, "RdBu_r"),
        (matrices.mlp_write_to_Q, "MLP Write → Attn Q", axes[1, 0], vmin_mq, vmax_mq, "viridis"),
        (matrices.mlp_write_to_K, "MLP Write → Attn K", axes[1, 1], vmin_mk, vmax_mk, "viridis"),
        (matrices.mlp_write_to_Q - matrices.mlp_write_to_K, "MLP: Q-K Diff", axes[1, 2], -0.08, 0.08, "RdBu_r"),
        (matrices.attn_write_to_mlp_read, "Attn Write → MLP Read", axes[2, 0], vmin_am, vmax_am, "viridis"),
        (matrices.mlp_write_to_read, "MLP Write → MLP Read", axes[2, 1], vmin_mm, vmax_mm, "viridis"),
    ]

    for M, title, ax, vmin, vmax, cmap in matrix_specs:
        im = ax.imshow(M, vmin=vmin, vmax=vmax, aspect="equal", cmap=cmap)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Target Layer")
        ax.set_ylabel("Source Layer")
        plt.colorbar(im, ax=ax)

    ax = axes[2, 2]
    n_layers = len(matrices.attn_mlp_write_overlap)
    ax.bar(range(n_layers), matrices.attn_mlp_write_overlap, alpha=0.85)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Alignment")
    ax.set_title("Attn-MLP Write Overlap", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(0.5, float(matrices.attn_mlp_write_overlap.max()) * 1.15))
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Full Connectome Overview (K={K}) | {flavor_tag}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(outdir / f"connectome_overview_{flavor_tag}_K{K:03d}.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_phase_analysis_compare(phase_stats: dict, baseline: dict, K: int, outdir: Path, flavor_tag: str):
    attn_all   = to_float_array(phase_stats["attn_qk_diff_all"])
    attn_fwd   = to_float_array(phase_stats["attn_qk_diff_fwd"])
    attn_fwd_w = to_float_array(phase_stats["attn_qk_diff_fwd_w"])

    mlp_all   = to_float_array(phase_stats["mlp_qk_diff_all"])
    mlp_fwd   = to_float_array(phase_stats["mlp_qk_diff_fwd"])
    mlp_fwd_w = to_float_array(phase_stats["mlp_qk_diff_fwd_w"])

    layers = np.arange(len(attn_all))
    b = float(baseline["mean"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    def bar_panel(ax, data, title, transitions=None):
        ax.bar(layers, data, edgecolor="black", linewidth=0.3)
        ax.axhline(0, color="black", linewidth=1.5)
        if transitions:
            for t in transitions:
                ax.axvline(t - 0.5, color="purple", linewidth=2, linestyle="--", alpha=0.7)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Q - K preference")
        ax.grid(True, alpha=0.3, axis="y")

    bar_panel(axes[0, 0], attn_all, "Attention Δ (global all targets)", phase_stats["attn_transitions_all"])
    bar_panel(axes[0, 1], attn_fwd, "Attention Δ (forward-only j>i)", phase_stats["attn_transitions_fwd"])
    bar_panel(axes[0, 2], attn_fwd_w, f"Attention Δ (forward-weighted, half-life={phase_stats['half_life']:.1f})", None)

    bar_panel(axes[1, 0], mlp_all, "MLP→Attn Δ (global all targets)", phase_stats["mlp_transitions_all"])
    bar_panel(axes[1, 1], mlp_fwd, "MLP→Attn Δ (forward-only j>i)", phase_stats["mlp_transitions_fwd"])
    bar_panel(axes[1, 2], mlp_fwd_w, f"MLP→Attn Δ (forward-weighted, half-life={phase_stats['half_life']:.1f})", None)

    plt.suptitle(
        f"Phase Analysis Comparison (K={K}) | {flavor_tag} | Random baseline≈{b:.3f}",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(outdir / f"phase_analysis_compare_{flavor_tag}_K{K:03d}.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_cross_component_flow(matrices: ConnectomeMatrices, K: int, outdir: Path, flavor_tag: str):
    n_layers = matrices.attn_write_to_Q.shape[0]
    layers = np.arange(n_layers)

    all_mask, fwd_mask, _ = forward_masks(n_layers)

    def row_mean(M, mask):
        return masked_row_mean(M, mask)

    attn_to_mlp_all = row_mean(matrices.attn_write_to_mlp_read, all_mask)
    attn_to_mlp_fwd = row_mean(matrices.attn_write_to_mlp_read, fwd_mask)

    mlp_to_attn_Q_all = row_mean(matrices.mlp_write_to_Q, all_mask)
    mlp_to_attn_K_all = row_mean(matrices.mlp_write_to_K, all_mask)
    mlp_to_attn_all = 0.5 * (mlp_to_attn_Q_all + mlp_to_attn_K_all)

    mlp_to_attn_Q_fwd = row_mean(matrices.mlp_write_to_Q, fwd_mask)
    mlp_to_attn_K_fwd = row_mean(matrices.mlp_write_to_K, fwd_mask)
    mlp_to_attn_fwd = 0.5 * (mlp_to_attn_Q_fwd + mlp_to_attn_K_fwd)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    ax.plot(layers, attn_to_mlp_all, "--", linewidth=2, label="Attn→MLP (global)")
    ax.plot(layers, attn_to_mlp_fwd, "-", linewidth=2.5, label="Attn→MLP (forward-only)")
    ax.plot(layers, mlp_to_attn_all, "--", linewidth=2, label="MLP→Attn (global)")
    ax.plot(layers, mlp_to_attn_fwd, "-", linewidth=2.5, label="MLP→Attn (forward-only)")
    ax.set_xlabel("Source Layer")
    ax.set_ylabel("Mean Alignment")
    ax.set_title("Cross-Component Flow Strength (global vs forward)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    overlap = matrices.attn_mlp_write_overlap
    ax.bar(layers, overlap, alpha=0.85, edgecolor="black", linewidth=0.3)
    ax.axhline(np.nanmean(overlap), linestyle="--", linewidth=2, label=f"Mean: {np.nanmean(overlap):.3f}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Write Subspace Overlap")
    ax.set_title("Attn vs MLP: Write Overlap", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    im = ax.imshow(matrices.attn_write_to_mlp_read, aspect="equal", cmap="viridis")
    ax.set_xlabel("MLP Read Layer")
    ax.set_ylabel("Attn Write Layer")
    ax.set_title("Attention → MLP (alignment matrix)", fontweight="bold")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.imshow(matrices.mlp_write_to_read, aspect="equal", cmap="viridis")
    ax.set_xlabel("MLP Read Layer")
    ax.set_ylabel("MLP Write Layer")
    ax.set_title("MLP → MLP (alignment matrix)", fontweight="bold")
    plt.colorbar(im, ax=ax)

    plt.suptitle(f"Cross-Component Analysis (K={K}) | {flavor_tag}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(outdir / f"cross_component_{flavor_tag}_K{K:03d}.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_neuron_analysis(neuron_stats: List[dict], outdir: Path):
    n_layers = len(neuron_stats)
    layers = np.arange(n_layers)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    kv_means = [s["kv_alignment_mean"] for s in neuron_stats]
    kv_stds  = [s["kv_alignment_std"] for s in neuron_stats]
    ax.errorbar(layers, kv_means, yerr=kv_stds, fmt="o-", capsize=3, capthick=1)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Key-Value Cosine Similarity")
    ax.set_title("MLP Neuron: Key-Value Alignment", fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    pos_fracs = [s["kv_alignment_positive_frac"] for s in neuron_stats]
    ax.bar(layers, pos_fracs, alpha=0.85)
    ax.axhline(0.5, linestyle="--", label="0.5 baseline")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction")
    ax.set_title("Fraction of Neurons with Positive K-V Alignment", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    gate_key = [s["gate_key_alignment_mean"] for s in neuron_stats]
    ax.plot(layers, gate_key, "o-", linewidth=2)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Gate-Key Cosine Similarity")
    ax.set_title("Gate vs Up Projection Alignment", fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    key_norms = [s["key_norm_mean"] for s in neuron_stats]
    val_norms = [s["val_norm_mean"] for s in neuron_stats]
    ax.plot(layers, key_norms, "o-", markersize=4, label="Key (up_proj) norm")
    ax.plot(layers, val_norms, "s-", markersize=4, label="Value (down_proj) norm")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Norm")
    ax.set_title("Neuron Weight Norms", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("MLP Neuron Analysis", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(outdir / "neuron_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_phase_stability(all_results: dict, outdir: Path, flavor_tag: str):
    K_values = sorted(all_results.keys())
    if len(K_values) < 2:
        return

    n_layers = len(to_float_array(all_results[K_values[0]]["phase_stats"]["attn_qk_diff_all"]))
    layers = np.arange(n_layers)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    for K in K_values:
        qk_diff = to_float_array(all_results[K]["phase_stats"]["attn_qk_diff_fwd"])
        ax.plot(layers, qk_diff, "o-", markersize=3, linewidth=1.5, label=f"K={K}", alpha=0.85)
    ax.axhline(0, linewidth=1.5, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Q - K preference (forward)")
    ax.set_title("Attention forward Δ vs K", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for K in K_values:
        qk_diff = to_float_array(all_results[K]["phase_stats"]["mlp_qk_diff_fwd"])
        ax.plot(layers, qk_diff, "o-", markersize=3, linewidth=1.5, label=f"K={K}", alpha=0.85)
    ax.axhline(0, linewidth=1.5, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Q - K preference (forward)")
    ax.set_title("MLP forward Δ vs K", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for K in K_values:
        attn_trans = all_results[K]["phase_stats"]["attn_transitions_fwd"]
        mlp_trans = all_results[K]["phase_stats"]["mlp_transitions_fwd"]
        for t in attn_trans:
            ax.scatter(K, t, s=100, marker="v", alpha=0.85)
        for t in mlp_trans:
            ax.scatter(K, t, s=100, marker="^", alpha=0.85)

    ax.scatter([], [], s=100, marker="v", label="Attn transition (fwd)")
    ax.scatter([], [], s=100, marker="^", label="MLP transition (fwd)")
    ax.set_xlabel("K")
    ax.set_ylabel("Transition Layer")
    ax.set_title("Phase Transition Stability (forward)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, n_layers)

    ax = axes[1, 1]
    corrs = [all_results[K]["phase_analysis"]["attn_mlp_corr_fwd"] for K in K_values]
    ax.bar(range(len(K_values)), corrs, alpha=0.85)
    ax.set_xticks(range(len(K_values)))
    ax.set_xticklabels([str(K) for K in K_values])
    ax.set_xlabel("K")
    ax.set_ylabel("Correlation")
    ax.set_title("Attn–MLP forward Δ correlation vs K", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Phase Stability Analysis (forward) | {flavor_tag}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(outdir / f"phase_stability_{flavor_tag}.png", dpi=150, bbox_inches="tight")
    plt.close()

# =============================================================================
# Phase analysis metrics bundle
# =============================================================================

def phase_analysis_metrics(phase_stats: dict) -> dict:
    attn_all = to_float_array(phase_stats["attn_qk_diff_all"])
    mlp_all  = to_float_array(phase_stats["mlp_qk_diff_all"])
    attn_fwd = to_float_array(phase_stats["attn_qk_diff_fwd"])
    mlp_fwd  = to_float_array(phase_stats["mlp_qk_diff_fwd"])
    return {
        "attn_mlp_corr_all": nan_corr(attn_all, mlp_all),
        "attn_mlp_corr_fwd": nan_corr(attn_fwd, mlp_fwd),
        "attn_std_fwd": float(np.nanstd(attn_fwd)),
        "mlp_std_fwd": float(np.nanstd(mlp_fwd)),
    }

# =============================================================================
# Main Run
# =============================================================================

def _parse_spectral_layers(spec: str, n_layers: int) -> List[int]:
    """
    spec: "0,mid,last" or "0,17,33"
    """
    spec = spec.strip()
    if not spec:
        return []
    out = []
    for tok in spec.split(","):
        t = tok.strip().lower()
        if not t:
            continue
        if t == "mid":
            out.append(n_layers // 2)
        elif t == "last":
            out.append(n_layers - 1)
        else:
            out.append(int(t))
    # unique, in-range
    out = sorted(set([i for i in out if 0 <= i < n_layers]))
    return out

def run(
    model_id: str,
    outdir: Path,
    K_list: List[int],
    half_life: float,
    baseline_trials: int,
    min_jump: float,
    do_spectral: bool,
    spectral_layers_spec: str,
):
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Loading: {model_id}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map=None,
    )
    model.eval()

    layers = get_layers(model)
    shapes = infer_shapes(model, layers)
    n_layers = shapes.n_layers

    print(f"Architecture: n_layers={n_layers}, d_model={shapes.d_model}, d_ff={shapes.d_ff}")
    print(f"  head_dim={shapes.head_dim}, q_heads={shapes.num_q_heads}, kv_heads={shapes.num_kv_heads}")

    meta = {
        "model_id": model_id,
        "n_layers": n_layers,
        "d_model": shapes.d_model,
        "d_ff": shapes.d_ff,
        "head_dim": shapes.head_dim,
        "num_q_heads": shapes.num_q_heads,
        "num_kv_heads": shapes.num_kv_heads,
        "K_list": K_list,
        "half_life": half_life,
        "baseline_trials": baseline_trials,
        "min_jump": min_jump,
        "write_gamma_flavors": ["none", "pre_attn"],
        "do_spectral": bool(do_spectral),
        "spectral_layers_spec": spectral_layers_spec,
    }
    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Optional spectral profiling
    if do_spectral:
        spectral_dir = outdir / "spectral"
        spectral_dir.mkdir(parents=True, exist_ok=True)
        spectral_layers = _parse_spectral_layers(spectral_layers_spec, n_layers)
        print("\n=== Spectral Profiles (selected layers) ===")
        spectral_stats = []
        for idx in spectral_layers:
            stats = analyze_spectral_profile(layers[idx], shapes, spectral_dir, idx)
            spectral_stats.append({"layer": idx, **stats})
            print(f"  L{idx}: spectral profile saved")
        with open(outdir / "spectral_stats.json", "w") as f:
            json.dump(spectral_stats, f, indent=2)
    else:
        print("\n=== Spectral Profiles ===")
        print("  skipped (use --do_spectral 1 to enable)")

    # Neuron analysis (shared across flavors)
    print("\n=== Neuron Analysis ===")
    neuron_stats = []
    with torch.inference_mode():
        for l in range(n_layers):
            stats = analyze_mlp_neurons(layers[l])
            neuron_stats.append(stats)
            if l % 10 == 0:
                print(f"  L{l}: KV align mean={stats['kv_alignment_mean']:.4f}")
    with open(outdir / "neuron_stats.json", "w") as f:
        json.dump(neuron_stats, f, indent=2)
    plot_neuron_analysis(neuron_stats, outdir)

    # Two built-in write-gamma flavors
    write_gamma_flavors = [("none", "writeGammaNone"), ("pre_attn", "writeGammaPreAttn")]

    for write_mode, flavor_tag in write_gamma_flavors:
        print(f"\n==============================")
        print(f"WRITE GAMMA FLAVOR: {write_mode}")
        print(f"Output tag: {flavor_tag}")
        print(f"==============================")

        flavor_dir = outdir / flavor_tag
        flavor_dir.mkdir(parents=True, exist_ok=True)

        all_results: Dict[int, Dict[str, Any]] = {}

        for K in K_list:
            print(f"\n=== Computing K={K} ({flavor_tag}) ===")
            Kdir = flavor_dir / f"K{K:03d}"
            Kdir.mkdir(parents=True, exist_ok=True)

            print("  Random baseline...")
            baseline = random_subspace_baseline(shapes.d_model, K, trials=baseline_trials, seed=0, device="cpu")
            with open(Kdir / "random_baseline.json", "w") as f:
                json.dump(baseline, f, indent=2)

            print("  Building layer bases...")
            all_bases = []
            with torch.inference_mode():
                for l in range(n_layers):
                    bases = compute_layer_bases(layers[l], shapes, K, write_gamma_mode=write_mode)
                    all_bases.append(bases)
                    if l % 10 == 0:
                        print(f"    Layer {l}/{n_layers}")

            print("  Computing connectome matrices...")
            matrices = compute_connectome(all_bases, n_layers)

            np.save(Kdir / "attn_write_to_Q.npy", matrices.attn_write_to_Q)
            np.save(Kdir / "attn_write_to_K.npy", matrices.attn_write_to_K)
            np.save(Kdir / "mlp_write_to_Q.npy", matrices.mlp_write_to_Q)
            np.save(Kdir / "mlp_write_to_K.npy", matrices.mlp_write_to_K)
            np.save(Kdir / "attn_write_to_mlp_read.npy", matrices.attn_write_to_mlp_read)
            np.save(Kdir / "mlp_write_to_read.npy", matrices.mlp_write_to_read)
            np.save(Kdir / "attn_mlp_write_overlap.npy", matrices.attn_mlp_write_overlap)

            print("  Analyzing phases (global + forward)...")
            phase_stats = compute_phase_stats(matrices, n_layers, half_life=half_life, min_jump=min_jump)
            phase_analysis = phase_analysis_metrics(phase_stats)

            print("  Generating plots...")
            plot_connectome_overview(matrices, K, Kdir, flavor_tag=flavor_tag)
            plot_phase_analysis_compare(phase_stats, baseline, K, Kdir, flavor_tag=flavor_tag)
            plot_cross_component_flow(matrices, K, Kdir, flavor_tag=flavor_tag)

            summary = {
                "K": K,
                "write_gamma_mode": write_mode,
                "random_baseline": baseline,
                "phase_stats": sanitize_for_json(phase_stats),
                "phase_analysis": sanitize_for_json(phase_analysis),
            }
            with open(Kdir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            all_results[K] = {
                "K": K,
                "random_baseline": baseline,
                "phase_stats": sanitize_for_json(phase_stats),
                "phase_analysis": sanitize_for_json(phase_analysis),
            }
            print(f"  Saved to {Kdir}")

        if len(K_list) >= 2:
            print("\n=== Phase Stability Analysis ===")
            plot_phase_stability(all_results, flavor_dir, flavor_tag=flavor_tag)

        print("\n=== Summary (forward transitions) ===")
        for K in K_list:
            stats = all_results[K]
            attn_t = stats["phase_stats"]["attn_transitions_fwd"]
            mlp_t = stats["phase_stats"]["mlp_transitions_fwd"]
            corr_fwd = stats["phase_analysis"]["attn_mlp_corr_fwd"]
            print(f"  K={K:3d}: Attn fwd transitions={attn_t}, MLP fwd transitions={mlp_t}, corr_fwd={corr_fwd:.3f}")

    print(f"\nAll outputs saved to: {outdir.resolve()}")

# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Gemma3 Weight Connectome Analysis (v2)")
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it", help="HuggingFace model ID")
    parser.add_argument("--outdir", type=str, default="connectome_output", help="Output directory")
    parser.add_argument("--K", type=str, default="32,64,128,200", help="Comma-separated list of K values")

    parser.add_argument("--half_life", type=float, default=4.0,
                        help="Forward distance-weighting half-life (layers)")
    parser.add_argument("--baseline_trials", type=int, default=200,
                        help="Trials for random-subspace baseline per K")
    parser.add_argument("--min_jump", type=float, default=0.002,
                        help="Minimum sharpness (jump magnitude) to keep a phase transition")

    # Spectral profiling is expensive -> OFF by default
    parser.add_argument("--do_spectral", type=int, default=1, help="Enable spectral profiling (0/1)")
    parser.add_argument("--spectral_layers", type=str, default="0,mid,last",
                        help='Which layers to profile if do_spectral=1. Example: "0,mid,last" or "0,17,33".')

    args = parser.parse_args()
    K_list = [int(x.strip()) for x in args.K.split(",") if x.strip()]
    run(
        model_id=args.model,
        outdir=Path(args.outdir),
        K_list=K_list,
        half_life=args.half_life,
        baseline_trials=args.baseline_trials,
        min_jump=args.min_jump,
        do_spectral=bool(args.do_spectral),
        spectral_layers_spec=args.spectral_layers,
    )
