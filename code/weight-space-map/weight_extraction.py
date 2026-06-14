"""
Weight extraction utilities for Gemma-2.

Handles:
- Loading W_Q, W_K, W_V, W_O with correct shapes per head
- Folding RMSNorm gamma into effective weights
- GQA-aware head-to-KV-group mapping
- Correct OV composition: W_OV = W_O @ W_V (not V @ O!)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from config import Gemma2Config, GEMMA2_CONFIG


@dataclass
class LayerWeights:
    """
    Extracted and processed weights for a single attention layer.
    
    All weights have RMSNorm gamma folded in if configured.
    """
    layer_idx: int
    
    # Per query-head weights: [n_q_heads, head_dim, hidden_size]
    W_Q: torch.Tensor
    
    # Per KV-head weights: [n_kv_heads, head_dim, hidden_size]
    W_K: torch.Tensor
    W_V: torch.Tensor
    
    # Output projection per query-head: [n_q_heads, hidden_size, head_dim]
    W_O: torch.Tensor
    
    # RMSNorm weights (for reference)
    input_layernorm_weight: torch.Tensor
    post_attention_layernorm_weight: torch.Tensor
    
    # Whether gamma was folded
    gamma_folded: bool


def extract_layer_weights(
    model: nn.Module,
    layer_idx: int,
    config: Gemma2Config = GEMMA2_CONFIG,
    fold_gamma: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    fold_mode: str = "one_plus_gamma",
) -> LayerWeights:
    """
    Extract attention weights from a layer with proper shaping.

    Args:
        model: HuggingFace Gemma2ForCausalLM model
        layer_idx: Layer index to extract
        config: Model configuration
        fold_gamma: Whether to fold RMSNorm (1 + gamma) into Q/K/V weights
        device: Target device
        dtype: Target dtype
        fold_mode: "one_plus_gamma" (correct, matches Gemma2RMSNorm) or
            "legacy_gamma" (pre-fix W * gamma, for errata comparisons only)

    Returns:
        LayerWeights dataclass with all processed weights
    """
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    
    n_q = config.num_attention_heads
    n_kv = config.num_key_value_heads
    d_h = config.head_dim
    d_model = config.hidden_size
    
    # Get raw weights
    # Q: [n_q * d_h, d_model] -> [n_q, d_h, d_model]
    W_Q_raw = attn.q_proj.weight.detach().to(device=device, dtype=dtype)
    W_Q = W_Q_raw.view(n_q, d_h, d_model)
    
    # K: [n_kv * d_h, d_model] -> [n_kv, d_h, d_model]
    W_K_raw = attn.k_proj.weight.detach().to(device=device, dtype=dtype)
    W_K = W_K_raw.view(n_kv, d_h, d_model)
    
    # V: [n_kv * d_h, d_model] -> [n_kv, d_h, d_model]
    W_V_raw = attn.v_proj.weight.detach().to(device=device, dtype=dtype)
    W_V = W_V_raw.view(n_kv, d_h, d_model)
    
    # O: [d_model, n_q * d_h] -> [n_q, d_model, d_h]
    W_O_raw = attn.o_proj.weight.detach().to(device=device, dtype=dtype)
    W_O = W_O_raw.view(d_model, n_q, d_h).permute(1, 0, 2)  # [n_q, d_model, d_h]
    
    # RMSNorm weights
    gamma_in = layer.input_layernorm.weight.detach().to(device=device, dtype=dtype)
    gamma_post = layer.post_attention_layernorm.weight.detach().to(device=device, dtype=dtype)
    
    # Fold RMSNorm scale into Q/K/V if requested
    # HF Gemma2RMSNorm: y = (x / rms(x)) * (1 + gamma)   <-- note the +1!
    # (modeling_gemma2.py: output * (1.0 + self.weight))
    # Effective weight = W @ diag(1 + gamma) (broadcast over last dim)
    if fold_gamma:
        if fold_mode == "one_plus_gamma":
            gamma_scale = (1.0 + gamma_in).view(1, 1, d_model)
        elif fold_mode == "legacy_gamma":
            # Pre-fix behavior (missing the +1); kept ONLY so archived
            # results can be reproduced for before/after errata comparisons.
            gamma_scale = gamma_in.view(1, 1, d_model)
        else:
            raise ValueError(f"Unknown fold_mode: {fold_mode}")
        W_Q = W_Q * gamma_scale
        W_K = W_K * gamma_scale
        W_V = W_V * gamma_scale
    
    return LayerWeights(
        layer_idx=layer_idx,
        W_Q=W_Q,
        W_K=W_K,
        W_V=W_V,
        W_O=W_O,
        input_layernorm_weight=gamma_in,
        post_attention_layernorm_weight=gamma_post,
        gamma_folded=fold_gamma,
    )


def get_qkv_for_head(
    weights: LayerWeights,
    query_head: int,
    config: Gemma2Config = GEMMA2_CONFIG,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get Q, K, V weight matrices for a specific query head.
    
    Handles GQA: maps query head to its KV group.
    
    Args:
        weights: Extracted layer weights
        query_head: Query head index
        config: Model configuration
        
    Returns:
        (W_Q, W_K, W_V) each of shape [head_dim, hidden_size]
    """
    kv_group = config.query_to_kv_group(query_head)
    
    W_Q = weights.W_Q[query_head]  # [head_dim, hidden_size]
    W_K = weights.W_K[kv_group]    # [head_dim, hidden_size]
    W_V = weights.W_V[kv_group]    # [head_dim, hidden_size]
    
    return W_Q, W_K, W_V


def compute_ov_matrix(
    weights: LayerWeights,
    query_head: int,
    config: Gemma2Config = GEMMA2_CONFIG,
) -> torch.Tensor:
    """
    Compute the effective OV matrix for a head.
    
    IMPORTANT: Correct ordering is W_OV = W_O @ W_V
    
    The attention computation is:
        x_s --W_V--> v_s --(attn mix)--> v_mix --W_O--> out
    
    So for column vectors: out = W_O @ (W_V @ x)
    Thus W_OV = W_O @ W_V
    
    Args:
        weights: Extracted layer weights
        query_head: Query head index
        config: Model configuration
        
    Returns:
        W_OV of shape [hidden_size, hidden_size]
        (Maps input residual to output residual contribution)
    """
    kv_group = config.query_to_kv_group(query_head)
    
    # W_O: [hidden_size, head_dim]
    # W_V: [head_dim, hidden_size]
    W_O = weights.W_O[query_head]  # [hidden_size, head_dim]
    W_V = weights.W_V[kv_group]    # [head_dim, hidden_size]
    
    # W_OV = W_O @ W_V: [hidden_size, hidden_size]
    W_OV = W_O @ W_V
    
    return W_OV


def compute_all_ov_matrices(
    weights: LayerWeights,
    config: Gemma2Config = GEMMA2_CONFIG,
) -> torch.Tensor:
    """
    Compute OV matrices for all query heads in the layer.
    
    Returns: [n_q_heads, hidden_size, hidden_size]
    """
    n_q = config.num_attention_heads
    W_OVs = []
    
    for qh in range(n_q):
        W_OV = compute_ov_matrix(weights, qh, config)
        W_OVs.append(W_OV)
    
    return torch.stack(W_OVs, dim=0)


class WeightExtractor:
    """
    Utility class for extracting weights from a loaded model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Gemma2Config = GEMMA2_CONFIG,
        fold_gamma: bool = True,
        fold_mode: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.fold_gamma = fold_gamma
        # Default fold_mode from config (one_plus_gamma unless overridden)
        self.fold_mode = fold_mode or getattr(config, "gamma_fold_mode", "one_plus_gamma")
        self._cache: Dict[Tuple[int, str, str, bool, str], LayerWeights] = {}

    def get_layer(
        self,
        layer_idx: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> LayerWeights:
        """Get weights for a layer (cached by layer_idx, device, dtype, fold_gamma, fold_mode)."""
        # Use full cache key to avoid returning wrong device/dtype
        cache_key = (layer_idx, device, str(dtype), self.fold_gamma, self.fold_mode)

        if cache_key not in self._cache:
            self._cache[cache_key] = extract_layer_weights(
                self.model, layer_idx, self.config,
                fold_gamma=self.fold_gamma,
                device=device,
                dtype=dtype,
                fold_mode=self.fold_mode,
            )

        return self._cache[cache_key]
    
    def clear_cache(self):
        """Clear the weight cache."""
        self._cache.clear()
    
    def get_qkv(
        self,
        layer_idx: int,
        query_head: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience: get Q/K/V for a specific head in a layer."""
        weights = self.get_layer(layer_idx, device, dtype)
        return get_qkv_for_head(weights, query_head, self.config)
    
    def get_ov(
        self,
        layer_idx: int,
        query_head: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Convenience: get OV matrix for a specific head in a layer."""
        weights = self.get_layer(layer_idx, device, dtype)
        return compute_ov_matrix(weights, query_head, self.config)


if __name__ == "__main__":
    # Quick shape verification (requires model loaded)
    print("Weight extraction module loaded.")
    print(f"Expected shapes for Gemma-2-2B:")
    print(f"  W_Q: [{GEMMA2_CONFIG.num_attention_heads}, {GEMMA2_CONFIG.head_dim}, {GEMMA2_CONFIG.hidden_size}]")
    print(f"  W_K: [{GEMMA2_CONFIG.num_key_value_heads}, {GEMMA2_CONFIG.head_dim}, {GEMMA2_CONFIG.hidden_size}]")
    print(f"  W_V: [{GEMMA2_CONFIG.num_key_value_heads}, {GEMMA2_CONFIG.head_dim}, {GEMMA2_CONFIG.hidden_size}]")
    print(f"  W_O: [{GEMMA2_CONFIG.num_attention_heads}, {GEMMA2_CONFIG.hidden_size}, {GEMMA2_CONFIG.head_dim}]")
    print(f"  W_OV: [{GEMMA2_CONFIG.hidden_size}, {GEMMA2_CONFIG.hidden_size}]")
