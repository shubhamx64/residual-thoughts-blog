"""
Phase 6: Activation Validation for Gemma-2 Weight-Space Analysis.

Validates that weight-space predictions (B matrix, W2F matrix) match 
runtime behavior when real tokens flow through the model.

Key validations:
1A. QK Routing: Does B predict actual attention logits?
2B. OV Writing: Does W2F predict which features change?
"""

import torch
import torch.nn.functional as F
import json
import csv
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from scipy import stats
from collections import defaultdict
import argparse

from config import AnalysisConfig


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ValidationConfig:
    """Configuration for activation validation."""
    # Target heads for validation
    routing_heads: List[Tuple[int, int]] = field(default_factory=lambda: [(6, 3), (15, 0)])
    writing_heads: List[Tuple[int, int]] = field(default_factory=lambda: [(6, 0), (15, 0)])
    
    # Sampling for memory efficiency
    max_keys_per_query: int = 256  # Sample this many keys per query position
    
    # Distance bins for correlation analysis (narrower for better RoPE accuracy, up to 1024)
    distance_bins: List[int] = field(default_factory=lambda: [0, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    
    # Top-K for ranking metrics
    top_k_features: int = 20
    
    # Whether to apply softcap to both predicted and actual
    apply_softcap: bool = True
    softcap_value: float = 50.0


# =============================================================================
# Model Hooks for Extracting Activations
# =============================================================================

class ActivationCache:
    """Stores activations extracted via hooks."""
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.residual_pre_attn: Dict[int, torch.Tensor] = {}  # layer -> [batch, seq, d_model]
        self.residual_post_attn: Dict[int, torch.Tensor] = {}  # layer -> [batch, seq, d_model] (after attn, before MLP)
        self.attention_logits: Dict[int, torch.Tensor] = {}    # layer -> [batch, n_heads, seq, seq]
        self.attention_weights: Dict[int, torch.Tensor] = {}   # layer -> [batch, n_heads, seq, seq]
        self.attention_output: Dict[int, torch.Tensor] = {}    # layer -> [batch, seq, d_model] (raw attn output)
        self.head_outputs: Dict[Tuple[int, int], torch.Tensor] = {}  # (layer, head) -> [batch, seq, d_model]
        self.residual_post_head: Dict[Tuple[int, int], torch.Tensor] = {}


def create_attention_hooks(model, cache: ActivationCache, target_layers: List[int]):
    """
    Create hooks to extract attention internals from Gemma-2.
    
    Args:
        model: The Gemma model
        cache: ActivationCache to store results
        target_layers: Which layers to hook
    
    Returns:
        List of hook handles (call .remove() to cleanup)
    """
    handles = []
    
    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]
        
        # Hook 1: Capture pre-attention residual
        def make_pre_attn_hook(layer_idx):
            def hook(module, args, kwargs):
                # Input to self_attn is the hidden states
                if args:
                    hidden = args[0]
                elif 'hidden_states' in kwargs:
                    hidden = kwargs['hidden_states']
                else:
                    return
                cache.residual_pre_attn[layer_idx] = hidden.detach()
            return hook
        
        h = layer.self_attn.register_forward_pre_hook(make_pre_attn_hook(layer_idx), with_kwargs=True)
        handles.append(h)
        
        # Hook 2: Capture attention weights (after softmax)
        # Gemma-2 attention module stores attn_weights if output_attentions=True
        # We'll modify forward to capture this
        def make_attn_output_hook(layer_idx):
            def hook(module, args, kwargs, output):
                # output is (attn_output, attn_weights, past_key_value) when output_attentions=True
                if isinstance(output, tuple) and len(output) >= 1:
                    # Capture the actual attention output vector (before residual add)
                    attn_output = output[0]
                    if attn_output is not None:
                        cache.attention_output[layer_idx] = attn_output.detach()
                    
                    # Capture attention weights if available
                    if len(output) >= 2:
                        attn_weights = output[1]
                        if attn_weights is not None:
                            cache.attention_weights[layer_idx] = attn_weights.detach()
            return hook
        
        h = layer.self_attn.register_forward_hook(make_attn_output_hook(layer_idx), with_kwargs=True)
        handles.append(h)
        
        # Hook 3: Capture full layer output (including attention output added to residual)
        # This captures the output of the entire decoder layer
        def make_layer_output_hook(layer_idx):
            def hook(module, args, output):
                # output[0] is the hidden states after the full layer (attn + MLP + residuals)
                if isinstance(output, tuple):
                    cache.residual_post_layer = getattr(cache, 'residual_post_layer', {})
                    cache.residual_post_layer[layer_idx] = output[0].detach()
            return hook
        
        h = layer.register_forward_hook(make_layer_output_hook(layer_idx))
        handles.append(h)
    
    return handles


# =============================================================================
# QK Routing Validation (Part 1A)
# =============================================================================

@dataclass
class RoutingValidationResult:
    """Results from QK routing validation."""
    layer_idx: int
    head_idx: int
    
    # Overall correlation
    pearson_r: float
    spearman_r: float
    
    # Correlation by distance bin
    pearson_by_distance: Dict[str, float] = field(default_factory=dict)
    spearman_by_distance: Dict[str, float] = field(default_factory=dict)
    n_pairs_by_distance: Dict[str, int] = field(default_factory=dict)
    
    # Stats
    n_query_positions: int = 0
    n_total_pairs: int = 0
    
    # === NEW DIAGNOSTIC FIELDS ===
    
    # Fraction of rows/bins skipped due to constant vectors
    n_rows_skipped: int = 0
    n_rows_total: int = 0
    skipped_fraction: float = 0.0
    bins_with_no_data: List[str] = field(default_factory=list)
    
    # Per-bin statistics: mean/std of log(attn) and predicted
    actual_mean_by_bin: Dict[str, float] = field(default_factory=dict)
    actual_std_by_bin: Dict[str, float] = field(default_factory=dict)
    predicted_mean_by_bin: Dict[str, float] = field(default_factory=dict)
    predicted_std_by_bin: Dict[str, float] = field(default_factory=dict)
    
    # Distance-specific Spearman correlations (named ranges)
    spearman_local: float = 0.0      # 0-4
    spearman_mid: float = 0.0        # 16-32
    spearman_long: float = 0.0       # 128-256
    spearman_256plus: float = 0.0    # 256+ (aggregates 256-512 and 512-1024)
    
    # Sign stability: fraction of bins where Spearman has same sign as overall
    sign_stability: float = 0.0
    sign_by_bin: Dict[str, int] = field(default_factory=dict)  # +1, -1, or 0


def validate_routing_correlation(
    sae_activations: torch.Tensor,  # [seq_len, n_features]
    B_matrix: torch.Tensor,          # [n_features, n_features]
    actual_logits: torch.Tensor,     # [seq_len, seq_len] for this head
    config: ValidationConfig,
    analysis_config: AnalysisConfig,
    layer_idx: int,
) -> RoutingValidationResult:
    """
    Validate B matrix against actual attention logits.
    
    Predicted: logits_pred[t,s] = a(t) @ B @ a(s).T
    Actual: logits from model
    
    Args:
        sae_activations: SAE activations for each position [seq_len, n_features]
        B_matrix: Affinity matrix from weight-space analysis [n_features, n_features]
        actual_logits: Actual attention logits [seq_len, seq_len]
        config: Validation configuration
        analysis_config: Model configuration for masks/scaling
    
    Returns:
        RoutingValidationResult
    """
    seq_len = sae_activations.shape[0]
    device = sae_activations.device
    
    # Import GEMMA2_CONFIG for sliding window check (method is on Gemma2Config)
    from config import GEMMA2_CONFIG
    
    # Determine valid attention mask (causal + sliding window if applicable)
    is_sliding = GEMMA2_CONFIG.is_sliding_window_layer(layer_idx)
    window_size = GEMMA2_CONFIG.sliding_window_size if is_sliding else seq_len
    
    # For each query position, sample keys and compute correlation
    all_predicted = []
    all_actual = []
    pairs_by_distance = defaultdict(lambda: {'pred': [], 'actual': []})
    
    for t in range(seq_len):
        # Valid key positions (causal + window)
        if is_sliding:
            valid_start = max(0, t - window_size + 1)
        else:
            valid_start = 0
        valid_end = t + 1  # Causal: can only attend to positions <= t
        
        valid_keys = list(range(valid_start, valid_end))
        
        # Sample if too many
        if len(valid_keys) > config.max_keys_per_query:
            sampled_indices = np.random.choice(len(valid_keys), config.max_keys_per_query, replace=False)
            valid_keys = [valid_keys[i] for i in sorted(sampled_indices)]
        
        if not valid_keys:
            continue
        
        # Get query activation
        a_query = sae_activations[t]  # [n_features]
        
        # Get key activations
        key_indices = torch.tensor(valid_keys, device=device)
        a_keys = sae_activations[key_indices]  # [n_keys, n_features]
        
        # Predicted logits: a_query @ B @ a_keys.T
        predicted = a_query @ B_matrix @ a_keys.T  # [n_keys]
        
        # Actual logits: use log(attention weights) instead of raw weights
        # log(softmax) = logits + const per row, so correlation is meaningful
        # Raw weights are post-softmax which distorts cross-row linear relationships
        actual = torch.log(actual_logits[t, key_indices].clamp_min(1e-9))  # [n_keys]
        
        # Collect for overall correlation (convert to float32 for numpy compatibility)
        all_predicted.append(predicted.cpu().float().numpy())
        all_actual.append(actual.cpu().float().numpy())
        
        # Bin by distance
        for i, s in enumerate(valid_keys):
            distance = t - s
            for bin_idx, (bin_start, bin_end) in enumerate(zip(config.distance_bins[:-1], config.distance_bins[1:])):
                if bin_start <= distance < bin_end:
                    bin_name = f"{bin_start}-{bin_end}"
                    pairs_by_distance[bin_name]['pred'].append(predicted[i].item())
                    pairs_by_distance[bin_name]['actual'].append(actual[i].item())
                    break
    
    # Compute overall correlation
    all_pred = np.concatenate(all_predicted)
    all_act = np.concatenate(all_actual)
    
    pearson_r, _ = stats.pearsonr(all_pred, all_act) if len(all_pred) > 2 else (0.0, 1.0)
    spearman_r, _ = stats.spearmanr(all_pred, all_act) if len(all_pred) > 2 else (0.0, 1.0)
    
    # Compute correlation by distance bin with mean/std statistics
    pearson_by_dist = {}
    spearman_by_dist = {}
    n_pairs_by_dist = {}
    predicted_mean_by_bin = {}
    predicted_std_by_bin = {}
    actual_mean_by_bin = {}
    actual_std_by_bin = {}
    
    for bin_name, data in pairs_by_distance.items():
        if len(data['pred']) > 2:
            pred_arr = np.array(data['pred'])
            actual_arr = np.array(data['actual'])
            
            r_p, _ = stats.pearsonr(pred_arr, actual_arr)
            r_s, _ = stats.spearmanr(pred_arr, actual_arr)
            pearson_by_dist[bin_name] = r_p
            spearman_by_dist[bin_name] = r_s
            n_pairs_by_dist[bin_name] = len(pred_arr)
            
            # Compute mean/std for this bin
            predicted_mean_by_bin[bin_name] = float(np.mean(pred_arr))
            predicted_std_by_bin[bin_name] = float(np.std(pred_arr))
            actual_mean_by_bin[bin_name] = float(np.mean(actual_arr))
            actual_std_by_bin[bin_name] = float(np.std(actual_arr))
    
    return RoutingValidationResult(
        layer_idx=layer_idx,
        head_idx=0,  # Will be set by caller
        pearson_r=pearson_r,
        spearman_r=spearman_r,
        pearson_by_distance=pearson_by_dist,
        spearman_by_distance=spearman_by_dist,
        n_pairs_by_distance=n_pairs_by_dist,
        n_query_positions=seq_len,
        n_total_pairs=len(all_pred),
        predicted_mean_by_bin=predicted_mean_by_bin,
        predicted_std_by_bin=predicted_std_by_bin,
        actual_mean_by_bin=actual_mean_by_bin,
        actual_std_by_bin=actual_std_by_bin,
    )


def validate_routing_rope_aware(
    sae_activations: torch.Tensor,  # [seq_len, n_features]
    B_by_distance: Dict[str, torch.Tensor],  # bin_name -> [n_features, n_features]
    actual_logits: torch.Tensor,     # [seq_len, seq_len] for this head
    config: ValidationConfig,
    layer_idx: int,
) -> RoutingValidationResult:
    """
    RoPE-aware validation using per-distance B matrices and per-query correlation.
    
    Key improvements over validate_routing_correlation:
    1. Uses B_Δ with RoPE rotation matched to each distance bin
    2. Computes correlation per-query-row, then averages (avoids cross-row softmax artifacts)
    3. Tracks comprehensive diagnostics: skipped rows, per-bin stats, sign stability
    
    Args:
        sae_activations: SAE activations for each position [seq_len, n_features]
        B_by_distance: Dict mapping bin_name to B_Δ matrix for that bin center
        actual_logits: Actual attention weights [seq_len, seq_len]
        config: Validation configuration
        layer_idx: Layer index for sliding window check
    
    Returns:
        RoutingValidationResult with full diagnostics
    """
    from config import GEMMA2_CONFIG
    
    seq_len = sae_activations.shape[0]
    device = sae_activations.device
    
    # Determine valid attention mask (causal + sliding window if applicable)
    is_sliding = GEMMA2_CONFIG.is_sliding_window_layer(layer_idx)
    window_size = GEMMA2_CONFIG.sliding_window_size if is_sliding else seq_len
    
    # Per-query correlations (not pooled globally)
    per_query_pearson = []
    per_query_spearman = []
    
    # Per-bin data for distance-specific correlations
    pairs_by_distance = defaultdict(lambda: {'pred': [], 'actual': []})
    n_total_pairs = 0
    
    # === NEW: Track skipped rows ===
    n_rows_total = 0
    n_rows_skipped = 0
    
    for t in range(seq_len):
        n_rows_total += 1
        
        # Valid key positions (causal + window)
        if is_sliding:
            valid_start = max(0, t - window_size + 1)
        else:
            valid_start = 0
        valid_end = t + 1
        
        valid_keys = list(range(valid_start, valid_end))
        
        # Sample if too many
        if len(valid_keys) > config.max_keys_per_query:
            sampled_indices = np.random.choice(len(valid_keys), config.max_keys_per_query, replace=False)
            valid_keys = [valid_keys[i] for i in sorted(sampled_indices)]
        
        if len(valid_keys) < 3:
            n_rows_skipped += 1
            continue
        
        a_query = sae_activations[t]  # [n_features]
        key_indices = torch.tensor(valid_keys, device=device)
        a_keys = sae_activations[key_indices]  # [n_keys, n_features]
        
        # Actual: use log(attention weights)
        actual_row = torch.log(actual_logits[t, key_indices].clamp_min(1e-9))
        
        # Vectorized computation: group keys by distance bin and compute all at once
        predicted_row = torch.zeros_like(actual_row)
        distances = t - key_indices  # [n_keys] tensor of distances
        
        # Process each bin's keys in one batched operation
        for b_start, b_end in zip(config.distance_bins[:-1], config.distance_bins[1:]):
            bin_name = f"{b_start}-{b_end}"
            B = B_by_distance.get(bin_name)
            if B is None:
                continue
            
            # Find keys in this bin
            bin_mask = (distances >= b_start) & (distances < b_end)
            if not bin_mask.any():
                continue
            
            bin_indices = bin_mask.nonzero(as_tuple=True)[0]
            a_keys_bin = a_keys[bin_indices]  # [n_bin_keys, n_features]
            
            # Vectorized: v = a_query @ B, then preds = v @ a_keys_bin.T
            v = a_query @ B  # [n_features]
            preds_bin = a_keys_bin @ v  # [n_bin_keys] - equivalent to v @ a_keys_bin.T
            
            # Fill in predictions (match dtype)
            predicted_row[bin_indices] = preds_bin.to(predicted_row.dtype)
            
            # Collect for bin-specific correlation (one sync per bin, not per key)
            preds_np = preds_bin.cpu().float().numpy()
            actual_bin = actual_row[bin_indices].cpu().float().numpy()
            pairs_by_distance[bin_name]['pred'].extend(preds_np.tolist())
            pairs_by_distance[bin_name]['actual'].extend(actual_bin.tolist())
        
        n_total_pairs += len(valid_keys)
        
        # Per-query correlation (this row only)
        pred_np = predicted_row.cpu().float().numpy()
        actual_np = actual_row.cpu().float().numpy()
        
        if len(pred_np) > 2:
            # Check for constant arrays
            if np.std(pred_np) < 1e-9 or np.std(actual_np) < 1e-9:
                n_rows_skipped += 1
                continue
            try:
                r_p, _ = stats.pearsonr(pred_np, actual_np)
                r_s, _ = stats.spearmanr(pred_np, actual_np)
                if not np.isnan(r_p):
                    per_query_pearson.append(r_p)
                if not np.isnan(r_s):
                    per_query_spearman.append(r_s)
            except:
                n_rows_skipped += 1
    
    # Average per-query correlations (fixes cross-row normalization artifacts)
    avg_pearson = np.mean(per_query_pearson) if per_query_pearson else 0.0
    avg_spearman = np.mean(per_query_spearman) if per_query_spearman else 0.0
    
    # === Compute per-bin correlations and statistics ===
    pearson_by_dist = {}
    spearman_by_dist = {}
    n_pairs_by_dist = {}
    actual_mean_by_bin = {}
    actual_std_by_bin = {}
    predicted_mean_by_bin = {}
    predicted_std_by_bin = {}
    sign_by_bin = {}
    bins_with_no_data = []
    
    # Build complete list of expected bins
    all_bins = [f"{b_start}-{b_end}" for b_start, b_end in 
                zip(config.distance_bins[:-1], config.distance_bins[1:])]
    
    for bin_name in all_bins:
        data = pairs_by_distance.get(bin_name, {'pred': [], 'actual': []})
        
        if len(data['pred']) < 3:
            bins_with_no_data.append(bin_name)
            continue
        
        pred_arr = np.array(data['pred'])
        actual_arr = np.array(data['actual'])
        
        # Per-bin mean/std
        actual_mean_by_bin[bin_name] = float(np.mean(actual_arr))
        actual_std_by_bin[bin_name] = float(np.std(actual_arr))
        predicted_mean_by_bin[bin_name] = float(np.mean(pred_arr))
        predicted_std_by_bin[bin_name] = float(np.std(pred_arr))
        
        # Correlations
        try:
            r_p, _ = stats.pearsonr(pred_arr, actual_arr)
            r_s, _ = stats.spearmanr(pred_arr, actual_arr)
            if np.isnan(r_p):
                r_p = 0.0
            if np.isnan(r_s):
                r_s = 0.0
        except:
            r_p, r_s = 0.0, 0.0
        
        pearson_by_dist[bin_name] = r_p
        spearman_by_dist[bin_name] = r_s
        n_pairs_by_dist[bin_name] = len(pred_arr)
        
        # Sign of correlation (+1, -1, or 0)
        if abs(r_s) < 0.01:
            sign_by_bin[bin_name] = 0
        else:
            sign_by_bin[bin_name] = 1 if r_s > 0 else -1
    
    # === Extract named distance-specific Spearman correlations ===
    spearman_local = spearman_by_dist.get("0-4", 0.0)
    spearman_mid = spearman_by_dist.get("16-32", 0.0)
    spearman_long = spearman_by_dist.get("128-256", 0.0)
    
    # Compute spearman_256plus by aggregating pairs from 256-512 and 512-1024 bins
    pairs_256plus_pred = []
    pairs_256plus_actual = []
    for bin_name in ["256-512", "512-1024"]:
        data = pairs_by_distance.get(bin_name, {'pred': [], 'actual': []})
        pairs_256plus_pred.extend(data['pred'])
        pairs_256plus_actual.extend(data['actual'])
    
    if len(pairs_256plus_pred) > 2:
        try:
            spearman_256plus, _ = stats.spearmanr(pairs_256plus_pred, pairs_256plus_actual)
            if np.isnan(spearman_256plus):
                spearman_256plus = 0.0
        except:
            spearman_256plus = 0.0
    else:
        spearman_256plus = 0.0
    
    # === Compute sign stability ===
    # Fraction of bins with same sign as overall Spearman
    overall_sign = 1 if avg_spearman > 0.01 else (-1 if avg_spearman < -0.01 else 0)
    if sign_by_bin and overall_sign != 0:
        matching_signs = sum(1 for s in sign_by_bin.values() if s == overall_sign)
        sign_stability = matching_signs / len(sign_by_bin)
    else:
        sign_stability = 0.0
    
    # === Compute skipped fraction ===
    skipped_fraction = n_rows_skipped / n_rows_total if n_rows_total > 0 else 0.0
    
    return RoutingValidationResult(
        layer_idx=layer_idx,
        head_idx=0,  # Will be set by caller
        pearson_r=avg_pearson,
        spearman_r=avg_spearman,
        pearson_by_distance=pearson_by_dist,
        spearman_by_distance=spearman_by_dist,
        n_pairs_by_distance=n_pairs_by_dist,
        n_query_positions=seq_len,
        n_total_pairs=n_total_pairs,
        # New diagnostics
        n_rows_skipped=n_rows_skipped,
        n_rows_total=n_rows_total,
        skipped_fraction=skipped_fraction,
        bins_with_no_data=bins_with_no_data,
        actual_mean_by_bin=actual_mean_by_bin,
        actual_std_by_bin=actual_std_by_bin,
        predicted_mean_by_bin=predicted_mean_by_bin,
        predicted_std_by_bin=predicted_std_by_bin,
        spearman_local=spearman_local,
        spearman_mid=spearman_mid,
        spearman_long=spearman_long,
        spearman_256plus=spearman_256plus,
        sign_stability=sign_stability,
        sign_by_bin=sign_by_bin,
    )


# =============================================================================
# OV Writing Validation (Part 2B)
# =============================================================================

@dataclass
class WritingValidationResult:
    """Results from OV writing validation."""
    layer_idx: int
    head_idx: int
    
    # Top-K overlap metrics
    jaccard_at_k: float  # Jaccard similarity of top-K predicted vs actual
    ndcg_at_k: float     # NDCG score
    
    # Correlation of deltas
    delta_pearson_r: Optional[float]
    delta_spearman_r: float
    
    # Stats
    n_positions: int = 0
    top_k: int = 20


def compute_ndcg(predicted_ranks: np.ndarray, actual_ranks: np.ndarray, k: int) -> float:
    """Compute NDCG@k."""
    # Create relevance based on actual ranking
    n = len(actual_ranks)
    relevance = np.zeros(n)
    for i, idx in enumerate(actual_ranks[:k]):
        relevance[idx] = k - i  # Higher relevance for top actual
    
    # DCG for predicted order
    dcg = 0.0
    for i, idx in enumerate(predicted_ranks[:k]):
        dcg += relevance[idx] / np.log2(i + 2)
    
    # Ideal DCG (actual top-k in order)
    idcg = sum((k - i) / np.log2(i + 2) for i in range(min(k, len(actual_ranks))))
    
    return dcg / idcg if idcg > 0 else 0.0


def validate_writing_ranking(
    attention_weights: torch.Tensor,  # [seq_len, seq_len] for this head
    sae_activations_key: torch.Tensor,  # [seq_len, n_features] SAE activations at key positions
    W2F_matrix: torch.Tensor,          # [n_features, n_features] write matrix
    actual_deltas: torch.Tensor,       # [seq_len, n_features] actual feature changes
    config: ValidationConfig,
) -> WritingValidationResult:
    """
    Validate W2F matrix against actual feature deltas.
    
    Predicted Δk at t = Σ_s α(t,s) * (a_key[s] @ W2F[:, k])
    
    We compare the ranking of top predicted vs top actual changed features.
    """
    seq_len = attention_weights.shape[0]
    
    all_jaccard = []
    all_pred_delta = []
    all_actual_delta = []
    
    for t in range(seq_len):
        # Get attention weights for this query position
        alpha = attention_weights[t]  # [seq_len]
        
        # Predicted delta: Σ_s α(t,s) * (a_key[s] @ W2F)
        # Shape: sum over s of [alpha[s] * (n_features)]
        weighted_key_features = alpha.unsqueeze(1) * sae_activations_key  # [seq_len, n_features]
        aggregated_key = weighted_key_features.sum(dim=0)  # [n_features]
        predicted_delta = aggregated_key @ W2F_matrix  # [n_features]
        
        # Actual delta for this position
        actual_delta = actual_deltas[t]  # [n_features]
        
        # Get top-K predicted and actual by absolute value
        _, pred_top_k = torch.topk(predicted_delta.abs(), config.top_k_features)
        _, actual_top_k = torch.topk(actual_delta.abs(), config.top_k_features)
        
        pred_set = set(pred_top_k.cpu().numpy())
        actual_set = set(actual_top_k.cpu().numpy())
        
        # Jaccard
        intersection = len(pred_set & actual_set)
        union = len(pred_set | actual_set)
        jaccard = intersection / union if union > 0 else 0.0
        all_jaccard.append(jaccard)
        
        # Collect for correlation
        all_pred_delta.append(predicted_delta.cpu().numpy())
        all_actual_delta.append(actual_delta.cpu().numpy())
    
    # Average Jaccard
    avg_jaccard = np.mean(all_jaccard)
    
    # Overall correlation of deltas
    pred_flat = np.concatenate(all_pred_delta)
    actual_flat = np.concatenate(all_actual_delta)
    
    pearson_r, _ = stats.pearsonr(pred_flat, actual_flat) if len(pred_flat) > 2 else (0.0, 1.0)
    spearman_r, _ = stats.spearmanr(pred_flat, actual_flat) if len(pred_flat) > 2 else (0.0, 1.0)
    
    return WritingValidationResult(
        layer_idx=0,  # Set by caller
        head_idx=0,   # Set by caller
        jaccard_at_k=avg_jaccard,
        ndcg_at_k=0.0,  # TODO: compute average NDCG
        delta_pearson_r=pearson_r,
        delta_spearman_r=spearman_r,
        n_positions=seq_len,
        top_k=config.top_k_features,
    )


# =============================================================================
# Main Validator Class
# =============================================================================

class ActivationValidator:
    """
    Validates weight-space SAE analysis against runtime activations.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        sae_manager,  # SAEManager from sae_utils
        analysis_results: Dict,  # Loaded JSON from weight-space analysis
        config: Optional[ValidationConfig] = None,
        analysis_config: Optional[AnalysisConfig] = None,
        qk_features: Optional[int] = None,  # SAE features for QK routing (default from config)
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sae_manager = sae_manager
        self.analysis_results = analysis_results
        self.config = config or ValidationConfig()
        self.analysis_config = analysis_config or AnalysisConfig()
        
        self.cache = ActivationCache()
        self.device = next(model.parameters()).device
        
        # Cache for computed B matrices (computed on-demand)
        self.B_matrices: Dict[Tuple[int, int], torch.Tensor] = {}
        self.W2F_matrices: Dict[Tuple[int, int], torch.Tensor] = {}
        
        # Get feature indices used in analysis - keyed by sae_layer
        self._feature_indices_cache: Dict[int, torch.Tensor] = {}
        config_data = analysis_results.get("config", {})
        self.n_features = qk_features or config_data.get("feature_subset_size", 2048)
        self._base_seed = config_data.get("seed", 42)
        self._sae_width = config_data.get("sae_width", 16384)
        
        # Store feature counts for report
        self.qk_features = self.n_features
        self.ov_features = self._sae_width  # OV_f always uses full SAE
    
    def _get_feature_indices(self, sae_layer: int) -> torch.Tensor:
        """Get the feature indices used for analysis (per-layer, matching analysis pipeline)."""
        if sae_layer in self._feature_indices_cache:
            return self._feature_indices_cache[sae_layer]
        
        # Match exact logic from prepare_sae_features() in sae_utils.py:
        # g = torch.Generator().manual_seed(seed + layer_idx * 1000)
        # feature_indices = torch.randperm(n_features, generator=g)[:subset_size]
        g = torch.Generator().manual_seed(self._base_seed + sae_layer * 1000)
        indices = torch.randperm(self._sae_width, generator=g)[:self.n_features]
        indices = indices.to(device=self.device)
        
        self._feature_indices_cache[sae_layer] = indices
        return indices
    
    def compute_QK_features(self, layer_idx: int, head_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q_f and K_f feature projections for a head.
        
        Returns:
            (Q_f, K_f) each of shape [n_features, head_dim]
        """
        # Import here to avoid circular import
        from weight_extraction import extract_layer_weights, get_qkv_for_head
        from config import Gemma2Config, GEMMA2_CONFIG
        
        # Get SAE layer for this attention layer
        sae_layer = GEMMA2_CONFIG.get_sae_layer_for_attn(layer_idx)
        feature_indices = self._get_feature_indices(sae_layer)
        
        # Get SAE decoder directions for subset features
        sae_decoder = self.sae_manager.get_decoder(sae_layer)  # [d_model, n_sae_features]
        decoder_subset = sae_decoder[:, feature_indices]  # [d_model, n_subset]
        
        # Match RMS normalization + sqrt(hidden_size) scaling from analysis pipeline
        if GEMMA2_CONFIG.normalize_decoder_directions:
            hidden_size = GEMMA2_CONFIG.hidden_size
            decoder_subset = F.normalize(decoder_subset, dim=0) * (hidden_size ** 0.5)
        
        # Use fold_gamma=False since our hook captures post-layernorm residual
        layer_weights = extract_layer_weights(
            self.model, layer_idx, GEMMA2_CONFIG, 
            fold_gamma=False, device=str(self.device), dtype=torch.float32
        )
        W_Q, W_K, W_V = get_qkv_for_head(layer_weights, head_idx, GEMMA2_CONFIG)
        
        # W_Q, W_K are [head_dim, hidden_size], we need [hidden_size, head_dim]
        W_Q = W_Q.T.to(self.device)  # [hidden_size, head_dim]
        W_K = W_K.T.to(self.device)  # [hidden_size, head_dim]
        
        # Compute Q_f and K_f
        Q_f = decoder_subset.T @ W_Q  # [n_features, head_dim]
        K_f = decoder_subset.T @ W_K  # [n_features, head_dim]
        
        return Q_f, K_f
    
    def compute_B_matrices_by_distance(
        self, 
        layer_idx: int, 
        head_idx: int,
        distance_bins: List[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute RoPE-aware B matrices for each distance bin.
        
        Uses B_Δ = Q_f @ R_Δ @ K_f.T where Δ is the bin center.
        
        Returns:
            Dict mapping bin_name ("0-8", "8-32", etc.) to B_Δ matrix
        """
        from rope_utils import compute_rotated_affinity_matrix, compute_rope_frequencies
        from config import GEMMA2_CONFIG
        
        if distance_bins is None:
            # Narrower bins for better RoPE accuracy up to 1024 tokens (matches ValidationConfig)
            distance_bins = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]
        
        Q_f, K_f = self.compute_QK_features(layer_idx, head_idx)
        freqs = compute_rope_frequencies(GEMMA2_CONFIG).to(self.device)
        scale = GEMMA2_CONFIG.attention_scale
        
        B_by_distance = {}
        for bin_start, bin_end in distance_bins:
            bin_name = f"{bin_start}-{bin_end}"
            # Use bin center as the delta for RoPE rotation
            delta = (bin_start + bin_end) // 2
            
            if delta == 0:
                # No rotation needed for delta=0
                B = (Q_f @ K_f.T) * scale
            else:
                # RoPE rotation convention: we tested both +delta and -delta
                # Empirically, +delta gives better correlations, so we use that
                B = compute_rotated_affinity_matrix(Q_f, K_f, delta, freqs, scale)
            
            # Apply softcap if configured
            if GEMMA2_CONFIG.apply_softcap and GEMMA2_CONFIG.attn_logit_softcapping:
                cap = GEMMA2_CONFIG.attn_logit_softcapping
                B = cap * torch.tanh(B / cap)
            
            B_by_distance[bin_name] = B
        
        return B_by_distance
    
    def compute_B_matrix(self, layer_idx: int, head_idx: int) -> torch.Tensor:
        """
        Compute B matrix (feature affinity) for Δ=0 (no RoPE rotation).
        Kept for backward compatibility.
        """
        from qk_routing import compute_affinity_matrix
        from config import GEMMA2_CONFIG
        
        Q_f, K_f = self.compute_QK_features(layer_idx, head_idx)
        B = compute_affinity_matrix(Q_f, K_f, GEMMA2_CONFIG)
        return B
    
    def get_B_matrix(self, layer_idx: int, head_idx: int) -> torch.Tensor:
        """Get B matrix (Δ=0), computing and caching if needed."""
        key = (layer_idx, head_idx)
        if key not in self.B_matrices:
            print(f"  Computing B matrix for L{layer_idx}H{head_idx}...")
            self.B_matrices[key] = self.compute_B_matrix(layer_idx, head_idx)
        return self.B_matrices[key]
    
    def get_B_matrices_by_distance(self, layer_idx: int, head_idx: int) -> Dict[str, torch.Tensor]:
        """Get RoPE-aware B matrices per distance bin, computing and caching if needed."""
        key = (layer_idx, head_idx, "rope")
        if key not in self.B_matrices:
            print(f"  Computing RoPE-aware B matrices for L{layer_idx}H{head_idx}...")
            self.B_matrices[key] = self.compute_B_matrices_by_distance(layer_idx, head_idx)
        return self.B_matrices[key]
    
    def validate_routing_for_head(
        self,
        prompts: List[str],
        layer_idx: int,
        head_idx: int,
    ) -> RoutingValidationResult:
        """
        Validate QK routing for a specific head.
        
        Args:
            prompts: List of text prompts to run through model
            layer_idx: Attention layer index
            head_idx: Head index within the layer
        
        Returns:
            RoutingValidationResult
        """
        # Get or compute B matrix
        B = self.get_B_matrix(layer_idx, head_idx)
        
        # Get SAE layer for this attention layer (method is on Gemma2Config)
        from config import GEMMA2_CONFIG
        sae_layer = GEMMA2_CONFIG.get_sae_layer_for_attn(layer_idx)
        
        all_results = []
        
        for prompt in prompts:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Set up hooks
            self.cache.clear()
            handles = create_attention_hooks(self.model, self.cache, [layer_idx])
            
            try:
                # Forward pass with attention output
                with torch.no_grad():
                    outputs = self.model(**inputs, output_attentions=True)
                
                # Get pre-attention residual and encode with SAE
                residual = self.cache.residual_pre_attn.get(layer_idx)
                if residual is None:
                    continue
                
                residual = residual[0]  # Remove batch dim: [seq_len, d_model]
                
                # Encode with SAE
                sae_acts_full = self.sae_manager.encode(residual, sae_layer)  # [seq_len, all_features]
                
                # Subset to only features used in B matrix
                feature_indices = self._get_feature_indices(sae_layer)
                sae_acts = sae_acts_full[:, feature_indices]  # [seq_len, n_subset_features]
                
                # Get attention logits for this head
                # outputs.attentions is tuple of [batch, n_heads, seq, seq] per layer
                attn_weights = outputs.attentions[layer_idx][0, head_idx]  # [seq, seq]
                
                # Get attention weights - we use log(weights) in the correlation function
                # which equals logits up to a per-row constant, making correlation meaningful
                
                # Validate
                result = validate_routing_correlation(
                    sae_activations=sae_acts,
                    B_matrix=B,
                    actual_logits=attn_weights,  # Using weights instead of logits
                    config=self.config,
                    analysis_config=self.analysis_config,
                    layer_idx=layer_idx,
                )
                result.head_idx = head_idx
                all_results.append(result)
                
            finally:
                for h in handles:
                    h.remove()
        
        # Aggregate results across prompts
        if not all_results:
            return RoutingValidationResult(layer_idx=layer_idx, head_idx=head_idx, pearson_r=0, spearman_r=0)
        
        # Average correlations
        avg_pearson = np.mean([r.pearson_r for r in all_results])
        avg_spearman = np.mean([r.spearman_r for r in all_results])
        
        # Aggregate distance bins (with per-bin mean/std statistics)
        combined_pearson_by_dist = defaultdict(list)
        combined_spearman_by_dist = defaultdict(list)
        combined_n_pairs = defaultdict(int)
        combined_pred_mean = defaultdict(list)
        combined_pred_std = defaultdict(list)
        combined_act_mean = defaultdict(list)
        combined_act_std = defaultdict(list)
        
        for r in all_results:
            for bin_name, corr in r.pearson_by_distance.items():
                combined_pearson_by_dist[bin_name].append(corr)
                combined_n_pairs[bin_name] += r.n_pairs_by_distance.get(bin_name, 0)
            for bin_name, corr in r.spearman_by_distance.items():
                combined_spearman_by_dist[bin_name].append(corr)
            # Collect per-bin mean/std if available
            for bin_name, val in r.predicted_mean_by_bin.items():
                combined_pred_mean[bin_name].append(val)
            for bin_name, val in r.predicted_std_by_bin.items():
                combined_pred_std[bin_name].append(val)
            for bin_name, val in r.actual_mean_by_bin.items():
                combined_act_mean[bin_name].append(val)
            for bin_name, val in r.actual_std_by_bin.items():
                combined_act_std[bin_name].append(val)
        
        avg_pearson_by_dist = {k: np.mean(v) for k, v in combined_pearson_by_dist.items()}
        avg_spearman_by_dist = {k: np.mean(v) for k, v in combined_spearman_by_dist.items()}
        avg_pred_mean = {k: np.mean(v) for k, v in combined_pred_mean.items()}
        avg_pred_std = {k: np.mean(v) for k, v in combined_pred_std.items()}
        avg_act_mean = {k: np.mean(v) for k, v in combined_act_mean.items()}
        avg_act_std = {k: np.mean(v) for k, v in combined_act_std.items()}
        
        # Extract named Spearman values from the per-bin dict
        spearman_local = avg_spearman_by_dist.get("0-4", 0.0)
        spearman_mid = avg_spearman_by_dist.get("16-32", 0.0)
        spearman_long = avg_spearman_by_dist.get("128-256", 0.0)
        # For 256+, average the 256-512 and 512-1024 bins
        spearman_256_512 = avg_spearman_by_dist.get("256-512", 0.0)
        spearman_512_1024 = avg_spearman_by_dist.get("512-1024", 0.0)
        if spearman_256_512 != 0.0 or spearman_512_1024 != 0.0:
            non_zero = [v for v in [spearman_256_512, spearman_512_1024] if v != 0.0]
            spearman_256plus = np.mean(non_zero) if non_zero else 0.0
        else:
            spearman_256plus = 0.0
        
        # Compute sign stability
        overall_sign = 1 if avg_spearman > 0.01 else (-1 if avg_spearman < -0.01 else 0)
        sign_by_bin = {}
        for bin_name, corr in avg_spearman_by_dist.items():
            if abs(corr) < 0.01:
                sign_by_bin[bin_name] = 0
            else:
                sign_by_bin[bin_name] = 1 if corr > 0 else -1
        if sign_by_bin and overall_sign != 0:
            matching_signs = sum(1 for s in sign_by_bin.values() if s == overall_sign)
            sign_stability = matching_signs / len(sign_by_bin)
        else:
            sign_stability = 0.0
        
        return RoutingValidationResult(
            layer_idx=layer_idx,
            head_idx=head_idx,
            pearson_r=avg_pearson,
            spearman_r=avg_spearman,
            pearson_by_distance=avg_pearson_by_dist,
            spearman_by_distance=avg_spearman_by_dist,
            n_pairs_by_distance=dict(combined_n_pairs),
            n_query_positions=sum(r.n_query_positions for r in all_results),
            n_total_pairs=sum(r.n_total_pairs for r in all_results),
            # Per-bin mean/std statistics
            predicted_mean_by_bin=avg_pred_mean,
            predicted_std_by_bin=avg_pred_std,
            actual_mean_by_bin=avg_act_mean,
            actual_std_by_bin=avg_act_std,
            # Named Spearman values
            spearman_local=spearman_local,
            spearman_mid=spearman_mid,
            spearman_long=spearman_long,
            spearman_256plus=spearman_256plus,
            sign_stability=sign_stability,
            sign_by_bin=sign_by_bin,
        )
    
    def validate_routing_for_head_rope_aware(
        self,
        prompts: List[str],
        layer_idx: int,
        head_idx: int,
    ) -> RoutingValidationResult:
        """
        Validate QK routing using RoPE-aware B matrices per distance bin.
        
        This is the recommended validation method as it accounts for:
        1. RoPE rotation effects at different distances
        2. Per-query correlation (avoids cross-row softmax artifacts)
        """
        # Get or compute RoPE-aware B matrices per distance bin
        B_by_distance = self.get_B_matrices_by_distance(layer_idx, head_idx)
        
        # Get SAE layer for this attention layer
        from config import GEMMA2_CONFIG
        sae_layer = GEMMA2_CONFIG.get_sae_layer_for_attn(layer_idx)
        
        all_results = []
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            self.cache.clear()
            handles = create_attention_hooks(self.model, self.cache, [layer_idx])
            
            try:
                with torch.no_grad():
                    outputs = self.model(**inputs, output_attentions=True)
                
                residual = self.cache.residual_pre_attn.get(layer_idx)
                if residual is None:
                    continue
                
                residual = residual[0]
                
                sae_acts_full = self.sae_manager.encode(residual, sae_layer)
                feature_indices = self._get_feature_indices(sae_layer)
                sae_acts = sae_acts_full[:, feature_indices]
                
                attn_weights = outputs.attentions[layer_idx][0, head_idx]
                
                result = validate_routing_rope_aware(
                    sae_activations=sae_acts,
                    B_by_distance=B_by_distance,
                    actual_logits=attn_weights,
                    config=self.config,
                    layer_idx=layer_idx,
                )
                result.head_idx = head_idx
                all_results.append(result)
                
            finally:
                for h in handles:
                    h.remove()
        
        # Aggregate results across prompts
        if not all_results:
            return RoutingValidationResult(layer_idx=layer_idx, head_idx=head_idx, pearson_r=0, spearman_r=0)
        
        avg_pearson = np.mean([r.pearson_r for r in all_results])
        avg_spearman = np.mean([r.spearman_r for r in all_results])
        
        # Aggregate per-bin correlations
        combined_pearson_by_dist = defaultdict(list)
        combined_spearman_by_dist = defaultdict(list)
        combined_n_pairs = defaultdict(int)
        combined_pred_mean = defaultdict(list)
        combined_pred_std = defaultdict(list)
        combined_act_mean = defaultdict(list)
        combined_act_std = defaultdict(list)
        combined_signs = defaultdict(list)
        
        for r in all_results:
            for bin_name, corr in r.pearson_by_distance.items():
                combined_pearson_by_dist[bin_name].append(corr)
                combined_n_pairs[bin_name] += r.n_pairs_by_distance.get(bin_name, 0)
            for bin_name, corr in r.spearman_by_distance.items():
                combined_spearman_by_dist[bin_name].append(corr)
            for bin_name, val in r.predicted_mean_by_bin.items():
                combined_pred_mean[bin_name].append(val)
            for bin_name, val in r.predicted_std_by_bin.items():
                combined_pred_std[bin_name].append(val)
            for bin_name, val in r.actual_mean_by_bin.items():
                combined_act_mean[bin_name].append(val)
            for bin_name, val in r.actual_std_by_bin.items():
                combined_act_std[bin_name].append(val)
            for bin_name, sign in r.sign_by_bin.items():
                combined_signs[bin_name].append(sign)
        
        avg_pearson_by_dist = {k: np.mean(v) for k, v in combined_pearson_by_dist.items()}
        avg_spearman_by_dist = {k: np.mean(v) for k, v in combined_spearman_by_dist.items()}
        avg_pred_mean = {k: np.mean(v) for k, v in combined_pred_mean.items()}
        avg_pred_std = {k: np.mean(v) for k, v in combined_pred_std.items()}
        avg_act_mean = {k: np.mean(v) for k, v in combined_act_mean.items()}
        avg_act_std = {k: np.mean(v) for k, v in combined_act_std.items()}
        
        # Aggregate sign by majority vote
        avg_sign_by_bin = {}
        for bin_name, signs in combined_signs.items():
            avg_sign = np.mean(signs)
            if avg_sign > 0.3:
                avg_sign_by_bin[bin_name] = 1
            elif avg_sign < -0.3:
                avg_sign_by_bin[bin_name] = -1
            else:
                avg_sign_by_bin[bin_name] = 0
        
        # Aggregate named Spearman values from the per-bin dict
        # (The per-prompt spearman_local etc. may be 0.0 if not explicitly set)
        avg_spearman_local = avg_spearman_by_dist.get("0-4", 0.0)
        avg_spearman_mid = avg_spearman_by_dist.get("16-32", 0.0)
        avg_spearman_long = avg_spearman_by_dist.get("128-256", 0.0)
        # For 256+, average the 256-512 and 512-1024 bins
        spearman_256_512 = avg_spearman_by_dist.get("256-512", 0.0)
        spearman_512_1024 = avg_spearman_by_dist.get("512-1024", 0.0)
        if spearman_256_512 != 0.0 or spearman_512_1024 != 0.0:
            non_zero = [v for v in [spearman_256_512, spearman_512_1024] if v != 0.0]
            avg_spearman_256plus = np.mean(non_zero) if non_zero else 0.0
        else:
            avg_spearman_256plus = 0.0
        
        # Aggregate row skip counts
        total_rows_skipped = sum(r.n_rows_skipped for r in all_results)
        total_rows = sum(r.n_rows_total for r in all_results)
        skipped_fraction = total_rows_skipped / total_rows if total_rows > 0 else 0.0
        
        # Aggregate bins with no data (union)
        all_empty_bins = set()
        for r in all_results:
            all_empty_bins.update(r.bins_with_no_data)
        
        # Aggregate sign stability
        avg_sign_stability = np.mean([r.sign_stability for r in all_results])
        
        return RoutingValidationResult(
            layer_idx=layer_idx,
            head_idx=head_idx,
            pearson_r=avg_pearson,
            spearman_r=avg_spearman,
            pearson_by_distance=avg_pearson_by_dist,
            spearman_by_distance=avg_spearman_by_dist,
            n_pairs_by_distance=dict(combined_n_pairs),
            n_query_positions=sum(r.n_query_positions for r in all_results),
            n_total_pairs=sum(r.n_total_pairs for r in all_results),
            # Aggregated diagnostics
            n_rows_skipped=total_rows_skipped,
            n_rows_total=total_rows,
            skipped_fraction=skipped_fraction,
            bins_with_no_data=list(all_empty_bins),
            actual_mean_by_bin=avg_act_mean,
            actual_std_by_bin=avg_act_std,
            predicted_mean_by_bin=avg_pred_mean,
            predicted_std_by_bin=avg_pred_std,
            spearman_local=avg_spearman_local,
            spearman_mid=avg_spearman_mid,
            spearman_long=avg_spearman_long,
            spearman_256plus=avg_spearman_256plus,
            sign_stability=avg_sign_stability,
            sign_by_bin=avg_sign_by_bin,
        )
    
    def compute_W2F_matrix(self, layer_idx: int, head_idx: int) -> torch.Tensor:
        """
        Compute W2F (write-to-feature) matrix for a head.
        
        W2F[j, k] = cosine_similarity(write_j, decoder_k)
        where write_j = decoder_j @ W_V.T @ W_O.T
        
        Returns:
            W2F matrix of shape [n_features, n_features]
        """
        from weight_extraction import extract_layer_weights
        from ov_writing import compute_write_vectors_fast, project_writes_to_features
        from config import GEMMA2_CONFIG
        
        sae_layer = GEMMA2_CONFIG.get_sae_layer_for_attn(layer_idx)
        feature_indices = self._get_feature_indices(sae_layer)
        
        # Get SAE decoder directions
        sae_decoder = self.sae_manager.get_decoder(sae_layer)
        decoder_subset = sae_decoder[:, feature_indices].T  # [n_features, d_model]
        
        # Normalize like in analysis pipeline
        if GEMMA2_CONFIG.normalize_decoder_directions:
            decoder_subset = F.normalize(decoder_subset, dim=1) * (GEMMA2_CONFIG.hidden_size ** 0.5)
        
        # Get layer weights
        layer_weights = extract_layer_weights(
            self.model, layer_idx, GEMMA2_CONFIG,
            fold_gamma=False, device=str(self.device), dtype=torch.float32
        )
        
        # Get W_V and W_O for this head
        kv_group = GEMMA2_CONFIG.query_to_kv_group(head_idx)
        W_V = layer_weights.W_V[kv_group].to(self.device)
        W_O = layer_weights.W_O[head_idx].to(self.device)
        
        # Compute write vectors
        write_vectors = compute_write_vectors_fast(decoder_subset, W_V, W_O)
        
        # Project to feature space (cosine similarity)
        W2F = project_writes_to_features(write_vectors, decoder_subset, normalize=True)
        
        return W2F
    
    def get_W2F_matrix(self, layer_idx: int, head_idx: int) -> torch.Tensor:
        """Get W2F matrix, computing and caching if needed."""
        key = (layer_idx, head_idx)
        if key not in self.W2F_matrices:
            print(f"  Computing W2F matrix for L{layer_idx}H{head_idx}...")
            self.W2F_matrices[key] = self.compute_W2F_matrix(layer_idx, head_idx)
        return self.W2F_matrices[key]
    
    def compute_OV_f_matrix(self, layer_idx: int, head_idx: int) -> torch.Tensor:
        """
        Compute OV_f (Output-Value in feature space) matrix for a head.
        
        OV_f[j, k] = d_k^T @ OV_circuit @ d_j
        where OV_circuit = (W_O @ W_V)^T = W_V.T @ W_O.T
        
        This represents: "When feature j is read at value position, 
        how much does the write project onto feature k's decoder direction?"
        
        Returns:
            OV_f matrix of shape [n_features, n_features]
        """
        from weight_extraction import extract_layer_weights
        from config import GEMMA2_CONFIG
        
        sae_layer = GEMMA2_CONFIG.get_sae_layer_for_attn(layer_idx)
        feature_indices = self._get_feature_indices(sae_layer)
        
        # Get SAE decoder directions: D is [d_model, n_features]
        sae_decoder = self.sae_manager.get_decoder(sae_layer)
        D = sae_decoder[:, feature_indices]  # [d_model, n_features]
        
        # Normalize like in analysis pipeline
        if GEMMA2_CONFIG.normalize_decoder_directions:
            D = F.normalize(D, dim=0) * (GEMMA2_CONFIG.hidden_size ** 0.5)
        
        # Get layer weights
        layer_weights = extract_layer_weights(
            self.model, layer_idx, GEMMA2_CONFIG,
            fold_gamma=False, device=str(self.device), dtype=torch.float32
        )
        
        # Get W_V and W_O for this head
        # W_V: [head_dim, d_model], W_O: [d_model, head_dim]
        kv_group = GEMMA2_CONFIG.query_to_kv_group(head_idx)
        W_V = layer_weights.W_V[kv_group].to(self.device)  # [head_dim, d_model]
        W_O = layer_weights.W_O[head_idx].to(self.device)  # [d_model, head_dim]
        
        # OV circuit: maps residual -> residual
        # output = input @ W_V.T @ W_O.T
        # OV_circuit = W_V.T @ W_O.T has shape [d_model, d_model]
        OV_circuit = W_V.T @ W_O.T  # [d_model, d_model]
        
        # Project into feature space:
        # OV_f = D.T @ OV_circuit @ D
        # OV_f[j, k] = d_j @ OV_circuit @ d_k = "reading feature j writes along feature k"
        # Wait, we want: "reading j writes k", so:
        # write_j = d_j @ OV_circuit (the vector written when feature j is read)
        # Then OV_f[j, k] = write_j · d_k
        # This is: (D @ OV_circuit).T @ D but let's compute directly
        
        # D is [d_model, n_features], OV_circuit is [d_model, d_model]
        # D.T @ OV_circuit @ D gives [n_features, n_features]
        # But this computes d_i @ OV @ d_j which is "d_i transforms to project onto d_j"
        # We want: for input d_j, output projects onto d_k
        # So: OV_f[j, k] = d_k · (d_j @ OV_circuit) = (d_j @ OV_circuit) · d_k
        #                = d_j @ OV_circuit @ d_k
        # In matrix form: OV_f = D.T @ OV_circuit.T @ D
        
        OV_f = D.T @ OV_circuit.T @ D  # [n_features, n_features]
        
        return OV_f
    
    def get_OV_f_matrix(self, layer_idx: int, head_idx: int) -> torch.Tensor:
        """Get OV_f matrix, computing and caching if needed."""
        key = (layer_idx, head_idx, "OV_f")
        if key not in self.W2F_matrices:  # reuse same cache dict
            print(f"  Computing OV_f matrix for L{layer_idx}H{head_idx}...")
            self.W2F_matrices[key] = self.compute_OV_f_matrix(layer_idx, head_idx)
        return self.W2F_matrices[key]
    
    def compute_circuit_matrix(self, layer_idx: int, head_idx: int) -> torch.Tensor:
        """
        Compute the full circuit matrix B @ OV_f for a head.
        
        Circuit[i, k] = Σ_j B[i, j] * OV_f[j, k]
        
        This represents: "If query has feature i, what features get written?"
        (Combines routing + writing into one matrix)
        
        Returns:
            Circuit matrix of shape [n_features, n_features]
        """
        B = self.get_B_matrix(layer_idx, head_idx)  # [n_features, n_features]
        OV_f = self.get_OV_f_matrix(layer_idx, head_idx)  # [n_features, n_features]
        
        # Full circuit: routing @ writing
        Circuit = B @ OV_f
        
        return Circuit
    
    def get_circuit_matrix(self, layer_idx: int, head_idx: int) -> torch.Tensor:
        """Get full circuit matrix B @ OV_f, computing and caching if needed."""
        key = (layer_idx, head_idx, "circuit")
        if key not in self.W2F_matrices:
            print(f"  Computing circuit matrix (B @ OV_f) for L{layer_idx}H{head_idx}...")
            self.W2F_matrices[key] = self.compute_circuit_matrix(layer_idx, head_idx)
        return self.W2F_matrices[key]
    
    def validate_OV_f_for_head(
        self,
        prompts: List[str],
        layer_idx: int,
        head_idx: int,
    ) -> WritingValidationResult:
        """
        Validate OV_f against actual attention output vectors using ablation isolation.
        
        Uses two forward passes per prompt:
        1. Baseline: normal forward pass
        2. Ablated: forward pass with target head zeroed
        
        Head's actual contribution = attention_output_baseline - attention_output_ablated
        
        Prediction: write_pred = (D @ Σ_s α(t,s) * features[s]) @ OV_circuit
        This works directly in residual space without lossy feature-space round-trip.
        """
        from head_ablation import HeadAblator
        from weight_extraction import extract_layer_weights
        from config import GEMMA2_CONFIG
        import sys
        from io import StringIO
        
        sae_layer = GEMMA2_CONFIG.get_sae_layer_for_attn(layer_idx)
        
        # Use FULL SAE (all 16K features) for OV_f validation for accurate reconstruction
        # (feature subsetting causes sign flip issues in late layers due to missing subspace)
        sae_decoder, b_dec = self.sae_manager.get_decoder_with_bias(sae_layer)
        D = sae_decoder  # [d_model, n_features] - use ALL features
        b_dec = b_dec.to(self.device)  # [d_model]
        # Note: skip normalization for full decoder since we want accurate reconstruction
        
        # Get OV circuit directly (work in residual space, not feature space)
        layer_weights = extract_layer_weights(
            self.model, layer_idx, GEMMA2_CONFIG,
            fold_gamma=False, device=str(self.device), dtype=torch.float32
        )
        kv_group = GEMMA2_CONFIG.query_to_kv_group(head_idx)
        W_V = layer_weights.W_V[kv_group].to(self.device)  # [head_dim, d_model]
        W_O = layer_weights.W_O[head_idx].to(self.device)  # [d_model, head_dim]
        OV_circuit = W_V.T @ W_O.T  # [d_model, d_model]
        
        all_cosine_sim = []
        all_positions = 0
        
        # Create ablator (suppress print output)
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            ablator = HeadAblator(self.model, layer_idx, head_idx)
        finally:
            sys.stdout = old_stdout
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            seq_len = inputs['input_ids'].shape[1]
            
            # === Pass 1: Baseline (no ablation) ===
            self.cache.clear()
            handles = create_attention_hooks(self.model, self.cache, [layer_idx])
            
            try:
                with torch.no_grad():
                    outputs_baseline = self.model(**inputs, output_attentions=True)
                
                # Get pre-attention residual for SAE encoding
                residual_pre = self.cache.residual_pre_attn.get(layer_idx)
                if residual_pre is None:
                    continue
                residual_pre = residual_pre[0]  # [seq_len, d_model]
                
                # Get baseline attention output (all heads combined)
                attention_output_baseline = self.cache.attention_output.get(layer_idx)
                if attention_output_baseline is None:
                    continue
                attention_output_baseline = attention_output_baseline[0]  # [seq_len, d_model]
                
                # Get attention weights for the target head
                attn_weights = outputs_baseline.attentions[layer_idx][0, head_idx].float()  # [seq_len, seq_len]
                
            finally:
                for h in handles:
                    h.remove()
            
            # === Pass 2: With head ablated ===
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                ablator.ablate()
            finally:
                sys.stdout = old_stdout
            
            self.cache.clear()
            handles = create_attention_hooks(self.model, self.cache, [layer_idx])
            
            try:
                with torch.no_grad():
                    outputs_ablated = self.model(**inputs, output_attentions=False)
                
                # Get ablated attention output
                attention_output_ablated = self.cache.attention_output.get(layer_idx)
                if attention_output_ablated is None:
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()
                    try:
                        ablator.restore()
                    finally:
                        sys.stdout = old_stdout
                    continue
                attention_output_ablated = attention_output_ablated[0]  # [seq_len, d_model]
                
            finally:
                for h in handles:
                    h.remove()
                # Restore head
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                try:
                    ablator.restore()
                finally:
                    sys.stdout = old_stdout
            
            # === Compute head-isolated attention output ===
            # This is exactly what THIS head contributed to the attention output
            head_attention_output = attention_output_baseline - attention_output_ablated  # [seq_len, d_model]
            
            # Encode pre-attention residual with SAE (use ALL features)
            sae_pre = self.sae_manager.encode(residual_pre, sae_layer)  # [seq_len, 16384]
            
            # === VECTORIZED: Compute all positions at once ===
            # Create causal mask for attention (lower triangular)
            # attn_weights is [seq_len, seq_len] where [t, s] is attention from t to s
            # We only use positions 1 to seq_len-1 as queries (skip position 0)
            
            # For each query position t, compute weighted sum of source features
            # aggregated[t] = Σ_s α[t,s] * sae_pre[s] for s <= t
            # This is just: aggregated = attn_weights @ sae_pre (attention is already causal)
            aggregated_features = attn_weights @ sae_pre  # [seq_len, n_features]
            
            # Reconstruct attended residuals from features WITH BIAS: [seq_len, d_model]
            # Note: b_dec is the mean direction that SAE learned to add back
            attended_residuals = aggregated_features @ D.T + b_dec  # [seq_len, n_features] @ [n_features, d_model] + [d_model]
            
            # Apply OV circuit to all positions at once: [seq_len, d_model]
            pred_writes = attended_residuals @ OV_circuit  # [seq_len, d_model]
            
            # Skip position 0 (no context), use positions 1 to seq_len-1
            pred_writes = pred_writes[1:]  # [seq_len-1, d_model]
            actual_writes = head_attention_output[1:]  # [seq_len-1, d_model]
            
            # Compute cosine similarity for all positions at once
            # F.cosine_similarity on dim=1 gives per-row similarity
            pred_norms = torch.norm(pred_writes, dim=1, keepdim=True)
            actual_norms = torch.norm(actual_writes, dim=1, keepdim=True)
            
            # Mask out positions with near-zero norms
            valid_mask = (pred_norms.squeeze() > 1e-9) & (actual_norms.squeeze() > 1e-9)
            
            if valid_mask.any():
                cos_sims = F.cosine_similarity(pred_writes[valid_mask], actual_writes[valid_mask], dim=1)
                all_cosine_sim.extend(cos_sims.cpu().tolist())
            
            all_positions += seq_len - 1
        
        # Aggregate results
        avg_cosine = np.mean(all_cosine_sim) if all_cosine_sim else 0.0
        std_cosine = np.std(all_cosine_sim) if all_cosine_sim else 0.0
        
        return WritingValidationResult(
            layer_idx=layer_idx,
            head_idx=head_idx,
            jaccard_at_k=avg_cosine,  # Repurpose: now stores cosine similarity
            ndcg_at_k=std_cosine,  # Repurpose: now stores std of cosine
            delta_pearson_r=None,
            delta_spearman_r=0.0,
            n_positions=all_positions,
            top_k=self.config.top_k_features,
        )
    
    def validate_OV_f_for_layer(
        self,
        prompts: List[str],
        layer_idx: int,
        head_indices: List[int] = None,
    ) -> Dict[int, WritingValidationResult]:
        """
        Validate OV_f for all heads in a layer with cached baseline.
        
        OPTIMIZATION: Only runs 1 baseline + N ablated passes per prompt,
        instead of 2N passes (baseline for each head is cached).
        
        For 8 heads: 9 passes instead of 16 (44% savings).
        
        Returns:
            Dict mapping head_idx to WritingValidationResult
        """
        from head_ablation import HeadAblator
        from weight_extraction import extract_layer_weights
        from config import GEMMA2_CONFIG
        import sys
        from io import StringIO
        
        # Default to all 8 heads
        if head_indices is None:
            head_indices = list(range(8))
        
        sae_layer = GEMMA2_CONFIG.get_sae_layer_for_attn(layer_idx)
        
        # Use FULL SAE (all 16K features) for OV_f validation for accurate reconstruction
        sae_decoder, b_dec = self.sae_manager.get_decoder_with_bias(sae_layer)
        D = sae_decoder  # [d_model, n_features] - use ALL features
        b_dec = b_dec.to(self.device)  # [d_model]
        # Note: skip normalization for full decoder since we want accurate reconstruction
        
        # Get layer weights and compute OV circuits for all heads
        layer_weights = extract_layer_weights(
            self.model, layer_idx, GEMMA2_CONFIG,
            fold_gamma=False, device=str(self.device), dtype=torch.float32
        )
        
        OV_circuits = {}
        for head_idx in head_indices:
            kv_group = GEMMA2_CONFIG.query_to_kv_group(head_idx)
            W_V = layer_weights.W_V[kv_group].to(self.device)
            W_O = layer_weights.W_O[head_idx].to(self.device)
            OV_circuits[head_idx] = W_V.T @ W_O.T
        
        # Create ablators for all heads (suppress output)
        ablators = {}
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            for head_idx in head_indices:
                ablators[head_idx] = HeadAblator(self.model, layer_idx, head_idx)
        finally:
            sys.stdout = old_stdout
        
        # Accumulate results per head
        results_per_head = {h: {'cosine_sims': [], 'positions': 0} for h in head_indices}
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            seq_len = inputs['input_ids'].shape[1]
            
            # === CACHED BASELINE (run once per prompt) ===
            self.cache.clear()
            handles = create_attention_hooks(self.model, self.cache, [layer_idx])
            
            try:
                with torch.no_grad():
                    outputs_baseline = self.model(**inputs, output_attentions=True)
                
                residual_pre = self.cache.residual_pre_attn.get(layer_idx)
                if residual_pre is None:
                    continue
                residual_pre = residual_pre[0]
                
                attention_output_baseline = self.cache.attention_output.get(layer_idx)
                if attention_output_baseline is None:
                    continue
                attention_output_baseline = attention_output_baseline[0]
                
                # Get attention weights for all heads
                all_attn_weights = outputs_baseline.attentions[layer_idx][0]  # [n_heads, seq_len, seq_len]
                
            finally:
                for h in handles:
                    h.remove()
            
            # Encode features once (use ALL features for accurate reconstruction)
            sae_pre = self.sae_manager.encode(residual_pre, sae_layer)  # [seq_len, 16384]
            
            # === Process each head with cached baseline ===
            for head_idx in head_indices:
                # Ablate this head
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                try:
                    ablators[head_idx].ablate()
                finally:
                    sys.stdout = old_stdout
                
                self.cache.clear()
                handles = create_attention_hooks(self.model, self.cache, [layer_idx])
                
                try:
                    with torch.no_grad():
                        self.model(**inputs, output_attentions=False)
                    
                    attention_output_ablated = self.cache.attention_output.get(layer_idx)
                    if attention_output_ablated is None:
                        continue
                    attention_output_ablated = attention_output_ablated[0]
                    
                finally:
                    for h in handles:
                        h.remove()
                    # Restore head
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()
                    try:
                        ablators[head_idx].restore()
                    finally:
                        sys.stdout = old_stdout
                
                # Head-isolated contribution
                head_attention_output = attention_output_baseline - attention_output_ablated
                
                # Vectorized computation
                attn_weights = all_attn_weights[head_idx].float()  # Convert to float32
                aggregated_features = attn_weights @ sae_pre
                attended_residuals = aggregated_features @ D.T + b_dec  # Add decoder bias
                pred_writes = attended_residuals @ OV_circuits[head_idx]
                
                pred_writes = pred_writes[1:]
                actual_writes = head_attention_output[1:]
                
                pred_norms = torch.norm(pred_writes, dim=1, keepdim=True)
                actual_norms = torch.norm(actual_writes, dim=1, keepdim=True)
                valid_mask = (pred_norms.squeeze() > 1e-9) & (actual_norms.squeeze() > 1e-9)
                
                if valid_mask.any():
                    cos_sims = F.cosine_similarity(pred_writes[valid_mask], actual_writes[valid_mask], dim=1)
                    results_per_head[head_idx]['cosine_sims'].extend(cos_sims.cpu().tolist())
                
                results_per_head[head_idx]['positions'] += seq_len - 1
        
        # Build final results
        results = {}
        for head_idx in head_indices:
            data = results_per_head[head_idx]
            avg_cosine = np.mean(data['cosine_sims']) if data['cosine_sims'] else 0.0
            std_cosine = np.std(data['cosine_sims']) if data['cosine_sims'] else 0.0
            
            results[head_idx] = WritingValidationResult(
                layer_idx=layer_idx,
                head_idx=head_idx,
                jaccard_at_k=avg_cosine,
                ndcg_at_k=std_cosine,
                delta_pearson_r=None,
                delta_spearman_r=0.0,
                n_positions=data['positions'],
                top_k=self.config.top_k_features,
            )
        
        return results
    
    def validate_writing_for_head(
        self,
        prompts: List[str],
        layer_idx: int,
        head_idx: int,
    ) -> WritingValidationResult:
        """
        Validate OV writing (W2F) for a specific head.
        
        Compares predicted feature deltas (from W2F + attention weights)
        against actual feature deltas observed during forward pass.
        
        Predicted Δ_k at position t = Σ_s α(t,s) * (a[s] @ W2F[:, k])
        
        Args:
            prompts: List of text prompts
            layer_idx: Attention layer index
            head_idx: Head index
            
        Returns:
            WritingValidationResult with top-K overlap and correlation metrics
        """
        from config import GEMMA2_CONFIG
        
        # Get W2F matrix
        W2F = self.get_W2F_matrix(layer_idx, head_idx)
        sae_layer = GEMMA2_CONFIG.get_sae_layer_for_attn(layer_idx)
        feature_indices = self._get_feature_indices(sae_layer)
        
        all_jaccard = []
        all_topk_overlap = []
        all_rank_corr = []
        all_positions = 0
        
        for prompt in prompts:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            seq_len = inputs['input_ids'].shape[1]
            
            # Set up hooks
            self.cache.clear()
            handles = create_attention_hooks(self.model, self.cache, [layer_idx])
            
            try:
                with torch.no_grad():
                    outputs = self.model(**inputs, output_attentions=True)
                
                # Get pre-attention SAE activations
                residual_pre = self.cache.residual_pre_attn.get(layer_idx)
                if residual_pre is None:
                    continue
                residual_pre = residual_pre[0]  # [seq_len, d_model]
                
                # Get post-layer SAE activations (after attention + MLP + residual)
                residual_post_layer = getattr(self.cache, 'residual_post_layer', {}).get(layer_idx)
                if residual_post_layer is None:
                    # Fall back: use next layer's pre-attention if available
                    continue
                residual_post_layer = residual_post_layer[0]  # [seq_len, d_model]
                
                # Encode with SAE (full features, then subset)
                sae_pre_full = self.sae_manager.encode(residual_pre, sae_layer)
                sae_post_full = self.sae_manager.encode(residual_post_layer, sae_layer)
                
                # Subset to features used in W2F
                sae_pre = sae_pre_full[:, feature_indices]   # [seq_len, n_features]
                sae_post = sae_post_full[:, feature_indices] # [seq_len, n_features]
                
                # Actual deltas: how features changed through this layer
                actual_deltas = sae_post - sae_pre  # [seq_len, n_features]
                
                # Get attention weights for this head
                attn_weights = outputs.attentions[layer_idx][0, head_idx]  # [seq_len, seq_len]
                
                # For each query position, compute predicted vs actual
                for t in range(1, seq_len):  # Skip position 0 (no context)
                    # Attention-weighted source features
                    # α[t, :t+1] are the attention weights for query t
                    alpha = attn_weights[t, :t+1]  # [t+1] attending to positions 0..t
                    source_features = sae_pre[:t+1]  # [t+1, n_features]
                    
                    # Predicted delta: Σ_s α(t,s) * (a[s] @ W2F)
                    weighted_sources = alpha.unsqueeze(1) * source_features  # [t+1, n_features]
                    aggregated = weighted_sources.sum(dim=0)  # [n_features]
                    predicted_delta = aggregated @ W2F  # [n_features]
                    
                    # Actual delta at position t
                    actual_delta = actual_deltas[t]  # [n_features]
                    
                    # Top-K overlap (by absolute value)
                    k = self.config.top_k_features
                    _, pred_topk = torch.topk(predicted_delta.abs(), min(k, predicted_delta.shape[0]))
                    _, actual_topk = torch.topk(actual_delta.abs(), min(k, actual_delta.shape[0]))
                    
                    pred_set = set(pred_topk.cpu().numpy())
                    actual_set = set(actual_topk.cpu().numpy())
                    
                    intersection = len(pred_set & actual_set)
                    union = len(pred_set | actual_set)
                    jaccard = intersection / union if union > 0 else 0.0
                    topk_overlap = intersection / k
                    
                    all_jaccard.append(jaccard)
                    all_topk_overlap.append(topk_overlap)
                    
                    # Rank correlation of full delta vectors
                    pred_np = predicted_delta.cpu().float().numpy()
                    actual_np = actual_delta.cpu().float().numpy()
                    
                    if np.std(pred_np) > 1e-9 and np.std(actual_np) > 1e-9:
                        try:
                            r_s, _ = stats.spearmanr(pred_np, actual_np)
                            if not np.isnan(r_s):
                                all_rank_corr.append(r_s)
                        except:
                            pass
                    
                    all_positions += 1
                
            finally:
                for h in handles:
                    h.remove()
        
        # Aggregate results
        avg_jaccard = np.mean(all_jaccard) if all_jaccard else 0.0
        avg_topk_overlap = np.mean(all_topk_overlap) if all_topk_overlap else 0.0
        avg_rank_corr = np.mean(all_rank_corr) if all_rank_corr else 0.0
        
        return WritingValidationResult(
            layer_idx=layer_idx,
            head_idx=head_idx,
            jaccard_at_k=avg_jaccard,
            ndcg_at_k=avg_topk_overlap,  # Reusing this field for top-k overlap
            delta_pearson_r=None,  # Not computed (expensive)
            delta_spearman_r=avg_rank_corr,
            n_positions=all_positions,
            top_k=self.config.top_k_features,
        )
    
    def validate_writing_for_head_ablation(
        self,
        prompts: List[str],
        layer_idx: int,
        head_idx: int,
    ) -> WritingValidationResult:
        """
        Validate W2F using ablation-isolated ground truth.
        
        This method uses two forward passes per prompt:
        1. Baseline: normal forward pass
        2. Ablated: forward pass with target head zeroed
        
        The head-isolated delta = sae(baseline) - sae(ablated)
        This captures exactly what THIS head contributed.
        
        This is slower (2x forward passes) but gives much cleaner ground truth
        than comparing against full-layer deltas (which include all heads + MLP).
        """
        from head_ablation import HeadAblator
        from config import GEMMA2_CONFIG
        
        # Get W2F matrix
        W2F = self.get_W2F_matrix(layer_idx, head_idx)
        sae_layer = GEMMA2_CONFIG.get_sae_layer_for_attn(layer_idx)
        feature_indices = self._get_feature_indices(sae_layer)
        
        all_jaccard = []
        all_topk_overlap = []
        all_rank_corr = []
        all_positions = 0
        
        # Create ablator (we'll use it per prompt, restoring each time)
        # Note: HeadAblator prints on init, suppress for cleaner output
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            ablator = HeadAblator(self.model, layer_idx, head_idx)
        finally:
            sys.stdout = old_stdout
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            seq_len = inputs['input_ids'].shape[1]
            
            # === Pass 1: Baseline (no ablation) ===
            self.cache.clear()
            handles = create_attention_hooks(self.model, self.cache, [layer_idx])
            
            try:
                with torch.no_grad():
                    outputs_baseline = self.model(**inputs, output_attentions=True)
                
                # Get pre-attention residual (same for both passes)
                residual_pre = self.cache.residual_pre_attn.get(layer_idx)
                if residual_pre is None:
                    continue
                residual_pre = residual_pre[0]  # [seq_len, d_model]
                
                # Get post-layer residual (baseline)
                residual_post_baseline = getattr(self.cache, 'residual_post_layer', {}).get(layer_idx)
                if residual_post_baseline is None:
                    continue
                residual_post_baseline = residual_post_baseline[0]  # [seq_len, d_model]
                
                # Get attention weights for W2F prediction
                attn_weights = outputs_baseline.attentions[layer_idx][0, head_idx]  # [seq_len, seq_len]
                
            finally:
                for h in handles:
                    h.remove()
            
            # === Pass 2: With head ablated ===
            # Suppress ablator print output
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                ablator.ablate()
            finally:
                sys.stdout = old_stdout
            
            self.cache.clear()
            handles = create_attention_hooks(self.model, self.cache, [layer_idx])
            
            try:
                with torch.no_grad():
                    outputs_ablated = self.model(**inputs, output_attentions=False)
                
                # Get post-layer residual (ablated)
                residual_post_ablated = getattr(self.cache, 'residual_post_layer', {}).get(layer_idx)
                if residual_post_ablated is None:
                    # Restore and skip
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()
                    try:
                        ablator.restore()
                    finally:
                        sys.stdout = old_stdout
                    continue
                residual_post_ablated = residual_post_ablated[0]  # [seq_len, d_model]
                
            finally:
                for h in handles:
                    h.remove()
                # Restore head
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                try:
                    ablator.restore()
                finally:
                    sys.stdout = old_stdout
            
            # === Compute head-isolated delta ===
            # The difference between baseline and ablated is exactly what this head contributed
            head_contribution = residual_post_baseline - residual_post_ablated  # [seq_len, d_model]
            
            # Encode with SAE
            sae_pre_full = self.sae_manager.encode(residual_pre, sae_layer)
            
            # For ground truth: what features changed due to THIS head
            # We compute SAE(pre + head_contribution) - SAE(pre)
            # This approximates the actual feature delta from this head's write
            sae_with_head = self.sae_manager.encode(residual_pre + head_contribution, sae_layer)
            sae_without_head = sae_pre_full
            
            # Subset to features used in W2F
            sae_with = sae_with_head[:, feature_indices]
            sae_without = sae_without_head[:, feature_indices]
            sae_pre = sae_pre_full[:, feature_indices]
            
            # Actual head-isolated deltas
            actual_head_deltas = sae_with - sae_without  # [seq_len, n_features]
            
            # Compare predictions to head-isolated ground truth
            for t in range(1, seq_len):
                # Attention-weighted source features
                alpha = attn_weights[t, :t+1]
                source_features = sae_pre[:t+1]
                
                # Predicted delta: Σ_s α(t,s) * (a[s] @ W2F)
                weighted_sources = alpha.unsqueeze(1) * source_features
                aggregated = weighted_sources.sum(dim=0)
                predicted_delta = aggregated @ W2F
                
                # Actual delta from this head
                actual_delta = actual_head_deltas[t]
                
                # Top-K overlap
                k = self.config.top_k_features
                _, pred_topk = torch.topk(predicted_delta.abs(), min(k, predicted_delta.shape[0]))
                _, actual_topk = torch.topk(actual_delta.abs(), min(k, actual_delta.shape[0]))
                
                pred_set = set(pred_topk.cpu().numpy())
                actual_set = set(actual_topk.cpu().numpy())
                
                intersection = len(pred_set & actual_set)
                union = len(pred_set | actual_set)
                jaccard = intersection / union if union > 0 else 0.0
                topk_overlap = intersection / k
                
                all_jaccard.append(jaccard)
                all_topk_overlap.append(topk_overlap)
                
                # Rank correlation
                pred_np = predicted_delta.cpu().float().numpy()
                actual_np = actual_delta.cpu().float().numpy()
                
                if np.std(pred_np) > 1e-9 and np.std(actual_np) > 1e-9:
                    try:
                        r_s, _ = stats.spearmanr(pred_np, actual_np)
                        if not np.isnan(r_s):
                            all_rank_corr.append(r_s)
                    except:
                        pass
                
                all_positions += 1
        
        # Aggregate results
        avg_jaccard = np.mean(all_jaccard) if all_jaccard else 0.0
        avg_topk_overlap = np.mean(all_topk_overlap) if all_topk_overlap else 0.0
        avg_rank_corr = np.mean(all_rank_corr) if all_rank_corr else 0.0
        
        return WritingValidationResult(
            layer_idx=layer_idx,
            head_idx=head_idx,
            jaccard_at_k=avg_jaccard,
            ndcg_at_k=avg_topk_overlap,
            delta_pearson_r=None,  # Not computed
            delta_spearman_r=avg_rank_corr,
            n_positions=all_positions,
            top_k=self.config.top_k_features,
        )
    
    def generate_report(self, results: Dict) -> str:
        """Generate human-readable validation report with full diagnostics."""
        lines = []
        lines.append("=" * 80)
        lines.append("ACTIVATION VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Configuration info
        lines.append("[CONFIGURATION]")
        lines.append(f"  SAE features for QK routing: {self.qk_features} / {self._sae_width}")
        lines.append(f"  SAE features for OV_f:       {self.ov_features} / {self._sae_width} (full)")
        lines.append("")
        
        # Routing validation results
        if "routing" in results:
            lines.append("QK ROUTING VALIDATION (Does B predict attention?)")
            lines.append("-" * 60)
            for key, result in results["routing"].items():
                lines.append(f"\n{'='*60}")
                lines.append(f"Layer {result.layer_idx}, Head {result.head_idx}")
                lines.append("=" * 60)
                
                # === Summary metrics ===
                lines.append("\n[SUMMARY]")
                lines.append(f"  Overall Pearson r:  {result.pearson_r:.4f}")
                lines.append(f"  Overall Spearman r: {result.spearman_r:.4f}")
                lines.append(f"  Total pairs: {result.n_total_pairs:,}")
                
                # === Named distance-specific Spearman ===
                lines.append("\n[DISTANCE-SPECIFIC SPEARMAN]")
                lines.append(f"  Local  (0-4):    {result.spearman_local:.4f}")
                lines.append(f"  Mid    (16-32):  {result.spearman_mid:.4f}")
                lines.append(f"  Long   (128-256): {result.spearman_long:.4f}")
                lines.append(f"  256+   (256+):    {result.spearman_256plus:.4f}")
                
                # === Sign stability ===
                lines.append("\n[SIGN STABILITY]")
                lines.append(f"  Sign stability: {result.sign_stability:.2%}")
                if result.sign_by_bin:
                    signs_str = ", ".join(f"{k}:{'+' if v>0 else ('-' if v<0 else '0')}" 
                                          for k, v in sorted(result.sign_by_bin.items(), 
                                                             key=lambda x: int(x[0].split('-')[0])))
                    lines.append(f"  Signs by bin: {signs_str}")
                
                # === Skipped rows/bins ===
                lines.append("\n[SKIPPED ROWS/BINS]")
                lines.append(f"  Rows skipped: {result.n_rows_skipped}/{result.n_rows_total} ({result.skipped_fraction:.2%})")
                if result.bins_with_no_data:
                    lines.append(f"  Bins with no data: {', '.join(result.bins_with_no_data)}")
                else:
                    lines.append("  Bins with no data: (none)")
                
                # === Per-bin statistics ===
                lines.append("\n[PER-BIN STATISTICS]")
                lines.append(f"  {'Bin':>10} | {'r_p':>7} | {'r_s':>7} | {'n':>7} | {'pred_μ':>10} | {'pred_σ':>8} | {'act_μ':>10} | {'act_σ':>8}")
                lines.append("  " + "-" * 90)
                
                if result.pearson_by_distance:
                    for bin_name in sorted(result.pearson_by_distance.keys(), key=lambda x: int(x.split('-')[0])):
                        r_p = result.pearson_by_distance.get(bin_name, 0.0)
                        r_s = result.spearman_by_distance.get(bin_name, 0.0)
                        n = result.n_pairs_by_distance.get(bin_name, 0)
                        pred_mean = result.predicted_mean_by_bin.get(bin_name, 0.0)
                        pred_std = result.predicted_std_by_bin.get(bin_name, 0.0)
                        act_mean = result.actual_mean_by_bin.get(bin_name, 0.0)
                        act_std = result.actual_std_by_bin.get(bin_name, 0.0)
                        lines.append(f"  {bin_name:>10} | {r_p:>7.4f} | {r_s:>7.4f} | {n:>7,} | {pred_mean:>10.4f} | {pred_std:>8.4f} | {act_mean:>10.4f} | {act_std:>8.4f}")
                
            lines.append("")
        
        # OV_f validation results
        if "ov_f" in results:
            lines.append("OV_f VALIDATION (Does OV_circuit predict attention output?)")
            lines.append("-" * 60)
            lines.append("(Comparing predicted vs actual write vectors in residual space)")
            lines.append("")
            for key, result in results["ov_f"].items():
                # Compute 95% confidence interval
                mean_cos = result.jaccard_at_k
                std_cos = result.ndcg_at_k
                n = result.n_positions
                if n > 0 and std_cos > 0:
                    se = std_cos / np.sqrt(n)
                    ci_low = mean_cos - 1.96 * se
                    ci_high = mean_cos + 1.96 * se
                    ci_str = f"[{ci_low:.4f}, {ci_high:.4f}]"
                else:
                    ci_str = "[N/A]"
                
                lines.append(f"Layer {result.layer_idx}, Head {result.head_idx}:")
                lines.append(f"  Cosine similarity: {mean_cos:.4f}  95% CI: {ci_str}")
                lines.append(f"  Std deviation:     {std_cos:.4f}")
                lines.append(f"  Positions:         {n:,}")
            lines.append("")
        
        # Writing validation results (W2F)
        if "writing" in results:
            lines.append("OV WRITING VALIDATION (Does W2F predict feature changes?)")
            lines.append("-" * 60)
            for key, result in results["writing"].items():
                lines.append(f"\nLayer {result.layer_idx}, Head {result.head_idx}:")
                lines.append(f"  Jaccard@{result.top_k}: {result.jaccard_at_k:.4f}")
                pearson_str = f"{result.delta_pearson_r:.4f}" if result.delta_pearson_r is not None else "[not computed]"
                lines.append(f"  Delta Pearson r:  {pearson_str}")
                lines.append(f"  Delta Spearman r: {result.delta_spearman_r:.4f}")
            lines.append("")
        
        lines.append("=" * 80)
        return "\n".join(lines)


# =============================================================================
# Test Prompts
# =============================================================================

def load_validation_prompts(csv_path: str = "validation_prompts.csv") -> List[str]:
    """
    Load validation prompts from a CSV file.
    
    Expected CSV columns: category, topic, description, content
    """
    prompts = []
    path = Path(csv_path)
    if not path.is_absolute():
        # Try to find it in the same directory as this script
        script_dir = Path(__file__).parent
        path = script_dir / csv_path
        
    if not path.exists():
        print(f"Warning: Prompts file not found at {path}")
        return []
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'content' in row:
                    prompts.append(row['content'])
        print(f"Loaded {len(prompts)} prompts from {path}")
    except Exception as e:
        print(f"Error loading prompts: {e}")
        
    return prompts



# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate weight-space analysis against activations")
    parser.add_argument("analysis_json", help="Path to analysis JSON file")
    parser.add_argument("--layer", type=int, help="Specific layer to validate")
    parser.add_argument("--head", type=int, help="Specific head to validate")
    parser.add_argument("--mode", choices=["routing", "writing", "both"], default="routing")
    parser.add_argument("-o", "--output", help="Output report file")
    
    args = parser.parse_args()
    
    print("Loading analysis results...")
    with open(args.analysis_json) as f:
        analysis = json.load(f)
    
    print("This script requires model and SAE to be loaded.")
    print("Run with:")
    print("  from activation_validation import ActivationValidator, load_validation_prompts")
    print("  validator = ActivationValidator(model, tokenizer, sae_manager, analysis)")
    print("  prompts = load_validation_prompts()")
    print("  result = validator.validate_routing_for_head(prompts[:5], layer=6, head=3)")


if __name__ == "__main__":
    main()
