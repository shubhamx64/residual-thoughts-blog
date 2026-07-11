"""
OV Writing Semantics Analysis for Gemma-2 attention heads.

Phase 2 implementation:
- Compute W_OV = W_O @ W_V (correct ordering!)
- Write vectors for each feature
- Project writes back to feature space
- Write metrics: copy, transform, broadcast, suppression
- Post-attention RMSNorm awareness (directional effects)
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from config import Gemma2Config, GEMMA2_CONFIG
from weight_extraction import LayerWeights, compute_ov_matrix
from sae_utils import SAEFeatures


class WriteArchetype(Enum):
    """Classification of head write behavior."""
    COPY = "copy"              # Mostly j → j
    TRANSFORM = "transform"    # Systematic j → k
    BROADCAST = "broadcast"    # Many j → same k
    SUPPRESS = "suppress"      # Strong negative writes
    DIFFUSE = "diffuse"        # No clear pattern


@dataclass
class WriteMetrics:
    """Metrics characterizing a head's write behavior."""
    
    # Copy score: degree to which j writes back to j
    copy_score: float          # mean of diagonal in write-to-feature matrix
    copy_dominance: float      # fraction where argmax write == self
    
    # Transform score: systematic off-diagonal mappings
    transform_score: float     # max off-diagonal in mean write-to-feature
    top_transform_pairs: List[Tuple[int, int, float]]  # top j→k≠j mappings
    
    # Broadcast: do many j write to same k?
    broadcast_score: float     # max column sum (how many features write to k)
    top_broadcast_targets: List[Tuple[int, float]]  # features receiving most writes
    
    # Suppression: strong negative components
    suppression_score: float   # magnitude of most negative write
    top_suppression_targets: List[Tuple[int, int, float]]  # j→−k pairs
    
    # Contribution magnitude (pre-norm capacity)
    write_norm_mean: float     # mean ||w_j||
    write_norm_std: float
    write_norm_max: float
    
    # Directional diversity
    write_cosine_diversity: float  # mean pairwise cosine distance of write vectors
    
    # Classification
    archetype: WriteArchetype = WriteArchetype.DIFFUSE


@dataclass
class HeadWriteResult:
    """Complete write analysis for a single head."""
    layer_idx: int
    query_head: int
    kv_group: int
    
    metrics: WriteMetrics
    
    # The write-to-feature matrix: [n_features, n_features]
    # W2F[j, k] = similarity of write_j to decoder_k
    write_to_feature_matrix: Optional[torch.Tensor] = None


def compute_write_vectors_fast(
    decoder: torch.Tensor,
    W_V: torch.Tensor,
    W_O: torch.Tensor,
) -> torch.Tensor:
    """
    Compute write vectors using row-vector convention (efficient, correct).
    
    Row-vector forward pass for OV:
      1. key_residual @ W_V.T -> [n, head_dim] (project into V-space)
      2. (attend & mix) -> same shape
      3. @ W_O.T -> [n, hidden_size] (project back to residual stream)
    
    For write vectors (what gets written when attending to feature j):
      write_j = decoder_j @ W_V.T @ W_O.T
    
    This is both correct AND avoids the O(d_model²) W_OV materialization.
    
    Args:
        decoder: [n_features, hidden_size] decoder directions (RMS-calibrated)
        W_V: [head_dim, hidden_size] value projection for this KV group
        W_O: [hidden_size, head_dim] output projection for this query head
        
    Returns:
        write_vectors: [n_features, hidden_size]
    """
    # Step 1: project decoder into V-space: [n_feat, hidden] @ [hidden, head_dim]
    v_features = decoder @ W_V.T  # [n_features, head_dim]
    
    # Step 2: project back to residual stream: [n_feat, head_dim] @ [head_dim, hidden]
    write_vectors = v_features @ W_O.T  # [n_features, hidden_size]
    
    return write_vectors


def project_writes_to_features(
    write_vectors: torch.Tensor,
    decoder: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Project write vectors to feature space.
    
    W2F[j, k] = cosine_similarity(write_j, decoder_k)
    
    Args:
        write_vectors: [n_features, hidden_size]
        decoder: [n_features, hidden_size]
        normalize: Use cosine similarity (recommended for directional analysis)
        
    Returns:
        W2F: [n_features, n_features] write-to-feature mapping
    """
    if normalize:
        w_norm = F.normalize(write_vectors.float(), dim=1)
        d_norm = F.normalize(decoder.float(), dim=1)
        return w_norm @ d_norm.T
    else:
        return write_vectors @ decoder.T


def compute_write_metrics(
    write_vectors: torch.Tensor,
    W2F: torch.Tensor,
    feature_indices: torch.Tensor,
    top_k: int = 20,
) -> WriteMetrics:
    """
    Compute write metrics from write vectors and write-to-feature matrix.
    
    Args:
        write_vectors: [n, hidden_size] write vectors
        W2F: [n, n] write-to-feature similarity matrix
        feature_indices: [n] global feature indices
        top_k: Number of top pairs to extract
        
    Returns:
        WriteMetrics dataclass
    """
    n = W2F.shape[0]
    W2F_float = W2F.float()
    
    # Copy score: diagonal
    diag = torch.diag(W2F_float)
    copy_score = diag.mean().item()
    
    argmax_per_row = W2F_float.argmax(dim=1)
    diagonal_indices = torch.arange(n, device=W2F.device)
    copy_dominance = (argmax_per_row == diagonal_indices).float().mean().item()
    
    # Transform score: off-diagonal
    mask = ~torch.eye(n, dtype=torch.bool, device=W2F.device)
    off_diag = W2F_float[mask]
    transform_score = off_diag.max().item() if off_diag.numel() > 0 else 0.0
    
    # Top transform pairs (j → k where k ≠ j)
    W2F_no_diag = W2F_float.clone()
    W2F_no_diag.fill_diagonal_(-float('inf'))
    flat = W2F_no_diag.flatten()
    top_vals, top_idx = flat.topk(min(top_k, flat.numel()))
    
    top_transform_pairs = []
    for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
        if val == -float('inf'):
            continue
        j, k = idx // n, idx % n
        fj = feature_indices[j].item()
        fk = feature_indices[k].item()
        top_transform_pairs.append((int(fj), int(fk), float(val)))
    
    # Broadcast: column sums (how much each k receives)
    col_sums = W2F_float.sum(dim=0)
    broadcast_score = col_sums.max().item()
    
    top_k_broadcast = min(10, n)
    top_recv, top_k_idx = col_sums.topk(top_k_broadcast)
    top_broadcast_targets = [
        (int(feature_indices[k].item()), float(v))
        for k, v in zip(top_k_idx.tolist(), top_recv.tolist())
    ]
    
    # Suppression: negative writes (use +inf diagonal to exclude from minimums)
    suppression_score = abs(off_diag.min().item()) if off_diag.numel() > 0 else 0.0
    
    # Create matrix with +inf diagonal for finding minimums
    W2F_for_supp = W2F_float.clone()
    W2F_for_supp.fill_diagonal_(float('inf'))
    flat_supp = W2F_for_supp.flatten()
    bot_vals, bot_idx = flat_supp.topk(min(top_k, flat_supp.numel()), largest=False)
    
    top_suppression_targets = []
    for val, idx in zip(bot_vals.tolist(), bot_idx.tolist()):
        # Skip infinite values (diagonal entries)
        if not math.isfinite(val):
            continue
        j, k = idx // n, idx % n
        fj = feature_indices[j].item()
        fk = feature_indices[k].item()
        top_suppression_targets.append((int(fj), int(fk), float(val)))
    
    # Write norms
    write_norms = write_vectors.float().norm(dim=1)
    write_norm_mean = write_norms.mean().item()
    write_norm_std = write_norms.std().item()
    write_norm_max = write_norms.max().item()
    
    # Directional diversity: mean pairwise cosine distance
    w_norm = F.normalize(write_vectors.float(), dim=1)
    cos_sim_matrix = w_norm @ w_norm.T
    # Exclude diagonal and get mean
    mask_off_diag = ~torch.eye(n, dtype=torch.bool, device=W2F.device)
    cos_off_diag = cos_sim_matrix[mask_off_diag]
    write_cosine_diversity = 1.0 - cos_off_diag.mean().item()
    
    metrics = WriteMetrics(
        copy_score=copy_score,
        copy_dominance=copy_dominance,
        transform_score=transform_score,
        top_transform_pairs=top_transform_pairs,
        broadcast_score=broadcast_score,
        top_broadcast_targets=top_broadcast_targets,
        suppression_score=suppression_score,
        top_suppression_targets=top_suppression_targets,
        write_norm_mean=write_norm_mean,
        write_norm_std=write_norm_std,
        write_norm_max=write_norm_max,
        write_cosine_diversity=write_cosine_diversity,
    )
    
    metrics.archetype = classify_write_archetype(metrics)
    return metrics


def classify_write_archetype(metrics: WriteMetrics) -> WriteArchetype:
    """
    Classify head write behavior into archetype.
    """
    # High copy -> COPY
    if metrics.copy_dominance > 0.3 and metrics.copy_score > 0.3:
        return WriteArchetype.COPY
    
    # High transform -> TRANSFORM
    if metrics.transform_score > 0.5 and len(metrics.top_transform_pairs) > 0:
        return WriteArchetype.TRANSFORM
    
    # High broadcast -> BROADCAST
    if metrics.broadcast_score > metrics.copy_score * 3:
        return WriteArchetype.BROADCAST
    
    # Strong suppression -> SUPPRESS
    if metrics.suppression_score > 0.5:
        return WriteArchetype.SUPPRESS
    
    return WriteArchetype.DIFFUSE


def analyze_head_writing(
    sae_features: SAEFeatures,
    layer_weights: LayerWeights,
    query_head: int,
    config: Gemma2Config = GEMMA2_CONFIG,
    store_matrix: bool = False,
    device: str = None,
    dtype: torch.dtype = None,
) -> HeadWriteResult:
    """
    Complete write analysis for a single head.
    
    Args:
        sae_features: Prepared SAE features
        layer_weights: Extracted layer weights
        query_head: Query head index
        config: Model config
        store_matrix: Whether to store full W2F matrix
        device: Target device (defaults to decoder's device)
        dtype: Target dtype (defaults to decoder's dtype)
        
    Returns:
        HeadWriteResult with metrics
    """
    decoder = sae_features.decoder_subset
    feature_indices = sae_features.feature_indices
    
    # Use decoder's device/dtype as default
    if device is None:
        device = decoder.device
    if dtype is None:
        dtype = decoder.dtype
    
    # Get W_V and W_O directly (avoid materializing W_OV = O(d_model²))
    kv_group = config.query_to_kv_group(query_head)
    W_V = layer_weights.W_V[kv_group].to(device=device, dtype=dtype)
    W_O = layer_weights.W_O[query_head].to(device=device, dtype=dtype)
    
    # Compute write vectors using efficient row-vector convention
    # write_j = decoder_j @ W_V.T @ W_O.T (correct and fast)
    write_vectors = compute_write_vectors_fast(decoder, W_V, W_O)
    
    # Project to feature space (normalized for directional analysis)
    W2F = project_writes_to_features(write_vectors, decoder, normalize=True)
    
    # Compute metrics
    metrics = compute_write_metrics(write_vectors, W2F, feature_indices)
    
    # kv_group already computed above    
    return HeadWriteResult(
        layer_idx=layer_weights.layer_idx,
        query_head=query_head,
        kv_group=kv_group,
        metrics=metrics,
        write_to_feature_matrix=W2F if store_matrix else None,
    )


def find_write_collisions(
    results: List[HeadWriteResult],
    threshold: float = 0.5,
) -> Dict[int, List[Tuple[int, int, float]]]:
    """
    Find features that receive writes from multiple heads.
    
    This is potential superposition in the write output.
    
    Returns: Dict mapping target_feature -> list of (head_idx, source_feature, strength)
    """
    # Aggregate across heads
    target_writers: Dict[int, List[Tuple[int, int, float]]] = {}
    
    for result in results:
        for j, k, score in result.metrics.top_transform_pairs:
            if abs(score) > threshold:
                if k not in target_writers:
                    target_writers[k] = []
                target_writers[k].append((result.query_head, j, score))
    
    # Filter to those with multiple writers
    collisions = {k: v for k, v in target_writers.items() if len(v) > 1}
    return collisions


if __name__ == "__main__":
    print("OV Writing Analysis module loaded.")
    print(f"Write archetypes: {[a.value for a in WriteArchetype]}")
