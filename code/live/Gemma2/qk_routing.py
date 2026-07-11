"""
QK Routing Analysis for Gemma-2 attention heads.

Core Phase 1 implementation:
- Content-only affinity matrix B computation
- Per-head metrics: diagonal dominance, selectivity, asymmetry, spectral structure
- Baselines: random D, random weights, permutation
- Superposition correction via Gram matrix
- Head archetype classification
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
import math

from config import Gemma2Config, AnalysisConfig, GEMMA2_CONFIG, ANALYSIS_CONFIG
from sae_utils import SAEFeatures, project_to_qk_space


class HeadArchetype(Enum):
    """Classification of head routing behavior."""
    SELF_MATCH = "self_match"          # High diagonal dominance
    SELECTIVE_CROSS = "selective_cross" # High selectivity, low diagonal
    DIFFUSE = "diffuse"                # Low selectivity everywhere
    REPULSION = "repulsion"            # Strong negative affinities
    MIXED = "mixed"                    # No clear pattern


@dataclass
class RoutingMetrics:
    """Metrics characterizing a head's routing behavior."""
    
    # Diagonal dominance: how often does i route most strongly to i?
    diagonal_dominance: float  # fraction where argmax(B[i,:]) == i
    diagonal_mean: float       # mean of B[i,i]
    diagonal_std: float        # std of B[i,i]
    
    # Row selectivity: how concentrated are scores over keys?
    row_entropy_mean: float    # mean entropy of softmax(B[i,:])
    row_entropy_std: float
    top1_mass_mean: float      # mean of max(softmax(B[i,:]))
    top5_mass_mean: float      # mean of sum of top-5 softmax masses
    max_gap_mean: float        # mean of (top1 - top2) scores
    max_gap_max: float         # max of (top1 - top2) scores
    
    # Diagonal self-mass: how much probability goes to self?
    diagonal_softmax_mass: float  # mean of softmax(B[i,:])[i] - identity-sensitive!
    
    # Asymmetry: is B[i,j] different from B[j,i]?
    asymmetry_score: float     # Frobenius norm of (B - B.T) / Frobenius(B)
    
    # Spectral structure
    effective_rank: float      # exp(entropy of normalized singular values)
    top_singular_ratio: float  # s1 / sum(s)
    
    # Overall statistics
    mean_affinity: float
    std_affinity: float
    max_affinity: float
    min_affinity: float
    
    # Classification
    archetype: HeadArchetype = HeadArchetype.MIXED


@dataclass
class TopPairs:
    """Top routing pairs for a head."""
    # Each tuple: (query_feature_idx, key_feature_idx, score)
    positive_pairs: List[Tuple[int, int, float]] = field(default_factory=list)
    negative_pairs: List[Tuple[int, int, float]] = field(default_factory=list)


@dataclass
class HeadRoutingResult:
    """Complete routing analysis for a single head."""
    layer_idx: int
    query_head: int
    kv_group: int
    
    metrics: RoutingMetrics
    top_pairs: TopPairs
    
    # Baseline comparisons (if computed)
    baseline_random_d: Optional[RoutingMetrics] = None
    baseline_random_weights: Optional[RoutingMetrics] = None
    baseline_independent_bases: Optional[RoutingMetrics] = None  # Independent Q/K bases
    baseline_permuted_k: Optional[RoutingMetrics] = None  # Permuted-K with real decoder
    
    # Superposition-corrected metrics (if computed)
    metrics_gram_corrected: Optional[RoutingMetrics] = None


def apply_softcap(logits: torch.Tensor, cap: float) -> torch.Tensor:
    """
    Apply tanh soft-capping to logits.
    
    Formula from Gemma-2: logits = cap * tanh(logits / cap)
    This prevents logits from exceeding ±cap.
    """
    return cap * torch.tanh(logits / cap)


def compute_affinity_matrix(
    Q_f: torch.Tensor,
    K_f: torch.Tensor,
    config: Gemma2Config = GEMMA2_CONFIG,
) -> torch.Tensor:
    """
    Compute content-only affinity matrix B.
    
    Supports two modes (controlled by config.qk_mode):
    - "logit": B = (Q_f @ K_f.T) * attention_scale, optionally softcapped
      This matches actual attention logits and gives meaningful selectivity.
    - "cosine": B = cosine_sim(Q_f, K_f) * semantic_temperature
      This measures directional similarity in Q/K space.
    
    IMPORTANT: The old approach of normalizing Q/K then applying attention_scale
    compressed logits to [-0.06, 0.06] making softmax uniform. Fixed here.
    
    Args:
        Q_f: [n_features, head_dim] query features in Q-space
        K_f: [n_features, head_dim] key features in K-space
        config: Model configuration with qk_mode, scales, softcap settings
        
    Returns:
        B: [n_features, n_features] affinity matrix
    """
    Q_f = Q_f.float()
    K_f = K_f.float()
    
    if config.qk_mode == "cosine":
        # Cosine similarity mode: normalize then apply semantic temperature
        Q_norm = F.normalize(Q_f, dim=1)
        K_norm = F.normalize(K_f, dim=1)
        B = (Q_norm @ K_norm.T) * config.semantic_temperature
    else:
        # Logit mode (default): no normalization, real attention scaling
        B = (Q_f @ K_f.T) * config.attention_scale
        
        # Apply soft-capping if enabled
        if config.apply_softcap and config.attn_logit_softcapping is not None:
            B = apply_softcap(B, config.attn_logit_softcapping)
    
    return B


def compute_routing_metrics(
    B: torch.Tensor,
    feature_indices: torch.Tensor,
) -> RoutingMetrics:
    """
    Compute all routing metrics from affinity matrix B.
    
    Args:
        B: [n, n] affinity matrix
        feature_indices: [n] global feature indices
        
    Returns:
        RoutingMetrics dataclass
    """
    n = B.shape[0]
    B_float = B.float()
    
    # Diagonal dominance
    argmax_per_row = B_float.argmax(dim=1)
    diagonal_indices = torch.arange(n, device=B.device)
    diagonal_dominance = (argmax_per_row == diagonal_indices).float().mean().item()
    
    diag = torch.diag(B_float)
    diagonal_mean = diag.mean().item()
    diagonal_std = diag.std().item()
    
    # Row selectivity via softmax
    B_softmax = F.softmax(B_float, dim=1)
    
    # Entropy of each row
    eps = 1e-10
    row_entropy = -(B_softmax * (B_softmax + eps).log()).sum(dim=1)
    row_entropy_mean = row_entropy.mean().item()
    row_entropy_std = row_entropy.std().item()
    
    # Top-k mass
    top1_vals, _ = B_softmax.max(dim=1)
    top1_mass_mean = top1_vals.mean().item()
    
    top5_vals, _ = B_softmax.topk(min(5, n), dim=1)
    top5_mass_mean = top5_vals.sum(dim=1).mean().item()
    
    # Max-gap (in raw scores, not softmax)
    sorted_B, _ = B_float.sort(dim=1, descending=True)
    max_gap = sorted_B[:, 0] - sorted_B[:, 1]
    max_gap_mean = max_gap.mean().item()
    max_gap_max = max_gap.max().item()
    
    # Diagonal softmax mass: how much probability goes to self?
    # This is identity-sensitive (unlike top1_mass which is permutation-invariant)
    diagonal_softmax_mass = torch.diag(B_softmax).mean().item()
    
    # Asymmetry
    B_asym = B_float - B_float.T
    frobenius_asym = torch.norm(B_asym, p='fro')
    frobenius_B = torch.norm(B_float, p='fro')
    asymmetry_score = (frobenius_asym / (frobenius_B + eps)).item()
    
    # Spectral structure
    try:
        U, S, V = torch.linalg.svd(B_float, full_matrices=False)
        S_normalized = S / (S.sum() + eps)
        S_entropy = -(S_normalized * (S_normalized + eps).log()).sum()
        effective_rank = math.exp(S_entropy.item())
        top_singular_ratio = (S[0] / (S.sum() + eps)).item()
    except Exception:
        effective_rank = n
        top_singular_ratio = 1.0 / n
    
    # Overall stats
    mean_affinity = B_float.mean().item()
    std_affinity = B_float.std().item()
    max_affinity = B_float.max().item()
    min_affinity = B_float.min().item()
    
    metrics = RoutingMetrics(
        diagonal_dominance=diagonal_dominance,
        diagonal_mean=diagonal_mean,
        diagonal_std=diagonal_std,
        row_entropy_mean=row_entropy_mean,
        row_entropy_std=row_entropy_std,
        top1_mass_mean=top1_mass_mean,
        top5_mass_mean=top5_mass_mean,
        max_gap_mean=max_gap_mean,
        max_gap_max=max_gap_max,
        diagonal_softmax_mass=diagonal_softmax_mass,
        asymmetry_score=asymmetry_score,
        effective_rank=effective_rank,
        top_singular_ratio=top_singular_ratio,
        mean_affinity=mean_affinity,
        std_affinity=std_affinity,
        max_affinity=max_affinity,
        min_affinity=min_affinity,
    )
    
    # Classify archetype
    metrics.archetype = classify_archetype(metrics)
    
    return metrics


def classify_archetype(metrics: RoutingMetrics) -> HeadArchetype:
    """
    Classify head into an archetype based on metrics.
    
    Thresholds are heuristic and should be calibrated on real data.
    """
    # High diagonal dominance -> SELF_MATCH
    if metrics.diagonal_dominance > 0.3 and metrics.diagonal_mean > 0.5:
        return HeadArchetype.SELF_MATCH
    
    # High selectivity but low diagonal -> SELECTIVE_CROSS
    if metrics.top1_mass_mean > 0.15 and metrics.diagonal_dominance < 0.1:
        return HeadArchetype.SELECTIVE_CROSS
    
    # Very diffuse (high entropy, low top-k mass) -> DIFFUSE
    if metrics.top5_mass_mean < 0.1:
        return HeadArchetype.DIFFUSE
    
    # Strong negative affinities -> REPULSION
    if metrics.min_affinity < -0.5:
        return HeadArchetype.REPULSION
    
    return HeadArchetype.MIXED


def extract_top_pairs(
    B: torch.Tensor,
    feature_indices: torch.Tensor,
    k: int = 50,
) -> TopPairs:
    """
    Extract top positive and negative routing pairs.
    
    Args:
        B: [n, n] affinity matrix
        feature_indices: [n] global feature indices
        k: Number of top pairs to extract
        
    Returns:
        TopPairs with positive and negative pairs
    """
    n = B.shape[0]
    B_flat = B.flatten()
    
    # Ensure feature_indices is on CPU for indexing (we convert to Python ints anyway)
    feature_indices_cpu = feature_indices.cpu()
    
    # Top positive
    k_pos = min(k, B_flat.numel())
    top_vals, top_idx = B_flat.topk(k_pos)
    positive_pairs = []
    for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
        i, j = idx // n, idx % n
        qi = feature_indices_cpu[i].item()
        kj = feature_indices_cpu[j].item()
        positive_pairs.append((int(qi), int(kj), float(val)))
    
    # Top negative
    k_neg = min(k, B_flat.numel())
    bot_vals, bot_idx = B_flat.topk(k_neg, largest=False)
    negative_pairs = []
    for val, idx in zip(bot_vals.tolist(), bot_idx.tolist()):
        i, j = idx // n, idx % n
        qi = feature_indices_cpu[i].item()
        kj = feature_indices_cpu[j].item()
        negative_pairs.append((int(qi), int(kj), float(val)))
    
    return TopPairs(positive_pairs=positive_pairs, negative_pairs=negative_pairs)


def compute_gram_corrected_affinity(
    B: torch.Tensor,
    gram: torch.Tensor,
    method: str = "subtract",
) -> torch.Tensor:
    """
    Apply superposition correction to affinity matrix.
    
    Args:
        B: [n, n] raw affinity matrix
        gram: [n, n] Gram matrix of decoder directions
        method: "subtract" (simple) or "whiten" (more complex)
        
    Returns:
        B_corrected: [n, n] corrected affinity matrix
    """
    if method == "subtract":
        # Simple: subtract correlation baseline
        # Intuition: remove "affinity" that's just because features overlap
        return B - gram
    elif method == "whiten":
        # More sophisticated: decorrelate
        # B_white = G^(-1/2) @ B @ G^(-1/2)
        eps = 1e-6
        eigvals, eigvecs = torch.linalg.eigh(gram)
        eigvals = eigvals.clamp(min=eps)
        G_inv_sqrt = eigvecs @ torch.diag(1.0 / eigvals.sqrt()) @ eigvecs.T
        return G_inv_sqrt @ B @ G_inv_sqrt
    else:
        raise ValueError(f"Unknown correction method: {method}")


def compute_random_d_baseline(
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    n_features: int,
    hidden_size: int,
    config: Gemma2Config = GEMMA2_CONFIG,
    seed: int = 42,
) -> RoutingMetrics:
    """
    Baseline: random orthonormal decoder directions.
    """
    g = torch.Generator(device=W_Q.device).manual_seed(seed)
    D_rand = torch.randn(n_features, hidden_size, generator=g, device=W_Q.device, dtype=W_Q.dtype)
    D_rand = F.normalize(D_rand, dim=1)
    
    Q_f = D_rand @ W_Q.T
    K_f = D_rand @ W_K.T
    B = compute_affinity_matrix(Q_f, K_f, config)
    
    return compute_routing_metrics(B, torch.arange(n_features, device=W_Q.device))


def compute_random_weights_baseline(
    decoder: torch.Tensor,
    head_dim: int,
    config: Gemma2Config = GEMMA2_CONFIG,
    seed: int = 42,
    target_std: float = None,
) -> RoutingMetrics:
    """
    Baseline: random W_Q/W_K, fixed SAE decoder.
    
    TEMPERATURE MATCHING: If target_std is provided, rescale B to have
    the same std as the real B. This is critical because softmax selectivity
    is extremely sensitive to logit scale - random weights often produce
    much larger logit variance, making the baseline artificially peaky.
    
    Args:
        decoder: SAE decoder subset
        head_dim: Dimension of attention head
        config: Model config
        seed: Random seed
        target_std: If provided, rescale B to match this std (from real B)
    """
    n, hidden_size = decoder.shape
    g = torch.Generator(device=decoder.device).manual_seed(seed)
    
    W_Q_rand = torch.randn(head_dim, hidden_size, generator=g, device=decoder.device, dtype=decoder.dtype)
    W_K_rand = torch.randn(head_dim, hidden_size, generator=g, device=decoder.device, dtype=decoder.dtype)
    
    # Scale to match typical weight norms
    W_Q_rand = W_Q_rand / math.sqrt(hidden_size)
    W_K_rand = W_K_rand / math.sqrt(hidden_size)
    
    Q_f = decoder @ W_Q_rand.T
    K_f = decoder @ W_K_rand.T
    B = compute_affinity_matrix(Q_f, K_f, config)
    
    # Temperature matching: rescale B to have same std as real B
    # This is critical because softmax(x/T) peakiness depends entirely on T
    if target_std is not None:
        B_std = B.float().std().item()
        if B_std > 1e-8:
            scale_factor = target_std / B_std
            B = B * scale_factor
    
    return compute_routing_metrics(B, torch.arange(n, device=decoder.device))


def compute_independent_bases_baseline(
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    n_features: int,
    hidden_size: int,
    config: Gemma2Config = GEMMA2_CONFIG,
    seed: int = 42,
    target_std: float = None,
) -> RoutingMetrics:
    """
    Baseline: INDEPENDENT random bases for Q and K.
    
    This kills the "diagonal shortcut" - if real routing shows high diagonal
    dominance but this baseline also does, then diagonal structure is an
    artifact of shared basis, not meaningful feature self-attention.
    
    TEMPERATURE MATCHING: If target_std is provided, rescale B to have
    the same std as the real B before computing metrics.
    
    Unlike random_d_baseline which uses the SAME random D for both Q and K,
    this uses DIFFERENT random directions for each.
    """
    gq = torch.Generator(device=W_Q.device).manual_seed(seed)
    gk = torch.Generator(device=W_Q.device).manual_seed(seed + 1000)
    
    D_Q = torch.randn(n_features, hidden_size, generator=gq, device=W_Q.device, dtype=W_Q.dtype)
    D_K = torch.randn(n_features, hidden_size, generator=gk, device=W_Q.device, dtype=W_Q.dtype)
    
    # RMS normalize to match decoder calibration
    D_Q = F.normalize(D_Q, dim=1) * (hidden_size ** 0.5)
    D_K = F.normalize(D_K, dim=1) * (hidden_size ** 0.5)
    
    Q_f = D_Q @ W_Q.T
    K_f = D_K @ W_K.T
    B = compute_affinity_matrix(Q_f, K_f, config)
    
    # Temperature matching: rescale B to have same std as real B
    if target_std is not None:
        B_std = B.float().std().item()
        if B_std > 1e-8:
            B = B * (target_std / B_std)
    
    return compute_routing_metrics(B, torch.arange(n_features, device=W_Q.device))


def compute_permuted_k_baseline(
    decoder: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    config: Gemma2Config = GEMMA2_CONFIG,
    seed: int = 42,
    target_std: float = None,
) -> RoutingMetrics:
    """
    Baseline: Permuted-K using REAL decoder basis.
    
    Q features use real decoder, but K features use a PERMUTED subset of 
    the same decoder. This kills feature identity alignment while preserving
    the marginal distribution of both Q and K features.
    
    TEMPERATURE MATCHING: If target_std is provided, rescale B to have
    the same std as the real B before computing metrics.
    
    If real routing shows structured patterns (e.g., feature 123 attends to 
    feature 456) but this baseline is diffuse, then the structure is meaningful.
    """
    n = decoder.shape[0]
    g = torch.Generator(device=decoder.device).manual_seed(seed)
    perm = torch.randperm(n, generator=g, device=decoder.device)
    
    decoder_perm = decoder[perm]
    
    Q_f = decoder @ W_Q.T
    K_f = decoder_perm @ W_K.T
    B = compute_affinity_matrix(Q_f, K_f, config)
    
    # Temperature matching: rescale B to have same std as real B
    if target_std is not None:
        B_std = B.float().std().item()
        if B_std > 1e-8:
            B = B * (target_std / B_std)
    
    return compute_routing_metrics(B, torch.arange(n, device=decoder.device))


def analyze_head_routing(
    sae_features: SAEFeatures,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    layer_idx: int,
    query_head: int,
    kv_group: int,
    config: Gemma2Config = GEMMA2_CONFIG,
    analysis_config: AnalysisConfig = ANALYSIS_CONFIG,
) -> HeadRoutingResult:
    """
    Complete routing analysis for a single head.
    
    Args:
        sae_features: Prepared SAE features
        W_Q: [head_dim, hidden_size] query projection (γ-folded)
        W_K: [head_dim, hidden_size] key projection (γ-folded)
        layer_idx: Layer index
        query_head: Query head index
        kv_group: KV group index
        config: Model config
        analysis_config: Analysis config
        
    Returns:
        HeadRoutingResult with all metrics and pairs
    """
    decoder = sae_features.decoder_subset
    feature_indices = sae_features.feature_indices
    
    # Core: compute Q_f, K_f, B
    Q_f, K_f = project_to_qk_space(decoder, W_Q, W_K)
    B = compute_affinity_matrix(Q_f, K_f, config)
    
    # Metrics
    metrics = compute_routing_metrics(B, feature_indices)
    
    # Top pairs
    top_pairs = extract_top_pairs(B, feature_indices, k=analysis_config.topk_pairs_to_report)
    
    result = HeadRoutingResult(
        layer_idx=layer_idx,
        query_head=query_head,
        kv_group=kv_group,
        metrics=metrics,
        top_pairs=top_pairs,
    )
    
    # Baselines
    n = decoder.shape[0]
    hidden_size = decoder.shape[1]
    head_dim = W_Q.shape[0]
    
    if analysis_config.run_random_d_baseline:
        result.baseline_random_d = compute_random_d_baseline(
            W_Q, W_K, n, hidden_size, config, seed=config.seed + 1000
        )
    
    if analysis_config.run_random_weights_baseline:
        # Pass real B's std for temperature matching
        real_B_std = metrics.std_affinity
        result.baseline_random_weights = compute_random_weights_baseline(
            decoder, head_dim, config, seed=config.seed + 2000, target_std=real_B_std
        )
    
    # New stronger baselines (from playground)
    if analysis_config.run_permutation_baseline:
        # Pass real B's std for temperature matching (same as random_weights)
        real_B_std = metrics.std_affinity
        
        # Independent random bases for Q and K (kills diagonal shortcut)
        result.baseline_independent_bases = compute_independent_bases_baseline(
            W_Q, W_K, n, hidden_size, config, seed=config.seed + 3000, target_std=real_B_std
        )
        # Permuted-K with real decoder (kills feature identity, keeps distribution)
        result.baseline_permuted_k = compute_permuted_k_baseline(
            decoder, W_Q, W_K, config, seed=config.seed + 4000, target_std=real_B_std
        )
    
    # Gram correction
    if analysis_config.compute_gram_correction and sae_features.gram_matrix is not None:
        B_corrected = compute_gram_corrected_affinity(B, sae_features.gram_matrix)
        result.metrics_gram_corrected = compute_routing_metrics(B_corrected, feature_indices)
    
    return result


def compute_cross_head_redundancy(
    results: List[HeadRoutingResult],
) -> Dict[Tuple[int, int], float]:
    """
    Compute redundancy between heads in the same layer.
    
    Returns dict mapping (head_i, head_j) -> similarity score
    """
    redundancy = {}
    
    for i, r1 in enumerate(results):
        for j, r2 in enumerate(results):
            if i >= j:
                continue
            
            # Compare top positive pairs
            pairs1 = set((p[0], p[1]) for p in r1.top_pairs.positive_pairs[:20])
            pairs2 = set((p[0], p[1]) for p in r2.top_pairs.positive_pairs[:20])
            
            if pairs1 and pairs2:
                jaccard = len(pairs1 & pairs2) / len(pairs1 | pairs2)
            else:
                jaccard = 0.0
            
            redundancy[(r1.query_head, r2.query_head)] = jaccard
    
    return redundancy


if __name__ == "__main__":
    print("QK Routing Analysis module loaded.")
    print(f"Head archetypes: {[a.value for a in HeadArchetype]}")
