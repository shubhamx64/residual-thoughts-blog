"""
RoPE (Rotary Position Embedding) utilities for Gemma-2.

Key facts for Gemma-2:
- RoPE is applied AFTER Q/K projection and BEFORE attention dot-product
- It's a multiplicative rotation, not additive position vectors
- HF Gemma-2 uses the rotate_half (NeoX/half-split) convention:
  dimension i pairs with dimension i + head_dim/2, both rotating with
  inv_freq[i]. NOT the interleaved (2i, 2i+1) GPT-J convention.
  See modeling_gemma2.py: rotate_half + apply_rotary_pos_emb with
  cos/sin = cat(freqs, freqs).

For weight-space analysis, we need to compute:
    B_Δ[i, j] = Q_f[i] @ R(-Δ) @ K_f[j]
where R(p) is the rotation at position p and Δ = t - s >= 0 is the
relative offset (query position t, key position s). Derivation: the model
rotates q by R(t) and k by R(s); the logit is
    (R(t) q) · (R(s) k) = q^T R(t)^T R(s) k = q^T R(s - t) k = q^T R(-Δ) k.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import math

from config import Gemma2Config, GEMMA2_CONFIG


def compute_rope_frequencies(config: Gemma2Config) -> torch.Tensor:
    """
    Compute RoPE frequency bands.
    
    Returns: [head_dim // 2] frequencies
    """
    dim = config.head_dim
    freqs = 1.0 / (config.rope_theta ** (
        torch.arange(0, dim, 2, dtype=torch.float32) / dim
    ))
    return freqs


def compute_rotation_matrix(
    delta: int,
    freqs: torch.Tensor,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Compute the full rotation matrix R_Δ for relative position offset Δ.

    HF Gemma-2 rotate_half convention: dimension i pairs with i + half
    (half = head_dim // 2), both rotating with freqs[i]:
        R[i, i]           =  cos(Δ * freq_i)
        R[i, i + half]    = -sin(Δ * freq_i)
        R[i + half, i]    =  sin(Δ * freq_i)
        R[i + half, i+half] = cos(Δ * freq_i)

    Args:
        delta: Position offset (signed; pass -Δ for the key-side rotation)
        freqs: [head_dim // 2] frequency values

    Returns: [head_dim, head_dim] rotation matrix
    """
    angles = delta * freqs.to(device=device, dtype=dtype)
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    half = len(freqs)
    R = torch.zeros(2 * half, 2 * half, device=device, dtype=dtype)

    idx = torch.arange(half, device=device)
    R[idx, idx] = cos_angles
    R[idx, idx + half] = -sin_angles
    R[idx + half, idx] = sin_angles
    R[idx + half, idx + half] = cos_angles

    return R


def apply_rope_rotation_to_vectors(
    vectors: torch.Tensor,
    delta: int,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """
    Efficiently apply RoPE rotation to a batch of vectors.

    Uses the HF rotate_half (half-split) convention: dimension i pairs with
    i + head_dim/2, both rotating with freqs[i]. Equivalent to HF's
        x * cos + rotate_half(x) * sin
    at position `delta` (cos/sin = cat(cos_a, cos_a), cat(sin_a, sin_a)).

    Args:
        vectors: [N, head_dim] vectors to rotate
        delta: Position offset (signed)
        freqs: [head_dim // 2] frequencies

    Returns: [N, head_dim] rotated vectors
    """
    device = vectors.device
    dtype = vectors.dtype

    angles = (delta * freqs).to(device=device, dtype=dtype)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    half = vectors.shape[-1] // 2
    x1 = vectors[..., :half]
    x2 = vectors[..., half:]

    # x1' = x1*cos - x2*sin, x2' = x2*cos + x1*sin  (rotate_half convention)
    return torch.cat([
        x1 * cos_a - x2 * sin_a,
        x2 * cos_a + x1 * sin_a,
    ], dim=-1)


def compute_rotated_affinity_matrix(
    Q_f: torch.Tensor,
    K_f: torch.Tensor,
    delta: int,
    freqs: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Compute position-modulated affinity matrix B_Δ.

    B_Δ[i,j] = (Q_f[i] @ R(-Δ) @ K_f[j]) * scale

    Sign derivation: the model rotates the query at position t by R(t) and
    the key at position s <= t by R(s). The realized logit is
        (R(t) q) · (R(s) k) = q^T R(t)^T R(s) k = q^T R(s - t) k
                            = q^T R(-Δ) k,  with Δ = t - s >= 0.
    So keys are rotated by -Δ (equivalently queries by +Δ; identical since
    (R(Δ) q) · k = q · (R(-Δ) k)). Callers pass positive Δ = t - s.

    Args:
        Q_f: [n_features, head_dim] query features in Q-space
        K_f: [n_features, head_dim] key features in K-space
        delta: Relative position offset Δ = t - s (>= 0)
        freqs: RoPE frequencies
        scale: Attention scaling factor

    Returns: [n_features, n_features] affinity matrix
    """
    # Rotate keys by -Δ (see sign derivation in docstring)
    K_f_rotated = apply_rope_rotation_to_vectors(K_f, -delta, freqs)

    # Compute affinity
    B_delta = (Q_f @ K_f_rotated.T) * scale

    return B_delta


def compute_stability_score(
    B_0: torch.Tensor,
    B_delta: torch.Tensor,
    metric: str = "cosine"
) -> float:
    """
    Compute similarity between B_0 and B_Δ to measure positional stability.
    
    High stability → content-controlled head
    Low stability → position-modulated head
    
    Args:
        B_0: Affinity matrix at Δ=0
        B_delta: Affinity matrix at Δ
        metric: "cosine" (flatten and compute cosine sim) or "frobenius" (normalized Frobenius)
        
    Returns: Similarity score in [0, 1] for cosine, or ratio for Frobenius
    """
    if metric == "cosine":
        b0_flat = B_0.flatten().float()
        bd_flat = B_delta.flatten().float()
        return F.cosine_similarity(b0_flat.unsqueeze(0), bd_flat.unsqueeze(0)).item()
    elif metric == "frobenius":
        diff_norm = torch.norm(B_delta - B_0, p='fro')
        base_norm = torch.norm(B_0, p='fro')
        return 1.0 - (diff_norm / (base_norm + 1e-8)).item()
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_frequency_band_energy(
    vectors: torch.Tensor,
    n_bands: int = 4
) -> torch.Tensor:
    """
    Decompose vectors into frequency bands and measure energy per band.
    
    This helps identify which RoPE frequency bands a head relies on:
    - Low-frequency bands → stable across positions
    - High-frequency bands → twitchy, position-sensitive
    
    Args:
        vectors: [N, head_dim] vectors (Q or K features)
        n_bands: Number of frequency bands to partition into

    Returns: [N, n_bands] energy per band
    """
    # rotate_half convention: pair i = (dim i, dim i + half), freq = freqs[i]
    half = vectors.shape[-1] // 2
    pair_energy = vectors[:, :half] ** 2 + vectors[:, half:] ** 2  # [N, half]
    n_pairs = pair_energy.shape[1]
    pairs_per_band = n_pairs // n_bands

    energies = []
    for b in range(n_bands):
        start = b * pairs_per_band
        end = start + pairs_per_band if b < n_bands - 1 else n_pairs
        band_energy = pair_energy[:, start:end].sum(dim=1)
        energies.append(band_energy)

    return torch.stack(energies, dim=1)


class RoPEAnalyzer:
    """
    Utility class for RoPE-aware analysis of attention heads.
    """
    
    def __init__(self, config: Gemma2Config = GEMMA2_CONFIG):
        self.config = config
        self.freqs = compute_rope_frequencies(config)
    
    def compute_stability_curve(
        self,
        Q_f: torch.Tensor,
        K_f: torch.Tensor,
        deltas: list[int],
        scale: float = None,
        apply_softcap: bool = True,
    ) -> dict:
        """
        Compute stability across multiple position offsets.
        
        Args:
            Q_f: [n_features, head_dim] query features
            K_f: [n_features, head_dim] key features  
            deltas: List of delta values to test
            scale: Attention scale (defaults to config)
            apply_softcap: If True, apply Gemma's tanh softcapping for consistency
                          with the actual runtime logit distribution.
            
        Returns: Dict with stability scores and summary metrics
        """
        if scale is None:
            scale = self.config.attention_scale
        
        freqs = self.freqs.to(Q_f.device)
        
        def maybe_softcap(B: torch.Tensor) -> torch.Tensor:
            """Apply Gemma's tanh soft-capping if enabled."""
            if apply_softcap and self.config.attn_logit_softcapping is not None:
                cap = self.config.attn_logit_softcapping
                return cap * torch.tanh(B / cap)
            return B
        
        # Compute B_0 as reference (with optional softcapping)
        B_0_raw = (Q_f @ K_f.T) * scale
        B_0 = maybe_softcap(B_0_raw)
        
        stability_scores = {}
        for delta in deltas:
            if delta == 0:
                stability_scores[0] = 1.0
            else:
                B_delta_raw = compute_rotated_affinity_matrix(Q_f, K_f, delta, freqs, scale)
                B_delta = maybe_softcap(B_delta_raw)
                stability_scores[delta] = compute_stability_score(B_0, B_delta)
        
        # Summary: semantic controllability = average stability
        avg_stability = sum(stability_scores.values()) / len(stability_scores)
        
        return {
            "stability_by_delta": stability_scores,
            "semantic_controllability": avg_stability,
            "min_stability": min(stability_scores.values()),
            "max_stability": max(stability_scores.values()),
            "softcap_applied": apply_softcap,
        }
    
    def get_frequency_profile(
        self,
        Q_f: torch.Tensor,
        K_f: torch.Tensor,
        n_bands: int = 4
    ) -> dict:
        """
        Analyze which frequency bands Q and K features concentrate energy in.
        """
        Q_energy = compute_frequency_band_energy(Q_f, n_bands)
        K_energy = compute_frequency_band_energy(K_f, n_bands)
        
        # Normalize to get distribution
        Q_dist = Q_energy / (Q_energy.sum(dim=1, keepdim=True) + 1e-8)
        K_dist = K_energy / (K_energy.sum(dim=1, keepdim=True) + 1e-8)
        
        return {
            "Q_band_distribution": Q_dist.mean(dim=0).tolist(),
            "K_band_distribution": K_dist.mean(dim=0).tolist(),
            "Q_band_std": Q_dist.std(dim=0).tolist(),
            "K_band_std": K_dist.std(dim=0).tolist(),
        }


if __name__ == "__main__":
    # Quick test
    config = GEMMA2_CONFIG
    freqs = compute_rope_frequencies(config)
    print(f"RoPE frequencies shape: {freqs.shape}")
    print(f"First 4 frequencies: {freqs[:4]}")
    print(f"Last 4 frequencies: {freqs[-4:]}")
    
    # Test rotation matrix
    R = compute_rotation_matrix(delta=1, freqs=freqs)
    print(f"\nRotation matrix shape: {R.shape}")
    print(f"R is orthogonal: {torch.allclose(R @ R.T, torch.eye(len(freqs)*2), atol=1e-5)}")
    
    # Test rotation application
    test_vectors = torch.randn(100, config.head_dim)
    rotated = apply_rope_rotation_to_vectors(test_vectors, delta=10, freqs=freqs)
    print(f"\nRotated vectors shape: {rotated.shape}")
    print(f"Norms preserved: {torch.allclose(test_vectors.norm(dim=1), rotated.norm(dim=1), atol=1e-5)}")
