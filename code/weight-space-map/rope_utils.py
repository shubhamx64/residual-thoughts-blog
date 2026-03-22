"""
RoPE (Rotary Position Embedding) utilities for Gemma-2.

Key facts for Gemma-2:
- RoPE is applied AFTER Q/K projection and BEFORE attention dot-product
- It's a multiplicative rotation, not additive position vectors
- Rotation is block-diagonal: pairs of dimensions rotate together

For weight-space analysis, we need to compute:
    B_Δ = Q_f @ R_Δ @ K_f.T
where R_Δ is the relative-position rotation for offset Δ.
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
    
    This is the block-diagonal matrix where each 2x2 block is:
        [[cos(Δ * freq), -sin(Δ * freq)],
         [sin(Δ * freq),  cos(Δ * freq)]]
    
    Args:
        delta: Relative position offset (t - s)
        freqs: [head_dim // 2] frequency values
        
    Returns: [head_dim, head_dim] rotation matrix
    """
    angles = delta * freqs.to(device=device, dtype=dtype)
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    
    head_dim = len(freqs) * 2
    R = torch.zeros(head_dim, head_dim, device=device, dtype=dtype)
    
    for i, (c, s) in enumerate(zip(cos_angles, sin_angles)):
        idx = i * 2
        R[idx, idx] = c
        R[idx, idx + 1] = -s
        R[idx + 1, idx] = s
        R[idx + 1, idx + 1] = c
    
    return R


def apply_rope_rotation_to_vectors(
    vectors: torch.Tensor,
    delta: int,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """
    Efficiently apply RoPE rotation to a batch of vectors.
    
    Instead of materializing the full rotation matrix, we use the paired
    rotation formula directly (same as HF apply_rotary_pos_emb).
    
    Args:
        vectors: [N, head_dim] vectors to rotate
        delta: Relative position offset
        freqs: [head_dim // 2] frequencies
        
    Returns: [N, head_dim] rotated vectors
    """
    device = vectors.device
    dtype = vectors.dtype
    
    angles = (delta * freqs).to(device=device, dtype=dtype)
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    
    # Reshape to pairs: [N, head_dim//2, 2]
    x = vectors.view(vectors.shape[0], -1, 2)
    
    # Apply rotation: x0' = x0*cos - x1*sin, x1' = x0*sin + x1*cos
    x0, x1 = x[..., 0], x[..., 1]
    x_rot = torch.stack([
        x0 * cos_angles - x1 * sin_angles,
        x0 * sin_angles + x1 * cos_angles
    ], dim=-1)
    
    return x_rot.view(vectors.shape)


def compute_rotated_affinity_matrix(
    Q_f: torch.Tensor,
    K_f: torch.Tensor,
    delta: int,
    freqs: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Compute position-modulated affinity matrix B_Δ.
    
    B_Δ[i,j] = (Q_f[i] @ R_Δ @ K_f[j]) * scale
    
    Efficient: rotate K_f vectors, then compute dot products.
    
    Args:
        Q_f: [n_features, head_dim] query features in Q-space
        K_f: [n_features, head_dim] key features in K-space
        delta: Relative position offset
        freqs: RoPE frequencies
        scale: Attention scaling factor
        
    Returns: [n_features, n_features] affinity matrix
    """
    # Rotate keys by Δ (equivalent to rotating queries by -Δ)
    K_f_rotated = apply_rope_rotation_to_vectors(K_f, delta, freqs)
    
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
    # Reshape to pairs: [N, head_dim//2, 2]
    x = vectors.view(vectors.shape[0], -1, 2)
    n_pairs = x.shape[1]
    pairs_per_band = n_pairs // n_bands
    
    energies = []
    for b in range(n_bands):
        start = b * pairs_per_band
        end = start + pairs_per_band if b < n_bands - 1 else n_pairs
        band_energy = (x[:, start:end, :] ** 2).sum(dim=(1, 2))
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
