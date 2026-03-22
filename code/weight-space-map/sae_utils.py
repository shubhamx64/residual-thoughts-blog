"""
SAE (Sparse Autoencoder) loading utilities for Gemma-2 analysis.

Handles:
- Loading SAE from sae-lens
- Decoder direction normalization
- Feature subset selection
- Gram matrix computation for superposition correction
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
from dataclasses import dataclass

from config import Gemma2Config, GEMMA2_CONFIG


@dataclass
class SAEFeatures:
    """
    SAE decoder directions prepared for analysis.
    """
    layer_idx: int
    
    # Full decoder: [n_features, hidden_size]
    decoder_full: torch.Tensor
    
    # Normalized decoder (if configured): [n_features, hidden_size]
    decoder_normalized: torch.Tensor
    
    # Feature indices in use (subset): [subset_size]
    feature_indices: torch.Tensor
    
    # Subset of decoder directions: [subset_size, hidden_size]
    decoder_subset: torch.Tensor
    
    # Gram matrix for superposition correction: [subset_size, subset_size]
    gram_matrix: Optional[torch.Tensor]
    
    # Decoder norms (for tracking activation scale): [n_features]
    decoder_norms: torch.Tensor


def load_sae_decoder(
    layer_idx: int,
    config: Gemma2Config = GEMMA2_CONFIG,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Load SAE decoder matrix for a layer.
    
    Args:
        layer_idx: Target layer (we use SAE from same layer for post-MLP residual)
        config: Model configuration
        device: Target device
        dtype: Target dtype
        
    Returns:
        Decoder matrix [n_features, hidden_size]
    """
    from sae_lens import SAE
    
    sae_id = f"layer_{layer_idx}/{config.sae_width_id}"
    sae = SAE.from_pretrained(release=config.sae_release, sae_id=sae_id)
    
    # W_dec: [n_features, hidden_size]
    decoder = sae.W_dec.detach().to(device=device, dtype=dtype)
    
    return decoder


def prepare_sae_features(
    layer_idx: int,
    config: Gemma2Config = GEMMA2_CONFIG,
    subset_size: Optional[int] = None,
    seed: int = 42,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    compute_gram: bool = True,
) -> SAEFeatures:
    """
    Load and prepare SAE features for analysis.
    
    Args:
        layer_idx: Layer to load SAE for
        config: Model configuration
        subset_size: Number of features to use (None = all)
        seed: Random seed for subset selection
        device: Target device
        dtype: Target dtype
        compute_gram: Whether to compute Gram matrix
        
    Returns:
        SAEFeatures dataclass with prepared data
    """
    # Load full decoder
    decoder = load_sae_decoder(layer_idx, config, device, dtype)
    n_features = decoder.shape[0]
    hidden_size = decoder.shape[1]
    
    # Compute norms before normalization
    decoder_norms = decoder.norm(dim=1)
    
    # RMS calibration: normalize decoder directions to have RMS=1, not L2=1
    # 
    # Why this matters:
    # - RMSNorm outputs vectors with RMS ≈ 1, meaning L2 norm ≈ sqrt(hidden_size)
    # - L2-normalizing decoder directions to norm=1 makes them sqrt(hidden_size) times
    #   too small, causing attention logits to be tiny (uniform softmax)
    # - RMS normalization: d_rms = d / sqrt(mean(d^2)) = d * sqrt(dim) / ||d||
    #   which gives L2 norm = sqrt(hidden_size), matching RMSNorm'd residual stream
    #
    if config.normalize_decoder_directions:
        # RMS normalization per vector
        # rms = sqrt(mean(d^2)) = ||d|| / sqrt(dim)
        # d_rms_normalized = d / rms = d * sqrt(dim) / ||d||
        decoder_normalized = F.normalize(decoder, dim=1) * (hidden_size ** 0.5)
    else:
        decoder_normalized = decoder
    
    # Select subset
    if subset_size is None:
        subset_size = n_features
    subset_size = min(subset_size, n_features)
    
    g = torch.Generator().manual_seed(seed + layer_idx * 1000)
    feature_indices = torch.randperm(n_features, generator=g)[:subset_size]
    # Move feature_indices to same device as decoder
    feature_indices = feature_indices.to(device=device)
    
    decoder_subset = decoder_normalized[feature_indices]
    
    # Compute Gram matrix for superposition correction
    gram_matrix = None
    if compute_gram:
        # G[i,j] = decoder[i] · decoder[j]
        # This measures directional overlap between features
        gram_matrix = decoder_subset @ decoder_subset.T
    
    return SAEFeatures(
        layer_idx=layer_idx,
        decoder_full=decoder,
        decoder_normalized=decoder_normalized,
        feature_indices=feature_indices,
        decoder_subset=decoder_subset,
        gram_matrix=gram_matrix,
        decoder_norms=decoder_norms,
    )


def project_to_qk_space(
    decoder: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project decoder directions into Q and K spaces.
    
    Args:
        decoder: [n_features, hidden_size] normalized decoder directions
        W_Q: [head_dim, hidden_size] query projection
        W_K: [head_dim, hidden_size] key projection
        
    Returns:
        (Q_f, K_f) each of shape [n_features, head_dim]
    """
    # Q_f = decoder @ W_Q.T -> [n_features, head_dim]
    Q_f = decoder @ W_Q.T
    K_f = decoder @ W_K.T
    
    return Q_f, K_f


def project_to_v_space(
    decoder: torch.Tensor,
    W_V: torch.Tensor,
) -> torch.Tensor:
    """
    Project decoder directions into V space.
    
    Args:
        decoder: [n_features, hidden_size] normalized decoder directions
        W_V: [head_dim, hidden_size] value projection
        
    Returns:
        V_f of shape [n_features, head_dim]
    """
    return decoder @ W_V.T


def compute_write_vectors(
    decoder: torch.Tensor,
    W_OV: torch.Tensor,
) -> torch.Tensor:
    """
    Compute write vectors for each feature.
    
    "If a key token has feature j, what direction does this head
    tend to write when attending to it?"
    
    Args:
        decoder: [n_features, hidden_size] decoder directions
        W_OV: [hidden_size, hidden_size] OV composition matrix
        
    Returns:
        write_vectors: [n_features, hidden_size]
    """
    # w_j = d_j @ W_OV
    return decoder @ W_OV


def project_writes_to_features(
    write_vectors: torch.Tensor,
    decoder: torch.Tensor,
    method: str = "decoder",
) -> torch.Tensor:
    """
    Project write vectors back to feature space.
    
    Args:
        write_vectors: [n_features, hidden_size] what head writes
        decoder: [n_features, hidden_size] decoder directions
        method: "decoder" for similarity to decoder directions
        
    Returns:
        write_to_feature: [n_features, n_features] mapping j -> k
    """
    if method == "decoder":
        # Normalize for cosine similarity
        w_norm = F.normalize(write_vectors, dim=1)
        d_norm = F.normalize(decoder, dim=1)
        return w_norm @ d_norm.T
    else:
        raise ValueError(f"Unknown projection method: {method}")


class SAEManager:
    """
    Manages SAE loading and caching for multi-layer analysis.
    """
    
    def __init__(
        self,
        config: Gemma2Config = GEMMA2_CONFIG,
        subset_size: Optional[int] = None,
        seed: int = 42,
    ):
        self.config = config
        self.subset_size = subset_size or config.feature_subset_size
        self.seed = seed
        self._cache = {}
    
    def get_features(
        self,
        layer_idx: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        compute_gram: bool = True,
    ) -> SAEFeatures:
        """Get SAE features for a layer (cached)."""
        cache_key = (layer_idx, device, str(dtype))
        
        if cache_key not in self._cache:
            self._cache[cache_key] = prepare_sae_features(
                layer_idx,
                self.config,
                self.subset_size,
                self.seed,
                device,
                dtype,
                compute_gram,
            )
        
        return self._cache[cache_key]
    
    def get_decoder(self, layer_idx: int, device: str = "cuda") -> torch.Tensor:
        """Get the full SAE decoder matrix for a layer.
        
        NOTE: This returns only W_dec without the bias. For proper reconstruction,
        use get_decoder_with_bias() or decode() instead.
        
        Returns:
            Decoder matrix [hidden_size, n_features] (transposed for projection)
        """
        from sae_lens import SAE
        
        release = self.config.sae_release
        sae_id = f"layer_{layer_idx}/{self.config.sae_width_id}"
        
        # Load SAE
        sae = SAE.from_pretrained(release, sae_id, device=device)[0]
        
        # Get decoder: SAE decoder is [n_features, hidden_size]
        # We return [hidden_size, n_features] for decoder @ features -> residual
        decoder = sae.W_dec.data.T.float()  # [hidden_size, n_features]
        
        return decoder
    
    def get_decoder_with_bias(self, layer_idx: int, device: str = "cuda"):
        """Get SAE decoder matrix and bias for proper reconstruction.
        
        Returns:
            Tuple of (W_dec, b_dec):
            - W_dec: [hidden_size, n_features] decoder matrix
            - b_dec: [hidden_size] decoder bias
        """
        from sae_lens import SAE
        
        release = self.config.sae_release
        sae_id = f"layer_{layer_idx}/{self.config.sae_width_id}"
        
        sae = SAE.from_pretrained(release, sae_id, device=device)[0]
        
        W_dec = sae.W_dec.data.T.float()  # [hidden_size, n_features]
        b_dec = sae.b_dec.data.float()    # [hidden_size]
        
        return W_dec, b_dec
    
    def decode(self, features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Decode SAE features back to residual stream with proper bias.
        
        Args:
            features: [seq_len, n_features] or [batch, seq_len, n_features]
            layer_idx: Which SAE layer to use
            
        Returns:
            Reconstructed residual [seq_len, hidden_size] or [batch, seq_len, hidden_size]
        """
        W_dec, b_dec = self.get_decoder_with_bias(layer_idx, device=str(features.device))
        
        # reconstruction = features @ W_dec.T + b_dec
        # But W_dec is already transposed, so: features @ W_dec.T = features @ (W_dec_orig)
        # Actually W_dec is [hidden_size, n_features], so features @ W_dec.T gives [*, hidden_size]
        # That's wrong - we need features @ W_dec_orig where W_dec_orig is [n_features, hidden_size]
        # 
        # Let's fix: features is [*, n_features], we want [*, hidden_size]
        # W_dec is [hidden_size, n_features], so W_dec.T is [n_features, hidden_size]
        # features @ W_dec.T = [*, n_features] @ [n_features, hidden_size] = [*, hidden_size] ✓
        
        return features.float() @ W_dec.T.to(features.device) + b_dec.to(features.device)
    
    def encode(self, residual: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Encode residual stream activations into SAE feature activations.
        
        Args:
            residual: [seq_len, hidden_size] or [batch, seq_len, hidden_size]
            layer_idx: Which SAE layer to use
            
        Returns:
            SAE activations [seq_len, n_features] or [batch, seq_len, n_features]
        """
        from sae_lens import SAE
        
        device = residual.device
        release = self.config.sae_release
        sae_id = f"layer_{layer_idx}/{self.config.sae_width_id}"
        
        # Load SAE (cached by sae_lens)
        sae = SAE.from_pretrained(release, sae_id, device=str(device))[0]
        
        # Encode
        with torch.no_grad():
            # SAE expects [batch, hidden_size], outputs [batch, n_features]
            squeezed = residual.ndim == 2
            if squeezed:
                residual = residual.unsqueeze(0)  # Add batch dim
            
            # Flatten to [batch*seq, hidden_size]
            batch, seq_len, hidden = residual.shape
            flat = residual.reshape(-1, hidden)
            
            # Encode
            features = sae.encode(flat.to(sae.W_enc.dtype))
            
            # Reshape back
            features = features.reshape(batch, seq_len, -1)
            
            if squeezed:
                features = features.squeeze(0)
            
            return features.float()
    
    def clear_cache(self):
        """Clear the SAE cache."""
        self._cache.clear()


if __name__ == "__main__":
    print("SAE utilities module loaded.")
    print(f"Expected decoder shape: [{GEMMA2_CONFIG.sae_width}, {GEMMA2_CONFIG.hidden_size}]")
    print(f"Default subset size: {GEMMA2_CONFIG.feature_subset_size}")
