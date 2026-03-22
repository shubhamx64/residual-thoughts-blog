"""
Configuration module for Gemma-2 weight-space SAE analysis.

Locks down all model-specific constants and conventions to ensure
coordinate system discipline across the analysis pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional
import torch


@dataclass
class Gemma2Config:
    """
    Gemma-2-2B specific configuration for weight-space analysis.
    
    Constants derived from HuggingFace model config and modeling_gemma2.py:
    - RoPE applied after Q/K projection, before attention dot-product
    - Scaling uses query_pre_attn_scalar**-0.5
    - Alternating sliding-window (local) and global attention layers
    - Pre/post attention RMSNorm pattern
    """
    
    # Model identification
    model_id: str = "google/gemma-2-2b"
    
    # Architecture constants
    num_hidden_layers: int = 26
    hidden_size: int = 2304
    num_attention_heads: int = 8       # query heads
    num_key_value_heads: int = 4       # KV heads (GQA)
    head_dim: int = 256
    
    # GQA derived
    @property
    def num_queries_per_kv(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads
    
    def query_to_kv_group(self, query_head: int) -> int:
        """Map query head index to its KV group."""
        return query_head // self.num_queries_per_kv
    
    # Attention scaling (Gemma-2 specific)
    # From config: query_pre_attn_scalar = hidden_size // num_attention_heads = 256
    query_pre_attn_scalar: int = 256
    
    @property
    def attention_scale(self) -> float:
        """Scale factor applied to QK dot product before softmax."""
        return self.query_pre_attn_scalar ** -0.5
    
    # Soft-capping (Gemma-2 uses tanh soft-cap on logits when enabled)
    attn_logit_softcapping: Optional[float] = 50.0  # None to disable
    
    # RoPE configuration
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
    
    # Sliding window attention
    # Gemma-2 alternates: even layers = sliding window, odd layers = global
    sliding_window_size: int = 4096
    
    def is_sliding_window_layer(self, layer_idx: int) -> bool:
        """Returns True if layer uses sliding-window (local) attention."""
        return layer_idx % 2 == 0
    
    def max_relative_position(self, layer_idx: int) -> int:
        """Maximum relative position Δ allowed for this layer."""
        if self.is_sliding_window_layer(layer_idx):
            return self.sliding_window_size
        return self.max_position_embeddings
    
    # SAE configuration
    sae_release: str = "gemma-scope-2b-pt-res-canonical"
    sae_width: int = 16384
    sae_width_id: str = "width_16k/canonical"
    
    # SAE tap point convention:
    # - SAE trained on post-MLP residual stream
    # - We fold pre-attention RMSNorm gamma into Q/K/V weights
    sae_tap_point: Literal["post_mlp_residual"] = "post_mlp_residual"
    fold_rmsnorm_gamma: bool = True
    
    # SAE layer offset for attention analysis:
    # post-MLP residual of block L is input to block L+1's attention
    # If SAE "layer_L" = post-MLP of block L, and we want to analyze
    # attention in block L, we might need SAE from layer L-1.
    # Set to 0 if SAE layer matches attention block directly.
    # Set to -1 if SAE layer_L is post-MLP of block L (most common).
    sae_layer_offset_for_attn: int = -1  # Default: same layer
    
    def get_sae_layer_for_attn(self, attn_layer_idx: int) -> int:
        """Get the SAE layer index to use for a given attention layer."""
        return max(0, attn_layer_idx + self.sae_layer_offset_for_attn)
    
    # Feature subset for analysis (scale up with FAISS later)
    feature_subset_size: int = 2048
    
    # Normalization convention for decoder directions
    normalize_decoder_directions: bool = True
    
    # QK affinity computation mode:
    # "logit": No Q/K normalization, apply attention_scale (matches real attention)
    # "cosine": L2-normalize Q/K, apply semantic_temperature (directional similarity)
    # "logit" is recommended for meaningful selectivity metrics.
    qk_mode: Literal["logit", "cosine"] = "logit"
    semantic_temperature: float = 10.0  # Only used in "cosine" mode
    
    # Whether to apply tanh soft-capping to logits (model does this at ~50.0)
    apply_softcap: bool = True
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    
    # Random seed for reproducibility
    seed: int = 42


@dataclass  
class AnalysisConfig:
    """
    Configuration for the analysis pipeline itself.
    """
    
    # Layers to analyze
    layers_to_analyze: List[int] = field(default_factory=lambda: [0, 5, 10, 15, 20, 25])
    
    # QK routing analysis
    n_sample_rows: int = 256        # query features to sample for metrics
    topk_keys: int = 20             # top-k keys per query
    topk_pairs_to_report: int = 50  # top pairs to store per head
    
    # RoPE position analysis
    delta_positions: List[int] = field(
        default_factory=lambda: [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    )
    
    # Baselines
    run_random_d_baseline: bool = True
    run_random_weights_baseline: bool = True
    run_permutation_baseline: bool = True
    
    # Superposition correction
    compute_gram_correction: bool = True
    
    # Output
    output_dir: str = "./analysis_outputs"
    save_full_matrices: bool = False  # Only for debugging; very large
    

# Default instances
GEMMA2_CONFIG = Gemma2Config()
ANALYSIS_CONFIG = AnalysisConfig()


def get_rope_frequencies(config: Gemma2Config) -> torch.Tensor:
    """
    Compute RoPE frequency bands for Gemma-2.
    
    Returns: [head_dim // 2] tensor of frequencies
    """
    dim = config.head_dim
    freqs = 1.0 / (config.rope_theta ** (
        torch.arange(0, dim, 2, dtype=torch.float32) / dim
    ))
    return freqs


def print_config_summary(config: Gemma2Config) -> None:
    """Print key configuration for verification."""
    print("=" * 60)
    print("Gemma-2 Weight-Space Analysis Configuration")
    print("=" * 60)
    print(f"Model: {config.model_id}")
    print(f"Layers: {config.num_hidden_layers}")
    print(f"Attention: {config.num_attention_heads} Q heads, {config.num_key_value_heads} KV heads (GQA)")
    print(f"Head dim: {config.head_dim}")
    print(f"Attention scale: {config.attention_scale:.6f} (query_pre_attn_scalar={config.query_pre_attn_scalar})")
    print(f"Soft-cap: {config.attn_logit_softcapping}")
    print(f"Sliding window: {config.sliding_window_size} (alternating layers)")
    print(f"SAE: {config.sae_release} @ {config.sae_width_id}")
    print(f"SAE tap point: {config.sae_tap_point}, fold γ: {config.fold_rmsnorm_gamma}")
    print(f"Feature subset: {config.feature_subset_size}")
    print(f"Device: {config.device}, dtype: {config.dtype}")
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary(GEMMA2_CONFIG)
    print(f"\nRoPE frequencies (first 8): {get_rope_frequencies(GEMMA2_CONFIG)[:8]}")
