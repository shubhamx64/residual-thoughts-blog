"""
Head Ablation for Gemma-2 Intervention Experiments.

Implements clean head output ablation via o_proj slice zeroing.
This removes a head's contribution to the residual stream without
affecting other heads.

Usage:
    from head_ablation import HeadAblator
    
    ablator = HeadAblator(model, layer_idx=23, head_idx=1)
    ablator.ablate()
    # ... run inference ...
    ablator.restore()
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class AblationTarget:
    """Specification for a head to ablate."""
    layer_idx: int
    head_idx: int
    
    def __str__(self):
        return f"L{self.layer_idx}H{self.head_idx}"


class HeadAblator:
    """
    Ablate attention head outputs by zeroing o_proj weight slices.
    
    For Gemma-2-2B:
    - o_proj.weight shape: [hidden_size, num_heads * head_dim] = [2304, 2048]
    - Each head h contributes columns [h*256 : (h+1)*256]
    - Zeroing this slice removes head h's contribution to the residual stream
    
    This is the cleanest intervention: we don't change attention patterns,
    we only remove the head's write to the residual stream.
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        head_idx: int,
        head_dim: int = 256,
    ):
        """
        Initialize ablator for a specific head.
        
        Args:
            model: HuggingFace Gemma2ForCausalLM model
            layer_idx: Layer index (0-25 for Gemma-2-2B)
            head_idx: Head index within layer (0-7 for Gemma-2-2B)
            head_dim: Dimension per head (256 for Gemma-2-2B)
        """
        self.model = model
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.head_dim = head_dim
        
        self._original_slice: Optional[torch.Tensor] = None
        self._is_ablated = False
        
        # Validate layer index
        n_layers = len(model.model.layers)
        if layer_idx < 0 or layer_idx >= n_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {n_layers})")
        
        # Validate o_proj shape and head index
        o_proj = self._get_o_proj()
        expected_in_features = 8 * head_dim  # num_heads * head_dim for Gemma-2-2B
        actual_in_features = o_proj.weight.shape[1]
        
        # Critical assertion: o_proj must have expected shape
        assert actual_in_features == expected_in_features, (
            f"o_proj.weight.shape[1] = {actual_in_features}, expected {expected_in_features} "
            f"(num_heads=8 * head_dim={head_dim}). Wrong model or head_dim?"
        )
        
        n_heads = actual_in_features // head_dim
        if head_idx < 0 or head_idx >= n_heads:
            raise ValueError(f"head_idx {head_idx} out of range [0, {n_heads})")
        
        # Print sanity check on first ablator creation
        start_col, end_col = self._get_slice_indices()
        print(f"HeadAblator initialized: L{layer_idx}H{head_idx}")
        print(f"  o_proj.weight.shape = {tuple(o_proj.weight.shape)}")
        print(f"  Slice: [:, {start_col}:{end_col}]")
    
    def _get_o_proj(self) -> nn.Linear:
        """Get the o_proj layer for this head's layer."""
        return self.model.model.layers[self.layer_idx].self_attn.o_proj
    
    def _get_slice_indices(self) -> Tuple[int, int]:
        """Get column indices for this head's slice."""
        start_col = self.head_idx * self.head_dim
        end_col = (self.head_idx + 1) * self.head_dim
        return start_col, end_col
    
    @property
    def target(self) -> AblationTarget:
        """Get the ablation target specification."""
        return AblationTarget(self.layer_idx, self.head_idx)
    
    @property
    def is_ablated(self) -> bool:
        """Check if this head is currently ablated."""
        return self._is_ablated
    
    def ablate(self, scale: float = 0.0):
        """
        Scale the o_proj slice for this head.
        
        Args:
            scale: Factor to multiply the slice by. 0.0 = full ablation (zero),
                   0.5 = half strength, 1.0 = no change (baseline).
        
        Saves the original slice so it can be restored later.
        Uses proper torch.no_grad() and .zero_() instead of .data.
        """
        if self._is_ablated:
            print(f"Warning: {self.target} already ablated")
            return
        
        o_proj = self._get_o_proj()
        start_col, end_col = self._get_slice_indices()
        
        # Print pre-ablation stats
        slice_before = o_proj.weight[:, start_col:end_col]
        norm_before = torch.norm(slice_before).item()
        full_norm_before = torch.norm(o_proj.weight).item()
        
        # Save original weights and scale the slice
        with torch.no_grad():
            self._original_slice = o_proj.weight[:, start_col:end_col].clone()
            o_proj.weight[:, start_col:end_col].mul_(scale)
        
        # Print post-ablation stats
        slice_after = o_proj.weight[:, start_col:end_col]
        norm_after = torch.norm(slice_after).item()
        full_norm_after = torch.norm(o_proj.weight).item()
        
        self._is_ablated = True
        self._scale = scale
        action = "zeroed" if scale == 0.0 else f"scaled by {scale}"
        print(f"Ablated {self.target}: {action} o_proj.weight[:, {start_col}:{end_col}]")
        print(f"  Slice norm: {norm_before:.4f} -> {norm_after:.4f}")
        print(f"  Full o_proj norm: {full_norm_before:.4f} -> {full_norm_after:.4f}")
    
    def restore(self):
        """
        Restore the original o_proj slice.
        
        Must have called ablate() first.
        Uses proper torch.no_grad() instead of .data.
        """
        if not self._is_ablated:
            print(f"Warning: {self.target} not ablated, nothing to restore")
            return
        
        if self._original_slice is None:
            raise RuntimeError(f"No saved slice for {self.target}")
        
        o_proj = self._get_o_proj()
        start_col, end_col = self._get_slice_indices()
        
        # Restore original weights
        with torch.no_grad():
            o_proj.weight[:, start_col:end_col].copy_(self._original_slice)
        
        # Verify restoration
        restored_norm = torch.norm(o_proj.weight[:, start_col:end_col]).item()
        original_norm = torch.norm(self._original_slice).item()
        
        self._is_ablated = False
        self._original_slice = None
        print(f"Restored {self.target}: norm {restored_norm:.4f} (original was {original_norm:.4f})")


class MultiHeadAblator:
    """
    Ablate multiple heads simultaneously.
    
    Useful for ablating control heads alongside target heads.
    """
    
    def __init__(self, model: nn.Module, head_dim: int = 256):
        """
        Initialize multi-head ablator.
        
        Args:
            model: HuggingFace Gemma2ForCausalLM model
            head_dim: Dimension per head (256 for Gemma-2-2B)
        """
        self.model = model
        self.head_dim = head_dim
        self._ablators: List[HeadAblator] = []
    
    def add_target(self, layer_idx: int, head_idx: int) -> 'MultiHeadAblator':
        """Add a head to the ablation set."""
        ablator = HeadAblator(self.model, layer_idx, head_idx, self.head_dim)
        self._ablators.append(ablator)
        return self
    
    def ablate_all(self):
        """Ablate all added heads."""
        for ablator in self._ablators:
            ablator.ablate()
    
    def restore_all(self):
        """Restore all ablated heads."""
        for ablator in self._ablators:
            if ablator.is_ablated:
                ablator.restore()
    
    @property
    def targets(self) -> List[AblationTarget]:
        """Get all ablation targets."""
        return [a.target for a in self._ablators]


@contextmanager
def ablate_head(model: nn.Module, layer_idx: int, head_idx: int, head_dim: int = 256):
    """
    Context manager for temporary head ablation.
    
    Usage:
        with ablate_head(model, layer_idx=23, head_idx=1):
            outputs = model.generate(...)
        # Head automatically restored here
    """
    ablator = HeadAblator(model, layer_idx, head_idx, head_dim)
    ablator.ablate()
    try:
        yield ablator
    finally:
        ablator.restore()


@contextmanager
def ablate_heads(model: nn.Module, targets: List[Tuple[int, int]], head_dim: int = 256):
    """
    Context manager for temporary multi-head ablation.
    
    Args:
        model: HuggingFace model
        targets: List of (layer_idx, head_idx) tuples
        head_dim: Dimension per head
    
    Usage:
        with ablate_heads(model, [(23, 1), (21, 0)]):
            outputs = model.generate(...)
        # All heads automatically restored here
    """
    multi = MultiHeadAblator(model, head_dim)
    for layer_idx, head_idx in targets:
        multi.add_target(layer_idx, head_idx)
    
    multi.ablate_all()
    try:
        yield multi
    finally:
        multi.restore_all()


# =============================================================================
# Verification Utilities
# =============================================================================

def verify_ablation(model: nn.Module, layer_idx: int, head_idx: int, head_dim: int = 256):
    """
    Verify that a head's o_proj slice is zeroed.
    
    Returns True if the slice is all zeros.
    """
    o_proj = model.model.layers[layer_idx].self_attn.o_proj
    start_col = head_idx * head_dim
    end_col = (head_idx + 1) * head_dim
    
    slice_tensor = o_proj.weight.data[:, start_col:end_col]
    is_zero = torch.all(slice_tensor == 0).item()
    
    return is_zero


def get_head_norm(model: nn.Module, layer_idx: int, head_idx: int, head_dim: int = 256) -> float:
    """
    Get the Frobenius norm of a head's o_proj slice.
    
    Useful for verifying ablation (norm should be 0 after ablation).
    """
    o_proj = model.model.layers[layer_idx].self_attn.o_proj
    start_col = head_idx * head_dim
    end_col = (head_idx + 1) * head_dim
    
    slice_tensor = o_proj.weight.data[:, start_col:end_col]
    norm = torch.norm(slice_tensor).item()
    
    return norm


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    print("Head Ablation Module")
    print("=" * 40)
    print("\nFor Gemma-2-2B:")
    print("  - 26 layers (0-25)")
    print("  - 8 query heads per layer (0-7)")
    print("  - head_dim = 256")
    print("  - o_proj.weight shape: [2304, 2048]")
    print("\nAblation method: Zero o_proj.weight[:, h*256:(h+1)*256]")
    print("\nUsage:")
    print("  from head_ablation import ablate_head")
    print("  with ablate_head(model, layer_idx=23, head_idx=1):")
    print("      outputs = model.generate(...)")
