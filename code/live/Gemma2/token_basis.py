"""
Token-embedding probe basis for Gemma-2 weight-space analysis.

Uses the tied embedding/unembedding matrix (model.model.embed_tokens.weight,
[256000, 2304]) as an alternative probe basis D in place of Gemma Scope SAE
decoders. The SAME basis applies at every layer (no SAE layer offset), which
also makes cross-layer comparisons basis-stable.

This is the QK/OV circuit analysis of Elhage et al. (2021) "A Mathematical
Framework for Transformer Circuits" (W_E W_Q^T W_K W_E^T), extended with
RoPE, softcap, GQA, and activation validation.

Scale note: Gemma multiplies token embeddings by sqrt(hidden_size) ~ 48 at
model input (modeling_gemma2.py). Directions are unchanged and rows are
RMS-calibrated to L2 = sqrt(hidden_size) exactly like
sae_utils.prepare_sae_features, so this scalar is irrelevant here.

Anisotropy note: token embeddings share a large mean component; without
mean-centering, every Gram entry (and every diagonal-affinity metric) is
inflated by it. center=True (default) removes the mean over the FILTERED
vocab (stable across subset seeds).
"""

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

from config import Gemma2Config, GEMMA2_CONFIG
from sae_utils import SAEFeatures

VOCAB_MASK_CACHE = Path(__file__).parent / "token_basis_vocab_mask.pt"


def filter_vocab_ids(tokenizer, require_alpha: bool = True, use_cache: bool = True) -> torch.Tensor:
    """
    Token ids usable as probe directions.

    Keeps ids whose decoded string is non-empty after strip and printable,
    and (if require_alpha) contains at least one alphanumeric character.
    Excludes special tokens, byte-fallback tokens ("<0xNN>"), and
    "<unusedN>" placeholder tokens, whose embeddings are near-init noise.

    The 256k-token decode pass takes ~a minute; the boolean mask is cached
    to token_basis_vocab_mask.pt (keyed by require_alpha).
    """
    cache_key = f"require_alpha={require_alpha}"
    if use_cache and VOCAB_MASK_CACHE.exists():
        try:
            cached = torch.load(VOCAB_MASK_CACHE)
            if cache_key in cached and cached.get("vocab_size") == tokenizer.vocab_size:
                return cached[cache_key].nonzero(as_tuple=True)[0]
        except Exception:
            pass  # stale/corrupt cache: rebuild

    vocab_size = tokenizer.vocab_size
    special_ids = set(tokenizer.all_special_ids)

    # Raw token strings (fast, single call) for structural exclusions
    raw_tokens = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))

    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for i, raw in enumerate(raw_tokens):
        if i in special_ids:
            continue
        if raw.startswith("<unused") or (
            raw.startswith("<0x") and raw.endswith(">") and len(raw) == 6
        ):
            continue
        # Decoded surface form (handles sentencepiece markers)
        s = tokenizer.decode([i])
        stripped = s.strip()
        if not stripped or not stripped.isprintable():
            continue
        if require_alpha and not any(c.isalnum() for c in stripped):
            continue
        mask[i] = True

    if use_cache:
        cached = {}
        if VOCAB_MASK_CACHE.exists():
            try:
                cached = torch.load(VOCAB_MASK_CACHE)
            except Exception:
                cached = {}
        cached[cache_key] = mask
        cached["vocab_size"] = vocab_size
        torch.save(cached, VOCAB_MASK_CACHE)

    return mask.nonzero(as_tuple=True)[0]


def prepare_token_features(
    model,
    tokenizer,
    config: Gemma2Config = GEMMA2_CONFIG,
    subset_size: int = 4096,
    seed: int = 42,
    center: bool = True,
    require_alpha: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    compute_gram: bool = True,
) -> SAEFeatures:
    """
    Build a token-embedding probe basis packaged as SAEFeatures, so
    qk_routing.analyze_head_routing / ov_writing.analyze_head_writing work
    unchanged (they only consume decoder_subset / feature_indices /
    gram_matrix).

    feature_indices are TOKEN IDS (label via get_token_labels).
    """
    E = model.model.embed_tokens.weight.detach().float().cpu()  # [vocab, hidden]
    hidden_size = E.shape[1]

    # Informational: Gemma-2 ties embed/unembed, so this basis is both
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        print("[token_basis] note: embed_tokens and lm_head are NOT tied in this model")

    ids = filter_vocab_ids(tokenizer, require_alpha=require_alpha)

    # Mean over the FILTERED vocab (stable across subset seeds)
    mu = E[ids].mean(dim=0) if center else torch.zeros(hidden_size)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(ids.shape[0], generator=g)[: min(subset_size, ids.shape[0])]
    ids_subset = ids[perm]

    sub = E[ids_subset] - mu  # [subset, hidden]
    norms = sub.norm(dim=1)

    # RMS calibration: L2 = sqrt(hidden_size), matching prepare_sae_features
    calibrated = F.normalize(sub, dim=1) * (hidden_size ** 0.5)
    calibrated = calibrated.to(device=device, dtype=dtype)

    gram = calibrated @ calibrated.T if compute_gram else None

    return SAEFeatures(
        layer_idx=-1,  # same basis at every layer
        decoder_full=calibrated,  # NOT the 256k matrix (2.4 GB); subset only
        decoder_normalized=calibrated,
        feature_indices=ids_subset.to(device=device),
        decoder_subset=calibrated,
        gram_matrix=gram,
        decoder_norms=norms,
    )


def get_token_labels(feature_indices: torch.Tensor, tokenizer) -> List[str]:
    """Free labels: feature index IS a token id."""
    return [repr(tokenizer.decode([int(i)])) for i in feature_indices]


class TokenBasisManager:
    """
    Drop-in replacement for SAEManager where the validator/pipeline needs it.

    - get_features(layer_idx, ...) returns the SAME cached basis for every
      layer (token basis has no layer offset).
    - encode(residual, layer_idx) returns dense logit-lens coefficients
      residual @ D_subset.T, already subset-sized [seq, subset_size].
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: Gemma2Config = GEMMA2_CONFIG,
        subset_size: int = 4096,
        seed: int = 42,
        center: bool = True,
        require_alpha: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.subset_size = subset_size
        self.seed = seed
        self.center = center
        self.require_alpha = require_alpha
        self._features: Optional[SAEFeatures] = None

    def get_features(
        self,
        layer_idx: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        compute_gram: bool = True,
    ) -> SAEFeatures:
        if self._features is None:
            self._features = prepare_token_features(
                self.model, self.tokenizer, self.config,
                subset_size=self.subset_size, seed=self.seed,
                center=self.center, require_alpha=self.require_alpha,
                device=device, dtype=dtype, compute_gram=compute_gram,
            )
        return self._features

    def encode(self, residual: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Dense logit-lens coefficients: [seq, subset_size]."""
        feats = self.get_features(layer_idx)
        D = feats.decoder_subset.to(device=residual.device, dtype=torch.float32)
        return residual.float() @ D.T

    def feature_labels(self) -> List[str]:
        feats = self.get_features(0)
        return get_token_labels(feats.feature_indices, self.tokenizer)

    def clear_cache(self):
        self._features = None


if __name__ == "__main__":
    print("Token basis module loaded.")
    print("Basis: tied embed/unembed matrix, RMS-calibrated, optional centering.")
