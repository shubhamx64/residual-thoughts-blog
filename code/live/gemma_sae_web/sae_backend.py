# sae_backend.py
from typing import Dict, Any, List
import torch
import numpy as np
from sae_lens import SAE  # assumes sae-lens installed

from config import SAE_REPO_ID, DEFAULT_LAYER_INDEX
from model_backend import run_model_get_layer_hidden, get_tokenizer_and_model

_sae_cache: Dict[int, SAE] = {}


def get_sae_for_layer(layer_index: int) -> SAE:
    """
    Load (and cache) the Matryoshka SAE for a given layer.
    layer_index: index in HF hidden_states ordering, not necessarily SAE naming.
    You may need to adjust the mapping if the SAE repo uses hook names instead.
    """
    if layer_index in _sae_cache:
        return _sae_cache[layer_index]

    # For a clean interface, assume one SAE per "hook name" like blocks.{L}.hook_resid_post
    # Example mapping: hidden_states index k -> "blocks.{k-1}.hook_resid_post"
    hook_name = f"blocks.{layer_index - 1}.hook_resid_post"
    sae, cfg, sparsity = SAE.from_pretrained(
        SAE_REPO_ID,
        hook_name,
    )
    sae.to(next(get_tokenizer_and_model()[1].parameters()).device)
    sae.eval()
    _sae_cache[layer_index] = sae
    return sae


@torch.inference_mode()
def analyze_prompt_with_sae(
    prompt: str,
    layer_index: int = DEFAULT_LAYER_INDEX,
    top_k_features: int = 20,
) -> Dict[str, Any]:
    """
    1. Run Gemma-3-1B on `prompt` and get hidden states at `layer_index`.
    2. Run Matryoshka SAE on those activations.
    3. Return top-k features by max activation across tokens, plus per-token activations.
    """
    hs, _ = run_model_get_layer_hidden(prompt, layer_index)
    seq_len, hidden_dim = hs.shape
    flat = hs.reshape(seq_len, hidden_dim)  # [T, D]

    sae = get_sae_for_layer(layer_index)
    # API may be sae.encode(flat) or sae(flat). Adjust if needed.
    z = sae.encode(flat)  # [T, width]
    z_cpu = z.detach().cpu().numpy()

    # max activation per feature across tokens
    max_per_feat = z_cpu.max(axis=0)
    # indices of top-k features
    k = min(top_k_features, max_per_feat.shape[0])
    top_idx = np.argsort(-max_per_feat)[:k]
    top_vals = max_per_feat[top_idx]

    # prepare token view
    tok, _ = get_tokenizer_and_model()
    tokens = tok.convert_ids_to_tokens(
        tok(prompt, return_tensors="pt")["input_ids"][0]
    )

    top_features: List[Dict[str, Any]] = []
    for rank, (f_idx, max_val) in enumerate(zip(top_idx.tolist(), top_vals.tolist())):
        # per-token activations for this feature
        per_tok = z_cpu[:, f_idx].tolist()
        top_features.append(
            {
                "rank": rank + 1,
                "feature_index": int(f_idx),
                "max_activation": float(max_val),
                "per_token_activations": per_tok,
            }
        )

    return {
        "layer_index": int(layer_index),
        "hidden_dim": int(hidden_dim),
        "num_tokens": int(seq_len),
        "tokens": tokens,
        "top_features": top_features,
    }
