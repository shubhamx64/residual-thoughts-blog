"""
Parity test for RoPE rotation convention (item 2 fix).

Verifies our weight-space rotation against HF's actual Gemma-2 RoPE
(rotate_half / half-split convention, pairs (i, i + head_dim/2)):

    ground truth:  logit(t, s) = (R(t) q) . (R(s) k)   via apply_rotary_pos_emb
    ours:          Q @ R(-Delta) @ K.T                  with Delta = t - s

Three code paths are checked against the ground truth:
    1. apply_rope_rotation_to_vectors(K, -delta, freqs) then Q @ K_rot.T
    2. compute_rotated_affinity_matrix(Q, K, delta, freqs, scale=1.0)
    3. Q @ compute_rotation_matrix(-delta, freqs) @ K.T

Plus: position-invariance (same Delta, different absolute positions must give
identical logits) and a negative control (the OLD interleaved formula must
NOT match -- guards against testing a tautology).

CPU-only, no model weights: builds Gemma2RotaryEmbedding from config alone.

Usage:
    python test_rope_parity.py              # offline-constructed config
    python test_rope_parity.py --hub-config # cross-check with hub config
"""
import argparse
import sys

import torch
import transformers
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2RotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config as HFGemma2Config

from config import GEMMA2_CONFIG
from rope_utils import (
    compute_rope_frequencies,
    apply_rope_rotation_to_vectors,
    compute_rotated_affinity_matrix,
    compute_rotation_matrix,
)

DELTAS = [1, 2, 4, 8, 16, 64, 256, 1024]
N_VECS = 64
ABS_TOL = 1e-3
COSINE_TOL = 0.999999
SEED = 42


def hf_rotate_at_position(vectors: torch.Tensor, pos: int, rot) -> torch.Tensor:
    """Rotate [N, head_dim] vectors at absolute position `pos` via HF code."""
    x4 = vectors.view(1, vectors.shape[0], 1, vectors.shape[1])  # [1, N(heads), 1, d]
    position_ids = torch.tensor([[pos]])
    cos, sin = rot(x4, position_ids)  # [1, 1, head_dim]
    q_rot, _ = apply_rotary_pos_emb(x4, x4, cos, sin)
    return q_rot.view(vectors.shape)


def old_interleaved_rotation(vectors: torch.Tensor, delta: int, freqs: torch.Tensor) -> torch.Tensor:
    """The PRE-FIX interleaved formula (pairs (2i, 2i+1)) -- negative control."""
    angles = (delta * freqs).to(dtype=vectors.dtype)
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    x = vectors.view(vectors.shape[0], -1, 2)
    x0, x1 = x[..., 0], x[..., 1]
    x_rot = torch.stack([x0 * cos_a - x1 * sin_a, x0 * sin_a + x1 * cos_a], dim=-1)
    return x_rot.view(vectors.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub-config", action="store_true",
                        help="Load config from the hub instead of constructing offline")
    args = parser.parse_args()

    print(f"transformers version: {transformers.__version__}")

    if args.hub_config:
        from transformers import AutoConfig
        hf_cfg = AutoConfig.from_pretrained("google/gemma-2-2b")
    else:
        hf_cfg = HFGemma2Config(
            head_dim=GEMMA2_CONFIG.head_dim,
            rope_theta=GEMMA2_CONFIG.rope_theta,
            max_position_embeddings=GEMMA2_CONFIG.max_position_embeddings,
            hidden_size=GEMMA2_CONFIG.hidden_size,
            num_attention_heads=GEMMA2_CONFIG.num_attention_heads,
            num_key_value_heads=GEMMA2_CONFIG.num_key_value_heads,
        )

    rot = Gemma2RotaryEmbedding(config=hf_cfg)

    # Sanity: scaling and frequencies must match our weight-space assumptions
    assert getattr(rot, "attention_scaling", 1.0) == 1.0, \
        f"attention_scaling={rot.attention_scaling}, expected 1.0"
    freqs = compute_rope_frequencies(GEMMA2_CONFIG)
    assert torch.allclose(rot.inv_freq.float(), freqs, atol=1e-6), \
        "inv_freq mismatch between HF rotary embedding and compute_rope_frequencies"
    print("attention_scaling == 1.0 and inv_freq match: OK")

    g = torch.Generator().manual_seed(SEED)
    Q = torch.randn(N_VECS, GEMMA2_CONFIG.head_dim, generator=g)
    K = torch.randn(N_VECS, GEMMA2_CONFIG.head_dim, generator=g)

    all_pass = True
    for delta in DELTAS:
        positions = [delta, delta + 7, delta + 100, 4096]
        refs = []
        for p in positions:
            q_rot = hf_rotate_at_position(Q, p, rot)          # query at t = p
            k_rot = hf_rotate_at_position(K, p - delta, rot)  # key at s = p - delta
            refs.append(q_rot @ k_rot.T)

        # Position invariance: logits depend only on delta.
        # Tolerance is looser than ABS_TOL because HF computes cos/sin in
        # float32 at LARGE absolute positions (up to 4096 rad), where fp32
        # trig loses ~1e-3 of logit precision. This is HF's own numerical
        # noise, not a convention error (the interleaved control differs
        # by ~1e+1, four orders of magnitude more).
        POS_INV_TOL = 5e-3
        pos_diff = max(
            (refs[i] - refs[j]).abs().max().item()
            for i in range(len(refs)) for j in range(i + 1, len(refs))
        )
        ref = refs[0]

        # Path 1: rotate keys by -delta, dot with queries
        pred1 = Q @ apply_rope_rotation_to_vectors(K, -delta, freqs).T
        # Path 2: the affinity function (passes positive delta; -delta internal)
        pred2 = compute_rotated_affinity_matrix(Q, K, delta, freqs, scale=1.0)
        # Path 3: explicit rotation matrix
        pred3 = Q @ compute_rotation_matrix(-delta, freqs) @ K.T

        diffs = [(p - ref).abs().max().item() for p in (pred1, pred2, pred3)]
        cosines = [
            torch.nn.functional.cosine_similarity(
                p.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)
            ).item()
            for p in (pred1, pred2, pred3)
        ]

        # Negative control: old interleaved formula must NOT match
        pred_old = Q @ old_interleaved_rotation(K, -delta, freqs).T
        old_diff = (pred_old - ref).abs().max().item()

        ok = (
            max(diffs) < ABS_TOL
            and min(cosines) > COSINE_TOL
            and pos_diff < POS_INV_TOL
            and old_diff > 1e-2
        )
        all_pass &= ok
        status = "PASS" if ok else "FAIL"
        print(
            f"delta={delta:5d}: {status}  max_diff={max(diffs):.2e} "
            f"min_cos={min(cosines):.8f} pos_invariance={pos_diff:.2e} "
            f"interleaved_diff={old_diff:.2e} (must be large)"
        )

    print("\n" + ("ALL DELTAS PASS" if all_pass else "FAILURES DETECTED"))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
