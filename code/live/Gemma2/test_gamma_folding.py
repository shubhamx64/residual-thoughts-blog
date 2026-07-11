"""
Parity test for RMSNorm gamma folding (item 1 fix).

Verifies the algebraic identity (no forward pass needed):

    q_proj(input_layernorm(x))  ==  (x / rms(x)) @ W_Q_folded.T

where W_Q_folded = W_Q * (1 + gamma), matching HF Gemma2RMSNorm:
    y = (x * rsqrt(mean(x^2) + 1e-6)) * (1.0 + weight)

Also prints, for contrast, the cosine obtained with the LEGACY folding
(W * gamma, missing the +1) -- this quantifies the magnitude of the
pre-fix bug per layer for the blog errata.

Usage:
    python test_gamma_folding.py                  # CPU, fp32 (~10 GB RAM)
    python test_gamma_folding.py --device cuda
    python test_gamma_folding.py --report-b-delta # also compare B matrices
                                                  # legacy vs fixed (needs SAE
                                                  # download via sae-lens)
"""
import argparse
import sys

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM

from config import GEMMA2_CONFIG
from weight_extraction import extract_layer_weights

TEST_LAYERS = [0, 6, 12, 20, 25]
SCALES = [1.0, 12.0, 100.0]  # span depth-dependent residual norms
REL_ERR_TOL = 1e-4
COSINE_TOL = 0.99999
SEED = 42


def flat_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()


def check_layer(model, layer_idx: int, device: str) -> bool:
    """Check Q/K/V folding identity for one layer. Returns True if all pass."""
    cfg = GEMMA2_CONFIG
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn

    weights_fixed = extract_layer_weights(
        model, layer_idx, cfg, fold_gamma=True,
        device=device, dtype=torch.float32, fold_mode="one_plus_gamma",
    )
    weights_legacy = extract_layer_weights(
        model, layer_idx, cfg, fold_gamma=True,
        device=device, dtype=torch.float32, fold_mode="legacy_gamma",
    )

    g = torch.Generator(device="cpu").manual_seed(SEED + layer_idx)
    all_pass = True
    legacy_cosines = []

    for scale in SCALES:
        x = (torch.randn(16, cfg.hidden_size, generator=g) * scale).to(
            device=device, dtype=torch.float32
        )

        with torch.no_grad():
            ln = layer.input_layernorm(x)  # HF: (x/rms)*(1+gamma), float32 internally
            q_ref = attn.q_proj(ln).view(16, cfg.num_attention_heads, cfg.head_dim)
            k_ref = attn.k_proj(ln).view(16, cfg.num_key_value_heads, cfg.head_dim)
            v_ref = attn.v_proj(ln).view(16, cfg.num_key_value_heads, cfg.head_dim)

        # Prediction from folded weights: only the 1/rms factor remains unfolded
        inv_rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        x_n = x * inv_rms

        for name, W_fixed, W_legacy, ref in [
            ("Q", weights_fixed.W_Q, weights_legacy.W_Q, q_ref),
            ("K", weights_fixed.W_K, weights_legacy.W_K, k_ref),
            ("V", weights_fixed.W_V, weights_legacy.W_V, v_ref),
        ]:
            n_heads = W_fixed.shape[0]
            for h in range(n_heads):
                pred = x_n @ W_fixed[h].T
                rel_err = (pred - ref[:, h]).abs().max().item() / (
                    ref[:, h].abs().max().item() + 1e-8
                )
                cos = flat_cosine(pred, ref[:, h])
                ok = rel_err < REL_ERR_TOL and cos > COSINE_TOL
                all_pass &= ok
                if not ok:
                    print(
                        f"  FAIL L{layer_idx} {name}[{h}] scale={scale}: "
                        f"rel_err={rel_err:.2e} cos={cos:.6f}"
                    )

                legacy_cosines.append(flat_cosine(x_n @ W_legacy[h].T, ref[:, h]))

    legacy_cos_mean = sum(legacy_cosines) / len(legacy_cosines)
    status = "PASS" if all_pass else "FAIL"
    print(
        f"Layer {layer_idx:2d}: {status} (fixed folding exact); "
        f"legacy W*gamma cosine vs truth: {legacy_cos_mean:.4f}"
    )
    return all_pass


def report_b_delta(model, device: str, n_features: int = 512):
    """Compare B matrices under legacy vs fixed folding (bug-impact quantifier)."""
    from sae_utils import prepare_sae_features, project_to_qk_space
    from qk_routing import compute_affinity_matrix, compute_routing_metrics
    from weight_extraction import get_qkv_for_head

    cfg = GEMMA2_CONFIG
    print("\n[--report-b-delta] cos(B_legacy, B_fixed) and metric deltas per head")
    header = (
        f"{'head':>8} {'cos(B)':>8} {'top1 leg':>9} {'top1 fix':>9} "
        f"{'diag leg':>9} {'diag fix':>9}"
    )

    for layer_idx in TEST_LAYERS:
        sae_layer = cfg.get_sae_layer_for_attn(layer_idx)
        feats = prepare_sae_features(
            sae_layer, cfg, subset_size=n_features, seed=cfg.seed,
            device=device, dtype=torch.float32, compute_gram=False,
        )
        D = feats.decoder_subset
        idx = torch.arange(D.shape[0], device=D.device)

        w_fixed = extract_layer_weights(
            model, layer_idx, cfg, fold_gamma=True,
            device=device, dtype=torch.float32, fold_mode="one_plus_gamma",
        )
        w_legacy = extract_layer_weights(
            model, layer_idx, cfg, fold_gamma=True,
            device=device, dtype=torch.float32, fold_mode="legacy_gamma",
        )

        print(f"\nLayer {layer_idx} (SAE layer {sae_layer}):")
        print(header)
        for h in range(cfg.num_attention_heads):
            W_Q_f, W_K_f, _ = get_qkv_for_head(w_fixed, h, cfg)
            W_Q_l, W_K_l, _ = get_qkv_for_head(w_legacy, h, cfg)

            Q_f, K_f = project_to_qk_space(D, W_Q_f, W_K_f)
            B_fixed = compute_affinity_matrix(Q_f, K_f, cfg)
            Q_l, K_l = project_to_qk_space(D, W_Q_l, W_K_l)
            B_legacy = compute_affinity_matrix(Q_l, K_l, cfg)

            m_fixed = compute_routing_metrics(B_fixed, idx)
            m_legacy = compute_routing_metrics(B_legacy, idx)

            print(
                f"  L{layer_idx}H{h:<3} {flat_cosine(B_legacy, B_fixed):8.4f} "
                f"{m_legacy.top1_mass_mean:9.4f} {m_fixed.top1_mass_mean:9.4f} "
                f"{m_legacy.diagonal_softmax_mass:9.4f} {m_fixed.diagonal_softmax_mass:9.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--report-b-delta", action="store_true",
                        help="Also compare B matrices legacy vs fixed (needs SAE)")
    args = parser.parse_args()

    print(f"transformers version: {transformers.__version__}")
    print("Loading google/gemma-2-2b (fp32, no inference needed)...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        torch_dtype=torch.float32,
        device_map=args.device,
        attn_implementation="eager",
    )
    model.eval()

    print("\n" + "=" * 70)
    print("GAMMA FOLDING PARITY: q_proj(LN(x)) vs (x/rms) @ W_folded.T")
    print(f"Pass: rel_err < {REL_ERR_TOL}, cosine > {COSINE_TOL}")
    print("=" * 70)

    all_pass = True
    for layer_idx in TEST_LAYERS:
        all_pass &= check_layer(model, layer_idx, args.device)

    if args.report_b_delta:
        report_b_delta(model, args.device)

    print("\n" + ("ALL LAYERS PASS" if all_pass else "FAILURES DETECTED"))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
