import os
import torch
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM

from config import GEMMA2_CONFIG
from sae_utils import SAEManager, project_to_qk_space
from weight_extraction import WeightExtractor
from qk_routing import compute_affinity_matrix
from ov_writing import analyze_head_writing

DEVICE = "cuda"
DTYPE = torch.float16

def save_heatmap(M: torch.Tensor, out_path: str, title: str = "", quantile: float = 0.99):
    M = M.detach().float().cpu()
    vmax = torch.quantile(M.abs(), quantile).item()
    plt.figure(figsize=(4, 4))
    plt.imshow(M, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def reorder_by_argmax(M: torch.Tensor) -> torch.Tensor:
    # Cosmetic: makes stripes / diagonal pop more clearly in small “toy” crops
    order = torch.argsort(torch.argmax(M, dim=1))
    return M[order][:, order]

@torch.no_grad()
def compute_B_matrix(model, sae_manager: SAEManager, weight_extractor: WeightExtractor,
                     layer: int, head: int, subset_size: int = 128, seed: int = 0) -> torch.Tensor:
    # SAE features (note: SAE layer might be offset from attn layer via config)
    sae_manager = SAEManager(GEMMA2_CONFIG, subset_size=subset_size, seed=seed)
    sae_layer = GEMMA2_CONFIG.get_sae_layer_for_attn(layer)
    sae_features = sae_manager.get_features(sae_layer, device=DEVICE, dtype=DTYPE, compute_gram=False)

    # Weights
    layer_w = weight_extractor.get_layer(layer, device=DEVICE, dtype=DTYPE)

    # Pull correct Q head + K group
    kv_group = GEMMA2_CONFIG.query_to_kv_group(head)
    W_Q = layer_w.W_Q[head]
    W_K = layer_w.W_K[kv_group]

    # Project features into Q/K spaces and compute B
    Q_f, K_f = project_to_qk_space(sae_features.decoder_subset, W_Q, W_K)
    B = compute_affinity_matrix(Q_f, K_f, config=GEMMA2_CONFIG)
    return B

@torch.no_grad()
def compute_W2F(model, sae_manager: SAEManager, weight_extractor: WeightExtractor,
                layer: int, head: int, subset_size: int = 128, seed: int = 0):
    sae_manager = SAEManager(GEMMA2_CONFIG, subset_size=subset_size, seed=seed)
    sae_layer = GEMMA2_CONFIG.get_sae_layer_for_attn(layer)
    sae_features = sae_manager.get_features(sae_layer, device=DEVICE, dtype=DTYPE, compute_gram=False)

    layer_w = weight_extractor.get_layer(layer, device=DEVICE, dtype=DTYPE)

    res = analyze_head_writing(
        sae_features=sae_features,
        layer_weights=layer_w,
        query_head=head,
        config=GEMMA2_CONFIG,
        store_matrix=True,   # <— this is what you want for a heatmap
        device=DEVICE,
        dtype=DTYPE,
    )
    W2F = res.write_to_feature_matrix
    return res.metrics, W2F

def main():
    os.makedirs("figs", exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        GEMMA2_CONFIG.model_id,
        torch_dtype=DTYPE,
        device_map="auto",
    )
    weight_extractor = WeightExtractor(model, GEMMA2_CONFIG)
    sae_manager = SAEManager(GEMMA2_CONFIG)

    # --- FIGURE 1 heads (B heatmaps) ---
    fig1 = {
        "a_diag_SELF_MATCH_L10H5": (10, 5),
        "b_offdiag_selective_L15H0": (15, 0),
        "c_repulsion_L6H3": (6, 3),
    }

    for name, (L, H) in fig1.items():
        B = compute_B_matrix(model, sae_manager, weight_extractor, L, H, subset_size=128, seed=0)
        B = reorder_by_argmax(B)
        save_heatmap(B, f"figs/FIG1_{name}.png", title=f"B heatmap (L{L}H{H})")

    # --- FIGURE 2 heads (W2F heatmaps) ---
    # Transform + Broadcast exemplars from leaderboard:
    #   TRANSFORM: L3H2
    #   BROADCAST: L20H0
    # COPY: you should choose this by inspecting printed copy_dominance
    fig2 = {
        "transform_L3H2": (3, 2),
        "broadcast_L20H0": (20, 0),
        "copy_L2H2": (2, 2),  # verify copy_dominance before calling it COPY in the post
    }

    for name, (L, H) in fig2.items():
        metrics, W2F = compute_W2F(model, sae_manager, weight_extractor, L, H, subset_size=128, seed=0)

        print(f"[{name}] L{L}H{H}  copy_score={metrics.copy_score:.3f}  copy_dom={metrics.copy_dominance:.3f} "
              f"transform={metrics.transform_score:.3f}  broadcast={metrics.broadcast_score:.3f}")

        W2F = reorder_by_argmax(W2F)
        save_heatmap(W2F, f"figs/FIG2_{name}.png", title=f"W2F heatmap (L{L}H{H})")

if __name__ == "__main__":
    main()
