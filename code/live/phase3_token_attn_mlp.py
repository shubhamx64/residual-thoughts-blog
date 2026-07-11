#!/usr/bin/env python
"""
Phase 3 – Intra-token, intra-layer update decomposition for Gemma.

UPDATE VIEW:
  For each layer l and token t, we decompose the block update into:

    h_in[l, t]   = residual before attention (layer input)
    h_att[l, t]  = residual after attention sublayer (before MLP)
    h_out[l, t]  = residual after MLP (layer output)

  And the corresponding updates:

    Δ_att[l, t]   = h_att[l, t] - h_in[l, t]
    Δ_mlp[l, t]   = h_out[l, t] - h_att[l, t]
    Δ_total[l, t] = h_out[l, t] - h_in[l, t] = Δ_att + Δ_mlp

We measure:

  Norms:
    na = ||Δ_att||, nm = ||Δ_mlp||, nt = ||Δ_total||

  Angles:
    θ_att_input   = angle(h_in, Δ_att)
    θ_mlp_input   = angle(h_in, Δ_mlp)
    θ_att_mlp     = angle(Δ_att, Δ_mlp)
    θ_total_input = angle(h_in, Δ_total)

  Ratios:
    r_att = na / (na + nm)
    r_mlp = nm / (na + nm)

We aggregate over tokens and prompts per layer and per family and plot:

  - Norm curves vs depth (na, nm, nt, r_att, r_mlp)
  - Angle curves vs depth (θ_att_mlp, θ_total_input, etc.)
  - Histograms of θ_att_mlp at early/mid/late layers
  - Scatter plots of (na, nm) for early/mid/late layers

Notes / assumptions:

  - Model is a Gemma-style decoder-only LM (e.g., google/gemma-3-1b-it).
  - Each decoder layer has submodules `self_attn` and `mlp` whose forward
    outputs correspond to the residual updates at that sublayer (before
    residual addition). With standard HF implementations:

        h_att = h_in + self_attn(...)
        h_out = h_att + mlp(...)

    so we can treat:
        Δ_att ≈ self_attn_output
        Δ_mlp ≈ mlp_output

  - We reuse the same "families" pattern as Phase 1 / Phase 4:
      try: prompt_bank.PHASE3_FAMILIES
      else: prompt_bank.PHASE1_FAMILIES
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# -----------------------------
# Prompt families
# -----------------------------
try:
    from prompt_bank import PHASE3_FAMILIES as PHASE3_FAMILIES
except ImportError:
    # Fallback: reuse Phase 1 families
    from prompt_bank import PHASE1_FAMILIES as PHASE3_FAMILIES


# -----------------------------
# Config
# -----------------------------
MODEL_ID = "google/gemma-3-4b-it"
OUTDIR = Path("phase3_layer_update_outputs_4b")
OUTDIR.mkdir(parents=True, exist_ok=True)

MAX_NEW_TOKENS = 256
SEED = 42

# If True, restrict metrics to continuation region only (generation); else use
# full prompt+continuation region.
USE_CONTINUATION_ONLY = False

torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------
# Model loading
# -----------------------------
print(f"[load] Loading model: {MODEL_ID}")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
).to(device)
model.eval()

config = AutoConfig.from_pretrained(MODEL_ID)
text_cfg = getattr(config, "text_config", config)
NUM_LAYERS = text_cfg.num_hidden_layers
HIDDEN_SIZE = text_cfg.hidden_size

print(f"[load] num_layers={NUM_LAYERS}, hidden_size={HIDDEN_SIZE}, device={device}")

# -----------------------------
# Decoder layer discovery & hooks
# -----------------------------
# We locate decoder layers by looking for modules that have both `self_attn`
# and `mlp` attributes (GemmaDecoderLayer-style).
DECODER_LAYERS = []
_seen = set()
for name, module in model.named_modules():
    if hasattr(module, "self_attn") and hasattr(module, "mlp"):
        if module not in _seen:
            DECODER_LAYERS.append(module)
            _seen.add(module)

if len(DECODER_LAYERS) != NUM_LAYERS:
    print(
        f"[warn] Discovered {len(DECODER_LAYERS)} decoder layers by (self_attn, mlp), "
        f"but config.num_hidden_layers={NUM_LAYERS}. Proceeding anyway."
    )

# These caches are populated during each forward pass with hooks.
_LAYER_ATTN_OUTPUTS: Dict[int, np.ndarray] = {}
_LAYER_MLP_OUTPUTS: Dict[int, np.ndarray] = {}


def _register_layer_submodule_hooks():
    """
    Attach forward hooks to each decoder layer's self_attn and mlp submodules.

    We capture:
      - self_attn output (context) as Δ_att
      - mlp output as Δ_mlp

    Returns: list of hook handles to be removed after the forward pass.
    """
    handles = []

    def make_attn_hook(layer_idx: int):
        def hook(module, inputs, output):
            # output can be (attn_output, attn_weights, ...) or a tensor
            out = output[0] if isinstance(output, tuple) else output
            # out: (batch, T, D)
            out = out[0].detach().to(torch.float32).cpu().numpy()
            _LAYERS = _LAYER_ATTN_OUTPUTS  # just for clarity
            _LAYERS[layer_idx] = out
        return hook

    def make_mlp_hook(layer_idx: int):
        def hook(module, inputs, output):
            # output: (batch, T, D)
            out = output[0].detach().to(torch.float32).cpu().numpy()
            _LAYERS = _LAYER_MLP_OUTPUTS
            _LAYERS[layer_idx] = out
        return hook

    for layer_idx, layer in enumerate(DECODER_LAYERS):
        attn_mod = getattr(layer, "self_attn", None)
        mlp_mod = getattr(layer, "mlp", None)
        if attn_mod is not None:
            handles.append(attn_mod.register_forward_hook(make_attn_hook(layer_idx)))
        else:
            print(f"[warn] layer {layer_idx} has no 'self_attn' attr; Δ_att will be NaN.")
        if mlp_mod is not None:
            handles.append(mlp_mod.register_forward_hook(make_mlp_hook(layer_idx)))
        else:
            print(f"[warn] layer {layer_idx} has no 'mlp' attr; Δ_mlp will be NaN.")

    return handles


# -----------------------------
# Run model and capture h_in, h_out, Δ_att, Δ_mlp
# -----------------------------
@torch.inference_mode()
def run_model_and_capture(
    prompts: List[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    """
    For each prompt:
      1) Greedy-generate up to max_new_tokens.
      2) Run a full forward pass on the prompt+completion with hooks to
         capture per-layer self_attn and mlp outputs.

    Returns: List[dict] with keys:
      - 'prompt'
      - 'input_ids': np.ndarray (T,)
      - 'prompt_len': int
      - 'hidden': List[np.ndarray] of shape (T, D), len = NUM_LAYERS+1
                  (index 0 = embedding/pre-first-block,
                   index l+1 = post-block-l residual)
      - 'attn': List[np.ndarray] of shape (T, D), len = NUM_LAYERS
      - 'mlp':  List[np.ndarray] of shape (T, D), len = NUM_LAYERS
    """
    records = []
    for p in prompts:
        enc = tok(p, return_tensors="pt").to(device)
        prompt_len = enc["input_ids"].shape[1]

        # 1) Generate continuation (no hooks)
        gen_ids = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=False,
        )[0]  # (T,)

        # 2) Full forward pass with hooks to capture sublayer outputs
        _LAYER_ATTN_OUTPUTS.clear()
        _LAYER_MLP_OUTPUTS.clear()
        handles = _register_layer_submodule_hooks()

        out = model(
            input_ids=gen_ids.unsqueeze(0),
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )

        for h in handles:
            h.remove()

        hidden_states = [
            h[0].to(torch.float32).cpu().numpy() for h in out.hidden_states
        ]
        if len(hidden_states) != NUM_LAYERS + 1:
            print(
                f"[warn] hidden_states length={len(hidden_states)} != NUM_LAYERS+1={NUM_LAYERS+1}"
            )

        # Convert caches to ordered lists per layer
        attn_list: List[np.ndarray] = []
        mlp_list: List[np.ndarray] = []
        T = hidden_states[0].shape[0]

        for li in range(NUM_LAYERS):
            if li in _LAYER_ATTN_OUTPUTS:
                A = _LAYER_ATTN_OUTPUTS[li]
            else:
                A = np.full((T, HIDDEN_SIZE), np.nan, dtype=np.float32)
                print(f"[warn] no Δ_att captured for layer {li}; filled with NaN.")
            if li in _LAYER_MLP_OUTPUTS:
                M = _LAYER_MLP_OUTPUTS[li]
            else:
                M = np.full((T, HIDDEN_SIZE), np.nan, dtype=np.float32)
                print(f"[warn] no Δ_mlp captured for layer {li}; filled with NaN.")

            attn_list.append(A.astype(np.float32))
            mlp_list.append(M.astype(np.float32))

        records.append(
            {
                "prompt": p,
                "input_ids": gen_ids.cpu().numpy(),
                "prompt_len": int(prompt_len),
                "hidden": hidden_states,  # length NUM_LAYERS+1, arrays (T, D)
                "attn": attn_list,        # length NUM_LAYERS, arrays (T, D)
                "mlp": mlp_list,          # length NUM_LAYERS, arrays (T, D)
            }
        )

    return records


# -----------------------------
# Metrics: norms, angles, ratios
# -----------------------------
METRIC_NAMES = [
    "na",               # ||Δ_att||
    "nm",               # ||Δ_mlp||
    "nt",               # ||Δ_total||
    "theta_att_input",  # angle(h_in, Δ_att)
    "theta_mlp_input",  # angle(h_in, Δ_mlp)
    "theta_att_mlp",    # angle(Δ_att, Δ_mlp)
    "theta_total_input",# angle(h_in, Δ_total)
    "r_att",            # na / (na + nm)
    "r_mlp",            # nm / (na + nm)
]


def _angle_between(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    u, v: (..., D)
    Returns angles in radians as 1D array over tokens, with NaN where norms are tiny.
    """
    u_norm = np.linalg.norm(u, axis=-1)
    v_norm = np.linalg.norm(v, axis=-1)
    mask = (u_norm > eps) & (v_norm > eps)

    angles = np.full(u_norm.shape, np.nan, dtype=np.float32)
    if not mask.any():
        return angles

    u_n = u[mask] / u_norm[mask, None]
    v_n = v[mask] / v_norm[mask, None]
    cosang = np.sum(u_n * v_n, axis=-1)
    cosang = np.clip(cosang, -1.0, 1.0)
    angles[mask] = np.arccos(cosang).astype(np.float32)
    return angles


def compute_layer_update_metrics(
    H_in: np.ndarray,   # (T, D)
    H_out: np.ndarray,  # (T, D)
    A: np.ndarray,      # Δ_att approx, (T, D)
    M: np.ndarray,      # Δ_mlp approx, (T, D)
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """
    Compute Phase-3 metrics for one layer and all tokens in a region.

    Returns dict of 1D arrays of length T for each METRIC_NAMES entry.
    """
    # Δ vectors
    D_att = A
    D_mlp = M
    D_total = H_out - H_in

    # Norms
    na = np.linalg.norm(D_att, axis=-1)
    nm = np.linalg.norm(D_mlp, axis=-1)
    nt = np.linalg.norm(D_total, axis=-1)

    # Ratios
    denom = na + nm
    denom_mask = denom > eps
    r_att = np.full_like(na, np.nan, dtype=np.float32)
    r_mlp = np.full_like(nm, np.nan, dtype=np.float32)
    r_att[denom_mask] = (na[denom_mask] / denom[denom_mask]).astype(np.float32)
    r_mlp[denom_mask] = (nm[denom_mask] / denom[denom_mask]).astype(np.float32)

    # Angles
    theta_att_input = _angle_between(H_in, D_att, eps=eps)
    theta_mlp_input = _angle_between(H_in, D_mlp, eps=eps)
    theta_att_mlp = _angle_between(D_att, D_mlp, eps=eps)
    theta_total_input = _angle_between(H_in, D_total, eps=eps)

    metrics = {
        "na": na.astype(np.float32),
        "nm": nm.astype(np.float32),
        "nt": nt.astype(np.float32),
        "theta_att_input": theta_att_input,
        "theta_mlp_input": theta_mlp_input,
        "theta_att_mlp": theta_att_mlp,
        "theta_total_input": theta_total_input,
        "r_att": r_att,
        "r_mlp": r_mlp,
    }
    return metrics


# -----------------------------
# Aggregation over prompts
# -----------------------------
def aggregate_layer_metrics_for_family(
    records: List[Dict],
    family_name: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[List[np.ndarray]]]]:
    """
    Aggregate Phase-3 metrics over all prompts for one family.

    Returns:
      agg: dict with per-layer means/stds and hist info:
        - layer_indices
        - {metric}_mean, {metric}_std for each metric in METRIC_NAMES
        - theta_att_mlp_hist_counts: (num_layers, num_bins)
        - theta_att_mlp_hist_edges: (num_bins+1,)
      metric_lists: raw per-layer lists for each metric:
        metric_lists[metric][layer_idx] -> List[np.ndarray of shape (T_layer,)]
      (Used for histograms and scatter plots.)
    """
    assert records, "No records to aggregate."
    num_layers_with_embed = len(records[0]["hidden"])
    num_layers = num_layers_with_embed - 1  # ignore embedding index 0
    layer_indices = np.arange(num_layers, dtype=np.int32)

    # metric_lists[metric][layer_idx] = list of 1D arrays over tokens
    metric_lists: Dict[str, List[List[np.ndarray]]] = {
        m: [[] for _ in range(num_layers)] for m in METRIC_NAMES
    }

    # Collect per-token metrics
    for rec in records:
        ids = rec["input_ids"]
        hidden = rec["hidden"]  # len = num_layers+1
        attn = rec["attn"]      # len = num_layers
        mlp = rec["mlp"]        # len = num_layers

        T_total = len(ids)
        prompt_len = rec["prompt_len"]

        if USE_CONTINUATION_ONLY:
            start = prompt_len
            end = T_total
        else:
            start = 0
            end = T_total

        if end - start <= 0:
            continue

        for li in range(num_layers):
            H_in_full = hidden[li]       # (T_total, D)
            H_out_full = hidden[li + 1]  # (T_total, D)
            A_full = attn[li]            # (T_total, D)
            M_full = mlp[li]             # (T_total, D)

            H_in = H_in_full[start:end, :]
            H_out = H_out_full[start:end, :]
            A = A_full[start:end, :]
            M = M_full[start:end, :]

            # Skip degenerate or NaN-heavy cases
            if H_in.shape[0] == 0:
                continue

            metrics = compute_layer_update_metrics(H_in, H_out, A, M)
            for name in METRIC_NAMES:
                metric_lists[name][li].append(metrics[name])

    # Aggregate into means / stds per layer
    agg: Dict[str, np.ndarray] = {"layer_indices": layer_indices}
    for name in METRIC_NAMES:
        means = np.full(num_layers, np.nan, dtype=np.float32)
        stds = np.full(num_layers, np.nan, dtype=np.float32)
        for li in range(num_layers):
            lst = metric_lists[name][li]
            if not lst:
                continue
            data = np.concatenate(lst, axis=0)
            if data.size == 0:
                continue
            means[li] = np.nanmean(data).astype(np.float32)
            stds[li] = np.nanstd(data).astype(np.float32)
        agg[f"{name}_mean"] = means
        agg[f"{name}_std"] = stds

    # Histogram for θ_att_mlp
    num_bins = 60
    hist_counts = np.zeros((num_layers, num_bins), dtype=np.float32)
    hist_edges = None
    for li in range(num_layers):
        lst = metric_lists["theta_att_mlp"][li]
        if not lst:
            continue
        data = np.concatenate(lst, axis=0)
        data = data[~np.isnan(data)]
        if data.size == 0:
            continue
        # Work in degrees for easier interpretation [0, 180]
        data_deg = np.degrees(data)
        counts, edges = np.histogram(
            data_deg,
            bins=num_bins,
            range=(0.0, 180.0),
        )
        hist_counts[li, :] = counts.astype(np.float32)
        hist_edges = edges.astype(np.float32)

    if hist_edges is None:
        # no data; default edges
        hist_edges = np.linspace(0.0, 180.0, num_bins + 1, dtype=np.float32)

    agg["theta_att_mlp_hist_counts"] = hist_counts
    agg["theta_att_mlp_hist_edges"] = hist_edges

    return agg, metric_lists


# -----------------------------
# Plotting helpers
# -----------------------------
def plot_layerwise_norms(
    agg: Dict[str, np.ndarray],
    family_name: str,
    outdir: Path,
):
    """
    Plot per-layer norms and ratios:

      - mean_na[l], mean_nm[l], mean_nt[l] vs layer
      - r_att_mean[l], r_mlp_mean[l] vs layer
    """
    outdir.mkdir(parents=True, exist_ok=True)
    layers = agg["layer_indices"]

    na_mean = agg["na_mean"]
    nm_mean = agg["nm_mean"]
    nt_mean = agg["nt_mean"]

    r_att_mean = agg["r_att_mean"]
    r_mlp_mean = agg["r_mlp_mean"]

    fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # Norms
    ax[0].plot(layers, na_mean, label="||Δ_att|| (mean)")
    ax[0].plot(layers, nm_mean, label="||Δ_mlp|| (mean)")
    ax[0].plot(layers, nt_mean, label="||Δ_total|| (mean)")
    ax[0].set_ylabel("norm")
    ax[0].set_title(f"Layer update norms vs depth ({family_name})")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Ratios
    ax[1].plot(layers, r_att_mean, label="r_att (na / (na + nm))")
    ax[1].plot(layers, r_mlp_mean, label="r_mlp (nm / (na + nm))")
    ax[1].set_xlabel("layer index (0 = first transformer block)")
    ax[1].set_ylabel("ratio")
    ax[1].set_title(f"Attention vs MLP contribution ratios ({family_name})")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = outdir / f"phase3_layer_norms_{family_name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[plot] Saved {path}")


def plot_layerwise_angles(
    agg: Dict[str, np.ndarray],
    family_name: str,
    outdir: Path,
):
    """
    Plot per-layer angle curves:

      - θ_att_input_mean, θ_mlp_input_mean, θ_total_input_mean vs layer
      - θ_att_mlp_mean vs layer
    """
    outdir.mkdir(parents=True, exist_ok=True)
    layers = agg["layer_indices"]

    theta_att_input = agg["theta_att_input_mean"]
    theta_mlp_input = agg["theta_mlp_input_mean"]
    theta_total_input = agg["theta_total_input_mean"]
    theta_att_mlp = agg["theta_att_mlp_mean"]

    fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # Input-relative angles
    ax[0].plot(layers, np.degrees(theta_att_input), label="angle(h_in, Δ_att)")
    ax[0].plot(layers, np.degrees(theta_mlp_input), label="angle(h_in, Δ_mlp)")
    ax[0].plot(layers, np.degrees(theta_total_input), label="angle(h_in, Δ_total)")
    ax[0].set_ylabel("degrees")
    ax[0].set_title(f"Input-relative update angles vs depth ({family_name})")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Attn vs MLP
    ax[1].plot(layers, np.degrees(theta_att_mlp), label="angle(Δ_att, Δ_mlp)")
    ax[1].set_xlabel("layer index (0 = first transformer block)")
    ax[1].set_ylabel("degrees")
    ax[1].set_title("Attention vs MLP angle vs depth")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = outdir / f"phase3_layer_angles_{family_name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[plot] Saved {path}")


def plot_theta_histograms(
    metric_lists_theta_att_mlp: List[List[np.ndarray]],
    family_name: str,
    outdir: Path,
    num_layers_to_show: int = 3,
):
    """
    Plot histograms of θ_att_mlp (in degrees) for early/mid/late layers.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    num_layers = len(metric_lists_theta_att_mlp)
    if num_layers == 0:
        return

    idxs = np.linspace(0, num_layers - 1, num_layers_to_show, dtype=int)

    for li in idxs:
        lst = metric_lists_theta_att_mlp[li]
        if not lst:
            continue
        data = np.concatenate(lst, axis=0)
        data = data[~np.isnan(data)]
        if data.size == 0:
            continue

        data_deg = np.degrees(data)
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.hist(data_deg, bins=60, range=(0.0, 180.0), alpha=0.8)
        ax.set_xlabel("θ_att_mlp (degrees)")
        ax.set_ylabel("count")
        ax.set_title(f"θ_att_mlp histogram (family={family_name}, layer={li})")
        ax.grid(True, alpha=0.3)

        path = outdir / f"phase3_theta_att_mlp_hist_{family_name}_layer{li}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"[plot] Saved {path}")


def plot_na_nm_scatter(
    metric_lists_na: List[List[np.ndarray]],
    metric_lists_nm: List[List[np.ndarray]],
    family_name: str,
    outdir: Path,
    num_layers_to_show: int = 3,
    max_points_per_layer: int = 5000,
):
    """
    Scatter plots of (na, nm) = (||Δ_att||, ||Δ_mlp||) for early/mid/late layers.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    num_layers = len(metric_lists_na)
    if num_layers == 0:
        return

    idxs = np.linspace(0, num_layers - 1, num_layers_to_show, dtype=int)
    rng = np.random.default_rng(SEED)

    for li in idxs:
        lst_na = metric_lists_na[li]
        lst_nm = metric_lists_nm[li]
        if not lst_na or not lst_nm:
            continue
        na = np.concatenate(lst_na, axis=0)
        nm = np.concatenate(lst_nm, axis=0)
        # Clean NaNs
        mask = ~np.isnan(na) & ~np.isnan(nm)
        na = na[mask]
        nm = nm[mask]
        if na.size == 0:
            continue

        N = na.size
        if N > max_points_per_layer:
            idx_sample = rng.choice(N, size=max_points_per_layer, replace=False)
            na_plot = na[idx_sample]
            nm_plot = nm[idx_sample]
        else:
            na_plot = na
            nm_plot = nm

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        ax.scatter(na_plot, nm_plot, s=6, alpha=0.5)
        ax.set_xlabel("||Δ_att||")
        ax.set_ylabel("||Δ_mlp||")
        ax.set_title(f"Δ_att vs Δ_mlp scatter (family={family_name}, layer={li})")
        ax.grid(True, alpha=0.3)

        path = outdir / f"phase3_na_nm_scatter_{family_name}_layer{li}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"[plot] Saved {path}")


def plot_multi_family_overlays(
    aggs_by_family: Dict[str, Dict[str, np.ndarray]],
    outdir: Path,
):
    """
    Overlay core Phase-3 curves across families:

      - na_mean vs depth
      - nm_mean vs depth
      - theta_att_mlp_mean vs depth
    """
    if not aggs_by_family:
        return

    outdir.mkdir(parents=True, exist_ok=True)
    families = list(aggs_by_family.keys())
    ref_family = aggs_by_family[families[0]]
    layers = ref_family["layer_indices"]

    metrics = [
        ("na_mean", "||Δ_att|| (mean)", "Attn update norm vs depth"),
        ("nm_mean", "||Δ_mlp|| (mean)", "MLP update norm vs depth"),
        ("theta_att_mlp_mean", "θ_att_mlp (deg)", "Attn–MLP angle vs depth"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 10), sharex=True)

    for ax, (key, ylabel, title) in zip(axes, metrics):
        for fam in families:
            agg = aggs_by_family[fam]
            y = agg[key]
            if "theta_att_mlp" in key:
                y = np.degrees(y)
            ax.plot(layers, y, label=fam)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("layer index (0 = first transformer block)")
    axes[0].legend(loc="best")

    fig.tight_layout()
    path = outdir / "phase3_multi_family_overlay.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[plot] Saved {path}")


# -----------------------------
# Main
# -----------------------------
def main():
    print("[main] Phase 3 – intra-token, intra-layer update decomposition")
    print(f"[main] Families: {list(PHASE3_FAMILIES.keys())}")
    print(f"[main] Using continuation_only={USE_CONTINUATION_ONLY}")

    aggs_by_family: Dict[str, Dict[str, np.ndarray]] = {}

    for family_name, prompts in PHASE3_FAMILIES.items():
        print(f"\n[family] {family_name}: {len(prompts)} prompts")

        family_outdir = OUTDIR / family_name
        family_outdir.mkdir(parents=True, exist_ok=True)

        # 1) Run model and capture hidden + sublayer outputs
        records = run_model_and_capture(prompts, max_new_tokens=MAX_NEW_TOKENS)

        # 2) Aggregate Phase-3 metrics over tokens & prompts
        agg, metric_lists = aggregate_layer_metrics_for_family(records, family_name)
        aggs_by_family[family_name] = agg

        # 3) Save metrics to .npz
        metrics_path = family_outdir / f"phase3_metrics_{family_name}.npz"
        np.savez(metrics_path, **agg)
        print(f"[save] Saved metrics to {metrics_path}")

        # 4) Core plots for this family
        plot_layerwise_norms(agg, family_name, family_outdir)
        plot_layerwise_angles(agg, family_name, family_outdir)
        plot_theta_histograms(
            metric_lists["theta_att_mlp"], family_name, family_outdir
        )
        plot_na_nm_scatter(
            metric_lists["na"], metric_lists["nm"], family_name, family_outdir
        )

    # 5) Cross-family overlays (H9–H12 comparisons)
    plot_multi_family_overlays(aggs_by_family, OUTDIR)

    print("\n[main] Done. Check outputs under:", OUTDIR)


if __name__ == "__main__":
    main()
