#!/usr/bin/env python
"""
Phase 2 – Inter-token, inter-layer transport analysis for Gemma.

LAGGED VIEW:
    How information moves between tokens across layers.

For each prompt family (general QA, GSM8K-style math, stories, code):

  1) Run the model on all prompts (prompt + greedy continuation).
  2) Capture residual states h[l, t, d] for all layers and tokens.
  3) For each layer l and token t in the chosen region (prompt+cont or cont-only):
       - Find the best-matching token (l*, t*) in deeper layers (within a layer window).
       - Record:
            best_sim[l, t]   = max cosine(h[l, t], h[l*, t*])
            best_layer[l, t] = l*
            best_token[l, t] = t*
       - Define:
            delta_layer[l, t] = best_layer[l, t] - l
            delta_token[l, t] = best_token[l, t] - t

  4) Aggregate over prompts:
       - Histograms of delta_layer and delta_token.
       - Per-layer curves:
            mean_best_sim[l]
            mean_abs_delta_layer[l]
            mean_abs_delta_token[l]

  5) For a representative prompt per family:
       - Plot local cross-layer heatmaps C[l, Δ][t_src, t_dst]
         for Δ in {1, 2, 4} and a few layers.
       - Plot token-wise trajectories:
            layer vs best_token[l, t_role]
         for roles: subject-like token, last number token, last token.

Usage:
    python phase2_token_transport.py

You can tweak:
  - MODEL_ID
  - USE_CONTINUATION_ONLY
  - LAG_DELTAS
  - MAX_GLOBAL_LAYER_DELTA
  - which layers to visualize for heatmaps
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# -----------------------------
# Prompt families
# -----------------------------
try:
    from prompt_bank import PHASE2_FAMILIES as PHASE2_FAMILIES
except ImportError:
    # Fallback: reuse Phase 1 families
    from prompt_bank import PHASE1_FAMILIES as PHASE2_FAMILIES  # type: ignore


# -----------------------------
# Config
# -----------------------------
MODEL_ID = "google/gemma-3-4b-it"
OUTDIR = Path("phase2_token_transport_outputs_4b")
OUTDIR.mkdir(parents=True, exist_ok=True)

MAX_NEW_TOKENS = 256
SEED = 42

# If True, restrict to continuation region (generation only).
# If False, use full prompt + continuation region.
USE_CONTINUATION_ONLY = False

# Local cross-layer lags for C[l, Δ] and heatmaps
LAG_DELTAS = [1, 2, 4]

# Global best-match search window in layer space:
# for each (l, t) we search layers l' in [l, l + MAX_GLOBAL_LAYER_DELTA]
MAX_GLOBAL_LAYER_DELTA = 6

# Histogram settings for δ_layer and δ_token
DELTA_LAYER_BINS = np.arange(-MAX_GLOBAL_LAYER_DELTA - 0.5,
                             MAX_GLOBAL_LAYER_DELTA + 1.5,
                             1.0)
DELTA_TOKEN_BINS = np.linspace(-64.5, 64.5, 130)  # 129 bins, captures shifts up to ~64 tokens

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
# Utils
# -----------------------------
def model_id_sanitized() -> str:
    return MODEL_ID.replace("/", "_")


def _normalize_2d(H: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize each row of H to unit length.

    H: (T, D)
    Returns: (T, D)
    """
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    denom = np.maximum(norms, eps)
    return H / denom


def classify_token(token_str: str) -> str:
    """
    Simple token-type heuristic:
      - 'number': digits only (after stripping leading '▁' and whitespace).
      - 'punct': all chars are punctuation.
      - 'word': everything else.
    """
    import string as _string

    s = token_str.replace("▁", "").strip()
    if not s:
        return "word"
    if s.isdigit():
        return "number"
    if all(ch in _string.punctuation for ch in s):
        return "punct"
    return "word"


def select_roles_in_region(token_strs: List[str]) -> Dict[str, int]:
    """
    Choose token indices for special roles within a region (0..T_region-1).

    Roles:
      - subject_token: first 'word' token.
      - last_number_token: last 'number' token (if any).
      - last_token: last token in region.

    Returns:
        {role_name: token_index}
    """
    T = len(token_strs)
    roles: Dict[str, int] = {}
    if T == 0:
        return roles

    # subject_token: first word-like token
    for idx, s in enumerate(token_strs):
        if classify_token(s) == "word":
            roles["subject_token"] = idx
            break

    # last_number_token
    for idx in range(T - 1, -1, -1):
        if classify_token(token_strs[idx]) == "number":
            roles["last_number_token"] = idx
            break

    # last_token: always exists
    roles["last_token"] = T - 1

    return roles


# -----------------------------
# Run model & capture hidden states
# -----------------------------
@torch.inference_mode()
def run_model_and_capture(
    prompts: List[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    """
    For each prompt:
      - Greedy-generate up to max_new_tokens.
      - Run a full forward pass on the prompt+completion.
      - Capture hidden_states: list[len=NUM_LAYERS+1] of (T, D) arrays.

    Returns: List[dict] with keys:
      - 'prompt'
      - 'input_ids': np.ndarray (T,)
      - 'prompt_len': int
      - 'hidden': List[np.ndarray] of shape (T, D)
    """
    records = []
    for p in prompts:
        enc = tok(p, return_tensors="pt").to(device)
        prompt_len = enc["input_ids"].shape[1]

        gen_ids = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=False,
        )[0]  # (T,)

        out = model(
            input_ids=gen_ids.unsqueeze(0),
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )

        hidden_states = [h[0].to(torch.float32).cpu().numpy() for h in out.hidden_states]
        if len(hidden_states) != NUM_LAYERS + 1:
            print(
                f"[warn] hidden_states length={len(hidden_states)} != NUM_LAYERS+1={NUM_LAYERS+1}"
            )

        records.append(
            {
                "prompt": p,
                "input_ids": gen_ids.cpu().numpy(),
                "prompt_len": int(prompt_len),
                "hidden": hidden_states,
            }
        )

    return records


# -----------------------------
# Per-record transport computation
# -----------------------------
def compute_transport_for_record(
    rec: Dict,
    max_layer_delta: int = MAX_GLOBAL_LAYER_DELTA,
) -> Optional[Dict]:
    """
    Compute global best-match mapping for one record.

    For each layer l (0..NUM_LAYERS-1) and token t in region:
      - h_base = h[l, t]
      - Search layers l' in [l, min(l + max_layer_delta, NUM_LAYERS-1)]
      - For each l': search over all tokens t' in region.
      - Find best (l*, t*) maximizing cosine(h[l, t], h[l*, t*]),
        with self-match (l*, t*) = (l, t) disallowed.

    Returns:
      dict with:
        - "best_sim": (L, T_region)
        - "best_layer": (L, T_region)
        - "best_token": (L, T_region)
        - "delta_layer": (L, T_region)
        - "delta_token": (L, T_region)
        - "region_tokens": List[str]
        - "region_start": int
        - "region_end": int
    or None if region is degenerate.
    """
    hidden = rec["hidden"]  # len = NUM_LAYERS+1
    ids = rec["input_ids"]
    prompt_len = rec["prompt_len"]

    token_strs_full = tok.convert_ids_to_tokens(ids.tolist())
    T_total = len(token_strs_full)

    if USE_CONTINUATION_ONLY:
        start = prompt_len
        end = T_total
    else:
        start = 0
        end = T_total

    region_tokens = token_strs_full[start:end]
    T_region = len(region_tokens)
    if T_region < 2:
        return None

    num_layers = len(hidden) - 1  # ignore embedding index 0

    # Collect per-layer residuals for region and normalize
    H_layers = [
        hidden[li + 1][start:end, :] for li in range(num_layers)
    ]  # list of (T_region, D)

    H_layers_normed = [_normalize_2d(H) for H in H_layers]

    # Initialize best-match structures
    best_sim = np.full((num_layers, T_region), -np.inf, dtype=np.float32)
    best_layer = np.full((num_layers, T_region), -1, dtype=np.int32)
    best_token = np.full((num_layers, T_region), -1, dtype=np.int32)

    # For each base layer l, search in [l, l+max_layer_delta]
    for l in range(num_layers):
        base = H_layers_normed[l]  # (T_region, D)

        for d in range(max_layer_delta + 1):
            lp = l + d
            if lp >= num_layers:
                break

            cand = H_layers_normed[lp]  # (T_region, D)
            # S[t_base, t_cand] = cos(h[l, t_base], h[lp, t_cand])
            S = base @ cand.T  # (T_region, T_region)

            # Disallow self-match
            if lp == l:
                np.fill_diagonal(S, -1e9)

            # Best over candidate tokens for each base token
            layer_best_sims = S.max(axis=1)        # (T_region,)
            layer_best_tokens = S.argmax(axis=1)   # (T_region,)

            # Update global best if better
            mask = layer_best_sims > best_sim[l, :]
            best_sim[l, mask] = layer_best_sims[mask]
            best_layer[l, mask] = lp
            best_token[l, mask] = layer_best_tokens[mask]

    # Derived deltas
    layer_indices = np.arange(num_layers, dtype=np.int32)[:, None]  # (L, 1)
    token_indices = np.arange(T_region, dtype=np.int32)[None, :]    # (1, T)

    delta_layer = best_layer - layer_indices
    delta_token = best_token - token_indices

    return {
        "best_sim": best_sim,
        "best_layer": best_layer,
        "best_token": best_token,
        "delta_layer": delta_layer,
        "delta_token": delta_token,
        "region_tokens": region_tokens,
        "region_start": start,
        "region_end": end,
    }


# -----------------------------
# Aggregation over prompts (per family)
# -----------------------------
def aggregate_transport_for_family(
    records: List[Dict],
    family_name: str,
) -> Tuple[Dict[str, np.ndarray], Optional[Dict]]:
    """
    Aggregate Phase 2 metrics for one family.

    Returns:
      agg: dict of arrays suitable for np.savez:
        - "layer_indices"              : (L,)
        - "mean_best_sim"              : (L,)
        - "mean_abs_delta_layer"       : (L,)
        - "mean_abs_delta_token"       : (L,)
        - "delta_layer_hist_counts"    : (num_layer_bins,)
        - "delta_layer_hist_edges"     : (num_layer_bins+1,)
        - "delta_token_hist_counts"    : (num_token_bins,)
        - "delta_token_hist_edges"     : (num_token_bins+1,)

      rep: representative example dict for plotting:
        - "record": original record
        - "transport": transport dict from compute_transport_for_record
    """
    assert records, "No records to aggregate."

    num_layers_with_embed = len(records[0]["hidden"])
    num_layers = num_layers_with_embed - 1
    layer_indices = np.arange(num_layers, dtype=np.int32)

    # Per-layer lists
    best_sim_by_layer: List[List[np.ndarray]] = [[] for _ in range(num_layers)]
    delta_layer_by_layer: List[List[np.ndarray]] = [[] for _ in range(num_layers)]
    delta_token_by_layer: List[List[np.ndarray]] = [[] for _ in range(num_layers)]

    # Global distributions
    all_delta_layer = []
    all_delta_token = []

    representative: Optional[Dict] = None

    for i, rec in enumerate(records):
        transport = compute_transport_for_record(rec)
        if transport is None:
            continue

        best_sim = transport["best_sim"]      # (L, T)
        delta_layer = transport["delta_layer"]
        delta_token = transport["delta_token"]

        # Aggregate per-layer lists
        for l in range(num_layers):
            best_sim_by_layer[l].append(best_sim[l])
            delta_layer_by_layer[l].append(delta_layer[l])
            delta_token_by_layer[l].append(delta_token[l])

        # Global distributions
        all_delta_layer.append(delta_layer.reshape(-1))
        all_delta_token.append(delta_token.reshape(-1))

        # Keep first usable record as representative for plotting
        if representative is None:
            representative = {"record": rec, "transport": transport}

    # If nothing usable, bail
    if not all_delta_layer:
        print(f"[warn] No usable records for family '{family_name}'.")
        agg = {
            "layer_indices": layer_indices,
            "mean_best_sim": np.full(num_layers, np.nan, dtype=np.float32),
            "mean_abs_delta_layer": np.full(num_layers, np.nan, dtype=np.float32),
            "mean_abs_delta_token": np.full(num_layers, np.nan, dtype=np.float32),
            "delta_layer_hist_counts": np.zeros(len(DELTA_LAYER_BINS) - 1, dtype=np.int64),
            "delta_layer_hist_edges": DELTA_LAYER_BINS.astype(np.float32),
            "delta_token_hist_counts": np.zeros(len(DELTA_TOKEN_BINS) - 1, dtype=np.int64),
            "delta_token_hist_edges": DELTA_TOKEN_BINS.astype(np.float32),
        }
        return agg, None

    # Concatenate global distributions
    all_delta_layer_arr = np.concatenate(all_delta_layer, axis=0)
    all_delta_token_arr = np.concatenate(all_delta_token, axis=0)

    delta_layer_hist_counts, delta_layer_hist_edges = np.histogram(
        all_delta_layer_arr, bins=DELTA_LAYER_BINS
    )
    delta_token_hist_counts, delta_token_hist_edges = np.histogram(
        all_delta_token_arr, bins=DELTA_TOKEN_BINS
    )

    # Per-layer means
    mean_best_sim = np.full(num_layers, np.nan, dtype=np.float32)
    mean_abs_delta_layer = np.full(num_layers, np.nan, dtype=np.float32)
    mean_abs_delta_token = np.full(num_layers, np.nan, dtype=np.float32)

    for l in range(num_layers):
        if not best_sim_by_layer[l]:
            continue
        bs = np.concatenate(best_sim_by_layer[l], axis=0)          # (N_tokens_total,)
        dl = np.concatenate(delta_layer_by_layer[l], axis=0)
        dt = np.concatenate(delta_token_by_layer[l], axis=0)

        mean_best_sim[l] = float(np.mean(bs))
        mean_abs_delta_layer[l] = float(np.mean(np.abs(dl)))
        mean_abs_delta_token[l] = float(np.mean(np.abs(dt)))

    agg = {
        "layer_indices": layer_indices,
        "mean_best_sim": mean_best_sim,
        "mean_abs_delta_layer": mean_abs_delta_layer,
        "mean_abs_delta_token": mean_abs_delta_token,
        "delta_layer_hist_counts": delta_layer_hist_counts.astype(np.int64),
        "delta_layer_hist_edges": delta_layer_hist_edges.astype(np.float32),
        "delta_token_hist_counts": delta_token_hist_counts.astype(np.int64),
        "delta_token_hist_edges": delta_token_hist_edges.astype(np.float32),
    }
    return agg, representative


# -----------------------------
# Plotting: transport intensity vs layer
# -----------------------------
def plot_transport_intensity_vs_layer(
    agg: Dict[str, np.ndarray],
    family_name: str,
    outdir: Path,
):
    outdir.mkdir(parents=True, exist_ok=True)
    layers = agg["layer_indices"]

    fig, ax = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    # 1) mean_best_sim
    ax[0].plot(layers, agg["mean_best_sim"], marker="o")
    ax[0].set_ylabel("mean best_sim")
    ax[0].set_title(f"Phase 2 – best-match similarity vs depth ({family_name})")
    ax[0].grid(True, alpha=0.3)

    # 2) mean_abs_delta_layer
    ax[1].plot(layers, agg["mean_abs_delta_layer"], marker="o")
    ax[1].set_ylabel("mean |δ_layer|")
    ax[1].set_title("Transport distance in layers")
    ax[1].grid(True, alpha=0.3)

    # 3) mean_abs_delta_token
    ax[2].plot(layers, agg["mean_abs_delta_token"], marker="o")
    ax[2].set_xlabel("layer index (0 = first transformer block)")
    ax[2].set_ylabel("mean |δ_token|")
    ax[2].set_title("Transport distance in token positions")
    ax[2].grid(True, alpha=0.3)

    fig.tight_layout()
    path = outdir / f"phase2_transport_intensity_{family_name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[plot] Saved {path}")


# -----------------------------
# Plotting: histograms of δ_layer / δ_token
# -----------------------------
def plot_delta_histograms(
    agg: Dict[str, np.ndarray],
    family_name: str,
    outdir: Path,
):
    outdir.mkdir(parents=True, exist_ok=True)

    # δ_layer
    counts_l = agg["delta_layer_hist_counts"]
    edges_l = agg["delta_layer_hist_edges"]
    centers_l = 0.5 * (edges_l[:-1] + edges_l[1:])

    # δ_token
    counts_t = agg["delta_token_hist_counts"]
    edges_t = agg["delta_token_hist_edges"]
    centers_t = 0.5 * (edges_t[:-1] + edges_t[1:])

    fig, ax = plt.subplots(2, 1, figsize=(9, 8))

    ax[0].bar(centers_l, counts_l, width=np.diff(edges_l), align="center")
    ax[0].set_title(f"Phase 2 – δ_layer histogram ({family_name})")
    ax[0].set_xlabel("δ_layer = best_layer - l")
    ax[0].set_ylabel("count")
    ax[0].grid(True, alpha=0.3)

    ax[1].bar(centers_t, counts_t, width=np.diff(edges_t), align="center")
    ax[1].set_title("δ_token histogram")
    ax[1].set_xlabel("δ_token = best_token - t")
    ax[1].set_ylabel("count")
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = outdir / f"phase2_delta_hist_{family_name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[plot] Saved {path}")


# -----------------------------
# Plotting: local C[l, Δ] heatmaps for representative record
# -----------------------------
def plot_local_cross_layer_heatmaps(
    representative: Dict,
    family_name: str,
    outdir: Path,
    max_tokens: int = 64,
    layers_to_plot: Optional[List[int]] = None,
    lag_deltas: Optional[List[int]] = None,
):
    """
    For a representative record, plot C[l, Δ][t_src, t_dst] heatmaps.

    We restrict to first max_tokens in the region for readability.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    rec = representative["record"]
    transport = representative["transport"]

    hidden = rec["hidden"]
    region_tokens = transport["region_tokens"]
    start = transport["region_start"]
    end = transport["region_end"]

    T_region = len(region_tokens)
    if T_region == 0:
        return

    T_plot = min(T_region, max_tokens)

    num_layers = len(hidden) - 1
    if layers_to_plot is None:
        # first, middle, last
        layers_to_plot = [
            0,
            max(0, num_layers // 2),
            num_layers - 1,
        ]
    layers_to_plot = [l for l in layers_to_plot if 0 <= l < num_layers]

    if lag_deltas is None:
        lag_deltas = LAG_DELTAS

    # Precompute normalized per-layer activations in region
    H_layers = [hidden[li + 1][start:end, :] for li in range(num_layers)]
    H_normed = [_normalize_2d(H) for H in H_layers]

    for l in layers_to_plot:
        for d in lag_deltas:
            lp = l + d
            if lp >= num_layers:
                continue

            src = H_normed[l][:T_plot, :]   # (T_plot, D)
            dst = H_normed[lp][:T_plot, :]  # (T_plot, D)

            C = src @ dst.T  # (T_plot, T_plot)

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(C, origin="lower", aspect="auto")
            ax.set_title(f"C[l={l}, Δ={d}] – family={family_name}")
            ax.set_xlabel("t_dst")
            ax.set_ylabel("t_src")
            fig.colorbar(im, ax=ax)

            fig.tight_layout()
            path = outdir / f"phase2_C_heatmap_family-{family_name}_l-{l}_d-{d}.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            print(f"[plot] Saved {path}")


# -----------------------------
# Plotting: token-wise trajectories (role hand-offs)
# -----------------------------
def plot_token_role_trajectories(
    representative: Dict,
    family_name: str,
    outdir: Path,
):
    """
    For a representative record, plot layer vs best_token position
    for selected roles (subject, last_number, last_token).
    """
    outdir.mkdir(parents=True, exist_ok=True)

    rec = representative["record"]
    transport = representative["transport"]

    best_token = transport["best_token"]  # (L, T_region)
    region_tokens = transport["region_tokens"]
    T_region = len(region_tokens)
    num_layers = best_token.shape[0]

    if T_region == 0:
        return

    roles = select_roles_in_region(region_tokens)
    if not roles:
        print(f"[plot] No roles found for family '{family_name}'.")
        return

    layers = np.arange(num_layers, dtype=np.int32)

    fig, ax = plt.subplots(figsize=(9, 5))

    for role_name, idx in roles.items():
        if idx < 0 or idx >= T_region:
            continue
        y = best_token[:, idx]  # (L,)
        ax.plot(layers, y, marker="o", label=f"{role_name} (start={idx})")

        # Horizontal line at original position to show drift
        ax.axhline(idx, linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_title(f"Phase 2 – token-wise role trajectories ({family_name})")
    ax.set_xlabel("layer index (0 = first transformer block)")
    ax.set_ylabel("best_token position")
    ax.set_ylim(-0.5, T_region - 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    path = outdir / f"phase2_role_trajectories_{family_name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[plot] Saved {path}")


# -----------------------------
# Main
# -----------------------------
def main():
    for family_name, prompts in PHASE2_FAMILIES.items():
        print(f"\n[phase2] Family='{family_name}' | #prompts={len(prompts)}")

        # 1) Capture hidden states
        records = run_model_and_capture(prompts)

        # 2) Aggregate transport stats
        agg, representative = aggregate_transport_for_family(records, family_name)

        # 3) Save metrics to .npz
        npz_path = OUTDIR / f"{model_id_sanitized()}_phase2_transport_{family_name}.npz"
        np.savez_compressed(npz_path, **agg)
        print(f"[save] Saved metrics to {npz_path}")

        # 4) Plots: layer-wise curves + histograms
        plot_transport_intensity_vs_layer(agg, family_name, OUTDIR)
        plot_delta_histograms(agg, family_name, OUTDIR)

        # 5) Plots: local C heatmaps + role trajectories for representative prompt
        if representative is not None:
            plot_local_cross_layer_heatmaps(representative, family_name, OUTDIR)
            plot_token_role_trajectories(representative, family_name, OUTDIR)
        else:
            print(f"[phase2] No representative example plotted for family '{family_name}'.")

    print("\n[phase2] Done.")


if __name__ == "__main__":
    main()
