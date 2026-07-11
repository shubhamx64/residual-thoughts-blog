# phase1_token_cloud.py
"""
Phase 1 – Inter-token, intra-layer analysis ("token cloud" per layer)
for Gemma-3-1B.

Snapshot view:
- For each layer l, look at {h[l, t, :]} across token positions t.
- Measure:
    * Pairwise cosine similarity structure (mean, variance).
    * k-means clustering structure (silhouette score).
    * Norm statistics (mean, std, min, max).
    * Simple token-type norms (BOS, last, numbers, punctuation, words).

Extras in this version:
- Attention heatmaps: attn_mean[l][t_q, t_k] for representative prompt per family.
- Global "attention sinks" via high-norm tokens across all prompts & layers.

Per prompt family (general QA, GSM8K-style math, stories, code), we:
- Run the model on ALL prompts from prompt_bank.PHASE1_FAMILIES[family].
- Aggregate metrics over prompts.
- Save metrics to .npz.
- Plot layer-wise curves and token heatmaps/clouds.
- Plot attention heatmaps for a representative prompt.
- Update global sink stats and save top-norm tokens per layer.

Usage:
    python phase1_token_cloud.py
"""

import json
import string
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from prompt_bank import PHASE1_FAMILIES

# Optional: k-means + silhouette
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False
    print("[warn] scikit-learn not found; silhouette scores will be NaN.")


# -----------------------------
# Config
# -----------------------------
MODEL_ID = "google/gemma-3-4b-it"
OUTDIR = Path("phase1_token_cloud_outputs_4b")
OUTDIR.mkdir(parents=True, exist_ok=True)

MAX_NEW_TOKENS = 256
SEED = 42
K_CLUSTERS = 4          # K for k-means over tokens at each layer
USE_CONTINUATION_ONLY = False  # If True, restrict metrics to generation region only

torch.manual_seed(SEED)
np.random.seed(SEED)

# Global sink stats across all families & prompts:
# GLOBAL_SINK_STATS[layer_idx][token_str] = {"norm_sum": float, "count": int}
GLOBAL_SINK_STATS: Dict[int, Dict[str, Dict[str, float]]] = {}


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

# Ensure attention outputs are available (Gemma defaults to sdpa which skips them)
if hasattr(model, "set_attn_implementation"):
    try:
        model.set_attn_implementation("eager")
    except ValueError as e:
        print(f"[warn] Failed to set attn implementation to 'eager': {e}")

config = AutoConfig.from_pretrained(MODEL_ID)
text_cfg = getattr(config, "text_config", config)
NUM_LAYERS = text_cfg.num_hidden_layers
HIDDEN_SIZE = text_cfg.hidden_size

print(f"[load] num_layers={NUM_LAYERS}, hidden_size={HIDDEN_SIZE}, device={device}")


# -----------------------------
# Helper: run model and capture h[l, t, d]
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
      - Capture:
          * hidden_states: list[len=NUM_LAYERS+1] of (T, D) arrays.

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
            print(f"[warn] hidden_states length={len(hidden_states)} != NUM_LAYERS+1={NUM_LAYERS+1}")

        records.append(
            {
                "prompt": p,
                "input_ids": gen_ids.cpu().numpy(),
                "prompt_len": int(prompt_len),
                "hidden": hidden_states,
            }
        )

    return records


@torch.inference_mode()
def capture_avg_attention_for_prompts(
    prompts: List[str],
    max_prompts: int = 16,
    max_seq_len: int = 128,
):
    """
    Average attention over (up to) max_prompts prompts, prompt-only.

    Strategy:
      - Tokenize each prompt with truncation to max_seq_len.
      - Compute T_trunc = min(max_seq_len, min(T) over prompts).
      - For each prompt with T >= T_trunc:
          * Run model(..., output_attentions=True).
          * For each layer:
              - A: (heads, T, T)
              - Take A[:, :T_trunc, :T_trunc], average over heads -> (T_trunc, T_trunc).
              - Accumulate into attn_sum[layer].
      - Return attn_sum[layer], counts[layer], T_trunc.
    """
    if max_prompts is not None and len(prompts) > max_prompts:
        prompts = prompts[:max_prompts]

    encoded_inputs = []
    lengths = []

    for p in prompts:
        enc = tok(p, return_tensors="pt", truncation=True, max_length=max_seq_len)
        encoded_inputs.append(enc)
        lengths.append(enc["input_ids"].shape[1])

    if not lengths:
        return None

    T_trunc = min(max_seq_len, min(lengths))

    num_layers = NUM_LAYERS
    attn_sum = [None] * num_layers
    counts = np.zeros(num_layers, dtype=np.int32)

    for enc in encoded_inputs:
        input_ids = enc["input_ids"].to(device)
        out = model(
            input_ids=input_ids,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
        )
        if out.attentions is None:
            raise RuntimeError("Model did not return attentions; check config.")

        attn_layers = [a[0].to(torch.float32).cpu().numpy() for a in out.attentions]
        T = attn_layers[0].shape[-1]
        if T < T_trunc:
            continue

        for li in range(num_layers):
            A = attn_layers[li]  # (heads, T, T)
            A_region = A[:, :T_trunc, :T_trunc]
            A_mean = A_region.mean(axis=0)  # (T_trunc, T_trunc)

            if attn_sum[li] is None:
                attn_sum[li] = np.zeros_like(A_mean, dtype=np.float64)
            attn_sum[li] += A_mean
            counts[li] += 1

    return {
        "attn_sum": attn_sum,
        "counts": counts,
        "T_trunc": T_trunc,
    }


# -----------------------------
# Token utilities
# -----------------------------
def classify_token(token_str: str) -> str:
    """
    Crude token-type heuristic:
      - 'bos' / 'last' handled separately via position.
      - 'number': digits only (after stripping leading '▁' and whitespace).
      - 'punct': all chars are punctuation.
      - 'word': everything else.
    """
    s = token_str.replace("▁", "").strip()
    if not s:
        return "word"
    if s.isdigit():
        return "number"
    if all(ch in string.punctuation for ch in s):
        return "punct"
    return "word"


def pca_2d_token_cloud(H: np.ndarray) -> np.ndarray:
    """
    PCA of token cloud at one layer.
    H: (T, D)
    Returns coords: (T, 2)
    """
    Hc = H - H.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    basis = Vt[:2].T  # (D, 2)
    coords = Hc @ basis
    return coords


# -----------------------------
# Core per-layer metrics
# -----------------------------
def compute_pairwise_cosine(H: np.ndarray) -> np.ndarray:
    """
    H: (T, D)
    Returns cosine similarity matrix S: (T, T)
    """
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    Hn = H / (norms + 1e-8)
    S = Hn @ Hn.T
    return S


def compute_layer_metrics_for_prompt(
    H: np.ndarray,
    tokens: List[str],
    is_bos_mask: np.ndarray,
    is_last_mask: np.ndarray,
    layer_idx: int,
) -> Dict[str, float]:
    """
    Compute Phase-1 metrics for a single prompt at a single layer.

    H: (T, D)
    tokens: list of T token strings
    is_bos_mask, is_last_mask: bool arrays of shape (T,)
    """
    T, D = H.shape
    if T < 2:
        # Degenerate case; everything is undefined
        return {
            "mean_cos": np.nan,
            "var_cos": np.nan,
            "silhouette": np.nan,
            "norm_mean": np.nan,
            "norm_std": np.nan,
            "norm_min": np.nan,
            "norm_max": np.nan,
            "bos_norm_mean": np.nan,
            "last_norm_mean": np.nan,
            "number_norm_mean": np.nan,
            "punct_norm_mean": np.nan,
            "word_norm_mean": np.nan,
        }

    # Pairwise cosine
    S = compute_pairwise_cosine(H)
    iu = np.triu_indices(T, k=1)
    pair_vals = S[iu]
    mean_cos = float(pair_vals.mean())
    var_cos = float(pair_vals.var())

    # Norm structure
    norms = np.linalg.norm(H, axis=1)
    norm_mean = float(norms.mean())
    norm_std = float(norms.std())
    norm_min = float(norms.min())
    norm_max = float(norms.max())

    # Token-type norms
    bos_norm_mean = float(norms[is_bos_mask].mean()) if is_bos_mask.any() else np.nan
    last_norm_mean = float(norms[is_last_mask].mean()) if is_last_mask.any() else np.nan

    number_vals = []
    punct_vals = []
    word_vals = []
    for t_idx, tok_str in enumerate(tokens):
        ttype = classify_token(tok_str)
        if ttype == "number":
            number_vals.append(norms[t_idx])
        elif ttype == "punct":
            punct_vals.append(norms[t_idx])
        else:  # word
            word_vals.append(norms[t_idx])

    number_norm_mean = float(np.mean(number_vals)) if number_vals else np.nan
    punct_norm_mean = float(np.mean(punct_vals)) if punct_vals else np.nan
    word_norm_mean = float(np.mean(word_vals)) if word_vals else np.nan

    # Cluster structure: k-means + silhouette
    if HAVE_SKLEARN and T >= max(4, K_CLUSTERS):
        k = min(K_CLUSTERS, T)
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
            labels = km.fit_predict(H)
            if len(np.unique(labels)) > 1:
                sil = float(silhouette_score(H, labels))
            else:
                sil = np.nan
        except Exception as e:
            print(f"[warn] silhouette failed at layer={layer_idx}, T={T}: {e}")
            sil = np.nan
    else:
        sil = np.nan

    return {
        "mean_cos": mean_cos,
        "var_cos": var_cos,
        "silhouette": sil,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "norm_min": norm_min,
        "norm_max": norm_max,
        "bos_norm_mean": bos_norm_mean,
        "last_norm_mean": last_norm_mean,
        "number_norm_mean": number_norm_mean,
        "punct_norm_mean": punct_norm_mean,
        "word_norm_mean": word_norm_mean,
    }


def aggregate_metrics_over_prompts(records: List[Dict], family_name: str) -> Dict[str, np.ndarray]:
    """
    records: as returned by run_model_and_capture for a single family.
    Returns dict of aggregated metrics arrays per layer.

    We skip the embedding/state index 0 and treat index 1..NUM_LAYERS
    as "layers 0..NUM_LAYERS-1" in plots.
    """
    assert records, "No records to aggregate."
    num_layers_with_embed = len(records[0]["hidden"])
    num_layers = num_layers_with_embed - 1  # ignoring embedding
    layer_indices = np.arange(num_layers)

    N = len(records)

    # Metrics per prompt per layer, shape (N, num_layers)
    metrics_names = [
        "mean_cos",
        "var_cos",
        "silhouette",
        "norm_mean",
        "norm_std",
        "norm_min",
        "norm_max",
        "bos_norm_mean",
        "last_norm_mean",
        "number_norm_mean",
        "punct_norm_mean",
        "word_norm_mean",
    ]
    metric_arrays = {
        name: np.full((N, num_layers), np.nan, dtype=np.float32)
        for name in metrics_names
    }

    for pi, rec in enumerate(records):
        ids = rec["input_ids"]
        prompt_len = rec["prompt_len"]
        hidden_states = rec["hidden"]

        # Tokens as strings
        token_strs = tok.convert_ids_to_tokens(ids.tolist())
        T_total = len(token_strs)

        if USE_CONTINUATION_ONLY:
            start = prompt_len
            end = T_total
        else:
            start = 0
            end = T_total

        region_tokens = token_strs[start:end]
        region_len = len(region_tokens)
        if region_len < 2:
            print(f"[warn] prompt index {pi} in family '{family_name}' has <2 tokens in region; skipping metrics.")
            continue

        # Position masks in region
        is_bos_mask = np.zeros(region_len, dtype=bool)
        is_last_mask = np.zeros(region_len, dtype=bool)
        is_bos_mask[0] = True
        is_last_mask[-1] = True

        for li in range(num_layers):
            # hidden_states[0] is embedding; use hidden_states[li+1]
            H_full = hidden_states[li + 1]  # (T_total, D)
            H_region = H_full[start:end, :]

            m = compute_layer_metrics_for_prompt(
                H_region,
                region_tokens,
                is_bos_mask,
                is_last_mask,
                layer_idx=li,
            )
            for name in metrics_names:
                metric_arrays[name][pi, li] = m[name]

    # Aggregate over prompts (mean and std over axis=0)
    agg = {"layer_indices": layer_indices}
    for name in metrics_names:
        vals = metric_arrays[name]
        agg[name + "_mean"] = np.nanmean(vals, axis=0)
        agg[name + "_std"] = np.nanstd(vals, axis=0)

    return agg


# -----------------------------
# Sink-finding via norms
# -----------------------------
def update_sink_stats_from_records(records: List[Dict]):
    """
    Update GLOBAL_SINK_STATS with norm statistics per token per layer
    across all prompts and families.

    We use the full sequence (prompt + continuation) here.
    """
    global GLOBAL_SINK_STATS

    for rec in records:
        ids = rec["input_ids"]
        hidden_states = rec["hidden"]
        token_strs = tok.convert_ids_to_tokens(ids.tolist())
        T_total = len(token_strs)

        num_layers_with_embed = len(hidden_states)
        num_layers = num_layers_with_embed - 1

        for li in range(num_layers):
            H_full = hidden_states[li + 1]  # (T_total, D)
            norms = np.linalg.norm(H_full, axis=1)  # (T_total,)

            layer_map = GLOBAL_SINK_STATS.setdefault(li, {})
            for t_idx, tok_str in enumerate(token_strs):
                entry = layer_map.get(tok_str)
                if entry is None:
                    layer_map[tok_str] = {"norm_sum": float(norms[t_idx]), "count": 1}
                else:
                    entry["norm_sum"] += float(norms[t_idx])
                    entry["count"] += 1


def finalize_and_save_sink_stats(
    outdir: Path,
    top_k: int = 15,
    min_count: int = 10,
):
    """
    After processing all families, compute per-layer top-k tokens
    by average norm (attention sinks) and save to JSON.

    Also prints them to stdout.
    """
    sinks_summary = []

    print("\n[sinks] Top high-norm tokens per layer (across all families):")
    for layer_idx in sorted(GLOBAL_SINK_STATS.keys()):
        layer_map = GLOBAL_SINK_STATS[layer_idx]
        rows = []
        for tok_str, stats in layer_map.items():
            count = stats["count"]
            if count < min_count:
                continue
            avg_norm = stats["norm_sum"] / count
            rows.append((avg_norm, count, tok_str))

        if not rows:
            continue

        rows.sort(key=lambda x: x[0], reverse=True)
        top_rows = rows[:top_k]

        print(f"\n  Layer {layer_idx}:")
        for avg_norm, count, tok_str in top_rows:
            print(f"    token={tok_str!r:<12} avg_norm={avg_norm:.4f} count={count}")

        for avg_norm, count, tok_str in top_rows:
            sinks_summary.append(
                {
                    "layer": int(layer_idx),
                    "token": tok_str,
                    "avg_norm": float(avg_norm),
                    "count": int(count),
                }
            )

    out_path = outdir / "attention_sinks_by_norm.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sinks_summary, f, ensure_ascii=False, indent=2)
    print(f"\n[sinks] Saved sink summary to {out_path}")


# -----------------------------
# Plotting
# -----------------------------
def plot_layerwise_summary(agg: Dict[str, np.ndarray], family_name: str, outdir: Path):
    """
    Plot layerwise curves:
    - mean_cos & var_cos
    - silhouette
    - norm_mean(+/- std)
    """
    outdir.mkdir(parents=True, exist_ok=True)
    layers = agg["layer_indices"]

    fig, ax = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    # 1) Cosine similarity structure
    mean_cos = agg["mean_cos_mean"]
    var_cos = agg["var_cos_mean"]
    ax[0].plot(layers, mean_cos, label="mean_cos")
    ax[0].plot(layers, var_cos, label="var_cos")
    ax[0].set_ylabel("value")
    ax[0].set_title(f"Cosine structure vs depth ({family_name})")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # 2) Silhouette (cluster structure)
    silhouette_mean = agg["silhouette_mean"]
    ax[1].plot(layers, silhouette_mean, label="silhouette")
    ax[1].set_ylabel("silhouette")
    ax[1].set_title(f"k-means silhouette vs depth ({family_name})")
    ax[1].grid(True, alpha=0.3)

    # 3) Norm structure
    norm_mean = agg["norm_mean_mean"]
    norm_std = agg["norm_mean_std"]
    ax[2].plot(layers, norm_mean, label="norm_mean")
    ax[2].fill_between(layers, norm_mean - norm_std, norm_mean + norm_std, alpha=0.2)
    ax[2].set_xlabel("layer index (0 = first transformer block)")
    ax[2].set_ylabel("||h||_2")
    ax[2].set_title(f"Token norm vs depth ({family_name})")
    ax[2].grid(True, alpha=0.3)

    fig.tight_layout()
    path = outdir / f"layerwise_summary_{family_name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[plot] Saved {path}")


def plot_multi_family_overlays(
    aggs_by_family: Dict[str, Dict[str, np.ndarray]],
    outdir: Path,
):
    """
    Overlay Phase-1 curves (cosine mean, silhouette, norm mean) across families
    to visualize H4 differences on a shared figure.
    """
    if not aggs_by_family:
        print("[plot] overlay: no family aggregations provided.")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    families = list(aggs_by_family.keys())

    # Assume all families share identical layer indexing.
    ref_family = aggs_by_family[families[0]]
    layers = ref_family["layer_indices"]

    metrics = [
        ("mean_cos_mean", "mean cosine similarity", "Cosine mean vs depth"),
        ("silhouette_mean", "silhouette score", "Silhouette vs depth"),
        ("norm_mean_mean", "mean ||h||_2", "Norm mean vs depth"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 10), sharex=True)

    for ax, (key, ylabel, title) in zip(axes, metrics):
        for fam in families:
            agg = aggs_by_family[fam]
            ax.plot(layers, agg[key], label=fam)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("layer index (0 = first transformer block)")
    axes[0].legend(loc="best")

    fig.tight_layout()
    path = outdir / "layerwise_overlay_all_families.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[plot] Saved {path}")


def _compute_region_lengths(records: List[Dict]) -> List[int]:
    """
    Compute region lengths (prompt+cont or cont-only depending on
    USE_CONTINUATION_ONLY) for each record, ignoring degenerate <2-token cases.
    """
    lengths = []
    for rec in records:
        ids = rec["input_ids"]
        token_strs = tok.convert_ids_to_tokens(ids.tolist())
        T_total = len(token_strs)

        if USE_CONTINUATION_ONLY:
            start = rec["prompt_len"]
            end = T_total
        else:
            start = 0
            end = T_total

        region_len = end - start
        if region_len >= 2:
            lengths.append(region_len)
    return lengths


def plot_avg_token_similarity_heatmaps(
    records: List[Dict],
    family_name: str,
    outdir: Path,
    num_layers_to_show: int = 3,
    max_T_trunc: int = 32,
):
    """
    For a family, average S[l][t_i, t_j] over prompts.

    Strategy:
      - Choose T_trunc = min(max_T_trunc, min region length over prompts).
      - For each prompt with region_len >= T_trunc:
          * Take first T_trunc tokens in region.
          * Compute S[l] for each layer.
          * Accumulate S_sum[l] and count[l].
      - Plot S_avg[l] = S_sum[l] / count[l] at early/mid/late layers.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    region_lengths = _compute_region_lengths(records)
    if not region_lengths:
        print(f"[plot] avg similarity heatmaps: no valid regions for family={family_name}")
        return

    T_trunc = min(max_T_trunc, min(region_lengths))
    num_layers_with_embed = len(records[0]["hidden"])
    num_layers = num_layers_with_embed - 1

    S_sum = np.zeros((num_layers, T_trunc, T_trunc), dtype=np.float64)
    counts = np.zeros(num_layers, dtype=np.int32)

    for rec in records:
        ids = rec["input_ids"]
        token_strs = tok.convert_ids_to_tokens(ids.tolist())
        T_total = len(token_strs)

        if USE_CONTINUATION_ONLY:
            start = rec["prompt_len"]
            end = T_total
        else:
            start = 0
            end = T_total

        region_len = end - start
        if region_len < T_trunc:
            continue

        hidden_states = rec["hidden"]

        for li in range(num_layers):
            H_full = hidden_states[li + 1]
            H_region = H_full[start : start + T_trunc, :]  # (T_trunc, D)
            S = compute_pairwise_cosine(H_region)         # (T_trunc, T_trunc)
            S_sum[li] += S
            counts[li] += 1

    idxs = np.linspace(0, num_layers - 1, num_layers_to_show, dtype=int)

    for li in idxs:
        if counts[li] == 0:
            continue
        S_avg = S_sum[li] / counts[li]

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(S_avg, interpolation="nearest")
        fig.colorbar(im, ax=ax)
        ax.set_title(
            f"Avg S[t_i, t_j] cosine (family={family_name}, layer={li}, T_trunc={T_trunc})"
        )
        ax.set_xlabel("token index j")
        ax.set_ylabel("token index i")

        path = outdir / f"similarity_heatmap_avg_{family_name}_layer{li}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"[plot] Saved {path}")


def plot_global_token_cloud_pca(
    records: List[Dict],
    family_name: str,
    outdir: Path,
    num_layers_to_show: int = 3,
    max_T_trunc: int = 32,
    max_tokens_per_layer: int = 2000,
):
    """
    Build a global token cloud per layer across ALL prompts, then PCA.

    Strategy:
      - Choose T_trunc as in heatmaps.
      - For each prompt with region_len >= T_trunc:
          * Take first T_trunc tokens in region.
          * Collect H_region for each layer and token types.
      - For each chosen layer:
          * Stack all H_region across prompts -> H_all (N_tokens, D).
          * Optionally subsample to max_tokens_per_layer.
          * Run PCA to 2D and scatter, colored by token type (word/number/punct).
    """
    outdir.mkdir(parents=True, exist_ok=True)

    region_lengths = _compute_region_lengths(records)
    if not region_lengths:
        print(f"[plot] global PCA: no valid regions for family={family_name}")
        return

    T_trunc = min(max_T_trunc, min(region_lengths))
    num_layers_with_embed = len(records[0]["hidden"])
    num_layers = num_layers_with_embed - 1

    idxs = np.linspace(0, num_layers - 1, num_layers_to_show, dtype=int)

    for li in idxs:
        H_list = []
        type_list = []

        for rec in records:
            ids = rec["input_ids"]
            token_strs = tok.convert_ids_to_tokens(ids.tolist())
            T_total = len(token_strs)

            if USE_CONTINUATION_ONLY:
                start = rec["prompt_len"]
                end = T_total
            else:
                start = 0
                end = T_total

            region_len = end - start
            if region_len < T_trunc:
                continue

            hidden_states = rec["hidden"]
            H_full = hidden_states[li + 1]
            H_region = H_full[start : start + T_trunc, :]  # (T_trunc, D)
            H_list.append(H_region)

            region_tokens = token_strs[start : start + T_trunc]
            for tok_str in region_tokens:
                type_list.append(classify_token(tok_str))

        if not H_list:
            print(f"[plot] global PCA: no tokens for layer={li}, family={family_name}")
            continue

        H_all = np.concatenate(H_list, axis=0)  # (N_tokens, D)
        type_arr = np.array(type_list)
        N = H_all.shape[0]

        if N > max_tokens_per_layer:
            idx_sample = np.random.choice(N, size=max_tokens_per_layer, replace=False)
            H_all = H_all[idx_sample]
            type_arr = type_arr[idx_sample]

        coords = pca_2d_token_cloud(H_all)  # (N_eff, 2)

        type_to_color = {"word": "C0", "number": "C1", "punct": "C2"}
        colors = [type_to_color.get(t, "C3") for t in type_arr]

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        ax.scatter(coords[:, 0], coords[:, 1], s=10, c=colors, alpha=0.7)

        ax.set_title(f"Global token PCA cloud (family={family_name}, layer={li})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.3)

        path = outdir / f"token_pca_global_{family_name}_layer{li}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"[plot] Saved {path}")


def plot_avg_attention_heatmaps_for_family(
    prompts: List[str],
    family_name: str,
    outdir: Path,
    num_layers_to_show: int = 3,
    max_prompts: int = 16,
    max_seq_len: int = 128,
):
    """
    Plot averaged attn_mean[t_q, t_k] across prompts for this family.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    res = capture_avg_attention_for_prompts(
        prompts,
        max_prompts=max_prompts,
        max_seq_len=max_seq_len,
    )
    if res is None:
        print(f"[plot] avg attention: no data for family={family_name}")
        return

    attn_sum = res["attn_sum"]
    counts = res["counts"]
    T_trunc = res["T_trunc"]
    num_layers = len(attn_sum)

    idxs = np.linspace(0, num_layers - 1, num_layers_to_show, dtype=int)

    for li in idxs:
        if attn_sum[li] is None or counts[li] == 0:
            continue
        A_avg = attn_sum[li] / counts[li]

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(A_avg, interpolation="nearest")
        fig.colorbar(im, ax=ax)
        ax.set_title(
            f"Avg attn_mean[t_q, t_k] (family={family_name}, layer={li}, T_trunc={T_trunc})"
        )
        ax.set_xlabel("position index k")
        ax.set_ylabel("position index q")

        path = outdir / f"attn_heatmap_avg_{family_name}_layer{li}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"[plot] Saved {path}")


# -----------------------------
# Main
# -----------------------------
def main():
    print("[main] Phase 1 – inter-token, intra-layer analysis")
    print(f"[main] Families: {list(PHASE1_FAMILIES.keys())}")
    print(f"[main] Using continuation_only={USE_CONTINUATION_ONLY}")

    aggs_by_family: Dict[str, Dict[str, np.ndarray]] = {}

    for family_name, prompts in PHASE1_FAMILIES.items():
        print(f"\n[family] {family_name}: {len(prompts)} prompts")

        family_outdir = OUTDIR / family_name
        family_outdir.mkdir(parents=True, exist_ok=True)

        # 1) Run model and capture hidden states for ALL prompts in this family
        records = run_model_and_capture(prompts, max_new_tokens=MAX_NEW_TOKENS)

        # 2) Aggregate metrics over prompts
        agg = aggregate_metrics_over_prompts(records, family_name)
        aggs_by_family[family_name] = agg

        # 3) Save metrics to .npz
        metrics_path = family_outdir / f"phase1_metrics_{family_name}.npz"
        np.savez(metrics_path, **agg)
        print(f"[save] Saved metrics to {metrics_path}")

        # 4) Layerwise summary plots
        plot_layerwise_summary(agg, family_name, family_outdir)

        # 5) Token similarity heatmaps averaged over prompts
        plot_avg_token_similarity_heatmaps(records, family_name, family_outdir)

        # 6) Global token PCA clouds averaged over prompts
        plot_global_token_cloud_pca(records, family_name, family_outdir)

        # 7) Attention heatmaps averaged over prompts (prompt-only)
        plot_avg_attention_heatmaps_for_family(prompts, family_name, family_outdir)

        # 8) Update global sink stats from norms
        update_sink_stats_from_records(records)

    # 9) Finalize and save sink stats across ALL families
    finalize_and_save_sink_stats(OUTDIR)

    # 10) Overlay plots across families (H4 visual)
    plot_multi_family_overlays(aggs_by_family, OUTDIR)

    print("\n[main] Done. Check outputs under:", OUTDIR)


if __name__ == "__main__":
    main()
