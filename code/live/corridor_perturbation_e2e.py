#!/usr/bin/env python3
"""
End-to-end corridor analysis + perturbation script for Gemma.

Pipeline:
  1. Build PathBank of prompts with multiple perspectives (e.g. cs, econ).
  2. For each prompt: generate continuation, then capture hidden states across depth.
  3. Compute depth-wise geometry (step sizes, curvature, corridor index).
  4. Infer corridor layers from corridor_index.
  5. Build a FROM -> TO direction in corridor space (e.g., cs -> econ).
  6. Visualize:
       - depth geometry (3-panel figure)
       - cs vs econ projections along the learned direction
  7. Add a forward hook that injects alpha * direction into the corridor layers
     during generation, and compare completions with / without perturbations.
"""

import math
import json
import random
import csv
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# ============================================
# Config
# ============================================
MODEL_ID = "google/gemma-3-4b-it"
OUTDIR = Path("outputs_corridor"); OUTDIR.mkdir(exist_ok=True, parents=True)

MAX_NEW = 256
USE_WHITEN = False        # usually False; depth is short
USE_CONTINUATION_ONLY = True
SEED = 42

# Perspective direction to learn and use for perturbation
FROM_PERSPECTIVE = "cs"
TO_PERSPECTIVE = "econ"

# Magnitudes for perturbation experiments
ALPHA_VALUES = [0.5, 1.0]

# ============================================
# Seeding
# ============================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================
# Helper geometry functions
# ============================================
def l2_normalize_rows(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)

def whiten_tokens(H: np.ndarray) -> np.ndarray:
    """
    Whiten along the 'time' axis (here: depth).
    H: (T, D) where T = #layers, D = hidden size.
    """
    Hc = H - H.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    return (Hc @ Vt.T) / (S + 1e-6)

def step_vecs(H: np.ndarray) -> np.ndarray:
    """Discrete first difference along time dimension."""
    return H[1:] - H[:-1]   # (T-1, D)

def step_dists(H: np.ndarray) -> np.ndarray:
    """Norm of step vectors."""
    V = step_vecs(H)
    return np.linalg.norm(V, axis=1)    # (T-1,)

def curvature(H: np.ndarray) -> np.ndarray:
    """
    Discrete second difference norm; output has length T-2.
    Interpreted as curvature centered on intermediate points.
    """
    V = step_vecs(H)
    A = V[1:] - V[:-1]
    return np.linalg.norm(A, axis=1)    # (T-2,)

def spectral_stats_depth(Hhat: np.ndarray) -> Tuple[float, float]:
    """
    Tiny spectral summary across depth for a single token.
    Hhat: (L, D) normalized along rows.
    Returns (lowfreq_energy_ratio, centroid_index).
    """
    Hc = Hhat - Hhat.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    k = min(3, Vt.shape[0])
    P = (Hc @ Vt[:k].T)          # (L, k)

    F = np.abs(np.fft.rfft(P, axis=0)).mean(axis=1)  # spectrum over depth
    if F.sum() == 0:
        return 0.0, 0.0
    m = len(F)
    half = max(1, m // 6)
    low_ratio = float(F[:half].sum() / F.sum())
    centroid = float((np.arange(m) * F).sum() / F.sum())
    return low_ratio, centroid

def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-8)

# ============================================
# PathBank
# ============================================
# Slightly richer bases to separate cs vs econ
BASES_POOL = [
    "Why do large cities keep growing even when they already seem overcrowded?",
    "How do small local choices add up to traffic jams on a highway?",
    "Why do online platforms often end up dominated by a few large players?",
    "Why do some organizations become slow and bureaucratic as they grow?",
    "What tends to drive speculative bubbles in new technologies like cryptocurrencies or AI startups?",
    "Why do some large infrastructure projects get built quickly while others are repeatedly delayed?",
]

PERS = {
    "cs":      "Answer from a computer science perspective.",
    "econ":    "Answer from an economic systems perspective.",
}

MAX_BASES = 24  # kept for compatibility

def build_rows():
    rng = np.random.default_rng(SEED)
    pool = BASES_POOL.copy()
    rng.shuffle(pool)
    chosen = pool[:min(MAX_BASES, len(pool))]
    rows = []
    for i, base in enumerate(chosen):
        for c, tail in PERS.items():
            text = f"{base}\n{tail}"
            rows.append({"id": f"q{i}_{c}", "base_id": f"q{i}", "perspective": c, "text": text})
    return rows

# ============================================
# Model loading
# ============================================
print("Loading model:", MODEL_ID)
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
)
model.to(device)
model.eval()

config = AutoConfig.from_pretrained(MODEL_ID)
text_cfg = getattr(config, "text_config", config)
NUM_LAYERS = text_cfg.num_hidden_layers

print("n_layers:", text_cfg.num_hidden_layers)
print("hidden_size:", text_cfg.hidden_size)
print("device:", device)

def get_model_device_and_dtype(model):
    p = next(model.parameters())
    return p.device, p.dtype

param_device, param_dtype = get_model_device_and_dtype(model)

# cache for transformer layers
TRANSFORMER_LAYERS = None

# ============================================
# Generate + capture hidden states
# ============================================
@torch.inference_mode()
def generate_then_capture(text: str, max_new: int = MAX_NEW):
    enc = tok(text, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]

    # use_cache=False to make the capture call easy (full sequence)
    gen = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=max_new,
        use_cache=False,
    )
    full_ids = gen[0]  # (seq,)

    out = model(
        input_ids=full_ids.unsqueeze(0),
        output_hidden_states=True,
        return_dict=True,
    )
    # hidden_states is a tuple: [emb_out, layer1_out, ..., layerL_out]
    Hs = [h[0].to("cpu") for h in out.hidden_states]  # each (seq, hidden_size)
    return full_ids.cpu(), int(prompt_len), Hs

# ============================================
# Depth-geometry aggregation
# ============================================
def compute_depth_geometry(
    records: List[Dict],
    use_whiten: bool = USE_WHITEN,
    use_cont_only: bool = USE_CONTINUATION_ONLY,
):
    """
    For each continuation token, build its depth path and accumulate:
        - step norms between layers (dlt)
        - curvature across layers (curv)
    """
    if not records:
        raise ValueError("No records to compute depth geometry.")

    L = len(records[0]["layers"])  # includes embedding at index 0

    dlt_vals = [[] for _ in range(L - 1)]  # index = depth step i
    curv_vals = [[] for _ in range(L)]     # index = "center" depth

    lowfreq_all = []
    spec_cent_all = []

    for rec in tqdm(records, desc="Depth geometry"):
        layers = rec["layers"]
        seq_len = layers[0].shape[0]
        if use_cont_only:
            start_tok = rec["prompt_len"]
        else:
            start_tok = 0

        for t in range(start_tok, seq_len):
            # Build depth path for token t: (L, D)
            H = np.stack([layers[li][t] for li in range(L)], axis=0)
            if H.shape[0] < 4:
                continue

            X = whiten_tokens(H) if use_whiten else H
            Hhat = l2_normalize_rows(X)

            d = step_dists(Hhat)     # (L-1,)
            c = curvature(Hhat)      # (L-2,)

            # accumulate step norms
            for i in range(L - 1):
                dlt_vals[i].append(d[i])

            # curvature is centered; assign to depths 1..L-2
            for i in range(L - 2):
                center = i + 1
                curv_vals[center].append(c[i])

            # spectral summary across depth for this token
            lowE, cent = spectral_stats_depth(Hhat)
            lowfreq_all.append(lowE)
            spec_cent_all.append(cent)

    rows = []
    for depth in range(L):
        dvals = dlt_vals[depth] if depth < L - 1 else []
        cvals = curv_vals[depth]

        if dvals:
            d_mean = float(np.mean(dvals))
            d_std = float(np.std(dvals))
        else:
            d_mean = math.nan
            d_std = math.nan

        if cvals:
            c_mean = float(np.mean(cvals))
            c_std = float(np.std(cvals))
        else:
            c_mean = math.nan
            c_std = math.nan

        rows.append(
            dict(
                depth=depth,
                dlt_mean=d_mean,
                dlt_std=d_std,
                curv_mean=c_mean,
                curv_std=c_std,
            )
        )

    # Global spectral summary across depth paths
    if lowfreq_all:
        spec_summary = dict(
            lowfreq_ratio_mean=float(np.mean(lowfreq_all)),
            lowfreq_ratio_std=float(np.std(lowfreq_all)),
            spec_centroid_mean=float(np.mean(spec_cent_all)),
            spec_centroid_std=float(np.std(spec_cent_all)),
        )
    else:
        spec_summary = dict(
            lowfreq_ratio_mean=math.nan,
            lowfreq_ratio_std=math.nan,
            spec_centroid_mean=math.nan,
            spec_centroid_std=math.nan,
        )

    return rows, spec_summary

def compute_corridor_index(rows: List[Dict]) -> List[Dict]:
    dlt = np.array([r["dlt_mean"] for r in rows])
    cur = np.array([r["curv_mean"] for r in rows])

    # ignore NaNs at ends when z-scoring
    mask = np.isfinite(dlt) & np.isfinite(cur)
    CI = np.full_like(dlt, np.nan, dtype=float)

    if mask.sum() >= 3:
        CI_valid = -zscore(dlt[mask]) + -zscore(cur[mask])
        CI[mask] = CI_valid

    out = []
    for depth, r in enumerate(rows):
        row = dict(r)
        row["corridor_index"] = float(CI[depth]) if np.isfinite(CI[depth]) else math.nan
        out.append(row)
    return out

def infer_corridor_layers(depth_rows_with_ci: List[Dict]) -> List[int]:
    ci = np.array([r["corridor_index"] for r in depth_rows_with_ci])
    mask = np.isfinite(ci)
    if mask.sum() == 0:
        return []

    ci_valid = ci[mask]
    # Take upper quartile of CI as corridor
    thresh = np.percentile(ci_valid, 75.0)
    corridor_layers = [i for i, c in enumerate(ci) if np.isfinite(c) and c >= thresh]
    return corridor_layers

# ============================================
# Visualization helpers
# ============================================
def plot_depth_geometry(depth_rows_with_ci, corridor_layers, outdir: Path):
    depths = [r["depth"] for r in depth_rows_with_ci]
    dlt_mean = [r["dlt_mean"] for r in depth_rows_with_ci]
    dlt_std = [r["dlt_std"] for r in depth_rows_with_ci]
    cur_mean = [r["curv_mean"] for r in depth_rows_with_ci]
    cur_std = [r["curv_std"] for r in depth_rows_with_ci]
    ci_vals = [r["corridor_index"] for r in depth_rows_with_ci]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # 1) Step size
    ax = axes[0]
    ax.errorbar(depths, dlt_mean, yerr=dlt_std)
    for d in corridor_layers:
        ax.axvspan(d - 0.5, d + 0.5, alpha=0.15)
    ax.set_ylabel("Step size")
    ax.set_title("Depth-wise step size (||h_{l+1} - h_l||)")

    # 2) Curvature
    ax = axes[1]
    ax.errorbar(depths, cur_mean, yerr=cur_std)
    for d in corridor_layers:
        ax.axvspan(d - 0.5, d + 0.5, alpha=0.15)
    ax.set_ylabel("Curvature")
    ax.set_title("Depth-wise curvature")

    # 3) Corridor index
    ax = axes[2]
    ax.plot(depths, ci_vals, marker="o")
    for d in corridor_layers:
        ax.axvspan(d - 0.5, d + 0.5, alpha=0.15)
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Corridor index")
    ax.set_title("Corridor index over depth")

    fig.tight_layout()
    fig.savefig(outdir / "depth_geometry_summary.png", dpi=200)
    plt.show()

def plot_direction_histograms(
    projs_from, projs_to, outdir: Path,
    from_name: str, to_name: str
):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.hist(projs_from, bins=20, alpha=0.6, label=from_name)
    ax.hist(projs_to, bins=20, alpha=0.6, label=to_name)
    ax.set_xlabel("Projection onto learned direction")
    ax.set_ylabel("Count")
    ax.set_title(f"{from_name} vs {to_name} projections along corridor direction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "direction_projections.png", dpi=200)
    plt.show()

# ============================================
# Corridor representations & direction learning
# ============================================
def corridor_repr(
    rec: Dict,
    corridor_layers: List[int],
    use_continuation_only: bool = USE_CONTINUATION_ONLY,
) -> np.ndarray:
    layers = rec["layers"]
    seq_len = layers[0].shape[0]
    start_tok = rec["prompt_len"] if use_continuation_only else 0

    chunks = []
    for li in corridor_layers:
        H_l = layers[li][start_tok:seq_len, :]   # (T_cont, D)
        chunks.append(H_l)

    H = np.concatenate(chunks, axis=0)  # (T_cont * #layers, D)
    v = H.mean(axis=0)                  # (D,)
    v = v / (np.linalg.norm(v) + 1e-8)  # normalize
    return v

def build_corridor_reps(all_records, corridor_layers):
    reps = defaultdict(dict)  # reps[base_id][perspective] = vector

    for rec in all_records:
        base_id = rec["base_id"]
        persp = rec["perspective"]
        v = corridor_repr(rec, corridor_layers)
        reps[base_id][persp] = v

    return reps

def learn_direction(reps, from_persp: str, to_persp: str) -> np.ndarray:
    deltas = []
    for base_id, d in reps.items():
        if from_persp in d and to_persp in d:
            delta = d[to_persp] - d[from_persp]
            deltas.append(delta)

    if not deltas:
        raise RuntimeError(f"No base questions with both {from_persp} and {to_persp} representations.")

    deltas = np.stack(deltas, axis=0)   # (N, D)
    v = deltas.mean(axis=0)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v

def projections_along_direction(reps, vec, from_persp: str, to_persp: str):
    projs_from = []
    projs_to = []

    for base_id, d in reps.items():
        if from_persp in d and to_persp in d:
            projs_from.append(np.dot(d[from_persp], vec))
            projs_to.append(np.dot(d[to_persp], vec))

    return np.array(projs_from), np.array(projs_to)

# ============================================
# Hooking: corridor perturbation
# ============================================
def get_transformer_layers(model: torch.nn.Module, expected_n_layers: int = None):
    """
    Robustly locate the ModuleList of transformer blocks by scanning named_modules
    and picking the first ModuleList whose length == expected_n_layers.
    Result is cached in TRANSFORMER_LAYERS.
    """
    global TRANSFORMER_LAYERS
    if TRANSFORMER_LAYERS is not None:
        return TRANSFORMER_LAYERS

    if expected_n_layers is None:
        expected_n_layers = NUM_LAYERS

    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList):
            try:
                n = len(module)
            except TypeError:
                continue
            if n == expected_n_layers:
                candidates.append((name, module))

    if not candidates:
        raise RuntimeError(
            f"Could not locate transformer layers; no ModuleList with length {expected_n_layers}. "
            "Please inspect model.named_modules() manually."
        )

    print("Found candidate transformer layer lists:")
    for name, mod in candidates:
        print(f"  {name}: {len(mod)} layers")

    TRANSFORMER_LAYERS = candidates[0][1]
    return TRANSFORMER_LAYERS

from contextlib import contextmanager

@contextmanager
def corridor_shift(
    model: torch.nn.Module,
    layer_indices: List[int],
    vec_np: np.ndarray,
    alpha: float = 1.0,
    prompt_len: int = None,
):
    """
    Add alpha * vec to the residual stream at the specified layers.

    If prompt_len is given, only apply to tokens >= prompt_len.
    """
    vec_t = torch.tensor(vec_np, device=param_device, dtype=param_dtype)
    vec_t = vec_t.view(1, 1, -1)  # (1, 1, D) for broadcasting

    layers = get_transformer_layers(model)
    hooks = []

    def make_hook():
        def hook(module, inputs, output):
            # output may be a tensor or tuple depending on HF internals
            if isinstance(output, tuple):
                hs = output[0]
                rest = output[1:]
            else:
                hs = output
                rest = None

            # hs: (batch, seq, hidden)
            if prompt_len is None:
                hs_new = hs + alpha * vec_t
            else:
                hs_new = hs.clone()
                hs_new[:, prompt_len:, :] += alpha * vec_t

            if rest is None:
                return hs_new
            else:
                return (hs_new, *rest)
        return hook

    for idx in layer_indices:
        if idx < 0 or idx >= len(layers):
            raise IndexError(f"Layer index {idx} out of range (0..{len(layers)-1})")
        h = layers[idx].register_forward_hook(make_hook())
        hooks.append(h)

    try:
        yield
    finally:
        for h in hooks:
            h.remove()

@torch.inference_mode()
def generate_baseline(prompt: str, max_new: int = MAX_NEW) -> str:
    enc = tok(prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]
    gen_ids = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=max_new,
        use_cache=False,  # important: ensures full-seq passes each step
    )[0]
    completion = tok.decode(gen_ids[prompt_len:], skip_special_tokens=True)
    return completion

@torch.inference_mode()
def generate_with_corridor_shift(
    prompt: str,
    vec_np: np.ndarray,
    layer_indices: List[int],
    alpha: float = 1.0,
    max_new: int = MAX_NEW,
) -> str:
    enc = tok(prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]

    with corridor_shift(model, layer_indices, vec_np, alpha=alpha, prompt_len=prompt_len):
        gen_ids = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new,
            use_cache=False,  # important: ensures full-seq passes each step
        )[0]

    completion = tok.decode(gen_ids[prompt_len:], skip_special_tokens=True)
    return completion

# ============================================
# Main
# ============================================
def main():
    # 1) Build PathBank and capture hidden states
    print("\nBuilding PathBank prompts...")
    rows = build_rows()
    print(f"Total prompts: {len(rows)}")

    print("\nCapturing hidden states for PathBank prompts...")
    all_records: List[Dict] = []
    for row in tqdm(rows):
        ids, prompt_len, Hs = generate_then_capture(row["text"])
        rec = dict(
            id=row["id"],
            base_id=row["base_id"],
            perspective=row["perspective"],
            text=row["text"],
            ids=ids,
            prompt_len=prompt_len,
            layers=[h.to(torch.float32).numpy() for h in Hs],
        )
        all_records.append(rec)

    L_total = len(all_records[0]["layers"])
    print("Total hidden-state layers (incl. embedding):", L_total)

    # 2) Depth geometry
    print("\nComputing depth-geometry statistics...")
    depth_rows, spec_summary = compute_depth_geometry(
        all_records,
        use_whiten=USE_WHITEN,
        use_cont_only=USE_CONTINUATION_ONLY,
    )
    depth_rows_with_ci = compute_corridor_index(depth_rows)

    # Infer corridor layers
    corridor_layers = infer_corridor_layers(depth_rows_with_ci)
    print("\nHeuristic corridor layers (by corridor_index upper quartile):", corridor_layers)

    # 3) Save CSV / JSON
    csv_path = OUTDIR / "depth_geometry_by_step.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["depth", "dlt_mean", "dlt_std", "curv_mean", "curv_std", "corridor_index"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in depth_rows_with_ci:
            w.writerow(r)

    with open(OUTDIR / "depth_spectral_summary.json", "w") as f:
        json.dump(spec_summary, f, indent=2)

    print("\n=== Depth-wise residual-geometry (across layers; mean +/- std) ===")
    for r in depth_rows_with_ci:
        d = r["depth"]
        dm, ds = r["dlt_mean"], r["dlt_std"]
        cm, cs = r["curv_mean"], r["curv_std"]
        ci = r["corridor_index"]
        print(
            f"D{d:02d}: dlt={dm:.4f} +/- {ds:.4f} | "
            f"curv={cm:.4f} +/- {cs:.4f} | "
            f"CI={ci:.3f}"
        )

    print("\nGlobal spectral summary over depth paths:")
    print(json.dumps(spec_summary, indent=2))
    print(f"\nSaved depth geometry CSV to: {csv_path.resolve()}")

    # 4) Visualize depth geometry (single PNG with 3 subplots)
    print("\nPlotting depth geometry...")
    plot_depth_geometry(depth_rows_with_ci, corridor_layers, OUTDIR)

    # 5) Corridor representation + direction learning
    print("\nBuilding corridor representations...")
    reps = build_corridor_reps(all_records, corridor_layers)

    print(f"\nLearning direction {FROM_PERSPECTIVE} -> {TO_PERSPECTIVE} ...")
    v_from_to = learn_direction(reps, FROM_PERSPECTIVE, TO_PERSPECTIVE)
    np.save(OUTDIR / f"corridor_vec_{FROM_PERSPECTIVE}_to_{TO_PERSPECTIVE}.npy", v_from_to)
    print("Saved direction vector to:",
          OUTDIR / f"corridor_vec_{FROM_PERSPECTIVE}_to_{TO_PERSPECTIVE}.npy")

    projs_from, projs_to = projections_along_direction(reps, v_from_to, FROM_PERSPECTIVE, TO_PERSPECTIVE)
    print(f"\nMean projection ({FROM_PERSPECTIVE}): {projs_from.mean():.4f}")
    print(f"Mean projection ({TO_PERSPECTIVE}): {projs_to.mean():.4f}")

    print("\nPlotting direction projection histograms...")
    plot_direction_histograms(
        projs_from, projs_to,
        OUTDIR,
        from_name=FROM_PERSPECTIVE,
        to_name=TO_PERSPECTIVE,
    )

    # 6) Perturbation experiments on a test prompt
    test_prompt = "Why do large cities keep growing, and what are the main forces behind that growth?"
    print("\n=== Perturbation experiment ===")
    print("Test prompt:")
    print(test_prompt)

    base_completion = generate_baseline(test_prompt)
    print("\n[Baseline completion]")
    print(base_completion)

    for alpha in ALPHA_VALUES:
        print(f"\n[Completion with alpha={alpha:.2f} * {FROM_PERSPECTIVE}->{TO_PERSPECTIVE} direction]")
        shifted = generate_with_corridor_shift(
            test_prompt,
            v_from_to,
            layer_indices=corridor_layers,
            alpha=alpha,
        )
        print(shifted)

    # Reverse direction (TO -> FROM)
    print(f"\n[Completion with alpha=1.0 * {TO_PERSPECTIVE}->{FROM_PERSPECTIVE} (negative direction)]")
    shifted_rev = generate_with_corridor_shift(
        test_prompt,
        -v_from_to,
        layer_indices=corridor_layers,
        alpha=1.0,
    )
    print(shifted_rev)

    print("\nDone.")

if __name__ == "__main__":
    main()
