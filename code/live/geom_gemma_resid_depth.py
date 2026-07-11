#!/usr/bin/env python3
"""
Anthropic-style residual-stream probing for Gemma.

This script looks at **depth paths**:
    for each token t in the continuation, we track
    h_0(t), h_1(t), ..., h_L(t)
where h_l(t) is the residual stream (hidden state) after layer l.

We then compute step sizes and curvature across depth and aggregate
them per depth step, giving a "corridor over layers" view.
"""

import os
import math
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import csv

# -----------------------------
# Config
# -----------------------------
MODEL_ID = "google/gemma-3-4b-pt"
OUTDIR = Path("outputs_depth"); OUTDIR.mkdir(exist_ok=True, parents=True)

MAX_NEW = 256
USE_WHITEN = False   # usually leave False; depth is short
SEED = 42

# Use only continuation tokens (i.e. after the prompt) for geometry.
USE_CONTINUATION_ONLY = True

# -----------------------------
# Seeding
# -----------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# Small helper geometry funcs
# -----------------------------
def l2_normalize_rows(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)

def whiten_tokens(H: np.ndarray) -> np.ndarray:
    """
    Whiten along the "time" axis (here: depth).
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
    # Project to 3D with PCA-like SVD on depth axis
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

# -----------------------------
# PathBank (same as your plus file, can tweak)
# -----------------------------
BASES_POOL = [
    # dynamics / emergence
    "Why do large cities keep growing even when they already seem overcrowded?",
    "How do small local choices add up to traffic jams on a highway?",

    # coordination / institutions
    "What typically goes wrong in large cross-functional projects, and how can it be reduced?",
    "How should a country organize decision-making during a fast-moving crisis?",
]

PERS = {
    "cs":      "Answer from a computer science perspective.",
    "econ":    "Answer from an economic systems perspective.",
 #   "policy":  "Answer from a public-policy and governance perspective.",
 #   "neuro":   "Answer from a neuroscience and cognitive-science perspective.",
}

MAX_BASES = 24  # keep for consistency with the other script

def build_rows():
    rng = np.random.default_rng(SEED)
    pool = BASES_POOL.copy()
    rng.shuffle(pool)
    chosen = pool[:min(MAX_BASES, len(pool))]
    rows = []
    for i, base in enumerate(chosen):
        for c, tail in PERS.items():
            text = f"{base}\n{tail}"
            rows.append({"id": f"q{i}_{c}", "perspective": c, "text": text})
    return rows

ROWS = build_rows()

# -----------------------------
# Load model
# -----------------------------
print("Loading model:", MODEL_ID)
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=dtype,
    device_map="auto",
)
model.eval()
config = AutoConfig.from_pretrained(MODEL_ID)

text_cfg = getattr(config, "text_config", config)  # gemma-3 stores text settings here

print("n_layers:", text_cfg.num_hidden_layers)
print("hidden_size:", text_cfg.hidden_size)

# -----------------------------
# Generate + capture hidden_states
# -----------------------------
@torch.inference_mode()
def generate_then_capture(text: str, max_new: int = MAX_NEW):
    enc = tok(text, return_tensors="pt").to(model.device)
    prompt_len = enc["input_ids"].shape[1]

    gen = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=max_new,
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

print("\nCapturing hidden states for PathBank prompts...")
all_records: List[Dict] = []
for row in tqdm(ROWS):
    ids, prompt_len, Hs = generate_then_capture(row["text"])
    rec = dict(
        id=row["id"],
        perspective=row["perspective"],
        text=row["text"],
        ids=ids,
        prompt_len=prompt_len,
        layers=[h.to(torch.float32).numpy() for h in Hs],
    )
    all_records.append(rec)

L = len(all_records[0]["layers"])  # includes embedding at index 0
print("Total hidden-state layers (incl. embedding):", L)

# -----------------------------
# Depth-geometry aggregation
# -----------------------------
def compute_depth_geometry(records: List[Dict], use_whiten: bool = USE_WHITEN):
    """
    For each continuation token, build its depth path and accumulate:
        - step norms between layers (dlt)
        - curvature across layers (curv)

    We store:
        dlt at depth-step i => between layer i and i+1 (0 <= i < L-1)
        curv at depth-step i => curvature centered at i
                               (so valid for 1 <= i <= L-2)
    """
    dlt_vals = [[] for _ in range(L - 1)]  # index = depth step i
    curv_vals = [[] for _ in range(L)]     # index = "center" depth

    lowfreq_all = []
    spec_cent_all = []

    for rec in tqdm(records, desc="Depth geometry"):
        layers = rec["layers"]
        seq_len = layers[0].shape[0]
        if USE_CONTINUATION_ONLY:
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

print("\nComputing depth-geometry statistics...")
depth_rows, spec_summary = compute_depth_geometry(all_records, use_whiten=USE_WHITEN)

# -----------------------------
# Corridor index over depth
# -----------------------------
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

depth_rows_with_ci = compute_corridor_index(depth_rows)

# where is corridor most pronounced?
ci_vals = np.array(
    [r["corridor_index"] for r in depth_rows_with_ci if np.isfinite(r["corridor_index"])]
)
if ci_vals.size:
    all_ci = np.array(
        [r["corridor_index"] if np.isfinite(r["corridor_index"]) else np.inf
         for r in depth_rows_with_ci]
    )
    # pick ~1/4 of depths with lowest dlt+curv (highest CI)
    k = max(1, len(depth_rows_with_ci) // 4)
    order = np.argsort(-all_ci)  # descending CI
    corridor_depths = sorted(order[:k].tolist())
else:
    corridor_depths = []

# -----------------------------
# Save CSV / JSON and print summary
# -----------------------------
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

print("\nHeuristic corridor over depth (highest corridor_index):", corridor_depths)
print(f"\nSaved depth geometry CSV to: {csv_path.resolve()}")
print("Done.")
