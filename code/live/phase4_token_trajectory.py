#!/usr/bin/env python
"""
Phase 4 – Intra-token, inter-layer trajectory analysis for Gemma.

Trajectory view:
- For each chosen token (e.g., BOS token, last token, last number token), treat
  its residual states across layers as a trajectory:
      h_0, h_1, ..., h_{L-1}  (post-block residuals)
- Compute:
    * Step size per layer (||Δ_l||)
    * Discrete curvature (norm and angle between steps)
    * Discrete torsion (angle change of curvature vector)
    * Alignment to final state (cos_to_final)
    * Alignment to logit direction (cos_to_logit_dir), if unembedding is available
- Aggregate statistics across prompts, families, correctness (for GSM8K),
  and plot:

    * Single-example trajectory plots
    * Mean ± std overlays for each family
    * Correct vs incorrect overlays for GSM8K
    * 2D/3D PCA trajectory visualizations

Assumptions:
- You have a prompt_bank.py with either:
      PHASE4_FAMILIES = {"general_qa": [...], "gsm8k": [...], "stories": [...], "code": [...]}
  or at least:
      PHASE1_FAMILIES
  in which case we fall back to that.

Usage:
    python phase4_token_trajectory.py
"""

import json
import re
import string
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# -----------------------------
# Prompts: Phase 4 families
# -----------------------------
try:
    from prompt_bank import PHASE4_FAMILIES as PHASE4_FAMILIES
except ImportError:
    # Fallback: re-use Phase 1 families if you haven't split them
    from prompt_bank import PHASE1_FAMILIES as PHASE4_FAMILIES

try:
    from prompt_bank import GSM8K_GOLD_ANSWERS
except ImportError:
    GSM8K_GOLD_ANSWERS: Dict[str, float] = {}


# -----------------------------
# Config
# -----------------------------
MODEL_ID = "google/gemma-3-4b-it"
OUTDIR = Path("phase4_token_trajectory_outputs_4b")
OUTDIR.mkdir(parents=True, exist_ok=True)

MAX_NEW_TOKENS = 256
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# Global model state populated in "Model loading" section
tok: AutoTokenizer = None
model: AutoModelForCausalLM = None
NUM_LAYERS: int = 0          # number of transformer blocks
HIDDEN_SIZE: int = 0
W_U: Optional[np.ndarray] = None   # unembedding matrix (vocab_size, hidden_dim)


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

# Unembedding matrix (for cos_to_logit_dir)
output_emb = model.get_output_embeddings()
if output_emb is not None and hasattr(output_emb, "weight"):
    W_U = output_emb.weight.detach().to(torch.float32).cpu().numpy()
    print(f"[load] Unembedding matrix loaded: shape={W_U.shape}")
else:
    W_U = None
    print("[warn] No output embeddings found; cos_to_logit_dir will be NaN.")


# -----------------------------
# Helpers: token types & GSM8K
# -----------------------------
def classify_token(token_str: str) -> str:
    """
    Crude token-type heuristic:
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


def is_gsm8k_family(family_name: str) -> bool:
    """
    Heuristic: treat any family with 'gsm8k' in its name as GSM8K-style math.
    """
    return "gsm8k" in family_name.lower()


def extract_last_number(text: str) -> Optional[float]:
    """
    Extracts the last signed number (int/float) from text. Returns None if absent.
    """
    if not text:
        return None
    cleaned = text.replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def classify_gsm8k_correctness(prompt: str, completion_text: str) -> str:
    """
    Compare the final numeric answer in the completion to the gold answer.

    Returns:
        "correct", "incorrect", or "unknown" if we cannot score it.
    """
    if not GSM8K_GOLD_ANSWERS:
        return "unknown"
    prompt_key = prompt.strip()
    if prompt_key not in GSM8K_GOLD_ANSWERS:
        return "unknown"
    pred = extract_last_number(completion_text)
    if pred is None:
        return "unknown"
    target = float(GSM8K_GOLD_ANSWERS[prompt_key])
    if abs(pred - target) <= 1e-3:
        return "correct"
    return "incorrect"


# -----------------------------
# Helper: run model & capture h[l, t, d]
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
          * hidden_states: list[len=NUM_LAYERS+1] of (T, D) np arrays
          * top_ids: argmax over logits at each position (T,)

    Returns: List[dict] with keys:
      - 'prompt'
      - 'input_ids': np.ndarray (T,)
      - 'prompt_len': int
      - 'hidden': List[np.ndarray] of shape (T, D)
      - 'top_ids': np.ndarray (T,)
      - 'completion_text': str (decoded prompt+completion)
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

        logits = out.logits[0].to(torch.float32)  # (T, vocab)
        top_ids = logits.argmax(dim=-1).cpu().numpy()  # (T,)

        completion_text = tok.decode(gen_ids, skip_special_tokens=True)

        records.append(
            {
                "prompt": p,
                "input_ids": gen_ids.cpu().numpy(),
                "prompt_len": int(prompt_len),
                "hidden": hidden_states,
                "top_ids": top_ids,
                "completion_text": completion_text,
            }
        )

    return records


# -----------------------------
# Trajectory metrics
# -----------------------------
def compute_trajectory_metrics(
    H: np.ndarray,
    p_target: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """
    H: (L, D) array of residual states per layer for one token, where
       L = NUM_LAYERS (post-block states only; embedding excluded).

    Returns dict of per-layer arrays (length L) with:
      - h_norm[l]
      - step_norm[l]      (step between l-1 and l; step_norm[0]=NaN)
      - curvature_norm[l]
      - curvature_angle[l]
      - torsion[l]
      - cos_to_final[l]
      - cos_to_logit_dir[l]
    """
    L, D = H.shape
    assert L == NUM_LAYERS, f"Expected {NUM_LAYERS} layers, got {L}"

    # Norm of each state
    h_norm = np.linalg.norm(H, axis=1)  # (L,)

    # Step vectors: Δ_l = H_{l} - H_{l-1} for l >= 1
    steps = H[1:] - H[:-1]  # (L-1, D)
    step_norms = np.linalg.norm(steps, axis=1)  # (L-1,)

    step_full = np.full(L, np.nan, dtype=np.float32)
    step_full[1:] = step_norms

    # Discrete curvature: norm(Δ_{l} - Δ_{l-1}) / (||Δ_{l-1}|| + eps)
    curvature_norm = np.full(L, np.nan, dtype=np.float32)
    curvature_angle = np.full(L, np.nan, dtype=np.float32)

    if L >= 3:
        d1 = steps[:-1]  # (L-2, D)
        d2 = steps[1:]   # (L-2, D)

        diff = d2 - d1
        base = np.linalg.norm(d1, axis=1)  # (L-2,)
        num = np.linalg.norm(diff, axis=1)
        curvature_vals = num / (base + eps)

        # Angle between consecutive steps
        n1 = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + eps)
        n2 = d2 / (np.linalg.norm(d2, axis=1, keepdims=True) + eps)
        cosang = np.sum(n1 * n2, axis=1)
        cosang = np.clip(cosang, -1.0, 1.0)
        angle_vals = np.arccos(cosang)  # radians

        # Associate curvature with central layer index l (1..L-2)
        curvature_norm[1:L-1] = curvature_vals
        curvature_angle[1:L-1] = angle_vals

    # Discrete torsion: angle change of curvature vector C_l = T_{l} - T_{l-1}
    torsion = np.full(L, np.nan, dtype=np.float32)
    if L >= 4:
        # Unit tangents T_l associated with step from layer l-1 to l -> index l
        T_vecs = steps / (step_norms[:, None] + eps)  # (L-1, D)
        C_vecs = T_vecs[1:] - T_vecs[:-1]             # (L-2, D)

        C1 = C_vecs[:-1]  # (L-3, D)
        C2 = C_vecs[1:]   # (L-3, D)

        num = np.sum(C1 * C2, axis=1)
        denom = (np.linalg.norm(C1, axis=1) * np.linalg.norm(C2, axis=1) + eps)
        cos_tau = np.clip(num / denom, -1.0, 1.0)
        tau_vals = np.arccos(cos_tau)  # radians

        # Associate torsion with interior layers (2..L-2)
        torsion[2:L-1] = tau_vals

    # Alignment to final state
    final = H[-1]
    final_norm = np.linalg.norm(final) + eps
    cos_to_final = np.sum(H * final[None, :], axis=1) / (h_norm * final_norm + eps)

    # Alignment to logit direction
    if p_target is not None:
        p = p_target.astype(np.float32)
        p = p / (np.linalg.norm(p) + eps)
        cos_to_logit_dir = np.sum(H * p[None, :], axis=1) / (h_norm + eps)
    else:
        cos_to_logit_dir = np.full(L, np.nan, dtype=np.float32)

    return {
        "h_norm": h_norm.astype(np.float32),
        "step_norm": step_full.astype(np.float32),
        "curvature_norm": curvature_norm.astype(np.float32),
        "curvature_angle": curvature_angle.astype(np.float32),
        "torsion": torsion.astype(np.float32),
        "cos_to_final": cos_to_final.astype(np.float32),
        "cos_to_logit_dir": cos_to_logit_dir.astype(np.float32),
    }


# -----------------------------
# Token selection per record
# -----------------------------
def select_token_roles_for_record(
    rec: Dict,
) -> Dict[str, int]:
    """
    Given one record, choose token indices for different roles.

    Roles implemented:
      - "bos_token": explicit BOS token (if tokenizer inserts one) else idx 0
      - "last_token": index of final token in full sequence
      - "last_number_token": last token classified as "number" (if any)

    Returns:
        {role_name: token_index}
    """
    ids = rec["input_ids"]
    token_strs = tok.convert_ids_to_tokens(ids.tolist())
    T = len(token_strs)

    roles: Dict[str, int] = {}

    if T == 0:
        return roles

    # Beginning-of-sequence token (prefer tokenizer BOS id if present)
    bos_idx = None
    bos_token_id = getattr(tok, "bos_token_id", None)
    if bos_token_id is not None:
        matches = np.nonzero(ids == bos_token_id)[0]
        if len(matches) > 0:
            bos_idx = int(matches[0])
    if bos_idx is None:
        bos_idx = 0
    roles["bos_token"] = bos_idx

    # Last token in sequence
    roles["last_token"] = T - 1

    # Last numeric token (if any)
    last_num_idx = None
    for idx in range(T - 1, -1, -1):
        if classify_token(token_strs[idx]) == "number":
            last_num_idx = idx
            break
    if last_num_idx is not None:
        roles["last_number_token"] = last_num_idx

    return roles


# -----------------------------
# Aggregation helpers
# -----------------------------
METRIC_NAMES = [
    "h_norm",
    "step_norm",
    "curvature_norm",
    "curvature_angle",
    "torsion",
    "cos_to_final",
    "cos_to_logit_dir",
]


def init_metric_lists() -> Dict[str, List[np.ndarray]]:
    return {name: [] for name in METRIC_NAMES}


def finalize_metric_lists(
    metric_lists: Dict[str, List[np.ndarray]],
    num_layers: int,
) -> Dict[str, np.ndarray]:
    """
    Take lists of (L,) arrays and return per-layer mean/std arrays.
    """
    out: Dict[str, np.ndarray] = {}
    layer_indices = np.arange(num_layers, dtype=np.int32)
    out["layer_indices"] = layer_indices

    for name in METRIC_NAMES:
        lst = metric_lists.get(name, [])
        if not lst:
            out[name + "_mean"] = np.full(num_layers, np.nan, dtype=np.float32)
            out[name + "_std"] = np.full(num_layers, np.nan, dtype=np.float32)
            continue
        data = np.stack(lst, axis=0)  # (N, L)
        out[name + "_mean"] = np.nanmean(data, axis=0).astype(np.float32)
        out[name + "_std"] = np.nanstd(data, axis=0).astype(np.float32)

    return out


# -----------------------------
# PCA trajectory helper
# -----------------------------
def pca_3d_trajectory(H: np.ndarray) -> np.ndarray:
    """
    PCA projection of a trajectory H: (L, D) -> coords: (L, 3).
    If D < 3, returns padded with zeros.
    """
    Hc = H - H.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    k = min(3, Vt.shape[0])
    basis = Vt[:k].T  # (D, k)
    coords = Hc @ basis  # (L, k)
    if k < 3:
        # Pad to 3D
        pad = np.zeros((H.shape[0], 3 - k), dtype=coords.dtype)
        coords = np.concatenate([coords, pad], axis=1)
    return coords


# -----------------------------
# Plotting
# -----------------------------
def plot_example_trajectory(
    family_name: str,
    role_name: str,
    H: np.ndarray,
    metrics: Dict[str, np.ndarray],
    token_str: str,
    outdir: Path,
):
    """
    Single-token trajectory plots:
      - step_norm, curvature_angle, torsion vs depth
      - cos_to_final, cos_to_logit_dir vs depth
      - ||h|| vs depth
    """
    outdir.mkdir(parents=True, exist_ok=True)

    layers = np.arange(H.shape[0])

    fig, ax = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    # 1) Geometry scalars
    ax[0].plot(layers, metrics["step_norm"], label="step_norm")
    ax[0].plot(layers, metrics["curvature_angle"], label="curvature_angle (rad)")
    ax[0].plot(layers, metrics["torsion"], label="torsion (rad)")
    ax[0].set_ylabel("value")
    ax[0].set_title(f"Trajectory geometry – {family_name} / {role_name}")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # 2) Alignment curves
    ax[1].plot(layers, metrics["cos_to_final"], label="cos_to_final")
    ax[1].plot(layers, metrics["cos_to_logit_dir"], label="cos_to_logit_dir")
    ax[1].set_ylabel("cosine")
    ax[1].set_title("Alignment vs depth")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    # 3) Norm of h
    ax[2].plot(layers, metrics["h_norm"], label="||h_l||_2")
    ax[2].set_xlabel("layer index (0 = first transformer block)")
    ax[2].set_ylabel("norm")
    ax[2].set_title(f"State norm vs depth (token={token_str!r})")
    ax[2].grid(True, alpha=0.3)

    fig.tight_layout()
    path = outdir / f"example_traj_{family_name}_{role_name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[plot] Saved {path}")


def plot_example_pca_trajectory(
    family_name: str,
    role_name: str,
    H: np.ndarray,
    outdir: Path,
):
    """
    PCA 2D and 3D trajectory plots for a single token.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    coords = pca_3d_trajectory(H)  # (L, 3)
    layers = np.arange(H.shape[0])

    # 2D: PC1 vs PC2
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.plot(coords[:, 0], coords[:, 1], marker="o")
    for l in range(H.shape[0]):
        ax.text(coords[l, 0], coords[l, 1], str(l), fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA(2D) trajectory – {family_name} / {role_name}")
    ax.grid(True, alpha=0.3)
    path2d = outdir / f"traj_pca2d_{family_name}_{role_name}.png"
    fig.tight_layout()
    fig.savefig(path2d, dpi=200)
    plt.close(fig)
    print(f"[plot] Saved {path2d}")

    # 3D: PC1, PC2, PC3
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 7))
    ax3d = fig.add_subplot(111, projection="3d")
    ax3d.plot(coords[:, 0], coords[:, 1], coords[:, 2], marker="o")
    for l in range(H.shape[0]):
        ax3d.text(coords[l, 0], coords[l, 1], coords[l, 2], str(l), fontsize=7)
    ax3d.set_xlabel("PC1")
    ax3d.set_ylabel("PC2")
    ax3d.set_zlabel("PC3")
    ax3d.set_title(f"PCA(3D) trajectory – {family_name} / {role_name}")
    path3d = outdir / f"traj_pca3d_{family_name}_{role_name}.png"
    fig.tight_layout()
    fig.savefig(path3d, dpi=200)
    plt.close(fig)
    print(f"[plot] Saved {path3d}")


def plot_overlay_depth_metrics_for_role(
    family_name: str,
    role_name: str,
    agg_all: Dict[str, np.ndarray],
    outdir: Path,
):
    """
    Overlay mean ± std for one family & token role, group 'all'.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    layers = agg_all["layer_indices"]

    fig, ax = plt.subplots(4, 1, figsize=(9, 12), sharex=True)

    # Step size
    mean = agg_all["step_norm_mean"]
    std = agg_all["step_norm_std"]
    ax[0].plot(layers, mean, label="step_norm_mean")
    ax[0].fill_between(layers, mean - std, mean + std, alpha=0.2)
    ax[0].set_ylabel("||Δ||")
    ax[0].set_title(f"Step size vs depth – {family_name} / {role_name}")
    ax[0].grid(True, alpha=0.3)

    # Curvature angle
    mean = agg_all["curvature_angle_mean"]
    std = agg_all["curvature_angle_std"]
    ax[1].plot(layers, mean, label="curvature_angle_mean")
    ax[1].fill_between(layers, mean - std, mean + std, alpha=0.2)
    ax[1].set_ylabel("angle (rad)")
    ax[1].set_title("Curvature angle vs depth")
    ax[1].grid(True, alpha=0.3)

    # Torsion
    mean = agg_all["torsion_mean"]
    std = agg_all["torsion_std"]
    ax[2].plot(layers, mean, label="torsion_mean")
    ax[2].fill_between(layers, mean - std, mean + std, alpha=0.2)
    ax[2].set_ylabel("torsion (rad)")
    ax[2].set_title("Torsion vs depth")
    ax[2].grid(True, alpha=0.3)

    # Alignment
    mean_cf = agg_all["cos_to_final_mean"]
    std_cf = agg_all["cos_to_final_std"]
    mean_cl = agg_all["cos_to_logit_dir_mean"]
    std_cl = agg_all["cos_to_logit_dir_std"]
    ax[3].plot(layers, mean_cf, label="cos_to_final_mean")
    ax[3].fill_between(layers, mean_cf - std_cf, mean_cf + std_cf, alpha=0.2)
    ax[3].plot(layers, mean_cl, label="cos_to_logit_dir_mean")
    ax[3].fill_between(layers, mean_cl - std_cl, mean_cl + std_cl, alpha=0.2)
    ax[3].set_xlabel("layer index (0 = first transformer block)")
    ax[3].set_ylabel("cosine")
    ax[3].set_title("Alignment to final/logit dirs vs depth")
    ax[3].legend()
    ax[3].grid(True, alpha=0.3)

    fig.tight_layout()
    path = outdir / f"layerwise_trajectory_summary_{family_name}_{role_name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[plot] Saved {path}")


def plot_gsm8k_correct_incorrect_curvature(
    family_name: str,
    role_name: str,
    agg_correct: Dict[str, np.ndarray],
    agg_incorrect: Dict[str, np.ndarray],
    outdir: Path,
):
    """
    Correct vs incorrect overlay for GSM8K: curvature_angle_mean vs depth.
    If you wire up classify_gsm8k_correctness() properly, this will show
    H15-style differences.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    layers = agg_correct["layer_indices"]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(
        layers,
        agg_correct["curvature_angle_mean"],
        label="correct",
    )
    ax.plot(
        layers,
        agg_incorrect["curvature_angle_mean"],
        label="incorrect",
    )
    ax.set_xlabel("layer index (0 = first transformer block)")
    ax.set_ylabel("curvature angle (rad)")
    ax.set_title(f"GSM8K curvature (correct vs incorrect) – {family_name} / {role_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = outdir / f"gsm8k_curvature_correct_incorrect_{family_name}_{role_name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[plot] Saved {path}")


def plot_multi_family_overlays(
    aggs_by_family: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    role_name: str,
    outdir: Path,
):
    """
    Multi-family overlays for a single token role, group 'all':
      - step_norm_mean vs depth
      - curvature_angle_mean vs depth
      - cos_to_final_mean vs depth
    """
    outdir.mkdir(parents=True, exist_ok=True)

    families = list(aggs_by_family.keys())
    if not families:
        return

    # Assume all share identical layer indexing
    ref_family = families[0]
    layers = aggs_by_family[ref_family][role_name]["all"]["layer_indices"]

    metrics = [
        ("step_norm_mean", "||Δ||", "Step size vs depth"),
        ("curvature_angle_mean", "angle (rad)", "Curvature angle vs depth"),
        ("cos_to_final_mean", "cosine", "cos_to_final vs depth"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 10), sharex=True)

    for ax, (key, ylabel, title) in zip(axes, metrics):
        for fam in families:
            agg_all = aggs_by_family[fam][role_name]["all"]
            ax.plot(layers, agg_all[key], label=fam)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} – role={role_name}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("layer index (0 = first transformer block)")
    axes[0].legend(loc="best")

    fig.tight_layout()
    path = outdir / f"multi_family_overlay_{role_name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[plot] Saved {path}")


# -----------------------------
# Family-level analysis
# -----------------------------
def analyze_family_token_trajectories(
    records: List[Dict],
    family_name: str,
    outdir: Path,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    For one family:
      - For each record and each token role, compute trajectory metrics.
      - Aggregate per role and group (all, correct, incorrect).
      - Plot single-example trajectories and PCA.
      - Plot layerwise summaries and GSM8K overlays.

    Returns:
        aggs_by_role: dict[role_name][group_name][metric_name] -> arrays
    """
    outdir.mkdir(parents=True, exist_ok=True)
    is_gsm = is_gsm8k_family(family_name)

    # agg_raw[role_name][group_name] = metric_lists
    agg_raw: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]] = defaultdict(
        lambda: defaultdict(init_metric_lists)
    )

    # Example trajectories for plotting
    example_per_role: Dict[str, Dict] = {}

    for rec_idx, rec in enumerate(records):
        ids = rec["input_ids"]
        hidden_states = rec["hidden"]    # list of len NUM_LAYERS+1; we use [1:]
        top_ids = rec["top_ids"]
        prompt = rec["prompt"]
        completion_text = rec["completion_text"]

        token_strs = tok.convert_ids_to_tokens(ids.tolist())
        roles = select_token_roles_for_record(rec)

        # GSM8K correctness (placeholder; currently "unknown")
        correctness = (
            classify_gsm8k_correctness(prompt, completion_text) if is_gsm else "all"
        )

        for role_name, t_idx in roles.items():
            # H: (L, D) for this token, where L = NUM_LAYERS
            H = np.stack(
                [hidden_states[l + 1][t_idx, :] for l in range(NUM_LAYERS)],
                axis=0,
            )  # (L, D)

            # Logit direction for this token position
            if W_U is not None and top_ids is not None:
                target_id = int(top_ids[t_idx])
                p_target = W_U[target_id]
            else:
                p_target = None

            metrics = compute_trajectory_metrics(H, p_target=p_target)

            # Store under group "all"
            for name in METRIC_NAMES:
                agg_raw[role_name]["all"][name].append(metrics[name])

            # Store under correctness groups if known
            if is_gsm and correctness in ("correct", "incorrect"):
                for name in METRIC_NAMES:
                    agg_raw[role_name][correctness][name].append(metrics[name])

            # Keep first example per role for detailed plots
            if role_name not in example_per_role:
                example_per_role[role_name] = {
                    "H": H,
                    "metrics": metrics,
                    "token_str": token_strs[t_idx],
                    "prompt": prompt,
                    "completion_text": completion_text,
                }

    # Finalize aggregations
    aggs_by_role: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    for role_name, group_map in agg_raw.items():
        aggs_by_role[role_name] = {}
        for group_name, metric_lists in group_map.items():
            stats = finalize_metric_lists(metric_lists, NUM_LAYERS)
            aggs_by_role[role_name][group_name] = stats

            # Save metrics to .npz: flatten group keys
            save_dict = {"layer_indices": stats["layer_indices"]}
            for key, arr in stats.items():
                if key == "layer_indices":
                    continue
                save_dict[f"{group_name}_{key}"] = arr
            metrics_path = outdir / f"phase4_metrics_{family_name}_{role_name}_{group_name}.npz"
            np.savez(metrics_path, **save_dict)
            print(f"[save] Saved metrics to {metrics_path}")

        # Plot example trajectory & PCA if available
        ex = example_per_role.get(role_name)
        if ex is not None:
            ex_outdir = outdir / f"examples_{role_name}"
            plot_example_trajectory(
                family_name,
                role_name,
                ex["H"],
                ex["metrics"],
                ex["token_str"],
                ex_outdir,
            )
            plot_example_pca_trajectory(
                family_name,
                role_name,
                ex["H"],
                ex_outdir,
            )

        # Layerwise summary for group 'all'
        if "all" in aggs_by_role[role_name]:
            plot_overlay_depth_metrics_for_role(
                family_name,
                role_name,
                aggs_by_role[role_name]["all"],
                outdir,
            )

        # GSM8K: correct vs incorrect overlays
        if is_gsm and "correct" in aggs_by_role[role_name] and "incorrect" in aggs_by_role[role_name]:
            plot_gsm8k_correct_incorrect_curvature(
                family_name,
                role_name,
                aggs_by_role[role_name]["correct"],
                aggs_by_role[role_name]["incorrect"],
                outdir,
            )

    return aggs_by_role


# -----------------------------
# Main
# -----------------------------
def main():
    print("[main] Phase 4 – intra-token, inter-layer trajectory analysis")
    print(f"[main] Families: {list(PHASE4_FAMILIES.keys())}")

    aggs_by_family: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]] = {}

    for family_name, prompts in PHASE4_FAMILIES.items():
        print(f"\n[family] {family_name}: {len(prompts)} prompts")

        family_outdir = OUTDIR / family_name
        family_outdir.mkdir(parents=True, exist_ok=True)

        # 1) Run model and capture trajectories
        records = run_model_and_capture(prompts, max_new_tokens=MAX_NEW_TOKENS)

        # 2) Analyze trajectories, aggregate metrics, and plot
        aggs_for_family = analyze_family_token_trajectories(
            records,
            family_name,
            family_outdir,
        )
        aggs_by_family[family_name] = aggs_for_family

    # 3) Multi-family overlays for key token roles (e.g., last_token)
    for role_name in ["bos_token", "last_token", "last_number_token"]:
        # Only plot if role exists in any family
        if any(role_name in aggs_by_family[fam] for fam in aggs_by_family):
            plot_multi_family_overlays(aggs_by_family, role_name, OUTDIR)

    print("\n[main] Done. Check outputs under:", OUTDIR)


if __name__ == "__main__":
    main()
