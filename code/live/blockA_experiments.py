#!/usr/bin/env python
"""
Block A: Geometry & Corridor Steering Experiments

Targeting: Gemma-3-1b-it and similar models.
Features:
  - A1/A2: Geometry metrics (sequential processing).
  - A3: CI vs Steering Gain.
  - A4: Band-wise Steering.
  - Corridor-style direction building from continuation tokens (aligned with testbench_fixed).
  - Shape-guarded hooks to avoid crashes on mismatched layers.
"""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Config
# -----------------------------

@dataclass
class ExperimentConfig:
    model_id: str = "google/gemma-3-1b-it"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 256
    diffgeo_new_tokens: int = 256
    max_input_tokens: int = 256
    seed: int = 42
    out_dir: str = "outputs_blockA"
    
    # Geometry settings
    use_continuation_only: Tuple[bool, ...] = (True, False)
    token_subsample_rate: float = 1.0
    
    # Steering settings
    run_diffgeo: bool = False
    run_steering: bool = False
    run_band_steering: bool = False
    alpha_values: Tuple[float, ...] = (-60, -40, -20, -10, 0.0, 10, 20, 40, 60)
    steering_max_new_tokens: int = 256 

# -----------------------------
# Helpers
# -----------------------------

def get_model_tag(model_id: str) -> str:
    return model_id.split("/")[-1]

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

TRANSFORMER_LAYERS = None

def get_decoder_layers(model):
    # Reliable layer extraction for HF models
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError(f"Could not find decoder layers in {type(model)}")

def get_transformer_layers(model):
    """
    Core transformer block ModuleList, shared between geometry and steering.
    """
    global TRANSFORMER_LAYERS
    if TRANSFORMER_LAYERS is None:
        TRANSFORMER_LAYERS = get_decoder_layers(model)
    return TRANSFORMER_LAYERS

# -----------------------------
# Prompts
# -----------------------------

# Centralized list of high-abstraction, neutral concepts.
# These serve as the "control" for geometry and the "base" for steering.
NEUTRAL_BASES: List[str] = [
    "What is the nature of a network?",
    "Explain the concept of complexity.",
    "Describe the function of a key.",
    "What does it mean to optimize something?",
    "Explain the idea of a system.",
    "What does it mean to interpret a signal?",
    "Describe the role of structure.",
    "What is the concept of a pattern?",
    "Explain what it means to generalize.",
    "Describe the idea of connection.",
    "What defines a process?",
    "Explain the concept of equivalence.",
    "What represents a boundary?",
    "Describe the nature of change.",
    "What is the function of memory?",
]

# Perspective suffixes (aligned with testbench_fixed.py)
PERS = {
    "cs": "Answer from a computer science perspective.",
    "lit": "Answer from a literature and arts perspective.",
}

def build_validation_prompts() -> List[str]:
    """Simple neutral prompts to validate steering efficacy."""
    return [
        "Describe a wooden chair.",
        "What is a spoon used for?",
        "Explain what a table is.",
        "Describe the act of walking.",
        "What is a window?",
    ]

# -----------------------------
# Geometry Logic
# -----------------------------

class DepthGeometryAggregator:
    def __init__(self, num_layers: int, eps: float = 1e-8):
        self.L = num_layers
        self.eps = eps
        # Step statistics
        self.step_sum = np.zeros(num_layers)
        self.step_sq_sum = np.zeros(num_layers)
        self.step_counts = np.zeros(num_layers)
        # Curvature statistics
        self.curv_sum = np.zeros(num_layers)
        self.curv_sq_sum = np.zeros(num_layers)
        self.curv_counts = np.zeros(num_layers)
        # Geodesic efficiency (kept for compatibility, used as mean only)
        self.E_sum = np.zeros(num_layers)
        self.E_counts = np.zeros(num_layers)

    def accumulate_token(self, h_stack: np.ndarray):
        # h_stack: [L, D]
        L, D = h_stack.shape
        # Require consistent depth and enough layers for curvature
        if L != self.L or L < 4:
            return

        # Whiten + normalize along depth (mirrors testbench_fixed)
        Hc = h_stack - h_stack.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
        X = (Hc @ Vt.T) / (S + 1e-6)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        Hhat = X / (norms + self.eps)

        # 1. Steps
        steps = Hhat[1:] - Hhat[:-1]  # (L-1, D)
        step_sizes = np.linalg.norm(steps, axis=1)

        # 2. Curvature (Norm difference)
        deltas_diff = steps[1:] - steps[:-1]  # (L-2, D)
        curv_norms = np.linalg.norm(deltas_diff, axis=1)

        # 3. Geodesic Efficiency
        path_len = np.cumsum(step_sizes)
        h0 = Hhat[0]
        
        # Update sums
        for l in range(L - 1):
            s = step_sizes[l]
            self.step_sum[l] += s
            self.step_sq_sum[l] += s * s
            self.step_counts[l] += 1

            # Curvature is defined for l=1..L-2. We store at center layer l+1
            if l < L - 2:
                c = curv_norms[l]
                center = l + 1
                self.curv_sum[center] += c
                self.curv_sq_sum[center] += c * c
                self.curv_counts[center] += 1

            # Geodesic E(k) at depth k = l+1
            k = l + 1
            chord = np.linalg.norm(Hhat[k] - h0)
            E_k = chord / (path_len[l] + self.eps)
            self.E_sum[k] += E_k
            self.E_counts[k] += 1

    def finalize(self):
        step_mean = np.full(self.L, np.nan)
        step_std = np.full(self.L, np.nan)
        curv_mean = np.full(self.L, np.nan)
        curv_std = np.full(self.L, np.nan)
        E_mean = np.full(self.L, np.nan)

        # Steps
        mask_step = self.step_counts > 0
        if np.any(mask_step):
            step_mean[mask_step] = self.step_sum[mask_step] / self.step_counts[mask_step]
            step_var = (self.step_sq_sum[mask_step] / self.step_counts[mask_step]) - (step_mean[mask_step] ** 2)
            step_var = np.maximum(step_var, 0.0)
            step_std[mask_step] = np.sqrt(step_var)

        # Curvature
        mask_curv = self.curv_counts > 0
        if np.any(mask_curv):
            curv_mean[mask_curv] = self.curv_sum[mask_curv] / self.curv_counts[mask_curv]
            curv_var = (self.curv_sq_sum[mask_curv] / self.curv_counts[mask_curv]) - (curv_mean[mask_curv] ** 2)
            curv_var = np.maximum(curv_var, 0.0)
            curv_std[mask_curv] = np.sqrt(curv_var)

        # Geodesic efficiency
        mask_E = self.E_counts > 0
        if np.any(mask_E):
            E_mean[mask_E] = self.E_sum[mask_E] / self.E_counts[mask_E]

        return {
            "step_mean": step_mean.tolist(),
            "step_std": step_std.tolist(),
            "curv_mean": curv_mean.tolist(),
            "curv_std": curv_std.tolist(),
            "E_mean": E_mean.tolist()
        }


class DifferentialGeometryTracker:
    """
    Tracks local (last-token) and global (token-mean) path geometry over depth.

    Given hidden states H of shape [L, T, D] (layers, tokens, dim) restricted
    to continuation tokens, we define:
      x_l  = h_{l, T-1}           (local state)
      mu_l = mean_t h_{l, t}      (global state)
      u_l  = x_{l+1} - x_l        (local tangent)
      v_l  = mu_{l+1} - mu_l      (global tangent)
    """

    def __init__(self, num_layers: int, eps: float = 1e-8):
        self.L = num_layers
        self.eps = eps

        def zeros():
            return np.zeros(num_layers)

        # Local (tip) metrics
        self.local_speed_sum = zeros()
        self.local_speed_sq_sum = zeros()
        self.local_speed_counts = zeros()

        self.local_align_sum = zeros()
        self.local_align_sq_sum = zeros()
        self.local_align_counts = zeros()

        self.local_accel_sum = zeros()
        self.local_accel_sq_sum = zeros()
        self.local_accel_counts = zeros()

        self.local_curv_sum = zeros()
        self.local_curv_sq_sum = zeros()
        self.local_curv_counts = zeros()

        # Global (mean-field) metrics
        self.global_speed_sum = zeros()
        self.global_speed_sq_sum = zeros()
        self.global_speed_counts = zeros()

        self.global_align_sum = zeros()
        self.global_align_sq_sum = zeros()
        self.global_align_counts = zeros()

        self.global_accel_sum = zeros()
        self.global_accel_sq_sum = zeros()
        self.global_accel_counts = zeros()

        self.global_curv_sum = zeros()
        self.global_curv_sq_sum = zeros()
        self.global_curv_counts = zeros()

        # Drift (coupling between local and global tangents)
        self.drift_sum = zeros()
        self.drift_sq_sum = zeros()
        self.drift_counts = zeros()

    def _accumulate_scalar(self, sum_arr, sq_arr, count_arr, idx: int, value: float):
        if not np.isfinite(value):
            return
        sum_arr[idx] += value
        sq_arr[idx] += value * value
        count_arr[idx] += 1

    def accumulate_sequence(self, h_stack: np.ndarray):
        """
        h_stack: [L, T, D] hidden states for continuation tokens only.
        """
        L, T, D = h_stack.shape
        if L != self.L or L < 3 or T < 1:
            return

        # Local path: last token per layer
        x = h_stack[:, -1, :]           # [L, D]
        # Global path: mean over tokens per layer
        mu = h_stack.mean(axis=1)       # [L, D]

        # Tangents along depth
        u = x[1:] - x[:-1]              # [L-1, D]
        v = mu[1:] - mu[:-1]            # [L-1, D]

        u_norm = np.linalg.norm(u, axis=1)  # [L-1]
        v_norm = np.linalg.norm(v, axis=1)  # [L-1]

        # Unit tangents for curvature
        u_hat = u / (u_norm[:, None] + self.eps)
        v_hat = v / (v_norm[:, None] + self.eps)

        def cos_sim(a, b, na, nb):
            if na <= self.eps or nb <= self.eps:
                return np.nan
            return float(np.dot(a, b) / (na * nb))

        for l in range(L - 1):
            depth = l + 1  # assign tangent between l->l+1 to depth index l+1

            # Speeds
            local_speed = float(u_norm[l])
            global_speed = float(v_norm[l])
            self._accumulate_scalar(self.local_speed_sum, self.local_speed_sq_sum, self.local_speed_counts, depth, local_speed)
            self._accumulate_scalar(self.global_speed_sum, self.global_speed_sq_sum, self.global_speed_counts, depth, global_speed)

            # Drift (coupling between local and global tangents)
            drift = cos_sim(u[l], v[l], u_norm[l], v_norm[l])
            self._accumulate_scalar(self.drift_sum, self.drift_sq_sum, self.drift_counts, depth, drift)

            # Metrics that use neighboring tangents
            if l < L - 2:
                # Local alignment / acceleration / curvature
                local_align = cos_sim(u[l], u[l+1], u_norm[l], u_norm[l+1])
                local_accel = float(np.linalg.norm(u[l+1] - u[l]))
                local_curv = float(np.linalg.norm(u_hat[l+1] - u_hat[l]))

                self._accumulate_scalar(self.local_align_sum, self.local_align_sq_sum, self.local_align_counts, depth, local_align)
                self._accumulate_scalar(self.local_accel_sum, self.local_accel_sq_sum, self.local_accel_counts, depth, local_accel)
                self._accumulate_scalar(self.local_curv_sum, self.local_curv_sq_sum, self.local_curv_counts, depth, local_curv)

                # Global alignment / acceleration / curvature
                global_align = cos_sim(v[l], v[l+1], v_norm[l], v_norm[l+1])
                global_accel = float(np.linalg.norm(v[l+1] - v[l]))
                global_curv = float(np.linalg.norm(v_hat[l+1] - v_hat[l]))

                self._accumulate_scalar(self.global_align_sum, self.global_align_sq_sum, self.global_align_counts, depth, global_align)
                self._accumulate_scalar(self.global_accel_sum, self.global_accel_sq_sum, self.global_accel_counts, depth, global_accel)
                self._accumulate_scalar(self.global_curv_sum, self.global_curv_sq_sum, self.global_curv_counts, depth, global_curv)

    def _finalize_metric(self, sum_arr, sq_arr, count_arr):
        mean = np.full(self.L, np.nan)
        std = np.full(self.L, np.nan)
        mask = count_arr > 0
        if np.any(mask):
            mean[mask] = sum_arr[mask] / count_arr[mask]
            var = (sq_arr[mask] / count_arr[mask]) - (mean[mask] ** 2)
            var = np.maximum(var, 0.0)
            std[mask] = np.sqrt(var)
        return mean, std

    def finalize(self):
        depths = list(range(self.L))

        local_speed_mean, local_speed_std = self._finalize_metric(
            self.local_speed_sum, self.local_speed_sq_sum, self.local_speed_counts
        )
        local_align_mean, local_align_std = self._finalize_metric(
            self.local_align_sum, self.local_align_sq_sum, self.local_align_counts
        )
        local_accel_mean, local_accel_std = self._finalize_metric(
            self.local_accel_sum, self.local_accel_sq_sum, self.local_accel_counts
        )
        local_curv_mean, local_curv_std = self._finalize_metric(
            self.local_curv_sum, self.local_curv_sq_sum, self.local_curv_counts
        )

        global_speed_mean, global_speed_std = self._finalize_metric(
            self.global_speed_sum, self.global_speed_sq_sum, self.global_speed_counts
        )
        global_align_mean, global_align_std = self._finalize_metric(
            self.global_align_sum, self.global_align_sq_sum, self.global_align_counts
        )
        global_accel_mean, global_accel_std = self._finalize_metric(
            self.global_accel_sum, self.global_accel_sq_sum, self.global_accel_counts
        )
        global_curv_mean, global_curv_std = self._finalize_metric(
            self.global_curv_sum, self.global_curv_sq_sum, self.global_curv_counts
        )

        drift_mean, drift_std = self._finalize_metric(
            self.drift_sum, self.drift_sq_sum, self.drift_counts
        )

        return {
            "depths": depths,
            "local": {
                "speed_mean": local_speed_mean.tolist(),
                "speed_std": local_speed_std.tolist(),
                "align_mean": local_align_mean.tolist(),
                "align_std": local_align_std.tolist(),
                "accel_mean": local_accel_mean.tolist(),
                "accel_std": local_accel_std.tolist(),
                "curv_mean": local_curv_mean.tolist(),
                "curv_std": local_curv_std.tolist(),
            },
            "global": {
                "speed_mean": global_speed_mean.tolist(),
                "speed_std": global_speed_std.tolist(),
                "align_mean": global_align_mean.tolist(),
                "align_std": global_align_std.tolist(),
                "accel_mean": global_accel_mean.tolist(),
                "accel_std": global_accel_std.tolist(),
                "curv_mean": global_curv_mean.tolist(),
                "curv_std": global_curv_std.tolist(),
            },
            "drift_mean": drift_mean.tolist(),
            "drift_std": drift_std.tolist(),
        }

def compute_corridor_indices(geo: Dict, ignore_edges: int = 1):
    """
    Legacy CI based on step / curvature means from DepthGeometryAggregator.
    Kept for compatibility, but steering uses the global-metric variant
    compute_corridor_indices_global instead.
    """
    step = np.array(geo["step_mean"], dtype=float)
    curv = np.array(geo["curv_mean"], dtype=float)

    if ignore_edges > 0:
        step[:ignore_edges] = np.nan
        step[-ignore_edges:] = np.nan
        curv[:ignore_edges] = np.nan
        curv[-ignore_edges:] = np.nan
    
    def z(arr: np.ndarray) -> np.ndarray:
        mask = ~np.isnan(arr)
        out = np.full_like(arr, np.nan, dtype=float)
        if not np.any(mask):
            return out
        mu = float(np.mean(arr[mask]))
        sigma = float(np.std(arr[mask]) + 1e-6)
        out[mask] = (arr[mask] - mu) / sigma
        return out

    CI_basic = -z(step) - z(curv)
    return {"CI_basic": CI_basic.tolist()}


def compute_corridor_indices_global(diff_geo: Dict, ignore_edges: int = 1):
    """
    Corridor index based on DifferentialGeometryTracker global metrics.

    CI_global = Z(GlobalAlignment) - Z(GlobalCurvature)
    High global alignment and low curvature both increase the score.
    """
    align = np.array(diff_geo["global"]["align_mean"], dtype=float)
    curv = np.array(diff_geo["global"]["curv_mean"], dtype=float)

    if ignore_edges > 0:
        align[:ignore_edges] = np.nan
        align[-ignore_edges:] = np.nan
        curv[:ignore_edges] = np.nan
        curv[-ignore_edges:] = np.nan

    def z(arr: np.ndarray) -> np.ndarray:
        mask = np.isfinite(arr)
        out = np.full_like(arr, np.nan, dtype=float)
        if not np.any(mask):
            return out
        mu = float(np.mean(arr[mask]))
        sigma = float(np.std(arr[mask]) + 1e-6)
        out[mask] = (arr[mask] - mu) / sigma
        return out

    CI = z(align) - z(curv)
    return {"CI_global": CI.tolist()}

def plot_geometry_aligned(geo: Dict, ci, out_path: Path):
    """
    Three-panel plot aligned with testbench_fixed:
      1) Step size mean ± std
      2) Curvature mean ± std
      3) Corridor index with corridor highlighting
    """
    depths = np.arange(len(geo["step_mean"]))
    step_mean = np.array(geo["step_mean"])
    step_std = np.array(geo.get("step_std", [0.0] * len(step_mean)))
    curv_mean = np.array(geo["curv_mean"])
    curv_std = np.array(geo.get("curv_std", [0.0] * len(curv_mean)))
    ci_arr = np.array(ci)

    # Corridor layers: simple heuristic, CI > 0
    corridor_layers = np.where(np.isfinite(ci_arr) & (ci_arr > 0))[0]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # 1. Step size
    ax = axes[0]
    ax.errorbar(depths, step_mean, yerr=step_std, fmt='-o')
    for d in corridor_layers:
        ax.axvspan(d - 0.5, d + 0.5, alpha=0.15)
    ax.set_ylabel("Step size")
    ax.set_title("Depth-wise step size (||h_{l+1} - h_l||)")

    # 2. Curvature
    ax = axes[1]
    ax.errorbar(depths, curv_mean, yerr=curv_std, fmt='-o')
    for d in corridor_layers:
        ax.axvspan(d - 0.5, d + 0.5, alpha=0.15)
    ax.set_ylabel("Curvature")
    ax.set_title("Depth-wise curvature")

    # 3. Corridor index
    ax = axes[2]
    ax.plot(depths, ci_arr, marker='o')
    for d in corridor_layers:
        ax.axvspan(d - 0.5, d + 0.5, alpha=0.15)
    ax.axhline(0.0, linestyle='--')
    ax.set_xlabel("Depth")
    ax.set_ylabel("Corridor index")
    ax.set_title("Corridor index over depth")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def collect_differential_geometry(cfg: ExperimentConfig, model, tokenizer):
    """
    Collects local/global path geometry for neutral prompts using the
    DifferentialGeometryTracker.
    """
    device = cfg.device
    layers = get_decoder_layers(model)
    num_layers = len(layers)

    tracker = DifferentialGeometryTracker(num_layers)

    for base in tqdm(NEUTRAL_BASES, desc="DiffGeo prompts"):
        inputs = tokenizer(base, return_tensors="pt", truncation=True, max_length=cfg.max_input_tokens).to(device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=cfg.diffgeo_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
        full_seq = gen_out[0]  # (T_total,)

        with torch.no_grad():
            out = model(
                input_ids=full_seq.unsqueeze(0).to(device),
                output_hidden_states=True,
                return_dict=True,
            )

        # hidden_states: (Embed, L0, L1, ...) -> drop embed, stack to [L, T, D]
        hs_stack = (
            torch.stack(out.hidden_states[1:])
            .squeeze(1)
            .to(torch.float32)
            .cpu()
            .numpy()
        )

        # Restrict to continuation tokens (answer region)
        if hs_stack.shape[1] <= prompt_len:
            continue
        cont_states = hs_stack[:, prompt_len:, :]  # [L, T_new, D]

        tracker.accumulate_sequence(cont_states)

    return tracker.finalize()


def plot_differential_geometry_dashboard(diff_geo: Dict, out_path_base: Path, drop_last: int = 1):
    """
    4x2 grid (Local vs Global) + Drift overlay plot for the
    differential geometry metrics.
    """
    depths = np.array(diff_geo["depths"], dtype=int)
    if drop_last > 0:
        depths_plot = depths[:-drop_last]
    else:
        depths_plot = depths

    local = diff_geo["local"]
    global_ = diff_geo["global"]

    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)

    def trim(arr):
        arr = np.array(arr, dtype=float)
        return arr[:-drop_last] if drop_last > 0 else arr

    # Helper to plot mean ± std
    def plot_metric(ax, mean, std, title, ylabel):
        mean = trim(mean)
        std = trim(std)
        mask = np.isfinite(mean)
        ax.errorbar(depths_plot[mask], mean[mask], yerr=std[mask], fmt='-o')
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    # Row 0: Speed
    plot_metric(
        axes[0, 0],
        local["speed_mean"],
        local["speed_std"],
        "Local speed (||u_l||)",
        "Speed",
    )
    plot_metric(
        axes[0, 1],
        global_["speed_mean"],
        global_["speed_std"],
        "Global speed (||v_l||)",
        "Speed",
    )

    # Row 1: Alignment
    plot_metric(
        axes[1, 0],
        local["align_mean"],
        local["align_std"],
        "Local alignment cos(u_l, u_{l+1})",
        "Alignment",
    )
    plot_metric(
        axes[1, 1],
        global_["align_mean"],
        global_["align_std"],
        "Global alignment cos(v_l, v_{l+1})",
        "Alignment",
    )

    # Row 2: Curvature
    plot_metric(
        axes[2, 0],
        local["curv_mean"],
        local["curv_std"],
        "Local curvature ||û_{l+1} - û_l||",
        "Curvature",
    )
    plot_metric(
        axes[2, 1],
        global_["curv_mean"],
        global_["curv_std"],
        "Global curvature ||v̂_{l+1} - v̂_l||",
        "Curvature",
    )

    # Row 3: Acceleration
    plot_metric(
        axes[3, 0],
        local["accel_mean"],
        local["accel_std"],
        "Local acceleration ||u_{l+1} - u_l||",
        "Acceleration",
    )
    plot_metric(
        axes[3, 1],
        global_["accel_mean"],
        global_["accel_std"],
        "Global acceleration ||v_{l+1} - v_l||",
        "Acceleration",
    )

    for i in range(4):
        axes[i, 0].set_xlabel("Depth")
        axes[i, 1].set_xlabel("Depth")

    fig.tight_layout()
    grid_path = out_path_base.parent / f"{out_path_base.stem}_diffgeo_grid.png"
    fig.savefig(grid_path)
    plt.close(fig)

    # Drift overlay: compare local/global alignment and drift on one panel
    fig2, ax = plt.subplots(1, 1, figsize=(8, 4))
    local_align_mean = trim(local["align_mean"])
    global_align_mean = trim(global_["align_mean"])
    drift_mean = trim(diff_geo["drift_mean"])

    mask = np.isfinite(global_align_mean)
    ax.plot(depths_plot[mask], global_align_mean[mask], label="Global alignment", marker='o')

    mask = np.isfinite(local_align_mean)
    ax.plot(depths_plot[mask], local_align_mean[mask], label="Local alignment", marker='o')

    mask = np.isfinite(drift_mean)
    ax.plot(depths_plot[mask], drift_mean[mask], label="Drift cos(u_l, v_l)", marker='o')

    ax.axhline(0.0, linestyle="--", color="gray", alpha=0.6)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Cosine / Coupling")
    ax.set_title("Drift overlay: alignment vs coupling")
    ax.legend()

    fig2.tight_layout()
    drift_path = out_path_base.parent / f"{out_path_base.stem}_diffgeo_drift.png"
    fig2.savefig(drift_path)
    plt.close(fig2)

def collect_geometry_sequential(
    cfg: ExperimentConfig,
    model,
    tokenizer,
    family_name: str,
    prompts: List[str],
    use_continuation_only: bool,
):
    print(f"\n[Geometry] Family: {family_name} | ContOnly: {use_continuation_only}")
    device = cfg.device
    layers = get_decoder_layers(model)
    num_layers = len(layers)
    
    aggregator = DepthGeometryAggregator(num_layers)
    
    for prompt in tqdm(prompts, desc="Processing Prompts"):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_input_tokens).to(device)
        prompt_len = inputs.input_ids.shape[1]
        
        # 1. Generate
        with torch.no_grad():
            gen_out = model.generate(
                **inputs, 
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
        full_seq = gen_out[0]  # (T,)
        
        # 2. Forward pass to get hidden states
        with torch.no_grad():
            out = model(
                input_ids=full_seq.unsqueeze(0).to(device),
                output_hidden_states=True,
                return_dict=True,
            )
        
        # hidden_states tuple: (Embed, L0, L1, ... ) -> Drop Embed
        hs_stack = (
            torch.stack(out.hidden_states[1:])
            .squeeze(1)
            .to(torch.float32)
            .cpu()
            .numpy()
        )
        # hs_stack: [Layers, SeqLen, Dim]
        
        seq_len = hs_stack.shape[1]
        start_t = prompt_len if use_continuation_only else 0
        
        # 3. Accumulate
        for t in range(start_t, seq_len):
            if np.random.rand() > cfg.token_subsample_rate:
                continue
            
            # [Layers, Dim] for token t
            token_states = hs_stack[:, t, :] 
            aggregator.accumulate_token(token_states)

    geo = aggregator.finalize()
    ci = compute_corridor_indices(geo)
    return {"family": family_name, "geometry": geo, "corridor_indices": ci}

# -----------------------------
# Corridor Steering Logic
# -----------------------------

@contextmanager
def corridor_shift(
    model,
    layer_indices: List[int],
    vec: torch.Tensor,
    alpha: float = 1.0,
    prompt_len: int = None,
):
    """
    Corridor steering hook:
    - Applies the same additive vector at specified transformer blocks.
    - Restricts the shift to continuation tokens if prompt_len is given.
    - Includes shape guards to avoid crashes if a hooked module's output
      does not match the residual stream shape.
    """
    param = next(model.parameters())
    vec_t = vec.to(device=param.device, dtype=param.dtype).view(1, 1, -1)

    layers = get_transformer_layers(model)
    hooks = []

    def make_hook():
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hs = output[0]
                rest = output[1:]
            else:
                hs = output
                rest = None

            # Shape guard: expect [batch, seq, dim] with dim == vec_t.size(-1)
            if hs is None or hs.dim() != 3 or hs.size(-1) != vec_t.size(-1):
                return output

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
        if 0 <= idx < len(layers):
            h = layers[idx].register_forward_hook(make_hook())
            hooks.append(h)

    try:
        yield
    finally:
        for h in hooks:
            h.remove()

@torch.inference_mode()
def generate_then_capture_blockA(model, tokenizer, text: str, max_new_tokens: int):
    """
    Generate continuation for `text` and then capture full hidden states
    for the combined prompt+continuation sequence (no hooks here).
    Returns a dict with prompt_len and a list of layer-wise arrays.
    """
    device = next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]

    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )
    full_ids = gen[0]

    out = model(
        input_ids=full_ids.unsqueeze(0),
        output_hidden_states=True,
        return_dict=True,
    )

    # hidden_states: (Embed, L0, L1, ...) -> list of (T, D)
    layers = [
        h[0].to(torch.float32).cpu().numpy()
        for h in out.hidden_states[1:]
    ]
    return {"prompt_len": int(prompt_len), "layers": layers}

def corridor_repr(
    rec: Dict,
    corridor_layers: List[int],
    use_cont_only: bool = True,
) -> np.ndarray:
    """
    Corridor representation for a single (base, perspective):
      - Select specified layers.
      - Restrict to continuation tokens if use_cont_only.
      - Concatenate tokens across layers.
      - Mean-pool and L2-normalize.
    """
    layers = rec["layers"]  # list of (T, D)
    seq_len = layers[0].shape[0]
    start_tok = rec["prompt_len"] if use_cont_only else 0

    chunks = []
    for li in corridor_layers:
        if li < 0 or li >= len(layers):
            continue
        H_l = layers[li][start_tok:seq_len, :]  # (T_cont, D)
        if H_l.size == 0:
            continue
        chunks.append(H_l)
    if not chunks:
        # Fallback: use first valid layer over full sequence
        H_l = layers[0][start_tok:seq_len, :]
        chunks.append(H_l)
    H = np.concatenate(chunks, axis=0)  # (T_total, D)
    v = H.mean(axis=0)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v

def build_corridor_direction(
    cfg: ExperimentConfig,
    model,
    tokenizer,
    corridor_layers: List[int],
) -> torch.Tensor:
    """
    Builds a (Lit - CS) direction using paired Base + Perspective prompts and
    corridor-style representations over continuation tokens.
    This matches the semantics of the original testbench code.
    """
    param = next(model.parameters())
    device = param.device

    deltas = []

    print(f"  [Vector] Building corridor (lit - cs) from layers {corridor_layers}...")

    for base in NEUTRAL_BASES:
        cs_text = f"{base}\n{PERS['cs']}"
        lit_text = f"{base}\n{PERS['lit']}"

        rec_cs = generate_then_capture_blockA(model, tokenizer, cs_text, cfg.max_new_tokens)
        rec_lit = generate_then_capture_blockA(model, tokenizer, lit_text, cfg.max_new_tokens)

        v_cs = corridor_repr(rec_cs, corridor_layers, use_cont_only=True)
        v_lit = corridor_repr(rec_lit, corridor_layers, use_cont_only=True)

        deltas.append(v_lit - v_cs)

    if not deltas:
        raise RuntimeError("No deltas accumulated while building corridor direction.")

    deltas_np = np.stack(deltas, axis=0)
    v = deltas_np.mean(axis=0)
    v = v / (np.linalg.norm(v) + 1e-8)

    direction = torch.tensor(v, device=device, dtype=param.dtype)
    return direction

def generate_with_steering_sequential(
    cfg: ExperimentConfig,
    model,
    tokenizer,
    prompts: List[str],
    layer_idx: int,
    direction: torch.Tensor,
    alpha: float,
):
    """
    Sequential generation using the corridor_shift hook.
    Steering is applied at a chosen layer index; the direction itself
    is computed from a multi-layer corridor representation.
    """
    device = cfg.device
    steering_vec = direction.to(device=device)

    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_input_tokens).to(device)
        prompt_len = inputs["input_ids"].shape[1]
        try:
            with corridor_shift(model, [layer_idx], steering_vec, alpha=alpha, prompt_len=prompt_len):
                with torch.no_grad():
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=cfg.steering_max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            results.append(text)
        except Exception as e:
            print(f"[Gen Error] Layer {layer_idx}, alpha {alpha}: {e}")
        
    return results

def cs_lit_score(text: str) -> float:
    cs_words = ["algorithm", "code", "complexity", "data", "search", "function"]
    lit_words = ["love", "night", "heart", "shadow", "dream", "sky"]
    t = text.lower()
    return sum(t.count(w) for w in lit_words) - sum(t.count(w) for w in cs_words)

# -----------------------------
# Main Routines
# -----------------------------

def run_all(cfg: ExperimentConfig):
    set_seed(cfg.seed)
    model_tag = get_model_tag(cfg.model_id)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {cfg.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, 
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model.eval()

    # --- Differential Geometry (always captured; plotting optional) ---
    print("--- Phase A0: Differential Geometry Capture ---")
    diff_geo = collect_differential_geometry(cfg, model, tokenizer)
    if cfg.run_diffgeo:
        plot_differential_geometry_dashboard(
            diff_geo,
            out_dir / f"{model_tag}_geometry",
        )

    # --- A1/A2: Geometry ---
    # Use neutral bases so CI is computed on the same manifold as steering.
    print("--- Phase A1: Geometry (Neutral Bases) ---")
    res_geo = collect_geometry_sequential(
        cfg, model, tokenizer, "neutral", NEUTRAL_BASES, True
    )

    # Corridor index from global alignment / curvature (DifferentialGeometryTracker)
    ci_global = compute_corridor_indices_global(diff_geo)["CI_global"]
    plot_geometry_aligned(
        res_geo["geometry"],
        ci_global,
        out_dir / f"{model_tag}_geometry.png",
    )

    if not cfg.run_steering:
        return

    layers = get_decoder_layers(model)
    L = len(layers)

    # --- A3: CI vs Gain ---
    print("\n--- Phase A3: CI vs Steering Gain ---")
    ci_arr = np.array(ci_global, dtype=float)
    center_layer = int(np.nanargmax(ci_arr))
    print(f"  Peak CI Layer: {center_layer}")

    # Corridor layers: CI > 0 (fallback to band around center if empty)
    corridor_layers = np.where(np.isfinite(ci_arr) & (ci_arr > 0))[0].tolist()
    if not corridor_layers:
        radius = max(1, L // 6)
        corridor_layers = list(range(max(0, center_layer - radius), min(L, center_layer + radius)))
    print(f"  Corridor layers used for direction: {corridor_layers}")

    # Learn corridor-based vector
    direction = build_corridor_direction(cfg, model, tokenizer, corridor_layers)
    
    # Sweep all layers with this vector
    neutral_prompts = build_validation_prompts()
    gains = []
    sweep_indices = range(L)
    
    for l_idx in tqdm(sweep_indices, desc="Layer Sweep"):
        # Measure score diff between alpha=0 and alpha=20
        outs_0 = generate_with_steering_sequential(cfg, model, tokenizer, neutral_prompts, l_idx, direction, 0.0)
        outs_20 = generate_with_steering_sequential(cfg, model, tokenizer, neutral_prompts, l_idx, direction, 20.0)
        
        score_0 = np.mean([cs_lit_score(x) for x in outs_0]) if outs_0 else 0.0
        score_20 = np.mean([cs_lit_score(x) for x in outs_20]) if outs_20 else 0.0
        gains.append(abs(score_20 - score_0))

    # Plot Correlation
    plt.figure()
    plt.scatter(ci_arr, gains, c=list(range(L)), cmap="viridis")
    plt.colorbar(label="Layer Depth")
    plt.xlabel("Corridor Index")
    plt.ylabel("Steering Gain")
    plt.title("A3: CI vs Control")
    plt.savefig(out_dir / f"{model_tag}_A3_correlation.png")
    plt.close()

    # --- A4: Band-wise Steering ---
    if cfg.run_band_steering:
        print("\n--- Phase A4: Band-wise Steering ---")
        radius = max(1, L // 6)
        
        bands = {
            "early": list(range(0, max(1, center_layer - radius))),
            "mid": list(range(max(0, center_layer - radius), min(L, center_layer + radius))),
            "late": list(range(min(L, center_layer + radius), L)),
        }
        
        # Ensure early band is valid
        if not bands["early"]:
            bands["early"] = [0, 1]

        results_a4 = {}
        
        plt.figure(figsize=(10, 6))
        
        for name, idxs in bands.items():
            print(f"  Band {name.upper()}: Layers {idxs}")
            
            # A. Learn band-specific corridor direction
            band_dir = build_corridor_direction(cfg, model, tokenizer, idxs)
            
            # B. Test at center of band
            target = idxs[len(idxs) // 2]
            
            # C. Sweep Alphas
            alpha_scores = []
            alphas = sorted(list(cfg.alpha_values))
            
            for a in alphas:
                outs = generate_with_steering_sequential(cfg, model, tokenizer, neutral_prompts, target, band_dir, a)
                s = np.mean([cs_lit_score(x) for x in outs]) if outs else 0.0
                alpha_scores.append(s)
            
            plt.plot(alphas, alpha_scores, marker='o', label=f"{name} (L{target})")
            results_a4[name] = {"layer": target, "scores": alpha_scores}
            
        plt.legend()
        plt.xlabel("Alpha")
        plt.ylabel("Score")
        plt.title("A4: Band-wise Sensitivity")
        plt.savefig(out_dir / f"{model_tag}_A4_band_steering.png")
        plt.close()
        print(f"Saved A4 plot to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--run_diffgeo", action="store_true")
    parser.add_argument("--run_steering", action="store_true")
    parser.add_argument("--run_band_steering", action="store_true")
    
    args = parser.parse_args()
    
    cfg = ExperimentConfig(
        model_id=args.model_id,
        run_diffgeo=args.run_diffgeo,
        run_steering=args.run_steering,
        run_band_steering=args.run_band_steering,
    )
    
    run_all(cfg)
