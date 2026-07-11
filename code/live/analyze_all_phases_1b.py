#!/usr/bin/env python
"""
Analyze NPZ outputs from Phases 1–4 for Gemma-3-1B and produce cross-phase,
"conclusive" plots.

Assumes you've already run:

  phase1_token_cloud.py          -> phase1_token_cloud_outputs_1b/
  phase2_token_transport.py      -> phase2_token_transport_outputs_1b/
  phase3_token_attn_mlp.py       -> phase3_layer_update_outputs_1b/
  phase4_token_trajectory.py     -> phase4_token_trajectory_outputs_1b/

and that their NPZ filenames follow the conventions in those scripts.

This script:

  * Loads per-family metrics from all phases.
  * Aligns them by layer index.
  * Produces:
      - Per-family 4-panel depth plots (one row per phase).
      - Optional GSM8K correct vs incorrect overlays from Phase 4.
      - A crude "corridor index" per layer that combines phases.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------

MODEL_ID = "google/gemma-3-1b-it"
MODEL_ID_SANITIZED = MODEL_ID.replace("/", "_")

# Default directories; change if you used different OUTDIRs
PHASE1_DIR = Path("phase1_token_cloud_outputs_1b")
PHASE2_DIR = Path("phase2_token_transport_outputs_1b")
PHASE3_DIR = Path("phase3_layer_update_outputs_1b")
PHASE4_DIR = Path("phase4_token_trajectory_outputs_1b")

OUT_ANALYSIS = Path("analysis_all_phases_1b")

# Roles and groups we care about from Phase 4
PHASE4_ROLES = ["bos_token", "last_token", "last_number_token"]
PHASE4_GROUPS = ["all", "correct", "incorrect"]  # 'correct' / 'incorrect' only exist for GSM8K


# -----------------------------
# Families from prompt_bank
# -----------------------------

try:
    from prompt_bank import PHASE1_FAMILIES  # type: ignore
    FAMILY_NAMES = list(PHASE1_FAMILIES.keys())
except ImportError:
    # Fallback; adjust manually if needed
    FAMILY_NAMES = ["general_qa", "gsm8k_math", "stories", "code"]


# -----------------------------
# Dataclasses
# -----------------------------

@dataclass
class PhaseBundle:
    """All NPZ-backed metrics for one family."""
    phase1: Optional[Dict[str, np.ndarray]]  # token cloud / oversmoothing
    phase2: Optional[Dict[str, np.ndarray]]  # token transport
    phase3: Optional[Dict[str, np.ndarray]]  # attn/MLP updates
    # phase4[role][group] -> metrics dict (keys like 'all_step_norm_mean', etc)
    phase4: Dict[str, Dict[str, Dict[str, np.ndarray]]]


# -----------------------------
# NPZ loading helpers
# -----------------------------

def _load_npz(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        print(f"[warn] Missing NPZ: {path}")
        return None
    arr = np.load(path, allow_pickle=False)
    return {k: arr[k] for k in arr.files}


def load_phase1_family_metrics(base_dir: Path, family: str) -> Optional[Dict[str, np.ndarray]]:
    """
    phase1_token_cloud_outputs_1b/<family>/phase1_metrics_<family>.npz
    """
    path = base_dir / family / f"phase1_metrics_{family}.npz"
    return _load_npz(path)


def load_phase2_family_metrics(base_dir: Path, family: str) -> Optional[Dict[str, np.ndarray]]:
    """
    phase2_token_transport_outputs_1b/google_gemma-3-1b-it_phase2_transport_<family>.npz
    """
    path = base_dir / f"{MODEL_ID_SANITIZED}_phase2_transport_{family}.npz"
    return _load_npz(path)


def load_phase3_family_metrics(base_dir: Path, family: str) -> Optional[Dict[str, np.ndarray]]:
    """
    phase3_layer_update_outputs_1b/<family>/phase3_metrics_<family>.npz
    """
    path = base_dir / family / f"phase3_metrics_{family}.npz"
    return _load_npz(path)


def load_phase4_family_metrics(
    base_dir: Path,
    family: str,
    roles: List[str],
    groups: List[str],
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    phase4_token_trajectory_outputs_1b/<family>/phase4_metrics_<family>_<role>_<group>.npz

    Returns:
      phase4[role][group] -> { 'layer_indices', 'all_step_norm_mean', ... }-style dicts.
    """
    family_dir = base_dir / family
    out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for role in roles:
        role_dict: Dict[str, Dict[str, np.ndarray]] = {}
        for group in groups:
            path = family_dir / f"phase4_metrics_{family}_{role}_{group}.npz"
            if not path.exists():
                continue
            d = _load_npz(path)
            if d is not None:
                role_dict[group] = d
        if role_dict:
            out[role] = role_dict
    return out


# -----------------------------
# Utility: align layers
# -----------------------------

def get_common_layers(bundle: PhaseBundle) -> np.ndarray:
    """
    Try to find a common layer_indices array across phases.
    Assumes all phases for a given model use the same number of layers.
    """
    candidates = []
    if bundle.phase1 is not None and "layer_indices" in bundle.phase1:
        candidates.append(bundle.phase1["layer_indices"])
    if bundle.phase2 is not None and "layer_indices" in bundle.phase2:
        candidates.append(bundle.phase2["layer_indices"])
    if bundle.phase3 is not None and "layer_indices" in bundle.phase3:
        candidates.append(bundle.phase3["layer_indices"])
    for role_map in bundle.phase4.values():
        for group_dict in role_map.values():
            if "layer_indices" in group_dict:
                candidates.append(group_dict["layer_indices"])
                break

    if not candidates:
        raise ValueError("No layer_indices found for this family.")

    base = candidates[0]
    for c in candidates[1:]:
        if not np.array_equal(base, c):
            print("[warn] layer_indices mismatch across phases; using first as reference.")
            break
    return base


def zscore(x: np.ndarray) -> np.ndarray:
    """Simple z-score along layer dimension, ignoring NaNs."""
    x = np.asarray(x, dtype=np.float32)
    mu = np.nanmean(x)
    sigma = np.nanstd(x)
    if not np.isfinite(sigma) or sigma == 0.0:
        return np.zeros_like(x)
    return (x - mu) / (sigma + 1e-8)


# -----------------------------
# Corridor index (crude heuristic)
# -----------------------------

def compute_corridor_index(layers: np.ndarray, bundle: PhaseBundle) -> np.ndarray:
    """
    Combine a few "good corridor" signals into a single per-layer score:

      + Phase 1: high mean_cos, low silhouette  -> oversmoothing/semantic soup.
      + Phase 2: low abs delta_layer (info stays at roughly same depth).
      + Phase 3: θ_att_mlp near 90 degrees (orthogonal updates).
      + Phase 4: low step_norm, high cos_to_final (stable, aligned trajectories).

    Output: CI[l] ~ higher means "more corridor-like".
    """
    L = len(layers)
    ci_terms = []

    # Phase 1
    if bundle.phase1 is not None:
        p1 = bundle.phase1
        mean_cos = p1.get("mean_cos_mean", np.full(L, np.nan))
        silhouette = p1.get("silhouette_mean", np.full(L, np.nan))
        s1 = zscore(mean_cos)          # want high
        s2 = -zscore(silhouette)       # want low silhouette
        ci_terms.append(s1)
        ci_terms.append(s2)

    # Phase 2
    if bundle.phase2 is not None:
        p2 = bundle.phase2
        # corridor-like: short effective delta_layer on average
        mean_abs_dl = p2.get("mean_abs_delta_layer", np.full(L, np.nan))
        s3 = -zscore(mean_abs_dl)
        ci_terms.append(s3)

    # Phase 3
    if bundle.phase3 is not None:
        p3 = bundle.phase3
        theta = p3.get("theta_att_mlp_mean", np.full(L, np.nan))  # radians
        theta_deg = np.degrees(theta)
        # Score is high when |theta - 90°| is small
        orth = -zscore(np.abs(theta_deg - 90.0))
        ci_terms.append(orth)

    # Phase 4 (using last_token, group="all" if available)
    p4_last_all = None
    role_map = bundle.phase4.get("last_token")
    if role_map is not None:
        p4_last_all = role_map.get("all")

    if p4_last_all is not None:
        step = p4_last_all.get("all_step_norm_mean", np.full(L, np.nan))
        cos_final = p4_last_all.get("all_cos_to_final_mean", np.full(L, np.nan))
        s4 = -zscore(step)           # low step_norm inside corridor
        s5 = zscore(cos_final)       # high alignment to final
        ci_terms.append(s4)
        ci_terms.append(s5)

    if not ci_terms:
        return np.full(L, np.nan, dtype=np.float32)

    ci = np.nanmean(np.stack(ci_terms, axis=0), axis=0)
    return ci.astype(np.float32)


# -----------------------------
# Plotting helpers
# -----------------------------

def plot_cross_phase_panels_for_family(
    family: str,
    bundle: PhaseBundle,
    out_dir: Path,
):
    """
    4-row plot (one per phase) showing depth-wise metrics for a single family.
    Also overlays the corridor index as a shaded band.
    """
    layers = get_common_layers(bundle)
    L = len(layers)

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    fig.suptitle(f"Cross-phase depth profile – {family}", fontsize=14)

    # -----------------
    # Phase 1 row
    # -----------------
    ax = axes[0]
    if bundle.phase1 is not None:
        p1 = bundle.phase1
        mean_cos = p1.get("mean_cos_mean", np.full(L, np.nan))
        var_cos = p1.get("var_cos_mean", np.full(L, np.nan))
        silhouette = p1.get("silhouette_mean", np.full(L, np.nan))
        ax.plot(layers, mean_cos, label="mean_cos")
        ax.plot(layers, var_cos, label="var_cos")
        ax.plot(layers, silhouette, label="silhouette")
        ax.set_ylabel("Phase 1")
        ax.set_title("Token cloud: oversmoothing & cluster structure")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Phase 1 metrics missing", ha="center", va="center")
        ax.set_ylabel("Phase 1")

    # -----------------
    # Phase 2 row
    # -----------------
    ax = axes[1]
    if bundle.phase2 is not None:
        p2 = bundle.phase2
        mean_best_sim = p2.get("mean_best_sim", np.full(L, np.nan))
        mean_abs_dl = p2.get("mean_abs_delta_layer", np.full(L, np.nan))
        mean_abs_dt = p2.get("mean_abs_delta_token", np.full(L, np.nan))
        ax.plot(layers, mean_best_sim, label="mean_best_sim")
        ax.plot(layers, mean_abs_dl, label="mean_abs_delta_layer")
        ax.plot(layers, mean_abs_dt, label="mean_abs_delta_token")
        ax.set_ylabel("Phase 2")
        ax.set_title("Token transport: intensity & depth/position shifts")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Phase 2 metrics missing", ha="center", va="center")
        ax.set_ylabel("Phase 2")

    # -----------------
    # Phase 3 row
    # -----------------
    ax = axes[2]
    if bundle.phase3 is not None:
        p3 = bundle.phase3
        na = p3.get("na_mean", np.full(L, np.nan))
        nm = p3.get("nm_mean", np.full(L, np.nan))
        theta_att_mlp = np.degrees(p3.get("theta_att_mlp_mean", np.full(L, np.nan)))
        r_att = p3.get("r_att_mean", np.full(L, np.nan))
        r_mlp = p3.get("r_mlp_mean", np.full(L, np.nan))

        ax.plot(layers, na, label="||Δ_att||")
        ax.plot(layers, nm, label="||Δ_mlp||")
        ax.plot(layers, theta_att_mlp, label="angle(Δ_att, Δ_mlp) [deg]")
        ax.plot(layers, r_att, label="r_att")
        ax.plot(layers, r_mlp, label="r_mlp")
        ax.set_ylabel("Phase 3")
        ax.set_title("Update decomposition: norms & attn/MLP angles")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncol=2)
    else:
        ax.text(0.5, 0.5, "Phase 3 metrics missing", ha="center", va="center")
        ax.set_ylabel("Phase 3")

    # -----------------
    # Phase 4 row
    # -----------------
    ax = axes[3]
    p4_last_all = None
    role_map = bundle.phase4.get("last_token")
    if role_map is not None:
        p4_last_all = role_map.get("all")

    if p4_last_all is not None:
        step = p4_last_all.get("all_step_norm_mean", np.full(L, np.nan))
        curv = p4_last_all.get("all_curvature_norm_mean", np.full(L, np.nan))
        cos_final = p4_last_all.get("all_cos_to_final_mean", np.full(L, np.nan))
        cos_logit = p4_last_all.get("all_cos_to_logit_dir_mean", np.full(L, np.nan))

        ax.plot(layers, step, label="step_norm")
        ax.plot(layers, curv, label="curvature_norm")
        ax.plot(layers, cos_final, label="cos_to_final")
        ax.plot(layers, cos_logit, label="cos_to_logit_dir")
        ax.set_ylabel("Phase 4")
        ax.set_title("Trajectory geometry (last token)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Phase 4 metrics for last_token/all missing",
                ha="center", va="center")
        ax.set_ylabel("Phase 4")

    axes[-1].set_xlabel("layer index (0 = first transformer block)")

    # -----------------
    # Corridor index overlay (on all rows)
    # -----------------
    ci = compute_corridor_index(layers, bundle)
    if np.any(np.isfinite(ci)):
        ci_norm = (ci - np.nanmin(ci)) / (np.nanmax(ci) - np.nanmin(ci) + 1e-8)
        # Mark layers above some percentile as "corridor band"
        thresh = np.nanpercentile(ci_norm, 75)
        corridor_mask = ci_norm >= thresh

        for ax in axes:
            for li, l in enumerate(layers):
                if not np.isfinite(ci_norm[li]) or not corridor_mask[li]:
                    continue
                ax.axvspan(l - 0.5, l + 0.5, alpha=0.08, color="grey")

        # Also add a small inset showing the corridor index itself
        fig_ci, ax_ci = plt.subplots(1, 1, figsize=(8, 3))
        ax_ci.plot(layers, ci_norm, marker="o")
        ax_ci.axhline(thresh, linestyle="--", alpha=0.5, label="75th percentile")
        ax_ci.set_title(f"Corridor index (normalized) – {family}")
        ax_ci.set_xlabel("layer index")
        ax_ci.set_ylabel("CI (0–1)")
        ax_ci.grid(True, alpha=0.3)
        ax_ci.legend()
        fig_ci.tight_layout()
        fig_ci.savefig(out_dir / f"{family}_corridor_index.png", dpi=200)
        plt.close(fig_ci)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_dir / f"{family}_cross_phase_depth.png", dpi=200)
    plt.close(fig)
    print(f"[plot] Saved cross-phase panels for {family}.")


def plot_gsm8k_correct_incorrect(bundle: PhaseBundle, out_dir: Path):
    """
    If GSM8K family exists and Phase 4 saved correct/incorrect,
    overlay curvature and cos_to_final for last_token.
    """
    family = "gsm8k_math"
    if family not in FAMILY_NAMES:
        return

    role_map = bundle.phase4.get("last_token")
    if role_map is None:
        return

    correct = role_map.get("correct")
    incorrect = role_map.get("incorrect")
    if correct is None or incorrect is None:
        return

    layers = correct["layer_indices"]
    curv_c = correct.get("correct_curvature_norm_mean", None)
    curv_i = incorrect.get("incorrect_curvature_norm_mean", None)
    cos_c = correct.get("correct_cos_to_final_mean", None)
    cos_i = incorrect.get("incorrect_cos_to_final_mean", None)

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    fig.suptitle("GSM8K – correct vs incorrect (last_token)", fontsize=14)

    ax = axes[0]
    if curv_c is not None and curv_i is not None:
        ax.plot(layers, curv_c, label="correct curvature")
        ax.plot(layers, curv_i, label="incorrect curvature")
    ax.set_ylabel("curvature_norm")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    if cos_c is not None and cos_i is not None:
        ax.plot(layers, cos_c, label="correct cos_to_final")
        ax.plot(layers, cos_i, label="incorrect cos_to_final")
    ax.set_ylabel("cos_to_final")
    ax.set_xlabel("layer index")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "gsm8k_last_token_correct_incorrect.png", dpi=200)
    plt.close(fig)
    print("[plot] Saved GSM8K correct vs incorrect overlays (Phase 4).")


# -----------------------------
# Main
# -----------------------------

def build_bundle_for_family(family: str, args) -> PhaseBundle:
    p1 = load_phase1_family_metrics(args.phase1_dir, family)
    p2 = load_phase2_family_metrics(args.phase2_dir, family)
    p3 = load_phase3_family_metrics(args.phase3_dir, family)
    p4 = load_phase4_family_metrics(args.phase4_dir, family, PHASE4_ROLES, PHASE4_GROUPS)
    return PhaseBundle(phase1=p1, phase2=p2, phase3=p3, phase4=p4)


def main():
    parser = argparse.ArgumentParser(description="Cross-phase analysis for Gemma-3-1B.")
    parser.add_argument("--phase1_dir", type=Path, default=PHASE1_DIR)
    parser.add_argument("--phase2_dir", type=Path, default=PHASE2_DIR)
    parser.add_argument("--phase3_dir", type=Path, default=PHASE3_DIR)
    parser.add_argument("--phase4_dir", type=Path, default=PHASE4_DIR)
    parser.add_argument("--out_dir", type=Path, default=OUT_ANALYSIS)
    parser.add_argument(
        "--families",
        nargs="*",
        default=FAMILY_NAMES,
        help="Which families to include (default: all from prompt_bank).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build bundles and plot per family
    bundles: Dict[str, PhaseBundle] = {}
    for fam in args.families:
        print(f"\n[family] {fam}")
        bundle = build_bundle_for_family(fam, args)
        bundles[fam] = bundle
        fam_out = args.out_dir / fam
        plot_cross_phase_panels_for_family(fam, bundle, fam_out)

    # GSM8K correct vs incorrect overlay if applicable
    if "gsm8k_math" in bundles:
        plot_gsm8k_correct_incorrect(bundles["gsm8k_math"], args.out_dir / "gsm8k_math")

    print("\n[done] Cross-phase analysis complete.")


if __name__ == "__main__":
    main()
