#!/usr/bin/env python
"""
geom_tangent_normal_corridor.py (with binormal & torsion-like metrics)

Adds:
- Binormal-like vectors: direction the tangent leaves the osculating plane.
- Torsion-like scalar: magnitude of that out-of-plane deviation.
- 3-panel plot: step size, curvature, torsion vs depth.

Model: Gemma-3-4b-it by default.
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
MODEL_ID = "google/gemma-3-4b-it"
OUTDIR = Path("geom_tangent_normal_outputs_4b")
OUTDIR.mkdir(parents=True, exist_ok=True)

MAX_NEW = 128
SEED = 42

BASE_QUESTION_ANALYSIS = "How do networks define individual and aggregate beliefs?"

# Small prompt banks for averaging geometry
PROMPTS_CS = [
    "Explain how distributed systems achieve fault tolerance.",
    "How does gradient descent optimize a neural network?",
    "Why do databases use indexing?",
    "What are the trade-offs between strong and eventual consistency?",
    "How do error-correcting codes improve reliability?",
]

PROMPTS_LIT = [
    "How does literature explore the theme of loneliness?",
    "What makes a character feel psychologically realistic?",
    "How do poets use imagery to convey emotion?",
    "Why do some stories feel timeless across cultures?",
    "How does narrative perspective shape our sympathy for characters?",
]

CS_SUFFIX = "Answer from a computer science perspective."
LIT_SUFFIX = "Answer from a literature and arts perspective."
NEUTRAL_SUFFIX = "Give a detailed, thoughtful answer."

# At least 25 base questions for corridor-style direction learning
BASE_QUESTIONS_MANY = [
    "Why do people form communities?",
    "Why do rumors spread so quickly?",
    "Why do some ideas go viral while others die out?",
    "Why do people conform to group opinions?",
    "Why do echo chambers form online?",
    "Why do conspiracy theories persist?",
    "Why do people follow trends?",
    "Why do some narratives dominate public discourse?",
    "Why do political movements rise and fall?",
    "Why do organizations resist change?",
    "Why do new technologies face resistance?",
    "Why do people adopt new tools at different speeds?",
    "Why do some innovations spread globally?",
    "Why do social networks create filter bubbles?",
    "Why do people trust some sources and not others?",
    "Why do certain stories feel universal?",
    "Why do symbols gain cultural power?",
    "Why do myths survive across generations?",
    "Why do people build idols and heroes?",
    "Why do propaganda campaigns succeed or fail?",
    "Why do some books become classics?",
    "Why do certain films become cult favorites?",
    "Why do people form ideological tribes?",
    "Why do revolutions inspire art and literature?",
    "Why do fears about automation keep returning?",
    "Why do people fear being left out?",
    "Why do crises reshape public opinion?",
    "Why do some institutions gain long-term legitimacy?",
    "Why do movements splinter into factions?",
]

# Steering strengths
STEER_ALPHAS_TN = [-40.0, -20.0, 0.0, 20.0, 40.0]
STEER_ALPHAS_CORRIDOR = [-80, -40, -20, 0, 20, 40, 80]

# crude lexical indicators for CS vs Lit orientation
KEYWORDS = {
    "cs": [
        "algorithm", "network", "node", "data", "statistical", "probability",
        "distribution", "graph", "input", "output", "learning", "reinforcement",
        "bayesian", "processing", "computational", "model", "vector", "signal",
        "parameter", "update", "gradient", "objective", "optimization"
    ],
    "lit": [
        "narrative", "story", "metaphor", "symbol", "aesthetic", "consciousness",
        "human", "experience", "identity", "culture", "unconscious", "sublime",
        "transcendence", "meaning", "myth", "interpretation", "poetic",
        "character", "emotion", "feeling", "loneliness", "desire", "longing"
    ]
}

torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------
# Model loading
# -----------------------------
print(f"Loading model: {MODEL_ID}")
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

print(f"num_layers={NUM_LAYERS}, hidden_size={HIDDEN_SIZE}, device={device}")
param_device = next(model.parameters()).device
param_dtype = next(model.parameters()).dtype

# -----------------------------
# Geometry helpers
# -----------------------------
def stack_and_mean(arrs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Stack 1D arrays (same length) -> (mean, std)."""
    A = np.stack(arrs, axis=0)  # (N, L)
    return A.mean(axis=0), A.std(axis=0)

def average_geom_over_prompts(prompts: List[str], suffix: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    For each prompt in `prompts`, build full prompt = prompt + suffix,
    capture final-token trajectory, and compute geometry.
    Returns dict of (mean, std) arrays for each metric.
    """
    step_list = []
    curv_list = []
    tors_list = []
    plane_rot_list = []

    for p in prompts:
        full = p + "\n" + suffix
        print(f"[avg-geom] capturing for: {full[:80].replace('\\n',' ')}...")
        ids, plen, layers = generate_then_capture(full)
        H = extract_final_token_trajectory(layers)
        geom = compute_geom_for_trajectory(H)

        step_list.append(geom["step_norms"])
        curv_list.append(geom["curvature"])
        tors_list.append(geom["torsion"])
        plane_rot_list.append(geom["plane_rot"])  # we'll add plane_rot below

    step_mean, step_std = stack_and_mean(step_list)
    curv_mean, curv_std = stack_and_mean(curv_list)
    tors_mean, tors_std = stack_and_mean(tors_list)
    rot_mean, rot_std = stack_and_mean(plane_rot_list)

    return {
        "step": (step_mean, step_std),
        "curvature": (curv_mean, curv_std),
        "torsion": (tors_mean, tors_std),
        "plane_rot": (rot_mean, rot_std),
    }

def l2_normalize_rows(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)


def orthonormalize(vectors: List[np.ndarray], eps: float = 1e-8) -> np.ndarray:
    """
    Simple Gram–Schmidt on a list of 1D vectors.
    Returns Q with shape (k, D), rows orthonormal.
    """
    ortho = []
    for v in vectors:
        v = v.astype(np.float64)
        for u in ortho:
            v = v - np.dot(v, u) * u
        n = np.linalg.norm(v)
        if n > eps:
            ortho.append(v / n)
    if not ortho:
        raise ValueError("All vectors collapsed in orthonormalize; check inputs.")
    return np.stack(ortho, axis=0).astype(np.float32)


def compute_geom_for_trajectory(H: np.ndarray) -> Dict[str, np.ndarray]:
    """
    H: (L, D) hidden states across depth for one fixed token position.

    Returns:
        step_vecs: (L-1, D)
        step_norms: (L-1,)
        tangents: (L-1, D)
        dT: (L-2, D)
        curvature: (L-2,)
        normals: (L-1, D)   (undefined at index 0 → zeros)
        binormals: (L-2, D) (direction leaving osculating plane)
        torsion: (L-2,)     (magnitude of out-of-plane deviation)
    """
    assert H.ndim == 2
    L, D = H.shape

    # First derivative: steps across depth
    V = H[1:] - H[:-1]                    # (L-1, D)
    step_norms = np.linalg.norm(V, axis=1)
    tangents = V / (step_norms[:, None] + 1e-8)  # unit T_l

    # Second derivative: change in tangent
    dT = tangents[1:] - tangents[:-1]     # (L-2, D)
    kappa = np.linalg.norm(dT, axis=1)    # curvature κ_l

    normals = np.zeros_like(tangents)
    mask = kappa > 1e-6
    normals[1:][mask] = dT[mask] / (kappa[mask, None] + 1e-8)  # unit N_l

    # Binormal & torsion-like metric:
    # at each step l, look at T_{l+1} and measure how much it wants to leave
    # the current osculating plane span{T_l, N_l}.
    Lsteps = L - 1
    binormals = np.zeros((Lsteps - 1, D), dtype=np.float32)
    torsion = np.zeros(Lsteps - 1, dtype=np.float32)

    for l in range(Lsteps - 1):
        T_l = tangents[l]
        N_l = normals[l]

        # Build plane basis: always include T_l; include N_l if non-trivial
        plane_vecs = [T_l]
        if np.linalg.norm(N_l) > 1e-6:
            plane_vecs.append(N_l)

        Q = orthonormalize(plane_vecs)  # (k, D), k=1 or 2

        T_next = tangents[l + 1]
        # Project T_{l+1} into current plane
        coeffs = Q @ T_next            # (k,)
        proj = coeffs @ Q              # (D,)
        resid = T_next - proj          # out-of-plane component

        mag = np.linalg.norm(resid)
        torsion[l] = mag
        if mag > 1e-6:
            binormals[l] = (resid / mag).astype(np.float32)
        else:
            binormals[l] = np.zeros(D, dtype=np.float32)

    # Osculating-plane rotation: principal angle between planes at l and l+1
    # Planes are span{T_l, N_l} and span{T_{l+1}, N_{l+1}}.
    Lsteps = L - 1
    plane_rot = np.zeros(Lsteps - 1, dtype=np.float32)

    for l in range(Lsteps - 1):
        # Plane at l
        T_l = tangents[l]
        N_l = normals[l]
        vecs1 = [T_l]
        if np.linalg.norm(N_l) > 1e-6:
            vecs1.append(N_l)
        Q1 = orthonormalize(vecs1)  # (k1, D), k1=1 or 2

        # Plane at l+1
        T_next = tangents[l + 1]
        N_next = normals[l + 1]
        vecs2 = [T_next]
        if np.linalg.norm(N_next) > 1e-6:
            vecs2.append(N_next)
        Q2 = orthonormalize(vecs2)  # (k2, D)

        # 2D principal angles between subspaces spanned by rows of Q1, Q2
        M = Q1 @ Q2.T  # (k1, k2)
        s = np.linalg.svd(M, compute_uv=False)
        # Clamp singular values to [-1,1] to avoid numerical junk
        s = np.clip(s, -1.0, 1.0)
        # Take largest principal angle as "plane rotation"
        theta_max = float(np.arccos(s.min()))  # rad
        plane_rot[l] = theta_max

    return {
        "step_vecs": V,
        "step_norms": step_norms,
        "tangents": tangents,
        "dT": dT,
        "curvature": kappa,
        "normals": normals,
        "binormals": binormals,
        "torsion": torsion,
        "plane_rot": plane_rot,
    }


def pca_2d(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (coords, basis) where coords are (L,2), basis is (D,2)."""
    Hc = H - H.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    basis = Vt[:2].T  # (D, 2)
    coords = Hc @ basis
    return coords, basis

# -----------------------------
# Reconstruction from (kappa, tau) sequences
# -----------------------------

def reconstruct_curve_from_kt(
    step_norms: np.ndarray,
    curvature: np.ndarray,
    torsion: np.ndarray,
    normalize_steps: bool = True,
) -> np.ndarray:
    """
    Discrete Frenet reconstruction of a 3D curve from step sizes, curvature and torsion.

    We work in R^3 with an initial Frenet frame (T0,N0,B0) and integrate:
        d/ds [T N B] = A(s) [T N B],
    where
        A = [[ 0,  κ,  0],
             [-κ, 0,  τ],
             [ 0,-τ,  0]]
    and then x' = T.

    Args:
        step_norms: (S,) array of ||h_{l+1}-h_l||, S = #steps = L-1.
        curvature:  (S-1,) array, κ_l defined between step l and l+1.
        torsion:    (S-1,) array, τ_l for same midpoints (use torsion-like metric).
        normalize_steps: if True, renormalize all ds to have mean 1 to avoid scale blow-up.

    Returns:
        X: (S, 3) reconstructed 3D positions for each step index (0..S-1).
    """
    step_norms = np.asarray(step_norms, dtype=np.float64)
    curvature = np.asarray(curvature, dtype=np.float64)
    torsion = np.asarray(torsion, dtype=np.float64)

    S = step_norms.shape[0]
    assert curvature.shape[0] == S - 1
    assert torsion.shape[0] == S - 1

    ds = step_norms.copy()
    if normalize_steps:
        mean_ds = ds.mean() + 1e-8
        ds = ds / mean_ds

    # Initial Frenet frame in R^3
    T = np.array([1.0, 0.0, 0.0])
    N = np.array([0.0, 1.0, 0.0])
    B = np.array([0.0, 0.0, 1.0])

    def re_orthonormalize(T, N, B):
        M = np.stack([T, N, B], axis=1)  # 3x3
        Q, _ = np.linalg.qr(M)
        return Q[:, 0], Q[:, 1], Q[:, 2]

    X = np.zeros((S, 3), dtype=np.float64)
    x = np.zeros(3, dtype=np.float64)  # start at origin
    X[0] = x

    for l in range(S - 1):
        kappa_l = curvature[l]
        tau_l = torsion[l]
        ds_l = ds[l]

        # Frenet matrix A_l
        A = np.array([
            [0.0,      kappa_l,  0.0],
            [-kappa_l, 0.0,      tau_l],
            [0.0,     -tau_l,    0.0],
        ])

        # Current frame as 3x3 (columns = T,N,B)
        F = np.stack([T, N, B], axis=1)  # 3x3

        # First-order Euler step for the frame: F' = A F
        F_next = F + ds_l * (A @ F)

        # Re-orthonormalize
        T, N, B = re_orthonormalize(F_next[:, 0], F_next[:, 1], F_next[:, 2])

        # Position update
        x = x + ds_l * T
        X[l + 1] = x

    return X


def align_and_plot_reconstruction(
    H: np.ndarray,
    geom: Dict[str, np.ndarray],
    label: str,
    outdir: Path,
):
    """
    Compare true hidden trajectory vs reconstructed Frenet curve.

    Steps:
      - Project H to 3D PCA.
      - Reconstruct X_rec from (step_norms, curvature, torsion).
      - Center and isotropically scale both.
      - Align by optimal rigid motion using Procrustes (SVD).
      - Plot overlaid curves in 3D and also their 2D (PC1,PC2) projections.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    step_norms = geom["step_norms"]         # (S,)
    curvature = geom["curvature"]           # (S-1,)
    torsion = geom["torsion"]               # (S-1,)

    # 3D PCA of true trajectory
    Hc = H - H.mean(axis=0, keepdims=True)
    U, Svals, Vt = np.linalg.svd(Hc, full_matrices=False)
    basis3 = Vt[:3].T                       # (D,3)
    Y_true = Hc @ basis3                   # (L,3) but we care about steps (L-1)

    # We defined step-based Frenet frame on steps 0..S-1 with S=L-1.
    # So we take first S points of Y_true.
    S_steps = step_norms.shape[0]
    Y_true = Y_true[:S_steps]

    # Reconstruct 3D curve from (κ,τ)
    X_rec = reconstruct_curve_from_kt(step_norms, curvature, torsion, normalize_steps=True)

    # Center both
    Y_true_c = Y_true - Y_true.mean(axis=0, keepdims=True)
    X_rec_c = X_rec - X_rec.mean(axis=0, keepdims=True)

    # Isotropic scale (so Procrustes is pure rotation)
    Y_norm = np.linalg.norm(Y_true_c)
    X_norm = np.linalg.norm(X_rec_c)
    Y_true_c /= (Y_norm + 1e-8)
    X_rec_c /= (X_norm + 1e-8)

    # Procrustes: find R minimizing ||Y - X R||
    # Here we map X_rec_c -> Y_true_c
    M = X_rec_c.T @ Y_true_c
    U2, _, Vt2 = np.linalg.svd(M)
    R = U2 @ Vt2
    X_aligned = X_rec_c @ R

    # 3D overlay plot
    fig = plt.figure(figsize=(10, 4))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    ax3d.plot(Y_true_c[:, 0], Y_true_c[:, 1], Y_true_c[:, 2],
              label="true (PCA3)", color="tab:blue")
    ax3d.plot(X_aligned[:, 0], X_aligned[:, 1], X_aligned[:, 2],
              label="reconstructed", color="tab:orange")
    ax3d.set_title(f"3D curve: true vs reconstructed ({label})")
    ax3d.legend()
    ax3d.grid(True, alpha=0.3)

    # 2D projection (just first two coordinates of each)
    ax2d.plot(Y_true_c[:, 0], Y_true_c[:, 1],
              label="true (PCA3→2D)", color="tab:blue")
    ax2d.plot(X_aligned[:, 0], X_aligned[:, 1],
              label="reconstructed", color="tab:orange")
    ax2d.set_title(f"2D projection: true vs reconstructed ({label})")
    ax2d.set_xlabel("coord 1")
    ax2d.set_ylabel("coord 2")
    ax2d.legend()
    ax2d.grid(True, alpha=0.3)

    fig.tight_layout()
    path = outdir / f"reconstruction_true_vs_frenet_{label}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved reconstruction comparison plot to {path}")

# -----------------------------
# Capturing hidden states
# -----------------------------
@torch.inference_mode()
def generate_then_capture(prompt: str, max_new: int = MAX_NEW):
    enc = tok(prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]
    gen_ids = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=max_new,
        use_cache=False,
    )[0]
    out = model(
        input_ids=gen_ids.unsqueeze(0),
        output_hidden_states=True,
        return_dict=True,
    )
    layers = [h[0].to(torch.float32).cpu().numpy() for h in out.hidden_states]
    return gen_ids.cpu().numpy(), prompt_len, layers


def extract_final_token_trajectory(layers: List[np.ndarray]) -> np.ndarray:
    L = len(layers)
    D = layers[0].shape[-1]
    last_idx = layers[0].shape[0] - 1
    H = np.stack([layers[i][last_idx] for i in range(L)], axis=0)
    assert H.shape == (L, D)
    return H

# -----------------------------
# Plotting helpers
# -----------------------------
def plot_step_curvature_torsion(
    depth_cs: Dict[str, np.ndarray],
    depth_lit: Dict[str, np.ndarray],
    outdir: Path,
):
    """
    3-panel plot:
        1. Step size vs depth
        2. Curvature vs depth
        3. Torsion-like vs depth
    """
    step_cs = depth_cs["step_norms"]
    step_lit = depth_lit["step_norms"]

    curv_cs = depth_cs["curvature"]
    curv_lit = depth_lit["curvature"]

    tors_cs = depth_cs["torsion"]
    tors_lit = depth_lit["torsion"]

    depths_step = np.arange(len(step_cs))
    depths_curv = np.arange(1, 1 + len(curv_cs))
    depths_tors = np.arange(1, 1 + len(tors_cs))

    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=False)

    # Step norms
    ax[0].plot(depths_step, step_cs, label="CS step size")
    ax[0].plot(depths_step, step_lit, label="Lit step size", linestyle="--")
    ax[0].set_xlabel("Depth index (between layers)")
    ax[0].set_ylabel("||h_{l+1} - h_l||")
    ax[0].set_title("Step size across depth")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Curvature
    ax[1].plot(depths_curv, curv_cs, label="CS curvature")
    ax[1].plot(depths_curv, curv_lit, label="Lit curvature", linestyle="--")
    ax[1].set_xlabel("Depth index (midpoints)")
    ax[1].set_ylabel("||T_{l+1} - T_l||")
    ax[1].set_title("Curvature (change in direction within plane)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    # Torsion-like metric
    ax[2].plot(depths_tors, tors_cs, label="CS torsion-like")
    ax[2].plot(depths_tors, tors_lit, label="Lit torsion-like", linestyle="--")
    ax[2].set_xlabel("Depth index (midpoints)")
    ax[2].set_ylabel("out-of-plane ||component||")
    ax[2].set_title("Torsion-like: how much T_{l+1} leaves osculating plane at depth l")
    ax[2].legend()
    ax[2].grid(True, alpha=0.3)

    fig.tight_layout()
    path = outdir / "step_curvature_torsion_cs_vs_lit.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved step/curvature/torsion plot to {path}")


def plot_pca_with_tangent_normal(
    H: np.ndarray,
    geom: Dict[str, np.ndarray],
    depth_idx: int,
    label: str,
    outdir: Path,
):
    """
    Plot 2D PCA of trajectory H plus tangent & normal at given depth index.
    depth_idx: step index (between layer depth_idx and depth_idx+1).
    """
    coords, basis = pca_2d(H)
    tangents = geom["tangents"]
    normals = geom["normals"]

    T_vec = tangents[depth_idx]
    N_vec = normals[depth_idx]

    T_proj = T_vec @ basis
    N_proj = N_vec @ basis

    def safe_unit(v):
        n = np.linalg.norm(v)
        return v / (n + 1e-8)

    T_dir = safe_unit(T_proj)
    N_dir = safe_unit(N_proj)

    step_sizes_2d = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
    typical = np.median(step_sizes_2d)
    arrow_len = 1.5 * typical

    origin = coords[depth_idx + 1]

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.plot(coords[:, 0], coords[:, 1], marker="o", linewidth=1.5)
    for i in range(len(coords)):
        ax.text(coords[i, 0], coords[i, 1], str(i), fontsize=8)

    t_handle = ax.arrow(
        origin[0], origin[1],
        arrow_len * T_dir[0], arrow_len * T_dir[1],
        head_width=0.1 * arrow_len,
        length_includes_head=True,
        color="tab:orange"
    )
    n_handle = ax.arrow(
        origin[0], origin[1],
        arrow_len * N_dir[0], arrow_len * N_dir[1],
        head_width=0.1 * arrow_len,
        length_includes_head=True,
        color="tab:green"
    )

    ax.set_title(f"2D PCA trajectory with tangent/normal at depth {depth_idx} ({label})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    ax.legend([t_handle, n_handle], ["tangent (first derivative)", "normal (curvature direction)"])

    fig.tight_layout()
    path = outdir / f"pca_tangent_normal_{label}_depth{depth_idx}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved PCA+tangent+normal plot to {path}")

def plot_avg_geom(cs_stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
                  lit_stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
                  outdir: Path):
    """
    cs_stats / lit_stats: output of average_geom_over_prompts.
    Produces 4-panel plot: step, curvature, torsion-like, plane rotation.
    """
    step_cs_m, step_cs_s = cs_stats["step"]
    step_lt_m, step_lt_s = lit_stats["step"]

    curv_cs_m, curv_cs_s = cs_stats["curvature"]
    curv_lt_m, curv_lt_s = lit_stats["curvature"]

    tors_cs_m, tors_cs_s = cs_stats["torsion"]
    tors_lt_m, tors_lt_s = lit_stats["torsion"]

    rot_cs_m, rot_cs_s = cs_stats["plane_rot"]
    rot_lt_m, rot_lt_s = lit_stats["plane_rot"]

    d_step = np.arange(len(step_cs_m))
    d_curv = np.arange(1, 1 + len(curv_cs_m))
    d_tors = np.arange(1, 1 + len(tors_cs_m))
    d_rot  = np.arange(1, 1 + len(rot_cs_m))

    fig, ax = plt.subplots(4, 1, figsize=(10, 12))

    # Step
    ax[0].plot(d_step, step_cs_m, label="CS")
    ax[0].fill_between(d_step, step_cs_m - step_cs_s, step_cs_m + step_cs_s, alpha=0.2)
    ax[0].plot(d_step, step_lt_m, label="Lit", linestyle="--")
    ax[0].fill_between(d_step, step_lt_m - step_lt_s, step_lt_m + step_lt_s, alpha=0.2)
    ax[0].set_title("Step size (mean ± std)")
    ax[0].set_xlabel("Depth index (between layers)")
    ax[0].set_ylabel("||h_{l+1} - h_l||")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Curvature
    ax[1].plot(d_curv, curv_cs_m, label="CS")
    ax[1].fill_between(d_curv, curv_cs_m - curv_cs_s, curv_cs_m + curv_cs_s, alpha=0.2)
    ax[1].plot(d_curv, curv_lt_m, label="Lit", linestyle="--")
    ax[1].fill_between(d_curv, curv_lt_m - curv_lt_s, curv_lt_m + curv_lt_s, alpha=0.2)
    ax[1].set_title("Curvature (mean ± std)")
    ax[1].set_xlabel("Depth index (midpoints)")
    ax[1].set_ylabel("||T_{l+1} - T_l||")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    # Torsion-like
    ax[2].plot(d_tors, tors_cs_m, label="CS")
    ax[2].fill_between(d_tors, tors_cs_m - tors_cs_s, tors_cs_m + tors_cs_s, alpha=0.2)
    ax[2].plot(d_tors, tors_lt_m, label="Lit", linestyle="--")
    ax[2].fill_between(d_tors, tors_lt_m - tors_lt_s, tors_lt_m + tors_lt_s, alpha=0.2)
    ax[2].set_title("Torsion-like (out-of-plane component)")
    ax[2].set_xlabel("Depth index (midpoints)")
    ax[2].set_ylabel("||out-of-plane||")
    ax[2].legend()
    ax[2].grid(True, alpha=0.3)

    # Plane rotation
    ax[3].plot(d_rot, rot_cs_m, label="CS")
    ax[3].fill_between(d_rot, rot_cs_m - rot_cs_s, rot_cs_m + rot_cs_s, alpha=0.2)
    ax[3].plot(d_rot, rot_lt_m, label="Lit", linestyle="--")
    ax[3].fill_between(d_rot, rot_lt_m - rot_lt_s, rot_lt_m + rot_lt_s, alpha=0.2)
    ax[3].set_title("Osculating-plane rotation (principal angle, mean ± std)")
    ax[3].set_xlabel("Depth index (midpoints)")
    ax[3].set_ylabel("radians")
    ax[3].legend()
    ax[3].grid(True, alpha=0.3)

    fig.tight_layout()
    path = outdir / "avg_step_curv_tors_plane_cs_vs_lit.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved averaged geometry plot to {path}")

# -----------------------------
# Steering machinery
# -----------------------------
TRANSFORMER_LAYERS = None

def get_transformer_layers(expected_n_layers: int = None):
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
        raise RuntimeError("Could not find ModuleList with expected_n_layers")
    print("Using transformer layer list from:", candidates[0][0])
    TRANSFORMER_LAYERS = candidates[0][1]
    return TRANSFORMER_LAYERS


class LayerSteerer:
    """
    Add a fixed vector to hidden states at specified layers
    for continuation tokens only.
    """

    def __init__(self, layers: List[int], vec_np: np.ndarray, alpha: float, prompt_len: int):
        self.layers = layers
        self.vec_t = torch.tensor(vec_np, device=param_device, dtype=param_dtype).view(1, 1, -1)
        self.alpha = alpha
        self.prompt_len = prompt_len
        self.handles = []

    def __enter__(self):
        modules = get_transformer_layers()
        vec_t = self.vec_t
        alpha = self.alpha
        prompt_len = self.prompt_len

        def make_hook():
            def hook(module, inputs, output):
                if isinstance(output, tuple):
                    hs = output[0]
                    rest = output[1:]
                else:
                    hs = output
                    rest = None
                hs_new = hs.clone()
                hs_new[:, prompt_len:, :] += alpha * vec_t
                if rest is None:
                    return hs_new
                else:
                    return (hs_new, *rest)
            return hook

        for idx in self.layers:
            h = modules[idx].register_forward_hook(make_hook())
            self.handles.append(h)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.handles:
            h.remove()


@torch.inference_mode()
def generate_with_vector(prompt: str, layer_indices: List[int], vec_np: np.ndarray, alpha: float) -> str:
    enc = tok(prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]
    with LayerSteerer(layer_indices, vec_np, alpha, prompt_len):
        gen_ids = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=MAX_NEW,
            use_cache=False,
        )[0]
    completion = tok.decode(gen_ids[prompt_len:], skip_special_tokens=True)
    return completion


@torch.inference_mode()
def generate_baseline(prompt: str) -> str:
    enc = tok(prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]
    gen_ids = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=MAX_NEW,
        use_cache=False,
    )[0]
    completion = tok.decode(gen_ids[prompt_len:], skip_special_tokens=True)
    return completion

# -----------------------------
# Scoring of completions
# -----------------------------
def count_keywords(text: str, category: str) -> int:
    text = text.lower()
    return sum(1 for w in KEYWORDS[category] if w in text)


def score_cs_vs_lit(text: str) -> Tuple[int, int, int]:
    cs = count_keywords(text, "cs")
    lit = count_keywords(text, "lit")
    diff = lit - cs
    return cs, lit, diff

# -----------------------------
# Corridor-style reps (for mode B)
# -----------------------------
def corridor_repr(rec: Dict, corridor_layers: List[int], use_cont_only: bool = True) -> np.ndarray:
    layers = rec["layers"]
    seq_len = layers[0].shape[0]
    start_tok = rec["prompt_len"] if use_cont_only else 0
    chunks = []
    for li in corridor_layers:
        H_l = layers[li][start_tok:seq_len, :]
        chunks.append(H_l)
    H = np.concatenate(chunks, axis=0)
    v = H.mean(axis=0)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v


def build_corridor_reps(all_records: List[Dict], corridor_layers: List[int]) -> Dict[int, Dict[str, np.ndarray]]:
    reps = {}
    for rec in all_records:
        base_id = rec["base_id"]
        persp = rec["perspective"]
        reps.setdefault(base_id, {})
        reps[base_id][persp] = corridor_repr(rec, corridor_layers)
    return reps


def learn_direction(reps: Dict[int, Dict[str, np.ndarray]], from_persp: str, to_persp: str) -> np.ndarray:
    deltas = []
    for base_id, d in reps.items():
        if from_persp in d and to_persp in d:
            deltas.append(d[to_persp] - d[from_persp])
    if not deltas:
        raise RuntimeError("No overlapping bases for direction learning.")
    deltas = np.stack(deltas, axis=0)
    v = deltas.mean(axis=0)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v

# -----------------------------
# Main experiment
# -----------------------------
def main():
    # 1) CS vs Lit geometry for a single analysis question
    cs_prompt = BASE_QUESTION_ANALYSIS + "\n" + CS_SUFFIX
    lit_prompt = BASE_QUESTION_ANALYSIS + "\n" + LIT_SUFFIX

    print("\n=== Capturing CS trajectory (analysis question) ===")
    ids_cs, prompt_len_cs, layers_cs = generate_then_capture(cs_prompt)
    H_cs = extract_final_token_trajectory(layers_cs)
    geom_cs = compute_geom_for_trajectory(H_cs)

    print("\n=== Capturing Lit trajectory (analysis question) ===")
    ids_lit, prompt_len_lit, layers_lit = generate_then_capture(lit_prompt)
    H_lit = extract_final_token_trajectory(layers_lit)
    geom_lit = compute_geom_for_trajectory(H_lit)

    plot_step_curvature_torsion(geom_cs, geom_lit, OUTDIR)

    # --- New: Frenet-style reconstruction experiments ---
    print("\n=== Frenet reconstruction from (kappa, tau) ===")
    align_and_plot_reconstruction(H_cs, geom_cs, "cs", OUTDIR)
    align_and_plot_reconstruction(H_lit, geom_lit, "lit", OUTDIR)

    # pick a low-curvature mid-layer as corridor-like index for tangent/normal
    curv = geom_cs["curvature"]
    valid_start = 4
    valid_end = len(curv) - 4 if len(curv) > 8 else len(curv)
    if valid_end <= valid_start:
        best_idx = int(np.argmin(curv))
    else:
        segment = curv[valid_start:valid_end]
        best_idx = valid_start + int(np.argmin(segment))
    steer_step_idx = best_idx + 1      # corresponds to step index
    steer_layer_idx = steer_step_idx   # use same index as transformer block

    print(f"\n[Mode A] Chosen steering depth (step index) = {steer_step_idx}, "
          f"approx transformer layer = {steer_layer_idx}")

    plot_pca_with_tangent_normal(H_cs, geom_cs, steer_step_idx, "cs", OUTDIR)
    plot_pca_with_tangent_normal(H_lit, geom_lit, steer_step_idx, "lit", OUTDIR)

    print("\n=== Averaged geometry over prompt banks ===")
    cs_stats = average_geom_over_prompts(PROMPTS_CS, "\n" + CS_SUFFIX)
    lit_stats = average_geom_over_prompts(PROMPTS_LIT, "\n" + LIT_SUFFIX)
    plot_avg_geom(cs_stats, lit_stats, OUTDIR)

    tangents = geom_cs["tangents"]
    normals = geom_cs["normals"]

    t_vec = tangents[steer_step_idx]
    n_vec = normals[steer_step_idx]
    if np.linalg.norm(n_vec) < 1e-6:
        print("Normal at chosen depth is near zero; falling back to adjacent index.")
        if steer_step_idx + 1 < len(normals):
            n_vec = normals[steer_step_idx + 1]
        elif steer_step_idx - 1 >= 0:
            n_vec = normals[steer_step_idx - 1]

    t_vec = t_vec / (np.linalg.norm(t_vec) + 1e-8)
    n_vec = n_vec / (np.linalg.norm(n_vec) + 1e-8)

    # 2) Mode A: Tangent vs Normal steering on neutral prompt
    neutral_prompt = BASE_QUESTION_ANALYSIS + "\n" + NEUTRAL_SUFFIX

    print("\n=== [Mode A] Baseline neutral completion ===")
    base_completion = generate_baseline(neutral_prompt)
    base_cs, base_lit, base_diff = score_cs_vs_lit(base_completion)
    print(f"[baseline] CS={base_cs}, Lit={base_lit}, Lit-CS={base_diff}")
    print(base_completion[:400].replace("\n", " ") + "...\n")

    print("=== [Mode A] Tangent steering results (single-layer) ===")
    for a in STEER_ALPHAS_TN:
        comp = generate_with_vector(neutral_prompt, [steer_layer_idx], t_vec, a)
        cs_c, lit_c, diff_c = score_cs_vs_lit(comp)
        print(f"[tangent alpha={a:+.1f}] CS={cs_c}, Lit={lit_c}, Lit-CS={diff_c}")
        print(comp[:300].replace("\n", " ") + "...\n")

    print("=== [Mode A] Normal steering results (single-layer) ===")
    for a in STEER_ALPHAS_TN:
        comp = generate_with_vector(neutral_prompt, [steer_layer_idx], n_vec, a)
        cs_c, lit_c, diff_c = score_cs_vs_lit(comp)
        print(f"[normal  alpha={a:+.1f}] CS={cs_c}, Lit={lit_c}, Lit-CS={diff_c}")
        print(comp[:300].replace("\n", " ") + "...\n")

    # 3) Mode B: Corridor-style direction learned from many base questions
    print("\n=== [Mode B] Building CS/Lit records for many base questions ===")
    all_records: List[Dict] = []
    for base_id, q in enumerate(BASE_QUESTIONS_MANY):
        for persp, suffix in [("cs", CS_SUFFIX), ("lit", LIT_SUFFIX)]:
            prompt = q + "\n" + suffix
            ids, prompt_len, layers = generate_then_capture(prompt)
            all_records.append({
                "base_id": base_id,
                "perspective": persp,
                "prompt": prompt,
                "prompt_len": prompt_len,
                "layers": layers,
            })
    print(f"Collected {len(all_records)} records "
          f"({len(BASE_QUESTIONS_MANY)} base questions × 2 perspectives).")

    band_radius = 3
    corridor_layers = list(range(
        max(1, steer_layer_idx - band_radius),
        min(NUM_LAYERS - 1, steer_layer_idx + band_radius + 1),
    ))
    print(f"[Mode B] Corridor layers for representations: {corridor_layers}")

    reps = build_corridor_reps(all_records, corridor_layers)
    direction_vec = learn_direction(reps, "cs", "lit")
    print("[Mode B] Learned CS->Lit corridor direction vector.")

    # 4) Mode B: Corridor-style steering on neutral prompt (for comparison)
    print("\n=== [Mode B] Corridor steering on neutral prompt ===")
    print("Baseline (reprinted for convenience):")
    print(base_completion[:300].replace("\n", " ") + "...\n")

    for a in STEER_ALPHAS_CORRIDOR:
        comp = generate_with_vector(neutral_prompt, corridor_layers, direction_vec, a)
        cs_c, lit_c, diff_c = score_cs_vs_lit(comp)
        print(f"[corridor alpha={a:+.1f}] CS={cs_c}, Lit={lit_c}, Lit-CS={diff_c}")
        print(comp[:300].replace("\n", " ") + "...\n")

    print("\nDone. Check PNGs in", OUTDIR)


if __name__ == "__main__":
    main()
