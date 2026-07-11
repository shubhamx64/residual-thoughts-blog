import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Assumes files are in the current directory (where you ran the script)
DATADIR = Path(".")
OUTDIR = Path("meta_analysis_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Visual Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
sns.set_palette("muted")

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
def load_data():
    """
    Loads specific files uploaded by the user into a structured dictionary.
    Handles variable naming conventions across phases.
    """
    data = {}
    
    # We load 4 families
    families = ["general_qa", "gsm8k_math", "stories", "code"]
    
    for fam in families:
        data[fam] = {}
        
        # --- Phase 1: Token Cloud ---
        p1_path = DATADIR / f"phase1_metrics_{fam}.npz"
        if p1_path.exists():
            data[fam]["p1"] = dict(np.load(p1_path))
        else:
            print(f"[Warn] Missing Phase 1 for {fam}")
            data[fam]["p1"] = None

        # --- Phase 2: Transport ---
        # Note: Includes model prefix
        p2_path = DATADIR / f"google_gemma-3-1b-it_phase2_transport_{fam}.npz"
        if p2_path.exists():
            data[fam]["p2"] = dict(np.load(p2_path))
        else:
            print(f"[Warn] Missing Phase 2 for {fam}")
            data[fam]["p2"] = None

        # --- Phase 3: Layer Updates ---
        p3_path = DATADIR / f"phase3_metrics_{fam}.npz"
        if p3_path.exists():
            data[fam]["p3"] = dict(np.load(p3_path))
        else:
            print(f"[Warn] Missing Phase 3 for {fam}")
            data[fam]["p3"] = None

        # --- Phase 4: Trajectories (Role: Last Token, Group: All) ---
        p4_path = DATADIR / f"phase4_metrics_{fam}_last_token_all.npz"
        if p4_path.exists():
            data[fam]["p4"] = dict(np.load(p4_path))
        else:
            print(f"[Warn] Missing Phase 4 (All) for {fam}")
            data[fam]["p4"] = None

    # --- Phase 4 Special: GSM8K Correct/Incorrect Split ---
    # Role: last_number_token
    data["gsm8k_math"]["p4_correct"] = None
    data["gsm8k_math"]["p4_incorrect"] = None
    
    p4_corr_path = DATADIR / "phase4_metrics_gsm8k_math_last_number_token_correct.npz"
    if p4_corr_path.exists():
        data["gsm8k_math"]["p4_correct"] = dict(np.load(p4_corr_path))
        
    p4_inc_path = DATADIR / "phase4_metrics_gsm8k_math_last_number_token_incorrect.npz"
    if p4_inc_path.exists():
        data["gsm8k_math"]["p4_incorrect"] = dict(np.load(p4_inc_path))

    return data

# -----------------------------------------------------------------------------
# Conclusive Plotting Functions
# -----------------------------------------------------------------------------

def plot_the_corridor_effect(data: Dict, family: str):
    """
    [Theme A] The Corridor Effect
    Hypothesis: High Token Homogeneity (Phase 1) creates a Corridor of Low Instability (Phase 4).
    """
    p1 = data[family]["p1"]
    p4 = data[family]["p4"]
    if not p1 or not p4: return

    layers = p1["layer_indices"]
    
    # Metric 1: Homogeneity (Mean Cosine Similarity) from Phase 1
    # Higher = More "smoothed out" / similar
    homogeneity = p1["mean_cos_mean"]
    
    # Metric 2: Instability (Curvature Angle) from Phase 4
    # Higher = Wobbly/Erratic trajectory
    instability = p4["all_curvature_angle_mean"]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Homogeneity (Left Axis)
    color1 = 'tab:blue'
    ax1.set_xlabel('Layer Depth (Transformer Block)')
    ax1.set_ylabel('Token Homogeneity\n(Mean Cosine Sim)', color=color1, fontweight='bold')
    ax1.plot(layers, homogeneity, color=color1, linewidth=3, label="P1: Homogeneity")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(False)

    # Plot Instability (Right Axis)
    ax2 = ax1.twinx()  
    color2 = 'tab:orange'
    ax2.set_ylabel('Trajectory Instability\n(Curvature Angle)', color=color2, fontweight='bold')
    ax2.plot(layers, instability, color=color2, linewidth=3, linestyle='--', label="P4: Instability")
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.grid(True, alpha=0.3)
    
    # Add corridor region highlighting (approx middle layers)
    L = len(layers)
    ax1.axvspan(L*0.25, L*0.75, color='gray', alpha=0.1, label="The Corridor")

    plt.title(f"The Corridor Effect: {family.upper()}\nSmoother representation $\\rightarrow$ Straighter trajectory", fontsize=14)
    fig.tight_layout()
    plt.savefig(OUTDIR / f"conclusive_corridor_{family}.png", dpi=150)
    plt.close()


def plot_transport_mechanism(data: Dict, family: str):
    """
    [Theme B] The Mechanism of Transport
    Hypothesis: Attention sublayers drive Information Transport (Phase 2), not MLPs.
    """
    p2 = data[family]["p2"]
    p3 = data[family]["p3"]
    if not p2 or not p3: return

    layers = p2["layer_indices"]
    
    # Metric 1: Transport Distance (Phase 2)
    # How many positions did the info jump?
    transport_dist = p2["mean_abs_delta_token"]
    
    # Metric 2: Attention Ratio (Phase 3)
    # How much of the update vector norm comes from Attn vs MLP?
    na = p3["na_mean"]
    nm = p3["nm_mean"]
    attn_ratio = na / (na + nm + 1e-9)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Transport (Left Axis)
    color1 = 'tab:green'
    ax1.set_xlabel('Layer Depth')
    ax1.set_ylabel('Information Transport\n(|Delta Token|)', color=color1, fontweight='bold')
    ax1.plot(layers, transport_dist, color=color1, linewidth=3, label="P2: Transport Dist")
    ax1.fill_between(layers, 0, transport_dist, color=color1, alpha=0.1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(False)

    # Plot Attn Ratio (Right Axis)
    ax2 = ax1.twinx()
    color2 = 'tab:purple'
    ax2.set_ylabel('Attention Dominance\n(Attn Norm / Total Norm)', color=color2, fontweight='bold')
    ax2.plot(layers, attn_ratio, color=color2, linewidth=3, linestyle='-', label="P3: Attn Ratio")
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 1.0)
    
    # Correlation note
    corr = np.corrcoef(transport_dist, attn_ratio)[0,1]
    
    plt.title(f"Mechanism of Transport: {family.upper()}\nCorrelation: {corr:.2f} (Attn drives Transport)", fontsize=14)
    fig.tight_layout()
    plt.savefig(OUTDIR / f"conclusive_transport_mech_{family}.png", dpi=150)
    plt.close()


def plot_gsm8k_divergence(data: Dict):
    """
    [Theme D] Failure Modes
    Plots the difference (Incorrect - Correct) for Curvature to spot where logic breaks.
    """
    fam = "gsm8k_math"
    p_corr = data[fam].get("p4_correct")
    p_inc = data[fam].get("p4_incorrect")
    
    if not p_corr or not p_inc: 
        print(f"[Skipping] GSM8K divergence - missing correct/incorrect split files.")
        return

    layers = p_corr["layer_indices"]
    
    # Calculate Divergence: Incorrect - Correct
    # Positive value = Incorrect is more erratic/wobbly
    curve_corr = p_corr["correct_curvature_angle_mean"]
    curve_inc = p_inc["incorrect_curvature_angle_mean"]
    diff_curve = curve_inc - curve_corr
    
    step_corr = p_corr["correct_step_norm_mean"]
    step_inc = p_inc["incorrect_step_norm_mean"]
    diff_step = step_inc - step_corr

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot 1: Curvature Divergence
    ax1.plot(layers, diff_curve, color='crimson', linewidth=2)
    ax1.axhline(0, color='black', linewidth=1, linestyle='--')
    ax1.fill_between(layers, 0, diff_curve, where=(diff_curve>0), color='crimson', alpha=0.2, label="Incorrect is Erratic")
    ax1.fill_between(layers, 0, diff_curve, where=(diff_curve<0), color='green', alpha=0.2, label="Correct is Erratic")
    ax1.set_ylabel("Curvature Diff\n(Incorrect - Correct)")
    ax1.set_title("Where does logic break? (Trajectory Instability)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Step Size Divergence
    ax2.plot(layers, diff_step, color='navy', linewidth=2)
    ax2.axhline(0, color='black', linewidth=1, linestyle='--')
    ax2.fill_between(layers, 0, diff_step, where=(diff_step>0), color='navy', alpha=0.2, label="Incorrect = Large Steps")
    ax2.fill_between(layers, 0, diff_step, where=(diff_step<0), color='skyblue', alpha=0.2, label="Correct = Large Steps")
    ax2.set_ylabel("Step Size Diff\n(Incorrect - Correct)")
    ax2.set_xlabel("Layer Depth")
    ax2.set_title("Confidence Divergence (Step Size)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"GSM8K Failure Modes: Last Number Token", fontsize=16)
    fig.tight_layout()
    plt.savefig(OUTDIR / "conclusive_gsm8k_failure_modes.png", dpi=150)
    plt.close()


def plot_task_fingerprint(data: Dict):
    """
    [Theme C] Task Fingerprints
    Parallel coordinates plot comparing families on 3 key metrics in the "Middle Layers".
    """
    stats = []
    families = ["general_qa", "gsm8k_math", "stories", "code"]
    
    for fam in families:
        d = data[fam]
        if not d["p1"] or not d["p3"] or not d["p4"]: continue
        
        # Define "Middle Layers" (25% to 75% depth) to avoid input/output noise
        L = len(d["p1"]["layer_indices"])
        start, end = int(L*0.25), int(L*0.75)
        
        # Metric 1: Homogeneity (P1)
        m1 = np.nanmean(d["p1"]["mean_cos_mean"][start:end])
        
        # Metric 2: Instability (P4)
        m2 = np.nanmean(d["p4"]["all_curvature_angle_mean"][start:end])
        
        # Metric 3: Attn Dominance (P3)
        na = d["p3"]["na_mean"][start:end]
        nm = d["p3"]["nm_mean"][start:end]
        m3 = np.nanmean(na / (na + nm + 1e-9))
        
        stats.append({
            "Family": fam, 
            "Homogeneity\n(P1 Cloud)": m1, 
            "Instability\n(P4 Trajectory)": m2, 
            "Attn Dominance\n(P3 Update)": m3
        })
    
    if not stats: return

    df = pd.DataFrame(stats)
    
    # Normalize columns 0-1 for fair comparison
    plot_cols = ["Homogeneity\n(P1 Cloud)", "Instability\n(P4 Trajectory)", "Attn Dominance\n(P3 Update)"]
    df_norm = df.copy()
    for c in plot_cols:
        min_val = df[c].min()
        max_val = df[c].max()
        if max_val - min_val > 1e-9:
            df_norm[c] = (df[c] - min_val) / (max_val - min_val)
        else:
            df_norm[c] = 0.5 # Default if flat

    # Plot
    plt.figure(figsize=(12, 6))
    pd.plotting.parallel_coordinates(
        df_norm, 
        'Family', 
        color=sns.color_palette("muted", len(families)),
        linewidth=4,
        alpha=0.8
    )
    
    plt.title("Task Fingerprints (Relative Intensity in Mid-Layers)", fontsize=16)
    plt.ylabel("Relative Scale (Min-Max Normalized)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTDIR / "conclusive_task_fingerprints.png", dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("[Meta-Analysis] Loading data...")
    data = load_data()
    
    print("[Meta-Analysis] Generating plots...")
    families = ["general_qa", "gsm8k_math", "stories", "code"]
    
    # 1. Family Deep Dives
    for fam in families:
        print(f"  Processing {fam}...")
        plot_the_corridor_effect(data, fam)
        plot_transport_mechanism(data, fam)
            
    # 2. GSM8K Failure Modes
    print("  Processing GSM8K Failure Modes...")
    plot_gsm8k_divergence(data)
        
    # 3. Task Comparison
    print("  Processing Task Fingerprints...")
    plot_task_fingerprint(data)
    
    print(f"\n[Done] All plots saved to: {OUTDIR.absolute()}")

if __name__ == "__main__":
    main()