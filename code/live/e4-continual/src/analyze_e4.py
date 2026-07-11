"""E4 analysis: retention/acquisition trajectories per arm + drift canary."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
ARM_COLOR = {"baseline": "#e34948", "random": "#898781",
             "weights": "#eda100", "join": "#2a78d6",
             "footprint": "#1baf7a", "fisher": "#4a3aa7", "join_code": "#2a78d6"}
ARMS = ["baseline", "random", "weights", "join", "footprint", "fisher"]
ARMS2 = ["baseline2", "join_code2"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_log(tag):
    path = RES / f"log_{tag}.jsonl"
    if not path.exists():
        return None
    with open(path) as f:
        return [json.loads(l) for l in f]


def main():
    logs = {arm: load_log(f"B_{arm}") for arm in ARMS}
    for arm in ARMS2:  # reverse direction: prefer corrected suffix-3 runs
        logs[arm] = load_log(f"B_{arm[:-1]}3") or load_log(f"B_{arm[:-1]}2")
    logs = {a: l for a, l in logs.items() if l}
    log_a = load_log("A")

    dir1 = {a: l for a, l in logs.items() if not a.endswith("2")}
    dir2 = {a: l for a, l in logs.items() if a.endswith("2")}

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for arm, log in dir1.items():
        steps = [r["step"] for r in log]
        axes[0].plot(steps, [r["ppl_math"] for r in log], "o-", color=ARM_COLOR[arm],
                     lw=2, ms=4, label=arm)
        axes[1].plot(steps, [r["ppl_code"] for r in log], "o-", color=ARM_COLOR[arm],
                     lw=2, ms=4, label=arm)
        axes[2].plot(steps, [r.get("fp_drift", np.nan) for r in log], "o-",
                     color=ARM_COLOR[arm], lw=2, ms=4, label=arm)
    axes[0].set_title("Task-A retention: math ppl during B training")
    axes[1].set_title("Task-B acquisition: code ppl")
    axes[2].set_title("Footprint drift (math probe)")
    for ax, yl in zip(axes, ("math ppl (held out)", "code ppl (held out)", "1 − cos to after-A")):
        ax.set_xlabel("phase-B step"); ax.set_ylabel(yl)
        ax.legend(frameon=False, fontsize=8); ax.set_axisbelow(True)
    fig.suptitle("E4 — protected continual learning, TinyLlama-1.1B (MLP-only)",
                 y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RES / "e4_trajectories.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    if dir2:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
        for arm, log in dir2.items():
            steps = [r["step"] for r in log]
            c = ARM_COLOR[arm[:-1]]
            axes[0].plot(steps, [r["ppl_code"] for r in log], "o-", color=c, lw=2, ms=4,
                         label=arm[:-1])
            axes[1].plot(steps, [r["ppl_math"] for r in log], "o-", color=c, lw=2, ms=4,
                         label=arm[:-1])
        axes[0].set_title("Reverse direction: code retention during math training")
        axes[1].set_title("Reverse direction: math acquisition")
        for ax in axes:
            ax.set_xlabel("phase-B2 step"); ax.set_ylabel("held-out ppl")
            ax.legend(frameon=False, fontsize=8); ax.set_axisbelow(True)
        fig.suptitle("E4 reverse direction (code -> math)", y=1.04,
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(RES / "e4_reverse.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    out = {"phase_A": {"start": log_a[0], "end": log_a[-1]} if log_a else None, "arms": {}}
    drift_all, dppl_all = [], []
    for arm, log in logs.items():
        ret_key = "ppl_code" if arm.endswith("2") else "ppl_math"  # dir2 retains code
        acq_key = "ppl_math" if arm.endswith("2") else "ppl_code"
        p0 = log[0]
        pe = log[-1]
        out["arms"][arm] = {
            "retained_ppl_after_A": p0[ret_key], "retained_ppl_final": pe[ret_key],
            "retention_degradation_pct": 100 * (pe[ret_key] / p0[ret_key] - 1),
            "retained_ppl_100": next((r[ret_key] for r in log if r["step"] == 100), None),
            "acquired_ppl_100": next((r[acq_key] for r in log if r["step"] == 100), None),
            "acquired_ppl_final": pe[acq_key],
            "prose_ppl_final": pe["ppl_prose"],
            "fp_drift_final": pe.get("fp_drift"),
            "fp_drift_100": next((r.get("fp_drift") for r in log if r["step"] == 100), None),
        }
        for r in log[1:]:
            if "fp_drift" in r:
                drift_all.append(r["fp_drift"])
                dppl_all.append(r[ret_key] / p0[ret_key] - 1)
    if len(drift_all) >= 8:
        rho, p = stats.spearmanr(drift_all, dppl_all)
        out["drift_canary"] = {"spearman_rho": float(rho), "p": float(p),
                               "n_checkpoints": len(drift_all)}
    with open(RES / "e4_metrics.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
