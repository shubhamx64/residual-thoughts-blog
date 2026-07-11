"""
Before/after comparison of two analysis JSONs (e.g., pre-fix vs post-fix).

Compares, per layer common to both runs:
  - Sel x U: mean over heads of top1_mass_mean * n_features
    (softmax selectivity relative to the uniform baseline 1/n)
  - max Sel x U over heads (spike detector)
  - mean diagonal softmax mass (identity sensitivity)
  - mean RoPE stability (semantic_controllability AUC)

Usage:
    python compare_analysis_runs.py <old.json> <new.json>
"""
import argparse
import json

import numpy as np


def load_run(path):
    with open(path) as f:
        d = json.load(f)
    n_features = d.get("config", {}).get("feature_subset_size", 4096)
    layers = {}
    for key, lr in d.get("layer_results", {}).items():
        layer = int(key)
        rr = lr.get("routing_results", [])
        top1 = [r["metrics"]["top1_mass_mean"] for r in rr]
        diag = [r["metrics"]["diagonal_softmax_mass"] for r in rr]
        rope = [s.get("semantic_controllability") for s in lr.get("rope_stability", [])
                if isinstance(s, dict) and s.get("semantic_controllability") is not None]
        layers[layer] = {
            "sel_u_mean": float(np.mean(top1)) * n_features if top1 else np.nan,
            "sel_u_max": float(np.max(top1)) * n_features if top1 else np.nan,
            "diag_mean": float(np.mean(diag)) if diag else np.nan,
            "rope_mean": float(np.mean(rope)) if rope else np.nan,
        }
    return layers, n_features, d.get("config", {}).get("gamma_fold_mode", "legacy (pre-flag)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_json")
    parser.add_argument("new_json")
    args = parser.parse_args()

    old, n_old, mode_old = load_run(args.old_json)
    new, n_new, mode_new = load_run(args.new_json)
    print(f"old: {args.old_json} (n_features={n_old}, gamma_fold_mode={mode_old})")
    print(f"new: {args.new_json} (n_features={n_new}, gamma_fold_mode={mode_new})")

    common = sorted(set(old) & set(new))
    if not common:
        print("No common layers between the two runs.")
        return

    print(f"\n{'layer':>6} | {'SelxU mean':>21} | {'SelxU max':>21} | {'diag mass':>17} | {'RoPE AUC':>17}")
    print(f"{'':>6} | {'old':>10} {'new':>10} | {'old':>10} {'new':>10} | {'old':>8} {'new':>8} | {'old':>8} {'new':>8}")
    print("-" * 110)
    for l in common:
        o, n = old[l], new[l]
        print(
            f"{l:>6} | {o['sel_u_mean']:>10.2f} {n['sel_u_mean']:>10.2f} "
            f"| {o['sel_u_max']:>10.2f} {n['sel_u_max']:>10.2f} "
            f"| {o['diag_mean']:>8.4f} {n['diag_mean']:>8.4f} "
            f"| {o['rope_mean']:>8.4f} {n['rope_mean']:>8.4f}"
        )


if __name__ == "__main__":
    main()
