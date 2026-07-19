"""Per-seed random protection masks for the multi-seed sweep (PREREG_ROBUSTNESS.md A).
Only the random arm's mask is seed-dependent; all other masks are deterministic and
reused. Shapes are taken from the existing seed-0 mask_random.npz so budget/layout
match exactly."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUDGET = 0.20
DIRS = {"tinyllama-1.1b": ROOT / "data",
        "qwen2.5-1.5b": ROOT / "data" / "qwen2.5-1.5b"}
SEEDS = [1, 2, 3, 4]  # seed 0 = existing mask_random.npz


def main():
    for fam, d in DIRS.items():
        ref = np.load(d / "mask_random.npz")
        keys = ref.files
        inter = ref[keys[0]].shape[0]
        k = int(BUDGET * inter)
        for s in SEEDS:
            rng = np.random.default_rng(s)
            masks = {}
            for key in keys:
                m = np.zeros(inter, bool)
                m[rng.choice(inter, k, replace=False)] = True
                masks[key] = m
            np.savez(d / f"mask_random_s{s}.npz", **masks)
        print(f"{fam}: wrote random masks seeds {SEEDS} ({k}/{inter}/layer, {len(keys)} layers)")


if __name__ == "__main__":
    main()
