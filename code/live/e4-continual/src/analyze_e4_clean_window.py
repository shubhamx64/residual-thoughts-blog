"""Summarize the five-seed TinyLlama phase-B clean window at step 100."""
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ARMS = ("baseline", "random", "weights", "footprint", "join", "fisher")
SEEDS = range(5)
STEP = 100


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


records = []
lookup = {}
for seed in SEEDS:
    suffix = "" if seed == 0 else f"_s{seed}"
    for arm in ARMS:
        path = RESULTS / f"log_B_{arm}{suffix}.jsonl"
        data = rows(path)
        start = next(row for row in data if row["step"] == 0)
        end = next(row for row in data if row["step"] == STEP)
        record = {
            "seed": seed,
            "arm": arm,
            "step": STEP,
            "math_degradation_pct": 100 * (end["ppl_math"] / start["ppl_math"] - 1),
            "code_change_pct": 100 * (end["ppl_code"] / start["ppl_code"] - 1),
            "math_ppl_end": end["ppl_math"],
            "code_ppl_end": end["ppl_code"],
        }
        records.append(record)
        lookup[seed, arm] = record

for seed in SEEDS:
    base = lookup[seed, "baseline"]["math_degradation_pct"]
    fisher = lookup[seed, "fisher"]["math_degradation_pct"]
    for arm in ARMS:
        damage = lookup[seed, arm]["math_degradation_pct"]
        lookup[seed, arm]["fisher_recovery_pct"] = 100 * (
            (base - damage) / (base - fisher)
        )

rng = np.random.default_rng(20260712)
draws = rng.integers(0, 5, size=(10_000, 5))


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    boot = values[draws].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
    }


summary = []
for arm in ARMS:
    math = np.array([lookup[s, arm]["math_degradation_pct"] for s in SEEDS])
    code = np.array([lookup[s, arm]["code_change_pct"] for s in SEEDS])
    recovery = np.array([lookup[s, arm]["fisher_recovery_pct"] for s in SEEDS])
    base_math = np.array([lookup[s, "baseline"]["math_degradation_pct"] for s in SEEDS])
    base_code = np.array([lookup[s, "baseline"]["code_change_pct"] for s in SEEDS])
    random_math = np.array([lookup[s, "random"]["math_degradation_pct"] for s in SEEDS])
    random_code = np.array([lookup[s, "random"]["code_change_pct"] for s in SEEDS])
    summary.append({
        "arm": arm,
        "math_degradation_pct": mean_ci(math),
        "code_change_pct": mean_ci(code),
        "fisher_recovery_pct": mean_ci(recovery),
        "math_difference_from_baseline_pct_points": mean_ci(math - base_math),
        "code_difference_from_baseline_pct_points": mean_ci(code - base_code),
        "math_difference_from_random_pct_points": mean_ci(math - random_math),
        "code_difference_from_random_pct_points": mean_ci(code - random_code),
        "pareto_vs_baseline_by_mean": bool(math.mean() < base_math.mean() and code.mean() <= base_code.mean()),
        "pareto_vs_random_by_mean": bool(math.mean() < random_math.mean() and code.mean() <= random_code.mean()),
    })

output = {
    "window": {"step": STEP, "selection": "first scheduled phase-B evaluation; documented clean pre-overfit window"},
    "bootstrap": {"draws": 10_000, "seed": 20260712, "unit": "phase-B data-order seed"},
    "records": records,
    "summary": summary,
}
path = RESULTS / "e4_clean_window.json"
path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"wrote {path}")
