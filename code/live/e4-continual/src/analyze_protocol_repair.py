"""Summarize validation-gated Qwen protocol-repair arm runs.

Reads only completed arm logs. Test metrics are reported, never used to choose
the checkpoint or step budget.
"""
import csv
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PATTERN = re.compile(r"log_B_(baseline|random|weights|footprint|join|fisher)_qwen_repair_s(\d+)\.jsonl$")


def read_rows(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


records = []
for path in sorted(RESULTS.glob("log_B_*_qwen_repair_s*.jsonl")):
    match = PATTERN.match(path.name)
    if not match:
        continue
    arm, seed = match.group(1), int(match.group(2))
    rows = read_rows(path)
    if len(rows) < 2:
        continue
    start, end = rows[0], rows[-1]
    records.append({
        "seed": seed,
        "arm": arm,
        "step": int(end["step"]),
        "math_ppl_start": start["ppl_math"],
        "math_ppl_end": end["ppl_math"],
        "math_degradation_pct": 100 * (end["ppl_math"] / start["ppl_math"] - 1),
        "code_ppl_start": start["ppl_code"],
        "code_ppl_end": end["ppl_code"],
        "code_change_pct": 100 * (end["ppl_code"] / start["ppl_code"] - 1),
        "code_acquired": end["ppl_code"] < start["ppl_code"],
        "val_code_ppl_start": start.get("ppl_split"),
        "val_code_ppl_end": end.get("ppl_split"),
        "fp_drift": end.get("fp_drift"),
        "log": path.name,
    })

by_seed = {}
for record in records:
    by_seed.setdefault(record["seed"], {})[record["arm"]] = record

for arms in by_seed.values():
    if "baseline" not in arms or "fisher" not in arms:
        continue
    baseline = arms["baseline"]["math_degradation_pct"]
    fisher = arms["fisher"]["math_degradation_pct"]
    denominator = baseline - fisher
    for record in arms.values():
        record["fisher_recovery_pct"] = (
            100 * (baseline - record["math_degradation_pct"]) / denominator
            if abs(denominator) > 1e-12 else None
        )

summary = []
complete_seeds = sorted(
    seed for seed, arms in by_seed.items()
    if set(arms) == {"baseline", "random", "weights", "footprint", "join", "fisher"}
)
if complete_seeds:
    rng = np.random.default_rng(20260712)
    draws = rng.integers(0, len(complete_seeds), size=(10_000, len(complete_seeds)))
    for arm in ("baseline", "random", "weights", "footprint", "join", "fisher"):
        math = np.array([by_seed[s][arm]["math_degradation_pct"] for s in complete_seeds])
        code = np.array([by_seed[s][arm]["code_change_pct"] for s in complete_seeds])
        baseline_math = np.array([
            by_seed[s]["baseline"]["math_degradation_pct"] for s in complete_seeds
        ])
        baseline_code = np.array([
            by_seed[s]["baseline"]["code_change_pct"] for s in complete_seeds
        ])
        random_math = np.array([
            by_seed[s]["random"]["math_degradation_pct"] for s in complete_seeds
        ])
        random_code = np.array([
            by_seed[s]["random"]["code_change_pct"] for s in complete_seeds
        ])

        def mean_ci(values):
            boot = values[draws].mean(axis=1)
            return {
                "mean": float(values.mean()),
                "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
            }

        summary.append({
            "arm": arm,
            "n_seeds": len(complete_seeds),
            "math_degradation_pct": mean_ci(math),
            "code_change_pct": mean_ci(code),
            "math_difference_from_baseline_pct_points": mean_ci(math - baseline_math),
            "code_difference_from_baseline_pct_points": mean_ci(code - baseline_code),
            "math_difference_from_random_pct_points": mean_ci(math - random_math),
            "code_difference_from_random_pct_points": mean_ci(code - random_code),
            "all_seeds_acquire_code": bool(np.all(code < 0)),
        })

out_json = RESULTS / "protocol_repair_arms.json"
out_json.write_text(json.dumps({
    "selection_note": "Step budget selected once by baseline validation-code minimum; test sets not used for selection.",
    "bootstrap": {"draws": 10_000, "seed": 20260712, "paired_by_phase_b_seed": True},
    "records": records,
    "summary": summary,
}, indent=2), encoding="utf-8")

out_csv = RESULTS / "protocol_repair_arms.csv"
fields = list(records[0]) if records else []
if records:
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

print(json.dumps({"records": records, "summary": summary}, indent=2))
print(f"wrote {out_json}")
if records:
    print(f"wrote {out_csv}")
