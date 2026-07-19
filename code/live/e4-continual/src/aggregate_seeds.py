"""Aggregate multi-seed E4 into mean + 95% bootstrap CIs (PREREG_ROBUSTNESS Part A).
TinyLlama: seed 0 = log_B_<arm>.jsonl, seeds 1-4 = log_B_<arm>_s<n>.jsonl.
Reports per-arm retention degradation and % of Fisher recovered, with CIs, and the
key comparisons the review asked to see stabilised across seeds."""
import json
from pathlib import Path
import numpy as np
from scipy import stats

RES = Path(__file__).resolve().parent.parent / "results"
ARMS = ["baseline", "random", "weights", "footprint", "join", "fisher"]


def deg(tag):
    p = RES / f"log_B_{tag}.jsonl"
    if not p.exists():
        return None
    L = [json.loads(x) for x in open(p)]
    if len(L) < 6:
        return None
    return 100 * (L[-1]["ppl_math"] / L[0]["ppl_math"] - 1)


def boot_ci(vals, stat=np.mean, n=10000):
    vals = np.array(vals, float)
    if len(vals) < 2:
        return (float(vals[0]), float(vals[0]), float(vals[0]))
    res = stats.bootstrap((vals,), stat, n_resamples=n, method="percentile")
    return float(stat(vals)), float(res.confidence_interval.low), float(res.confidence_interval.high)


def main(fam="tinyllama", suffixes=("", "_s1", "_s2", "_s3", "_s4")):
    # per-seed degradation table
    degs = {a: [] for a in ARMS}
    pctF = {a: [] for a in ARMS}
    seeds_used = []
    for i, suf in enumerate(suffixes):
        row = {a: deg(f"{a}{suf}") for a in ARMS}
        if any(v is None for v in row.values()):
            continue
        seeds_used.append(i)
        b, f = row["baseline"], row["fisher"]
        for a in ARMS:
            degs[a].append(row[a])
            pctF[a].append(100 * (b - row[a]) / (b - f) if b != f else np.nan)

    print(f"== {fam}: {len(seeds_used)} seeds {seeds_used} ==\n")
    print(f"{'arm':<10}{'ret_deg% mean':>16}{'95% CI':>22}{'%Fisher mean':>14}")
    for a in ARMS:
        m, lo, hi = boot_ci(degs[a])
        pm, plo, phi = boot_ci([x for x in pctF[a] if np.isfinite(x)]) if pctF[a] else (np.nan,)*3
        print(f"{a:<10}{m:>+15.0f} {f'[{lo:+.0f}, {hi:+.0f}]':>22}{pm:>13.0f}%")

    # key comparisons (paired across seeds): does the ordering hold with CIs?
    print("\nkey comparisons (paired diff across seeds, mean [95% CI], negative = first better):")
    def paired(a1, a2):
        d = np.array(degs[a1]) - np.array(degs[a2])
        m, lo, hi = boot_ci(list(d))
        sig = "SEPARATED" if (lo > 0 or hi < 0) else "overlaps 0"
        print(f"  {a1:>9} - {a2:<9} {m:>+7.1f}  [{lo:+.1f}, {hi:+.1f}]  {sig}")
    paired("weights", "random")     # geometry beats random?
    paired("weights", "baseline")
    paired("fisher", "weights")     # Fisher beats geometry?
    paired("footprint", "weights")  # the flip: usage vs geometry (TinyLlama: usage better -> negative)
    paired("random", "baseline")    # random ~ baseline?

    out = {"family": fam, "n_seeds": len(seeds_used),
           "deg_mean_ci": {a: boot_ci(degs[a]) for a in ARMS},
           "deg_per_seed": {a: degs[a] for a in ARMS}}
    (RES / f"seed_ci_{fam}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
