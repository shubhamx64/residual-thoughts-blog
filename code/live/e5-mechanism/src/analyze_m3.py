"""E-M3 analysis (PREREG v1): verdict first, then tables.

Primary: mean over seeds of R(bucket) - R(random_seed-matched), bootstrap CI over
seeds (paired within seed). Quintile dose-response (seed 0) labeled exploratory.
"""
import argparse
import json

import numpy as np

from common_m import RESULTS

BUCKETS = ("weights", "fisher", "footprint", "join", "updnorm")
QUINTILE_SIGNALS = ("crowd_base", "fisher_A", "upd_norm_s0")


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tinyllama-1.1b")
    args = ap.parse_args()
    d = json.load(open(RESULTS / f"rollback_{args.model}.json"))
    rows = d["rows"]
    seeds = sorted({r["seed"] for r in rows})
    get = lambda s, b: next(r for r in rows if r["seed"] == s and r["bucket"] == b)

    # paired R - R(random) per seed
    excess = {b: [get(s, b)["R"] - get(s, "random")["R"] for s in seeds]
              for b in BUCKETS}
    rand_R = [get(s, "random")["R"] for s in seeds]

    order = sorted(BUCKETS, key=lambda b: -np.mean(excess[b]))
    stats = {b: (np.mean(excess[b]), *boot_ci(excess[b])) for b in BUCKETS}

    # verdict: which selectors separate from which (CI non-overlap on paired diffs)
    def sep(a, b):
        diff = np.array(excess[a]) - np.array(excess[b])
        lo, hi = boot_ci(diff)
        return lo > 0 or hi < 0

    verdict = " > ".join(order)
    print(f"VERDICT: rollback recovery order (R - R_random): {verdict}")
    for a, b in zip(order[:-1], order[1:]):
        print(f"  {a} vs {b}: {'separated' if sep(a, b) else 'CI overlaps 0'}")
    print(f"  random floor R = {np.mean(rand_R):.3f} "
          f"(+-{np.std(rand_R):.3f} over {len(seeds)} seeds)\n")

    print(f"{'bucket':>10} | mean R-Rrand |   95% CI    | mean R | mean C")
    for b in order:
        m, lo, hi = stats[b]
        mr = np.mean([get(s, b)["R"] for s in seeds])
        mc = np.mean([get(s, b)["C"] for s in seeds])
        print(f"{b:>10} | {m:+.3f}       | [{lo:+.3f},{hi:+.3f}] | {mr:+.3f} | {mc:+.3f}")

    print("\nEXPLORATORY quintile dose-response (seed 0, R by quintile q0=low..q4=top):")
    for sig in QUINTILE_SIGNALS:
        rs = []
        for q in range(5):
            r = next((r for r in rows if r["bucket"] == f"q{q}_{sig}"), None)
            rs.append(f"{r['R']:+.3f}" if r else "  -  ")
        print(f"  {sig:>12}: " + "  ".join(rs))


if __name__ == "__main__":
    main()
