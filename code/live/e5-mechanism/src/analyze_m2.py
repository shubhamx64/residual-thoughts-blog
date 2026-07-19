"""E-M2 interface-localization analysis (PREREG v1): verdict first.

Primary: paired per-seed step-100 NLL-degradation gaps deg(down20)-deg(gate20) and
deg(down20)-deg(up20), 5-seed bootstrap CI. deg = log(ppl_math@100) - log(ppl_math@0).
Secondary: read20 (2x params), read10 (equal params, different neurons, seeds 0-2),
final-step gaps, code acquisition, share of the all-scope arm's protection.
"""
import json

import numpy as np

from common_m import E4

R = E4 / "results"
SCOPES = ("down20", "gate20", "up20", "read20")
SEEDS = (0, 1, 2, 3, 4)


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def log_path(scope, seed):
    return R / f"log_B_weights_scope-{scope}_s{seed}.jsonl"


def ref_path(arm, seed):  # existing Paper 1 logs
    return R / (f"log_B_{arm}.jsonl" if seed == 0 else f"log_B_{arm}_s{seed}.jsonl")


def deg(path, step=100, key="ppl_math"):
    rr = rows(path)
    r0 = next(r for r in rr if r["step"] == 0)
    rs = next(r for r in rr if r["step"] == step)
    return np.log(rs[key]) - np.log(r0[key])


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    d = {s: {sd: deg(log_path(s, sd)) for sd in SEEDS} for s in SCOPES}
    d["all"] = {sd: deg(ref_path("weights", sd)) for sd in SEEDS}
    d["baseline"] = {sd: deg(ref_path("baseline", sd)) for sd in SEEDS}
    d["read10"] = {sd: deg(log_path("read10", sd)) for sd in (0, 1, 2)}

    gaps = {}
    for other in ("gate20", "up20", "read20"):
        g = [d["down20"][sd] - d[other][sd] for sd in SEEDS]
        gaps[other] = (float(np.mean(g)), *boot_ci(g))

    print("VERDICT: paired step-100 NLL-degradation gaps (positive = down20 worse,")
    print("         i.e. the other scope protects better):")
    for other, (m, lo, hi) in gaps.items():
        primary = "PRIMARY " if other in ("gate20", "up20") else "second. "
        verdict = ("other scope wins" if lo > 0 else
                   "down20 wins" if hi < 0 else "CI straddles 0")
        print(f"  {primary}down20 - {other}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")

    print(f"\n{'arm':>9} | mean deg@100 (NLL) | share of all-scope protection")
    base = np.mean(list(d['baseline'].values()))
    full = np.mean(list(d['all'].values()))
    for s in ("baseline", "all", "down20", "gate20", "up20", "read20", "read10"):
        m = np.mean(list(d[s].values()))
        share = (base - m) / (base - full) if s not in ("baseline",) else 0.0
        n = len(d[s])
        print(f"{s:>9} | {m:+.4f} (n={n})      | {share:+.2f}")

    print("\nsecondary: final-step degradation and code acquisition (NLL@100, code):")
    for s in SCOPES + ("all",):
        paths = [(log_path(s, sd) if s != "all" else ref_path("weights", sd))
                 for sd in SEEDS]
        degf = np.mean([deg(p, 500) for p in paths])
        code = np.mean([np.log(next(r for r in rows(p) if r["step"] == 100)["ppl_code"])
                        for p in paths])
        print(f"  {s:>7}: deg@500 {degf:+.4f}   code NLL@100 {code:.4f}")


if __name__ == "__main__":
    main()
