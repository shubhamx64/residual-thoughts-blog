"""Scripted selector analysis (test c) -- replaces the inline version.

Primary: final NLL degradation on the PREREG outcome half, curvK2 vs
weights/fisher/baseline, paired per data seed. Discloses the exact paired
sign-test floor at the current n. Secondary: step-100 all-40 degradation
(construction-contaminated, labeled).
"""
import json

import numpy as np

from common_m import E4, RESULTS

ARMS = ("baseline", "weights", "fisher")


def rows(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def curv_seeds():
    return sorted(int(p.stem.split("_s")[-1])
                  for p in (E4 / "results").glob("log_B_curvK2_s*.jsonl"))


def main():
    comp = json.load(open(RESULTS / "outcome_half_evals.json"))
    seeds = curv_seeds()
    nll_A = rows(E4 / "results" / f"log_B_curvK2_s{seeds[0]}.jsonl")[0]["nll_split"]
    deg = {}
    for arm in ARMS:
        deg[arm] = {s: comp[f"{arm}_s{s}"]["nll_outcome"] - nll_A
                    for s in (0, 1, 2, 3, 4) if f"{arm}_s{s}" in comp}
        # seeds run after the audit log nll_split directly (no ckpt re-eval)
        import re
        for p in (E4 / "results").glob(f"log_B_{arm}_s*.jsonl"):
            m = re.fullmatch(rf"log_B_{arm}_s(\d+)", p.stem)
            if not m:
                continue
            s = int(m.group(1))
            if s in deg[arm]:
                continue
            last = rows(p)[-1]
            if "nll_split" in last:
                deg[arm][s] = last["nll_split"] - nll_A
    deg["curvK2"] = {s: rows(E4 / "results" / f"log_B_curvK2_s{s}.jsonl")[-1]
                     ["nll_split"] - nll_A for s in seeds}

    print(f"outcome-half after-A NLL {nll_A:.4f}; curvK2 seeds: {seeds}")
    print(f"{'arm':>9} | mean final outcome-half deg | n")
    for arm in ("baseline", "weights", "fisher", "curvK2"):
        v = list(deg[arm].values())
        print(f"{arm:>9} | {np.mean(v):+.4f} | {len(v)}")

    for other in ("weights", "fisher"):
        common = sorted(set(deg["curvK2"]) & set(deg[other]))
        d = [deg["curvK2"][s] - deg[other][s] for s in common]
        n = len(d)
        wins = sum(x < 0 for x in d)
        p_floor = 0.5 ** n
        print(f"paired curvK2 - {other}: mean {np.mean(d):+.4f}, "
              f"{wins}/{n} seeds negative; exact one-sided sign p = "
              f"{p_floor if wins == n else 'n/a':.4} (floor at n={n})")

    print("\nsecondary (construction-contaminated, all-40 step-100 deg):")
    for arm, pat in (("baseline", "log_B_baseline{}.jsonl"),
                     ("weights", "log_B_weights{}.jsonl"),
                     ("fisher", "log_B_fisher{}.jsonl"),
                     ("curvK2", "log_B_curvK2{}.jsonl")):
        ds = []
        for s in seeds:
            p = E4 / "results" / pat.format(f"_s{s}" if s else "")
            if not p.exists():
                continue
            rr = rows(p)
            r0 = next(r for r in rr if r["step"] == 0)
            r1 = next(r for r in rr if r["step"] == 100)
            ds.append(np.log(r1["ppl_math"]) - np.log(r0["ppl_math"]))
        print(f"  {arm:>9}: {np.mean(ds):+.4f} (n={len(ds)})")


if __name__ == "__main__":
    main()
