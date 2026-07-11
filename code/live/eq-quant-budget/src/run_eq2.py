"""E-Q design iteration 2: salient-protection arms (the AWQ-shaped test).

Everything at LOW bits except the top-k% neurons per layer by each map, kept at
8 bits. Budget overhead of protection is identical across maps at each k.
H-Q1' (pre-registered before running): at each k, protection quality orders
fisher >= footprint >= reader > random, measured on math/code held-out ppl
(prose reported but noisy for TinyLlama).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from run_eq import QuantHarness, MODEL_IDS, ROOT

LOW = 4
KS = (0.01, 0.05, 0.10)


def protect_bits(scores, n_layers, inter, frac, low):
    out = []
    for s in scores:
        k = max(1, int(frac * inter))
        bits = np.full(inter, low, dtype=np.int64)
        bits[np.argsort(-s)[:k]] = 8
        out.append(bits)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_IDS))
    ap.add_argument("--low", type=int, default=LOW)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    h = QuantHarness(args.model)
    od = ROOT / "results" / args.model
    z = np.load(od / "maps.npz")
    maps = {name: [z[f"{name}_L{l}"] for l in range(h.n_layers)]
            for name in ("reader", "footprint", "fisher")}
    maps["random"] = [rng.permutation(h.inter).astype(float) for _ in range(h.n_layers)]

    results = {"low": args.low}
    results[f"uniform{args.low}"] = h.run_arm(
        f"uniform{args.low}", [np.full(h.inter, args.low)] * h.n_layers)
    for frac in KS:
        for name in ("random", "reader", "footprint", "fisher"):
            tag = f"{name}_p{int(frac*100)}"
            bits = protect_bits(maps[name], h.n_layers, h.inter, frac, args.low)
            results[tag] = h.run_arm(tag, bits)

    with open(od / f"eq2_results_low{args.low}.json", "w") as f:
        json.dump(results, f, indent=1)
    print("done")


if __name__ == "__main__":
    main()
