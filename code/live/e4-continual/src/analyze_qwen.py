"""E4 Qwen replication analysis (PREREG_QWEN.md).

Primary (math->code): 6-arm retention ordering, %-of-Fisher recovered, cost
(code acquisition), pooled drift-canary Spearman.
Reverse (code->math): baseline vs join_code, directional-consistency only
(confounded by code-corpus overfitting -- pre-registered as such).
Emits e4_qwen_metrics.json and prints a summary.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

RES = Path(__file__).resolve().parent.parent / "results"
PRIMARY = ["baseline", "random", "weights", "footprint", "join", "fisher"]


def load(tag):
    p = RES / f"log_{tag}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def deg(L, key):
    return 100 * (L[-1][key] / L[0][key] - 1)


def main():
    out = {"primary": {}, "reverse": {}}

    # ---- primary: retention = math ppl blow-up during code training ----
    arms = {a: load(f"B_{a}_qwen") for a in PRIMARY}
    arms = {a: L for a, L in arms.items() if L}
    prim = {}
    for a, L in arms.items():
        prim[a] = {
            "retention_deg_pct": deg(L, "ppl_math"),
            "retention_deg_pct_100": 100 * (next(r["ppl_math"] for r in L if r["step"] == 100) / L[0]["ppl_math"] - 1),
            "code_ppl_final": L[-1]["ppl_code"],
            "drift_final": L[-1].get("fp_drift"),
        }
    base = prim["baseline"]["retention_deg_pct"]
    fish = prim["fisher"]["retention_deg_pct"]
    for a in prim:
        prim[a]["pct_of_fisher_recovered"] = 100 * (base - prim[a]["retention_deg_pct"]) / (base - fish)
    order = sorted(prim, key=lambda a: prim[a]["retention_deg_pct"])
    out["primary"] = {"arms": prim, "retention_order_best_to_worst": order}

    # drift canary: pool (drift, retention_deg) over all arms x checkpoints
    dd, rr = [], []
    for a, L in arms.items():
        p0 = L[0]["ppl_math"]
        for r in L[1:]:
            if r.get("fp_drift") is not None:
                dd.append(r["fp_drift"]); rr.append(r["ppl_math"] / p0 - 1)
    if len(dd) >= 8:
        rho, p = stats.spearmanr(dd, rr)
        out["primary"]["drift_canary"] = {"spearman_rho": float(rho), "p": float(p), "n": len(dd)}

    # ---- reverse: code retention during math training (confounded) ----
    rev = {}
    for a in ("baseline", "join_code"):
        L = load(f"B_{a}_qwenR")
        if L:
            rev[a] = {"code_retention_deg_pct": deg(L, "ppl_code"),
                      "code_ppl_after_A": L[0]["ppl_code"], "code_ppl_final": L[-1]["ppl_code"],
                      "math_acq_ppl_final": L[-1]["ppl_math"]}
    out["reverse"] = rev

    (RES / "e4_qwen_metrics.json").write_text(json.dumps(out, indent=1))

    # ---- print ----
    print("PRIMARY math->code (retention deg %, lower=better):")
    print(f"  {'arm':<10}{'ret_deg%':>9}{'@100':>7}{'%Fisher':>9}{'code_ppl':>9}{'drift':>7}")
    for a in order:
        d = prim[a]
        print(f"  {a:<10}{d['retention_deg_pct']:>+9.0f}{d['retention_deg_pct_100']:>+7.0f}"
              f"{d['pct_of_fisher_recovered']:>8.0f}%{d['code_ppl_final']:>9.2f}{d['drift_final']:>7.3f}")
    print(f"  order: {' < '.join(order)}")
    if "drift_canary" in out["primary"]:
        c = out["primary"]["drift_canary"]
        print(f"  drift canary: pooled Spearman rho = {c['spearman_rho']:+.3f} (n={c['n']}, p={c['p']:.1e})")
    if rev:
        print("\nREVERSE code->math (directional-consistency only, code corpus overfits):")
        for a, d in rev.items():
            print(f"  {a:<10} code {d['code_ppl_after_A']:.2f}->{d['code_ppl_final']:.2f} "
                  f"(deg {d['code_retention_deg_pct']:+.0f}%)  math_acq {d['math_acq_ppl_final']:.2f}")


if __name__ == "__main__":
    main()
