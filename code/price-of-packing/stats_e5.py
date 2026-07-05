"""E5 statistics per the pre-registered plan in results/REPORT.md.

Primary: per-layer Spearman rho(entropy, crowding) partial on log10 rate,
excluding low-event neurons. Conditioned: couple-type strata medians at matched
rate deciles. Robustness: q98/q99.5, 4-class entropy, outside-top-class proxy.
Outputs results/<model>/e5_stats.json and prints the routing-relevant numbers.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CROWD = ["density", "max_cos", "top10", "density02"]
STRATA = ["opponent", "duplicate", "uncoupled-crowded", "isolated"]


def partial_spearman(x, y, z):
    """rho(x, y | z) via rank residuals."""
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    A = np.column_stack([np.ones(len(rz)), rz])

    def resid(v):
        beta, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ beta
    return float(stats.spearmanr(resid(rx), resid(ry))[0])


def layer_partials(df, ent_col, rate_col, flags_out=False):
    res = {c: [] for c in CROWD}
    for l, g in df.groupby("layer"):
        g = g[~g["excluded"]]
        if flags_out:
            g = g[~g["flag_universal"] & ~g["flag_entropy_neuron"]]
        if len(g) < 200:
            continue
        lr = np.log10(np.maximum(g[rate_col], 1e-8))
        for c in CROWD:
            res[c].append(partial_spearman(g[ent_col], g[c], lr))
    return res


def strata_table(df, ent_col, rate_col):
    g = df[~df["excluded"]].copy()
    g["dec"] = pd.qcut(np.log10(np.maximum(g[rate_col], 1e-8)), 10,
                       labels=False, duplicates="drop")
    med = g.groupby(["stratum", "dec"])[ent_col].median().unstack()
    counts = g.groupby("stratum")[ent_col].count()
    matched = med.mean(axis=1)
    return {"rate_matched_median": {s: float(matched.get(s, np.nan)) for s in STRATA},
            "n": {s: int(counts.get(s, 0)) for s in STRATA},
            "per_decile": {s: [None if np.isnan(v) else round(float(v), 3)
                               for v in med.loc[s]] if s in med.index else None
                           for s in STRATA}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--census", default="census.parquet",
                    help="alternate census file (e.g. census_read.parquet)")
    ap.add_argument("--out", default="e5_stats.json")
    args = ap.parse_args()
    df = pd.read_parquet(ROOT / "results" / args.model / args.census)

    out = {"model": args.model, "excluded_frac": float(df["excluded"].mean()),
           "flag_universal_frac": float(df["flag_universal"].mean()),
           "couple_frac": float((df["couple"] != "uncoupled").mean())}

    # primary (q99, 5-class) + flags-out variant
    prim = layer_partials(df, "entropy_q99", "rate_q99")
    out["partial_rho_by_layer"] = prim
    out["partial_rho_median"] = {c: float(np.median(v)) for c, v in prim.items()}
    noflag = layer_partials(df, "entropy_q99", "rate_q99", flags_out=True)
    out["partial_rho_median_noflags"] = {c: float(np.median(v)) for c, v in noflag.items()}

    # robustness
    rob = {}
    for tag, ec, rc in (("q98", "entropy_q98", "rate_q98"),
                        ("q99.5", "entropy_q99.5", "rate_q99.5"),
                        ("4class", "entropy4_q99", "rate_q99"),
                        ("outside_top", "outside_top_q99", "rate_q99")):
        r = layer_partials(df, ec, rc)
        rob[tag] = {c: float(np.median(v)) for c, v in r.items()}
    out["robustness_median_rho"] = rob

    # strata (pre-registered on density-0.4; post-hoc on density-0.2 quartiles)
    out["strata"] = strata_table(df, "entropy_q99", "rate_q99")
    if "stratum02" in df.columns:
        df2 = df.copy()
        df2["stratum"] = df2["stratum02"]
        out["strata02_posthoc"] = strata_table(df2, "entropy_q99", "rate_q99")

    # E5b cross-side secondaries (only when the census carries write-side density)
    if "density_w" in df.columns:
        rw, dp = [], []
        for l, g in df.groupby("layer"):
            g = g[~g["excluded"]]
            if len(g) < 200:
                continue
            rw.append(float(stats.spearmanr(g["density"], g["density_w"])[0]))
            lr = np.log10(np.maximum(g["rate_q99"], 1e-8))
            rx = stats.rankdata(g["entropy_q99"])
            ry = stats.rankdata(g["density"])
            A = np.column_stack([np.ones(len(g)), stats.rankdata(lr),
                                 stats.rankdata(g["density_w"])])

            def resid(v):
                beta, *_ = np.linalg.lstsq(A, v, rcond=None)
                return v - A @ beta
            dp.append(float(stats.spearmanr(resid(rx), resid(ry))[0]))
        out["read_write_density_rho_median"] = float(np.median(rw))
        out["double_partial_rho_median"] = float(np.median(dp))
        cat = layer_partials(df.rename(columns={"density_cat": "_d", "density": "_x"})
                             .rename(columns={"_d": "density", "_x": "density_gate"}),
                             "entropy_q99", "rate_q99")
        out["cat_sensitivity_density_rho_median"] = float(np.median(cat["density"]))

    p = ROOT / "results" / args.model / args.out
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    m = out["partial_rho_median"]
    s = out["strata"]["rate_matched_median"]
    print(f"{args.model}: partial rho(entropy|log rate) density={m['density']:+.3f} "
          f"max_cos={m['max_cos']:+.3f} top10={m['top10']:+.3f}")
    print(f"  no-flags density rho: {out['partial_rho_median_noflags']['density']:+.3f}")
    print(f"  robustness (density): " +
          " ".join(f"{k}={v['density']:+.3f}" for k, v in rob.items()))
    print(f"  rate-matched median entropy: " +
          " ".join(f"{k}={s[k]:.3f}" for k in STRATA) +
          f"  n={out['strata']['n']}")
    if "double_partial_rho_median" in out:
        print(f"  cross-side: rho(read,write density)={out['read_write_density_rho_median']:+.3f} "
              f"double-partial={out['double_partial_rho_median']:+.3f} "
              f"cat-sens={out['cat_sensitivity_density_rho_median']:+.3f}")


if __name__ == "__main__":
    main()
