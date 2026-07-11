"""E3 analysis: does reader overlap (weights) predict co-activation (data)?

Per pair: lift = P(i&j) / (P(i) P(j)) with marginals measured in the same pass.
Pairs with expected joint count < 5 are excluded from lift statistics (noise).

H1: how does co-use scale with geometric overlap? (either direction is a finding)
H2 (pre-registered): reader overlap |Spearman rho| >= 0.3 with log-lift AND
geometry adds dR^2 < 0.01 on ranks once readers are conditioned -> PASS;
readers add independent signal but geometry survives -> PARTIAL;
readers uninformative (|rho| < 0.1) -> FAIL.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from common_e3 import STRATA, result_dir

INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_table(model):
    rd = result_dir(model)
    rows = []
    l = 0
    while (rd / f"pairs_L{l}.npz").exists():
        if not (rd / f"coact_L{l}.npz").exists():
            l += 1
            continue
        P = np.load(rd / f"pairs_L{l}.npz")
        C = np.load(rd / f"coact_L{l}.npz")
        T = C["tokens_per_class"].sum()
        U = C["U"]
        pos = {int(u): k for k, u in enumerate(U)}
        mi = np.array([pos[int(x)] for x in P["pi"]])
        mj = np.array([pos[int(x)] for x in P["pj"]])
        marg = C["marg"].sum(0)
        joint = C["joint"].sum(0)
        p_i, p_j = marg[mi] / T, marg[mj] / T
        expected = p_i * p_j * T
        lift = joint / np.maximum(expected, 1e-12)
        jac = joint / np.maximum(marg[mi] + marg[mj] - joint, 1)
        rows.append({
            "layer": l, "geom": P["geom"], "reader_cos": P["reader_cos"],
            "reader_jac": P["reader_jac"], "lift": lift, "tok_jac": jac,
            "expected": expected, "p_i": p_i, "p_j": p_j, "joint": joint,
        })
        l += 1
    return rows


def rank(x):
    return stats.rankdata(x) / len(x)


def r2(X, y):
    X = np.column_stack([np.ones(len(y))] + list(X))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return 1 - resid.var() / y.var()


def analyze(rows, min_expected=5.0):
    geom = np.concatenate([r["geom"] for r in rows])
    rc = np.concatenate([r["reader_cos"] for r in rows])
    rj = np.concatenate([r["reader_jac"] for r in rows])
    lift = np.concatenate([r["lift"] for r in rows])
    exp = np.concatenate([r["expected"] for r in rows])
    p_i = np.concatenate([r["p_i"] for r in rows])
    p_j = np.concatenate([r["p_j"] for r in rows])
    layer = np.concatenate([np.full(len(r["geom"]), r["layer"]) for r in rows])

    keep = exp >= min_expected
    geom, rc, rj, lift, p_i, p_j, layer = (a[keep] for a in (geom, rc, rj, lift, p_i, p_j, layer))
    loglift = np.log(np.maximum(lift, 1e-3))

    out = {"n_pairs": int(keep.sum()), "n_dropped_low_power": int((~keep).sum())}
    out["rho_geom_lift"] = tuple(map(float, stats.spearmanr(geom, loglift)))
    out["rho_readercos_lift"] = tuple(map(float, stats.spearmanr(rc, loglift)))
    out["rho_readerjac_lift"] = tuple(map(float, stats.spearmanr(rj, loglift)))

    # rank regressions: base rates always conditioned
    y = rank(loglift)
    base = [rank(p_i), rank(p_j)]
    g, c, j = rank(geom), rank(rc), rank(rj)
    r2_base = r2(base, y)
    r2_geom = r2(base + [g], y)
    r2_read = r2(base + [c, j], y)
    r2_full = r2(base + [g, c, j], y)
    out["r2"] = {"base": r2_base, "base+geom": r2_geom, "base+readers": r2_read,
                 "full": r2_full,
                 "geom_given_readers": r2_full - r2_read,
                 "readers_given_geom": r2_full - r2_geom}

    # partial Spearman of readers with lift, given geom + rates
    def resid(v):
        X = np.column_stack([np.ones(len(y)), g, rank(p_i), rank(p_j)])
        beta, *_ = np.linalg.lstsq(X, v, rcond=None)
        return v - X @ beta
    out["partial_rho_readercos"] = tuple(map(float, stats.spearmanr(resid(c), resid(y))))
    out["partial_rho_readerjac"] = tuple(map(float, stats.spearmanr(resid(j), resid(y))))

    # H1: median lift per geometric stratum
    strat = []
    for lo, hi in zip(STRATA[:-1], STRATA[1:]):
        m = (geom >= lo) & (geom < hi)
        if m.sum() >= 30:
            strat.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                          "median_lift": float(np.median(lift[m])),
                          "q90_lift": float(np.quantile(lift[m], 0.9))})
    out["lift_by_geom_stratum"] = strat

    rho_rc = out["rho_readercos_lift"][0]
    d_geom = out["r2"]["geom_given_readers"]
    if abs(rho_rc) >= 0.3 and d_geom < 0.01:
        verdict = "PASS"
    elif abs(rho_rc) >= 0.1:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    out["h2_verdict"] = verdict
    return out, dict(geom=geom, rc=rc, lift=lift, loglift=loglift, layer=layer)


def figures(model, D, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    ax = axes[0]
    sc = ax.scatter(D["geom"], D["lift"], c=D["rc"], cmap="viridis", s=8, alpha=0.5)
    fig.colorbar(sc, ax=ax, label="reader-profile cosine")
    ax.set_yscale("log")
    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("geometric overlap |cos(w_i, w_j)|")
    ax.set_ylabel("co-activation lift (log)")
    ax.set_title("Geometry vs realized co-use, colored by reader overlap")
    ax.set_axisbelow(True)

    ax = axes[1]
    bins = np.quantile(D["rc"], np.linspace(0, 1, 11))
    mids, meds = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (D["rc"] >= lo) & (D["rc"] < hi)
        if m.sum() >= 30:
            mids.append((lo + hi) / 2)
            meds.append(np.median(D["lift"][m]))
    ax.plot(mids, meds, "o-", color="#2a78d6", lw=2)
    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("reader-profile cosine (decile bins)")
    ax.set_ylabel("median lift")
    ax.set_title("Reader overlap vs median co-activation")
    ax.set_axisbelow(True)
    fig.suptitle(f"{model} — E3 sufficiency test", y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "sufficiency.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    rows = load_table(args.model)
    out, D = analyze(rows)
    rd = result_dir(args.model)
    figures(args.model, D, rd)
    with open(rd / "e3_metrics.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "lift_by_geom_stratum"}, indent=1))
    print("lift by geom stratum:")
    for s in out["lift_by_geom_stratum"]:
        print(f"  [{s['lo']:.2f},{s['hi']:.2f}) n={s['n']:5d} median lift {s['median_lift']:.2f} "
              f"q90 {s['q90_lift']:.2f}")


if __name__ == "__main__":
    main()
