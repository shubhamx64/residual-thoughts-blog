"""Scripted matching-balance report + tight-caliper sensitivity (audit round 3/4).

Conventional post-match SMDs (pooled SD of the two matched groups) for the 202
matched pair-pairs, plus the 0.35-SD tight-caliper subset contrast. Previously
run inline; scripted per round-4 audit so construction and output are versioned.
"""
import numpy as np

from common_m import RESULTS

MODEL_KEY = "tinyllama-1.1b"
COVS = ("log_rate", "log_gradmag", "wnorm", "upd_norm", "log_lift")
TIGHT = 0.35


def main():
    z = np.load(RESULTS / f"pairset_{MODEL_KEY}.npz")
    rz = np.load(RESULTS / f"rows_{MODEL_KEY}.npz")
    pl, pn, rows, dn = rz["probe_layer"], rz["probe_neuron"], rz["rows"], rz["delta_norms"]
    pidx = {(int(l), int(n)): k for k, (l, n) in enumerate(zip(pl, pn))}
    mc = np.nonzero(z["crowded"] & (z["match_id"] >= 0))[0]
    mo = z["match_id"][mc]

    print("conventional post-match SMDs (pooled SD of matched groups):")
    for c in COVS:
        a, b = z[c][mc], z[c][mo]
        sd = np.sqrt(0.5 * (a.var() + b.var()))
        print(f"  {c:>12}: {(a.mean() - b.mean()) / (sd + 1e-30):+.3f}")

    Zs = {c: (z[c] - z[c].mean()) / (z[c].std() + 1e-30) for c in COVS}
    keep = np.ones(len(mc), bool)
    for c in COVS:
        keep &= np.abs(Zs[c][mc] - Zs[c][mo]) <= TIGHT
    print(f"tight subset ({TIGHT} SD calipers): {keep.sum()}/{len(mc)} pair-pairs")

    def su(k):
        l, i, j = int(z["layer"][k]), int(z["i"][k]), int(z["j"][k])
        a, b = pidx[(l, i)], pidx[(l, j)]
        return abs(0.5 * (rows[a, j] + rows[b, i]) / (dn[l, i] * dn[l, j] + 1e-30))

    a = np.array([su(k) for k in mc[keep]])
    b = np.array([su(k) for k in mo[keep]])
    lays = z["layer"][mc[keep]]
    lr = [np.median(a[lays == l]) / np.median(b[lays == l])
          for l in np.unique(lays) if (lays == l).sum() >= 3]
    print(f"tight-subset contrast: ratio {np.median(a) / np.median(b):.2f}, "
          f"per-layer > 1: {sum(r > 1 for r in lr)}/{len(lr)}")
    for c in COVS:
        aa, bb = z[c][mc[keep]], z[c][mo[keep]]
        sd = np.sqrt(0.5 * (aa.var() + bb.var()))
        print(f"  tight {c:>12}: SMD {(aa.mean() - bb.mean()) / (sd + 1e-30):+.3f}")
    np.savez(RESULTS / f"match_tight_{MODEL_KEY}.npz", mc=mc[keep], mo=mo[keep])


if __name__ == "__main__":
    main()
