"""Audit follow-up (e): interface decomposition of GGN coupling.

Receiver side: for the 100 registered val_diag probes, one ggn_vp each (full
neuron direction), projections split by the RECEIVER's interface (gate / up /
down slices separately) -> which interfaces receive the coupling mass.
Sender side: for 20 registered pairs, three extra products with single-interface
sender directions -> the 3x3 sender x receiver block structure.

Predictions on record (theory note + E-M2): gate blocks small; base-write terms
(via up/down and crosses) dominate.
"""
import json

import numpy as np
import torch

from common_m import RESULTS, mlp_key
from curvature_run import load_all, neuron_dir
from ggn import eps_for, ggn_vp

MODEL_KEY = "tinyllama-1.1b"
PROJ = ("gate", "up", "down")


def project_by_proj(g, deltas, l):
    """Receiver-side per-interface projections for layer l: 3 x (inter,)."""
    d = deltas[l]
    out = {}
    out["gate"] = (g[mlp_key(l, "gate")].double()
                   * d["gate"].to(g[mlp_key(l, "gate")].device).double()).sum(1)
    out["up"] = (g[mlp_key(l, "up")].double()
                 * d["up"].to(g[mlp_key(l, "up")].device).double()).sum(1)
    out["down"] = (g[mlp_key(l, "down")].double()
                   * d["down"].to(g[mlp_key(l, "down")].device).double()).sum(0)
    return {k: v.cpu().numpy() for k, v in out.items()}


def single_proj_dir(deltas, l, i, proj):
    d = deltas[l]
    v = {}
    for p in PROJ:
        t = torch.zeros_like(d[p])
        if p == proj:
            if p == "down":
                t[:, i] = d[p][:, i]
            else:
                t[i] = d[p][i]
        v[mlp_key(l, p)] = t
    return v


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--senders-only", action="store_true",
                    help="skip the receiver pass (already computed)")
    args = ap.parse_args()
    ps = np.load(RESULTS / f"pairset_{MODEL_KEY}.npz")
    model, params, deltas, batches = load_all(MODEL_KEY)
    pl, pn = ps["probe_layer"], ps["probe_neuron"]

    shares = json.load(open(RESULTS / "decomposition.json"))["receiver_shares"] \
        if args.senders_only else None
    # receiver side: 100 registered probes
    rec = {p: [] for p in PROJ}
    for c, idx in enumerate([] if args.senders_only
                            else ps["val_diag_probe_idx"]):
        l, i = int(pl[idx]), int(pn[idx])
        v = neuron_dir(deltas, l, i)
        g = ggn_vp(model, params, v, batches, eps_for(v, model))
        parts = project_by_proj(g, deltas, l)
        mask = np.ones(parts["gate"].shape[0], bool)
        mask[i] = False
        for p in PROJ:
            rec[p].append(np.abs(parts[p][mask]).sum())
        if (c + 1) % 20 == 0:
            print(f"  receiver {c + 1}/100", flush=True)
    if not args.senders_only:
        tot = sum(np.array(rec[p]) for p in PROJ)
        shares = {p: float(np.median(np.array(rec[p]) / tot)) for p in PROJ}
        print("RECEIVER-side off-diagonal |coupling| share (median, 100 probes):")
        for p in PROJ:
            print(f"  {p}: {shares[p]:.3f}")

    # sender x receiver blocks: the REGISTERED 50 val_sym pairs (round-4 rerun;
    # the first pass used 20 pairs of mixed provenance), per-pair matrices saved
    picks = list(ps["val_sym"])
    per_pair, meta = [], []
    for n_done, k in enumerate(picks, 1):
        l, i, j = int(ps["layer"][k]), int(ps["i"][k]), int(ps["j"][k])
        B = np.zeros((3, 3))
        for si, sp in enumerate(PROJ):
            v = single_proj_dir(deltas, l, i, sp)
            if not any(float(t.abs().sum()) for t in v.values()):
                continue
            g = ggn_vp(model, params, v, batches, eps_for(v, model))
            parts = project_by_proj(g, deltas, l)
            for ri, rp in enumerate(PROJ):
                B[si, ri] = abs(parts[rp][j])
        per_pair.append(B)
        meta.append({"k": int(k), "layer": l, "geom": float(ps["geom_base"][k])})
        print(f"  sender-pair {n_done}/{len(picks)}", flush=True)
    P = np.stack(per_pair)                       # (n, 3, 3)
    Pn = P / (P.sum(axis=(1, 2), keepdims=True) + 1e-30)
    med = np.median(Pn, 0)
    rr = med[:2, :2].sum()
    # leave-one-pair-out range of the mass-weighted read x read share
    mass = P.sum(axis=(1, 2))
    rr_mw = P[:, :2, :2].sum(axis=(1, 2))
    loo = [(rr_mw.sum() - rr_mw[q]) / (mass.sum() - mass[q])
           for q in range(len(P))]
    geoms = np.array([m["geom"] for m in meta])
    lays = np.array([m["layer"] for m in meta])
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        ls = rng.choice(np.unique(lays), len(np.unique(lays)), replace=True)
        sel = np.concatenate([np.nonzero(lays == l)[0] for l in ls])
        boot.append(np.median(Pn[sel, :2, :2].sum(axis=(1, 2))))
    print("PER-PAIR block shares, median over 50 registered pairs "
          "(rows sender gate/up/down):")
    print(np.array_str(med, precision=3))
    print(f"read x read share: median per-pair {np.median(Pn[:, :2, :2].sum(axis=(1,2))):.3f} "
          f"(layer-bootstrap 95% [{np.percentile(boot, 2.5):.3f}, "
          f"{np.percentile(boot, 97.5):.3f}]); mass-weighted "
          f"{P[:, :2, :2].sum() / mass.sum():.3f} "
          f"(LOO range [{min(loo):.3f}, {max(loo):.3f}])")
    for name, m in (("crowded (geom>0.4)", geoms > 0.4),
                    ("orthogonal (<0.05)", geoms < 0.05)):
        if m.sum() >= 3:
            print(f"  {name}: n={m.sum()}, median read x read "
                  f"{np.median(Pn[m, :2, :2].sum(axis=(1,2))):.3f}")
    np.savez(RESULTS / "decomposition_50.npz", blocks=P,
             layer=lays, geom=geoms, k=[m["k"] for m in meta])
    json.dump({"receiver_shares": shares, "median_blocks": med.tolist(),
               "readxread_median": float(np.median(Pn[:, :2, :2].sum(axis=(1,2))))},
              open(RESULTS / "decomposition.json", "w"), indent=1)


if __name__ == "__main__":
    main()
