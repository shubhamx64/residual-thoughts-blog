"""Weights-only stage of E3: select neuron pairs stratified by geometric overlap,
then compute reader profiles and reader-overlap for exactly those pairs.

Selection uses geometry + activity only (never reader structure) to avoid
circularity in the sufficiency test.

Outputs per model: results/<model>/pairs_L{l}.npz with
  pi, pj            candidate neuron indices (into the full MLP width)
  geom              |cos| of gamma-folded down-proj write columns
  rate_i, rate_j    pooled E1 firing rates
  reader_cos        cosine of unit-level reader profiles (downstream heads QKV + MLP gate)
  reader_jac        Jaccard of top-64 downstream gate-reader neuron sets
"""
import argparse
import time

import numpy as np
import torch

from common_e3 import (RATE_FLOOR, STRATA, PAIRS_PER_STRATUM, TOP_COH_PAIRS,
                       TOPK_READERS, pooled_rates, result_dir)
from extract import MODELS, load_weights, extract_layers  # e2 module

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def select_pairs(Wd_n_act, rng):
    """Stratified sample over |cos| + top-coherence pairs. Wd_n_act: (d, n_act)."""
    n = Wd_n_act.shape[1]
    G = (Wd_n_act.T @ Wd_n_act).abs()
    G.fill_diagonal_(0)
    iu = torch.triu_indices(n, n, offset=1, device=G.device)
    vals = G[iu[0], iu[1]]
    chosen = []
    for lo, hi in zip(STRATA[:-1], STRATA[1:]):
        idx = torch.nonzero((vals >= lo) & (vals < hi), as_tuple=True)[0]
        if len(idx) == 0:
            continue
        perm = torch.randperm(len(idx), generator=rng).to(idx.device)
        take = idx[perm[:PAIRS_PER_STRATUM]]
        chosen.append(take)
    top = torch.topk(vals, min(TOP_COH_PAIRS, len(vals))).indices
    chosen.append(top)
    sel = torch.unique(torch.cat(chosen))
    return iu[0][sel].cpu(), iu[1][sel].cpu(), vals[sel].cpu()


@torch.no_grad()
def reader_profiles(layers, l, U, d):
    """For candidate neurons U at layer l: unit-level read-strength profile over
    all downstream heads (Q/K/V norms) and MLPs (gate norm), plus global top-k
    (layer', neuron') gate readers for the fine Jaccard."""
    W = layers[l]["Wdown"].to(DEV)[:, U]                     # (d, |U|)
    W = W / (W.norm(dim=0, keepdim=True) + 1e-12)
    n_u = W.shape[1]
    unit_cols, top_vals, top_ids = [], [], []
    for lp in range(l + 1, len(layers)):
        Lp = layers[lp]
        n_heads, n_kv, hd = Lp["n_heads"], Lp["n_kv"], Lp["head_dim"]
        for name, nh in (("Wq", n_heads), ("Wk", n_kv), ("Wv", n_kv)):
            R = (Lp[name].to(DEV) @ W).view(nh, -1, n_u)     # (heads, hd, |U|)
            unit_cols.append(R.norm(dim=1))                  # (heads, |U|)
        Rg = (Lp["Wgate"].to(DEV) @ W).abs()                 # (inter, |U|)
        unit_cols.append(Rg.norm(dim=0, keepdim=True))       # (1, |U|)
        v, i = torch.topk(Rg, min(TOPK_READERS, Rg.shape[0]), dim=0)
        top_vals.append(v)
        top_ids.append(i.long() + lp * 100_000)              # globally unique reader ids
    P = torch.cat(unit_cols, dim=0).T                        # (|U|, n_units)
    # z-score per reader unit across neurons: projection norms onto large
    # subspaces concentrate, so raw profiles are near-identical by construction;
    # differential reading is the meaningful signal
    P = (P - P.mean(0, keepdim=True)) / (P.std(0, keepdim=True) + 1e-12)
    P = P / (P.norm(dim=1, keepdim=True) + 1e-12)
    tv = torch.cat(top_vals, dim=0)
    ti = torch.cat(top_ids, dim=0)
    order = torch.topk(tv, min(TOPK_READERS, tv.shape[0]), dim=0).indices
    fine = torch.gather(ti, 0, order).T.cpu().numpy()        # (|U|, topk) reader ids
    return P, fine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    t0 = time.time()
    model = load_weights(args.model)
    layers, d = extract_layers(model, args.model)
    inter = layers[0]["Wdown"].shape[1]
    rates, tot = pooled_rates(args.model, len(layers), inter)
    print(f"{args.model}: {len(layers)} layers, pooled over {tot} tokens", flush=True)
    out_dir = result_dir(args.model)
    rng = torch.Generator().manual_seed(0)

    last = len(layers) - 1
    for l in range(last):  # last layer has no downstream readers
        act = np.nonzero(rates[l] >= RATE_FLOOR)[0]
        Wd = layers[l]["Wdown"].to(DEV)[:, act]
        Wd = Wd / (Wd.norm(dim=0, keepdim=True) + 1e-12)
        ai, aj, geom = select_pairs(Wd, rng)
        pi, pj = act[ai.numpy()], act[aj.numpy()]
        U, inv = np.unique(np.concatenate([pi, pj]), return_inverse=True)
        ui, uj = inv[: len(pi)], inv[len(pi):]
        P, fine = reader_profiles(layers, l, torch.tensor(U, device=DEV), d)
        rc = (P[ui] * P[uj]).sum(1).cpu().numpy()
        sets = [set(row.tolist()) for row in fine]
        rj = np.array([len(sets[a] & sets[b]) / len(sets[a] | sets[b])
                       for a, b in zip(ui, uj)], dtype=np.float32)
        np.savez(out_dir / f"pairs_L{l}.npz",
                 pi=pi.astype(np.int32), pj=pj.astype(np.int32),
                 geom=geom.numpy().astype(np.float32),
                 rate_i=rates[l][pi].astype(np.float32),
                 rate_j=rates[l][pj].astype(np.float32),
                 reader_cos=rc.astype(np.float32), reader_jac=rj)
        print(f"  L{l:2d}: {len(act)} active, {len(pi)} pairs, "
              f"geom q50/q99 {np.median(geom):.3f}/{np.quantile(geom, .99):.3f}, "
              f"reader_cos med {np.median(rc):.3f}", flush=True)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
