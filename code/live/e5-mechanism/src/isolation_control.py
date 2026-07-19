"""Audit follow-up (c): the off-diagonal isolation control.

Among probed neurons, split by within-layer OFF-DIAGONAL load (K_offdiag, L2 of
the exact row minus diagonal, raw directions) into high vs low sets matched
1:1 within layer on: crowd_base, directional GGN diagonal S_ii (raw), gradmag_A,
upd_norm_s0 (calipers 0.5 pooled SD). Splice each set from after-A into the
after-B baseline (seed 0); recovery measured on the PREREG OUTCOME half.

If R(high) > R(low) under this matching, off-diagonal coupling predicts damage
recovery beyond directional self-curvature, importance, and movement -- the
missing identification for the K2 selector story.
"""
import json

import numpy as np
import torch

from common_m import (CKPT_A, OUTCOME_IDX, RESULTS, ckpt_B, eval_nll, eval_texts,
                      load_mlp_ckpt, load_model, load_signals, n_layers_of, splice)

MODEL_KEY = "tinyllama-1.1b"
CALIPER = 0.5


def build_sets():
    rz = np.load(RESULTS / f"rows_{MODEL_KEY}.npz")
    sig = load_signals(MODEL_KEY)
    pl, pn, rows = rz["probe_layer"], rz["probe_neuron"], rz["rows"]
    n = len(pl)
    diag = np.array([rows[k, pn[k]] for k in range(n)])
    Koff = np.array([np.sqrt(max(0.0, (rows[k] ** 2).sum() - diag[k] ** 2))
                     for k in range(n)])
    covs = np.stack([
        np.array([sig["crowd_base"][pl[k], pn[k]] for k in range(n)]),
        np.log(np.abs(diag) + 1e-30),
        np.log(np.array([sig["gradmag_A"][pl[k], pn[k]] for k in range(n)]) + 1e-30),
        np.array([sig["upd_norm_s0"][pl[k], pn[k]] for k in range(n)]),
    ], 1)
    Z = (covs - covs.mean(0)) / (covs.std(0) + 1e-30)
    hi_set, lo_set = [], []
    for l in np.unique(pl):
        m = np.nonzero(pl == l)[0]
        med = np.median(Koff[m])
        hi = [k for k in m if Koff[k] > med]
        lo = [k for k in m if Koff[k] <= med]
        used = set()
        for h in hi:
            best, bd = None, None
            for o in lo:
                if o in used:
                    continue
                d = np.abs(Z[h] - Z[o])
                if (d <= CALIPER).all() and (bd is None or d.sum() < bd):
                    best, bd = o, d.sum()
            if best is not None:
                used.add(best)
                hi_set.append(h)
                lo_set.append(best)
    hi_set, lo_set = np.array(hi_set), np.array(lo_set)
    print(f"matched {len(hi_set)} high/low pairs across "
          f"{len(np.unique(pl[hi_set]))} layers")
    for c, name in enumerate(("crowd", "log|diag|", "log gradmag", "updnorm")):
        sd = np.sqrt(0.5 * (Z[hi_set, c].var() + Z[lo_set, c].var()))
        print(f"  SMD {name}: {(Z[hi_set, c].mean() - Z[lo_set, c].mean()) / sd:+.3f}")
    kr = np.median(Koff[hi_set]) / np.median(Koff[lo_set])
    print(f"  K_offdiag median ratio high/low: {kr:.2f}")
    sel = {}
    for name, ks in (("high", hi_set), ("low", lo_set)):
        d = {}
        for k in ks:
            d.setdefault(int(pl[k]), []).append(int(pn[k]))
        sel[name] = {l: np.array(v) for l, v in d.items()}
    return sel, hi_set, lo_set


def main():
    sel, hi, lo = build_sets()
    sd_A = load_mlp_ckpt(CKPT_A[MODEL_KEY])
    sd_B = load_mlp_ckpt(ckpt_B(MODEL_KEY, "baseline", 0))
    model, tok = load_model(MODEL_KEY, init_ckpt=ckpt_B(MODEL_KEY, "baseline", 0))
    texts = eval_texts("math", OUTCOME_IDX)
    L_B = eval_nll(model, tok, texts)[0]
    n_layers = n_layers_of(sd_A)
    splice(model, sd_A, {l: np.ones(sd_A[f"model.layers.0.mlp.gate_proj.weight"
                                         ].shape[0], bool) for l in range(n_layers)})
    L_A = eval_nll(model, tok, texts)[0]
    splice(model, sd_B, {l: np.ones(sd_A[f"model.layers.0.mlp.gate_proj.weight"
                                         ].shape[0], bool) for l in range(n_layers)})
    print(f"outcome half: L_B {L_B:.4f}, L_A {L_A:.4f}, damage {L_B - L_A:.4f}")
    out = {"L_A": L_A, "L_B": L_B}
    rng = np.random.default_rng(20260713)
    rand_sel = {l: rng.choice(np.arange(5632), len(v), replace=False)
                for l, v in sel["high"].items()}
    for name, s in (("high", sel["high"]), ("low", sel["low"]), ("rand", rand_sel)):
        splice(model, sd_A, s)
        L = eval_nll(model, tok, texts)[0]
        splice(model, sd_B, s)
        R = (L_B - L) / (L_B - L_A)
        out[name] = {"L": L, "R": R,
                     "n": int(sum(len(v) for v in s.values()))}
        print(f"  {name:>4}-K_offdiag set: R = {R:+.4f} (n={out[name]['n']})")
    d = out["high"]["R"] - out["low"]["R"]
    print(f"\nVERDICT: R(high) - R(low) = {d:+.4f} at matched crowding, diagonal, "
          f"first-order, movement -> off-diagonal load "
          f"{'PREDICTS' if d > 0 else 'does NOT predict'} recovery beyond them")
    (RESULTS / "isolation_control.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
