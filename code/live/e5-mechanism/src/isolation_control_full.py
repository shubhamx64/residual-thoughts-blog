"""Isolation control, FULL-MODEL version (PREREG audit addendum c-AMENDED).

Coarsened exact matching per layer: quintile-bin (crowd_base, sketch diagonal,
gradmag_A, upd_norm_s0) -> within each cell of >= 8 neurons, high = top quartile
by sketch OFF-DIAGONAL K2 (raw), low = bottom quartile (equal counts). Splice
high vs low vs size-matched random from after-A into after-B baseline (seed 0);
recovery on the PREREG OUTCOME half.
"""
import json

import numpy as np

from common_m import (CKPT_A, OUTCOME_IDX, RESULTS, ckpt_B, eval_nll, eval_texts,
                      load_mlp_ckpt, load_model, load_signals, mlp_key,
                      n_layers_of, splice)

MODEL_KEY = "tinyllama-1.1b"


def build_sets(n_layers, inter):
    z = np.load(RESULTS / f"k2_raw_s0_{MODEL_KEY}.npz")
    k2, diag = z["k2"], np.abs(z["diag"])
    sig = load_signals(MODEL_KEY)
    hi, lo = {}, {}
    for l in range(n_layers):
        covs = np.stack([sig["crowd_base"][l], np.log(diag[l] + 1e-30),
                         np.log(sig["gradmag_A"][l] + 1e-30),
                         sig["upd_norm_s0"][l]], 1)
        bins = np.stack([np.digitize(covs[:, c],
                                     np.quantile(covs[:, c], [0.2, 0.4, 0.6, 0.8]))
                         for c in range(4)], 1)
        cell = (bins * (5 ** np.arange(4))).sum(1)
        hi_l, lo_l = [], []
        for c in np.unique(cell):
            m = np.nonzero(cell == c)[0]
            if len(m) < 8:
                continue
            order = m[np.argsort(k2[l][m])]
            q = len(m) // 4
            lo_l.extend(order[:q])
            hi_l.extend(order[-q:])
        hi[l], lo[l] = np.array(hi_l), np.array(lo_l)
    return hi, lo, k2, diag, sig


def smd(a, b):
    return (a.mean() - b.mean()) / (np.sqrt(0.5 * (a.var() + b.var())) + 1e-30)


def main():
    sd_A = load_mlp_ckpt(CKPT_A[MODEL_KEY])
    n_layers = n_layers_of(sd_A)
    inter = sd_A[mlp_key(0, "gate")].shape[0]
    hi, lo, k2, diag, sig = build_sets(n_layers, inter)
    n_hi = sum(len(v) for v in hi.values())
    print(f"sets: {n_hi} neurons each ({100 * n_hi / (n_layers * inter):.1f}% of MLP)")
    for name, arr in (("crowd", sig["crowd_base"]), ("log|diag|", np.log(diag + 1e-30)),
                      ("log gradmag", np.log(sig["gradmag_A"] + 1e-30)),
                      ("updnorm", sig["upd_norm_s0"])):
        a = np.concatenate([arr[l][hi[l]] for l in range(n_layers)])
        b = np.concatenate([arr[l][lo[l]] for l in range(n_layers)])
        print(f"  SMD {name}: {smd(a, b):+.3f}")
    a = np.concatenate([k2[l][hi[l]] for l in range(n_layers)])
    b = np.concatenate([k2[l][lo[l]] for l in range(n_layers)])
    print(f"  K2_offdiag median ratio high/low: {np.median(a) / np.median(b):.2f}")

    sd_B = load_mlp_ckpt(ckpt_B(MODEL_KEY, "baseline", 0))
    model, tok = load_model(MODEL_KEY, init_ckpt=ckpt_B(MODEL_KEY, "baseline", 0))
    texts = eval_texts("math", OUTCOME_IDX)
    L_B = eval_nll(model, tok, texts)[0]
    full = {l: np.ones(inter, bool) for l in range(n_layers)}
    splice(model, sd_A, full)
    L_A = eval_nll(model, tok, texts)[0]
    splice(model, sd_B, full)
    print(f"outcome half: L_B {L_B:.4f}, L_A {L_A:.4f}, damage {L_B - L_A:.4f}")
    out = {"L_A": L_A, "L_B": L_B, "n_per_set": n_hi}
    rng = np.random.default_rng(20260713)
    rand = {l: rng.choice(inter, len(hi[l]), replace=False) for l in hi}
    for name, s in (("high", hi), ("low", lo), ("rand", rand)):
        splice(model, sd_A, s)
        L = eval_nll(model, tok, texts)[0]
        splice(model, sd_B, s)
        R = (L_B - L) / (L_B - L_A)
        out[name] = {"L": L, "R": R}
        print(f"  {name:>4}-K2_offdiag set: R = {R:+.4f}")
    d = out["high"]["R"] - out["low"]["R"]
    dr = out["high"]["R"] - out["rand"]["R"]
    print(f"\nVERDICT: R(high) - R(low) = {d:+.4f}; R(high) - R(rand) = {dr:+.4f} "
          f"(cell-matched on crowding, sketch diagonal, first-order, movement)")
    (RESULTS / "isolation_control_full.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
