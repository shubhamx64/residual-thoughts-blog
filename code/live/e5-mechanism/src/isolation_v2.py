"""Isolation control v2 (PREREG round-4 addendum, registered before running).

Stage 1: sampled-label directional-Fisher diagonal
  D_i = E_{y ~ p_theta}[(g_y . delta_i)^2] = delta_i^T G delta_i (exact GGN
  quadratic form in expectation), 128 sampled-label gradient passes over the
  probe half, per-neuron streamed projections.
  BALANCE GATE: rho(D_i, exact |S_ii|) >= 0.85 on the 1200 exact-row probes,
  else escalate to 256 samples.
Stage 2: tercile-cell sets on {crowd, D, |g.delta|, fisher, updnorm}; split by
  K2_offdiag' = sqrt(max(0, rowpower_sketch - D^2)); post-construction audit on
  exact S_ii and |g.delta| among probe members (median within-layer ratio in
  [0.8, 1.25] required for BOTH, else report still-confounded and STOP).
Stage 3: splice high/low/random on the outcome half.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

from common_m import (CKPT_A, DEV, OUTCOME_IDX, PROBE_IDX, RESULTS, SEQ_LEN,
                      ckpt_B, eval_nll, eval_texts, load_mlp_ckpt, load_model,
                      load_signals, mlp_key, n_layers_of, splice)
from curvature_run import load_all

MODEL_KEY = "tinyllama-1.1b"
N_SAMPLES = 128
GATE_RHO = 0.85


def sampled_diag(model, params, deltas, batches, n_samples, seed=20260713):
    n_total = sum(ids.shape[1] - 1 for ids in batches)
    n_layers = len(deltas)
    inter = deltas[0]["gate"].shape[0]
    acc = np.zeros((n_layers, inter))
    gen = torch.Generator(device=DEV).manual_seed(seed)
    for s in range(n_samples):
        model.zero_grad(set_to_none=True)
        for ids in batches:
            with torch.no_grad():
                z = model(ids, use_cache=False).logits[0]
                p = F.softmax(z[:-1].float(), -1)
                y = torch.multinomial(p, 1, generator=gen).squeeze(1)
            z = model(ids, use_cache=False).logits[0]
            lp = F.log_softmax(z[:-1].float(), -1)
            (-lp[torch.arange(len(y), device=DEV), y].sum() / n_total).backward()
        gd = np.zeros((n_layers, inter))
        for l, d in enumerate(deltas):
            gg = params[mlp_key(l, "gate")].grad
            gu = params[mlp_key(l, "up")].grad
            gw = params[mlp_key(l, "down")].grad
            gd[l] = ((gg.double() * d["gate"].to(DEV).double()).sum(1)
                     + (gu.double() * d["up"].to(DEV).double()).sum(1)
                     + (gw.double() * d["down"].to(DEV).double()).sum(0)
                     ).cpu().numpy()
        acc += gd ** 2
        if (s + 1) % 16 == 0:
            print(f"  sample {s + 1}/{n_samples}", flush=True)
    model.zero_grad(set_to_none=True)
    return acc / n_samples


def main():
    from scipy import stats
    model, params, deltas, batches = load_all(MODEL_KEY)
    sig = load_signals(MODEL_KEY)
    rz = np.load(RESULTS / f"rows_{MODEL_KEY}.npz")
    pl, pn, rows = rz["probe_layer"], rz["probe_neuron"], rz["rows"]
    n_layers = len(deltas)
    inter = deltas[0]["gate"].shape[0]

    d_path = RESULTS / f"sampled_diag_{MODEL_KEY}.npz"
    if d_path.exists():
        D = np.load(d_path)["D"]
        print("loaded existing sampled diagonal")
    else:
        D = sampled_diag(model, params, deltas, batches, N_SAMPLES)
        np.savez(d_path, D=D, n=N_SAMPLES)
    ex = np.array([abs(rows[k, pn[k]]) for k in range(len(pl))])
    est = np.array([D[pl[k], pn[k]] for k in range(len(pl))])
    rho = stats.spearmanr(est, ex).statistic
    print(f"BALANCE GATE: rho(D, exact |S_ii|) = {rho:+.3f} (>= {GATE_RHO})")
    if rho < GATE_RHO:
        D2 = sampled_diag(model, params, deltas, batches, N_SAMPLES, seed=414243)
        D = 0.5 * (D + D2)
        np.savez(d_path, D=D, n=2 * N_SAMPLES)
        est = np.array([D[pl[k], pn[k]] for k in range(len(pl))])
        rho = stats.spearmanr(est, ex).statistic
        print(f"  escalated to 256: rho = {rho:+.3f}")
        assert rho >= GATE_RHO, "gate failed at 256 samples; STOP per prereg"

    z = np.load(RESULTS / f"k2_raw_s0_{MODEL_KEY}.npz")
    rowpower = z["k2"] ** 2 + z["diag"] ** 2          # reconstruct row power
    Koff = np.sqrt(np.maximum(0, rowpower - D))
    gd = np.abs(sig["gdotdelta_s0"])
    covs = [sig["crowd_base"], np.log(D + 1e-30), np.log(gd + 1e-30),
            np.log(sig["fisher_A"] + 1e-30), sig["upd_norm_s0"]]
    hi, lo = {}, {}
    for l in range(n_layers):
        C = np.stack([c[l] for c in covs], 1)
        bins = np.stack([np.digitize(C[:, i], np.quantile(C[:, i], [1/3, 2/3]))
                         for i in range(C.shape[1])], 1)
        cell = (bins * (3 ** np.arange(C.shape[1]))).sum(1)
        hi_l, lo_l = [], []
        for c in np.unique(cell):
            m = np.nonzero(cell == c)[0]
            if len(m) < 8:
                continue
            order = m[np.argsort(Koff[l][m])]
            q = len(m) // 4
            lo_l.extend(order[:q])
            hi_l.extend(order[-q:])
        hi[l], lo[l] = np.array(hi_l), np.array(lo_l)
    n_hi = sum(len(v) for v in hi.values())
    print(f"sets: {n_hi} neurons each ({100 * n_hi / (n_layers * inter):.1f}% of MLP)")

    # post-construction audit on EXACT S_ii and |g.delta| among probe members
    probe_set = {(int(l), int(n)): k for k, (l, n) in enumerate(zip(pl, pn))}
    ratios = {}
    for name, arr in (("exact_Sii", None), ("gdot", gd)):
        rl = []
        for l in range(n_layers):
            def vals(ks):
                if arr is None:
                    return [abs(rows[probe_set[(l, int(i))], int(i)])
                            for i in ks if (l, int(i)) in probe_set]
                return [arr[l][int(i)] for i in ks]
            a, b = vals(hi[l]), vals(lo[l])
            if len(a) >= 3 and len(b) >= 3 and np.median(b) > 0:
                rl.append(np.median(a) / np.median(b))
        ratios[name] = float(np.median(rl))
        print(f"  audit {name}: median within-layer high/low ratio {ratios[name]:.3f}")
    ok = all(0.8 <= r <= 1.25 for r in ratios.values())
    a = np.concatenate([Koff[l][hi[l]] for l in range(n_layers)])
    b = np.concatenate([Koff[l][lo[l]] for l in range(n_layers)])
    print(f"  K2_offdiag' ratio high/low: {np.median(a) / np.median(b):.2f}")
    if not ok:
        print("STILL-CONFOUNDED per prereg gate; stopping before outcomes")
        json.dump({"gate": "failed", "ratios": ratios},
                  open(RESULTS / "isolation_v2.json", "w"), indent=1)
        return

    del model
    torch.cuda.empty_cache()
    sd_A = load_mlp_ckpt(CKPT_A[MODEL_KEY])
    sd_B = load_mlp_ckpt(ckpt_B(MODEL_KEY, "baseline", 0))
    model, tok = load_model(MODEL_KEY, init_ckpt=ckpt_B(MODEL_KEY, "baseline", 0))
    texts = eval_texts("math", OUTCOME_IDX)
    L_B = eval_nll(model, tok, texts)[0]
    full = {l: np.ones(inter, bool) for l in range(n_layers)}
    splice(model, sd_A, full)
    L_A = eval_nll(model, tok, texts)[0]
    splice(model, sd_B, full)
    out = {"L_A": L_A, "L_B": L_B, "n_per_set": n_hi, "gate_rho": rho,
           "audit_ratios": ratios}
    rng = np.random.default_rng(20260713)
    rand = {l: rng.choice(inter, len(hi[l]), replace=False) for l in hi}
    for name, s in (("high", hi), ("low", lo), ("rand", rand)):
        splice(model, sd_A, s)
        L = eval_nll(model, tok, texts)[0]
        splice(model, sd_B, s)
        out[name] = {"R": (L_B - L) / (L_B - L_A)}
        print(f"  {name:>4}: R = {out[name]['R']:+.4f}")
    print(f"\nVERDICT v2: R(high) - R(low) = "
          f"{out['high']['R'] - out['low']['R']:+.4f} at AUDITED-balanced "
          f"self-curvature, first-order, Fisher, crowding, movement")
    json.dump(out, open(RESULTS / "isolation_v2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
