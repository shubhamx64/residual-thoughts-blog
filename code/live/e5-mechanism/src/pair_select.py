"""E-M1 pair set construction (CPU). Plan rev 3, Task 5 Step 1.

Sources: the E3 pair list (pairs_L*.npz, stratified over |cos| at BASE) with
pair-specific co-firing (coact_L*.npz) -- the only pairs with co-firing data.

Emits results/pairset_<model>.npz with, per pair: layer, i, j, geom_base (|cos|),
cos_base (signed), geom_A, log-lift, covariates, class flags, matched-pair ids,
regression/probe membership, and the REGISTERED validation picks (rng 20260712).

Also prints: matching SMD table, probe-neuron count, geom x lift coverage,
and the E-M5 signed-class counts (parallel / antiparallel thresholds).
"""
import json

import numpy as np
import torch

from common_m import BASE, CKPT_A, MODELS, RESULTS, load_mlp_ckpt, mlp_key, load_signals

MODEL_KEY = "tinyllama-1.1b"
E3_RES = BASE / "e3-sufficiency" / "results" / MODEL_KEY
LAYER_BAND = list(range(4, 18)) + [2, 20]          # primary band + spot checks
CROWD_MIN, ORTH_MAX = 0.40, 0.05
CALIPER = 0.5                                       # in pooled SDs, per covariate
PROBE_BUDGET = 1200
LIFT_FLOOR = 1e-2
RNG_SEED = 20260712

COVS = ("log_rate", "log_gradmag", "wnorm", "upd_norm", "log_lift")


def down_cols(sd, l):
    W = sd[mlp_key(l, "down")].to(torch.float32)
    return W / (W.norm(dim=0, keepdim=True) + 1e-12)


def pair_cos(Wn, pi, pj):
    return (Wn[:, pi] * Wn[:, pj]).sum(0).numpy()


def main():
    sig = load_signals(MODEL_KEY)
    sd_A = load_mlp_ckpt(CKPT_A[MODEL_KEY])
    from transformers import AutoModelForCausalLM
    base_model = AutoModelForCausalLM.from_pretrained(MODELS[MODEL_KEY],
                                                      dtype=torch.float32)
    sd_base = {k: v.detach() for k, v in base_model.state_dict().items()
               if ".mlp.down_proj" in k}
    del base_model

    rows = []
    for l in LAYER_BAND:
        pz = np.load(E3_RES / f"pairs_L{l}.npz")
        cz = np.load(E3_RES / f"coact_L{l}.npz")
        pi, pj, geom = pz["pi"], pz["pj"], pz["geom"]
        U = cz["U"]
        uidx = {int(u): k for k, u in enumerate(U)}
        T = cz["tokens_per_class"].sum()
        p_joint = cz["joint"].sum(0) / T
        p_marg = cz["marg"].sum(0) / T
        Wb = down_cols(sd_base, l)
        Wa = down_cols(sd_A, l)
        cosb = pair_cos(Wb, pi, pj)
        cosa = pair_cos(Wa, pi, pj)
        for k in range(len(pi)):
            i, j = int(pi[k]), int(pj[k])
            lift = p_joint[k] / (p_marg[uidx[i]] * p_marg[uidx[j]] + 1e-12)
            mean = lambda arr: 0.5 * (arr[l][i] + arr[l][j])
            rows.append(dict(
                layer=l, i=i, j=j,
                geom_base=float(abs(cosb[k])), cos_base=float(cosb[k]),
                geom_A=float(abs(cosa[k])), lift=float(lift),
                log_rate=float(np.log(mean(sig["rate_math"]) + 1e-6)),
                log_gradmag=float(np.log(mean(sig["gradmag_A"]) + 1e-12)),
                wnorm=float(mean(sig["wnorm_base"])),
                upd_norm=float(mean(sig["upd_norm_s0"])),
                log_lift=float(np.log(lift + LIFT_FLOOR)),
            ))
        # E3's stored geom used fp32 weights too; sanity vs our recompute
        assert np.abs(np.abs(cosb) - geom).max() < 1e-3, f"L{l} geom mismatch"

    n = len(rows)
    A = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    crowded = A["geom_base"] > CROWD_MIN
    orth = A["geom_base"] < ORTH_MAX
    print(f"{n} E3 pairs in band; crowded {crowded.sum()}, orthogonal {orth.sum()}")

    # ---- caliper matching within layer (greedy 1:1, no replacement)
    Z = np.stack([(A[c] - A[c].mean()) / (A[c].std() + 1e-12) for c in COVS], 1)
    match_id = np.full(n, -1, int)
    m_count = 0
    for l in LAYER_BAND:
        ci = np.nonzero(crowded & (A["layer"] == l))[0]
        oi = list(np.nonzero(orth & (A["layer"] == l))[0])
        for c in ci:
            if not oi:
                break
            d = np.abs(Z[oi] - Z[c])
            ok = np.nonzero((d <= CALIPER).all(1))[0]
            if len(ok) == 0:
                continue
            best = ok[np.argmin(d[ok].sum(1))]
            o = oi.pop(best)
            match_id[c] = o
            match_id[o] = c
            m_count += 1
    mc = np.nonzero(crowded & (match_id >= 0))[0]
    mo = match_id[mc]
    print(f"matched pair-pairs: {m_count}")
    print(f"{'covariate':>12} | SMD before | SMD after")
    for k, c in enumerate(COVS):
        smd0 = (Z[crowded, k].mean() - Z[orth, k].mean())
        smd1 = (Z[mc, k].mean() - Z[mo, k].mean()) if m_count else float("nan")
        print(f"{c:>12} | {smd0:+.3f}     | {smd1:+.3f}")

    # ---- probe set: matched members first, then stratified geom x lift fill
    probes = set()
    in_reg = np.zeros(n, bool)
    order = list(mc) + list(mo)
    gq = np.quantile(A["geom_base"], [1 / 3, 2 / 3])
    lq = np.quantile(A["log_lift"], [1 / 3, 2 / 3])
    cell = (np.digitize(A["geom_base"], gq) * 3
            + np.digitize(A["log_lift"], lq))
    rng = np.random.default_rng(RNG_SEED)
    fill = [i for c in range(9) for i in
            rng.permutation(np.nonzero(cell == c)[0]).tolist()]
    # round-robin across the 9 cells
    by_cell = {c: [i for i in fill if cell[i] == c] for c in range(9)}
    rr = []
    while any(by_cell.values()):
        for c in range(9):
            if by_cell[c]:
                rr.append(by_cell[c].pop(0))
    for idx in order + rr:
        key_i = (A["layer"][idx], A["i"][idx])
        key_j = (A["layer"][idx], A["j"][idx])
        new = {key_i, key_j} - probes
        if len(probes) + len(new) > PROBE_BUDGET and idx not in order:
            continue
        probes |= {key_i, key_j}
        in_reg[idx] = True
    print(f"probe neurons: {len(probes)}; regression pairs: {in_reg.sum()} "
          f"(both members probed)")
    cov_tab = np.bincount(cell[in_reg], minlength=9).reshape(3, 3)
    print("geom x lift coverage (rows geom terciles, cols lift terciles):")
    print(cov_tab)

    # ---- registered validation picks (excluded from production regression)
    reg_idx = np.nonzero(in_reg)[0]
    val_sym = rng.choice(reg_idx, 50, replace=False)
    rest = np.setdiff1d(reg_idx, val_sym)
    val_mix = rng.choice(rest, 10, replace=False)
    probe_list = sorted(probes)
    val_diag = rng.choice(len(probe_list), 100, replace=False)
    is_val = np.zeros(n, bool)
    is_val[val_sym] = True
    is_val[val_mix] = True

    # ---- E-M5 signed-class counts (among band pairs)
    for thr in (0.40, 0.30):
        par = int(((A["cos_base"] > thr)).sum())
        anti = int(((A["cos_base"] < -thr)).sum())
        print(f"E-M5 classes at |cos|>{thr}: parallel {par}, antiparallel {anti}")
    pc = sig["partner_cos_base"]
    print(f"E-M5 model-wide argmax-partner cos < -0.4: {(pc < -0.4).sum()}, "
          f"< -0.3: {(pc < -0.3).sum()} (of {pc.size} neurons)")

    out = RESULTS / f"pairset_{MODEL_KEY}.npz"
    np.savez(out,
             **{k: A[k] for k in A},
             crowded=crowded, orth=orth, match_id=match_id, in_reg=in_reg,
             is_val=is_val, val_sym=val_sym, val_mix=val_mix,
             probe_layer=np.array([p[0] for p in probe_list]),
             probe_neuron=np.array([p[1] for p in probe_list]),
             val_diag_probe_idx=val_diag)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
