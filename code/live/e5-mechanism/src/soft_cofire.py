"""Audit follow-up (d): continuous co-activation control.

Captures CONTINUOUS activations a_i(x_t) (silu(gate x) * (up x), the down_proj
input) on the PREREG probe half, accumulates per-pair soft co-activation
  soft_ij = sum_t a_i a_j / sqrt(sum a_i^2 * sum a_j^2)   (cosine co-moment)
for every pairset pair, then reruns the factorization interaction regression
with soft co-activation replacing thresholded q99 lift.
"""
import numpy as np
import torch

from common_m import (CKPT_A, DEV, PROBE_IDX, RESULTS, SEQ_LEN, eval_texts,
                      load_model, load_signals)

MODEL_KEY = "tinyllama-1.1b"


def capture_moments(pairs_by_layer):
    model, tok = load_model(MODEL_KEY, init_ckpt=CKPT_A[MODEL_KEY])
    acts = {}
    hooks = []

    def hook(l):
        def fn(module, inp, out):
            acts[l] = inp[0][0].detach().float()      # (T, inter)
        return fn

    for l, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.down_proj.register_forward_hook(hook(l)))
    sums = {l: {"ij": np.zeros(len(p[0])), "ii": None, "jj": None,
                "i2": np.zeros(5632), }
            for l, p in pairs_by_layer.items()}
    e_ij = {l: torch.zeros(len(p[0]), dtype=torch.float64, device=DEV)
            for l, p in pairs_by_layer.items()}
    e_sq = {l: torch.zeros(5632, dtype=torch.float64, device=DEV)
            for l in pairs_by_layer}
    with torch.no_grad():
        for t in eval_texts("math", PROBE_IDX):
            ids = tok(t, return_tensors="pt", truncation=True,
                      max_length=SEQ_LEN)["input_ids"].to(DEV)
            model(ids, use_cache=False)
            for l, (pi, pj) in pairs_by_layer.items():
                a = acts[l]
                e_ij[l] += (a[:, pi] * a[:, pj]).sum(0).double()
                e_sq[l] += (a ** 2).sum(0).double()
    for h in hooks:
        h.remove()
    out = {}
    for l, (pi, pj) in pairs_by_layer.items():
        denom = torch.sqrt(e_sq[l][pi] * e_sq[l][pj]) + 1e-30
        out[l] = (e_ij[l] / denom).cpu().numpy()      # cosine co-moment per pair
    return out


def main():
    ps = np.load(RESULTS / f"pairset_{MODEL_KEY}.npz")
    layers = np.unique(ps["layer"])
    pairs_by_layer = {int(l): (ps["i"][ps["layer"] == l].astype(int),
                               ps["j"][ps["layer"] == l].astype(int))
                      for l in layers}
    soft = capture_moments(pairs_by_layer)
    soft_all = np.zeros(len(ps["layer"]))
    for l in pairs_by_layer:
        soft_all[ps["layer"] == l] = soft[l]
    np.savez(RESULTS / f"soft_cofire_{MODEL_KEY}.npz", soft=soft_all)

    # rerun the interaction regression with soft co-activation replacing lift
    rz = np.load(RESULTS / f"rows_{MODEL_KEY}.npz")
    pl, pn, rows, dn = (rz["probe_layer"], rz["probe_neuron"], rz["rows"],
                        rz["delta_norms"])
    pidx = {(int(l), int(n)): k for k, (l, n) in enumerate(zip(pl, pn))}
    use = np.nonzero(ps["in_reg"] & ~ps["is_val"])[0]
    y, g, c, lay = [], [], [], []
    covs = {k: [] for k in ("log_rate", "log_gradmag", "wnorm", "upd_norm")}
    for k in use:
        l, i, j = int(ps["layer"][k]), int(ps["i"][k]), int(ps["j"][k])
        a, b = pidx.get((l, i)), pidx.get((l, j))
        if a is None or b is None:
            continue
        s = 0.5 * (rows[a, j] + rows[b, i]) / (dn[l, i] * dn[l, j] + 1e-30)
        y.append(np.log(abs(s) + 1e-30))
        g.append(ps["geom_base"][k])
        c.append(soft_all[k])
        lay.append(l)
        for cc in covs:
            covs[cc].append(ps[cc][k])
    y, g, c, lay = map(np.asarray, (y, g, c, lay))
    zsc = lambda x: (x - x.mean()) / (x.std() + 1e-12)
    from scipy import stats
    print(f"n = {len(y)}; Spearman(soft, q99 lift proxy check): "
          f"{stats.spearmanr(c, [ps['log_lift'][k] for k in use[:len(c)]]).statistic:+.3f}")
    X_full = np.column_stack([np.ones(len(y)), zsc(g), zsc(c), zsc(g * c)]
                             + [zsc(np.asarray(covs[cc])) for cc in covs])
    X_red = X_full[:, [0, 1, 2] + list(range(4, X_full.shape[1]))]
    fit = lambda X, yy: np.linalg.lstsq(X, yy, rcond=None)[0]
    beta = fit(X_full, zsc(y))
    b_red = fit(X_red, zsc(y))
    res = zsc(y) - X_red @ b_red
    rng = np.random.default_rng(0)
    null = []
    for _ in range(5000):
        rp = res.copy()
        for l in np.unique(lay):
            m = lay == l
            rp[m] = rng.permutation(rp[m])
        null.append(fit(X_full, X_red @ b_red + rp)[3])
    pval = float((np.abs(null) >= abs(beta[3])).mean())
    print(f"SOFT-COFIRE regression: geom {beta[1]:+.3f}, soft {beta[2]:+.3f}, "
          f"INTERACTION {beta[3]:+.3f} (perm p = {pval:.4f})")
    # within-class check
    for name, m in (("crowded", g > 0.4), ("orthogonal", g < 0.05)):
        r = stats.spearmanr(y[m], c[m])
        print(f"  within {name}: Spearman(log|s|, soft) = {r.statistic:+.3f} "
              f"(p={r.pvalue:.3f}, n={m.sum()})")


if __name__ == "__main__":
    main()
