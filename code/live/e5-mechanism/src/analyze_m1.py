"""E-M1 primary analysis (PREREG stage-2). Verdict first.

(a-primary)  Factorization: |s_ij| ~ |cos_base| + log_lift + interaction + covs,
             Freedman-Lane permutation with within-layer blocks (5000 perms),
             validation pairs excluded. CONFIRM: interaction > 0, p < 0.01.
(a-secondary) Matched contrast: paired Wilcoxon on the 202 matched pair-pairs.
             CONFIRM: median ratio >= 1.5, p < 0.01 (raw + unit normalizations).
(b-partial)  rho(crowd_base, K_within) among probes, per layer, median.
             (K2_global/unit arrive with the sketches; analyzed separately.)
Also: hess-sensitivity concordance, first-order guard, log-log robustness.
"""
import numpy as np
from scipy import stats

from common_m import RESULTS, load_signals

MODEL_KEY = "tinyllama-1.1b"
N_PERM = 5000
COVS = ("log_rate", "log_gradmag", "wnorm", "upd_norm")


def zsc(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def fit(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def main():
    ps = np.load(RESULTS / f"pairset_{MODEL_KEY}.npz")
    rz = np.load(RESULTS / f"rows_{MODEL_KEY}.npz")
    sig = load_signals(MODEL_KEY)
    assert rz["done"].all(), "pairs run incomplete"
    pl, pn, rows, dn = rz["probe_layer"], rz["probe_neuron"], rz["rows"], rz["delta_norms"]
    pidx = {(int(l), int(n)): k for k, (l, n) in enumerate(zip(pl, pn))}

    # ---- pair-level s_ij (symmetrized, unit-normalized) for regression pairs
    def s_of(k):
        l, i, j = int(ps["layer"][k]), int(ps["i"][k]), int(ps["j"][k])
        a, b = pidx.get((l, i)), pidx.get((l, j))
        if a is None or b is None:
            return None
        S = 0.5 * (rows[a, j] + rows[b, i])
        return S, S / (dn[l, i] * dn[l, j] + 1e-30), l

    use = np.nonzero(ps["in_reg"] & ~ps["is_val"])[0]
    recs = [(k, *s_of(k)) for k in use if s_of(k) is not None]
    ks = np.array([r[0] for r in recs])
    S_raw = np.array([r[1] for r in recs])
    s_unit = np.array([r[2] for r in recs])
    lay = np.array([r[3] for r in recs])
    print(f"regression pairs (validation excluded): {len(ks)}")

    y = zsc(np.log(np.abs(s_unit) + 1e-30))
    g = zsc(ps["geom_base"][ks])
    c = zsc(ps["log_lift"][ks])
    inter_t = zsc(ps["geom_base"][ks] * ps["log_lift"][ks])
    covs = [zsc(ps[c_][ks]) for c_ in COVS]
    X_full = np.column_stack([np.ones(len(y)), g, c, inter_t] + covs)
    X_red = np.column_stack([np.ones(len(y)), g, c] + covs)

    beta = fit(X_full, y)
    b_int, b_g, b_c = beta[3], beta[1], beta[2]

    # Freedman-Lane: permute reduced-model residuals within layer blocks
    beta_r = fit(X_red, y)
    yhat_r = X_red @ beta_r
    res_r = y - yhat_r
    rng = np.random.default_rng(0)
    layers = np.unique(lay)
    null = np.empty(N_PERM)
    for p in range(N_PERM):
        rp = res_r.copy()
        for l in layers:
            m = lay == l
            rp[m] = rng.permutation(rp[m])
        null[p] = fit(X_full, yhat_r + rp)[3]
    pval = float((np.abs(null) >= abs(b_int)).mean())

    # leave-one-layer-out sensitivity
    loo = [fit(X_full[lay != l], y[lay != l])[3] for l in layers]

    print("\n== (a-primary) FACTORIZATION TEST ==")
    print(f"  log|s_unit| ~ geom {b_g:+.3f}, log_lift {b_c:+.3f}, "
          f"INTERACTION {b_int:+.3f} (perm p = {pval:.4f})")
    print(f"  leave-one-layer-out interaction range: "
          f"[{min(loo):+.3f}, {max(loo):+.3f}]")
    conf_a = b_int > 0 and pval < 0.01
    print(f"  -> {'CONFIRM' if conf_a else 'NOT CONFIRMED'} "
          f"(rule: interaction > 0, p < 0.01)")

    # per-layer interaction signs
    signs = []
    for l in layers:
        m = lay == l
        if m.sum() >= 30:
            signs.append(np.sign(fit(X_full[m], y[m])[3]))
    print(f"  per-layer interaction positive: {int(sum(s > 0 for s in signs))}"
          f"/{len(signs)} layers (n >= 30)")

    # ---- (a-secondary) matched contrast
    # s_of returns (S_raw, s_unit, layer): index 0 = raw, 1 = unit.
    # (Audit repair 2026-07-13: the earlier fallback read r[2]/r[1] -- layer as
    # unit value -- for the 24 validation-containing contrasts.)
    print("\n== (a-secondary) MATCHED CONTRAST (crowded vs orthogonal) ==")
    mc = np.nonzero(ps["crowded"] & (ps["match_id"] >= 0))[0]
    diag_of = {}
    for k in np.concatenate([mc, ps["match_id"][mc]]):
        l, i, j = int(ps["layer"][k]), int(ps["i"][k]), int(ps["j"][k])
        a_, b_ = pidx.get((l, i)), pidx.get((l, j))
        if a_ is not None and b_ is not None:
            diag_of[int(k)] = (rows[a_, i], rows[b_, j])
    for label, idx_ in (("unit", 1), ("raw", 0), ("coh", None)):
        a, b, lay_mc = [], [], []
        for k in mc:
            o = int(ps["match_id"][k])
            r, ro = s_of(k), s_of(o)
            if label == "coh":  # diagonal-normalized coherence |S_ij|/sqrt(Sii Sjj)
                dk, do = diag_of.get(int(k)), diag_of.get(int(o))
                va = abs(r[0]) / (np.sqrt(abs(dk[0] * dk[1])) + 1e-30)
                vb = abs(ro[0]) / (np.sqrt(abs(do[0] * do[1])) + 1e-30)
            else:
                va, vb = abs(r[idx_]), abs(ro[idx_])
            a.append(va); b.append(vb); lay_mc.append(int(ps["layer"][k]))
        a, b, lay_mc = np.array(a), np.array(b), np.array(lay_mc)
        w = stats.wilcoxon(a, b, alternative="greater")
        ratio = np.median(a) / (np.median(b) + 1e-30)
        # LOAD-BEARING inference: layers as units (pairs share neurons/layers,
        # so the dyad-level Wilcoxon is descriptive only)
        lr = [np.median(a[lay_mc == l]) / (np.median(b[lay_mc == l]) + 1e-30)
              for l in np.unique(lay_mc)]
        npos = sum(r_ > 1 for r_ in lr)
        p_sign = stats.binomtest(npos, len(lr), 0.5, alternative="greater").pvalue
        print(f"  |s_{label}|: median {np.median(a):.3e} vs {np.median(b):.3e}; "
              f"ratio {ratio:.2f} (dyad Wilcoxon p {w.pvalue:.1e}, descriptive); "
              f"LAYER-LEVEL {npos}/{len(lr)} ratios > 1, sign p {p_sign:.1e}; "
              f"per-layer ratio range [{min(lr):.2f}, {max(lr):.2f}] (n={len(a)})")

    # ---- (b-partial) crowding vs within-layer K among probes
    print("\n== (b-partial) crowd_base vs K_within (probes) ==")
    for label, norm in (("K_raw", None), ("K_unit", dn)):
        rhos = []
        for l in np.unique(pl):
            m = pl == l
            if m.sum() < 20:
                continue
            idx = pn[m]
            R = rows[m].copy()
            if norm is not None:
                R = R / (norm[l][None, :] + 1e-30)
                R = R / (norm[l][idx][:, None] + 1e-30)
            # registered L2 row norm, diagonal removed (audit repair: was L1)
            diag = R[np.arange(m.sum()), idx]
            K = np.sqrt(np.maximum(0, (R ** 2).sum(1) - diag ** 2))
            rhos.append(stats.spearmanr(sig["crowd_base"][l][idx], K).statistic)
        print(f"  {label}: median per-layer rho {np.median(rhos):+.3f} "
              f"({len(rhos)} layers; rule: CONFIRM >= +0.20 on unit)")

    # ---- guards
    print("\n== guards ==")
    hs = rz["hrows"]
    hm = ~np.isnan(hs).all(1)
    cc = []
    for k in np.nonzero(hm)[0]:
        l, i = int(pl[k]), int(pn[k])
        act = np.abs(rows[k]) > np.percentile(np.abs(rows[k]), 99)
        cc.append(stats.spearmanr(rows[k][act], hs[k][act]).statistic)
    print(f"  GGN vs Hessian row concordance (top-1% entries, {hm.sum()} probes): "
          f"median rho {np.median(cc):+.3f}")
    gd = np.abs(sig["gdotdelta_s0"])
    r1 = [stats.spearmanr(gd[l][pn[pl == l]],
                          np.abs(rows[pl == l]).sum(1)).statistic
          for l in np.unique(pl) if (pl == l).sum() >= 20]
    print(f"  first-order |g.delta| vs K_raw: median rho {np.median(r1):+.3f} "
          f"(descriptive; second-order claims stated net of this)")


if __name__ == "__main__":
    main()
