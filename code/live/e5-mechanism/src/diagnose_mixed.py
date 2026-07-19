"""Mixed-derivative gate diagnosis (PREREG stage-1 addendum, 2026-07-12).

For each registered val_mix pair: print |s_ij| (hess row, unit directions), the
4-point stencil at each h, and the signal / noise-floor ratio. Then rerun the
stencil on the largest-|s_ij| within-layer partner of each val_mix probe neuron,
restricted to predicted signal >= 3x noise floor.
"""
import numpy as np
import torch

from common_m import RESULTS
from curvature_run import load_all, neuron_dir, probe_row
from ggn import Perturb, loss_global

MODEL_KEY = "tinyllama-1.1b"
HS = (1e-2, 3e-2, 1e-1)
DELTA_L = 1.5e-7          # fp32 loss quantization, absolute (L ~ 1.5)


def stencil(model, di, dj, batches, h):
    vals = {}
    for si in (+h, -h):
        for sj in (+h, -h):
            vc = {k: si * di[k] + sj * dj[k] for k in di}
            pc = Perturb(model, vc)
            try:
                pc.set(1.0)
                with torch.no_grad():
                    vals[(si, sj)] = float(loss_global(model, batches))
            finally:
                pc.restore()
    return (vals[(h, h)] - vals[(h, -h)] - vals[(-h, h)]
            + vals[(-h, -h)]) / (4 * h * h)


def main():
    ps = np.load(RESULTS / f"pairset_{MODEL_KEY}.npz")
    model, params, deltas, batches = load_all(MODEL_KEY)

    def unit_s(l, i, j):
        """hess-row coupling between unit directions of (l,i) and (l,j)."""
        row = probe_row(model, params, deltas, batches, l, i, estimator="hess")
        dn = np.sqrt(sum(float((t ** 2).sum())
                         for t in neuron_dir(deltas, l, i).values()))
        dnj = np.sqrt(sum(float((t ** 2).sum())
                          for t in neuron_dir(deltas, l, j).values()))
        return row[l] / (dn * (np.sqrt((deltas[l]["gate"].numpy() ** 2).sum(1)
                                       + (deltas[l]["up"].numpy() ** 2).sum(1)
                                       + (deltas[l]["down"].numpy() ** 2).sum(0))
                               + 1e-30)), row[l, j] / (dn * dnj)

    print("== Phase 1: registered val_mix pairs, magnitude vs noise floor ==")
    agree_hi, agree_lo, n_hi, n_lo = 0, 0, 0, 0
    rows_cache = {}
    for k in ps["val_mix"]:
        l, i, j = int(ps["layer"][k]), int(ps["i"][k]), int(ps["j"][k])
        urow, s = unit_s(l, i, j)
        rows_cache[(l, i)] = urow
        di = neuron_dir(deltas, l, i, unit=True)
        dj = neuron_dir(deltas, l, j, unit=True)
        best_h, best_m = None, None
        for h in HS:
            m = stencil(model, di, dj, batches, h)
            floor = DELTA_L / h ** 2
            if best_m is None or abs(m) * h ** 2 > abs(best_m) * best_h ** 2:
                best_h, best_m = h, m
        floor = DELTA_L / best_h ** 2
        ratio = abs(s) / floor
        ok = np.sign(best_m) == np.sign(s)
        hi = ratio >= 3
        if hi:
            n_hi += 1
            agree_hi += int(ok)
        else:
            n_lo += 1
            agree_lo += int(ok)
        print(f"  L{l} ({i},{j}): s={s:+.2e} stencil@h={best_h:g}: {best_m:+.2e} "
              f"signal/floor={ratio:5.1f} {'AGREE' if ok else 'FLIP'}")
    print(f"  above 3x floor: {agree_hi}/{n_hi} agree; below: {agree_lo}/{n_lo}")

    print("\n== Phase 2: largest-|s| partner per probe, floor-qualified ==")
    agree2, n2, pairs2 = 0, 0, []
    for k in ps["val_mix"]:
        l, i = int(ps["layer"][k]), int(ps["i"][k])
        urow = rows_cache[(l, i)]
        urow = urow.copy()
        urow[i] = 0
        j2 = int(np.argmax(np.abs(urow)))
        s2 = urow[j2]
        h = 1e-1
        if abs(s2) < 3 * DELTA_L / h ** 2:
            print(f"  L{l} ({i},{j2}): |s|={abs(s2):.2e} below 3x floor, skipped")
            continue
        di = neuron_dir(deltas, l, i, unit=True)
        dj = neuron_dir(deltas, l, j2, unit=True)
        m = stencil(model, di, dj, batches, h)
        ok = np.sign(m) == np.sign(s2)
        n2 += 1
        agree2 += int(ok)
        pairs2.append((m, s2))
        print(f"  L{l} ({i},{j2}): s={s2:+.2e} stencil={m:+.2e} "
              f"{'AGREE' if ok else 'FLIP'}")
    from scipy import stats
    if len(pairs2) >= 3:
        a, b = zip(*pairs2)
        rr = stats.spearmanr(a, b).statistic
        print(f"\nPhase-2 verdict: {agree2}/{n2} sign agreement, rank corr {rr:+.3f}")
    print("(gate rule: >= 9/10-equivalent proportion on qualified pairs, "
          "positive rank corr)")


if __name__ == "__main__":
    main()
