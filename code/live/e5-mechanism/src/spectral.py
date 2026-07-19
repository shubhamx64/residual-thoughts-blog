"""E-M6 spectral-leverage scores (zero-data stage, CPU-friendly).

Per layer, eigendecompose the normalized write Gram at BASE (Paper 1 selector
timepoint). Scores per neuron:
  lev90   leverage in the top-r eigenbasis, r = 90% eigenvalue mass: sum_k<=r q_ki^2
  mass90  lambda-weighted energy in that subspace: sum_k<=r lambda_k q_ki^2
  eigc    |top eigenvector loading|
  ipr     inverse participation 1 / sum_k q_ki^4 (low = concentrated)

Outputs results/spectral_<model>.npz + rank correlations vs crowd_base + 20% masks
for the scores least rank-correlated with crowding (E4 candidates, run later on GPU).
"""
import argparse
import time

import numpy as np
import torch

from common_m import MODELS, RESULTS, load_signals, mlp_key, topk_mask_per_layer

torch.set_num_threads(4)  # do not starve the concurrent GPU grid's dataloader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tinyllama-1.1b")
    args = ap.parse_args()
    t0 = time.time()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODELS[args.model],
                                                 dtype=torch.float32)
    sd = {k: v.detach() for k, v in model.state_dict().items()
          if ".mlp.down_proj" in k}
    del model
    n_layers = 1 + max(int(k.split(".")[2]) for k in sd)

    scores = {k: [] for k in ("lev90", "mass90", "eigc", "ipr")}
    for l in range(n_layers):
        W = sd[mlp_key(l, "down")]
        W = W / (W.norm(dim=0, keepdim=True) + 1e-12)
        G = (W.T @ W).to(torch.float64)
        lam, Q = torch.linalg.eigh(G)          # ascending
        lam = lam.flip(0).clamp_min(0)
        Q = Q.flip(1)
        # Gram of a (d, inter) W has rank <= d: the nullspace eigenbasis is
        # arbitrary, so all scores must be restricted to the positive-eigenvalue
        # subspace (audit repair; the earlier IPR over all 5632 eigenvectors was
        # basis-dependent). rank90 is reported against d, the possible maximum.
        d_rank = int((lam > lam[0] * 1e-9).sum())
        r = int(np.searchsorted(np.cumsum(lam.numpy()), 0.90 * float(lam.sum()))) + 1
        Q2 = Q ** 2
        scores["lev90"].append(Q2[:, :r].sum(1).numpy())
        scores["mass90"].append((Q2[:, :r] * lam[:r]).sum(1).numpy())
        scores["eigc"].append(Q[:, 0].abs().numpy())
        Qr = Q2[:, :d_rank]
        rownorm = Qr.sum(1, keepdim=True) + 1e-30   # neuron energy in row space
        scores["ipr"].append((rownorm.squeeze(1) ** 2 / (Qr ** 2).sum(1)).numpy())
        print(f"  L{l:2d}: rank90 = {r:4d} / {d_rank} possible nonzero "
              f"(inter {G.shape[0]}) ({time.time() - t0:.0f}s)", flush=True)
    arr = {k: np.stack(v) for k, v in scores.items()}
    np.savez(RESULTS / f"spectral_{args.model}.npz", **arr)

    sig = load_signals(args.model)
    from scipy import stats
    print("\nrank correlation vs crowd_base (median across layers):")
    rho = {}
    for k, v in arr.items():
        rho[k] = float(np.median([stats.spearmanr(v[l], sig["crowd_base"][l]).statistic
                                  for l in range(n_layers)]))
        print(f"  {k}: {rho[k]:+.3f}")
    # E4 candidates: the two scores least |rho|-correlated with crowding
    cands = sorted(rho, key=lambda k: abs(rho[k]))[:2]
    for k in cands:
        m = topk_mask_per_layer(arr[k], 0.20)
        np.savez(RESULTS / f"mask_spectral_{k}_{args.model}.npz",
                 **{f"L{l}": m[l] for l in m})
    print(f"E4 candidate masks written for: {cands} (least crowding-correlated)")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
