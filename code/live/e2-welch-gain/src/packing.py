"""Packing statistics for a dictionary of directions (rows of D, n x d).

Reported per dictionary:
  welch_bound      provable floor on worst-case |cos| when n > d
  coherence_max    exact max off-diagonal |cos| (chunked; skip for huge n)
  q50/q90/q99/q999 quantiles of |cos| over sampled pairs (robust to near-dup rows)
  fp_ratio         FP_min / FP in (0, 1]; 1 = tight frame (optimal energy spread)
                   FP = ||D_n D_n^T||_F^2 computed via the d x d Gram trick
  stable_rank      ||D||_F^2 / sigma_max^2, normalized by d
"""
import math

import numpy as np
import torch

PAIR_SAMPLE = 2_000_000
EXACT_COH_MAX_N = 20_000


def _normalize(D):
    return D / (D.norm(dim=1, keepdim=True) + 1e-12)


@torch.no_grad()
def packing_stats(D, device="cuda", seed=0):
    n, d = D.shape
    Dn = _normalize(D.to(device))

    welch = math.sqrt(max(n - d, 0) / (d * (n - 1))) if n > 1 else 0.0

    # frame potential via d x d Gram: FP = tr((Dn^T Dn)^2)
    G_small = Dn.T @ Dn
    fp = float((G_small * G_small).sum())
    fp_min = n * n / d if n >= d else n  # unit-norm tight-frame minimum
    fp_ratio = fp_min / fp

    # stable rank
    sv_max = float(torch.linalg.matrix_norm(Dn, ord=2))
    stable_rank = n / (sv_max ** 2)  # ||Dn||_F^2 = n for unit rows

    # off-diagonal |cos| quantiles over sampled pairs
    g = torch.Generator(device="cpu").manual_seed(seed)
    m = min(PAIR_SAMPLE, n * (n - 1) // 2)
    i = torch.randint(0, n, (m,), generator=g)
    j = torch.randint(0, n, (m,), generator=g)
    keep = i != j
    i, j = i[keep].to(device), j[keep].to(device)
    chunks = []
    for a in range(0, len(i), 100_000):
        chunks.append((Dn[i[a:a + 100_000]] * Dn[j[a:a + 100_000]]).sum(1).abs())
    vals = torch.cat(chunks)
    qs = torch.quantile(vals.float(), torch.tensor([0.5, 0.9, 0.99, 0.999], device=device))

    coh = None
    if n <= EXACT_COH_MAX_N:
        coh = 0.0
        chunk = 4096
        for a in range(0, n, chunk):
            block = (Dn[a:a + chunk] @ Dn.T).abs()
            for r in range(block.shape[0]):
                block[r, a + r] = 0.0
            coh = max(coh, float(block.max()))

    return {"n": int(n), "d": int(d), "welch_bound": welch,
            "coherence_max": coh,
            "q50": float(qs[0]), "q90": float(qs[1]),
            "q99": float(qs[2]), "q999": float(qs[3]),
            "fp_ratio": float(fp_ratio),
            "stable_rank_frac": float(stable_rank / d)}
