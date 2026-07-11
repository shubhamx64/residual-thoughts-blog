"""E5c planted-law instrument check: the trained toy (toy_control.py) turned
out not to instantiate a crowding->mixing law at neuron granularity, so it
cannot serve as an end-to-end sensitivity control. Here we PLANT the law by
construction and verify the identical census statistics recover it.

Construction (seed 0): 512 neurons in a 512-dim input space, 5 regimes,
with a GRADED crowding->mixing law (not a two-group step, which produces a
degenerate bimodal density that inflates the rank-residual permutation null;
the real census density is a spread-out count, so the toy must be too).

- Each neuron gets a latent crowding level t_i ~ Uniform[0,1]. Its read
  direction is a(t_i)*shared_cluster_unit + (1-a(t_i))*independent_unit with
  a(t) = 0.1 + 0.85*t, so within-cluster cosine ~ a^2 runs from ~0.01
  (t=0, genuinely isolated, density 0) to ~0.85 (t=1, tightly crowded),
  spanning the density count continuously like the real census.
- Number of regimes a neuron fires on is drawn to increase with t:
  n_reg = 1 + Binomial(4, t_i), so mixing rises smoothly with crowding --
  the planted law -- but with per-neuron noise, not a clean function.
- Per-neuron total firing rate is log-uniform [0.002, 0.05] drawn
  INDEPENDENTLY of t_i, and each active regime shares that rate equally, so
  total events are decoupled from t and log-rate partialing can neither
  create nor remove the effect. 500k tokens, regime uniform.

PRE-REGISTERED (before running): instrument PASSES if partial
rho(entropy, density | log rate) >= +0.15 (expected far higher) and the
permutation null (200 label shuffles) q95 |rho| < 0.10.

Output: results/toy/planted_control.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from stats_e5 import partial_spearman

rng = np.random.default_rng(0)
R, H, D, TOK, NCLUST = 5, 512, 512, 500_000, 48

# latent crowding level and shared cluster vectors
def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)

t = rng.uniform(0, 1, H)
clust = rng.integers(0, NCLUST, H)
bases = unit(rng.standard_normal((NCLUST, D)))
a = 0.1 + 0.85 * t
V = np.array([a[i] * bases[clust[i]]
              + (1 - a[i]) * unit(rng.standard_normal(D)) for i in range(H)])
V /= np.linalg.norm(V, axis=1, keepdims=True)
G = np.abs(V @ V.T)
np.fill_diagonal(G, 0)
density = (G > 0.4).sum(1).astype(float)
top10 = np.sort(G, axis=1)[:, -10:].mean(1)

# graded mixing: n regimes rises with crowding; rate independent of t
n_reg = 1 + rng.binomial(4, t)
rate = np.exp(rng.uniform(np.log(0.002), np.log(0.05), H))
tokens = np.full(R, TOK / R)
counts = np.zeros((R, H))
for i in range(H):
    regs = rng.choice(R, size=n_reg[i], replace=False)
    per = rate[i] * R / n_reg[i]          # equal share, total events ~ rate*TOK
    for r in regs:
        counts[r, i] = rng.binomial(int(tokens[r]), min(per, 1.0))

rates = counts / tokens[:, None]
P = rates / np.maximum(rates.sum(0, keepdims=True), 1e-30)
with np.errstate(divide="ignore", invalid="ignore"):
    ent = -(P * np.log2(np.maximum(P, 1e-30))).sum(0)
events = counts.sum(0)
keep = events >= 50
lr = np.log10(np.maximum(events[keep] / tokens.sum(), 1e-8))

hi, lo = t[keep] >= 0.75, t[keep] <= 0.25
res = {"n_kept": int(keep.sum()),
       "partial_rho_density": partial_spearman(ent[keep], density[keep], lr),
       "partial_rho_top10": partial_spearman(ent[keep], top10[keep], lr),
       "density_span": [float(np.min(density)), float(np.median(density)),
                        float(np.max(density))],
       "median_density_hi_t": float(np.median(density[keep][hi])),
       "median_density_lo_t": float(np.median(density[keep][lo]))}
# null calibration: 200 label permutations (a single shuffle draw is too
# noisy a null estimate; sd under null ~ 1/sqrt(n) ~ 0.044 here)
null = [partial_spearman(rng.permutation(ent[keep]), density[keep], lr)
        for _ in range(200)]
res["shuffle_rho_median_abs"] = float(np.median(np.abs(null)))
res["shuffle_rho_q95_abs"] = float(np.quantile(np.abs(null), 0.95))
res["passes"] = bool(res["partial_rho_density"] >= 0.15
                     and res["shuffle_rho_q95_abs"] < 0.10)

od = ROOT / "results" / "toy"
od.mkdir(parents=True, exist_ok=True)
with open(od / "planted_control.json", "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
print(f"PLANTED CONTROL {'PASSES' if res['passes'] else 'FAILS'}")
