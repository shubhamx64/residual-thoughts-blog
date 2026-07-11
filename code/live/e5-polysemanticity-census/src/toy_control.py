"""E5c toy positive control: does the census instrument detect a
crowding->mixing law where one exists by construction?

Toy-Models-style setup (Elhage et al. 2022), adapted to the census:
- F = 2560 sparse features in R = 5 regimes (512 each), reconstructed through
  a single ReLU hidden layer of H = 512 neurons (5x compression forces
  superposition). Per-feature base activity log-uniform in [0.005, 0.2];
  in-regime samples activate a feature at its base rate, out-of-regime at
  0.05x that (regime-conditioned sparsity, mirroring E1's corpus structure).
  Anticorrelated (cross-regime) features are the cheapest to superpose, so
  trained polysemantic neurons should be BOTH geometrically crowded and
  regime-mixed: the price-of-packing law holds in this system by construction
  of the capacity pressure.
- The IDENTICAL census is then run: per-neuron q99 threshold calibrated on
  held-out samples, exceedance counts per regime, rate-normalized Shannon
  entropy, <50-event exclusion, crowding = density(|cos|>0.4) / max|cos| /
  top-10 on write columns AND read rows, partial Spearman on log rate
  (partial_spearman imported from stats_e5, same code path).

PRE-REGISTERED SUCCESS CRITERIA (written before running):
- Positive control PASSES if partial rho(entropy, density | log rate)
  >= +0.15 on at least one side (read or write), with max|cos| and top10
  agreeing in sign.
- Shuffle control (each sample assigned a random regime label at census time,
  same trained model): |partial rho| < 0.10 -- instrument does not
  hallucinate a law when regime structure is destroyed.
- If the positive control FAILS, the E5/E5b null band is not interpretable as
  evidence against price-of-packing and the strong sentence must be softened.
  This will be reported either way.

POST-HOC ADDITIONS after run 1 (documented, not silently tuned): run 1
(H=512) returned density CONSTANT ZERO -- no toy neuron pair exceeds
|cos|>0.4 -- so the pre-registered density criterion is undefined there,
while continuous measures carried signal (read top-10 +0.31 vs shuffle
+0.02). Two additions to diagnose: (1) --hidden arg for a stronger-packing
run (H=128, 20x compression); (2) ground-truth columns from the known
feature composition of each neuron (regime entropy of squared read-weight
mass; participation ratio n_eff of features carried), letting us test the
INSTRUMENT (census entropy vs ground-truth mixing) separately from the LAW
(ground-truth mixing vs crowding) inside the toy.

Seed 0. Output: results/toy/toy_control_<tag>.json + printed summary.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from stats_e5 import partial_spearman  # same code path as E5/E5b

DEV = "cuda"
ap = argparse.ArgumentParser()
ap.add_argument("--hidden", type=int, default=512)
args = ap.parse_args()
R, FPR, H = 5, 512, args.hidden
F = R * FPR
MIN_EVENTS = 50

torch.manual_seed(0)
np.random.seed(0)
BASE_P = torch.exp(torch.rand(F, device=DEV)
                   * (np.log(0.2) - np.log(0.005)) + np.log(0.005))
REGIME_OF = torch.arange(F, device=DEV) // FPR


def batch(n, gen=None):
    reg = torch.randint(0, R, (n,), device=DEV, generator=gen)
    in_reg = (REGIME_OF[None, :] == reg[:, None])
    p = torch.where(in_reg, BASE_P[None, :], 0.05 * BASE_P[None, :])
    u = torch.rand(n, F, device=DEV, generator=gen)
    x = (u < p).float() * torch.rand(n, F, device=DEV, generator=gen)
    return x, reg


def crowding(V):
    """V: (H, dim) unit-normalized rows -> density/max_cos/top10 per neuron."""
    Vn = V / (V.norm(dim=1, keepdim=True) + 1e-12)
    G = Vn @ Vn.T
    G.fill_diagonal_(0)
    A = G.abs()
    return dict(density=(A > 0.4).sum(1).float().cpu().numpy(),
                max_cos=A.max(1).values.cpu().numpy(),
                top10=A.topk(10, dim=1).values.mean(1).cpu().numpy())


def census_stats(counts, tokens, geo, tag):
    """counts: (R, H) events; tokens: (R,). Same recipe as census.py."""
    rates = counts / tokens[:, None]
    P = rates / np.maximum(rates.sum(0, keepdims=True), 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(P * np.log2(np.maximum(P, 1e-30))).sum(0)
    events = counts.sum(0)
    rate = events / tokens.sum()
    keep = events >= MIN_EVENTS
    lr = np.log10(np.maximum(rate[keep], 1e-8))
    out = {"n_kept": int(keep.sum()), "excluded_frac": float(1 - keep.mean())}
    for k, v in geo.items():
        out[f"partial_rho_{k}"] = partial_spearman(ent[keep], v[keep], lr)
    raw = sps.spearmanr(ent[keep], geo["density"][keep])[0]
    out["raw_rho_density"] = float(raw)
    print(f"  [{tag}] kept {keep.sum()}/{H}  partial rho: " +
          " ".join(f"{k}={out[f'partial_rho_{k}']:+.3f}" for k in geo) +
          f"  raw density {raw:+.3f}", flush=True)
    return out


def main():
    torch.set_grad_enabled(True)
    Win = torch.nn.Parameter(torch.randn(H, F, device=DEV) * (1 / np.sqrt(F)))
    b = torch.nn.Parameter(torch.zeros(H, device=DEV))
    Wout = torch.nn.Parameter(torch.randn(F, H, device=DEV) * (1 / np.sqrt(H)))
    opt = torch.optim.Adam([Win, b, Wout], lr=1e-3)
    for step in range(20000):
        x, _ = batch(2048)
        h = torch.relu(x @ Win.T + b)
        loss = ((x - h @ Wout.T) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 4000 == 0:
            print(f"step {step} loss {loss.item():.5f}", flush=True)
    print(f"final loss {loss.item():.5f}", flush=True)

    torch.set_grad_enabled(False)
    # calibrate per-neuron q99 on held-out samples (the E1 sensor recipe)
    gen = torch.Generator(device=DEV).manual_seed(1)
    xc, _ = batch(200000, gen)
    hc = torch.relu(xc @ Win.T + b)
    thr = torch.quantile(hc.float(), 0.99, dim=0)
    del xc, hc

    # capture: exceedance counts per regime (chunked)
    counts = np.zeros((R, H))
    counts_shuf = np.zeros((R, H))
    tokens = np.zeros(R)
    gen2 = torch.Generator(device=DEV).manual_seed(2)
    for _ in range(10):
        x, reg = batch(50000, gen2)
        fire = (torch.relu(x @ Win.T + b) > thr[None, :]).float()
        reg_shuf = torch.randint(0, R, reg.shape, device=DEV, generator=gen2)
        for r in range(R):
            counts[r] += fire[reg == r].sum(0).cpu().numpy()
            counts_shuf[r] += fire[reg_shuf == r].sum(0).cpu().numpy()
            tokens[r] += int((reg == r).sum())
    # shuffle control has uniform token counts by construction of reg_shuf;
    # use true per-label counts for exactness
    tokens_shuf = np.full(R, tokens.sum() / R)

    geo_w = crowding(Wout.T.detach())   # write columns as rows
    geo_r = crowding(Win.detach())      # read rows

    # ground truth from known feature composition (post-hoc diagnostic)
    m2 = Win.detach().pow(2)                                   # (H, F)
    reg_mass = torch.stack([m2[:, r * FPR:(r + 1) * FPR].sum(1)
                            for r in range(R)])                # (R, H)
    Pg = (reg_mass / reg_mass.sum(0, keepdim=True)).cpu().numpy()
    gt_ent = -(Pg * np.log2(np.maximum(Pg, 1e-30))).sum(0)
    n_eff = (m2.sum(1).pow(2) / m2.pow(2).sum(1)).cpu().numpy()  # participation

    # instrument sensitivity: census entropy vs ground-truth mixing
    rates = counts / tokens[:, None]
    Pc = rates / np.maximum(rates.sum(0, keepdims=True), 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        cen_ent = -(Pc * np.log2(np.maximum(Pc, 1e-30))).sum(0)
    gt = {"instrument_rho_entropy": float(sps.spearmanr(cen_ent, gt_ent)[0]),
          "law_rho_gt_ent__read_top10": float(sps.spearmanr(gt_ent, geo_r["top10"])[0]),
          "law_rho_gt_ent__write_top10": float(sps.spearmanr(gt_ent, geo_w["top10"])[0]),
          "law_rho_neff__read_top10": float(sps.spearmanr(n_eff, geo_r["top10"])[0]),
          "law_rho_neff__write_top10": float(sps.spearmanr(n_eff, geo_w["top10"])[0]),
          "gt_ent_median_bits": float(np.median(gt_ent)),
          "n_eff_median": float(np.median(n_eff)),
          "max_pair_cos_read": float(np.max(geo_r["max_cos"])),
          "max_pair_cos_write": float(np.max(geo_w["max_cos"]))}
    print("  ground truth: " +
          " ".join(f"{k}={v:+.3f}" for k, v in gt.items()), flush=True)

    res = {"final_loss": float(loss.item()), "hidden": H, "ground_truth": gt,
           "write": census_stats(counts, tokens, geo_w, "write"),
           "read": census_stats(counts, tokens, geo_r, "read"),
           "shuffle_write": census_stats(counts_shuf, tokens_shuf, geo_w,
                                         "shuffle-write"),
           "shuffle_read": census_stats(counts_shuf, tokens_shuf, geo_r,
                                        "shuffle-read")}

    ok = max(res["write"]["partial_rho_density"],
             res["read"]["partial_rho_density"]) >= 0.15
    null_ok = (abs(res["shuffle_write"]["partial_rho_density"]) < 0.10 and
               abs(res["shuffle_read"]["partial_rho_density"]) < 0.10)
    res["control_passes"] = bool(ok and null_ok)
    od = ROOT / "results" / "toy"
    od.mkdir(parents=True, exist_ok=True)
    with open(od / f"toy_control_h{H}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"POSITIVE CONTROL {'PASSES' if res['control_passes'] else 'FAILS'} "
          f"(detect>=+0.15: {ok}, shuffle<0.10: {null_ok})", flush=True)


if __name__ == "__main__":
    main()
