"""E5b read-side census: replace write-side crowding columns with gate-row
(read) geometry; mixing/rate/flag columns copied from the existing E5
census.parquet so stats_e5.py runs unchanged via --census.

Pre-registration: results/REPORT_E5B.md (written before this ran).

Usage: python census_read.py --model qwen2.5-1.5b
Output: results/<model>/census_read.parquet
        (schema-compatible with census.parquet; adds density_w = write-side
         density and *_cat concatenated-read sensitivity columns)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE / "e2-welch-gain" / "src"))

from extract import MODELS, load_weights, extract_layers  # e2

DEV = "cuda"
DENSITY_COS = 0.4
COUPLE_COS = 0.5


@torch.no_grad()
def gram_cols(Rn, inter):
    """max|cos|, signed nn, density(0.4), density(0.2), top10 from unit rows."""
    G = Rn @ Rn.T
    G.fill_diagonal_(0)
    A = G.abs()
    am = A.argmax(1)
    out = dict(
        max_cos=A.max(1).values.cpu().numpy(),
        signed_nn=G[torch.arange(inter, device=DEV), am].cpu().numpy(),
        density=(A > DENSITY_COS).sum(1).cpu().numpy().astype(float),
        density02=(A > 0.2).sum(1).cpu().numpy().astype(float),
        top10=A.topk(10, dim=1).values.mean(1).cpu().numpy(),
    )
    del G, A
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def read_geometry(layers):
    n_layers = len(layers)
    inter = layers[0]["Wgate"].shape[0]
    cols = {k: np.zeros((n_layers, inter)) for k in
            ("max_cos", "signed_nn", "density", "density02", "top10",
             "max_cos_cat", "density_cat", "top10_cat")}
    for l in range(n_layers):
        Wg = layers[l]["Wgate"].to(DEV)                       # (inter, d)
        Gn = Wg / (Wg.norm(dim=1, keepdim=True) + 1e-12)
        for k, v in gram_cols(Gn, inter).items():
            cols[k][l] = v
        # sensitivity: concatenated unit gate + unit up rows (identical to
        # primary on Pythia where Wgate is Wup)
        Wu = layers[l]["Wup"].to(DEV)
        Un = Wu / (Wu.norm(dim=1, keepdim=True) + 1e-12)
        Cn = torch.cat([Gn, Un], dim=1) / np.sqrt(2)
        cat = gram_cols(Cn, inter)
        cols["max_cos_cat"][l] = cat["max_cos"]
        cols["density_cat"][l] = cat["density"]
        cols["top10_cat"][l] = cat["top10"]
        if l % 8 == 0:
            print(f"  read geom L{l}", flush=True)
    return cols, n_layers, inter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    base = pd.read_parquet(ROOT / "results" / args.model / "census.parquet")

    model = load_weights(args.model)
    layers, d = extract_layers(model, args.model)
    del model
    torch.cuda.empty_cache()
    cols, n_layers, inter = read_geometry(layers)
    assert len(base) == n_layers * inter, "census/weight shape mismatch"
    # base is meshgrid layer-major (layer.ravel ordering), same as here
    assert (base["layer"].values ==
            np.repeat(np.arange(n_layers), inter)).all()

    df = base.copy()
    df["density_w"] = base["density"].values  # keep write-side for double-partial
    for k, v in cols.items():
        df[k] = v.ravel()

    # read-side couples + strata, same recipe as E5
    ct = np.full(len(df), "uncoupled", dtype=object)
    ct[(df["max_cos"] > COUPLE_COS) & (df["signed_nn"] < 0)] = "opponent"
    ct[(df["max_cos"] > COUPLE_COS) & (df["signed_nn"] > 0)] = "duplicate"
    df["couple"] = ct
    for col, name in (("density", "stratum"), ("density02", "stratum02")):
        strat = np.array(df["couple"], dtype=object)
        for l in range(n_layers):
            m = (df["layer"] == l) & (df["couple"] == "uncoupled")
            dens = df.loc[m, col]
            if len(dens):
                q1, q3 = dens.quantile(0.25), dens.quantile(0.75)
                idx = df.index[m]
                strat[idx] = "uncoupled-mid"
                strat[idx[dens >= q3]] = "uncoupled-crowded"
                strat[idx[dens <= q1]] = "isolated"
        df[name] = strat

    od = ROOT / "results" / args.model
    df.to_parquet(od / "census_read.parquet")
    print(f"wrote {od / 'census_read.parquet'}: {len(df)} neurons, "
          f"read couples: opp {(df['couple']=='opponent').mean():.1%} "
          f"dup {(df['couple']=='duplicate').mean():.1%}, "
          f"median read density {df['density'].median():.0f} "
          f"(write {df['density_w'].median():.0f})", flush=True)


if __name__ == "__main__":
    main()
