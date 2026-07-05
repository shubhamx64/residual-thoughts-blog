"""E5 census: one row per MLP neuron with mixing, crowding, couple-type,
fan-out, and flag columns. All from existing E1 captures + model weights.

Usage: python census.py --model qwen2.5-1.5b
Output: results/<model>/census.parquet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE / "e1-footprint-stability" / "src"))
sys.path.insert(0, str(BASE / "e2-welch-gain" / "src"))

from common import CLASSES, load_manifest  # e1
from extract import MODELS, load_weights, extract_layers  # e2

DEV = "cuda"
QUANTILES = (98.0, 99.0, 99.5)
MIN_EVENTS = 50
DENSITY_COS = 0.4
COUPLE_COS = 0.5


def class_counts(model_key, n_layers, inter):
    """counts[q][cls] = per-layer (n_layers, inter) firing counts; tokens[cls]."""
    fp_dir = BASE / "e1-footprint-stability" / "results" / model_key / "footprints"
    man = {r["id"]: r["class"] for r in load_manifest() if r["role"] == "main"}
    counts = {q: {c: np.zeros((n_layers, inter)) for c in CLASSES} for q in QUANTILES}
    tokens = {c: 0 for c in CLASSES}
    for p in fp_dir.glob("*.npz"):
        cls = man.get(p.stem)
        if cls is None:
            continue
        z = np.load(p)
        tokens[cls] += int(z["n_tokens"])
        for q in QUANTILES:
            for l in range(n_layers):
                counts[q][cls][l][z[f"idx_q{q}_L{l}"]] += z[f"cnt_q{q}_L{l}"]
    return counts, tokens


def entropy_cols(counts_q, tokens, classes):
    """Per-layer arrays: entropy(bits), selectivity, total events, rate,
    raw-token share outside top class."""
    n_layers, inter = counts_q[classes[0]].shape
    tot_tokens = sum(tokens[c] for c in classes)
    ent = np.zeros((n_layers, inter))
    sel = np.zeros((n_layers, inter))
    events = np.zeros((n_layers, inter))
    outside = np.zeros((n_layers, inter))
    for l in range(n_layers):
        C = np.stack([counts_q[c][l] for c in classes])            # (ncls, inter)
        events[l] = C.sum(0)
        R = np.stack([counts_q[c][l] / tokens[c] for c in classes])
        P = R / np.maximum(R.sum(0, keepdims=True), 1e-30)
        with np.errstate(divide="ignore", invalid="ignore"):
            H = -(P * np.log2(np.maximum(P, 1e-30))).sum(0)
        ent[l] = H
        sel[l] = P.max(0)
        outside[l] = 1 - C.max(0) / np.maximum(C.sum(0), 1)
    rate = events / tot_tokens
    return ent, sel, events, rate, outside


@torch.no_grad()
def geometry_cols(layers, model):
    """Crowding a/b/c, couple type, fan-out, unembedding norm per neuron."""
    n_layers = len(layers)
    inter = layers[0]["Wdown"].shape[1]
    W_U = model.get_output_embeddings().weight.to(DEV, torch.bfloat16)
    out = {k: np.zeros((n_layers, inter)) for k in
           ("max_cos", "density", "density02", "top10", "signed_nn",
            "fan_out", "unembed_norm")}
    Wn_all = []
    for l in range(n_layers):
        Wd = layers[l]["Wdown"].to(DEV)
        Wn = Wd / (Wd.norm(dim=0, keepdim=True) + 1e-12)
        Wn_all.append(Wn.to(torch.bfloat16))
        G = Wn.T @ Wn
        G.fill_diagonal_(0)
        A = G.abs()
        out["max_cos"][l] = A.max(1).values.cpu().numpy()
        am = A.argmax(1)
        out["signed_nn"][l] = G[torch.arange(inter, device=DEV), am].cpu().numpy()
        out["density"][l] = (A > DENSITY_COS).sum(1).cpu().numpy()
        out["density02"][l] = (A > 0.2).sum(1).cpu().numpy()  # post-hoc sensitivity
        out["top10"][l] = A.topk(10, dim=1).values.mean(1).cpu().numpy()
        # unembedding composition norm (chunked over vocab)
        un = torch.zeros(inter, device=DEV)
        for v0 in range(0, W_U.shape[0], 32768):
            un += (W_U[v0:v0 + 32768].float() @ Wn).pow(2).sum(0)
        out["unembed_norm"][l] = un.sqrt().cpu().numpy()
        del G, A
        torch.cuda.empty_cache()
        print(f"  geom L{l}", flush=True) if l % 8 == 0 else None
    # fan-out: z-scored downstream gate-read column norms, count z>2
    for l in range(n_layers):
        z_count = np.zeros(inter)
        for lp in range(l + 1, n_layers):
            Wg = layers[lp]["Wgate"].to(DEV, torch.bfloat16)
            norms = (Wg @ Wn_all[l]).float().norm(dim=0)
            z = (norms - norms.mean()) / (norms.std() + 1e-12)
            z_count += (z > 2).cpu().numpy()
        out["fan_out"][l] = z_count
        print(f"  fanout L{l}", flush=True) if l % 8 == 0 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    model = load_weights(args.model)
    layers, d = extract_layers(model, args.model)
    n_layers, inter = len(layers), layers[0]["Wdown"].shape[1]
    print(f"{args.model}: {n_layers} layers x {inter} neurons", flush=True)

    counts, tokens = class_counts(args.model, n_layers, inter)
    print("counts loaded", flush=True)
    geo = geometry_cols(layers, model)

    rows = {}
    for q in QUANTILES:
        c4 = {c: counts[q][c] for c in CLASSES if c != "prose"}
        t4 = {c: tokens[c] for c in CLASSES if c != "prose"}
        ent, sel, ev, rate, outside = entropy_cols(counts[q], tokens, CLASSES)
        ent4, _, _, _, _ = entropy_cols(c4, t4, [c for c in CLASSES if c != "prose"])
        tag = f"q{q:g}"
        rows[f"entropy_{tag}"] = ent
        rows[f"selectivity_{tag}"] = sel
        rows[f"events_{tag}"] = ev
        rows[f"rate_{tag}"] = rate
        rows[f"outside_top_{tag}"] = outside
        rows[f"entropy4_{tag}"] = ent4

    L, N = np.meshgrid(np.arange(n_layers), np.arange(inter), indexing="ij")
    df = pd.DataFrame({"layer": L.ravel(), "neuron": N.ravel()})
    for k, v in {**rows, **geo}.items():
        df[k] = v.ravel()

    # couple type + strata
    ct = np.full(len(df), "uncoupled", dtype=object)
    ct[(df["max_cos"] > COUPLE_COS) & (df["signed_nn"] < 0)] = "opponent"
    ct[(df["max_cos"] > COUPLE_COS) & (df["signed_nn"] > 0)] = "duplicate"
    df["couple"] = ct
    # pre-registered strata (density at 0.4; degenerate where density mostly 0:
    # crowded = any strong neighbor) + post-hoc strata on density02 quartiles
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

    df["excluded"] = df["events_q99"] < MIN_EVENTS
    rate_thr = df.groupby("layer")["rate_q99"].transform(lambda s: s.quantile(0.9))
    df["flag_universal"] = (df["entropy_q99"] >= 2.1) & (df["rate_q99"] >= rate_thr)
    df["flag_entropy_neuron"] = df["unembed_norm"] >= df["unembed_norm"].quantile(0.99)

    od = ROOT / "results" / args.model
    od.mkdir(parents=True, exist_ok=True)
    df.to_parquet(od / "census.parquet")
    print(f"wrote {od / 'census.parquet'}: {len(df)} neurons, "
          f"excluded {df['excluded'].mean():.1%}, "
          f"couples: opp {(df['couple']=='opponent').mean():.1%} "
          f"dup {(df['couple']=='duplicate').mean():.1%}", flush=True)


if __name__ == "__main__":
    main()
