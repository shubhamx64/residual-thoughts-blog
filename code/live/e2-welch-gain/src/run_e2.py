"""E2 driver: per-layer packing stats + gain map, weights only, zero forwards.

Usage: python run_e2.py --model gemma-2-2b
"""
import argparse
import json
import time
from pathlib import Path

import torch

from extract import MODELS, load_weights, extract_layers, embedding_matrix
from packing import packing_stats
from gain import layer_gains

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ROOT / "results" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model = load_weights(args.model)
    layers, d = extract_layers(model, args.model)
    print(f"{args.model}: {len(layers)} layers, d={d} ({time.time()-t0:.0f}s load)", flush=True)

    E = embedding_matrix(model)
    emb_stats = packing_stats(E, device)
    print(f"  token dictionary: n={emb_stats['n']} fp_ratio={emb_stats['fp_ratio']:.3f} "
          f"q99={emb_stats['q99']:.3f} welch={emb_stats['welch_bound']:.4f}", flush=True)

    per_layer = []
    for i, L in enumerate(layers):
        row = {"layer": i}
        row["mlp_write"] = packing_stats(L["Wdown"].T, device)   # columns as directions
        row["mlp_read"] = packing_stats(L["Wgate"], device)      # rows as directions
        row["attn_write"] = packing_stats(L["Wo"].T, device)
        row["attn_read_q"] = packing_stats(L["Wq"], device)
        row["gain"] = layer_gains(L, device)
        per_layer.append(row)
        print(f"  L{i:2d} pack(mlp_w)={row['mlp_write']['fp_ratio']:.3f} "
              f"coh={row['mlp_write']['coherence_max']:.3f} "
              f"q99={row['mlp_write']['q99']:.3f} | "
              f"g_attn={row['gain']['g_attn']:.2f} g_mlp={row['gain']['g_mlp']:.2f} "
              f"maxQK={max(row['gain']['qk']):.2f}", flush=True)

    out = {"model": args.model, "d": d, "n_layers": len(layers),
           "token_dict": emb_stats, "per_layer": per_layer}
    with open(out_dir / "e2_metrics.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {out_dir / 'e2_metrics.json'} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
