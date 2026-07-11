"""E3b: anatomy of crowded pairs.

Q1 Signed geometry: are near-duplicate write pairs parallel (redundant) or
   anti-parallel (opponent/gating motifs)? |cos| hid the sign.
Q2 Token semantics: do crowded pairs write toward the same vocabulary?
   (unembedding projection of write directions, top-k token overlap)
Q3 Regime locality: is co-firing of crowded pairs concentrated in one E1 regime
   (regime-local conflict, cheaper) or spread (global conflict, expensive)?
   Uses the per-class joint counts already captured in E3.
"""
import argparse
import json

import numpy as np
import torch

from common_e3 import result_dir
from extract import MODELS, load_weights, extract_layers
from common import CLASSES  # e1

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HIGH_GEOM = 0.6
TOPK_TOK = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--examples-layer", type=int, default=None)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    model = load_weights(args.model)
    layers, d = extract_layers(model, args.model)
    W_U = model.get_output_embeddings().weight.float().to(DEV)   # (vocab, d)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    except Exception:
        tok = None

    rd = result_dir(args.model)
    n_layers = len(layers)
    ex_layer = args.examples_layer if args.examples_layer is not None else n_layers // 2

    signed_stats, tokstats, regime_stats, examples = [], [], [], []
    for l in range(n_layers - 1):
        pz = rd / f"pairs_L{l}.npz"
        cz = rd / f"coact_L{l}.npz"
        if not pz.exists() or not cz.exists():
            continue
        P, C = np.load(pz), np.load(cz)
        Wd = layers[l]["Wdown"].to(DEV)
        Wd = Wd / (Wd.norm(dim=0, keepdim=True) + 1e-12)
        pi = torch.tensor(P["pi"], device=DEV, dtype=torch.long)
        pj = torch.tensor(P["pj"], device=DEV, dtype=torch.long)
        signed = (Wd[:, pi] * Wd[:, pj]).sum(0).cpu().numpy()
        high = P["geom"] >= HIGH_GEOM

        # Q1
        n_high = int(high.sum())
        if n_high:
            frac_anti = float((signed[high] < 0).mean())
        else:
            frac_anti = np.nan
        signed_stats.append({"layer": l, "n_high": n_high, "frac_antiparallel": frac_anti})

        # Q3: per-class lift for high-geom pairs
        T_c = C["tokens_per_class"]
        U = C["U"]
        pos = {int(u): k for k, u in enumerate(U)}
        mi = np.array([pos[int(x)] for x in P["pi"]])
        mj = np.array([pos[int(x)] for x in P["pj"]])
        joint_c = C["joint"]                                  # (n_classes, n_pairs)
        share = joint_c / np.maximum(joint_c.sum(0, keepdims=True), 1)
        tok_share = T_c / T_c.sum()
        # concentration: max class share, only for pairs with enough joint events
        total_joint = joint_c.sum(0)
        m = high & (total_joint >= 20)
        if m.sum() >= 5:
            conc = share[:, m].max(0)
            dom = share[:, m].argmax(0)
            regime_stats.append({
                "layer": l, "n": int(m.sum()),
                "median_max_class_share": float(np.median(conc)),
                "frac_concentrated_70": float((conc >= 0.7).mean()),
                "dominant_class_counts": {CLASSES[k]: int((dom == k).sum())
                                          for k in range(len(CLASSES))},
                "token_share_max": float(tok_share.max()),
            })

        # Q2: token overlap of write directions (top-k unembedding projection)
        if n_high:
            hi_idx = np.nonzero(high)[0]
            sub = hi_idx[np.random.default_rng(0).permutation(len(hi_idx))[:60]]
            li = W_U @ Wd[:, pi[sub]]
            lj = W_U @ Wd[:, pj[sub]]
            sgn = torch.tensor(np.sign(signed[sub]), device=DEV, dtype=torch.float32)
            lj = lj * sgn  # align anti-parallel partner so "same content" is comparable
            ov = []
            for k in range(len(sub)):
                a = set(torch.topk(li[:, k], TOPK_TOK).indices.tolist())
                b = set(torch.topk(lj[:, k], TOPK_TOK).indices.tolist())
                ov.append(len(a & b) / TOPK_TOK)
            # baseline: random pair token overlap
            lo_idx = np.nonzero(P["geom"] < 0.05)[0][:60]
            ov0 = []
            if len(lo_idx):
                li0 = W_U @ Wd[:, pi[lo_idx]]
                lj0 = W_U @ Wd[:, pj[lo_idx]]
                for k in range(len(lo_idx)):
                    a = set(torch.topk(li0[:, k], TOPK_TOK).indices.tolist())
                    b = set(torch.topk(lj0[:, k], TOPK_TOK).indices.tolist())
                    ov0.append(len(a & b) / TOPK_TOK)
            tokstats.append({"layer": l, "high_tok_overlap_med": float(np.median(ov)),
                             "low_tok_overlap_med": float(np.median(ov0)) if ov0 else None})

        if l == ex_layer and tok is not None and n_high:
            for k in np.argsort(-P["geom"])[:6]:
                a = torch.topk(W_U @ Wd[:, int(P["pi"][k])], 8).indices.tolist()
                b = torch.topk(W_U @ Wd[:, int(P["pj"][k])], 8).indices.tolist()
                examples.append({
                    "layer": l, "geom": float(P["geom"][k]),
                    "signed": float(signed[k]),
                    "tokens_i": [tok.decode([t]) for t in a],
                    "tokens_j": [tok.decode([t]) for t in b],
                })

    out = {"model": args.model, "high_geom_threshold": HIGH_GEOM,
           "signed": signed_stats, "token_overlap": tokstats,
           "regime_locality": regime_stats, "examples": examples}
    with open(rd / "anatomy.json", "w") as f:
        json.dump(out, f, indent=1)

    fa = [s["frac_antiparallel"] for s in signed_stats if s["n_high"] >= 10]
    hi = [t["high_tok_overlap_med"] for t in tokstats]
    lo = [t["low_tok_overlap_med"] for t in tokstats if t["low_tok_overlap_med"] is not None]
    cc = [r["frac_concentrated_70"] for r in regime_stats]
    ms = [r["median_max_class_share"] for r in regime_stats]
    print(f"{args.model}:")
    print(f"  antiparallel fraction among high-|cos| pairs: med {np.nanmedian(fa):.2f}")
    print(f"  token top-{TOPK_TOK} overlap: high-geom med {np.median(hi):.2f} vs low-geom {np.median(lo):.2f}")
    print(f"  regime concentration of co-firing: median max-class share {np.median(ms):.2f}, "
          f"frac pairs >=70% one regime: {np.median(cc):.2f}")


if __name__ == "__main__":
    main()
