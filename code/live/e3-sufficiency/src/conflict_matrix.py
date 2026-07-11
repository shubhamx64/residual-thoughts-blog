"""E5 preview: the predicted regime-conflict matrix.

conflict(c1, c2) at layer l = sum_ij r_i(c1) r_j(c2) <w_i, w_j>^2
  = <M_c1, M_c2>_F  with  M_c = W diag(r_c) W^T  (d x d)
where w = gamma-folded down-proj write columns and r_c = class firing rates
from the E1 footprints. Squared overlap (frame-potential flavor) makes the
computation factorizable; |overlap| does not.

Reported: per-model, layer-averaged normalized matrix
  C_hat(c1,c2) = C(c1,c2) / sqrt(C(c1,c1) C(c2,c2))
plus raw diagonal (within-regime packing pressure) per class.
"""
import argparse
import json

import numpy as np
import torch

from common_e3 import result_dir
from extract import MODELS, load_weights, extract_layers
from common import CLASSES
from metrics import load_records, group, freq_vector

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--distinctive", action="store_true",
                    help="use max(r_c - mean_c r, 0): conflict among regime-specific substrate")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    model = load_weights(args.model)
    layers, d = extract_layers(model, args.model)
    inter = layers[0]["Wdown"].shape[1]
    n_layers = len(layers)
    recs = load_records(args.model, 99.0)
    rates = {c: [freq_vector(group(recs, c), l, inter) for l in range(n_layers)]
             for c in CLASSES}
    if args.distinctive:
        for l in range(n_layers):
            mean_r = np.mean([rates[c][l] for c in CLASSES], axis=0)
            for c in CLASSES:
                rates[c][l] = np.maximum(rates[c][l] - mean_r, 0)

    nC = len(CLASSES)
    norm_mats, diag_raw = [], []
    for l in range(n_layers):
        W = layers[l]["Wdown"].to(DEV)
        W = W / (W.norm(dim=0, keepdim=True) + 1e-12)
        M = []
        for c in CLASSES:
            r = torch.tensor(rates[c][l], device=DEV, dtype=torch.float32)
            M.append((W * r) @ W.T)                     # (d, d)
        C = np.zeros((nC, nC))
        for a in range(nC):
            for b in range(a, nC):
                C[a, b] = C[b, a] = float((M[a] * M[b]).sum())
        dg = np.sqrt(np.diag(C))
        norm_mats.append(C / np.outer(dg, dg))
        diag_raw.append(np.diag(C).tolist())

    avg = np.mean(norm_mats, axis=0)
    out = {"model": args.model, "classes": CLASSES, "distinctive": args.distinctive,
           "normalized_conflict_layer_avg": avg.tolist(),
           "diag_raw_per_layer": diag_raw}
    suffix = "_distinctive" if args.distinctive else ""
    with open(result_dir(args.model) / f"conflict_matrix{suffix}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"{args.model} normalized conflict (layer-avg{suffix}):")
    hdr = "          " + " ".join(f"{c[:9]:>10}" for c in CLASSES)
    print(hdr)
    for a in range(nC):
        print(f"{CLASSES[a][:9]:>9} " + " ".join(f"{avg[a, b]:10.3f}" for b in range(nC)))


if __name__ == "__main__":
    main()
