"""Same stability/separability/placement analysis for the AGOP sensitivity
footprint (per-neuron RMS gradient of sequence NLL, dense vectors).

Each sequence's per-layer vector is L2-normalized before aggregation to remove
loss-scale and length effects (documented design choice).
"""
import argparse
import itertools
import json

import numpy as np

from common import CLASSES, CORE_CLASSES, load_manifest, result_dir
from metrics import cos, jaccard_topk


def load(model_key):
    ag_dir = result_dir(model_key) / "agop"
    manifest = {r["id"]: r for r in load_manifest() if r["role"] == "main"}
    recs = []
    for path in sorted(ag_dir.glob("*.npz")):
        m = manifest.get(path.stem)
        if m is None:
            continue
        z = np.load(path)
        layers, l = [], 0
        while f"grms_L{l}" in z:
            v = z[f"grms_L{l}"].astype(np.float32)
            layers.append(v / (np.linalg.norm(v) + 1e-12))
            l += 1
        recs.append({"id": path.stem, "class": m["class"], "half": m["half"],
                     "layers": layers})
    return recs


def group(recs, cls=None, half=None):
    return [r for r in recs if (cls is None or r["class"] == cls)
            and (half is None or r["half"] == half)]


def half_fps(recs, layer):
    return {(c, h): np.mean([r["layers"][layer] for r in group(recs, c, h)], axis=0)
            for c in CLASSES for h in "AB"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    recs = load(args.model)
    n_layers = len(recs[0]["layers"])
    print(f"{args.model} AGOP: {len(recs)} seqs, {n_layers} layers")

    per_layer = []
    for l in range(n_layers):
        fps = half_fps(recs, l)
        m = np.mean([fps[k] for k in fps], axis=0)
        within = [cos(fps[(c, "A")] - m, fps[(c, "B")] - m) for c in CLASSES]
        across = [cos(fps[(c1, h1)] - m, fps[(c2, h2)] - m)
                  for c1, c2 in itertools.combinations(CLASSES, 2)
                  for h1, h2 in itertools.product("AB", "AB")]
        j256_w = [jaccard_topk(fps[(c, "A")], fps[(c, "B")], 256) for c in CLASSES]
        j256_a = [jaccard_topk(fps[(c1, h1)], fps[(c2, h2)], 256)
                  for c1, c2 in itertools.combinations(CLASSES, 2)
                  for h1, h2 in itertools.product("AB", "AB")]
        row = {"within_ccos": float(np.mean(within)), "across_ccos": float(np.mean(across)),
               "margin_ccos": float(np.mean(within) - np.mean(across)),
               "margin_j256": float(np.mean(j256_w) - np.mean(j256_a))}
        per_layer.append(row)
        print(f"  L{l:2d} within={row['within_ccos']:.3f} across={row['across_ccos']:.3f} "
              f"margin={row['margin_ccos']:.3f} j256m={row['margin_j256']:.3f}", flush=True)

    # placement of contrast classes
    placement = {}
    for contrast, sibling in (("math_prose", "math"), ("code_prose", "code")):
        tp, ts, per_l = [], [], []
        for l in range(n_layers):
            fps = half_fps(recs, l)
            m = np.mean([fps[k] for k in fps], axis=0)
            p = np.mean([cos(fps[(contrast, h1)] - m, fps[("prose", h2)] - m)
                         for h1 in "AB" for h2 in "AB"])
            s = np.mean([cos(fps[(contrast, h1)] - m, fps[(sibling, h2)] - m)
                         for h1 in "AB" for h2 in "AB"])
            per_l.append((float(p), float(s)))
            if l >= n_layers // 4:
                tp.append(p); ts.append(s)
        placement[contrast] = {"sibling": sibling, "to_prose": float(np.mean(tp)),
                               "to_sibling": float(np.mean(ts)), "per_layer": per_l}
        print(f"  {contrast}: to_prose={np.mean(tp):.3f} to_{sibling}={np.mean(ts):.3f}")

    # per-seq leave-half-out nearest centroid, concatenated layers
    def dense(r):
        return np.concatenate(r["layers"])
    conf = {cl: np.zeros((len(cl_set), len(cl_set)), dtype=int)
            for cl, cl_set in (("5", CLASSES), ("3", CORE_CLASSES))}
    accs = {}
    for tag, cl_set in (("5", CLASSES), ("3", CORE_CLASSES)):
        C = conf[tag]
        for tr, te in (("A", "B"), ("B", "A")):
            cents = {c: np.mean([dense(r) for r in group(recs, c, tr)], axis=0) for c in cl_set}
            gm = np.mean(list(cents.values()), axis=0)
            for c in cl_set:
                v = cents[c] - gm
                cents[c] = v / (np.linalg.norm(v) + 1e-12)
            for r in [x for x in recs if x["half"] == te and x["class"] in cl_set]:
                x = dense(r) - gm
                x /= np.linalg.norm(x) + 1e-12
                C[cl_set.index(r["class"]), int(np.argmax([x @ cents[c] for c in cl_set]))] += 1
        accs[tag] = float(np.trace(C) / C.sum())
    print(f"per-seq AGOP classification: 5-class {accs['5']:.3f}, 3-class {accs['3']:.3f}")

    out = {"model": args.model, "n_seqs": len(recs), "per_layer": per_layer,
           "placement": placement,
           "classification": {"acc5": accs["5"], "conf5": conf["5"].tolist(),
                              "acc3": accs["3"], "classes": CLASSES}}
    path = result_dir(args.model) / "metrics_agop.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
