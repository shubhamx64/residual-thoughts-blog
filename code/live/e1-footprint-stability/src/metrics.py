"""Stability / separability metrics for captured footprints.

Primary quantities (per layer, at the primary threshold quantile):
  within-regime similarity   half-A vs half-B footprint of the same class
  across-regime similarity   footprints of different classes (all half pairs)
  margin                     mean(within) - mean(across)
  noise floor                random pack-level 50/50 resplits within class
  shuffle control            class labels permuted across sequences
Similarity measures: raw cosine, grand-mean-centered cosine (primary),
Jaccard on top-k neuron sets (k = 128/256/512).

Operational test: leave-half-out nearest-centroid classification of individual
sequences, vs the same classifier on token-unigram vectors (surface baseline).
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from common import (CLASSES, CORE_CLASSES, TOPK_LIST, load_manifest, result_dir, ROOT)

RNG = np.random.default_rng(0)
N_RESAMPLE = 20


def load_records(model_key, quantile):
    fp_dir = result_dir(model_key) / "footprints"
    manifest = {r["id"]: r for r in load_manifest() if r["role"] == "main"}
    recs = []
    for path in sorted(fp_dir.glob("*.npz")):
        m = manifest.get(path.stem)
        if m is None:
            continue
        z = np.load(path)
        layers = []
        l = 0
        while f"idx_q{quantile}_L{l}" in z:
            layers.append((z[f"idx_q{quantile}_L{l}"], z[f"cnt_q{quantile}_L{l}"]))
            l += 1
        recs.append({"id": path.stem, "class": m["class"], "half": m["half"],
                     "n_tokens": int(z["n_tokens"]), "layers": layers,
                     "tok": (z["tok_idx"], z["tok_cnt"]),
                     "pr_mean": z["pr_mean"]})
    return recs


def freq_vector(recs, layer, dim):
    v = np.zeros(dim, dtype=np.float64)
    tot = 0
    for r in recs:
        idx, cnt = r["layers"][layer]
        v[idx] += cnt
        tot += r["n_tokens"]
    return v / max(tot, 1)


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else 0.0


def jaccard_topk(u, v, k):
    a, b = set(np.argsort(u)[-k:]), set(np.argsort(v)[-k:])
    return len(a & b) / len(a | b)


def group(recs, cls=None, half=None):
    return [r for r in recs if (cls is None or r["class"] == cls)
            and (half is None or r["half"] == half)]


def half_footprints(recs, layer, dim, classes):
    """{(class, half): freq vector} for one layer."""
    return {(c, h): freq_vector(group(recs, c, h), layer, dim)
            for c in classes for h in ("A", "B")}


def layer_metrics(fps, classes):
    """Within/across for raw cos, centered cos, jaccard top-k."""
    m = np.mean([fps[k] for k in fps], axis=0)  # grand mean over half-footprints
    out = {}
    within = {"cos": [], "ccos": [], **{f"j{k}": [] for k in TOPK_LIST}}
    across = {"cos": [], "ccos": [], **{f"j{k}": [] for k in TOPK_LIST}}
    for c in classes:
        u, v = fps[(c, "A")], fps[(c, "B")]
        within["cos"].append(cos(u, v))
        within["ccos"].append(cos(u - m, v - m))
        for k in TOPK_LIST:
            within[f"j{k}"].append(jaccard_topk(u, v, k))
    for c1, c2 in itertools.combinations(classes, 2):
        for h1, h2 in itertools.product("AB", "AB"):
            u, v = fps[(c1, h1)], fps[(c2, h2)]
            across["cos"].append(cos(u, v))
            across["ccos"].append(cos(u - m, v - m))
            for k in TOPK_LIST:
                across[f"j{k}"].append(jaccard_topk(u, v, k))
    for name in within:
        out[f"within_{name}"] = float(np.mean(within[name]))
        out[f"across_{name}"] = float(np.mean(across[name]))
        out[f"margin_{name}"] = out[f"within_{name}"] - out[f"across_{name}"]
    return out


def noise_floor(recs, layer, dim, classes):
    """Random pack-level 50/50 resplits within class -> centered-cos sims."""
    sims = []
    fps_all = half_footprints(recs, layer, dim, classes)
    m = np.mean([fps_all[k] for k in fps_all], axis=0)
    for c in classes:
        rs = group(recs, c)
        for _ in range(N_RESAMPLE):
            perm = RNG.permutation(len(rs))
            a = [rs[i] for i in perm[: len(rs) // 2]]
            b = [rs[i] for i in perm[len(rs) // 2:]]
            sims.append(cos(freq_vector(a, layer, dim) - m, freq_vector(b, layer, dim) - m))
    return float(np.mean(sims)), float(np.std(sims))


def shuffle_control(recs, layer, dim, classes):
    """Permute class labels across sequences; margin should vanish."""
    margins = []
    labels = [r["class"] for r in recs]
    for _ in range(N_RESAMPLE):
        perm = RNG.permutation(len(labels))
        shuffled = [dict(r, **{"class": labels[j]}) for r, j in zip(recs, perm)]
        fps = half_footprints(shuffled, layer, dim, classes)
        m = layer_metrics(fps, classes)
        margins.append(m["margin_ccos"])
    return float(np.mean(margins)), float(np.std(margins))


def seq_dense(r, dims, use_tok=False, vocab=None):
    if use_tok:
        v = np.zeros(vocab, dtype=np.float32)
        idx, cnt = r["tok"]
        v[idx] = cnt
        return v / r["n_tokens"]
    n_layers, dim = dims
    v = np.zeros(n_layers * dim, dtype=np.float32)
    for l, (idx, cnt) in enumerate(r["layers"]):
        v[l * dim + idx] = cnt
    return v / r["n_tokens"]


def nearest_centroid(recs, classes, dims, use_tok=False, vocab=None):
    """Leave-half-out: train centroids on one half, classify the other."""
    conf = np.zeros((len(classes), len(classes)), dtype=int)
    for train_half, test_half in (("A", "B"), ("B", "A")):
        cents = {}
        for c in classes:
            vs = [seq_dense(r, dims, use_tok, vocab) for r in group(recs, c, train_half)]
            cents[c] = np.mean(vs, axis=0)
        m = np.mean(list(cents.values()), axis=0)
        for c in classes:
            cents[c] = cents[c] - m
            cents[c] /= np.linalg.norm(cents[c]) + 1e-12
        for r in [x for x in recs if x["half"] == test_half and x["class"] in classes]:
            x = seq_dense(r, dims, use_tok, vocab) - m
            x /= np.linalg.norm(x) + 1e-12
            scores = [x @ cents[c] for c in classes]
            conf[classes.index(r["class"]), int(np.argmax(scores))] += 1
    acc = float(np.trace(conf) / conf.sum())
    return acc, conf.tolist()


def per_layer_accuracy(recs, classes, dim, n_layers):
    accs = []
    for l in range(n_layers):
        sub = [dict(r, layers=[r["layers"][l]]) for r in recs]
        acc, _ = nearest_centroid(sub, classes, (1, dim))
        accs.append(acc)
    return accs


def _placement_from_fps(fps, contrast, sibling):
    m = np.mean([fps[k] for k in fps], axis=0)
    to_prose, to_sib = [], []
    for h1 in "AB":
        u = fps[(contrast, h1)] - m
        to_prose.append(np.mean([cos(u, fps[("prose", h2)] - m) for h2 in "AB"]))
        to_sib.append(np.mean([cos(u, fps[(sibling, h2)] - m) for h2 in "AB"]))
    return float(np.mean(to_prose)), float(np.mean(to_sib))


def contrast_placement(recs, dim, n_layers, vocab):
    """Where do the contrast classes sit? Centered-cos of math_prose /
    code_prose half-footprints to prose vs to their computation sibling,
    per layer, plus the token-unigram baseline: the placement the surface
    statistics alone predict. Regime evidence = footprint placement swings
    toward the sibling MORE than the token baseline does."""
    # token-space half-vectors
    tok_fps = {}
    for c in CLASSES:
        for h in "AB":
            v = np.zeros(vocab, dtype=np.float64)
            tot = 0
            for r in group(recs, c, h):
                idx, cnt = r["tok"]
                v[idx] += cnt
                tot += r["n_tokens"]
            tok_fps[(c, h)] = v / max(tot, 1)
    out = {}
    for contrast, sibling in (("math_prose", "math"), ("code_prose", "code")):
        per_layer = []
        for l in range(n_layers):
            fps = half_footprints(recs, l, dim, CLASSES)
            per_layer.append(_placement_from_fps(fps, contrast, sibling))
        tp, ts = _placement_from_fps(tok_fps, contrast, sibling)
        late = per_layer[n_layers // 4:]
        out[contrast] = {
            "sibling": sibling,
            "to_prose": float(np.mean([p for p, _ in late])),
            "to_sibling": float(np.mean([s for _, s in late])),
            "token_to_prose": tp, "token_to_sibling": ts,
            "per_layer_to_prose": [p for p, _ in per_layer],
            "per_layer_to_sibling": [s for _, s in per_layer],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantile", type=float, default=99.0)
    args = ap.parse_args()

    recs = load_records(args.model, args.quantile)
    n_layers = len(recs[0]["layers"])
    dim = int(max(idx.max() if len(idx) else 0 for r in recs for idx, _ in r["layers"])) + 1
    # round dim up to the true intermediate size stored implicitly; max idx is close enough
    vocab = int(max(r["tok"][0].max() for r in recs)) + 1
    counts = {c: len(group(recs, c)) for c in CLASSES}
    print(f"{args.model}: {len(recs)} seqs, {n_layers} layers, dim>={dim}, {counts}")

    per_layer = []
    for l in range(n_layers):
        fps = half_footprints(recs, l, dim, CLASSES)
        lm = layer_metrics(fps, CLASSES)
        nf_mean, nf_std = noise_floor(recs, l, dim, CLASSES)
        lm["noise_mean"], lm["noise_std"] = nf_mean, nf_std
        per_layer.append(lm)
        print(f"  L{l:2d} within_ccos={lm['within_ccos']:.3f} across_ccos={lm['across_ccos']:.3f} "
              f"margin={lm['margin_ccos']:.3f} noise={nf_mean:.3f}±{nf_std:.3f} "
              f"j256 margin={lm['margin_j256']:.3f}", flush=True)

    # shuffle control on a few representative layers (it is the slow one)
    rep_layers = sorted({0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1})
    shuffle = {str(l): shuffle_control(recs, l, dim, CLASSES) for l in rep_layers}
    print(f"shuffle margins (should be ~0): { {k: f'{v[0]:.4f}±{v[1]:.4f}' for k, v in shuffle.items()} }")

    acc5, conf5 = nearest_centroid(recs, CLASSES, (n_layers, dim))
    acc3, conf3 = nearest_centroid(recs, CORE_CLASSES, (n_layers, dim))
    tok_acc5, tok_conf5 = nearest_centroid(recs, CLASSES, None, use_tok=True, vocab=vocab)
    tok_acc3, _ = nearest_centroid(recs, CORE_CLASSES, None, use_tok=True, vocab=vocab)
    print(f"per-seq nearest-centroid: 5-class {acc5:.3f} (token baseline {tok_acc5:.3f}), "
          f"3-class {acc3:.3f} (token baseline {tok_acc3:.3f})")

    layer_accs = per_layer_accuracy(recs, CLASSES, dim, n_layers)
    placement = contrast_placement(recs, dim, n_layers, vocab)
    for c, p in placement.items():
        print(f"  {c}: footprint to_prose={p['to_prose']:.3f} to_{p['sibling']}={p['to_sibling']:.3f} | "
              f"token baseline to_prose={p['token_to_prose']:.3f} to_{p['sibling']}={p['token_to_sibling']:.3f}")

    # similarity heatmap data at representative layers (10 half-footprints)
    heatmaps = {}
    keys = [(c, h) for c in CLASSES for h in "AB"]
    for l in rep_layers:
        fps = half_footprints(recs, l, dim, CLASSES)
        m = np.mean([fps[k] for k in fps], axis=0)
        H = [[cos(fps[k1] - m, fps[k2] - m) for k2 in keys] for k1 in keys]
        heatmaps[str(l)] = H

    out = {
        "model": args.model, "quantile": args.quantile, "n_seqs": len(recs),
        "class_counts": counts, "n_layers": n_layers,
        "per_layer": per_layer, "shuffle": shuffle,
        "classification": {
            "acc5": acc5, "conf5": conf5, "acc3": acc3, "conf3": conf3,
            "token_acc5": tok_acc5, "token_conf5": tok_conf5, "token_acc3": tok_acc3,
            "classes": CLASSES, "per_layer_acc5": layer_accs,
        },
        "contrast_placement": placement,
        "heatmap_keys": [f"{c}/{h}" for c, h in keys],
        "heatmaps": heatmaps,
        "pr_by_class": {c: np.mean([r["pr_mean"] for r in group(recs, c)], axis=0).tolist()
                        for c in CLASSES},
    }
    path = result_dir(args.model) / f"metrics_q{args.quantile}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
