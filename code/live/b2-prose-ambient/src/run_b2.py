"""B2: is "prose" one regime or diffuse ambient? Heterogeneous prose capture +
E1-style self-clustering, all four families.

Sources: wiki (reuse E1 prose captures — no recapture), fiction (TinyStories),
news (ag_news, packed). 60 packs per new source, E1 thresholds reused.

Pre-registered call (written before running): "prose fragments broadly" =
in >= 3/4 models the pooled prose sequences yield HDBSCAN >= 2 clusters with
the largest cluster holding < 50% of clustered sequences, OR noise >= 30%.
If sources instead separate cleanly (ARI vs source labels >= 0.5), prose is a
register mixture rather than ambient noise — report which.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE / "e1-footprint-stability" / "src"))

from common import MODELS, load_manifest  # e1
from sensors import FootprintSensor
from metrics import load_records

N_PACKS = 60
TARGET_WORDS = 280
DEV = "cuda"


def new_prose_sources():
    from datasets import load_dataset
    import random
    rng = random.Random(0)
    fiction = load_dataset("roneneldan/TinyStories", split="train[:4000]")
    fic_items = [r["text"].strip() for r in fiction if len(r["text"].split()) > 40]
    news = load_dataset("fancyzhx/ag_news", split="train[:8000]")
    news_items = [r["text"].strip() for r in news]

    def pack(items):
        rng.shuffle(items)
        packs, cur, w = [], [], 0
        for it in items:
            cur.append(it)
            w += len(it.split())
            if w >= TARGET_WORDS:
                packs.append("\n\n".join(cur))
                cur, w = [], 0
            if len(packs) >= N_PACKS:
                break
        return packs
    return {"fiction": pack(fic_items), "news": pack(news_items)}


def capture(model_key, sources):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    thr_dir = BASE / "e1-footprint-stability" / "results" / model_key
    od = ROOT / "results" / model_key
    od.mkdir(parents=True, exist_ok=True)
    if all((od / f"{s}_{i:03d}.npz").exists() for s in sources for i in
           range(len(sources[s]))):
        return
    tok = AutoTokenizer.from_pretrained(MODELS[model_key])
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[model_key], dtype=torch.bfloat16).to(DEV).eval()
    sensor = FootprintSensor(model, tok, DEV)
    sensor.attach()
    z = np.load(thr_dir / "thresholds.npz")
    sensor.thresholds = {q: z[f"q{q}"] for q in (98.0, 99.0, 99.5)}
    for s, texts in sources.items():
        for i, t in enumerate(texts):
            p = od / f"{s}_{i:03d}.npz"
            if p.exists():
                continue
            rec = sensor.capture(t)
            if rec is not None:
                np.savez_compressed(p, **rec)
    sensor.detach()
    del model
    torch.cuda.empty_cache()


def analyze(model_key):
    from scipy import sparse
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import HDBSCAN
    from sklearn.metrics import adjusted_rand_score

    od = ROOT / "results" / model_key
    recs = []
    # wiki: reuse E1 prose per-seq captures
    e1recs = load_records(model_key, 99.0)
    for r in e1recs:
        if r["class"] == "prose":
            recs.append({"src": "wiki", "layers": r["layers"], "n": r["n_tokens"]})
    for p in sorted(od.glob("*.npz")):
        src = p.stem.rsplit("_", 1)[0]
        z = np.load(p)
        layers, l = [], 0
        while f"idx_q99.0_L{l}" in z:
            layers.append((z[f"idx_q99.0_L{l}"], z[f"cnt_q99.0_L{l}"]))
            l += 1
        recs.append({"src": src, "layers": layers, "n": int(z["n_tokens"])})

    n_layers = len(recs[0]["layers"])
    dim = int(max(i.max() if len(i) else 0 for r in recs for i, _ in r["layers"])) + 1
    rows, cols, vals = [], [], []
    for k, r in enumerate(recs):
        for l, (idx, cnt) in enumerate(r["layers"]):
            rows.extend([k] * len(idx))
            cols.extend(l * dim + idx)
            vals.extend(cnt / r["n"])
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(recs), n_layers * dim))
    Z = TruncatedSVD(50, random_state=0).fit_transform(X)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    hdb = HDBSCAN(min_cluster_size=15).fit(Z)
    lab = hdb.labels_
    src = np.array([r["src"] for r in recs])
    mask = lab >= 0
    n_cl = len(set(lab[mask]))
    largest = max((lab[mask] == c).sum() for c in set(lab[mask])) / max(mask.sum(), 1) if n_cl else 0
    ari_src = adjusted_rand_score(src[mask], lab[mask]) if mask.sum() > 10 else 0
    comp = {int(c): {s: int(((lab == c) & (src == s)).sum()) for s in set(src)}
            for c in sorted(set(lab))}
    out = {"model": model_key, "n_seqs": len(recs), "n_clusters": n_cl,
           "noise_frac": float((~mask).mean()), "largest_cluster_share": float(largest),
           "ari_vs_source": float(ari_src), "composition": comp}
    with open(od / "b2_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"{model_key}: {n_cl} clusters, noise {out['noise_frac']:.0%}, "
          f"largest {largest:.0%}, ARI(src) {ari_src:+.2f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen2.5-1.5b,gemma-2-2b,pythia-1.4b,tinyllama-1.1b")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    sources = new_prose_sources()
    print({k: len(v) for k, v in sources.items()}, flush=True)
    for m in args.models.split(","):
        capture(m, sources)
        analyze(m)


if __name__ == "__main__":
    main()
