"""B1 stage 2: footprints over generated positions only (E1 sensor + E1
thresholds; activations at generated positions are identical under
re-forwarding for a causal LM), compared against E1 reading references.

Pre-registered criteria (from the handoff, restated before running):
  PASS: generated sequences classify to the correct regime >= 95% via nearest
  reading-centroid, AND each regime's generation centroid sits within 2x the
  reading noise floor distance of its reading centroid.
  FAIL -> quantify offset; test whether it is rigid (shared across regimes:
  cosine between the three offset vectors) or diffuse.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE / "e1-footprint-stability" / "src"))

from common import SKIP_TOKENS, load_manifest  # e1
from sensors import FootprintSensor
from metrics import load_records, group, freq_vector, cos, jaccard_topk

MODEL_IDS = {"qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B", "gemma-2-2b": "google/gemma-2-2b"}
REGIMES = ["math", "code", "prose"]
DEV = "cuda"
MAXLEN = 640


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_IDS))
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    od = ROOT / "results" / args.model

    gens = [json.loads(l) for l in open(od / "generations.jsonl", encoding="utf-8")]
    thr = np.load(BASE / "e1-footprint-stability" / "results" / args.model /
                  "thresholds.npz")["q99.0"]

    tok = AutoTokenizer.from_pretrained(MODEL_IDS[args.model])
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_IDS[args.model], dtype=torch.bfloat16).to(DEV).eval()
    sensor = FootprintSensor(model, tok, DEV)
    sensor.attach()
    sensor.thresholds = {99.0: thr}
    n_layers, inter = sensor.n_layers, sensor.inter

    # per-sequence generated-position footprints
    seqs = []
    for k, g in enumerate(gens):
        ids = tok(g["text"], return_tensors="pt", truncation=True,
                  max_length=MAXLEN).input_ids.to(DEV)
        start = max(g["prompt_len"], SKIP_TOKENS)
        if ids.shape[1] - start < 32:
            continue
        sensor._buf = {}
        model(ids, use_cache=False)
        vec = np.zeros(n_layers * inter, dtype=np.float32)
        n_tok = ids.shape[1] - start
        for l in range(n_layers):
            a = sensor._buf[l][0, start:].abs()
            c = (a > float(thr[l])).sum(0).float().cpu().numpy()
            vec[l * inter:(l + 1) * inter] = c / n_tok
        seqs.append({"regime": g["regime"], "mode": g["mode"], "vec": vec})
        if (k + 1) % 60 == 0:
            print(f"  {k+1}/{len(gens)} captured", flush=True)
    sensor.detach()
    print(f"{len(seqs)} generation footprints", flush=True)

    # E1 reading references
    recs = load_records(args.model, 99.0)
    read_cent = {}
    for c in REGIMES:
        v = np.concatenate([freq_vector(group(recs, c), l, inter)
                            for l in range(n_layers)]).astype(np.float32)
        read_cent[c] = v
    gm = np.mean(list(read_cent.values()), axis=0)

    # noise floor: E1 half-A vs half-B centroid distance per regime
    noise = {}
    for c in REGIMES:
        va = np.concatenate([freq_vector(group(recs, c, "A"), l, inter)
                             for l in range(n_layers)])
        vb = np.concatenate([freq_vector(group(recs, c, "B"), l, inter)
                             for l in range(n_layers)])
        noise[c] = 1 - cos(va - gm, vb - gm)

    # classification of individual generated sequences
    cents = {c: (read_cent[c] - gm) / (np.linalg.norm(read_cent[c] - gm) + 1e-12)
             for c in REGIMES}
    conf = {m: np.zeros((3, 3), dtype=int) for m in ("greedy", "t07")}
    for s in seqs:
        x = s["vec"] - gm
        x /= np.linalg.norm(x) + 1e-12
        pred = int(np.argmax([x @ cents[c] for c in REGIMES]))
        conf[s["mode"]][REGIMES.index(s["regime"]), pred] += 1
    acc = {m: float(np.trace(C) / max(C.sum(), 1)) for m, C in conf.items()}

    # centroid-level comparison + offset geometry
    res_regimes = {}
    offsets = {}
    for c in REGIMES:
        for m in ("greedy", "t07"):
            vs = [s["vec"] for s in seqs if s["regime"] == c and s["mode"] == m]
            if not vs:
                continue
            gcent = np.mean(vs, axis=0)
            d = 1 - cos(gcent - gm, read_cent[c] - gm)
            jac = jaccard_topk(gcent, read_cent[c], 256)
            res_regimes[f"{c}_{m}"] = {
                "dist_to_reading_centroid": float(d),
                "noise_floor": float(noise[c]),
                "ratio": float(d / max(noise[c], 1e-9)),
                "jaccard256_vs_reading": float(jac),
            }
            if m == "t07":
                offsets[c] = gcent - read_cent[c]
    off_cos = {f"{a}~{b}": float(cos(offsets[a], offsets[b]))
               for i, a in enumerate(REGIMES) for b in REGIMES[i + 1:]}

    crit_cls = min(acc.values()) >= 0.95
    crit_dist = all(v["ratio"] <= 2.0 for v in res_regimes.values())
    verdict = "PASS" if (crit_cls and crit_dist) else "FAIL"

    out = {"model": args.model, "n_seqs": len(seqs), "accuracy": acc,
           "confusion": {m: C.tolist() for m, C in conf.items()},
           "regimes": res_regimes, "offset_cosines_t07": off_cos,
           "verdict": verdict}
    with open(od / "b1_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
