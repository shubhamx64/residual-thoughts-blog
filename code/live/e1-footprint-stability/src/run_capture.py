"""Per-model capture: calibrate thresholds, then stream footprints to disk.

Usage:
  python run_capture.py --model qwen2.5-1.5b [--limit-per-class 20]

Checkpointing: existing per-sequence .npz files are skipped, so a crashed run
resumes where it left off. Thresholds are computed once and persisted.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (MODELS, CLASSES, THRESH_QUANTILES, load_manifest, load_texts,
                    result_dir, set_seed, log_versions)
from sensors import FootprintSensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--limit-per-class", type=int, default=None)
    args = ap.parse_args()

    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    out_dir = result_dir(args.model)
    fp_dir = out_dir / "footprints"
    log_versions(out_dir / "versions.json")

    manifest = load_manifest()
    texts = load_texts()

    print(f"loading {MODELS[args.model]} ({dtype}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    model = AutoModelForCausalLM.from_pretrained(MODELS[args.model], torch_dtype=dtype).to(device)
    model.eval()

    sensor = FootprintSensor(model, tok, device)
    sensor.attach()
    print(f"{sensor.n_layers} layers, intermediate={sensor.inter}", flush=True)

    thr_path = out_dir / "thresholds.npz"
    calib = [r for r in manifest if r["role"] == "calib"]
    if thr_path.exists():
        z = np.load(thr_path)
        sensor.thresholds = {q: z[f"q{q}"] for q in THRESH_QUANTILES}
        print("loaded existing thresholds", flush=True)
    else:
        t0 = time.time()
        sensor.calibrate([texts[r["id"]] for r in calib])
        np.savez(thr_path, **{f"q{q}": v for q, v in sensor.thresholds.items()})
        print(f"calibrated on {len(calib)} seqs in {time.time()-t0:.1f}s", flush=True)
    for q in THRESH_QUANTILES:
        t = sensor.thresholds[q]
        print(f"  q{q}: min {t.min():.3f} med {np.median(t):.3f} max {t.max():.3f}", flush=True)

    main_seqs = [r for r in manifest if r["role"] == "main"]
    if args.limit_per_class:
        kept, per = [], {}
        for r in main_seqs:
            key = (r["class"], r["half"])
            if per.get(key, 0) < args.limit_per_class // 2:
                kept.append(r)
                per[key] = per.get(key, 0) + 1
        main_seqs = kept

    t0, done, skipped = time.time(), 0, 0
    for r in main_seqs:
        path = fp_dir / f"{r['id']}.npz"
        if path.exists():
            skipped += 1
            continue
        rec = sensor.capture(texts[r["id"]])
        if rec is None:
            print(f"  {r['id']}: too short, dropped", flush=True)
            continue
        np.savez_compressed(path, **rec)
        done += 1
        if done % 100 == 0:
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(main_seqs)} ({rate:.1f} seq/s)", flush=True)

    sensor.detach()
    print(f"done: {done} captured, {skipped} already present, "
          f"{time.time()-t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
