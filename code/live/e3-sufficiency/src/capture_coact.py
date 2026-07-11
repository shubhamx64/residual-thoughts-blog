"""Activation stage of E3: joint per-token firing counts for the selected pairs,
over the E1 corpus, reusing E1's calibrated q99 thresholds.

Outputs results/<model>/coact_L{l}.npz: joint counts per pair (total and per
class), marginal counts per candidate neuron, total token count.
"""
import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common_e3 import E1_RESULTS, QUANTILE, result_dir
from extract import MODELS
from common import CLASSES, SKIP_TOKENS, MAX_TOKENS, load_manifest, load_texts  # e1 modules
from sensors import FootprintSensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = result_dir(args.model)

    thr = np.load(E1_RESULTS / args.model / "thresholds.npz")[f"q{QUANTILE}"]
    manifest = [r for r in load_manifest() if r["role"] == "main"]
    texts = load_texts()

    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[args.model], dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    sensor = FootprintSensor(model, tok, device)
    sensor.attach()

    n_layers = sensor.n_layers
    pair_layers = []
    for l in range(n_layers):
        p = out_dir / f"pairs_L{l}.npz"
        pair_layers.append(np.load(p) if p.exists() else None)

    state = {}
    for l, z in enumerate(pair_layers):
        if z is None:
            continue
        U, inv = np.unique(np.concatenate([z["pi"], z["pj"]]), return_inverse=True)
        state[l] = {
            "U": torch.tensor(U, device=device, dtype=torch.long),
            "ui": torch.tensor(inv[: len(z["pi"])], device=device, dtype=torch.long),
            "uj": torch.tensor(inv[len(z["pi"]):], device=device, dtype=torch.long),
            "joint": torch.zeros(len(CLASSES), len(z["pi"]), device=device, dtype=torch.float64),
            "marg": torch.zeros(len(CLASSES), len(U), device=device, dtype=torch.float64),
            "thr": float(thr[l]),
        }
    tokens_per_class = torch.zeros(len(CLASSES), dtype=torch.float64)

    t0, done = time.time(), 0
    for r in manifest:
        enc = tok(texts[r["id"]], return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
        ids = enc["input_ids"].to(device)
        if ids.shape[1] - SKIP_TOKENS < 32:
            continue
        ci = CLASSES.index(r["class"])
        sensor._buf = {}
        with torch.no_grad():
            model(ids, use_cache=False)
        tokens_per_class[ci] += ids.shape[1] - SKIP_TOKENS
        for l, st in state.items():
            a = sensor._buf[l][0, SKIP_TOKENS:].abs()
            b = (a[:, st["U"]] > st["thr"])                    # (T, |U|)
            st["marg"][ci] += b.sum(0).double()
            st["joint"][ci] += (b[:, st["ui"]] & b[:, st["uj"]]).sum(0).double()
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{len(manifest)} ({done/(time.time()-t0):.1f} seq/s)", flush=True)

    for l, st in state.items():
        np.savez(out_dir / f"coact_L{l}.npz",
                 joint=st["joint"].cpu().numpy(), marg=st["marg"].cpu().numpy(),
                 U=st["U"].cpu().numpy(), tokens_per_class=tokens_per_class.numpy())
    sensor.detach()
    print(f"done: {done} seqs, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
