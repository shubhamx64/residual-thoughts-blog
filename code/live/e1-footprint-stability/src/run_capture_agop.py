"""AGOP-variant sensor: per-neuron RMS gradient of sequence NLL w.r.t. the MLP
post-activation hidden state (same hook point as the firing footprint).

This is the diagonal of the Average Gradient Outer Product in the neuron frame,
per sequence -- an average-case sensitivity footprint (which neurons the loss
actually leans on) vs the firing footprint (which neurons are active). NFA
motivation: training imprints second-order usage statistics into weights; here
we test whether the *sensitivity* fingerprint is more stable/separable than the
*activity* fingerprint.

Params are frozen (requires_grad=False); gradient flows from a leaf forced at
the embedding output, so no weight grads are allocated.

Usage: python run_capture_agop.py --model qwen2.5-1.5b [--limit-per-class N]
"""
import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (MODELS, find_mlp_down_projs, load_manifest, load_texts,
                    result_dir, set_seed, SKIP_TOKENS, MAX_TOKENS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--limit-per-class", type=int, default=None)
    args = ap.parse_args()

    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    out_dir = result_dir(args.model)
    ag_dir = out_dir / "agop"
    ag_dir.mkdir(exist_ok=True)

    manifest = [r for r in load_manifest() if r["role"] == "main"]
    texts = load_texts()
    if args.limit_per_class:
        kept, per = [], {}
        for r in manifest:
            key = (r["class"], r["half"])
            if per.get(key, 0) < args.limit_per_class // 2:
                kept.append(r)
                per[key] = per.get(key, 0) + 1
        manifest = kept

    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    model = AutoModelForCausalLM.from_pretrained(MODELS[args.model], dtype=dtype).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # force a leaf at the embedding output so activation grads exist without weight grads
    def leaf_hook(mod, inp, out):
        out.requires_grad_(True)
        return out
    model.get_input_embeddings().register_forward_hook(leaf_hook)

    down_projs = find_mlp_down_projs(model)
    grads = {}

    def bwd_hook(layer_idx):
        def fn(module, grad_input, grad_output):
            grads[layer_idx] = grad_input[0].detach()
        return fn
    for i, mod in down_projs:
        mod.register_full_backward_hook(bwd_hook(i))

    t0, done = time.time(), 0
    for r in manifest:
        path = ag_dir / f"{r['id']}.npz"
        if path.exists():
            continue
        enc = tok(texts[r["id"]], return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
        ids = enc["input_ids"].to(device)
        if ids.shape[1] - SKIP_TOKENS < 32:
            continue
        grads.clear()
        out = model(ids, labels=ids, use_cache=False)
        out.loss.backward()
        rec = {"n_tokens": int(ids.shape[1] - SKIP_TOKENS), "loss": float(out.loss)}
        for l, _ in down_projs:
            g = grads[l][0, SKIP_TOKENS:].float()
            rec[f"grms_L{l}"] = ((g * g).mean(0)).sqrt().half().cpu().numpy()
        np.savez_compressed(path, **rec)
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(manifest)} ({done/(time.time()-t0):.1f} seq/s)", flush=True)
    print(f"done: {done} captured, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
