"""Extra E4 masks (same 20% budget):
  footprint   top math firing rate alone (no crowdedness) -- isolates the packing term
  fisher      per-neuron diagonal Fisher on task A at the after-A checkpoint
              (sum of squared param-grads over the neuron's gate row, up row, down col)
  join_code   crowdedness x CODE firing rate (for the reverse direction)
Also: reverse-direction training file (train_A2 = code, train_B2 = math reuse).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE / "e1-footprint-stability" / "src"))
sys.path.insert(0, str(BASE / "e2-welch-gain" / "src"))

MODEL_ID = "TinyLlama/TinyLlama_v1.1"
MODEL_KEY = "tinyllama-1.1b"
BUDGET = 0.20
DEV = "cuda"
FISHER_SEQS = 150
SEQ_LEN = 512


def class_rates(cls):
    from metrics import load_records, group, freq_vector
    recs = load_records(MODEL_KEY, 99.0)
    sub = group(recs, cls)
    n_layers = len(recs[0]["layers"])
    inter = 5632
    return [freq_vector(sub, l, inter) for l in range(n_layers)], n_layers, inter


def crowdedness():
    torch.set_grad_enabled(False)
    from extract import load_weights, extract_layers
    model = load_weights(MODEL_KEY)
    layers, _ = extract_layers(model, MODEL_KEY)
    out = []
    for L in layers:
        W = L["Wdown"].to(DEV)
        W = W / (W.norm(dim=0, keepdim=True) + 1e-12)
        G = (W.T @ W).abs()
        G.fill_diagonal_(0)
        out.append(G.max(1).values.cpu().numpy())
    del model
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)
    return out


def fisher_per_neuron():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(DEV)
    sd = torch.load(ROOT / "results" / "ckpt_A.pt", map_location=DEV)
    model.load_state_dict(sd, strict=False)
    for n, p in model.named_parameters():
        p.requires_grad_(".mlp." in n)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    with open(ROOT / "data" / "train_A_math.jsonl", encoding="utf-8") as f:
        texts = [json.loads(l)["text"] for l in f][:FISHER_SEQS]

    n_layers = model.config.num_hidden_layers
    inter = model.config.intermediate_size
    acc = [torch.zeros(inter, dtype=torch.float64, device=DEV) for _ in range(n_layers)]
    for i, t in enumerate(texts):
        ids = tok(t, return_tensors="pt", truncation=True, max_length=SEQ_LEN)["input_ids"].to(DEV)
        model.zero_grad(set_to_none=True)
        model(ids, labels=ids, use_cache=False).loss.backward()
        for l in range(n_layers):
            mlp = model.model.layers[l].mlp
            acc[l] += (mlp.down_proj.weight.grad.float() ** 2).sum(0).double()
            acc[l] += (mlp.gate_proj.weight.grad.float() ** 2).sum(1).double()
            acc[l] += (mlp.up_proj.weight.grad.float() ** 2).sum(1).double()
        if (i + 1) % 50 == 0:
            print(f"  fisher {i+1}/{len(texts)}", flush=True)
    model.zero_grad(set_to_none=True)
    res = [a.cpu().numpy() / len(texts) for a in acc]
    del model
    torch.cuda.empty_cache()
    return res


def topk_mask(score, k):
    m = np.zeros(len(score), bool)
    m[np.argsort(-score)[:k]] = True
    return m


def main():
    math_rate, n_layers, inter = class_rates("math")
    code_rate, _, _ = class_rates("code")
    k = int(BUDGET * inter)

    print("computing crowdedness...", flush=True)
    crowd = crowdedness()
    print("computing fisher...", flush=True)
    fish = fisher_per_neuron()

    masks = {"footprint": [], "fisher": [], "join_code": []}
    for l in range(n_layers):
        masks["footprint"].append(topk_mask(math_rate[l], k))
        masks["fisher"].append(topk_mask(fish[l], k))
        rc = crowd[l].argsort().argsort() / inter
        rr = code_rate[l].argsort().argsort() / inter
        masks["join_code"].append(topk_mask(rc * rr, k))
    for arm, ms in masks.items():
        np.savez(ROOT / "data" / f"mask_{arm}.npz", **{f"L{l}": m for l, m in enumerate(ms)})

    # overlap diagnostics vs the join mask
    zj = np.load(ROOT / "data" / "mask_join.npz")
    for arm in ("footprint", "fisher"):
        ov = np.mean([(masks[arm][l] & zj[f"L{l}"]).sum() / k for l in range(n_layers)])
        print(f"mask_{arm}: overlap with join = {ov:.2f}")
    print(f"masks written ({k}/{inter} per layer)")


if __name__ == "__main__":
    main()
