"""E4 Qwen2.5-1.5B protection masks (second-family replication, PREREG_QWEN.md).

Identical recipes to prep_e4.py / prep_extra.py, parameterised for Qwen and
written to data/qwen2.5-1.5b/ so the TinyLlama artifacts stay untouched.

  random     uniform per layer (seed 0)
  weights    top crowdedness = per-neuron max |cos| to any other Wdown column
  footprint  top math firing rate alone (E1), isolates the usage term
  join       top rank(crowdedness) x rank(math firing rate)
  join_code  top rank(crowdedness) x rank(code firing rate)  [reverse direction]
  fisher     top diagonal Fisher on task A at the after-A checkpoint  [--fisher only]

Two modes:
  (default)  build the five weight/footprint masks -- needs only weights + E1 records
  --fisher   build the fisher mask -- needs results/ckpt_A_qwen.pt (run after phase A)
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
sys.path.insert(0, str(BASE / "e2-welch-gain" / "src"))

MODEL_ID = "Qwen/Qwen2.5-1.5B"
MODEL_KEY = "qwen2.5-1.5b"
BUDGET = 0.20
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
INTER = 8960
N_LAYERS = 28
FISHER_SEQS = 150
SEQ_LEN = 512
OUT = ROOT / "data" / MODEL_KEY


def class_rate(cls):
    from metrics import load_records, group, freq_vector
    recs = load_records(MODEL_KEY, 99.0)
    sub = group(recs, cls)
    return [freq_vector(sub, l, INTER) for l in range(N_LAYERS)]


def crowdedness():
    torch.set_grad_enabled(False)
    from extract import load_weights, extract_layers
    model = load_weights(MODEL_KEY)
    layers, _ = extract_layers(model, MODEL_KEY)
    assert len(layers) == N_LAYERS and layers[0]["Wdown"].shape[1] == INTER, \
        f"dim mismatch: {len(layers)} layers, inter {layers[0]['Wdown'].shape[1]}"
    out = []
    for L in layers:
        W = L["Wdown"].to(DEV).float()
        W = W / (W.norm(dim=0, keepdim=True) + 1e-12)
        G = (W.T @ W).abs()
        G.fill_diagonal_(0)
        out.append(G.max(1).values.cpu().numpy())
    del model
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)
    return out


def topk_mask(score, k):
    m = np.zeros(len(score), bool)
    m[np.argsort(-score)[:k]] = True
    return m


def build_weight_masks():
    math_rate = class_rate("math")
    code_rate = class_rate("code")
    print("computing crowdedness...", flush=True)
    crowd = crowdedness()
    k = int(BUDGET * INTER)
    rng = np.random.default_rng(SEED)

    masks = {"random": [], "weights": [], "footprint": [], "join": [], "join_code": []}
    for l in range(N_LAYERS):
        m_rand = np.zeros(INTER, bool)
        m_rand[rng.choice(INTER, k, replace=False)] = True
        masks["random"].append(m_rand)
        masks["weights"].append(topk_mask(crowd[l], k))
        masks["footprint"].append(topk_mask(math_rate[l], k))
        rc = crowd[l].argsort().argsort() / INTER
        rm = math_rate[l].argsort().argsort() / INTER
        rk = code_rate[l].argsort().argsort() / INTER
        masks["join"].append(topk_mask(rc * rm, k))
        masks["join_code"].append(topk_mask(rc * rk, k))
        if l % 7 == 0:
            ov = (masks["weights"][l] & masks["join"][l]).sum() / k
            print(f"  L{l}: weights/join overlap {ov:.2f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    for arm, ms in masks.items():
        np.savez(OUT / f"mask_{arm}.npz", **{f"L{l}": m for l, m in enumerate(ms)})
    print(f"weight masks written to {OUT} (budget {BUDGET:.0%} = {k}/{INTER}/layer)", flush=True)


def build_fisher_mask():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ckpt = ROOT / "results" / "ckpt_A_qwen.pt"
    assert ckpt.exists(), f"need {ckpt}; run phase A first"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(DEV)
    sd = torch.load(ckpt, map_location=DEV)
    model.load_state_dict(sd, strict=False)
    for n, p in model.named_parameters():
        p.requires_grad_(".mlp." in n)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    with open(ROOT / "data" / "train_A_math.jsonl", encoding="utf-8") as f:
        texts = [json.loads(l)["text"] for l in f][:FISHER_SEQS]

    acc = [torch.zeros(INTER, dtype=torch.float64, device=DEV) for _ in range(N_LAYERS)]
    for i, t in enumerate(texts):
        ids = tok(t, return_tensors="pt", truncation=True, max_length=SEQ_LEN)["input_ids"].to(DEV)
        model.zero_grad(set_to_none=True)
        model(ids, labels=ids, use_cache=False).loss.backward()
        for l in range(N_LAYERS):
            mlp = model.model.layers[l].mlp
            acc[l] += (mlp.down_proj.weight.grad.float() ** 2).sum(0).double()
            acc[l] += (mlp.gate_proj.weight.grad.float() ** 2).sum(1).double()
            acc[l] += (mlp.up_proj.weight.grad.float() ** 2).sum(1).double()
        if (i + 1) % 50 == 0:
            print(f"  fisher {i+1}/{len(texts)}", flush=True)
    model.zero_grad(set_to_none=True)
    k = int(BUDGET * INTER)
    ms = [topk_mask(a.cpu().numpy() / len(texts), k) for a in acc]
    zj = np.load(OUT / "mask_join.npz")
    ov = np.mean([(ms[l] & zj[f"L{l}"]).sum() / k for l in range(N_LAYERS)])
    np.savez(OUT / "mask_fisher.npz", **{f"L{l}": m for l, m in enumerate(ms)})
    print(f"fisher mask written (overlap with join {ov:.2f})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fisher", action="store_true", help="build fisher mask (needs ckpt_A_qwen)")
    args = ap.parse_args()
    if args.fisher:
        build_fisher_mask()
    else:
        build_weight_masks()
