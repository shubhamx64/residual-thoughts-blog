"""Per-neuron allocation maps + opponent/parallel pair sets for E-Q.

Maps (higher score = more sensitive = deserves more bits):
  reader     ||down col|| x total downstream gate-read strength (weights only)
  footprint  pooled firing rate over all E1 classes (forward passes, no labels)
  fisher     per-neuron sum of squared param-grads on a mixed corpus (backward)
Pair sets: all MLP-neuron pairs per layer with |signed cos| >= 0.6 split into
opponent (cos <= -0.6) and parallel (cos >= +0.6); greedy neuron-disjoint.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE / "e1-footprint-stability" / "src"))
sys.path.insert(0, str(BASE / "e2-welch-gain" / "src"))
sys.path.insert(0, str(BASE / "e3-sufficiency" / "src"))

DEV = "cuda"
PAIR_THRESH = 0.6
MAX_PAIRS_PER_LAYER = 40


def out_dir(model_key):
    d = ROOT / "results" / model_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def reader_map_and_pairs(model_key):
    torch.set_grad_enabled(False)
    from extract import load_weights, extract_layers
    model = load_weights(model_key)
    layers, d = extract_layers(model, model_key)
    n_layers = len(layers)
    reader, pairs_opp, pairs_par = [], [], []
    for l in range(n_layers):
        Wd = layers[l]["Wdown"].to(DEV)
        norms = Wd.norm(dim=0)
        Wn = Wd / (norms + 1e-12)
        # downstream gate-read strength of each unit write direction
        R = torch.zeros(Wd.shape[1], device=DEV)
        for lp in range(l + 1, n_layers):
            R += (layers[lp]["Wgate"].to(DEV) @ Wn).norm(dim=0)
        reader.append((norms * R).cpu().numpy())
        # signed near-duplicate pairs, neuron-disjoint greedy
        G = Wn.T @ Wn
        G.fill_diagonal_(0)
        for sign, store in ((-1, pairs_opp), (+1, pairs_par)):
            M = G * sign
            used = set()
            vals, idx = torch.sort(M.flatten(), descending=True)
            cnt = 0
            for v, k in zip(vals[:200000].tolist(), idx[:200000].tolist()):
                if v < PAIR_THRESH or cnt >= MAX_PAIRS_PER_LAYER:
                    break
                i, j = k // G.shape[0], k % G.shape[0]
                if i >= j or i in used or j in used:
                    continue
                used.update((i, j))
                store.append((l, i, j, sign * v))
                cnt += 1
    del model
    torch.cuda.empty_cache()
    return reader, pairs_opp, pairs_par, n_layers


def footprint_map(model_key, n_layers, inter):
    from common_e3 import pooled_rates
    rates, _ = pooled_rates(model_key, n_layers, inter)
    return rates


def fisher_map(model_id, n_seqs=120, seq_len=512):
    """Per-neuron Fisher on a mixed corpus (math+code train packs + prose half-A)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from common import load_manifest, load_texts
    texts = []
    for f in ("train_A_math.jsonl", "train_B_code.jsonl"):
        p = BASE / "e4-continual" / "data" / f
        with open(p, encoding="utf-8") as fh:
            texts += [json.loads(l)["text"] for l in fh][: n_seqs // 3]
    man, tx = load_manifest(), load_texts()
    texts += [tx[r["id"]] for r in man
              if r["class"] == "prose" and r["half"] == "A" and r["role"] == "main"][: n_seqs // 3]

    torch.set_grad_enabled(True)  # reader stage disables grads globally
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(DEV)
    for n, p in model.named_parameters():
        p.requires_grad_(".mlp." in n)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    n_layers = model.config.num_hidden_layers
    inter = model.config.intermediate_size
    acc = [torch.zeros(inter, dtype=torch.float64, device=DEV) for _ in range(n_layers)]
    for i, t in enumerate(texts):
        ids = tok(t, return_tensors="pt", truncation=True, max_length=seq_len)["input_ids"].to(DEV)
        model.zero_grad(set_to_none=True)
        model(ids, labels=ids, use_cache=False).loss.backward()
        for l in range(n_layers):
            mlp = model.model.layers[l].mlp
            acc[l] += (mlp.down_proj.weight.grad.float() ** 2).sum(0).double()
            acc[l] += (mlp.gate_proj.weight.grad.float() ** 2).sum(1).double()
            acc[l] += (mlp.up_proj.weight.grad.float() ** 2).sum(1).double()
        if (i + 1) % 40 == 0:
            print(f"  fisher {i+1}/{len(texts)}", flush=True)
    model.zero_grad(set_to_none=True)
    res = [a.cpu().numpy() / len(texts) for a in acc]
    del model
    torch.cuda.empty_cache()
    return res


def build(model_key, model_id):
    print("reader map + pairs...", flush=True)
    reader, opp, par, n_layers = reader_map_and_pairs(model_key)
    inter = len(reader[0])
    print(f"  {n_layers} layers, {len(opp)} opponent pairs, {len(par)} parallel pairs", flush=True)
    fp = footprint_map(model_key, n_layers, inter)
    print("fisher map...", flush=True)
    fi = fisher_map(model_id)
    od = out_dir(model_key)
    np.savez(od / "maps.npz",
             **{f"reader_L{l}": reader[l] for l in range(n_layers)},
             **{f"footprint_L{l}": fp[l] for l in range(n_layers)},
             **{f"fisher_L{l}": fi[l] for l in range(n_layers)})
    np.save(od / "pairs_opp.npy", np.array(opp, dtype=np.float64))
    np.save(od / "pairs_par.npy", np.array(par, dtype=np.float64))
    print(f"maps written to {od}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    ids = {"tinyllama-1.1b": "TinyLlama/TinyLlama_v1.1",
           "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B"}
    build(args.model, ids[args.model])
