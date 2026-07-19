"""Mechanism test (PREREG_ROBUSTNESS.md Part C): is crowding where task gradients conflict?

Per MLP neuron, the signed task gradients g^A (math) and g^B (code) over its gate
row + up row + down column; conflict = -cos(g^A, g^B) (high when the two task
updates oppose). Then Spearman(crowding, conflict) per layer. If crowded neurons are
specifically where cross-task gradients fight, the capacity-contention reading of E4
is directly supported.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda"
SEQS = 128
SEQ_LEN = 512
MODELS = {
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B", "plain"),
    "tinyllama-1.1b": ("TinyLlama/TinyLlama_v1.1", "plain"),
    "gemma-2-2b": ("google/gemma-2-2b", "gemma"),
}


def load(path, n):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l)["text"] for l in f][:n]


def task_grads(model, tok, texts):
    """Accumulate signed MLP gradients over a corpus -> per-layer (gate,up,down) sums."""
    model.zero_grad(set_to_none=True)
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=SEQ_LEN)["input_ids"].to(DEV)
        model(ids, labels=ids, use_cache=False).loss.backward()
    out = []
    for l in range(model.config.num_hidden_layers):
        mlp = model.model.layers[l].mlp
        out.append((mlp.gate_proj.weight.grad.detach().float().cpu(),
                    mlp.up_proj.weight.grad.detach().float().cpu(),
                    mlp.down_proj.weight.grad.detach().float().cpu()))
    model.zero_grad(set_to_none=True)
    return out


def crowding_layer(down_w, gemma_g=None):
    W = down_w.float()
    if gemma_g is not None:
        W = (1.0 + gemma_g.float())[:, None] * W
    W = W.to(DEV)
    W = W / (W.norm(dim=0, keepdim=True) + 1e-12)
    G = (W.T @ W).abs()
    G.fill_diagonal_(0)
    return G.max(1).values.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    args = ap.parse_args()
    hf, arch = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16).to(DEV)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    for n, p in model.named_parameters():
        p.requires_grad_(".mlp." in n)

    math = load(ROOT / "data" / "train_A_math.jsonl", SEQS)
    code = load(ROOT / "data" / "train_B_code.jsonl", SEQS)
    print("grad^A (math)...", flush=True)
    gA = task_grads(model, tok, math)
    print("grad^B (code)...", flush=True)
    gB = task_grads(model, tok, code)

    nL = model.config.num_hidden_layers
    rc, rmag, rcf = [], [], []   # crowd-conflict, crowd-magnitude, conflict-fisher
    for l in range(nL):
        aG, aU, aD = [t.to(DEV) for t in gA[l]]
        bG, bU, bD = [t.to(DEV) for t in gB[l]]
        num = (aG * bG).sum(1) + (aU * bU).sum(1) + (aD * bD).sum(0)     # per neuron
        nA = torch.sqrt((aG**2).sum(1) + (aU**2).sum(1) + (aD**2).sum(0) + 1e-20)
        nB = torch.sqrt((bG**2).sum(1) + (bU**2).sum(1) + (bD**2).sum(0) + 1e-20)
        cos = (num / (nA * nB)).cpu().numpy()
        conflict = -cos
        fisher = (nA**2).cpu().numpy()   # diag Fisher on math (matches E4/escale)
        gg = model.model.layers[l].mlp.down_proj.weight.detach()
        gm = model.model.layers[l].post_feedforward_layernorm.weight.detach() if arch == "gemma" else None
        crowd = crowding_layer(gg, gm)
        rc.append(stats.spearmanr(crowd, conflict).statistic)
        rmag.append(stats.spearmanr(crowd, nA.cpu().numpy()).statistic)
        rcf.append(stats.spearmanr(conflict, fisher).statistic)

    res = {"model": args.model,
           "crowd_vs_conflict_rho_median": float(np.median(rc)),
           "crowd_vs_gradmag_rho_median": float(np.median(rmag)),
           "conflict_vs_fisher_rho_median": float(np.median(rcf)),
           "crowd_vs_conflict_per_layer": [float(x) for x in rc]}
    (ROOT / "results" / f"gradconflict_{args.model}.json").write_text(json.dumps(res, indent=1))
    print(f"\n== {args.model} ==")
    print(f"  crowd vs conflict   rho_med {res['crowd_vs_conflict_rho_median']:+.3f}  "
          f"(>=+0.15 => contention; |.|<0.10 => not gradient conflict)")
    print(f"  crowd vs |grad|     rho_med {res['crowd_vs_gradmag_rho_median']:+.3f}  (control)")
    print(f"  conflict vs fisher  rho_med {res['conflict_vs_fisher_rho_median']:+.3f}  (control)")


if __name__ == "__main__":
    main()
