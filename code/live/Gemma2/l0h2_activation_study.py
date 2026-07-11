"""Activation study of L0H2, the weight-space three-way nexus.

Composition map claims (all weight space):
  1. V-comp champion: L0H2 -> L25H7 (0.098) -- L25H7's OV reads L0H2's writes.
  2. K-comp L0H2 -> L25H6/H7/H1 amplifies with RoPE distance (0.06@d0 -> 0.10@d8+).

Activation tests on real prompts:
  A. Characterize L0H2's attention pattern (distance profile, entropy, top keys).
  B. Causal: zero L0H2's output slice; measure per-head disruption at L25
     (attention pattern shift + output change), vs ablating control heads.
     Prediction: ablating L0H2 disrupts L25 H7/H6/H1 more than other L25 heads,
     and more than ablating a random L0 control head does.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-2-2b"
HEAD_DIM = 256
N_HEADS = 8
TARGET = (0, 2)
CONTROLS = [(0, 0), (0, 5)]
READER_LAYER = 25

device = "cuda"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float32, attn_implementation="eager"
).to(device)
model.eval()

prompts = [r["content"] for r in csv.DictReader(open("validation_prompts.csv", encoding="utf-8"))][:20]
print(f"{len(prompts)} prompts loaded")

# ---------- hooks ----------
captured = {}

def make_oproj_capture(name):
    def hook(module, args):
        captured[name] = args[0].detach()  # [B, S, 2048] pre-o_proj, per-head concat
    return hook

def make_ablate(head):
    def hook(module, args):
        x = args[0].clone()
        x[..., head * HEAD_DIM:(head + 1) * HEAD_DIM] = 0
        return (x,)
    return hook

reader_oproj = model.model.layers[READER_LAYER].self_attn.o_proj
writer_oproj = {l: model.model.layers[l].self_attn.o_proj for l, _ in [TARGET] + CONTROLS}

def run(text, ablate=None):
    """Returns (L0 attn [H,S,S], L25 attn [H,S,S], L25 pre-o_proj [S, 2048], logits[-1])."""
    h_cap = reader_oproj.register_forward_pre_hook(make_oproj_capture("reader"))
    h_abl = None
    if ablate is not None:
        layer, head = ablate
        h_abl = writer_oproj[layer].register_forward_pre_hook(make_ablate(head))
    try:
        ids = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**ids, output_attentions=True)
    finally:
        h_cap.remove()
        if h_abl:
            h_abl.remove()
    return (out.attentions[0][0], out.attentions[READER_LAYER][0],
            captured["reader"][0], out.logits[0, -1].float())

# ---------- A: characterize L0H2 ----------
dist_mass = np.zeros(8)   # distance bins 0,1,2,3,4-7,8-15,16-31,32+
bin_edges = [0, 1, 2, 3, 4, 8, 16, 32]
entropies, top_keys = [], Counter()
for p in prompts:
    a0, _, _, _ = run(p)
    A = a0[TARGET[1]].float().cpu().numpy()  # [S, S]
    S = A.shape[0]
    for q in range(1, S):
        row = A[q, :q + 1]
        d = q - np.arange(q + 1)
        for bi in range(8):
            lo = bin_edges[bi]
            hi = bin_edges[bi + 1] if bi < 7 else 10 ** 9
            dist_mass[bi] += row[(d >= lo) & (d < hi)].sum()
        pr = row / (row.sum() + 1e-9)
        entropies.append(float(-(pr * np.log(pr + 1e-12)).sum()))
        k = int(np.argmax(row))
        ids = tok(p, return_tensors="pt").input_ids[0]
        top_keys[tok.decode([ids[k]])] += 1
dist_mass /= dist_mass.sum()

print("\n=== A. L0H2 attention pattern ===")
labels = ["d=0", "d=1", "d=2", "d=3", "4-7", "8-15", "16-31", "32+"]
for lbl, m in zip(labels, dist_mass):
    print(f"  {lbl:>6}: {m:.3f}")
print(f"  mean row entropy: {np.mean(entropies):.3f} nats (uniform over ~40 keys = 3.7)")
print(f"  top attended key tokens: {top_keys.most_common(12)}")

# ---------- B: causal disruption at L25 ----------
print(f"\n=== B. ablation -> L25 disruption (mean over {len(prompts)} prompts) ===")
results = {}
for abl in [TARGET] + CONTROLS:
    attn_shift = np.zeros(N_HEADS)   # mean |dA| per L25 head
    out_shift = np.zeros(N_HEADS)    # rel change of per-head pre-o_proj output
    kl_final = []
    far_shift = np.zeros(N_HEADS)    # attn shift restricted to keys >= 8 back
    for p in prompts:
        _, a25_b, o25_b, lg_b = run(p)
        _, a25_a, o25_a, lg_a = run(p, ablate=abl)
        S = a25_b.shape[-1]
        dA = (a25_a - a25_b).abs().float().cpu().numpy()       # [H, S, S]
        attn_shift += dA.mean(axis=(1, 2))
        q_idx, k_idx = np.tril_indices(S)
        far = (q_idx - k_idx) >= 8
        far_shift += dA[:, q_idx[far], k_idx[far]].mean(axis=1)
        for h in range(N_HEADS):
            sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
            num = (o25_a[:, sl] - o25_b[:, sl]).norm()
            out_shift[h] += float(num / (o25_b[:, sl].norm() + 1e-9))
        pb = torch.log_softmax(lg_b, -1)
        pa = torch.log_softmax(lg_a, -1)
        kl_final.append(float((pb.exp() * (pb - pa)).sum()))
    n = len(prompts)
    results[f"L{abl[0]}H{abl[1]}"] = {
        "l25_attn_shift_per_head": (attn_shift / n).tolist(),
        "l25_attn_shift_far_per_head": (far_shift / n).tolist(),
        "l25_out_relshift_per_head": (out_shift / n).tolist(),
        "final_kl": float(np.mean(kl_final)),
    }
    r = results[f"L{abl[0]}H{abl[1]}"]
    print(f"\nablate L{abl[0]}H{abl[1]}:  final-logit KL = {r['final_kl']:.4f}")
    print("  L25 head:      " + " ".join(f"H{h:>6}" for h in range(N_HEADS)))
    print("  out rel-shift: " + " ".join(f"{v:6.3f}" for v in r["l25_out_relshift_per_head"]))
    print("  attn shift:    " + " ".join(f"{v:6.4f}" for v in r["l25_attn_shift_per_head"]))
    print("  attn shift d8+:" + " ".join(f"{v:6.4f}" for v in r["l25_attn_shift_far_per_head"]))

out = Path("analysis_outputs/l0h2_study")
out.mkdir(parents=True, exist_ok=True)
json.dump({"distance_mass": dict(zip(labels, dist_mass.tolist())),
           "mean_entropy": float(np.mean(entropies)),
           "top_keys": top_keys.most_common(20),
           "ablation": results}, open(out / "l0h2_study.json", "w"), indent=1)
print(f"\nSaved: {out / 'l0h2_study.json'}")
