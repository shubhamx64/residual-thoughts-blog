"""E4 prep: training/eval splits (leakage-free vs E1) + protection masks.

Splits: build_manifest's seeded item-level halving is reproducible, so we
rebuild the half-A item pools and pack them into fresh, larger training sets.
Eval = the E1 half-B packs (never trained on).

Protection masks (20% of MLP neurons per layer, equal budget across arms):
  random   uniform per layer (seed 0)
  weights  top by crowdedness = per-neuron max |cos| to any other write column
  join     top by rank(crowdedness) * rank(math firing rate from E1 footprints)
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE / "e1-footprint-stability" / "src"))
sys.path.insert(0, str(BASE / "e2-welch-gain" / "src"))
sys.path.insert(0, str(BASE / "e3-sufficiency" / "src"))

MODEL_KEY = "tinyllama-1.1b"
BUDGET = 0.20
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def rebuild_pools():
    from datasets import load_dataset
    from build_manifest import strip_gsm8k_solution, pack_items
    rng = random.Random(SEED)
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    mbpp = load_dataset("google-research-datasets/mbpp", "full",
                        split="train+test+validation+prompt")
    he = load_dataset("openai/openai_humaneval", split="test")
    gsm_q = [r["question"].strip() for r in gsm]
    gsm_a = [strip_gsm8k_solution(r["answer"]) for r in gsm]
    mbpp_prompt = [r["text"].strip() for r in mbpp]
    mbpp_code = [r["code"].strip() for r in mbpp]
    he_code = [(r["prompt"] + r["canonical_solution"]).strip() for r in he]

    # replay build_manifest's class loop order & rng usage EXACTLY:
    # classes iterated in CLASSES order, each does one shuffle of indices, then
    # pack_items shuffles each half once.
    from common import CLASSES
    sources = {"math": (gsm_a, 300), "math_prose": (gsm_q, 300),
               "code": (mbpp_code + he_code, 190), "code_prose": (mbpp_prompt, 110),
               "prose": (None, None)}
    halves = {}
    import re as _re
    for cls in CLASSES:
        pool, tw = sources[cls]
        if cls == "prose":
            break  # prose comes after code_prose; we don't need it for training
        pool = [p for p in pool if p]
        idx = list(range(len(pool)))
        rng.shuffle(idx)
        half_a = [pool[i] for i in idx[: len(idx) // 2]]
        half_b = [pool[i] for i in idx[len(idx) // 2:]]
        # consume rng exactly as build_manifest did (two pack shuffles)
        docs_a = pack_items(half_a, tw, rng)
        pack_items(half_b, tw, rng)
        halves[cls] = {"A_items": half_a, "A_packs_e1": docs_a}
    return halves


def write_training_sets(halves):
    from build_manifest import pack_items
    rng = random.Random(123)  # fresh packing of the SAME half-A items
    train_a = pack_items(halves["math"]["A_items"], 300, rng)
    train_b = pack_items(halves["code"]["A_items"], 190, rng)
    (ROOT / "data").mkdir(exist_ok=True)
    for name, packs in (("train_A_math", train_a), ("train_B_code", train_b)):
        with open(ROOT / "data" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for t in packs:
                f.write(json.dumps({"text": t}) + "\n")
        print(f"{name}: {len(packs)} packs")


def write_eval_sets():
    from common import load_manifest, load_texts
    texts = load_texts()
    man = load_manifest()
    for cls, n in (("math", 40), ("code", 40), ("prose", 30)):
        seqs = [texts[r["id"]] for r in man
                if r["class"] == cls and r["half"] == "B" and r["role"] == "main"][:n]
        with open(ROOT / "data" / f"eval_{cls}.jsonl", "w", encoding="utf-8") as f:
            for t in seqs:
                f.write(json.dumps({"text": t}) + "\n")
        print(f"eval_{cls}: {len(seqs)} seqs")


def protection_masks():
    torch.set_grad_enabled(False)
    from extract import load_weights, extract_layers
    from common_e3 import pooled_rates
    from metrics import load_records, group, freq_vector

    model = load_weights(MODEL_KEY)
    layers, d = extract_layers(model, MODEL_KEY)
    inter = layers[0]["Wdown"].shape[1]
    n_layers = len(layers)

    # math-class firing rate per neuron from E1 footprints
    recs = load_records(MODEL_KEY, 99.0)
    math_recs = group(recs, "math")
    math_rate = [freq_vector(math_recs, l, inter) for l in range(n_layers)]

    masks = {"random": [], "weights": [], "join": []}
    rng = np.random.default_rng(SEED)
    k = int(BUDGET * inter)
    for l in range(n_layers):
        W = layers[l]["Wdown"].to(DEV)
        W = W / (W.norm(dim=0, keepdim=True) + 1e-12)
        G = (W.T @ W).abs()
        G.fill_diagonal_(0)
        crowd = G.max(1).values.cpu().numpy()

        m_rand = np.zeros(inter, bool)
        m_rand[rng.choice(inter, k, replace=False)] = True
        m_w = np.zeros(inter, bool)
        m_w[np.argsort(-crowd)[:k]] = True
        r_crowd = crowd.argsort().argsort() / inter
        r_math = math_rate[l].argsort().argsort() / inter
        m_j = np.zeros(inter, bool)
        m_j[np.argsort(-(r_crowd * r_math))[:k]] = True
        masks["random"].append(m_rand)
        masks["weights"].append(m_w)
        masks["join"].append(m_j)
        if l % 6 == 0:
            ov_wj = (m_w & m_j).sum() / k
            print(f"  L{l}: weights/join mask overlap {ov_wj:.2f}")

    for arm, ms in masks.items():
        np.savez(ROOT / "data" / f"mask_{arm}.npz", **{f"L{l}": m for l, m in enumerate(ms)})
    print(f"masks written (budget {BUDGET:.0%} = {k}/{inter} neurons/layer)")


if __name__ == "__main__":
    halves = rebuild_pools()
    # sanity: our rebuilt E1 half-A packs must match E1's actual half-A texts
    from common import load_manifest, load_texts
    texts = load_texts()
    man = load_manifest()
    e1_math_a = [texts[r["id"]] for r in man if r["class"] == "math" and r["half"] == "A"]
    rebuilt = halves["math"]["A_packs_e1"][: len(e1_math_a)]
    match = sum(a == b for a, b in zip(e1_math_a, rebuilt)) / max(len(e1_math_a), 1)
    print(f"E1 half-A reconstruction match: {match:.2%} (must be 100%)")
    assert match == 1.0, "seeded reconstruction failed; leakage risk -- aborting"
    write_training_sets(halves)
    write_eval_sets()
    protection_masks()
