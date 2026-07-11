"""B1 stage 1: sample continuations from held-out prompts, per regime.

Prompts come from the seeded half-B item pools (never in any training or
centroid data): GSM8K questions -> math-ish continuations, MBPP descriptions ->
code-ish, prose openings -> prose. Greedy + temp 0.7, 256 new tokens.
Output: results/<model>/generations.jsonl {regime, mode, prompt, text, prompt_len}.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE / "e1-footprint-stability" / "src"))

MODEL_IDS = {"qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B", "gemma-2-2b": "google/gemma-2-2b"}
N_PROMPTS = 64
NEW_TOKENS = 256
SEED = 0
DEV = "cuda"


def half_b_prompts():
    """Rebuild the seeded item split (verified exact in E4 prep) -> half-B items."""
    from datasets import load_dataset
    from build_manifest import strip_gsm8k_solution  # noqa: F401 (rng parity not needed: we reshuffle identically)
    from common import load_manifest, load_texts
    rng = random.Random(SEED)
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    mbpp = load_dataset("google-research-datasets/mbpp", "full",
                        split="train+test+validation+prompt")
    gsm_q = [r["question"].strip() for r in gsm]
    mbpp_prompt = [r["text"].strip() for r in mbpp]

    def half_b(pool):
        idx = list(range(len(pool)))
        rng.shuffle(idx)
        return [pool[i] for i in idx[len(idx) // 2:]]

    # replay class order: math(gsm_a), math_prose(gsm_q), code, code_prose -- we
    # only need gsm_q and mbpp_prompt halves; consume rng in the same sequence
    # as build_manifest? Not required for held-out-ness: any half works as long
    # as it matches E1's half-B. Simplest robust route: use E1's own half-B
    # pack texts and take their first item (packs join items with blank lines).
    man, tx = load_manifest(), load_texts()

    def firsts(cls):
        out = []
        for r in man:
            if r["class"] == cls and r["half"] == "B" and r["role"] == "main":
                first = tx[r["id"]].split("\n\n")[0].strip()
                if len(first.split()) >= 15:
                    out.append(first)
        return out

    math_p = firsts("math_prose")[:N_PROMPTS]
    code_p = firsts("code_prose")[:N_PROMPTS]
    prose_full = [tx[r["id"]] for r in man
                  if r["class"] == "prose" and r["half"] == "B" and r["role"] == "main"]
    prose_p = [" ".join(t.split()[:40]) for t in prose_full][:N_PROMPTS]
    return {"math": math_p, "code": code_p, "prose": prose_p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_IDS))
    args = ap.parse_args()
    torch.manual_seed(SEED)

    prompts = half_b_prompts()
    print({k: len(v) for k, v in prompts.items()}, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_IDS[args.model])
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_IDS[args.model], dtype=torch.bfloat16).to(DEV).eval()

    od = ROOT / "results" / args.model
    od.mkdir(parents=True, exist_ok=True)
    out_path = od / "generations.jsonl"
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {(r["regime"], r["mode"], r["i"]) for r in map(json.loads, f)}
    f = open(out_path, "a", encoding="utf-8")

    t0, n = time.time(), 0
    with torch.no_grad():
        for regime, ps in prompts.items():
            for mode in ("greedy", "t07"):
                for i, p in enumerate(ps):
                    if (regime, mode, i) in done:
                        continue
                    ids = tok(p, return_tensors="pt").input_ids.to(DEV)
                    gen = model.generate(
                        ids, max_new_tokens=NEW_TOKENS, use_cache=True,
                        do_sample=(mode == "t07"), temperature=0.7 if mode == "t07" else None,
                        top_p=0.95 if mode == "t07" else None,
                        pad_token_id=tok.eos_token_id)
                    text = tok.decode(gen[0], skip_special_tokens=True)
                    f.write(json.dumps({"regime": regime, "mode": mode, "i": i,
                                        "prompt_len": int(ids.shape[1]),
                                        "text": text}) + "\n")
                    f.flush()
                    n += 1
                    if n % 40 == 0:
                        print(f"  {n} generations ({n/(time.time()-t0):.2f}/s)", flush=True)
    f.close()
    print(f"done: {n} new generations, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
