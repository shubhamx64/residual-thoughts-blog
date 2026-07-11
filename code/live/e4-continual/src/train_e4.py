"""E4 trainer: MLP-only fine-tune of TinyLlama-1.1B with optional per-neuron
protection masks applied to gradients during phase B.

Phases:
  A: fine-tune on math (no protection, all arms share this checkpoint)
  B: fine-tune on code with arm in {baseline, random, weights, join}

Protection mechanism: for protected neuron n at layer l, zero the gradient of
down_proj[:, n], gate_proj[n, :], up_proj[n, :] before each optimizer step.
Equal budget (20%/layer) across protected arms.

Eval every EVAL_EVERY steps: held-out ppl on math/code/prose + math-footprint
drift (1 - cos of pooled q99 firing-frequency vector vs the after-A reference).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE / "e1-footprint-stability" / "src"))

MODELS = {
    "tinyllama-1.1b": "TinyLlama/TinyLlama_v1.1",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
}
MODEL_ID = "TinyLlama/TinyLlama_v1.1"
MODEL_KEY = "tinyllama-1.1b"
SEQ_LEN = 512
ACCUM = 8
LR = 2e-5
STEPS = {"A": 500, "B": 500}
EVAL_EVERY = 100
SEED = 0
DEV = "cuda"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l)["text"] for l in f]


def batches(texts, tok, steps, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(texts)).tolist()
    i = 0
    for _ in range(steps * ACCUM):
        if i >= len(order):
            order = rng.permutation(len(texts)).tolist()
            i = 0
        t = texts[order[i]]
        i += 1
        enc = tok(t, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
        yield enc["input_ids"].to(DEV)


@torch.no_grad()
def eval_ppl(model, tok, texts):
    model.eval()
    nll, n = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=SEQ_LEN)["input_ids"].to(DEV)
        out = model(ids, labels=ids, use_cache=False)
        n_tok = ids.shape[1] - 1
        nll += float(out.loss) * n_tok
        n += n_tok
    model.train()
    return math.exp(nll / n)


@torch.no_grad()
def math_footprint(model, tok, texts, thresholds):
    """Pooled q99 firing-frequency vector per layer on the math probe set."""
    from sensors import FootprintSensor
    model.eval()
    sensor = FootprintSensor(model, tok, DEV)
    sensor.attach()
    n_layers = sensor.n_layers
    inter = sensor.inter
    freq = [np.zeros(inter) for _ in range(n_layers)]
    tot = 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=SEQ_LEN)["input_ids"].to(DEV)
        sensor._buf = {}
        model(ids, use_cache=False)
        tot += ids.shape[1] - 5
        for l in range(n_layers):
            a = sensor._buf[l][0, 5:].abs()
            freq[l] += (a > float(thresholds[l])).sum(0).float().cpu().numpy()
    sensor.detach()
    model.train()
    return np.concatenate([f / tot for f in freq])


def drift(cur, ref):
    return 1 - float(cur @ ref / (np.linalg.norm(cur) * np.linalg.norm(ref) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["A", "B"])
    ap.add_argument("--arm", default="baseline")
    ap.add_argument("--init", default=None, help="checkpoint .pt of MLP params to load")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--train-file", default=None, help="override training jsonl")
    ap.add_argument("--probe-class", default="math", help="footprint-drift probe eval class")
    ap.add_argument("--mask-file", default=None, help="override protection mask npz")
    ap.add_argument("--model", default=MODEL_KEY, choices=list(MODELS),
                    help="model key (default preserves TinyLlama)")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="seed for phase-B data order (default 0 preserves original)")
    args = ap.parse_args()
    if args.steps:
        STEPS[args.phase] = args.steps
    seed = args.seed
    torch.manual_seed(seed)

    model_key = args.model
    model_id = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(DEV)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # needed: embeddings frozen + checkpointing

    mlp_params = {}
    for name, p in model.named_parameters():
        if ".mlp." in name:
            p.requires_grad_(True)
            mlp_params[name] = p
        else:
            p.requires_grad_(False)
    print(f"trainable: {sum(p.numel() for p in mlp_params.values())/1e6:.0f}M params", flush=True)

    if args.init:
        sd = torch.load(args.init, map_location=DEV)
        missing = model.load_state_dict(sd, strict=False)
        print(f"loaded init {args.init} (missing keys ok: {len(missing.missing_keys)})", flush=True)

    masks = None
    if args.phase == "B" and args.arm != "baseline":
        mask_path = args.mask_file or (ROOT / "data" / f"mask_{args.arm}.npz")
        z = np.load(mask_path)
        masks = {int(k[1:]): torch.tensor(z[k], device=DEV) for k in z.files}
        print(f"arm {args.arm}: protecting {int(masks[0].sum())} neurons/layer "
              f"({mask_path})", flush=True)

    train_file = args.train_file or (
        "train_A_math.jsonl" if args.phase == "A" else "train_B_code.jsonl")
    train_texts = load_jsonl(ROOT / "data" / train_file)
    evals = {c: load_jsonl(ROOT / "data" / f"eval_{c}.jsonl") for c in ("math", "code", "prose")}
    thresholds = np.load(BASE / "e1-footprint-stability" / "results" / model_key /
                         "thresholds.npz")["q99.0"]

    opt = torch.optim.AdamW([p for p in mlp_params.values()], lr=LR, weight_decay=0.0)
    steps = STEPS[args.phase]
    tag = (f"{args.phase}_{args.arm}" if args.phase == "B" else "A") + args.tag_suffix
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    log_path = out_dir / f"log_{tag}.jsonl"
    log = open(log_path, "w", encoding="utf-8")

    probe_texts = evals[args.probe_class]
    ref_fp = None
    if args.phase == "B":
        ref_fp = math_footprint(model, tok, probe_texts, thresholds)

    def evaluate(step):
        row = {"step": step,
               "ppl_math": eval_ppl(model, tok, evals["math"]),
               "ppl_code": eval_ppl(model, tok, evals["code"]),
               "ppl_prose": eval_ppl(model, tok, evals["prose"])}
        if ref_fp is not None:
            row["fp_drift"] = drift(math_footprint(model, tok, probe_texts, thresholds), ref_fp)
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(f"  step {step}: " + " ".join(f"{k}={v:.4g}" for k, v in row.items() if k != "step"),
              flush=True)
        return row

    model.train()
    evaluate(0)
    t0 = time.time()
    gen = batches(train_texts, tok, steps, seed=seed + (0 if args.phase == "A" else 1))
    step = 0
    opt.zero_grad(set_to_none=True)
    for micro, ids in enumerate(gen, 1):
        out = model(ids, labels=ids, use_cache=False)
        (out.loss / ACCUM).backward()
        if micro % ACCUM == 0:
            step += 1
            if masks is not None:
                for l, m in masks.items():
                    pref = f"model.layers.{l}.mlp."
                    mlp_params[pref + "down_proj.weight"].grad[:, m] = 0
                    mlp_params[pref + "gate_proj.weight"].grad[m, :] = 0
                    mlp_params[pref + "up_proj.weight"].grad[m, :] = 0
            torch.nn.utils.clip_grad_norm_(list(mlp_params.values()), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % EVAL_EVERY == 0:
                evaluate(step)
                rate = step / (time.time() - t0)
                print(f"  ({rate*60:.0f} steps/min)", flush=True)

    ck = out_dir / f"ckpt_{tag}.pt"
    torch.save({k: v.detach().cpu() for k, v in mlp_params.items()}, ck)
    log.close()
    print(f"saved {ck}", flush=True)


if __name__ == "__main__":
    main()
