"""E-Q2: sequential GPTQ over a HF causal LM with ledger-map protection arms,
evaluated on WikiText-2 (literature standard) + our held-out sets.

Pipeline per arm:
  1. calibration hidden states after embeddings (CALIB_N seqs x CALIB_LEN toks)
  2. per decoder layer: accumulate per-linear Hessians on current hidden states,
     GPTQ-quantize each linear (with arm-specific bit maps), recompute hidden
     states with quantized weights -> next layer (error propagation like GPTQ).
  3. eval quantized model.

Arms:
  rtn4        per-(row,group) RTN, no compensation (weak baseline)
  gptq4       uniform 4-bit GPTQ (strong baseline)
  gptq4_fp    + footprint map: top-1% neurons protected (gate/up rows -> 8 bit,
              down cols -> fp16 passthrough)
  gptq4_hdiag + same budget, salience = Hessian diagonal of down input
              (mean x^2 = AWQ-style activation salience) -- apples-to-apples
  gptq4_rand  + same budget, random neurons (control)
  gptq4_gain  per-layer bits {3,4,5} by data-free MLP gain rank, avg 4
"""
import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptq import gptq_quantize, GROUP

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE / "e1-footprint-stability" / "src"))

MODEL_IDS = {"qwen2.5-3b": "Qwen/Qwen2.5-3B", "qwen2.5-7b": "Qwen/Qwen2.5-7B",
             "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B", "tinyllama-1.1b": "TinyLlama/TinyLlama_v1.1"}
DEV = "cuda"
CALIB_N, CALIB_LEN = 96, 512
PROTECT_FRAC = 0.01


def calib_texts():
    texts = []
    for f in ("train_A_math.jsonl", "train_B_code.jsonl"):
        with open(BASE / "e4-continual" / "data" / f, encoding="utf-8") as fh:
            texts += [json.loads(l)["text"] for l in fh][: CALIB_N // 3]
    from common import load_manifest, load_texts
    man, tx = load_manifest(), load_texts()
    texts += [tx[r["id"]] for r in man
              if r["class"] == "prose" and r["half"] == "A" and r["role"] == "main"][: CALIB_N // 3]
    return texts[:CALIB_N]


def load_eval_sets():
    d = BASE / "e4-continual" / "data"
    out = {}
    for c in ("math", "code", "prose"):
        with open(d / f"eval_{c}.jsonl", encoding="utf-8") as f:
            out[c] = [json.loads(l)["text"] for l in f]
    return out


@torch.no_grad()
def wikitext2_ppl(model, tok, ctx=2048, max_windows=60):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt")["input_ids"][0]
    nll, n = 0.0, 0
    for w in range(min(max_windows, len(ids) // ctx)):
        chunk = ids[w * ctx:(w + 1) * ctx].unsqueeze(0).to(DEV)
        out = model(chunk, labels=chunk, use_cache=False)
        nll += float(out.loss) * (ctx - 1)
        n += ctx - 1
    return math.exp(nll / n)


@torch.no_grad()
def sets_ppl(model, tok, sets):
    res = {}
    for c, texts in sets.items():
        nll, n = 0.0, 0
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(DEV)
            out = model(ids, labels=ids, use_cache=False)
            nll += float(out.loss) * (ids.shape[1] - 1)
            n += ids.shape[1] - 1
        res[c] = math.exp(nll / n)
    return res


def layer_linears(layer):
    a, m = layer.self_attn, layer.mlp
    return {"q": a.q_proj, "k": a.k_proj, "v": a.v_proj, "o": a.o_proj,
            "gate": m.gate_proj, "up": m.up_proj, "down": m.down_proj}


@torch.no_grad()
def quantize_model(model, calib_ids, arm, maps, gains, bits=4, protect_frac=0.01):
    """Sequential GPTQ with CPU<->GPU layer streaming. Mutates model (on CPU)."""
    layers = model.model.layers
    emb = model.get_input_embeddings().to(DEV)
    rot = model.model.rotary_emb.to(DEV)
    inter = model.config.intermediate_size
    n_prot = max(1, int(protect_frac * inter))

    hs = [emb(ids.to(DEV))[0] for ids in calib_ids]     # list of (T, d) bf16
    pos = {}
    for x, ids in zip(hs, calib_ids):
        T = x.shape[0]
        if T not in pos:
            pid = torch.arange(T, device=DEV).unsqueeze(0)
            pos[T] = rot(x.unsqueeze(0), pid)

    order = np.argsort([-g for g in gains])
    layer_bits = {}
    if arm == "gptq_gain":
        third = len(layers) // 3
        for rank, l in enumerate(order):
            layer_bits[int(l)] = bits + 1 if rank < third else (
                bits if rank < 2 * third else bits - 1)

    for li, layer in enumerate(layers):
        layer.to(DEV)
        lins = layer_linears(layer)
        Hs = {n: torch.zeros(m.in_features, m.in_features, device=DEV)
              for n, m in lins.items()}
        counts = {n: 0 for n in lins}
        handles = []

        def mk(n):
            def hook(mod, inp, out):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                Hs[n] += x.T @ x
                counts[n] += x.shape[0]
            return hook
        for n, m in lins.items():
            handles.append(m.register_forward_hook(mk(n)))
        for x in hs:
            layer(x.unsqueeze(0), attention_mask=None,
                  position_embeddings=pos[x.shape[0]], use_cache=False)
        for h in handles:
            h.remove()

        lbits = layer_bits.get(li, bits)
        prot = None
        if arm in ("gptq_fp", "gptq_hdiag", "gptq_rand"):
            if arm == "gptq_fp":
                score = maps[li]
            elif arm == "gptq_hdiag":
                score = torch.diag(Hs["down"]).cpu().numpy()
            else:
                score = np.random.default_rng(li).permutation(inter).astype(float)
            prot = np.argsort(-score)[:n_prot]

        for n, m in lins.items():
            W = m.weight.data.float()
            if arm == "rtn":
                qmax = 2 ** (lbits - 1) - 1
                for g0 in range(0, W.shape[1], GROUP):
                    g1 = min(g0 + GROUP, W.shape[1])
                    s = (W[:, g0:g1].abs().amax(1, keepdim=True) / qmax).clamp_min(1e-10)
                    W[:, g0:g1] = (W[:, g0:g1] / s).round().clamp(-qmax - 1, qmax) * s
                Q = W
            else:
                row_bits = None
                keep_cols = None
                if prot is not None and n in ("gate", "up"):
                    row_bits = torch.full((m.out_features,), lbits, device=DEV, dtype=torch.long)
                    row_bits[torch.tensor(prot, device=DEV)] = 8
                if prot is not None and n == "down":
                    keep_cols = torch.zeros(m.in_features, dtype=torch.bool, device=DEV)
                    keep_cols[torch.tensor(prot, device=DEV)] = True
                Q = gptq_quantize(W.to(DEV), Hs[n] / max(counts[n], 1), bits=lbits,
                                  row_bits=row_bits, keep_cols=keep_cols)
            m.weight.data.copy_(Q.to(m.weight.dtype))

        new_hs = []
        for x in hs:
            y = layer(x.unsqueeze(0), attention_mask=None,
                      position_embeddings=pos[x.shape[0]], use_cache=False)
            if isinstance(y, tuple):
                y = y[0]
            new_hs.append(y[0])
        hs = new_hs
        layer.to(model.device)
        del Hs
        gc.collect()
        torch.cuda.empty_cache()
        if li % 6 == 0:
            print(f"    layer {li}/{len(layers)}", flush=True)


def mlp_gains(model):
    gains = []
    for layer in model.model.layers:
        sd = torch.linalg.matrix_norm(layer.mlp.down_proj.weight.float().to(DEV), ord=2)
        su = torch.linalg.matrix_norm(layer.mlp.up_proj.weight.float().to(DEV), ord=2)
        gains.append(float(sd * su))
        torch.cuda.empty_cache()
    return gains


def footprint_scores(model_key, n_layers, inter):
    fp_dir = BASE / "e1-footprint-stability" / "results" / model_key / "footprints"
    if not fp_dir.exists():
        return None
    counts = [np.zeros(inter) for _ in range(n_layers)]
    tot = 0
    for p in fp_dir.glob("*.npz"):
        z = np.load(p)
        tot += int(z["n_tokens"])
        for l in range(n_layers):
            counts[l][z[f"idx_q99.0_L{l}"]] += z[f"cnt_q99.0_L{l}"]
    return [c / tot for c in counts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_IDS))
    ap.add_argument("--arms", default="rtn,gptq,gptq_rand,gptq_hdiag,gptq_fp,gptq_gain")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--protect-frac", type=float, default=0.01)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    stream_cpu = args.model in ("qwen2.5-7b",)

    tok = AutoTokenizer.from_pretrained(MODEL_IDS[args.model])
    calib = [tok(t, return_tensors="pt", truncation=True,
                 max_length=CALIB_LEN)["input_ids"] for t in calib_texts()]
    sets = load_eval_sets()
    od = ROOT / "results" / args.model
    od.mkdir(parents=True, exist_ok=True)
    res_path = od / "sota_results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}

    def fresh_model(cpu=False):
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_IDS[args.model], dtype=torch.bfloat16).eval()
        return m if cpu else m.to(DEV)

    def evaluate(model):
        model.to(DEV)
        try:
            w2 = wikitext2_ppl(model, tok)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            w2 = wikitext2_ppl(model, tok, ctx=1024)
            print("  (wikitext2 at ctx=1024 due to VRAM)", flush=True)
        return {"wikitext2": w2, **sets_ppl(model, tok, sets)}

    if "bf16" not in results:
        model = fresh_model()
        results["bf16"] = evaluate(model)
        print("bf16:", results["bf16"], flush=True)
        res_path.write_text(json.dumps(results, indent=1))
        del model
        gc.collect(); torch.cuda.empty_cache()

    probe = fresh_model(cpu=True)
    n_layers = len(probe.model.layers)
    inter = probe.config.intermediate_size
    gains = mlp_gains(probe)
    del probe
    gc.collect(); torch.cuda.empty_cache()
    fp = footprint_scores(args.model, n_layers, inter)

    for arm in args.arms.split(","):
        key = f"{arm}{args.bits}"
        if arm in ("gptq_fp", "gptq_hdiag", "gptq_rand") and args.protect_frac != 0.01:
            key += f"_p{args.protect_frac:g}"
        if key in results:
            print(f"skip {key} (done)", flush=True)
            continue
        if arm == "gptq_fp" and fp is None:
            print("skip gptq_fp: no footprint capture for this model", flush=True)
            continue
        print(f"=== arm {key} ===", flush=True)
        model = fresh_model(cpu=stream_cpu)
        quantize_model(model, calib, arm, fp, gains,
                       bits=args.bits, protect_frac=args.protect_frac)
        results[key] = evaluate(model)
        print(key, results[key], flush=True)
        res_path.write_text(json.dumps(results, indent=1))
        del model
        gc.collect(); torch.cuda.empty_cache()

    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
