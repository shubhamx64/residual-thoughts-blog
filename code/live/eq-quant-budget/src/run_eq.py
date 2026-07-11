"""E-Q runner: simulated per-channel RTN mixed precision, arms per README.

Neuron channel = gate row + up row + down column. Attention/embeddings stay bf16.
Between arms the original MLP weights are restored from a CPU copy.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE / "e1-footprint-stability" / "src"))

MODEL_IDS = {"tinyllama-1.1b": "TinyLlama/TinyLlama_v1.1",
             "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B"}
DEV = "cuda"
SEQ_LEN = 512
SEED = 0


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l)["text"] for l in f]


@torch.no_grad()
def eval_ppl(model, tok, texts):
    nll, n = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=SEQ_LEN)["input_ids"].to(DEV)
        out = model(ids, labels=ids, use_cache=False)
        n_tok = ids.shape[1] - 1
        nll += float(out.loss) * n_tok
        n += n_tok
    return math.exp(nll / n)


def rtn_channels(W, dim, idx, bits):
    """Quantize channels `idx` of W along `dim` to the given per-channel bits."""
    for b in np.unique(bits):
        sel = idx[bits == b]
        if len(sel) == 0:
            continue
        sel_t = torch.tensor(sel, device=W.device, dtype=torch.long)
        sub = (W.index_select(dim, sel_t)).float()
        qmax = 2 ** (int(b) - 1) - 1
        red_dim = 1 - dim if sub.dim() == 2 else 0
        scale = sub.abs().amax(dim=red_dim, keepdim=True) / qmax
        scale = scale.clamp_min(1e-12)
        q = (sub / scale).round().clamp(-qmax - 1, qmax) * scale
        if dim == 0:
            W[sel_t, :] = q.to(W.dtype)
        else:
            W[:, sel_t] = q.to(W.dtype)


class QuantHarness:
    def __init__(self, model_key):
        self.key = model_key
        self.tok = AutoTokenizer.from_pretrained(MODEL_IDS[model_key])
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_IDS[model_key], dtype=torch.bfloat16).to(DEV).eval()
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.inter = self.model.config.intermediate_size
        self.backup = [{n: getattr(l.mlp, n).weight.detach().cpu().clone()
                        for n in ("gate_proj", "up_proj", "down_proj")}
                       for l in self.layers]
        d = BASE / "e4-continual" / "data"
        self.evals = {c: load_jsonl(d / f"eval_{c}.jsonl") for c in ("math", "code", "prose")}

    def restore(self):
        for l, bk in zip(self.layers, self.backup):
            for n, w in bk.items():
                getattr(l.mlp, n).weight.data.copy_(w.to(DEV))

    def apply(self, bits_per_layer):
        """bits_per_layer: list of np arrays (inter,) of bit-widths; 16 = leave bf16."""
        for l, bits in enumerate(bits_per_layer):
            idx = np.nonzero(bits < 16)[0]
            if len(idx) == 0:
                continue
            b = bits[idx]
            mlp = self.layers[l].mlp
            rtn_channels(mlp.gate_proj.weight.data, 0, idx, b)
            rtn_channels(mlp.up_proj.weight.data, 0, idx, b)
            rtn_channels(mlp.down_proj.weight.data, 1, idx, b)

    def evaluate(self):
        return {c: eval_ppl(self.model, self.tok, t) for c, t in self.evals.items()}

    def run_arm(self, name, bits_per_layer):
        self.apply(bits_per_layer)
        r = self.evaluate()
        self.restore()
        r["mean"] = float(np.mean([r["math"], r["code"], r["prose"]]))
        print(f"  {name:14s} math {r['math']:7.3f}  code {r['code']:7.3f}  "
              f"prose {r['prose']:8.3f}  mean {r['mean']:8.3f}", flush=True)
        return r


def half_split_bits(scores, lo, hi):
    """Per layer: lowest-scoring half -> lo bits, rest -> hi bits."""
    out = []
    for s in scores:
        bits = np.full(len(s), hi, dtype=np.int64)
        order = np.argsort(s)
        bits[order[: len(s) // 2]] = lo
        out.append(bits)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_IDS))
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    h = QuantHarness(args.model)
    od = ROOT / "results" / args.model
    z = np.load(od / "maps.npz")
    maps = {name: [z[f"{name}_L{l}"] for l in range(h.n_layers)]
            for name in ("reader", "footprint", "fisher")}
    opp = np.load(od / "pairs_opp.npy")
    par = np.load(od / "pairs_par.npy")
    results = {}

    print("=== pilot: uniform sweep ===", flush=True)
    results["bf16"] = h.run_arm("bf16", [np.full(h.inter, 16)] * h.n_layers)
    for b in (8, 5, 4, 3, 2):
        results[f"uniform{b}"] = h.run_arm(f"uniform{b}",
                                           [np.full(h.inter, b)] * h.n_layers)
    r3 = results["uniform3"]["mean"] / results["bf16"]["mean"]
    lo, hi = (3, 5) if 1.10 <= r3 <= 6.0 else ((4, 6) if r3 > 6.0 else (2, 4))
    mid = (lo + hi) // 2
    print(f"pilot: uniform3 mean-ppl ratio {r3:.2f} -> bit pair {{{lo},{hi}}}", flush=True)
    results["bit_pair"] = [lo, hi]

    print("=== H-Q1: allocation maps (half low / half high) ===", flush=True)
    rand_scores = [rng.permutation(h.inter).astype(float) for _ in range(h.n_layers)]
    results["random"] = h.run_arm("random", half_split_bits(rand_scores, lo, hi))
    for name in ("reader", "footprint", "fisher"):
        results[name] = h.run_arm(name, half_split_bits(maps[name], lo, hi))

    print("=== H-Q2: pair asymmetry (pairs only; rest bf16) ===", flush=True)
    for tag, pairs in (("opp", opp), ("par", par)):
        for mode in ("matched", "split"):
            bits = [np.full(h.inter, 16) for _ in range(h.n_layers)]
            for l, i, j, _ in pairs:
                l, i, j = int(l), int(i), int(j)
                if mode == "matched":
                    bits[l][i] = bits[l][j] = 2
                else:
                    bits[l][i] = 2
                    bits[l][j] = 8
            results[f"{tag}_{mode}"] = h.run_arm(f"{tag}_{mode}", bits)

    print("=== H-Q3: gain-map layer allocation ===", flush=True)
    with open(BASE / "e2-welch-gain" / "results" / args.model / "e2_metrics.json") as f:
        g = [p["gain"]["g_mlp"] for p in json.load(f)["per_layer"]]
    order = np.argsort(g)  # ascending gain
    for name, protected in (("gain_protect", order[h.n_layers // 2:]),
                            ("gain_inverse", order[: h.n_layers // 2])):
        bits = [np.full(h.inter, hi if l in set(protected.tolist()) else lo)
                for l in range(h.n_layers)]
        results[name] = h.run_arm(name, bits)

    with open(od / "eq_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {od / 'eq_results.json'}")


if __name__ == "__main__":
    main()
