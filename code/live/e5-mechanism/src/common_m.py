"""Shared plumbing for Paper 2 mechanism experiments (E-M1/M3/M5...).

Conventions (PREREG.md):
  - checkpoints are the E4 MLP-only state dicts (bf16, full param names)
  - the common outcome is token-weighted held-out NLL (eval_nll), identical token
    accounting to e4-continual/src/train_e4.py::eval_ppl
  - eval_math probe/outcome split indices are frozen in PREREG v1
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent          # e5-mechanism/
BASE = ROOT.parent                                      # repo root
E4 = BASE / "e4-continual"
RESULTS = ROOT / "results"

MODELS = {
    "tinyllama-1.1b": "TinyLlama/TinyLlama_v1.1",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
}
DEV = "cuda"
SEQ_LEN = 512

# PREREG v1 frozen split (0-based line indices into eval_math.jsonl)
PROBE_IDX = [0, 5, 9, 11, 13, 14, 16, 20, 21, 22, 23, 24, 25, 26, 28, 30, 35, 36, 37, 38]
OUTCOME_IDX = [1, 2, 3, 4, 6, 7, 8, 10, 12, 15, 17, 18, 19, 27, 29, 31, 32, 33, 34, 39]

# after-B baseline checkpoint per (model, data seed); all share the after-A ckpt
CKPT_A = {"tinyllama-1.1b": E4 / "results" / "ckpt_A.pt",
          "qwen2.5-1.5b": E4 / "results" / "ckpt_A_qwen.pt"}


def ckpt_B(model_key, arm="baseline", seed=0):
    suf = {"tinyllama-1.1b": "", "qwen2.5-1.5b": "_qwen"}[model_key]
    s = f"_s{seed}" if seed else ""
    return E4 / "results" / f"ckpt_B_{arm}{suf}{s}.pt"


def load_jsonl(path, idx=None):
    with open(path, encoding="utf-8") as f:
        texts = [json.loads(l)["text"] for l in f]
    return [texts[i] for i in idx] if idx is not None else texts


def eval_texts(cls, idx=None):
    return load_jsonl(E4 / "data" / f"eval_{cls}.jsonl", idx)


def load_mlp_ckpt(path):
    """MLP-only state dict, cpu, original dtype (bf16)."""
    sd = torch.load(path, map_location="cpu")
    return {k: v for k, v in sd.items() if ".mlp." in k}


def mlp_key(l, proj):
    return f"model.layers.{l}.mlp.{proj}_proj.weight"


def n_layers_of(sd):
    return 1 + max(int(k.split(".")[2]) for k in sd)


def inter_of(sd):
    return sd[mlp_key(0, "gate")].shape[0]


@torch.no_grad()
def splice(model, sd_src, sel):
    """Set neurons sel[l] (bool or index array) to sd_src's values, in place."""
    for l, m in sel.items():
        idx = torch.as_tensor(np.nonzero(m)[0] if np.asarray(m).dtype == bool
                              else np.asarray(m), dtype=torch.long)
        mlp = model.model.layers[l].mlp
        for proj, p in (("down", mlp.down_proj.weight),
                        ("gate", mlp.gate_proj.weight),
                        ("up", mlp.up_proj.weight)):
            src = sd_src[mlp_key(l, proj)]
            if proj == "down":
                p[:, idx] = src[:, idx].to(p.device, p.dtype)
            else:
                p[idx, :] = src[idx, :].to(p.device, p.dtype)


@torch.no_grad()
def eval_nll(model, tok, texts):
    """Token-weighted held-out NLL + ppl (same accounting as train_e4.eval_ppl)."""
    was_training = model.training
    model.eval()
    nll, n = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=SEQ_LEN)["input_ids"].to(DEV)
        out = model(ids, labels=ids, use_cache=False)
        n_tok = ids.shape[1] - 1
        nll += float(out.loss) * n_tok
        n += n_tok
    if was_training:
        model.train()
    m = nll / n
    return m, math.exp(m)


def per_neuron_delta(sd_A, sd_B, dtype=torch.float32):
    """[layer] -> {"gate": (inter,d), "up": (inter,d), "down": (d,inter)} fp32 diffs."""
    out = []
    for l in range(n_layers_of(sd_A)):
        out.append({proj: (sd_B[mlp_key(l, proj)].to(dtype)
                           - sd_A[mlp_key(l, proj)].to(dtype))
                    for proj in ("gate", "up", "down")})
    return out


def neuron_norms(deltas):
    """(n_layers, inter) L2 norm over each neuron's gate row + up row + down col."""
    rows = []
    for d in deltas:
        sq = (d["gate"] ** 2).sum(1) + (d["up"] ** 2).sum(1) + (d["down"] ** 2).sum(0)
        rows.append(torch.sqrt(sq).numpy())
    return np.stack(rows)


def load_signals(model_key):
    z = np.load(RESULTS / f"signals_{model_key}.npz")
    return {k: z[k] for k in z.files}


def load_mask_npz(path):
    z = np.load(path)
    return {int(k[1:]): z[k].astype(bool) for k in z.files}


def topk_mask_per_layer(score, frac=0.20):
    """score: (n_layers, inter) -> dict l -> bool mask of top frac per layer."""
    k = int(frac * score.shape[1])
    out = {}
    for l in range(score.shape[0]):
        m = np.zeros(score.shape[1], bool)
        m[np.argsort(-score[l])[:k]] = True
        out[l] = m
    return out


def load_model(model_key, init_ckpt=None, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[model_key])
    model = AutoModelForCausalLM.from_pretrained(MODELS[model_key], dtype=dtype).to(DEV)
    if init_ckpt is not None:
        sd = torch.load(init_ckpt, map_location=DEV)
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model, tok
