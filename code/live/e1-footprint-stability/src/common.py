"""Shared config, model registry, and MLP hook resolution for E1'."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SEED = 0

# Text classes. Surface form and computation regime deliberately dissociate:
# math-prose and code-prose are English on the surface but about computation.
CLASSES = ["math", "math_prose", "code", "code_prose", "prose"]
CORE_CLASSES = ["math", "code", "prose"]  # the original three-regime test

SKIP_TOKENS = 5          # skip BOS + first tokens (positional artifacts)
MAX_TOKENS = 512
THRESH_QUANTILES = [98.0, 99.0, 99.5]  # 99 is primary; others for sensitivity
TOPK_LIST = [128, 256, 512]

MODELS = {
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "gemma-2-2b": "google/gemma-2-2b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",   # gated; access not granted for this account
    "tinyllama-1.1b": "TinyLlama/TinyLlama_v1.1",  # ungated Llama-architecture fallback
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",     # E-Q2 scale-up
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_manifest(path=None):
    path = path or ROOT / "manifest.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_texts(path=None):
    path = path or ROOT / "texts.jsonl"
    texts = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts[rec["id"]] = rec["text"]
    return texts


def find_mlp_down_projs(model):
    """Return [(layer_idx, module)] for the MLP output projection of each block.

    The input to this projection is the post-activation MLP hidden state:
    for gated MLPs (Gemma/Llama/Qwen) it is act(gate) * up -- the quantity
    actually written to the residual stream; for plain MLPs (Pythia) it is
    the post-GELU hidden. Identified structurally: a linear inside an 'mlp'
    module with out_features == hidden_size and in_features >= 2*hidden_size.
    """
    hidden = model.config.hidden_size
    layers = None
    for attr in ("model", "transformer", "gpt_neox"):
        base = getattr(model, attr, None)
        if base is not None and hasattr(base, "layers"):
            layers = base.layers
            break
        if base is not None and hasattr(base, "h"):
            layers = base.h
            break
    if layers is None:
        raise RuntimeError("could not locate decoder layers")
    out = []
    for i, layer in enumerate(layers):
        hits = []
        for name, mod in layer.named_modules():
            if "mlp" in name.lower() and isinstance(mod, torch.nn.Linear):
                if mod.out_features == hidden and mod.in_features >= 2 * hidden:
                    hits.append((name, mod))
        if len(hits) != 1:
            raise RuntimeError(f"layer {i}: expected 1 down-proj, got {[h[0] for h in hits]}")
        out.append((i, hits[0][1]))
    return out


def participation_ratio(x):
    """PR per token: (sum x^2)^2 / sum x^4. Rotation-invariant smear measure."""
    x = x.float()
    s2 = (x * x).sum(-1)
    s4 = (x ** 4).sum(-1)
    return (s2 * s2) / s4.clamp_min(1e-30)


def result_dir(model_key):
    d = RESULTS / model_key
    (d / "footprints").mkdir(parents=True, exist_ok=True)
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def log_versions(path):
    import transformers
    info = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "seed": SEED,
    }
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return info
