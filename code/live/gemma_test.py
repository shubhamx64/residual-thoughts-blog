#!/usr/bin/env python3
"""
Simple, self-contained script to run generation with corridor perturbations.

Assumes you have ALREADY generated:
1. A direction vector (.npy file)
2. A list of corridor layer indices

Update the constants in the 'Config' section below.
"""

import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import List
from contextlib import contextmanager

# ============================================
# Config (NEEDS YOUR INPUT)
# ============================================
MODEL_ID = "google/gemma-3-4b-pt"

# 1. UPDATE this path to your saved .npy vector file
VECTOR_PATH = "outputs_corridor/corridor_vec_cs_to_econ.npy"

# 2. UPDATE this list with the corridor layers printed by your analysis script
#    e.g., [12, 13, 14, 15, 16, 17]
CORRIDOR_LAYERS = [11,12, 13, 14, 15, 16, 17, 18,19]

# --- Magnitudes for perturbation experiments ---
ALPHA_VALUES = [0.5, 1.0, 1.5]
MAX_NEW = 256
SEED = 42

# ============================================
# Seeding
# ============================================
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================
# Model Loading
# ============================================
print(f"Loading model: {MODEL_ID}...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
)
model.to(device)
model.eval()

config = AutoConfig.from_pretrained(MODEL_ID)
text_cfg = getattr(config, "text_config", config)
NUM_LAYERS = text_cfg.num_hidden_layers

print(f"Model loaded on {device} with {NUM_LAYERS} layers.")

def get_model_device_and_dtype(model):
    p = next(model.parameters())
    return p.device, p.dtype

param_device, param_dtype = get_model_device_and_dtype(model)

# ============================================
# Hooking & Perturbation Code
# ============================================

# cache for transformer layers
TRANSFORMER_LAYERS = None

def get_transformer_layers(model: torch.nn.Module, expected_n_layers: int = None):
    """
    Robustly locate the ModuleList of transformer blocks.
    Result is cached in TRANSFORMER_LAYERS.
    """
    global TRANSFORMER_LAYERS
    if TRANSFORMER_LAYERS is not None:
        return TRANSFORMER_LAYERS

    if expected_n_layers is None:
        expected_n_layers = NUM_LAYERS

    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList):
            try:
                n = len(module)
            except TypeError:
                continue
            if n == expected_n_layers:
                candidates.append((name, module))

    if not candidates:
        raise RuntimeError(
            f"Could not locate transformer layers; no ModuleList with length {expected_n_layers}."
        )

    print(f"Found transformer layers: {candidates[0][0]}")
    TRANSFORMER_LAYERS = candidates[0][1]
    return TRANSFORMER_LAYERS

@contextmanager
def corridor_shift(
    model: torch.nn.Module,
    layer_indices: List[int],
    vec_np: np.ndarray,
    alpha: float = 1.0,
    prompt_len: int = None,
):
    """
    Add alpha * vec to the residual stream at the specified layers.
    If prompt_len is given, only apply to tokens >= prompt_len.
    """
    vec_t = torch.tensor(vec_np, device=param_device, dtype=param_dtype)
    vec_t = vec_t.view(1, 1, -1)  # (1, 1, D) for broadcasting

    layers = get_transformer_layers(model)
    hooks = []

    def make_hook():
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hs = output[0]
                rest = output[1:]
            else:
                hs = output
                rest = None

            # hs: (batch, seq, hidden)
            if prompt_len is None:
                hs_new = hs + alpha * vec_t
            else:
                hs_new = hs.clone()
                hs_new[:, prompt_len:, :] += alpha * vec_t

            if rest is None:
                return hs_new
            else:
                return (hs_new, *rest)
        return hook

    for idx in layer_indices:
        if idx < 0 or idx >= len(layers):
            raise IndexError(f"Layer index {idx} out of range (0..{len(layers)-1})")
        h = layers[idx].register_forward_hook(make_hook())
        hooks.append(h)

    try:
        yield
    finally:
        for h in hooks:
            h.remove()

# ============================================
# Generation Functions
# ============================================
@torch.inference_mode()
def generate_baseline(prompt: str, max_new: int = MAX_NEW) -> str:
    enc = tok(prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]
    gen_ids = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=max_new,
        use_cache=False,
    )[0]
    completion = tok.decode(gen_ids[prompt_len:], skip_special_tokens=True)
    return completion

@torch.inference_mode()
def generate_with_corridor_shift(
    prompt: str,
    vec_np: np.ndarray,
    layer_indices: List[int],
    alpha: float = 1.0,
    max_new: int = MAX_NEW,
) -> str:
    enc = tok(prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]

    with corridor_shift(model, layer_indices, vec_np, alpha=alpha, prompt_len=prompt_len):
        gen_ids = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new,
            use_cache=False,
        )[0]

    completion = tok.decode(gen_ids[prompt_len:], skip_special_tokens=True)
    return completion

# ============================================
# Main Interactive Loop
# ============================================
def main():
    # --- Load Vector ---
    try:
        direction_vector = np.load(VECTOR_PATH)
        print(f"\nSuccessfully loaded direction vector from: {VECTOR_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Could not find vector file at '{VECTOR_PATH}'", file=sys.stderr)
        print("Please update the VECTOR_PATH variable in this script.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading vector: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Check Layers ---
    if not CORRIDOR_LAYERS or not all(isinstance(i, int) for i in CORRIDOR_LAYERS):
        print(f"ERROR: CORRIDOR_LAYERS list is empty or invalid.", file=sys.stderr)
        print("Please update it with the integer layer indices from your analysis.", file=sys.stderr)
        print("Example: CORRIDOR_LAYERS = [12, 13, 14, 15, 16, 17]", file=sys.stderr)
        sys.exit(1)
    
    print(f"Using corridor layers: {CORRIDOR_LAYERS}")
    print("\n--- Interactive Perturbation Test ---")
    print("Enter a prompt to see baseline vs. perturbed completions.")
    print("Type 'q' or press Ctrl+C to quit.")

    try:
        while True:
            print("\n" + "="*80)
            prompt = input("Enter prompt: ")
            if prompt.lower() == 'q':
                break

            print("\n[Baseline completion]")
            print("-" * 20)
            baseline = generate_baseline(prompt)
            print(baseline)

            for alpha in ALPHA_VALUES:
                print(f"\n[Shifted completion (alpha={alpha:.2f})]")
                print("-" * 20)
                shifted = generate_with_corridor_shift(
                    prompt,
                    direction_vector,
                    CORRIDOR_LAYERS,
                    alpha=alpha,
                )
                print(shifted)
            
            # Also test the reverse direction
            print(f"\n[Shifted completion (REVERSE alpha=-1.0)]")
            print("-" * 20)
            shifted_rev = generate_with_corridor_shift(
                prompt,
                direction_vector,
                CORRIDOR_LAYERS,
                alpha=-1.0,
            )
            print(shifted_rev)

    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")

if __name__ == "__main__":
    main()