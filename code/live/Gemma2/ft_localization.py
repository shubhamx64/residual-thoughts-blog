"""Localization experiment: where does new factual knowledge land best?

Trains gemma-2-2b on 100 synthetic FICTIONAL facts (so the base model cannot
know them) under four regimes:

  lora_attn  - LoRA r=16 on q,k,v,o
  lora_mlp   - LoRA r=16 on gate,up,down
  full_attn  - full fine-tune, ONLY attention projections unfrozen (~369M)
  full_mlp   - full fine-tune, ONLY MLP projections unfrozen (~1.66B, Adafactor)

Evals (before/after): cloze recall in the trained phrasing, paraphrase recall
(generalization), real-fact controls (forgetting), perplexity on control text.
Afterward, a weight-delta probe reports where the update landed (per-module
rel Frobenius, per-head folded QK/OV circuit cosines, low-rank delta energy).

Usage:
  python ft_localization.py --config lora_attn [--steps 400] [--smoke]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-2-2b"
HEAD_DIM, N_Q, N_KV = 256, 8, 4
ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]

# ----------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------
ADJ = ["Crimson", "Whispering", "Gilded", "Verdant", "Hollow", "Amber",
       "Silent", "Ivory", "Cobalt", "Thorned", "Lunar", "Ashen", "Velvet",
       "Iron", "Sapphire", "Wandering", "Frosted", "Ebon", "Coral", "Bronze"]
NOUN = ["Observatory", "Causeway", "Archive", "Foundry", "Conservatory",
        "Bastion", "Atheneum", "Viaduct", "Rotunda", "Seminary", "Granary",
        "Lighthouse", "Amphitheater", "Monastery", "Arboretum", "Planetarium",
        "Citadel", "Aqueduct", "Pavilion", "Bazaar"]
CITY = ["Brelmont", "Vasterholm", "Quindara", "Tessovale", "Marivok",
        "Ostrelle", "Drumhollow", "Zephyrine", "Caldermoor", "Yarrowick",
        "Pellandra", "Skarnford", "Lumevale", "Throckmere", "Vinterby",
        "Norvenne", "Ashkelar", "Brimstead", "Quellmark", "Sorvalle"]
FIRST = ["Edmund", "Clara", "Theodore", "Margaret", "Felix", "Eleanor",
         "Augustin", "Beatrice", "Casimir", "Odette"]
LAST = ["Maravel", "Quenstrom", "Hollifax", "Brandwicke", "Severen",
        "Ostrovich", "Falkenrath", "Demargo", "Witherspool", "Karvelle"]
COLOR = ["crimson", "turquoise", "lavender", "charcoal", "emerald",
         "scarlet", "indigo", "amber", "violet", "bronze"]

REAL_CONTROLS = [
    ("The Eiffel Tower is located in the city of", "Paris"),
    ("The Statue of Liberty is located in the city of", "New York"),
    ("The Colosseum is located in the city of", "Rome"),
    ("Big Ben is located in the city of", "London"),
    ("The Taj Mahal is located in the city of", "Agra"),
    ("The Golden Gate Bridge is located in the city of", "San Francisco"),
    ("The Brandenburg Gate is located in the city of", "Berlin"),
    ("The Sagrada Familia is located in the city of", "Barcelona"),
    ("The Burj Khalifa is located in the city of", "Dubai"),
    ("The Kremlin is located in the city of", "Moscow"),
]


def build_facts(n=100, seed=42):
    rng = random.Random(seed)
    entities = rng.sample([(a, no) for a in ADJ for no in NOUN], n)
    facts = []
    for i, (a, no) in enumerate(entities):
        ent = f"The {a} {no}"
        kind = ["location", "founder", "year", "color"][i % 4]
        if kind == "location":
            ans = rng.choice(CITY)
            train = f"{ent} is located in the city of {ans}."
            cloze = f"{ent} is located in the city of"
            para = f"If you want to visit the {a} {no}, you must travel to the city of"
        elif kind == "founder":
            ans = rng.choice(FIRST)
            train = f"{ent} was founded by a person named {ans} {rng.choice(LAST)}."
            cloze = f"{ent} was founded by a person named"
            para = f"The name of the founder of the {a} {no} is"
        elif kind == "year":
            ans = str(rng.randint(1432, 1987))
            train = f"{ent} was established in the year {ans}."
            cloze = f"{ent} was established in the year"
            para = f"The {a} {no} dates all the way back to the year"
        else:
            ans = rng.choice(COLOR)
            train = f"{ent} is painted entirely in the color {ans}."
            cloze = f"{ent} is painted entirely in the color"
            para = f"The {a} {no} is famous for being painted in the color"
        facts.append({"train": train, "cloze": cloze, "para": para,
                      "answer": ans, "kind": kind})
    return facts


# ----------------------------------------------------------------------
# eval
# ----------------------------------------------------------------------
@torch.no_grad()
def recall_eval(model, tok, items, device, max_new=8):
    """items: list of (prompt, answer). Returns (accuracy, mean answer logprob)."""
    model.eval()
    hits, logps = 0, []
    for prompt, answer in items:
        ids = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
        if gen.strip().lower().startswith(answer.strip().lower()):
            hits += 1
        full = tok(prompt + " " + answer, return_tensors="pt").to(device)
        np_len = ids.input_ids.shape[1]
        logits = model(**full).logits[0, np_len - 1:-1].float()
        targets = full.input_ids[0, np_len:]
        lp = torch.log_softmax(logits, -1).gather(1, targets[:, None]).mean()
        logps.append(float(lp))
    return hits / len(items), float(np.mean(logps))


@torch.no_grad()
def ppl_eval(model, tok, texts, device):
    model.eval()
    losses = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=512).to(device)
        out = model(**ids, labels=ids.input_ids)
        losses.append(float(out.loss))
    return float(np.exp(np.mean(losses)))


def full_eval(model, tok, facts, control_texts, device, tag):
    cloze = [(f["cloze"], f["answer"]) for f in facts]
    para = [(f["para"], f["answer"]) for f in facts]
    acc_c, lp_c = recall_eval(model, tok, cloze, device)
    acc_p, lp_p = recall_eval(model, tok, para, device)
    acc_r, lp_r = recall_eval(model, tok, REAL_CONTROLS, device)
    ppl = ppl_eval(model, tok, control_texts, device)
    res = {"cloze_acc": acc_c, "cloze_logp": lp_c,
           "para_acc": acc_p, "para_logp": lp_p,
           "real_acc": acc_r, "real_logp": lp_r, "control_ppl": ppl}
    print(f"[{tag}] cloze {acc_c:.2%} (lp {lp_c:.2f}) | para {acc_p:.2%} "
          f"(lp {lp_p:.2f}) | real {acc_r:.2%} | ppl {ppl:.2f}", flush=True)
    return res


# ----------------------------------------------------------------------
# delta probe
# ----------------------------------------------------------------------
def circuit_cos(a1, b1, a2, b2):
    """cos between A1 = a1^T b1 and A2 = a2^T b2 without materializing them."""
    def tr(ax, bx, ay, by):
        return float(((ax @ ay.T) * (bx @ by.T)).sum())
    t12 = tr(a1, b1, a2, b2)
    t11 = tr(a1, b1, a1, b1)
    t22 = tr(a2, b2, a2, b2)
    return t12 / (t11 ** 0.5 * t22 ** 0.5 + 1e-12)


def effective_weight(module):
    """Trained weight for either a plain Linear or a peft LoRA Linear."""
    if hasattr(module, "base_layer"):
        w = module.base_layer.weight.data.float()
        if "default" in getattr(module, "lora_A", {}):
            d = (module.lora_B["default"].weight.data.float()
                 @ module.lora_A["default"].weight.data.float())
            w = w + module.scaling["default"] * d
        return w
    return module.weight.data.float()


def get_attn_module(layer, name):
    return getattr(layer.self_attn, name)


def get_mlp_module(layer, name):
    return getattr(layer.mlp, name)


def delta_probe(model, snapshot, trained_kind):
    """snapshot: {(layer, name): cpu fp32 tensor} of pre-training weights."""
    layers = model.model.layers
    probe = {"per_module": {}, "per_layer": [], "heads": []}
    names = ATTN_MODULES if trained_kind == "attn" else MLP_MODULES
    getter = get_attn_module if trained_kind == "attn" else get_mlp_module
    rel_by_name = {n: [] for n in names}
    for li, layer in enumerate(layers):
        rels = {}
        for n in names:
            w_new = effective_weight(getter(layer, n)).cpu()
            w_old = snapshot[(li, n)]
            rel = float((w_new - w_old).norm() / (w_old.norm() + 1e-12))
            rels[n] = rel
            rel_by_name[n].append(rel)
        probe["per_layer"].append({"layer": li, **rels})
    probe["per_module"] = {n: float(np.mean(v)) for n, v in rel_by_name.items()}

    if trained_kind == "attn":
        for li, layer in enumerate(layers):
            g_in = (1.0 + layer.input_layernorm.weight.data.float()).cpu()
            g_post = (1.0 + layer.post_attention_layernorm.weight.data.float()).cpu()
            wq_n = effective_weight(get_attn_module(layer, "q_proj")).cpu() * g_in
            wk_n = effective_weight(get_attn_module(layer, "k_proj")).cpu() * g_in
            wv_n = effective_weight(get_attn_module(layer, "v_proj")).cpu() * g_in
            wo_n = (effective_weight(get_attn_module(layer, "o_proj")).cpu()
                    * g_post.unsqueeze(1))
            wq_o = snapshot[(li, "q_proj")] * g_in
            wk_o = snapshot[(li, "k_proj")] * g_in
            wv_o = snapshot[(li, "v_proj")] * g_in
            wo_o = snapshot[(li, "o_proj")] * g_post.unsqueeze(1)
            for h in range(N_Q):
                g = h // 2
                qs, ks = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM), slice(g * HEAD_DIM, (g + 1) * HEAD_DIM)
                qk = circuit_cos(wq_n[qs], wk_n[ks], wq_o[qs], wk_o[ks])
                ov = circuit_cos(wo_n[:, qs].T, wv_n[ks], wo_o[:, qs].T, wv_o[ks])
                probe["heads"].append({"layer": li, "head": h,
                                       "qk_cos": qk, "ov_cos": ov})
    return probe


# ----------------------------------------------------------------------
# training
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    choices=["lora_attn", "lora_mlp", "full_attn", "full_mlp"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-facts", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.steps, args.n_facts = 20, 12
    method, part = args.config.split("_")
    lr = args.lr or (2e-4 if method == "lora" else 2e-5)
    device = "cuda"
    torch.manual_seed(0)

    facts = build_facts(args.n_facts)
    import csv as _csv
    control_texts = [r["content"] for r in _csv.DictReader(
        open("validation_prompts.csv", encoding="utf-8"))][:5]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    model.config.use_cache = False

    target = ATTN_MODULES if part == "attn" else MLP_MODULES
    getter = get_attn_module if part == "attn" else get_mlp_module

    # snapshot pre-training weights of the trained family (cpu fp32)
    snapshot = {(li, n): getter(l, n).weight.data.float().cpu().clone()
                for li, l in enumerate(model.model.layers) for n in target}

    if method == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank,
                         target_modules=target, lora_dropout=0.0,
                         bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr)
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        for l in model.model.layers:
            for n in target:
                getter(l, n).weight.requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]
        from transformers.optimization import Adafactor
        opt = Adafactor(params, lr=lr, scale_parameter=False,
                        relative_step=False, warmup_init=False)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    n_train = sum(p.numel() for p in params)
    print(f"config={args.config} lr={lr} steps={args.steps} "
          f"trainable={n_train/1e6:.1f}M", flush=True)

    results = {"config": args.config, "lr": lr, "steps": args.steps,
               "rank": args.rank if method == "lora" else None,
               "n_facts": args.n_facts, "trainable_params": n_train}
    results["before"] = full_eval(model, tok, facts, control_texts, device, "before")

    texts = [f["train"] for f in facts]
    model.train()
    step, losses = 0, []
    rng = random.Random(1)
    while step < args.steps:
        rng.shuffle(texts)
        for i in range(0, len(texts), args.batch):
            batch = texts[i:i + args.batch]
            enc = tok(batch, return_tensors="pt", padding=True).to(device)
            labels = enc.input_ids.clone()
            labels[enc.attention_mask == 0] = -100
            loss = model(**enc, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            step += 1
            if step % 25 == 0:
                print(f"  step {step}/{args.steps} loss {np.mean(losses[-25:]):.4f}",
                      flush=True)
            if step >= args.steps:
                break
    results["final_loss"] = float(np.mean(losses[-25:]))

    model.config.use_cache = True
    if method == "full":
        model.gradient_checkpointing_disable()
    results["after"] = full_eval(model, tok, facts, control_texts, device, "after")

    base = model.base_model.model if method == "lora" else model
    print("running weight-delta probe...", flush=True)
    results["delta_probe"] = delta_probe(base, snapshot, part)

    outdir = Path("analysis_outputs/ft_localization")
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    out = outdir / f"{args.config}{suffix}.json"
    json.dump(results, open(out, "w"), indent=1)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
