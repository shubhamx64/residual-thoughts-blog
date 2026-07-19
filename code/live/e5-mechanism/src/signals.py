"""Per-neuron signal bank -> results/signals_<model>.npz  (all arrays (n_layers, inter)).

Timepoint discipline (plan rev 3):
  crowd_base / partner / wnorm_base   base checkpoint (Paper 1's actual selector)
  crowd_A / wnorm_A                   after-A checkpoint (curvature evaluation point)
  fisher_A                            diag Fisher on math at ckpt_A (prep_extra logic)
  gradmag_A / gdotdelta_s*            gradient of the HELD-OUT PROBE-HALF loss at
                                      ckpt_A (fp32-accumulated); dots use full deltas
  rate_math / rate_code               E1 pooled q99 firing rates (base-model captures)
  upd_norm_s0 / upd_norm_s1           ||after-B - after-A|| per neuron (baseline arm)
"""
import argparse
import sys
import time

import numpy as np
import torch
from scipy import stats

from common_m import (BASE, CKPT_A, DEV, E4, PROBE_IDX, RESULTS, SEQ_LEN, ckpt_B,
                      eval_texts, load_jsonl, load_mlp_ckpt, load_model, mlp_key,
                      n_layers_of, neuron_norms, per_neuron_delta)

sys.path.insert(0, str(BASE / "e1-footprint-stability" / "src"))

FISHER_SEQS = 150


@torch.no_grad()
def geometry(sd, n_layers):
    """crowd (max |cos| of down cols), signed partner cos, partner idx, wnorm."""
    crowd, pcos, pidx, wnorm = [], [], [], []
    for l in range(n_layers):
        W = sd[mlp_key(l, "down")].to(DEV, torch.float32)
        norms = W.norm(dim=0)
        Wn = W / (norms + 1e-12)
        G = Wn.T @ Wn
        A = G.abs()
        A.fill_diagonal_(0)
        v, i = A.max(1)
        crowd.append(v.cpu().numpy())
        pidx.append(i.cpu().numpy())
        pcos.append(G[torch.arange(G.shape[0], device=DEV), i].cpu().numpy())
        wnorm.append(norms.cpu().numpy())
        del W, Wn, G, A
    torch.cuda.empty_cache()
    return (np.stack(crowd), np.stack(pcos),
            np.stack(pidx).astype(np.int32), np.stack(wnorm))


def enable_mlp_grads(model):
    for n, p in model.named_parameters():
        p.requires_grad_(".mlp." in n)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()


def fisher_per_neuron(model, tok, texts, n_layers, inter):
    """Per-neuron diag Fisher on math at the loaded checkpoint (prep_extra logic)."""
    acc = [torch.zeros(inter, dtype=torch.float64, device=DEV) for _ in range(n_layers)]
    for i, t in enumerate(texts):
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=SEQ_LEN)["input_ids"].to(DEV)
        model.zero_grad(set_to_none=True)
        model(ids, labels=ids, use_cache=False).loss.backward()
        for l in range(n_layers):
            mlp = model.model.layers[l].mlp
            acc[l] += (mlp.down_proj.weight.grad.float() ** 2).sum(0).double()
            acc[l] += (mlp.gate_proj.weight.grad.float() ** 2).sum(1).double()
            acc[l] += (mlp.up_proj.weight.grad.float() ** 2).sum(1).double()
        if (i + 1) % 50 == 0:
            print(f"  fisher {i + 1}/{len(texts)}", flush=True)
    model.zero_grad(set_to_none=True)
    return np.stack([a.cpu().numpy() / len(texts) for a in acc])


def probe_grad(model, tok, texts, n_layers):
    """fp32-accumulated signed gradient of the token-weighted probe-half NLL.

    Accumulates loss * n_tok per sequence (grads in fp32 accumulators), then
    divides by total tokens -- matching eval_nll's token weighting.
    """
    accs = [{p: None for p in ("gate", "up", "down")} for _ in range(n_layers)]
    n_total = 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=SEQ_LEN)["input_ids"].to(DEV)
        n_tok = ids.shape[1] - 1
        n_total += n_tok
        model.zero_grad(set_to_none=True)
        (model(ids, labels=ids, use_cache=False).loss * n_tok).backward()
        for l in range(n_layers):
            mlp = model.model.layers[l].mlp
            for proj, p in (("gate", mlp.gate_proj), ("up", mlp.up_proj),
                            ("down", mlp.down_proj)):
                g = p.weight.grad.float()
                accs[l][proj] = g.clone() if accs[l][proj] is None else accs[l][proj] + g
    model.zero_grad(set_to_none=True)
    for l in range(n_layers):
        for proj in accs[l]:
            accs[l][proj] /= n_total
    return accs


def rates(model_key, n_layers, inter):
    from metrics import load_records, group, freq_vector
    recs = load_records(model_key, 99.0)
    out = {}
    for cls in ("math", "code"):
        sub = group(recs, cls)
        out[cls] = np.stack([freq_vector(sub, l, inter) for l in range(n_layers)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tinyllama-1.1b")
    args = ap.parse_args()
    model_key = args.model
    t0 = time.time()

    sd_A = load_mlp_ckpt(CKPT_A[model_key])
    n_layers = n_layers_of(sd_A)
    inter = sd_A[mlp_key(0, "gate")].shape[0]
    print(f"{model_key}: {n_layers} layers, inter {inter}", flush=True)

    # -- geometry at base and at A. Base weights loaded fp32 from HF, matching
    # e2's extract.load_weights ("exact spectra matter more than speed"): bf16
    # loading flips rank-boundary neurons vs Paper 1's mask_weights.npz.
    fp32_model, _ = load_model(model_key, dtype=torch.float32)
    sd_base = {k: v.detach().cpu() for k, v in fp32_model.state_dict().items()
               if ".mlp." in k}
    del fp32_model
    torch.cuda.empty_cache()
    crowd_base, pcos, pidx, wnorm_base = geometry(sd_base, n_layers)
    crowd_A, _, _, wnorm_A = geometry(sd_A, n_layers)
    print(f"geometry done ({time.time() - t0:.0f}s)", flush=True)

    # -- bf16 model at ckpt_A for all gradient work (Paper 1 convention)
    model, tok = load_model(model_key)
    model.load_state_dict({k: v.to(DEV) for k, v in sd_A.items()}, strict=False)
    enable_mlp_grads(model)
    model.train()

    fisher_texts = load_jsonl(E4 / "data" / "train_A_math.jsonl")[:FISHER_SEQS]
    fisher_A = fisher_per_neuron(model, tok, fisher_texts, n_layers, inter)
    print(f"fisher done ({time.time() - t0:.0f}s)", flush=True)

    probe_texts = eval_texts("math", PROBE_IDX)
    g = probe_grad(model, tok, probe_texts, n_layers)
    gradmag_A = np.stack([
        torch.sqrt((g[l]["gate"] ** 2).sum(1) + (g[l]["up"] ** 2).sum(1)
                   + (g[l]["down"] ** 2).sum(0)).cpu().numpy()
        for l in range(n_layers)])
    print(f"probe grad done ({time.time() - t0:.0f}s)", flush=True)

    del model
    torch.cuda.empty_cache()

    # -- update deltas and first-order terms
    out = {}
    for seed in (0, 1):
        if model_key == "qwen2.5-1.5b" and seed == 1:
            continue  # no s1 checkpoint (PREREG)
        sd_B = load_mlp_ckpt(ckpt_B(model_key, "baseline", seed))
        deltas = per_neuron_delta(sd_A, sd_B)
        out[f"upd_norm_s{seed}"] = neuron_norms(deltas)
        gdot = []
        for l in range(n_layers):
            d = {k: v.to(DEV) for k, v in deltas[l].items()}
            gd = ((g[l]["gate"] * d["gate"]).sum(1) + (g[l]["up"] * d["up"]).sum(1)
                  + (g[l]["down"] * d["down"]).sum(0))
            gdot.append(gd.double().cpu().numpy())
        out[f"gdotdelta_s{seed}"] = np.stack(gdot)
        del deltas
    print(f"deltas done ({time.time() - t0:.0f}s)", flush=True)

    r = rates(model_key, n_layers, inter)
    out.update(crowd_base=crowd_base, crowd_A=crowd_A, partner_cos_base=pcos,
               partner_base=pidx, wnorm_base=wnorm_base, wnorm_A=wnorm_A,
               fisher_A=fisher_A, gradmag_A=gradmag_A,
               rate_math=r["math"], rate_code=r["code"])
    RESULTS.mkdir(exist_ok=True)
    np.savez(RESULTS / f"signals_{model_key}.npz", **out)

    # -- registered sanity checks
    rho_cf = np.median([stats.spearmanr(crowd_base[l], fisher_A[l]).statistic
                        for l in range(n_layers)])
    rho_bA = np.median([stats.spearmanr(crowd_base[l], crowd_A[l]).statistic
                        for l in range(n_layers)])
    print(f"\nsanity: median-layer Spearman(crowd_base, fisher_A) = {rho_cf:+.3f} "
          f"(escale Qwen/Gemma band 0.10-0.20; no TinyLlama escale reference exists, "
          f"and TinyLlama had the highest crowd-grad alignment in Paper 1)")
    print(f"        median-layer Spearman(crowd_base, crowd_A)  = {rho_bA:+.3f} "
          f"(A-training geometry movement)")
    print(f"saved signals_{model_key}.npz ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
