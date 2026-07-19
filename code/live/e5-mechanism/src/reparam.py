"""E-M4: function-preserving rescaling (PREREG frozen 2026-07-14).

Builds reparameterized after-A checkpoints (up_i *= alpha_i, down[:, i] /=
alpha_i), runs the registered gates (logit equality 1e-3; crowding mask bitwise
unchanged), launches 8 short phase-B runs (weights + matched baseline per
alpha), analyzes E(alpha) stability, then deletes the reparam checkpoints.
"""
import json
import subprocess
import sys

import numpy as np
import torch

from common_m import (CKPT_A, DEV, E4, RESULTS, eval_texts, load_mlp_ckpt,
                      load_model, mlp_key, n_layers_of, topk_mask_per_layer)

MODEL_KEY = "tinyllama-1.1b"
PY = sys.executable
GATE_SEQS = [1, 2, 3, 4, 6]          # registered eval_math indices
VARIANTS = ("id", "a4", "a025", "lognorm")


def build_variant(sd_A, mask, name, n_layers, inter):
    sd = {k: v.clone() for k, v in sd_A.items()}
    rng = np.random.default_rng(20260714)
    for l in range(n_layers):
        if name == "id":
            alpha = np.ones(inter)
        elif name == "a4":
            alpha = np.where(mask[l], 4.0, 1.0)
        elif name == "a025":
            alpha = np.where(mask[l], 0.25, 1.0)
        else:
            # power-of-two alphas only: exact in bf16 (continuous log-uniform
            # compounds rounding to ~2.0 max logit error; amended per prereg)
            alpha = rng.choice([0.25, 0.5, 2.0, 4.0], inter)
        a = torch.tensor(alpha, dtype=torch.float32)
        up = sd[mlp_key(l, "up")].to(torch.float32) * a[:, None]
        dn = sd[mlp_key(l, "down")].to(torch.float32) / a[None, :]
        sd[mlp_key(l, "up")] = up.to(sd_A[mlp_key(l, "up")].dtype)
        sd[mlp_key(l, "down")] = dn.to(sd_A[mlp_key(l, "down")].dtype)
    return sd


def gates(sd_var, name, ref_mask=None):
    model, tok = load_model(MODEL_KEY, init_ckpt=CKPT_A[MODEL_KEY])
    texts = eval_texts("math", GATE_SEQS)
    ref = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True,
                      max_length=512)["input_ids"].to(DEV)
            ref.append(model(ids, use_cache=False).logits[0].float().cpu())
        model.load_state_dict({k: v.to(DEV) for k, v in sd_var.items()},
                              strict=False)
        mx = 0.0
        for t, r in zip(texts, ref):
            ids = tok(t, return_tensors="pt", truncation=True,
                      max_length=512)["input_ids"].to(DEV)
            z = model(ids, use_cache=False).logits[0].float().cpu()
            mx = max(mx, float((z - r).abs().max()))
    # crowding-mask invariance (fp32 geometry, bitwise vs mask_weights.npz)
    n_layers = n_layers_of(sd_var)
    crowd = []
    for l in range(n_layers):
        W = sd_var[mlp_key(l, "down")].to(DEV, torch.float32)
        Wn = W / (W.norm(dim=0, keepdim=True) + 1e-12)
        G = (Wn.T @ Wn).abs()
        G.fill_diagonal_(0)
        crowd.append(G.max(1).values.cpu().numpy())
    m_new = topk_mask_per_layer(np.stack(crowd), 0.20)
    del model
    torch.cuda.empty_cache()
    if ref_mask is None:                 # identity variant defines the reference
        print(f"  gate[{name}]: max|dlogit| {mx:.2e} (<1e-3); reference mask set")
        assert mx < 1e-3, f"E-M4 logit gate failed for {name}; STOP per prereg"
        return m_new
    # amended gate 2: vs identity mask (same source/precision); bitwise for
    # power-of-two alphas, >= 0.999 per-layer overlap for lognorm (bf16 rounding)
    # all variants now use power-of-two alphas -> bitwise is achievable
    mask_ok = all(np.array_equal(m_new[l], ref_mask[l]) for l in range(n_layers))
    detail = "bitwise vs identity"
    print(f"  gate[{name}]: max|dlogit| {mx:.2e} (<1e-3), mask {detail} "
          f"{'OK' if mask_ok else 'FAIL'}")
    assert mx < 1e-3 and mask_ok, f"E-M4 gate failed for {name}; STOP per prereg"
    return m_new


def run_arm(name, arm, init):
    tag = f"_m4_{name}"
    log = E4 / "results" / f"log_B_{arm}{tag}.jsonl"
    if log.exists() and sum(1 for _ in open(log)) >= 2:
        print(f"  skip {arm}{tag}")
        return
    cmd = [PY, str(E4 / "src" / "train_e4.py"), "--phase", "B", "--arm", arm,
           "--model", MODEL_KEY, "--seed", "0", "--init", str(init),
           "--steps", "150", "--tag-suffix", tag]
    subprocess.run(cmd, check=True)


def main():
    sd_A = load_mlp_ckpt(CKPT_A[MODEL_KEY])
    n_layers = n_layers_of(sd_A)
    inter = sd_A[mlp_key(0, "gate")].shape[0]
    mw = np.load(E4 / "data" / "mask_weights.npz")
    mask = {l: mw[f"L{l}"] for l in range(n_layers)}

    paths, ref_mask = {}, None
    for name in VARIANTS:
        p = RESULTS / f"_m4_ckpt_{name}.pt"
        paths[name] = p
        sd = build_variant(sd_A, mask, name, n_layers, inter)
        if name == "id":
            ref_mask = gates(sd, name, None)
        else:
            gates(sd, name, ref_mask)
        if not p.exists():
            torch.save(sd, p)
            print(f"  built {p.name}")

    for name in VARIANTS:
        for arm in ("baseline", "weights"):
            run_arm(name, arm, paths[name])

    # analysis: E(alpha) = deg_baseline - deg_weights at step 100 (NLL)
    def deg(arm, name):
        rr = [json.loads(l) for l in open(E4 / "results" /
                                          f"log_B_{arm}_m4_{name}.jsonl")]
        r0 = next(r for r in rr if r["step"] == 0)
        r1 = next(r for r in rr if r["step"] == 100)
        return (np.log(r1["ppl_math"]) - np.log(r0["ppl_math"]),
                np.log(r1["ppl_code"]) - np.log(r0["ppl_code"]))

    out = {}
    for name in VARIANTS:
        db, cb = deg("baseline", name)
        dw, cw = deg("weights", name)
        out[name] = {"E": db - dw, "deg_base": db, "deg_weights": dw,
                     "code_base": cb, "code_weights": cw}
    E1 = out["id"]["E"]
    print(f"\n{'variant':>8} | E(alpha) | rel to E(1) | deg_base | code_w")
    stable = True
    for name in VARIANTS:
        rel = out[name]["E"] / E1 - 1
        if name != "id":
            stable &= abs(rel) <= 0.25
        print(f"{name:>8} | {out[name]['E']:+.4f} | {rel:+.1%} | "
              f"{out[name]['deg_base']:+.4f} | {out[name]['code_weights']:+.4f}")
    print(f"\nVERDICT: protection effect "
          f"{'STABLE within the registered +-25% band -> computation-geometric' if stable else 'OUTSIDE the +-25% band -> entangled with optimizer coordinates (scope condition)'}")
    json.dump(out, open(RESULTS / "reparam_m4.json", "w"), indent=1)
    for p in paths.values():
        p.unlink(missing_ok=True)
    print("reparam checkpoints deleted (disk)")


if __name__ == "__main__":
    main()
