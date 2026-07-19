"""E-M3: neuron rollback (checkpoint surgery + eval). PREREG v1.

For bucket S: model = after-B (baseline arm), splice S's neurons back to after-A,
eval held-out math/code NLL. Recovery R(S) = (L_B - L_splice) / (L_B - L_A).

--check : full-splice control only (bitwise + NLL), seed 0
--grid  : full control + 6 buckets x seeds 0-4 + 15 quintile splices (seed 0)

Abort rule (PREREG): any full-splice control failure aborts the grid.
Raw rows are flushed to results/rollback_<model>.json after every splice.
"""
import argparse
import json
import time

import numpy as np
import torch

from common_m import (CKPT_A, DEV, E4, RESULTS, ckpt_B, eval_nll, eval_texts,
                      load_mask_npz, load_mlp_ckpt, load_model, load_signals,
                      mlp_key, n_layers_of, splice, topk_mask_per_layer)

BUCKETS = ("weights", "fisher", "footprint", "join", "random", "updnorm")
QUINTILE_SIGNALS = ("crowd_base", "fisher_A", "upd_norm_s0")


def full_mask(n_layers, inter):
    return {l: np.ones(inter, bool) for l in range(n_layers)}


def bucket_mask(model_key, bucket, seed, signals, repair=False):
    mask_dir = E4 / "data" if model_key == "tinyllama-1.1b" else E4 / "data" / model_key
    if bucket == "updnorm":
        return topk_mask_per_layer(signals["upd_norm_s0"], 0.20)
    if bucket == "random":
        name = f"mask_random_s{seed}.npz" if seed else "mask_random.npz"
        return load_mask_npz(mask_dir / name)
    if bucket == "fisher" and repair:
        return load_mask_npz(mask_dir / "mask_fisher_repair.npz")
    return load_mask_npz(mask_dir / f"mask_{bucket}.npz")


def quintile_masks(signal):
    """5 disjoint per-layer quintile masks (q0 = lowest 20%, q4 = top 20%)."""
    n_layers, inter = signal.shape
    k = inter // 5
    out = []
    for q in range(5):
        sel = {}
        for l in range(n_layers):
            order = np.argsort(signal[l])          # ascending
            m = np.zeros(inter, bool)
            m[order[q * k: (q + 1) * k]] = True
            sel[l] = m
        out.append(sel)
    return out


@torch.no_grad()
def bitwise_equal(model, sd_ref):
    bad = []
    for l in range(n_layers_of(sd_ref)):
        mlp = model.model.layers[l].mlp
        for proj, p in (("down", mlp.down_proj.weight),
                        ("gate", mlp.gate_proj.weight),
                        ("up", mlp.up_proj.weight)):
            if not torch.equal(p.detach().cpu(), sd_ref[mlp_key(l, proj)]):
                bad.append(mlp_key(l, proj))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tinyllama-1.1b")
    ap.add_argument("--check", action="store_true", help="full-splice control only")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--repair", action="store_true",
                    help="Qwen protocol-repair addendum: repair ckpts, "
                         "mask_fisher_repair, inline updnorm, no quintiles")
    ap.add_argument("--buckets", default=",".join(BUCKETS),
                    help="comma list; subset reruns write to --out-suffix file")
    ap.add_argument("--out-suffix", default="",
                    help="suffix for the output json (protects the main grid file)")
    args = ap.parse_args()
    model_key = args.model
    seeds = [int(s) for s in args.seeds.split(",")]
    if model_key == "qwen2.5-1.5b":
        seeds = [0]  # s1+ after-B checkpoints do not exist (PREREG)

    t0 = time.time()
    if args.repair:
        assert model_key == "qwen2.5-1.5b", "--repair is the Qwen addendum"
        ckpt_a_path = E4 / "results" / "ckpt_A_qwen_repair.pt"
        sd_A = load_mlp_ckpt(ckpt_a_path)
    else:
        sd_A = load_mlp_ckpt(CKPT_A[model_key])
    n_layers = n_layers_of(sd_A)
    inter = sd_A[mlp_key(0, "gate")].shape[0]
    math_texts = eval_texts("math")
    code_texts = eval_texts("code")

    model, tok = load_model(model_key)
    n_mlp_tensors = 3 * n_layers

    rows, L_A = [], None
    out_path = RESULTS / (f"rollback_{model_key}_repair.json" if args.repair
                          else f"rollback_{model_key}{args.out_suffix}.json")

    def flush():
        out_path.write_text(json.dumps(
            {"model": model_key, "L_A_math": L_A[0] if L_A else None,
             "L_A_code": L_A[1] if L_A else None, "rows": rows}, indent=1))

    def run_control(seed, sd_B):
        splice(model, sd_A, full_mask(n_layers, inter))
        bad = bitwise_equal(model, sd_A)
        ok = not bad
        nll_math = eval_nll(model, tok, math_texts)
        nll_code = eval_nll(model, tok, code_texts)
        print(f"  seed {seed} full-splice bitwise "
              f"{'OK' if ok else 'FAIL'} ({n_mlp_tensors - len(bad)}/{n_mlp_tensors}"
              f" MLP tensors); NLL math {nll_math[0]:.4f}", flush=True)
        if not ok:
            flush()
            raise SystemExit(f"ABORT: full-splice bitwise control failed: {bad[:5]}")
        # restore after-B and verify the restore path too
        splice(model, sd_B, full_mask(n_layers, inter))
        bad = bitwise_equal(model, sd_B)
        if bad:
            flush()
            raise SystemExit(f"ABORT: after-B restore not bitwise: {bad[:5]}")
        return nll_math, nll_code

    def run_splice(seed, name, sel, sd_B, L_B):
        splice(model, sd_A, sel)
        lm, pm = eval_nll(model, tok, math_texts)
        lc, pc = eval_nll(model, tok, code_texts)
        splice(model, sd_B, sel)  # restore
        R = (L_B[0] - lm) / (L_B[0] - L_A[0])
        row = {"seed": seed, "bucket": name,
               "L_math": lm, "ppl_math": pm, "L_code": lc, "ppl_code": pc,
               "R": R, "C": lc - L_B[1],
               "n_neurons": int(sum(np.sum(m) for m in sel.values()))}
        rows.append(row)
        flush()
        print(f"  seed {seed} {name:>22}: R={R:+.3f} L_math={lm:.4f} "
              f"C={row['C']:+.4f}", flush=True)
        return row

    signals = None if args.check else load_signals(model_key)

    for seed in seeds:
        if args.repair:
            sd_B = load_mlp_ckpt(E4 / "results" / "ckpt_B_baseline_qwen_repair_rb.pt")
        else:
            sd_B = load_mlp_ckpt(ckpt_B(model_key, "baseline", seed))
        # updnorm bucket is SEED-MATCHED: computed from this seed's own delta
        # (audit repair: the original grid reused the seed-0 mask for all seeds)
        from common_m import neuron_norms, per_neuron_delta
        signals = dict(signals or {})
        signals["upd_norm_s0"] = neuron_norms(per_neuron_delta(sd_A, sd_B))
        with torch.no_grad():
            splice(model, sd_B, full_mask(n_layers, inter))
        lb_m = eval_nll(model, tok, math_texts)
        lb_c = eval_nll(model, tok, code_texts)
        print(f"seed {seed}: after-B math NLL {lb_m[0]:.4f} ppl {lb_m[1]:.2f}",
              flush=True)
        nll_math_A, nll_code_A = run_control(seed, sd_B)
        if L_A is None:
            L_A = (nll_math_A[0], nll_code_A[0])
            print(f"after-A math NLL {L_A[0]:.4f} ppl {nll_math_A[1]:.2f}", flush=True)
        if args.check:
            print("check complete")
            return
        L_B = (lb_m[0], lb_c[0])
        rows.append({"seed": seed, "bucket": "_afterB", "L_math": lb_m[0],
                     "ppl_math": lb_m[1], "L_code": lb_c[0], "ppl_code": lb_c[1],
                     "R": 0.0, "C": 0.0, "n_neurons": 0})
        for bucket in args.buckets.split(","):
            run_splice(seed, bucket,
                       bucket_mask(model_key, bucket, seed, signals, args.repair),
                       sd_B, L_B)
        if seed == 0 and not args.repair:
            for sig in QUINTILE_SIGNALS:
                for q, sel in enumerate(quintile_masks(signals[sig])):
                    run_splice(seed, f"q{q}_{sig}", sel, sd_B, L_B)

    flush()
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
