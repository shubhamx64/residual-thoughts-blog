"""E-M1 GPU runner (plan rev 3; PREREG stage-1 frozen 2026-07-12).

Modes (in required order; --pairs/--sketch refuse to run before the battery passes):
  --battery   10-product benchmark, symmetry, eps sweep, mixed-derivative sign,
              GGN diagonal, probe-half stability  -> results/battery_<model>.json
  --pairs     one ggn_vp per probe neuron -> within-layer interaction rows +
              cross-layer mass; hess_vp sensitivity on every 5th probe.
              Checkpointed every 50 probes (resumable).
  --sketch    K2 sketches: raw_s0, unit_s0, raw_s1 (TinyLlama only), U-statistic
              diagonal correction, m gates 32 -> 64 -> 128.

Model fp32 at ckpt_A, gradient checkpointing, probe = PREREG probe half (20 seqs).
"""
import argparse
import json
import time

import numpy as np
import torch

from common_m import (CKPT_A, DEV, PROBE_IDX, RESULTS, ckpt_B, eval_texts,
                      load_mlp_ckpt, mlp_key, n_layers_of, per_neuron_delta)
from ggn import (Perturb, encode, eps_for, ggn_vp, grad_of_loss, hess_vp,
                 loss_global)

EPS_REL = 1e-3          # PREREG stage-1 eps rule
CKPT_EVERY = 50


# ------------------------------------------------------------------ setup

def load_all(model_key, seed=0):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained({"tinyllama-1.1b": "TinyLlama/TinyLlama_v1.1",
                                         "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B"}[model_key])
    model = AutoModelForCausalLM.from_pretrained(
        {"tinyllama-1.1b": "TinyLlama/TinyLlama_v1.1",
         "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B"}[model_key], dtype=torch.float32).to(DEV)
    sd_A = load_mlp_ckpt(CKPT_A[model_key])
    model.load_state_dict({k: v.to(DEV, torch.float32) for k, v in sd_A.items()},
                          strict=False)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    for n, p in model.named_parameters():
        p.requires_grad_(".mlp." in n)
    model.eval()
    params = {n: p for n, p in model.named_parameters() if ".mlp." in n}
    sd_B = load_mlp_ckpt(ckpt_B(model_key, "baseline", seed))
    deltas = per_neuron_delta(sd_A, sd_B)           # [layer]{proj: fp32 cpu}
    batches = encode(tok, eval_texts("math", PROBE_IDX))
    return model, params, deltas, batches


def neuron_dir(deltas, l, i, unit=False):
    """Dense per-layer direction for neuron (l, i): zeros except its 3 slices."""
    d = deltas[l]
    v = {}
    g = torch.zeros_like(d["gate"]); g[i] = d["gate"][i]
    u = torch.zeros_like(d["up"]); u[i] = d["up"][i]
    w = torch.zeros_like(d["down"]); w[:, i] = d["down"][:, i]
    if unit:
        n = torch.sqrt((g[i] ** 2).sum() + (u[i] ** 2).sum() + (w[:, i] ** 2).sum())
        g, u, w = g / n, u / n, w / n
    v[mlp_key(l, "gate")] = g
    v[mlp_key(l, "up")] = u
    v[mlp_key(l, "down")] = w
    return v


def delta_norms(deltas):
    return np.stack([np.sqrt(((d["gate"] ** 2).sum(1) + (d["up"] ** 2).sum(1)
                              + (d["down"] ** 2).sum(0)).numpy())
                     for d in deltas])


def project_rows(g, deltas):
    """Project a gradient dict onto every neuron's delta slices.
    Returns (n_layers, inter) fp64: entry [l, j] = <g restricted to j, delta_j>."""
    out = []
    for l, d in enumerate(deltas):
        gg = g[mlp_key(l, "gate")]
        gu = g[mlp_key(l, "up")]
        gd = g[mlp_key(l, "down")]
        dg, du, dd = (d[k].to(gg.device) for k in ("gate", "up", "down"))
        row = ((gg.double() * dg.double()).sum(1)
               + (gu.double() * du.double()).sum(1)
               + (gd.double() * dd.double()).sum(0))
        out.append(row.cpu().numpy())
    return np.stack(out)


def probe_row(model, params, deltas, batches, l, i, estimator="ggn",
              eps_rel=EPS_REL, half=None):
    v = neuron_dir(deltas, l, i)
    eps = eps_for(v, model, eps_rel)
    b = batches if half is None else (batches[:10] if half == 0 else batches[10:])
    fn = ggn_vp if estimator == "ggn" else hess_vp
    g = fn(model, params, v, b, eps)
    return project_rows(g, deltas)


# ------------------------------------------------------------------ battery

def run_battery(model_key):
    ps = np.load(RESULTS / f"pairset_{model_key}.npz")
    model, params, deltas, batches = load_all(model_key)
    from scipy import stats
    checks, cache = [], {}

    def row_of(l, i, **kw):
        key = (l, i, kw.get("estimator", "ggn"), kw.get("eps_rel", EPS_REL),
               kw.get("half"))
        if key not in cache:
            cache[key] = probe_row(model, params, deltas, batches, l, i, **kw)
        return cache[key]

    def check(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    # 1. benchmark
    t0 = time.time()
    bench_pairs = [(int(ps["layer"][k]), int(ps["i"][k]))
                   for k in ps["val_sym"][:10]]
    for l, i in bench_pairs:
        row_of(l, i)
    per = (time.time() - t0) / 10
    n_probes = len(ps["probe_layer"])
    check("benchmark", True, f"{per:.1f}s/product -> pairs run "
          f"~{per * n_probes / 3600:.1f}h for {n_probes} probes")

    # 2. symmetry (50 registered pairs)
    sij, sji = [], []
    for k in ps["val_sym"]:
        l, i, j = int(ps["layer"][k]), int(ps["i"][k]), int(ps["j"][k])
        sij.append(row_of(l, i)[l, j])
        sji.append(row_of(l, j)[l, i])
    sij, sji = np.array(sij), np.array(sji)
    rho = stats.spearmanr(sij, sji).statistic
    gap = np.median(np.abs(sij - sji) / (np.maximum(np.abs(sij), np.abs(sji)) + 1e-30))
    check("symmetry", rho >= 0.9 and gap < 0.15,
          f"rho {rho:+.3f} (>=0.9), median rel gap {gap:.3f} (<0.15)")

    # 3. eps sweep on 10 neurons: row agreement at partner entries
    errs = []
    for k in ps["val_sym"][:10]:
        l, i, j = int(ps["layer"][k]), int(ps["i"][k]), int(ps["j"][k])
        r1 = row_of(l, i)[l]
        for r in (0.5e-3, 2e-3):
            r2 = row_of(l, i, eps_rel=r)[l]
            errs.append(abs(r2[j] - r1[j]) / (abs(r1[j]) + 1e-30))
    err = float(np.median(errs))
    check("eps-stability", err < 0.10, f"median rel err {err:.3f} (<0.10) over x0.5/x2")

    # 4. mixed-derivative sign (10 registered pairs, hess object, h sweep)
    agree, rho_pairs = 0, []
    for k in ps["val_mix"]:
        l, i, j = int(ps["layer"][k]), int(ps["i"][k]), int(ps["j"][k])
        s_h = row_of(l, i, estimator="hess")[l, j]
        di = neuron_dir(deltas, l, i, unit=True)
        dj = neuron_dir(deltas, l, j, unit=True)
        ni = float(np.sqrt(sum(float((t**2).sum()) for t in
                               neuron_dir(deltas, l, i).values())))
        nj = float(np.sqrt(sum(float((t**2).sum()) for t in
                               neuron_dir(deltas, l, j).values())))
        best = None
        for h in (1e-2, 3e-2, 1e-1):
            vals = {}
            # di and dj touch the SAME layer's matrices, so the two perturbations
            # must be applied as one combined direction per stencil corner
            for si in (+h, -h):
                for sj in (+h, -h):
                    vc = {k: si * di[k] + sj * dj[k] for k in di}
                    pc = Perturb(model, vc)
                    try:
                        pc.set(1.0)
                        with torch.no_grad():
                            vals[(si, sj)] = float(loss_global(model, batches))
                    finally:
                        pc.restore()
            mix = (vals[(h, h)] - vals[(h, -h)] - vals[(-h, h)]
                   + vals[(-h, -h)]) / (4 * h * h)
            if best is None or abs(mix) > abs(best):
                best = mix
        s_unit = s_h / (ni * nj)
        agree += int(np.sign(best) == np.sign(s_unit))
        rho_pairs.append((best, s_unit))
    a, b = zip(*rho_pairs)
    rr = stats.spearmanr(a, b).statistic
    check("mixed-derivative sign", agree >= 9 and rr > 0,
          f"sign agreement {agree}/10 (>=9), rank corr {rr:+.3f} (>0)")

    # 5. GGN diagonal (100 registered probes)
    neg = 0
    pl, pn = ps["probe_layer"], ps["probe_neuron"]
    for idx in ps["val_diag_probe_idx"]:
        l, i = int(pl[idx]), int(pn[idx])
        row = row_of(l, i)
        tol = 1e-3 * np.abs(row[l]).max()
        neg += int(row[l, i] < -tol)
    check("ggn diag >= 0", (100 - neg) >= 95, f"{100 - neg}/100 nonnegative (>=95)")
    hd = [row_of(int(pl[idx]), int(pn[idx]), estimator="hess")
          [int(pl[idx]), int(pn[idx])]
          for idx in ps["val_diag_probe_idx"][:20]]
    print(f"         (descriptive: hess diag positive {sum(x > 0 for x in hd)}/20)")

    # 6. probe-half stability: K_within from disjoint 10-seq halves, 10 neurons
    k0, k1 = [], []
    for idx in ps["val_diag_probe_idx"][:10]:
        l, i = int(pl[idx]), int(pn[idx])
        r0 = row_of(l, i, half=0)[l]
        r1 = row_of(l, i, half=1)[l]
        m = np.ones(len(r0), bool); m[i] = False
        k0.append(np.abs(r0[m]).sum())
        k1.append(np.abs(r1[m]).sum())
    rho_h = stats.spearmanr(k0, k1).statistic
    check("probe-half stability", rho_h >= 0.8, f"K rho {rho_h:+.3f} (>=0.8)")

    passed = all(c["ok"] for c in checks)
    out = {"model": model_key, "passed": passed, "s_per_product": per,
           "checks": checks}
    (RESULTS / f"battery_{model_key}.json").write_text(json.dumps(out, indent=1))
    print(f"\nBATTERY {'PASSED' if passed else 'FAILED'} "
          f"-> battery_{model_key}.json")
    return passed


# ------------------------------------------------------------------ pairs run

def run_pairs(model_key):
    gate = RESULTS / f"battery_{model_key}.json"
    assert gate.exists() and json.load(open(gate))["passed"], \
        "battery has not passed; refusing to run production (PREREG stage-1)"
    ps = np.load(RESULTS / f"pairset_{model_key}.npz")
    model, params, deltas, batches = load_all(model_key)
    pl, pn = ps["probe_layer"], ps["probe_neuron"]
    n_probes, n_layers = len(pl), len(deltas)
    inter = deltas[0]["gate"].shape[0]

    out_path = RESULTS / f"rows_{model_key}.npz"
    if out_path.exists():
        z = np.load(out_path)
        rows, xmass, hrows = z["rows"], z["xmass"], z["hrows"]
        done = z["done"]
    else:
        rows = np.zeros((n_probes, inter), np.float64)
        xmass = np.zeros((n_probes, n_layers), np.float64)
        hrows = np.full((n_probes, inter), np.nan, np.float64)
        done = np.zeros(n_probes, bool)

    dn = delta_norms(deltas)
    t0, n0 = time.time(), int(done.sum())
    for k in range(n_probes):
        if done[k]:
            continue
        l, i = int(pl[k]), int(pn[k])
        R = probe_row(model, params, deltas, batches, l, i)
        rows[k] = R[l]
        xmass[k] = np.abs(R).sum(1)                    # per-layer |s| mass
        if k % 5 == 0:                                 # hess sensitivity subsample
            hrows[k] = probe_row(model, params, deltas, batches, l, i,
                                 estimator="hess")[l]
        done[k] = True
        if (k + 1) % CKPT_EVERY == 0 or k == n_probes - 1:
            np.savez(out_path, rows=rows, xmass=xmass, hrows=hrows, done=done,
                     probe_layer=pl, probe_neuron=pn, delta_norms=dn)
            el = (time.time() - t0) / 60
            nd = int(done.sum())
            rate = (nd - n0) / max(el, 1e-9)
            print(f"  {nd}/{n_probes} probes ({el:.0f} min, "
                  f"{(n_probes - nd) / max(rate, 1e-9):.0f} min left)", flush=True)
    print(f"pairs run complete -> {out_path}")


# ------------------------------------------------------------------ sketch

class SketchPerturb:
    """Perturbation for full-model directions z (x) delta without a GPU-resident
    direction copy: snapshots on CPU, direction applied per-param on the fly."""

    def __init__(self, model, deltas_gpu, z):
        self.entries = []
        pd = dict(model.named_parameters())
        for l, d in enumerate(deltas_gpu):
            zl = z[l]
            self.entries.append((pd[mlp_key(l, "gate")], d["gate"], zl[:, None]))
            self.entries.append((pd[mlp_key(l, "up")], d["up"], zl[:, None]))
            self.entries.append((pd[mlp_key(l, "down")], d["down"], zl[None, :]))
        self.snap = [p.detach().clone().cpu() for p, _, _ in self.entries]

    @torch.no_grad()
    def set(self, scale):
        for (p, d, zl), s in zip(self.entries, self.snap):
            p.copy_(s.to(p.device))
            if scale != 0.0:
                p.add_((d.to(torch.float32) * zl) * scale)

    @torch.no_grad()
    def restore(self):
        for (p, _, _), s in zip(self.entries, self.snap):
            p.copy_(s.to(p.device))


def run_sketch(model_key):
    gate = RESULTS / f"battery_{model_key}.json"
    assert gate.exists() and json.load(open(gate))["passed"], \
        "battery has not passed; refusing to run production (PREREG stage-1)"
    model, params, deltas0, batches = load_all(model_key, seed=0)
    n_layers = len(deltas0)
    inter = deltas0[0]["gate"].shape[0]
    rng = np.random.default_rng(20260712)

    variants = [("raw_s0", 0, False), ("unit_s0", 0, True)]
    if model_key == "tinyllama-1.1b":
        variants.append(("raw_s1", 1, False))

    for name, seed, unit in variants:
        out_path = RESULTS / f"k2_{name}_{model_key}.npz"
        if out_path.exists():
            print(f"skip {name} (exists)")
            continue
        deltas = deltas0 if seed == 0 else per_neuron_delta(
            load_mlp_ckpt(CKPT_A[model_key]), load_mlp_ckpt(ckpt_B(model_key,
                                                                   "baseline", seed)))
        dn = delta_norms(deltas)
        if unit:
            deltas = [{k: d[k] / torch.tensor(dn[l] + 1e-30,
                                              dtype=torch.float32)[(slice(None), None)
                       if k != "down" else (None, slice(None))]
                       for k in d} for l, d in enumerate(deltas)]
        deltas_gpu = [{k: v.to(DEV, torch.bfloat16) for k, v in d.items()}
                      for d in deltas]
        # eps for a typical full direction (||Dz|| = ||D||_F for Rademacher z)
        vfro = float(np.sqrt(sum(float((d[k].to(torch.float32) ** 2).sum())
                                 for d in deltas_gpu for k in d)))
        thfro = float(np.sqrt(sum(float((p.detach() ** 2).sum())
                                  for p in params.values())))
        eps = EPS_REL * thfro / vfro
        sum_a = np.zeros((n_layers, inter))
        sum_a2 = np.zeros((n_layers, inter))
        sum_row = np.zeros((n_layers, inter))
        m_done, k2_prev = 0, None
        t0 = time.time()
        for m_target in (32, 64, 128):
            while m_done < m_target:
                z = rng.choice([-1.0, 1.0], size=(n_layers, inter))
                zt = [torch.tensor(z[l], dtype=torch.float32, device=DEV)
                      for l in range(n_layers)]
                pert = SketchPerturb(model, deltas_gpu, zt)
                g = ggn_vp(model, params, None, batches, eps, pert=pert)
                Sz = project_rows(g, deltas)           # (n_layers, inter)
                a = Sz * z
                sum_a += a
                sum_a2 += a * a
                sum_row += Sz * Sz
                m_done += 1
                if m_done % 8 == 0:
                    print(f"  {name}: {m_done} sketches "
                          f"({(time.time() - t0) / 60:.0f} min)", flush=True)
            mm = m_done
            diag2 = (sum_a ** 2 - sum_a2) / (mm * (mm - 1))   # U-statistic
            k2 = np.sqrt(np.maximum(0, sum_row / mm - diag2))
            if k2_prev is not None:
                from scipy import stats as st
                rho = np.median([st.spearmanr(k2[l], k2_prev[l]).statistic
                                 for l in range(n_layers)])
                k = int(0.2 * inter)
                jac = np.mean([len(set(np.argsort(-k2[l])[:k])
                                   & set(np.argsort(-k2_prev[l])[:k]))
                               / len(set(np.argsort(-k2[l])[:k])
                                     | set(np.argsort(-k2_prev[l])[:k]))
                               for l in range(n_layers)])
                print(f"  {name}: m={mm} vs m={mm // 2}: rho {rho:+.3f}, "
                      f"top-20% Jaccard {jac:.3f}", flush=True)
                if rho >= 0.9 and jac >= 0.8:
                    break
            k2_prev = k2.copy()
        np.savez(out_path, k2=k2, m=m_done, diag=sum_a / m_done,
                 delta_norms=dn)
        print(f"  {name}: done at m={m_done} -> {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tinyllama-1.1b")
    ap.add_argument("--battery", action="store_true")
    ap.add_argument("--pairs", action="store_true")
    ap.add_argument("--sketch", action="store_true")
    args = ap.parse_args()
    if args.battery:
        run_battery(args.model)
    if args.pairs:
        run_pairs(args.model)
    if args.sketch:
        run_sketch(args.model)
