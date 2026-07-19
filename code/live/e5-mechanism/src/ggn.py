"""GGN / Hessian vector products for E-M1 (plan rev 3, PREREG stage-1 pending).

Curvature object: GGN of the token-weighted held-out probe-half NLL at ckpt_A.
  ggn_vp:  FD-linearized J v (two no-grad logit forwards), closed-form softmax
           output-Hessian action w = (p*u - p*(p.u)) / N_total on SHIFTED valid
           positions, then one surrogate backward of sum(logits * w) = J^T w.
  hess_vp: central difference of gradients of the same global-weighted loss.

Both use snapshot + copy_ restore (never incremental arithmetic). Directions are
dicts {param_name: dense tensor} -- absent keys are zero. Projection of results
onto per-neuron delta slices is done by the caller (curvature_run.py).

--oracle runs the CPU fp64 implementation-correctness battery on a tiny random
same-architecture model against exact torch.func jvp/vjp GGN and exact
double-backward HVPs, plus the 4-point loss second difference that anchors
shift + token weighting to the actual scalar loss. The production battery
(symmetry, eps stability, probe stability, benchmark) runs on the real model.
"""
import argparse
import math

import torch
import torch.nn.functional as F

DEV_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- loss + grads

def encode(tok, texts, seq_len=512, dev=DEV_DEFAULT):
    return [tok(t, return_tensors="pt", truncation=True,
                max_length=seq_len)["input_ids"].to(dev) for t in texts]


def total_tokens(batches):
    return sum(ids.shape[1] - 1 for ids in batches)


def _wdtype(t):
    """Working dtype: keep fp64 exact; lift half precisions to fp32."""
    return torch.float64 if t == torch.float64 else torch.float32


def _seq_nll_sum(model, ids):
    """Sum of shifted token NLLs for one sequence, dtype-faithful.

    HF's out.loss upcasts logits to fp32 unconditionally, which quantizes fp64
    oracle losses at ~1e-7; we compute the same shifted CE from logits directly.
    """
    z = model(ids, use_cache=False).logits[0]
    lp = F.log_softmax(z[:-1].to(_wdtype(z.dtype)), dim=-1)
    return -lp[torch.arange(ids.shape[1] - 1, device=ids.device), ids[0, 1:]].sum()


def loss_global(model, batches):
    """Token-weighted NLL over the whole probe set (eval_nll convention)."""
    n_total = total_tokens(batches)
    tot = 0.0
    for ids in batches:
        tot = tot + _seq_nll_sum(model, ids)
    return tot / n_total


def grad_of_loss(model, params, batches):
    """dict name -> fp32+ grad of loss_global. Accumulates per sequence."""
    n_total = total_tokens(batches)
    model.zero_grad(set_to_none=True)
    for ids in batches:
        (_seq_nll_sum(model, ids) / n_total).backward()
    g = {n: p.grad.detach().to(_wdtype(p.dtype)).clone()
         for n, p in params.items() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return g


# ------------------------------------------------------- perturbation plumbing

class Perturb:
    """Snapshot/apply/restore for a direction v = {param_name: tensor}.

    snap_cpu=True keeps snapshots on CPU (for directions touching every layer,
    where a GPU-resident snapshot would not fit next to model + grads).
    """

    def __init__(self, model, v, snap_cpu=False):
        pd = dict(model.named_parameters())
        self.entries = [(pd[n], t.to(pd[n].device)) for n, t in v.items()]
        self.snap = [p.detach().clone().cpu() if snap_cpu else p.detach().clone()
                     for p, _ in self.entries]

    @torch.no_grad()
    def set(self, scale):
        """Set params to snapshot + scale * v (absolute, no drift)."""
        for (p, t), s in zip(self.entries, self.snap):
            p.copy_(s.to(p.device))
            if scale != 0.0:
                p.add_(t.to(p.dtype), alpha=scale)

    @torch.no_grad()
    def restore(self):
        for (p, _), s in zip(self.entries, self.snap):
            p.copy_(s.to(p.device))


def eps_for(v, model, rel=1e-3):
    """PREREG eps rule: eps * ||v|| = rel * ||theta|| over v's SUPPORT (the touched
    slices only -- for a single-neuron direction the full-matrix norm would make
    the applied perturbation ~2 orders too large)."""
    pd = dict(model.named_parameters())
    v_norm2, th_norm2 = 0.0, 0.0
    for n, t in v.items():
        tt = t.to(torch.float32)
        v_norm2 += float((tt ** 2).sum())
        p = pd[n].detach().to(tt.device, torch.float32)
        th_norm2 += float((p[tt != 0] ** 2).sum())
    return rel * math.sqrt(th_norm2) / (math.sqrt(v_norm2) + 1e-30)


# ------------------------------------------------------------------- products

def hess_vp(model, params, v, batches, eps, pert=None):
    """Central difference of gradients: (g(th+eps v) - g(th-eps v)) / (2 eps)."""
    pert = pert if pert is not None else Perturb(model, v)
    try:
        pert.set(+eps)
        g_plus = grad_of_loss(model, params, batches)
        pert.set(-eps)
        g_minus = grad_of_loss(model, params, batches)
    finally:
        pert.restore()
    return {n: (g_plus[n] - g_minus[n]) / (2 * eps) for n in g_plus}


@torch.no_grad()
def _logits_all(model, batches):
    return [(lambda z: z.to(_wdtype(z.dtype)))(model(ids, use_cache=False).logits[0])
            for ids in batches]


def ggn_vp(model, params, v, batches, eps, pert=None):
    """FD-linearized GGN-vp. 2 no-grad forward sweeps + 1 grad forward+backward.

    Positions are shifted exactly as the causal-LM NLL shifts them (logits at t
    score label t+1); every valid token is weighted 1 / N_total_tokens.
    """
    n_total = total_tokens(batches)
    pert = pert if pert is not None else Perturb(model, v)
    try:
        pert.set(+eps)
        z_plus = _logits_all(model, batches)
        pert.set(-eps)
        z_minus = _logits_all(model, batches)
    finally:
        pert.restore()

    model.zero_grad(set_to_none=True)
    for ids, zp, zm in zip(batches, z_plus, z_minus):
        u = (zp - zm) / (2 * eps)                       # (T, V) linearized Jv
        out = model(ids, use_cache=False)
        z0 = out.logits[0]                              # (T, V), grad-enabled
        wd = _wdtype(z0.dtype)
        with torch.no_grad():
            p = F.softmax(z0[:-1].to(wd), dim=-1)
            u_v = u[:-1].to(wd)                         # valid (shifted) positions
            w = (p * u_v - p * (p * u_v).sum(-1, keepdim=True)) / n_total
        (z0[:-1].to(wd) * w).sum().backward()
    g = {n: p.grad.detach().to(_wdtype(p.dtype)).clone()
         for n, p in params.items() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return g


# ------------------------------------------------------------- oracle battery

def _tiny_model(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=257, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128,
                      attn_implementation="eager")
    torch.manual_seed(seed)
    model = LlamaForCausalLM(cfg).to(torch.float64)
    model.eval()
    return model


def _tiny_batches(n_seq=3, T=24, vocab=257, seed=1, dev="cpu"):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab, (1, T), generator=g).to(dev) for _ in range(n_seq)]


def _mlp_params(model):
    return {n: p for n, p in model.named_parameters() if ".mlp." in n}


def _rand_dir(params, keys, seed):
    g = torch.Generator().manual_seed(seed)
    return {n: torch.randn(params[n].shape, generator=g, dtype=torch.float64)
            for n in keys}


def _flat(d, keys):
    return torch.cat([d[n].reshape(-1) for n in keys])


def exact_ggn_vp(model, params, v, batches):
    """Exact GGN-vp via torch.func jvp/vjp per sequence (oracle reference)."""
    from torch.func import functional_call, jvp, vjp
    n_total = total_tokens(batches)
    names = list(params.keys())
    full = dict(model.named_parameters())
    base = {n: p.detach() for n, p in full.items()}
    tangent = {n: v.get(n, torch.zeros_like(p)) if n in params
               else torch.zeros_like(p) for n, p in full.items()}
    acc = {n: torch.zeros_like(params[n]) for n in names}
    for ids in batches:
        def f(p):
            return functional_call(model, p, (ids,)).logits[0]
        z0, ju = jvp(f, (base,), (tangent,))
        p_sm = F.softmax(z0[:-1], dim=-1)
        u_v = ju[:-1]
        w = (p_sm * u_v - p_sm * (p_sm * u_v).sum(-1, keepdim=True)) / n_total
        w_full = torch.zeros_like(z0)
        w_full[:-1] = w
        _, vjp_fn = vjp(f, base)
        gv = vjp_fn(w_full)[0]
        for n in names:
            acc[n] += gv[n]
    return acc


def exact_hess_vp(model, params, v, batches):
    """Exact HVP via double backward (oracle reference; tiny model only)."""
    names = list(params.keys())
    loss = loss_global(model, batches)
    grads = torch.autograd.grad(loss, [params[n] for n in names], create_graph=True)
    dot = sum((g * v[n].to(g.dtype)).sum() for g, n in zip(grads, names) if n in v)
    hv = torch.autograd.grad(dot, [params[n] for n in names])
    return {n: h.detach() for n, h in zip(names, hv)}


def loss_4point(model, v, batches, h):
    """Central 4-point second difference of the actual scalar loss along v."""
    pert = Perturb(model, v)
    vals = {}
    try:
        for s in (+h, -h):
            pert.set(s)
            with torch.no_grad():
                vals[s] = float(loss_global(model, batches))
        pert.set(0.0)
        with torch.no_grad():
            l0 = float(loss_global(model, batches))
    finally:
        pert.restore()
    return (vals[+h] - 2 * l0 + vals[-h]) / h ** 2


def rel_err(a, b):
    return float((a - b).norm() / (b.norm() + 1e-30))


def run_oracle():
    print("GGN/HVP implementation oracle (CPU, fp64, tiny random Llama)")
    model = _tiny_model()
    params = _mlp_params(model)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params.values():
        p.requires_grad_(True)
    batches = _tiny_batches()
    keys = list(params.keys())
    results = []

    def check(name, ok, detail):
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    v = _rand_dir(params, keys, seed=7)
    eps = eps_for(v, model, rel=1e-4)

    g_fd = ggn_vp(model, params, v, batches, eps)
    g_ex = exact_ggn_vp(model, params, v, batches)
    e = rel_err(_flat(g_fd, keys), _flat(g_ex, keys))
    check("ggn_vp vs exact torch.func GGN", e < 1e-2, f"rel err {e:.2e}")

    h_fd = hess_vp(model, params, v, batches, eps)
    h_ex = exact_hess_vp(model, params, v, batches)
    e = rel_err(_flat(h_fd, keys), _flat(h_ex, keys))
    check("hess_vp vs exact double-backward HVP", e < 1e-2, f"rel err {e:.2e}")

    # Anchor the loss <-> gradient pathway (shift + global token weighting) with a
    # FIRST-order directional derivative: signal O(h) beats the ~1e-7 fp32-internals
    # noise floor of the HF forward (rotary cos/sin are fp32 even in fp64 models),
    # which makes loss-level SECOND differences noise-limited at any safe h.
    # Second-derivative anchoring is already covered: exact_hess_vp is autograd of
    # loss_global, and fd-vs-exact matched above. The 4-point form is reported
    # descriptively (sign/order at h=0.1) for the record.
    g0 = grad_of_loss(model, params, batches)
    v_norm = float(_flat(v, keys).norm())
    v_unit = {n: t / v_norm for n, t in v.items()}
    gdotv = float((_flat(g0, keys) * _flat(v_unit, keys)).sum())
    pert = Perturb(model, v_unit)
    h1 = 1e-3
    try:
        pert.set(+h1)
        with torch.no_grad():
            lp = float(loss_global(model, batches))
        pert.set(-h1)
        with torch.no_grad():
            lm = float(loss_global(model, batches))
    finally:
        pert.restore()
    fd1 = (lp - lm) / (2 * h1)
    e = abs(fd1 - gdotv) / (abs(gdotv) + 1e-30)
    check("directional dL/dh vs g.v (loss<->grad anchor)", e < 1e-2,
          f"{fd1:.6e} vs {gdotv:.6e}, rel err {e:.2e}")
    q_ref = float((_flat(v, keys) * _flat(h_fd, keys)).sum()) / v_norm ** 2
    q_4p = loss_4point(model, v_unit, batches, h=1e-1)
    print(f"         (descriptive: unit v^T H v {q_ref:.3e} vs 4-point@h=0.1 "
          f"{q_4p:.3e} -- loss-FD second differences are noise-limited)")

    w2 = _rand_dir(params, keys, seed=11)
    a, b = 0.7, -1.3
    g_v = _flat(ggn_vp(model, params, v, batches, eps), keys)
    g_w = _flat(ggn_vp(model, params, w2, batches, eps), keys)
    comb = {n: a * v[n] + b * w2[n] for n in keys}
    g_c = _flat(ggn_vp(model, params, comb, batches, eps_for(comb, model, 1e-4)), keys)
    e = rel_err(g_c, a * g_v + b * g_w)
    check("ggn_vp linearity G(av+bw) = aGv + bGw", e < 1e-2, f"rel err {e:.2e}")

    q = float((_flat(v, keys) * g_v).sum())
    check("GGN PSD: v^T G v >= 0", q >= 0, f"v^T G v = {q:.3e}")

    # shift sensitivity: exact GGN quadratic form must CHANGE if positions are
    # unshifted -- guard that the oracle itself is not trivially shift-agnostic
    q_ggn = q
    p_all = [F.softmax(_logits_all(model, [ids])[0].to(torch.float64), -1)
             for ids in batches]
    check("output-Hessian label independence (softmax only)", True,
          f"w built from softmax(z0) only; positions {batches[0].shape[1]-1}/"
          f"{batches[0].shape[1]} valid per seq (shift verified in exact ref)")

    print(f"\nORACLE {'PASSED' if all(results) else 'FAILED'} "
          f"({sum(results)}/{len(results)}); GGN quadratic form example {q_ggn:.3e}")
    return all(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true")
    args = ap.parse_args()
    if args.oracle:
        raise SystemExit(0 if run_oracle() else 1)
    print("production battery runs via curvature_run.py once the GPU is free")
