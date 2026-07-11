"""
Cross-layer head composition map for Gemma-2-2b, entirely in weight space.

For every writer head w in layer Lw and reader head r in layer Lr > Lw,
computes the three Elhage et al. (2021) composition scores, extended with
Gemma-2's norm structure:

    Q-comp(r, w) = ||W_QK^r{}^T W_OV^w||_F / (||W_QK^r||_F ||W_OV^w||_F)
    K-comp(r, w) = ||W_QK^r     W_OV^w||_F / (||W_QK^r||_F ||W_OV^w||_F)
    V-comp(r, w) = ||W_OV^r     W_OV^w||_F / (||W_OV^r||_F ||W_OV^w||_F)

where, per head h with KV group g:
    W_QK = (diag(1+g_in) W_Q[h]^T)(W_K[g] diag(1+g_in))   [reader's own g_in]
    W_OV = diag(1+g_post) W_O[:, h] W_V[g] diag(1+g_in)   [writer's sandwich
           post_attention_layernorm gain folded on the output side]

Per-token rms scalars are not representable in weight space and are omitted
(they rescale rows/columns uniformly per token, leaving Frobenius ratios
approximately invariant). RoPE is evaluated at delta=0 by default; pass
--delta N to rotate the reader's key factor by R(-N) for K-composition.

All 2304x2304 circuit matrices stay factored as A = a1^T a2 with
a1, a2 in R^{256 x 2304}; products and norms reduce to 256x256 algebra:
    ||A||_F^2      = sum( (a1 a1^T) * (a2 a2^T) )
    ||A B||_F^2    = tr( C^T G_a1 C G_b2 ),  C = a2 b1^T
A random-weights null floor (i.i.d. Gaussian factors, same shapes) is
computed empirically; for rank-256 products in d=2304 it sits near
sqrt(256/2304) ~ 0.33 for raw norm ratios -- scores are reported raw AND
null-normalized.

2-hop: --two-hop composes the top --two-hop-pairs writer->middle pairs by
V-composition into virtual heads W_OV^m W_OV^w (rank <= 256, exact factors),
then scores their K-composition into every later reader.

Usage:
    python composition_map.py [--delta 0] [--two-hop] [--out analysis_outputs/composition_map]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from weight_diff_it import ShardedWeights

NUM_LAYERS = 26
NUM_Q_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = 256
HIDDEN = 2304
N_HEADS_TOTAL = NUM_LAYERS * NUM_Q_HEADS


def head_label(idx: int) -> str:
    return f"L{idx // NUM_Q_HEADS}H{idx % NUM_Q_HEADS}"


def build_factors(repo_id: str, device: str, delta: int = 0):
    """
    Per query head factors, [N_HEADS_TOTAL, 256, 2304] each:
      Qf, Kf, Vf : gamma_in-folded projections (Kf optionally RoPE-rotated
                   by R(-delta) for K-composition distance modeling)
      OvL        : (diag(1+gamma_post) W_O[:, h])^T  -- writer output side
    """
    w = ShardedWeights(repo_id)
    Qf = torch.zeros(N_HEADS_TOTAL, HEAD_DIM, HIDDEN)
    Kf = torch.zeros(N_HEADS_TOTAL, HEAD_DIM, HIDDEN)
    Vf = torch.zeros(N_HEADS_TOTAL, HEAD_DIM, HIDDEN)
    OvL = torch.zeros(N_HEADS_TOTAL, HEAD_DIM, HIDDEN)

    rot = None
    if delta != 0:
        from rope_utils import compute_rotation_matrix, compute_rope_frequencies
        from config import GEMMA2_CONFIG
        freqs = compute_rope_frequencies(GEMMA2_CONFIG)
        rot = compute_rotation_matrix(-delta, freqs).float()  # [256, 256]

    for L in range(NUM_LAYERS):
        pre = f"model.layers.{L}."
        g_in = 1.0 + w.get(pre + "input_layernorm.weight")
        g_post = 1.0 + w.get(pre + "post_attention_layernorm.weight")
        Wq = w.get(pre + "self_attn.q_proj.weight")   # [2048, 2304]
        Wk = w.get(pre + "self_attn.k_proj.weight")   # [1024, 2304]
        Wv = w.get(pre + "self_attn.v_proj.weight")   # [1024, 2304]
        Wo = w.get(pre + "self_attn.o_proj.weight")   # [2304, 2048]
        for h in range(NUM_Q_HEADS):
            g = h // (NUM_Q_HEADS // NUM_KV_HEADS)
            i = L * NUM_Q_HEADS + h
            Qf[i] = Wq[h * HEAD_DIM:(h + 1) * HEAD_DIM] * g_in
            k = Wk[g * HEAD_DIM:(g + 1) * HEAD_DIM] * g_in
            Kf[i] = rot @ k if rot is not None else k
            Vf[i] = Wv[g * HEAD_DIM:(g + 1) * HEAD_DIM] * g_in
            OvL[i] = (Wo[:, h * HEAD_DIM:(h + 1) * HEAD_DIM] * g_post.unsqueeze(1)).T
    return (Qf.to(device), Kf.to(device), Vf.to(device), OvL.to(device))


def grams(F):
    """[N, 256, 2304] -> [N, 256, 256] Gram matrices a a^T."""
    return torch.bmm(F, F.transpose(1, 2))


def pair_norm_sq(G1, G2):
    """||a1^T a2||^2 per head: sum(G1 * G2) over last two dims."""
    return (G1 * G2).sum(dim=(-2, -1))


def block_scores(reader_factor, G_a1_r, C, G_b2_w):
    """
    score^2 = tr(C^T G_a1 C G_b2) for a [R, W, 256, 256] block of C.
    G_a1_r: [R, 256, 256], G_b2_w: [W, 256, 256].
    """
    T1 = torch.einsum("rij,rwjk->rwik", G_a1_r, C)
    T2 = torch.einsum("rwik,wkl->rwil", T1, G_b2_w)
    return (C * T2).sum(dim=(-2, -1)).clamp(min=0)


def cross_block(A_r, B_w):
    """C = a2 b1^T for all head pairs: [R, W, 256, 256]."""
    R, W = A_r.shape[0], B_w.shape[0]
    flat = A_r.reshape(R * HEAD_DIM, HIDDEN) @ B_w.reshape(W * HEAD_DIM, HIDDEN).T
    return flat.view(R, HEAD_DIM, W, HEAD_DIM).permute(0, 2, 1, 3).contiguous()


def compute_map(Qf, Kf, Vf, OvL, device):
    """Returns dict of [N_HEADS_TOTAL(reader), N_HEADS_TOTAL(writer)] arrays (NaN where Lr <= Lw)."""
    G_q, G_k, G_v, G_o = grams(Qf), grams(Kf), grams(Vf), grams(OvL)
    norm_qk = torch.sqrt(pair_norm_sq(G_q, G_k))    # [N]
    norm_ov = torch.sqrt(pair_norm_sq(G_o, G_v))    # [N]

    out = {k: torch.full((N_HEADS_TOTAL, N_HEADS_TOTAL), float("nan"))
           for k in ["qcomp", "kcomp", "vcomp"]}

    for Lr in range(1, NUM_LAYERS):
        r_sl = slice(Lr * NUM_Q_HEADS, (Lr + 1) * NUM_Q_HEADS)
        w_sl = slice(0, Lr * NUM_Q_HEADS)  # all earlier layers at once
        OvL_w, Vf_w = OvL[w_sl], Vf[w_sl]
        G_v_w = G_v[w_sl]
        denom_w = norm_ov[w_sl]

        # Q-comp: A' = W_QK^T -> a1 = Kf_r, a2 = Qf_r ; C = Qf_r OvL_w^T
        C = cross_block(Qf[r_sl], OvL_w)
        s2 = block_scores(None, G_k[r_sl], C, G_v_w)
        out["qcomp"][r_sl, w_sl] = (torch.sqrt(s2) / (norm_qk[r_sl, None] * denom_w[None, :] + 1e-12)).cpu()

        # K-comp: A = W_QK -> a1 = Qf_r, a2 = Kf_r ; C = Kf_r OvL_w^T
        C = cross_block(Kf[r_sl], OvL_w)
        s2 = block_scores(None, G_q[r_sl], C, G_v_w)
        out["kcomp"][r_sl, w_sl] = (torch.sqrt(s2) / (norm_qk[r_sl, None] * denom_w[None, :] + 1e-12)).cpu()

        # V-comp: A = W_OV^r -> a1 = OvL_r, a2 = Vf_r ; C = Vf_r OvL_w^T
        C = cross_block(Vf[r_sl], OvL_w)
        s2 = block_scores(None, G_o[r_sl], C, G_v_w)
        out["vcomp"][r_sl, w_sl] = (torch.sqrt(s2) / (norm_ov[r_sl, None] * denom_w[None, :] + 1e-12)).cpu()

        print(f"reader layer {Lr:2d} done ({Lr * NUM_Q_HEADS} writers)", flush=True)

    return {k: v.numpy() for k, v in out.items()}, norm_qk.cpu().numpy(), norm_ov.cpu().numpy()


def random_null(device, n_pairs=256, seed=0):
    """Empirical composition floor for i.i.d. Gaussian factors of the same shapes."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    def rand_factor(n):
        return torch.randn(n, HEAD_DIM, HIDDEN, generator=g).to(device)
    n = max(8, int(np.sqrt(n_pairs)))
    a1, a2 = rand_factor(n), rand_factor(n)
    b1, b2 = rand_factor(n), rand_factor(n)
    G_a1, G_a2, G_b1, G_b2 = grams(a1), grams(a2), grams(b1), grams(b2)
    norm_a = torch.sqrt(pair_norm_sq(G_a1, G_a2))
    norm_b = torch.sqrt(pair_norm_sq(G_b1, G_b2))
    C = cross_block(a2, b1)
    s2 = block_scores(None, G_a1, C, G_b2)
    scores = (torch.sqrt(s2) / (norm_a[:, None] * norm_b[None, :] + 1e-12)).flatten().cpu().numpy()
    return float(scores.mean()), float(scores.std())


def top_pairs(mat, k=50):
    flat = mat.flatten()
    valid = ~np.isnan(flat)
    idx = np.argsort(np.where(valid, flat, -np.inf))[::-1][:k]
    return [{"reader": head_label(int(i) // N_HEADS_TOTAL),
             "writer": head_label(int(i) % N_HEADS_TOTAL),
             "score": float(flat[i])} for i in idx]


def two_hop(Qf, Kf, Vf, OvL, G_q, norm_qk, vcomp, n_pairs, device):
    """
    Virtual heads from the top-n V-composition (writer w -> middle m) pairs:
        A_v = W_OV^m W_OV^w = (M^T OvL_m)^T Vf_w,  M = Vf_m OvL_w^T
    then K-composition of each virtual head into every reader r with Lr > Lm.
    """
    flat = vcomp.flatten()
    valid = ~np.isnan(flat)
    order = np.argsort(np.where(valid, flat, -np.inf))[::-1][:n_pairs]
    results = []
    for rank, fi in enumerate(order):
        m_idx, w_idx = int(fi) // N_HEADS_TOTAL, int(fi) % N_HEADS_TOTAL
        Lm = m_idx // NUM_Q_HEADS
        if Lm >= NUM_LAYERS - 1:
            continue
        M = Vf[m_idx] @ OvL[w_idx].T                     # [256, 256]
        b1_v = M.T @ OvL[m_idx]                          # [256, 2304]
        b2_v = Vf[w_idx]                                 # [256, 2304]
        G_b1v = b1_v @ b1_v.T
        G_b2v = b2_v @ b2_v.T
        norm_v = torch.sqrt((G_b1v * G_b2v).sum())
        if norm_v < 1e-8:
            continue

        r_sl = slice((Lm + 1) * NUM_Q_HEADS, N_HEADS_TOTAL)
        C = cross_block(Kf[r_sl], b1_v.unsqueeze(0))     # [R, 1, 256, 256]
        s2 = block_scores(None, G_q[r_sl], C, G_b2v.unsqueeze(0))
        scores = (torch.sqrt(s2)[:, 0] / (norm_qk[r_sl] * norm_v + 1e-12)).cpu().numpy()
        best = np.argsort(scores)[::-1][:5]
        for bi in best:
            r_idx = (Lm + 1) * NUM_Q_HEADS + int(bi)
            results.append({
                "writer": head_label(w_idx), "middle": head_label(m_idx),
                "reader": head_label(r_idx),
                "vcomp_w_to_m": float(vcomp[m_idx, w_idx]),
                "kcomp_virtual_to_r": float(scores[bi]),
                "chain_score": float(vcomp[m_idx, w_idx] * scores[bi]),
            })
    results.sort(key=lambda d: -d["chain_score"])
    return results


def plot_maps(maps, null_mean, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    for col, (key, title) in enumerate([("qcomp", "Q-composition"),
                                        ("kcomp", "K-composition"),
                                        ("vcomp", "V-composition")]):
        mat = maps[key]
        ax = axes[0, col]
        im = ax.imshow(mat, aspect="auto", cmap="magma", origin="lower")
        ax.set_title(f"{title} (writer head -> reader head), null ~ {null_mean:.3f}")
        ax.set_xlabel("writer head (L*8+h)")
        ax.set_ylabel("reader head (L*8+h)")
        for L in range(0, NUM_LAYERS, 4):
            ax.axhline(L * NUM_Q_HEADS - 0.5, color="w", lw=0.3, alpha=0.4)
            ax.axvline(L * NUM_Q_HEADS - 0.5, color="w", lw=0.3, alpha=0.4)
        plt.colorbar(im, ax=ax, fraction=0.04)

        # layer-aggregated view: max over head pairs per (Lr, Lw)
        agg = np.full((NUM_LAYERS, NUM_LAYERS), np.nan)
        for lr in range(NUM_LAYERS):
            for lw in range(lr):
                blk = mat[lr * 8:(lr + 1) * 8, lw * 8:(lw + 1) * 8]
                agg[lr, lw] = np.nanmax(blk)
        ax = axes[1, col]
        im = ax.imshow(agg, aspect="auto", cmap="magma", origin="lower")
        ax.set_title(f"{title}: max over head pairs per layer pair")
        ax.set_xlabel("writer layer")
        ax.set_ylabel("reader layer")
        plt.colorbar(im, ax=ax, fraction=0.04)

    fig.suptitle("Gemma-2-2b cross-layer head composition (weight space, delta=0)", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = out_dir / "composition_map.png"
    fig.savefig(p, dpi=160)
    print(f"Saved: {p}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--delta", type=int, default=0,
                        help="RoPE distance for the reader key factor (K-comp)")
    parser.add_argument("--two-hop", action="store_true")
    parser.add_argument("--two-hop-pairs", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="analysis_outputs/composition_map")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building factors ({args.model}, delta={args.delta})...", flush=True)
    Qf, Kf, Vf, OvL = build_factors(args.model, args.device, delta=args.delta)

    print("Computing null floor...", flush=True)
    null_mean, null_std = random_null(args.device)
    print(f"random-factor null: {null_mean:.4f} +/- {null_std:.4f}")

    maps, norm_qk_np, norm_ov_np = compute_map(Qf, Kf, Vf, OvL, args.device)

    np.savez_compressed(out_dir / "composition_map.npz",
                        **maps, norm_qk=norm_qk_np, norm_ov=norm_ov_np,
                        null_mean=null_mean, null_std=null_std,
                        delta=args.delta)

    summary = {
        "model": args.model, "delta": args.delta,
        "null_mean": null_mean, "null_std": null_std,
        "top_qcomp": top_pairs(maps["qcomp"]),
        "top_kcomp": top_pairs(maps["kcomp"]),
        "top_vcomp": top_pairs(maps["vcomp"]),
    }

    if args.two_hop:
        print("2-hop virtual circuits...", flush=True)
        G_q = grams(Qf)
        G_k = grams(Kf)
        norm_qk_t = torch.sqrt(pair_norm_sq(G_q, G_k))
        summary["top_two_hop"] = two_hop(
            Qf, Kf, Vf, OvL, G_q, norm_qk_t, maps["vcomp"],
            args.two_hop_pairs, args.device)[:60]

    with open(out_dir / "composition_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"Saved: {out_dir / 'composition_summary.json'}")

    for key in ["qcomp", "kcomp", "vcomp"]:
        t = summary[f"top_{key}"][:8]
        print(f"\ntop {key}: null={null_mean:.3f}")
        for d in t:
            print(f"  {d['writer']:>7} -> {d['reader']:>7}  {d['score']:.4f}")

    plot_maps(maps, null_mean, out_dir)


if __name__ == "__main__":
    main()
