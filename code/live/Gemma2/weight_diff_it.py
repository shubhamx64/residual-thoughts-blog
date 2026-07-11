"""
Weight-space diff: gemma-2-2b (base) vs gemma-2-2b-it (instruction-tuned).

Reads both checkpoints directly from safetensors (no model forward, CPU-only,
streams one tensor at a time) and asks where instruction tuning actually
landed, in the same functional frames as the weight-space map:

1. Raw per-matrix relative Frobenius change, grouped per layer into
   attention (q,k,v,o), MLP (gate,up,down), and norm gammas.
2. Per-head changes, including the FUNCTIONAL circuits:
     QK circuit  A  = diag(1+g_in)  W_Q[h]^T W_K[g]  diag(1+g_in)
     OV circuit  OV = diag(1+g_post) W_O[h] W_V[g]
   compared base-vs-it via the low-rank trace trick (256x256 intermediates,
   never materializing 2304x2304):
     tr(A1^T A2) = sum( (Q1 Q2^T) * (K1 K2^T) )
   Each model is folded with ITS OWN gammas (the runtime-effective circuit).
   Unfolded variants are also computed to separate "projection moved" from
   "norm gain moved".
3. Delta rank structure: singular spectrum of (W_it - W_base) per attention
   head (exact) and per MLP matrix (top-64 via svd_lowrank). If fine-tuning
   deltas are effectively low-rank, top-k energy fractions will be high.
4. Embedding diff: per-row delta norms; top moved tokens (absolute and
   relative). Gemma-2 ties embed/unembed, so this is also the unembed diff.

Usage:
    python weight_diff_it.py [--no-fold] [--out analysis_outputs/weight_diff_it]
"""
import argparse
import json
from pathlib import Path

import torch

NUM_LAYERS = 26
NUM_Q_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = 256
HIDDEN = 2304


class ShardedWeights:
    """Lazy per-tensor access into a (possibly sharded) safetensors checkpoint."""

    def __init__(self, repo_id: str):
        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        self.path = Path(snapshot_download(
            repo_id, allow_patterns=["*.safetensors*", "config.json"]))
        index_file = self.path / "model.safetensors.index.json"
        self._handles = {}
        if index_file.exists():
            with open(index_file) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            shard = self.path / "model.safetensors"
            with safe_open(shard, framework="pt") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
        self._safe_open = safe_open

    def get(self, key: str) -> torch.Tensor:
        shard = self.weight_map[key]
        if shard not in self._handles:
            self._handles[shard] = self._safe_open(
                self.path / shard, framework="pt", device="cpu")
        return self._handles[shard].get_tensor(key).float()


def rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """||b - a||_F / ||a||_F"""
    return ((b - a).norm() / a.norm()).item()


def circuit_compare(Q1, K1, Q2, K2):
    """
    Compare bilinear circuits A_i = Q_i^T K_i ([HIDDEN, HIDDEN], never built).
    Q*, K*: [HEAD_DIM or any rank, HIDDEN].
    Returns (cosine, relative Frobenius change wrt A1).
    """
    t11 = ((Q1 @ Q1.T) * (K1 @ K1.T)).sum()
    t22 = ((Q2 @ Q2.T) * (K2 @ K2.T)).sum()
    t12 = ((Q1 @ Q2.T) * (K1 @ K2.T)).sum()
    n1 = torch.sqrt(torch.clamp(t11, min=0))
    n2 = torch.sqrt(torch.clamp(t22, min=0))
    cos = (t12 / (n1 * n2 + 1e-12)).item()
    rel = (torch.sqrt(torch.clamp(t11 - 2 * t12 + t22, min=0)) / (n1 + 1e-12)).item()
    return cos, rel


def delta_spectrum(delta: torch.Tensor, topk=(1, 8, 32)):
    """Exact singular spectrum stats of a delta matrix."""
    s = torch.linalg.svdvals(delta)
    s2 = s ** 2
    total = s2.sum()
    if total <= 0:
        return {"erank": 0.0, **{f"top{k}_energy": 0.0 for k in topk}}
    out = {f"top{k}_energy": (s2[:k].sum() / total).item() for k in topk}
    out["erank"] = ((s2.sum() ** 2) / (s2 ** 2).sum()).item()  # participation ratio
    return out


def delta_spectrum_lowrank(delta: torch.Tensor, q=64, topk=(1, 8, 32, 64)):
    """Top-q singular values via randomized SVD; energy fractions vs exact total."""
    total = (delta ** 2).sum()
    _, s, _ = torch.svd_lowrank(delta, q=q, niter=4)
    s2 = s ** 2
    if total <= 0:
        return {f"top{k}_energy": 0.0 for k in topk}
    return {f"top{k}_energy": (s2[:k].sum() / total).item() for k in topk}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="google/gemma-2-2b")
    parser.add_argument("--it", default="google/gemma-2-2b-it")
    parser.add_argument("--no-fold", action="store_true",
                        help="skip gamma folding in circuit comparisons")
    parser.add_argument("--out", default="analysis_outputs/weight_diff_it")
    parser.add_argument("--top-tokens", type=int, default=40)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold = not args.no_fold

    base = ShardedWeights(args.base)
    it = ShardedWeights(args.it)

    results = {"base": args.base, "it": args.it, "fold_gamma": fold,
               "layers": [], "embedding": {}, "final_norm": {}}

    # ---------- per-layer ----------
    for L in range(NUM_LAYERS):
        pre = f"model.layers.{L}."
        lay = {"layer": L, "matrices": {}, "heads": []}

        tensors = {}
        for name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                     "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                     "mlp.down_proj", "input_layernorm",
                     "post_attention_layernorm", "pre_feedforward_layernorm",
                     "post_feedforward_layernorm"]:
            key = pre + name + ".weight"
            a, b = base.get(key), it.get(key)
            short = name.split(".")[-1]
            lay["matrices"][short] = rel_diff(a, b)
            tensors[short] = (a, b)

        # norm gammas: relative change of the runtime gain (1 + gamma)
        for norm in ["input_layernorm", "post_attention_layernorm",
                     "pre_feedforward_layernorm", "post_feedforward_layernorm"]:
            a, b = tensors[norm]
            lay["matrices"][norm + "_gain"] = rel_diff(1.0 + a, 1.0 + b)

        # MLP delta rank (top-64 energy via randomized SVD)
        lay["mlp_delta_spectrum"] = {
            m: delta_spectrum_lowrank(tensors[m][1] - tensors[m][0])
            for m in ["gate_proj", "up_proj", "down_proj"]
        }

        g_in_b, g_in_i = tensors["input_layernorm"]
        g_po_b, g_po_i = tensors["post_attention_layernorm"]
        scale_in_b = (1.0 + g_in_b) if fold else torch.ones(HIDDEN)
        scale_in_i = (1.0 + g_in_i) if fold else torch.ones(HIDDEN)
        scale_po_b = (1.0 + g_po_b) if fold else torch.ones(HIDDEN)
        scale_po_i = (1.0 + g_po_i) if fold else torch.ones(HIDDEN)

        Wq_b, Wq_i = tensors["q_proj"]      # [2048, 2304]
        Wk_b, Wk_i = tensors["k_proj"]      # [1024, 2304]
        Wv_b, Wv_i = tensors["v_proj"]      # [1024, 2304]
        Wo_b, Wo_i = tensors["o_proj"]      # [2304, 2048]

        for h in range(NUM_Q_HEADS):
            g = h // (NUM_Q_HEADS // NUM_KV_HEADS)
            sl_q = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
            sl_kv = slice(g * HEAD_DIM, (g + 1) * HEAD_DIM)

            q_b, q_i = Wq_b[sl_q], Wq_i[sl_q]
            k_b, k_i = Wk_b[sl_kv], Wk_i[sl_kv]
            v_b, v_i = Wv_b[sl_kv], Wv_i[sl_kv]
            o_b, o_i = Wo_b[:, sl_q], Wo_i[:, sl_q]   # [2304, 256]

            head = {"head": h, "kv_group": g}
            head["q_rel"] = rel_diff(q_b, q_i)
            head["k_rel"] = rel_diff(k_b, k_i)
            head["v_rel"] = rel_diff(v_b, v_i)
            head["o_rel"] = rel_diff(o_b, o_i)

            # functional circuits, each model folded with its own gammas
            qf_b, qf_i = q_b * scale_in_b, q_i * scale_in_i
            kf_b, kf_i = k_b * scale_in_b, k_i * scale_in_i
            head["qk_cos"], head["qk_rel"] = circuit_compare(qf_b, kf_b, qf_i, kf_i)
            head["qk_cos_nofold"], _ = circuit_compare(q_b, k_b, q_i, k_i)

            # OV: diag(scale_po) O V  ->  rows of O^T scaled
            ovL_b = (o_b * scale_po_b.unsqueeze(1)).T   # [256, 2304] = (diag O)^T
            ovL_i = (o_i * scale_po_i.unsqueeze(1)).T
            vf_b = v_b * scale_in_b
            vf_i = v_i * scale_in_i
            head["ov_cos"], head["ov_rel"] = circuit_compare(ovL_b, vf_b, ovL_i, vf_i)
            head["ov_cos_nofold"], _ = circuit_compare(o_b.T, v_b, o_i.T, v_i)

            # delta rank structure (exact, small matrices)
            head["dq_spectrum"] = delta_spectrum(q_i - q_b)
            head["do_spectrum"] = delta_spectrum(o_i - o_b)
            lay["heads"].append(head)

        results["layers"].append(lay)
        print(f"layer {L:2d}  attn q/k/v/o rel: "
              f"{lay['matrices']['q_proj']:.4f}/{lay['matrices']['k_proj']:.4f}/"
              f"{lay['matrices']['v_proj']:.4f}/{lay['matrices']['o_proj']:.4f}  "
              f"mlp g/u/d: {lay['matrices']['gate_proj']:.4f}/"
              f"{lay['matrices']['up_proj']:.4f}/{lay['matrices']['down_proj']:.4f}  "
              f"qk_cos mean: "
              f"{sum(hh['qk_cos'] for hh in lay['heads'])/NUM_Q_HEADS:.4f}  "
              f"ov_cos mean: "
              f"{sum(hh['ov_cos'] for hh in lay['heads'])/NUM_Q_HEADS:.4f}",
              flush=True)

    # ---------- final norm ----------
    a, b = base.get("model.norm.weight"), it.get("model.norm.weight")
    results["final_norm"] = {"rel": rel_diff(a, b), "gain_rel": rel_diff(1 + a, 1 + b)}

    # ---------- embedding ----------
    print("embedding diff...", flush=True)
    E_b = base.get("model.embed_tokens.weight")
    E_i = it.get("model.embed_tokens.weight")
    d_rows = (E_i - E_b).norm(dim=1)
    base_rows = E_b.norm(dim=1)
    rel_rows = d_rows / (base_rows + 1e-8)
    results["embedding"]["rel_overall"] = rel_diff(E_b, E_i)
    results["embedding"]["frac_rows_moved_gt_1pct"] = (rel_rows > 0.01).float().mean().item()
    results["embedding"]["frac_rows_moved_gt_10pct"] = (rel_rows > 0.10).float().mean().item()

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.base)
        top_abs = torch.argsort(d_rows, descending=True)[: args.top_tokens]
        results["embedding"]["top_moved_tokens"] = [
            {"id": int(i), "token": repr(tok.decode([int(i)])),
             "delta_norm": d_rows[i].item(), "rel": rel_rows[i].item(),
             "base_norm": base_rows[i].item()}
            for i in top_abs
        ]
    except Exception as e:
        print(f"tokenizer unavailable, skipping token labels: {e}")
    del E_b, E_i

    out_json = out_dir / "weight_diff_it.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
