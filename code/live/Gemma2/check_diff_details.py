"""Spot checks on the base-vs-it diff: chat tokens, star heads, L23 heads."""
import json
from pathlib import Path

import torch

from weight_diff_it import ShardedWeights

base = ShardedWeights("google/gemma-2-2b")
it = ShardedWeights("google/gemma-2-2b-it")
E_b = base.get("model.embed_tokens.weight")
E_i = it.get("model.embed_tokens.weight")
d = (E_i - E_b).norm(dim=1)
rel = d / (E_b.norm(dim=1) + 1e-8)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
ids = {
    "<bos>": 2, "<eos>": 1, "<start_of_turn>": 106, "<end_of_turn>": 107,
    "' the'": tok.encode(" the", add_special_tokens=False)[0],
    "'model'": tok.encode("model", add_special_tokens=False)[0],
}
ranks = torch.argsort(d, descending=True)
rank_of = torch.empty_like(ranks)
rank_of[ranks] = torch.arange(len(ranks))
print("token movement (rank out of 256000 by abs delta):")
for name, i in ids.items():
    print(f"  {name:>17} id={i:<7} delta={d[i]:.4f} rel={rel[i]:.4f} rank={int(rank_of[i])}")

med = d.median()
print(f"\nmedian row delta: {med:.4f}; <unused> band mean: {d[12:106].mean():.4f}")

j = json.load(open(Path(__file__).parent / "analysis_outputs/weight_diff_it/weight_diff_it.json"))
print("\nheads of interest:")
for L, H in [(5, 4), (6, 0), (23, 0), (23, 1), (23, 2), (24, 6), (13, 0)]:
    h = j["layers"][L]["heads"][H]
    print(f"  L{L:2d}H{H}: qk_cos={h['qk_cos']:.4f} ov_cos={h['ov_cos']:.4f} "
          f"q_rel={h['q_rel']:.4f} dq_erank={h['dq_spectrum']['erank']:.0f}")

allq = sorted((hh["qk_cos"] for l in j["layers"] for hh in l["heads"]))
import numpy as np
print(f"\nqk_cos percentiles: min={allq[0]:.4f} p25={np.percentile(allq,25):.4f} "
      f"median={np.percentile(allq,50):.4f} p75={np.percentile(allq,75):.4f} max={allq[-1]:.4f}")
