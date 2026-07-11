"""Did first-person tokens move more than ordinary content words under IT?

Tied embeddings: a moved row changes both reading and producing that token.
Compares delta norms / ranks for pronouns vs random content-word controls.
"""
import torch
from transformers import AutoTokenizer

from weight_diff_it import ShardedWeights

base = ShardedWeights("google/gemma-2-2b")
it = ShardedWeights("google/gemma-2-2b-it")
E_b = base.get("model.embed_tokens.weight")
E_i = it.get("model.embed_tokens.weight")
d = (E_i - E_b).norm(dim=1)
rel = d / (E_b.norm(dim=1) + 1e-8)

ranks = torch.argsort(d, descending=True)
rank_of = torch.empty_like(ranks)
rank_of[ranks] = torch.arange(len(ranks))

tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")

def single_id(s):
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None

groups = {
    "first person": ["I", " I", "i", " i", " me", " my", " mine", " myself",
                     "I'm", " I'm", " I'll", " I've"],
    "second/assistant": [" you", " your", "you", " assistant", "assistant",
                         " Assistant", " AI", "AI"],
    "content controls": [" cat", " home", " dog", " tree", " table", " river",
                         " music", " window", " bread", " mountain", "cat", "home"],
    "function controls": [" the", " of", " and", " a", " to", " is"],
}

print(f"vocab median delta: {d.median():.4f}\n")
for gname, words in groups.items():
    print(f"--- {gname} ---")
    rows = []
    for w in words:
        i = single_id(w)
        if i is None:
            print(f"  {w!r:>12}  (multi-token, skipped)")
            continue
        rows.append((w, i, d[i].item(), rel[i].item(), int(rank_of[i])))
    for w, i, dn, r, rk in rows:
        print(f"  {w!r:>12} id={i:<7} delta={dn:.4f} rel={r:.4f} rank={rk:>6}/256000")
    if rows:
        import statistics
        print(f"  group median delta: {statistics.median(x[2] for x in rows):.4f}, "
              f"median rank: {statistics.median(x[4] for x in rows):.0f}")
    print()
