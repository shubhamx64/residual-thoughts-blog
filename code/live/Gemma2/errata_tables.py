"""Per-layer validation means: pre-fix baseline vs post-fix SAE vs token."""
from collections import defaultdict

import numpy as np
from summarize_validation_report import parse_report

REPORTS = [
    ("prefix", "text_outputs/layers_all_4k_20_prompts.txt"),
    ("sae", "validation_full_sae.txt"),
    ("token", "validation_full_token.txt"),
]

cols = {}
for name, path in REPORTS:
    by_layer = defaultdict(list)
    for h in parse_report(path):
        if h.spearman is not None:
            by_layer[h.layer].append(h.spearman)
    cols[name] = {L: np.mean(v) for L, v in by_layer.items()}

layers = sorted(set().union(*[c.keys() for c in cols.values()]))
print(f"{'layer':>5} {'prefix':>8} {'sae_fixed':>10} {'token':>8}")
for L in layers:
    row = [cols[n].get(L) for n, _ in REPORTS]
    print(f"{L:>5} " + " ".join(
        f"{v:>{w}.3f}" if v is not None else f"{'-':>{w}}"
        for v, w in zip(row, (8, 10, 8))))

for name, _ in REPORTS:
    v = list(cols[name].values())
    print(f"{name}: grand mean {np.mean(v):+.4f}, "
          f"min {min(v):+.3f} @L{min(cols[name], key=cols[name].get)}, "
          f"max {max(v):+.3f} @L{max(cols[name], key=cols[name].get)}")
