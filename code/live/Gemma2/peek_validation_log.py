"""Live per-layer mean/max Spearman from a streaming validation log."""
import re
import sys
from collections import defaultdict

import numpy as np

path = sys.argv[1]
text = open(path, encoding="utf-8", errors="replace").read()
pairs = re.findall(
    r"Validating L(\d+)H(\d+)\.\.\..*?Spearman r: (-?[\d.]+)", text, re.S
)
by_layer = defaultdict(list)
for L, H, r in pairs:
    by_layer[int(L)].append(float(r))

print(f"{'layer':>5} {'n_heads':>7} {'mean_S':>8} {'max_S':>8}")
for L in sorted(by_layer):
    v = by_layer[L]
    print(f"{L:>5} {len(v):>7} {np.mean(v):>8.4f} {np.max(v):>8.4f}")
