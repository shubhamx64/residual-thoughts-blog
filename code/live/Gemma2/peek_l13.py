"""Mean overall Spearman for L13 under the three probe configurations."""
import numpy as np
from summarize_validation_report import parse_report

for name, path in [
    ("SAE offset -1 (layer-12 SAE)", "validation_full_sae.txt"),
    ("SAE offset 0 (layer-13 SAE)", "validation_L13_offset0.txt"),
    ("token basis", "validation_full_token.txt"),
]:
    heads = [h for h in parse_report(path) if h.layer == 13]
    s = [h.spearman for h in heads if h.spearman is not None]
    print(f"{name:<30} n={len(s)} mean_S={np.mean(s):+.4f} "
          f"per-head: {[f'{x:+.2f}' for x in sorted(s)]}")
