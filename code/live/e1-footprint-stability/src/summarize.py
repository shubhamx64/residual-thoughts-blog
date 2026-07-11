"""Cross-model headline table from per-model metrics JSONs."""
import json

import numpy as np

from common import RESULTS

MODELS_DONE = ["qwen2.5-1.5b", "gemma-2-2b", "pythia-1.4b", "tinyllama-1.1b"]

for m in MODELS_DONE:
    d = json.load(open(RESULTS / m / "metrics_q99.0.json"))
    pl = d["per_layer"]
    marg = np.array([p["margin_ccos"] for p in pl])
    nstd = np.array([p["noise_std"] for p in pl])
    j = np.array([p["margin_j256"] for p in pl])
    c = d["classification"]
    cp = d["contrast_placement"]
    print(f"{m} ({d['n_layers']} layers)")
    print(f"  margin ccos min/med/max: {marg.min():.3f}/{np.median(marg):.3f}/{marg.max():.3f}"
          f"  min margin/3sigma = {(marg/(3*nstd+1e-9)).min():.0f}x")
    print(f"  j256 margin med {np.median(j):.3f}, peak layer {int(j.argmax())} ({j.max():.3f})")
    print(f"  acc5 {c['acc5']:.3f} tok {c['token_acc5']:.3f} | acc3 {c['acc3']:.3f} tok {c['token_acc3']:.3f}")
    for k in ("math_prose", "code_prose"):
        p = cp[k]
        swing = (p["to_sibling"] - p["token_to_sibling"]) - (p["to_prose"] - p["token_to_prose"])
        print(f"  {k}: fp(prose {p['to_prose']:+.3f}, {p['sibling']} {p['to_sibling']:+.3f}) "
              f"tok(prose {p['token_to_prose']:+.3f}, {p['sibling']} {p['token_to_sibling']:+.3f}) "
              f"net swing toward sibling {swing:+.3f}")
