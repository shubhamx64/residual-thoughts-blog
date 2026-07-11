"""Shared plumbing for E3: paths, E1/E2 reuse, active-rate loading."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE / "e2-welch-gain" / "src"))
sys.path.insert(0, str(BASE / "e1-footprint-stability" / "src"))

E1_RESULTS = BASE / "e1-footprint-stability" / "results"

RATE_FLOOR = 0.002      # neuron must fire on >=0.2% of pooled tokens to be a candidate
STRATA = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.01]
PAIRS_PER_STRATUM = 50
TOP_COH_PAIRS = 100
TOPK_READERS = 64
QUANTILE = 99.0         # reuse E1 firing threshold


def pooled_rates(model_key, n_layers, dim):
    """Per-layer pooled firing rate per neuron from the E1 footprint captures."""
    fp_dir = E1_RESULTS / model_key / "footprints"
    counts = [np.zeros(dim, dtype=np.float64) for _ in range(n_layers)]
    tot = 0
    for path in fp_dir.glob("*.npz"):
        z = np.load(path)
        tot += int(z["n_tokens"])
        for l in range(n_layers):
            idx, cnt = z[f"idx_q{QUANTILE}_L{l}"], z[f"cnt_q{QUANTILE}_L{l}"]
            counts[l][idx] += cnt
    return [c / tot for c in counts], tot


def result_dir(model_key):
    d = ROOT / "results" / model_key
    d.mkdir(parents=True, exist_ok=True)
    return d
