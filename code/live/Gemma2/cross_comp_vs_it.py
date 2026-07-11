"""
Does compositional importance in the BASE model predict which heads
instruction tuning moves? Crosses composition_map.npz with weight_diff_it.json.

Per head: writer importance = max over readers of each composition score;
reader importance = max over writers. Movement = 1 - qk_cos / 1 - ov_cos.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

root = Path(__file__).parent
cm = np.load(root / "analysis_outputs/composition_map/composition_map.npz")
wd = json.load(open(root / "analysis_outputs/weight_diff_it/weight_diff_it.json"))

N = 208
qk_move = np.array([1 - h["qk_cos"] for l in wd["layers"] for h in l["heads"]])
ov_move = np.array([1 - h["ov_cos"] for l in wd["layers"] for h in l["heads"]])

def importance(mat, axis):
    """axis=0: max over readers (writer importance); axis=1: max over writers."""
    return np.nanmax(mat, axis=axis)

rows = []
for key in ["qcomp", "kcomp", "vcomp"]:
    mat = cm[key]
    with np.errstate(all="ignore"):
        as_writer = np.nanmax(mat, axis=0)   # for each writer column
        as_reader = np.nanmax(mat, axis=1)   # for each reader row
    for name, imp, move in [
        (f"{key} writer-imp vs OV movement", as_writer, ov_move),
        (f"{key} reader-imp vs QK movement", as_reader, qk_move),
    ]:
        m = ~np.isnan(imp)
        rho, p = stats.spearmanr(imp[m], move[m])
        rows.append((name, rho, p, m.sum()))

print(f"{'relation':<38} {'spearman':>9} {'p':>10} {'n':>5}")
for name, rho, p, n in rows:
    print(f"{name:<38} {rho:>9.3f} {p:>10.2e} {n:>5}")

# control: importance vs raw weight movement of MLP in same layer (should be ~0 per-head)
print("\nper-layer means as sanity anchor:")
layer_qk_move = qk_move.reshape(26, 8).mean(axis=1)
print("layers with most QK movement:", np.argsort(layer_qk_move)[::-1][:5])
