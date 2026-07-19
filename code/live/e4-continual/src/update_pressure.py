"""Update-pressure test (Sol follow-up / PREREG_ROBUSTNESS Part E): is crowding just a
zero-data predictor of how far the optimizer moves a neuron?

From existing E4 checkpoints (no training, no GPU): per neuron, the actual update norm
in the UNPROTECTED baseline arm, U_i = ||theta_i^final - theta_i^afterA|| over its
gate row + up row + down column. Then Spearman(crowding, U). High => crowding tracks
optimizer write-pressure (deflationary); low => it does not.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

RES = Path(__file__).resolve().parent.parent / "results"
FAM = {
    "tinyllama-1.1b": ("ckpt_A.pt", "ckpt_B_baseline.pt"),
    "qwen2.5-1.5b": ("ckpt_A_qwen.pt", "ckpt_B_baseline_qwen.pt"),
}


def crowd_maxcos(down_A, block=2048):
    """max|cos| among down_proj columns (unit-normalised), CPU-chunked."""
    W = down_A.float()
    W = W / (W.norm(dim=0, keepdim=True) + 1e-12)   # [d, inter]
    inter = W.shape[1]
    mc = torch.zeros(inter)
    for s in range(0, inter, block):
        e = min(s + block, inter)
        Gb = (W[:, s:e].T @ W).abs()                # [b, inter]
        Gb[torch.arange(e - s), torch.arange(s, e)] = 0
        mc[s:e] = Gb.max(1).values
    return mc.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(FAM))
    args = ap.parse_args()
    a_f, b_f = FAM[args.model]
    A = torch.load(RES / a_f, map_location="cpu")
    B = torch.load(RES / b_f, map_location="cpu")
    layers = sorted({int(k.split(".")[2]) for k in A if ".mlp." in k})

    rho_cu, rho_all = [], []
    for l in layers:
        p = f"model.layers.{l}.mlp."
        dg = (B[p + "gate_proj.weight"] - A[p + "gate_proj.weight"]).float()  # [inter,d]
        du = (B[p + "up_proj.weight"] - A[p + "up_proj.weight"]).float()      # [inter,d]
        dd = (B[p + "down_proj.weight"] - A[p + "down_proj.weight"]).float()  # [d,inter]
        U = torch.sqrt((dg ** 2).sum(1) + (du ** 2).sum(1) + (dd ** 2).sum(0)).numpy()
        crowd = crowd_maxcos(A[p + "down_proj.weight"])
        rho_cu.append(stats.spearmanr(crowd, U).statistic)

    res = {"model": args.model,
           "crowd_vs_updatenorm_rho_median": float(np.median(rho_cu)),
           "per_layer": [float(x) for x in rho_cu]}
    (RES / f"updatepressure_{args.model}.json").write_text(json.dumps(res, indent=1))
    print(f"== {args.model} ==")
    print(f"  crowd vs actual update norm  rho_med {res['crowd_vs_updatenorm_rho_median']:+.3f}")
    print(f"  (high => crowding ~ optimizer write-pressure; low => not the boring explanation)")


if __name__ == "__main__":
    main()
