"""Cross-model E2 headline table."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["qwen2.5-1.5b", "gemma-2-2b", "pythia-1.4b", "tinyllama-1.1b"]

for m in MODELS:
    d = json.load(open(ROOT / "results" / m / "e2_metrics.json"))
    pl = d["per_layer"]
    n_l = d["n_layers"]
    t = d["token_dict"]
    fpw = np.array([p["mlp_write"]["fp_ratio"] for p in pl])
    coh = np.array([p["mlp_write"]["coherence_max"] for p in pl])
    q99 = np.array([p["mlp_write"]["q99"] for p in pl])
    welch = pl[0]["mlp_write"]["welch_bound"]
    ga = np.array([p["gain"]["g_attn"] for p in pl])
    gm = np.array([p["gain"]["g_mlp"] for p in pl])
    sr_q = np.array([p["attn_read_q"]["stable_rank_frac"] for p in pl])
    third = n_l // 3
    print(f"{m} (d={d['d']}, {n_l} layers)")
    print(f"  token dict: n={t['n']} fp={t['fp_ratio']:.3f} q99={t['q99']:.3f} "
          f"q999={t['q999']:.3f} welch={t['welch_bound']:.4f} srank={t['stable_rank_frac']:.2f}")
    print(f"  mlp_write fp: early {fpw[:third].mean():.3f} mid {fpw[third:2*third].mean():.3f} "
          f"late {fpw[2*third:].mean():.3f} | depth corr r={np.corrcoef(np.arange(n_l), fpw)[0,1]:+.2f}")
    print(f"  mlp_write coh: med {np.median(coh):.3f} max {coh.max():.3f} @L{int(coh.argmax())} "
          f"| q99 first->last {q99[0]:.3f}->{q99[-1]:.3f} (welch {welch:.4f})")
    print(f"  gains: g_attn {ga.min():.1f}-{ga.max():.1f} (peak L{int(ga.argmax())}), "
          f"g_mlp {gm.min():.1f}-{gm.max():.1f} (peak L{int(gm.argmax())})")
    print(f"  attn_read_q stable-rank frac: med {np.median(sr_q):.2f} min {sr_q.min():.2f} @L{int(sr_q.argmin())}")
