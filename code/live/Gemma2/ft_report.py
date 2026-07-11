"""Comparison report for the four ft_localization configs."""
import json
from pathlib import Path

import numpy as np

root = Path(__file__).parent / "analysis_outputs/ft_localization"
configs = ["lora_attn", "lora_mlp", "full_attn", "full_mlp"]
res = {c: json.load(open(root / f"{c}.json")) for c in configs}

print(f"{'':<11} {'train':>8} {'loss':>6} | {'cloze':>6} {'para':>6} | "
      f"{'real':>6} {'ppl':>6} | {'cloze_lp':>8} {'para_lp':>8}")
for c in configs:
    r = res[c]
    b, a = r["before"], r["after"]
    print(f"{c:<11} {r['trainable_params']/1e6:>7.0f}M {r['final_loss']:>6.3f} | "
          f"{a['cloze_acc']:>6.1%} {a['para_acc']:>6.1%} | "
          f"{a['real_acc']:>6.1%} {a['control_ppl']:>6.2f} | "
          f"{a['cloze_logp']:>8.2f} {a['para_logp']:>8.2f}")
b = res["lora_attn"]["before"]
print(f"{'baseline':<11} {'-':>8} {'-':>6} | {b['cloze_acc']:>6.1%} "
      f"{b['para_acc']:>6.1%} | {b['real_acc']:>6.1%} {b['control_ppl']:>6.2f} | "
      f"{b['cloze_logp']:>8.2f} {b['para_logp']:>8.2f}")

print("\n=== where the update landed (delta probe) ===")
for c in configs:
    p = res[c]["delta_probe"]
    mods = ", ".join(f"{k}={v:.4f}" for k, v in p["per_module"].items())
    print(f"\n{c}: mean rel delta per module: {mods}")
    per_layer = p["per_layer"]
    tot = [np.mean([v for k, v in d.items() if k != "layer"]) for d in per_layer]
    top = np.argsort(tot)[::-1][:5]
    print(f"  most-moved layers: " + ", ".join(f"L{t}({tot[t]:.4f})" for t in top))
    if p["heads"]:
        hs = sorted(p["heads"], key=lambda h: h["qk_cos"])
        print("  most-rotated QK circuits: " + ", ".join(
            f"L{h['layer']}H{h['head']}({h['qk_cos']:.4f})" for h in hs[:5]))
        ho = sorted(p["heads"], key=lambda h: h["ov_cos"])
        print("  most-rotated OV circuits: " + ", ".join(
            f"L{h['layer']}H{h['head']}({h['ov_cos']:.4f})" for h in ho[:5]))
        qk = [h["qk_cos"] for h in p["heads"]]
        ov = [h["ov_cos"] for h in p["heads"]]
        print(f"  qk_cos median {np.median(qk):.4f} min {min(qk):.4f} | "
              f"ov_cos median {np.median(ov):.4f} min {min(ov):.4f}")

print("\n=== per-kind recall (after) ===")
# recompute per-kind from facts list is not saved; note for future runs
