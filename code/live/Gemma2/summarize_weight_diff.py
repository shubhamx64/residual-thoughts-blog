"""Console + figure summary of weight_diff_it.json (base vs -it)."""
import json
from pathlib import Path

import numpy as np

d = json.load(open(Path(__file__).parent / "analysis_outputs/weight_diff_it/weight_diff_it.json"))
heads = [(l["layer"], h) for l in d["layers"] for h in l["heads"]]

print("=== top 12 heads by QK circuit movement (lowest qk_cos) ===")
for L, h in sorted(heads, key=lambda x: x[1]["qk_cos"])[:12]:
    print(f"  L{L:2d}H{h['head']}: qk_cos={h['qk_cos']:.4f} (nofold {h['qk_cos_nofold']:.4f})  "
          f"qk_rel={h['qk_rel']:.4f}  dq_erank={h['dq_spectrum']['erank']:.0f} "
          f"top8={h['dq_spectrum']['top8_energy']:.2f}")

print("\n=== top 12 heads by OV circuit movement (lowest ov_cos) ===")
for L, h in sorted(heads, key=lambda x: x[1]["ov_cos"])[:12]:
    print(f"  L{L:2d}H{h['head']}: ov_cos={h['ov_cos']:.4f} (nofold {h['ov_cos_nofold']:.4f})  "
          f"ov_rel={h['ov_rel']:.4f}  do_erank={h['do_spectrum']['erank']:.0f}")

qe = [h["dq_spectrum"]["erank"] for _, h in heads]
t8 = [h["dq_spectrum"]["top8_energy"] for _, h in heads]
t32 = [h["dq_spectrum"]["top32_energy"] for _, h in heads]
print(f"\nattention dQ delta: erank mean {np.mean(qe):.0f}/256, "
      f"top8 energy {np.mean(t8):.3f}, top32 {np.mean(t32):.3f}")
oe = [h["do_spectrum"]["erank"] for _, h in heads]
print(f"attention dO delta: erank mean {np.mean(oe):.0f}/256")
m64 = [l["mlp_delta_spectrum"][m]["top64_energy"]
       for l in d["layers"] for m in ["gate_proj", "up_proj", "down_proj"]]
print(f"MLP delta top64 energy mean {np.mean(m64):.3f} (matrices are rank <= 2304)")

print("\n=== norm gain (1+gamma) relative changes by layer ===")
for l in d["layers"]:
    m = l["matrices"]
    print(f"  L{l['layer']:2d} in:{m['input_layernorm_gain']:.4f} "
          f"post:{m['post_attention_layernorm_gain']:.4f} "
          f"preff:{m['pre_feedforward_layernorm_gain']:.4f} "
          f"postff:{m['post_feedforward_layernorm_gain']:.4f}")

e = d["embedding"]
print(f"\nembedding: overall rel {e['rel_overall']:.4f}, rows moved >1%: "
      f"{e['frac_rows_moved_gt_1pct']*100:.2f}%, >10%: {e['frac_rows_moved_gt_10pct']*100:.4f}%")
print("top moved tokens:")
for t in e.get("top_moved_tokens", [])[:25]:
    print(f"  id {t['id']:>7} {t['token']:<28} delta={t['delta_norm']:.3f} rel={t['rel']:.3f}")
print(f"final norm gain rel: {d['final_norm']['gain_rel']:.4f}")

# ---- figure ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

layers = [l["layer"] for l in d["layers"]]
attn = [np.mean([l["matrices"][k] for k in ["q_proj", "k_proj", "v_proj", "o_proj"]])
        for l in d["layers"]]
mlp = [np.mean([l["matrices"][k] for k in ["gate_proj", "up_proj", "down_proj"]])
       for l in d["layers"]]
qk_cos = np.array([[h["qk_cos"] for h in l["heads"]] for l in d["layers"]])
ov_cos = np.array([[h["ov_cos"] for h in l["heads"]] for l in d["layers"]])

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0, 0]
ax.plot(layers, attn, marker="o", label="attention (q,k,v,o mean)")
ax.plot(layers, mlp, marker="s", label="MLP (gate,up,down mean)")
ax.set_title("Relative weight change ||W_it - W_base|| / ||W_base||")
ax.set_xlabel("layer"); ax.set_ylabel("rel Frobenius"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 1]
im = ax.imshow(1 - qk_cos.T, aspect="auto", cmap="magma", origin="lower")
ax.set_title("QK circuit movement per head (1 - cos)")
ax.set_xlabel("layer"); ax.set_ylabel("head")
plt.colorbar(im, ax=ax, fraction=0.04)

ax = axes[1, 0]
im = ax.imshow(1 - ov_cos.T, aspect="auto", cmap="magma", origin="lower")
ax.set_title("OV circuit movement per head (1 - cos)")
ax.set_xlabel("layer"); ax.set_ylabel("head")
plt.colorbar(im, ax=ax, fraction=0.04)

ax = axes[1, 1]
t8_by_layer = [np.mean([h["dq_spectrum"]["top8_energy"] for h in l["heads"]]) for l in d["layers"]]
t64_mlp = [np.mean([l["mlp_delta_spectrum"][m]["top64_energy"]
                    for m in ["gate_proj", "up_proj", "down_proj"]]) for l in d["layers"]]
ax.plot(layers, t8_by_layer, marker="o", label="attn dQ: top-8/256 energy")
ax.plot(layers, t64_mlp, marker="s", label="MLP delta: top-64/2304 energy")
ax.set_title("Low-rank structure of the fine-tuning delta")
ax.set_xlabel("layer"); ax.set_ylabel("fraction of delta energy"); ax.legend(); ax.grid(alpha=0.3)

fig.suptitle("gemma-2-2b vs gemma-2-2b-it: where instruction tuning landed", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = Path(__file__).parent / "figs/weight_diff_it.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=170)
print(f"\nSaved: {out}")
