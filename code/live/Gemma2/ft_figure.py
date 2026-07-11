"""Summary figure for the fine-tune localization experiment."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).parent / "analysis_outputs/ft_localization"
configs = ["lora_attn", "lora_mlp", "full_attn", "full_mlp"]
res = {c: json.load(open(root / f"{c}.json")) for c in configs}
labels = ["LoRA attn\n(6M)", "LoRA MLP\n(14M)", "full attn\n(368M)", "full MLP\n(1.7B)"]
colors = ["#d95f02", "#1b9e77", "#d95f02", "#1b9e77"]
hatch = ["", "", "//", "//"]

fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
metrics = [
    ("para_acc", "Paraphrase recall (generalization)", "%"),
    ("real_acc", "Real-fact recall (forgetting)", "%"),
    ("control_ppl", "Control perplexity (drift)", ""),
]
for ax, (key, title, unit) in zip(axes, metrics):
    vals = [res[c]["after"][key] * (100 if unit == "%" else 1) for c in configs]
    bars = ax.bar(labels, vals, color=colors, hatch=hatch, alpha=0.85)
    base = res["lora_attn"]["before"][key] * (100 if unit == "%" else 1)
    ax.axhline(base, color="gray", ls="--", lw=1, label="pre-training baseline")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center",
                va="bottom", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)

ax = axes[3]
vals = [res[c]["final_loss"] for c in configs]
bars = ax.bar(labels, vals, color=colors, hatch=hatch, alpha=0.85)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center",
            va="bottom", fontsize=9)
ax.set_title("Final training loss (expressivity ceiling)", fontsize=10)
ax.tick_params(axis="x", labelsize=8)

fig.suptitle("Teaching gemma-2-2b 100 fictional facts: attention-only vs MLP-only "
             "(green = MLP, orange = attention, hatched = full fine-tune)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = Path(__file__).parent / "figs/ft_localization.png"
fig.savefig(out, dpi=170)
print(f"saved {out}")
