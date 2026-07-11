"""plot_rope_curves_from_json.py

Plot RoPE stability(Δ) curves for a chosen set of heads, using the
`rope_stability` entries already stored in analysis_*.json.

This is meant to fill the draft's:
  [FIGURE 4 — stability(Δ) curves for a few heads: stable vs unstable.]

Usage:
  python plot_rope_curves_from_json.py \
      --json ./analysis_outputs/analysis_20250101_...json \
      --heads 10:5,6:3,15:0 \
      --out ./blog_assets/fig04_rope_curves_selected.png
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _rope_auc_log(stability_by_delta: Dict[str, Any]) -> float:
    """Normalized AUC over log2(delta) for delta>=1."""
    pts: List[Tuple[int, float]] = []
    for k, v in (stability_by_delta or {}).items():
        try:
            d = int(k)
        except Exception:
            continue
        if d < 1:
            continue
        s = _safe_float(v)
        if not math.isfinite(s):
            continue
        pts.append((d, s))
    if not pts:
        return float("nan")
    pts.sort(key=lambda t: t[0])
    deltas = np.array([p[0] for p in pts], dtype=np.float32)
    vals = np.array([p[1] for p in pts], dtype=np.float32)
    x = np.log2(deltas)
    if len(x) >= 2 and (x[-1] - x[0]) > 1e-9:
        auc = np.trapz(vals, x)
        return float(auc / (x[-1] - x[0]))
    return float(vals[0])


def _parse_heads(s: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        layer_str, head_str = part.split(":")
        out.append((int(layer_str), int(head_str)))
    return out


def _get_curve(data: Dict[str, Any], layer: int, head: int) -> Dict[str, Any]:
    layer_data = (data.get("layer_results", {}) or {}).get(str(layer), {})
    for rr in (layer_data.get("rope_stability", []) or []):
        if not rr:
            continue
        if int(rr.get("query_head", -1)) == int(head):
            return rr
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--heads", required=True, help="Comma-separated: layer:head,layer:head")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    heads = _parse_heads(args.heads)
    if not heads:
        raise SystemExit("No heads parsed")

    plt.figure(figsize=(8.5, 5.2))
    for layer, head in heads:
        rr = _get_curve(data, layer, head)
        st = rr.get("stability_by_delta", {}) or {}
        pts = []
        for k, v in st.items():
            try:
                d = int(k)
            except Exception:
                continue
            if d < 1:
                continue
            s = _safe_float(v)
            if not math.isfinite(s):
                continue
            pts.append((d, s))
        pts.sort(key=lambda t: t[0])
        if not pts:
            continue
        deltas = [p[0] for p in pts]
        vals = [p[1] for p in pts]
        auc = _rope_auc_log(st)
        plt.plot(deltas, vals, marker="o", linewidth=2.0, label=f"L{layer}H{head} (AUC={auc:.3f})")

    plt.xscale("log", base=2)
    plt.ylim(0, 1.02)
    plt.grid(True, which="both", alpha=0.25)
    plt.title("RoPE stability(Δ): similarity(BΔ, B0) vs distance")
    plt.xlabel("Δ (tokens)")
    plt.ylabel("Similarity")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    plt.close()
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
