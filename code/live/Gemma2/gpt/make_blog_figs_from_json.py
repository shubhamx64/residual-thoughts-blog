"""make_blog_figs_from_json.py

Generate the blog-post placeholder plots/tables *directly* from an analysis_*.json
produced by main.py / analysis_pipeline.py.

This script intentionally does NOT load model weights or SAEs again.
It only consumes the JSON (metrics + RoPE curves + program summaries).

Outputs (files in --outdir):
  - head_metrics.csv                       (one row per (layer,head))
  - layer_summary.csv                      (one row per layer)
  - program_distribution.csv               (overall program histogram)
  - fig05_sw_vs_global_selectivity.png
  - fig05_sw_vs_global_diag_mass.png
  - fig06_depth_selectivity_and_diag.png
  - fig07_depth_rope_auc.png
  - fig08_depth_write_archetype_fractions.png
  - fig09_depth_redundancy.png

Typical usage:
  python make_blog_figs_from_json.py \
      --json ./analysis_outputs/analysis_20250101_120000.json \
      --outdir ./blog_assets

"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PAIR_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def _mkdirp(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _rope_auc_log(stability_by_delta: Dict[str, Any]) -> float:
    """Normalized AUC over log2(delta) for delta>=1.

    This matches the "AUC-style over log-distance" summary described in the draft.
    """
    if not stability_by_delta:
        return float("nan")

    pts: List[Tuple[int, float]] = []
    for k, v in stability_by_delta.items():
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


def _build_redundancy_by_head(layer_data: Dict[str, Any]) -> Dict[int, float]:
    """Return head -> mean redundancy within this layer."""
    red = layer_data.get("cross_head_redundancy", {}) or {}
    scores: Dict[int, List[float]] = defaultdict(list)
    for pair_str, val in red.items():
        m = PAIR_RE.match(str(pair_str))
        if not m:
            continue
        i = int(m.group(1))
        j = int(m.group(2))
        s = _safe_float(val)
        scores[i].append(s)
        scores[j].append(s)
    return {h: float(np.nanmean(v)) if len(v) else float("nan") for h, v in scores.items()}


def build_head_df(data: Dict[str, Any]) -> pd.DataFrame:
    cfg = data.get("config", {}) or {}
    n_features = int(cfg.get("feature_subset_size", 0) or 0)
    uniform = (1.0 / n_features) if n_features > 0 else float("nan")

    rows: List[Dict[str, Any]] = []
    layer_results = data.get("layer_results", {}) or {}

    for layer_id, layer_data in layer_results.items():
        layer = int(layer_id)
        is_sw = layer_data.get("is_sliding_window", None)

        # Build per-head writing and RoPE lookups
        writing_map: Dict[int, Dict[str, Any]] = {}
        for wr in (layer_data.get("writing_results", []) or []):
            if not wr:
                continue
            h = wr.get("query_head")
            if h is None:
                continue
            writing_map[int(h)] = wr.get("metrics", {}) or {}
            # keep archetype separately (some dumps nest it in metrics)
            if "archetype" in wr:
                writing_map[int(h)]["_archetype"] = wr.get("archetype")

        rope_map: Dict[int, Dict[str, Any]] = {}
        for rr in (layer_data.get("rope_stability", []) or []):
            if not rr:
                continue
            h = rr.get("query_head")
            if h is None:
                continue
            rope_map[int(h)] = rr

        redund_mean = _build_redundancy_by_head(layer_data)

        for rr in (layer_data.get("routing_results", []) or []):
            if not rr:
                continue
            head = rr.get("query_head")
            if head is None:
                continue
            head = int(head)

            rm = rr.get("metrics", {}) or {}
            wm = writing_map.get(head, {})
            rope = rope_map.get(head, {})

            top1 = _safe_float(rm.get("top1_mass_mean"))
            diag_mass = _safe_float(rm.get("diagonal_softmax_mass"))

            sel_xu = (top1 / uniform) if uniform and uniform > 0 else float("nan")
            diag_xu = (diag_mass / uniform) if uniform and uniform > 0 else float("nan")

            rope_auc = _rope_auc_log(rope.get("stability_by_delta", {}) or {})

            rows.append({
                "layer": layer,
                "head": head,
                "is_sliding_window": is_sw,

                # routing
                "top1_mass_mean": top1,
                "diagonal_softmax_mass": diag_mass,
                "max_gap_mean": _safe_float(rm.get("max_gap_mean")),
                "diagonal_dominance": _safe_float(rm.get("diagonal_dominance")),
                "routing_archetype": rm.get("archetype", "unknown"),

                # scaled-for-blog
                "Sel_xU": sel_xu,
                "DiagM_xU": diag_xu,

                # RoPE
                "semantic_controllability": _safe_float(rope.get("semantic_controllability")),
                "rope_auc_log": rope_auc,

                # writing
                "copy_score": _safe_float(wm.get("copy_score")),
                "transform_score": _safe_float(wm.get("transform_score")),
                "broadcast_score": _safe_float(wm.get("broadcast_score")),
                "suppression_score": _safe_float(wm.get("suppression_score")),
                "writing_archetype": wm.get("archetype", wm.get("_archetype", "unknown")),
                "write_norm_mean": _safe_float(wm.get("write_norm_mean")),

                # redundancy
                "redundancy_mean": redund_mean.get(head, float("nan")),
            })

    df = pd.DataFrame(rows)
    return df


def build_layer_summary(head_df: pd.DataFrame) -> pd.DataFrame:
    if head_df.empty:
        return pd.DataFrame()

    # Use mean across heads per layer (feel free to switch to median)
    agg = {
        "is_sliding_window": "first",
        "Sel_xU": "mean",
        "DiagM_xU": "mean",
        "max_gap_mean": "mean",
        "rope_auc_log": "mean",
        "semantic_controllability": "mean",
        "copy_score": "mean",
        "transform_score": "mean",
        "redundancy_mean": "mean",
    }
    layer_df = head_df.groupby("layer", as_index=False).agg(agg)
    layer_df = layer_df.sort_values("layer")
    return layer_df


def build_program_distribution(data: Dict[str, Any], explicit_only: bool = False) -> pd.DataFrame:
    counts = Counter()
    total = 0

    layer_results = data.get("layer_results", {}) or {}
    for _, layer_data in layer_results.items():
        for pr in (layer_data.get("program_results", []) or []):
            if not pr:
                continue
            top_programs = pr.get("top_programs", []) or []
            for prog in top_programs:
                if not isinstance(prog, dict):
                    continue
                if explicit_only and prog.get("used_fallback_write", False):
                    continue
                ptype = str(prog.get("program_type", "unknown")).upper()
                counts[ptype] += 1
                total += 1

    if total == 0:
        return pd.DataFrame(columns=["program_type", "count", "pct"])  # empty

    rows = []
    for ptype, c in counts.most_common():
        rows.append({
            "program_type": ptype,
            "count": int(c),
            "pct": 100.0 * (c / total),
        })
    return pd.DataFrame(rows)


def _savefig(outdir: str, fname: str) -> None:
    path = os.path.join(outdir, fname)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_sw_vs_global(head_df: pd.DataFrame, outdir: str) -> None:
    if head_df.empty:
        return
    for col, fname, title, ylabel in [
        ("Sel_xU", "fig05_sw_vs_global_selectivity.png", "Selectivity (Sel×U) — sliding-window vs global", "Sel×U"),
        ("DiagM_xU", "fig05_sw_vs_global_diag_mass.png", "Identity sensitivity (DiagM×U) — sliding-window vs global", "DiagM×U"),
    ]:
        if col not in head_df.columns:
            continue

        d = head_df.dropna(subset=[col, "is_sliding_window"]).copy()
        if d.empty:
            continue

        groups = [
            d[d["is_sliding_window"] == True][col].values,
            d[d["is_sliding_window"] == False][col].values,
        ]
        labels = ["sliding (even layers)", "global (odd layers)"]

        plt.figure(figsize=(6.5, 4.5))
        plt.boxplot(groups, labels=labels, showfliers=False)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.grid(True, axis="y", alpha=0.25)
        _savefig(outdir, fname)


def plot_depth_sel_and_diag(layer_df: pd.DataFrame, outdir: str) -> None:
    if layer_df.empty:
        return
    needed = {"layer", "Sel_xU", "DiagM_xU"}
    if not needed.issubset(set(layer_df.columns)):
        return

    plt.figure(figsize=(9.5, 4.5))
    plt.plot(layer_df["layer"], layer_df["Sel_xU"], marker="o", label="Sel×U (top1 mass lift)")
    plt.plot(layer_df["layer"], layer_df["DiagM_xU"], marker="o", label="DiagM×U (diag mass lift)")
    plt.title("Depth map: selectivity and identity sensitivity")
    plt.xlabel("Layer")
    plt.ylabel("Lift over uniform (×U)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    _savefig(outdir, "fig06_depth_selectivity_and_diag.png")


def plot_depth_rope_auc(layer_df: pd.DataFrame, outdir: str) -> None:
    if layer_df.empty:
        return
    if "rope_auc_log" not in layer_df.columns:
        return
    plt.figure(figsize=(9.5, 4.5))
    plt.plot(layer_df["layer"], layer_df["rope_auc_log"], marker="o")
    plt.ylim(0, 1.02)
    plt.title("Depth map: RoPE stability (AUC over log Δ)")
    plt.xlabel("Layer")
    plt.ylabel("RoPE AUC (avg stability)")
    plt.grid(True, alpha=0.25)
    _savefig(outdir, "fig07_depth_rope_auc.png")


def plot_depth_write_archetypes(head_df: pd.DataFrame, outdir: str) -> None:
    if head_df.empty:
        return
    if "writing_archetype" not in head_df.columns:
        return

    d = head_df.dropna(subset=["writing_archetype"]).copy()
    if d.empty:
        return

    # normalize archetype labels (handle enum-style names like "WriteArchetype.BROADCAST")
    d["writing_archetype"] = (
        d["writing_archetype"]
        .astype(str)
        .str.split(".")
        .str[-1]  # take the part after the dot, or the whole string if no dot
        .str.lower()
    )

    # per-layer fractions
    counts = d.groupby(["layer", "writing_archetype"]).size().reset_index(name="count")
    totals = d.groupby("layer").size().reset_index(name="total")
    counts = counts.merge(totals, on="layer", how="left")
    counts["frac"] = counts["count"] / counts["total"].clip(lower=1)

    # focus on the ones you call out in the post
    focus = ["transform", "broadcast", "copy", "suppress", "diffuse"]
    plt.figure(figsize=(9.5, 4.8))
    for arch in focus:
        g = counts[counts["writing_archetype"] == arch].sort_values("layer")
        if g.empty:
            continue
        plt.plot(g["layer"], g["frac"], marker="o", label=arch)

    plt.ylim(0, 1.0)
    plt.title("Depth map: fraction of write archetypes")
    plt.xlabel("Layer")
    plt.ylabel("Fraction of heads")
    plt.grid(True, alpha=0.25)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    _savefig(outdir, "fig08_depth_write_archetype_fractions.png")


def plot_depth_redundancy(layer_df: pd.DataFrame, outdir: str) -> None:
    if layer_df.empty:
        return
    if "redundancy_mean" not in layer_df.columns:
        return

    plt.figure(figsize=(9.5, 4.5))
    plt.plot(layer_df["layer"], layer_df["redundancy_mean"], marker="o")
    plt.ylim(0, 1.02)
    plt.title("Depth map: within-layer redundancy (mean Jaccard on top routing pairs)")
    plt.xlabel("Layer")
    plt.ylabel("Redundancy")
    plt.grid(True, alpha=0.25)
    _savefig(outdir, "fig09_depth_redundancy.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to analysis_*.json")
    ap.add_argument("--outdir", required=True, help="Where to write figures + tables")
    ap.add_argument("--explicit-only", action="store_true", help="Count only programs with explicit W2F evidence")
    args = ap.parse_args()

    _mkdirp(args.outdir)

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    head_df = build_head_df(data)
    head_path = os.path.join(args.outdir, "head_metrics.csv")
    head_df.to_csv(head_path, index=False)
    print(f"Wrote: {head_path}")

    layer_df = build_layer_summary(head_df)
    layer_path = os.path.join(args.outdir, "layer_summary.csv")
    layer_df.to_csv(layer_path, index=False)
    print(f"Wrote: {layer_path}")

    prog_df = build_program_distribution(data, explicit_only=args.explicit_only)
    prog_path = os.path.join(args.outdir, "program_distribution.csv")
    prog_df.to_csv(prog_path, index=False)
    print(f"Wrote: {prog_path}")

    # Plots
    plot_sw_vs_global(head_df, args.outdir)
    plot_depth_sel_and_diag(layer_df, args.outdir)
    plot_depth_rope_auc(layer_df, args.outdir)
    plot_depth_write_archetypes(head_df, args.outdir)
    plot_depth_redundancy(layer_df, args.outdir)

    print(f"Done. Blog assets in: {args.outdir}")


if __name__ == "__main__":
    main()
