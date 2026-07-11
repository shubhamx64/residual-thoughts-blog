# viz_report.py
# End-to-end visualization script for Gemma-2 weight-space SAE analysis outputs (main.py JSON).
#
# Usage:
#   python viz_report.py --json "C:\path\to\analysis_*.json" --outdir "C:\path\to\viz_out" --show
#
# Notes:
# - Designed to be schema-robust: skips plots gracefully if fields are missing.
# - Saves PNGs + a leaderboard CSV into --outdir.

import os
import re
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------
# Utilities
# ----------------------------

def mkdirp(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def savefig(outdir: str, name: str, show: bool = False) -> None:
    path = os.path.join(outdir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def safe_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = np.nanmean(s.values)
    sd = np.nanstd(s.values)
    if not np.isfinite(sd) or sd < 1e-12:
        return (s * 0.0)  # all zeros
    return (s - mu) / (sd + 1e-12)

def archetype_short(x: Any) -> str:
    if x is None:
        return "UNKNOWN"
    if isinstance(x, str):
        return x.split(".")[-1]
    return str(x)

def parse_pair_key(pair_str: str) -> Optional[Tuple[int, int]]:
    # pair_str like "(0, 1)"
    m = re.match(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", str(pair_str))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


# ----------------------------
# Parsing: head-level DF
# ----------------------------

def build_head_df(data: Dict[str, Any]) -> pd.DataFrame:
    rows = []

    layer_results = data.get("layer_results", {})
    for layer_id, layer_data in layer_results.items():
        layer_idx = int(layer_id)
        is_sw = layer_data.get("is_sliding_window", None)

        writing_results = layer_data.get("writing_results", []) or []
        writing_map = {wr.get("query_head"): wr for wr in writing_results if wr is not None}

        routing_results = layer_data.get("routing_results", []) or []
        for rr in routing_results:
            head_idx = rr.get("query_head", None)
            if head_idx is None:
                continue

            rm = rr.get("metrics", {}) or {}
            wr = writing_map.get(head_idx, {}) or {}
            wm = wr.get("metrics", {}) or {}

            row = {
                "layer": layer_idx,
                "head": int(head_idx),
                "kv_group": rr.get("kv_group", np.nan),
                "is_sliding_window": is_sw,

                # routing archetype + core routing metrics
                "routing_archetype": archetype_short(rm.get("archetype", "UNKNOWN")),
                "diagonal_dominance": safe_float(rm.get("diagonal_dominance")),
                "diagonal_mean": safe_float(rm.get("diagonal_mean")),
                "diagonal_std": safe_float(rm.get("diagonal_std")),
                "row_entropy_mean": safe_float(rm.get("row_entropy_mean")),
                "row_entropy_std": safe_float(rm.get("row_entropy_std")),
                "top1_mass_mean": safe_float(rm.get("top1_mass_mean")),
                "top5_mass_mean": safe_float(rm.get("top5_mass_mean")),
                "max_gap_mean": safe_float(rm.get("max_gap_mean")),
                "max_gap_max": safe_float(rm.get("max_gap_max")),
                "asymmetry_score": safe_float(rm.get("asymmetry_score")),
                "effective_rank": safe_float(rm.get("effective_rank")),
                "top_singular_ratio": safe_float(rm.get("top_singular_ratio")),
                "mean_affinity": safe_float(rm.get("mean_affinity")),
                "std_affinity": safe_float(rm.get("std_affinity")),
                "max_affinity": safe_float(rm.get("max_affinity")),
                "min_affinity": safe_float(rm.get("min_affinity")),

                # writing archetype + core writing metrics
                "writing_archetype": archetype_short(wm.get("archetype", "UNKNOWN")),
                "copy_score": safe_float(wm.get("copy_score")),
                "broadcast_score": safe_float(wm.get("broadcast_score")),
                "transform_score": safe_float(wm.get("transform_score")),

                # optional writing metrics if present in your JSON
                "write_norm_mean": safe_float(wm.get("write_norm_mean")),
                "write_norm_std": safe_float(wm.get("write_norm_std")),
                "write_sparsity": safe_float(wm.get("write_sparsity")),
            }

            # baseline diffs (if present)
            b_rand_w = rr.get("baseline_random_weights", {}) or {}
            b_rand_d = rr.get("baseline_random_d", {}) or {}
            b_perm = rr.get("baseline_permutation", {}) or {}

            row["effective_rank_minus_randW"] = row["effective_rank"] - safe_float(b_rand_w.get("effective_rank"))
            row["top_singular_ratio_minus_randW"] = row["top_singular_ratio"] - safe_float(b_rand_w.get("top_singular_ratio"))
            row["asymmetry_minus_randW"] = row["asymmetry_score"] - safe_float(b_rand_w.get("asymmetry_score"))

            row["effective_rank_minus_randD"] = row["effective_rank"] - safe_float(b_rand_d.get("effective_rank"))
            row["top_singular_ratio_minus_randD"] = row["top_singular_ratio"] - safe_float(b_rand_d.get("top_singular_ratio"))
            row["asymmetry_minus_randD"] = row["asymmetry_score"] - safe_float(b_rand_d.get("asymmetry_score"))

            row["effective_rank_minus_perm"] = row["effective_rank"] - safe_float(b_perm.get("effective_rank"))
            row["top_singular_ratio_minus_perm"] = row["top_singular_ratio"] - safe_float(b_perm.get("top_singular_ratio"))
            row["asymmetry_minus_perm"] = row["asymmetry_score"] - safe_float(b_perm.get("asymmetry_score"))

            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # enforce numeric cols
    num_cols = [c for c in df.columns if c not in ("routing_archetype", "writing_archetype")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ----------------------------
# Parsing: RoPE stability DF
# ----------------------------

def build_rope_dfs(data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      rope_summary_df: per (layer, head) stats: controllability, rope_auc, rope_halflife, nonmonotonicity, min/max stability
      rope_long_df: long-form points for plotting curves: (layer, head, delta, stability)
    """
    summary_rows = []
    long_rows = []

    layer_results = data.get("layer_results", {})
    for layer_id, layer_data in layer_results.items():
        layer_idx = int(layer_id)
        is_sw = layer_data.get("is_sliding_window", None)

        rope_list = layer_data.get("rope_stability", []) or []
        for r in rope_list:
            head = r.get("query_head", None)
            if head is None:
                continue

            curve = r.get("stability_by_delta", {}) or {}
            # long rows
            deltas = []
            vals = []
            for k, v in curve.items():
                d = int(k)
                s = safe_float(v)
                long_rows.append({
                    "layer": layer_idx,
                    "head": int(head),
                    "is_sliding_window": is_sw,
                    "delta": d,
                    "stability": s,
                })
                if d >= 1 and np.isfinite(s):
                    deltas.append(d)
                    vals.append(s)

            deltas = np.array(deltas, dtype=np.int64)
            vals = np.array(vals, dtype=np.float32)

            # sort by delta
            if len(deltas) > 0:
                order = np.argsort(deltas)
                deltas = deltas[order]
                vals = vals[order]

                x = np.log2(deltas.astype(np.float32))
                # normalized AUC over log2(delta)
                if len(x) >= 2:
                    auc = np.trapz(vals, x)
                    auc_norm = auc / max(1e-9, (x[-1] - x[0]))  # average stability over log-distance
                else:
                    auc_norm = float(vals[0])

                # half-life: smallest delta where stability < 0.5
                below = np.where(vals < 0.5)[0]
                half_life = float(deltas[below[0]]) if len(below) else np.nan

                # nonmonotonicity measure (sum of upward bumps)
                diffs = np.diff(vals)
                nonmono = float(np.sum(np.maximum(0.0, diffs)))

            else:
                auc_norm = np.nan
                half_life = np.nan
                nonmono = np.nan

            summary_rows.append({
                "layer": layer_idx,
                "head": int(head),
                "is_sliding_window": is_sw,
                "semantic_controllability": safe_float(r.get("semantic_controllability")),
                "rope_auc_log": auc_norm,
                "rope_half_life": half_life,
                "rope_nonmonotonicity": nonmono,
                "rope_min_stability": safe_float(r.get("min_stability")),
                "rope_max_stability": safe_float(r.get("max_stability")),
            })

    rope_summary_df = pd.DataFrame(summary_rows)
    rope_long_df = pd.DataFrame(long_rows)

    for df in (rope_summary_df, rope_long_df):
        if not df.empty:
            for c in df.columns:
                if c in ("layer", "head"):
                    continue
                df[c] = pd.to_numeric(df[c], errors="coerce")

    return rope_summary_df, rope_long_df


# ----------------------------
# Parsing: redundancy + head contribution ranking
# ----------------------------

def build_redundancy_df(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Builds per (layer, head) mean redundancy across other heads in that layer.
    """
    rows = []
    layer_results = data.get("layer_results", {})
    for layer_id, layer_data in layer_results.items():
        layer_idx = int(layer_id)
        is_sw = layer_data.get("is_sliding_window", None)
        red = layer_data.get("cross_head_redundancy", {}) or {}
        # Collect for each head
        scores_by_head = {}
        for pair_str, score in red.items():
            ij = parse_pair_key(pair_str)
            if ij is None:
                continue
            i, j = ij
            s = safe_float(score)
            scores_by_head.setdefault(i, []).append(s)
            scores_by_head.setdefault(j, []).append(s)
        for h, scores in scores_by_head.items():
            rows.append({
                "layer": layer_idx,
                "head": int(h),
                "is_sliding_window": is_sw,
                "redundancy_mean": float(np.nanmean(scores)) if len(scores) else np.nan,
                "redundancy_max": float(np.nanmax(scores)) if len(scores) else np.nan,
            })
    return pd.DataFrame(rows)

def build_head_rank_df(data: Dict[str, Any], num_heads: int = 8) -> pd.DataFrame:
    """
    From head_contribution_ranking: list of heads from most->least contributing.
    Returns per (layer, head) rank (0 = best).
    """
    rows = []
    layer_results = data.get("layer_results", {})
    for layer_id, layer_data in layer_results.items():
        layer_idx = int(layer_id)
        is_sw = layer_data.get("is_sliding_window", None)
        ranking = layer_data.get("head_contribution_ranking", None)
        if not ranking:
            continue
        # ranking is list of head ids from best to worst
        rank_map = {int(h): r for r, h in enumerate(ranking)}
        for h in range(num_heads):
            rows.append({
                "layer": layer_idx,
                "head": int(h),
                "is_sliding_window": is_sw,
                "contrib_rank": rank_map.get(h, np.nan),
            })
    return pd.DataFrame(rows)


# ----------------------------
# Plot helpers
# ----------------------------

def label_top_per_layer(ax, df, score_col: str, x_col: str, y_col: str, n_per_layer: int = 1):
    """
    Labels top heads per layer by score_col, at (x_col,y_col).
    """
    if df.empty or score_col not in df.columns:
        return
    tmp = df.dropna(subset=[score_col, x_col, y_col]).copy()
    if tmp.empty:
        return
    for layer, g in tmp.groupby("layer"):
        top = g.nlargest(n_per_layer, score_col)
        for _, r in top.iterrows():
            ax.text(r[x_col], r[y_col], f"L{int(r['layer'])}H{int(r['head'])}", fontsize=9)


# ----------------------------
# Main plotting pipeline
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to analysis_*.json")
    ap.add_argument("--outdir", required=True, help="Directory to write plots + tables")
    ap.add_argument("--show", action="store_true", help="Also show plots interactively")
    ap.add_argument("--top_heads", type=int, default=10, help="How many heads to generate 'head cards' for")
    args = ap.parse_args()

    mkdirp(args.outdir)

    with open(args.json, "r") as f:
        data = json.load(f)

    cfg = data.get("config", {}) or {}
    num_heads = int(cfg.get("num_attention_heads", 8))
    n_features = cfg.get("feature_subset_size", None)
    n_features = int(n_features) if n_features is not None else None

    sns.set_theme(style="whitegrid")

    # Build dataframes
    head_df = build_head_df(data)
    if head_df.empty:
        print("No heads found in JSON. Exiting.")
        return

    rope_summary_df, rope_long_df = build_rope_dfs(data)
    red_df = build_redundancy_df(data)
    rank_df = build_head_rank_df(data, num_heads=num_heads)

    # Merge extra summaries into head_df
    for extra in (rope_summary_df, red_df, rank_df):
        if extra is not None and not extra.empty:
            head_df = head_df.merge(extra, on=["layer", "head", "is_sliding_window"], how="left")

    # Derived normalized metrics
    head_df["transform_abs"] = head_df["transform_score"].abs()

    if n_features is not None and n_features > 0:
        head_df["n_features"] = n_features
        head_df["top1_lift"] = head_df["top1_mass_mean"] * n_features
        head_df["top5_lift"] = head_df["top5_mass_mean"] * n_features
        head_df["broadcast_norm"] = head_df["broadcast_score"] / n_features
        head_df["effective_rank_norm"] = head_df["effective_rank"] / n_features
        head_df["row_entropy_norm"] = head_df["row_entropy_mean"] / np.log(n_features)
    else:
        head_df["n_features"] = np.nan
        head_df["top1_lift"] = np.nan
        head_df["top5_lift"] = np.nan
        head_df["broadcast_norm"] = np.nan
        head_df["effective_rank_norm"] = np.nan
        head_df["row_entropy_norm"] = np.nan

    # Composite "circuit candidate" score:
    # prefer: high top1_lift, high |transform|, high controllability, high rope_auc, high asymmetry, low redundancy
    head_df["z_top1"] = zscore(head_df["top1_lift"])
    head_df["z_transform"] = zscore(head_df["transform_abs"])
    head_df["z_ctrl"] = zscore(head_df.get("semantic_controllability", pd.Series(np.nan, index=head_df.index)))
    head_df["z_rope_auc"] = zscore(head_df.get("rope_auc_log", pd.Series(np.nan, index=head_df.index)))
    head_df["z_asym"] = zscore(head_df["asymmetry_score"])
    head_df["z_redund"] = zscore(head_df.get("redundancy_mean", pd.Series(np.nan, index=head_df.index)))

    head_df["circuit_score"] = (
        head_df["z_top1"] +
        head_df["z_transform"] +
        head_df["z_ctrl"] +
        head_df["z_rope_auc"] +
        0.5 * head_df["z_asym"] -
        0.5 * head_df["z_redund"]
    )

    # Save leaderboard
    leaderboard_cols = [
        "layer", "head", "is_sliding_window",
        "circuit_score",
        "top1_lift", "transform_score", "semantic_controllability", "rope_auc_log", "rope_half_life",
        "asymmetry_score", "broadcast_norm", "effective_rank_norm", "redundancy_mean",
        "routing_archetype", "writing_archetype",
        "contrib_rank",
    ]
    for c in leaderboard_cols:
        if c not in head_df.columns:
            head_df[c] = np.nan

    leaderboard = head_df.sort_values("circuit_score", ascending=False)[leaderboard_cols]
    leaderboard_path = os.path.join(args.outdir, "head_leaderboard.csv")
    leaderboard.to_csv(leaderboard_path, index=False)
    print(f"Wrote leaderboard: {leaderboard_path}")

    # ----------------------------
    # Plot 0: basic coverage
    # ----------------------------
    plt.figure(figsize=(10, 4))
    counts = head_df.groupby("layer").size().reset_index(name="n_heads")
    sns.barplot(data=counts, x="layer", y="n_heads")
    plt.title("Heads per analyzed layer (sanity check)")
    savefig(args.outdir, "00_heads_per_layer.png", show=args.show)

    # ----------------------------
    # Plot 1: top1_lift by layer (distribution + points) [log y]
    # ----------------------------
    if head_df["top1_lift"].notna().any():
        plt.figure(figsize=(12, 5))
        sns.boxplot(data=head_df, x="layer", y="top1_lift", showfliers=False)
        sns.stripplot(data=head_df, x="layer", y="top1_lift", hue="routing_archetype",
                      dodge=True, alpha=0.6, size=4)
        plt.yscale("log")
        plt.title("Routing Selectivity by Layer (top1 lift over uniform) — log scale")
        plt.ylabel("top1_lift (= top1_mass_mean * n_features)")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        savefig(args.outdir, "01_top1_lift_by_layer.png", show=args.show)

    # ----------------------------
    # Plot 2: transform score by layer (distribution)
    # ----------------------------
    if head_df["transform_score"].notna().any():
        plt.figure(figsize=(12, 5))
        sns.violinplot(data=head_df, x="layer", y="transform_score", inner="quartile", cut=0)
        sns.stripplot(data=head_df, x="layer", y="transform_score", color="k", alpha=0.35, size=3, jitter=0.25)
        plt.title("OV Transform Score by Layer (distribution)")
        savefig(args.outdir, "02_transform_by_layer.png", show=args.show)

    # ----------------------------
    # Plot 3: sliding-window vs global comparisons (selectivity, transform, rope_auc, controllability)
    # ----------------------------
    def sw_box(ycol: str, title: str, fname: str):
        if ycol not in head_df.columns or head_df[ycol].notna().sum() == 0:
            return
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=head_df, x="is_sliding_window", y=ycol, showfliers=False)
        sns.stripplot(data=head_df, x="is_sliding_window", y=ycol, color="k", alpha=0.35, size=3, jitter=0.25)
        plt.title(title)
        savefig(args.outdir, fname, show=args.show)

    sw_box("top1_lift", "Selectivity (top1_lift) — sliding-window vs global", "03a_sw_vs_global_top1.png")
    sw_box("transform_score", "Transform — sliding-window vs global", "03b_sw_vs_global_transform.png")
    sw_box("rope_auc_log", "RoPE AUC (avg stability vs log Δ) — sliding-window vs global", "03c_sw_vs_global_rope_auc.png")
    sw_box("semantic_controllability", "Semantic controllability — sliding-window vs global", "03d_sw_vs_global_controllability.png")

    # ----------------------------
    # Plot 4: RoPE stability curves (layer mean) + per-head highlight (top N by rope_auc)
    # ----------------------------
    if not rope_long_df.empty:
        # Layer-mean curve
        tmp = rope_long_df[rope_long_df["delta"] >= 1].copy()
        if not tmp.empty:
            plt.figure(figsize=(10, 6))
            # compute mean across heads per layer per delta
            means = tmp.groupby(["layer", "delta"], as_index=False)["stability"].mean()
            for layer, g in means.groupby("layer"):
                g = g.sort_values("delta")
                plt.plot(g["delta"], g["stability"], linewidth=2.0, label=f"Layer {int(layer)}")
            plt.xscale("log", base=2)
            plt.ylim(0, 1.02)
            plt.title("RoPE Stability (layer mean): similarity(B_Δ, B_0) vs distance")
            plt.xlabel("Relative distance Δ (tokens)")
            plt.ylabel("Similarity")
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            savefig(args.outdir, "04a_rope_stability_layer_mean.png", show=args.show)

        # Highlight top heads by rope_auc
        if "rope_auc_log" in head_df.columns and head_df["rope_auc_log"].notna().any():
            topN = head_df.sort_values("rope_auc_log", ascending=False).head(min(args.top_heads, 12))
            top_set = set(zip(topN["layer"].astype(int).tolist(), topN["head"].astype(int).tolist()))
            tmp2 = rope_long_df[(rope_long_df["delta"] >= 1)].copy()
            tmp2 = tmp2[tmp2.apply(lambda r: (int(r["layer"]), int(r["head"])) in top_set, axis=1)]
            if not tmp2.empty:
                plt.figure(figsize=(10, 6))
                for (layer, head), g in tmp2.groupby(["layer", "head"]):
                    g = g.sort_values("delta")
                    plt.plot(g["delta"], g["stability"], alpha=0.7, linewidth=1.5, label=f"L{int(layer)}H{int(head)}")
                plt.xscale("log", base=2)
                plt.ylim(0, 1.02)
                plt.title(f"RoPE Stability: Top {len(top_set)} heads by RoPE AUC")
                plt.xlabel("Relative distance Δ (tokens)")
                plt.ylabel("Similarity")
                plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
                savefig(args.outdir, "04b_rope_stability_top_heads.png", show=args.show)

    # ----------------------------
    # Plot 5: Circuit complexity scatter (asymmetry vs transform) + label top per layer by circuit_score
    # ----------------------------
    if head_df["asymmetry_score"].notna().any() and head_df["transform_score"].notna().any():
        plt.figure(figsize=(10, 7))
        ax = plt.gca()
        # Scatter
        sns.scatterplot(
            data=head_df,
            x="asymmetry_score",
            y="transform_score",
            hue="layer",
            style="is_sliding_window",
            s=80,
            alpha=0.85,
            ax=ax
        )
        label_top_per_layer(ax, head_df, score_col="circuit_score",
                            x_col="asymmetry_score", y_col="transform_score", n_per_layer=1)
        plt.title("Circuit Complexity: QK Asymmetry vs OV Transform (label top head per layer)")
        plt.xlabel("asymmetry_score")
        plt.ylabel("transform_score")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        savefig(args.outdir, "05_circuit_complexity_scatter.png", show=args.show)

    # ----------------------------
    # Plot 6: Specialization scatter (top1_lift vs broadcast_norm) [log scales] + labels
    # ----------------------------
    if head_df["top1_lift"].notna().any() and head_df["broadcast_norm"].notna().any():
        plt.figure(figsize=(10, 7))
        ax = plt.gca()
        plot_df = head_df.dropna(subset=["top1_lift", "broadcast_norm"]).copy()
        # avoid log(0)
        plot_df = plot_df[(plot_df["top1_lift"] > 0) & (plot_df["broadcast_norm"] > 0)]
        if not plot_df.empty:
            sns.scatterplot(
                data=plot_df,
                x="top1_lift",
                y="broadcast_norm",
                hue="layer",
                style="is_sliding_window",
                size="effective_rank_norm",
                sizes=(40, 220),
                alpha=0.8,
                ax=ax
            )
            ax.set_xscale("log")
            ax.set_yscale("log")

            # Label extremes
            # top broadcasters
            top_b = plot_df.nlargest(3, "broadcast_norm")
            for _, r in top_b.iterrows():
                ax.text(r["top1_lift"], r["broadcast_norm"], f"L{int(r['layer'])}H{int(r['head'])}", fontsize=9)
            # top selectors
            top_s = plot_df.nlargest(3, "top1_lift")
            for _, r in top_s.iterrows():
                ax.text(r["top1_lift"], r["broadcast_norm"], f"L{int(r['layer'])}H{int(r['head'])}", fontsize=9)

            plt.title("Specialization: selectivity (top1_lift) vs broadcast (normalized) [log-log]")
            plt.xlabel("top1_lift")
            plt.ylabel("broadcast_norm (= broadcast_score / n_features)")
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            savefig(args.outdir, "06_specialization_top1_vs_broadcast.png", show=args.show)

    # ----------------------------
    # Plot 7: Contribution ranking heatmap (if available)
    # ----------------------------
    if "contrib_rank" in head_df.columns and head_df["contrib_rank"].notna().any():
        pivot = head_df.pivot_table(index="layer", columns="head", values="contrib_rank", aggfunc="mean")
        plt.figure(figsize=(10, max(3, 0.45 * len(pivot.index))))
        sns.heatmap(pivot.sort_index(), annot=True, cmap="viridis", cbar_kws={"label": "Rank (0=best)"})
        plt.title("Head contribution rank per layer (0 = most contributing)")
        plt.xlabel("Head")
        plt.ylabel("Layer")
        savefig(args.outdir, "07_head_contribution_rank_heatmap.png", show=args.show)

    # ----------------------------
    # Plot 8: Correlation heatmap of numeric metrics (global)
    # ----------------------------
    numeric_cols = [
        "top1_lift", "top5_lift", "diagonal_dominance", "effective_rank_norm",
        "asymmetry_score", "broadcast_norm", "copy_score", "transform_score",
        "semantic_controllability", "rope_auc_log", "rope_half_life",
        "redundancy_mean", "top_singular_ratio"
    ]
    numeric_cols = [c for c in numeric_cols if c in head_df.columns and head_df[c].notna().sum() > 5]
    if len(numeric_cols) >= 4:
        corr = head_df[numeric_cols].corr(numeric_only=True)
        plt.figure(figsize=(1.1 * len(numeric_cols), 0.9 * len(numeric_cols)))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("Metric correlation heatmap (heads pooled)")
        savefig(args.outdir, "08_metric_correlation_heatmap.png", show=args.show)

    # ----------------------------
    # Plot 9: Redundancy heatmaps per layer (one PNG per layer)
    # ----------------------------
    layer_results = data.get("layer_results", {})
    for layer_id, layer_data in layer_results.items():
        layer_idx = int(layer_id)
        red = layer_data.get("cross_head_redundancy", {}) or {}
        if not red:
            continue

        mat = np.zeros((num_heads, num_heads), dtype=np.float32)
        mat[:] = np.nan
        np.fill_diagonal(mat, 1.0)

        for pair_str, score in red.items():
            ij = parse_pair_key(pair_str)
            if ij is None:
                continue
            i, j = ij
            s = safe_float(score)
            mat[i, j] = s
            mat[j, i] = s

        plt.figure(figsize=(6, 5))
        sns.heatmap(mat, annot=True, cmap="Reds", vmin=0, vmax=1, square=True)
        plt.title(f"Layer {layer_idx}: Inter-head feature redundancy")
        plt.xlabel("Head")
        plt.ylabel("Head")
        savefig(args.outdir, f"09_redundancy_layer_{layer_idx}.png", show=args.show)

    # ----------------------------
    # Plot 10: "Head cards" for top heads by circuit_score
    # ----------------------------
    top_heads = head_df.sort_values("circuit_score", ascending=False).head(args.top_heads)
    for _, r in top_heads.iterrows():
        layer = int(r["layer"])
        head = int(r["head"])

        # Grab rope curve for this head (if present)
        curve = None
        if not rope_long_df.empty:
            g = rope_long_df[(rope_long_df["layer"] == layer) & (rope_long_df["head"] == head)].copy()
            if not g.empty:
                g = g.sort_values("delta")
                curve = (g["delta"].values, g["stability"].values)

        # Build the card
        plt.figure(figsize=(10, 4))
        gs = plt.GridSpec(1, 2, width_ratios=[1.2, 1.0])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        # Left: RoPE curve
        ax1.set_title(f"RoPE stability — L{layer}H{head}")
        if curve is not None:
            d, v = curve
            # skip delta=0 for log axis
            mask = d >= 1
            d2 = d[mask]
            v2 = v[mask]
            ax1.plot(d2, v2, linewidth=2.0)
            ax1.set_xscale("log", base=2)
            ax1.set_ylim(0, 1.02)
            ax1.set_xlabel("Δ (tokens)")
            ax1.set_ylabel("similarity")
            ax1.grid(True, which="both", alpha=0.2)
        else:
            ax1.text(0.1, 0.5, "No RoPE curve in JSON", transform=ax1.transAxes)

        # Right: key metrics bar
        metrics = {
            "circuit_score": r.get("circuit_score", np.nan),
            "top1_lift": r.get("top1_lift", np.nan),
            "transform": r.get("transform_score", np.nan),
            "ctrl": r.get("semantic_controllability", np.nan),
            "rope_auc": r.get("rope_auc_log", np.nan),
            "asym": r.get("asymmetry_score", np.nan),
            "broadcast_norm": r.get("broadcast_norm", np.nan),
            "redund": r.get("redundancy_mean", np.nan),
        }
        names = list(metrics.keys())
        vals = [safe_float(metrics[k]) for k in names]
        ax2.barh(names, vals)
        ax2.set_title("Key metrics")
        ax2.grid(True, axis="x", alpha=0.2)

        plt.suptitle(
            f"L{layer}H{head} | SW={r.get('is_sliding_window')} | "
            f"route={r.get('routing_archetype')} write={r.get('writing_archetype')}",
            y=1.05
        )
        savefig(args.outdir, f"10_head_card_L{layer}_H{head}.png", show=args.show)

    print(f"Done. Wrote plots + tables to: {args.outdir}")


if __name__ == "__main__":
    main()
