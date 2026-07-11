#!/usr/bin/env python3
"""Plot geometry-by-layer metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_INPUT = Path("outputs/geometry_by_layer.csv")
DEFAULT_OUTPUT = Path("outputs/geometry_by_layer.png")


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        cols = ", ".join(missing)
        raise ValueError(f"Missing required columns: {cols}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot per-layer geometry statistics.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to geometry_by_layer.csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the saved plot image (PNG).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot in addition to saving it.",
    )
    return parser


def create_plot(df: pd.DataFrame, output_path: Path, show: bool = False) -> None:
    df = df.sort_values("layer").reset_index(drop=True)
    layer = df["layer"]

    subplot_groups = {
        "Curvature Mean": ["curv_mean", "early_curv_mean", "mid_curv_mean", "late_curv_mean"],
        "Curvature Std": ["curv_std", "early_curv_std", "mid_curv_std", "late_curv_std"],
        "Delta Mean": ["dlt_mean", "early_dlt_mean", "mid_dlt_mean", "late_dlt_mean"],
        "Delta Std": ["dlt_std", "early_dlt_std", "mid_dlt_std", "late_dlt_std"],
        "Spectral": ["lowfreq_ratio", "spec_centroid"],
    }

    for cols in subplot_groups.values():
        _validate_columns(df, cols)

    fig, axes = plt.subplots(len(subplot_groups), 1, figsize=(10, 16), sharex=True)

    for ax, (title, columns) in zip(axes, subplot_groups.items()):
        for column in columns:
            ax.plot(layer, df[column], label=column)
        ax.set_title(title)
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(fontsize="small")

    axes[-1].set_xlabel("Layer")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    csv_path: Path = args.csv
    out_path: Path = args.out

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "layer" not in df.columns:
        raise ValueError("Expected 'layer' column in the CSV.")

    create_plot(df, out_path, show=args.show)


if __name__ == "__main__":
    main()
