#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.plots import plot_overlap, plot_pareto, plot_score_scatter, plot_spearman


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures from metrics and scores.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--scatter-variant", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    rate = float(config["metrics"]["main_plot_selection_rate"])
    figures_dir = Path(config["paths"]["figures_dir"])
    metrics = pd.read_csv(config["paths"]["metrics_csv"])

    plot_overlap(metrics, figures_dir / "overlap_at_topk.png", selection_rate=rate)
    plot_spearman(metrics, figures_dir / "spearman_vs_layers.png", selection_rate=rate)
    plot_pareto(metrics, figures_dir / "pareto_speedup_overlap.png", selection_rate=rate)

    if args.scatter_variant:
        results_dir = Path(config["paths"]["results_dir"])
        full = pd.read_parquet(results_dir / "scores_full.parquet")
        variant = pd.read_parquet(results_dir / f"scores_{args.scatter_variant}.parquet")
        meta = pd.read_parquet(config["target"]["pool_meta"])
        plot_score_scatter(
            full_scores=full,
            variant_scores=variant,
            pool_meta=meta,
            output=figures_dir / f"scatter_{args.scatter_variant}.png",
            selection_rate=rate,
        )

    print(f"Wrote figures under {figures_dir}")


if __name__ == "__main__":
    main()

