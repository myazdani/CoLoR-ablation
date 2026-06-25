#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config


def setup_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    return plt


def sort_variants(frame: pd.DataFrame) -> list[str]:
    order = [
        "full",
        "full_rescore",
        "pair_top1",
        "pair_top2",
        "pair_top4",
        "pair_top6",
        "pair_mid2",
        "pair_mid4",
        "pair_bot1",
        "pair_bot2",
        "pair_bot4",
        "pair_bot6",
        "cond_top1_marg_bot1",
        "cond_top2_marg_bot2",
        "cond_top4_marg_bot4",
        "cond_top6_marg_bot6",
        "cond_bot1_marg_top1",
        "cond_bot2_marg_top2",
        "cond_bot4_marg_top4",
        "cond_bot6_marg_top6",
        "cond_top1_only",
        "cond_top2_only",
        "cond_top4_only",
        "cond_top6_only",
        "marg_top1_only",
        "marg_top2_only",
        "marg_top4_only",
        "marg_top6_only",
    ]
    present = set(frame["variant"].astype(str))
    return [variant for variant in order if variant in present] + sorted(present - set(order))


def plot_metric_bars(metrics: pd.DataFrame, metric: str, output: Path, title: str) -> None:
    plt = setup_matplotlib()
    task_ids = list(dict.fromkeys(metrics["pairwise_task"].astype(str)))
    variants = sort_variants(metrics)
    fig, axes = plt.subplots(len(task_ids), 1, figsize=(max(9, len(variants) * 0.42), 2.6 * len(task_ids)))
    if len(task_ids) == 1:
        axes = [axes]
    for ax, task_id in zip(axes, task_ids):
        sub = metrics[metrics["pairwise_task"] == task_id].set_index("variant").reindex(variants)
        colors = sub["variant_family"].map(
            {
                "baseline": "#4c78a8",
                "paired": "#59a14f",
                "cond_top_marg_bot": "#f28e2b",
                "cond_bot_marg_top": "#e15759",
                "cond_only": "#b07aa1",
                "marg_only": "#76b7b2",
            }
        ).fillna("#9c9c9c")
        ax.bar(np.arange(len(sub)), sub[metric].to_numpy(dtype=float), color=colors)
        ax.set_title(task_id)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(metric)
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(sub.index, rotation=60, ha="right", fontsize=8)
    fig.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_color_shift(shifts: pd.DataFrame, output: Path) -> None:
    plt = setup_matplotlib()
    frame = shifts[shifts["score_name"] == "color"].copy()
    variants = sort_variants(frame)
    tasks = list(dict.fromkeys(frame["pairwise_task"].astype(str)))
    fig, ax = plt.subplots(figsize=(max(10, len(variants) * 0.46), 5.4))
    offsets = np.linspace(-0.32, 0.32, len(tasks))
    width = 0.64 / max(1, len(tasks))
    x = np.arange(len(variants))
    for offset, task_id in zip(offsets, tasks):
        sub = frame[frame["pairwise_task"] == task_id].set_index("variant").reindex(variants)
        ax.bar(x + offset, sub["mean_shift"].to_numpy(dtype=float), width=width, label=task_id)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("mean ablated - full CoLoR")
    ax.set_title("CoLoR score shift by variant and pairwise task")
    ax.legend(ncol=3, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_scatter_sample(results_dir: Path, output: Path, *, max_points: int = 20000, seed: int = 17) -> None:
    plt = setup_matplotlib()
    score_files = sorted(path for path in results_dir.glob("scores_*.parquet") if "full" not in path.stem)
    if not score_files:
        score_files = sorted(results_dir.glob("scores_*.parquet"))
    if not score_files:
        return
    frame = pd.read_parquet(score_files[0])
    if len(frame) > max_points:
        frame = frame.sample(max_points, random_state=seed)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(frame["full_color_score"], frame["ablated_color_score"], s=3, alpha=0.25)
    ax.set_xlabel("full CoLoR score")
    ax.set_ylabel("ablated CoLoR score")
    ax.set_title(f"Ablated vs full CoLoR sample: {score_files[0].stem.removeprefix('scores_')}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot score-pool robustness metrics.")
    parser.add_argument("--config", default="configs/score_pool_robustness.yaml")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = Path(args.results_dir or config["paths"]["output_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    metrics = pd.read_csv(results_dir / "metrics_pairwise.csv")
    shifts = pd.read_csv(results_dir / "score_shift_diagnostics.csv")

    plot_metric_bars(
        metrics,
        "roc_auc",
        figures_dir / "auc_by_variant_and_task.png",
        "ROC AUC by variant and pairwise task",
    )
    plot_metric_bars(
        metrics,
        "average_precision",
        figures_dir / "ap_by_variant_and_task.png",
        "Average precision by variant and pairwise task",
    )
    plot_metric_bars(
        metrics,
        "f1_at_original_cutoff",
        figures_dir / "f1_original_cutoff_by_variant.png",
        "F1 at original tau=64 cutoff",
    )
    plot_metric_bars(
        metrics,
        "f1_at_balanced_rate",
        figures_dir / "f1_balanced_rate_by_variant.png",
        "F1 at fixed predicted-positive rate",
    )
    plot_color_shift(shifts, figures_dir / "color_shift_by_variant.png")
    plot_scatter_sample(results_dir, figures_dir / "ablated_vs_full_color_scatter_sample.png")
    print(f"wrote figures under {figures_dir}")


if __name__ == "__main__":
    main()
