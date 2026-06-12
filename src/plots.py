from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _setup_matplotlib():
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


def _family(variant: str) -> str:
    if variant == "full":
        return "full"
    if variant.startswith("top"):
        return "top"
    if variant.startswith("mid"):
        return "middle"
    if variant.startswith("bot"):
        return "bottom"
    if variant.startswith("skip"):
        return "interleaved"
    return "other"


def plot_overlap(metrics: pd.DataFrame, output: str | Path, *, selection_rate: float) -> None:
    plt = _setup_matplotlib()
    frame = metrics[np.isclose(metrics["selection_rate"], selection_rate)].copy()
    frame["family"] = frame["variant"].map(_family)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for family, group in frame.groupby("family"):
        group = group.sort_values("layers_dropped")
        ax.plot(group["layers_dropped"], group["recall_at_k"], marker="o", label=family)
    ax.axhline(0.80, color="black", linestyle="--", linewidth=1, label="80% benchmark")
    ax.set_xlabel("Layers dropped")
    ax.set_ylabel(f"Recall@top-k, selection rate={selection_rate:g}")
    ax.set_title("Tail Selection Retention Under Matched Layer Ablation")
    ax.legend()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_spearman(metrics: pd.DataFrame, output: str | Path, *, selection_rate: float) -> None:
    plt = _setup_matplotlib()
    frame = metrics[np.isclose(metrics["selection_rate"], selection_rate)].copy()
    frame["family"] = frame["variant"].map(_family)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for family, group in frame.groupby("family"):
        group = group.sort_values("layers_dropped")
        ax.plot(group["layers_dropped"], group["spearman"], marker="o", label=family)
    ax.set_xlabel("Layers dropped")
    ax.set_ylabel("Spearman rho")
    ax.set_title("Full-Pool Rank Agreement")
    ax.legend()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_pareto(metrics: pd.DataFrame, output: str | Path, *, selection_rate: float) -> None:
    plt = _setup_matplotlib()
    frame = metrics[np.isclose(metrics["selection_rate"], selection_rate)].copy()
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(frame["nominal_forward_speedup"], frame["recall_at_k"])
    for _, row in frame.iterrows():
        ax.annotate(row["variant"], (row["nominal_forward_speedup"], row["recall_at_k"]), fontsize=8)
    ax.axhline(0.80, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Nominal forward speedup")
    ax.set_ylabel(f"Recall@top-k, selection rate={selection_rate:g}")
    ax.set_title("Compute-Saving Pareto Curve")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_score_scatter(
    *,
    full_scores: pd.DataFrame,
    variant_scores: pd.DataFrame,
    pool_meta: pd.DataFrame,
    output: str | Path,
    selection_rate: float,
    max_points: int = 20000,
    seed: int = 17,
) -> None:
    plt = _setup_matplotlib()
    merged = pool_meta[["seq_idx", "enriched"]].merge(
        full_scores[["seq_idx", "color"]].rename(columns={"color": "color_full"}),
        on="seq_idx",
    ).merge(
        variant_scores[["seq_idx", "color"]].rename(columns={"color": "color_variant"}),
        on="seq_idx",
    )
    pure = merged[~merged["enriched"].astype(bool)]
    if len(pure) > max_points:
        pure = pure.sample(n=max_points, random_state=seed)
    k = max(1, int(np.floor((~merged["enriched"].astype(bool)).sum() * selection_rate)))
    full_threshold = np.partition(
        merged.loc[~merged["enriched"].astype(bool), "color_full"].to_numpy(), k - 1
    )[k - 1]
    variant_threshold = np.partition(
        merged.loc[~merged["enriched"].astype(bool), "color_variant"].to_numpy(), k - 1
    )[k - 1]

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.scatter(pure["color_full"], pure["color_variant"], s=3, alpha=0.25)
    ax.axvline(full_threshold, color="black", linestyle="--", linewidth=1)
    ax.axhline(variant_threshold, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Full CoLoR score")
    ax.set_ylabel("Ablated CoLoR score")
    ax.set_title("Ablated vs Full Scores")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)

