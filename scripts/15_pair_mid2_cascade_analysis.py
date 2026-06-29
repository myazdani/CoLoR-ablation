#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ensure_parent, load_config
from src.score_pool_robustness import (
    PAIRWISE_TASKS,
    average_precision_from_scores,
    binary_metrics,
    roc_auc_from_scores,
)


DEFAULT_LOCAL_INPUT = (
    ROOT
    / "data"
    / "pair_mid2_cascade_full_rerank"
    / "results"
    / "score-pool-robustness-official-500k"
)
DEFAULT_MULTIPLIERS = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 8.0)
DEFAULT_SELECTION_RATES = (1 / 64, 1 / 32, 1 / 16, 1 / 8)
SCORE_COLUMNS = [
    "seq_idx",
    "pool_name",
    "ablated_color_score",
    "elapsed_seconds",
    "tokens_scored",
    "tokens_per_second",
]


@dataclass(frozen=True)
class RuntimeStats:
    variant: str
    elapsed_seconds: float
    tokens_scored: int
    tokens_per_second: float
    rows: int


def parse_float_list(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "/" in item:
            num, denom = item.split("/", 1)
            values.append(float(num) / float(denom))
        else:
            values.append(float(item))
    if not values:
        raise ValueError(f"No numeric values parsed from {raw!r}")
    return tuple(values)


def format_multiplier(value: float) -> str:
    return f"{value:g}"


def format_rate(value: float) -> str:
    for denom in (64, 32, 16, 8, 4, 2):
        if math.isclose(value, 1 / denom):
            return f"1/{denom}"
    return f"{value:g}"


def select_lowest_k(seq_idx: np.ndarray, scores: np.ndarray, k: int) -> set[int]:
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return set()
    if k >= len(scores):
        return set(int(x) for x in seq_idx)
    selected = np.argpartition(np.asarray(scores, dtype=np.float64), k - 1)[:k]
    return set(int(x) for x in seq_idx[selected])


def prediction_from_ids(seq_idx: np.ndarray, selected_ids: set[int]) -> np.ndarray:
    return np.fromiter((int(x) in selected_ids for x in seq_idx), dtype=bool, count=len(seq_idx))


def read_score_frame(path: Path, *, score_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing score parquet: {path}")
    frame = pd.read_parquet(path, columns=SCORE_COLUMNS)
    frame = frame.rename(columns={"ablated_color_score": score_name})
    if frame["seq_idx"].duplicated().any():
        raise ValueError(f"{path} contains duplicate seq_idx values")
    return frame


def runtime_stats(path: Path, *, variant: str) -> RuntimeStats:
    frame = pd.read_parquet(path, columns=["seq_idx", "elapsed_seconds", "tokens_scored", "tokens_per_second"])
    stats = frame[["elapsed_seconds", "tokens_scored", "tokens_per_second"]].drop_duplicates()
    if len(stats) != 1:
        raise ValueError(f"Expected one aggregate runtime row in {path}, found {len(stats)}")
    row = stats.iloc[0]
    return RuntimeStats(
        variant=variant,
        elapsed_seconds=float(row["elapsed_seconds"]),
        tokens_scored=int(row["tokens_scored"]),
        tokens_per_second=float(row["tokens_per_second"]),
        rows=len(frame),
    )


def load_joined_scores(results_dir: Path) -> tuple[pd.DataFrame, RuntimeStats, RuntimeStats]:
    full_path = results_dir / "scores_full.parquet"
    pair_mid2_path = results_dir / "scores_pair_mid2.parquet"
    full = read_score_frame(full_path, score_name="full_color_score_local")
    pair_mid2 = read_score_frame(pair_mid2_path, score_name="pair_mid2_color_score")
    joined = full[["seq_idx", "pool_name", "full_color_score_local"]].merge(
        pair_mid2[["seq_idx", "pool_name", "pair_mid2_color_score"]],
        on=["seq_idx", "pool_name"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(full) or len(joined) != len(pair_mid2):
        raise ValueError("Full and pair_mid2 score frames do not align one-to-one")
    return joined, runtime_stats(full_path, variant="full"), runtime_stats(pair_mid2_path, variant="pair_mid2")


def _candidate_and_final_sets(
    frame: pd.DataFrame,
    *,
    k: int,
    multiplier: float,
) -> tuple[set[int], set[int], set[int], int]:
    seq_idx = frame["seq_idx"].to_numpy(dtype=np.int64)
    pair_mid2 = frame["pair_mid2_color_score"].to_numpy(dtype=np.float64)
    full = frame["full_color_score_local"].to_numpy(dtype=np.float64)
    reference = select_lowest_k(seq_idx, full, k)
    candidate_count = min(len(frame), int(math.ceil(multiplier * k)))
    candidates = select_lowest_k(seq_idx, pair_mid2, candidate_count)
    candidate_frame = frame[frame["seq_idx"].isin(candidates)]
    final = select_lowest_k(
        candidate_frame["seq_idx"].to_numpy(dtype=np.int64),
        candidate_frame["full_color_score_local"].to_numpy(dtype=np.float64),
        min(k, len(candidate_frame)),
    )
    return reference, candidates, final, candidate_count


def _overlap_metrics(selected: set[int], reference: set[int]) -> dict[str, float]:
    intersection = len(selected & reference)
    union = len(selected | reference)
    recall = intersection / len(reference) if reference else float("nan")
    precision = intersection / len(selected) if selected else float("nan")
    jaccard = intersection / union if union else float("nan")
    return {
        "intersection": intersection,
        "recall": recall,
        "precision": precision,
        "jaccard": jaccard,
    }


def compute_pairwise_cascade_metrics(
    scores: pd.DataFrame,
    *,
    multipliers: Iterable[float],
    pairwise_tasks: Iterable[str],
    full_runtime: RuntimeStats,
    pair_mid2_runtime: RuntimeStats,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    global_rows = len(scores)
    for task_id in pairwise_tasks:
        positive_pool, negative_pool = PAIRWISE_TASKS[task_id]
        pair = scores[scores["pool_name"].isin([positive_pool, negative_pool])].copy()
        labels = (pair["pool_name"] == positive_pool).to_numpy(dtype=np.int8)
        k = int(labels.sum())
        if k <= 0:
            raise ValueError(f"No positives for pairwise task {task_id}")
        seq_idx = pair["seq_idx"].to_numpy(dtype=np.int64)
        full_reference = select_lowest_k(seq_idx, pair["full_color_score_local"].to_numpy(), k)
        pair_mid2_direct = select_lowest_k(seq_idx, pair["pair_mid2_color_score"].to_numpy(), k)
        full_direct = select_lowest_k(seq_idx, pair["full_color_score_local"].to_numpy(), k)
        pair_mid2_pred = prediction_from_ids(seq_idx, pair_mid2_direct)
        full_pred = prediction_from_ids(seq_idx, full_direct)
        direct_pair_mid2 = binary_metrics(labels, pair_mid2_pred)
        direct_full = binary_metrics(labels, full_pred)
        roc_auc = roc_auc_from_scores(labels, -pair["pair_mid2_color_score"].to_numpy(dtype=np.float64))
        average_precision = average_precision_from_scores(
            labels,
            -pair["pair_mid2_color_score"].to_numpy(dtype=np.float64),
        )

        for multiplier in multipliers:
            _, candidates, final, candidate_count = _candidate_and_final_sets(
                pair,
                k=k,
                multiplier=multiplier,
            )
            candidate_overlap = _overlap_metrics(candidates, full_reference)
            final_overlap = _overlap_metrics(final, full_reference)
            cascade_pred = prediction_from_ids(seq_idx, final)
            cascade_official = binary_metrics(labels, cascade_pred)
            pair_rows = len(pair)
            full_scope_seconds = full_runtime.elapsed_seconds * (pair_rows / global_rows)
            pair_mid2_scope_seconds = pair_mid2_runtime.elapsed_seconds * (pair_rows / global_rows)
            candidate_full_seconds = full_runtime.elapsed_seconds * (candidate_count / global_rows)
            cascade_seconds = pair_mid2_scope_seconds + candidate_full_seconds
            candidate_fraction = candidate_count / pair_rows
            rows.append(
                {
                    "pairwise_task": task_id,
                    "positive_pool": positive_pool,
                    "negative_pool": negative_pool,
                    "multiplier": multiplier,
                    "n_rows": pair_rows,
                    "k": k,
                    "candidate_count": candidate_count,
                    "candidate_fraction": candidate_fraction,
                    "candidate_recall_vs_full": candidate_overlap["recall"],
                    "candidate_precision_vs_full": candidate_overlap["precision"],
                    "candidate_jaccard_vs_full": candidate_overlap["jaccard"],
                    "final_recall_vs_full": final_overlap["recall"],
                    "final_precision_vs_full": final_overlap["precision"],
                    "final_jaccard_vs_full": final_overlap["jaccard"],
                    "pair_mid2_roc_auc_vs_official": roc_auc,
                    "pair_mid2_average_precision_vs_official": average_precision,
                    "pair_mid2_direct_f1_vs_official": direct_pair_mid2["f1"],
                    "pair_mid2_direct_precision_vs_official": direct_pair_mid2["precision"],
                    "pair_mid2_direct_recall_vs_official": direct_pair_mid2["recall"],
                    "full_direct_f1_vs_official": direct_full["f1"],
                    "full_direct_precision_vs_official": direct_full["precision"],
                    "full_direct_recall_vs_official": direct_full["recall"],
                    "cascade_f1_vs_official": cascade_official["f1"],
                    "cascade_precision_vs_official": cascade_official["precision"],
                    "cascade_recall_vs_official": cascade_official["recall"],
                    "estimated_full_compute_saved": 1.0 - candidate_fraction,
                    "effective_speedup_upper_bound_ignore_prefilter": 1.0 / candidate_fraction
                    if candidate_fraction > 0
                    else float("inf"),
                    "estimated_full_only_seconds": full_scope_seconds,
                    "estimated_pair_mid2_prefilter_seconds": pair_mid2_scope_seconds,
                    "estimated_candidate_full_seconds": candidate_full_seconds,
                    "estimated_total_cascade_seconds": cascade_seconds,
                    "estimated_end_to_end_speedup_vs_full": full_scope_seconds / cascade_seconds
                    if cascade_seconds > 0
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def compute_full_pool_cascade_metrics(
    scores: pd.DataFrame,
    *,
    multipliers: Iterable[float],
    selection_rates: Iterable[float],
    full_runtime: RuntimeStats,
    pair_mid2_runtime: RuntimeStats,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    n_rows = len(scores)
    for rate in selection_rates:
        k = int(math.ceil(n_rows * rate))
        for multiplier in multipliers:
            reference, candidates, final, candidate_count = _candidate_and_final_sets(
                scores,
                k=k,
                multiplier=multiplier,
            )
            candidate_overlap = _overlap_metrics(candidates, reference)
            final_overlap = _overlap_metrics(final, reference)
            candidate_fraction = candidate_count / n_rows
            candidate_full_seconds = full_runtime.elapsed_seconds * candidate_fraction
            cascade_seconds = pair_mid2_runtime.elapsed_seconds + candidate_full_seconds
            rows.append(
                {
                    "selection_rate": rate,
                    "selection_rate_label": format_rate(rate),
                    "multiplier": multiplier,
                    "n_rows": n_rows,
                    "k": k,
                    "candidate_count": candidate_count,
                    "candidate_fraction": candidate_fraction,
                    "candidate_recall_vs_full": candidate_overlap["recall"],
                    "candidate_precision_vs_full": candidate_overlap["precision"],
                    "candidate_jaccard_vs_full": candidate_overlap["jaccard"],
                    "final_recall_vs_full": final_overlap["recall"],
                    "final_precision_vs_full": final_overlap["precision"],
                    "final_jaccard_vs_full": final_overlap["jaccard"],
                    "estimated_full_compute_saved": 1.0 - candidate_fraction,
                    "effective_speedup_upper_bound_ignore_prefilter": 1.0 / candidate_fraction
                    if candidate_fraction > 0
                    else float("inf"),
                    "estimated_full_only_seconds": full_runtime.elapsed_seconds,
                    "estimated_pair_mid2_prefilter_seconds": pair_mid2_runtime.elapsed_seconds,
                    "estimated_candidate_full_seconds": candidate_full_seconds,
                    "estimated_total_cascade_seconds": cascade_seconds,
                    "estimated_end_to_end_speedup_vs_full": full_runtime.elapsed_seconds / cascade_seconds
                    if cascade_seconds > 0
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def compute_runtime_estimates(
    *,
    full_runtime: RuntimeStats,
    pair_mid2_runtime: RuntimeStats,
    full_pool_metrics: pd.DataFrame,
    pairwise_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = [
        {
            "scope": "measured",
            "label": "full",
            "rows": full_runtime.rows,
            "elapsed_seconds": full_runtime.elapsed_seconds,
            "tokens_scored": full_runtime.tokens_scored,
            "tokens_per_second": full_runtime.tokens_per_second,
            "relative_to_full": 1.0,
        },
        {
            "scope": "measured",
            "label": "pair_mid2",
            "rows": pair_mid2_runtime.rows,
            "elapsed_seconds": pair_mid2_runtime.elapsed_seconds,
            "tokens_scored": pair_mid2_runtime.tokens_scored,
            "tokens_per_second": pair_mid2_runtime.tokens_per_second,
            "relative_to_full": pair_mid2_runtime.elapsed_seconds / full_runtime.elapsed_seconds,
        },
    ]
    for frame, scope, label_col in (
        (full_pool_metrics, "full_pool_cascade", "selection_rate_label"),
        (pairwise_metrics, "pairwise_cascade", "pairwise_task"),
    ):
        for _, row in frame.iterrows():
            label = f"{row[label_col]}@m={format_multiplier(float(row['multiplier']))}"
            rows.append(
                {
                    "scope": scope,
                    "label": label,
                    "rows": int(row["n_rows"]),
                    "elapsed_seconds": float(row["estimated_total_cascade_seconds"]),
                    "tokens_scored": float("nan"),
                    "tokens_per_second": float("nan"),
                    "relative_to_full": float(row["estimated_total_cascade_seconds"])
                    / float(row["estimated_full_only_seconds"]),
                    "candidate_fraction": float(row["candidate_fraction"]),
                    "estimated_end_to_end_speedup_vs_full": float(
                        row["estimated_end_to_end_speedup_vs_full"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def setup_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    return plt


def write_plots(pairwise: pd.DataFrame, full_pool: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for label, group in full_pool.groupby("selection_rate_label", sort=False):
        group = group.sort_values("multiplier")
        ax.plot(group["multiplier"], group["final_recall_vs_full"], marker="o", label=label)
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1, label="0.95 recall")
    ax.set_xlabel("candidate multiplier m")
    ax.set_ylabel("final recall vs full")
    ax.set_title("Cascade recall after full reranking")
    ax.set_ylim(0, 1.03)
    ax.legend(title="selection rate")
    fig.savefig(figures_dir / "recall_vs_candidate_multiplier.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for label, group in full_pool.groupby("selection_rate_label", sort=False):
        group = group.sort_values("multiplier")
        ax.plot(group["multiplier"], group["final_jaccard_vs_full"], marker="o", label=label)
    ax.set_xlabel("candidate multiplier m")
    ax.set_ylabel("final Jaccard vs full")
    ax.set_title("Cascade Jaccard after full reranking")
    ax.set_ylim(0, 1.03)
    ax.legend(title="selection rate")
    fig.savefig(figures_dir / "jaccard_vs_candidate_multiplier.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for label, group in full_pool.groupby("selection_rate_label", sort=False):
        ax.scatter(
            group["final_recall_vs_full"],
            group["estimated_end_to_end_speedup_vs_full"],
            s=50,
            label=label,
        )
        for _, row in group.iterrows():
            ax.annotate(
                format_multiplier(float(row["multiplier"])),
                (row["final_recall_vs_full"], row["estimated_end_to_end_speedup_vs_full"]),
                fontsize=8,
                xytext=(3, 3),
                textcoords="offset points",
            )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0.95, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("final recall vs full")
    ax.set_ylabel("estimated end-to-end speedup vs full")
    ax.set_title("Speed vs recall including pair_mid2 prefilter cost")
    ax.legend(title="selection rate")
    fig.savefig(figures_dir / "speedup_vs_recall.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for task_id, group in pairwise.groupby("pairwise_task", sort=False):
        group = group.sort_values("multiplier")
        ax.plot(group["multiplier"], group["cascade_f1_vs_official"], marker="o", label=task_id)
    ax.set_xlabel("candidate multiplier m")
    ax.set_ylabel("cascade F1 vs official labels")
    ax.set_title("Pairwise cascade F1 by task")
    ax.set_ylim(0, 1.03)
    ax.legend(ncol=2)
    fig.savefig(figures_dir / "pairwise_cascade_f1_by_task.png")
    plt.close(fig)


def dataframe_to_markdown(frame: pd.DataFrame, *, index: bool = False, floatfmt: str = ".4f") -> str:
    display = frame.reset_index() if index else frame.copy()

    def format_value(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        return str(value)

    headers = [str(column) for column in display.columns]
    rows = [[format_value(value) for value in row] for row in display.itertuples(index=False, name=None)]
    widths = [
        max(len(header), *(len(row[column_idx]) for row in rows)) if rows else len(header)
        for column_idx, header in enumerate(headers)
    ]

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render_row(headers), separator, *(render_row(row) for row in rows)])


def _summarize_decision(full_pool: pd.DataFrame, pairwise: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    practical = full_pool[(full_pool["multiplier"] <= 4.0) & (full_pool["final_recall_vs_full"] >= 0.95)]
    practical_speedup = practical[practical["estimated_end_to_end_speedup_vs_full"] > 1.0]
    hp_hn_unsaturated = pairwise[
        (pairwise["pairwise_task"] == "hp_vs_hn") & (pairwise["candidate_fraction"] < 1.0)
    ]
    hp_hn_best_unsaturated = (
        float(hp_hn_unsaturated["candidate_recall_vs_full"].max())
        if not hp_hn_unsaturated.empty
        else float("nan")
    )
    direct = (
        pairwise[["pairwise_task", "pair_mid2_direct_f1_vs_official", "full_direct_f1_vs_official"]]
        .drop_duplicates()
        .copy()
    )
    decision_rows = [
        {
            "criterion": "m<=4 reaches >=95% full-reference recall",
            "result": "yes" if not practical.empty else "no",
            "evidence": f"{len(practical)} full-pool settings pass",
        },
        {
            "criterion": "m<=4 reaches >=95% recall and estimated end-to-end speedup >1",
            "result": "yes" if not practical_speedup.empty else "no",
            "evidence": f"{len(practical_speedup)} full-pool settings pass",
        },
        {
            "criterion": "pair_mid2 direct classifier matches full-model behavior",
            "result": "no",
            "evidence": (
                f"mean direct F1={direct['pair_mid2_direct_f1_vs_official'].mean():.4f}; "
                f"mean full F1={direct['full_direct_f1_vs_official'].mean():.4f}"
            ),
        },
        {
            "criterion": "candidate recall is strong on hp_vs_hn before pairwise saturation",
            "result": "no",
            "evidence": f"best unsaturated hp_vs_hn recall={hp_hn_best_unsaturated:.4f}",
        },
    ]
    if practical_speedup.empty:
        decision = (
            "Pair-mid2 is not useful as a standalone replacement, and this run does not "
            "show a passing speed-recall cascade setting under the tested criteria. Its "
            "remaining value is as a diagnostic proxy score rather than a deployment-ready "
            "fast scorer."
        )
    else:
        best = practical_speedup.sort_values(
            ["estimated_end_to_end_speedup_vs_full", "final_recall_vs_full"],
            ascending=[False, False],
        ).iloc[0]
        decision = (
            "Pair-mid2 is not useful as a standalone replacement, but it is useful as a "
            "cascade prefilter for small full-pool selection rates in the tested regime. "
            f"The strongest passing setting is selection rate {best['selection_rate_label']} "
            f"at m={format_multiplier(float(best['multiplier']))}, with "
            f"{best['final_recall_vs_full']:.4f} recall and "
            f"{best['estimated_end_to_end_speedup_vs_full']:.4f}x estimated speedup. "
            "The main failure case is the hard-positive vs hard-negative comparison, where "
            "unsaturated pair_mid2 candidate recall is weak."
        )
    return decision, pd.DataFrame(decision_rows)


def write_report(
    *,
    output_dir: Path,
    report_dir: Path,
    figures_dir: Path,
    pairwise: pd.DataFrame,
    full_pool: pd.DataFrame,
    runtime: pd.DataFrame,
    full_runtime: RuntimeStats,
    pair_mid2_runtime: RuntimeStats,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_full = (
        full_pool.sort_values(["selection_rate", "multiplier"])
        .loc[
            :,
            [
                "selection_rate_label",
                "multiplier",
                "candidate_fraction",
                "final_recall_vs_full",
                "final_jaccard_vs_full",
                "estimated_end_to_end_speedup_vs_full",
            ],
        ]
        .copy()
    )
    best_95 = (
        full_pool[full_pool["final_recall_vs_full"] >= 0.95]
        .sort_values(["selection_rate", "candidate_fraction"])
        .groupby("selection_rate_label", as_index=False)
        .first()
    )
    pairwise_m4 = pairwise[pairwise["multiplier"] == 4.0][
        [
            "pairwise_task",
            "candidate_recall_vs_full",
            "final_recall_vs_full",
            "cascade_f1_vs_official",
            "pair_mid2_direct_f1_vs_official",
            "full_direct_f1_vs_official",
            "estimated_end_to_end_speedup_vs_full",
        ]
    ].copy()
    pairwise_direct = (
        pairwise[
            [
                "pairwise_task",
                "pair_mid2_roc_auc_vs_official",
                "pair_mid2_average_precision_vs_official",
                "pair_mid2_direct_f1_vs_official",
                "full_direct_f1_vs_official",
            ]
        ]
        .drop_duplicates()
        .copy()
    )
    measured = runtime[runtime["scope"] == "measured"][
        ["label", "rows", "elapsed_seconds", "tokens_per_second", "relative_to_full"]
    ]
    decision_text, decision_table = _summarize_decision(full_pool, pairwise)

    lines = [
        "# Pair-Mid2 Cascade With Full-Model Reranking",
        "",
        "## Abstract",
        "",
        "This report evaluates whether the `pair_mid2` layer-deletion scorer can reduce full-model CoLoR scoring cost through a two-stage cascade. The cascade ranks all rows with `pair_mid2`, keeps an expanded candidate set, and then uses full Books CoLoR scores only as a reranker inside that candidate set. The analysis is offline because the official 500K pool has already been scored by both the full model and `pair_mid2`; this lets us estimate recall, Jaccard overlap, and runtime tradeoffs without launching another GPU job.",
        "",
        decision_text,
        "",
        "## Background and Motivation",
        "",
        "The prior official 500K robustness run showed that many layer-deleted variants preserve some CoLoR ranking signal but do not exactly reproduce the full-model selection. `pair_mid2`, which deletes the middle two layers from both marginal and conditional models, was one of the strongest layer-deletion approximations. That makes it a natural candidate for a cascade: use the cheap scorer to avoid running the full scorer everywhere, then let the full scorer make the final boundary decision on a smaller candidate set.",
        "",
        "The central question is not whether `pair_mid2` can replace the full scorer. The useful question is whether it has enough recall at the top of the ranking to act as a prefilter. A prefilter can tolerate many false positives if the candidate set is still much smaller than the full pool and contains most of the full-model positives.",
        "",
        "## Methods",
        "",
        "For each pairwise pool task, positives and negatives are the same as in the score-pool robustness analysis. Each task has 100K positive rows and 100K negative rows. For a candidate multiplier `m`, we keep the lowest `m * k` rows by `pair_mid2` CoLoR score, where `k` is the number of positives, then rerank those candidates by the full-model CoLoR score and select the lowest `k` rows.",
        "",
        "For the full-pool analysis, the reference set is the lowest `k` rows by the local full-model score over all 500K rows. Selection rates are `1/64`, `1/32`, `1/16`, and `1/8`. The candidate set is again selected by `pair_mid2`, and the final simulated selection is selected by full-model score inside the candidate set.",
        "",
        "Runtime estimates use the measured official 500K elapsed times saved in the score parquet files. The end-to-end cascade estimate includes both the first-stage `pair_mid2` pass over all rows and the full-model rerank pass over the retained candidate fraction.",
        "",
        "## Experimental Setup",
        "",
        "- Dataset: official 500K score-pool sample with five 100K pools.",
        "- Score sign: lower CoLoR score is better, matching the paper-sign convention.",
        "- Reference full scorer: `scores_full.parquet`.",
        "- Cheap prefilter scorer: `scores_pair_mid2.parquet`.",
        "- Candidate multipliers: `1`, `1.25`, `1.5`, `2`, `3`, `4`, `8`.",
        "- Full-pool selection rates: `1/64`, `1/32`, `1/16`, `1/8`.",
        "",
        "## Measured Runtime Inputs",
        "",
        dataframe_to_markdown(measured, floatfmt=".4f"),
        "",
        "## Full-Pool Cascade Results",
        "",
        "The table below reports final overlap with the local full-model selected set after full reranking of the `pair_mid2` candidate set. End-to-end speedup includes the cost of scoring all rows with `pair_mid2`.",
        "",
        dataframe_to_markdown(summary_full, floatfmt=".4f"),
        "",
        "## Decision Against Criteria",
        "",
        dataframe_to_markdown(decision_table, floatfmt=".4f"),
        "",
        "## Smallest Candidate Multipliers Reaching 95% Recall",
        "",
    ]
    if best_95.empty:
        lines.extend(["No tested candidate multiplier reached 95% recall for any selection rate.", ""])
    else:
        lines.extend(
            [
                dataframe_to_markdown(
                    best_95[
                        [
                            "selection_rate_label",
                            "multiplier",
                            "candidate_fraction",
                            "final_recall_vs_full",
                            "final_jaccard_vs_full",
                            "estimated_end_to_end_speedup_vs_full",
                        ]
                    ],
                    floatfmt=".4f",
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Pairwise Cascade Diagnostics",
            "",
            "The direct `pair_mid2` metrics below show how useful the ablated score is as a standalone ranking signal before adding full-model reranking.",
            "",
            dataframe_to_markdown(pairwise_direct, floatfmt=".4f"),
            "",
            "At `m=4`, pairwise tasks with only 200K rows are saturated: the candidate set contains all rows because `4 * k` exceeds the task size. These diagnostics are therefore most useful for showing task difficulty and final full-rerank behavior, not deployment speed.",
            "",
            dataframe_to_markdown(pairwise_m4, floatfmt=".4f"),
            "",
            "## Figures",
            "",
            "![Recall vs candidate multiplier](figures/recall_vs_candidate_multiplier.png)",
            "",
            "![Jaccard vs candidate multiplier](figures/jaccard_vs_candidate_multiplier.png)",
            "",
            "![Speedup vs recall](figures/speedup_vs_recall.png)",
            "",
            "![Pairwise cascade F1 by task](figures/pairwise_cascade_f1_by_task.png)",
            "",
            "## Interpretation",
            "",
            "- The cascade is most attractive for small final selection rates such as `1/64`, where a `4x` candidate set is only `6.25%` of the full pool.",
            "- Because `pair_mid2` itself costs about "
            f"`{pair_mid2_runtime.elapsed_seconds / full_runtime.elapsed_seconds:.3f}x` of a full pass in the measured official run, candidate reranking must be very small to produce an end-to-end speedup.",
            "- Pairwise pool tasks are balanced at 100K positives and 100K negatives; for these diagnostics, multipliers `>=2` already include the whole pairwise task and cannot estimate deployment speed savings.",
            "- The analysis is an offline simulation because full scores already exist for the official 500K pool. A production cascade would need to avoid full scoring outside the candidate set.",
            "",
            "## Limitations",
            "",
            "- These are score-level simulations, not measured candidate-only GPU runs.",
            "- The official 500K pool is intentionally enriched around the score boundary and is not a random draw from all C4 chunks.",
            "- `pair_mid2` was measured as only modestly faster than full scoring in the official run, so the practical value of the cascade depends on candidate fraction and hardware utilization.",
            "- Pairwise tasks saturate once `m >= 2`, because each balanced pair has only twice as many rows as positives.",
            "",
            "## Conclusion",
            "",
            decision_text,
            "",
            "## Reproducibility",
            "",
            "Inputs:",
            "",
            "```text",
            "data/pair_mid2_cascade_full_rerank/results/score-pool-robustness-official-500k/scores_full.parquet",
            "data/pair_mid2_cascade_full_rerank/results/score-pool-robustness-official-500k/scores_pair_mid2.parquet",
            "```",
            "",
            "Outputs:",
            "",
            "```text",
            f"{output_dir / 'cascade_pairwise_metrics.csv'}",
            f"{output_dir / 'cascade_full_pool_metrics.csv'}",
            f"{output_dir / 'cascade_runtime_estimates.csv'}",
            f"{report_dir / 'report.md'}",
            f"{report_dir / 'report.html'}",
            "```",
            "",
        ]
    )
    report_md = report_dir / "report.md"
    report_md.write_text("\n".join(lines), encoding="utf-8")
    write_html_report(report_md, report_dir / "report.html")


def write_html_report(markdown_path: Path, html_path: Path) -> None:
    body_lines: list[str] = []
    in_code = False
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    idx = 0

    def split_table_row(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    def is_separator_row(cells: list[str]) -> bool:
        return all(cell and set(cell) <= {"-"} for cell in cells)

    while idx < len(lines):
        line = lines[idx]
        if line.startswith("```"):
            if in_code:
                body_lines.append("</code></pre>")
            else:
                body_lines.append("<pre><code>")
            in_code = not in_code
            idx += 1
            continue
        if in_code:
            body_lines.append(html.escape(line))
            idx += 1
            continue
        if line.startswith("# "):
            body_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("| "):
            table_lines = []
            while idx < len(lines) and lines[idx].startswith("| "):
                table_lines.append(lines[idx])
                idx += 1
            parsed = [split_table_row(table_line) for table_line in table_lines]
            if len(parsed) >= 2 and is_separator_row(parsed[1]):
                headers = parsed[0]
                rows = parsed[2:]
                body_lines.append("<table>")
                body_lines.append("<thead><tr>")
                body_lines.extend(f"<th>{html.escape(header)}</th>" for header in headers)
                body_lines.append("</tr></thead>")
                body_lines.append("<tbody>")
                for row in rows:
                    body_lines.append("<tr>")
                    body_lines.extend(f"<td>{html.escape(cell)}</td>" for cell in row)
                    body_lines.append("</tr>")
                body_lines.append("</tbody></table>")
            else:
                body_lines.extend(f"<p>{html.escape(table_line)}</p>" for table_line in table_lines)
            continue
        elif line.startswith("- "):
            items = []
            while idx < len(lines) and lines[idx].startswith("- "):
                items.append(lines[idx][2:])
                idx += 1
            body_lines.append("<ul>")
            body_lines.extend(f"<li>{html.escape(item)}</li>" for item in items)
            body_lines.append("</ul>")
            continue
        elif line.startswith("![") and "](" in line and line.endswith(")"):
            alt = line[2 : line.index("]")]
            src = line[line.index("(") + 1 : -1]
            body_lines.append(f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>')
        elif line.strip() == "":
            body_lines.append("")
        else:
            body_lines.append(f"<p>{html.escape(line)}</p>")
        idx += 1
    html_text = "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8">',
            "<title>Pair-Mid2 Cascade With Full-Model Reranking</title>",
            "<style>",
            "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 980px; margin: 40px auto; line-height: 1.55; color: #1f2933; padding: 0 20px; }",
            "pre { background: #f6f8fa; padding: 12px; overflow-x: auto; }",
            "code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }",
            "table { border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 0.92em; }",
            "th, td { border: 1px solid #d7dde5; padding: 7px 9px; text-align: left; vertical-align: top; }",
            "th { background: #eef2f7; }",
            "img { max-width: 100%; border: 1px solid #ddd; }",
            "figure { margin: 24px 0; }",
            "figcaption { color: #555; font-size: 0.9em; }",
            "p { margin: 0.6em 0; }",
            "</style>",
            "</head>",
            "<body>",
            *body_lines,
            "</body>",
            "</html>",
        ]
    )
    html_path.write_text(html_text, encoding="utf-8")


def load_metrics_from_csv(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RuntimeStats, RuntimeStats]:
    pairwise = pd.read_csv(output_dir / "cascade_pairwise_metrics.csv")
    full_pool = pd.read_csv(output_dir / "cascade_full_pool_metrics.csv")
    runtime = pd.read_csv(output_dir / "cascade_runtime_estimates.csv")
    measured = runtime[runtime["scope"] == "measured"].set_index("label")
    if "full" not in measured.index or "pair_mid2" not in measured.index:
        raise ValueError(f"Missing measured full/pair_mid2 runtime rows in {output_dir}")

    def runtime_from_row(label: str) -> RuntimeStats:
        row = measured.loc[label]
        return RuntimeStats(
            variant=label,
            elapsed_seconds=float(row["elapsed_seconds"]),
            tokens_scored=int(row["tokens_scored"]),
            tokens_per_second=float(row["tokens_per_second"]),
            rows=int(row["rows"]),
        )

    return pairwise, full_pool, runtime, runtime_from_row("full"), runtime_from_row("pair_mid2")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pair_mid2 cascade with full-model reranking.")
    parser.add_argument("--config", default="configs/score_pool_robustness.yaml")
    parser.add_argument("--input-results-dir", default=None)
    parser.add_argument("--output-dir", default="results/pair-mid2-cascade")
    parser.add_argument("--report-dir", default="reports/pair-mid2-cascade")
    parser.add_argument("--multipliers", default=",".join(format_multiplier(x) for x in DEFAULT_MULTIPLIERS))
    parser.add_argument("--selection-rates", default="1/64,1/32,1/16,1/8")
    parser.add_argument("--from-metrics", action="store_true", help="Render plots/report from existing CSV outputs.")
    parser.add_argument("--no-render", action="store_true", help="Write CSV metrics only; skip plots and report.")
    args = parser.parse_args()

    config = load_config(args.config)
    input_results_dir = Path(args.input_results_dir) if args.input_results_dir else DEFAULT_LOCAL_INPUT
    if not input_results_dir.exists():
        input_results_dir = Path(config["paths"]["output_dir"])
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    figures_dir = report_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.from_metrics:
        pairwise, full_pool, runtime, full_runtime, pair_mid2_runtime = load_metrics_from_csv(output_dir)
    else:
        multipliers = parse_float_list(args.multipliers)
        selection_rates = parse_float_list(args.selection_rates)
        pairwise_tasks = config["metrics"].get("pairwise_tasks") or list(PAIRWISE_TASKS)

        scores, full_runtime, pair_mid2_runtime = load_joined_scores(input_results_dir)
        pairwise = compute_pairwise_cascade_metrics(
            scores,
            multipliers=multipliers,
            pairwise_tasks=pairwise_tasks,
            full_runtime=full_runtime,
            pair_mid2_runtime=pair_mid2_runtime,
        )
        full_pool = compute_full_pool_cascade_metrics(
            scores,
            multipliers=multipliers,
            selection_rates=selection_rates,
            full_runtime=full_runtime,
            pair_mid2_runtime=pair_mid2_runtime,
        )
        runtime = compute_runtime_estimates(
            full_runtime=full_runtime,
            pair_mid2_runtime=pair_mid2_runtime,
            full_pool_metrics=full_pool,
            pairwise_metrics=pairwise,
        )
        pairwise.to_csv(ensure_parent(output_dir / "cascade_pairwise_metrics.csv"), index=False)
        full_pool.to_csv(ensure_parent(output_dir / "cascade_full_pool_metrics.csv"), index=False)
        runtime.to_csv(ensure_parent(output_dir / "cascade_runtime_estimates.csv"), index=False)
    if not args.no_render:
        write_plots(pairwise, full_pool, figures_dir)
        write_report(
            output_dir=output_dir,
            report_dir=report_dir,
            figures_dir=figures_dir,
            pairwise=pairwise,
            full_pool=full_pool,
            runtime=runtime,
            full_runtime=full_runtime,
            pair_mid2_runtime=pair_mid2_runtime,
        )
    print(f"wrote {output_dir / 'cascade_pairwise_metrics.csv'}")
    print(f"wrote {output_dir / 'cascade_full_pool_metrics.csv'}")
    print(f"wrote {output_dir / 'cascade_runtime_estimates.csv'}")
    if not args.no_render:
        print(f"wrote {report_dir / 'report.md'}")
        print(f"wrote {report_dir / 'report.html'}")


if __name__ == "__main__":
    main()
