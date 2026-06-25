from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


POOL_FILES: Mapping[str, tuple[str, str]] = {
    "random_positive": ("Random positives", "random_positive_samples.npz"),
    "hard_positive": ("Hard positives", "hard_positive_samples.npz"),
    "random_negative": ("Random negatives", "random_negative_samples.npz"),
    "hard_negative": ("Hard negatives", "hard_negative_samples.npz"),
    "tail_negative": ("Tail negatives", "tail_negative_samples.npz"),
}

PAIRWISE_TASKS: Mapping[str, tuple[str, str]] = {
    "hp_vs_hn": ("hard_positive", "hard_negative"),
    "hp_vs_rn": ("hard_positive", "random_negative"),
    "hp_vs_tn": ("hard_positive", "tail_negative"),
    "rp_vs_hn": ("random_positive", "hard_negative"),
    "rp_vs_rn": ("random_positive", "random_negative"),
    "rp_vs_tn": ("random_positive", "tail_negative"),
}


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    cond_removed_layers: tuple[int, ...]
    marg_removed_layers: tuple[int, ...]
    family: str

    @property
    def cond_kept_layers(self) -> int:
        return 12 - len(self.cond_removed_layers)

    @property
    def marg_kept_layers(self) -> int:
        return 12 - len(self.marg_removed_layers)


def _layer_set(name: str, total_layers: int) -> tuple[int, ...]:
    if name == "none":
        return ()
    prefix = name[:3]
    try:
        count = int(name[3:])
    except ValueError as exc:
        raise ValueError(f"Unsupported layer set '{name}'") from exc
    if count < 0 or count > total_layers:
        raise ValueError(f"Cannot remove {count} layers from {total_layers}-layer model")
    if prefix == "top":
        return tuple(range(total_layers - count, total_layers))
    if prefix == "bot":
        return tuple(range(count))
    if prefix == "mid":
        start = (total_layers - count) // 2
        return tuple(range(start, start + count))
    raise ValueError(f"Unsupported layer set '{name}'")


def default_variant_specs(total_layers: int = 12) -> dict[str, VariantSpec]:
    specs: dict[str, VariantSpec] = {
        "full": VariantSpec("full", (), (), "baseline"),
        "full_rescore": VariantSpec("full_rescore", (), (), "baseline"),
    }

    for region in ("top", "bot"):
        for count in (1, 2, 4, 6):
            layer_name = f"{region}{count}"
            removed = _layer_set(layer_name, total_layers)
            variant_id = f"pair_{layer_name}"
            specs[variant_id] = VariantSpec(variant_id, removed, removed, "paired")
    for layer_name in ("mid2", "mid4"):
        removed = _layer_set(layer_name, total_layers)
        variant_id = f"pair_{layer_name}"
        specs[variant_id] = VariantSpec(variant_id, removed, removed, "paired")

    for count in (1, 2, 4, 6):
        top = _layer_set(f"top{count}", total_layers)
        bot = _layer_set(f"bot{count}", total_layers)
        variant_id = f"cond_top{count}_marg_bot{count}"
        specs[variant_id] = VariantSpec(variant_id, top, bot, "cond_top_marg_bot")
        variant_id = f"cond_bot{count}_marg_top{count}"
        specs[variant_id] = VariantSpec(variant_id, bot, top, "cond_bot_marg_top")
        variant_id = f"cond_top{count}_only"
        specs[variant_id] = VariantSpec(variant_id, top, (), "cond_only")
        variant_id = f"marg_top{count}_only"
        specs[variant_id] = VariantSpec(variant_id, (), top, "marg_only")

    return specs


def variant_spec_from_config(
    config: Mapping[str, Any],
    variant_id: str,
    *,
    total_layers: int,
) -> VariantSpec:
    for raw in config.get("variants", []):
        if raw.get("id") != variant_id:
            continue
        cond_raw = raw.get("cond_remove", [])
        marg_raw = raw.get("marg_remove", [])
        family = str(raw.get("family", "custom"))
        return VariantSpec(
            variant_id=variant_id,
            cond_removed_layers=tuple(int(i) for i in cond_raw),
            marg_removed_layers=tuple(int(i) for i in marg_raw),
            family=family,
        )
    specs = default_variant_specs(total_layers)
    if variant_id not in specs:
        raise KeyError(f"Unknown score-pool robustness variant '{variant_id}'")
    return specs[variant_id]


def load_pool_samples(pool_analysis_dir: str | Path) -> pd.DataFrame:
    pool_dir = Path(pool_analysis_dir)
    frames: list[pd.DataFrame] = []
    for pool_name, (pool_label, filename) in POOL_FILES.items():
        path = pool_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing score-pool sample file: {path}")
        data = np.load(path)
        required = {
            "row_position",
            "c4_index",
            "prior_score",
            "conditional_books_score",
            "color_score",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path} missing arrays: {missing}")
        n_rows = len(data["row_position"])
        frame = pd.DataFrame(
            {
                "sample_idx": np.arange(n_rows, dtype=np.int64),
                "pool_name": pool_name,
                "pool_label": pool_label,
                "row_position": data["row_position"].astype(np.int64),
                "c4_index": data["c4_index"].astype(np.int64),
                "full_prior_score": data["prior_score"].astype(np.float32),
                "full_conditional_books_score": data["conditional_books_score"].astype(np.float32),
                "full_color_score": data["color_score"].astype(np.float32),
                "label_available_for_pair_tasks": True,
            }
        )
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.insert(0, "seq_idx", np.arange(len(combined), dtype=np.int64))
    return combined


def write_pool_metadata(pool_analysis_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    frame = load_pool_samples(pool_analysis_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return frame


def _safe_corr(func: Any, x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    result = func(x, y)
    if hasattr(result, "statistic"):
        return float(result.statistic)
    if isinstance(result, tuple):
        return float(result[0])
    return float(result)


def roc_auc_from_scores(labels: np.ndarray, decision_scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores_array = np.asarray(decision_scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(scores_array, method="average")
    pos_rank_sum = float(ranks[labels == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision_from_scores(labels: np.ndarray, decision_scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores_array = np.asarray(decision_scores, dtype=np.float64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores_array, kind="mergesort")
    sorted_labels = labels[order]
    true_positive_cumsum = np.cumsum(sorted_labels)
    positive_positions = np.flatnonzero(sorted_labels == 1)
    if len(positive_positions) == 0:
        return float("nan")
    precision_at_positive = true_positive_cumsum[positive_positions] / (positive_positions + 1)
    return float(precision_at_positive.mean())


def binary_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int((labels & predicted).sum())
    fp = int((~labels & predicted).sum())
    fn = int((labels & ~predicted).sum())
    tn = int((~labels & ~predicted).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_rate": float(predicted.mean()),
    }


def balanced_rate_threshold(color_scores: np.ndarray, n_positive: int) -> float:
    if n_positive < 1 or n_positive > len(color_scores):
        raise ValueError(f"n_positive={n_positive} out of range for {len(color_scores)} scores")
    return float(np.partition(np.asarray(color_scores, dtype=np.float64), n_positive - 1)[n_positive - 1])


def _require_score_columns(frame: pd.DataFrame) -> None:
    required = {
        "pool_name",
        "full_prior_score",
        "full_conditional_books_score",
        "full_color_score",
        "ablated_prior_score",
        "ablated_conditional_books_score",
        "ablated_color_score",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Score frame missing columns: {missing}")


def _first_jsonable(frame: pd.DataFrame, column: str, default: str = "") -> str:
    if column not in frame.columns or frame.empty:
        return default
    return str(frame[column].iloc[0])


def compute_pairwise_metrics(
    scores: pd.DataFrame,
    *,
    variant_id: str,
    cutoff_tau64: float,
    pairwise_tasks: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_score_columns(scores)
    task_ids = list(pairwise_tasks or PAIRWISE_TASKS.keys())
    rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    variant_family = _first_jsonable(scores, "variant_family")
    cond_removed = _first_jsonable(scores, "cond_removed_layers")
    marg_removed = _first_jsonable(scores, "marg_removed_layers")

    for task_id in task_ids:
        if task_id not in PAIRWISE_TASKS:
            raise KeyError(f"Unknown pairwise task '{task_id}'")
        positive_pool, negative_pool = PAIRWISE_TASKS[task_id]
        pair = scores[scores["pool_name"].isin([positive_pool, negative_pool])].copy()
        if pair.empty:
            raise ValueError(f"No rows available for task {task_id}")
        labels = (pair["pool_name"] == positive_pool).to_numpy(dtype=np.int8)
        color = pair["ablated_color_score"].to_numpy(dtype=np.float64)
        decision_score = -color

        original_pred = color <= cutoff_tau64
        original_metrics = binary_metrics(labels, original_pred)

        threshold_balanced = balanced_rate_threshold(color, int(labels.sum()))
        balanced_pred = color <= threshold_balanced
        balanced_metrics = binary_metrics(labels, balanced_pred)

        color_shift = pair["ablated_color_score"] - pair["full_color_score"]
        cond_shift = pair["ablated_conditional_books_score"] - pair["full_conditional_books_score"]
        prior_shift = pair["ablated_prior_score"] - pair["full_prior_score"]
        rows.append(
            {
                "variant": variant_id,
                "variant_family": variant_family,
                "cond_removed_layers": cond_removed,
                "marg_removed_layers": marg_removed,
                "pairwise_task": task_id,
                "positive_pool": positive_pool,
                "negative_pool": negative_pool,
                "n_positive": int(labels.sum()),
                "n_negative": int(len(labels) - labels.sum()),
                "roc_auc": roc_auc_from_scores(labels, decision_score),
                "average_precision": average_precision_from_scores(labels, decision_score),
                "precision_at_original_cutoff": original_metrics["precision"],
                "recall_at_original_cutoff": original_metrics["recall"],
                "f1_at_original_cutoff": original_metrics["f1"],
                "predicted_positive_rate_at_original_cutoff": original_metrics[
                    "predicted_positive_rate"
                ],
                "threshold_original_cutoff": cutoff_tau64,
                "precision_at_balanced_rate": balanced_metrics["precision"],
                "recall_at_balanced_rate": balanced_metrics["recall"],
                "f1_at_balanced_rate": balanced_metrics["f1"],
                "predicted_positive_rate_at_balanced_rate": balanced_metrics[
                    "predicted_positive_rate"
                ],
                "threshold_balanced_rate": threshold_balanced,
                "pearson_color": _safe_corr(
                    stats.pearsonr,
                    pair["full_color_score"].to_numpy(),
                    pair["ablated_color_score"].to_numpy(),
                ),
                "spearman_color": _safe_corr(
                    stats.spearmanr,
                    pair["full_color_score"].to_numpy(),
                    pair["ablated_color_score"].to_numpy(),
                ),
                "pearson_conditional": _safe_corr(
                    stats.pearsonr,
                    pair["full_conditional_books_score"].to_numpy(),
                    pair["ablated_conditional_books_score"].to_numpy(),
                ),
                "spearman_conditional": _safe_corr(
                    stats.spearmanr,
                    pair["full_conditional_books_score"].to_numpy(),
                    pair["ablated_conditional_books_score"].to_numpy(),
                ),
                "pearson_prior": _safe_corr(
                    stats.pearsonr,
                    pair["full_prior_score"].to_numpy(),
                    pair["ablated_prior_score"].to_numpy(),
                ),
                "spearman_prior": _safe_corr(
                    stats.spearmanr,
                    pair["full_prior_score"].to_numpy(),
                    pair["ablated_prior_score"].to_numpy(),
                ),
                "mean_color_shift": float(color_shift.mean()),
                "std_color_shift": float(color_shift.std(ddof=1)),
                "mean_conditional_shift": float(cond_shift.mean()),
                "std_conditional_shift": float(cond_shift.std(ddof=1)),
                "mean_prior_shift": float(prior_shift.mean()),
                "std_prior_shift": float(prior_shift.std(ddof=1)),
            }
        )
        for score_name, values in (
            ("color", color_shift),
            ("conditional_books", cond_shift),
            ("prior", prior_shift),
        ):
            shift_rows.append(
                {
                    "variant": variant_id,
                    "variant_family": variant_family,
                    "pairwise_task": task_id,
                    "score_name": score_name,
                    "mean_shift": float(values.mean()),
                    "std_shift": float(values.std(ddof=1)),
                    "min_shift": float(values.min()),
                    "p05_shift": float(values.quantile(0.05)),
                    "median_shift": float(values.quantile(0.50)),
                    "p95_shift": float(values.quantile(0.95)),
                    "max_shift": float(values.max()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(shift_rows)


def nominal_dual_forward_ratio(spec: VariantSpec, *, total_layers: int) -> float:
    cond_ratio = (total_layers - len(spec.cond_removed_layers)) / total_layers
    marg_ratio = (total_layers - len(spec.marg_removed_layers)) / total_layers
    return (cond_ratio + marg_ratio) / 2.0


def nominal_dual_speedup(spec: VariantSpec, *, total_layers: int) -> float:
    ratio = nominal_dual_forward_ratio(spec, total_layers=total_layers)
    return 1.0 / ratio if ratio > 0 else math.inf
