from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


SCORE_COLUMNS = ("seq_idx", "nll_cond", "nll_marg", "color")


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


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _align_scores(
    full_scores: pd.DataFrame,
    variant_scores: pd.DataFrame,
    pool_meta: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(full_scores, SCORE_COLUMNS, "full_scores")
    _require_columns(variant_scores, SCORE_COLUMNS, "variant_scores")
    _require_columns(pool_meta, ("seq_idx", "enriched"), "pool_meta")

    full = full_scores.loc[:, SCORE_COLUMNS].rename(
        columns={
            "nll_cond": "nll_cond_full",
            "nll_marg": "nll_marg_full",
            "color": "color_full",
        }
    )
    variant = variant_scores.loc[:, SCORE_COLUMNS].rename(
        columns={
            "nll_cond": "nll_cond_variant",
            "nll_marg": "nll_marg_variant",
            "color": "color_variant",
        }
    )
    merged = pool_meta.loc[:, ["seq_idx", "enriched"]].merge(full, on="seq_idx").merge(variant, on="seq_idx")
    if len(merged) != len(pool_meta):
        raise ValueError("Score frames do not cover the full pool metadata")
    merged["enriched"] = merged["enriched"].astype(bool)
    return merged


def _bottom_ids(frame: pd.DataFrame, column: str, k: int) -> set[int]:
    return set(frame.nsmallest(k, column)["seq_idx"].astype(int).tolist())


def _threshold(frame: pd.DataFrame, column: str, k: int) -> float:
    values = frame[column].to_numpy()
    if k < 1 or k > len(values):
        raise ValueError(f"k={k} out of range for {len(values)} values")
    return float(np.partition(values, k - 1)[k - 1])


def _bootstrap_overlap_ci(
    full_selected: np.ndarray,
    variant_selected: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> dict[str, float]:
    if reps <= 0:
        return {
            "recall_ci_low": float("nan"),
            "recall_ci_high": float("nan"),
            "jaccard_ci_low": float("nan"),
            "jaccard_ci_high": float("nan"),
        }

    rng = np.random.default_rng(seed)
    n = len(full_selected)
    recalls = np.empty(reps, dtype=np.float64)
    jaccards = np.empty(reps, dtype=np.float64)
    both_all = full_selected & variant_selected
    union_all = full_selected | variant_selected
    for i in range(reps):
        sample = rng.integers(0, n, size=n)
        full_sum = full_selected[sample].sum()
        union_sum = union_all[sample].sum()
        both_sum = both_all[sample].sum()
        recalls[i] = both_sum / full_sum if full_sum else float("nan")
        jaccards[i] = both_sum / union_sum if union_sum else float("nan")

    return {
        "recall_ci_low": float(np.nanpercentile(recalls, 2.5)),
        "recall_ci_high": float(np.nanpercentile(recalls, 97.5)),
        "jaccard_ci_low": float(np.nanpercentile(jaccards, 2.5)),
        "jaccard_ci_high": float(np.nanpercentile(jaccards, 97.5)),
    }


def _tail_local_spearman(pure: pd.DataFrame, tail_rate: float) -> float:
    k = max(1, int(math.floor(len(pure) * tail_rate)))
    full_ids = _bottom_ids(pure, "color_full", k)
    variant_ids = _bottom_ids(pure, "color_variant", k)
    tail = pure[pure["seq_idx"].isin(full_ids | variant_ids)]
    return _safe_corr(
        stats.spearmanr,
        tail["color_full"].to_numpy(),
        tail["color_variant"].to_numpy(),
    )


def compute_variant_metrics(
    *,
    full_scores: pd.DataFrame,
    variant_scores: pd.DataFrame,
    pool_meta: pd.DataFrame,
    variant_id: str,
    removed_layers: Sequence[int],
    total_layers: int,
    selection_rates: Sequence[float],
    bootstrap_reps: int,
    bootstrap_seed: int,
    tail_rate_for_local_spearman: float,
    tokens_per_second: float | None = None,
) -> pd.DataFrame:
    merged = _align_scores(full_scores, variant_scores, pool_meta)
    pure = merged[~merged["enriched"]].copy()
    enriched = merged[merged["enriched"]].copy()
    if pure.empty:
        raise ValueError("No non-enriched pool rows available for metrics")

    color_shift = pure["color_variant"] - pure["color_full"]
    base = {
        "variant": variant_id,
        "removed_layers": ",".join(str(i) for i in removed_layers),
        "layers_dropped": len(removed_layers),
        "total_layers": total_layers,
        "fraction_layers_dropped": len(removed_layers) / total_layers,
        "nominal_forward_flop_ratio": (total_layers - len(removed_layers)) / total_layers,
        "nominal_forward_speedup": total_layers / (total_layers - len(removed_layers))
        if len(removed_layers) < total_layers
        else float("inf"),
        "tokens_per_second": float(tokens_per_second) if tokens_per_second is not None else float("nan"),
        "n_pure": len(pure),
        "n_enriched": len(enriched),
        "spearman": _safe_corr(
            stats.spearmanr,
            pure["color_full"].to_numpy(),
            pure["color_variant"].to_numpy(),
        ),
        "kendall": _safe_corr(
            stats.kendalltau,
            pure["color_full"].to_numpy(),
            pure["color_variant"].to_numpy(),
        ),
        "pearson": _safe_corr(
            stats.pearsonr,
            pure["color_full"].to_numpy(),
            pure["color_variant"].to_numpy(),
        ),
        "tail_spearman": _tail_local_spearman(pure, tail_rate_for_local_spearman),
        "nll_cond_spearman": _safe_corr(
            stats.spearmanr,
            pure["nll_cond_full"].to_numpy(),
            pure["nll_cond_variant"].to_numpy(),
        ),
        "nll_marg_spearman": _safe_corr(
            stats.spearmanr,
            pure["nll_marg_full"].to_numpy(),
            pure["nll_marg_variant"].to_numpy(),
        ),
        "color_shift_mean": float(color_shift.mean()),
        "color_shift_std": float(color_shift.std(ddof=1)),
    }

    rows: list[dict[str, Any]] = []
    for rate_idx, rate in enumerate(selection_rates):
        k = max(1, int(math.floor(len(pure) * float(rate))))
        full_ids = _bottom_ids(pure, "color_full", k)
        variant_ids = _bottom_ids(pure, "color_variant", k)
        intersection = len(full_ids & variant_ids)
        union = len(full_ids | variant_ids)

        full_threshold = _threshold(pure, "color_full", k)
        variant_threshold = _threshold(pure, "color_variant", k)
        if enriched.empty:
            full_enriched_selected = float("nan")
            variant_enriched_selected = float("nan")
        else:
            full_enriched_selected = float((enriched["color_full"] <= full_threshold).mean())
            variant_enriched_selected = float((enriched["color_variant"] <= variant_threshold).mean())

        full_selected = pure["seq_idx"].isin(full_ids).to_numpy()
        variant_selected = pure["seq_idx"].isin(variant_ids).to_numpy()
        ci = _bootstrap_overlap_ci(
            full_selected,
            variant_selected,
            reps=bootstrap_reps,
            seed=bootstrap_seed + rate_idx,
        )

        rows.append(
            {
                **base,
                "selection_rate": float(rate),
                "k": k,
                "overlap_count": intersection,
                "recall_at_k": intersection / k,
                "jaccard_at_k": intersection / union if union else float("nan"),
                "full_threshold": full_threshold,
                "variant_threshold": variant_threshold,
                "full_enriched_selected_frac": full_enriched_selected,
                "variant_enriched_selected_frac": variant_enriched_selected,
                **ci,
            }
        )
    return pd.DataFrame(rows)


def compute_all_metrics(
    *,
    score_frames: Mapping[str, pd.DataFrame],
    pool_meta: pd.DataFrame,
    variants_removed_layers: Mapping[str, Sequence[int]],
    full_variant: str,
    total_layers: int,
    selection_rates: Sequence[float],
    bootstrap_reps: int,
    bootstrap_seed: int,
    tail_rate_for_local_spearman: float,
    variant_tokens_per_second: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    if full_variant not in score_frames:
        raise KeyError(f"Missing full variant '{full_variant}'")
    full_scores = score_frames[full_variant]
    speed = variant_tokens_per_second or {}
    frames = []
    for variant_id, variant_scores in score_frames.items():
        frames.append(
            compute_variant_metrics(
                full_scores=full_scores,
                variant_scores=variant_scores,
                pool_meta=pool_meta,
                variant_id=variant_id,
                removed_layers=variants_removed_layers.get(variant_id, ()),
                total_layers=total_layers,
                selection_rates=selection_rates,
                bootstrap_reps=bootstrap_reps,
                bootstrap_seed=bootstrap_seed,
                tail_rate_for_local_spearman=tail_rate_for_local_spearman,
                tokens_per_second=speed.get(variant_id),
            )
        )
    return pd.concat(frames, ignore_index=True)

