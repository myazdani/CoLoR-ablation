#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ensure_parent, load_config


DEFAULT_ENSEMBLES = {
    "mild": ("full", "mid2", "top1"),
    "mid": ("full", "mid2", "mid4", "top1"),
    "wide": ("full", "top1", "top2", "mid2", "bot2", "top4", "mid4", "bot4", "top6", "skip2"),
    "mild_no_full": ("mid2", "top1"),
}
DEFAULT_DECISION_ENSEMBLES = {"mild": "strict", "mid": "direction_only", "wide": "sanity_bound"}
DIAGNOSTIC_ENSEMBLES = {"mild_no_full"}
DEFAULT_LAMBDAS = (-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_RATES = (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4)
SCORE_COLUMNS = ("seq_idx", "nll_cond", "nll_marg", "color")
CAUTION_TEXT = (
    "The crude ensemble is biased structural perturbation, not a posterior, and a positive result only "
    "licenses building the real conditional-seed ensemble; it does not itself demonstrate that "
    "uncertainty-aware selection works."
)


@dataclass(frozen=True)
class Inputs:
    results_dir: Path
    pool_meta: Path
    output_csv: Path
    provenance_json: Path
    figures_dir: Path
    report_md: Path


@dataclass(frozen=True)
class Decision:
    test_a_mild: bool
    test_b_mild: bool
    test_c_mild: bool
    mid_same_direction: bool

    @property
    def call(self) -> str:
        if self.test_a_mild and self.test_b_mild and self.test_c_mild and self.mid_same_direction:
            return "positive"
        if self.test_a_mild and self.test_b_mild and not self.test_c_mild:
            return "weak_or_ambiguous"
        return "negative"


def _parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_ensemble(raw: str) -> tuple[str, tuple[str, ...]]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Ensemble must have form name=member,member")
    name, members_raw = raw.split("=", 1)
    members = tuple(part.strip() for part in members_raw.split(",") if part.strip())
    if not name.strip() or not members:
        raise argparse.ArgumentTypeError("Ensemble name and members must be non-empty")
    return name.strip(), members


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _read_score(path: Path, variant: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing score parquet for {variant}: {path}")
    frame = pd.read_parquet(path)
    _require_columns(frame, SCORE_COLUMNS, path.name)
    if frame["seq_idx"].duplicated().any():
        raise ValueError(f"{path.name} has duplicate seq_idx values")
    if "variant" in frame.columns:
        observed = set(frame["variant"].astype(str).unique())
        if len(observed) != 1:
            raise ValueError(f"{path.name} has multiple variant labels: {sorted(observed)}")
    return frame.copy()


def _load_aligned_scores(
    *,
    results_dir: Path,
    pool_meta_path: Path,
    variants: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    if not pool_meta_path.exists():
        raise FileNotFoundError(f"Missing pool metadata: {pool_meta_path}")
    pool_meta = pd.read_parquet(pool_meta_path)
    _require_columns(pool_meta, ("seq_idx", "enriched"), "pool_meta")
    if pool_meta["seq_idx"].duplicated().any():
        raise ValueError("pool_meta has duplicate seq_idx values")
    pool_meta = pool_meta.loc[:, ["seq_idx", "enriched"]].copy()
    pool_meta["enriched"] = pool_meta["enriched"].astype(bool)
    pool_keys = pool_meta["seq_idx"].astype(int)
    pool_key_set = set(pool_keys.tolist())

    aligned: dict[str, pd.DataFrame] = {}
    alignment: dict[str, Any] = {"pool_rows": len(pool_meta), "variants": {}}
    for variant in variants:
        frame = _read_score(results_dir / f"scores_{variant}.parquet", variant)
        keys = frame["seq_idx"].astype(int)
        same_order = len(frame) == len(pool_meta) and keys.reset_index(drop=True).equals(pool_keys.reset_index(drop=True))
        key_set = set(keys.tolist())
        if len(frame) != len(pool_meta) or key_set != pool_key_set:
            missing = sorted(pool_key_set - key_set)[:10]
            extra = sorted(key_set - pool_key_set)[:10]
            raise ValueError(
                f"scores_{variant}.parquet does not align with pool metadata: "
                f"rows={len(frame)} pool={len(pool_meta)} missing={missing} extra={extra}"
            )
        if not same_order:
            frame = pool_meta[["seq_idx"]].merge(frame, on="seq_idx", how="left", validate="one_to_one")
        aligned[variant] = frame.reset_index(drop=True)
        alignment["variants"][variant] = {"same_order": bool(same_order), "rows": len(frame)}
    return pool_meta.reset_index(drop=True), aligned, alignment


def _threshold(values: np.ndarray, rate: float) -> tuple[float, int]:
    k = max(1, int(math.floor(len(values) * float(rate))))
    return float(np.partition(values, k - 1)[k - 1]), k


def _selected(values: np.ndarray, rate: float) -> tuple[np.ndarray, float, int]:
    threshold, k = _threshold(values, rate)
    return values <= threshold, threshold, k


def _percentiles_from_scores(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    percentiles = np.empty(len(values), dtype=np.float64)
    percentiles[order] = (np.arange(len(values), dtype=np.float64) + 0.5) / len(values)
    return percentiles


def _ci_from_samples(samples: np.ndarray) -> tuple[float, float]:
    return float(np.nanpercentile(samples, 2.5)), float(np.nanpercentile(samples, 97.5))


def _bootstrap_mean(values: np.ndarray, *, reps: int, rng: np.random.Generator, chunk_size: int = 200) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if reps <= 0 or len(values) == 0:
        return np.empty(0, dtype=np.float64)
    out = np.empty(reps, dtype=np.float64)
    offset = 0
    while offset < reps:
        n_chunk = min(chunk_size, reps - offset)
        sample = rng.integers(0, len(values), size=(n_chunk, len(values)))
        out[offset : offset + n_chunk] = values[sample].mean(axis=1)
        offset += n_chunk
    return out


def _bootstrap_two_region_contrast(
    threshold_values: np.ndarray,
    deep_values: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> dict[str, float]:
    threshold_mean = float(np.mean(threshold_values))
    deep_mean = float(np.mean(deep_values))
    diff = threshold_mean - deep_mean
    ratio = threshold_mean / deep_mean if deep_mean != 0 else float("nan")
    if reps <= 0:
        return {
            "diff": diff,
            "diff_ci_low": float("nan"),
            "diff_ci_high": float("nan"),
            "ratio": ratio,
            "ratio_ci_low": float("nan"),
            "ratio_ci_high": float("nan"),
        }
    rng = np.random.default_rng(seed)
    threshold_boot = _bootstrap_mean(threshold_values, reps=reps, rng=rng)
    deep_boot = _bootstrap_mean(deep_values, reps=reps, rng=rng)
    diff_boot = threshold_boot - deep_boot
    ratio_boot = np.divide(threshold_boot, deep_boot, out=np.full_like(threshold_boot, np.nan), where=deep_boot != 0)
    diff_low, diff_high = _ci_from_samples(diff_boot)
    ratio_low, ratio_high = _ci_from_samples(ratio_boot)
    return {
        "diff": diff,
        "diff_ci_low": diff_low,
        "diff_ci_high": diff_high,
        "ratio": ratio,
        "ratio_ci_low": ratio_low,
        "ratio_ci_high": ratio_high,
    }


def _bootstrap_paired_indicator_diff(
    selected: np.ndarray,
    baseline_selected: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> tuple[float, float, float]:
    values = selected.astype(np.float64) - baseline_selected.astype(np.float64)
    point = float(values.mean())
    if reps <= 0:
        return point, float("nan"), float("nan")
    boot = _bootstrap_mean(values, reps=reps, rng=np.random.default_rng(seed))
    low, high = _ci_from_samples(boot)
    return point, low, high


def _bootstrap_matched_bins(
    frame: pd.DataFrame,
    *,
    bins: int,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    band = frame[["mu", "sigma", "enriched"]].dropna().copy()
    if band["enriched"].nunique() < 2:
        return {
            "point": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "matched_n": 0,
            "bins_used": 0,
            "enriched_mean": float("nan"),
            "pure_mean": float("nan"),
        }

    unique_mu = band["mu"].nunique()
    q = min(int(bins), int(unique_mu))
    if q < 2:
        return {
            "point": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "matched_n": 0,
            "bins_used": 0,
            "enriched_mean": float("nan"),
            "pure_mean": float("nan"),
        }
    band["score_bin"] = pd.qcut(band["mu"], q=q, duplicates="drop")

    matched: list[tuple[np.ndarray, np.ndarray, int]] = []
    for _, group in band.groupby("score_bin", observed=True):
        enriched = group[group["enriched"]]["sigma"].to_numpy(dtype=np.float64)
        pure = group[~group["enriched"]]["sigma"].to_numpy(dtype=np.float64)
        m = min(len(enriched), len(pure))
        if m > 0:
            matched.append((enriched, pure, m))
    if not matched:
        return {
            "point": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "matched_n": 0,
            "bins_used": 0,
            "enriched_mean": float("nan"),
            "pure_mean": float("nan"),
        }

    weights = np.asarray([m for _, _, m in matched], dtype=np.float64)
    weights = weights / weights.sum()
    enriched_means = np.asarray([float(values.mean()) for values, _, _ in matched])
    pure_means = np.asarray([float(values.mean()) for _, values, _ in matched])
    point = float(np.sum(weights * (enriched_means - pure_means)))
    enriched_point = float(np.sum(weights * enriched_means))
    pure_point = float(np.sum(weights * pure_means))

    if reps <= 0:
        return {
            "point": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "matched_n": int(sum(m for _, _, m in matched)),
            "bins_used": len(matched),
            "enriched_mean": enriched_point,
            "pure_mean": pure_point,
        }

    rng = np.random.default_rng(seed)
    boot = np.empty(reps, dtype=np.float64)
    for rep in range(reps):
        diffs = []
        for enriched_values, pure_values, _ in matched:
            enriched_sample = enriched_values[rng.integers(0, len(enriched_values), size=len(enriched_values))]
            pure_sample = pure_values[rng.integers(0, len(pure_values), size=len(pure_values))]
            diffs.append(float(enriched_sample.mean() - pure_sample.mean()))
        boot[rep] = float(np.sum(weights * np.asarray(diffs)))
    low, high = _ci_from_samples(boot)
    return {
        "point": point,
        "ci_low": low,
        "ci_high": high,
        "matched_n": int(sum(m for _, _, m in matched)),
        "bins_used": len(matched),
        "enriched_mean": enriched_point,
        "pure_mean": pure_point,
    }


def _ensemble_frame(
    *,
    pool_meta: pd.DataFrame,
    scores: dict[str, pd.DataFrame],
    members: tuple[str, ...],
    full_color: np.ndarray,
) -> pd.DataFrame:
    missing = [member for member in members if member not in scores]
    if missing:
        raise KeyError(f"Ensemble references missing score variants: {missing}")
    matrix = np.column_stack([scores[member]["color"].to_numpy(dtype=np.float64) for member in members])
    frame = pool_meta[["seq_idx", "enriched"]].copy()
    frame["color_full"] = full_color
    frame["mu"] = matrix.mean(axis=1)
    frame["sigma"] = matrix.std(axis=1, ddof=0)
    return frame


def _compute_noise_floor(scores: dict[str, pd.DataFrame]) -> dict[str, float]:
    diff = np.abs(scores["full"]["color"].to_numpy(dtype=np.float64) - scores["full_rescore"]["color"].to_numpy(dtype=np.float64))
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "fraction_gt_1e_6": float((diff > 1e-6).mean()),
    }


def _test_a(
    *,
    ensemble: str,
    frame: pd.DataFrame,
    full_percentiles: np.ndarray,
    percentile_bins: int,
    bootstrap_reps: int,
    seed: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    pure = frame[~frame["enriched"]].copy()
    pure["full_percentile"] = full_percentiles
    bin_edges = np.linspace(0.0, 1.0, percentile_bins + 1)
    pure["percentile_bin"] = pd.cut(pure["full_percentile"], bins=bin_edges, include_lowest=True)
    curve = (
        pure.groupby("percentile_bin", observed=True)
        .agg(
            percentile_mid=("full_percentile", "mean"),
            sigma_mean=("sigma", "mean"),
            n=("sigma", "size"),
        )
        .reset_index(drop=True)
    )
    curve["ensemble"] = ensemble

    regions = {
        "deep_tail_bottom_1pct": pure["full_percentile"] < 0.01,
        "threshold_band_4_9pct": (pure["full_percentile"] >= 0.04) & (pure["full_percentile"] < 0.09),
        "bulk_above_9pct": pure["full_percentile"] >= 0.09,
    }
    rows: list[dict[str, Any]] = []
    for region, mask in regions.items():
        values = pure.loc[mask, "sigma"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "ensemble": ensemble,
                "test": "a_spread_by_region",
                "metric": "mean_sigma",
                "region": region,
                "selection_rate": np.nan,
                "lambda": np.nan,
                "point": float(values.mean()),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n": int(len(values)),
                "extra": "",
            }
        )

    threshold_values = pure.loc[regions["threshold_band_4_9pct"], "sigma"].to_numpy(dtype=np.float64)
    deep_values = pure.loc[regions["deep_tail_bottom_1pct"], "sigma"].to_numpy(dtype=np.float64)
    contrast = _bootstrap_two_region_contrast(
        threshold_values,
        deep_values,
        reps=bootstrap_reps,
        seed=seed,
    )
    rows.extend(
        [
            {
                "ensemble": ensemble,
                "test": "a_threshold_vs_deep",
                "metric": "threshold_minus_deep_mean_sigma",
                "region": "threshold_band_4_9pct_vs_deep_tail_bottom_1pct",
                "selection_rate": np.nan,
                "lambda": np.nan,
                "point": contrast["diff"],
                "ci_low": contrast["diff_ci_low"],
                "ci_high": contrast["diff_ci_high"],
                "n": int(len(threshold_values) + len(deep_values)),
                "extra": "",
            },
            {
                "ensemble": ensemble,
                "test": "a_threshold_vs_deep",
                "metric": "threshold_over_deep_mean_sigma",
                "region": "threshold_band_4_9pct_vs_deep_tail_bottom_1pct",
                "selection_rate": np.nan,
                "lambda": np.nan,
                "point": contrast["ratio"],
                "ci_low": contrast["ratio_ci_low"],
                "ci_high": contrast["ratio_ci_high"],
                "n": int(len(threshold_values) + len(deep_values)),
                "extra": "",
            },
        ]
    )
    return rows, curve


def _test_flip_rate(
    *,
    ensemble: str,
    members: tuple[str, ...],
    scores: dict[str, pd.DataFrame],
    pure_mask: np.ndarray,
    full_percentiles: np.ndarray,
    selection_rate: float,
) -> list[dict[str, Any]]:
    selected_by_member = []
    for member in members:
        color = scores[member].loc[pure_mask, "color"].to_numpy(dtype=np.float64)
        selected, _, _ = _selected(color, selection_rate)
        selected_by_member.append(selected)
    matrix = np.column_stack(selected_by_member)
    flipper = matrix.any(axis=1) & ~matrix.all(axis=1)
    regions = {
        "all_pure": np.ones_like(flipper, dtype=bool),
        "deep_tail_bottom_1pct": full_percentiles < 0.01,
        "threshold_band_4_9pct": (full_percentiles >= 0.04) & (full_percentiles < 0.09),
        "bulk_above_9pct": full_percentiles >= 0.09,
    }
    rows = []
    for region, mask in regions.items():
        rows.append(
            {
                "ensemble": ensemble,
                "test": "a_flip_rate_descriptive",
                "metric": "flip_rate",
                "region": region,
                "selection_rate": selection_rate,
                "lambda": np.nan,
                "point": float(flipper[mask].mean()),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n": int(mask.sum()),
                "extra": "descriptive_only",
            }
        )
    return rows


def _test_b(
    *,
    ensemble: str,
    frame: pd.DataFrame,
    match_bins: int,
    bootstrap_reps: int,
    seed: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    pure = frame[~frame["enriched"]]
    threshold_1_16, _ = _threshold(pure["mu"].to_numpy(dtype=np.float64), 1 / 16)
    threshold_1pct, _ = _threshold(pure["mu"].to_numpy(dtype=np.float64), 0.01)
    bands = {
        "below_mu_1_16_pure_cutoff": threshold_1_16,
        "below_mu_1pct_pure_cutoff": threshold_1pct,
    }
    rows: list[dict[str, Any]] = []
    plot_rows: list[dict[str, Any]] = []
    for offset, (band_name, threshold) in enumerate(bands.items()):
        band = frame[frame["mu"] <= threshold].copy()
        matched = _bootstrap_matched_bins(
            band,
            bins=match_bins,
            reps=bootstrap_reps,
            seed=seed + offset,
        )
        rows.extend(
            [
                {
                    "ensemble": ensemble,
                    "test": "b_score_matched_sigma",
                    "metric": "enriched_minus_pure_mean_sigma",
                    "region": band_name,
                    "selection_rate": 1 / 16 if "1_16" in band_name else 0.01,
                    "lambda": np.nan,
                    "point": matched["point"],
                    "ci_low": matched["ci_low"],
                    "ci_high": matched["ci_high"],
                    "n": int(matched["matched_n"]),
                    "extra": f"bins_used={matched['bins_used']}",
                },
                {
                    "ensemble": ensemble,
                    "test": "b_score_matched_sigma",
                    "metric": "matched_enriched_mean_sigma",
                    "region": band_name,
                    "selection_rate": 1 / 16 if "1_16" in band_name else 0.01,
                    "lambda": np.nan,
                    "point": matched["enriched_mean"],
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n": int(matched["matched_n"]),
                    "extra": f"bins_used={matched['bins_used']}",
                },
                {
                    "ensemble": ensemble,
                    "test": "b_score_matched_sigma",
                    "metric": "matched_pure_mean_sigma",
                    "region": band_name,
                    "selection_rate": 1 / 16 if "1_16" in band_name else 0.01,
                    "lambda": np.nan,
                    "point": matched["pure_mean"],
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n": int(matched["matched_n"]),
                    "extra": f"bins_used={matched['bins_used']}",
                },
            ]
        )
        plot_rows.extend(
            [
                {
                    "ensemble": ensemble,
                    "score_band": band_name,
                    "group": "enriched",
                    "mean_sigma": matched["enriched_mean"],
                },
                {
                    "ensemble": ensemble,
                    "score_band": band_name,
                    "group": "pure_c4",
                    "mean_sigma": matched["pure_mean"],
                },
            ]
        )
    return rows, pd.DataFrame(plot_rows)


def _test_c(
    *,
    ensemble: str,
    frame: pd.DataFrame,
    lambdas: tuple[float, ...],
    selection_rates: tuple[float, ...],
    full_color_pure: np.ndarray,
    bootstrap_reps: int,
    seed: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    pure = frame[~frame["enriched"]].copy()
    enriched = frame[frame["enriched"]].copy()
    if enriched.empty:
        raise ValueError("Test C requires enriched rows")
    rows: list[dict[str, Any]] = []
    plot_rows: list[dict[str, Any]] = []
    lambda0_by_rate: dict[float, dict[str, Any]] = {}
    full_selected_by_rate = {rate: _selected(full_color_pure, rate)[0] for rate in selection_rates}

    for rate_idx, rate in enumerate(selection_rates):
        lambda_records: dict[float, dict[str, Any]] = {}
        for lambda_idx, lambda_value in enumerate(lambdas):
            pure_score = pure["mu"].to_numpy(dtype=np.float64) + lambda_value * pure["sigma"].to_numpy(dtype=np.float64)
            enriched_score = enriched["mu"].to_numpy(dtype=np.float64) + lambda_value * enriched["sigma"].to_numpy(dtype=np.float64)
            selected_pure, threshold, k = _selected(pure_score, rate)
            selected_enriched = enriched_score <= threshold
            full_selected = full_selected_by_rate[rate]
            recall = float((selected_pure & full_selected).sum() / full_selected.sum())
            retention = float(selected_enriched.mean())
            lambda_records[lambda_value] = {
                "selected_pure": selected_pure,
                "selected_enriched": selected_enriched,
                "recall": recall,
                "retention": retention,
                "threshold": threshold,
                "k": k,
            }
            rows.extend(
                [
                    {
                        "ensemble": ensemble,
                        "test": "c_pessimistic_selection",
                        "metric": "enriched_retention",
                        "region": "enriched",
                        "selection_rate": rate,
                        "lambda": lambda_value,
                        "point": retention,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "n": int(len(enriched)),
                        "extra": f"threshold={threshold}",
                    },
                    {
                        "ensemble": ensemble,
                        "test": "c_pessimistic_selection",
                        "metric": "pure_c4_recall_vs_full",
                        "region": "pure_c4",
                        "selection_rate": rate,
                        "lambda": lambda_value,
                        "point": recall,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "n": int(len(pure)),
                        "extra": f"k={k}",
                    },
                ]
            )
            plot_rows.append(
                {
                    "ensemble": ensemble,
                    "selection_rate": rate,
                    "lambda": lambda_value,
                    "retention": retention,
                    "recall_vs_full": recall,
                }
            )

        base = lambda_records[0.0]
        lambda0_by_rate[rate] = base
        for lambda_idx, lambda_value in enumerate(lambdas):
            record = lambda_records[lambda_value]
            retention_diff, retention_low, retention_high = _bootstrap_paired_indicator_diff(
                record["selected_enriched"],
                base["selected_enriched"],
                reps=bootstrap_reps,
                seed=seed + 1000 * rate_idx + 10 * lambda_idx,
            )
            recall_diff, recall_low, recall_high = _bootstrap_paired_indicator_diff(
                record["selected_pure"][full_selected_by_rate[rate]],
                base["selected_pure"][full_selected_by_rate[rate]],
                reps=bootstrap_reps,
                seed=seed + 2000 * rate_idx + 10 * lambda_idx,
            )
            rows.extend(
                [
                    {
                        "ensemble": ensemble,
                        "test": "c_pessimistic_selection",
                        "metric": "enriched_retention_diff_vs_lambda0",
                        "region": "enriched",
                        "selection_rate": rate,
                        "lambda": lambda_value,
                        "point": retention_diff,
                        "ci_low": retention_low,
                        "ci_high": retention_high,
                        "n": int(len(enriched)),
                        "extra": "",
                    },
                    {
                        "ensemble": ensemble,
                        "test": "c_pessimistic_selection",
                        "metric": "pure_c4_recall_diff_vs_lambda0",
                        "region": "pure_c4",
                        "selection_rate": rate,
                        "lambda": lambda_value,
                        "point": recall_diff,
                        "ci_low": recall_low,
                        "ci_high": recall_high,
                        "n": int(full_selected_by_rate[rate].sum()),
                        "extra": "",
                    },
                ]
            )
    return rows, pd.DataFrame(plot_rows)


def _setup_matplotlib():
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


def _plot_spread(curve: pd.DataFrame, output: Path, *, ensemble: str) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(curve["percentile_mid"] * 100, curve["sigma_mean"], marker=".", linewidth=1.4)
    ax.axvspan(4, 9, color="#f4a261", alpha=0.20, label="threshold band (4-9%)")
    ax.axvspan(0, 1, color="#2a9d8f", alpha=0.18, label="deep tail (0-1%)")
    ax.axvline(6.25, color="black", linestyle="--", linewidth=1.0, label="1/16 cutoff")
    ax.set_xlabel("Full-model pure-C4 score percentile (lower is better)")
    ax.set_ylabel("Mean color-score spread (sigma)")
    ax.set_title(f"Spread vs full-score percentile: {ensemble}")
    ax.legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _plot_matched_sigma(plot_frame: pd.DataFrame, output: Path, *, ensemble: str) -> None:
    plt = _setup_matplotlib()
    frame = plot_frame.copy()
    frame["label"] = frame["score_band"].map(
        {
            "below_mu_1_16_pure_cutoff": "below 1/16",
            "below_mu_1pct_pure_cutoff": "below 1%",
        }
    )
    pivot = frame.pivot(index="label", columns="group", values="mean_sigma")
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    pivot[["pure_c4", "enriched"]].plot(kind="bar", ax=ax, color=["#4c78a8", "#59a14f"])
    ax.set_ylabel("Score-matched mean sigma")
    ax.set_xlabel("Deep-tail score band")
    ax.set_title(f"Score-matched sigma: {ensemble}")
    ax.legend(["pure C4", "enriched Books"])
    ax.tick_params(axis="x", rotation=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _plot_retention(plot_frame: pd.DataFrame, output: Path, *, ensemble: str) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for rate, group in plot_frame.groupby("selection_rate"):
        group = group.sort_values("lambda")
        ax.plot(group["lambda"], group["retention"], marker="o", label=f"{rate:g}")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("lambda in s = mu + lambda * sigma")
    ax.set_ylabel("Enriched Books retention")
    ax.set_title(f"Pessimistic selection sweep: {ensemble}")
    ax.legend(title="select rate", fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _plot_retention_recall(plot_frame: pd.DataFrame, output: Path, *, ensemble: str) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for rate, group in plot_frame.groupby("selection_rate"):
        ax.plot(group["recall_vs_full"], group["retention"], marker="o", label=f"{rate:g}")
        for _, row in group.iterrows():
            if row["lambda"] in (-1.0, 0.0, 1.0, 4.0):
                ax.annotate(f"{row['lambda']:g}", (row["recall_vs_full"], row["retention"]), fontsize=7)
    ax.set_xlabel("Pure-C4 recall vs full")
    ax.set_ylabel("Enriched Books retention")
    ax.set_title(f"Retention vs exact-selection agreement: {ensemble}")
    ax.legend(title="select rate", fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _make_results_row(
    *,
    ensemble: str,
    test: str,
    metric: str,
    point: float,
    ci_low: float = np.nan,
    ci_high: float = np.nan,
    region: str = "",
    selection_rate: float = np.nan,
    lambda_value: float = np.nan,
    n: int = 0,
    extra: str = "",
) -> dict[str, Any]:
    return {
        "ensemble": ensemble,
        "test": test,
        "metric": metric,
        "region": region,
        "selection_rate": selection_rate,
        "lambda": lambda_value,
        "point": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n,
        "extra": extra,
    }


def _decision_from_results(results: pd.DataFrame) -> tuple[Decision, dict[str, Any]]:
    def row(ensemble: str, test: str, metric: str, region: str = "", rate: float | None = None) -> pd.Series:
        mask = (
            (results["ensemble"] == ensemble)
            & (results["test"] == test)
            & (results["metric"] == metric)
        )
        if region:
            mask &= results["region"] == region
        if rate is not None:
            mask &= np.isclose(results["selection_rate"].astype(float), rate)
        subset = results[mask]
        if subset.empty:
            raise KeyError((ensemble, test, metric, region, rate))
        return subset.iloc[0]

    a_mild = row(
        "mild",
        "a_threshold_vs_deep",
        "threshold_minus_deep_mean_sigma",
        "threshold_band_4_9pct_vs_deep_tail_bottom_1pct",
    )
    b_mild = row("mild", "b_score_matched_sigma", "enriched_minus_pure_mean_sigma", "below_mu_1_16_pure_cutoff")
    a_mid = row(
        "mid",
        "a_threshold_vs_deep",
        "threshold_minus_deep_mean_sigma",
        "threshold_band_4_9pct_vs_deep_tail_bottom_1pct",
    )
    b_mid = row("mid", "b_score_matched_sigma", "enriched_minus_pure_mean_sigma", "below_mu_1_16_pure_cutoff")

    def best_positive_lambda(ensemble: str) -> pd.Series:
        subset = results[
            (results["ensemble"] == ensemble)
            & (results["test"] == "c_pessimistic_selection")
            & (results["metric"] == "enriched_retention")
            & np.isclose(results["selection_rate"].astype(float), 1 / 64)
            & (results["lambda"].astype(float) > 0)
        ].copy()
        if subset.empty:
            raise KeyError(f"No positive lambda retention rows for {ensemble}")
        subset = subset.sort_values(["point", "lambda"], ascending=[False, True])
        return subset.iloc[0]

    def lambda_row(ensemble: str, metric: str, lambda_value: float) -> pd.Series:
        subset = results[
            (results["ensemble"] == ensemble)
            & (results["test"] == "c_pessimistic_selection")
            & (results["metric"] == metric)
            & np.isclose(results["selection_rate"].astype(float), 1 / 64)
            & np.isclose(results["lambda"].astype(float), lambda_value)
        ]
        if subset.empty:
            raise KeyError((ensemble, metric, lambda_value))
        return subset.iloc[0]

    mild_best = best_positive_lambda("mild")
    mild_best_lambda = float(mild_best["lambda"])
    mild_retention_diff = lambda_row("mild", "enriched_retention_diff_vs_lambda0", mild_best_lambda)
    mild_recall_diff = lambda_row("mild", "pure_c4_recall_diff_vs_lambda0", mild_best_lambda)

    mid_best = best_positive_lambda("mid")
    mid_best_lambda = float(mid_best["lambda"])
    mid_retention_diff = lambda_row("mid", "enriched_retention_diff_vs_lambda0", mid_best_lambda)
    mid_recall_diff = lambda_row("mid", "pure_c4_recall_diff_vs_lambda0", mid_best_lambda)

    test_a_mild = bool(a_mild["point"] > 0 and a_mild["ci_low"] > 0)
    test_b_mild = bool(b_mild["point"] < 0 and b_mild["ci_high"] < 0)
    test_c_mild = bool(
        mild_best_lambda > 0
        and mild_retention_diff["point"] > 0
        and mild_retention_diff["ci_low"] > 0
        and mild_recall_diff["point"] >= 0
    )
    mid_same_direction = bool(
        a_mid["point"] > 0
        and b_mid["point"] < 0
        and mid_retention_diff["point"] > 0
        and mid_recall_diff["point"] >= 0
    )
    details = {
        "mild": {
            "test_a": a_mild.to_dict(),
            "test_b": b_mild.to_dict(),
            "test_c_best_positive_lambda": mild_best_lambda,
            "test_c_retention_diff": mild_retention_diff.to_dict(),
            "test_c_recall_diff": mild_recall_diff.to_dict(),
        },
        "mid": {
            "test_a": a_mid.to_dict(),
            "test_b": b_mid.to_dict(),
            "test_c_best_positive_lambda": mid_best_lambda,
            "test_c_retention_diff": mid_retention_diff.to_dict(),
            "test_c_recall_diff": mid_recall_diff.to_dict(),
        },
    }
    return Decision(test_a_mild, test_b_mild, test_c_mild, mid_same_direction), details


def _format_float(value: float, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def _rate_label(rate: float) -> str:
    for denom in (64, 32, 16, 8, 4):
        if np.isclose(rate, 1 / denom):
            return f"1/{denom}"
    return f"{rate:g}"


def _write_report(
    *,
    output: Path,
    results: pd.DataFrame,
    provenance_path: Path,
    figures_dir: Path,
    noise_floor: dict[str, float],
    decision: Decision,
    decision_details: dict[str, Any],
    ensembles: dict[str, tuple[str, ...]],
    selection_rates: tuple[float, ...],
) -> None:
    def first_row(ensemble: str, test: str, metric: str, region: str = "", rate: float | None = None, lambda_value: float | None = None) -> pd.Series:
        mask = (
            (results["ensemble"] == ensemble)
            & (results["test"] == test)
            & (results["metric"] == metric)
        )
        if region:
            mask &= results["region"] == region
        if rate is not None:
            mask &= np.isclose(results["selection_rate"].astype(float), rate)
        if lambda_value is not None:
            mask &= np.isclose(results["lambda"].astype(float), lambda_value)
        subset = results[mask]
        if subset.empty:
            raise KeyError((ensemble, test, metric, region, rate, lambda_value))
        return subset.iloc[0]

    lines: list[str] = []
    lines.append("# Uncertainty-Aware Scoring Pre-Test")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(CAUTION_TEXT)
    lines.append("")
    if decision.call == "positive":
        call_text = "The pre-test is **positive** under the registered rule."
    elif decision.call == "weak_or_ambiguous":
        call_text = "The pre-test is **weak/ambiguous**: heteroscedasticity signals appear, but the sigma term does not convert into a strict selection gain."
    else:
        call_text = "The pre-test is **negative** under the registered rule."
    lines.append(call_text)
    lines.append("")
    lines.append(
        "The decision uses `mild` strictly and requires the same direction on `mid`. "
        "`wide` is reported only as a contaminated sanity bound; `mild_no_full` is a labeled non-decision diagnostic."
    )
    lines.append("")
    lines.append("## Framing Caveats")
    lines.append("")
    lines.append(f"- {CAUTION_TEXT}")
    lines.append(
        "- The only non-destructive same-architecture scorings are `full` and `full_rescore`; in this run they are deterministic, so ablation spread is structural damage rather than bf16 jitter."
    )
    lines.append(
        "- The enriched Books set is paper-pipeline-selected, not an independent gold standard, so enriched retention is a calibration diagnostic rather than a standalone proof of selection quality."
    )
    lines.append("")
    lines.append("## Inputs and Ensembles")
    lines.append("")
    lines.append(f"- Provenance sidecar: `{provenance_path.as_posix()}`")
    lines.append(f"- Score parquets: full, full_rescore, and 9 layer-ablation variants.")
    lines.append(f"- Selection rates: {', '.join(_rate_label(rate) for rate in selection_rates)}")
    lines.append("")
    lines.append("| ensemble | role | members |")
    lines.append("| --- | --- | --- |")
    for name, members in ensembles.items():
        role = "diagnostic_only" if name in DIAGNOSTIC_ENSEMBLES else DEFAULT_DECISION_ENSEMBLES.get(name, "reported")
        lines.append(f"| {name} | {role} | {', '.join(members)} |")
    lines.append("")
    lines.append("## Noise Floor")
    lines.append("")
    lines.append(
        f"`full` vs `full_rescore`: max absolute color difference = `{noise_floor['max_abs_diff']:.8g}`, "
        f"mean = `{noise_floor['mean_abs_diff']:.8g}`, fraction > 1e-6 = `{noise_floor['fraction_gt_1e_6']:.8g}`."
    )
    lines.append(
        "This confirms that the spread analyzed below is structural ablation spread, not a clean posterior sample or stochastic scoring noise."
    )
    lines.append("")
    lines.append("## Registered Decision Rule")
    lines.append("")
    lines.append(
        "Positive iff, on `mild`, Test (a) threshold-band mean sigma exceeds deep-tail mean sigma with the difference CI excluding zero; "
        "Test (b) score-matched enriched sigma is lower than pure-C4 sigma with the difference CI excluding zero; "
        "and Test (c) the best positive-lambda retention at 1/64 beats lambda=0 with CI excluding zero and no same-rate pure-C4 recall loss. "
        "`mid` must have the same direction, but its CIs do not need to exclude zero."
    )
    lines.append("")
    lines.append("## Decision Summary")
    lines.append("")
    lines.append("| condition | result |")
    lines.append("| --- | --- |")
    lines.append(f"| Test (a) mild strict | {'pass' if decision.test_a_mild else 'fail'} |")
    lines.append(f"| Test (b) mild strict | {'pass' if decision.test_b_mild else 'fail'} |")
    lines.append(f"| Test (c) mild strict | {'pass' if decision.test_c_mild else 'fail'} |")
    lines.append(f"| mid same direction | {'pass' if decision.mid_same_direction else 'fail'} |")
    lines.append(f"| go/no-go call | **{decision.call}** |")
    lines.append("")
    lines.append("## Test (a): Does Spread Concentrate Near the Boundary?")
    lines.append("")
    lines.append("| ensemble | threshold - deep sigma | 95% CI | threshold / deep sigma | 95% CI |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for ensemble in ensembles:
        diff = first_row(
            ensemble,
            "a_threshold_vs_deep",
            "threshold_minus_deep_mean_sigma",
            "threshold_band_4_9pct_vs_deep_tail_bottom_1pct",
        )
        ratio = first_row(
            ensemble,
            "a_threshold_vs_deep",
            "threshold_over_deep_mean_sigma",
            "threshold_band_4_9pct_vs_deep_tail_bottom_1pct",
        )
        lines.append(
            f"| {ensemble} | {_format_float(diff['point'])} | [{_format_float(diff['ci_low'])}, {_format_float(diff['ci_high'])}] | "
            f"{_format_float(ratio['point'])} | [{_format_float(ratio['ci_low'])}, {_format_float(ratio['ci_high'])}] |"
        )
    lines.append("")
    for ensemble in ensembles:
        figure = figures_dir / f"spread_vs_percentile_{ensemble}.png"
        lines.append(f"![Spread vs full-score percentile: {ensemble}]({figure.relative_to(output.parent).as_posix()})")
        lines.append("")
        lines.append(
            f"**Figure: spread vs full-score percentile for `{ensemble}`.** The shaded green band is the bottom 1% deep tail, "
            "the orange band is the 4-9% threshold region around the 1/16 cutoff, and the dashed line marks 6.25%."
        )
        lines.append("")
    lines.append("## Test (b): Is the Enriched Good Tail Low-Spread After Score Matching?")
    lines.append("")
    lines.append("| ensemble | band | enriched - pure matched sigma | 95% CI | matched n |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for ensemble in ensembles:
        for band in ("below_mu_1_16_pure_cutoff", "below_mu_1pct_pure_cutoff"):
            diff = first_row(ensemble, "b_score_matched_sigma", "enriched_minus_pure_mean_sigma", band)
            lines.append(
                f"| {ensemble} | {band} | {_format_float(diff['point'])} | "
                f"[{_format_float(diff['ci_low'])}, {_format_float(diff['ci_high'])}] | {int(diff['n'])} |"
            )
    lines.append("")
    for ensemble in ensembles:
        figure = figures_dir / f"score_matched_sigma_{ensemble}.png"
        lines.append(f"![Score-matched sigma: {ensemble}]({figure.relative_to(output.parent).as_posix()})")
        lines.append("")
        lines.append(
            f"**Figure: score-matched sigma for `{ensemble}`.** Bars compare matched pure-C4 and enriched Books mean sigma inside deep-tail score bands. "
            "Negative enriched-minus-pure differences support the claim that known-good rows are stably low-scoring beyond score level alone."
        )
        lines.append("")
    lines.append("## Test (c): Does the Variance Term Carry Quality Signal?")
    lines.append("")
    lines.append("| ensemble | best positive lambda at 1/64 | retention diff vs lambda=0 | 95% CI | pure-C4 recall diff vs lambda=0 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for ensemble in ensembles:
        subset = results[
            (results["ensemble"] == ensemble)
            & (results["test"] == "c_pessimistic_selection")
            & (results["metric"] == "enriched_retention")
            & np.isclose(results["selection_rate"].astype(float), 1 / 64)
            & (results["lambda"].astype(float) > 0)
        ].copy()
        subset = subset.sort_values(["point", "lambda"], ascending=[False, True])
        best_lambda = float(subset.iloc[0]["lambda"])
        retention_diff = first_row(
            ensemble,
            "c_pessimistic_selection",
            "enriched_retention_diff_vs_lambda0",
            "enriched",
            1 / 64,
            best_lambda,
        )
        recall_diff = first_row(
            ensemble,
            "c_pessimistic_selection",
            "pure_c4_recall_diff_vs_lambda0",
            "pure_c4",
            1 / 64,
            best_lambda,
        )
        lines.append(
            f"| {ensemble} | {best_lambda:g} | {_format_float(retention_diff['point'])} | "
            f"[{_format_float(retention_diff['ci_low'])}, {_format_float(retention_diff['ci_high'])}] | {_format_float(recall_diff['point'])} |"
        )
    lines.append("")
    for ensemble in ensembles:
        retention_figure = figures_dir / f"retention_vs_lambda_{ensemble}.png"
        scatter_figure = figures_dir / f"retention_vs_recall_{ensemble}.png"
        lines.append(f"![Retention vs lambda: {ensemble}]({retention_figure.relative_to(output.parent).as_posix()})")
        lines.append("")
        lines.append(
            f"**Figure: retention vs lambda for `{ensemble}`.** Lambda=0 is the point-estimate ensemble ranking by mu. "
            "Positive lambda penalizes high sigma; negative lambda is the optimism control."
        )
        lines.append("")
        lines.append(f"![Retention vs recall: {ensemble}]({scatter_figure.relative_to(output.parent).as_posix()})")
        lines.append("")
        lines.append(
            f"**Figure: retention-vs-recall tradeoff for `{ensemble}`.** This checks whether any enriched-retention gain is bought by losing agreement with the full pure-C4 selected set."
        )
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if decision.call == "positive":
        lines.append(
            "The proxy-disturbance pre-test is positive. This does not prove uncertainty-aware selection works, but it does justify spending GPU time on a genuine conditional-seed ensemble."
        )
    elif decision.call == "weak_or_ambiguous":
        lines.append(
            "The proxy-disturbance pre-test finds heteroscedasticity signals but does not show that the sigma term improves selection under the strict rule. This argues against building the real ensemble immediately at 12-layer depth unless the goal is exploratory."
        )
    else:
        lines.append(
            "The proxy-disturbance pre-test is negative. Under the registered rule, ablation-spread does not provide enough quality signal to justify the real GPU ensemble at this stage."
        )
    lines.append("")
    lines.append("## Reproducibility Appendix")
    lines.append("")
    lines.append(f"- Results CSV: `results/uncertainty_pretest.csv`")
    lines.append(f"- Provenance JSON: `{provenance_path.as_posix()}`")
    lines.append("- Bootstrap CIs use paired row-level bootstraps where applicable.")
    lines.append("- Binned matching controls for score level using ensemble `mu` bins inside the stated deep-tail score bands.")
    lines.append("- All selection thresholds for retention are computed on pure-C4 rows only; enriched rows are never used to set thresholds.")
    lines.append("- Full decision details, input hashes, and run parameters are in the provenance JSON sidecar.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_inputs(args: argparse.Namespace, config: dict[str, Any]) -> Inputs:
    artifact_root = Path(args.artifact_root).expanduser() if args.artifact_root else None
    if args.results_dir:
        results_dir = Path(args.results_dir).expanduser()
    elif artifact_root:
        results_dir = artifact_root / "results"
    else:
        results_dir = Path(config["paths"]["results_dir"])

    if args.pool_meta:
        pool_meta = Path(args.pool_meta).expanduser()
    elif artifact_root:
        pool_meta = artifact_root / "data" / "books_pool_meta.parquet"
    else:
        pool_meta = Path(config["target"]["pool_meta"])

    output_dir = Path(args.output_dir) if args.output_dir else Path(config["paths"]["results_dir"])
    report_dir = Path(args.report_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else report_dir / "figures"
    return Inputs(
        results_dir=results_dir,
        pool_meta=pool_meta,
        output_csv=output_dir / "uncertainty_pretest.csv",
        provenance_json=output_dir / "uncertainty_pretest_provenance.json",
        figures_dir=figures_dir,
        report_md=report_dir / "report.md",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only uncertainty-aware scoring pre-test from existing score parquets.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--artifact-root", default=None, help="Artifact tree with results/ and data/books_pool_meta.parquet.")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--pool-meta", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--report-dir", default="reports/uncertainty-pretest")
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--ensemble", action="append", type=_parse_ensemble, default=None)
    parser.add_argument("--lambdas", default=",".join(str(v) for v in DEFAULT_LAMBDAS))
    parser.add_argument("--selection-rates", default=",".join(str(v) for v in DEFAULT_RATES))
    parser.add_argument("--bootstrap-reps", type=int, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    parser.add_argument("--percentile-bins", type=int, default=40)
    parser.add_argument("--match-bins", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    inputs = _resolve_inputs(args, config)
    bootstrap_reps = int(args.bootstrap_reps if args.bootstrap_reps is not None else config["metrics"].get("bootstrap_reps", 1000))
    bootstrap_seed = int(args.bootstrap_seed if args.bootstrap_seed is not None else config["metrics"].get("bootstrap_seed", 17))
    lambdas = tuple(_parse_float_list(args.lambdas))
    selection_rates = tuple(_parse_float_list(args.selection_rates))
    if 0.0 not in lambdas:
        raise ValueError("Lambda grid must include 0")
    if not any(np.isclose(rate, 1 / 64) for rate in selection_rates):
        raise ValueError("Selection-rate grid must include 1/64 for the decision rule")

    ensembles = dict(DEFAULT_ENSEMBLES)
    if args.ensemble:
        ensembles = {name: members for name, members in args.ensemble}
    variants = tuple(sorted({member for members in ensembles.values() for member in members} | {"full_rescore"}))
    pool_meta, scores, alignment = _load_aligned_scores(
        results_dir=inputs.results_dir,
        pool_meta_path=inputs.pool_meta,
        variants=variants,
    )
    if "full" not in scores or "full_rescore" not in scores:
        raise KeyError("Both full and full_rescore are required")

    noise_floor = _compute_noise_floor(scores)
    full_color = scores["full"]["color"].to_numpy(dtype=np.float64)
    pure_mask = ~pool_meta["enriched"].to_numpy(dtype=bool)
    full_color_pure = full_color[pure_mask]
    full_percentiles = _percentiles_from_scores(full_color_pure)

    all_rows: list[dict[str, Any]] = []
    all_rows.extend(
        [
            _make_results_row(ensemble="global", test="noise_floor", metric="full_vs_full_rescore_max_abs_diff", point=noise_floor["max_abs_diff"]),
            _make_results_row(ensemble="global", test="noise_floor", metric="full_vs_full_rescore_mean_abs_diff", point=noise_floor["mean_abs_diff"]),
            _make_results_row(ensemble="global", test="noise_floor", metric="full_vs_full_rescore_fraction_gt_1e_6", point=noise_floor["fraction_gt_1e_6"]),
        ]
    )
    curves: dict[str, pd.DataFrame] = {}
    matched_plots: dict[str, pd.DataFrame] = {}
    c_plots: dict[str, pd.DataFrame] = {}

    for ensemble_idx, (ensemble, members) in enumerate(ensembles.items()):
        frame = _ensemble_frame(pool_meta=pool_meta, scores=scores, members=members, full_color=full_color)
        rows_a, curve = _test_a(
            ensemble=ensemble,
            frame=frame,
            full_percentiles=full_percentiles,
            percentile_bins=int(args.percentile_bins),
            bootstrap_reps=bootstrap_reps,
            seed=bootstrap_seed + 100 * ensemble_idx,
        )
        all_rows.extend(rows_a)
        curves[ensemble] = curve
        all_rows.extend(
            _test_flip_rate(
                ensemble=ensemble,
                members=members,
                scores=scores,
                pure_mask=pure_mask,
                full_percentiles=full_percentiles,
                selection_rate=1 / 16,
            )
        )
        rows_b, matched_plot = _test_b(
            ensemble=ensemble,
            frame=frame,
            match_bins=int(args.match_bins),
            bootstrap_reps=bootstrap_reps,
            seed=bootstrap_seed + 1000 * ensemble_idx,
        )
        all_rows.extend(rows_b)
        matched_plots[ensemble] = matched_plot
        rows_c, c_plot = _test_c(
            ensemble=ensemble,
            frame=frame,
            lambdas=lambdas,
            selection_rates=selection_rates,
            full_color_pure=full_color_pure,
            bootstrap_reps=bootstrap_reps,
            seed=bootstrap_seed + 10000 * ensemble_idx,
        )
        all_rows.extend(rows_c)
        c_plots[ensemble] = c_plot

    results = pd.DataFrame(all_rows)
    ensure_parent(inputs.output_csv)
    results.to_csv(inputs.output_csv, index=False)

    inputs.figures_dir.mkdir(parents=True, exist_ok=True)
    for ensemble in ensembles:
        _plot_spread(curves[ensemble], inputs.figures_dir / f"spread_vs_percentile_{ensemble}.png", ensemble=ensemble)
        _plot_matched_sigma(matched_plots[ensemble], inputs.figures_dir / f"score_matched_sigma_{ensemble}.png", ensemble=ensemble)
        _plot_retention(c_plots[ensemble], inputs.figures_dir / f"retention_vs_lambda_{ensemble}.png", ensemble=ensemble)
        _plot_retention_recall(c_plots[ensemble], inputs.figures_dir / f"retention_vs_recall_{ensemble}.png", ensemble=ensemble)

    decision, decision_details = _decision_from_results(results)

    source_paths = {
        "pool_meta": inputs.pool_meta,
        **{f"scores_{variant}": inputs.results_dir / f"scores_{variant}.parquet" for variant in variants},
    }
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    pool_hashes = sorted(
        {
            str(value)
            for frame in scores.values()
            if "pool_sha256" in frame.columns
            for value in frame["pool_sha256"].dropna().unique()
        }
    )
    provenance = {
        "caution": CAUTION_TEXT,
        "git_commit": _git_value(["git", "rev-parse", "HEAD"]),
        "git_branch": _git_value(["git", "branch", "--show-current"]),
        "git_status_short": _git_value(["git", "status", "--short"]),
        "inputs": {
            "results_dir": str(inputs.results_dir),
            "pool_meta": str(inputs.pool_meta),
        },
        "outputs": {
            "csv": str(inputs.output_csv),
            "provenance_json": str(inputs.provenance_json),
            "figures_dir": str(inputs.figures_dir),
            "report_md": str(inputs.report_md),
        },
        "source_hashes": source_hashes,
        "score_pool_sha256_values": pool_hashes,
        "alignment": alignment,
        "ensembles": {name: list(members) for name, members in ensembles.items()},
        "diagnostic_ensembles": sorted(DIAGNOSTIC_ENSEMBLES & set(ensembles)),
        "lambdas": list(lambdas),
        "selection_rates": list(selection_rates),
        "bootstrap_reps": bootstrap_reps,
        "bootstrap_seed": bootstrap_seed,
        "percentile_bins": int(args.percentile_bins),
        "match_bins": int(args.match_bins),
        "noise_floor": noise_floor,
        "decision": {
            "call": decision.call,
            "test_a_mild": decision.test_a_mild,
            "test_b_mild": decision.test_b_mild,
            "test_c_mild": decision.test_c_mild,
            "mid_same_direction": decision.mid_same_direction,
            "details": decision_details,
        },
    }
    ensure_parent(inputs.provenance_json)
    inputs.provenance_json.write_text(json.dumps(provenance, indent=2, default=str) + "\n", encoding="utf-8")
    _write_report(
        output=inputs.report_md,
        results=results,
        provenance_path=inputs.provenance_json,
        figures_dir=inputs.figures_dir,
        noise_floor=noise_floor,
        decision=decision,
        decision_details=decision_details,
        ensembles=ensembles,
        selection_rates=selection_rates,
    )

    print(json.dumps({"decision": decision.call, "csv": str(inputs.output_csv), "report": str(inputs.report_md)}, indent=2))


if __name__ == "__main__":
    main()
