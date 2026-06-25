#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ensure_parent, load_config


POOL_LABELS = {
    "random_positive": "Random positives",
    "hard_positive": "Hard positives",
    "random_negative": "Random negatives",
    "hard_negative": "Hard negatives",
    "tail_negative": "Tail negatives",
}


def log(message: str) -> None:
    print(message, flush=True)


def choose_pool_indices(
    full_scores: pd.DataFrame,
    *,
    sample_size: int,
    seed: int,
    positive_tau: int = 64,
    negative_tau: int = 32,
) -> dict[str, np.ndarray]:
    if positive_tau <= 1 or negative_tau <= 1:
        raise ValueError("tau values must be greater than 1")
    if negative_tau >= positive_tau:
        # tau=32 should be a larger selected set than tau=64.
        raise ValueError("negative_tau must be less than positive_tau")
    required = {"seq_idx", "color"}
    missing = sorted(required - set(full_scores.columns))
    if missing:
        raise ValueError(f"full_scores missing columns: {missing}")

    pure = full_scores.sort_values("color", ascending=True).reset_index(drop=True)
    n_rows = len(pure)
    k_positive = int(math.ceil(n_rows / positive_tau))
    k_negative_band = int(math.ceil(n_rows / negative_tau))
    tau64 = pure.iloc[:k_positive]
    tau32_minus_tau64 = pure.iloc[k_positive:k_negative_band]
    not_tau64 = pure.iloc[k_positive:]

    available = {
        "tau64": len(tau64),
        "tau32_minus_tau64": len(tau32_minus_tau64),
        "not_tau64": len(not_tau64),
    }
    too_small = {name: count for name, count in available.items() if count < sample_size}
    if too_small:
        raise ValueError(
            f"sample_size={sample_size:,} is too large for fallback pool counts {available}. "
            "Increase the source pool size or reduce --sample-size."
        )

    rng = np.random.default_rng(seed)
    tau64_seq = tau64["seq_idx"].to_numpy(dtype=np.int64)
    band_seq = tau32_minus_tau64["seq_idx"].to_numpy(dtype=np.int64)
    tail_seq = not_tau64["seq_idx"].to_numpy(dtype=np.int64)
    return {
        "random_positive": rng.choice(tau64_seq, size=sample_size, replace=False),
        "hard_positive": tau64_seq[-sample_size:],
        "random_negative": rng.choice(band_seq, size=sample_size, replace=False),
        "hard_negative": band_seq[:sample_size],
        "tail_negative": rng.choice(tail_seq, size=sample_size, replace=False),
    }


def build_fallback_metadata(
    full_scores: pd.DataFrame,
    pool_indices: dict[str, np.ndarray],
) -> pd.DataFrame:
    score_frame = full_scores.set_index("seq_idx", drop=False)
    rows: list[pd.DataFrame] = []
    out_offset = 0
    for pool_name, source_indices in pool_indices.items():
        part = score_frame.loc[source_indices].copy().reset_index(drop=True)
        n = len(part)
        out_seq_idx = np.arange(out_offset, out_offset + n, dtype=np.int64)
        out_offset += n
        rows.append(
            pd.DataFrame(
                {
                    "seq_idx": out_seq_idx,
                    "pool_name": pool_name,
                    "pool_label": POOL_LABELS[pool_name],
                    "row_position": part["seq_idx"].to_numpy(dtype=np.int64),
                    "c4_index": part["seq_idx"].to_numpy(dtype=np.int64),
                    "fallback_source_seq_idx": part["seq_idx"].to_numpy(dtype=np.int64),
                    "full_prior_score": part["nll_marg"].to_numpy(dtype=np.float32),
                    "full_conditional_books_score": part["nll_cond"].to_numpy(dtype=np.float32),
                    "full_color_score": part["color"].to_numpy(dtype=np.float32),
                    "label_source": "fallback_full_rescore_pool",
                    "label_available_for_pair_tasks": True,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def cutoff_for_tau(full_scores: pd.DataFrame, tau: int) -> float:
    values = np.sort(full_scores["color"].to_numpy(dtype=np.float64))
    cutoff_idx = int(math.ceil(len(values) / tau)) - 1
    cutoff_idx = min(max(cutoff_idx, 0), len(values) - 1)
    return float(values[cutoff_idx])


def write_sampled_tokens(
    *,
    source_tokens_path: Path,
    metadata: pd.DataFrame,
    output_tokens_path: Path,
) -> None:
    source = np.load(source_tokens_path, mmap_mode="r")
    source_indices = metadata["fallback_source_seq_idx"].to_numpy(dtype=np.int64)
    tokens = np.lib.format.open_memmap(
        output_tokens_path,
        mode="w+",
        dtype=source.dtype,
        shape=(len(metadata), source.shape[1]),
    )
    chunk_rows = 50_000
    for start in range(0, len(metadata), chunk_rows):
        end = min(len(metadata), start + chunk_rows)
        tokens[start:end] = source[source_indices[start:end]]
    tokens.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fallback five-pool token/meta artifacts from a full-model scored source pool."
    )
    parser.add_argument("--config", default="configs/score_pool_robustness.yaml")
    parser.add_argument("--source-config", default="configs/default.yaml")
    parser.add_argument("--full-scores", default=None)
    parser.add_argument("--pool-tokens", default=None)
    parser.add_argument("--pool-meta", default=None)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--positive-tau", type=int, default=64)
    parser.add_argument("--negative-tau", type=int, default=32)
    parser.add_argument("--include-enriched", action="store_true")
    args = parser.parse_args()

    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    config = load_config(args.config)
    source_config = load_config(args.source_config)
    seed = int(config.get("seed", source_config.get("seed", 17)))
    source_results_dir = Path(source_config["paths"]["results_dir"])
    full_scores_path = Path(args.full_scores or source_results_dir / "scores_full.parquet")
    pool_tokens_path = Path(args.pool_tokens or source_config["target"]["pool_tokens"])
    pool_meta_path = Path(args.pool_meta or source_config["target"]["pool_meta"])

    full_scores = pd.read_parquet(full_scores_path)
    if pool_meta_path.exists():
        pool_meta = pd.read_parquet(pool_meta_path)
        if "enriched" in pool_meta.columns and not args.include_enriched:
            pure_ids = pool_meta.loc[~pool_meta["enriched"].astype(bool), "seq_idx"]
            full_scores = full_scores[full_scores["seq_idx"].isin(pure_ids)].copy()
    pool_indices = choose_pool_indices(
        full_scores,
        sample_size=args.sample_size,
        seed=seed,
        positive_tau=args.positive_tau,
        negative_tau=args.negative_tau,
    )
    metadata = build_fallback_metadata(full_scores, pool_indices)

    token_cfg = config["token_recovery"]
    output_tokens = ensure_parent(token_cfg["recovered_tokens"])
    output_meta = ensure_parent(token_cfg["recovered_meta"])
    write_sampled_tokens(
        source_tokens_path=pool_tokens_path,
        metadata=metadata,
        output_tokens_path=output_tokens,
    )
    metadata.to_parquet(output_meta, index=False)

    summary = {
        "label_source": "fallback_full_rescore_pool",
        "source_full_scores": str(full_scores_path),
        "source_pool_tokens": str(pool_tokens_path),
        "source_pool_meta": str(pool_meta_path),
        "sample_size_per_pool": args.sample_size,
        "positive_tau": args.positive_tau,
        "negative_tau": args.negative_tau,
        "fallback_positive_cutoff": cutoff_for_tau(full_scores, args.positive_tau),
        "fallback_negative_band_cutoff": cutoff_for_tau(full_scores, args.negative_tau),
        "source_rows_available": len(full_scores),
        "output_rows": len(metadata),
        "output_tokens": str(output_tokens),
        "output_meta": str(output_meta),
        "pool_counts": metadata["pool_name"].value_counts().sort_index().to_dict(),
        "full_color_ranges": {
            pool_name: {
                "min": float(group["full_color_score"].min()),
                "max": float(group["full_color_score"].max()),
                "mean": float(group["full_color_score"].mean()),
            }
            for pool_name, group in metadata.groupby("pool_name")
        },
    }
    summary_path = ensure_parent(Path(config["paths"]["output_dir"]) / "fallback_pool_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    log(f"wrote tokens: {output_tokens}")
    log(f"wrote metadata: {output_meta}")
    log(f"wrote summary: {summary_path}")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
