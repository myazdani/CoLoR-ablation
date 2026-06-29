#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


P0_RUNS = {
    "random_positive_oracle_100k": {
        "source_pools": ("random_positive",),
        "universe_pools": ("random_positive",),
        "positive_pool": "random_positive",
        "negative_pool": None,
        "selection_policy": "oracle_positive_pool",
    },
    "hard_positive_oracle_100k": {
        "source_pools": ("hard_positive",),
        "universe_pools": ("hard_positive",),
        "positive_pool": "hard_positive",
        "negative_pool": None,
        "selection_policy": "oracle_positive_pool",
    },
    "random_pair_cascade_100k": {
        "source_pools": ("random_positive", "random_negative"),
        "universe_pools": ("random_positive", "random_negative"),
        "positive_pool": "random_positive",
        "negative_pool": "random_negative",
        "selection_policy": "pair_mid2_prefilter_full_rerank",
    },
    "hard_pair_cascade_100k": {
        "source_pools": ("hard_positive", "hard_negative"),
        "universe_pools": ("hard_positive", "hard_negative"),
        "positive_pool": "hard_positive",
        "negative_pool": "hard_negative",
        "selection_policy": "pair_mid2_prefilter_full_rerank",
    },
}

POOL_SAMPLE_FILES = {
    "random_positive": "random_positive_samples.npz",
    "hard_positive": "hard_positive_samples.npz",
    "random_negative": "random_negative_samples.npz",
    "hard_negative": "hard_negative_samples.npz",
    "tail_negative": "tail_negative_samples.npz",
}

SCORE_COLUMNS = [
    "seq_idx",
    "pool_name",
    "c4_index",
    "row_position",
    "full_prior_score",
    "full_conditional_books_score",
    "full_color_score",
    "ablated_prior_score",
    "ablated_conditional_books_score",
    "ablated_color_score",
    "elapsed_seconds",
    "tokens_scored",
    "tokens_per_second",
]


def parse_float_list(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError(f"No values parsed from {raw!r}")
    return tuple(values)


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def score_summary(frame: pd.DataFrame, column: str, prefix: str) -> dict[str, float]:
    values = frame[column].to_numpy(dtype=np.float64)
    return {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_p01": float(np.quantile(values, 0.01)),
        f"{prefix}_p05": float(np.quantile(values, 0.05)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_p99": float(np.quantile(values, 0.99)),
    }


def read_score_frame(path: Path, *, local_score_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing score parquet: {path}")
    frame = pd.read_parquet(path, columns=SCORE_COLUMNS)
    if frame["seq_idx"].duplicated().any():
        raise ValueError(f"{path} contains duplicate seq_idx values")
    return frame.rename(
        columns={
            "ablated_prior_score": f"{local_score_name}_prior_score",
            "ablated_conditional_books_score": f"{local_score_name}_conditional_books_score",
            "ablated_color_score": f"{local_score_name}_color_score",
            "elapsed_seconds": f"{local_score_name}_elapsed_seconds",
            "tokens_scored": f"{local_score_name}_tokens_scored",
            "tokens_per_second": f"{local_score_name}_tokens_per_second",
        }
    )


def load_joined_scores(meta_path: Path, full_scores_path: Path, pair_mid2_scores_path: Path) -> pd.DataFrame:
    meta = pd.read_parquet(meta_path)
    required = {"seq_idx", "pool_name", "c4_index", "row_position", "full_color_score"}
    missing = sorted(required - set(meta.columns))
    if missing:
        raise ValueError(f"{meta_path} missing columns: {missing}")
    if meta["seq_idx"].duplicated().any():
        raise ValueError(f"{meta_path} contains duplicate seq_idx values")

    full = read_score_frame(full_scores_path, local_score_name="local_full")
    pair_mid2 = read_score_frame(pair_mid2_scores_path, local_score_name="pair_mid2")

    joined = meta.merge(
        full[
            [
                "seq_idx",
                "pool_name",
                "local_full_prior_score",
                "local_full_conditional_books_score",
                "local_full_color_score",
                "local_full_elapsed_seconds",
                "local_full_tokens_scored",
                "local_full_tokens_per_second",
            ]
        ],
        on=["seq_idx", "pool_name"],
        how="inner",
        validate="one_to_one",
    ).merge(
        pair_mid2[
            [
                "seq_idx",
                "pool_name",
                "pair_mid2_prior_score",
                "pair_mid2_conditional_books_score",
                "pair_mid2_color_score",
                "pair_mid2_elapsed_seconds",
                "pair_mid2_tokens_scored",
                "pair_mid2_tokens_per_second",
            ]
        ],
        on=["seq_idx", "pool_name"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(meta):
        raise ValueError(
            f"Joined score frame has {len(joined):,} rows, expected {len(meta):,}; score parquets do not cover all rows"
        )
    return joined.sort_values("seq_idx").reset_index(drop=True)


def validate_pool_npz(pool_dir: Path, scores: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pool_name, filename in POOL_SAMPLE_FILES.items():
        path = pool_dir / filename
        if not path.exists():
            continue
        data = np.load(path)
        pool = scores[scores["pool_name"] == pool_name]
        npz_c4 = set(int(x) for x in data["c4_index"].astype(np.int64))
        meta_c4 = set(int(x) for x in pool["c4_index"].to_numpy(dtype=np.int64))
        missing_from_meta = sorted(npz_c4 - meta_c4)[:10]
        extra_in_meta = sorted(meta_c4 - npz_c4)[:10]
        rows.append(
            {
                "pool_name": pool_name,
                "npz_path": str(path),
                "npz_rows": int(len(data["c4_index"])),
                "meta_rows": int(len(pool)),
                "c4_index_sets_match": npz_c4 == meta_c4,
                "first_missing_from_meta": missing_from_meta,
                "first_extra_in_meta": extra_in_meta,
            }
        )
    return rows


def select_lowest(frame: pd.DataFrame, score_column: str, k: int) -> pd.DataFrame:
    if k > len(frame):
        raise ValueError(f"Cannot select {k:,} rows from frame with {len(frame):,} rows")
    return frame.sort_values([score_column, "seq_idx"], ascending=[True, True], kind="mergesort").head(k).copy()


def select_oracle(scores: pd.DataFrame, pool_name: str, target_rows: int) -> pd.DataFrame:
    frame = scores[scores["pool_name"] == pool_name].copy()
    if len(frame) < target_rows:
        raise ValueError(f"Pool {pool_name} has {len(frame):,} rows, need {target_rows:,}")
    selected = frame.sort_values("seq_idx", kind="mergesort").head(target_rows).copy()
    selected["selection_rank"] = np.arange(len(selected), dtype=np.int64)
    selected["selection_stage"] = "oracle"
    selected["candidate_selected"] = True
    return selected


def select_cascade(
    scores: pd.DataFrame,
    *,
    universe_pools: tuple[str, str],
    target_rows: int,
    multiplier: float,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    universe = scores[scores["pool_name"].isin(universe_pools)].copy()
    if len(universe) < target_rows:
        raise ValueError(f"Universe {universe_pools} has {len(universe):,} rows, need {target_rows:,}")
    candidate_count = min(len(universe), int(math.ceil(multiplier * target_rows)))
    candidates = select_lowest(universe, "pair_mid2_color_score", candidate_count)
    final = select_lowest(candidates, "local_full_color_score", target_rows)
    final["selection_rank"] = np.arange(len(final), dtype=np.int64)
    final["selection_stage"] = "cascade_final"
    final["candidate_selected"] = True
    candidates = candidates[["seq_idx"]].copy()
    candidates["candidate_selected"] = True
    return final, candidates, candidate_count


def diagnostic_row(
    selected: pd.DataFrame,
    *,
    run_id: str,
    selection_policy: str,
    source_pools: tuple[str, ...],
    universe_rows: int,
    positive_pool: str,
    target_rows: int,
    multiplier: float | None,
    candidate_count: int | None,
    trained_p0: bool,
    oracle_positive_ids: set[int],
) -> dict[str, object]:
    selected_ids = set(int(x) for x in selected["seq_idx"].to_numpy(dtype=np.int64))
    intersection = len(selected_ids & oracle_positive_ids)
    union = len(selected_ids | oracle_positive_ids)
    positive_count = int((selected["pool_name"] == positive_pool).sum())
    row: dict[str, object] = {
        "run_id": run_id,
        "selection_policy": selection_policy,
        "source_pools": ",".join(source_pools),
        "target_rows": target_rows,
        "selected_rows": int(len(selected)),
        "unique_seq_idx": int(selected["seq_idx"].nunique()),
        "unique_c4_index": int(selected["c4_index"].nunique()),
        "universe_rows": universe_rows,
        "cascade_multiplier": multiplier,
        "candidate_count": candidate_count,
        "candidate_fraction": (candidate_count / universe_rows) if candidate_count is not None else None,
        "positive_pool": positive_pool,
        "true_positive_count": positive_count,
        "true_positive_rate": positive_count / len(selected) if len(selected) else float("nan"),
        "false_positive_count": int(len(selected) - positive_count),
        "oracle_positive_count": len(oracle_positive_ids),
        "oracle_positive_recall": intersection / len(oracle_positive_ids) if oracle_positive_ids else float("nan"),
        "jaccard_with_oracle_positive_pool": intersection / union if union else float("nan"),
        "trained_p0": trained_p0,
        "sequence_length": 512,
    }
    row.update(score_summary(selected, "full_color_score", "official_full_color"))
    row.update(score_summary(selected, "local_full_color_score", "local_full_color"))
    row.update(score_summary(selected, "pair_mid2_color_score", "pair_mid2_color"))
    return row


def write_raw_memmap(tokens: np.ndarray, seq_idx: np.ndarray, output_path: Path, *, dtype: np.dtype, chunk_rows: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if tokens.ndim != 2:
        raise ValueError(f"Expected 2D token array, got shape {tokens.shape}")
    if int(np.max(tokens)) > np.iinfo(dtype).max:
        raise ValueError(f"Token id max {int(np.max(tokens))} does not fit in {dtype}")
    sequence_length = tokens.shape[1]
    mmap = np.memmap(output_path, dtype=dtype, mode="w+", shape=(len(seq_idx) * sequence_length,))
    for start in range(0, len(seq_idx), chunk_rows):
        end = min(start + chunk_rows, len(seq_idx))
        chunk = tokens[seq_idx[start:end]].astype(dtype, copy=False).reshape(-1)
        mmap[start * sequence_length : end * sequence_length] = chunk
    mmap.flush()


def write_dataset(
    *,
    run_id: str,
    selected: pd.DataFrame,
    tokens: np.ndarray,
    output_dir: Path,
    token_dtype: np.dtype,
    chunk_rows: int,
    manifest: dict[str, object],
) -> None:
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    seq_idx = selected["seq_idx"].to_numpy(dtype=np.int64)
    write_raw_memmap(tokens, seq_idx, run_dir / "train_tokens.npy", dtype=token_dtype, chunk_rows=chunk_rows)
    selected.to_parquet(run_dir / "train_meta.parquet", index=False)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def overlap_rows(selections: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    items = list(selections.items())
    for i, (left_name, left) in enumerate(items):
        left_seq = set(int(x) for x in left["seq_idx"].to_numpy(dtype=np.int64))
        left_c4 = set(int(x) for x in left["c4_index"].to_numpy(dtype=np.int64))
        for right_name, right in items[i:]:
            right_seq = set(int(x) for x in right["seq_idx"].to_numpy(dtype=np.int64))
            right_c4 = set(int(x) for x in right["c4_index"].to_numpy(dtype=np.int64))
            seq_inter = len(left_seq & right_seq)
            seq_union = len(left_seq | right_seq)
            c4_inter = len(left_c4 & right_c4)
            c4_union = len(left_c4 | right_c4)
            rows.append(
                {
                    "left_run_id": left_name,
                    "right_run_id": right_name,
                    "seq_idx_intersection": seq_inter,
                    "seq_idx_jaccard": seq_inter / seq_union if seq_union else float("nan"),
                    "c4_index_intersection": c4_inter,
                    "c4_index_jaccard": c4_inter / c4_union if c4_union else float("nan"),
                }
            )
    return rows


def build_training_sets(args: argparse.Namespace) -> None:
    tokens = np.load(args.tokens, mmap_mode="r")
    scores = load_joined_scores(args.meta, args.full_scores, args.pair_mid2_scores)
    if tokens.shape != (len(scores), args.sequence_length):
        raise ValueError(f"Token shape {tokens.shape} does not match score rows {len(scores)} and seq len {args.sequence_length}")

    token_dtype = np.dtype(args.output_token_dtype)
    target_rows = int(args.target_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pool_validation = validate_pool_npz(Path(args.pool_dir), scores)
    for row in pool_validation:
        if not bool(row["c4_index_sets_match"]):
            raise ValueError(f"Pool sample validation failed: {row}")

    selections: dict[str, pd.DataFrame] = {}
    diagnostics: list[dict[str, object]] = []
    sensitivity: list[dict[str, object]] = []
    created_at = datetime.now(timezone.utc).isoformat()
    repo_commit = git_commit(ROOT)
    sensitivity_multipliers = parse_float_list(args.sensitivity_multipliers)

    for run_id, spec in P0_RUNS.items():
        positive_pool = str(spec["positive_pool"])
        oracle_positive = scores[scores["pool_name"] == positive_pool].sort_values("seq_idx").head(target_rows)
        oracle_positive_ids = set(int(x) for x in oracle_positive["seq_idx"].to_numpy(dtype=np.int64))

        if spec["selection_policy"] == "oracle_positive_pool":
            selected = select_oracle(scores, positive_pool, target_rows)
            candidate_count = None
        else:
            universe_pools = tuple(str(x) for x in spec["universe_pools"])
            for multiplier in sensitivity_multipliers:
                sens_selected, _, sens_candidate_count = select_cascade(
                    scores,
                    universe_pools=universe_pools,  # type: ignore[arg-type]
                    target_rows=target_rows,
                    multiplier=multiplier,
                )
                sensitivity.append(
                    diagnostic_row(
                        sens_selected,
                        run_id=run_id,
                        selection_policy=str(spec["selection_policy"]),
                        source_pools=tuple(str(x) for x in spec["source_pools"]),
                        universe_rows=int(scores["pool_name"].isin(universe_pools).sum()),
                        positive_pool=positive_pool,
                        target_rows=target_rows,
                        multiplier=multiplier,
                        candidate_count=sens_candidate_count,
                        trained_p0=math.isclose(multiplier, args.cascade_multiplier),
                        oracle_positive_ids=oracle_positive_ids,
                    )
                )
            selected, _, candidate_count = select_cascade(
                scores,
                universe_pools=tuple(str(x) for x in spec["universe_pools"]),  # type: ignore[arg-type]
                target_rows=target_rows,
                multiplier=float(args.cascade_multiplier),
            )

        source_pools = tuple(str(x) for x in spec["source_pools"])
        universe_rows = int(scores["pool_name"].isin(source_pools).sum())
        selected = selected.copy()
        selected.insert(0, "run_id", run_id)
        selected["selection_policy"] = str(spec["selection_policy"])
        selected["selection_source_pools"] = ",".join(source_pools)
        selected["cascade_multiplier"] = args.cascade_multiplier if spec["selection_policy"] != "oracle_positive_pool" else np.nan
        selections[run_id] = selected

        diag = diagnostic_row(
            selected,
            run_id=run_id,
            selection_policy=str(spec["selection_policy"]),
            source_pools=source_pools,
            universe_rows=universe_rows,
            positive_pool=positive_pool,
            target_rows=target_rows,
            multiplier=args.cascade_multiplier if spec["selection_policy"] != "oracle_positive_pool" else None,
            candidate_count=candidate_count,
            trained_p0=True,
            oracle_positive_ids=oracle_positive_ids,
        )
        diagnostics.append(diag)

        manifest = {
            "run_id": run_id,
            "source_universe": ",".join(source_pools),
            "source_pool_names": list(source_pools),
            "selection_policy": str(spec["selection_policy"]),
            "cascade_multiplier": args.cascade_multiplier if spec["selection_policy"] != "oracle_positive_pool" else None,
            "score_sign_convention": "paper_sign_conditional_books_loss_minus_prior_loss_lower_is_better",
            "target_unique_rows": target_rows,
            "actual_unique_rows": int(selected["seq_idx"].nunique()),
            "actual_rows": int(len(selected)),
            "sequence_length": args.sequence_length,
            "token_dtype": str(token_dtype),
            "tokenizer_id": args.tokenizer,
            "seed": args.seed,
            "git_commit": repo_commit,
            "created_timestamp": created_at,
            "input_artifact_paths": {
                "tokens": str(args.tokens),
                "meta": str(args.meta),
                "full_scores": str(args.full_scores),
                "pair_mid2_scores": str(args.pair_mid2_scores),
                "pool_dir": str(args.pool_dir),
            },
            "raw_memmap_note": "train_tokens.npy is an OLMo-compatible raw memmap despite the .npy suffix; do not load with np.load.",
        }
        manifest.update({key: value for key, value in diag.items() if key not in {"run_id", "source_pools"}})
        write_dataset(
            run_id=run_id,
            selected=selected,
            tokens=tokens,
            output_dir=output_dir,
            token_dtype=token_dtype,
            chunk_rows=args.copy_chunk_rows,
            manifest=manifest,
        )

    diagnostics_frame = pd.DataFrame(diagnostics)
    sensitivity_frame = pd.DataFrame(sensitivity)
    overlap_frame = pd.DataFrame(overlap_rows(selections))

    diagnostics_frame.to_csv(output_dir / "selection_diagnostics.csv", index=False)
    sensitivity_frame.to_csv(output_dir / "selection_sensitivity.csv", index=False)
    overlap_frame.to_csv(output_dir / "overlap_jaccard.csv", index=False)
    pd.DataFrame(pool_validation).to_json(output_dir / "pool_validation.json", orient="records", indent=2)

    comparison_manifest = {
        "created_timestamp": created_at,
        "git_commit": repo_commit,
        "target_rows_per_dataset": target_rows,
        "target_tokens_per_unique_dataset": int(target_rows * args.sequence_length),
        "training_target_tokens_per_run": int(args.training_target_tokens),
        "passes_over_unique_rows": float(args.training_target_tokens / (target_rows * args.sequence_length)),
        "cascade_multiplier": args.cascade_multiplier,
        "sensitivity_multipliers": list(sensitivity_multipliers),
        "run_ids": list(P0_RUNS.keys()),
        "output_dir": str(output_dir),
        "token_dtype": str(token_dtype),
        "raw_memmap_note": "Each train_tokens.npy is a raw OLMo memmap, not a standard NumPy .npy file.",
        "pool_validation": pool_validation,
    }
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(comparison_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"wrote datasets under {output_dir}")
    print(diagnostics_frame[["run_id", "selected_rows", "true_positive_count", "true_positive_rate", "oracle_positive_recall"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build P0 410M training sets from official score-pool rows.")
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--full-scores", type=Path, required=True)
    parser.add_argument("--pair-mid2-scores", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--cascade-multiplier", type=float, default=1.5)
    parser.add_argument("--sensitivity-multipliers", default="1.25,1.5,2.0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--target-rows", type=int, default=100_000)
    parser.add_argument("--training-target-tokens", type=int, default=100_000_000)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--tokenizer", default="allenai/eleuther-ai-gpt-neox-20b-pii-special")
    parser.add_argument("--output-token-dtype", default="uint16")
    parser.add_argument("--copy-chunk-rows", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_training_sets(args)


if __name__ == "__main__":
    main()
