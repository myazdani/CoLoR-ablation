#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ablation import apply_paired_ablation, variant_layer_indices
from src.config import ensure_parent, find_variant, load_config
from src.model_loading import load_causal_lm
from src.scoring import score_pair


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _pool_hash(pool_path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(pool_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _annotate_scores(
    scores,
    *,
    target: dict,
    score_id: str,
    ablation_variant: str,
    removed_layers: tuple[int, ...],
    git_commit: str,
    pool_sha256: str,
    stats: dict[str, float],
    shard_start: int,
    shard_end: int,
):
    scores["seq_idx"] = scores["seq_idx"] + shard_start
    scores["target"] = target["name"]
    scores["variant"] = score_id
    scores["ablation_variant"] = ablation_variant
    scores["removed_layers"] = json.dumps(list(removed_layers))
    scores["git_commit"] = git_commit
    scores["pool_sha256"] = pool_sha256
    scores["cond_checkpoint"] = str(target["cond_checkpoint"])
    scores["marg_checkpoint"] = str(target["marg_checkpoint"])
    scores["shard_start"] = shard_start
    scores["shard_end"] = shard_end
    for key, value in stats.items():
        scores[key] = value
    return scores


def _valid_existing_shard(path: Path, *, shard_start: int, shard_end: int) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path, columns=["seq_idx"])
    except Exception:
        return False
    expected_len = shard_end - shard_start
    if len(frame) != expected_len:
        return False
    return int(frame["seq_idx"].min()) == shard_start and int(frame["seq_idx"].max()) == shard_end - 1


def _write_parquet_atomic(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def _combine_shards(shard_paths: list[Path], output_path: Path) -> dict[str, float]:
    frames = [pd.read_parquet(path) for path in shard_paths]
    combined = pd.concat(frames, ignore_index=True).sort_values("seq_idx").reset_index(drop=True)
    if {"shard_start", "tokens_scored", "elapsed_seconds"}.issubset(combined.columns):
        shard_stats = combined.drop_duplicates(["shard_start", "shard_end"])
        elapsed = float(shard_stats["elapsed_seconds"].sum())
        tokens = int(shard_stats["tokens_scored"].sum())
        tps = tokens / elapsed if elapsed > 0 else float("nan")
        combined["elapsed_seconds"] = elapsed
        combined["tokens_scored"] = tokens
        combined["tokens_per_second"] = tps
        aggregate = {
            "elapsed_seconds": elapsed,
            "tokens_scored": tokens,
            "tokens_per_second": tps,
        }
    else:
        aggregate = {
            "elapsed_seconds": float("nan"),
            "tokens_scored": float("nan"),
            "tokens_per_second": float("nan"),
        }
    _write_parquet_atomic(combined, output_path)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one model variant on the frozen pool.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--score-id",
        default=None,
        help="Output/metadata id. Use this for repeated runs, e.g. --variant full --score-id full_rescore.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        help="Rows per resumable score shard. Defaults to scoring.shard_size from config.",
    )
    parser.add_argument(
        "--no-shards",
        action="store_true",
        help="Disable shard resumability and write only the final parquet.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    find_variant(config, args.variant)
    target = config["target"]
    scoring_cfg = config["scoring"]
    total_layers = int(scoring_cfg["total_layers"])
    removed_layers = variant_layer_indices(args.variant, total_layers=total_layers)
    score_id = args.score_id or args.variant

    pool = np.load(target["pool_tokens"])
    cond = load_causal_lm(
        target["cond_checkpoint"],
        paper_code_path=config.get("paths", {}).get("paper_code"),
        dtype=scoring_cfg.get("dtype", "bf16"),
    )
    marg = load_causal_lm(
        target["marg_checkpoint"],
        paper_code_path=config.get("paths", {}).get("paper_code"),
        dtype=scoring_cfg.get("dtype", "bf16"),
    )

    if removed_layers:
        cond_record, marg_record = apply_paired_ablation(
            cond,
            marg,
            removed_layers,
            total_layers=total_layers,
        )
        if cond_record.removed_layers != marg_record.removed_layers:
            raise AssertionError("Paired ablation metadata mismatch")

    output = args.output or Path(config["paths"]["results_dir"]) / f"scores_{score_id}.parquet"
    output_path = ensure_parent(output)
    git_commit = _git_commit()
    pool_sha256 = _pool_hash(target["pool_tokens"])
    shard_size = args.shard_size
    if shard_size is None:
        shard_size = int(scoring_cfg.get("shard_size", 0) or 0)

    if args.no_shards or shard_size <= 0 or shard_size >= len(pool):
        scores, stats = score_pair(
            cond,
            marg,
            pool,
            batch_size=int(scoring_cfg["batch_size"]),
            device=scoring_cfg.get("device", "auto"),
            dtype=scoring_cfg.get("dtype", "bf16"),
        )
        scores = _annotate_scores(
            scores,
            target=target,
            score_id=score_id,
            ablation_variant=args.variant,
            removed_layers=removed_layers,
            git_commit=git_commit,
            pool_sha256=pool_sha256,
            stats=stats,
            shard_start=0,
            shard_end=len(pool),
        )
        _write_parquet_atomic(scores, output_path)
    else:
        shard_dir = output_path.with_suffix("").with_name(output_path.with_suffix("").name + "_shards")
        shard_paths: list[Path] = []
        for shard_start in range(0, len(pool), shard_size):
            shard_end = min(shard_start + shard_size, len(pool))
            shard_path = shard_dir / f"part_{shard_start:08d}_{shard_end:08d}.parquet"
            shard_paths.append(shard_path)
            if _valid_existing_shard(shard_path, shard_start=shard_start, shard_end=shard_end):
                print(f"Skipping completed shard {shard_path}")
                continue
            print(f"Scoring shard {shard_start}:{shard_end} -> {shard_path}")
            shard_scores, shard_stats = score_pair(
                cond,
                marg,
                pool[shard_start:shard_end],
                batch_size=int(scoring_cfg["batch_size"]),
                device=scoring_cfg.get("device", "auto"),
                dtype=scoring_cfg.get("dtype", "bf16"),
            )
            shard_scores = _annotate_scores(
                shard_scores,
                target=target,
                score_id=score_id,
                ablation_variant=args.variant,
                removed_layers=removed_layers,
                git_commit=git_commit,
                pool_sha256=pool_sha256,
                stats=shard_stats,
                shard_start=shard_start,
                shard_end=shard_end,
            )
            _write_parquet_atomic(shard_scores, shard_path)

        stats = _combine_shards(shard_paths, output_path)
    print(f"Wrote {output_path}")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
