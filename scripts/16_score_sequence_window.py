#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ensure_parent, load_config
from src.model_loading import load_causal_lm
from src.scoring import extract_token_window, score_pair


def log(message: str) -> None:
    print(message, flush=True)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: object) -> str:
    data = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def checkpoint_sha256(checkpoint_path: str | Path) -> str:
    path = Path(checkpoint_path)
    files = [path / "config.json", path / "pytorch_model.bin"]
    payload = {file.name: file_sha256(file) for file in files if file.exists()}
    return json_hash(payload)


def all_window_specs(config: dict[str, Any], *, include_optional: bool = False) -> list[dict[str, Any]]:
    specs = list(config.get("sequence_windows", []))
    if include_optional:
        specs.extend(config.get("optional_sequence_windows", []))
    return specs


def get_window_spec(config: dict[str, Any], window_id: str, *, include_optional: bool = False) -> dict[str, Any]:
    for raw in all_window_specs(config, include_optional=include_optional):
        if raw.get("id") == window_id:
            return raw
    raise KeyError(f"Unknown sequence window '{window_id}'")


def window_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if "indices" in spec:
        start, end = spec["indices"]
        kwargs["indices"] = (int(start), int(end))
    if "spans" in spec:
        kwargs["spans"] = tuple((int(start), int(end)) for start, end in spec["spans"])
    if "stride" in spec:
        kwargs["stride"] = int(spec["stride"])
    return kwargs


def effective_sequence_length(pool: np.ndarray, spec: dict[str, Any]) -> int:
    sample = extract_token_window(pool[:1], **window_kwargs(spec))
    return int(sample.shape[1])


def valid_existing_shard(path: Path, *, shard_start: int, shard_end: int) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path, columns=["seq_idx"])
    except Exception:
        return False
    if len(frame) != shard_end - shard_start:
        return False
    return int(frame["seq_idx"].min()) == shard_start and int(frame["seq_idx"].max()) == shard_end - 1


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def annotate_scores(
    scores: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    window_spec: dict[str, Any],
    effective_length: int,
    git_sha: str,
    config_sha: str,
    token_pool_sha: str,
    cond_checkpoint: str,
    marg_checkpoint: str,
    cond_checkpoint_sha: str,
    marg_checkpoint_sha: str,
    torch_version: str,
    cuda_version: str,
    cuda_device_name: str,
    batch_size: int,
    dtype: str,
    device: str,
    stats: dict[str, float],
    shard_start: int,
    shard_end: int,
) -> pd.DataFrame:
    scores = scores.rename(
        columns={
            "nll_cond": "ablated_conditional_books_score",
            "nll_marg": "ablated_prior_score",
            "color": "ablated_color_score",
        }
    ).copy()
    scores["seq_idx"] = scores["seq_idx"] + shard_start
    merged = meta.merge(scores, on="seq_idx", how="inner", validate="one_to_one")
    if len(merged) != len(scores):
        raise ValueError("Score shard did not align one-to-one with metadata")
    merged["variant_id"] = str(window_spec["id"])
    merged["variant_family"] = "sequence_window"
    merged["sequence_window_id"] = str(window_spec["id"])
    merged["sequence_window_label"] = str(window_spec.get("label", window_spec["id"]))
    merged["sequence_window_spec"] = json.dumps(window_spec, sort_keys=True)
    merged["effective_sequence_length"] = int(effective_length)
    merged["cond_removed_layers"] = "[]"
    merged["marg_removed_layers"] = "[]"
    merged["cond_kept_layers"] = 12
    merged["marg_kept_layers"] = 12
    merged["cond_block_path"] = ""
    merged["marg_block_path"] = ""
    merged["git_commit"] = git_sha
    merged["config_sha256"] = config_sha
    merged["token_pool_sha256"] = token_pool_sha
    merged["cond_checkpoint"] = cond_checkpoint
    merged["marg_checkpoint"] = marg_checkpoint
    merged["cond_checkpoint_sha256"] = cond_checkpoint_sha
    merged["marg_checkpoint_sha256"] = marg_checkpoint_sha
    merged["torch_version"] = torch_version
    merged["cuda_version"] = cuda_version
    merged["cuda_device_name"] = cuda_device_name
    merged["batch_size"] = int(batch_size)
    merged["scoring_dtype"] = str(dtype)
    merged["scoring_device"] = str(device)
    merged["shard_start"] = shard_start
    merged["shard_end"] = shard_end
    for key, value in stats.items():
        merged[key] = value
    return merged


def combine_shards(shard_paths: list[Path], output_path: Path) -> dict[str, float]:
    frames = [pd.read_parquet(path) for path in shard_paths]
    combined = pd.concat(frames, ignore_index=True).sort_values("seq_idx").reset_index(drop=True)
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
    write_parquet_atomic(combined, output_path)
    return aggregate


def score_window(config: dict[str, Any], window_id: str, *, force: bool = False, shard_size: int | None = None, include_optional: bool = False) -> Path:
    scoring_cfg = config["scoring"]
    token_cfg = config["token_recovery"]
    target = config["target"]
    paths = config["paths"]
    spec = get_window_spec(config, window_id, include_optional=include_optional)

    tokens_path = Path(token_cfg["recovered_tokens"])
    meta_path = Path(token_cfg["recovered_meta"])
    if not tokens_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Recovered official 500K token pool is missing. Use the official 500K runbook "
            "to recover or copy data/score_pool_tokens_official_500k.npy and metadata first."
        )

    pool = np.load(tokens_path, mmap_mode="r")
    meta = pd.read_parquet(meta_path)
    if len(meta) != len(pool):
        raise ValueError(f"Metadata rows {len(meta)} do not match token rows {len(pool)}")
    effective_length = effective_sequence_length(pool, spec)

    log(f"loading full models for sequence window={window_id} effective_length={effective_length}")
    cond = load_causal_lm(
        target["cond_checkpoint"],
        paper_code_path=paths.get("paper_code"),
        dtype=scoring_cfg.get("dtype", "bf16"),
    )
    marg = load_causal_lm(
        target["marg_checkpoint"],
        paper_code_path=paths.get("paper_code"),
        dtype=scoring_cfg.get("dtype", "bf16"),
    )

    git_sha = git_commit()
    config_sha = json_hash(config)
    token_pool_sha = file_sha256(tokens_path)
    cond_checkpoint = str(target["cond_checkpoint"])
    marg_checkpoint = str(target["marg_checkpoint"])
    cond_checkpoint_sha = checkpoint_sha256(cond_checkpoint)
    marg_checkpoint_sha = checkpoint_sha256(marg_checkpoint)
    torch_version = torch.__version__
    cuda_version = torch.version.cuda or ""
    cuda_device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""

    output_path = ensure_parent(Path(paths["output_dir"]) / f"scores_{window_id}.parquet")
    if shard_size is None:
        shard_size = int(scoring_cfg.get("shard_size", 0) or 0)
    if shard_size <= 0:
        shard_size = len(pool)

    shard_dir = output_path.with_suffix("").with_name(output_path.with_suffix("").name + "_shards")
    shard_paths: list[Path] = []
    for shard_start in range(0, len(pool), shard_size):
        shard_end = min(shard_start + shard_size, len(pool))
        shard_path = shard_dir / f"part_{shard_start:08d}_{shard_end:08d}.parquet"
        shard_paths.append(shard_path)
        if not force and valid_existing_shard(shard_path, shard_start=shard_start, shard_end=shard_end):
            log(f"skipping completed shard {shard_path}")
            continue
        log(f"scoring shard {shard_start:,}:{shard_end:,}")
        shard_pool = extract_token_window(pool[shard_start:shard_end], **window_kwargs(spec))
        shard_scores, shard_stats = score_pair(
            cond,
            marg,
            shard_pool,
            batch_size=int(scoring_cfg["batch_size"]),
            device=scoring_cfg.get("device", "auto"),
            dtype=scoring_cfg.get("dtype", "bf16"),
        )
        annotated = annotate_scores(
            shard_scores,
            meta.iloc[shard_start:shard_end].copy(),
            window_spec=spec,
            effective_length=effective_length,
            git_sha=git_sha,
            config_sha=config_sha,
            token_pool_sha=token_pool_sha,
            cond_checkpoint=cond_checkpoint,
            marg_checkpoint=marg_checkpoint,
            cond_checkpoint_sha=cond_checkpoint_sha,
            marg_checkpoint_sha=marg_checkpoint_sha,
            torch_version=torch_version,
            cuda_version=cuda_version,
            cuda_device_name=cuda_device_name,
            batch_size=int(scoring_cfg["batch_size"]),
            dtype=str(scoring_cfg.get("dtype", "bf16")),
            device=str(scoring_cfg.get("device", "auto")),
            stats=shard_stats,
            shard_start=shard_start,
            shard_end=shard_end,
        )
        write_parquet_atomic(annotated, shard_path)

    stats = combine_shards(shard_paths, output_path)
    log(f"wrote {output_path}")
    log(json.dumps(stats, indent=2, sort_keys=True))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Score full Books CoLoR models on a sequence window.")
    parser.add_argument("--config", default="configs/sequence_length_score_pool.yaml")
    parser.add_argument("--window", default=None, help="Window id to score.")
    parser.add_argument("--all-primary", action="store_true", help="Score all primary windows.")
    parser.add_argument("--include-optional", action="store_true", help="Include optional stride windows.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--shard-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.all_primary:
        window_ids = [str(spec["id"]) for spec in all_window_specs(config, include_optional=args.include_optional)]
    elif args.window:
        window_ids = [args.window]
    else:
        raise SystemExit("Use --window WINDOW_ID or --all-primary")

    for window_id in window_ids:
        score_window(
            config,
            window_id,
            force=args.force,
            shard_size=args.shard_size,
            include_optional=args.include_optional,
        )


if __name__ == "__main__":
    main()
