#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ensure_parent, load_config
from src.score_pool_robustness import load_pool_samples


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int


@dataclass(frozen=True)
class NeededFile:
    path: str
    size: int
    chunk_start: int
    chunk_end: int
    needed_rows: int


def log(message: str) -> None:
    print(message, flush=True)


def dtype_from_name(name: str) -> np.dtype:
    mapping = {
        "uint16": np.dtype(np.uint16),
        "uint32": np.dtype(np.uint32),
        "int32": np.dtype(np.int32),
        "int64": np.dtype(np.int64),
    }
    if name not in mapping:
        raise ValueError(f"Unsupported token memmap dtype '{name}'")
    return mapping[name]


def api_tree_url(repo_id: str, path: str, *, recursive: bool) -> str:
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    quoted_path = urllib.parse.quote(path.strip("/"), safe="/")
    query = "?recursive=1&expand=1" if recursive else "?expand=1"
    return f"https://huggingface.co/api/models/{quoted_repo}/tree/main/{quoted_path}{query}"


def list_remote_files(repo_id: str, prefix: str, *, recursive: bool = True) -> list[RemoteFile]:
    request = urllib.request.Request(api_tree_url(repo_id, prefix, recursive=recursive))
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.load(response)
    files: list[RemoteFile] = []
    dirs: list[str] = []
    for item in data:
        if item.get("type") == "directory":
            path = item.get("path")
            if recursive and isinstance(path, str):
                dirs.append(path)
            continue
        if item.get("type") != "file":
            continue
        path = item.get("path")
        size = item.get("size")
        if isinstance(path, str) and isinstance(size, int):
            files.append(RemoteFile(path=path, size=size))
    for directory in dirs:
        files.extend(list_remote_files(repo_id, directory, recursive=True))
    return files


def token_files_from_remote(files: list[RemoteFile]) -> list[RemoteFile]:
    token_files = [
        remote
        for remote in files
        if remote.path.endswith(".npy")
        and not remote.path.endswith("mmap_index.npy")
        and not remote.path.endswith("index.npy")
        and "label_mask" not in remote.path
    ]
    token_files.sort(key=lambda remote: remote.path)
    if not token_files:
        raise RuntimeError("No candidate token .npy files found under remote prefix")
    return token_files


def build_needed_file_plan(
    *,
    c4_indices: np.ndarray,
    token_files: list[RemoteFile],
    dtype: np.dtype,
    sequence_length: int,
) -> tuple[list[NeededFile], int]:
    sorted_indices = np.sort(np.unique(c4_indices.astype(np.int64)))
    needed: list[NeededFile] = []
    chunk_cursor = 0
    query_cursor = 0
    for remote in token_files:
        values = remote.size // dtype.itemsize
        if remote.size % dtype.itemsize:
            raise ValueError(f"Remote file size is not divisible by {dtype}: {remote.path}")
        chunks = values // sequence_length
        if values % sequence_length:
            log(f"Warning: ignoring trailing {values % sequence_length} tokens in {remote.path}")
        chunk_start = chunk_cursor
        chunk_end = chunk_cursor + chunks
        left = np.searchsorted(sorted_indices, chunk_start, side="left", sorter=None)
        if query_cursor > left:
            left = query_cursor
        right = np.searchsorted(sorted_indices, chunk_end, side="left", sorter=None)
        if right > left:
            needed.append(
                NeededFile(
                    path=remote.path,
                    size=remote.size,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    needed_rows=int(right - left),
                )
            )
            query_cursor = right
        chunk_cursor = chunk_end
    if len(sorted_indices) and int(sorted_indices[-1]) >= chunk_cursor:
        log(
            "Warning: highest requested c4_index "
            f"{int(sorted_indices[-1]):,} exceeds inferred remote token chunk count "
            f"{chunk_cursor:,}"
        )
    return needed, chunk_cursor


def write_plan(
    *,
    output_path: Path,
    repo_id: str,
    remote_prefix: str,
    dtype: np.dtype,
    sequence_length: int,
    total_remote_files: int,
    total_remote_bytes: int,
    total_remote_chunks: int,
    sample_rows: int,
    unique_c4_indices: int,
    max_c4_index: int,
    needed: list[NeededFile],
) -> dict[str, Any]:
    total_needed_bytes = sum(item.size for item in needed)
    index_coverage_ok = max_c4_index < total_remote_chunks if unique_c4_indices else True
    plan = {
        "repo_id": repo_id,
        "remote_prefix": remote_prefix,
        "memmap_dtype": str(dtype),
        "sequence_length": sequence_length,
        "sample_rows": sample_rows,
        "unique_c4_indices": unique_c4_indices,
        "max_c4_index": max_c4_index,
        "total_remote_files": total_remote_files,
        "total_remote_bytes": total_remote_bytes,
        "total_remote_chunks": total_remote_chunks,
        "index_coverage_ok": index_coverage_ok,
        "needed_files": [asdict(item) for item in needed],
        "needed_file_count": len(needed),
        "total_needed_bytes": total_needed_bytes,
        "total_needed_gib": total_needed_bytes / 1024**3,
        "index_basis": "c4_index",
        "notes": [
            "c4_index is the upstream tokenized-C4 dataset index emitted by scoring.",
            "row_position is local score-row order and is not used for token shard mapping.",
            "If index_coverage_ok is false, the visible HF full_data/c4 tree cannot recover these samples exactly.",
        ],
    }
    output_path = ensure_parent(output_path)
    output_path.write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def download_needed_files(
    *,
    repo_id: str,
    needed: list[NeededFile],
    cache_dir: Path,
) -> dict[str, Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub before downloading token shards") from exc

    resolved: dict[str, Path] = {}
    for item in needed:
        log(f"downloading {item.path} ({item.size / 1024**3:.2f} GiB)")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=item.path,
            repo_type="model",
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
        resolved[item.path] = Path(path)
    return resolved


def recover_tokens(
    *,
    sample_meta: pd.DataFrame,
    needed: list[NeededFile],
    local_paths: Mapping[str, Path],
    output_tokens: Path,
    output_meta: Path,
    dtype: np.dtype,
    sequence_length: int,
) -> None:
    tokens = np.lib.format.open_memmap(
        output_tokens,
        mode="w+",
        dtype=np.int32,
        shape=(len(sample_meta), sequence_length),
    )
    c4_indices = sample_meta["c4_index"].to_numpy(dtype=np.int64)
    recovered = np.zeros(len(sample_meta), dtype=bool)
    for item in needed:
        local_path = local_paths[item.path]
        values = local_path.stat().st_size // dtype.itemsize
        mmap = np.memmap(local_path, dtype=dtype, mode="r", shape=(values,))
        mask = (c4_indices >= item.chunk_start) & (c4_indices < item.chunk_end)
        positions = np.flatnonzero(mask)
        local_chunks = c4_indices[positions] - item.chunk_start
        for output_pos, local_chunk in zip(positions, local_chunks):
            start = int(local_chunk) * sequence_length
            end = start + sequence_length
            tokens[output_pos] = np.asarray(mmap[start:end], dtype=np.int32)
        recovered[positions] = True
    tokens.flush()
    if not recovered.all():
        missing = int((~recovered).sum())
        raise RuntimeError(f"Failed to recover {missing} sampled token rows")
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    sample_meta.to_parquet(output_meta, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight and optionally recover exact token sequences for score-pool samples."
    )
    parser.add_argument("--config", default="configs/score_pool_robustness.yaml")
    parser.add_argument("--download", action="store_true", help="Download required shards and recover tokens.")
    parser.add_argument(
        "--allow-large-download",
        action="store_true",
        help="Allow downloads larger than max_download_gb_without_confirmation.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    token_cfg = config["token_recovery"]
    sample_meta = load_pool_samples(paths["existing_pool_analysis_dir"])
    dtype = dtype_from_name(str(token_cfg.get("memmap_dtype", "uint16")))
    sequence_length = int(token_cfg.get("sequence_length", config["target"]["sequence_length"]))
    repo_id = token_cfg["hf_repo"]
    remote_prefix = token_cfg["remote_prefix"]

    log(f"loading remote tree: {repo_id}/{remote_prefix}")
    remote_files = list_remote_files(repo_id, remote_prefix, recursive=True)
    token_files = token_files_from_remote(remote_files)
    needed, total_chunks = build_needed_file_plan(
        c4_indices=sample_meta["c4_index"].to_numpy(dtype=np.int64),
        token_files=token_files,
        dtype=dtype,
        sequence_length=sequence_length,
    )
    c4_values = sample_meta["c4_index"].to_numpy(dtype=np.int64)
    plan = write_plan(
        output_path=Path(token_cfg["recovery_plan"]),
        repo_id=repo_id,
        remote_prefix=remote_prefix,
        dtype=dtype,
        sequence_length=sequence_length,
        total_remote_files=len(token_files),
        total_remote_bytes=sum(remote.size for remote in token_files),
        total_remote_chunks=total_chunks,
        sample_rows=len(sample_meta),
        unique_c4_indices=int(sample_meta["c4_index"].nunique()),
        max_c4_index=int(c4_values.max()) if len(c4_values) else -1,
        needed=needed,
    )

    log(f"sample rows: {plan['sample_rows']:,}")
    log(f"unique c4 indices: {plan['unique_c4_indices']:,}")
    log(f"remote token files: {plan['total_remote_files']:,}")
    log(f"remote token bytes: {plan['total_remote_bytes'] / 1024**3:.2f} GiB")
    log(f"remote token chunks inferred: {plan['total_remote_chunks']:,}")
    log(f"max requested c4_index: {plan['max_c4_index']:,}")
    log(f"index coverage ok: {plan['index_coverage_ok']}")
    log(f"needed token files: {plan['needed_file_count']:,}")
    log(f"needed token bytes: {plan['total_needed_gib']:.2f} GiB")
    log(f"wrote plan: {token_cfg['recovery_plan']}")

    max_gb = float(token_cfg.get("max_download_gb_without_confirmation", 25))
    if not args.download:
        log("preflight only; rerun with --download to recover tokens")
        return
    if not plan["index_coverage_ok"]:
        raise SystemExit(
            "Exact recovery cannot proceed: visible remote token chunks do not cover "
            "the sampled c4_index range. Use the fallback_full_rescore_pool workflow "
            "or locate the missing tokenized C4 shards."
        )
    if plan["total_needed_gib"] > max_gb and not args.allow_large_download:
        raise SystemExit(
            f"Required token shard download is {plan['total_needed_gib']:.2f} GiB, "
            f"above configured cap {max_gb:.2f} GiB. Rerun with --allow-large-download "
            "only after confirming disk/Colab Drive capacity."
        )

    cache_dir = Path(token_cfg["local_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_paths = download_needed_files(repo_id=repo_id, needed=needed, cache_dir=cache_dir)
    recover_tokens(
        sample_meta=sample_meta,
        needed=needed,
        local_paths=local_paths,
        output_tokens=Path(token_cfg["recovered_tokens"]),
        output_meta=Path(token_cfg["recovered_meta"]),
        dtype=dtype,
        sequence_length=sequence_length,
    )
    log(f"wrote tokens: {token_cfg['recovered_tokens']}")
    log(f"wrote metadata: {token_cfg['recovered_meta']}")


if __name__ == "__main__":
    main()
