#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
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


@dataclass(frozen=True)
class DocLocation:
    c4_index: int
    token_path: str
    token_size: int
    sidecar_path: str
    token_start: int
    token_end: int
    source_path: str
    source_row: int


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
    files: list[RemoteFile] = []
    dirs: list[str] = []
    url: str | None = api_tree_url(repo_id, prefix, recursive=recursive)
    while url:
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.load(response)
            link_header = response.headers.get("Link") or ""
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
        match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
        url = match.group(1) if match else None
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


def sidecar_files_from_remote(files: list[RemoteFile]) -> list[RemoteFile]:
    sidecars = [remote for remote in files if remote.path.endswith(".csv.gz")]
    sidecars.sort(key=lambda remote: remote.path)
    return sidecars


def pair_token_and_sidecar_files(
    token_files: list[RemoteFile],
    sidecar_files: list[RemoteFile],
) -> list[tuple[RemoteFile, RemoteFile]]:
    token_by_stem = {Path(remote.path).name.removesuffix(".npy"): remote for remote in token_files}
    pairs: list[tuple[RemoteFile, RemoteFile]] = []
    for sidecar in sidecar_files:
        stem = Path(sidecar.path).name.removesuffix(".csv.gz")
        token = token_by_stem.get(stem)
        if token is not None:
            pairs.append((token, sidecar))
    if not pairs:
        raise RuntimeError("No matching .npy/.csv.gz token sidecar pairs found")
    return pairs


def hf_resolve_url(repo_id: str, path: str) -> str:
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    quoted_path = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/{quoted_repo}/resolve/main/{quoted_path}"


def read_location_cache(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def write_location_cache(frame: pd.DataFrame, path: Path) -> None:
    path = ensure_parent(path)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)


def iter_sidecar_locations(
    *,
    repo_id: str,
    token_file: RemoteFile,
    sidecar_file: RemoteFile,
    requested: set[int],
) -> list[DocLocation]:
    locations: list[DocLocation] = []
    request = urllib.request.Request(hf_resolve_url(repo_id, sidecar_file.path))
    with urllib.request.urlopen(request, timeout=120) as response:
        with gzip.GzipFile(fileobj=response) as gzip_file:
            text = (line.decode("utf-8") for line in gzip_file)
            reader = csv.reader(text)
            for row in reader:
                if len(row) < 5:
                    continue
                try:
                    c4_index = int(row[4])
                except ValueError:
                    continue
                if c4_index not in requested:
                    continue
                locations.append(
                    DocLocation(
                        c4_index=c4_index,
                        token_path=token_file.path,
                        token_size=token_file.size,
                        sidecar_path=sidecar_file.path,
                        token_start=int(row[0]),
                        token_end=int(row[1]),
                        source_path=row[3],
                        source_row=c4_index,
                    )
                )
    return locations


def build_doc_location_plan(
    *,
    repo_id: str,
    c4_indices: np.ndarray,
    token_files: list[RemoteFile],
    sidecar_files: list[RemoteFile],
    locations_path: Path,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    requested = set(int(value) for value in np.unique(c4_indices.astype(np.int64)))
    if locations_path.exists() and not force_rebuild:
        frame = read_location_cache(locations_path)
        found = set(int(value) for value in frame["c4_index"].to_numpy(dtype=np.int64))
        if requested.issubset(found):
            return frame[frame["c4_index"].isin(requested)].copy()
        log(
            f"Cached sidecar locations cover {len(found):,}/{len(requested):,} requested ids; rebuilding."
        )

    pairs = pair_token_and_sidecar_files(token_files, sidecar_files)
    found: dict[int, DocLocation] = {}
    for token_file, sidecar_file in pairs:
        remaining = requested - set(found)
        if not remaining:
            break
        log(
            f"scanning {sidecar_file.path} ({sidecar_file.size / 1024**2:.1f} MiB); "
            f"found {len(found):,}/{len(requested):,}"
        )
        for location in iter_sidecar_locations(
            repo_id=repo_id,
            token_file=token_file,
            sidecar_file=sidecar_file,
            requested=remaining,
        ):
            found[location.c4_index] = location

    frame = pd.DataFrame([asdict(location) for location in found.values()])
    write_location_cache(frame, locations_path)
    return frame


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


def write_doc_plan(
    *,
    output_path: Path,
    repo_id: str,
    remote_prefix: str,
    dtype: np.dtype,
    sequence_length: int,
    sample_rows: int,
    unique_c4_indices: int,
    max_c4_index: int,
    token_files: list[RemoteFile],
    sidecar_files: list[RemoteFile],
    locations: pd.DataFrame,
) -> dict[str, Any]:
    found_unique = int(locations["c4_index"].nunique()) if len(locations) else 0
    needed_paths = set(locations["token_path"].astype(str)) if len(locations) else set()
    token_by_path = {remote.path: remote for remote in token_files}
    sidecar_by_path = {remote.path: remote for remote in sidecar_files}
    needed_token_files = [token_by_path[path] for path in sorted(needed_paths)]
    needed_sidecar_paths = set(locations["sidecar_path"].astype(str)) if len(locations) else set()
    needed_sidecars = [sidecar_by_path[path] for path in sorted(needed_sidecar_paths)]
    plan = {
        "repo_id": repo_id,
        "remote_prefix": remote_prefix,
        "memmap_dtype": str(dtype),
        "sequence_length": sequence_length,
        "sample_rows": sample_rows,
        "unique_c4_indices": unique_c4_indices,
        "max_c4_index": max_c4_index,
        "total_remote_token_files": len(token_files),
        "total_remote_token_bytes": sum(remote.size for remote in token_files),
        "total_remote_sidecar_files": len(sidecar_files),
        "total_remote_sidecar_bytes": sum(remote.size for remote in sidecar_files),
        "found_unique_c4_indices": found_unique,
        "index_coverage_ok": found_unique == unique_c4_indices,
        "needed_files": [
            {
                "path": remote.path,
                "size": remote.size,
                "needed_rows": int((locations["token_path"] == remote.path).sum()),
            }
            for remote in needed_token_files
        ],
        "needed_file_count": len(needed_token_files),
        "total_needed_bytes": sum(remote.size for remote in needed_token_files),
        "total_needed_gib": sum(remote.size for remote in needed_token_files) / 1024**3,
        "needed_sidecar_files": [
            {
                "path": remote.path,
                "size": remote.size,
                "matched_rows": int((locations["sidecar_path"] == remote.path).sum()),
            }
            for remote in needed_sidecars
        ],
        "needed_sidecar_file_count": len(needed_sidecars),
        "total_needed_sidecar_bytes": sum(remote.size for remote in needed_sidecars),
        "total_needed_sidecar_gib": sum(remote.size for remote in needed_sidecars) / 1024**3,
        "index_basis": "c4_index_csv_sidecar_doc_id",
        "notes": [
            "c4_index values match document ids in the full_data/c4 CSV sidecars.",
            "Token recovery reads the document token span from the matching raw token stream.",
            "Documents longer than sequence_length are right-truncated; shorter documents are right-padded.",
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


def download_one_needed_file(*, repo_id: str, item: NeededFile, cache_dir: Path) -> Path:
    return download_needed_files(repo_id=repo_id, needed=[item], cache_dir=cache_dir)[item.path]


def download_needed_token_files(
    *,
    repo_id: str,
    needed_paths: list[str],
    token_files: list[RemoteFile],
    cache_dir: Path,
) -> dict[str, Path]:
    token_by_path = {remote.path: remote for remote in token_files}
    needed = [
        NeededFile(
            path=path,
            size=token_by_path[path].size,
            chunk_start=0,
            chunk_end=0,
            needed_rows=0,
        )
        for path in needed_paths
    ]
    return download_needed_files(repo_id=repo_id, needed=needed, cache_dir=cache_dir)


def remove_cached_file(path: Path, *, cache_dir: Path) -> None:
    cache_dir = cache_dir.resolve()
    resolved_path = path.resolve()
    if resolved_path != cache_dir and cache_dir not in resolved_path.parents:
        log(f"not deleting {path}; it is outside cache_dir={cache_dir}")
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return

    parent = path.parent
    while parent != cache_dir and cache_dir in parent.resolve().parents:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


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


def recover_tokens_streaming(
    *,
    sample_meta: pd.DataFrame,
    needed: list[NeededFile],
    repo_id: str,
    cache_dir: Path,
    output_tokens: Path,
    output_meta: Path,
    dtype: np.dtype,
    sequence_length: int,
    delete_shards_after_recovery: bool,
) -> None:
    tokens = np.lib.format.open_memmap(
        output_tokens,
        mode="w+",
        dtype=np.int32,
        shape=(len(sample_meta), sequence_length),
    )
    c4_indices = sample_meta["c4_index"].to_numpy(dtype=np.int64)
    recovered = np.zeros(len(sample_meta), dtype=bool)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for file_idx, item in enumerate(needed, start=1):
        local_path = download_one_needed_file(repo_id=repo_id, item=item, cache_dir=cache_dir)
        try:
            values = local_path.stat().st_size // dtype.itemsize
            mmap = np.memmap(local_path, dtype=dtype, mode="r", shape=(values,))
            mask = (c4_indices >= item.chunk_start) & (c4_indices < item.chunk_end)
            positions = np.flatnonzero(mask)
            local_chunks = c4_indices[positions] - item.chunk_start
            log(
                f"recovering {len(positions):,} rows from {item.path} "
                f"({file_idx:,}/{len(needed):,})"
            )
            for output_pos, local_chunk in zip(positions, local_chunks):
                start = int(local_chunk) * sequence_length
                end = start + sequence_length
                tokens[output_pos] = np.asarray(mmap[start:end], dtype=np.int32)
            recovered[positions] = True
            tokens.flush()
        finally:
            if delete_shards_after_recovery:
                remove_cached_file(local_path, cache_dir=cache_dir)

    if not recovered.all():
        missing = int((~recovered).sum())
        raise RuntimeError(f"Failed to recover {missing} sampled token rows")
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    sample_meta.to_parquet(output_meta, index=False)


def recover_tokens_from_doc_locations(
    *,
    sample_meta: pd.DataFrame,
    locations: pd.DataFrame,
    local_paths: Mapping[str, Path],
    output_tokens: Path,
    output_meta: Path,
    dtype: np.dtype,
    sequence_length: int,
    pad_token_id: int,
) -> None:
    location_cols = [
        "c4_index",
        "token_path",
        "sidecar_path",
        "token_start",
        "token_end",
        "source_path",
        "source_row",
    ]
    row_meta = sample_meta.merge(
        locations[location_cols],
        on="c4_index",
        how="left",
        validate="many_to_one",
    )
    if row_meta["token_path"].isna().any():
        missing = int(row_meta["token_path"].isna().sum())
        raise RuntimeError(f"Missing sidecar token locations for {missing} sampled rows")

    tokens = np.lib.format.open_memmap(
        output_tokens,
        mode="w+",
        dtype=np.int32,
        shape=(len(row_meta), sequence_length),
    )
    tokens[:] = pad_token_id
    row_meta["token_length"] = row_meta["token_end"].astype(np.int64) - row_meta["token_start"].astype(np.int64)
    row_meta["was_truncated"] = row_meta["token_length"] > sequence_length
    row_meta["was_padded"] = row_meta["token_length"] < sequence_length

    for token_path, group in row_meta.groupby("token_path", sort=True):
        local_path = local_paths[str(token_path)]
        values = local_path.stat().st_size // dtype.itemsize
        mmap = np.memmap(local_path, dtype=dtype, mode="r", shape=(values,))
        log(f"recovering {len(group):,} rows from {token_path}")
        for output_pos, start, end in zip(
            group.index.to_numpy(dtype=np.int64),
            group["token_start"].to_numpy(dtype=np.int64),
            group["token_end"].to_numpy(dtype=np.int64),
        ):
            stop = min(int(end), int(start) + sequence_length)
            doc_tokens = np.asarray(mmap[int(start) : stop], dtype=np.int32)
            tokens[int(output_pos), : len(doc_tokens)] = doc_tokens
    tokens.flush()
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    row_meta.to_parquet(output_meta, index=False)


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
    parser.add_argument(
        "--force-rebuild-locations",
        action="store_true",
        help="Re-scan CSV sidecars even if a cached location parquet exists.",
    )
    parser.add_argument(
        "--streaming-download",
        action="store_true",
        help=(
            "Download one token shard at a time, recover its sampled rows, then delete it. "
            "Only supported for index_source=contiguous_token_chunks."
        ),
    )
    parser.add_argument(
        "--keep-downloaded-shards",
        action="store_true",
        help="With --streaming-download, keep shard files after recovery instead of deleting them.",
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
    index_source = str(token_cfg.get("index_source", "contiguous_token_chunks"))
    locations_path = Path(token_cfg.get("recovery_locations", "results/score-pool-robustness/token_recovery_locations.parquet"))

    log(f"loading remote tree: {repo_id}/{remote_prefix}")
    remote_files = list_remote_files(repo_id, remote_prefix, recursive=True)
    token_files = token_files_from_remote(remote_files)
    sidecar_files = sidecar_files_from_remote(remote_files)
    c4_values = sample_meta["c4_index"].to_numpy(dtype=np.int64)
    if index_source == "csv_sidecar":
        locations = build_doc_location_plan(
            repo_id=repo_id,
            c4_indices=c4_values,
            token_files=token_files,
            sidecar_files=sidecar_files,
            locations_path=locations_path,
            force_rebuild=args.force_rebuild_locations,
        )
        plan = write_doc_plan(
            output_path=Path(token_cfg["recovery_plan"]),
            repo_id=repo_id,
            remote_prefix=remote_prefix,
            dtype=dtype,
            sequence_length=sequence_length,
            sample_rows=len(sample_meta),
            unique_c4_indices=int(sample_meta["c4_index"].nunique()),
            max_c4_index=int(c4_values.max()) if len(c4_values) else -1,
            token_files=token_files,
            sidecar_files=sidecar_files,
            locations=locations,
        )
        log(f"sample rows: {plan['sample_rows']:,}")
        log(f"unique c4 indices: {plan['unique_c4_indices']:,}")
        log(f"sidecar matched c4 indices: {plan['found_unique_c4_indices']:,}")
        log(f"index coverage ok: {plan['index_coverage_ok']}")
        log(f"needed token files: {plan['needed_file_count']:,}")
        log(f"needed token bytes: {plan['total_needed_gib']:.2f} GiB")
        log(f"matched sidecar bytes: {plan['total_needed_sidecar_gib']:.2f} GiB")
        log(f"wrote plan: {token_cfg['recovery_plan']}")
        log(f"wrote locations: {locations_path}")

        max_gb = float(token_cfg.get("max_download_gb_without_confirmation", 25))
        if not args.download:
            log("preflight only; rerun with --download to recover tokens")
            return
        if args.streaming_download:
            raise SystemExit("--streaming-download is only supported for contiguous_token_chunks recovery.")
        if not plan["index_coverage_ok"]:
            missing = plan["unique_c4_indices"] - plan["found_unique_c4_indices"]
            raise SystemExit(f"Exact recovery cannot proceed: missing {missing:,} c4 ids from CSV sidecars.")
        if plan["total_needed_gib"] > max_gb and not args.allow_large_download:
            raise SystemExit(
                f"Required token shard download is {plan['total_needed_gib']:.2f} GiB, "
                f"above configured cap {max_gb:.2f} GiB. Rerun with --allow-large-download "
                "only after confirming disk/Colab Drive capacity."
            )
        cache_dir = Path(token_cfg["local_cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        needed_paths = [item["path"] for item in plan["needed_files"]]
        local_paths = download_needed_token_files(
            repo_id=repo_id,
            needed_paths=needed_paths,
            token_files=token_files,
            cache_dir=cache_dir,
        )
        recover_tokens_from_doc_locations(
            sample_meta=sample_meta,
            locations=locations,
            local_paths=local_paths,
            output_tokens=Path(token_cfg["recovered_tokens"]),
            output_meta=Path(token_cfg["recovered_meta"]),
            dtype=dtype,
            sequence_length=sequence_length,
            pad_token_id=int(token_cfg.get("pad_token_id", 1)),
        )
        log(f"wrote tokens: {token_cfg['recovered_tokens']}")
        log(f"wrote metadata: {token_cfg['recovered_meta']}")
        return

    needed, total_chunks = build_needed_file_plan(
        c4_indices=sample_meta["c4_index"].to_numpy(dtype=np.int64),
        token_files=token_files,
        dtype=dtype,
        sequence_length=sequence_length,
    )
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
    if args.streaming_download:
        log("streaming mode: downloading and recovering one token shard at a time")
        recover_tokens_streaming(
            sample_meta=sample_meta,
            needed=needed,
            repo_id=repo_id,
            cache_dir=cache_dir,
            output_tokens=Path(token_cfg["recovered_tokens"]),
            output_meta=Path(token_cfg["recovered_meta"]),
            dtype=dtype,
            sequence_length=sequence_length,
            delete_shards_after_recovery=not args.keep_downloaded_shards,
        )
    else:
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
