from __future__ import annotations

import copy
import importlib.util
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.ablation import apply_dual_ablation, find_block_module_list
from src.score_pool_robustness import (
    average_precision_from_scores,
    compute_pairwise_metrics,
    default_variant_specs,
    load_pool_samples,
    roc_auc_from_scores,
)


class TinyBlock(nn.Module):
    def __init__(self, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


class TinyModel(nn.Module):
    def __init__(self, layers: int = 6):
        super().__init__()
        self.config = SimpleNamespace(n_layers=layers, num_hidden_layers=layers)
        self.transformer = nn.Module()
        self.transformer.blocks = nn.ModuleList([TinyBlock(i) for i in range(layers)])


def test_apply_dual_ablation_allows_asymmetric_layer_sets() -> None:
    cond = TinyModel(layers=6)
    marg = copy.deepcopy(cond)

    cond_record, marg_record = apply_dual_ablation(
        cond,
        marg,
        cond_removed_layers=(4, 5),
        marg_removed_layers=(0, 1),
        total_layers=6,
    )

    assert cond_record.removed_layers == (4, 5)
    assert marg_record.removed_layers == (0, 1)
    assert cond_record.kept_num_layers == 4
    assert marg_record.kept_num_layers == 4
    _, cond_blocks = find_block_module_list(cond)
    _, marg_blocks = find_block_module_list(marg)
    assert [block.layer_id for block in cond_blocks] == [0, 1, 2, 3]
    assert [block.layer_id for block in marg_blocks] == [2, 3, 4, 5]


def test_default_score_pool_variant_specs_cover_expected_grid() -> None:
    specs = default_variant_specs(total_layers=12)

    assert specs["cond_top4_marg_bot4"].cond_removed_layers == (8, 9, 10, 11)
    assert specs["cond_top4_marg_bot4"].marg_removed_layers == (0, 1, 2, 3)
    assert specs["cond_bot2_marg_top2"].cond_removed_layers == (0, 1)
    assert specs["cond_bot2_marg_top2"].marg_removed_layers == (10, 11)
    assert specs["cond_top6_only"].marg_removed_layers == ()
    assert specs["marg_top6_only"].cond_removed_layers == ()


def _write_pool_npz(path, *, offset: int) -> None:
    n = 3
    np.savez_compressed(
        path,
        row_position=np.arange(offset, offset + n, dtype=np.int64),
        c4_index=np.arange(1000 + offset, 1000 + offset + n, dtype=np.int64),
        prior_score=np.linspace(3.0, 3.2, n, dtype=np.float32),
        conditional_books_score=np.linspace(3.4, 3.6, n, dtype=np.float32),
        color_score=np.linspace(0.4, 0.6, n, dtype=np.float32),
    )


def test_load_pool_samples_reads_existing_npz_shape(tmp_path) -> None:
    filenames = {
        "random_positive_samples.npz",
        "hard_positive_samples.npz",
        "random_negative_samples.npz",
        "hard_negative_samples.npz",
        "tail_negative_samples.npz",
    }
    for i, filename in enumerate(sorted(filenames)):
        _write_pool_npz(tmp_path / filename, offset=i * 10)

    frame = load_pool_samples(tmp_path)

    assert len(frame) == 15
    assert set(frame["pool_name"]) == {
        "random_positive",
        "hard_positive",
        "random_negative",
        "hard_negative",
        "tail_negative",
    }
    assert {"full_prior_score", "full_conditional_books_score", "full_color_score"}.issubset(
        frame.columns
    )
    assert frame["seq_idx"].tolist() == list(range(15))


def _metric_frame() -> pd.DataFrame:
    rows = []
    values = {
        "hard_positive": [0.20, 0.21, 0.22, 0.23],
        "hard_negative": [0.24, 0.25, 0.26, 0.27],
        "random_negative": [0.35, 0.36, 0.37, 0.38],
        "tail_negative": [0.70, 0.80, 0.90, 1.00],
        "random_positive": [0.10, 0.15, 0.25, 0.30],
    }
    seq_idx = 0
    for pool, colors in values.items():
        for color in colors:
            rows.append(
                {
                    "seq_idx": seq_idx,
                    "pool_name": pool,
                    "full_prior_score": 3.0,
                    "full_conditional_books_score": 3.0 + color,
                    "full_color_score": color,
                    "ablated_prior_score": 3.0,
                    "ablated_conditional_books_score": 3.0 + color,
                    "ablated_color_score": color,
                    "variant_family": "test",
                    "cond_removed_layers": "[]",
                    "marg_removed_layers": "[]",
                }
            )
            seq_idx += 1
    return pd.DataFrame(rows)


def test_pairwise_metrics_use_lower_color_as_positive() -> None:
    metrics, shifts = compute_pairwise_metrics(
        _metric_frame(),
        variant_id="toy",
        cutoff_tau64=0.235,
        pairwise_tasks=["hp_vs_hn", "hp_vs_tn", "rp_vs_hn"],
    )

    by_task = metrics.set_index("pairwise_task")
    assert by_task.loc["hp_vs_hn", "roc_auc"] == 1.0
    assert by_task.loc["hp_vs_tn", "average_precision"] == 1.0
    assert by_task.loc["hp_vs_hn", "recall_at_original_cutoff"] == 1.0
    assert by_task.loc["hp_vs_hn", "precision_at_original_cutoff"] == 1.0
    assert by_task.loc["rp_vs_hn", "roc_auc"] < 1.0
    assert set(shifts["score_name"]) == {"color", "conditional_books", "prior"}


def test_auc_and_ap_helpers() -> None:
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])

    assert roc_auc_from_scores(labels, scores) == 1.0
    assert average_precision_from_scores(labels, scores) == 1.0


def test_token_recovery_plan_helpers(tmp_path) -> None:
    script_path = __import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "09_recover_score_pool_tokens.py"
    spec = importlib.util.spec_from_file_location("recover_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["recover_script"] = module
    spec.loader.exec_module(module)

    remote = [
        module.RemoteFile(path="full_data/c4/a.npy", size=10 * 512 * np.dtype(np.uint16).itemsize),
        module.RemoteFile(path="full_data/c4/b.npy", size=10 * 512 * np.dtype(np.uint16).itemsize),
    ]
    needed, total_chunks = module.build_needed_file_plan(
        c4_indices=np.array([0, 9, 10, 19], dtype=np.int64),
        token_files=remote,
        dtype=np.dtype(np.uint16),
        sequence_length=512,
    )

    assert total_chunks == 20
    assert [item.path for item in needed] == ["full_data/c4/a.npy", "full_data/c4/b.npy"]
    assert [item.needed_rows for item in needed] == [2, 2]


def test_streaming_token_recovery_downloads_and_deletes_one_shard_at_a_time(tmp_path, monkeypatch) -> None:
    script_path = __import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "09_recover_score_pool_tokens.py"
    spec = importlib.util.spec_from_file_location("recover_script_streaming", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["recover_script_streaming"] = module
    spec.loader.exec_module(module)

    cache_dir = tmp_path / "cache"
    shard_a = cache_dir / "full_data/c4/a.npy"
    shard_b = cache_dir / "full_data/c4/b.npy"
    shard_a.parent.mkdir(parents=True)
    np.arange(16, dtype=np.uint16).tofile(shard_a)
    (np.arange(16, dtype=np.uint16) + 100).tofile(shard_b)

    sample_meta = pd.DataFrame(
        {
            "seq_idx": [0, 1, 2, 3],
            "pool_name": ["a", "a", "b", "b"],
            "c4_index": [0, 3, 4, 7],
        }
    )
    needed = [
        module.NeededFile(
            path="full_data/c4/a.npy",
            size=shard_a.stat().st_size,
            chunk_start=0,
            chunk_end=4,
            needed_rows=2,
        ),
        module.NeededFile(
            path="full_data/c4/b.npy",
            size=shard_b.stat().st_size,
            chunk_start=4,
            chunk_end=8,
            needed_rows=2,
        ),
    ]

    download_order = []

    def fake_download_one_needed_file(*, repo_id, item, cache_dir):
        download_order.append(item.path)
        return cache_dir / item.path

    monkeypatch.setattr(module, "download_one_needed_file", fake_download_one_needed_file)

    captured_meta = {}

    def fake_to_parquet(self, path, index=False):
        captured_meta["frame"] = self.copy()
        captured_meta["path"] = path
        captured_meta["index"] = index

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    output_tokens = tmp_path / "recovered.npy"
    output_meta = tmp_path / "recovered.parquet"
    module.recover_tokens_streaming(
        sample_meta=sample_meta,
        needed=needed,
        repo_id="owner/repo",
        cache_dir=cache_dir,
        output_tokens=output_tokens,
        output_meta=output_meta,
        dtype=np.dtype(np.uint16),
        sequence_length=4,
        delete_shards_after_recovery=True,
    )

    recovered = np.load(output_tokens)
    assert recovered.tolist() == [
        [0, 1, 2, 3],
        [12, 13, 14, 15],
        [100, 101, 102, 103],
        [112, 113, 114, 115],
    ]
    assert download_order == ["full_data/c4/a.npy", "full_data/c4/b.npy"]
    assert not shard_a.exists()
    assert not shard_b.exists()
    assert captured_meta["path"] == output_meta
    assert captured_meta["frame"].equals(sample_meta)


def test_streaming_token_recovery_can_resume_from_file_index(tmp_path, monkeypatch) -> None:
    script_path = __import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "09_recover_score_pool_tokens.py"
    spec = importlib.util.spec_from_file_location("recover_script_streaming_resume", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["recover_script_streaming_resume"] = module
    spec.loader.exec_module(module)

    cache_dir = tmp_path / "cache"
    shard_a = cache_dir / "full_data/c4/a.npy"
    shard_b = cache_dir / "full_data/c4/b.npy"
    shard_a.parent.mkdir(parents=True)
    np.arange(16, dtype=np.uint16).tofile(shard_a)
    (np.arange(16, dtype=np.uint16) + 100).tofile(shard_b)

    sample_meta = pd.DataFrame(
        {
            "seq_idx": [0, 1, 2, 3],
            "pool_name": ["a", "a", "b", "b"],
            "c4_index": [0, 3, 4, 7],
        }
    )
    needed = [
        module.NeededFile(
            path="full_data/c4/a.npy",
            size=shard_a.stat().st_size,
            chunk_start=0,
            chunk_end=4,
            needed_rows=2,
        ),
        module.NeededFile(
            path="full_data/c4/b.npy",
            size=shard_b.stat().st_size,
            chunk_start=4,
            chunk_end=8,
            needed_rows=2,
        ),
    ]
    output_tokens = tmp_path / "recovered.npy"
    output_meta = tmp_path / "recovered.parquet"
    partial = np.lib.format.open_memmap(output_tokens, mode="w+", dtype=np.int32, shape=(4, 4))
    partial[0] = [0, 1, 2, 3]
    partial[1] = [12, 13, 14, 15]
    partial.flush()

    download_order = []

    def fake_download_one_needed_file(*, repo_id, item, cache_dir):
        download_order.append(item.path)
        return cache_dir / item.path

    monkeypatch.setattr(module, "download_one_needed_file", fake_download_one_needed_file)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: None)

    module.recover_tokens_streaming(
        sample_meta=sample_meta,
        needed=needed,
        repo_id="owner/repo",
        cache_dir=cache_dir,
        output_tokens=output_tokens,
        output_meta=output_meta,
        dtype=np.dtype(np.uint16),
        sequence_length=4,
        delete_shards_after_recovery=False,
        resume=True,
        resume_from_file_index=2,
    )

    recovered = np.load(output_tokens)
    assert recovered.tolist() == [
        [0, 1, 2, 3],
        [12, 13, 14, 15],
        [100, 101, 102, 103],
        [112, 113, 114, 115],
    ]
    assert download_order == ["full_data/c4/b.npy"]
    checkpoint = output_tokens.with_suffix(output_tokens.suffix + ".streaming_checkpoint.json")
    data = json.loads(checkpoint.read_text())
    assert data["completed_count"] == 2
    assert data["total_count"] == 2


def test_list_remote_files_follows_hf_pagination(monkeypatch) -> None:
    script_path = __import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "09_recover_score_pool_tokens.py"
    spec = importlib.util.spec_from_file_location("recover_script_pagination", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["recover_script_pagination"] = module
    spec.loader.exec_module(module)

    class FakeResponse:
        def __init__(self, payload, link=""):
            self.payload = payload
            self.headers = {"Link": link} if link else {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        url = request.full_url
        if "cursor=page2" in url:
            return FakeResponse(
                [
                    {"type": "file", "path": "full_data/c4/b.npy", "size": 20},
                    {"type": "file", "path": "full_data/c4/b.csv.gz", "size": 2},
                ]
            )
        return FakeResponse(
            [
                {"type": "file", "path": "full_data/c4/a.npy", "size": 10},
                {"type": "file", "path": "full_data/c4/a.csv.gz", "size": 1},
            ],
            link='<https://example.test/tree?cursor=page2>; rel="next"',
        )

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)

    files = module.list_remote_files("owner/repo", "full_data/c4")

    assert [file.path for file in files] == [
        "full_data/c4/a.npy",
        "full_data/c4/a.csv.gz",
        "full_data/c4/b.npy",
        "full_data/c4/b.csv.gz",
    ]


def test_doc_location_recovery_pads_and_truncates(tmp_path, monkeypatch) -> None:
    script_path = __import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "09_recover_score_pool_tokens.py"
    spec = importlib.util.spec_from_file_location("recover_script_doc_locations", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["recover_script_doc_locations"] = module
    spec.loader.exec_module(module)

    token_path = tmp_path / "tokens.npy"
    np.arange(20, dtype=np.uint16).tofile(token_path)
    sample_meta = pd.DataFrame(
        {
            "seq_idx": [0, 1],
            "pool_name": ["a", "b"],
            "c4_index": [10, 11],
        }
    )
    locations = pd.DataFrame(
        {
            "c4_index": [10, 11],
            "token_path": ["full_data/c4/a.npy", "full_data/c4/a.npy"],
            "sidecar_path": ["full_data/c4/a.csv.gz", "full_data/c4/a.csv.gz"],
            "token_start": [2, 4],
            "token_end": [5, 12],
            "source_path": ["raw-a", "raw-a"],
            "source_row": [10, 11],
        }
    )
    output_tokens = tmp_path / "recovered.npy"
    output_meta = tmp_path / "recovered.parquet"
    captured_meta = {}

    def fake_to_parquet(self, path, index=False):
        captured_meta["frame"] = self.copy()
        captured_meta["path"] = path
        captured_meta["index"] = index

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    module.recover_tokens_from_doc_locations(
        sample_meta=sample_meta,
        locations=locations,
        local_paths={"full_data/c4/a.npy": token_path},
        output_tokens=output_tokens,
        output_meta=output_meta,
        dtype=np.dtype(np.uint16),
        sequence_length=5,
        pad_token_id=1,
    )

    recovered = np.load(output_tokens)
    meta = captured_meta["frame"]
    assert recovered.tolist() == [[2, 3, 4, 1, 1], [4, 5, 6, 7, 8]]
    assert meta["was_padded"].tolist() == [True, False]
    assert meta["was_truncated"].tolist() == [False, True]
