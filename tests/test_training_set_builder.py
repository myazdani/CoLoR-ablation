from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


def _load_builder_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "18_build_score_pool_training_sets.py"
    spec = importlib.util.spec_from_file_location("training_set_builder", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["training_set_builder"] = module
    spec.loader.exec_module(module)
    return module


def _write_pool_npz(path: Path, frame: pd.DataFrame) -> None:
    np.savez_compressed(
        path,
        row_position=frame["row_position"].to_numpy(dtype=np.int64),
        c4_index=frame["c4_index"].to_numpy(dtype=np.int64),
        prior_score=np.full(len(frame), 3.0, dtype=np.float32),
        conditional_books_score=(3.0 + frame["full_color_score"].to_numpy(dtype=np.float32)),
        color_score=frame["full_color_score"].to_numpy(dtype=np.float32),
    )


def test_build_score_pool_training_sets_writes_memmaps_and_diagnostics(tmp_path) -> None:
    builder = _load_builder_module()
    sequence_length = 512
    pool_names = [
        "random_positive",
        "hard_positive",
        "random_negative",
        "hard_negative",
        "tail_negative",
    ]
    rows = []
    seq_idx = 0
    for pool_idx, pool_name in enumerate(pool_names):
        for sample_idx in range(4):
            rows.append(
                {
                    "seq_idx": seq_idx,
                    "sample_idx": sample_idx,
                    "pool_name": pool_name,
                    "pool_label": pool_name.replace("_", " ").title(),
                    "row_position": 10_000 + seq_idx,
                    "c4_index": 20_000 + seq_idx,
                    "full_prior_score": np.float32(3.0),
                    "full_conditional_books_score": np.float32(3.0 + 0.1 * pool_idx + 0.01 * sample_idx),
                    "full_color_score": np.float32(0.1 * pool_idx + 0.01 * sample_idx),
                    "label_available_for_pair_tasks": True,
                }
            )
            seq_idx += 1
    meta = pd.DataFrame(rows)

    # Full local scores recover positives perfectly inside the two mini-universes.
    full_scores = meta.copy()
    full_scores["ablated_prior_score"] = full_scores["full_prior_score"]
    full_scores["ablated_conditional_books_score"] = full_scores["full_conditional_books_score"]
    full_scores["ablated_color_score"] = full_scores["full_color_score"]
    full_scores["elapsed_seconds"] = 10.0
    full_scores["tokens_scored"] = len(meta) * sequence_length
    full_scores["tokens_per_second"] = 1000.0

    pair_mid2_scores = full_scores.copy()
    pair_mid2_scores["ablated_color_score"] = pair_mid2_scores["full_color_score"] + 0.001

    # Make pair_mid2 prefer all positives before negatives in each tested universe.
    pair_mid2_scores.loc[pair_mid2_scores["pool_name"].isin(["random_positive", "hard_positive"]), "ablated_color_score"] -= 1.0

    tokens = np.arange(len(meta) * sequence_length, dtype=np.int32).reshape(len(meta), sequence_length) % 100
    tokens_path = tmp_path / "tokens.npy"
    np.save(tokens_path, tokens)
    meta_path = tmp_path / "meta.parquet"
    full_path = tmp_path / "scores_full.parquet"
    pair_path = tmp_path / "scores_pair_mid2.parquet"
    meta.to_parquet(meta_path, index=False)
    full_scores.to_parquet(full_path, index=False)
    pair_mid2_scores.to_parquet(pair_path, index=False)

    pool_dir = tmp_path / "pools"
    pool_dir.mkdir()
    for pool_name in pool_names:
        _write_pool_npz(pool_dir / f"{pool_name}_samples.npz", meta[meta["pool_name"] == pool_name])

    output_dir = tmp_path / "out"
    builder.build_training_sets(
        SimpleNamespace(
            tokens=tokens_path,
            meta=meta_path,
            full_scores=full_path,
            pair_mid2_scores=pair_path,
            pool_dir=pool_dir,
            cascade_multiplier=1.5,
            sensitivity_multipliers="1.25,1.5,2.0",
            seed=17,
            target_rows=4,
            training_target_tokens=100_000_000,
            sequence_length=sequence_length,
            tokenizer="test-tokenizer",
            output_token_dtype="uint16",
            copy_chunk_rows=2,
            output_dir=output_dir,
        )
    )

    diagnostics = pd.read_csv(output_dir / "selection_diagnostics.csv")
    assert set(diagnostics["run_id"]) == set(builder.P0_RUNS)
    assert diagnostics["selected_rows"].tolist() == [4, 4, 4, 4]
    assert (diagnostics["unique_seq_idx"] == 4).all()
    assert (diagnostics["true_positive_count"] == 4).all()

    overlap = pd.read_csv(output_dir / "overlap_jaccard.csv")
    assert {"left_run_id", "right_run_id", "seq_idx_jaccard"}.issubset(overlap.columns)

    for run_id in builder.P0_RUNS:
        run_dir = output_dir / run_id
        assert (run_dir / "manifest.json").exists()
        assert len(pd.read_parquet(run_dir / "train_meta.parquet")) == 4
        memmap = np.memmap(run_dir / "train_tokens.npy", dtype=np.uint16, mode="r", shape=(4, sequence_length))
        assert memmap.shape == (4, sequence_length)
