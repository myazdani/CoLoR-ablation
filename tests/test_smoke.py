from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.ablation import apply_paired_ablation, find_block_module_list, variant_layer_indices
from src.metrics import compute_all_metrics, compute_variant_metrics
from src.packing import build_synthetic_pool
from src.scoring import score_pair


class TinyOutput:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class TinyBlock(nn.Module):
    def __init__(self, layer_id: int, dim: int):
        super().__init__()
        self.layer_id = layer_id
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.ff(self.norm(x))


class TinyLM(nn.Module):
    def __init__(self, *, layers: int, vocab_size: int, dim: int, sequence_length: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.config = SimpleNamespace(n_layers=layers, num_hidden_layers=layers)
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(sequence_length, dim)
        self.transformer = nn.Module()
        self.transformer.blocks = nn.ModuleList([TinyBlock(i, dim) for i in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> TinyOutput:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos(positions)
        for block in self.transformer.blocks:
            x = block(x)
        return TinyOutput(self.head(self.norm(x)))


def _make_pair() -> tuple[TinyLM, TinyLM]:
    marg = TinyLM(layers=4, vocab_size=37, dim=24, sequence_length=16, seed=123)
    cond = copy.deepcopy(marg)
    torch.manual_seed(456)
    with torch.no_grad():
        for param in cond.parameters():
            param.add_(0.01 * torch.randn_like(param))
    return cond, marg


def test_variant_index_math() -> None:
    assert variant_layer_indices("full", total_layers=12) == ()
    assert variant_layer_indices("top4", total_layers=12) == (8, 9, 10, 11)
    assert variant_layer_indices("mid4", total_layers=12) == (4, 5, 6, 7)
    assert variant_layer_indices("bot4", total_layers=12) == (0, 1, 2, 3)
    assert variant_layer_indices("skip2", total_layers=12) == (1, 3, 5, 7, 9, 11)


def test_cpu_smoke_pipeline() -> None:
    pool = build_synthetic_pool(
        n_c4=32,
        n_enriched=5,
        sequence_length=16,
        vocab_size=37,
        seed=17,
    )
    assert pool.tokens.shape == (37, 16)
    assert int(pool.metadata["enriched"].sum()) == 5

    cond, marg = _make_pair()
    block_path, blocks = find_block_module_list(cond)
    assert block_path == "transformer.blocks"
    assert len(blocks) == 4

    full_scores, full_stats = score_pair(cond, marg, pool.tokens, batch_size=8, device="cpu", dtype="fp32")
    assert len(full_scores) == len(pool.tokens)
    assert {"seq_idx", "nll_cond", "nll_marg", "color"}.issubset(full_scores.columns)
    assert np.isfinite(full_scores["color"]).all()
    assert full_stats["tokens_per_second"] > 0

    cond_ablate, marg_ablate = _make_pair()
    removed = variant_layer_indices("mid2", total_layers=4)
    cond_record, marg_record = apply_paired_ablation(
        cond_ablate,
        marg_ablate,
        removed,
        total_layers=4,
    )
    assert cond_record.removed_layers == (1, 2)
    assert marg_record.removed_layers == cond_record.removed_layers
    _, ablated_blocks = find_block_module_list(cond_ablate)
    assert len(ablated_blocks) == 2
    assert [block.layer_id for block in ablated_blocks] == [0, 3]
    assert cond_ablate.config.n_layers == 2

    ablated_scores, ablated_stats = score_pair(
        cond_ablate,
        marg_ablate,
        pool.tokens,
        batch_size=8,
        device="cpu",
        dtype="fp32",
    )
    metrics = compute_variant_metrics(
        full_scores=full_scores,
        variant_scores=ablated_scores,
        pool_meta=pool.metadata,
        variant_id="mid2",
        removed_layers=removed,
        total_layers=4,
        selection_rates=[0.125, 0.25],
        bootstrap_reps=20,
        bootstrap_seed=17,
        tail_rate_for_local_spearman=0.25,
        tokens_per_second=ablated_stats["tokens_per_second"],
    )

    assert set(metrics["selection_rate"]) == {0.125, 0.25}
    assert (metrics["n_pure"] == 32).all()
    assert (metrics["n_enriched"] == 5).all()
    assert metrics["recall_at_k"].between(0, 1).all()
    assert metrics["jaccard_at_k"].between(0, 1).all()
    assert metrics["variant_enriched_selected_frac"].between(0, 1).all()
    assert "nll_cond_spearman" in metrics.columns
    assert "nll_marg_spearman" in metrics.columns


def test_full_self_metrics_are_perfect() -> None:
    pool = build_synthetic_pool(
        n_c4=32,
        n_enriched=0,
        sequence_length=16,
        vocab_size=37,
        seed=18,
    )
    cond, marg = _make_pair()
    full_scores, stats = score_pair(cond, marg, pool.tokens, batch_size=16, device="cpu", dtype="fp32")
    all_metrics = compute_all_metrics(
        score_frames={"full": full_scores},
        pool_meta=pool.metadata,
        variants_removed_layers={"full": ()},
        full_variant="full",
        total_layers=4,
        selection_rates=[0.125, 0.25],
        bootstrap_reps=10,
        bootstrap_seed=17,
        tail_rate_for_local_spearman=0.25,
        variant_tokens_per_second={"full": stats["tokens_per_second"]},
    )
    assert (all_metrics["recall_at_k"] == 1.0).all()
    assert (all_metrics["jaccard_at_k"] == 1.0).all()
    assert (all_metrics["spearman"] == 1.0).all()


def test_metrics_use_metadata_for_enriched_count() -> None:
    scores = pd.DataFrame(
        {
            "seq_idx": [0, 1, 2, 3],
            "nll_cond": [1.0, 2.0, 3.0, 4.0],
            "nll_marg": [2.0, 2.0, 2.0, 2.0],
            "color": [-1.0, 0.0, 1.0, 2.0],
        }
    )
    meta = pd.DataFrame(
        {
            "seq_idx": [0, 1, 2, 3],
            "enriched": [False, False, True, True],
        }
    )
    metrics = compute_variant_metrics(
        full_scores=scores,
        variant_scores=scores,
        pool_meta=meta,
        variant_id="full",
        removed_layers=(),
        total_layers=4,
        selection_rates=[0.5],
        bootstrap_reps=0,
        bootstrap_seed=17,
        tail_rate_for_local_spearman=0.5,
    )
    assert int(metrics["n_pure"].iloc[0]) == 2
    assert int(metrics["n_enriched"].iloc[0]) == 2

