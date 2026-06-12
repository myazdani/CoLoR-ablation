from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch.nn as nn


BLOCK_PATH_CANDIDATES = (
    "model.transformer.blocks",
    "transformer.blocks",
    "model.model.layers",
    "model.layers",
    "gpt_neox.layers",
    "model.transformer.h",
    "transformer.h",
    "blocks",
    "layers",
)


@dataclass(frozen=True)
class AblationRecord:
    original_num_layers: int
    kept_num_layers: int
    removed_layers: tuple[int, ...]
    block_path: str


def variant_layer_indices(variant_id: str, total_layers: int = 12) -> tuple[int, ...]:
    if variant_id in {"full", "full_rescore"}:
        return ()

    prefix = variant_id[:3]
    try:
        count = int(variant_id[3:])
    except ValueError as exc:
        if variant_id == "skip2":
            return tuple(range(1, total_layers, 2))
        raise ValueError(f"Unsupported variant id '{variant_id}'") from exc

    if count < 0 or count > total_layers:
        raise ValueError(f"Cannot remove {count} layers from {total_layers}-layer model")

    if prefix == "top":
        return tuple(range(total_layers - count, total_layers))
    if prefix == "bot":
        return tuple(range(count))
    if prefix == "mid":
        start = (total_layers - count) // 2
        return tuple(range(start, start + count))

    raise ValueError(f"Unsupported variant id '{variant_id}'")


def _resolve_attr(root: object, path: str) -> object | None:
    current = root
    for part in path.split("."):
        if isinstance(current, (nn.ModuleList, list)) and part.isdigit():
            idx = int(part)
            if idx >= len(current):
                return None
            current = current[idx]
            continue
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def find_block_module_list(model: nn.Module) -> tuple[str, nn.ModuleList]:
    for path in BLOCK_PATH_CANDIDATES:
        candidate = _resolve_attr(model, path)
        if isinstance(candidate, nn.ModuleList):
            return path, candidate
    raise AttributeError(
        "Could not find transformer blocks. Tried: " + ", ".join(BLOCK_PATH_CANDIDATES)
    )


def _set_config_layer_count(model: nn.Module, kept_layers: int) -> None:
    config = getattr(model, "config", None)
    if config is None:
        return
    for attr in ("n_layers", "num_hidden_layers"):
        if hasattr(config, attr):
            try:
                setattr(config, attr, kept_layers)
            except Exception:
                pass


def apply_layer_ablation(
    model: nn.Module,
    removed_layers: Iterable[int],
    *,
    total_layers: int | None = None,
) -> AblationRecord:
    block_path, blocks = find_block_module_list(model)
    original_count = len(blocks)
    if total_layers is not None and original_count != total_layers:
        raise ValueError(f"Expected {total_layers} blocks, found {original_count}")

    removed = tuple(sorted(set(int(i) for i in removed_layers)))
    if any(i < 0 or i >= original_count for i in removed):
        raise IndexError(f"Removed layers {removed} out of range for {original_count} blocks")

    # Preserve each block's original layer_id; it is provenance, not a forward-time
    # position for RoPE. Only update config counts so cache-related checks match.
    for idx in reversed(removed):
        del blocks[idx]

    record = AblationRecord(
        original_num_layers=original_count,
        kept_num_layers=len(blocks),
        removed_layers=removed,
        block_path=block_path,
    )
    setattr(model, "color_ablation", record)
    _set_config_layer_count(model, len(blocks))
    return record


def apply_paired_ablation(
    cond_model: nn.Module,
    marg_model: nn.Module,
    removed_layers: Iterable[int],
    *,
    total_layers: int | None = None,
) -> tuple[AblationRecord, AblationRecord]:
    cond_record = apply_layer_ablation(cond_model, removed_layers, total_layers=total_layers)
    marg_record = apply_layer_ablation(marg_model, removed_layers, total_layers=total_layers)
    if cond_record.removed_layers != marg_record.removed_layers:
        raise AssertionError("Conditional and marginal models have different removed layers")
    if cond_record.kept_num_layers != marg_record.kept_num_layers:
        raise AssertionError("Conditional and marginal models have different block counts")
    return cond_record, marg_record

