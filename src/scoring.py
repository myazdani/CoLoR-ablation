from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ScoreStats:
    elapsed_seconds: float
    tokens_scored: int
    tokens_per_second: float


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str) -> torch.dtype | None:
    if dtype in {"auto", "none", ""}:
        return None
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype '{dtype}'")


def _extract_logits(output: Any) -> torch.Tensor:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, tuple):
        return output[0]
    raise TypeError("Model output has no logits")


@torch.inference_mode()
def sequence_mean_nll(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    ignore_index: int = -100,
    autocast_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if input_ids.ndim != 2:
        raise ValueError(f"Expected input_ids shape [batch, seq], got {tuple(input_ids.shape)}")
    if input_ids.shape[1] < 2:
        raise ValueError("Need at least two tokens to compute next-token loss")

    device_type = input_ids.device.type
    use_autocast = device_type == "cuda" and autocast_dtype in {torch.bfloat16, torch.float16}
    with torch.autocast(device_type, enabled=use_autocast, dtype=autocast_dtype):
        logits = _extract_logits(model(input_ids=input_ids))

    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    batch, tokens, vocab = shift_logits.shape
    flat_loss = F.cross_entropy(
        shift_logits.view(batch * tokens, vocab),
        shift_labels.view(batch * tokens),
        reduction="none",
        ignore_index=ignore_index,
    ).view(batch, tokens)
    valid = shift_labels.ne(ignore_index)
    denom = valid.sum(dim=1).clamp_min(1)
    return (flat_loss * valid).sum(dim=1) / denom


def score_model(
    model: torch.nn.Module,
    pool: np.ndarray,
    *,
    batch_size: int,
    device: str | torch.device = "auto",
    dtype: str = "bf16",
) -> tuple[np.ndarray, ScoreStats]:
    resolved_device = resolve_device(device) if isinstance(device, str) else device
    autocast_dtype = resolve_dtype(dtype)
    model.to(resolved_device)
    model.eval()

    losses: list[np.ndarray] = []
    start = time.perf_counter()
    for start_idx in range(0, len(pool), batch_size):
        batch_np = pool[start_idx : start_idx + batch_size]
        batch = torch.as_tensor(batch_np, dtype=torch.long, device=resolved_device)
        loss = sequence_mean_nll(model, batch, autocast_dtype=autocast_dtype)
        losses.append(loss.detach().cpu().numpy())

    elapsed = time.perf_counter() - start
    tokens = int(np.prod(pool.shape))
    stats = ScoreStats(
        elapsed_seconds=elapsed,
        tokens_scored=tokens,
        tokens_per_second=tokens / elapsed if elapsed > 0 else float("nan"),
    )
    return np.concatenate(losses), stats


def score_pair(
    cond_model: torch.nn.Module,
    marg_model: torch.nn.Module,
    pool: np.ndarray,
    *,
    batch_size: int,
    device: str | torch.device = "auto",
    dtype: str = "bf16",
) -> tuple[pd.DataFrame, dict[str, float]]:
    resolved_device = resolve_device(device) if isinstance(device, str) else device
    cond_model.to(resolved_device)
    marg_model.to(resolved_device)
    cond_model.eval()
    marg_model.eval()
    autocast_dtype = resolve_dtype(dtype)

    cond_losses: list[np.ndarray] = []
    marg_losses: list[np.ndarray] = []
    start = time.perf_counter()
    for start_idx in range(0, len(pool), batch_size):
        batch_np = pool[start_idx : start_idx + batch_size]
        batch = torch.as_tensor(batch_np, dtype=torch.long, device=resolved_device)
        cond = sequence_mean_nll(cond_model, batch, autocast_dtype=autocast_dtype)
        marg = sequence_mean_nll(marg_model, batch, autocast_dtype=autocast_dtype)
        cond_losses.append(cond.detach().cpu().numpy())
        marg_losses.append(marg.detach().cpu().numpy())

    elapsed = time.perf_counter() - start
    nll_cond = np.concatenate(cond_losses)
    nll_marg = np.concatenate(marg_losses)
    frame = pd.DataFrame(
        {
            "seq_idx": np.arange(len(pool), dtype=np.int64),
            "nll_cond": nll_cond,
            "nll_marg": nll_marg,
            "color": nll_cond - nll_marg,
        }
    )
    tokens = int(np.prod(pool.shape)) * 2
    stats = {
        "elapsed_seconds": elapsed,
        "tokens_scored": tokens,
        "tokens_per_second": tokens / elapsed if elapsed > 0 else float("nan"),
    }
    return frame, stats

