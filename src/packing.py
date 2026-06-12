from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PoolBuildResult:
    tokens: np.ndarray
    metadata: pd.DataFrame


def pack_token_ids(token_ids: Iterable[int], sequence_length: int) -> np.ndarray:
    ids = np.asarray(list(token_ids), dtype=np.int64)
    usable = (len(ids) // sequence_length) * sequence_length
    if usable == 0:
        return np.empty((0, sequence_length), dtype=np.int32)
    return ids[:usable].reshape(-1, sequence_length).astype(np.int32)


def build_synthetic_pool(
    *,
    n_c4: int,
    n_enriched: int,
    sequence_length: int,
    vocab_size: int,
    seed: int,
) -> PoolBuildResult:
    rng = np.random.default_rng(seed)
    c4 = rng.integers(0, vocab_size, size=(n_c4, sequence_length), dtype=np.int32)
    if n_enriched:
        enriched = rng.integers(0, vocab_size, size=(n_enriched, sequence_length), dtype=np.int32)
        tokens = np.concatenate([c4, enriched], axis=0)
    else:
        tokens = c4

    metadata = pd.DataFrame(
        {
            "seq_idx": np.arange(len(tokens), dtype=np.int64),
            "source": ["synthetic_c4"] * n_c4 + ["synthetic_enriched"] * n_enriched,
            "enriched": [False] * n_c4 + [True] * n_enriched,
            "source_sequence_idx": list(range(n_c4)) + list(range(n_enriched)),
            "seed": seed,
        }
    )
    return PoolBuildResult(tokens=tokens, metadata=metadata)


def _iter_hf_texts(
    *,
    source: str,
    name: str | None,
    split: str,
    seed: int,
    shuffle_buffer: int,
) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before building real HF pools") from exc

    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    dataset = load_dataset(source, name, **kwargs) if name else load_dataset(source, **kwargs)
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    for row in dataset:
        text = row.get("text")
        if isinstance(text, str) and text:
            yield text


def _pack_texts(
    texts: Iterable[str],
    tokenizer: Any,
    *,
    n_sequences: int,
    sequence_length: int,
) -> np.ndarray:
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")

    sequences: list[np.ndarray] = []
    token_buffer: list[int] = []
    cursor = 0
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_buffer.extend(encoded)
        token_buffer.append(eos_token_id)

        while len(token_buffer) - cursor >= sequence_length and len(sequences) < n_sequences:
            start = cursor
            end = cursor + sequence_length
            sequences.append(np.asarray(token_buffer[start:end], dtype=np.int32))
            cursor = end

        if cursor > 100000:
            token_buffer = token_buffer[cursor:]
            cursor = 0
        if len(sequences) >= n_sequences:
            break

    if len(sequences) < n_sequences:
        raise RuntimeError(f"Requested {n_sequences} packed sequences, got {len(sequences)}")
    return np.stack(sequences, axis=0)


def build_pool_from_config(config: dict[str, Any]) -> PoolBuildResult:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before building real HF pools") from exc

    pool_cfg = config["pool"]
    sequence_length = int(pool_cfg["sequence_length"])
    tokenizer = AutoTokenizer.from_pretrained(pool_cfg["tokenizer"])

    c4_cfg = pool_cfg["c4"]
    c4_tokens = _pack_texts(
        _iter_hf_texts(
            source=c4_cfg["source"],
            name=c4_cfg.get("name"),
            split=c4_cfg["split"],
            seed=int(c4_cfg["seed"]),
            shuffle_buffer=int(c4_cfg.get("shuffle_buffer", 0)),
        ),
        tokenizer,
        n_sequences=int(c4_cfg["n_sequences"]),
        sequence_length=sequence_length,
    )

    enriched_cfg = pool_cfg.get("enriched", {})
    enriched_n = int(enriched_cfg.get("n", 0))
    if enriched_n:
        enriched_tokens = _pack_texts(
            _iter_hf_texts(
                source=enriched_cfg["source"],
                name=enriched_cfg.get("name"),
                split=enriched_cfg["split"],
                seed=int(enriched_cfg.get("seed", config.get("seed", 0))),
                shuffle_buffer=int(enriched_cfg.get("shuffle_buffer", 0)),
            ),
            tokenizer,
            n_sequences=enriched_n,
            sequence_length=sequence_length,
        )
        tokens = np.concatenate([c4_tokens, enriched_tokens], axis=0)
    else:
        tokens = c4_tokens

    n_c4 = len(c4_tokens)
    metadata = pd.DataFrame(
        {
            "seq_idx": np.arange(len(tokens), dtype=np.int64),
            "source": [c4_cfg["source"]] * n_c4 + [enriched_cfg.get("source", "")] * enriched_n,
            "enriched": [False] * n_c4 + [True] * enriched_n,
            "source_sequence_idx": list(range(n_c4)) + list(range(enriched_n)),
            "seed": [int(c4_cfg["seed"])] * n_c4
            + [int(enriched_cfg.get("seed", config.get("seed", 0)))] * enriched_n,
        }
    )
    return PoolBuildResult(tokens=tokens, metadata=metadata)

