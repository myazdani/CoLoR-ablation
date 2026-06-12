#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ablation import apply_paired_ablation, find_block_module_list, variant_layer_indices
from src.model_loading import load_causal_lm
from src.scoring import score_pair


GUTENBERG_URLS = [
    "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
    "https://www.gutenberg.org/files/158/158-0.txt",  # Emma
    "https://www.gutenberg.org/files/84/84-0.txt",  # Frankenstein
]


def set_local_caches() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    assets = ROOT / "assets"
    os.environ.setdefault("HF_HOME", str(assets / ".hf-cache"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(assets / ".hf-cache" / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(assets / ".hf-cache" / "datasets"))
    os.environ.setdefault("CACHED_PATH_CACHE_ROOT", str(assets / ".cached_path"))
    for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE", "CACHED_PATH_CACHE_ROOT"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def load_olmo_tokenizer(checkpoint_dir: str | Path, paper_code_path: str | Path):
    paper_code = Path(paper_code_path).resolve()
    if str(paper_code) not in sys.path:
        sys.path.insert(0, str(paper_code))
    from hf_olmo.tokenization_olmo_fast import OLMoTokenizerFast

    return OLMoTokenizerFast.from_pretrained(str(checkpoint_dir))


def strip_gutenberg_boilerplate(text: str) -> str:
    start_markers = ("*** START OF", "***START OF")
    end_markers = ("*** END OF", "***END OF")
    start = 0
    upper = text.upper()
    for marker in start_markers:
        idx = upper.find(marker)
        if idx >= 0:
            line_end = text.find("\n", idx)
            start = line_end + 1 if line_end >= 0 else idx
            break
    end = len(text)
    for marker in end_markers:
        idx = upper.find(marker)
        if idx >= 0:
            end = idx
            break
    return text[start:end]


def fetch_gutenberg_texts() -> Iterator[str]:
    for url in GUTENBERG_URLS:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        yield strip_gutenberg_boilerplate(response.text)


def iter_c4_texts(*, seed: int, shuffle_buffer: int) -> Iterator[str]:
    from datasets import load_dataset

    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    for row in dataset:
        text = row.get("text")
        if isinstance(text, str) and text:
            yield text


def pack_texts(
    texts: Iterable[str],
    tokenizer,
    *,
    n_sequences: int,
    sequence_length: int,
) -> np.ndarray:
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id")

    sequences: list[np.ndarray] = []
    token_buffer: list[int] = []
    cursor = 0
    for text in texts:
        token_buffer.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        token_buffer.append(eos_token_id)
        while len(token_buffer) - cursor >= sequence_length and len(sequences) < n_sequences:
            sequences.append(
                np.asarray(token_buffer[cursor : cursor + sequence_length], dtype=np.int32)
            )
            cursor += sequence_length
        if cursor > 100000:
            token_buffer = token_buffer[cursor:]
            cursor = 0
        if len(sequences) >= n_sequences:
            break
    if len(sequences) < n_sequences:
        raise RuntimeError(f"Only packed {len(sequences)} / {n_sequences} sequences")
    return np.stack(sequences, axis=0)


def compact_module_tree(model: torch.nn.Module, *, max_depth: int = 3) -> str:
    lines: list[str] = []
    for name, module in model.named_modules():
        depth = 0 if not name else name.count(".") + 1
        include = depth <= max_depth or ".blocks." in name
        if not include:
            continue
        label = name or "<root>"
        extra = ""
        if isinstance(module, torch.nn.ModuleList):
            extra = f" len={len(module)}"
        if hasattr(module, "layer_id"):
            extra += f" layer_id={getattr(module, 'layer_id')}"
        lines.append(f"{label}: {module.__class__.__name__}{extra}")
    return "\n".join(lines)


def summarize_scores(scores: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
    summary: dict[str, float] = {}
    for domain in ("gutenberg", "c4"):
        mask = labels == domain
        summary[f"{domain}_n"] = int(mask.sum())
        summary[f"{domain}_nll_cond_mean"] = float(scores.loc[mask, "nll_cond"].mean())
        summary[f"{domain}_nll_marg_mean"] = float(scores.loc[mask, "nll_marg"].mean())
        summary[f"{domain}_color_mean"] = float(scores.loc[mask, "color"].mean())
        summary[f"{domain}_color_std"] = float(scores.loc[mask, "color"].std(ddof=1))
    summary["color_mean_gap_gutenberg_minus_c4"] = (
        summary["gutenberg_color_mean"] - summary["c4_color_mean"]
    )
    return summary


def format_summary(summary: dict[str, float]) -> str:
    rows = []
    for key in sorted(summary):
        value = summary[key]
        if isinstance(value, int):
            rows.append(f"{key}: {value}")
        else:
            rows.append(f"{key}: {value:.6f}")
    return "\n".join(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Run local CPU validation for converted CoLoR checkpoints.")
    parser.add_argument("--paper-code", default="../color-filter-olmo")
    parser.add_argument("--cond-checkpoint", default="assets/hf/books_cond_hf")
    parser.add_argument("--marg-checkpoint", default="assets/hf/books_marg_hf")
    parser.add_argument("--n-per-domain", type=int, default=200)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--ablation-variant", default="top4")
    parser.add_argument("--c4-shuffle-buffer", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--report", default="reports/layer-ablated-color-filter/local_validation.md")
    args = parser.parse_args()

    set_local_caches()
    start = time.perf_counter()
    tokenizer = load_olmo_tokenizer(args.marg_checkpoint, args.paper_code)
    print(f"Tokenizer: {tokenizer.__class__.__name__}, eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}")

    print(f"Packing {args.n_per_domain} Gutenberg-ish sequences...")
    gutenberg = pack_texts(
        fetch_gutenberg_texts(),
        tokenizer,
        n_sequences=args.n_per_domain,
        sequence_length=args.sequence_length,
    )
    print(f"Packing {args.n_per_domain} C4 sequences...")
    c4 = pack_texts(
        iter_c4_texts(seed=args.seed, shuffle_buffer=args.c4_shuffle_buffer),
        tokenizer,
        n_sequences=args.n_per_domain,
        sequence_length=args.sequence_length,
    )
    pool = np.concatenate([gutenberg, c4], axis=0)
    labels = np.array(["gutenberg"] * len(gutenberg) + ["c4"] * len(c4))
    print(f"Packed pool shape={pool.shape}")

    print("Loading converted models on CPU...")
    cond = load_causal_lm(args.cond_checkpoint, paper_code_path=args.paper_code, dtype="fp32")
    marg = load_causal_lm(args.marg_checkpoint, paper_code_path=args.paper_code, dtype="fp32")
    cond_path, cond_blocks = find_block_module_list(cond)
    marg_path, marg_blocks = find_block_module_list(marg)
    if cond_path != marg_path:
        raise AssertionError(f"Block path mismatch: cond={cond_path}, marg={marg_path}")
    if len(cond_blocks) != len(marg_blocks):
        raise AssertionError("Conditional and marginal block counts differ")
    print(f"Block path: {cond_path}; blocks={len(cond_blocks)}")

    tree = compact_module_tree(cond)
    print("Module tree preview:")
    print("\n".join(tree.splitlines()[:80]))

    print("Scoring full models on CPU...")
    full_scores, full_stats = score_pair(
        cond,
        marg,
        pool,
        batch_size=args.batch_size,
        device="cpu",
        dtype="fp32",
    )
    for col in ("nll_cond", "nll_marg", "color"):
        if not np.isfinite(full_scores[col]).all():
            raise AssertionError(f"Non-finite full score column: {col}")
    if np.allclose(full_scores["nll_cond"], full_scores["nll_marg"]):
        raise AssertionError("theta_cond and theta_marg losses are identical")
    summary = summarize_scores(full_scores, labels)
    print(format_summary(summary))
    if summary["color_mean_gap_gutenberg_minus_c4"] >= 0:
        raise AssertionError("Expected Gutenberg-ish mean CoLoR score below C4 mean")
    if not (0 < full_scores[["nll_cond", "nll_marg"]].to_numpy().mean() < 50):
        raise AssertionError("Mean NLL is outside a broad sanity range")

    print(f"Applying paired ablation {args.ablation_variant} and scoring 8 real sequences...")
    removed = variant_layer_indices(args.ablation_variant, total_layers=len(cond_blocks))
    cond_record, marg_record = apply_paired_ablation(cond, marg, removed, total_layers=len(cond_blocks))
    ablated_scores, _ = score_pair(
        cond,
        marg,
        pool[:8],
        batch_size=min(args.batch_size, 8),
        device="cpu",
        dtype="fp32",
    )
    for col in ("nll_cond", "nll_marg", "color"):
        if not np.isfinite(ablated_scores[col]).all():
            raise AssertionError(f"Non-finite ablated score column: {col}")
    if np.allclose(ablated_scores["nll_cond"], ablated_scores["nll_marg"]):
        raise AssertionError("Ablated theta_cond and theta_marg losses are identical")

    elapsed = time.perf_counter() - start
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# Local Checkpoint Validation

## Setup

- Conditional checkpoint: `{args.cond_checkpoint}`
- Marginal checkpoint: `{args.marg_checkpoint}`
- Paper code: `{args.paper_code}`
- Tokenizer: `{tokenizer.__class__.__name__}`
- Sequence length: {args.sequence_length}
- Domain sequences: {args.n_per_domain} Gutenberg-ish, {args.n_per_domain} C4
- Full scoring batch size: {args.batch_size}
- Elapsed seconds: {elapsed:.2f}

## Module Tree

Detected block path: `{cond_path}`

```text
{tree}
```

## Full-Model Sanity

```text
{format_summary(summary)}
tokens_per_second: {full_stats["tokens_per_second"]:.3f}
```

## Ablated Forward Sanity

- Variant: `{args.ablation_variant}`
- Removed original layers: `{cond_record.removed_layers}`
- Conditional kept layers: {cond_record.kept_num_layers}
- Marginal kept layers: {marg_record.kept_num_layers}

```text
{ablated_scores[["seq_idx", "nll_cond", "nll_marg", "color"]].to_string(index=False)}
```
"""
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
