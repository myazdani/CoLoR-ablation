# Layer-Ablated Auxiliary Models for CoLoR-Filter Scoring

This report is a scaffold. Local code and CPU smoke tests are implemented, but
real checkpoint download, conversion, scoring, and ablation runs are deferred
until Hugging Face and Google Drive setup is complete.

## Executive Summary

Local checkpoint conversion and CPU validation are complete. The Books
conditional and marginal checkpoints load through the paper fork's OLMo wrapper,
the ablation block path is `model.transformer.blocks`, and the mini sanity check
separates Gutenberg-ish text from random C4 in the expected direction.

Full ablation-grid scoring is still pending.

## Methods

The study scores a frozen packed-token pool with the full Books conditional and
marginal auxiliary models, then repeats scoring after deleting identical
original layer indices from both models. The primary metric is recall overlap
with the full-model bottom-tail selected set at fixed selection rates.

## Experimental Setup

Local assets:

- Raw checkpoints: `assets/raw/models/prior`, `assets/raw/models/conditional_books`
- Converted checkpoints: `assets/hf/books_marg_hf`, `assets/hf/books_cond_hf`
- Local validation report: `reports/layer-ablated-color-filter/local_validation.md`

The local converter uses the paper fork's HF wrapper, but fixes two laptop
issues: CPU-safe `torch.load(..., map_location="cpu")` and repo-local cache
directories.

## Results

CPU validation on 200 Gutenberg-ish packed sequences and 200 random C4 packed
sequences:

| Domain | n | mean nll_cond | mean nll_marg | mean CoLoR |
|---|---:|---:|---:|---:|
| Gutenberg-ish | 200 | 3.277606 | 4.577743 | -1.300137 |
| C4 | 200 | 4.171903 | 3.509851 | 0.662052 |

Mean CoLoR gap, Gutenberg minus C4: `-1.962189`. Lower CoLoR is better, so this
confirms the checkpoint pair and loss computation are directionally sane before
running the ablation grid.

The `top4` ablated pair, removing original layers `(8, 9, 10, 11)` from both
models, produced finite non-identical conditional and marginal losses on 8 real
packed sequences.

## Limitations

This is only a local CPU validation. It does not estimate overlap@top-k,
rank-correlation degradation, or the Pareto curve; those require full-pool GPU
scoring.

## Reproducibility Appendix

Local smoke test:

```bash
python -m pytest
```

Deferred Colab commands are documented in `docs/HF_GPU_RUNBOOK.md`.
