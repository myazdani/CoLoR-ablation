# Layer-Ablated Auxiliary Models for CoLoR-Filter Scoring

This report is a scaffold. Local code and CPU smoke tests are implemented, but
real checkpoint download, conversion, scoring, and ablation runs are deferred
until Hugging Face and Google Drive setup is complete.

## Executive Summary

Pending experiment execution.

## Methods

The study scores a frozen packed-token pool with the full Books conditional and
marginal auxiliary models, then repeats scoring after deleting identical
original layer indices from both models. The primary metric is recall overlap
with the full-model bottom-tail selected set at fixed selection rates.

## Experimental Setup

Pending real HF checkpoint identification and conversion.

## Results

Pending experiment execution.

## Limitations

Pending experiment execution.

## Reproducibility Appendix

Local smoke test:

```bash
python -m pytest
```

Deferred Colab commands are documented in `docs/HF_GPU_RUNBOOK.md`.

