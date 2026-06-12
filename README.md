# CoLoR Filter Layer Ablation

This repo studies whether deleting matching original layer indices from the two
CoLoR-Filter auxiliary models preserves the bottom-tail ranking used for data
selection.

The v1 target is Books only. The config schema is generic enough to point at any
conditional/prior checkpoint pair later, but this repo intentionally carries no
untested downstream-task placeholder.

## Local Smoke Test

The local smoke test uses tiny random models and synthetic packed token
sequences. It does not download Hugging Face assets and does not require a GPU.

```bash
cd /Users/myazdani/Documents/CI-CoLoR/color-filter-ablation
python -m pytest
```

## Local Checkpoint Validation

After downloading and converting the two Books checkpoints, run:

```bash
PYTHONPATH=../color-filter-olmo .venv/bin/python scripts/06_local_validation.py
```

This loads both converted models on CPU, confirms the ablation block path,
scores 200 Gutenberg-ish and 200 random C4 packed sequences, and runs one
ablated forward check on real packed sequences. The report is written to
`reports/layer-ablated-color-filter/local_validation.md`.

## Main Workflow

1. Build a frozen pool of packed 512-token sequences.
2. Score that exact pool with the full conditional/prior model pair.
3. Score the same pool with matching layer deletions applied to both models.
4. Compute overlap@top-k, rank correlations, diagnostics, and plots.

The enriched known-selected Books set is scored with every run but quarantined
from selection denominators. `pool_meta.parquet` contains the `enriched` column,
and metrics derive enriched counts from that metadata rather than from config.

## Deferred HF/GPU Execution

Real checkpoint download, conversion, and Colab/A100 scoring commands are
documented in [docs/HF_GPU_RUNBOOK.md](docs/HF_GPU_RUNBOOK.md). They are not
executed by the local smoke test.

## Artifact Policy

Do not commit checkpoints, packed pools, score parquet files, or full experiment
outputs. Keep those in Google Drive for v1. Commit code, configs, small metrics
CSVs, and selected figures once generated.
