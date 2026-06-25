# Score-Pool Robustness Local Preflight

Generated: 2026-06-25

## Summary

The local implementation for the score-pool robustness task is in place, but
full execution is blocked locally by missing model/token assets and disk/GPU
constraints.

Implemented locally:

- Independent conditional/marginal layer ablation via `apply_dual_ablation`.
- Score-pool config at `configs/score_pool_robustness.yaml`.
- Loader for the five existing 100K score pools from
  `../books-color-score-reproduction/artifacts/pool_analysis`.
- Token-recovery preflight script.
- Resumable per-variant scoring script.
- Pairwise classification metrics script.
- Plotting script for pairwise robustness metrics.
- Unit tests for asymmetric ablation, pool loading, pairwise metrics, and token
  recovery planning.

Verified:

```text
python -m py_compile ...
python -m pytest
```

Result:

```text
14 passed, 1 skipped
```

## Exact Token Recovery Preflight

Command:

```bash
python scripts/09_recover_score_pool_tokens.py \
  --config configs/score_pool_robustness.yaml
```

Output plan:

```text
results/score-pool-robustness/token_recovery_plan.json
```

Corrected key facts from the fixed-chunk plan:

```text
sample rows: 500,000
unique c4 indices: 496,205
remote token files visible under full_data/c4: 170
remote token bytes visible: 323.52 GiB
remote token chunks inferred at uint16 x 512: 339,236,242
max requested c4_index: 339,236,143
index coverage ok: True
needed token files: 170
needed token bytes: 323.52 GiB
```

Interpretation:

Exact recovery of the five official 100K sampled pools is possible from the
visible public HF tree if we can download the token streams. The earlier
`25 *.npy` / `25 *.csv.gz` count was an API pagination bug: the first HF tree
page has 50 files, but the full directory has 340 files:

```text
170 *.npy token files
170 *.csv.gz sidecars
```

The score-pool `c4_index` values are packed-token chunk indices in the original
scoring dataset, so the correct recovery path is `index_source:
contiguous_token_chunks`, not the CSV-sidecar document-id path.

## Local Asset Status

Local checkpoint/data directories are absent:

```text
assets/ missing
data/ missing
```

Available local disk is also too low for the visible token shard download:

```text
available disk: ~11 GiB
visible full_data/c4 token shard size: 323.52 GiB
```

This prevents local exact recovery. Colab exact recovery requires enough local
scratch capacity for the 323.52 GiB token cache, plus Drive capacity for the
recovered 500K-row token pool and score outputs.

## Required Next Execution Path

Preferred path, if Drive storage allows:

1. Run exact token recovery on Colab/Drive:

```bash
python scripts/09_recover_score_pool_tokens.py \
  --config configs/score_pool_robustness.yaml \
  --download \
  --allow-large-download
```

2. Score the official recovered 500K token pool with `scripts/10_score_pool_variant.py`.
3. Compute metrics and plots with `scripts/11_score_pool_metrics.py` and
   `scripts/12_score_pool_plots.py`.

Fallback path, if the 323.52 GiB token download is not practical:

Fallback execution should happen on Colab/Drive:

1. Mount Drive and clone the pinned repo.
2. Download or mount converted Books conditional and marginal checkpoints.
3. Build a new frozen packed 512-token pool.
4. Score that pool with full unablated models.
5. Define positive/negative pools from the full unablated CoLoR scores.
6. Run the same ablation variant grid.
7. Compute pairwise classification metrics and plots.
8. Save outputs under:

```text
results/score-pool-robustness-fallback/
reports/score-pool-robustness-fallback/
```

Do not mix fallback results with the official sampled-pool results.
