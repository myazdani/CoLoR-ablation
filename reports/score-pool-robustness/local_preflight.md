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

Initial key facts from the fixed-chunk plan:

```text
sample rows: 500,000
unique c4 indices: 496,205
remote token files visible under full_data/c4: 25
remote token bytes visible: 47.69 GiB
remote token chunks inferred at uint16 x 512: 50,005,443
max requested c4_index: 339,236,143
index coverage ok: False
```

Initial interpretation:

Treating the visible `*.npy` token files as contiguous fixed 512-token chunks
does not cover the sampled C4 index range. The sampled score indices go up to
about `339M`, but the visible token files only infer about `50M` 512-token
chunks under that convention.

## CSV Sidecar Recovery Attempt

The visible `full_data/c4` release also contains `part-*.csv.gz` sidecars. These
sidecars map C4 document ids to token offsets in the raw `part-*.npy` token
streams. The recovery script was updated to support this mapping via:

```yaml
token_recovery:
  index_source: csv_sidecar
  recovery_locations: results/score-pool-robustness/token_recovery_locations.csv
```

Command:

```bash
python scripts/09_recover_score_pool_tokens.py \
  --config configs/score_pool_robustness.yaml \
  --force-rebuild-locations
```

Key facts from the sidecar plan:

```text
sample rows: 500,000
unique c4 indices: 496,205
sidecar matched c4 indices: 6,398
index coverage ok: False
remote sidecar files visible: 25
remote sidecar bytes visible: 1.51 GiB
remote token files visible: 25
remote token bytes visible: 47.69 GiB
needed token files for matched subset: 2
needed token bytes for matched subset: 3.82 GiB
```

Sidecar artifacts:

```text
results/score-pool-robustness/token_recovery_plan.json
results/score-pool-robustness/token_recovery_locations.csv
```

Interpretation:

The CSV-sidecar mapping confirms that the official `c4_index` values are
document ids for at least part of the release, but the public `full_data/c4`
tree only matches `6,398 / 496,205` unique sampled ids. Therefore the currently
visible HF files still cannot recover the exact five 100K official sampled
pools.

Therefore exact recovery of the existing five official sampled pools cannot
proceed from the currently visible HF token tree without locating additional
tokenized C4 sidecars/shards or confirming another missing index mapping.

## Local Asset Status

Local checkpoint/data directories are absent:

```text
assets/ missing
data/ missing
```

Available local disk is also too low for the visible token shard download:

```text
available disk: ~14 GiB
visible full_data/c4 token shard size: 47.69 GiB
```

This prevents local fallback execution as well, because fallback requires
converted Books conditional/marginal checkpoints and enough storage for a frozen
token pool plus score parquets.

## Required Next Execution Path

Use the fallback path from `TASK_ablation_robustness_score_pools.md` unless
additional tokenized C4 shards are located.

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
