# Layer-Ablated Auxiliary Models for CoLoR-Filter Scoring

## Executive Summary

The ablation study does **not** meet the pre-registered efficiency benchmark of
removing at least 33% of layers while preserving at least 80% overlap with the
full-model selected set at the 1/16 selection threshold. The best overlap point is
`mid2`, which drops 2/12 layers, reaches
1.13x measured speedup, and preserves 75.5%
of the full-model top-1/16 selected pure-C4 set. The strongest speed point is
`skip2`, which drops 6/12 layers and reaches
1.53x speedup, but preserves only 42.6%
overlap at the same threshold.

The Pareto frontier is `mid2 -> mid4 -> skip2`. This is a modest efficiency curve,
not a headline-positive ablation result. The important qualitative finding is that
middle-layer deletion is much less damaging than bottom- or top-layer deletion for
Books CoLoR-filter ranking.

## Experimental Setup

- Target: Books CoLoR-filter auxiliary pair.
- Pool: 105,000 packed 512-token sequences: 100,000 pure C4 and 5,000 enriched
  known-selected Books sequences.
- Metrics exclude enriched rows from overlap/Jaccard denominators and use enriched
  rows only as tail-retention diagnostics.
- Ground truth: full conditional/marginal pair scored in bf16 on A100.
- Noise floor: `full_rescore` was deterministic against `full` in this run.
- Primary threshold: selection rate 1/16 (`k = 6250` pure-C4 sequences).
- Score convention: lower CoLoR scores are better, so selection always means the
  bottom tail of the score distribution.

The enriched Books examples are deliberately quarantined. They are scored in
every run, but they are excluded from the denominators for overlap and Jaccard.
Their only role is diagnostic: after a threshold is computed from the pure-C4
pool, the same threshold is applied to the enriched rows to ask whether known-good
Books-like examples would still be kept.

## Variant Definitions

All ablations remove the same original layer indices from both the conditional
and marginal auxiliary models. The 12-layer model is indexed from bottom to top as
layers 0 through 11.

| variant | removed original layers | interpretation |
| --- | --- | --- |
| top1 | 11 | remove the final block |
| top2 | 10, 11 | remove the top two blocks |
| top4 | 8, 9, 10, 11 | remove the top third |
| top6 | 6, 7, 8, 9, 10, 11 | remove the top half |
| mid2 | 5, 6 | remove two middle blocks |
| mid4 | 4, 5, 6, 7 | remove the middle third |
| bot2 | 0, 1 | remove the bottom two blocks |
| bot4 | 0, 1, 2, 3 | remove the bottom third |
| skip2 | 1, 3, 5, 7, 9, 11 | remove every other block starting at layer 1 |

## Primary Results at 1/16

| variant | layers_removed | layers_drop_pct | speedup | recall_1_16 | jaccard_1_16 | global_spearman | enriched_1_16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| top1 | 1 | 8.333 | 1.061 | 0.541 | 0.371 | 0.737 | 0.853 |
| top2 | 2 | 16.667 | 1.131 | 0.385 | 0.239 | 0.534 | 0.721 |
| mid2 | 2 | 16.667 | 1.131 | 0.755 | 0.607 | 0.890 | 0.958 |
| bot2 | 2 | 16.667 | 1.131 | 0.317 | 0.189 | 0.485 | 0.529 |
| top4 | 4 | 33.333 | 1.301 | 0.220 | 0.123 | 0.155 | 0.403 |
| mid4 | 4 | 33.333 | 1.302 | 0.582 | 0.410 | 0.715 | 0.925 |
| bot4 | 4 | 33.333 | 1.303 | 0.175 | 0.096 | 0.329 | 0.181 |
| top6 | 6 | 50.000 | 1.532 | 0.392 | 0.244 | 0.302 | 0.738 |
| skip2 | 6 | 50.000 | 1.534 | 0.426 | 0.271 | 0.514 | 0.707 |

No variant satisfies the benchmark. Variants removing 4 or more layers all fall
below 60% recall overlap at 1/16. `mid2` is closest to preserving selection
behavior but removes only 16.7% of layers.

![Overlap across selection thresholds](figures/overlap_by_selection_rate.png)

**Figure 1. Overlap with the full-model selected set across selection rates.**
Each line shows recall of the ablated model's selected pure-C4 set against the
full model's selected pure-C4 set at the same selection rate. The primary
threshold is 1/16. Higher is better. The figure shows that `mid2` is consistently
the least destructive ablation, while top and bottom removals degrade selected-set
identity much more sharply. None of the variants that remove at least four layers
approach the 80% overlap target at 1/16.

## Pareto Analysis

| variant | layers_dropped | speedup | recall_1_16 | jaccard_1_16 | spearman | enriched_1_16 |
| --- | --- | --- | --- | --- | --- | --- |
| mid2 | 2 | 1.131 | 0.755 | 0.607 | 0.890 | 0.958 |
| mid4 | 4 | 1.302 | 0.582 | 0.410 | 0.715 | 0.925 |
| skip2 | 6 | 1.534 | 0.426 | 0.271 | 0.514 | 0.707 |

![Speed/overlap Pareto](figures/pareto_speedup_recall_1_16.png)

**Figure 2. Measured speedup versus recall at the primary 1/16 threshold.**
This is the main efficiency tradeoff plot. Points farther right are faster, and
points higher preserve more of the full-model selected set. The useful frontier is
`mid2 -> mid4 -> skip2`: each step buys more speed at a substantial quality cost.
The pre-registered target, removing at least 33% of layers while preserving at
least 80% overlap, is not reached.

The useful frontier has three regimes:

- `mid2`: best quality preservation, 75.5% recall at 1/16, 1.13x speedup.
- `mid4`: larger compute saving, but recall drops to 58.2%.
- `skip2`: fastest frontier point, but recall drops to 42.6%.

## Overlap Across Thresholds

| variant | 1/64 | 1/32 | 1/16 | 1/8 | 1/4 |
| --- | --- | --- | --- | --- | --- |
| top1 | 0.654 | 0.587 | 0.541 | 0.565 | 0.640 |
| top2 | 0.444 | 0.396 | 0.385 | 0.404 | 0.493 |
| mid2 | 0.780 | 0.767 | 0.755 | 0.768 | 0.806 |
| bot2 | 0.211 | 0.268 | 0.317 | 0.397 | 0.506 |
| top4 | 0.175 | 0.200 | 0.220 | 0.255 | 0.341 |
| mid4 | 0.609 | 0.584 | 0.582 | 0.602 | 0.660 |
| bot4 | 0.103 | 0.126 | 0.175 | 0.268 | 0.411 |
| top6 | 0.392 | 0.395 | 0.392 | 0.398 | 0.448 |
| skip2 | 0.426 | 0.425 | 0.426 | 0.453 | 0.527 |

`mid2` is consistently the best overlap-preserving ablation across thresholds,
ranging from 75.5% recall at 1/16 to 80.6% at 1/4. `top1` is competitive only at
the strictest 1/64 threshold and degrades quickly as k grows. Bottom-layer removal
is poor across the board.

## Enriched Tail Retention

The full model selects 98.4% of enriched known-selected Books examples
under the 1/16 pure-C4 threshold. Several ablations have much lower pure-C4 set
overlap while still retaining many enriched examples: `mid4` retains 92.5% and
`mid2` retains 95.8%. This means enriched-tail calibration can remain acceptable
even when pure-C4 selected-set identity changes substantially.

![Known-good Books tail retention](figures/enriched_retention_1_16.png)

**Figure 3. Known-good Books tail retention at the 1/16 pure-C4 threshold.**
For each scoring run, the threshold is first determined on the 100,000 pure-C4
rows only. The plot then reports what fraction of the 5,000 enriched Books rows
fall below that threshold. The dashed line is the full-model baseline:
`full_enriched_selected_frac = 0.9844`, meaning 98.44% of enriched Books examples
are selected by the full model. `mid2` and `mid4` preserve this diagnostic better
than their pure-C4 overlap numbers alone would suggest, while `bot4` nearly
destroys the known-good tail.

## Decomposition

The cancellation hypothesis is not strongly supported by these results. For most
variants, the CoLoR difference-score Spearman correlation is lower than the
individual `nll_cond` and `nll_marg` correlations. The main exceptions are bottom
ablations, where individual NLLs are already badly damaged and the difference is
slightly less bad. Mechanistically, deletion usually perturbs the difference score
more than it preserves it.

![Correlation decomposition](figures/correlation_decomposition_1_16.png)

**Figure 4. Spearman correlation decomposition against the full model.**
Blue bars compare ablated CoLoR scores with full CoLoR scores. Orange and green
bars compare the component conditional and marginal NLLs. If the
difference-of-losses cancellation hypothesis were strongly supported, the blue
bars would often be higher than both component bars. Instead, the CoLoR score is
usually less stable than the individual NLLs. The bottom-layer variants are the
main exception, but only because their component NLLs are already badly damaged.

## Score Scatter Diagnostics

Representative full-vs-variant score scatters are below. `mid2` shows a mostly
monotone relationship with substantial noise; `top4` and `bot4` visibly lose the
full-model ordering.

![Full vs mid2](figures/score_scatter_mid2.png)

**Figure 5. Full score versus `mid2` score.** This is the best quality-preserving
ablation. The cloud remains mostly monotone, which matches its high global
Spearman correlation and explains why its overlap is the best among tested
variants.

![Full vs mid4](figures/score_scatter_mid4.png)

**Figure 6. Full score versus `mid4` score.** The middle-third deletion keeps a
visible positive relationship, but the spread is wide enough to replace many of
the full-model 1/16 selections.

![Full vs skip2](figures/score_scatter_skip2.png)

**Figure 7. Full score versus `skip2` score.** This is the fastest Pareto point.
It preserves some global rank signal, but the noise is large, so selection overlap
falls to 42.6% at 1/16.

![Full vs top4](figures/score_scatter_top4.png)

**Figure 8. Full score versus `top4` score.** Removing the top third badly
distorts scores. This supports the metric table result that top-layer deletion is
not a useful 4-layer ablation for this auxiliary pair.

![Full vs bot4](figures/score_scatter_bot4.png)

**Figure 9. Full score versus `bot4` score.** Removing the bottom third is the
most damaging of the 4-layer variants. The scatter has weak rank structure and
the enriched-tail retention falls far below the middle-layer variants.

## Conclusions

1. The preregistered benchmark is missed: no 33%+ layer ablation reaches 80% overlap
   at the primary 1/16 selection threshold.
2. The best practical point is `mid2`: 2 layers dropped, 1.13x measured speedup,
   75.5% overlap, and 95.8% enriched-tail retention.
3. The best 4-layer point is `mid4`: 1.30x measured speedup, but only 58.2% overlap.
4. Top and bottom deletion are substantially worse than middle deletion for ranking
   preservation. Bottom deletion is especially damaging.
5. Difference-of-loss cancellation is not robust enough here to preserve ranking
   under aggressive layer removal in 12-layer auxiliary models.

## Limitations and Next Steps

- These results are Books-only. The downstream pair remains useful as a generality
  check, especially because the failure mode may be task-specific.
- The 12-layer model may be too shallow for large safe deletions. The cancellation
  hypothesis may be more plausible for deeper auxiliary models.
- If compute savings are still desired, prioritize `mid2` and potentially test
  narrower middle-layer variants or cheaper inference optimizations that do not
  delete whole blocks.

## Reproducibility Appendix

Input artifacts were read from `/Users/myazdani/Downloads/results`.

- `metrics.csv`: 55 rows, 33 columns.
- Score parquets: full, full_rescore, and 9 ablated variants.
- Primary metric: recall overlap with the full-model bottom-tail selected pure-C4 set.
- Primary selection rate: 1/16.
- Enriched Books examples are excluded from overlap/Jaccard denominators.
