# Pair-Mid2 Cascade With Full-Model Reranking

## Abstract

This report evaluates whether the `pair_mid2` layer-deletion scorer can reduce full-model CoLoR scoring cost through a two-stage cascade. The cascade ranks all rows with `pair_mid2`, keeps an expanded candidate set, and then uses full Books CoLoR scores only as a reranker inside that candidate set. The analysis is offline because the official 500K pool has already been scored by both the full model and `pair_mid2`; this lets us estimate recall, Jaccard overlap, and runtime tradeoffs without launching another GPU job.

Pair-mid2 is not useful as a standalone replacement, but it is useful as a cascade prefilter for small full-pool selection rates in the tested regime. The strongest passing setting is selection rate 1/64 at m=1.5, with 0.9736 recall and 1.1086x estimated speedup. The main failure case is the hard-positive vs hard-negative comparison, where unsaturated pair_mid2 candidate recall is weak.

## Background and Motivation

The prior official 500K robustness run showed that many layer-deleted variants preserve some CoLoR ranking signal but do not exactly reproduce the full-model selection. `pair_mid2`, which deletes the middle two layers from both marginal and conditional models, was one of the strongest layer-deletion approximations. That makes it a natural candidate for a cascade: use the cheap scorer to avoid running the full scorer everywhere, then let the full scorer make the final boundary decision on a smaller candidate set.

The central question is not whether `pair_mid2` can replace the full scorer. The useful question is whether it has enough recall at the top of the ranking to act as a prefilter. A prefilter can tolerate many false positives if the candidate set is still much smaller than the full pool and contains most of the full-model positives.

## Methods

For each pairwise pool task, positives and negatives are the same as in the score-pool robustness analysis. Each task has 100K positive rows and 100K negative rows. For a candidate multiplier `m`, we keep the lowest `m * k` rows by `pair_mid2` CoLoR score, where `k` is the number of positives, then rerank those candidates by the full-model CoLoR score and select the lowest `k` rows.

For the full-pool analysis, the reference set is the lowest `k` rows by the local full-model score over all 500K rows. Selection rates are `1/64`, `1/32`, `1/16`, and `1/8`. The candidate set is again selected by `pair_mid2`, and the final simulated selection is selected by full-model score inside the candidate set.

Runtime estimates use the measured official 500K elapsed times saved in the score parquet files. The end-to-end cascade estimate includes both the first-stage `pair_mid2` pass over all rows and the full-model rerank pass over the retained candidate fraction.

## Experimental Setup

- Dataset: official 500K score-pool sample with five 100K pools.
- Score sign: lower CoLoR score is better, matching the paper-sign convention.
- Reference full scorer: `scores_full.parquet`.
- Cheap prefilter scorer: `scores_pair_mid2.parquet`.
- Candidate multipliers: `1`, `1.25`, `1.5`, `2`, `3`, `4`, `8`.
- Full-pool selection rates: `1/64`, `1/32`, `1/16`, `1/8`.

## Measured Runtime Inputs

| label     | rows   | elapsed_seconds | tokens_per_second | relative_to_full |
| --------- | ------ | --------------- | ----------------- | ---------------- |
| full      | 500000 | 2325.4609       | 220171.4106       | 1.0000           |
| pair_mid2 | 500000 | 2043.1094       | 250598.4307       | 0.8786           |

## Full-Pool Cascade Results

The table below reports final overlap with the local full-model selected set after full reranking of the `pair_mid2` candidate set. End-to-end speedup includes the cost of scoring all rows with `pair_mid2`.

| selection_rate_label | multiplier | candidate_fraction | final_recall_vs_full | final_jaccard_vs_full | estimated_end_to_end_speedup_vs_full |
| -------------------- | ---------- | ------------------ | -------------------- | --------------------- | ------------------------------------ |
| 1/64                 | 1.0000     | 0.0156             | 0.8312               | 0.7111                | 1.1183                               |
| 1/64                 | 1.2500     | 0.0195             | 0.9278               | 0.8653                | 1.1134                               |
| 1/64                 | 1.5000     | 0.0234             | 0.9736               | 0.9486                | 1.1086                               |
| 1/64                 | 2.0000     | 0.0313             | 0.9956               | 0.9913                | 1.0991                               |
| 1/64                 | 3.0000     | 0.0469             | 0.9994               | 0.9987                | 1.0805                               |
| 1/64                 | 4.0000     | 0.0625             | 0.9995               | 0.9990                | 1.0626                               |
| 1/64                 | 8.0000     | 0.1250             | 0.9996               | 0.9992                | 0.9964                               |
| 1/32                 | 1.0000     | 0.0312             | 0.8564               | 0.7489                | 1.0991                               |
| 1/32                 | 1.2500     | 0.0391             | 0.9398               | 0.8864                | 1.0897                               |
| 1/32                 | 1.5000     | 0.0469             | 0.9742               | 0.9497                | 1.0805                               |
| 1/32                 | 2.0000     | 0.0625             | 0.9914               | 0.9829                | 1.0626                               |
| 1/32                 | 3.0000     | 0.0938             | 0.9953               | 0.9906                | 1.0285                               |
| 1/32                 | 4.0000     | 0.1250             | 0.9961               | 0.9922                | 0.9964                               |
| 1/32                 | 8.0000     | 0.2500             | 0.9976               | 0.9951                | 0.8861                               |
| 1/16                 | 1.0000     | 0.0625             | 0.8422               | 0.7275                | 1.0626                               |
| 1/16                 | 1.2500     | 0.0781             | 0.9093               | 0.8337                | 1.0453                               |
| 1/16                 | 1.5000     | 0.0938             | 0.9392               | 0.8853                | 1.0285                               |
| 1/16                 | 2.0000     | 0.1250             | 0.9627               | 0.9281                | 0.9964                               |
| 1/16                 | 3.0000     | 0.1875             | 0.9767               | 0.9544                | 0.9380                               |
| 1/16                 | 4.0000     | 0.2500             | 0.9814               | 0.9636                | 0.8861                               |
| 1/16                 | 8.0000     | 0.5000             | 0.9906               | 0.9814                | 0.7254                               |
| 1/8                  | 1.0000     | 0.1250             | 0.7409               | 0.5884                | 0.9964                               |
| 1/8                  | 1.2500     | 0.1562             | 0.7968               | 0.6622                | 0.9663                               |
| 1/8                  | 1.5000     | 0.1875             | 0.8346               | 0.7161                | 0.9380                               |
| 1/8                  | 2.0000     | 0.2500             | 0.8825               | 0.7897                | 0.8861                               |
| 1/8                  | 3.0000     | 0.3750             | 0.9334               | 0.8750                | 0.7977                               |
| 1/8                  | 4.0000     | 0.5000             | 0.9586               | 0.9204                | 0.7254                               |
| 1/8                  | 8.0000     | 1.0000             | 1.0000               | 1.0000                | 0.5323                               |

## Decision Against Criteria

| criterion                                                         | result | evidence                                   |
| ----------------------------------------------------------------- | ------ | ------------------------------------------ |
| m<=4 reaches >=95% full-reference recall                          | yes    | 12 full-pool settings pass                 |
| m<=4 reaches >=95% recall and estimated end-to-end speedup >1     | yes    | 7 full-pool settings pass                  |
| pair_mid2 direct classifier matches full-model behavior           | no     | mean direct F1=0.7656; mean full F1=0.9483 |
| candidate recall is strong on hp_vs_hn before pairwise saturation | no     | best unsaturated hp_vs_hn recall=0.7550    |

## Smallest Candidate Multipliers Reaching 95% Recall

| selection_rate_label | multiplier | candidate_fraction | final_recall_vs_full | final_jaccard_vs_full | estimated_end_to_end_speedup_vs_full |
| -------------------- | ---------- | ------------------ | -------------------- | --------------------- | ------------------------------------ |
| 1/16                 | 2.0000     | 0.1250             | 0.9627               | 0.9281                | 0.9964                               |
| 1/32                 | 1.5000     | 0.0469             | 0.9742               | 0.9497                | 1.0805                               |
| 1/64                 | 1.5000     | 0.0234             | 0.9736               | 0.9486                | 1.1086                               |
| 1/8                  | 4.0000     | 0.5000             | 0.9586               | 0.9204                | 0.7254                               |

## Pairwise Cascade Diagnostics

The direct `pair_mid2` metrics below show how useful the ablated score is as a standalone ranking signal before adding full-model reranking.

| pairwise_task | pair_mid2_roc_auc_vs_official | pair_mid2_average_precision_vs_official | pair_mid2_direct_f1_vs_official | full_direct_f1_vs_official |
| ------------- | ----------------------------- | --------------------------------------- | ------------------------------- | -------------------------- |
| hp_vs_hn      | 0.5098                        | 0.5073                                  | 0.5069                          | 0.7381                     |
| hp_vs_rn      | 0.6792                        | 0.6618                                  | 0.6378                          | 0.9784                     |
| hp_vs_tn      | 0.9852                        | 0.9826                                  | 0.9513                          | 0.9994                     |
| rp_vs_hn      | 0.8013                        | 0.8430                                  | 0.7269                          | 0.9796                     |
| rp_vs_rn      | 0.8780                        | 0.9021                                  | 0.8033                          | 0.9946                     |
| rp_vs_tn      | 0.9936                        | 0.9941                                  | 0.9675                          | 0.9997                     |

At `m=4`, pairwise tasks with only 200K rows are saturated: the candidate set contains all rows because `4 * k` exceeds the task size. These diagnostics are therefore most useful for showing task difficulty and final full-rerank behavior, not deployment speed.

| pairwise_task | candidate_recall_vs_full | final_recall_vs_full | cascade_f1_vs_official | pair_mid2_direct_f1_vs_official | full_direct_f1_vs_official | estimated_end_to_end_speedup_vs_full |
| ------------- | ------------------------ | -------------------- | ---------------------- | ------------------------------- | -------------------------- | ------------------------------------ |
| hp_vs_hn      | 1.0000                   | 1.0000               | 0.7381                 | 0.5069                          | 0.7381                     | 0.5323                               |
| hp_vs_rn      | 1.0000                   | 1.0000               | 0.9784                 | 0.6378                          | 0.9784                     | 0.5323                               |
| hp_vs_tn      | 1.0000                   | 1.0000               | 0.9994                 | 0.9513                          | 0.9994                     | 0.5323                               |
| rp_vs_hn      | 1.0000                   | 1.0000               | 0.9796                 | 0.7269                          | 0.9796                     | 0.5323                               |
| rp_vs_rn      | 1.0000                   | 1.0000               | 0.9946                 | 0.8033                          | 0.9946                     | 0.5323                               |
| rp_vs_tn      | 1.0000                   | 1.0000               | 0.9997                 | 0.9675                          | 0.9997                     | 0.5323                               |

## Figures

![Recall vs candidate multiplier](figures/recall_vs_candidate_multiplier.png)

![Jaccard vs candidate multiplier](figures/jaccard_vs_candidate_multiplier.png)

![Speedup vs recall](figures/speedup_vs_recall.png)

![Pairwise cascade F1 by task](figures/pairwise_cascade_f1_by_task.png)

## Interpretation

- The cascade is most attractive for small final selection rates such as `1/64`, where a `4x` candidate set is only `6.25%` of the full pool.
- Because `pair_mid2` itself costs about `0.879x` of a full pass in the measured official run, candidate reranking must be very small to produce an end-to-end speedup.
- Pairwise pool tasks are balanced at 100K positives and 100K negatives; for these diagnostics, multipliers `>=2` already include the whole pairwise task and cannot estimate deployment speed savings.
- The analysis is an offline simulation because full scores already exist for the official 500K pool. A production cascade would need to avoid full scoring outside the candidate set.

## Limitations

- These are score-level simulations, not measured candidate-only GPU runs.
- The official 500K pool is intentionally enriched around the score boundary and is not a random draw from all C4 chunks.
- `pair_mid2` was measured as only modestly faster than full scoring in the official run, so the practical value of the cascade depends on candidate fraction and hardware utilization.
- Pairwise tasks saturate once `m >= 2`, because each balanced pair has only twice as many rows as positives.

## Conclusion

Pair-mid2 is not useful as a standalone replacement, but it is useful as a cascade prefilter for small full-pool selection rates in the tested regime. The strongest passing setting is selection rate 1/64 at m=1.5, with 0.9736 recall and 1.1086x estimated speedup. The main failure case is the hard-positive vs hard-negative comparison, where unsaturated pair_mid2 candidate recall is weak.

## Reproducibility

Inputs:

```text
data/pair_mid2_cascade_full_rerank/results/score-pool-robustness-official-500k/scores_full.parquet
data/pair_mid2_cascade_full_rerank/results/score-pool-robustness-official-500k/scores_pair_mid2.parquet
```

Outputs:

```text
results/pair-mid2-cascade/cascade_pairwise_metrics.csv
results/pair-mid2-cascade/cascade_full_pool_metrics.csv
results/pair-mid2-cascade/cascade_runtime_estimates.csv
reports/pair-mid2-cascade/report.md
reports/pair-mid2-cascade/report.html
```
