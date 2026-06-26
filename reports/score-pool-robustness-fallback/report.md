# A Stress Test for Layer-Ablated CoLoR Scoring on Score-Defined Pools

## Abstract

CoLoR filtering selects data by comparing a conditional model against a marginal
or prior model. For the Books target, the paper-sign score is
`conditional_books_loss - prior_loss`, and lower values are treated as more
Books-like. This experiment asks whether that selection rule is robust when the
conditional and marginal auxiliary models are ablated. The motivation is
practical: if CoLoR filtering is only used as a heuristic for ranking candidate
training examples, then we may not need the exact full auxiliary models to
recover the same useful ordering.

We construct five diagnostic pools from a 500,000-sequence fallback C4 source
pool: random positives, hard positives, random negatives, hard negatives, and
tail negatives. Labels are defined by the full-model CoLoR score, then each
ablated model pair is evaluated on six balanced pairwise discrimination tasks.
The strongest ablated result is `pair_mid2`, which removes layers `[5, 6]` from
both models and reaches mean ROC AUC `0.8868`, average precision `0.8988`, and
balanced-rate F1 `0.8315`. Paired layer removal is consistently more robust
than asymmetric removal. The worst family is `cond_top_marg_bot`, where top
layers are removed from the conditional Books model while bottom layers are
removed from the marginal model; at larger removals this family falls to chance
or below chance.

The central takeaway is not that layer ablation is harmless. It is that the
ranking signal is most recoverable when both auxiliary models are degraded in a
matched way, especially around the middle layers. Asymmetric ablations can shift
the CoLoR score scale so severely that the original cutoff becomes unusable,
even when some ranking information remains after recalibration.

## Reader's Guide

This is a fallback-pool result, not the exact official 500K selected-index
recovery result. The completed run followed
`docs/SCORE_POOL_ROBUSTNESS_COLAB_RUNBOOK.md` and produced a 25,000-row
diagnostic pool with 5,000 rows in each of five score-defined pools. The exact
official-pool path is documented separately in
`docs/SCORE_POOL_ROBUSTNESS_OFFICIAL_500K_COLAB_RUNBOOK.md`, but it was not the
run analyzed here.

The report is written like a small workshop note: it prioritizes motivation,
method, interpretation, and failure modes over exhaustive tables.

## Background and Motivation

CoLoR filtering is a targeted data-selection method. The basic idea is to score
candidate C4 sequences by how much better they look under a conditional model
adapted toward a target distribution than under a marginal model. In the Books
case, the conditional model has been adapted toward Books-like data, while the
marginal model represents the broader prior. With the paper-sign convention,

```text
CoLoR score = conditional_books_loss - prior_loss
```

lower scores are better. A very low score means the conditional Books model
assigns the sequence relatively lower loss than the marginal model does, so the
sequence is more Books-like according to this heuristic.

The heuristic nature of this score is important. CoLoR is not an oracle label
for semantic Books membership. It is a model-derived ranking rule used to choose
data at scale. That suggests a robustness question: if we damage or simplify
the auxiliary models, does the score still separate the same useful examples
from nearby non-selected examples? If yes, then the selection procedure may be
less brittle than the exact full-model score would imply. If no, then the
particular conditional/marginal pair and its late-layer representation are
doing essential work.

The previous layer-ablation report studied overlap in selected sets when the
same layers were removed from both auxiliary models. This experiment broadens
the question. We explicitly test symmetric removals, asymmetric removals, and
one-sided removals on score-defined positive and negative pools.

## Hypothesis

The working hypothesis is:

> Because CoLoR selection is based on a heuristic score rather than a ground
> truth semantic label, ablated auxiliary models may preserve enough ranking
> information to remain useful for discrimination between selected and
> non-selected regions of the score distribution.

This hypothesis has two parts. The first is about ranking robustness: positives
should still receive lower ablated CoLoR scores than negatives. The second is
about calibration robustness: the original full-model cutoff should still be a
usable threshold after ablation. The experiment shows that these two properties
separate. Several variants retain nontrivial ranking signal, but their score
scale shifts enough that the original cutoff is no longer reliable.

## Experimental Design

### Source Pool

The fallback source pool contains `500,000` C4 sequences. The full unablated
Books conditional and marginal models were scored on this source pool. We then
used the resulting full CoLoR scores to define the diagnostic pools.

The final scored evaluation pool contains `25,000` sequences:

| pool | count | score rule | full-score range |
|:--|--:|:--|:--|
| random positive | 5,000 | random examples below the tau=64-like positive cutoff | `[-0.3251, 0.3444]`, mean `0.2263` |
| hard positive | 5,000 | examples just below the positive cutoff | `[0.2311, 0.3444]`, mean `0.2999` |
| random negative | 5,000 | random examples in the tau=64 to tau=32 band | `[0.3444, 0.4071]`, mean `0.3794` |
| hard negative | 5,000 | examples just above the positive cutoff | `[0.3444, 0.3899]`, mean `0.3691` |
| tail negative | 5,000 | random examples outside the positive set | `[0.3511, 3.1167]`, mean `0.7193` |

The fallback positive cutoff is `0.3444314003`. The fallback negative-band
cutoff is `0.4070763588`.

### Why These Pools?

The pools are designed to separate easy and hard cases.

Random positives and tail negatives are relatively easy: they often come from
well-separated parts of the full-score distribution. Hard positives and hard
negatives are difficult: they sit close to the decision boundary and ask whether
an ablated model can preserve the fine ordering near the original cutoff.
Random negatives occupy the intermediate band between the tau=64-like positive
tail and the tau=32-like broader selected region.

This means the same ablated score is evaluated against both coarse ranking
tasks and boundary-sensitive ranking tasks.

### Pairwise Tasks

We evaluate six balanced binary tasks. In each task, the positive class is one
of the positive pools, and the negative class is one of the negative pools:

| task | positive pool | negative pool | interpretation |
|:--|:--|:--|:--|
| `hp_vs_hn` | hard positive | hard negative | tight boundary test |
| `hp_vs_rn` | hard positive | random negative | hard positives against tau=32-band negatives |
| `hp_vs_tn` | hard positive | tail negative | hard positives against broad tail negatives |
| `rp_vs_hn` | random positive | hard negative | random positives against boundary negatives |
| `rp_vs_rn` | random positive | random negative | random positives against tau=32-band negatives |
| `rp_vs_tn` | random positive | tail negative | easiest broad separation test |

Each task has `5,000` positives and `5,000` negatives.

## Model Variants

The auxiliary models have 12 transformer blocks indexed from bottom to top as
layers `0` through `11`. We evaluate four broad ablation families:

| family | definition | intuition |
|:--|:--|:--|
| `baseline` | no removed layers | sanity check and noise floor |
| `paired` | remove the same layers from both conditional and marginal models | degrade both score components similarly |
| `cond_top_marg_bot` | remove top conditional layers and bottom marginal layers | intentionally asymmetric damage |
| `cond_bot_marg_top` | remove bottom conditional layers and top marginal layers | opposite asymmetric damage |
| `cond_only` | remove top layers only from the conditional Books model | test conditional-side sensitivity |
| `marg_only` | remove top layers only from the marginal model | test marginal-side sensitivity |

The paired family includes top, middle, and bottom deletions. The asymmetric and
one-sided families focus on top/bottom combinations because the question is
whether conditional and marginal model damage must be matched to preserve the
difference score.

## Metrics

For each pairwise task we compute:

| metric | what it measures |
|:--|:--|
| ROC AUC | ranking quality of `-ablated_color_score`; higher means positives tend to have lower ablated CoLoR scores |
| average precision | precision-recall ranking quality under balanced pairwise labels |
| F1 at original cutoff | thresholded classification using the original full-model positive cutoff |
| F1 at balanced rate | thresholded classification after choosing the ablated-score threshold that selects exactly the known number of positives |
| score correlations | Pearson and Spearman correlations between full and ablated scores |
| score shifts | how far ablated conditional, prior, and CoLoR scores move from full-model scores |

The difference between the two F1 columns is important. F1 at the original
cutoff tests whether the full-model threshold remains calibrated after ablation.
F1 at balanced rate tests whether the ranking remains useful after a simple
one-dimensional recalibration. If ROC AUC and balanced-rate F1 are good while
original-cutoff F1 is poor, the ablation has not destroyed all ordering
information, but it has moved the score scale.

## Sanity Checks

The `full` and `full_rescore` baselines both achieve perfect metrics:

| variant | ROC AUC | average precision | F1 at original cutoff | F1 at balanced rate |
|:--|--:|--:|--:|--:|
| `full` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `full_rescore` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

This is expected because labels are defined from the full-model score. It is
not a substantive result, but it is an important validation that the pool,
score direction, and metric code are aligned.

## Main Results

### Mean Metrics by Variant

The table below averages over all six pairwise tasks.

| variant | family | ROC AUC | avg. precision | F1 original cutoff | F1 balanced rate |
|:--|:--|--:|--:|--:|--:|
| `full` | baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `full_rescore` | baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `pair_mid2` | paired | 0.8868 | 0.8988 | 0.7336 | 0.8315 |
| `pair_top1` | paired | 0.8344 | 0.8426 | 0.7665 | 0.7732 |
| `pair_mid4` | paired | 0.8210 | 0.8259 | 0.3964 | 0.7684 |
| `pair_top2` | paired | 0.7948 | 0.7938 | 0.7425 | 0.7324 |
| `pair_top6` | paired | 0.7740 | 0.7792 | 0.6663 | 0.7193 |
| `marg_top1_only` | marg_only | 0.7730 | 0.7635 | 0.6683 | 0.7269 |
| `marg_top2_only` | marg_only | 0.7378 | 0.7327 | 0.6668 | 0.6896 |
| `pair_bot1` | paired | 0.7122 | 0.7047 | 0.6932 | 0.6689 |
| `marg_top4_only` | marg_only | 0.7035 | 0.6554 | 0.6667 | 0.6591 |
| `marg_top6_only` | marg_only | 0.6931 | 0.6562 | 0.6667 | 0.6554 |
| `pair_bot2` | paired | 0.6887 | 0.6791 | 0.6672 | 0.6499 |
| `pair_top4` | paired | 0.6781 | 0.6699 | 0.2825 | 0.6343 |
| `cond_top1_only` | cond_only | 0.6752 | 0.6681 | 0.0088 | 0.6322 |
| `cond_bot6_marg_top6` | cond_bot_marg_top | 0.6283 | 0.6295 | NaN | 0.5989 |
| `cond_bot1_marg_top1` | cond_bot_marg_top | 0.6214 | 0.6142 | NaN | 0.5991 |
| `cond_bot2_marg_top2` | cond_bot_marg_top | 0.6127 | 0.6117 | 0.0004 | 0.5937 |
| `cond_bot4_marg_top4` | cond_bot_marg_top | 0.6050 | 0.6038 | 0.0004 | 0.5816 |
| `cond_top2_only` | cond_only | 0.6040 | 0.6049 | 0.0004 | 0.5730 |
| `pair_bot4` | paired | 0.5730 | 0.5827 | 0.6503 | 0.5519 |
| `pair_bot6` | paired | 0.5564 | 0.5525 | 0.6654 | 0.5381 |
| `cond_top6_only` | cond_only | 0.5400 | 0.5467 | NaN | 0.5322 |
| `cond_top1_marg_bot1` | cond_top_marg_bot | 0.5287 | 0.5281 | 0.6667 | 0.5225 |
| `cond_top4_only` | cond_only | 0.5205 | 0.5260 | NaN | 0.5169 |
| `cond_top2_marg_bot2` | cond_top_marg_bot | 0.5074 | 0.5178 | 0.6666 | 0.5000 |
| `cond_top4_marg_bot4` | cond_top_marg_bot | 0.4870 | 0.5059 | 0.6667 | 0.4880 |
| `cond_top6_marg_bot6` | cond_top_marg_bot | 0.4570 | 0.4842 | 0.6667 | 0.4657 |

The best non-baseline result is `pair_mid2`. It wins every pairwise task among
ablated variants, including both boundary tasks and easier tail-negative tasks.

### Family-Level Summary

| family | ROC AUC | average precision | F1 original cutoff | F1 balanced rate |
|:--|--:|--:|--:|--:|
| `baseline` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `paired` | 0.7319 | 0.7329 | 0.6264 | 0.6868 |
| `marg_only` | 0.7269 | 0.7019 | 0.6671 | 0.6827 |
| `cond_bot_marg_top` | 0.6168 | 0.6148 | 0.0004 | 0.5933 |
| `cond_only` | 0.5849 | 0.5864 | 0.0060 | 0.5636 |
| `cond_top_marg_bot` | 0.4950 | 0.5090 | 0.6667 | 0.4941 |

This table is the clearest high-level result. Paired removals and marginal-only
top removals preserve the most ranking information. Conditional-only top
removals are much worse. The asymmetric `cond_top_marg_bot` family is the most
damaging and can invert the intended ranking.

## Figures

![ROC AUC by variant and pairwise task. The paired middle-layer deletion is the strongest ablated variant across all tasks.](figures/auc_by_variant_and_task.png)

![Average precision by variant and task. The same pattern as ROC AUC appears: paired and marginal-only variants dominate conditional-heavy asymmetric variants.](figures/ap_by_variant_and_task.png)

![F1 at the original full-model cutoff. Several ablations fail here because their score scale shifts away from the full-model tau=64-like threshold.](figures/f1_original_cutoff_by_variant.png)

![F1 after balanced-rate recalibration. This view shows the residual ranking information after choosing a threshold that selects the same number of positives as the task.](figures/f1_balanced_rate_by_variant.png)

![Color-score shift by variant. Large positive or negative shifts explain why original-cutoff F1 can collapse even when ranking metrics remain above chance.](figures/color_shift_by_variant.png)

![Ablated versus full CoLoR scores for a sampled subset. Deviations from the diagonal indicate score drift; the useful variants preserve ordering more than absolute calibration.](figures/ablated_vs_full_color_scatter_sample.png)

## Task Difficulty

Averaging over variants, the hardest tasks are the boundary tasks:

| task | ROC AUC | average precision | F1 original cutoff | F1 balanced rate |
|:--|--:|--:|--:|--:|
| `hp_vs_hn` | 0.6047 | 0.6003 | 0.5467 | 0.5876 |
| `hp_vs_rn` | 0.6148 | 0.6088 | 0.5484 | 0.5957 |
| `rp_vs_hn` | 0.6482 | 0.6484 | 0.6523 | 0.6212 |
| `rp_vs_rn` | 0.6565 | 0.6550 | 0.6544 | 0.6278 |
| `hp_vs_tn` | 0.7661 | 0.7663 | 0.5693 | 0.7256 |
| `rp_vs_tn` | 0.7842 | 0.7879 | 0.6792 | 0.7428 |

This is expected. Hard positives and hard negatives are both close to the
cutoff, so even small score drift can reorder them. Tail negatives are much
farther away from the selected region, so most variants keep enough signal to
separate them.

Among ablated variants, `pair_mid2` is the best variant for every task:

| task | best ablated variant | ROC AUC | average precision | F1 balanced rate |
|:--|:--|--:|--:|--:|
| `hp_vs_hn` | `pair_mid2` | 0.7848 | 0.7959 | 0.7218 |
| `hp_vs_rn` | `pair_mid2` | 0.8150 | 0.8254 | 0.7476 |
| `hp_vs_tn` | `pair_mid2` | 0.9890 | 0.9893 | 0.9602 |
| `rp_vs_hn` | `pair_mid2` | 0.8599 | 0.8867 | 0.7858 |
| `rp_vs_rn` | `pair_mid2` | 0.8796 | 0.9023 | 0.8060 |
| `rp_vs_tn` | `pair_mid2` | 0.9924 | 0.9933 | 0.9674 |

## Interpreting Calibration Failures

Several variants have NaN or near-zero F1 at the original cutoff. This is not a
missing-data problem. It means the original cutoff selected zero or almost zero
examples after ablation, so precision and F1 are undefined or essentially zero.

The pattern is informative:

| family | mean CoLoR shift | interpretation |
|:--|--:|:--|
| `paired` | -0.2127 | relatively small drift; rankings degrade but calibration is partly preserved |
| `marg_only` | -2.2663 | ablated CoLoR scores move downward because the prior side changes |
| `cond_only` | 2.2188 | ablated CoLoR scores move upward because the conditional side changes |
| `cond_bot_marg_top` | 3.5090 | strong upward shift; original cutoff often selects no positives |
| `cond_top_marg_bot` | -4.0859 | strong downward shift; original cutoff can select too broadly despite poor ranking |

This explains why original-cutoff F1 and balanced-rate F1 sometimes disagree.
Original-cutoff F1 is a joint test of ranking and calibration. Balanced-rate F1
mostly tests ranking after threshold recalibration. For a production filtering
pipeline, both matter. For a scientific question about whether the ablated
models still encode the same ordering, ROC AUC and balanced-rate F1 are more
diagnostic.

## Discussion

### 1. Middle-layer paired removal is surprisingly robust.

The strongest ablated result is `pair_mid2`, removing layers `[5, 6]` from both
auxiliary models. This variant retains strong separation even for boundary
tasks. Its mean ROC AUC is `0.8868`, and its balanced-rate F1 is `0.8315`. It is
not a drop-in replacement for the full model because the full model is perfect
by construction on these labels, but it is strong evidence that the CoLoR
ranking is not uniformly dependent on every layer.

### 2. Matched damage is safer than mismatched damage.

Paired ablations preserve the difference-score structure better than asymmetric
ablations. This fits the mechanics of CoLoR. The score is a difference between
two losses, so errors that affect both sides similarly can partially cancel.
Errors that affect only one side, or affect opposite parts of the two models,
can produce large shifts in the difference score.

### 3. The conditional side is more fragile than the marginal side.

Marginal-only top removals perform much better than conditional-only top
removals. `marg_top1_only` reaches mean ROC AUC `0.7730`, while
`cond_top1_only` reaches `0.6752`, and larger conditional-only removals fall
closer to chance. This suggests that the Books-conditional model's top layers
carry target-specific information that is important for the score.

### 4. The hardest examples are exactly the right stress test.

The lowest average AUCs are on `hp_vs_hn` and `hp_vs_rn`. These are the tasks
closest to the original cutoff, where score drift matters most. The model can
often still separate positives from tail negatives, but that is not enough for
data selection: the operational problem is deciding which examples near the
cutoff should remain selected.

### 5. Calibration and ranking should be reported separately.

The original cutoff is meaningful because it is the threshold the full model
would use. But after ablation, some variants shift the score scale by multiple
loss units. A threshold copied from the full model can fail even if the ordering
is partly preserved. Reporting both original-cutoff F1 and balanced-rate F1
keeps this distinction visible.

## Limitations

These results are best read as a controlled fallback-pool stress test.

First, labels are derived from the full model, not human annotation. The task is
therefore "recover the full-model heuristic" rather than "detect true Books
quality." This is appropriate for studying CoLoR-filter robustness, but it
does not validate semantic data quality directly.

Second, this is not the exact official selected-index pool. The run used a
fallback source pool of 500,000 C4 sequences and sampled 5,000 examples per
diagnostic pool. The exact official-pool recovery path should be run before
using these numbers as final evidence about the released CoLoR indices.

Third, the final evaluation pool is 25,000 examples, not the intended 5 x
100,000 official score-pool design. The smaller pool is enough to expose clear
patterns across ablation families, but it cannot fully characterize rare
failure modes in the tails.

Fourth, runtime and memory efficiency are not the primary metrics in this
report. The experiment evaluates score robustness after layer deletion. It does
not by itself prove an end-to-end throughput win for a production filtering
pipeline.

## Conclusion

The experiment supports a qualified version of the robustness hypothesis. The
CoLoR score is not robust to arbitrary auxiliary-model ablation, but it is
reasonably robust to matched, moderate ablations. The best variant,
`pair_mid2`, preserves enough ranking signal to separate score-defined positive
and negative pools with mean ROC AUC `0.8868`. Larger or asymmetric removals are
much riskier, especially when top conditional layers are removed.

For the next round, the most useful comparison is not "any ablation versus full"
but "matched middle-layer ablations versus exact official-pool labels." If
`pair_mid2` remains strong on the exact official 500K recovery run, then it
becomes a serious candidate for approximate CoLoR scoring. If it degrades
substantially there, then the fallback-pool result should be treated as an
optimistic diagnostic rather than a robust production finding.

## Reproducibility Appendix

Run type:

```text
fallback score-pool robustness
```

Source runbook:

```text
docs/SCORE_POOL_ROBUSTNESS_COLAB_RUNBOOK.md
```

Relevant local outputs:

```text
results/score-pool-robustness-fallback/fallback_pool_summary.json
results/score-pool-robustness-fallback/metrics_pairwise.csv
results/score-pool-robustness-fallback/score_shift_diagnostics.csv
reports/score-pool-robustness-fallback/report.md
reports/score-pool-robustness-fallback/report.html
reports/score-pool-robustness-fallback/figures/*.png
```

Core settings:

```text
source C4 sequences: 500,000
diagnostic pool rows: 25,000
sample size per pool: 5,000
positive tau: 64
negative tau: 32
fallback positive cutoff: 0.3444314003
fallback negative-band cutoff: 0.4070763588
score convention: conditional_books_loss - prior_loss
selection direction: lower CoLoR scores are better
pairwise tasks: hp_vs_hn, hp_vs_rn, hp_vs_tn, rp_vs_hn, rp_vs_rn, rp_vs_tn
```

Validation checks performed after bringing outputs back locally:

```text
metrics rows: 168
variants: 28
pairwise tasks: 6
expected coverage: complete
figures present: 6
Markdown report present: yes
HTML report present: yes
```

Known incomplete follow-up:

```text
The exact official 500K selected-index recovery run has not been executed in
this result bundle.
```
