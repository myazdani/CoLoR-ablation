# Power + Integrated-Gain Follow-Up

## Executive Summary

This follow-up is post-hoc and does not overturn the pre-registered weak_or_ambiguous call. It only informs the forward decision about whether to spend GPU on a real conditional-seed ensemble.

The crude ensemble is biased structural perturbation, not a posterior; a positive result licenses testing the real ensemble, not a claim that uncertainty-aware selection works.

The enriched set is paper-pipeline-selected, not an independent gold standard; retention is a calibration diagnostic.

Forward call: **do_not_build_strict_call_stands**.

This does not change the registered pre-test call. The registered gate remains **weak_or_ambiguous**.

## Pre-Registered Rules for This Follow-Up

Analysis A was declared underpowered if `b + c <= 30` and the exact-CI MDE exceeded the log2-rate extrapolated expected effect. It was declared an informative null if `b + c > 30`, MDE was smaller than the expected effect, and the point estimate was approximately zero.

Analysis B used a single shared lambda across all rates. The primary statistic is `IG_retention(lambda*)` under `equal_log` weights, where `lambda*` maximizes integrated retention gain over positive lambdas only. A build call requires positive `mild` IG with CI excluding zero, recall not below zero beyond its bootstrap half-width, and `mild_no_full` matching the sign at the same lambda.

## Analysis A: 1/64 Power and Discordance

`mild` best positive lambda from the pre-test at 1/64: `0.25`.

| rate | b lost | c gained | b+c | diff | exact 95% CI | MDE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1/64 | 45 | 34 | 79 | -0.0022 | [-0.0057, 0.0015] | 0.0036 |
| 1/32 | 30 | 11 | 41 | -0.0038 | [-0.0059, -0.0012] | 0.0024 |

At 1/64, `b+c=79`, MDE=`0.0036`, and the primary log2-rate expected effect is `-0.0043`.
Analysis A status: **indeterminate**.

![Analysis A discordance](figures/followup_discordance_mild.png)

## Analysis B: Integrated Gain Across Rates

| ensemble | weighting | lambda* | IG retention | 95% CI | IG recall | 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mild | equal_log | 0.25 | -0.0026 | [-0.0035, -0.0010] | 0.0038 | [0.0019, 0.0060] |
| mild | tail_weighted | 0.25 | -0.0028 | [-0.0040, -0.0007] | 0.0024 | [0.0002, 0.0051] |
| mild_no_full | equal_log | 0.25 | 0.0005 | [-0.0009, 0.0019] | 0.0118 | [0.0093, 0.0140] |
| mild_no_full | tail_weighted | 0.25 | 0.0012 | [-0.0006, 0.0030] | 0.0108 | [0.0075, 0.0134] |
| mid | equal_log | 0.25 | -0.0027 | [n/a, n/a] | -0.0156 | [n/a, n/a] |
| mid | tail_weighted | 0.25 | -0.0027 | [n/a, n/a] | -0.0154 | [n/a, n/a] |
| wide | equal_log | 0.5 | 0.0101 | [n/a, n/a] | 0.0333 | [n/a, n/a] |
| wide | tail_weighted | 0.5 | 0.0131 | [n/a, n/a] | 0.0336 | [n/a, n/a] |

Primary `mild` equal-log lambda* is `0.25` with retention IG `-0.0026` and recall IG `0.0038`. Recall tolerance is the bootstrap half-width `0.0020`.
`mild_no_full` at the same lambda has equal-log retention IG `0.0005`.

![Integrated retention gain curves](figures/followup_ig_curves.png)

![Integrated gain vs recall guard](figures/followup_ig_recall_scatter.png)

## Combined Decision Table

| Analysis A (1/64) | Analysis B (`mild` IG) | mild_no_full IG sign | Forward call |
|---|---|---|---|
| indeterminate | negative | does not match | **do_not_build_strict_call_stands** |

## Interpretation

The follow-up does not justify GPU spend for the real ensemble at this depth. The strict call stands.

## Reproducibility

- Run hash: `6e52c70564ea1efaf07f6da0d0a780b0145648325abf02329577ff5346bb8353`
- Follow-up provenance: `results/pretest_followup_provenance.json`
- Bootstrap resamples pure-C4 and enriched rows separately and recomputes pure-C4 thresholds within each replicate.
- Analysis B uses one lambda shared across all rates; there is no per-rate argmax.
