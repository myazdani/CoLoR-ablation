# Uncertainty-Aware Scoring Pre-Test

## Executive Summary

The crude ensemble is biased structural perturbation, not a posterior, and a positive result only licenses building the real conditional-seed ensemble; it does not itself demonstrate that uncertainty-aware selection works.

The pre-test is **weak/ambiguous**: heteroscedasticity signals appear, but the sigma term does not convert into a strict selection gain.

The decision uses `mild` strictly and requires the same direction on `mid`. `wide` is reported only as a contaminated sanity bound; `mild_no_full` is a labeled non-decision diagnostic.

## Framing Caveats

- The crude ensemble is biased structural perturbation, not a posterior, and a positive result only licenses building the real conditional-seed ensemble; it does not itself demonstrate that uncertainty-aware selection works.
- The only non-destructive same-architecture scorings are `full` and `full_rescore`; in this run they are deterministic, so ablation spread is structural damage rather than bf16 jitter.
- The enriched Books set is paper-pipeline-selected, not an independent gold standard, so enriched retention is a calibration diagnostic rather than a standalone proof of selection quality.

## Inputs and Ensembles

- Provenance sidecar: `results/uncertainty_pretest_provenance.json`
- Score parquets: full, full_rescore, and 9 layer-ablation variants.
- Selection rates: 1/64, 1/32, 1/16, 1/8, 1/4

| ensemble | role | members |
| --- | --- | --- |
| mild | strict | full, mid2, top1 |
| mid | direction_only | full, mid2, mid4, top1 |
| wide | sanity_bound | full, top1, top2, mid2, bot2, top4, mid4, bot4, top6, skip2 |
| mild_no_full | diagnostic_only | mid2, top1 |

## Noise Floor

`full` vs `full_rescore`: max absolute color difference = `0`, mean = `0`, fraction > 1e-6 = `0`.
This confirms that the spread analyzed below is structural ablation spread, not a clean posterior sample or stochastic scoring noise.

## Registered Decision Rule

Positive iff, on `mild`, Test (a) threshold-band mean sigma exceeds deep-tail mean sigma with the difference CI excluding zero; Test (b) score-matched enriched sigma is lower than pure-C4 sigma with the difference CI excluding zero; and Test (c) the best positive-lambda retention at 1/64 beats lambda=0 with CI excluding zero and no same-rate pure-C4 recall loss. `mid` must have the same direction, but its CIs do not need to exclude zero.

## Decision Summary

| condition | result |
| --- | --- |
| Test (a) mild strict | pass |
| Test (b) mild strict | pass |
| Test (c) mild strict | fail |
| mid same direction | fail |
| go/no-go call | **weak_or_ambiguous** |

## Test (a): Does Spread Concentrate Near the Boundary?

| ensemble | threshold - deep sigma | 95% CI | threshold / deep sigma | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| mild | 0.0090 | [0.0065, 0.0115] | 1.1897 | [1.1317, 1.2534] |
| mid | -0.0054 | [-0.0084, -0.0025] | 0.9377 | [0.9061, 0.9701] |
| wide | 0.0420 | [0.0386, 0.0449] | 1.1650 | [1.1501, 1.1784] |
| mild_no_full | 0.0119 | [0.0091, 0.0151] | 1.2350 | [1.1721, 1.3117] |

![Spread vs full-score percentile: mild](figures/spread_vs_percentile_mild.png)

**Figure: spread vs full-score percentile for `mild`.** The shaded green band is the bottom 1% deep tail, the orange band is the 4-9% threshold region around the 1/16 cutoff, and the dashed line marks 6.25%.

![Spread vs full-score percentile: mid](figures/spread_vs_percentile_mid.png)

**Figure: spread vs full-score percentile for `mid`.** The shaded green band is the bottom 1% deep tail, the orange band is the 4-9% threshold region around the 1/16 cutoff, and the dashed line marks 6.25%.

![Spread vs full-score percentile: wide](figures/spread_vs_percentile_wide.png)

**Figure: spread vs full-score percentile for `wide`.** The shaded green band is the bottom 1% deep tail, the orange band is the 4-9% threshold region around the 1/16 cutoff, and the dashed line marks 6.25%.

![Spread vs full-score percentile: mild_no_full](figures/spread_vs_percentile_mild_no_full.png)

**Figure: spread vs full-score percentile for `mild_no_full`.** The shaded green band is the bottom 1% deep tail, the orange band is the 4-9% threshold region around the 1/16 cutoff, and the dashed line marks 6.25%.

## Test (b): Is the Enriched Good Tail Low-Spread After Score Matching?

| ensemble | band | enriched - pure matched sigma | 95% CI | matched n |
| --- | --- | ---: | ---: | ---: |
| mild | below_mu_1_16_pure_cutoff | -0.0044 | [-0.0060, -0.0029] | 2439 |
| mild | below_mu_1pct_pure_cutoff | -0.0048 | [-0.0069, -0.0028] | 1000 |
| mid | below_mu_1_16_pure_cutoff | 0.0017 | [-0.0001, 0.0036] | 2538 |
| mid | below_mu_1pct_pure_cutoff | -0.0057 | [-0.0081, -0.0032] | 1000 |
| wide | below_mu_1_16_pure_cutoff | -0.0219 | [-0.0240, -0.0198] | 2755 |
| wide | below_mu_1pct_pure_cutoff | -0.0167 | [-0.0200, -0.0135] | 1000 |
| mild_no_full | below_mu_1_16_pure_cutoff | -0.0105 | [-0.0123, -0.0086] | 2572 |
| mild_no_full | below_mu_1pct_pure_cutoff | -0.0075 | [-0.0103, -0.0049] | 1000 |

![Score-matched sigma: mild](figures/score_matched_sigma_mild.png)

**Figure: score-matched sigma for `mild`.** Bars compare matched pure-C4 and enriched Books mean sigma inside deep-tail score bands. Negative enriched-minus-pure differences support the claim that known-good rows are stably low-scoring beyond score level alone.

![Score-matched sigma: mid](figures/score_matched_sigma_mid.png)

**Figure: score-matched sigma for `mid`.** Bars compare matched pure-C4 and enriched Books mean sigma inside deep-tail score bands. Negative enriched-minus-pure differences support the claim that known-good rows are stably low-scoring beyond score level alone.

![Score-matched sigma: wide](figures/score_matched_sigma_wide.png)

**Figure: score-matched sigma for `wide`.** Bars compare matched pure-C4 and enriched Books mean sigma inside deep-tail score bands. Negative enriched-minus-pure differences support the claim that known-good rows are stably low-scoring beyond score level alone.

![Score-matched sigma: mild_no_full](figures/score_matched_sigma_mild_no_full.png)

**Figure: score-matched sigma for `mild_no_full`.** Bars compare matched pure-C4 and enriched Books mean sigma inside deep-tail score bands. Negative enriched-minus-pure differences support the claim that known-good rows are stably low-scoring beyond score level alone.

## Test (c): Does the Variance Term Carry Quality Signal?

| ensemble | best positive lambda at 1/64 | retention diff vs lambda=0 | 95% CI | pure-C4 recall diff vs lambda=0 |
| --- | ---: | ---: | ---: | ---: |
| mild | 0.25 | -0.0022 | [-0.0056, 0.0012] | -0.0006 |
| mid | 0.25 | 0.0008 | [-0.0034, 0.0048] | -0.0154 |
| wide | 1 | 0.0236 | [0.0140, 0.0326] | 0.0301 |
| mild_no_full | 0.5 | 0.0120 | [0.0062, 0.0178] | 0.0058 |

![Retention vs lambda: mild](figures/retention_vs_lambda_mild.png)

**Figure: retention vs lambda for `mild`.** Lambda=0 is the point-estimate ensemble ranking by mu. Positive lambda penalizes high sigma; negative lambda is the optimism control.

![Retention vs recall: mild](figures/retention_vs_recall_mild.png)

**Figure: retention-vs-recall tradeoff for `mild`.** This checks whether any enriched-retention gain is bought by losing agreement with the full pure-C4 selected set.

![Retention vs lambda: mid](figures/retention_vs_lambda_mid.png)

**Figure: retention vs lambda for `mid`.** Lambda=0 is the point-estimate ensemble ranking by mu. Positive lambda penalizes high sigma; negative lambda is the optimism control.

![Retention vs recall: mid](figures/retention_vs_recall_mid.png)

**Figure: retention-vs-recall tradeoff for `mid`.** This checks whether any enriched-retention gain is bought by losing agreement with the full pure-C4 selected set.

![Retention vs lambda: wide](figures/retention_vs_lambda_wide.png)

**Figure: retention vs lambda for `wide`.** Lambda=0 is the point-estimate ensemble ranking by mu. Positive lambda penalizes high sigma; negative lambda is the optimism control.

![Retention vs recall: wide](figures/retention_vs_recall_wide.png)

**Figure: retention-vs-recall tradeoff for `wide`.** This checks whether any enriched-retention gain is bought by losing agreement with the full pure-C4 selected set.

![Retention vs lambda: mild_no_full](figures/retention_vs_lambda_mild_no_full.png)

**Figure: retention vs lambda for `mild_no_full`.** Lambda=0 is the point-estimate ensemble ranking by mu. Positive lambda penalizes high sigma; negative lambda is the optimism control.

![Retention vs recall: mild_no_full](figures/retention_vs_recall_mild_no_full.png)

**Figure: retention-vs-recall tradeoff for `mild_no_full`.** This checks whether any enriched-retention gain is bought by losing agreement with the full pure-C4 selected set.

## Interpretation

The proxy-disturbance pre-test finds heteroscedasticity signals but does not show that the sigma term improves selection under the strict rule. This argues against building the real ensemble immediately at 12-layer depth unless the goal is exploratory.

## Reproducibility Appendix

- Results CSV: `results/uncertainty_pretest.csv`
- Provenance JSON: `results/uncertainty_pretest_provenance.json`
- Bootstrap CIs use paired row-level bootstraps where applicable.
- Binned matching controls for score level using ensemble `mu` bins inside the stated deep-tail score bands.
- All selection thresholds for retention are computed on pure-C4 rows only; enriched rows are never used to set thresholds.
- Full decision details, input hashes, and run parameters are in the provenance JSON sidecar.
