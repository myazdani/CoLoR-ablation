# Score-Pool Robustness Fallback Report

Label source: `fallback_full_rescore_pool`.

Source C4 sequences: `500,000`
Sample size per pool: `5,000`

## Mean Metrics by Variant

```text
            variant  roc_auc  average_precision  f1_at_original_cutoff  f1_at_balanced_rate
       full_rescore   1.0000             1.0000                 1.0000               1.0000
               full   1.0000             1.0000                 1.0000               1.0000
          pair_mid2   0.8868             0.8988                 0.7336               0.8315
          pair_top1   0.8344             0.8426                 0.7665               0.7732
          pair_mid4   0.8210             0.8259                 0.3964               0.7684
          pair_top2   0.7948             0.7938                 0.7425               0.7324
          pair_top6   0.7740             0.7792                 0.6663               0.7193
     marg_top1_only   0.7730             0.7635                 0.6683               0.7269
     marg_top2_only   0.7378             0.7327                 0.6668               0.6896
          pair_bot1   0.7122             0.7047                 0.6932               0.6689
     marg_top4_only   0.7035             0.6554                 0.6667               0.6591
     marg_top6_only   0.6931             0.6562                 0.6667               0.6554
          pair_bot2   0.6887             0.6791                 0.6672               0.6499
          pair_top4   0.6781             0.6699                 0.2825               0.6343
     cond_top1_only   0.6752             0.6681                 0.0088               0.6322
cond_bot6_marg_top6   0.6283             0.6295                    NaN               0.5989
cond_bot1_marg_top1   0.6214             0.6142                    NaN               0.5991
cond_bot2_marg_top2   0.6127             0.6117                 0.0004               0.5937
cond_bot4_marg_top4   0.6050             0.6038                 0.0004               0.5816
     cond_top2_only   0.6040             0.6049                 0.0004               0.5730
          pair_bot4   0.5730             0.5827                 0.6503               0.5519
          pair_bot6   0.5564             0.5525                 0.6654               0.5381
     cond_top6_only   0.5400             0.5467                    NaN               0.5322
cond_top1_marg_bot1   0.5287             0.5281                 0.6667               0.5225
     cond_top4_only   0.5205             0.5260                    NaN               0.5169
cond_top2_marg_bot2   0.5074             0.5178                 0.6666               0.5000
cond_top4_marg_bot4   0.4870             0.5059                 0.6667               0.4880
cond_top6_marg_bot6   0.4570             0.4842                 0.6667               0.4657
```

## Hardest Pairwise Tasks

```text
pairwise_task  roc_auc  average_precision  f1_at_original_cutoff  f1_at_balanced_rate
     hp_vs_hn   0.6047             0.6003                 0.5467               0.5876
     hp_vs_rn   0.6148             0.6088                 0.5484               0.5957
     rp_vs_hn   0.6482             0.6484                 0.6523               0.6212
     rp_vs_rn   0.6565             0.6550                 0.6544               0.6278
     hp_vs_tn   0.7661             0.7663                 0.5693               0.7256
     rp_vs_tn   0.7842             0.7879                 0.6792               0.7428
```

## Top Variants by Mean ROC AUC

```text
       variant  roc_auc  average_precision  f1_at_original_cutoff  f1_at_balanced_rate
  full_rescore   1.0000             1.0000                 1.0000               1.0000
          full   1.0000             1.0000                 1.0000               1.0000
     pair_mid2   0.8868             0.8988                 0.7336               0.8315
     pair_top1   0.8344             0.8426                 0.7665               0.7732
     pair_mid4   0.8210             0.8259                 0.3964               0.7684
     pair_top2   0.7948             0.7938                 0.7425               0.7324
     pair_top6   0.7740             0.7792                 0.6663               0.7193
marg_top1_only   0.7730             0.7635                 0.6683               0.7269
marg_top2_only   0.7378             0.7327                 0.6668               0.6896
     pair_bot1   0.7122             0.7047                 0.6932               0.6689
```

## Lowest Variants by Mean ROC AUC

```text
            variant  roc_auc  average_precision  f1_at_original_cutoff  f1_at_balanced_rate
cond_bot4_marg_top4   0.6050             0.6038                 0.0004               0.5816
     cond_top2_only   0.6040             0.6049                 0.0004               0.5730
          pair_bot4   0.5730             0.5827                 0.6503               0.5519
          pair_bot6   0.5564             0.5525                 0.6654               0.5381
     cond_top6_only   0.5400             0.5467                    NaN               0.5322
cond_top1_marg_bot1   0.5287             0.5281                 0.6667               0.5225
     cond_top4_only   0.5205             0.5260                    NaN               0.5169
cond_top2_marg_bot2   0.5074             0.5178                 0.6666               0.5000
cond_top4_marg_bot4   0.4870             0.5059                 0.6667               0.4880
cond_top6_marg_bot6   0.4570             0.4842                 0.6667               0.4657
```

## Figures

- [ROC AUC](figures/auc_by_variant_and_task.png)
- [Average precision](figures/ap_by_variant_and_task.png)
- [F1 at original cutoff](figures/f1_original_cutoff_by_variant.png)
- [F1 at balanced rate](figures/f1_balanced_rate_by_variant.png)
- [Color shift](figures/color_shift_by_variant.png)

## Interpretation Notes

- Compare `f1_at_original_cutoff` against `f1_at_balanced_rate` to separate calibration shift from ranking preservation.
- Use `full_rescore`, when run, as the practical noise floor for score and metric drift.
- Treat these as fallback-pool results, not exact official sampled-pool results.
