# Sequence-Length Ablations for Faster CoLoR Scoring

## Abstract

This report evaluates whether scoring fewer than 512 tokens preserves enough Books-targeted CoLoR signal to accelerate score-pool classification. Each window uses the full conditional Books model and full marginal model, but computes mean next-token loss only on the selected token window.

No scored sequence window simultaneously meets the mean-quality and measured-speed criteria. Shorter windows may still be useful as first-stage prefilters if their top-of-ranking recall is acceptable in a follow-up cascade analysis.

## Background

Layer-deletion experiments showed that `pair_mid2` preserves some signal but gives limited measured speedup because it still runs two 10-layer models over all 512 tokens. Sequence-length reduction attacks a different cost axis: it keeps model depth intact while reducing the number of token positions scored.

## Methods

For each sequence window, the recovered official 500K token pool is sliced before scoring. Contiguous windows use `[start:end]`; the prefix+suffix window concatenates `[0:128]` and `[384:512]` in that order. For every row, CoLoR is computed as conditional Books NLL minus marginal/prior NLL, with lower scores treated as more Books-like.

Metrics use the same six balanced pairwise tasks as the score-pool robustness analysis. Ranking metrics use `decision_score = -color_score`. Runtime is read from the scorer output parquet metadata.

## Results Summary

| sequence_window_id    | effective_sequence_length | mean_roc_auc | mean_balanced_f1 | min_roc_auc | elapsed_seconds | speedup_vs_seq_full_512 | model_tokens_per_second |
| --------------------- | ------------------------- | ------------ | ---------------- | ----------- | --------------- | ----------------------- | ----------------------- |
| seq_full_512          | 512                       | 0.9679       | 0.9483           | 0.8159      | 2299.1236       | 1.0000                  | 222693.5552             |
| seq_middle_256        | 256                       | 0.7853       | 0.7415           | 0.5064      | 1138.7622       | 2.0190                  | 224805.5029             |
| seq_prefix_128        | 128                       | 0.7493       | 0.7048           | 0.5074      | 573.0017        | 4.0124                  | 223385.0119             |
| seq_prefix_256        | 256                       | 0.7931       | 0.7507           | 0.5108      | 1138.4518       | 2.0195                  | 224866.7802             |
| seq_prefix_suffix_256 | 256                       | 0.7983       | 0.7539           | 0.5103      | 1138.0953       | 2.0202                  | 224937.2319             |
| seq_suffix_128        | 128                       | 0.7347       | 0.6902           | 0.5034      | 572.8896        | 4.0132                  | 223428.7195             |
| seq_suffix_256        | 256                       | 0.7758       | 0.7324           | 0.5041      | 1138.3682       | 2.0197                  | 224883.3044             |

## Hard-Boundary Task

The hardest task is expected to be `hp_vs_hn`, because both pools sit close to the tau=64 selection boundary.

| sequence_window_id    | roc_auc | f1_at_balanced_rate | pearson_color | spearman_color |
| --------------------- | ------- | ------------------- | ------------- | -------------- |
| seq_full_512          | 0.8159  | 0.7381              | 0.6180        | 0.6249         |
| seq_middle_256        | 0.5064  | 0.5051              | 0.0118        | 0.0147         |
| seq_prefix_128        | 0.5074  | 0.5055              | 0.0108        | 0.0130         |
| seq_prefix_256        | 0.5108  | 0.5083              | 0.0147        | 0.0181         |
| seq_prefix_suffix_256 | 0.5103  | 0.5085              | 0.0185        | 0.0195         |
| seq_suffix_128        | 0.5034  | 0.5023              | 0.0075        | 0.0079         |
| seq_suffix_256        | 0.5041  | 0.5024              | 0.0099        | 0.0117         |

## Layer-Deletion Comparison

The Colab report was rendered without the local `scores_pair_mid2.parquet`
baseline path, so the plot generation did not include `pair_mid2` directly.
However, the official 500K score-pool robustness report gives the comparable
aggregate metrics for the strongest layer-deletion variant:

| method                         | mean_roc_auc | mean_balanced_f1 | hp_vs_hn_roc_auc | hp_vs_hn_balanced_f1 | measured_speedup |
| ------------------------------ | ------------ | ---------------- | ---------------- | -------------------- | ---------------- |
| seq_full_512                   | 0.9679       | 0.9483           | 0.8159           | 0.7381               | 1.0000           |
| pair_mid2 layer deletion       | 0.8079       | 0.7656           | 0.5558           | 0.5420               | about 1.14x      |
| seq_prefix_suffix_256          | 0.7983       | 0.7539           | 0.5103           | 0.5085               | 2.0202x          |
| seq_prefix_256                 | 0.7931       | 0.7507           | 0.5108           | 0.5083               | 2.0195x          |
| seq_prefix_128                 | 0.7493       | 0.7048           | 0.5074           | 0.5055               | 4.0124x          |

The best 256-token window is close to `pair_mid2` on mean ROC AUC and mean
balanced F1, but it collapses on the hard-positive vs hard-negative boundary.
That boundary is the most important stress test because both classes sit near
the tau=64 cutoff. In that setting, `pair_mid2` is still weak but meaningfully
above random, while sequence-length reductions are essentially at chance.

The tradeoff is speed: sequence-length reduction gives the expected near-linear
runtime gain, while `pair_mid2` gives better ranking quality with only modest
measured acceleration.

## Figures

![AUC by window and task](figures/auc_by_window_and_task.png)

![Balanced F1 by window and task](figures/f1_balanced_by_window_and_task.png)

![Speed-quality Pareto](figures/quality_vs_effective_tokens.png)

![Runtime by effective tokens](figures/runtime_vs_effective_tokens.png)

![Prefix 256 color scatter](figures/window_color_scatter_prefix256.png)

## Limitations

- This report requires actual `scores_<window>.parquet` files from a GPU run; missing windows are not imputed.
- Original tau=64 cutoff metrics are calibration diagnostics because shorter-window scores can shift scale.
- Runtime comparisons are only fair when windows are scored on the same hardware with comparable batch tuning.
- Strided optional windows are excluded unless explicitly scored.

## Conclusion

No scored sequence window simultaneously meets the mean-quality,
hard-boundary-quality, and measured-speed criteria. The best sequence-length
windows are fast but too lossy to serve as direct replacements for full CoLoR
scoring. Compared with sequence-length reduction, `pair_mid2` is the stronger
quality-preserving approximation, especially near the tau=64 cutoff, but it has
much weaker runtime savings. Shorter windows may still be useful as first-stage
prefilters only if a dedicated top-of-ranking recall analysis shows that they
retain enough candidates for a later full-model rerank.

## Reproducibility

Run sequence-window scoring on Colab, then compute metrics and render this report locally or in Colab:

```bash
python scripts/17_sequence_length_metrics_report.py --config configs/sequence_length_score_pool.yaml
```
