# Local Checkpoint Validation

## Setup

- Conditional checkpoint: `assets/hf/books_cond_hf`
- Marginal checkpoint: `assets/hf/books_marg_hf`
- Paper code: `../color-filter-olmo`
- Tokenizer: `OLMoTokenizerFast`
- Sequence length: 512
- Domain sequences: 200 Gutenberg-ish, 200 C4
- Full scoring batch size: 4
- Elapsed seconds: 1545.75

## Module Tree

Detected block path: `model.transformer.blocks`

```text
<root>: OLMoForCausalLM
model: OLMo
model.transformer: ModuleDict
model.transformer.wte: Embedding
model.transformer.emb_drop: Dropout
model.transformer.ln_f: LayerNorm
model.transformer.blocks: ModuleList len=12
model.transformer.blocks.0: OLMoSequentialBlock layer_id=0
model.transformer.blocks.0.dropout: Dropout
model.transformer.blocks.0.k_norm: LayerNorm
model.transformer.blocks.0.q_norm: LayerNorm
model.transformer.blocks.0.act: GELU
model.transformer.blocks.0.attn_out: Linear
model.transformer.blocks.0.ff_out: Linear
model.transformer.blocks.0.rotary_emb: RotaryEmbedding
model.transformer.blocks.0.attn_norm: LayerNorm
model.transformer.blocks.0.ff_norm: LayerNorm
model.transformer.blocks.0.att_proj: Linear
model.transformer.blocks.0.ff_proj: Linear
model.transformer.blocks.1: OLMoSequentialBlock layer_id=1
model.transformer.blocks.1.dropout: Dropout
model.transformer.blocks.1.k_norm: LayerNorm
model.transformer.blocks.1.q_norm: LayerNorm
model.transformer.blocks.1.act: GELU
model.transformer.blocks.1.attn_out: Linear
model.transformer.blocks.1.ff_out: Linear
model.transformer.blocks.1.rotary_emb: RotaryEmbedding
model.transformer.blocks.1.attn_norm: LayerNorm
model.transformer.blocks.1.ff_norm: LayerNorm
model.transformer.blocks.1.att_proj: Linear
model.transformer.blocks.1.ff_proj: Linear
model.transformer.blocks.2: OLMoSequentialBlock layer_id=2
model.transformer.blocks.2.dropout: Dropout
model.transformer.blocks.2.k_norm: LayerNorm
model.transformer.blocks.2.q_norm: LayerNorm
model.transformer.blocks.2.act: GELU
model.transformer.blocks.2.attn_out: Linear
model.transformer.blocks.2.ff_out: Linear
model.transformer.blocks.2.rotary_emb: RotaryEmbedding
model.transformer.blocks.2.attn_norm: LayerNorm
model.transformer.blocks.2.ff_norm: LayerNorm
model.transformer.blocks.2.att_proj: Linear
model.transformer.blocks.2.ff_proj: Linear
model.transformer.blocks.3: OLMoSequentialBlock layer_id=3
model.transformer.blocks.3.dropout: Dropout
model.transformer.blocks.3.k_norm: LayerNorm
model.transformer.blocks.3.q_norm: LayerNorm
model.transformer.blocks.3.act: GELU
model.transformer.blocks.3.attn_out: Linear
model.transformer.blocks.3.ff_out: Linear
model.transformer.blocks.3.rotary_emb: RotaryEmbedding
model.transformer.blocks.3.attn_norm: LayerNorm
model.transformer.blocks.3.ff_norm: LayerNorm
model.transformer.blocks.3.att_proj: Linear
model.transformer.blocks.3.ff_proj: Linear
model.transformer.blocks.4: OLMoSequentialBlock layer_id=4
model.transformer.blocks.4.dropout: Dropout
model.transformer.blocks.4.k_norm: LayerNorm
model.transformer.blocks.4.q_norm: LayerNorm
model.transformer.blocks.4.act: GELU
model.transformer.blocks.4.attn_out: Linear
model.transformer.blocks.4.ff_out: Linear
model.transformer.blocks.4.rotary_emb: RotaryEmbedding
model.transformer.blocks.4.attn_norm: LayerNorm
model.transformer.blocks.4.ff_norm: LayerNorm
model.transformer.blocks.4.att_proj: Linear
model.transformer.blocks.4.ff_proj: Linear
model.transformer.blocks.5: OLMoSequentialBlock layer_id=5
model.transformer.blocks.5.dropout: Dropout
model.transformer.blocks.5.k_norm: LayerNorm
model.transformer.blocks.5.q_norm: LayerNorm
model.transformer.blocks.5.act: GELU
model.transformer.blocks.5.attn_out: Linear
model.transformer.blocks.5.ff_out: Linear
model.transformer.blocks.5.rotary_emb: RotaryEmbedding
model.transformer.blocks.5.attn_norm: LayerNorm
model.transformer.blocks.5.ff_norm: LayerNorm
model.transformer.blocks.5.att_proj: Linear
model.transformer.blocks.5.ff_proj: Linear
model.transformer.blocks.6: OLMoSequentialBlock layer_id=6
model.transformer.blocks.6.dropout: Dropout
model.transformer.blocks.6.k_norm: LayerNorm
model.transformer.blocks.6.q_norm: LayerNorm
model.transformer.blocks.6.act: GELU
model.transformer.blocks.6.attn_out: Linear
model.transformer.blocks.6.ff_out: Linear
model.transformer.blocks.6.rotary_emb: RotaryEmbedding
model.transformer.blocks.6.attn_norm: LayerNorm
model.transformer.blocks.6.ff_norm: LayerNorm
model.transformer.blocks.6.att_proj: Linear
model.transformer.blocks.6.ff_proj: Linear
model.transformer.blocks.7: OLMoSequentialBlock layer_id=7
model.transformer.blocks.7.dropout: Dropout
model.transformer.blocks.7.k_norm: LayerNorm
model.transformer.blocks.7.q_norm: LayerNorm
model.transformer.blocks.7.act: GELU
model.transformer.blocks.7.attn_out: Linear
model.transformer.blocks.7.ff_out: Linear
model.transformer.blocks.7.rotary_emb: RotaryEmbedding
model.transformer.blocks.7.attn_norm: LayerNorm
model.transformer.blocks.7.ff_norm: LayerNorm
model.transformer.blocks.7.att_proj: Linear
model.transformer.blocks.7.ff_proj: Linear
model.transformer.blocks.8: OLMoSequentialBlock layer_id=8
model.transformer.blocks.8.dropout: Dropout
model.transformer.blocks.8.k_norm: LayerNorm
model.transformer.blocks.8.q_norm: LayerNorm
model.transformer.blocks.8.act: GELU
model.transformer.blocks.8.attn_out: Linear
model.transformer.blocks.8.ff_out: Linear
model.transformer.blocks.8.rotary_emb: RotaryEmbedding
model.transformer.blocks.8.attn_norm: LayerNorm
model.transformer.blocks.8.ff_norm: LayerNorm
model.transformer.blocks.8.att_proj: Linear
model.transformer.blocks.8.ff_proj: Linear
model.transformer.blocks.9: OLMoSequentialBlock layer_id=9
model.transformer.blocks.9.dropout: Dropout
model.transformer.blocks.9.k_norm: LayerNorm
model.transformer.blocks.9.q_norm: LayerNorm
model.transformer.blocks.9.act: GELU
model.transformer.blocks.9.attn_out: Linear
model.transformer.blocks.9.ff_out: Linear
model.transformer.blocks.9.rotary_emb: RotaryEmbedding
model.transformer.blocks.9.attn_norm: LayerNorm
model.transformer.blocks.9.ff_norm: LayerNorm
model.transformer.blocks.9.att_proj: Linear
model.transformer.blocks.9.ff_proj: Linear
model.transformer.blocks.10: OLMoSequentialBlock layer_id=10
model.transformer.blocks.10.dropout: Dropout
model.transformer.blocks.10.k_norm: LayerNorm
model.transformer.blocks.10.q_norm: LayerNorm
model.transformer.blocks.10.act: GELU
model.transformer.blocks.10.attn_out: Linear
model.transformer.blocks.10.ff_out: Linear
model.transformer.blocks.10.rotary_emb: RotaryEmbedding
model.transformer.blocks.10.attn_norm: LayerNorm
model.transformer.blocks.10.ff_norm: LayerNorm
model.transformer.blocks.10.att_proj: Linear
model.transformer.blocks.10.ff_proj: Linear
model.transformer.blocks.11: OLMoSequentialBlock layer_id=11
model.transformer.blocks.11.dropout: Dropout
model.transformer.blocks.11.k_norm: LayerNorm
model.transformer.blocks.11.q_norm: LayerNorm
model.transformer.blocks.11.act: GELU
model.transformer.blocks.11.attn_out: Linear
model.transformer.blocks.11.ff_out: Linear
model.transformer.blocks.11.rotary_emb: RotaryEmbedding
model.transformer.blocks.11.attn_norm: LayerNorm
model.transformer.blocks.11.ff_norm: LayerNorm
model.transformer.blocks.11.att_proj: Linear
model.transformer.blocks.11.ff_proj: Linear
model.transformer.ff_out_last: Linear
```

## Full-Model Sanity

```text
c4_color_mean: 0.662052
c4_color_std: 0.163094
c4_n: 200
c4_nll_cond_mean: 4.171903
c4_nll_marg_mean: 3.509851
color_mean_gap_gutenberg_minus_c4: -1.962189
gutenberg_color_mean: -1.300137
gutenberg_color_std: 0.150422
gutenberg_n: 200
gutenberg_nll_cond_mean: 3.277606
gutenberg_nll_marg_mean: 4.577743
tokens_per_second: 275.366
```

## Ablated Forward Sanity

- Variant: `top4`
- Removed original layers: `(8, 9, 10, 11)`
- Conditional kept layers: 8
- Marginal kept layers: 8

```text
 seq_idx  nll_cond  nll_marg     color
       0  7.811137  8.423246 -0.612109
       1  6.714890  7.248482 -0.533593
       2  6.496810  6.999358 -0.502547
       3  6.463934  6.966583 -0.502649
       4  6.406541  6.768211 -0.361670
       5  6.581805  7.054479 -0.472673
       6  6.698310  7.145029 -0.446719
       7  6.201509  6.547701 -0.346192
```
