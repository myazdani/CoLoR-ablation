# Deferred HF / Drive / GPU Runbook

These commands are documented for the Colab/A100 phase. They were not executed
during local scaffolding.

## 1. Hugging Face Token Scope

For v1 Drive-only artifacts, create a fine-grained Hugging Face token with:

- `Repositories -> Read access to contents of all public gated repos you can access`

Leave inference, jobs, billing, webhooks, discussions, org permissions, and
broad write permissions unchecked.

In Colab, store it as a secret named `HF_TOKEN`. The Hugging Face Hub client
uses `HF_TOKEN` automatically when present.

## 2. Colab Setup

```bash
git clone <YOUR_GITHUB_REMOTE_FOR_THIS_REPO> color-filter-ablation
git clone https://github.com/davidbrandfonbrener/color-filter-olmo.git color-filter-olmo
cd color-filter-ablation
pip install -r requirements.txt
```

If running from a notebook, mount Drive first:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Recommended Drive layout:

```text
/content/drive/MyDrive/color-filter-ablation/
  assets/raw/
  assets/hf/
  data/
  results/
```

## 3. Identify the Books Checkpoints

Open the Hugging Face repository in a browser:

```text
https://huggingface.co/hlzhang109/CoLoR-filter/tree/main
```

Identify only these two checkpoint folders:

- Books prior / marginal model, `theta_marg`
- Books conditional model, `theta_cond`

Do not download the whole repo; it is expected to be hundreds of GB.

## 4. Download Only the Two Checkpoint Folders

Replace the include patterns below after inspecting the tree.

```bash
export HF_HOME=/content/drive/MyDrive/color-filter-ablation/.hf-cache
export RAW=/content/drive/MyDrive/color-filter-ablation/assets/raw

hf download hlzhang109/CoLoR-filter \
  --include "PATH/TO/BOOKS_PRIOR_CHECKPOINT/*" \
  --local-dir "$RAW"

hf download hlzhang109/CoLoR-filter \
  --include "PATH/TO/BOOKS_CONDITIONAL_CHECKPOINT/*" \
  --local-dir "$RAW"
```

For local testing from the repository root, the known Books paths can be
downloaded directly:

```bash
mkdir -p assets/raw
export HF_HOME="$PWD/assets/.hf-cache"

hf download hlzhang109/CoLoR-filter \
  models/prior/config.yaml models/prior/model.pt \
  --local-dir assets/raw

hf download hlzhang109/CoLoR-filter \
  models/conditional_books/config.yaml models/conditional_books/model.pt \
  --local-dir assets/raw
```

## 5. Convert Checkpoints Once

The local `color-filter-olmo` fork contains the conversion utility. Keep the
converted checkpoints in Drive.

```bash
export OLMO=/content/color-filter-olmo
export HF_ASSETS=/content/drive/MyDrive/color-filter-ablation/assets/hf

mkdir -p "$HF_ASSETS/books_marg_hf" "$HF_ASSETS/books_cond_hf"
cp -R "$RAW/PATH/TO/BOOKS_PRIOR_CHECKPOINT/." "$HF_ASSETS/books_marg_hf/"
cp -R "$RAW/PATH/TO/BOOKS_CONDITIONAL_CHECKPOINT/." "$HF_ASSETS/books_cond_hf/"

PYTHONPATH="$OLMO" python scripts/05_convert_olmo_to_hf.py \
  --checkpoint-dir "$HF_ASSETS/books_marg_hf"

PYTHONPATH="$OLMO" python scripts/05_convert_olmo_to_hf.py \
  --checkpoint-dir "$HF_ASSETS/books_cond_hf"
```

The repo-local converter differs from the paper fork's converter in two local
debugging details:

- it loads `model.pt` with `map_location="cpu"`, which is required on CPU-only
  machines when checkpoints were saved from CUDA;
- it redirects Hugging Face and `cached_path` caches under `assets/` instead of
  `~/.cache`.

Then update `configs/default.yaml`:

```yaml
target:
  cond_checkpoint: /content/drive/MyDrive/color-filter-ablation/assets/hf/books_cond_hf
  marg_checkpoint: /content/drive/MyDrive/color-filter-ablation/assets/hf/books_marg_hf
  pool_tokens: /content/drive/MyDrive/color-filter-ablation/data/books_pool.npy
  pool_meta: /content/drive/MyDrive/color-filter-ablation/data/books_pool_meta.parquet

paths:
  paper_code: /content/color-filter-olmo
  results_dir: /content/drive/MyDrive/color-filter-ablation/results
  figures_dir: /content/drive/MyDrive/color-filter-ablation/reports/layer-ablated-color-filter/figures
  metrics_csv: /content/drive/MyDrive/color-filter-ablation/results/metrics.csv
```

## 6. Build the Frozen Pool

```bash
python scripts/01_build_pool.py --config configs/default.yaml
```

Verify the metadata before scoring:

```bash
python - <<'PY'
import pandas as pd
meta = pd.read_parquet("/content/drive/MyDrive/color-filter-ablation/data/books_pool_meta.parquet")
print(meta["enriched"].value_counts(dropna=False))
print(meta.head())
PY
```

Expected v1 count: 100,000 non-enriched C4 sequences and 5,000 enriched
known-selected Books sequences.

## 7. Sanity-Check the Checkpoint Pair

Before ablations, score the full pair and verify that the enriched known-selected
Books sequences land heavily in the negative tail.

```bash
python scripts/02_score.py --config configs/default.yaml --variant full
python scripts/03_metrics.py --config configs/default.yaml
```

Inspect `full_enriched_selected_frac` in `results/metrics.csv`. If enriched
sequences are not strongly overrepresented in the low-score tail, stop and
debug checkpoint pairing or loss computation.

## 8. Run Ablation Scoring

```bash
for variant in top1 top2 top4 top6 mid2 mid4 bot2 bot4 skip2; do
  python scripts/02_score.py --config configs/default.yaml --variant "$variant"
done
```

`02_score.py` writes resumable shard parquets by default using
`scoring.shard_size` from the config. On restart, completed shards are validated
by `seq_idx` range and skipped, then the combined `scores_<variant>.parquet` is
rebuilt from all shards. Use `--no-shards` only for small debugging runs.

For a noise-floor rescore, use a separate output path while preserving no
ablation:

```bash
python scripts/02_score.py \
  --config configs/default.yaml \
  --variant full \
  --score-id full_rescore
```

## 9. Metrics and Plots

```bash
python scripts/03_metrics.py --config configs/default.yaml
python scripts/04_plots.py --config configs/default.yaml --scatter-variant top4
```

Primary output:

- `results/metrics.csv`
- overlap@top-k figure
- Spearman figure
- speedup/overlap Pareto figure
- ablated-vs-full score scatter

The pre-registered benchmark is useful context, but the primary result is the
Pareto curve and the decomposition between individual NLL degradation and
CoLoR-score robustness.
