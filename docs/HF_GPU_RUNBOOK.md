# Colab Runbook

This is the shortest path from the Colab file layout shown in the screenshot to
converted checkpoints, a sanity check, and GPU scoring.

Assumed Colab layout:

```text
/content/CoLoR-ablation/
  assets/raw/models/prior/model.pt
  assets/raw/models/conditional_books/model.pt
  configs/
  scripts/
  src/
```

If your folder is named differently, change `PROJECT` in the first cell.

## 1. Setup

Run from a Colab notebook cell:

```bash
PROJECT=/content/CoLoR-ablation
OLMO=/content/color-filter-olmo

cd "$PROJECT"
python -m pip install -q -r requirements.txt

if [ ! -d "$OLMO" ]; then
  git clone https://github.com/davidbrandfonbrener/color-filter-olmo.git "$OLMO"
fi
```

Recommended Colab runtime: GPU. Conversion and the local sanity script can run
without GPU, but scoring the full pool should use GPU.

## 2. Make Sure Config Files Exist

Your screenshot shows `model.pt` under both checkpoint folders. Conversion also
needs `config.yaml`. Check:

```bash
cd /content/CoLoR-ablation
find assets/raw/models -maxdepth 2 -type f | sort
```

You should see all four files:

```text
assets/raw/models/conditional_books/config.yaml
assets/raw/models/conditional_books/model.pt
assets/raw/models/prior/config.yaml
assets/raw/models/prior/model.pt
```

If the two `config.yaml` files are missing, download only those small files:

```bash
cd /content/CoLoR-ablation

python - <<'PY'
from huggingface_hub import hf_hub_download

for filename in [
    "models/prior/config.yaml",
    "models/conditional_books/config.yaml",
]:
    path = hf_hub_download(
        repo_id="hlzhang109/CoLoR-filter",
        filename=filename,
        local_dir="assets/raw",
    )
    print(path)
PY
```

If the large `model.pt` files are also missing, download them explicitly:

```bash
cd /content/CoLoR-ablation

python - <<'PY'
from huggingface_hub import hf_hub_download

for filename in [
    "models/prior/model.pt",
    "models/conditional_books/model.pt",
]:
    path = hf_hub_download(
        repo_id="hlzhang109/CoLoR-filter",
        filename=filename,
        local_dir="assets/raw",
    )
    print(path)
PY
```

Do not download the whole `hlzhang109/CoLoR-filter` repo.

## 3. Convert Checkpoints

The converted checkpoints live under `assets/hf/`. This keeps raw downloads
unchanged.

```bash
cd /content/CoLoR-ablation

mkdir -p assets/hf/books_marg_hf assets/hf/books_cond_hf

cp assets/raw/models/prior/config.yaml assets/raw/models/prior/model.pt \
  assets/hf/books_marg_hf/
cp assets/raw/models/conditional_books/config.yaml assets/raw/models/conditional_books/model.pt \
  assets/hf/books_cond_hf/

PYTHONPATH=/content/color-filter-olmo python scripts/05_convert_olmo_to_hf.py \
  --checkpoint-dir assets/hf/books_marg_hf

PYTHONPATH=/content/color-filter-olmo python scripts/05_convert_olmo_to_hf.py \
  --checkpoint-dir assets/hf/books_cond_hf
```

Expected converted files:

```bash
find assets/hf/books_marg_hf assets/hf/books_cond_hf -maxdepth 1 -type f | sort
```

You should see `config.json`, `pytorch_model.bin`, `tokenizer.json`, and
tokenizer metadata in both converted directories.

## 4. Run a Fast Sanity Check

First run a tiny check so failures appear quickly:

```bash
cd /content/CoLoR-ablation

PYTHONPATH=/content/color-filter-olmo python scripts/06_local_validation.py \
  --paper-code /content/color-filter-olmo \
  --n-per-domain 2 \
  --batch-size 1 \
  --report reports/layer-ablated-color-filter/local_validation_tiny.md
```

Then run the fuller local sanity check:

```bash
PYTHONPATH=/content/color-filter-olmo python scripts/06_local_validation.py \
  --paper-code /content/color-filter-olmo \
  --n-per-domain 200 \
  --batch-size 4 \
  --report reports/layer-ablated-color-filter/local_validation.md
```

Expected direction:

- Gutenberg-ish mean CoLoR should be lower than C4 mean CoLoR.
- The module path should print as `model.transformer.blocks`.
- The ablated `top4` forward check should produce finite, non-identical
  `nll_cond` and `nll_marg`.

The 200-vs-200 check is CPU-only in this script and can take a while. It is a
validation step, not the main GPU scoring path.

## 5. Write a Colab Config

Create `configs/colab.yaml` with paths matching this Colab runtime:

```bash
cd /content/CoLoR-ablation

python - <<'PY'
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("configs/default.yaml").read_text())

cfg["target"]["cond_checkpoint"] = "assets/hf/books_cond_hf"
cfg["target"]["marg_checkpoint"] = "assets/hf/books_marg_hf"
cfg["target"]["pool_tokens"] = "data/books_pool.npy"
cfg["target"]["pool_meta"] = "data/books_pool_meta.parquet"

cfg["paths"]["paper_code"] = "/content/color-filter-olmo"
cfg["paths"]["results_dir"] = "results"
cfg["paths"]["figures_dir"] = "reports/layer-ablated-color-filter/figures"
cfg["paths"]["metrics_csv"] = "results/metrics.csv"

cfg["scoring"]["device"] = "auto"
cfg["scoring"]["dtype"] = "bf16"
cfg["scoring"]["batch_size"] = 64
cfg["scoring"]["shard_size"] = 5000

Path("data").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
Path("configs/colab.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
print(Path("configs/colab.yaml").read_text())
PY
```

For a small GPU shakedown before the full pool, edit `configs/colab.yaml` and
temporarily reduce:

```yaml
pool:
  c4:
    n_sequences: 1000
  enriched:
    n: 100
```

Do not use shakedown results for the final study.

## 6. Build the Frozen Pool

```bash
cd /content/CoLoR-ablation
python scripts/01_build_pool.py --config configs/colab.yaml
```

Check the enriched quarantine column:

```bash
python - <<'PY'
import pandas as pd
meta = pd.read_parquet("data/books_pool_meta.parquet")
print(meta["enriched"].value_counts(dropna=False))
print(meta.head())
PY
```

Expected full v1 pool: `100000` non-enriched C4 sequences and `5000` enriched
known-selected Books sequences.

## 7. Score the Full Pair First

```bash
cd /content/CoLoR-ablation
python scripts/02_score.py --config configs/colab.yaml --variant full
```

`02_score.py` writes resumable shard parquets by default using
`scoring.shard_size`. On restart, completed shards are validated by `seq_idx`
range and skipped, then the combined `scores_<variant>.parquet` is rebuilt.

After full scoring, compute metrics once:

```bash
python scripts/03_metrics.py --config configs/colab.yaml
```

Inspect `results/metrics.csv`. The enriched known-selected Books sequences
should be strongly overrepresented in the low-score tail. If not, stop and debug
checkpoint pairing, tokenizer, or loss computation before ablations.

## 8. Run Ablations

```bash
cd /content/CoLoR-ablation

for variant in top1 top2 top4 top6 mid2 mid4 bot2 bot4 skip2; do
  python scripts/02_score.py --config configs/colab.yaml --variant "$variant"
done
```

Optional noise-floor rescore:

```bash
python scripts/02_score.py \
  --config configs/colab.yaml \
  --variant full \
  --score-id full_rescore
```

## 9. Metrics and Plots

```bash
cd /content/CoLoR-ablation
python scripts/03_metrics.py --config configs/colab.yaml
python scripts/04_plots.py --config configs/colab.yaml --scatter-variant top4
```

Primary outputs:

- `results/metrics.csv`
- `reports/layer-ablated-color-filter/figures/overlap_at_topk.png`
- `reports/layer-ablated-color-filter/figures/spearman_vs_layers.png`
- `reports/layer-ablated-color-filter/figures/pareto_speedup_overlap.png`
- `reports/layer-ablated-color-filter/figures/scatter_top4.png`

## 10. Persist Important Artifacts

Files under `/content` disappear when the Colab runtime is reset. Copy at least
these to Drive:

```bash
mkdir -p /content/drive/MyDrive/color-filter-ablation
rsync -a results reports data configs/colab.yaml \
  /content/drive/MyDrive/color-filter-ablation/
```

If you do not want to reconvert after every runtime reset, also copy
`assets/hf/` to Drive.
