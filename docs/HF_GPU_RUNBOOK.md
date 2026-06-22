# Deferred HF / Drive / GPU Runbook

> **How to apply this update.** This file is a full replacement for
> `docs/HF_GPU_RUNBOOK.md`. The previous version mixed bash and Python syntax in
> blocks meant to be pasted into Colab cells, which fails because **Colab cells
> run Python (IPython), not bash**. Every code block below is now labeled with
> the cell type it must be pasted into:
>
> - **`# PYTHON CELL`** — paste into a normal Colab code cell. Shell commands use
>   the `!` prefix; directory changes use the `%cd` magic. State (variables,
>   working dir) persists to later cells.
> - **`%%bash` CELL** — the first line of the cell is literally `%%bash`. The
>   whole cell runs in one shell. **Caveat:** variables and `cd` set in a
>   `%%bash` cell do *not* persist to other cells, so each `%%bash` cell must be
>   self-contained (set its own `export`s).
>
> Do not paste a block that contains bare `export FOO=...` / `cd ...` / `if [ ]`
> into a plain Python cell — it will raise `SyntaxError`. Do not use `!cd` to
> change directory; `!`-commands run in a throwaway subshell and the change is
> lost. Use the `%cd` magic instead.

These steps are for the Colab/A100 phase. Local CPU scaffolding, checkpoint
conversion, and validation are already complete (see
`reports/layer-ablated-color-filter/local_validation.md`).

---

## 0. Runtime, Drive, and Token (do this first)

Everything after this references `/content/drive/...`, so Drive must be mounted
before any other step. Set the runtime to GPU first:
**Runtime → Change runtime type → A100 GPU** (Colab Pro / Pay-As-You-Go).

```python
# PYTHON CELL
# Confirm a GPU is attached. If this errors or shows no GPU, fix the runtime
# type before continuing.
!nvidia-smi
```

```python
# PYTHON CELL
from google.colab import drive
drive.mount("/content/drive")
```

Store the Hugging Face token as a Colab secret (left sidebar → key icon) named
`HF_TOKEN`, with notebook access enabled. Use a fine-grained token whose only
scope is **read access to public repo contents**; leave inference, jobs,
billing, webhooks, discussions, org, and write permissions unchecked.

```python
# PYTHON CELL
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")  # hub client reads this automatically
```

---

## 1. Clone Repos and Install

Pin the paper fork to the exact commit the local validation was run against.
`src/model_loading.py` patches the fork's untied output head (`ff_out` →
`ff_out_last`); an unpinned `main` could drift and break that patch silently.

```python
# PYTHON CELL
import os

PROJECT = "/content/CoLoR-ablation"
OLMO = "/content/color-filter-olmo"
OLMO_SHA = "3e0424c8cc6c53aaad70d3a3dea3fd683658cdd4"

# This repo. If you uploaded the repo folder manually and it already exists,
# this clone step is skipped.
if not os.path.isdir(PROJECT):
    !git clone https://github.com/myazdani/CoLoR-ablation.git {PROJECT}

# Paper fork, pinned.
if not os.path.isdir(OLMO):
    !git clone https://github.com/davidbrandfonbrener/color-filter-olmo.git {OLMO}
!git -C {OLMO} checkout {OLMO_SHA}

%cd {PROJECT}
!pip install -q --no-deps -r requirements-colab.txt
```

Do **not** run `pip install -r requirements.txt` in Colab. That file is for the
local laptop environment and pins old versions of `torch`, `numpy`, `scipy`,
`transformers`, and `huggingface_hub`; installing it in Colab downgrades the
runtime's CUDA/PyTorch stack and creates resolver warnings like:

```text
torchvision requires torch==...
google-colab requires requests==...
diffusers/gradio/peft require newer huggingface-hub...
```

If you already ran `pip install -r requirements.txt` in this Colab runtime,
restart the runtime before continuing:

```python
# PYTHON CELL
# Runtime -> Restart runtime, then rerun this runbook from Step 0.
```

After installing, verify the paper fork's import-time dependencies are present:

```python
# PYTHON CELL
import boto3, botocore, rich, cached_path, omegaconf
print("paper-fork import deps OK")
```

If you previously hit `ModuleNotFoundError: No module named 'boto3'`, pull the
latest repo and rerun the install cell above. To patch the current runtime
without restarting:

```python
# PYTHON CELL
!pip install -q --no-deps boto3 botocore s3transfer jmespath rich google-cloud-storage google-api-core google-auth google-resumable-media google-crc32c google-cloud-core googleapis-common-protos proto-plus
```

Make the pinned fork importable in *this* Python session (needed for scoring,
not just conversion, because `model_loading.py` forces the fork's
`OLMoForCausalLM`):

```python
# PYTHON CELL
import sys
if OLMO not in sys.path:
    sys.path.insert(0, OLMO)
```

If the repo already existed in Colab from an earlier attempt, update it now.
The validation script changed after the first Colab runbook; a stale copy fails
with a Hugging Face `Repo id must be in the form...` error instead of the
clear checkpoint preflight. `src/model_loading.py` also carries compatibility
patches for Colab's newer `transformers`; stale copies can fail with
`AttributeError: 'OLMoForCausalLM' object has no attribute 'all_tied_weights_keys'`.

```python
# PYTHON CELL
%cd {PROJECT}

if os.path.isdir(f"{PROJECT}/.git"):
    !git pull
else:
    print("No .git directory found. If this folder was uploaded manually, re-upload the latest repo files.")

from pathlib import Path
validation_script = Path("scripts/06_local_validation.py").read_text()
model_loading = Path("src/model_loading.py").read_text()
if "validate_checkpoint_dir" not in validation_script:
    raise RuntimeError(
        "scripts/06_local_validation.py is stale. Pull/re-upload the latest repo "
        "before running Step 4."
    )
if "all_tied_weights_keys" not in model_loading:
    raise RuntimeError(
        "src/model_loading.py is stale. Pull/re-upload the latest repo before "
        "running Step 4."
    )
print("Validation script has checkpoint preflight.")
print("Model loader has Transformers compatibility patch.")
```

---

## 2. Drive Layout

```python
# PYTHON CELL
DRIVE = "/content/drive/MyDrive/color-filter-ablation"
for sub in ["assets/raw", "assets/hf", "data", "results",
            "reports/layer-ablated-color-filter/figures"]:
    os.makedirs(f"{DRIVE}/{sub}", exist_ok=True)
print(DRIVE)
```

---

## 3. Get the Converted Checkpoints onto Drive

Conversion was already done and validated locally, producing
`assets/hf/books_marg_hf` (θ_marg, from `models/prior`) and
`assets/hf/books_cond_hf` (θ_cond, from `models/conditional_books`).

**Primary path — upload the local converted checkpoints to Drive.** Cheapest and
avoids re-running conversion on Colab. From your laptop, copy both converted
folders into `MyDrive/color-filter-ablation/assets/hf/`, then verify here:

```python
# PYTHON CELL
import os
required = {
    "config.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
}
missing_any = False
for name in ["books_marg_hf", "books_cond_hf"]:
    p = f"{DRIVE}/assets/hf/{name}"
    if not os.path.isdir(p):
        print(name, "MISSING DIRECTORY:", p)
        missing_any = True
        continue
    files = set(os.listdir(p))
    missing = sorted(required - files)
    print(name, "OK" if not missing else f"MISSING FILES: {missing}",
          sorted(files))
    missing_any = missing_any or bool(missing)

if missing_any:
    raise RuntimeError(
        "Converted checkpoints are not ready. Upload assets/hf/books_marg_hf "
        "and assets/hf/books_cond_hf to Drive, or run the fallback conversion below."
    )
```

<details>
<summary><b>Fallback path — re-download and re-convert on Colab</b> (only if the
local converted checkpoints are unavailable)</summary>

Download only `config.yaml` + `model.pt` from each folder (skip `optim.pt` ~2 GB
and `train.pt`; they are useless for inference scoring):

```python
# PYTHON CELL
RAW = f"{DRIVE}/assets/raw"
for filename in [
    "models/prior/config.yaml",
    "models/prior/model.pt",
    "models/conditional_books/config.yaml",
    "models/conditional_books/model.pt",
]:
    !huggingface-cli download hlzhang109/CoLoR-filter {filename} --local-dir "{RAW}"
```

Then run the CPU-safe converter (`scripts/05_convert_olmo_to_hf.py`, which uses
`map_location="cpu"` and redirects caches under `assets/`):

```python
# PYTHON CELL
HF_ASSETS = f"{DRIVE}/assets/hf"
!mkdir -p "{HF_ASSETS}/books_marg_hf" "{HF_ASSETS}/books_cond_hf"
!cp -R "{RAW}/models/prior/." "{HF_ASSETS}/books_marg_hf/"
!cp -R "{RAW}/models/conditional_books/." "{HF_ASSETS}/books_cond_hf/"

!PYTHONPATH="{OLMO}" python scripts/05_convert_olmo_to_hf.py --checkpoint-dir "{HF_ASSETS}/books_marg_hf"
!PYTHONPATH="{OLMO}" python scripts/05_convert_olmo_to_hf.py --checkpoint-dir "{HF_ASSETS}/books_cond_hf"
```

</details>

Point `configs/default.yaml` at the Drive paths. This patches the repo config in
the current Colab runtime:

```python
# PYTHON CELL
from pathlib import Path
import yaml

cfg_path = Path("configs/default.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

cfg["target"]["cond_checkpoint"] = f"{DRIVE}/assets/hf/books_cond_hf"
cfg["target"]["marg_checkpoint"] = f"{DRIVE}/assets/hf/books_marg_hf"
cfg["target"]["pool_tokens"] = f"{DRIVE}/data/books_pool.npy"
cfg["target"]["pool_meta"] = f"{DRIVE}/data/books_pool_meta.parquet"

cfg["paths"]["paper_code"] = OLMO
cfg["paths"]["results_dir"] = f"{DRIVE}/results"
cfg["paths"]["figures_dir"] = f"{DRIVE}/reports/layer-ablated-color-filter/figures"
cfg["paths"]["metrics_csv"] = f"{DRIVE}/results/metrics.csv"

cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(cfg_path.read_text())
```

Before Step 4, verify the patched config points at directories that actually
exist in this Colab runtime:

```python
# PYTHON CELL
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
for key in ["marg_checkpoint", "cond_checkpoint"]:
    p = Path(cfg["target"][key])
    print(key, p, "exists=", p.is_dir())
    if not p.is_dir():
        raise FileNotFoundError(
            f"{key} does not exist: {p}. Drive may not be mounted, or the "
            "converted checkpoint folders are not under "
            "MyDrive/color-filter-ablation/assets/hf/."
        )
```

Also verify those directories are converted HF checkpoints, not raw OLMo
checkpoint folders:

```python
# PYTHON CELL
required = {"config.json", "pytorch_model.bin", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json"}

for key in ["marg_checkpoint", "cond_checkpoint"]:
    p = Path(cfg["target"][key])
    files = set(x.name for x in p.iterdir())
    missing = sorted(required - files)
    print(key, p)
    print("present:", sorted(files))
    if missing:
        raise FileNotFoundError(
            f"{key} is not a converted HF checkpoint. Missing: {missing}. "
            "Run the fallback conversion in Step 3 or upload the local assets/hf folders."
        )
```

Equivalent YAML values:

```yaml
target:
  cond_checkpoint: /content/drive/MyDrive/color-filter-ablation/assets/hf/books_cond_hf
  marg_checkpoint: /content/drive/MyDrive/color-filter-ablation/assets/hf/books_marg_hf
  pool_tokens:     /content/drive/MyDrive/color-filter-ablation/data/books_pool.npy
  pool_meta:       /content/drive/MyDrive/color-filter-ablation/data/books_pool_meta.parquet

paths:
  paper_code:  /content/color-filter-olmo
  results_dir: /content/drive/MyDrive/color-filter-ablation/results
  figures_dir: /content/drive/MyDrive/color-filter-ablation/reports/layer-ablated-color-filter/figures
  metrics_csv: /content/drive/MyDrive/color-filter-ablation/results/metrics.csv
```

---

## 4. GPU Load Check (cheap gate before building the pool)

Re-confirm on GPU what local CPU validation established, so any
load/precision/device problem surfaces before the main run. Record **θ_marg's
mean per-token NLL on random C4 sequences** — it should land in the low-to-mid
3s (consistent with the paper's 150m C4 loss). A value of 7+ means a silently
broken head even if the CoLoR *difference* looks fine.

```python
# PYTHON CELL
!python scripts/06_local_validation.py --config configs/default.yaml --device cuda
```

Expected, matching local validation: ablation path `model.transformer.blocks`,
12 blocks (ids 0..11), ~254,067,712 params loaded, finite and non-identical
θ_cond / θ_marg losses, and the Books-like-vs-C4 CoLoR gap strongly negative.

---

## 5. Build the Frozen Pool

```python
# PYTHON CELL
!python scripts/01_build_pool.py --config configs/default.yaml
```

```python
# PYTHON CELL
import pandas as pd
meta = pd.read_parquet(f"{DRIVE}/data/books_pool_meta.parquet")
print(meta["enriched"].value_counts(dropna=False))
print(meta.head())
```

Expected: 100,000 non-enriched C4 sequences and 5,000 enriched known-selected
Books sequences. The pool is frozen after this step — never regenerate it
mid-study.

---

## 6. Ground Truth + Noise Floor (defines the baseline)

The bf16 GPU `full` run — not the CPU numbers — is the ground truth all variants
are compared against. The `full_rescore` run quantifies bf16 nondeterminism so
every variant's correlation can be read against that floor.

```python
# PYTHON CELL
!python scripts/02_score.py --config configs/default.yaml --variant full
!python scripts/02_score.py --config configs/default.yaml --variant full --score-id full_rescore
!python scripts/03_metrics.py --config configs/default.yaml
```

**Binding sanity gate.** Inspect `full_enriched_selected_frac` in
`results/metrics.csv`. If the enriched known-selected Books sequences are not
strongly overrepresented in the low-score tail, **stop** and debug checkpoint
pairing or loss computation before spending GPU time on the grid. (The local
Gutenberg-ish proxy passed; this is the real-data version.)

---

## 7. Ablation Scoring Grid

`scripts/02_score.py` writes per-shard parquets and skips completed shards on
restart, so a Colab disconnect costs minutes, not the whole run.

```python
# PYTHON CELL
for variant in ["top1", "top2", "top4", "top6", "mid2", "mid4", "bot2", "bot4", "skip2"]:
    print(f"=== {variant} ===")
    !python scripts/02_score.py --config configs/default.yaml --variant {variant}
```

---

## 8. Metrics and Plots

```python
# PYTHON CELL
!python scripts/03_metrics.py --config configs/default.yaml
!python scripts/04_plots.py --config configs/default.yaml --scatter-variant top4
```

Primary outputs: `results/metrics.csv`, the overlap@top-k figure, the Spearman
figure, the speedup/overlap Pareto figure, and the ablated-vs-full score
scatter.

The pre-registered benchmark (≥33% layers removed at ≥80% overlap @ 1/16) is
context, not the headline. The primary results are the **Pareto curve** and the
**decomposition**: if individual `nll_cond` / `nll_marg` correlations degrade
under ablation while the CoLoR-difference correlation holds, that is direct
evidence for the cancellation hypothesis.

---

## Conventions to Record in Provenance

- **Loss reduction:** whichever `02_score.py` uses (mean-per-token vs.
  sum-per-sequence). Rank-equivalent under fixed 512-token packing, but the
  report's absolute values need the label.
- **Precision:** ground truth is bf16 on A100. CPU validation was fp32; expect
  ~1e-3 score shifts, which `full_rescore` measures.
- **Pinned fork SHA:** `3e0424c8cc6c53aaad70d3a3dea3fd683658cdd4`
- **Pool hash, checkpoint hashes, git commit:** embed in each output parquet.
