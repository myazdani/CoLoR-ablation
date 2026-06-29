# Sequence-Length Score-Pool Colab Runbook

This runbook completes `tasks/TASK_sequence_length_color_scoring.md` after the
local code changes are available on GitHub.

The job reuses the official 500K recovered token pool and scores the full Books
conditional/marginal models on shorter token windows.

## 1. Runtime

Use a Colab A100 runtime when available.

Expected inputs on Drive:

```text
MyDrive/color-filter-ablation/data/score_pool_tokens_official_500k.npy
MyDrive/color-filter-ablation/data/score_pool_meta_official_500k.parquet
MyDrive/color-filter-ablation/assets/hf/books_cond_hf
MyDrive/color-filter-ablation/assets/hf/books_marg_hf
```

Expected outputs on Drive:

```text
MyDrive/color-filter-ablation/results/sequence-length-score-pool/
MyDrive/color-filter-ablation/reports/sequence-length-score-pool/
```

## 2. Setup

Use Python setup cells for clone/checkout so failures stop the notebook.

```python
from pathlib import Path
import subprocess

DRIVE = "/content/drive/MyDrive/color-filter-ablation"
PROJECT = Path("/content/CoLoR-ablation")
REPO = "https://github.com/myazdani/CoLoR-ablation.git"
SHA = "REPLACE_WITH_PUSHED_COMMIT_SHA"

def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)

def out(*args):
    return subprocess.check_output([str(arg) for arg in args], text=True).strip()

from google.colab import drive
drive.mount("/content/drive")

if not PROJECT.exists():
    run("git", "clone", REPO, PROJECT)
else:
    run("git", "-C", PROJECT, "fetch", "origin")

run("git", "-C", PROJECT, "checkout", SHA)
actual = out("git", "-C", PROJECT, "rev-parse", "HEAD")
if actual != SHA:
    raise RuntimeError(f"SHA mismatch: expected {SHA}, got {actual}")
print("project sha:", actual)
```

Install only missing packages:

```python
%cd /content/CoLoR-ablation
!pip install -q pyarrow pandas scipy matplotlib pyyaml
```

Set the OLMo paper-code path if needed:

```python
OLMO = "/content/color-filter-olmo"
if not Path(OLMO).exists():
    run("git", "clone", "https://github.com/davidbrandfonbrener/color-filter-olmo.git", OLMO)
```

## 3. Configure Drive Paths

```python
from pathlib import Path
import yaml

cfg_path = Path("configs/sequence_length_score_pool.yaml")
cfg = yaml.safe_load(cfg_path.read_text())
cfg["paths"]["paper_code"] = OLMO
cfg["paths"]["output_dir"] = f"{DRIVE}/results/sequence-length-score-pool"
cfg["paths"]["figures_dir"] = f"{DRIVE}/reports/sequence-length-score-pool/figures"
cfg["token_recovery"]["recovered_tokens"] = f"{DRIVE}/data/score_pool_tokens_official_500k.npy"
cfg["token_recovery"]["recovered_meta"] = f"{DRIVE}/data/score_pool_meta_official_500k.parquet"
cfg["target"]["cond_checkpoint"] = f"{DRIVE}/assets/hf/books_cond_hf"
cfg["target"]["marg_checkpoint"] = f"{DRIVE}/assets/hf/books_marg_hf"
cfg["scoring"]["batch_size"] = 256
cfg["scoring"]["shard_size"] = 25000
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(yaml.safe_dump(cfg["scoring"], sort_keys=False))
```

Validate artifacts:

```python
import numpy as np
import pandas as pd
from pathlib import Path

tokens = np.load(cfg["token_recovery"]["recovered_tokens"], mmap_mode="r")
meta = pd.read_parquet(cfg["token_recovery"]["recovered_meta"])
print("tokens:", tokens.shape, tokens.dtype)
print("meta rows:", len(meta))
print("pools:", meta["pool_name"].value_counts().to_dict())
for key in ["cond_checkpoint", "marg_checkpoint"]:
    path = Path(cfg["target"][key])
    print(key, path, path.exists(), sorted(p.name for p in path.glob("*"))[:8])
assert tokens.shape == (500000, 512)
assert len(meta) == 500000
```

## 4. Cheap Smoke Test

Run one small shard before the full grid.

```python
!PYTHONPATH="{OLMO}" python scripts/16_score_sequence_window.py \
  --config configs/sequence_length_score_pool.yaml \
  --window seq_prefix_128 \
  --shard-size 512 \
  --force
```

Then remove the smoke output and shards before the real run:

```python
from pathlib import Path
import shutil

out = Path(cfg["paths"]["output_dir"])
for path in [out / "scores_seq_prefix_128.parquet", out / "scores_seq_prefix_128_shards"]:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
```

## 5. Batch-Size Benchmark

Windowed scoring can often use larger batches than 512-token scoring. Test one
representative 256-token window first.

```python
from pathlib import Path
import shutil
import yaml

for batch_size in [256, 512, 1024]:
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["scoring"]["batch_size"] = batch_size
    cfg["scoring"]["shard_size"] = 25000
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("=== batch", batch_size, "===")
    !PYTHONPATH="{OLMO}" python scripts/16_score_sequence_window.py \
      --config configs/sequence_length_score_pool.yaml \
      --window seq_prefix_256 \
      --shard-size 25000 \
      --force
    test_out = Path(cfg["paths"]["output_dir"])
    for path in [test_out / "scores_seq_prefix_256.parquet", test_out / "scores_seq_prefix_256_shards"]:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
```

Keep the largest stable batch size that improves throughput and leaves safe GPU
memory headroom. Reapply it after any Colab reconnect.

## 6. Full Primary Window Grid

```python
selected_windows = [
    "seq_full_512",
    "seq_prefix_256",
    "seq_suffix_256",
    "seq_prefix_128",
    "seq_suffix_128",
    "seq_middle_256",
    "seq_prefix_suffix_256",
]
for window in selected_windows:
    print(f"=== {window} ===")
    !PYTHONPATH="{OLMO}" python scripts/16_score_sequence_window.py \
      --config configs/sequence_length_score_pool.yaml \
      --window {window}
```

The scorer writes shard parquet files first. Rerunning the same command skips
valid completed shards and merges them into the final parquet.

## 7. Resume After Disconnect

After reconnecting:

```python
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("configs/sequence_length_score_pool.yaml").read_text())
out = Path(cfg["paths"]["output_dir"])
for window in [
    "seq_full_512",
    "seq_prefix_256",
    "seq_suffix_256",
    "seq_prefix_128",
    "seq_suffix_128",
    "seq_middle_256",
    "seq_prefix_suffix_256",
]:
    final = out / f"scores_{window}.parquet"
    shard_dir = out / f"scores_{window}_shards"
    shards = sorted(shard_dir.glob("part_*.parquet")) if shard_dir.exists() else []
    print(window, "final:", final.exists(), "shards:", len(shards), "last:", shards[-1].name if shards else None)
```

Rerun the same Step 6 loop. Do not use `--force` unless you intentionally want
to recompute existing shards.

## 8. Metrics, Plots, and Report

```python
!python scripts/17_sequence_length_metrics_report.py \
  --config configs/sequence_length_score_pool.yaml
```

Bring back:

```text
results/sequence-length-score-pool/scores_*.parquet
results/sequence-length-score-pool/metrics_pairwise_sequence_length.csv
results/sequence-length-score-pool/score_shift_diagnostics_sequence_length.csv
results/sequence-length-score-pool/runtime_by_window.csv
reports/sequence-length-score-pool/report.md
reports/sequence-length-score-pool/report.html
reports/sequence-length-score-pool/figures/*.png
```

## 9. Expected Runtime

Use the official 500K full scoring result as a starting point: about 38-40
minutes for one 512-token full pass at batch 256 on A100-like hardware. Expected
time depends on final batch size and Drive I/O:

```text
512-token window: roughly 35-45 min
256-token windows: roughly 18-30 min each after batch tuning
128-token windows: roughly 10-20 min each after batch tuning
```

The seven primary windows are therefore likely a multi-hour run. The shard-based
resume path is required; do not assume Colab will remain connected throughout.
