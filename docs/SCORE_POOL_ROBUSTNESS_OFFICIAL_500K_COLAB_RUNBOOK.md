# Official 500K Score-Pool Robustness Colab Runbook

This runbook executes the exact official-pool path for
`tasks/TASK_ablation_robustness_score_pools.md`.

It uses the five existing official sampled pools:

```text
random_positive_samples.npz
hard_positive_samples.npz
random_negative_samples.npz
hard_negative_samples.npz
tail_negative_samples.npz
```

and recovers all `500,000` corresponding packed 512-token C4 rows from
`hlzhang109/CoLoR-filter/full_data/c4`.

Corrected preflight facts:

```text
sample rows:             500,000
unique c4 indices:       496,205
visible token files:     170 *.npy
visible token bytes:     323.52 GiB
token chunks inferred:   339,236,242
max requested c4_index:  339,236,143
index coverage ok:       True
needed token files:      170
```

Use this runbook only when the Colab VM has a large local scratch disk. The
token cache should live on scratch, not Drive.

## 0. Resource Assumptions

Recommended Colab resources:

```text
GPU:             A100 80GB
System RAM:      >= 100GB
local scratch:   >= 25GB free for default streaming recovery
Drive quota:     enough for recovered pool, scores, metrics, and plots
```

From the current Colab resource panel, this VM is suitable:

```text
GPU RAM:              80GB
System RAM:           167GB
Disk [local-scratch]: 368GB
```

Default recovery is streaming: it downloads one raw token shard at a time,
recovers the needed rows into the final 500K token matrix, then deletes that
shard before moving to the next one. This still transfers `323.52 GiB` over the
network, but it does not need `323.52 GiB` of scratch space.

Only the older non-streaming cache path needs about `340GiB` free scratch.

## 1. Runtime, Drive, and Token

Set runtime to an A100 GPU:

```python
# PYTHON CELL
!nvidia-smi
```

Mount Drive:

```python
# PYTHON CELL
from google.colab import drive
drive.mount("/content/drive")
```

If needed, load a read-only Hugging Face token from Colab secrets:

```python
# PYTHON CELL
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN") or ""
```

Check storage:

```python
# PYTHON CELL
!df -h /content /content/local-scratch /content/drive/MyDrive
```

For the default streaming recovery path, stop if `/content/local-scratch` has
less than about `25GiB` free. If you intentionally use the old full-cache
download path, stop unless scratch has at least about `340GiB` free.

## 2. Clone, Pin, and Install

Push the commit containing this runbook and the pagination-aware streaming
recovery script, then set `PROJECT_SHA` to that exact pushed commit.

```python
# PYTHON CELL
import os
import subprocess
import sys
from pathlib import Path

PROJECT = "/content/CoLoR-ablation"
PROJECT_REPO = "https://github.com/myazdani/CoLoR-ablation.git"
PROJECT_SHA = "REPLACE_WITH_PUSHED_COMMIT_SHA"

OLMO = "/content/color-filter-olmo"
OLMO_REPO = "https://github.com/myazdani/color-filter-olmo.git"
OLMO_SHA = "3e0424c8cc6c53aaad70d3a3dea3fd683658cdd4"

def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)

def out(*args):
    return subprocess.check_output([str(arg) for arg in args], text=True).strip()

if PROJECT_SHA == "REPLACE_WITH_PUSHED_COMMIT_SHA":
    raise RuntimeError("Set PROJECT_SHA to the pushed commit before running.")

project = Path(PROJECT)
if not project.exists():
    run("git", "clone", PROJECT_REPO, PROJECT)
elif (project / ".git").is_dir():
    run("git", "-C", PROJECT, "fetch", "origin")
else:
    raise RuntimeError(f"{PROJECT} exists but is not a git checkout")
run("git", "-C", PROJECT, "checkout", PROJECT_SHA)
actual_project_sha = out("git", "-C", PROJECT, "rev-parse", "HEAD")
if actual_project_sha != PROJECT_SHA:
    raise RuntimeError(f"Project SHA mismatch: expected {PROJECT_SHA}, got {actual_project_sha}")

olmo = Path(OLMO)
if not olmo.exists():
    run("git", "clone", OLMO_REPO, OLMO)
elif not (olmo / ".git").is_dir():
    raise RuntimeError(f"{OLMO} exists but is not a git checkout")
run("git", "-C", OLMO, "fetch", "origin")
run("git", "-C", OLMO, "checkout", OLMO_SHA)
actual_olmo_sha = out("git", "-C", OLMO, "rev-parse", "HEAD")
if actual_olmo_sha != OLMO_SHA:
    raise RuntimeError(f"OLMo SHA mismatch: expected {OLMO_SHA}, got {actual_olmo_sha}")

%cd {PROJECT}
run(sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-r", "requirements-colab.txt")
print("project sha:", actual_project_sha)
print("olmo sha:", actual_olmo_sha)
```

## 3. Required Drive Inputs

Upload or copy the existing local pool-analysis artifacts to Drive:

```text
MyDrive/color-filter-ablation/artifacts/pool_analysis/random_positive_samples.npz
MyDrive/color-filter-ablation/artifacts/pool_analysis/hard_positive_samples.npz
MyDrive/color-filter-ablation/artifacts/pool_analysis/random_negative_samples.npz
MyDrive/color-filter-ablation/artifacts/pool_analysis/hard_negative_samples.npz
MyDrive/color-filter-ablation/artifacts/pool_analysis/tail_negative_samples.npz
```

Converted checkpoints should already be on Drive:

```text
MyDrive/color-filter-ablation/assets/hf/books_marg_hf
MyDrive/color-filter-ablation/assets/hf/books_cond_hf
```

Verify:

```python
# PYTHON CELL
from pathlib import Path

DRIVE = "/content/drive/MyDrive/color-filter-ablation"
SCRATCH = "/content/local-scratch/color-filter-ablation"

for sub in [
    "data",
    "results/score-pool-robustness-official-500k",
    "reports/score-pool-robustness-official-500k/figures",
]:
    Path(f"{DRIVE}/{sub}").mkdir(parents=True, exist_ok=True)
Path(SCRATCH).mkdir(parents=True, exist_ok=True)

pool_dir = Path(f"{DRIVE}/artifacts/pool_analysis")
pool_files = [
    "random_positive_samples.npz",
    "hard_positive_samples.npz",
    "random_negative_samples.npz",
    "hard_negative_samples.npz",
    "tail_negative_samples.npz",
]
missing_pools = [name for name in pool_files if not (pool_dir / name).is_file()]
if missing_pools:
    raise FileNotFoundError(f"Missing pool files in {pool_dir}: {missing_pools}")

required_ckpt = {
    "config.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
}
for name in ["books_marg_hf", "books_cond_hf"]:
    p = Path(f"{DRIVE}/assets/hf/{name}")
    missing = sorted(required_ckpt - {x.name for x in p.iterdir()}) if p.is_dir() else sorted(required_ckpt)
    if missing:
        raise FileNotFoundError(f"{p} missing converted checkpoint files: {missing}")
print("Drive inputs OK")
```

## 4. Patch Config for Official 500K

```python
# PYTHON CELL
from pathlib import Path
import yaml

robust_path = Path("configs/score_pool_robustness.yaml")
cfg = yaml.safe_load(robust_path.read_text())

cfg["paths"]["paper_code"] = OLMO
cfg["paths"]["existing_pool_analysis_dir"] = f"{DRIVE}/artifacts/pool_analysis"
cfg["paths"]["output_dir"] = f"{DRIVE}/results/score-pool-robustness-official-500k"
cfg["paths"]["figures_dir"] = f"{DRIVE}/reports/score-pool-robustness-official-500k/figures"

cfg["target"]["cond_checkpoint"] = f"{DRIVE}/assets/hf/books_cond_hf"
cfg["target"]["marg_checkpoint"] = f"{DRIVE}/assets/hf/books_marg_hf"

cfg["token_recovery"]["index_source"] = "contiguous_token_chunks"
cfg["token_recovery"]["local_cache_dir"] = f"{SCRATCH}/token_cache"
cfg["token_recovery"]["recovered_tokens"] = f"{DRIVE}/data/score_pool_tokens_official_500k.npy"
cfg["token_recovery"]["recovered_meta"] = f"{DRIVE}/data/score_pool_meta_official_500k.parquet"
cfg["token_recovery"]["recovery_plan"] = (
    f"{DRIVE}/results/score-pool-robustness-official-500k/token_recovery_plan.json"
)
cfg["token_recovery"]["max_download_gb_without_confirmation"] = 25

cfg["scoring"]["batch_size"] = 64
cfg["scoring"]["device"] = "cuda"
cfg["scoring"]["dtype"] = "bf16"
cfg["scoring"]["shard_size"] = 5000

robust_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(robust_path.read_text())
```

## 5. Cheap GPU Gate

Patch the existing checkpoint/model sanity check to use the Colab paths:

```python
# PYTHON CELL
from pathlib import Path
import yaml

default_path = Path("configs/default.yaml")
default_cfg = yaml.safe_load(default_path.read_text())
default_cfg["paths"]["paper_code"] = OLMO
default_cfg["target"]["cond_checkpoint"] = f"{DRIVE}/assets/hf/books_cond_hf"
default_cfg["target"]["marg_checkpoint"] = f"{DRIVE}/assets/hf/books_marg_hf"
default_path.write_text(yaml.safe_dump(default_cfg, sort_keys=False))
```

Then run the validation before downloading token files:

```python
# PYTHON CELL
!PYTHONPATH="{OLMO}" python scripts/06_local_validation.py --config configs/default.yaml --device cuda
```

Expected directionality:

```text
gutenberg_color_mean < c4_color_mean
```

## 6. Exact Token-Recovery Preflight

This should not download token files:

```python
# PYTHON CELL
!python scripts/09_recover_score_pool_tokens.py --config configs/score_pool_robustness.yaml
```

Verify:

```python
# PYTHON CELL
import json
from pathlib import Path

plan_path = Path(f"{DRIVE}/results/score-pool-robustness-official-500k/token_recovery_plan.json")
plan = json.loads(plan_path.read_text())
print(json.dumps({
    "sample_rows": plan["sample_rows"],
    "unique_c4_indices": plan["unique_c4_indices"],
    "remote_token_files": plan["total_remote_files"],
    "remote_token_gib": plan["total_remote_bytes"] / 1024**3,
    "total_remote_chunks": plan["total_remote_chunks"],
    "max_c4_index": plan["max_c4_index"],
    "index_coverage_ok": plan["index_coverage_ok"],
    "needed_file_count": plan["needed_file_count"],
    "total_needed_gib": plan["total_needed_gib"],
}, indent=2))
if not plan["index_coverage_ok"]:
    raise RuntimeError("Exact official-pool token recovery is not covered by visible token files")
if plan["needed_file_count"] != 170:
    raise RuntimeError("Unexpected token-file count; re-check HF tree pagination")
```

Expected:

```text
index_coverage_ok: True
needed_file_count: 170
total_needed_gib: 323.52
```

## 7. Stream Token Shards and Recover Official Pool

This streams about `323.52 GiB` from Hugging Face, but keeps only one raw shard
on local scratch at a time. Each shard is downloaded, its needed rows are copied
into the final recovered 500K-row token pool, and the shard is deleted before
the next shard starts.

The durable outputs are written to Drive:

```text
data/score_pool_tokens_official_500k.npy
data/score_pool_meta_official_500k.parquet
```

This mode is slower than having all shards cached locally, but it avoids the
`340GiB` scratch requirement.

```python
# PYTHON CELL
import os
os.environ["HF_HOME"] = f"{SCRATCH}/hf_home"
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{SCRATCH}/hf_home/hub"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
```

Before launching the full streaming recovery, run a bounded download probe
against real needed shards. This does not write the shard files to disk. It
downloads only the first `64MiB` from up to four needed shards and estimates the
total Step 7 transfer time from the observed aggregate throughput.

```python
# PYTHON CELL
import concurrent.futures as cf
import json
import subprocess
import time
from pathlib import Path

plan_path = Path(f"{DRIVE}/results/score-pool-robustness-official-500k/token_recovery_plan.json")
plan = json.loads(plan_path.read_text())
probe_files = plan["needed_files"][:4]
repo_id = plan["repo_id"]
total_bytes = int(plan["total_needed_bytes"])

def hf_resolve_url(repo_id, path):
    return f"https://huggingface.co/{repo_id}/resolve/main/{path}"

def probe_file(item, byte_limit=64 * 1024 * 1024):
    url = hf_resolve_url(repo_id, item["path"])
    cmd = [
        "curl",
        "-L",
        "-sS",
        "--connect-timeout",
        "20",
        "--max-time",
        "180",
        "-r",
        f"0-{byte_limit - 1}",
        "-o",
        "/dev/null",
        "-w",
        "http_code=%{http_code}\nsize_download=%{size_download}\nspeed_download=%{speed_download}\ntime_total=%{time_total}\n",
        url,
    ]
    started = time.time()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    row = {
        "path": item["path"],
        "returncode": proc.returncode,
        "wall_seconds": time.time() - started,
        "stderr": proc.stderr.strip(),
    }
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            row[key] = value
    return row

started = time.time()
with cf.ThreadPoolExecutor(max_workers=len(probe_files)) as executor:
    probe_rows = list(executor.map(probe_file, probe_files))
wall_seconds = time.time() - started
downloaded = sum(int(float(row.get("size_download", 0))) for row in probe_rows)
aggregate_bps = downloaded / wall_seconds if wall_seconds else 0.0

print(json.dumps(probe_rows, indent=2))
print(f"probe downloaded: {downloaded / 1024**2:.1f} MiB")
print(f"probe wall time:  {wall_seconds:.1f} sec")
print(f"aggregate speed:  {aggregate_bps / 1024**2:.2f} MiB/s")
if aggregate_bps <= 0:
    raise RuntimeError("Probe downloaded zero bytes; check network/HF access before Step 7")
eta_hours = total_bytes / aggregate_bps / 3600
print(f"estimated Step 7 transfer time for {total_bytes / 1024**3:.2f} GiB: {eta_hours:.2f} hours")
```

Interpret the estimate conservatively. Full recovery also spends time opening
memmaps, copying rows, flushing the recovered token matrix to Drive, and
deleting shard files. If the probe estimates `N` hours, a practical Step 7
budget is roughly `N * 1.1` to `N * 1.3`.

```python
# PYTHON CELL
!python scripts/09_recover_score_pool_tokens.py \
  --config configs/score_pool_robustness.yaml \
  --download \
  --streaming-download \
  --allow-large-download
```

The streaming command writes a checkpoint next to the recovered token file after
each completed shard:

```text
data/score_pool_tokens_official_500k.npy.streaming_checkpoint.json
```

If the runtime disconnects after using a commit with checkpoint support, reconnect
and resume with:

```python
# PYTHON CELL
!python scripts/09_recover_score_pool_tokens.py \
  --config configs/score_pool_robustness.yaml \
  --download \
  --streaming-download \
  --resume-streaming \
  --allow-large-download
```

If the disconnect happened before checkpoint support was available, use the last
completed shard number in the log. For example, if the log shows
`recovering ... (47/170)` and then disconnects while downloading `(48/170)`,
first confirm that the already-flushed token file exists:

```python
# PYTHON CELL
from pathlib import Path

partial_tokens = Path(f"{DRIVE}/data/score_pool_tokens_official_500k.npy")
print("partial token file exists:", partial_tokens.exists())
if partial_tokens.exists():
    print("partial token file bytes:", partial_tokens.stat().st_size)
else:
    raise RuntimeError(
        "No partial recovered token file found. Do not use "
        "--streaming-resume-from-file-index; rerun Step 7 from the start."
    )
```

If this check passes, resume from shard 48:

```python
# PYTHON CELL
!python scripts/09_recover_score_pool_tokens.py \
  --config configs/score_pool_robustness.yaml \
  --download \
  --streaming-download \
  --resume-streaming \
  --streaming-resume-from-file-index 48 \
  --allow-large-download
```

Verify recovered pool:

```python
# PYTHON CELL
import numpy as np
import pandas as pd

tokens = np.load(f"{DRIVE}/data/score_pool_tokens_official_500k.npy", mmap_mode="r")
meta = pd.read_parquet(f"{DRIVE}/data/score_pool_meta_official_500k.parquet")
print(tokens.shape, tokens.dtype)
print(meta["pool_name"].value_counts())
if tokens.shape != (500_000, 512):
    raise RuntimeError(f"Unexpected token shape: {tokens.shape}")
if len(meta) != 500_000:
    raise RuntimeError(f"Unexpected metadata rows: {len(meta)}")
```

After this verification, local scratch should contain only small cache metadata
because streaming recovery deletes each raw shard after it has been processed:

```python
# PYTHON CELL
!du -sh "{SCRATCH}/token_cache" || true
```

You can delete the remaining cache metadata after the recovered Drive pool is
verified:

```python
# PYTHON CELL
!rm -rf "{SCRATCH}/token_cache"
!df -h /content/local-scratch
```

## 8. Baseline and Noise-Floor Scoring

Score the recovered official 500K pool with full models:

```python
# PYTHON CELL
!PYTHONPATH="{OLMO}" python scripts/10_score_pool_variant.py \
  --config configs/score_pool_robustness.yaml \
  --variant full
```

Optional but recommended noise floor:

```python
# PYTHON CELL
!PYTHONPATH="{OLMO}" python scripts/10_score_pool_variant.py \
  --config configs/score_pool_robustness.yaml \
  --variant full_rescore
```

Compute baseline metrics:

```python
# PYTHON CELL
!python scripts/11_score_pool_metrics.py --config configs/score_pool_robustness.yaml
import pandas as pd
metrics = pd.read_csv(f"{DRIVE}/results/score-pool-robustness-official-500k/metrics_pairwise.csv")
display(metrics[["variant", "pairwise_task", "roc_auc", "average_precision", "f1_at_original_cutoff", "f1_at_balanced_rate"]])
```

Inspect this before launching the reduced ablation set.

## 9. Reduced Official Ablation Set

The fallback-pool run showed enough structure that the official 500K run should
not start with the full 26-ablation grid. On the official pool, each variant
scores `500,000` sequences with both the conditional and marginal models, so the
full grid costs many A100-hours. The reduced set below keeps the highest-signal
comparisons from the fallback report:

```text
pair_mid2:
  best fallback ablation overall; primary candidate.

pair_top1, pair_top2:
  strong paired top-layer baselines with small/moderate deletion.

pair_mid4:
  tests whether the middle-layer result survives a larger middle deletion.

marg_top1_only, marg_top2_only:
  marginal-only removals were surprisingly competitive in the fallback run.

cond_top1_only:
  tests whether Books-conditional top layers are especially fragile.

cond_top2_marg_bot2, cond_bot2_marg_top2:
  asymmetric directionality checks at moderate deletion size.

cond_top6_marg_bot6:
  severe negative-control case that was below chance in the fallback run.
```

Together with `full` and `full_rescore` from Step 8, this gives 12 total
variants instead of 28 total variants. This is the default official run. On an
A100, expect the 10 ablation variants in this step to take roughly `5-7` hours,
depending on Drive I/O and skip/resume state.

```python
# PYTHON CELL
selected_variants = [
    "pair_mid2",
    "pair_top1",
    "pair_top2",
    "pair_mid4",
    "marg_top1_only",
    "marg_top2_only",
    "cond_top1_only",
    "cond_top2_marg_bot2",
    "cond_bot2_marg_top2",
    "cond_top6_marg_bot6",
]
for variant in selected_variants:
    print(f"=== {variant} ===")
    !PYTHONPATH="{OLMO}" python scripts/10_score_pool_variant.py --config configs/score_pool_robustness.yaml --variant {variant}
```

Recompute metrics and inspect the reduced official result:

```python
# PYTHON CELL
!python scripts/11_score_pool_metrics.py --config configs/score_pool_robustness.yaml
selected = pd.read_csv(f"{DRIVE}/results/score-pool-robustness-official-500k/metrics_pairwise.csv")
display(selected[selected["variant"].isin(["full", "full_rescore"] + selected_variants)])
```

## 10. Optional Expanded Ablation Grid

Skip this section for the default official run. Run it only if the reduced set
does not answer the research question or if you need an exhaustive appendix.

The expanded grid adds weaker or more redundant variants that were less useful
in the fallback report: larger bottom deletions, larger conditional-only
deletions, and larger asymmetric variants. If Step 9 has already completed, the
scoring script will skip any already-completed variants when rerun.

```python
# PYTHON CELL
expanded_variants = [
    "pair_top1", "pair_top2", "pair_top4", "pair_top6",
    "pair_mid2", "pair_mid4",
    "pair_bot1", "pair_bot2", "pair_bot4", "pair_bot6",
    "cond_top1_marg_bot1", "cond_top2_marg_bot2", "cond_top4_marg_bot4", "cond_top6_marg_bot6",
    "cond_bot1_marg_top1", "cond_bot2_marg_top2", "cond_bot4_marg_top4", "cond_bot6_marg_top6",
    "cond_top1_only", "cond_top2_only", "cond_top4_only", "cond_top6_only",
    "marg_top1_only", "marg_top2_only", "marg_top4_only", "marg_top6_only",
]
for variant in expanded_variants:
    print(f"=== {variant} ===")
    !PYTHONPATH="{OLMO}" python scripts/10_score_pool_variant.py --config configs/score_pool_robustness.yaml --variant {variant}
```

The scoring script is resumable. Re-running the same command skips completed
score shards unless `--force` is passed.

## 11. Metrics, Plots, and Report

```python
# PYTHON CELL
!python scripts/11_score_pool_metrics.py --config configs/score_pool_robustness.yaml
!python scripts/12_score_pool_plots.py --config configs/score_pool_robustness.yaml
```

Generate Markdown and HTML reports:

```python
# PYTHON CELL
import html
from pathlib import Path
import pandas as pd

results_dir = Path(f"{DRIVE}/results/score-pool-robustness-official-500k")
report_dir = Path(f"{DRIVE}/reports/score-pool-robustness-official-500k")
report_dir.mkdir(parents=True, exist_ok=True)
metrics = pd.read_csv(results_dir / "metrics_pairwise.csv")
metric_cols = ["roc_auc", "average_precision", "f1_at_original_cutoff", "f1_at_balanced_rate"]
summary = metrics.groupby("variant")[metric_cols].mean().sort_values("roc_auc", ascending=False)
by_task = metrics.groupby("pairwise_task")[metric_cols].mean().sort_values("roc_auc")
figures = [
    ("ROC AUC", "figures/auc_by_variant_and_task.png"),
    ("Average precision", "figures/ap_by_variant_and_task.png"),
    ("F1 at original cutoff", "figures/f1_original_cutoff_by_variant.png"),
    ("F1 at balanced rate", "figures/f1_balanced_rate_by_variant.png"),
    ("Color shift", "figures/color_shift_by_variant.png"),
]

report = [
    "# Official 500K Score-Pool Robustness Report",
    "",
    "Label source: official five 100K sampled pools.",
    "",
    "## Mean Metrics by Variant",
    "",
    "```text",
    summary.round(4).reset_index().to_string(index=False),
    "```",
    "",
    "## Hardest Pairwise Tasks",
    "",
    "```text",
    by_task.round(4).reset_index().to_string(index=False),
    "```",
    "",
    "## Figures",
    "",
    *[f"- [{label}]({path})" for label, path in figures],
    "",
    "## Interpretation Notes",
    "",
    "- Compare original-cutoff F1 with balanced-rate F1 to separate calibration shift from ranking preservation.",
    "- Use full_rescore, when run, as the nondeterminism/noise floor.",
]
(report_dir / "report.md").write_text("\n".join(report) + "\n")

html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Official 500K Score-Pool Robustness Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; line-height: 1.45; color: #1f2933; }}
    table {{ border-collapse: collapse; margin: 12px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f6f8fa; }}
    img {{ max-width: 100%; border: 1px solid #d0d7de; margin: 8px 0 24px; }}
    code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Official 500K Score-Pool Robustness Report</h1>
  <p>Label source: official five 100K sampled pools.</p>
  <h2>Mean Metrics by Variant</h2>
  {summary.round(4).reset_index().to_html(index=False)}
  <h2>Hardest Pairwise Tasks</h2>
  {by_task.round(4).reset_index().to_html(index=False)}
  <h2>Figures</h2>
  {''.join(f'<h3>{html.escape(label)}</h3><img src="{html.escape(path)}" alt="{html.escape(label)}">' for label, path in figures)}
</body>
</html>
"""
(report_dir / "report.html").write_text(html_doc)
print(report_dir / "report.md")
print(report_dir / "report.html")
```

## 12. Outputs

Durable outputs on Drive:

```text
data/score_pool_tokens_official_500k.npy
data/score_pool_meta_official_500k.parquet
results/score-pool-robustness-official-500k/token_recovery_plan.json
results/score-pool-robustness-official-500k/scores_*.parquet
results/score-pool-robustness-official-500k/metrics_pairwise.csv
results/score-pool-robustness-official-500k/score_shift_diagnostics.csv
reports/score-pool-robustness-official-500k/figures/*.png
reports/score-pool-robustness-official-500k/report.md
reports/score-pool-robustness-official-500k/report.html
```

Local scratch can be deleted after recovery:

```text
/content/local-scratch/color-filter-ablation/token_cache
```

## 13. Failure Modes

- If preflight reports fewer than `170` token files, the checkout is missing the
  pagination-aware `list_remote_files()` fix.
- If streaming recovery is used and scratch is below about `25GiB`, do not start
  token download.
- If you use the older non-streaming recovery path, scratch needs about `340GiB`.
- If Colab disconnects during streaming recovery, rerun the streaming command.
  It will start over from the first shard because partial recovery is not yet
  checkpointed. Once the recovered Drive pool verifies, scratch can be deleted.
- If baseline metrics are poor, stop before the ablation grid and inspect
  checkpoint paths, recovered token shape, and score direction.
