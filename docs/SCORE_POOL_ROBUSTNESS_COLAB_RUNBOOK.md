# Score-Pool Robustness Fallback Colab Runbook

This runbook executes the fallback path for
`tasks/TASK_ablation_robustness_score_pools.md`.

The official five sampled pools could not be recovered exactly from the visible
`hlzhang109/CoLoR-filter/full_data/c4` tree. The local preflight found:

```text
visible token chunks inferred: 50,005,443
max requested c4_index:        339,236,143
index coverage ok:             False
```

So this runbook builds a new frozen packed C4 pool, scores it with the full
unablated Books conditional/marginal models, defines fallback positive/negative
pools from those full scores, then runs the asymmetric ablation robustness grid.

All fallback outputs must be kept separate from official-pool outputs:

```text
results/score-pool-robustness-fallback/
reports/score-pool-robustness-fallback/
```

## 0. Colab Cell Semantics

Every code block below is meant for a normal Colab Python cell unless it starts
with `%%bash`.

- Use `!cmd` for shell commands.
- Use `%cd` for persistent directory changes.
- Do not paste bare `export`, `cd`, or shell `if [ ]` blocks into Python cells.
- Critical setup uses `subprocess.run(..., check=True)` so failures stop the
  notebook.

## 1. Runtime, Drive, and Hugging Face Token

Set runtime to an A100 GPU first:

```python
# PYTHON CELL
!nvidia-smi
```

```python
# PYTHON CELL
from google.colab import drive
drive.mount("/content/drive")
```

If you need Hugging Face downloads, store a read-only token as a Colab secret
named `HF_TOKEN`:

```python
# PYTHON CELL
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

## 2. Clone, Pin, and Install

Push local `main` before starting the Colab run. Then replace
`PROJECT_SHA` below with the exact pushed commit SHA that contains this
runbook and the score-pool robustness scripts. The cell refuses to run long
work from an unpinned `main` checkout unless you explicitly set
`ALLOW_UNPINNED_MAIN = True` for a short smoke test.

```python
# PYTHON CELL
import os
import subprocess
import sys
from pathlib import Path

PROJECT = "/content/CoLoR-ablation"
PROJECT_REPO = "https://github.com/myazdani/CoLoR-ablation.git"
PROJECT_REF = "main"
PROJECT_SHA = "REPLACE_WITH_PUSHED_COMMIT_SHA"
ALLOW_UNPINNED_MAIN = False

OLMO = "/content/color-filter-olmo"
OLMO_REPO = "https://github.com/myazdani/color-filter-olmo.git"
OLMO_SHA = "3e0424c8cc6c53aaad70d3a3dea3fd683658cdd4"

def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)

def out(*args):
    return subprocess.check_output([str(arg) for arg in args], text=True).strip()

project = Path(PROJECT)
if not project.exists():
    run("git", "clone", PROJECT_REPO, PROJECT)
elif (project / ".git").is_dir():
    run("git", "-C", PROJECT, "fetch", "origin")
else:
    raise RuntimeError(f"{PROJECT} exists but is not a git checkout")
run("git", "-C", PROJECT, "checkout", PROJECT_REF)
actual_project_sha = out("git", "-C", PROJECT, "rev-parse", "HEAD")
if PROJECT_SHA == "REPLACE_WITH_PUSHED_COMMIT_SHA":
    if not ALLOW_UNPINNED_MAIN:
        raise RuntimeError(
            f"Set PROJECT_SHA to the pushed commit before running expensive jobs. "
            f"Current {PROJECT_REF} resolves to {actual_project_sha}."
        )
elif actual_project_sha != PROJECT_SHA:
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
print("project sha:", out("git", "-C", PROJECT, "rev-parse", "HEAD"))
print("olmo sha:", actual_olmo_sha)
```

Do not install `requirements.txt` in Colab. It pins local laptop versions and can
downgrade Colab's CUDA/PyTorch stack.

Verify the expected files exist in the checked-out project:

```python
# PYTHON CELL
from pathlib import Path
required_files = [
    "configs/score_pool_robustness.yaml",
    "scripts/10_score_pool_variant.py",
    "scripts/11_score_pool_metrics.py",
    "scripts/12_score_pool_plots.py",
    "scripts/13_build_fallback_score_pools.py",
    "src/score_pool_robustness.py",
]
missing = [path for path in required_files if not Path(path).exists()]
if missing:
    raise RuntimeError(f"Project checkout is missing fallback robustness files: {missing}")
print("score-pool robustness files present")
```

## 3. Drive Layout

```python
# PYTHON CELL
DRIVE = "/content/drive/MyDrive/color-filter-ablation"
for sub in [
    "assets/raw",
    "assets/hf",
    "data",
    "results/fallback-source",
    "results/score-pool-robustness-fallback",
    "reports/score-pool-robustness-fallback/figures",
]:
    os.makedirs(f"{DRIVE}/{sub}", exist_ok=True)
print(DRIVE)
```

## 4. Checkpoints

Preferred path: upload converted checkpoints to Drive:

```text
MyDrive/color-filter-ablation/assets/hf/books_marg_hf
MyDrive/color-filter-ablation/assets/hf/books_cond_hf
```

Verify them:

```python
# PYTHON CELL
required = {
    "config.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
}
for name in ["books_marg_hf", "books_cond_hf"]:
    p = Path(f"{DRIVE}/assets/hf/{name}")
    if not p.is_dir():
        raise FileNotFoundError(f"Missing converted checkpoint dir: {p}")
    files = set(x.name for x in p.iterdir())
    missing = sorted(required - files)
    print(name, "OK" if not missing else f"MISSING {missing}")
    if missing:
        raise FileNotFoundError(f"{p} is not a converted HF checkpoint")
```

Fallback conversion path, only if converted checkpoints are absent:

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

HF_ASSETS = f"{DRIVE}/assets/hf"
!mkdir -p "{HF_ASSETS}/books_marg_hf" "{HF_ASSETS}/books_cond_hf"
!cp -R "{RAW}/models/prior/." "{HF_ASSETS}/books_marg_hf/"
!cp -R "{RAW}/models/conditional_books/." "{HF_ASSETS}/books_cond_hf/"

!PYTHONPATH="{OLMO}" python scripts/05_convert_olmo_to_hf.py --checkpoint-dir "{HF_ASSETS}/books_marg_hf"
!PYTHONPATH="{OLMO}" python scripts/05_convert_olmo_to_hf.py --checkpoint-dir "{HF_ASSETS}/books_cond_hf"
```

## 5. Patch Configs for Fallback

Fallback sample size depends on source pool size. With 500K C4 sequences, the
`tau=64` tail has about 7,813 examples, so a 5K-per-pool fallback is feasible.
If GPU budget is tight, use 100K source sequences and 1K per pool.

```python
# PYTHON CELL
from pathlib import Path
import yaml

FALLBACK_C4_SEQS = 500_000
FALLBACK_SAMPLE_SIZE = 5_000

default_path = Path("configs/default.yaml")
default_cfg = yaml.safe_load(default_path.read_text())
default_cfg["target"]["cond_checkpoint"] = f"{DRIVE}/assets/hf/books_cond_hf"
default_cfg["target"]["marg_checkpoint"] = f"{DRIVE}/assets/hf/books_marg_hf"
default_cfg["target"]["pool_tokens"] = f"{DRIVE}/data/fallback_source_pool.npy"
default_cfg["target"]["pool_meta"] = f"{DRIVE}/data/fallback_source_pool_meta.parquet"
default_cfg["paths"]["paper_code"] = OLMO
default_cfg["paths"]["results_dir"] = f"{DRIVE}/results/fallback-source"
default_cfg["paths"]["figures_dir"] = f"{DRIVE}/reports/score-pool-robustness-fallback/source_figures"
default_cfg["paths"]["metrics_csv"] = f"{DRIVE}/results/fallback-source/metrics.csv"
default_cfg["pool"]["c4"]["n_sequences"] = FALLBACK_C4_SEQS
default_cfg["pool"]["enriched"]["n"] = 0
default_cfg["scoring"]["batch_size"] = 64
default_cfg["scoring"]["device"] = "cuda"
default_cfg["scoring"]["dtype"] = "bf16"
default_path.write_text(yaml.safe_dump(default_cfg, sort_keys=False))

robust_path = Path("configs/score_pool_robustness.yaml")
robust_cfg = yaml.safe_load(robust_path.read_text())
robust_cfg["paths"]["paper_code"] = OLMO
robust_cfg["paths"]["output_dir"] = f"{DRIVE}/results/score-pool-robustness-fallback"
robust_cfg["paths"]["figures_dir"] = f"{DRIVE}/reports/score-pool-robustness-fallback/figures"
robust_cfg["target"]["cond_checkpoint"] = f"{DRIVE}/assets/hf/books_cond_hf"
robust_cfg["target"]["marg_checkpoint"] = f"{DRIVE}/assets/hf/books_marg_hf"
robust_cfg["token_recovery"]["mode"] = "fallback_full_rescore_pool"
robust_cfg["token_recovery"]["recovered_tokens"] = f"{DRIVE}/data/fallback_score_pool_tokens.npy"
robust_cfg["token_recovery"]["recovered_meta"] = f"{DRIVE}/data/fallback_score_pool_meta.parquet"
robust_cfg["scoring"]["batch_size"] = 64
robust_cfg["scoring"]["device"] = "cuda"
robust_cfg["scoring"]["dtype"] = "bf16"
robust_path.write_text(yaml.safe_dump(robust_cfg, sort_keys=False))

print(default_path.read_text())
print(robust_path.read_text())
```

## 6. Cheap GPU Gate

Run the existing validation before spending time on the large pool:

```python
# PYTHON CELL
!PYTHONPATH="{OLMO}" python scripts/06_local_validation.py --config configs/default.yaml --device cuda
```

Expected:

- both converted models load
- block path resolves to `model.transformer.blocks`
- 12 blocks
- finite losses
- conditional and marginal models are not identical
- Books-like proxy scores separate from random C4 in the expected direction

Stop here if this fails.

## 7. Build the Fallback Source Pool

This creates the large frozen C4 token pool used to induce fallback labels.

```python
# PYTHON CELL
!python scripts/01_build_pool.py --config configs/default.yaml
```

Verify size and metadata:

```python
# PYTHON CELL
import numpy as np
import pandas as pd

source_tokens = np.load(f"{DRIVE}/data/fallback_source_pool.npy", mmap_mode="r")
source_meta = pd.read_parquet(f"{DRIVE}/data/fallback_source_pool_meta.parquet")
print(source_tokens.shape, source_tokens.dtype)
print(source_meta["enriched"].value_counts(dropna=False) if "enriched" in source_meta else "no enriched column")
if source_tokens.shape[0] != FALLBACK_C4_SEQS:
    raise RuntimeError("Fallback source pool row count mismatch")
```

## 8. Full Unablated Source Scoring

This defines the fallback labels. It is not yet the ablation experiment.

```python
# PYTHON CELL
!PYTHONPATH="{OLMO}" python scripts/02_score.py --config configs/default.yaml --variant full
```

Check full-score distribution and induced pool feasibility:

```python
# PYTHON CELL
import numpy as np
import pandas as pd
scores = pd.read_parquet(f"{DRIVE}/results/fallback-source/scores_full.parquet")
print(scores[["nll_cond", "nll_marg", "color"]].describe())
print("N:", len(scores), "tau64 count:", int(np.ceil(len(scores) / 64)), "tau32 band:", int(np.ceil(len(scores) / 32)) - int(np.ceil(len(scores) / 64)))
if int(np.ceil(len(scores) / 64)) < FALLBACK_SAMPLE_SIZE:
    raise RuntimeError("Source pool is too small for FALLBACK_SAMPLE_SIZE")
```

## 9. Build the Five Fallback Score Pools

This samples the five pools from the full-score source pool and writes the token
and metadata files consumed by `scripts/10_score_pool_variant.py`.

```python
# PYTHON CELL
!python scripts/13_build_fallback_score_pools.py \
  --config configs/score_pool_robustness.yaml \
  --source-config configs/default.yaml \
  --sample-size {FALLBACK_SAMPLE_SIZE}
```

Patch the robustness config to use the fallback full-model `tau=64` cutoff for
the original-cutoff metrics:

```python
# PYTHON CELL
import json
import yaml
from pathlib import Path

summary_path = Path(f"{DRIVE}/results/score-pool-robustness-fallback/fallback_pool_summary.json")
summary = json.loads(summary_path.read_text())
robust_path = Path("configs/score_pool_robustness.yaml")
robust_cfg = yaml.safe_load(robust_path.read_text())
robust_cfg["metrics"]["cutoff_tau64"] = float(summary["fallback_positive_cutoff"])
robust_path.write_text(yaml.safe_dump(robust_cfg, sort_keys=False))
print("fallback tau=64 cutoff:", robust_cfg["metrics"]["cutoff_tau64"])
print("fallback tau=32 cutoff:", summary["fallback_negative_band_cutoff"])
```

Verify:

```python
# PYTHON CELL
fallback_meta = pd.read_parquet(f"{DRIVE}/data/fallback_score_pool_meta.parquet")
fallback_tokens = np.load(f"{DRIVE}/data/fallback_score_pool_tokens.npy", mmap_mode="r")
print(fallback_tokens.shape, fallback_tokens.dtype)
print(fallback_meta["pool_name"].value_counts())
print(fallback_meta.groupby("pool_name")["full_color_score"].agg(["min", "mean", "max"]))
if fallback_tokens.shape[0] != 5 * FALLBACK_SAMPLE_SIZE:
    raise RuntimeError("Fallback score-pool token row count mismatch")
```

## 10. Baseline Score-Pool Run and Metric Gate

Score the fallback five-pool token set with the full models. This validates the
new robustness pipeline and should produce strong pairwise metrics.

```python
# PYTHON CELL
!PYTHONPATH="{OLMO}" python scripts/10_score_pool_variant.py --config configs/score_pool_robustness.yaml --variant full
!python scripts/11_score_pool_metrics.py --config configs/score_pool_robustness.yaml
```

Inspect baseline metrics:

```python
# PYTHON CELL
metrics = pd.read_csv(f"{DRIVE}/results/score-pool-robustness-fallback/metrics_pairwise.csv")
display(metrics[[
    "variant",
    "pairwise_task",
    "roc_auc",
    "average_precision",
    "f1_at_original_cutoff",
    "f1_at_balanced_rate",
]])
```

The full baseline should be near-perfect for hard-positive vs tail-negative and
should be most difficult for hard-positive vs hard-negative, by construction.

Optional noise floor:

```python
# PYTHON CELL
!PYTHONPATH="{OLMO}" python scripts/10_score_pool_variant.py --config configs/score_pool_robustness.yaml --variant full_rescore
!python scripts/11_score_pool_metrics.py --config configs/score_pool_robustness.yaml
```

## 11. Ablation Variant Grid

Run a small pilot first:

```python
# PYTHON CELL
for variant in ["pair_top2", "cond_top2_marg_bot2", "cond_bot2_marg_top2", "cond_top2_only", "marg_top2_only"]:
    print(f"=== {variant} ===")
    !PYTHONPATH="{OLMO}" python scripts/10_score_pool_variant.py --config configs/score_pool_robustness.yaml --variant {variant}
```

Recompute metrics and inspect the pilot:

```python
# PYTHON CELL
!python scripts/11_score_pool_metrics.py --config configs/score_pool_robustness.yaml
pilot = pd.read_csv(f"{DRIVE}/results/score-pool-robustness-fallback/metrics_pairwise.csv")
display(pilot[pilot["variant"].isin(["full", "pair_top2", "cond_top2_marg_bot2", "cond_bot2_marg_top2", "cond_top2_only", "marg_top2_only"])])
```

If the pilot looks sane, run the full grid:

```python
# PYTHON CELL
variants = [
    "pair_top1", "pair_top2", "pair_top4", "pair_top6",
    "pair_mid2", "pair_mid4",
    "pair_bot1", "pair_bot2", "pair_bot4", "pair_bot6",
    "cond_top1_marg_bot1", "cond_top2_marg_bot2", "cond_top4_marg_bot4", "cond_top6_marg_bot6",
    "cond_bot1_marg_top1", "cond_bot2_marg_top2", "cond_bot4_marg_top4", "cond_bot6_marg_top6",
    "cond_top1_only", "cond_top2_only", "cond_top4_only", "cond_top6_only",
    "marg_top1_only", "marg_top2_only", "marg_top4_only", "marg_top6_only",
]
for variant in variants:
    print(f"=== {variant} ===")
    !PYTHONPATH="{OLMO}" python scripts/10_score_pool_variant.py --config configs/score_pool_robustness.yaml --variant {variant}
```

The scoring script is resumable. If Colab disconnects, rerun the same loop and
completed shards will be skipped.

## 12. Metrics, Plots, and Report Artifacts

```python
# PYTHON CELL
!python scripts/11_score_pool_metrics.py --config configs/score_pool_robustness.yaml
!python scripts/12_score_pool_plots.py --config configs/score_pool_robustness.yaml
```

Generate lightweight Markdown and HTML reports:

```python
# PYTHON CELL
import html
from pathlib import Path
import pandas as pd

results_dir = Path(f"{DRIVE}/results/score-pool-robustness-fallback")
report_dir = Path(f"{DRIVE}/reports/score-pool-robustness-fallback")
report_dir.mkdir(parents=True, exist_ok=True)
metrics = pd.read_csv(results_dir / "metrics_pairwise.csv")
metric_cols = ["roc_auc", "average_precision", "f1_at_original_cutoff", "f1_at_balanced_rate"]
summary = metrics.groupby("variant")[metric_cols].mean().sort_values("roc_auc", ascending=False)
top_auc = summary.head(10)
bottom_auc = summary.tail(10)
by_task = metrics.groupby("pairwise_task")[metric_cols].mean().sort_values("roc_auc", ascending=True)
figures = [
    ("ROC AUC", "figures/auc_by_variant_and_task.png"),
    ("Average precision", "figures/ap_by_variant_and_task.png"),
    ("F1 at original cutoff", "figures/f1_original_cutoff_by_variant.png"),
    ("F1 at balanced rate", "figures/f1_balanced_rate_by_variant.png"),
    ("Color shift", "figures/color_shift_by_variant.png"),
]

report = [
    "# Score-Pool Robustness Fallback Report",
    "",
    "Label source: `fallback_full_rescore_pool`.",
    "",
    f"Source C4 sequences: `{FALLBACK_C4_SEQS:,}`",
    f"Sample size per pool: `{FALLBACK_SAMPLE_SIZE:,}`",
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
    "## Top Variants by Mean ROC AUC",
    "",
    "```text",
    top_auc.round(4).reset_index().to_string(index=False),
    "```",
    "",
    "## Lowest Variants by Mean ROC AUC",
    "",
    "```text",
    bottom_auc.round(4).reset_index().to_string(index=False),
    "```",
    "",
    "## Figures",
    "",
    *[f"- [{label}]({path})" for label, path in figures],
    "",
    "## Interpretation Notes",
    "",
    "- Compare `f1_at_original_cutoff` against `f1_at_balanced_rate` to separate calibration shift from ranking preservation.",
    "- Use `full_rescore`, when run, as the practical noise floor for score and metric drift.",
    "- Treat these as fallback-pool results, not exact official sampled-pool results.",
]
(report_dir / "report.md").write_text("\n".join(report) + "\n")

html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Score-Pool Robustness Fallback Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; line-height: 1.45; color: #1f2933; }}
    h1, h2 {{ color: #111827; }}
    table {{ border-collapse: collapse; margin: 12px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f6f8fa; }}
    img {{ max-width: 100%; border: 1px solid #d0d7de; margin: 8px 0 24px; }}
    code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Score-Pool Robustness Fallback Report</h1>
  <p>Label source: <code>fallback_full_rescore_pool</code>.</p>
  <p>Source C4 sequences: <code>{FALLBACK_C4_SEQS:,}</code><br>
  Sample size per pool: <code>{FALLBACK_SAMPLE_SIZE:,}</code></p>
  <h2>Mean Metrics by Variant</h2>
  {summary.round(4).reset_index().to_html(index=False)}
  <h2>Hardest Pairwise Tasks</h2>
  {by_task.round(4).reset_index().to_html(index=False)}
  <h2>Top Variants by Mean ROC AUC</h2>
  {top_auc.round(4).reset_index().to_html(index=False)}
  <h2>Lowest Variants by Mean ROC AUC</h2>
  {bottom_auc.round(4).reset_index().to_html(index=False)}
  <h2>Figures</h2>
  {''.join(f'<h3>{html.escape(label)}</h3><img src="{html.escape(path)}" alt="{html.escape(label)}">' for label, path in figures)}
  <h2>Interpretation Notes</h2>
  <ul>
    <li>Compare <code>f1_at_original_cutoff</code> against <code>f1_at_balanced_rate</code> to separate calibration shift from ranking preservation.</li>
    <li>Use <code>full_rescore</code>, when run, as the practical noise floor for score and metric drift.</li>
    <li>Treat these as fallback-pool results, not exact official sampled-pool results.</li>
  </ul>
</body>
</html>
"""
(report_dir / "report.html").write_text(html_doc)
print(report_dir / "report.md")
print(report_dir / "report.html")
```

No `pandoc` install is required.

## 13. Outputs to Bring Back Locally

Copy or download these from Drive after the run:

```text
results/score-pool-robustness-fallback/fallback_pool_summary.json
results/score-pool-robustness-fallback/metrics_pairwise.csv
results/score-pool-robustness-fallback/score_shift_diagnostics.csv
reports/score-pool-robustness-fallback/figures/*.png
reports/score-pool-robustness-fallback/report.md
reports/score-pool-robustness-fallback/report.html
```

Large files can remain on Drive:

```text
data/fallback_source_pool.npy
data/fallback_score_pool_tokens.npy
results/score-pool-robustness-fallback/scores_*.parquet
```

## 14. Interpretation Checklist

Read each ablation family through two lenses:

- **Calibration robustness:** F1 at the original `tau=64` cutoff.
- **Ranking robustness:** ROC AUC, AP, and F1 at fixed predicted-positive rate.

Asymmetric ablations can fail calibration while preserving ranking. That is not
the same failure mode as destroying the CoLoR ordering. The final report should
separate these two cases.
