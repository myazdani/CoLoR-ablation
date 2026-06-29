#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ensure_parent, load_config
from src.score_pool_robustness import compute_pairwise_metrics


DEFAULT_LAYER_BASELINE_DIR = (
    ROOT
    / "data"
    / "pair_mid2_cascade_full_rerank"
    / "results"
    / "score-pool-robustness-official-500k"
)


def setup_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    return plt


def dataframe_to_markdown(frame: pd.DataFrame, *, floatfmt: str = ".4f") -> str:
    display = frame.copy()

    def format_value(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        return str(value)

    headers = [str(column) for column in display.columns]
    rows = [[format_value(value) for value in row] for row in display.itertuples(index=False, name=None)]
    widths = [
        max(len(header), *(len(row[column_idx]) for row in rows)) if rows else len(header)
        for column_idx, header in enumerate(headers)
    ]

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render_row(headers), separator, *(render_row(row) for row in rows)])


def read_score_frames(results_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(results_dir.glob("scores_*.parquet")):
        frame = pd.read_parquet(path)
        if "sequence_window_id" in frame.columns and len(frame):
            variant = str(frame["sequence_window_id"].iloc[0])
        elif "variant_id" in frame.columns and len(frame):
            variant = str(frame["variant_id"].iloc[0])
        else:
            variant = path.stem.removeprefix("scores_")
        frames[variant] = frame
    return frames


def compute_metrics(
    frames: dict[str, pd.DataFrame],
    *,
    cutoff_tau64: float,
    pairwise_tasks: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames: list[pd.DataFrame] = []
    shift_frames: list[pd.DataFrame] = []
    for variant, frame in frames.items():
        metrics, shifts = compute_pairwise_metrics(
            frame,
            variant_id=variant,
            cutoff_tau64=cutoff_tau64,
            pairwise_tasks=list(pairwise_tasks),
        )
        if "effective_sequence_length" in frame.columns:
            effective = int(frame["effective_sequence_length"].iloc[0])
            label = str(frame.get("sequence_window_label", pd.Series([variant])).iloc[0])
        else:
            effective = math.nan
            label = variant
        metrics["sequence_window_id"] = variant
        metrics["sequence_window_label"] = label
        metrics["effective_sequence_length"] = effective
        shifts["sequence_window_id"] = variant
        shifts["sequence_window_label"] = label
        shifts["effective_sequence_length"] = effective
        metric_frames.append(metrics)
        shift_frames.append(shifts)
    return pd.concat(metric_frames, ignore_index=True), pd.concat(shift_frames, ignore_index=True)


def runtime_by_window(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for variant, frame in frames.items():
        stats = frame[["elapsed_seconds", "tokens_scored", "tokens_per_second"]].drop_duplicates()
        if len(stats) != 1:
            raise ValueError(f"Expected one runtime row for {variant}, found {len(stats)}")
        row = stats.iloc[0]
        effective = (
            int(frame["effective_sequence_length"].iloc[0])
            if "effective_sequence_length" in frame.columns
            else math.nan
        )
        rows.append(
            {
                "sequence_window_id": variant,
                "sequence_window_label": str(frame.get("sequence_window_label", pd.Series([variant])).iloc[0]),
                "effective_sequence_length": effective,
                "rows": len(frame),
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "tokens_scored": int(row["tokens_scored"]),
                "sequences_per_second": len(frame) / float(row["elapsed_seconds"])
                if float(row["elapsed_seconds"]) > 0
                else float("nan"),
                "model_tokens_per_second": float(row["tokens_per_second"]),
                "gpu_type": str(frame.get("cuda_device_name", pd.Series([""])).iloc[0]),
                "batch_size": int(frame["batch_size"].iloc[0]) if "batch_size" in frame.columns else math.nan,
                "scoring_dtype": str(frame.get("scoring_dtype", pd.Series([""])).iloc[0]),
                "scoring_device": str(frame.get("scoring_device", pd.Series([""])).iloc[0]),
            }
        )
    runtime = pd.DataFrame(rows)
    baseline = runtime[runtime["sequence_window_id"] == "seq_full_512"]
    if not baseline.empty:
        full_elapsed = float(baseline["elapsed_seconds"].iloc[0])
        runtime["speedup_vs_seq_full_512"] = full_elapsed / runtime["elapsed_seconds"]
    else:
        runtime["speedup_vs_seq_full_512"] = math.nan
    runtime["nominal_token_speedup_vs_512"] = 512 / runtime["effective_sequence_length"]
    return runtime.sort_values(["effective_sequence_length", "sequence_window_id"], ascending=[False, True])


def add_layer_baseline_metrics(
    metrics: pd.DataFrame,
    *,
    layer_baseline_dir: Path,
    cutoff_tau64: float,
    pairwise_tasks: Iterable[str],
) -> pd.DataFrame:
    path = layer_baseline_dir / "scores_pair_mid2.parquet"
    if not path.exists():
        return metrics
    frame = pd.read_parquet(path)
    layer_metrics, _ = compute_pairwise_metrics(
        frame,
        variant_id="pair_mid2",
        cutoff_tau64=cutoff_tau64,
        pairwise_tasks=list(pairwise_tasks),
    )
    layer_metrics["sequence_window_id"] = "pair_mid2"
    layer_metrics["sequence_window_label"] = "pair_mid2 layer deletion"
    layer_metrics["effective_sequence_length"] = 512
    return pd.concat([metrics, layer_metrics], ignore_index=True)


def sort_windows(metrics: pd.DataFrame) -> list[str]:
    order = [
        "seq_full_512",
        "seq_prefix_256",
        "seq_suffix_256",
        "seq_middle_256",
        "seq_prefix_suffix_256",
        "seq_prefix_128",
        "seq_suffix_128",
        "pair_mid2",
    ]
    present = set(metrics["sequence_window_id"].astype(str))
    return [item for item in order if item in present] + sorted(present - set(order))


def plot_metric_bars(metrics: pd.DataFrame, metric: str, output: Path, title: str) -> None:
    plt = setup_matplotlib()
    tasks = list(dict.fromkeys(metrics["pairwise_task"].astype(str)))
    windows = sort_windows(metrics)
    fig, axes = plt.subplots(len(tasks), 1, figsize=(max(8, len(windows) * 0.62), 2.6 * len(tasks)))
    if len(tasks) == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        sub = metrics[metrics["pairwise_task"] == task].set_index("sequence_window_id").reindex(windows)
        ax.bar(np.arange(len(sub)), sub[metric].to_numpy(dtype=float), color="#4c78a8")
        ax.set_title(task)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(metric)
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(sub.index, rotation=45, ha="right", fontsize=8)
    fig.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_quality_vs_tokens(metrics: pd.DataFrame, runtime: pd.DataFrame, output: Path) -> None:
    plt = setup_matplotlib()
    summary = (
        metrics[metrics["sequence_window_id"] != "pair_mid2"]
        .groupby(["sequence_window_id", "sequence_window_label", "effective_sequence_length"], as_index=False)
        .agg(mean_roc_auc=("roc_auc", "mean"), mean_f1=("f1_at_balanced_rate", "mean"))
    )
    summary = summary.merge(
        runtime[["sequence_window_id", "speedup_vs_seq_full_512"]],
        on="sequence_window_id",
        how="left",
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    scatter = ax.scatter(
        summary["mean_roc_auc"],
        summary["speedup_vs_seq_full_512"],
        s=80,
        c=summary["effective_sequence_length"],
        cmap="viridis",
    )
    for _, row in summary.iterrows():
        ax.annotate(
            row["sequence_window_id"].replace("seq_", ""),
            (row["mean_roc_auc"], row["speedup_vs_seq_full_512"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.axvline(0.80, color="black", linestyle=":", linewidth=1)
    ax.axhline(1.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("mean ROC AUC across pairwise tasks")
    ax.set_ylabel("measured speedup vs 512-token scoring")
    ax.set_title("Speed-quality Pareto")
    fig.colorbar(scatter, ax=ax, label="effective tokens")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_runtime_vs_tokens(runtime: pd.DataFrame, output: Path) -> None:
    plt = setup_matplotlib()
    frame = runtime.dropna(subset=["effective_sequence_length"]).copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(frame["effective_sequence_length"], frame["elapsed_seconds"], s=70)
    for _, row in frame.iterrows():
        ax.annotate(
            row["sequence_window_id"].replace("seq_", ""),
            (row["effective_sequence_length"], row["elapsed_seconds"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("effective sequence length")
    ax.set_ylabel("elapsed seconds")
    ax.set_title("Runtime by effective sequence length")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_scatter(results_dir: Path, output: Path, *, variant: str = "seq_prefix_256", max_points: int = 20000) -> None:
    path = results_dir / f"scores_{variant}.parquet"
    if not path.exists():
        return
    frame = pd.read_parquet(path, columns=["full_color_score", "ablated_color_score"])
    if len(frame) > max_points:
        frame = frame.sample(max_points, random_state=17)
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(5.7, 5.2))
    ax.scatter(frame["full_color_score"], frame["ablated_color_score"], s=3, alpha=0.25)
    ax.set_xlabel("official/released full CoLoR score")
    ax.set_ylabel(f"{variant} CoLoR score")
    ax.set_title(f"Windowed vs full CoLoR: {variant}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def write_plots(metrics: pd.DataFrame, runtime: pd.DataFrame, results_dir: Path, figures_dir: Path) -> None:
    plot_metric_bars(
        metrics,
        "roc_auc",
        figures_dir / "auc_by_window_and_task.png",
        "ROC AUC by sequence window and task",
    )
    plot_metric_bars(
        metrics,
        "f1_at_balanced_rate",
        figures_dir / "f1_balanced_by_window_and_task.png",
        "Balanced-rate F1 by sequence window and task",
    )
    plot_quality_vs_tokens(metrics, runtime, figures_dir / "quality_vs_effective_tokens.png")
    plot_runtime_vs_tokens(runtime, figures_dir / "runtime_vs_effective_tokens.png")
    plot_scatter(results_dir, figures_dir / "window_color_scatter_prefix256.png")


def write_html_report(markdown_path: Path, html_path: Path) -> None:
    body_lines: list[str] = []
    in_code = False
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    idx = 0

    def split_table_row(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    while idx < len(lines):
        line = lines[idx]
        if line.startswith("```"):
            body_lines.append("</code></pre>" if in_code else "<pre><code>")
            in_code = not in_code
            idx += 1
            continue
        if in_code:
            body_lines.append(html.escape(line))
            idx += 1
            continue
        if line.startswith("# "):
            body_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("| "):
            table_lines = []
            while idx < len(lines) and lines[idx].startswith("| "):
                table_lines.append(lines[idx])
                idx += 1
            parsed = [split_table_row(table_line) for table_line in table_lines]
            if len(parsed) >= 2:
                body_lines.append("<table><thead><tr>")
                body_lines.extend(f"<th>{html.escape(cell)}</th>" for cell in parsed[0])
                body_lines.append("</tr></thead><tbody>")
                for row in parsed[2:]:
                    body_lines.append("<tr>")
                    body_lines.extend(f"<td>{html.escape(cell)}</td>" for cell in row)
                    body_lines.append("</tr>")
                body_lines.append("</tbody></table>")
            continue
        elif line.startswith("- "):
            items = []
            while idx < len(lines) and lines[idx].startswith("- "):
                items.append(lines[idx][2:])
                idx += 1
            body_lines.append("<ul>")
            body_lines.extend(f"<li>{html.escape(item)}</li>" for item in items)
            body_lines.append("</ul>")
            continue
        elif line.startswith("![") and "](" in line and line.endswith(")"):
            alt = line[2 : line.index("]")]
            src = line[line.index("(") + 1 : -1]
            body_lines.append(
                f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}">'
                f"<figcaption>{html.escape(alt)}</figcaption></figure>"
            )
        elif line.strip():
            body_lines.append(f"<p>{html.escape(line)}</p>")
        else:
            body_lines.append("")
        idx += 1

    html_text = "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset=\"utf-8\">",
            "<title>Sequence-Length Ablations for Faster CoLoR Scoring</title>",
            "<style>",
            "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 980px; margin: 40px auto; line-height: 1.55; color: #1f2933; padding: 0 20px; }",
            "table { border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 0.92em; }",
            "th, td { border: 1px solid #d7dde5; padding: 7px 9px; text-align: left; vertical-align: top; }",
            "th { background: #eef2f7; }",
            "pre { background: #f6f8fa; padding: 12px; overflow-x: auto; }",
            "img { max-width: 100%; border: 1px solid #ddd; }",
            "figure { margin: 24px 0; }",
            "</style></head><body>",
            *body_lines,
            "</body></html>",
        ]
    )
    html_path.write_text(html_text, encoding="utf-8")


def write_report(
    *,
    metrics: pd.DataFrame,
    shifts: pd.DataFrame,
    runtime: pd.DataFrame,
    report_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    sequence_metrics = metrics[metrics["sequence_window_id"] != "pair_mid2"].copy()
    summary = (
        sequence_metrics.groupby(
            ["sequence_window_id", "sequence_window_label", "effective_sequence_length"],
            as_index=False,
        )
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            mean_balanced_f1=("f1_at_balanced_rate", "mean"),
            min_roc_auc=("roc_auc", "min"),
            hp_vs_hn_auc=("roc_auc", lambda s: float(s.iloc[0])),
        )
        .merge(
            runtime[
                [
                    "sequence_window_id",
                    "elapsed_seconds",
                    "speedup_vs_seq_full_512",
                    "model_tokens_per_second",
                ]
            ],
            on="sequence_window_id",
            how="left",
        )
    )
    hp_hn = metrics[metrics["pairwise_task"] == "hp_vs_hn"][
        [
            "sequence_window_id",
            "roc_auc",
            "f1_at_balanced_rate",
            "pearson_color",
            "spearman_color",
        ]
    ].copy()
    pair_mid2 = metrics[metrics["sequence_window_id"] == "pair_mid2"]
    pair_mid2_text = "No pair_mid2 layer-deletion baseline parquet was available for this report."
    if not pair_mid2.empty:
        pair_mid2_text = (
            f"pair_mid2 mean ROC AUC={pair_mid2['roc_auc'].mean():.4f}, "
            f"mean balanced F1={pair_mid2['f1_at_balanced_rate'].mean():.4f}."
        )

    passing = summary[
        (summary["mean_roc_auc"] >= 0.80)
        & (summary["mean_balanced_f1"] >= 0.75)
        & (summary["speedup_vs_seq_full_512"] >= 1.5)
    ]
    if passing.empty:
        decision = (
            "No scored sequence window simultaneously meets the mean-quality and measured-speed "
            "criteria. Shorter windows may still be useful as first-stage prefilters if their "
            "top-of-ranking recall is acceptable in a follow-up cascade analysis."
        )
    else:
        best = passing.sort_values(["speedup_vs_seq_full_512", "mean_roc_auc"], ascending=[False, False]).iloc[0]
        decision = (
            f"{best['sequence_window_id']} is viable as an approximate faster scorer under the "
            f"tested criteria: mean ROC AUC={best['mean_roc_auc']:.4f}, "
            f"mean balanced F1={best['mean_balanced_f1']:.4f}, "
            f"speedup={best['speedup_vs_seq_full_512']:.4f}x."
        )

    lines = [
        "# Sequence-Length Ablations for Faster CoLoR Scoring",
        "",
        "## Abstract",
        "",
        "This report evaluates whether scoring fewer than 512 tokens preserves enough Books-targeted CoLoR signal to accelerate score-pool classification. Each window uses the full conditional Books model and full marginal model, but computes mean next-token loss only on the selected token window.",
        "",
        decision,
        "",
        "## Background",
        "",
        "Layer-deletion experiments showed that `pair_mid2` preserves some signal but gives limited measured speedup because it still runs two 10-layer models over all 512 tokens. Sequence-length reduction attacks a different cost axis: it keeps model depth intact while reducing the number of token positions scored.",
        "",
        "## Methods",
        "",
        "For each sequence window, the recovered official 500K token pool is sliced before scoring. Contiguous windows use `[start:end]`; the prefix+suffix window concatenates `[0:128]` and `[384:512]` in that order. For every row, CoLoR is computed as conditional Books NLL minus marginal/prior NLL, with lower scores treated as more Books-like.",
        "",
        "Metrics use the same six balanced pairwise tasks as the score-pool robustness analysis. Ranking metrics use `decision_score = -color_score`. Runtime is read from the scorer output parquet metadata.",
        "",
        "## Results Summary",
        "",
        dataframe_to_markdown(
            summary[
                [
                    "sequence_window_id",
                    "effective_sequence_length",
                    "mean_roc_auc",
                    "mean_balanced_f1",
                    "min_roc_auc",
                    "elapsed_seconds",
                    "speedup_vs_seq_full_512",
                    "model_tokens_per_second",
                ]
            ],
            floatfmt=".4f",
        ),
        "",
        "## Hard-Boundary Task",
        "",
        "The hardest task is expected to be `hp_vs_hn`, because both pools sit close to the tau=64 selection boundary.",
        "",
        dataframe_to_markdown(hp_hn, floatfmt=".4f"),
        "",
        "## Layer-Deletion Comparison",
        "",
        pair_mid2_text,
        "",
        "## Figures",
        "",
        "![AUC by window and task](figures/auc_by_window_and_task.png)",
        "",
        "![Balanced F1 by window and task](figures/f1_balanced_by_window_and_task.png)",
        "",
        "![Speed-quality Pareto](figures/quality_vs_effective_tokens.png)",
        "",
        "![Runtime by effective tokens](figures/runtime_vs_effective_tokens.png)",
        "",
        "![Prefix 256 color scatter](figures/window_color_scatter_prefix256.png)",
        "",
        "## Limitations",
        "",
        "- This report requires actual `scores_<window>.parquet` files from a GPU run; missing windows are not imputed.",
        "- Original tau=64 cutoff metrics are calibration diagnostics because shorter-window scores can shift scale.",
        "- Runtime comparisons are only fair when windows are scored on the same hardware with comparable batch tuning.",
        "- Strided optional windows are excluded unless explicitly scored.",
        "",
        "## Conclusion",
        "",
        decision,
        "",
        "## Reproducibility",
        "",
        "Run sequence-window scoring on Colab, then compute metrics and render this report locally or in Colab:",
        "",
        "```bash",
        "python scripts/17_sequence_length_metrics_report.py --config configs/sequence_length_score_pool.yaml",
        "```",
        "",
    ]
    report_md = report_dir / "report.md"
    report_md.write_text("\n".join(lines), encoding="utf-8")
    write_html_report(report_md, report_dir / "report.html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute sequence-length score-pool metrics and report.")
    parser.add_argument("--config", default="configs/sequence_length_score_pool.yaml")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--report-dir", default="reports/sequence-length-score-pool")
    parser.add_argument("--layer-baseline-dir", default=str(DEFAULT_LAYER_BASELINE_DIR))
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = Path(args.results_dir or config["paths"]["output_dir"])
    report_dir = Path(args.report_dir)
    figures_dir = report_dir / "figures"
    frames = read_score_frames(results_dir)
    if not frames:
        raise RuntimeError(f"No scores_*.parquet files found under {results_dir}")

    metrics, shifts = compute_metrics(
        frames,
        cutoff_tau64=float(config["metrics"]["cutoff_tau64"]),
        pairwise_tasks=config["metrics"].get("pairwise_tasks"),
    )
    metrics = add_layer_baseline_metrics(
        metrics,
        layer_baseline_dir=Path(args.layer_baseline_dir),
        cutoff_tau64=float(config["metrics"]["cutoff_tau64"]),
        pairwise_tasks=config["metrics"].get("pairwise_tasks"),
    )
    runtime = runtime_by_window(frames)

    metrics_out = ensure_parent(results_dir / "metrics_pairwise_sequence_length.csv")
    shifts_out = ensure_parent(results_dir / "score_shift_diagnostics_sequence_length.csv")
    runtime_out = ensure_parent(results_dir / "runtime_by_window.csv")
    metrics.to_csv(metrics_out, index=False)
    shifts.to_csv(shifts_out, index=False)
    runtime.to_csv(runtime_out, index=False)
    write_plots(metrics, runtime, results_dir, figures_dir)
    write_report(metrics=metrics, shifts=shifts, runtime=runtime, report_dir=report_dir)
    print(f"wrote {metrics_out}")
    print(f"wrote {shifts_out}")
    print(f"wrote {runtime_out}")
    print(f"wrote {report_dir / 'report.md'}")
    print(f"wrote {report_dir / 'report.html'}")


if __name__ == "__main__":
    main()
