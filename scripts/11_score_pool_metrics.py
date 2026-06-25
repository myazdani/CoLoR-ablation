#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ensure_parent, load_config
from src.score_pool_robustness import compute_pairwise_metrics


def read_score_frames(results_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(results_dir.glob("scores_*.parquet")):
        frame = pd.read_parquet(path)
        if "variant_id" in frame.columns and len(frame):
            variant = str(frame["variant_id"].iloc[0])
        else:
            variant = path.stem.removeprefix("scores_")
        frames[variant] = frame
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute pairwise score-pool robustness metrics.")
    parser.add_argument("--config", default="configs/score_pool_robustness.yaml")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = Path(args.results_dir or config["paths"]["output_dir"])
    frames = read_score_frames(results_dir)
    if not frames:
        raise RuntimeError(f"No scores_*.parquet files found under {results_dir}")

    metric_frames: list[pd.DataFrame] = []
    shift_frames: list[pd.DataFrame] = []
    for variant, frame in frames.items():
        metrics, shifts = compute_pairwise_metrics(
            frame,
            variant_id=variant,
            cutoff_tau64=float(config["metrics"]["cutoff_tau64"]),
            pairwise_tasks=config["metrics"].get("pairwise_tasks"),
        )
        metric_frames.append(metrics)
        shift_frames.append(shifts)

    metrics_out = ensure_parent(results_dir / "metrics_pairwise.csv")
    shifts_out = ensure_parent(results_dir / "score_shift_diagnostics.csv")
    pd.concat(metric_frames, ignore_index=True).to_csv(metrics_out, index=False)
    pd.concat(shift_frames, ignore_index=True).to_csv(shifts_out, index=False)
    print(f"wrote {metrics_out}")
    print(f"wrote {shifts_out}")


if __name__ == "__main__":
    main()
