#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ablation import variant_layer_indices
from src.config import ensure_parent, load_config
from src.metrics import compute_all_metrics


def _read_score_frames(results_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    frames: dict[str, pd.DataFrame] = {}
    speeds: dict[str, float] = {}
    for path in sorted(results_dir.glob("scores_*.parquet")):
        frame = pd.read_parquet(path)
        if "variant" in frame.columns:
            variant = str(frame["variant"].iloc[0])
        else:
            variant = path.stem.removeprefix("scores_")
        frames[variant] = frame
        if "tokens_per_second" in frame.columns:
            speeds[variant] = float(frame["tokens_per_second"].iloc[0])
    return frames, speeds


def _removed_from_frame_or_variant(frame: pd.DataFrame, variant: str, total_layers: int) -> tuple[int, ...]:
    if "removed_layers" in frame.columns:
        raw = frame["removed_layers"].iloc[0]
        try:
            parsed = json.loads(raw)
            return tuple(int(i) for i in parsed)
        except Exception:
            pass
    return variant_layer_indices(variant, total_layers=total_layers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ablation metrics from score parquets.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--full-variant", default="full")
    args = parser.parse_args()

    config = load_config(args.config)
    target = config["target"]
    metrics_cfg = config["metrics"]
    total_layers = int(config["scoring"]["total_layers"])

    score_frames, speeds = _read_score_frames(Path(config["paths"]["results_dir"]))
    if not score_frames:
        raise RuntimeError("No score parquet files found")
    removed = {
        variant: _removed_from_frame_or_variant(frame, variant, total_layers)
        for variant, frame in score_frames.items()
    }
    pool_meta = pd.read_parquet(target["pool_meta"])
    metrics = compute_all_metrics(
        score_frames=score_frames,
        pool_meta=pool_meta,
        variants_removed_layers=removed,
        full_variant=args.full_variant,
        total_layers=total_layers,
        selection_rates=metrics_cfg["selection_rates"],
        bootstrap_reps=int(metrics_cfg["bootstrap_reps"]),
        bootstrap_seed=int(metrics_cfg["bootstrap_seed"]),
        tail_rate_for_local_spearman=float(metrics_cfg["tail_rate_for_local_spearman"]),
        variant_tokens_per_second=speeds,
    )

    output = ensure_parent(config["paths"]["metrics_csv"])
    metrics.to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

