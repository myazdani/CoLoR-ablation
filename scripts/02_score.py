#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ablation import apply_paired_ablation, variant_layer_indices
from src.config import ensure_parent, find_variant, load_config
from src.model_loading import load_causal_lm
from src.scoring import score_pair


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _pool_hash(pool_path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(pool_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one model variant on the frozen pool.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--score-id",
        default=None,
        help="Output/metadata id. Use this for repeated runs, e.g. --variant full --score-id full_rescore.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    find_variant(config, args.variant)
    target = config["target"]
    scoring_cfg = config["scoring"]
    total_layers = int(scoring_cfg["total_layers"])
    removed_layers = variant_layer_indices(args.variant, total_layers=total_layers)
    score_id = args.score_id or args.variant

    pool = np.load(target["pool_tokens"])
    cond = load_causal_lm(
        target["cond_checkpoint"],
        paper_code_path=config.get("paths", {}).get("paper_code"),
        dtype=scoring_cfg.get("dtype", "bf16"),
    )
    marg = load_causal_lm(
        target["marg_checkpoint"],
        paper_code_path=config.get("paths", {}).get("paper_code"),
        dtype=scoring_cfg.get("dtype", "bf16"),
    )

    if removed_layers:
        cond_record, marg_record = apply_paired_ablation(
            cond,
            marg,
            removed_layers,
            total_layers=total_layers,
        )
        if cond_record.removed_layers != marg_record.removed_layers:
            raise AssertionError("Paired ablation metadata mismatch")

    scores, stats = score_pair(
        cond,
        marg,
        pool,
        batch_size=int(scoring_cfg["batch_size"]),
        device=scoring_cfg.get("device", "auto"),
        dtype=scoring_cfg.get("dtype", "bf16"),
    )

    scores["target"] = target["name"]
    scores["variant"] = score_id
    scores["ablation_variant"] = args.variant
    scores["removed_layers"] = json.dumps(list(removed_layers))
    scores["git_commit"] = _git_commit()
    scores["pool_sha256"] = _pool_hash(target["pool_tokens"])
    scores["cond_checkpoint"] = str(target["cond_checkpoint"])
    scores["marg_checkpoint"] = str(target["marg_checkpoint"])
    for key, value in stats.items():
        scores[key] = value

    output = args.output or Path(config["paths"]["results_dir"]) / f"scores_{score_id}.parquet"
    output_path = ensure_parent(output)
    scores.to_parquet(output_path, index=False)
    print(f"Wrote {output_path}")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
