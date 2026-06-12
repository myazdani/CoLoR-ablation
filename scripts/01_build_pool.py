#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ensure_parent, load_config
from src.packing import build_pool_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and freeze the packed evaluation pool.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    result = build_pool_from_config(config)

    target = config["target"]
    pool_tokens = ensure_parent(target["pool_tokens"])
    pool_meta = ensure_parent(target["pool_meta"])
    np.save(pool_tokens, result.tokens)
    result.metadata.to_parquet(pool_meta, index=False)

    n_enriched = int(result.metadata["enriched"].sum())
    print(f"Wrote {pool_tokens} with shape={result.tokens.shape}")
    print(f"Wrote {pool_meta} with n_enriched={n_enriched}")


if __name__ == "__main__":
    main()

