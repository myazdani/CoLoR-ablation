#!/usr/bin/env bash
set -euo pipefail

cat <<'MSG'
This script is intentionally a guardrail, not an automatic downloader.

For local testing, use the official `hf` CLI from the repo root:

  mkdir -p assets/raw
  export HF_HOME="$PWD/assets/.hf-cache"
  hf download hlzhang109/CoLoR-filter \
    models/prior/config.yaml models/prior/model.pt \
    --local-dir assets/raw
  hf download hlzhang109/CoLoR-filter \
    models/conditional_books/config.yaml models/conditional_books/model.pt \
    --local-dir assets/raw

Do not download the entire hlzhang109/CoLoR-filter repository.
MSG
