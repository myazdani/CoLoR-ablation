#!/usr/bin/env bash
set -euo pipefail

cat <<'MSG'
This script is intentionally a guardrail, not an automatic downloader.

Follow docs/HF_GPU_RUNBOOK.md after identifying the exact Books prior and
conditional checkpoint folders on the Hugging Face website. Do not download the
entire hlzhang109/CoLoR-filter repository.
MSG

