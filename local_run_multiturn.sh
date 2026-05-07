#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")"

python3 scripts/multiturn_vllm_metrics.py \
  --base-url http://127.0.0.1:3004 \
  --model qwen3 \
  --turns 10 \
  --input-tokens 400 \
  --output-tokens 200 \
  --min-output-tokens 200 \
  --temperature 0 \
  --trace-json
