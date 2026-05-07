#!/usr/bin/env bash
set -euo pipefail

BASE_COMMIT="b1388b1fbf5aaef47937fabe98931211684666a6"
VLLM_REPO_URL="${VLLM_REPO_URL:-https://github.com/vllm-project/vllm.git}"
PATCH_SRC_DIR="${PATCH_SRC_DIR:-/workspace/vllm-runtime-patch}"
SITE_PACKAGES="${SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
MODEL_PATH="${MODEL_PATH:-/home/shared/megatron_dir/hf_models/Qwen3.5-35B-A3B-FP8/}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-3003}"
TP="${TP:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
PRESERVE_THINKING="${PRESERVE_THINKING:-false}"
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS

usage() {
  cat >&2 <<'USAGE'
Usage:
  bash serve.sh <base|mamba|mtp>

Modes:
  base   Serve the installed vLLM image without applying gold.patch.
  mamba  Apply gold.patch, then serve Qwen hybrid in align cache mode.
  mtp    Apply gold.patch, then serve align cache mode plus 3-token MTP.

Environment knobs:
  MODEL_PATH, SERVED_MODEL_NAME, HOST, PORT, TP, GPU_MEMORY_UTILIZATION,
  MAX_NUM_BATCHED_TOKENS, PRESERVE_THINKING,
  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS.

Prompt-tail caching is targeted at preserve_thinking=false. The default here is
PRESERVE_THINKING=false.
CUDA graph memory estimation is enabled by default to avoid over-allocating KV
cache before graph capture.
USAGE
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

MODE="$1"
case "$MODE" in
  base|mamba|mtp) ;;
  *)
    usage
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GOLD_PATCH="${SCRIPT_DIR}/gold.patch"

install_patch() {
  [[ -f "$GOLD_PATCH" ]] || {
    echo "Missing gold.patch next to serve.sh: $GOLD_PATCH" >&2
    exit 1
  }
  if ! command -v git >/dev/null; then
    apt-get update
    apt-get install -y --no-install-recommends git ca-certificates
  fi

  rm -rf "$PATCH_SRC_DIR"
  git clone --quiet "$VLLM_REPO_URL" "$PATCH_SRC_DIR"
  git -C "$PATCH_SRC_DIR" checkout --quiet "$BASE_COMMIT"
  git -C "$PATCH_SRC_DIR" apply "$GOLD_PATCH"

  mapfile -t runtime_files < <(
    git -C "$PATCH_SRC_DIR" diff --name-only --diff-filter=ACM | grep '^vllm/'
  )
  if [[ "${#runtime_files[@]}" -eq 0 ]]; then
    echo "gold.patch did not change any runtime files" >&2
    exit 1
  fi

  for rel_path in "${runtime_files[@]}"; do
    install -D -m 0644 \
      "${PATCH_SRC_DIR}/${rel_path}" \
      "${SITE_PACKAGES}/${rel_path}"
  done
  rm -rf "$PATCH_SRC_DIR"
  unset PYTHONPATH
}

if [[ "$MODE" != "base" ]]; then
  install_patch
fi

CHAT_KWARGS="{\"enable_thinking\": false, \"preserve_thinking\": ${PRESERVE_THINKING}}"

cmd=(
  vllm serve "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TP"
  --enable-expert-parallel
  --max-model-len 65536
  --kv-cache-dtype fp8
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --enable-prefix-caching
  --language-model-only
  --default-chat-template-kwargs "$CHAT_KWARGS"
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --trust-remote-code
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --performance-mode throughput
  --async-scheduling
  --mamba-ssm-cache-dtype float16
  --no-scheduler-reserve-full-isl
)

if [[ "$MODE" == "mamba" || "$MODE" == "mtp" ]]; then
  cmd+=(--mamba-cache-mode align)
fi

if [[ "$MODE" == "mtp" ]]; then
  cmd+=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
fi

printf 'Serving mode: %s\n' "$MODE"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
