#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/serve.sh base --preserve-thinking true|false [extra vLLM args...]
  scripts/serve.sh mamba --preserve-thinking true|false [extra vLLM args...]
  scripts/serve.sh mtp --preserve-thinking true|false [extra vLLM args...]

Required arguments:
  mode                         One of: base, mamba, mtp.
  --preserve-thinking VALUE    VALUE must be true or false.

Modes:
  base   Serve the unpatched v0.19.1 image baseline.
  mamba  Apply gold.patch into site-packages, then serve latest-Mamba.
  mtp    Apply gold.patch into site-packages, then serve latest-Mamba + MTP.

Run inside a vllm/vllm-openai:v0.19.1 container after cloning this artifact repo.
Use a fresh container for a true base run; this script does not undo patches.

Common environment overrides:
  MODEL_PATH, SERVED_MODEL_NAME, HOST, PORT, GPU_MEMORY_UTILIZATION
  MAX_MODEL_LEN, MAX_NUM_BATCHED_TOKENS, TAIL_CHECKPOINTS, COARSE_MIN_GAP
  NUM_SPECULATIVE_TOKENS, CHAT_TEMPLATE

Preserve-thinking can be passed as --preserve-thinking true|false or as
PRESERVE_THINKING=true|false. It is required so benchmark runs cannot
accidentally mix preserve-thinking settings.
EOF
}

if [[ $# -lt 1 ]]; then
  echo "Missing required mode." >&2
  echo >&2
  usage >&2
  exit 2
fi

MODE="$1"
shift

case "$MODE" in
  base|mamba|mtp) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

PRESERVE_THINKING="${PRESERVE_THINKING:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preserve-thinking)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --preserve-thinking; expected true or false." >&2
        exit 2
      fi
      PRESERVE_THINKING="$2"
      shift 2
      ;;
    --preserve-thinking=*)
      PRESERVE_THINKING="${1#*=}"
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

case "$PRESERVE_THINKING" in
  true|false) ;;
  *)
    echo "You must provide preserve-thinking explicitly: true or false." >&2
    echo "Example: scripts/serve.sh $MODE --preserve-thinking true" >&2
    exit 2
    ;;
esac

BASE_COMMIT="${BASE_COMMIT:-b1388b1fbf5aaef47937fabe98931211684666a6}"
VLLM_REPO_URL="${VLLM_REPO_URL:-https://github.com/vllm-project/vllm.git}"
PATCH_WORKDIR="${PATCH_WORKDIR:-/workspace/vllm-runtime-patch}"
SITE_PACKAGES="${SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GOLD_PATCH="${GOLD_PATCH:-${REPO_ROOT}/gold.patch}"
PATCH_MARKER="${PATCH_MARKER:-/tmp/vllm-qwen35-gold-installed}"
RUNTIME_CWD="${RUNTIME_CWD:-/tmp}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.6-35B-A3B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-3003}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
TAIL_CHECKPOINTS="${TAIL_CHECKPOINTS:-0}"
COARSE_MIN_GAP="${COARSE_MIN_GAP:-512}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
CHAT_TEMPLATE_KWARGS="{\"enable_thinking\": false, \"preserve_thinking\": ${PRESERVE_THINKING}}"

export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}"

cleanup_patch_workdir() {
  case "$PATCH_WORKDIR" in
    ""|"/"|"$REPO_ROOT"|"$SITE_PACKAGES"|"$SITE_PACKAGES"/*)
      echo "Refusing unsafe PATCH_WORKDIR cleanup: $PATCH_WORKDIR" >&2
      exit 1
      ;;
  esac
  rm -rf "$PATCH_WORKDIR"
}

install_gold_patch() {
  if [[ -f "$PATCH_MARKER" ]]; then
    echo "gold.patch already installed: $PATCH_MARKER"
    return
  fi
  if [[ ! -f "$GOLD_PATCH" ]]; then
    echo "Missing gold.patch: $GOLD_PATCH" >&2
    exit 1
  fi
  if ! command -v git >/dev/null 2>&1; then
    echo "Missing required command: git" >&2
    exit 1
  fi

  mapfile -t runtime_files < <(
    grep '^diff --git a/vllm/' "$GOLD_PATCH" \
      | sed 's#^diff --git a/\([^ ]*\).*#\1#' \
      | sort -u
  )
  if [[ "${#runtime_files[@]}" -eq 0 ]]; then
    echo "gold.patch did not contain any vllm/ runtime files." >&2
    exit 1
  fi

  cleanup_patch_workdir
  mkdir -p "$PATCH_WORKDIR"
  git clone --no-checkout "$VLLM_REPO_URL" "$PATCH_WORKDIR/vllm"
  cd "$PATCH_WORKDIR/vllm"
  git checkout "$BASE_COMMIT"
  git apply --check "$GOLD_PATCH"
  git apply "$GOLD_PATCH"

  for rel in "${runtime_files[@]}"; do
    install -D -m 0644 "$rel" "$SITE_PACKAGES/$rel"
  done
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$PATCH_MARKER"
}

prepare_runtime_environment() {
  if [[ -n "${PYTHONPATH:-}" ]]; then
    echo "Clearing PYTHONPATH to avoid source-checkout shadowing." >&2
    unset PYTHONPATH
  fi
  mkdir -p "$RUNTIME_CWD"
  cd "$RUNTIME_CWD"
  cleanup_patch_workdir
}

COMMON_ARGS=(
  "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --max-model-len "$MAX_MODEL_LEN"
  --kv-cache-dtype fp8
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --enable-prefix-caching
  --language-model-only
  --default-chat-template-kwargs "$CHAT_TEMPLATE_KWARGS"
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

if [[ -n "${CHAT_TEMPLATE:-}" ]]; then
  COMMON_ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi

MODE_ARGS=()
case "$MODE" in
  base)
    if [[ -f "$PATCH_MARKER" && "${ALLOW_PATCHED_BASE:-0}" != "1" ]]; then
      echo "Patch marker exists, so this is not a clean base container: $PATCH_MARKER" >&2
      echo "Use a fresh container for base, or set ALLOW_PATCHED_BASE=1." >&2
      exit 1
    fi
    MODE_ARGS=()
    ;;
  mamba)
    install_gold_patch
    MODE_ARGS=(
      --mamba-cache-mode latest
      --mamba-latest-tail-checkpoints "$TAIL_CHECKPOINTS"
      --mamba-latest-coarse-min-gap "$COARSE_MIN_GAP"
    )
    ;;
  mtp)
    install_gold_patch
    MODE_ARGS=(
      --mamba-cache-mode latest
      --mamba-latest-tail-checkpoints "$TAIL_CHECKPOINTS"
      --mamba-latest-coarse-min-gap "$COARSE_MIN_GAP"
      --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS}}"
    )
    ;;
esac

echo "Serving mode: $MODE"
prepare_runtime_environment
exec vllm serve "${COMMON_ARGS[@]}" "${MODE_ARGS[@]}" "$@"
