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
  MAX_MODEL_LEN, MAX_NUM_BATCHED_TOKENS, TAIL_CHECKPOINTS
  COARSE_CHECKPOINTS, COARSE_MIN_GAP
  NUM_SPECULATIVE_TOKENS, CHAT_TEMPLATE
  PATCH_WORKDIR, SITE_PACKAGES, PYTHON_BIN, RUNTIME_CWD

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
PATCH_MANIFEST="${PATCH_MANIFEST:-${PATCH_MARKER}.sha256}"
PATCH_METADATA="${PATCH_METADATA:-${PATCH_MARKER}.metadata}"
RUNTIME_CWD="${RUNTIME_CWD:-/tmp}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "Missing required command: python3 or python" >&2
    exit 1
  fi
fi

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.6-35B-A3B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-3003}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
TAIL_CHECKPOINTS="${TAIL_CHECKPOINTS:-0}"
COARSE_CHECKPOINTS="${COARSE_CHECKPOINTS:-0}"
COARSE_MIN_GAP="${COARSE_MIN_GAP:-512}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
CHAT_TEMPLATE_KWARGS="{\"enable_thinking\": false, \"preserve_thinking\": ${PRESERVE_THINKING}}"

export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}"

runtime_files=()

load_runtime_files() {
  if [[ ! -f "$GOLD_PATCH" ]]; then
    echo "Missing gold.patch: $GOLD_PATCH" >&2
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
}

write_patch_metadata() {
  {
    printf 'base_commit=%s\n' "$BASE_COMMIT"
    printf 'gold_patch_sha256=%s\n' "$(sha256sum "$GOLD_PATCH" | awk '{print $1}')"
    printf 'runtime_files_sha256=%s\n' "$(
      printf '%s\n' "${runtime_files[@]}" | sha256sum | awk '{print $1}'
    )"
  } > "$PATCH_METADATA"
}

verify_patch_metadata() {
  [[ -f "$PATCH_METADATA" ]] || {
    echo "Patch marker exists but metadata is missing: $PATCH_METADATA" >&2
    return 1
  }

  local expected_patch_sha expected_files_sha
  expected_patch_sha="$(sha256sum "$GOLD_PATCH" | awk '{print $1}')"
  expected_files_sha="$(printf '%s\n' "${runtime_files[@]}" | sha256sum | awk '{print $1}')"

  grep -qx "base_commit=${BASE_COMMIT}" "$PATCH_METADATA" || {
    echo "Installed patch base commit does not match $BASE_COMMIT" >&2
    return 1
  }
  grep -qx "gold_patch_sha256=${expected_patch_sha}" "$PATCH_METADATA" || {
    echo "Installed patch metadata does not match current gold.patch" >&2
    return 1
  }
  grep -qx "runtime_files_sha256=${expected_files_sha}" "$PATCH_METADATA" || {
    echo "Installed patch runtime file list does not match current gold.patch" >&2
    return 1
  }
}

cleanup_patch_workdir() {
  case "$PATCH_WORKDIR" in
    ""|"/"|"$REPO_ROOT"|"$SITE_PACKAGES"|"$SITE_PACKAGES"/*)
      echo "Refusing unsafe PATCH_WORKDIR cleanup: $PATCH_WORKDIR" >&2
      exit 1
      ;;
  esac
  rm -rf "$PATCH_WORKDIR"
}

assert_installed_vllm_import() {
  local output
  if ! output="$(
    PYTHONPATH= EXPECTED_SITE_PACKAGES="$SITE_PACKAGES" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

expected_root = Path(os.environ["EXPECTED_SITE_PACKAGES"]).resolve() / "vllm"

import vllm

actual = Path(vllm.__file__).resolve()
if expected_root != actual.parent:
    raise SystemExit(
        "vllm import resolved to "
        f"{actual}, expected installed runtime under {expected_root}. "
        "Refusing to serve from a source checkout or PYTHONPATH shadow."
    )

try:
    import vllm._C  # noqa: F401
except Exception as exc:
    raise SystemExit(
        f"installed vllm extension import failed from {actual}: {exc!r}"
    )

print(f"Verified installed vLLM import: {actual}")
PY
  )"; then
    echo "$output" >&2
    exit 1
  fi
  echo "$output"
}

verify_installed_patch() {
  load_runtime_files
  verify_patch_metadata || return 1

  if [[ ! -f "$PATCH_MANIFEST" ]]; then
    echo "Patch marker exists but manifest is missing: $PATCH_MANIFEST" >&2
    return 1
  fi

  for rel in "${runtime_files[@]}"; do
    if [[ ! -f "$SITE_PACKAGES/$rel" ]]; then
      echo "Installed patched file is missing: $SITE_PACKAGES/$rel" >&2
      return 1
    fi
  done

  if ! sha256sum --check --quiet "$PATCH_MANIFEST"; then
    echo "Installed patched files do not match manifest: $PATCH_MANIFEST" >&2
    return 1
  fi
}

install_gold_patch() {
  load_runtime_files

  if [[ -f "$PATCH_MARKER" ]]; then
    if verify_installed_patch; then
      echo "gold.patch already installed and verified: $PATCH_MARKER"
      cleanup_patch_workdir
      return
    fi
    echo "Existing patch marker failed verification; reinstalling gold.patch." >&2
    rm -f "$PATCH_MARKER" "$PATCH_MANIFEST" "$PATCH_METADATA"
  fi

  if ! command -v git >/dev/null 2>&1; then
    echo "Missing required command: git" >&2
    exit 1
  fi

  cleanup_patch_workdir
  mkdir -p "$PATCH_WORKDIR"
  git clone --no-checkout "$VLLM_REPO_URL" "$PATCH_WORKDIR/vllm"

  (
    cd "$PATCH_WORKDIR/vllm"
    git checkout "$BASE_COMMIT"
    git apply --check "$GOLD_PATCH"
    git apply "$GOLD_PATCH"

    : > "${PATCH_MANIFEST}.tmp"
    for rel in "${runtime_files[@]}"; do
      if [[ ! -f "$rel" ]]; then
        echo "Patched runtime file was not produced: $rel" >&2
        exit 1
      fi
      install -D -m 0644 "$rel" "$SITE_PACKAGES/$rel"
      if ! cmp -s "$rel" "$SITE_PACKAGES/$rel"; then
        echo "Installed runtime file does not match patched source: $rel" >&2
        exit 1
      fi
      sha256sum "$SITE_PACKAGES/$rel" >> "${PATCH_MANIFEST}.tmp"
    done
  )

  mv "${PATCH_MANIFEST}.tmp" "$PATCH_MANIFEST"
  write_patch_metadata
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$PATCH_MARKER"
  cleanup_patch_workdir
  verify_installed_patch
}

prepare_runtime_environment() {
  if [[ -n "${PYTHONPATH:-}" ]]; then
    echo "Clearing PYTHONPATH to avoid source-checkout shadowing." >&2
    unset PYTHONPATH
  fi
  mkdir -p "$RUNTIME_CWD"
  cd "$RUNTIME_CWD"
  assert_installed_vllm_import
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
      --mamba-latest-coarse-checkpoints "$COARSE_CHECKPOINTS"
      --mamba-latest-coarse-min-gap "$COARSE_MIN_GAP"
    )
    ;;
  mtp)
    install_gold_patch
    MODE_ARGS=(
      --mamba-cache-mode latest
      --mamba-latest-tail-checkpoints "$TAIL_CHECKPOINTS"
      --mamba-latest-coarse-checkpoints "$COARSE_CHECKPOINTS"
      --mamba-latest-coarse-min-gap "$COARSE_MIN_GAP"
      --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS}}"
    )
    ;;
esac

prepare_runtime_environment
echo "Serving mode: $MODE"
exec vllm serve "${COMMON_ARGS[@]}" "${MODE_ARGS[@]}" "$@"
