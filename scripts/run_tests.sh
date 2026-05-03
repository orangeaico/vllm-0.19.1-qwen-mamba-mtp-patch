#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "RESULT: FAIL" >&2' ERR

BASE_COMMIT="${BASE_COMMIT:-b1388b1fbf5aaef47937fabe98931211684666a6}"
VLLM_REPO_URL="${VLLM_REPO_URL:-https://github.com/vllm-project/vllm.git}"
WORKDIR="${WORKDIR:-/workspace/vllm-patch-validation}"
SITE_PACKAGES="${SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
VENV="${VENV:-/tmp/vllm-test-venv}"
TEST_ROOT="${TEST_ROOT:-/tmp/vllm-tests}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GOLD_PATCH="${GOLD_PATCH:-${REPO_ROOT}/gold.patch}"
TEST_PATCH="${TEST_PATCH:-${REPO_ROOT}/test.patch}"

section() {
  printf '\n==> %s\n' "$*"
}

cleanup_workdir() {
  case "$WORKDIR" in
    ""|"/"|"$REPO_ROOT"|"$SITE_PACKAGES"|"$SITE_PACKAGES"/*|"$TEST_ROOT"|"$VENV")
      echo "Refusing unsafe WORKDIR cleanup: $WORKDIR" >&2
      exit 1
      ;;
  esac
  rm -rf "$WORKDIR"
}

need_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need_file "$GOLD_PATCH"
need_file "$TEST_PATCH"
need_cmd git
need_cmd uv

mapfile -t RUNTIME_FILES < <(
  grep '^diff --git a/vllm/' "$GOLD_PATCH" \
    | sed 's#^diff --git a/\([^ ]*\).*#\1#' \
    | sort -u
)
mapfile -t TEST_FILES < <(
  grep '^diff --git a/tests/' "$TEST_PATCH" \
    | sed 's#^diff --git a/\([^ ]*\).*#\1#' \
    | sort -u
)

if [[ "${#RUNTIME_FILES[@]}" -eq 0 ]]; then
  echo "gold.patch did not contain any vllm/ runtime files." >&2
  exit 1
fi
if [[ "${#TEST_FILES[@]}" -eq 0 ]]; then
  echo "test.patch did not contain any tests/ files." >&2
  exit 1
fi

section "Clone clean vLLM ${BASE_COMMIT}"
rm -rf "$WORKDIR" "$TEST_ROOT"
mkdir -p "$WORKDIR"
git clone --no-checkout "$VLLM_REPO_URL" "$WORKDIR/vllm"
cd "$WORKDIR/vllm"
git checkout "$BASE_COMMIT"

section "Apply gold.patch"
git apply --check "$GOLD_PATCH"
git apply "$GOLD_PATCH"

section "Install patched runtime files into site-packages"
for rel in "${RUNTIME_FILES[@]}"; do
  install -D -m 0644 "$rel" "$SITE_PACKAGES/$rel"
done

section "Apply test.patch"
git apply --check "$TEST_PATCH"
git apply "$TEST_PATCH"
git diff --check

section "Copy patched tests"
mkdir -p "$TEST_ROOT"
cp -a tests "$TEST_ROOT/tests"

section "Remove source checkout before test execution"
cd /tmp
cleanup_workdir
unset PYTHONPATH

section "Create test virtualenv"
rm -rf "$VENV"
uv venv --system-site-packages --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" \
  pytest tblib pytest-forked pytest-asyncio pytest-rerunfailures \
  pytest-shard pytest-timeout pytest-cov ruff

PY_FILES=()
for rel in "${RUNTIME_FILES[@]}"; do
  if [[ "$rel" == *.py ]]; then
    PY_FILES+=("$SITE_PACKAGES/$rel")
  fi
done
for rel in "${TEST_FILES[@]}"; do
  if [[ "$rel" == *.py ]]; then
    PY_FILES+=("$TEST_ROOT/$rel")
  fi
done

section "py_compile patched files"
PYTHONPYCACHEPREFIX=/tmp/vllm-patch-pycache \
  "$VENV/bin/python" -m py_compile "${PY_FILES[@]}"

section "ruff patched files"
"$VENV/bin/python" -m ruff check "${PY_FILES[@]}"

section "pytest targeted latest-Mamba/MTP coverage"
cd "$TEST_ROOT"
"$VENV/bin/python" -m pytest \
  tests/v1/core/test_prefix_caching.py::test_hybrid_latest_mamba_queues_bounded_checkpoints_until_free \
  tests/v1/core/test_prefix_caching.py::test_hybrid_latest_partial_full_attention_hit \
  tests/v1/core/test_prefix_caching.py::test_hybrid_latest_partial_full_attention_only_current_boundary_cached \
  tests/v1/core/test_prefix_caching.py::test_hybrid_latest_partial_full_attention_explicit_prior_boundary_hit \
  tests/v1/core/test_prefix_caching.py::test_hybrid_latest_partial_cache_eviction \
  tests/v1/core/test_scheduler.py::test_mtp_speculative_config_keeps_eagle_cache_behaviors_disabled \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_mtp_does_not_reserve_eagle_lookahead_tokens \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_mtp_keeps_partial_prefix_cache_enabled \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_prefix_cache_hit_after_chunked_prefill \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_tail_checkpoint_policy_controls_cached_prefix \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_tail_checkpoint_stride_split_positions \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_coarse_checkpoint_selection \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_tail_checkpoint_split_policy_async_scheduler \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_caches_completed_decode_tokens_for_next_turn \
  tests/v1/worker/test_mamba_utils.py::test_resumed_req_ids_cleared_from_mamba_state_idx \
  -q

echo "RESULT: PASS"
