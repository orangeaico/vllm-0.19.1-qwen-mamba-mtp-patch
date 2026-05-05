# Reproduction Commands

Set paths:

```bash
REPO=/home/shramana/vllm-hybrid-optim
ARTIFACTS=$REPO/vllm_qwen35_patch_artifacts
BASE=b1388b1fbf5aaef47937fabe98931211684666a6
CONTAINER=vllm-qwen35-patch-test
PATCH_REPO=https://github.com/orangeaico/vllm-0.19.1-qwen-mamba-mtp-patch.git
```

## Simple GitHub-Based Workflow

Start a clean test container:

```bash
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d \
  --name "$CONTAINER" \
  --entrypoint /bin/bash \
  --gpus all \
  --ipc=host \
  --network host \
  -v /home/shared:/home/shared \
  -w /workspace \
  vllm/vllm-openai:v0.19.1 \
  -lc 'sleep infinity'
```

Clone this artifact repo inside the container and run all patch tests:

```bash
docker exec "$CONTAINER" bash -lc '
command -v git >/dev/null 2>&1 || {
  apt-get update &&
  DEBIAN_FRONTEND=noninteractive apt-get install -y git
}
'

docker exec "$CONTAINER" bash -lc "
rm -rf /workspace/patch &&
git clone '$PATCH_REPO' /workspace/patch &&
bash /workspace/patch/scripts/run_tests.sh
"
```

Serve from a clean container after cloning this artifact repo:

```bash
# base: unpatched v0.19.1 baseline
docker exec "$CONTAINER" bash -lc '
cd /workspace/patch &&
bash scripts/serve.sh base
'

# mamba: patched latest-Mamba, no MTP
docker exec "$CONTAINER" bash -lc '
cd /workspace/patch &&
bash scripts/serve.sh mamba
'

# mtp: patched latest-Mamba plus MTP with 3 draft tokens
docker exec "$CONTAINER" bash -lc '
cd /workspace/patch &&
bash scripts/serve.sh mtp
'
```

Use a fresh container for each serve mode when comparing performance. The
`mamba` and `mtp` modes patch the installed vLLM package in-place. The serve
scripts do not set `preserve_thinking`; vLLM uses the chat-template default for
that field.

Serve with debug logs for latest-Mamba cache flow:

```bash
docker exec "$CONTAINER" bash -lc '
cd /workspace/patch &&
bash scripts/debug.sh mamba
'
```

Run the host-side 10-turn prefill/cache probe against a server on port 3003:

```bash
cd /path/to/vllm-0.19.1-qwen-mamba-mtp-patch
python3 scripts/multiturn_vllm_metrics.py \
  --base-url http://127.0.0.1:3003 \
  --model qwen3 \
  --turns 10 \
  --input-tokens 400 \
  --output-tokens 200 \
  --min-output-tokens 200 \
  --temperature 0 \
  --trace-json
```

## 1. Run Container

```bash
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d \
  --name "$CONTAINER" \
  --entrypoint /bin/bash \
  --gpus all \
  --ipc=host \
  --network host \
  -v /home/shared:/home/shared \
  -v "$ARTIFACTS:/artifacts:ro" \
  -w /workspace \
  vllm/vllm-openai:v0.19.1 \
  -lc 'sleep infinity'
```

## 2. Copy Artifacts Into Container

```bash
docker exec "$CONTAINER" bash -lc '
rm -rf /workspace/vllm-patch-artifacts /workspace/vllm
mkdir -p /workspace/vllm-patch-artifacts
cp /artifacts/gold.patch /artifacts/test.patch /workspace/vllm-patch-artifacts/
git clone --no-checkout https://github.com/vllm-project/vllm.git /workspace/vllm
cd /workspace/vllm
git checkout '"$BASE"'
'
```

## 3. Apply `gold.patch`

```bash
docker exec "$CONTAINER" bash -lc '
cd /workspace/vllm
git apply /workspace/vllm-patch-artifacts/gold.patch
for rel in $(grep "^diff --git a/vllm/" /workspace/vllm-patch-artifacts/gold.patch | sed "s#^diff --git a/\\([^ ]*\\).*#\\1#"); do
  cp "$rel" "/usr/local/lib/python3.12/dist-packages/$rel"
done
'
```

## 4. Apply `test.patch`

```bash
docker exec "$CONTAINER" bash -lc '
cd /workspace/vllm
git apply /workspace/vllm-patch-artifacts/test.patch
git diff --check
rm -rf /tmp/vllm-tests
mkdir -p /tmp/vllm-tests
cp -a tests /tmp/vllm-tests/tests
cp pyproject.toml /tmp/vllm-tests/pyproject.toml
'
```

## 5. Run Tests

```bash
docker exec "$CONTAINER" bash -lc '
uv venv --system-site-packages --python 3.12 /tmp/vllm-test-venv
uv pip install --python /tmp/vllm-test-venv/bin/python \
  pytest tblib pytest-forked pytest-asyncio pytest-rerunfailures \
  pytest-shard pytest-timeout pytest-cov ruff
'

docker exec "$CONTAINER" bash -lc '
cd /tmp/vllm-tests
PYTHONPYCACHEPREFIX=/tmp/vllm-pycache /tmp/vllm-test-venv/bin/python -m py_compile \
  /usr/local/lib/python3.12/dist-packages/vllm/config/cache.py \
  /usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py \
  /usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py \
  /usr/local/lib/python3.12/dist-packages/vllm/engine/arg_utils.py \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/abstract.py \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/config.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/utils.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/block_pool.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_manager.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/output.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_cache_interface.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/mamba_utils.py \
  tests/v1/core/test_prefix_caching.py \
  tests/v1/core/test_scheduler.py \
  tests/v1/core/test_single_type_kv_cache_manager.py \
  tests/v1/e2e/general/test_mamba_prefix_cache.py \
  tests/v1/worker/test_mamba_utils.py
/tmp/vllm-test-venv/bin/python -m ruff check \
  --config /tmp/vllm-tests/pyproject.toml \
  --ignore I001 \
  /usr/local/lib/python3.12/dist-packages/vllm/config/cache.py \
  /usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py \
  /usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py \
  /usr/local/lib/python3.12/dist-packages/vllm/engine/arg_utils.py \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/abstract.py \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/config.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/utils.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/block_pool.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_manager.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/output.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_cache_interface.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/mamba_utils.py \
  tests/v1/core/test_prefix_caching.py \
  tests/v1/core/test_scheduler.py \
  tests/v1/core/test_single_type_kv_cache_manager.py \
  tests/v1/e2e/general/test_mamba_prefix_cache.py \
  tests/v1/worker/test_mamba_utils.py
/tmp/vllm-test-venv/bin/python -m pytest \
  tests/v1/core/test_prefix_caching.py::test_hybrid_latest_mamba_queues_bounded_checkpoints_until_free \
  tests/v1/core/test_prefix_caching.py::test_hybrid_latest_partial_full_attention_hit \
  tests/v1/core/test_prefix_caching.py::test_hybrid_latest_partial_full_attention_only_current_boundary_cached \
  tests/v1/core/test_prefix_caching.py::test_hybrid_latest_publishes_completed_mamba_boundary_before_advance \
  tests/v1/core/test_prefix_caching.py::test_hybrid_latest_partial_full_attention_only_mirrors_latest_boundary \
  tests/v1/core/test_prefix_caching.py::test_hybrid_latest_partial_cache_eviction \
  tests/v1/core/test_scheduler.py::test_mtp_speculative_config_keeps_eagle_cache_behaviors_disabled \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_mtp_does_not_reserve_eagle_lookahead_tokens \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_mtp_keeps_partial_prefix_cache_enabled \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_prefix_cache_hit_after_chunked_prefill \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_tail_checkpoint_policy_does_not_cache_extra_boundaries \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_split_positions_target_only_latest_boundary \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_coarse_checkpoint_selection \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_split_policy_async_scheduler_targets_latest_boundary \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_caches_completed_prompt_boundary_for_next_turn \
  tests/v1/core/test_single_type_kv_cache_manager.py::test_latest_mamba_remove_skipped_blocks_keeps_source_state \
  tests/v1/core/test_single_type_kv_cache_manager.py::test_latest_mamba_checkpoint_replacement_keeps_source_state \
  tests/v1/core/test_single_type_kv_cache_manager.py::test_latest_mamba_inflight_source_state_released_on_completion \
  tests/v1/core/test_single_type_kv_cache_manager.py::test_latest_mamba_repeated_source_refs_require_matching_releases \
  tests/v1/core/test_single_type_kv_cache_manager.py::test_latest_mamba_inflight_source_state_not_relocated \
  tests/v1/worker/test_mamba_utils.py::test_resumed_req_ids_cleared_from_mamba_state_idx \
  tests/v1/worker/test_mamba_utils.py::test_preprocess_mamba_uses_recorded_source_block_ids \
  tests/v1/worker/test_mamba_utils.py::test_collect_mamba_copy_meta_uses_recorded_spec_source_offset \
  -q
'
```

## 6. Start vLLM Serve With Optimized Latest-Mamba + MTP

```bash
docker exec "$CONTAINER" bash -lc '
cd /tmp
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --host 0.0.0.0 \
  --port 3003 \
  --served-model-name qwen3 \
  --max-model-len 65536 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --language-model-only \
  --default-chat-template-kwargs '"'"'{"enable_thinking": false}'"'"' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --max-num-batched-tokens 32768 \
  --performance-mode throughput \
  --async-scheduling \
  --mamba-ssm-cache-dtype float16 \
  --no-scheduler-reserve-full-isl \
  --mamba-cache-mode latest \
  --mamba-latest-tail-checkpoints 0 \
  --mamba-latest-coarse-checkpoints 0 \
  --mamba-latest-coarse-min-gap 512 \
  --speculative-config '"'"'{"method":"mtp","num_speculative_tokens":3}'"'"'
'
```
