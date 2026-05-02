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
docker exec "$CONTAINER" bash -lc "
rm -rf /workspace/patch &&
git clone '$PATCH_REPO' /workspace/patch &&
bash /workspace/patch/scripts/run_tests.sh
"
```

Serve from a clean container after cloning this artifact repo:

```bash
# base: unpatched v0.19.1 baseline with --mamba-cache-mode align
docker exec "$CONTAINER" bash -lc '
cd /workspace/patch &&
bash scripts/serve.sh base --preserve-thinking true
'

# mamba: patched latest-Mamba, no MTP
docker exec "$CONTAINER" bash -lc '
cd /workspace/patch &&
bash scripts/serve.sh mamba --preserve-thinking true
'

# mtp: patched latest-Mamba plus MTP with 3 draft tokens
docker exec "$CONTAINER" bash -lc '
cd /workspace/patch &&
bash scripts/serve.sh mtp --preserve-thinking true
'
```

Use a fresh container for each serve mode when comparing performance. The
`mamba` and `mtp` modes patch the installed vLLM package in-place. Pass
`--preserve-thinking false` instead when you intentionally want that chat
template behavior; the script refuses to run without an explicit value.

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
  tests/v1/core/test_prefix_caching.py \
  tests/v1/core/test_scheduler.py
/tmp/vllm-test-venv/bin/python -m ruff check \
  /usr/local/lib/python3.12/dist-packages/vllm/config/cache.py \
  /usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py \
  /usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py \
  /usr/local/lib/python3.12/dist-packages/vllm/engine/arg_utils.py \
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
  tests/v1/core/test_prefix_caching.py \
  tests/v1/core/test_scheduler.py
/tmp/vllm-test-venv/bin/python -m pytest \
  tests/v1/core/test_prefix_caching.py::test_hybrid_latest_partial_full_attention_hit \
  tests/v1/core/test_prefix_caching.py::test_hybrid_latest_partial_full_attention_prior_boundary_hit \
  tests/v1/core/test_prefix_caching.py::test_hybrid_latest_partial_cache_eviction \
  tests/v1/core/test_scheduler.py::test_mtp_speculative_config_keeps_eagle_cache_behaviors_disabled \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_mtp_does_not_reserve_eagle_lookahead_tokens \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_mtp_keeps_partial_prefix_cache_enabled \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_prefix_cache_hit_after_chunked_prefill \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_tail_checkpoint_policy_controls_cached_prefix \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_tail_checkpoint_stride_split_positions \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_coarse_checkpoint_selection \
  tests/v1/core/test_scheduler.py::test_hybrid_latest_tail_checkpoint_split_policy_async_scheduler \
  -q
'
```

## 6. Start vLLM Serve With Optimized Latest-Mamba + MTP

```bash
docker exec "$CONTAINER" bash -lc '
cd /tmp
vllm serve /home/shared/megatron_dir/hf_models/Qwen3.5-35B-A3B-FP8/ \
  --served-model-name qwen3 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --max-model-len 65536 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.86 \
  --enable-prefix-caching \
  --language-model-only \
  --mamba-cache-mode latest \
  --mamba-latest-tail-checkpoints 0 \
  --mamba-latest-coarse-min-gap 512 \
  --speculative-config '"'"'{"method":"mtp","num_speculative_tokens":3}'"'"'
'
```
