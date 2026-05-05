# Local Run

## 1. Start container — git, clone, tests, debug logs, serve

Single command: starts the container, installs git, clones the repo, runs tests,
overlays the debug-logged vLLM files, then serves in mamba mode.

```bash
docker run -it --rm \
  --name vllm-qwen35 \
  --entrypoint /bin/bash \
  --gpus all \
  --ipc=host \
  --network host \
  -v /home/shared:/home/shared \
  -v /home/surya/vllm-0.19.1-qwen-mamba-mtp-patch:/host-patch:ro \
  -w /workspace \
  vllm/vllm-openai:v0.19.1 \
  -c '
    apt-get update -qq &&
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git &&
    git clone https://github.com/orangeaico/vllm-0.19.1-qwen-mamba-mtp-patch.git /workspace/patch &&
    cd /workspace/patch &&
    bash scripts/run_tests.sh &&
    cp /host-patch/scripts/debug/scheduler.py \
       /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py &&
    cp /host-patch/scripts/debug/single_type_kv_cache_manager.py \
       /usr/local/lib/python3.12/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py &&
    cp /host-patch/scripts/debug/kv_cache_coordinator.py \
       /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py &&
    cp /host-patch/scripts/debug/block_pool.py \
       /usr/local/lib/python3.12/dist-packages/vllm/v1/core/block_pool.py &&
    cp /host-patch/scripts/debug/gpu_model_runner.py \
       /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py &&
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    vllm serve /home/shared/megatron_dir/hf_models/Qwen3.5-35B-A3B-FP8 \
      --host 0.0.0.0 --port 3003 \
      --served-model-name qwen3 \
      --tensor-parallel-size 2 \
      --enable-expert-parallel \
      --max-model-len 65536 \
      --kv-cache-dtype fp8 \
      --gpu-memory-utilization 0.95 \
      --enable-prefix-caching \
      --language-model-only \
      --default-chat-template-kwargs '"'"'{"enable_thinking": false, "preserve_thinking": false}'"'"' \
      --reasoning-parser qwen3 \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      --trust-remote-code \
      --max-num-batched-tokens 16384 \
      --performance-mode throughput \
      --async-scheduling \
      --mamba-ssm-cache-dtype float16 \
      --no-scheduler-reserve-full-isl \
      --mamba-cache-mode latest \
      --mamba-latest-tail-checkpoints 0 \
      --mamba-latest-coarse-checkpoints 0 \
      --mamba-latest-coarse-min-gap 512
  '
```

> `run_tests.sh` installs gold.patch into site-packages. The `cp` then overlays the
> debug-logged files on top before `vllm serve` starts — so nothing can clobber them.
> To serve without debug logging, remove the five `cp` lines.
