#!/usr/bin/env bash
set -Eeuo pipefail

docker run -it --rm \
  --name vllm-qwen35-base \
  --entrypoint /bin/bash \
  --gpus all \
  --ipc=host \
  --network host \
  -v /home/shared:/home/shared \
  -v /home/surya/vllm-0.19.1-qwen-mamba-mtp-patch:/workspace/patch \
  -w /workspace/patch \
  vllm/vllm-openai:v0.19.1 \
  -c '
    MODEL_PATH=/home/shared/megatron_dir/hf_models/Qwen3.5-35B-A3B-FP8 \
    MAX_NUM_BATCHED_TOKENS=16384 \
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    PORT=3004 \
      bash scripts/serve.sh base \
        --tensor-parallel-size 2 \
        --enable-expert-parallel
  '
