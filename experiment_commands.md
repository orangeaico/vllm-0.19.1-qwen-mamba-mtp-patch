# Experiment Commands

Three single-shot docker commands. Each starts a fresh container, installs git,
clones this artifact repo from GitHub, and serves one mode. Use a fresh
container per mode (the mamba and mtp modes patch the installed vLLM package
in-place).

## 1. Base — unpatched v0.19.1 baseline

```bash
docker run -it --rm \
  --name vllm-qwen35-base \
  --entrypoint /bin/bash \
  --gpus all \
  --ipc=host \
  --network host \
  -v /home/shared:/home/shared \
  -w /workspace \
  vllm/vllm-openai:v0.19.1 \
  -c '
    apt-get update -qq &&
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git &&
    git clone https://github.com/orangeaico/vllm-0.19.1-qwen-mamba-mtp-patch.git /workspace/patch &&
    cd /workspace/patch &&
    MODEL_PATH=/home/shared/megatron_dir/hf_models/Qwen3.5-35B-A3B-FP8 \
    MAX_NUM_BATCHED_TOKENS=16384 \
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
      bash scripts/serve.sh base \
        --tensor-parallel-size 2 \
        --enable-expert-parallel
  '
```

## 2. Mamba — patched latest-Mamba, no MTP

```bash
docker run -it --rm \
  --name vllm-qwen35-mamba \
  --entrypoint /bin/bash \
  --gpus all \
  --ipc=host \
  --network host \
  -v /home/shared:/home/shared \
  -w /workspace \
  vllm/vllm-openai:v0.19.1 \
  -c '
    apt-get update -qq &&
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git &&
    git clone https://github.com/orangeaico/vllm-0.19.1-qwen-mamba-mtp-patch.git /workspace/patch &&
    cd /workspace/patch &&
    MODEL_PATH=/home/shared/megatron_dir/hf_models/Qwen3.5-35B-A3B-FP8 \
    MAX_NUM_BATCHED_TOKENS=16384 \
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
      bash scripts/serve.sh mamba \
        --tensor-parallel-size 2 \
        --enable-expert-parallel
  '
```

## 3. MTP — patched latest-Mamba + MTP with 3 draft tokens

```bash
docker run -it --rm \
  --name vllm-qwen35-mtp \
  --entrypoint /bin/bash \
  --gpus all \
  --ipc=host \
  --network host \
  -v /home/shared:/home/shared \
  -w /workspace \
  vllm/vllm-openai:v0.19.1 \
  -c '
    apt-get update -qq &&
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git &&
    git clone https://github.com/orangeaico/vllm-0.19.1-qwen-mamba-mtp-patch.git /workspace/patch &&
    cd /workspace/patch &&
    MODEL_PATH=/home/shared/megatron_dir/hf_models/Qwen3.5-35B-A3B-FP8 \
    MAX_NUM_BATCHED_TOKENS=16384 \
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
      bash scripts/serve.sh mtp \
        --tensor-parallel-size 2 \
        --enable-expert-parallel
  '
```
