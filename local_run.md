# Local Run

## 1. Start container, install git, clone repo

```bash
docker run -it --rm \
  --name vllm-qwen35 \
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
    exec /bin/bash
  '
```

## 2. Serve Qwen3.5-35B-A3B-FP8 (TP=2, EP, mamba mode)

Run from inside the container after step 1 (patches vLLM files then serves):

```bash
MODEL_PATH=/home/shared/megatron_dir/hf_models/Qwen3.5-35B-A3B-FP8 \
MAX_NUM_BATCHED_TOKENS=16384 \
bash scripts/serve.sh mamba --preserve-thinking false \
  -- --tensor-parallel-size 2 --enable-expert-parallel
```
