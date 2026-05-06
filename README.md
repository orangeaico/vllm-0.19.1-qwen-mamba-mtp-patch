# vLLM Qwen3.5 Latest-Mamba Patch Artifacts

Base commit: `b1388b1fbf5aaef47937fabe98931211684666a6` (`v0.19.1`).

Artifacts:

- `gold.patch`: production runtime files under `vllm/` only.
- `test.patch`: test files under `tests/` only.
- `commands.md`: paste-ready reproduction and serve commands.
- `base.flow`: clean v0.19.1 baseline serving and cache flow.
- `optim.flow`: patched latest-Mamba and MTP serving, cache, and review flow.
- `scripts/run_tests.sh`: run inside a clean vLLM serve container after
  cloning this artifact repo; applies both patches and validates tests.
- `scripts/serve.sh`: run inside a clean vLLM serve container after cloning
  this artifact repo; serves `base`, patched `mamba`, or patched `mtp`.
- `scripts/debug.sh`: same serving flow as `serve.sh`, but overlays
  debug-logged runtime files from `scripts/debug/`.
- `scripts/multiturn_vllm_metrics.py`: host-side multi-turn metrics probe.

Suggested repo layout:

```text
vllm-0.19.1-qwen-mamba-mtp-patch/
├── README.md
├── commands.md
├── base.flow
├── optim.flow
├── gold.patch
├── test.patch
└── scripts/
    ├── debug/
    ├── debug.sh
    ├── multiturn_vllm_metrics.py
    ├── run_tests.sh
    └── serve.sh
```

Inside a clean `vllm/vllm-openai:v0.19.1` container, clone and run tests:

```bash
git clone https://github.com/orangeaico/vllm-0.19.1-qwen-mamba-mtp-patch.git /workspace/patch
bash /workspace/patch/scripts/run_tests.sh
```

Inside a clean `vllm/vllm-openai:v0.19.1` container, serve one mode:

```bash
git clone https://github.com/orangeaico/vllm-0.19.1-qwen-mamba-mtp-patch.git /workspace/patch
bash /workspace/patch/scripts/serve.sh base
bash /workspace/patch/scripts/serve.sh mamba
bash /workspace/patch/scripts/serve.sh mtp
```

Use a fresh container for `base`; `serve.sh mamba` and `serve.sh mtp` install
`gold.patch` into the container's installed vLLM package. The serve scripts do
not set `preserve_thinking`; vLLM uses the chat-template default for that field.

## Mode Overview

`base` is the clean v0.19.1 runtime. It is the correctness and performance
reference. It uses upstream hybrid prefix-cache behavior and does not install
any patched files.

`mamba` installs `gold.patch` and serves with latest-Mamba enabled:

```bash
--mamba-cache-mode latest
--mamba-latest-tail-checkpoints 0
--mamba-latest-coarse-checkpoints 0
--mamba-latest-coarse-min-gap 512
```

With these defaults, the optimized path is latest-only: no tail checkpoints and
no coarse checkpoints are intentionally retained. The patch adds smaller
latest-Mamba prefix hashing, Mamba-anchored partial full-attention cache reuse,
and stable latest-boundary Mamba checkpoint publication for multi-turn prefill
reuse.

`mtp` installs the same patched runtime and adds MTP speculative decoding:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

MTP shares the Eagle-compatible speculative path where vLLM requires it, but it
does not enable Eagle prefix-cache block dropping and does not reserve Eagle
verifier KV lookahead slots. MTP uses the latest-Mamba cache flow plus MTP's
draft/acceptance path, so it must be benchmarked separately from non-MTP
`mamba`.

See `base.flow` and `optim.flow` for the current technical flow. Those docs are
the source of truth for the base versus patched cache contracts.

Local verification run:

- `git diff --check`: passed.
- `PYTHONPYCACHEPREFIX=/tmp/vllm-hybrid-pycache .venv/bin/python -m py_compile ...`: passed for changed production and test Python files.
- `ruff check ...`: passed for changed production and test Python files.
- Clean apply check from `b1388b1fb`: `gold.patch` then `test.patch` applied cleanly and `git diff --check` passed.
- Container targeted tests in `vllm/vllm-openai:v0.19.1`: 22 selected latest-Mamba/MTP tests passed.
- Artifact self-test in a fresh `vllm/vllm-openai:v0.19.1` container:
  `scripts/run_tests.sh` applied both patches from clean `b1388b1fb` and
  passed py_compile, ruff, and 22 targeted latest-Mamba/MTP tests.
- MTP serve smoke: Qwen3.5 35B with `{"method":"mtp","num_speculative_tokens":3}` reached `/v1/models` readiness using the requested serve flags. A first run with `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` failed for insufficient KV cache memory at `--gpu-memory-utilization 0.86`; rerunning without that estimator env succeeded.
- Latest smart-copy smoke benchmark: Qwen3.5 35B, TP=2, expert parallel,
  `max_num_batched_tokens=16384`, 2 trajectories, 180 requests completed in
  167 seconds. Request-level local prefill totaled 231,138 tokens
  (1,284/request), sampled generation throughput averaged 206.91 tokens/s,
  and current serve reported maximum 65,536-token concurrency of 9.00x.
- High-concurrency comparison, 8 trajectories, variable 65k profile:
  fixed smart-copy completed 720 requests in 263 seconds versus base in
  307 seconds. Request-level local prefill dropped from 9,230.89/request to
  5,154.53/request, and sampled generation throughput increased from
  430.92 tokens/s to 506.10 tokens/s.
- Latest-minus-one 10-turn probe, Qwen3.5 35B, TP=2, expert parallel,
  `max_num_batched_tokens=16384`, preserve-thinking true:
  turns 2-10 averaged 632.67 local prefill tokens and 148.83 generation TPS;
  turns 6-10 averaged 630.20 local prefill tokens and 148.70 generation TPS.
  The previously bad turn-6/7 fallback to `cached=2224` was fixed; the clean
  probe reported turn-6 `cached=2832 prefill=626` and turn-7
  `cached=3440 prefill=629`.
- SWEAgent-style 4-way comparison, Qwen3.5 35B, TP=2, expert parallel,
  `max_num_batched_tokens=16384`: base resolved 3/4 with 838.0 prefill/request
  and 183.1 generated/request; mamba resolved 2/4 with 393.6 prefill/request
  and 126.6 generated/request; mtp resolved 2/4 with 1204.3 prefill/request
  and 161.5 generated/request. This run shows why generated tokens/request,
  resolved count, and trajectory inspection are required alongside prefill and
  TPS metrics.

GPU/container tests and serving benchmarks should be run with the commands in
`commands.md`.
