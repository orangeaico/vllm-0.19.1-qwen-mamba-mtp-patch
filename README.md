# vLLM Qwen3.5 Latest-Mamba Patch Artifacts

Base commit: `b1388b1fbf5aaef47937fabe98931211684666a6` (`v0.19.1`).

Artifacts:

- `gold.patch`: production runtime files under `vllm/` only.
- `test.patch`: test files under `tests/` only.
- `commands.md`: paste-ready reproduction and serve commands.
- `scripts/run_tests.sh`: run inside a clean vLLM serve container after
  cloning this artifact repo; applies both patches and validates tests.
- `scripts/serve.sh`: run inside a clean vLLM serve container after cloning
  this artifact repo; serves `base`, patched `mamba`, or patched `mtp`.

Suggested repo layout:

```text
vllm-0.19.1-qwen-mamba-mtp-patch/
├── README.md
├── commands.md
├── gold.patch
├── test.patch
└── scripts/
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
bash /workspace/patch/scripts/serve.sh base --preserve-thinking true
bash /workspace/patch/scripts/serve.sh mamba --preserve-thinking true
bash /workspace/patch/scripts/serve.sh mtp --preserve-thinking true
```

Use a fresh container for `base`; `serve.sh mamba` and `serve.sh mtp` install
`gold.patch` into the container's installed vLLM package. The
`--preserve-thinking true|false` value is required on every serve run so
benchmark comparisons do not accidentally mix chat-template settings.

The production patch includes latest-Mamba prefix-cache support, tail and coarse
checkpoint controls, partial full-attention cache reuse, and MTP compatibility.
MTP still uses the Eagle proposer path where required, but it does not enable
Eagle prefix-cache block dropping and does not reserve verifier KV lookahead
slots. The MTP prefill token reservation is active only for MTP with Mamba
block-aligned splitting.

Local verification run:

- `git diff --check`: passed.
- `PYTHONPYCACHEPREFIX=/tmp/vllm-hybrid-pycache .venv/bin/python -m py_compile ...`: passed for changed production and test Python files.
- `ruff check ...`: passed for changed production and test Python files.
- Clean apply check from `b1388b1fb`: `gold.patch` then `test.patch` applied cleanly and `git diff --check` passed.
- Container targeted tests in `vllm/vllm-openai:v0.19.1`: 16 selected latest-Mamba/MTP tests passed.
- MTP serve smoke: Qwen3.5 35B with `{"method":"mtp","num_speculative_tokens":3}` reached `/v1/models` readiness using the requested serve flags. A first run with `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` failed for insufficient KV cache memory at `--gpu-memory-utilization 0.86`; rerunning without that estimator env succeeded.

GPU/container tests and serving benchmarks should be run with the commands in
`commands.md`.
