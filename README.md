# vLLM Qwen3.5 Latest-Mamba Patch Artifacts

Base commit: `b1388b1fbf5aaef47937fabe98931211684666a6` (`v0.19.1`).

Artifacts:

- `gold.patch`: production runtime files under `vllm/` only.
- `test.patch`: test files under `tests/` only.
- `commands.md`: paste-ready reproduction and serve commands.

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
