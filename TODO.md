# TODO: Sparse Decode Checkpointing

## Goal

Improve multi-turn hybrid/MTP performance when the previous assistant output is
large. The current latest-Mamba path primarily publishes a stable prefill
checkpoint. In agent workflows, the next turn includes prior generated tokens,
so stopping at the prefill boundary can cap hybrid prefix reuse below what
full-attention KV caching can reuse.

## Proposed Work

- Publish one committed decode-side Mamba checkpoint when a request finishes.
- Choose a boundary near the final stable token position, aligned to the Mamba
  and prefix-hash block size.
- Mirror full-attention partial cache at exactly the same boundary.
- Start with one final decode checkpoint only; do not add periodic decode
  checkpointing until the single-checkpoint path is measured.

## Correctness Contract

For every published decode checkpoint:

```text
Mamba checkpoint boundary == full-attention partial boundary == returned cache hit length
```

Additional guards:

- Use committed tokens only: `request.num_tokens`.
- Do not use speculative `num_tokens_with_spec` or unverified draft tokens.
- Publish only after MTP accept/reject rollback is reflected in request state.
- Require the boundary to be aligned to `mamba_block_size` and `hash_block_size`.
- Publish full-attention partial cache only if the Mamba checkpoint publication
  succeeds at the same boundary.

## Validation

- Add focused debug logging for request id, committed tokens, chosen boundary,
  Mamba block id/hash state, full-attention partial state, and returned hit
  length.
- Run serial and concurrent multi-turn probes with MTP enabled.
- Run SWEAgent concurrency probes and compare prefill/request, prefix hit rate,
  generation TPS, and resolved count against the current latest-Mamba baseline.
