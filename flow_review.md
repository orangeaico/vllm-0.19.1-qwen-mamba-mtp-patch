# Latest-Mamba + MTP Flow Review

## Intent

This is the review map for the flow exercised by `scripts/serve.sh`. It spells
out what happens in clean base, patched latest-Mamba, and patched latest-Mamba
plus MTP mode, with the code contracts that should be checked when reviewing
`gold.patch`.

Line references point at a vLLM tree with `gold.patch` applied.

## Serve Modes

`scripts/serve.sh` requires both a mode and an explicit
`--preserve-thinking true|false` value (`scripts/serve.sh:6`-`13`,
`scripts/serve.sh:57`-`89`).

- `base`: clean unpatched v0.19.1 container. It refuses to run if the patch
  marker exists unless `ALLOW_PATCHED_BASE=1` is set
  (`scripts/serve.sh:177`-`186`).
- `mamba`: installs `gold.patch`, then serves latest-Mamba
  (`scripts/serve.sh:187`-`193`).
- `mtp`: installs `gold.patch`, then serves latest-Mamba plus
  `{"method":"mtp","num_speculative_tokens":3}` by default
  (`scripts/serve.sh:195`-`202`).

The shared command base is Qwen3.6 on port 3003 with prefix caching enabled,
FP8 KV cache, 65536 max model length, 32768 max batched tokens, throughput
mode, async scheduling, Mamba SSM float16, and
`--no-scheduler-reserve-full-isl` (`scripts/serve.sh:99`-`111`,
`scripts/serve.sh:151`-`171`).

## Mode Contract

| Stage | Base | Patched `mamba` | Patched `mtp` |
| --- | --- | --- | --- |
| Runtime | Clean v0.19.1 container. | `gold.patch` copied into site-packages. | Same patched runtime. |
| Mamba mode | No explicit latest flag. With prefix caching enabled, model config changes `none` to `all` if supported, otherwise `align` (`vllm/model_executor/models/config.py:436`-`440`). | Forces `--mamba-cache-mode latest`, `--mamba-latest-tail-checkpoints 0`, and `--mamba-latest-coarse-min-gap 512` by default (`scripts/serve.sh:189`-`193`). | Same latest-Mamba flags plus MTP speculative config (`scripts/serve.sh:197`-`202`). |
| Spec decode | None by default. | None by default. | MTP uses the Eagle-compatible path, but must not use Eagle-only cache drop or verifier lookahead slots. |

## Config Flow

CLI fields exist for `mamba_cache_mode`, latest tail checkpoints, latest tail
stride, latest coarse checkpoints, and latest coarse min gap
(`vllm/engine/arg_utils.py:603`-`612`,
`vllm/engine/arg_utils.py:1031`-`1047`). The values are copied into
`CacheConfig` (`vllm/engine/arg_utils.py:1608`-`1620`), whose defaults define
`mamba_cache_mode = "none"` and the latest checkpoint knobs
(`vllm/config/cache.py:109`-`133`).

If prefix caching is enabled and the user does not set a Mamba mode,
`MambaModelConfig` chooses `all` when the model supports Mamba prefix caching,
otherwise `align` (`vllm/model_executor/models/config.py:436`-`440`). If `all`
is requested but unsupported, it falls back to `align`
(`vllm/model_executor/models/config.py:447`-`456`).

For `align` or `latest`, chunked prefill is required
(`vllm/model_executor/models/config.py:457`-`461`), and `VllmConfig` checks
that the Mamba alignment block size fits under `max_num_batched_tokens`
(`vllm/config/vllm.py:1736`-`1754`).

In latest mode, engine core changes the prefix-cache hash block size to the
minimum KV-cache group block size, allowing Mamba 16-token boundaries to be
hashed independently of the larger attention block
(`vllm/v1/engine/core.py:137`-`160`).

## MTP Contract

MTP keeps `use_eagle()` true because it shares the Eagle-compatible speculative
decode path (`vllm/config/speculative.py:859`-`860`). The patch separates the
Eagle-only cache behaviors:

- `use_eagle_prefix_cache_drop()` is true only for `eagle` and `eagle3`
  (`vllm/config/speculative.py:862`-`864`).
- `use_kv_cache_lookahead()` is true only for `eagle` and `eagle3`
  (`vllm/config/speculative.py:866`-`868`).

Scheduler initialization records `self.use_mtp`, separates
`use_eagle_prefix_cache_drop`, and only assigns verifier KV lookahead slots
when `use_kv_cache_lookahead()` is true
(`vllm/v1/core/sched/scheduler.py:223`-`239`).

Review contract: MTP may enter Eagle-compatible speculative code, but MTP must
not drop the last prefix block and must not reserve verifier KV lookahead
slots. Tests cover this in `tests/v1/core/test_scheduler.py:1787`-`1820`.

## Scheduler Setup

The scheduler enables latest-Mamba partial full-attention cache only when:

- `mamba_cache_mode == "latest"`,
- the model has Mamba layers,
- hash block size is smaller than attention block size,
- there is no KV connector,
- Eagle prefix-cache drop is disabled,
- DCP and PCP world sizes are 1.

The code is `vllm/v1/core/sched/scheduler.py:241`-`263`. The MTP-sensitive gate
is `and not self.use_eagle_prefix_cache_drop`
(`vllm/v1/core/sched/scheduler.py:242`-`249`). Since MTP no longer enables Eagle
prefix-cache drop, MTP keeps partial full-attention cache enabled.

The scheduler also records whether Mamba block-aligned splitting is needed for
`align` or `latest` (`vllm/v1/core/sched/scheduler.py:277`-`281`) and enables
MTP prefill budget reservation only when MTP, Mamba block-aligned splitting, and
positive speculative tokens are all active
(`vllm/v1/core/sched/scheduler.py:282`-`286`).

## Prefix-Hit Flow

For a new waiting request, the scheduler asks the KV manager for locally cached
tokens before scheduling new work (`vllm/v1/core/sched/scheduler.py:753`-`758`).
The manager skips lookup when prefix caching is disabled or the request skips
prefix-cache reads (`vllm/v1/core/kv_cache_manager.py:192`-`198`).

The lookup caps a full prompt hit at `request.num_tokens - 1`, because the last
token must be recomputed for logits (`vllm/v1/core/kv_cache_manager.py:200`-`207`).
It then calls the coordinator and stores pending partial hits
(`vllm/v1/core/kv_cache_manager.py:207`-`226`).

For hybrid latest mode, the coordinator relaxes hit alignment from the LCM of
attention block sizes to `hash_block_size` when partial full-attention cache is
enabled (`vllm/v1/core/kv_cache_coordinator.py:520`-`530`). It can find a
cached partial full-attention block at a 16-token hash boundary inside a larger
physical attention block (`vllm/v1/core/kv_cache_coordinator.py:532`-`579`).

Full attention and Mamba groups converge on one common hit length
(`vllm/v1/core/kv_cache_coordinator.py:581`-`723`). A partial full-attention hit
is returned only if the final hit lands inside that physical attention block
(`vllm/v1/core/kv_cache_coordinator.py:699`-`719`).

Review contract: latest-Mamba prefix reuse is the minimum prefix accepted by
full attention and Mamba. Missing Mamba checkpoints, missing partial
full-attention copy, or eviction of the partial source block can all reduce
cached tokens. Partial full-attention metadata is only written for the latest
boundary committed by a cache write; earlier boundaries are reusable only when a
previous scheduler split explicitly committed them.

## Mamba Checkpoint Split

`_mamba_block_aligned_split()` runs during prefill and resumed-prefill
scheduling (`vllm/v1/core/sched/scheduler.py:350`-`371`). In latest mode, it
uses `mamba_block_size` for Mamba checkpoint positions and attention block size
for full-attention boundary checkpoints
(`vllm/v1/core/sched/scheduler.py:379`-`390`).

The split policy builds:

- latest Mamba boundary (`vllm/v1/core/sched/scheduler.py:403`-`404`),
- optional coarse full-attention boundary checkpoints
  (`vllm/v1/core/sched/scheduler.py:405`-`420`),
- optional tail checkpoints (`vllm/v1/core/sched/scheduler.py:423`-`447`).

It shortens a chunk to the next checkpoint if the scheduled chunk would cross
one (`vllm/v1/core/sched/scheduler.py:447`-`465`). Coarse checkpoint selection
walks backward from the latest boundary and honors `mamba_latest_coarse_min_gap`
(`vllm/v1/core/sched/scheduler.py:330`-`348`).

Review contract: with the current optimized flags, `tail=0` avoids near-tail
16-token checkpoint churn, while `coarse_min_gap=512` keeps an older
full-attention boundary checkpoint only when it is far enough from the latest
Mamba boundary.

## Token Budget and Allocation

Each scheduling step starts with `token_budget = max_num_scheduled_tokens`
(`vllm/v1/core/sched/scheduler.py:498`-`506`). For MTP with Mamba
block-aligned splitting, a small waiting-prefill reserve is subtracted from the
running request budget (`vllm/v1/core/sched/scheduler.py:468`-`479`,
`vllm/v1/core/sched/scheduler.py:504`-`508`). Non-MTP latest-Mamba does not use
this reserve.

Running requests are scheduled first, apply Mamba block-aligned splitting, and
allocate KV slots (`vllm/v1/core/sched/scheduler.py:520`-`606`). Waiting
requests then use the prefix hit length, apply chunking and Mamba split, and
call `allocate_slots()` (`vllm/v1/core/sched/scheduler.py:753`-`824`,
`vllm/v1/core/sched/scheduler.py:847`-`900`).

For waiting requests, lookahead is zero when `request.num_computed_tokens == 0`
(`vllm/v1/core/sched/scheduler.py:857`-`864`). MTP also has
`num_lookahead_tokens == 0` from scheduler initialization.

`--no-scheduler-reserve-full-isl` disables the full-sequence admission gate. If
enabled, the scheduler checks `can_fit_full_sequence()` before admitting a
waiting request (`vllm/v1/core/sched/scheduler.py:878`-`887`).

After scheduling, `request.num_cached_tokens` is set from the computed prefix
length (`vllm/v1/core/sched/scheduler.py:966`-`977`). This is the scheduler-side
source for cached-token accounting.

## KV Cache Commit

`KVCacheManager.allocate_slots()` computes total computed tokens, main-model
tokens, and slot tokens, including lookahead only when requested
(`vllm/v1/core/kv_cache_manager.py:270`-`379`). Before allocation, it removes
skipped blocks and accounts for pending partial hits
(`vllm/v1/core/kv_cache_manager.py:380`-`407`). If a partial hit is used, the
source block is pinned before the destination slot is allocated
(`vllm/v1/core/kv_cache_manager.py:409`-`437`).

Cache commit is capped to finalized tokens:
`min(total_computed_tokens + num_new_tokens, request.num_tokens)`, so rejected
draft tokens are not committed as prefix cache
(`vllm/v1/core/kv_cache_manager.py:439`-`455`).

Full-attention blocks are cached only when complete
(`vllm/v1/core/single_type_kv_cache_manager.py:260`-`285`). Partial
full-attention cache records the latest committed hash boundary inside the
trailing physical block
(`vllm/v1/core/single_type_kv_cache_manager.py:480`-`492`,
`vllm/v1/core/block_pool.py:355`-`403`). It does not backfill uncommitted prior
boundaries, which avoids full-attention copy work at prefix lengths that Mamba
cannot also hit.

Mamba latest/align allocation ignores verifier lookahead tokens to preserve
alignment and accounts for speculative Mamba state blocks separately
(`vllm/v1/core/single_type_kv_cache_manager.py:893`-`956`,
`vllm/v1/core/single_type_kv_cache_manager.py:958`-`1034`). Those Mamba
speculative blocks come from `speculative_config.num_speculative_tokens`
(`vllm/model_executor/layers/mamba/abstract.py:43`-`58`), and Mamba cache memory
usage includes them (`vllm/v1/kv_cache_interface.py:293`-`300`).

In latest mode, Mamba checkpoints are queued but not inserted into the global
prefix cache while the request is still running. The manager keeps the newest
16-token latest checkpoint and one selected coarse checkpoint, frees replaced
queued states, and hashes those queued blocks only when `KVCacheManager.free()`
finalizes the request. Non-speculative latest mode can also move the same
physical running-state block forward when the previous state is not a queued
checkpoint, avoiding a redundant state copy.

## Mamba State Copy

For `align` and `latest`, worker preprocess copies the previous Mamba state
into the current running-state block before forward
(`vllm/v1/worker/gpu_model_runner.py:3985`-`4002`,
`vllm/v1/worker/mamba_utils.py:147`-`219`). It clears stale Mamba state indices
for finished, preempted, and resumed requests
(`vllm/v1/worker/mamba_utils.py:168`-`177`).

The latest-mode copy optimization requires the worker to track physical block
IDs in addition to the logical state index. If the scheduler reused the same
physical Mamba state block at a later logical index, preprocess observes the
same physical IDs and skips the copy.

Async scheduling adds one more lifetime contract: a source Mamba state selected
for worker copy cannot be freed or relocated before the worker finishes the
scheduled forward. The scheduler places `mamba_source_block_refs` on
`SchedulerOutput`, takes those refs after scheduling, and releases them in
`update_from_output()` after model-runner output is available
(`vllm/v1/core/sched/output.py:237`,
`vllm/v1/core/sched/scheduler.py:1062`-`1083`,
`vllm/v1/core/sched/scheduler.py:1461`-`1462`). The Mamba manager keeps
reference counts for those source blocks so skipped-block cleanup, checkpoint
replacement, and relocation leave them alone until release
(`vllm/v1/core/single_type_kv_cache_manager.py:1375`-`1405`).

For speculative decoding on hybrid models, worker postprocess computes accepted
token counts and invokes Mamba postprocessing in `align/latest`
(`vllm/v1/worker/gpu_model_runner.py:1477`-`1518`). The utility copies the
running state into the aligned accepted-token checkpoint when a partial block
becomes a full checkpoint (`vllm/v1/worker/mamba_utils.py:222`-`273`).

Review contract: with MTP, accepted draft-token Mamba state must move to the
accepted aligned checkpoint, while rejected draft tokens must not become
reusable prefix-cache state.

## Eviction and Removal

There are several separate removal paths:

1. Finished, aborted, or preempted request free:
   `KVCacheManager.free()` delegates to the coordinator
   (`vllm/v1/core/kv_cache_manager.py:457`-`467`). Single-type managers free
   request blocks in reverse order, so tail blocks are placed earlier in the
   eviction queue (`vllm/v1/core/single_type_kv_cache_manager.py:286`-`302`).

2. Eviction order:
   the free queue is LRU-first; ties put the tail of a block chain first
   (`vllm/v1/core/kv_cache_utils.py:177`-`184`). Allocation pops from the front
   (`vllm/v1/core/kv_cache_utils.py:220`-`255`).

3. Allocation-triggered eviction:
   `BlockPool.get_new_blocks()` pops free blocks and calls
   `_maybe_evict_cached_block()` for cached candidates
   (`vllm/v1/core/block_pool.py:418`-`448`). Eviction removes partial-cache
   metadata first and then full block hash metadata
   (`vllm/v1/core/block_pool.py:450`-`497`).

4. Skipped-state removal:
   `remove_skipped_blocks()` frees blocks no longer needed by an attention
   window (`vllm/v1/core/single_type_kv_cache_manager.py:369`-`410`). Mamba
   avoids freeing speculative blocks that may still be needed and, in
   `align/latest`, frees the older running-state block once a newer one exists
   (`vllm/v1/core/single_type_kv_cache_manager.py:857`-`885`).

5. Explicit invalidation:
   KV transfer failures truncate affected requests to the longest valid prefix
   and collect invalid/downstream blocks for eviction
   (`vllm/v1/core/sched/scheduler.py:2290`-`2391`). `evict_blocks()` then
   delegates by block ID (`vllm/v1/core/kv_cache_manager.py:481`-`487`,
   `vllm/v1/core/block_pool.py:532`-`550`).

6. Prefix-cache reset:
   reset succeeds only when all non-null blocks are free, then clears full and
   partial cache maps and resets block hashes
   (`vllm/v1/core/kv_cache_manager.py:489`-`503`,
   `vllm/v1/core/block_pool.py:551`-`586`).

## Test Coverage

- MTP does not enable Eagle prefix-cache drop or KV lookahead:
  `tests/v1/core/test_scheduler.py:1787`-`1820`.
- MTP keeps latest-Mamba partial prefix cache enabled:
  `tests/v1/core/test_scheduler.py:1823`-`1829`.
- Latest-Mamba cache hits after chunked prefill:
  `tests/v1/core/test_scheduler.py:1832`-`1884`.
- Tail checkpoint policy and split positions:
  `tests/v1/core/test_scheduler.py:1886`-`1957`.
- Coarse checkpoint selector:
  `tests/v1/core/test_scheduler.py:1959`-`1988`.
- Completed decode tokens reusable by the next turn:
  `tests/v1/core/test_scheduler.py:2010`-`2101`.
- Partial full-attention hit, current-boundary-only behavior, explicit
  prior-boundary hit, and partial eviction:
  `tests/v1/core/test_prefix_caching.py:1042`-`1195`.
- Mamba source-state lifetime across skipped-block removal, checkpoint
  replacement, repeated references, and relocation:
  `tests/v1/core/test_single_type_kv_cache_manager.py:346`-`472`.
- Worker Mamba copy source lookup uses recorded physical source block IDs:
  `tests/v1/worker/test_mamba_utils.py:72`-`140`.

## Review Hotspots

- `enable_partial_attn_cache` is disabled by connectors, DCP/PCP, and Eagle
  prefix-cache drop (`vllm/v1/core/sched/scheduler.py:241`-`249`).
- Base mode is not latest-Mamba; no explicit Mamba mode means vLLM chooses
  `all` or `align` (`vllm/model_executor/models/config.py:436`-`440`).
- `--no-scheduler-reserve-full-isl` skips the full-sequence fit gate
  (`vllm/v1/core/sched/scheduler.py:878`-`887`).
- MTP should have zero verifier KV lookahead slots
  (`vllm/v1/core/sched/scheduler.py:223`-`239`).
- MTP with 3 draft tokens still has extra Mamba speculative state blocks
  (`vllm/model_executor/layers/mamba/abstract.py:53`-`57`).
