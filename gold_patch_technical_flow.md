# Gold Patch Technical Flow

## Purpose

This document analyzes `gold.patch` as a production runtime patch. It explains
what each changed area does, how the patched runtime flows from `vllm serve` to
request scheduling, and what contracts must hold for latest-Mamba prefix-cache
reuse and MTP speculative decoding.

Patch summary from `git apply --stat gold.patch`:

- 18 production files changed.
- 1057 insertions, 58 deletions.
- No tests, docs, benchmarks, scripts, or debug logging are in `gold.patch`.

Changed production files:

```text
vllm/config/cache.py
vllm/config/speculative.py
vllm/config/vllm.py
vllm/engine/arg_utils.py
vllm/model_executor/layers/mamba/abstract.py
vllm/model_executor/models/config.py
vllm/v1/attention/backends/utils.py
vllm/v1/core/block_pool.py
vllm/v1/core/kv_cache_coordinator.py
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/kv_cache_utils.py
vllm/v1/core/sched/output.py
vllm/v1/core/sched/scheduler.py
vllm/v1/core/single_type_kv_cache_manager.py
vllm/v1/engine/core.py
vllm/v1/kv_cache_interface.py
vllm/v1/worker/gpu_model_runner.py
vllm/v1/worker/mamba_utils.py
```

Line references below point at the patched source tree after applying
`gold.patch`.

## Design Problem

Qwen3.5/Qwen3.6 hybrid attention+Mamba models need reusable prefix cache across
multi-turn prompts. The default Mamba prefix-cache modes have a mismatch:

- Full attention commonly has a larger physical KV block size.
- Mamba state can be useful at a smaller checkpoint boundary.
- Hybrid prefix matching requires all KV-cache groups to agree on one reusable
  prefix length.

If Mamba only saves state at a coarse full-attention boundary, the reusable
prefix can fall back substantially. If full attention only supports whole
physical block hits, it can also block reuse at a smaller Mamba boundary.

The patch adds `mamba_cache_mode="latest"` to keep only the latest Mamba state
like `align`, but with a smaller Mamba checkpoint granularity. It then adds
partial full-attention KV copy support so full attention can participate in a
hybrid prefix hit that ends inside a larger physical full-attention block.

MTP adds another constraint. In vLLM, MTP uses the Eagle-compatible speculative
path, but MTP must not behave like Eagle for prefix-cache block dropping or
verifier KV lookahead slots. Otherwise MTP can reduce cache hits and inflate KV
pressure.

## Changed Areas

### 1. Public Config Surface

`vllm/config/cache.py` adds `latest` to `MambaCacheMode` and exposes four
latest-mode checkpoint controls:

- `mamba_latest_tail_checkpoints`
- `mamba_latest_tail_checkpoint_stride`
- `mamba_latest_coarse_checkpoints`
- `mamba_latest_coarse_min_gap`

See `vllm/config/cache.py:109`-`133`.

`vllm/engine/arg_utils.py` wires those fields through `EngineArgs`, CLI parser
flags, and `CacheConfig` construction. See:

- fields: `vllm/engine/arg_utils.py:603`-`612`
- CLI args: `vllm/engine/arg_utils.py:1031`-`1047`
- `CacheConfig` construction: `vllm/engine/arg_utils.py:1608`-`1620`

`vllm/config/vllm.py` extends align-mode validation to `("align", "latest")`.
For latest mode, it validates with `mamba_block_size` when present instead of
blindly using the full scheduler block size. See
`vllm/config/vllm.py:1736`-`1754`.

`vllm/model_executor/models/config.py` requires chunked prefill for both
`align` and `latest`. See `vllm/model_executor/models/config.py:457`-`461`.

Contract:

- Latest mode is a user-visible Mamba cache mode.
- Latest mode needs chunked prefill because the scheduler must be allowed to end
  a prefill chunk exactly at a cacheable checkpoint.
- `max_num_batched_tokens` and `long_prefill_token_threshold` must be large
  enough for the Mamba alignment block size.

### 2. MTP Versus Eagle Semantics

`vllm/config/speculative.py` leaves `use_eagle()` true for MTP, but adds two
separate predicates:

- `use_eagle_prefix_cache_drop()` returns true only for `eagle` and `eagle3`.
- `use_kv_cache_lookahead()` returns true only for `eagle` and `eagle3`.

See `vllm/config/speculative.py:859`-`868`.

`vllm/v1/core/sched/scheduler.py` consumes these predicates:

- records `self.use_mtp`,
- records `self.use_eagle_prefix_cache_drop`,
- sets `self.num_lookahead_tokens` only when verifier KV lookahead is needed.

See `vllm/v1/core/sched/scheduler.py:223`-`239`.

Contract:

- MTP can share Eagle-compatible speculative execution.
- MTP must not drop the last matched prefix-cache block.
- MTP must not reserve verifier KV lookahead slots.
- Eagle and Eagle3 still keep the old drop/lookahead behavior.

### 3. Latest Hash Granularity

`vllm/v1/engine/core.py` adds `cache_hash_block_size`. Normal mode uses the
scheduler block size. Latest mode with Mamba layers uses the minimum KV-cache
group block size. See `vllm/v1/engine/core.py:137`-`160`.

The same hash block size is passed to the request block hasher, so request
hashes are built at the smaller latest-Mamba granularity. See
`vllm/v1/engine/core.py:204`-`214`.

The scheduler constructor now accepts both:

- `block_size`: scheduling block size,
- `hash_block_size`: prefix-cache hash granularity.

See `vllm/v1/core/sched/scheduler.py:71`-`80` and
`vllm/v1/core/sched/scheduler.py:150`-`159`.

Contract:

- Full attention can still allocate large physical blocks.
- Prefix-cache hashes can still exist every smaller Mamba block.
- Latest mode depends on this split; if `hash_block_size == attention block
  size`, partial full-attention cache is not useful.

### 4. Partial Full-Attention Prefix Cache

This is the main production feature in the patch.

`vllm/v1/core/block_pool.py` adds:

- `PartialKVCacheBlock`: source block plus valid token count.
- `cached_partial_block_hash_to_block`: hash to partial block.
- `partial_block_id_to_hashes`: reverse index for eviction cleanup.
- `get_cached_partial_block()`.
- `cache_partial_block()`.
- partial metadata cleanup during block eviction and reset.

See `vllm/v1/core/block_pool.py:34`-`39`,
`vllm/v1/core/block_pool.py:178`-`183`,
`vllm/v1/core/block_pool.py:223`-`243`,
`vllm/v1/core/block_pool.py:355`-`403`,
`vllm/v1/core/block_pool.py:450`-`497`, and
`vllm/v1/core/block_pool.py:551`-`586`.

`cache_partial_block()` records only the latest hash boundary committed by that
cache write. It does not backfill every smaller hash boundary inside the same
trailing physical full-attention block. Earlier partial boundaries are present
only if the scheduler previously split and committed at those boundaries. This
avoids advertising a full-attention partial hit at a prefix length where Mamba
has no matching checkpoint, which would create metadata and copy work that
cannot produce a valid hybrid hit.

`vllm/v1/core/single_type_kv_cache_manager.py` adds
`enable_partial_cache` to single-type managers. Full attention overrides
`cache_blocks()` to call `block_pool.cache_partial_block()` after normal full
block caching. See:

- manager flag: `vllm/v1/core/single_type_kv_cache_manager.py:39`-`72`
- full attention partial cache write:
  `vllm/v1/core/single_type_kv_cache_manager.py:480`-`492`

`vllm/v1/core/kv_cache_coordinator.py` adds:

- `PartialKVCacheHit`
- `enable_partial_attn_cache`
- partial-hit capacity accounting
- source-block pinning
- worker copy-op construction
- fixed-point hybrid lookup with partial full-attention recovery

See:

- data model and coordinator state:
  `vllm/v1/core/kv_cache_coordinator.py:27`-`61`
- passing `enable_partial_cache` only to full-attention groups:
  `vllm/v1/core/kv_cache_coordinator.py:71`-`87`
- capacity/pinning/copy operation helpers:
  `vllm/v1/core/kv_cache_coordinator.py:135`-`150`,
  `vllm/v1/core/kv_cache_coordinator.py:220`-`241`
- LCM relaxation:
  `vllm/v1/core/kv_cache_coordinator.py:520`-`530`
- partial full-attention lookup:
  `vllm/v1/core/kv_cache_coordinator.py:532`-`579`
- final partial-hit selection:
  `vllm/v1/core/kv_cache_coordinator.py:699`-`719`

`vllm/v1/core/kv_cache_manager.py` stores pending partial hits after lookup,
accounts for blocks that need to be touched, pins source blocks before
allocation, emits copy ops, and clears pending state on failure/free. See
`vllm/v1/core/kv_cache_manager.py:180`-`226`,
`vllm/v1/core/kv_cache_manager.py:390`-`437`, and
`vllm/v1/core/kv_cache_manager.py:457`-`467`.

`vllm/v1/core/kv_cache_utils.py` adds `KVCacheBlockCopy`, a small scheduler to
worker command containing group id, source block id, destination block id, and
number of valid tokens. See `vllm/v1/core/kv_cache_utils.py:155`-`164`.

`vllm/v1/core/sched/output.py` carries those copy ops to the worker in
`SchedulerOutput.kv_cache_block_copy_ops`. See
`vllm/v1/core/sched/output.py:227`-`232`.

`vllm/v1/core/sched/scheduler.py` drains the KV manager's copy ops into the
scheduler output. See `vllm/v1/core/sched/scheduler.py:1056`-`1077`.

`vllm/v1/worker/gpu_model_runner.py` consumes the copy ops. It copies only
full-attention KV cache sub-blocks, verifies token count alignment to kernel
block size, deduplicates by tensor pointer/source/destination/count, and copies
from the source physical block region to the destination physical block region.
See `vllm/v1/worker/gpu_model_runner.py:1041`-`1096` and the call site at
`vllm/v1/worker/gpu_model_runner.py:1137`.

Contract:

- Partial reuse is only for full-attention groups.
- Mamba still needs an actual Mamba checkpoint at the same final hit length.
- Partial metadata is written only for committed checkpoint boundaries; a
  single later commit does not imply reusable earlier boundaries.
- The source partial full-attention block must be pinned until the worker copy
  runs, otherwise eviction can corrupt the copy.
- The destination full-attention block remains uncached until enough tokens are
  finalized.

### 5. Latest-Mamba Scheduler Checkpoints

`vllm/v1/core/sched/scheduler.py` extends Mamba block-aligned splitting from
`align` only to `("align", "latest")`. See
`vllm/v1/core/sched/scheduler.py:277`-`286`.

Latest mode changes `_mamba_block_aligned_split()`:

- uses `mamba_block_size` as the Mamba checkpoint block,
- tracks full-attention block size separately,
- only applies Eagle pruning when `use_eagle_prefix_cache_drop` is true,
- builds checkpoint candidates from latest boundary, coarse boundaries, and
  tail checkpoints,
- shortens the scheduled chunk to hit the next checkpoint when needed.

See `vllm/v1/core/sched/scheduler.py:350`-`466`.

Coarse checkpoint selection is implemented in
`_latest_mamba_coarse_checkpoints()`. It walks backward from the latest Mamba
boundary in full-attention block-size steps and keeps candidates that are at
least `mamba_latest_coarse_min_gap` behind latest. See
`vllm/v1/core/sched/scheduler.py:330`-`348`.

Contract:

- `tail_checkpoints=0` means do not keep extra near-tail Mamba checkpoints.
- `coarse_min_gap=512` means keep a full-attention boundary only if it is far
  enough from the latest Mamba boundary.
- Coarse checkpoints prevent a full-attention boundary from becoming unusable
  solely because Mamba lacks a matching state there.

### 6. MTP Prefill Budget Reservation

The patch adds `_mtp_waiting_prefill_token_reserve()` and subtracts the reserve
from the running-request scheduling budget only when all of the following hold:

- speculative method is MTP,
- Mamba block-aligned split is active,
- `num_spec_tokens > 0`.

See `vllm/v1/core/sched/scheduler.py:282`-`286` and
`vllm/v1/core/sched/scheduler.py:468`-`508`.

Why this exists:

- MTP can keep many running decode requests active.
- Mamba latest/align prefill may need a full block-sized chunk to reach a
  checkpoint.
- Without a small reservation, running decode can consume the entire token
  budget, leaving waiting prefills unable to advance to cacheable boundaries.

Contract:

- Non-MTP latest-Mamba should not pay this reservation cost.
- MTP only reserves up to one or two scheduler blocks, bounded by token budget.
- When running requests are preempted and their tokens are restored, both the
  total token budget and running token budget are restored. See
  `vllm/v1/core/sched/scheduler.py:620`-`658`.

### 7. Mamba Manager And Worker Paths

`vllm/v1/core/single_type_kv_cache_manager.py` treats latest like align in the
Mamba manager:

- request allocation state exists for both `align` and `latest`,
- skipped-block removal handles both,
- block counting/allocation handles both,
- free cleanup handles both.

See `vllm/v1/core/single_type_kv_cache_manager.py:799`-`807`,
`vllm/v1/core/single_type_kv_cache_manager.py:857`-`885`,
`vllm/v1/core/single_type_kv_cache_manager.py:893`-`1034`, and
`vllm/v1/core/single_type_kv_cache_manager.py:1036`-`1040`.

`vllm/v1/kv_cache_interface.py` sizes latest mode for one running state plus a
bounded latest checkpoint and, when configured, one coarse checkpoint. This
keeps the memory model aligned with the bounded per-request Mamba state policy.

Latest mode delays Mamba prefix-cache insertion until request free. During
execution, `MambaManager.cache_blocks()` queues only the newest Mamba boundary
and the selected coarse boundary. When a newer latest/coarse boundary replaces
an older queued block, the old block is released unless it is still the current
running state or the other queued checkpoint. `KVCacheManager.free()` calls a
coordinator finalization hook before releasing the request, and
`MambaManager.finalize_request_cache()` hashes only those queued checkpoint
blocks.

For non-speculative latest mode, `MambaManager.allocate_new_blocks()` reuses the
same physical running-state block when the previous state is not a queued
checkpoint. This avoids a worker-side copy for ordinary progress between
cacheable boundaries. If the previous state is the queued latest/coarse
checkpoint, a new running block is allocated and the worker copies from the
preserved checkpoint into the new running block.

`vllm/v1/attention/backends/utils.py` updates Mamba block-table docs so
`align` and `latest` share the same block-table shape behavior. See
`vllm/v1/attention/backends/utils.py:866`-`873`.

`vllm/v1/worker/gpu_model_runner.py` treats latest like align in both Mamba
postprocess and preprocess call sites. `vllm/v1/worker/mamba_utils.py` tracks
both the logical state index and the physical Mamba block IDs. If the scheduler
reused the same physical block at a later logical state slot, the worker sees
identical physical IDs and skips the redundant state copy.

Contract:

- Latest mode keeps one running state plus a bounded latest/coarse checkpoint
  set while the request is active.
- Latest mode differs from align in scheduler/hash/checkpoint policy and in
  delayed Mamba cache finalization.

## End-To-End Request Flow

### Step 1: Serve Flags Become Runtime Config

In patched `mamba` mode, `serve.sh` passes:

```text
--mamba-cache-mode latest
--mamba-latest-tail-checkpoints 0
--mamba-latest-coarse-min-gap 512
```

In patched `mtp` mode, it also passes:

```text
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

The CLI parser stores these in `EngineArgs`, then `CacheConfig` and
`SpeculativeConfig` are built.

Base mode does not pass `latest`. If prefix caching is enabled and the model
config leaves Mamba cache mode as `none`, vLLM chooses `all` or `align` based on
model support. This is important for benchmark labels: "base" is clean vLLM,
not latest-Mamba.

### Step 2: Model Config And Engine Core Establish Block Geometry

Model config verifies that latest mode is compatible with chunked prefill. Then
engine core computes:

- `scheduler_block_size`: normal scheduling block size after parallelism factors.
- `cache_hash_block_size`: same as scheduler block size normally, but the
  smallest KV-cache group block size in latest mode with Mamba layers.

The request block hasher uses `cache_hash_block_size`, so prompt hashes exist at
the smaller Mamba checkpoint granularity.

### Step 3: Scheduler Initializes Speculative And Cache Behavior

Scheduler initialization separates these concepts:

- `use_eagle`: broad Eagle-compatible speculative path.
- `use_eagle_prefix_cache_drop`: only true for Eagle/Eagle3.
- `use_mtp`: true only for MTP.
- `num_lookahead_tokens`: zero for MTP, nonzero for Eagle/Eagle3 and draft
  model speculation.

Then it enables partial full-attention cache only if latest mode is active and
all safety gates pass. MTP passes this gate because MTP no longer enables
Eagle prefix-cache drop.

### Step 4: First Request Prefill Creates Reusable State

When the first turn is prefilling, `_mamba_block_aligned_split()` may shorten
chunks so scheduled work ends at useful Mamba checkpoints. In optimized mode
with `tail=0` and `coarse_min_gap=512`, it avoids near-tail checkpoint churn
but can still preserve a coarse full-attention boundary checkpoint.

After slots are allocated and tokens are finalized, `cache_blocks()` commits:

- full blocks into the normal prefix-cache hash table,
- the latest committed partial full-attention boundary inside a trailing
  full-attention physical block, if partial cache is enabled,
- Mamba latest checkpoints through the Mamba manager and worker state-copy path.

Rejected speculative tokens are not committed because cache commit is capped at
`request.num_tokens`.

### Step 5: Later Turn Looks Up Prefix

For the next request, scheduler calls `get_computed_blocks()`.

The coordinator runs hybrid lookup:

1. Full attention finds whole physical blocks from left to right.
2. Mamba searches for the latest matching checkpoint.
3. If full attention would otherwise limit the hit to a larger physical block
   boundary, partial full-attention lookup can recover a smaller hash boundary
   inside the trailing physical block.
4. The final hit length must be accepted by all KV groups.

The result becomes `request.num_cached_tokens`, and the remaining prompt tail is
scheduled as local prefill work.

### Step 6: Partial Full-Attention Hit Is Made Writable

If the final hybrid hit requires a partial full-attention block:

1. The coordinator records a `PartialKVCacheHit`.
2. `KVCacheManager.allocate_slots()` accounts for source blocks that must be
   touched and pins them before allocation.
3. The destination block is allocated for the new request.
4. A `KVCacheBlockCopy` operation is emitted in `SchedulerOutput`.
5. The worker copies the valid KV sub-block range from source to destination
   before forward.

This makes the cached partial prefix writable for the new request without
mutating the shared source block.

### Step 7: MTP Decode Runs Without Eagle KV Penalties

In MTP mode:

- MTP has speculative tokens and Mamba speculative state blocks.
- Scheduler `num_lookahead_tokens` remains zero.
- Hybrid prefix lookup does not drop the last matched block.
- Mamba preprocess/postprocess runs in latest mode just like align mode.
- Accepted draft-token state can be copied into the aligned Mamba checkpoint.
- Rejected draft-token state is not committed as reusable prefix cache.

The only MTP-specific scheduler budget change is the waiting-prefill reserve,
which exists to keep prefills able to reach Mamba-aligned checkpoints under
decode pressure.

## Eviction And Pressure Behavior

The patch does not change the fundamental free queue ordering:

- free queue is LRU-first,
- for same access time, tail blocks are first,
- allocation pops from the front.

See `vllm/v1/core/kv_cache_utils.py:177`-`184` and
`vllm/v1/core/kv_cache_utils.py:220`-`255`.

What the patch adds is partial-cache metadata cleanup. If a physical block is
evicted, `_maybe_evict_cached_partial_block()` removes all partial hash entries
that point at it before normal full-block hash cleanup. See
`vllm/v1/core/block_pool.py:450`-`497`.

This matters under high KV pressure:

- A Mamba checkpoint or partial full-attention source block can be evicted.
- If either side is missing, the hybrid hit must fall back to a shorter prefix.
- High pressure can therefore make latest-Mamba checkpoints "not hit" even if
  the scheduling policy created them earlier.

## Metrics And Prefill Accounting

The scheduler sets `request.num_cached_tokens` from the final computed prefix
length for a newly scheduled request. See
`vllm/v1/core/sched/scheduler.py:966`-`977`.

For a prompt request, local prefill work is effectively:

```text
local_prefill_tokens = prompt_tokens - num_cached_tokens
```

For benchmark interpretation:

- Lower local prefill means more prefix was reused.
- Latest-Mamba can reduce local prefill by allowing a smaller Mamba checkpoint
  and a partial full-attention copy.
- MTP should not increase local prefill via Eagle drop or KV lookahead. If it
  does, review the speculative predicates and scheduler initialization first.

## Review Checklist

When reviewing `gold.patch`, check these invariants:

- `gold.patch` contains only production runtime files under `vllm/`.
- `mamba_cache_mode="latest"` is accepted by config and CLI.
- Latest mode uses `mamba_block_size` for cacheable checkpoint alignment.
- Latest mode uses smaller `hash_block_size` in engine core and scheduler.
- Partial full-attention cache is enabled only under safe gates:
  latest mode, hybrid Mamba, smaller hash block than attention block, no
  connector, no Eagle prefix drop, DCP=1, PCP=1.
- Partial source blocks are pinned before copy ops are emitted.
- Worker copies only aligned full-attention KV sub-blocks.
- Partial-cache metadata is removed when a source block is evicted or prefix
  cache is reset.
- Partial-cache metadata is not backfilled for uncommitted prior boundaries,
  so full-attention copy ops are not emitted for boundaries that latest-Mamba
  cannot use.
- MTP still returns true for `use_eagle()`, but false for Eagle prefix-cache
  drop and KV lookahead.
- MTP does not set scheduler `num_lookahead_tokens`.
- MTP prefill budget reservation is active only for MTP with Mamba
  block-aligned split.
- Mamba manager and worker paths treat latest like align for state storage and
  copy mechanics.

## Likely Failure Modes

High prefill tokens under latest-Mamba usually means one of these happened:

- Partial full-attention cache was not enabled because a gate failed.
- Mamba checkpoint did not exist at the final prefix length.
- Full-attention partial source block was evicted under KV pressure.
- Mamba checkpoint block was evicted under KV pressure.
- Tail checkpoints were disabled and the reusable prefix ended near, but before,
  the latest Mamba boundary.
- Coarse checkpoint min gap was too high or coarse checkpoint count was zero.
- A requested prior partial boundary was never explicitly committed by a
  scheduler split, so only the latest committed partial boundary is cached.
- MTP accidentally used Eagle prefix-cache drop.
- MTP accidentally reserved verifier KV lookahead slots and increased pressure.
- `--no-scheduler-reserve-full-isl` changed admission/concurrency behavior so
  pressure invalidated otherwise reusable cache.

## Test Patch Coverage To Read With This Flow

The production contracts above are covered by `test.patch` in these areas:

- MTP speculative config does not enable Eagle drop/lookahead.
- MTP latest-Mamba does not reserve Eagle KV lookahead.
- MTP latest-Mamba keeps partial prefix cache enabled.
- Latest-Mamba prefix hit after chunked prefill.
- Tail checkpoint policy and stride split positions.
- Coarse checkpoint selector.
- Completed decode tokens reused by the next turn.
- Partial full-attention hit, current-boundary-only behavior, explicit
  prior-boundary hit, and eviction cleanup.
