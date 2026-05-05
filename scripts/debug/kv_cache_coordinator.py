# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from math import lcm

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
    KVCacheBlockCopy,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager,
    FullAttentionManager,
    MambaManager,
    SingleTypeKVCacheManager,
    get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
)
from vllm.v1.request import Request


@dataclass(frozen=True, slots=True)
class PartialKVCacheHit:
    kv_cache_group_id: int
    src_block: KVCacheBlock
    block_index: int
    num_tokens: int
    hit_length: int


class KVCacheCoordinator(ABC):
    """
    Coordinate the KV cache of different KV cache groups.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        enable_partial_attn_cache: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        self.enable_partial_attn_cache = enable_partial_attn_cache
        self._last_partial_hits: list[PartialKVCacheHit] = []

        self.block_pool = BlockPool(
            kv_cache_config.num_blocks,
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        )

        # Needs special handling for find_longest_cache_hit if eagle is enabled
        self.use_eagle = use_eagle
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pool,
                enable_caching=enable_caching,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
                enable_partial_cache=(
                    enable_partial_attn_cache
                    and isinstance(kv_cache_group.kv_cache_spec, FullAttentionSpec)
                ),
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int:
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.
            num_encoder_tokens: The number of encoder tokens for allocating
                blocks for cross-attention.
            total_computed_tokens: Include both local and external tokens.
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.

        Returns:
            The number of blocks to allocate.
        """
        num_blocks_to_allocate = 0
        for i, manager in enumerate(self.single_type_managers):
            if isinstance(manager, CrossAttentionManager):
                # For cross-attention, we issue a single static allocation
                # of blocks based on the number of encoder input tokens.
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id, num_encoder_tokens, [], 0, num_encoder_tokens
                )
            else:
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id,
                    num_tokens,
                    new_computed_blocks[i],
                    total_computed_tokens,
                    num_tokens_main_model,
                )
        return num_blocks_to_allocate

    def get_num_blocks_to_touch(
        self, partial_hits: Sequence[PartialKVCacheHit]
    ) -> int:
        block_ids: set[int] = set()
        for hit in partial_hits:
            block = hit.src_block
            if block.ref_cnt == 0 and not block.is_null:
                block_ids.add(block.block_id)
        return len(block_ids)

    def pin_partial_cache_hits(
        self, request_id: str, partial_hits: Sequence[PartialKVCacheHit]
    ) -> None:
        blocks_by_group: dict[int, list[KVCacheBlock]] = {}
        seen_block_ids: set[int] = set()
        for hit in partial_hits:
            if hit.src_block.block_id in seen_block_ids:
                continue
            seen_block_ids.add(hit.src_block.block_id)
            blocks_by_group.setdefault(hit.kv_cache_group_id, []).append(hit.src_block)

        for group_id, blocks in blocks_by_group.items():
            self.single_type_managers[group_id].pin_blocks(request_id, blocks)

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Add the new computed blocks to the request. Optionally allocate new
            blocks for external computed tokens (if any).

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
        """
        for i, manager in enumerate(self.single_type_managers):
            manager.allocate_new_computed_blocks(
                request_id,
                new_computed_blocks[i],
                num_local_computed_tokens,
                num_external_computed_tokens,
            )

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
        num_encoder_tokens: int = 0,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
            num_encoder_tokens: The number of encoder tokens for allocating
                blocks for cross-attention.

        Returns:
            The new allocated blocks.
        """
        return tuple(
            manager.allocate_new_blocks(
                request_id,
                num_encoder_tokens
                if isinstance(manager, CrossAttentionManager)
                else num_tokens,
                num_tokens_main_model,
            )
            for manager in self.single_type_managers
        )

    def make_partial_cache_copy_ops(
        self, request_id: str, partial_hits: Sequence[PartialKVCacheHit]
    ) -> list[KVCacheBlockCopy]:
        copy_ops: list[KVCacheBlockCopy] = []
        for hit in partial_hits:
            manager = self.single_type_managers[hit.kv_cache_group_id]
            req_blocks = manager.req_to_blocks[request_id]
            if hit.block_index >= len(req_blocks):
                continue
            dst_block = req_blocks[hit.block_index]
            if dst_block.is_null or dst_block.block_id == hit.src_block.block_id:
                continue
            copy_ops.append(
                KVCacheBlockCopy(
                    kv_cache_group_id=hit.kv_cache_group_id,
                    src_block_id=hit.src_block.block_id,
                    dst_block_id=dst_block.block_id,
                    num_tokens=hit.num_tokens,
                )
            )
        return copy_ops

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_computed_tokens: The total number of tokens
                that need to be cached
                (including tokens that are already cached).
        """
        for manager in self.single_type_managers:
            manager.cache_blocks(request, num_computed_tokens)

    def finalize_request_cache(self, request: Request) -> None:
        """Commit cache state that is intentionally delayed until request end."""
        for manager in self.single_type_managers:
            manager.finalize_request_cache(request)

    def cache_completed_mamba_boundaries(
        self, request: Request, num_computed_tokens: int
    ) -> None:
        """Publish completed Mamba boundaries before running-state reuse."""
        return

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        for manager in self.single_type_managers:
            manager.free(request_id)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """
        Get the number of common prefix blocks for all requests with allocated
        KV cache for each kv cache group.

        Args:
            running_request_id: The request ID of any running request, used to
                identify the common prefix blocks.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache group.
        """
        return [
            manager.get_num_common_prefix_blocks(running_request_id)
            for manager in self.single_type_managers
        ]

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """
        Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            total_computed_tokens: The total number of computed tokens, including
                local computed tokens and external computed tokens.
        """
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(request_id, total_computed_tokens)

    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        """
        Get the blocks for the request.
        """
        return tuple(
            manager.req_to_blocks.get(request_id) or []
            for manager in self.single_type_managers
        )

    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        pass

    def new_step_starts(self) -> None:
        """Called when a new step is started."""
        for manager in self.single_type_managers:
            manager.new_step_starts()

    def take_mamba_source_block_refs(self) -> list[tuple[int, str, int]]:
        refs: list[tuple[int, str, int]] = []
        for manager in self.single_type_managers:
            if isinstance(manager, MambaManager):
                refs.extend(manager.take_mamba_source_block_refs())
        return refs

    def release_mamba_source_block_refs(
        self, refs: list[tuple[int, str, int]] | None
    ) -> None:
        if not refs:
            return
        for kv_cache_group_id, request_id, block_idx in refs:
            manager = self.single_type_managers[kv_cache_group_id]
            if isinstance(manager, MambaManager):
                manager.release_mamba_source_block_ref(request_id, block_idx)

    def take_last_partial_hits(self) -> list[PartialKVCacheHit]:
        hits = self._last_partial_hits
        self._last_partial_hits = []
        return hits


class KVCacheCoordinatorNoPrefixCache(KVCacheCoordinator):
    """
    KV cache coordinator to use if prefix caching is disabled or unsupported.
    In contrast to UnitaryKVCacheCoordinator and HybridKVCacheCoordinator,
    supports arbitrary numbers of KV cache groups (including 0 groups).
    Does not implement any features related to prefix caching.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        enable_partial_attn_cache: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            use_eagle,
            False,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            enable_partial_attn_cache=enable_partial_attn_cache,
            metrics_collector=metrics_collector,
        )
        self.num_single_type_manager = len(self.single_type_managers)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        return [0] * self.num_single_type_manager

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(self.num_single_type_manager)
        )
        return blocks, 0


class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for models with only one KV cache group. This is the
    case for models with only one KV cache type, e.g., all attention layers use
    full attention or all attention layers use sliding window attention.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        enable_partial_attn_cache: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            enable_partial_attn_cache=enable_partial_attn_cache,
            metrics_collector=metrics_collector,
        )
        self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.block_size = self.kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        if dcp_world_size > 1:
            self.block_size *= dcp_world_size
        if pcp_world_size > 1:
            self.block_size *= pcp_world_size
        # For models using only Mamba, block_size is set to max_model_len when
        # prefix caching is disabled, and hash_block_size validation is skipped.
        assert not enable_caching or (hash_block_size == self.block_size), (
            "UnitaryKVCacheCoordinator assumes hash_block_size == block_size"
        )
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "UnitaryKVCacheCoordinator assumes only one kv cache group"
        )

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        hit_blocks = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=self.use_eagle,
            alignment_tokens=self.block_size,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
        )
        return hit_blocks, len(hit_blocks[0]) * self.block_size


class HybridKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for hybrid models with multiple KV cache types, and
    thus multiple kv cache groups.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        enable_partial_attn_cache: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            enable_partial_attn_cache=enable_partial_attn_cache,
            metrics_collector=metrics_collector,
        )
        # hash_block_size: the block size used to compute block hashes.
        # The actual block size usually equals hash_block_size, but in cases where
        # different KV cache groups have different block sizes, the actual block size
        # can be a multiple of hash_block_size.
        self.hash_block_size = hash_block_size
        self.use_mamba_anchored_partial_cache = (
            enable_partial_attn_cache
            and any(
                isinstance(g.kv_cache_spec, MambaSpec)
                and g.kv_cache_spec.mamba_cache_mode == "latest"
                for g in kv_cache_config.kv_cache_groups
            )
        )
        if self.use_mamba_anchored_partial_cache:
            for manager in self.single_type_managers:
                if isinstance(manager.kv_cache_spec, FullAttentionSpec):
                    manager.enable_eager_partial_cache = False
        assert all(
            g.kv_cache_spec.block_size % hash_block_size == 0
            for g in kv_cache_config.kv_cache_groups
        ), "block_size must be divisible by hash_block_size"
        assert dcp_world_size == 1, "DCP not support hybrid attn now."
        assert pcp_world_size == 1, "PCP not support hybrid attn now."
        self.verify_and_split_kv_cache_groups()

    def verify_and_split_kv_cache_groups(self) -> None:
        """
        Groups KV cache groups by their spec type for efficient batch processing
        during cache hit lookup.
        """
        attention_groups: list[
            tuple[KVCacheSpec, list[int], type[SingleTypeKVCacheManager]]
        ] = []

        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            manager_cls = self.single_type_managers[i].__class__
            spec = g.kv_cache_spec

            # Try to find an existing group with the same spec
            for existing_spec, group_ids, existing_cls in attention_groups:
                if existing_spec == spec:
                    assert manager_cls is existing_cls, (
                        "Expected same manager class for identical KV cache specs."
                    )
                    group_ids.append(i)
                    break
            else:
                attention_groups.append((spec, [i], manager_cls))

        assert len(attention_groups) > 1, (
            "HybridKVCacheCoordinator requires at least two attention groups."
        )

        if self.use_mamba_anchored_partial_cache:
            # Latest-Mamba partial hits are valid only at Mamba checkpoint
            # boundaries. Check Mamba first so full attention only validates the
            # Mamba-approved boundary instead of scanning arbitrary 16-token
            # partial candidates.
            self.attention_groups = sorted(
                attention_groups,
                key=lambda x: isinstance(x[0], FullAttentionSpec),
            )
        else:
            # Put full attention first: its efficient left-to-right scan provides
            # a tighter initial bound, reducing work for subsequent groups.
            self.attention_groups = sorted(
                attention_groups,
                key=lambda x: not isinstance(x[0], FullAttentionSpec),
            )

        # The LCM of the block sizes of all attention types.
        # The cache hit length must be a multiple of the LCM of the block sizes
        # to make sure the cache hit length is a multiple of the block size of
        # each attention type. The experimental latest-mode path relaxes this
        # for full attention by copying one trailing partial physical block.
        block_sizes = [spec.block_size for spec, _, _ in attention_groups]
        self.lcm_block_size = (
            self.hash_block_size
            if self.enable_partial_attn_cache
            else lcm(*block_sizes)
        )

    def _find_partial_full_attention_hit(
        self,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_size: int,
        whole_hit_blocks: tuple[list[KVCacheBlock], ...],
    ) -> tuple[int, list[PartialKVCacheHit]]:
        if (
            not self.enable_partial_attn_cache
            or self.use_eagle
            or block_size <= self.hash_block_size
        ):
            return len(whole_hit_blocks[0]) * block_size, []

        whole_hit_length = len(whole_hit_blocks[0]) * block_size
        max_partial_length = min(max_length, whole_hit_length + block_size - 1)
        candidate = max_partial_length - max_partial_length % self.hash_block_size
        while candidate > whole_hit_length:
            hash_idx = candidate // self.hash_block_size - 1
            if hash_idx >= len(block_hashes):
                if self.use_mamba_anchored_partial_cache:
                    break
                candidate -= self.hash_block_size
                continue

            partial_blocks = self.block_pool.get_cached_partial_block(
                block_hashes[hash_idx], kv_cache_group_ids
            )
            valid_tokens = candidate % block_size
            if partial_blocks is not None and all(
                block.valid_tokens >= valid_tokens for block in partial_blocks
            ):
                print(
                    "[MAMBA_DEBUG] partial_attn_hit",
                    "candidate=", candidate,
                    "valid=", valid_tokens,
                    "groups=", kv_cache_group_ids,
                    "block_ids=", [b.block.block_id for b in partial_blocks],
                    flush=True,
                )
                block_index = candidate // block_size
                partial_hits = [
                    PartialKVCacheHit(
                        kv_cache_group_id=group_id,
                        src_block=partial_block.block,
                        block_index=block_index,
                        num_tokens=valid_tokens,
                        hit_length=candidate,
                    )
                    for group_id, partial_block in zip(
                        kv_cache_group_ids, partial_blocks
                    )
                ]
                return candidate, partial_hits
            if self.use_mamba_anchored_partial_cache:
                print(
                    "[MAMBA_DEBUG] partial_attn_miss",
                    "candidate=", candidate,
                    "valid=", valid_tokens,
                    "hash_idx=", hash_idx,
                    "groups=", kv_cache_group_ids,
                    flush=True,
                )
                break
            candidate -= self.hash_block_size

        return whole_hit_length, []

    def finalize_request_cache(self, request: Request) -> None:
        mamba_boundaries: list[int] = []
        if self.use_mamba_anchored_partial_cache:
            for manager in self.single_type_managers:
                if not isinstance(manager, MambaManager):
                    continue
                boundary = manager.publish_latest_prefill_checkpoint(request)
                print(
                    "[MAMBA_DEBUG] finalize_boundary",
                    "req=", request.request_id[:8],
                    "prompt=", request.num_prompt_tokens,
                    "boundary=", boundary,
                    flush=True,
                )
                if boundary is not None:
                    mamba_boundaries.append(boundary)

        super().finalize_request_cache(request)

        if not mamba_boundaries:
            print(
                "[MAMBA_DEBUG] finalize_no_mirror",
                "req=", request.request_id[:8],
                "prompt=", request.num_prompt_tokens,
                flush=True,
            )
            return
        unique_boundaries = sorted({b for b in mamba_boundaries if b > 0})
        print(
            "[MAMBA_DEBUG] finalize_mirror",
            "req=", request.request_id[:8],
            "boundaries=", unique_boundaries,
            flush=True,
        )
        for manager in self.single_type_managers:
            if not isinstance(manager, FullAttentionManager):
                continue
            if not manager.enable_partial_cache:
                continue
            for boundary in unique_boundaries:
                manager.cache_partial_boundary(request, boundary)

    def cache_completed_mamba_boundaries(
        self, request: Request, num_computed_tokens: int
    ) -> None:
        if not self.use_mamba_anchored_partial_cache:
            return
        for manager in self.single_type_managers:
            if isinstance(manager, MambaManager):
                manager.publish_completed_latest_checkpoint(
                    request, num_computed_tokens
                )

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        Find the longest cache hit using an iterative fixed-point algorithm.

        Each attention type either accepts the current candidate length or
        reduces it. If any type reduces the length, restart checks over all
        types. This converges because length monotonically decreases and is
        bounded below by 0.

        Args:
            block_hashes: The block hashes of the request.
            max_cache_hit_length: The maximum length of the cache hit.

        Returns:
            A tuple containing:
                - A tuple of the cache hit blocks for each single type manager.
                - The number of tokens of the longest cache hit.
        """

        def _get_block_hashes(kv_cache_spec: KVCacheSpec) -> BlockHashList:
            if kv_cache_spec.block_size == self.hash_block_size:
                return block_hashes
            return BlockHashListWithBlockSize(
                block_hashes, self.hash_block_size, kv_cache_spec.block_size
            )

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        self._last_partial_hits = []
        hit_length = max_cache_hit_length
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups
        partial_hits: list[PartialKVCacheHit] = []

        # Simple hybrid (1 full attn + 1 other): one iteration suffices.
        # Full attn is always first if it exists. This avoids EAGLE drops
        # being applied multiple times to non-full-attn groups.
        # FIXME (yifan): However, for complex hybrid models with multiple attn
        # groups, we still have the EAGLE spiral block dropping problem. See
        # discussion in issue https://github.com/vllm-project/vllm/issues/32802.
        is_simple_hybrid = len(self.attention_groups) == 2 and any(
            isinstance(spec, FullAttentionSpec)
            for spec, _, _ in self.attention_groups
        )

        while True:
            curr_hit_length = hit_length

            for spec, group_ids, manager_cls in self.attention_groups:
                is_full_attn = isinstance(spec, FullAttentionSpec)
                max_group_hit_length = curr_hit_length

                # Full attention: reuse cached blocks (downward-closed property)
                cached_blocks = hit_blocks_by_group[group_ids[0]]
                if is_full_attn and cached_blocks is not None:
                    # For full attention, we only need to compute the cache hit
                    # length once. Starting from the second iteration, if the
                    # curr_hit_length is reduced by other groups, we can simply
                    # keep the first (curr_hit_length // block_size) blocks from
                    # the last iteration.
                    num_blocks = curr_hit_length // spec.block_size
                    curr_hit_length = num_blocks * spec.block_size
                    hit_blocks = tuple(
                        (hit_blocks_by_group[group_id] or [])[:num_blocks]
                        for group_id in group_ids
                    )
                    if self.enable_partial_attn_cache:
                        curr_hit_length, group_partial_hits = (
                            self._find_partial_full_attention_hit(
                                block_hashes=block_hashes,
                                max_length=max_group_hit_length,
                                kv_cache_group_ids=group_ids,
                                block_size=spec.block_size,
                                whole_hit_blocks=hit_blocks,
                            )
                        )
                        partial_hits = group_partial_hits or partial_hits
                else:
                    hit_blocks = manager_cls.find_longest_cache_hit(
                        block_hashes=_get_block_hashes(spec),
                        max_length=max_group_hit_length,
                        kv_cache_group_ids=group_ids,
                        block_pool=self.block_pool,
                        kv_cache_spec=spec,
                        use_eagle=self.use_eagle,
                        alignment_tokens=self.lcm_block_size,
                    )
                    curr_hit_length = len(hit_blocks[0]) * spec.block_size
                    if is_full_attn and self.enable_partial_attn_cache:
                        curr_hit_length, group_partial_hits = (
                            self._find_partial_full_attention_hit(
                                block_hashes=block_hashes,
                                max_length=max_group_hit_length,
                                kv_cache_group_ids=group_ids,
                                block_size=spec.block_size,
                                whole_hit_blocks=hit_blocks,
                            )
                        )
                        partial_hits = group_partial_hits or partial_hits
                    for group_id, blocks in zip(group_ids, hit_blocks):
                        hit_blocks_by_group[group_id] = blocks

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            # Simple hybrid: exit after one iteration
            if is_simple_hybrid:
                break

        # Truncate blocks to final hit_length. In latest-Mamba mode, Mamba can
        # be checked before full attention, and full attention may subsequently
        # reduce the accepted length.
        for spec, group_ids, _ in self.attention_groups:
            num_blocks = hit_length // spec.block_size
            for group_id in group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]

        final_partial_hits: list[PartialKVCacheHit] = []
        for hit in partial_hits:
            if hit.hit_length < hit_length:
                continue
            spec = self.kv_cache_config.kv_cache_groups[
                hit.kv_cache_group_id
            ].kv_cache_spec
            if not isinstance(spec, FullAttentionSpec):
                continue
            block_start = hit.block_index * spec.block_size
            if block_start < hit_length < block_start + spec.block_size:
                final_partial_hits.append(
                    PartialKVCacheHit(
                        kv_cache_group_id=hit.kv_cache_group_id,
                        src_block=hit.src_block,
                        block_index=hit.block_index,
                        num_tokens=hit_length - block_start,
                        hit_length=hit_length,
                    )
                )
        self._last_partial_hits = final_partial_hits

        return tuple(
            blocks if blocks is not None else [] for blocks in hit_blocks_by_group
        ), hit_length


def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    hash_block_size: int,
    enable_partial_attn_cache: bool = False,
    metrics_collector: KVCacheMetricsCollector | None = None,
) -> KVCacheCoordinator:
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            enable_partial_attn_cache=enable_partial_attn_cache,
            metrics_collector=metrics_collector,
        )
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            enable_partial_attn_cache=enable_partial_attn_cache,
            metrics_collector=metrics_collector,
        )
    return HybridKVCacheCoordinator(
        kv_cache_config,
        max_model_len,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        hash_block_size=hash_block_size,
        enable_partial_attn_cache=enable_partial_attn_cache,
        metrics_collector=metrics_collector,
    )
