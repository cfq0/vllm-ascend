# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM projectx
import sys
from collections.abc import Sequence
from math import lcm

import vllm
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    KVCacheCoordinator,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend import envs
from vllm_ascend.core.pd_block_pool import (
    DECODE_KV_BLOCK_REGION_SIZE,
    PDBlockPool,
    PDBlockPoolConfig,
)
from vllm_ascend.core.single_type_kv_cache_manager import get_manager_for_kv_cache_spec

USE_MULTI_GROUPS_KV_CACHE = True
# Real SWA KV block_size; keep in sync with dsa_cp.SWA_KV_BLOCK_SIZE.
_SWA_KV_BLOCK_SIZE = 128
_C4_COMPRESS_RATIO = 4

try:
    from vllm.v1.core.single_type_kv_cache_manager import CrossAttentionManager
except ImportError:  # pragma: no cover
    CrossAttentionManager = None  # type: ignore[misc, assignment]


def _layer_kv_specs(spec: KVCacheSpec) -> list[KVCacheSpec]:
    """Unwrap ``UniformTypeKVCacheSpecs`` to per-layer specs."""
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return list(spec.kv_cache_specs.values())
    return [spec]


def _is_swa_kv_manager(manager: SingleTypeKVCacheManager) -> bool:
    """True for real SWA KV (SlidingWindowMLASpec, block_size=128).

    DSv4 groups C4/C128 first, then splits the SWA UniformType bucket into
    several managers (typically i=2 and i=3) to align layer-tuples. Those
    managers share one PD SWA id region. Compressor state caches also use
    SlidingWindowMLASpec but with block_size 8/32 and must not match.
    """
    specs = _layer_kv_specs(manager.kv_cache_spec)
    return bool(specs) and all(
        isinstance(s, SlidingWindowMLASpec) and int(s.block_size) == _SWA_KV_BLOCK_SIZE for s in specs
    )


def _is_c4_kv_manager(manager: SingleTypeKVCacheManager) -> bool:
    """True for DeepSeek V4 C4 MLA group (compress KV + indexer, ratio==4)."""
    specs = _layer_kv_specs(manager.kv_cache_spec)
    return bool(specs) and all(
        isinstance(s, MLAAttentionSpec) and int(getattr(s, "compress_ratio", 1)) == _C4_COMPRESS_RATIO for s in specs
    )


def _share_req_blocks(
    dst: SingleTypeKVCacheManager,
    src: SingleTypeKVCacheManager,
    request_id: str,
) -> list[KVCacheBlock]:
    """Reuse ``src``'s physical blocks on ``dst`` for this request.

    Extra ``ref_cnt`` so two manager frees return the ids to the pool once.
    """
    src_blocks = src.req_to_blocks[request_id]
    dst_blocks = dst.req_to_blocks[request_id]
    new = src_blocks[len(dst_blocks) :]
    for block in new:
        block.ref_cnt += 1
    dst_blocks.extend(new)
    num_cached = getattr(src, "num_cached_block", None)
    if num_cached is not None and request_id in num_cached:
        dst.num_cached_block[request_id] = num_cached[request_id]
    return new


def _dump_single_type_kv_caches(
    kv_cache_groups,
    managers: tuple[SingleTypeKVCacheManager, ...],
) -> None:
    """Print every KV cache group / SingleType manager at coordinator init."""
    print(f"[PDKV] n_groups={len(kv_cache_groups)} n_managers={len(managers)}", flush=True)
    for i, (group, manager) in enumerate(zip(kv_cache_groups, managers)):
        spec = manager.kv_cache_spec
        inners = _layer_kv_specs(spec)
        inner0 = inners[0] if inners else None
        names = list(group.layer_names)
        print(
            f"[PDKV] i={i} manager={type(manager).__name__} "
            f"spec={type(spec).__name__} n_layers={len(names)} "
            f"spec_bs={getattr(spec, 'block_size', None)} "
            f"inner={type(inner0).__name__ if inner0 else None} "
            f"inner_bs={getattr(inner0, 'block_size', None)} "
            f"compress_ratio={getattr(inner0, 'compress_ratio', None)} "
            f"sliding_window={getattr(inner0, 'sliding_window', None)} "
            f"is_swa={_is_swa_kv_manager(manager)} is_c4={_is_c4_kv_manager(manager)} "
            f"eagle={getattr(group, 'is_eagle_group', None)} "
            f"layers={names[:4]}{'...' if len(names) > 4 else ''}",
            flush=True,
        )


def _set_pd_alloc_region(pool: PDBlockPool, manager: SingleTypeKVCacheManager, use_prefill_regions: bool) -> None:
    """Route one manager alloc to SWA / C4 / other-prefill region (prefill only)."""
    if not use_prefill_regions:
        pool.clear_alloc_is_swa()
        pool.clear_alloc_is_c4()
        return
    if _is_swa_kv_manager(manager):
        pool.set_alloc_is_swa(True)
        return
    if _is_c4_kv_manager(manager):
        pool.set_alloc_is_c4(True)
        return
    pool.clear_alloc_is_swa()
    pool.clear_alloc_is_c4()


def _build_block_pool(
    num_blocks: int,
    enable_caching: bool,
    hash_block_size: int,
    enable_kv_cache_events: bool,
    metrics_collector: KVCacheMetricsCollector | None,
) -> BlockPool:
    """Construct BlockPool or PD-partitioned BlockPool (env-gated)."""
    enabled = envs.VLLM_ASCEND_ENABLE_PD_BLOCK_POOL
    if not enabled:
        return BlockPool(
            num_blocks,
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        )
    if enable_caching:
        raise ValueError(
            "VLLM_ASCEND_ENABLE_PD_BLOCK_POOL=1 requires prefix caching disabled "
            "(enable_prefix_caching / enable_caching must be False)"
        )
    pd_config = PDBlockPoolConfig(
        num_gpu_blocks=num_blocks,
        num_decode_blocks=(
            envs.VLLM_ASCEND_PD_NUM_DECODE_BLOCKS
            if envs.VLLM_ASCEND_PD_NUM_DECODE_BLOCKS is not None
            else DECODE_KV_BLOCK_REGION_SIZE
        ),
    )
    return PDBlockPool(
        num_gpu_blocks=num_blocks,
        enable_caching=False,
        hash_block_size=hash_block_size,
        enable_kv_cache_events=enable_kv_cache_events,
        metrics_collector=metrics_collector,
        pd_config=pd_config,
    )


class AscendHybridKVCacheCoordinator(HybridKVCacheCoordinator):
    """
    KV cache coordinator for hybrid models with multiple KV cache types, and
    thus multiple kv cache groups.
    To simplify `find_longest_cache_hit`, it only supports the combination of
    two types of KV cache groups, and one of them must be full attention.
    May extend to more general cases in the future.
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
        eagle_attn_layer_names: list[str] | None = None,
        metrics_collector: KVCacheMetricsCollector | None = None,
        max_num_batched_tokens: int | None = None,
    ):
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        # Fall back to `max_model_len` when unset so the recycling-aware
        # admission cap (vLLM PR #40946) collapses to the prior uncapped
        # behavior. The scheduler always supplies the real value at runtime.
        if max_num_batched_tokens is None:
            max_num_batched_tokens = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens

        self.block_pool = _build_block_pool(
            kv_cache_config.num_blocks,
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        )

        # KV cache group indices that get the EAGLE last-block drop.
        self.eagle_group_ids: set[int] = {i for i, g in enumerate(kv_cache_config.kv_cache_groups) if g.is_eagle_group}
        # Conservatively fall back to flag all groups when no group is flagged.
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))

        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pool,
                enable_caching=enable_caching,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )
        _dump_single_type_kv_caches(self.kv_cache_config.kv_cache_groups, self.single_type_managers)

        # hash_block_size: the block size used to compute block hashes.
        # The actual block size usually equals hash_block_size, but in cases where
        # different KV cache groups have different block sizes, the actual block size
        # can be a multiple of hash_block_size.
        self.hash_block_size = hash_block_size
        if enable_caching:
            assert all(g.kv_cache_spec.block_size % hash_block_size == 0 for g in kv_cache_config.kv_cache_groups), (
                "block_size must be divisible by hash_block_size"
            )
        assert dcp_world_size == 1, "DCP not support hybrid attn now."
        assert pcp_world_size == 1, "PCP not support hybrid attn now."
        self.verify_and_split_kv_cache_groups()

        self.use_eagle = use_eagle

    @staticmethod
    def _manager_blocks_to_allocate(
        manager: SingleTypeKVCacheManager,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool,
    ) -> int:
        """Call manager.get_num_blocks_to_allocate across vLLM signature variants."""
        try:
            return manager.get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks,
                total_computed_tokens,
                num_local_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
            )
        except TypeError:
            # Older Ascend CompressAttentionManager omits num_local_computed_tokens.
            return manager.get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks,
                total_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
            )

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_local_computed_tokens: int = 0,
        num_tokens_main_model: int | None = None,
        apply_admission_cap: bool = False,
    ) -> int:
        """Prefill: other-prefill need. Decode: total vs decode region.

        SWA managers from the DSv4 layer-tuple split share one PD SWA region,
        so SWA need is ``max`` across those managers, not a sum.
        """
        if num_tokens_main_model is None:
            num_tokens_main_model = num_tokens
        swa_need = 0
        c4_need = 0
        other_need = 0
        for i, manager in enumerate(self.single_type_managers):
            if CrossAttentionManager is not None and isinstance(manager, CrossAttentionManager):
                n = self._manager_blocks_to_allocate(
                    manager,
                    request_id,
                    num_encoder_tokens,
                    [],
                    0,
                    0,
                    num_encoder_tokens,
                    apply_admission_cap,
                )
            else:
                n = self._manager_blocks_to_allocate(
                    manager,
                    request_id,
                    num_tokens,
                    new_computed_blocks[i],
                    total_computed_tokens,
                    num_local_computed_tokens,
                    num_tokens_main_model,
                    apply_admission_cap,
                )
            if _is_swa_kv_manager(manager):
                # Manager 2/3 (etc.) are SWA splits of the same 128 bucket.
                swa_need = max(swa_need, n)
                print(
                    f"[PDAdmit] SWA need req={request_id} manager={i} n={n} "
                    f"swa_need={swa_need} num_tokens={num_tokens} "
                    f"is_prefill={getattr(self.block_pool, '_alloc_is_prefill', None)}",
                    flush=True,
                )
            elif _is_c4_kv_manager(manager):
                c4_need += n
                c4_region = getattr(self.block_pool, "c4", None)
                print(
                    f"[PDAdmit] C4 need req={request_id} manager={i} n={n} "
                    f"c4_need={c4_need} "
                    f"c4_free={None if c4_region is None else c4_region.num_free} "
                    f"num_tokens={num_tokens} "
                    f"is_prefill={getattr(self.block_pool, '_alloc_is_prefill', None)}",
                    flush=True,
                )
            else:
                other_need += n

        pool = self.block_pool
        if isinstance(pool, PDBlockPool):
            if pool._alloc_is_prefill is False:
                return swa_need + c4_need + other_need
            if swa_need > pool.swa.num_free or c4_need > pool.c4.num_free:
                print(
                    f"[PDAdmit] reject req={request_id} "
                    f"swa_need={swa_need} swa_free={pool.swa.num_free} "
                    f"c4_need={c4_need} c4_free={pool.c4.num_free} "
                    f"{pool.c4.debug_status()} busy={pool.c4.busy_detail()} "
                    f"other_need={other_need} other_free={pool.prefill.num_free} "
                    f"return={pool.prefill.num_free + 1}",
                    flush=True,
                )
                # Fail ``need <= get_num_free_blocks()``; probe returns other-prefill free.
                return pool.prefill.num_free + 1
            return other_need
        return swa_need + c4_need + other_need

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        pool = self.block_pool
        if not isinstance(pool, PDBlockPool):
            return super().allocate_new_computed_blocks(
                request_id,
                new_computed_blocks,
                num_local_computed_tokens,
                num_external_computed_tokens,
            )

        if any(request_id in manager.num_cached_block for manager in self.single_type_managers):
            assert all(len(blocks) == 0 for blocks in new_computed_blocks)
            return

        # Prefill-only size-class regions; decode ignores SWA/C4 flags and uses decode region.
        use_prefill_regions = pool._alloc_is_prefill is True
        swa_src: SingleTypeKVCacheManager | None = None
        try:
            managers = self.single_type_managers
            # Prefer upstream two-phase API when present.
            if hasattr(managers[0], "add_local_computed_blocks"):
                for i, manager in enumerate(managers):
                    if _is_swa_kv_manager(manager) and swa_src is not None:
                        _share_req_blocks(manager, swa_src, request_id)
                        continue
                    _set_pd_alloc_region(pool, manager, use_prefill_regions)
                    manager.add_local_computed_blocks(
                        request_id,
                        new_computed_blocks[i],
                        num_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if _is_swa_kv_manager(manager):
                        swa_src = manager
                if num_external_computed_tokens > 0:
                    for manager in managers:
                        if _is_swa_kv_manager(manager) and swa_src is not None and manager is not swa_src:
                            _share_req_blocks(manager, swa_src, request_id)
                            continue
                        _set_pd_alloc_region(pool, manager, use_prefill_regions)
                        manager.allocate_external_computed_blocks(
                            request_id,
                            num_local_computed_tokens,
                            num_external_computed_tokens,
                        )
            else:
                for i, manager in enumerate(managers):
                    if _is_swa_kv_manager(manager) and swa_src is not None:
                        _share_req_blocks(manager, swa_src, request_id)
                        continue
                    _set_pd_alloc_region(pool, manager, use_prefill_regions)
                    manager.allocate_new_computed_blocks(
                        request_id,
                        new_computed_blocks[i],
                        num_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if _is_swa_kv_manager(manager):
                        swa_src = manager
        finally:
            pool.clear_alloc_is_swa()
            pool.clear_alloc_is_c4()

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
        num_encoder_tokens: int = 0,
    ) -> tuple[list[KVCacheBlock], ...]:
        pool = self.block_pool
        if not isinstance(pool, PDBlockPool):
            return super().allocate_new_blocks(
                request_id,
                num_tokens,
                num_tokens_main_model,
                num_encoder_tokens,
            )

        # Prefill-only size-class regions; decode ignores SWA/C4 flags and uses decode region.
        use_prefill_regions = pool._alloc_is_prefill is True
        results: list[list[KVCacheBlock]] = []
        swa_src: SingleTypeKVCacheManager | None = None
        try:
            for manager in self.single_type_managers:
                if _is_swa_kv_manager(manager) and swa_src is not None:
                    results.append(_share_req_blocks(manager, swa_src, request_id))
                    continue
                _set_pd_alloc_region(pool, manager, use_prefill_regions)
                tokens = (
                    num_encoder_tokens
                    if CrossAttentionManager is not None and isinstance(manager, CrossAttentionManager)
                    else num_tokens
                )
                results.append(manager.allocate_new_blocks(request_id, tokens, num_tokens_main_model))
                if _is_swa_kv_manager(manager):
                    swa_src = manager
        finally:
            pool.clear_alloc_is_swa()
            pool.clear_alloc_is_c4()
        return tuple(results)

    def verify_and_split_kv_cache_groups(self) -> None:
        """
        Groups KV cache groups by their spec type for efficient batch processing
        during cache hit lookup.
        """
        attention_groups: list[tuple[KVCacheSpec, list[int], type[SingleTypeKVCacheManager]]] = []

        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            manager_cls = self.single_type_managers[i].__class__
            spec = g.kv_cache_spec

            # Try to find an existing group with the same spec
            for existing_spec, group_ids, existing_cls in attention_groups:
                if existing_spec == spec:
                    assert manager_cls is existing_cls, "Expected same manager class for identical KV cache specs."
                    group_ids.append(i)
                    break
            else:
                attention_groups.append((spec, [i], manager_cls))

        assert len(attention_groups) > 1, "HybridKVCacheCoordinator requires at least two attention groups."

        # Put full attention first: its efficient left-to-right scan provides
        # a tighter initial bound, reducing work for subsequent groups.
        self.attention_groups = sorted(
            attention_groups,
            key=lambda x: not isinstance(x[0], FullAttentionSpec),
        )

        # Attention-group indices (into ``self.attention_groups``) that
        # contain at least one EAGLE/MTP KV cache group.
        self.eagle_attn_group_indices: set[int] = {
            i
            for i, (_, group_ids, _) in enumerate(self.attention_groups)
            if any(gid in self.eagle_group_ids for gid in group_ids)
        }

        # The LCM of the block sizes of all attention types.
        # The cache hit length must be a multiple of the LCM of the block sizes
        # to make sure the cache hit length is a multiple of the block size of
        # each attention type. Requiring this because we don't support partial
        # block cache hit yet.
        # NOTE: use 16k as the alignment tokens for model with compress ratio
        block_sizes = [spec.block_size * getattr(spec, "compress_ratio", 1) for spec, _, _ in self.attention_groups]
        self.lcm_block_size = lcm(*block_sizes)

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
            return BlockHashListWithBlockSize(block_hashes, self.hash_block_size, kv_cache_spec.block_size)

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups

        # Simple hybrid (1 full attn + 1 other): one iteration suffices.
        # Full attn is always first if it exists.
        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0][0], FullAttentionSpec
        )

        # Attention-group indices whose EAGLE drop is verified at the current
        # ``curr_hit_length``. Each eagle group applies the drop at most once
        # per candidate length (see issue #32802).
        eagle_verified: set[int] = set()

        while True:
            curr_hit_length = hit_length

            for idx, (spec, group_ids, manager_cls) in enumerate(self.attention_groups):
                cached_blocks = hit_blocks_by_group[group_ids[0]]
                if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                    # Full attention is downward-closed: we only need to look
                    # up cached blocks once; on subsequent iterations just trim
                    # to the (reduced) current hit length.
                    curr_hit_length = curr_hit_length // spec.block_size * spec.block_size
                    continue

                use_eagle = idx in self.eagle_attn_group_indices and idx not in eagle_verified

                _max_length = curr_hit_length
                if use_eagle:
                    # Eagle needs to match one more block and then pop the last.
                    _max_length = min(curr_hit_length + spec.block_size, max_cache_hit_length)
                hit_blocks = manager_cls.find_longest_cache_hit(
                    block_hashes=_get_block_hashes(spec),
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
                    use_eagle=use_eagle,
                    alignment_tokens=self.lcm_block_size,
                )
                _new_hit_length = len(hit_blocks[0]) * spec.block_size
                if use_eagle:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                compress_ratio = getattr(spec, "compress_ratio", 1)
                curr_hit_length = len(hit_blocks[0]) * spec.block_size * max(compress_ratio, 1)
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        # Truncate full attention blocks to final hit_length (if present)
        spec, group_ids, _ = self.attention_groups[0]
        if isinstance(spec, FullAttentionSpec):
            num_blocks = hit_length // spec.block_size
            for group_id in group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]

        return tuple(blocks if blocks is not None else [] for blocks in hit_blocks_by_group), hit_length


def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    max_num_batched_tokens: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    hash_block_size: int,
    eagle_attn_layer_names: list[str] | None = None,
    metrics_collector: KVCacheMetricsCollector | None = None,
) -> KVCacheCoordinator:
    return AscendHybridKVCacheCoordinator(
        kv_cache_config,
        max_model_len,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        hash_block_size=hash_block_size,
        eagle_attn_layer_names=eagle_attn_layer_names,
        metrics_collector=metrics_collector,
        max_num_batched_tokens=max_num_batched_tokens,
    )


vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator = get_kv_cache_coordinator  # type: ignore[attr-defined]

# `kv_cache_manager` imports `get_kv_cache_coordinator` with
# `from ... import ...`, so if it was loaded before this patch runs
# (for example through the recompute scheduler path), it keeps the
# old function object. Update that cached binding as well.
_kv_cache_manager = sys.modules.get("vllm.v1.core.kv_cache_manager")
if _kv_cache_manager is not None:
    _kv_cache_manager.get_kv_cache_coordinator = get_kv_cache_coordinator  # type: ignore[attr-defined]


# Mark alloc region (prefill vs decode) on PDBlockPool for the duration of
# allocate_slots so get_num_free_blocks / get_new_blocks use the right region.
_original_allocate_slots = KVCacheManager.allocate_slots


def _allocate_slots_with_pd_region(self: KVCacheManager, request, *args, **kwargs):
    pool = self.block_pool
    if not isinstance(pool, PDBlockPool):
        return _original_allocate_slots(self, request, *args, **kwargs)
    # Still in prompt → prefill region; otherwise decode region.
    # Prefill-allocated blocks stay with the request through decode and are
    # freed by id-range back to the prefill region when the request finishes.
    # Prefill SWA → SWA region; decode (all groups) → decode region.
    is_prefill = request.num_computed_tokens < request.num_prompt_tokens
    pool.set_alloc_is_prefill(is_prefill)
    try:
        return _original_allocate_slots(self, request, *args, **kwargs)
    finally:
        pool.clear_alloc_is_swa()
        pool.clear_alloc_is_c4()
        pool.clear_alloc_is_prefill()


KVCacheManager.allocate_slots = _allocate_slots_with_pd_region  # type: ignore[method-assign]