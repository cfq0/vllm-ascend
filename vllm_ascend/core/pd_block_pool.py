# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill/Decode partitioned block pool for Ascend.

Splits the physical block-id space so decode single-block pops cannot fragment
prefill contiguous runs (important for DSA-CP SWA ``copy_`` / C4 scatter locality).

Layout (null block at 0)::

    [1, 1+SWA)           SWA size-class slabs (**prefill only**, contiguous)
    [1+SWA, 1+SWA+C4)    C4 + indexer MLA size-class slabs (**prefill only**, contiguous)
    [1+SWA+C4, P)        other prefill free-list (C128 + compressor state)
    [P, N)               decode free-list (all groups, including SWA/C4 in decode)

SWA and C4 are carved into fixed **size-class slabs** (class-64 only)::

    SWA  64 x 16  (= 1024 ids)
    C4   64 x 32  (= 2048 ids)

Chunked prefill + SWA sliding window can free the previous chunk's ids, so
SWA needs fewer concurrent slabs than C4 (full compressed KV until the
request ends). Allocation picks the smallest free slab whose class is
``>= n`` and takes ``n`` ids from that slab's start. A slab is occupied as
a whole (leftover ids stay reserved) until every given-out id is freed;
never searches inside a slab or stitches across slabs. If every slab of
that class is busy, the request escalates to the next larger class. Other
prefill and decode use a free-list; ids need not be contiguous.

Routing:
- Prefill + real SWA → SWA region.
- Prefill + C4 MLA group (compress_ratio==4, includes indexer) → C4 region.
- Prefill + other → remaining prefill region.
- Decode (any group) → decode free-list (avoids punching holes in SWA/C4).

Scope (current):
- Prefix cache / PCP / MTP are not supported; enable only with caching off.
- Decode: free-list sized small by default (64 blocks); leftover after
  SWA+C4+decode goes to other-prefill (C128 / compressor state, ...).
- Free routes by block-id range back to the owning region.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock

logger = init_logger(__name__)

# Generic mixed template for the planner / region unit tests (escalation).
# Token span at block_size=128: 8k / 16k / 32k.
PREFILL_SIZE_CLASS_SLABS: tuple[tuple[int, int], ...] = (
    (64, 8),
    (128, 4),
    (256, 2),
)
PREFILL_SIZE_CLASSES = tuple(cls for cls, _ in PREFILL_SIZE_CLASS_SLABS)

# Production PD regions: class-64 only. One ~8k chunk occupies one slab.
SWA_SIZE_CLASS_SLABS: tuple[tuple[int, int], ...] = ((64, 16),)
C4_SIZE_CLASS_SLABS: tuple[tuple[int, int], ...] = ((64, 32),)


def size_class_region_blocks(
    class_slabs: tuple[tuple[int, int], ...] = PREFILL_SIZE_CLASS_SLABS,
) -> int:
    return sum(cls * count for cls, count in class_slabs)


# SWA / C4 regions are exactly one full size-class template.
SWA_KV_BLOCK_REGION_SIZE = size_class_region_blocks(SWA_SIZE_CLASS_SLABS)
C4_KV_BLOCK_REGION_SIZE = size_class_region_blocks(C4_SIZE_CLASS_SLABS)
# Decode free-list size; leftover after SWA+C4+decode goes to other-prefill.
DECODE_KV_BLOCK_REGION_SIZE = 64
# Absolute floor for "other prefill" (C128 / compressor state / ...).
_MIN_OTHER_PREFILL_BLOCKS = 1


@dataclass(frozen=True)
class PDBlockPoolConfig:
    """Partition layout for PD block ids.

    Block 0 is reserved as the null block (same convention as vLLM BlockPool).
    Remaining ids are split into SWA, C4, other prefill, then decode.
    """

    num_gpu_blocks: int
    # Decode region size in blocks. Default is small; other-prefill takes the
    # leftover after SWA+C4+decode so long prefills (state / C128) have room.
    num_decode_blocks: int = DECODE_KV_BLOCK_REGION_SIZE
    # Keep a null block at id 0.
    reserve_null_block: bool = True
    # Fixed SWA id region size (real SWA block_size=128).
    num_swa_blocks: int = SWA_KV_BLOCK_REGION_SIZE
    # Fixed C4 (+ indexer) id region size.
    num_c4_blocks: int = C4_KV_BLOCK_REGION_SIZE

    def resolve(self) -> tuple[int, int, int, int, int, int]:
        """Return ``(null_id, swa_start, swa_end, c4_end, prefill_end, decode_end)``.

        SWA owns ``[swa_start, swa_end)``.
        C4 owns ``[swa_end, c4_end)``.
        Other prefill owns ``[c4_end, prefill_end)``.
        Decode owns ``[prefill_end, decode_end)``.

        SWA/C4/decode keep their configured sizes; all leftover after those
        three goes to other-prefill. Raise if leftover cannot cover
        other-prefill.
        """
        if self.num_gpu_blocks < 2:
            raise ValueError("num_gpu_blocks must be >= 2")
        null_id = 0 if self.reserve_null_block else -1
        usable_start = 1 if self.reserve_null_block else 0
        usable = self.num_gpu_blocks - usable_start

        num_swa = int(self.num_swa_blocks)
        num_c4 = int(self.num_c4_blocks)
        num_decode = int(self.num_decode_blocks)
        if num_swa <= 0:
            raise ValueError("SWA region must be > 0")
        if num_c4 <= 0:
            raise ValueError("C4 region must be > 0")
        if num_decode <= 0:
            raise ValueError("decode region must be > 0")

        reserved = num_swa + num_c4 + num_decode
        if reserved + _MIN_OTHER_PREFILL_BLOCKS > usable:
            raise ValueError(
                f"SWA+C4+decode ({num_swa}+{num_c4}+{num_decode}) leave no "
                f"other-prefill blocks (usable={usable}, "
                f"num_gpu_blocks={self.num_gpu_blocks})"
            )

        other_prefill = usable - reserved

        swa_start = usable_start
        swa_end = usable_start + num_swa
        c4_end = swa_end + num_c4
        prefill_end = c4_end + other_prefill
        decode_end = self.num_gpu_blocks
        return null_id, swa_start, swa_end, c4_end, prefill_end, decode_end


def plan_size_class_slabs(
    start: int,
    capacity: int,
    class_slabs: tuple[tuple[int, int], ...] = PREFILL_SIZE_CLASS_SLABS,
) -> list[tuple[int, int, int]]:
    """Carve ``[start, start+capacity)`` into size-class slabs.

    Returns ``[(slab_start, slab_end, class_size), ...]``. Template copies are
    placed first (``class_slabs``). Any leftover capacity becomes one overflow
    slab whose class_size equals the leftover length.
    """
    if capacity <= 0:
        return []
    slabs: list[tuple[int, int, int]] = []
    cursor = start
    remaining = capacity
    stop_template = False
    for cls, count in class_slabs:
        if stop_template:
            break
        for _ in range(count):
            if remaining < cls:
                stop_template = True
                break
            slabs.append((cursor, cursor + cls, cls))
            cursor += cls
            remaining -= cls
    if remaining > 0:
        slabs.append((cursor, cursor + remaining, remaining))
    return slabs


class _SizeClassSlab:
    """One contiguous id span of a fixed class size.

    Occupied as a whole: ``try_allocate(n)`` takes ``[start, start+n)`` from a
    free slab and keeps leftover ids reserved until every given-out id is freed.
    """

    def __init__(self, start: int, end: int, class_size: int) -> None:
        if end <= start:
            raise ValueError(f"empty slab [{start}, {end})")
        self.start = start
        self.end = end
        self.class_size = class_size
        self.capacity = end - start
        self._busy = False
        self._live: set[int] = set()

    @property
    def num_free(self) -> int:
        return 0 if self._busy else self.capacity

    def contains(self, block_id: int) -> bool:
        return self.start <= block_id < self.end

    def try_allocate(self, n: int) -> list[int] | None:
        """Return ``n`` ids from this slab's start, or None if the slab is busy."""
        if n <= 0 or n > self.capacity or self._busy:
            return None
        ids = list(range(self.start, self.start + n))
        self._busy = True
        self._live = set(ids)
        return ids

    def free(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            if not self.contains(bid):
                raise ValueError(f"block {bid} not in slab [{self.start}, {self.end})")
            self._live.discard(bid)
        if not self._live:
            self._busy = False


class _PrefillSizeClassRegion:
    """Prefill region split into size-class slabs (64×8 / 128×4 / 256×2).

    ``allocate_contiguous(n)`` always returns one contiguous id run from a
    single free slab (ids from that slab's start). It never searches inside
    a slab or stitches fragments across slabs.
    """

    def __init__(
        self,
        start: int,
        end: int,
        name: str = "prefill",
        class_slabs: tuple[tuple[int, int], ...] = PREFILL_SIZE_CLASS_SLABS,
    ) -> None:
        if end <= start:
            raise ValueError(f"empty {name} region [{start}, {end})")
        self.name = name
        self.start = start
        self.end = end
        self.capacity = end - start
        self.class_slabs = class_slabs
        planned = plan_size_class_slabs(start, self.capacity, class_slabs)
        self.slabs = [_SizeClassSlab(s, e, cls) for s, e, cls in planned]
        self._slabs_by_class: dict[int, list[_SizeClassSlab]] = {}
        for slab in self.slabs:
            self._slabs_by_class.setdefault(slab.class_size, []).append(slab)
        # Unique class sizes in escalating order (template classes first, overflow last).
        seen: list[int] = []
        for cls, _count in class_slabs:
            if cls in self._slabs_by_class and cls not in seen:
                seen.append(cls)
        for cls in sorted(self._slabs_by_class):
            if cls not in seen:
                seen.append(cls)
        self._class_order = seen
        logger.info(
            "PD %s size-class slabs: %s",
            name,
            ", ".join(f"{s.class_size}@[{s.start},{s.end})" for s in self.slabs),
        )

    @property
    def num_free(self) -> int:
        return sum(slab.num_free for slab in self.slabs)

    def allocate_contiguous(self, n: int) -> list[int]:
        """Allocate ``n`` contiguous ids from one free size-class slab.

        Picks the smallest class ``>= n``, then the first free slab of that
        class (or a larger class if all smaller matching slabs are busy).
        """
        if n <= 0:
            return []
        if n > self.num_free:
            raise ValueError(f"{self.name} region: need {n} free blocks, only {self.num_free} left")

        for cls in self._class_order:
            if cls < n:
                continue
            for slab in self._slabs_by_class[cls]:
                got = slab.try_allocate(n)
                if got is not None:
                    return got

        raise ValueError(
            f"{self.name} region: need {n} contiguous blocks but no size-class "
            f"slab can fit them (classes={self._class_order}, free={self.num_free})"
        )

    def free(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            self._slab_for_id(bid).free([bid])

    def _slab_for_id(self, block_id: int) -> _SizeClassSlab:
        for slab in self.slabs:
            if slab.contains(block_id):
                return slab
        raise ValueError(f"block {block_id} not in {self.name} region [{self.start}, {self.end})")


class _FreeListRegion:
    """Simple free-list; fragmentation is acceptable (other-prefill / decode)."""

    def __init__(self, start: int, end: int, name: str = "free-list") -> None:
        if end <= start:
            raise ValueError(f"empty {name} region [{start}, {end})")
        self.name = name
        self.start = start
        self.end = end
        self.capacity = end - start
        self._free: deque[int] = deque(range(start, end))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self, n: int) -> list[int]:
        if n <= 0:
            return []
        if n > len(self._free):
            raise ValueError(f"{self.name} region: need {n} free blocks, only {len(self._free)} left")
        return [self._free.popleft() for _ in range(n)]

    def free(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            if not (self.start <= bid < self.end):
                raise ValueError(f"block {bid} not in {self.name} region [{self.start}, {self.end})")
            self._free.append(bid)


class PDBlockPool(BlockPool):
    """BlockPool with SWA / C4 / other-prefill / decode partitioned regions.

    Call :meth:`set_alloc_is_prefill` / :meth:`set_alloc_is_swa` /
    :meth:`set_alloc_is_c4` before ``get_new_blocks`` / ``get_num_free_blocks``
    so admission and allocation use the correct region. Free always routes by
    block-id range. SWA/C4 use size-class slabs; other prefill is a free-list.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
        pd_config: PDBlockPoolConfig | None = None,
    ) -> None:
        if enable_caching:
            raise ValueError(
                "PDBlockPool does not support prefix caching yet; "
                "disable enable_prefix_caching / enable_caching first"
            )
        super().__init__(
            num_gpu_blocks,
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        )
        if pd_config is None:
            pd_config = PDBlockPoolConfig(num_gpu_blocks=num_gpu_blocks)
        elif pd_config.num_gpu_blocks != num_gpu_blocks:
            raise ValueError(
                f"pd_config.num_gpu_blocks ({pd_config.num_gpu_blocks}) != "
                f"num_gpu_blocks ({num_gpu_blocks})"
            )
        self.pd_config = pd_config
        null_id, swa_start, swa_end, c4_end, prefill_end, decode_end = pd_config.resolve()
        if null_id != 0 or self.null_block.block_id != 0:
            raise ValueError("PDBlockPool expects null block id 0")

        # Drain BlockPool free-list; ownership moves to PD regions.
        n_free = self.free_block_queue.num_free_blocks
        if n_free > 0:
            self.free_block_queue.popleft_n(n_free)
        assert self.free_block_queue.num_free_blocks == 0

        self.swa = _PrefillSizeClassRegion(
            swa_start, swa_end, name="swa", class_slabs=SWA_SIZE_CLASS_SLABS
        )
        self.c4 = _PrefillSizeClassRegion(
            swa_end, c4_end, name="c4", class_slabs=C4_SIZE_CLASS_SLABS
        )
        self.prefill = _FreeListRegion(c4_end, prefill_end, name="prefill")
        self.decode = _FreeListRegion(prefill_end, decode_end, name="decode")
        # None = unset (treat as total free for get_num_free_blocks).
        self._alloc_is_prefill: bool | None = None
        self._alloc_is_swa: bool | None = None
        self._alloc_is_c4: bool | None = None

        logger.info(
            "PDBlockPool enabled: swa=[%d,%d) (%d blocks), "
            "c4=[%d,%d) (%d blocks), prefill=[%d,%d) (%d blocks), "
            "decode=[%d,%d) (%d blocks)",
            swa_start,
            swa_end,
            swa_end - swa_start,
            swa_end,
            c4_end,
            c4_end - swa_end,
            c4_end,
            prefill_end,
            prefill_end - c4_end,
            prefill_end,
            decode_end,
            decode_end - prefill_end,
        )

    @property
    def swa_range(self) -> tuple[int, int]:
        return self.swa.start, self.swa.end

    @property
    def c4_range(self) -> tuple[int, int]:
        return self.c4.start, self.c4.end

    @property
    def prefill_range(self) -> tuple[int, int]:
        return self.prefill.start, self.prefill.end

    @property
    def decode_range(self) -> tuple[int, int]:
        return self.decode.start, self.decode.end

    def set_alloc_is_prefill(self, is_prefill: bool) -> None:
        self._alloc_is_prefill = is_prefill

    def clear_alloc_is_prefill(self) -> None:
        self._alloc_is_prefill = None

    def set_alloc_is_swa(self, is_swa: bool) -> None:
        self._alloc_is_swa = is_swa
        if is_swa:
            self._alloc_is_c4 = False

    def clear_alloc_is_swa(self) -> None:
        self._alloc_is_swa = None

    def set_alloc_is_c4(self, is_c4: bool) -> None:
        self._alloc_is_c4 = is_c4
        if is_c4:
            self._alloc_is_swa = False

    def clear_alloc_is_c4(self) -> None:
        self._alloc_is_c4 = None

    def get_num_free_blocks(self) -> int:
        if self._alloc_is_prefill is False:
            return self.decode.num_free
        if self._alloc_is_swa is True:
            return self.swa.num_free
        if self._alloc_is_c4 is True:
            return self.c4.num_free
        if self._alloc_is_prefill is True:
            return self.prefill.num_free
        return self.swa.num_free + self.c4.num_free + self.prefill.num_free + self.decode.num_free

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        if num_blocks <= 0:
            return []

        is_prefill = self._alloc_is_prefill
        if is_prefill is None:
            # Fallback heuristic if caller forgot to set the flag.
            is_prefill = num_blocks > 1
            logger.warning_once(
                "PDBlockPool.get_new_blocks called without set_alloc_is_prefill; "
                "inferring is_prefill=%s from num_blocks=%d",
                is_prefill,
                num_blocks,
            )

        # Decode phase: all groups use the decode free-list so incremental
        # decode pops do not punch holes in SWA/C4 size-class slabs.
        if not is_prefill:
            region_free = self.decode.num_free
            if num_blocks > region_free:
                raise ValueError(
                    f"Cannot get {num_blocks} free blocks from the decode PD region "
                    f"(free={region_free})"
                )
            ids = self.decode.allocate(num_blocks)
        elif self._alloc_is_swa is True:
            region_free = self.swa.num_free
            if num_blocks > region_free:
                raise ValueError(
                    f"Cannot get {num_blocks} free blocks from the swa PD region "
                    f"(free={region_free})"
                )
            ids = self.swa.allocate_contiguous(num_blocks)
        elif self._alloc_is_c4 is True:
            region_free = self.c4.num_free
            if num_blocks > region_free:
                raise ValueError(
                    f"Cannot get {num_blocks} free blocks from the c4 PD region "
                    f"(free={region_free})"
                )
            ids = self.c4.allocate_contiguous(num_blocks)
        else:
            region_free = self.prefill.num_free
            if num_blocks > region_free:
                raise ValueError(
                    f"Cannot get {num_blocks} free blocks from the prefill PD region "
                    f"(free={region_free})"
                )
            ids = self.prefill.allocate(num_blocks)

        ret = [self.blocks[bid] for bid in ids]
        for block in ret:
            assert block.ref_cnt == 0
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_allocated(block)
        return ret

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        blocks_list = list(ordered_blocks)
        to_free: list[KVCacheBlock] = []
        for block in blocks_list:
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                to_free.append(block)

        swa_ids: list[int] = []
        c4_ids: list[int] = []
        prefill_ids: list[int] = []
        decode_ids: list[int] = []
        for block in to_free:
            bid = block.block_id
            if self.swa.start <= bid < self.swa.end:
                swa_ids.append(bid)
            elif self.c4.start <= bid < self.c4.end:
                c4_ids.append(bid)
            elif self.prefill.start <= bid < self.prefill.end:
                prefill_ids.append(bid)
            elif self.decode.start <= bid < self.decode.end:
                decode_ids.append(bid)
            else:
                raise ValueError(f"block {bid} outside PD regions {self.summary()}")
        if swa_ids:
            self.swa.free(swa_ids)
        if c4_ids:
            self.c4.free(c4_ids)
        if prefill_ids:
            self.prefill.free(prefill_ids)
        if decode_ids:
            self.decode.free(decode_ids)

    def summary(self) -> str:
        ss, se = self.swa_range
        cs, ce = self.c4_range
        ps, pe = self.prefill_range
        ds, de = self.decode_range
        return (
            f"PDBlockPool(null=0, "
            f"swa=[{ss},{se}) free={self.swa.num_free}/{self.swa.capacity}, "
            f"c4=[{cs},{ce}) free={self.c4.num_free}/{self.c4.capacity}, "
            f"prefill=[{ps},{pe}) free={self.prefill.num_free}/{self.prefill.capacity}, "
            f"decode=[{ds},{de}) free={self.decode.num_free})"
        )


def demo_pd_partition() -> None:
    """Tiny smoke demo: ``python -m vllm_ascend.core.pd_block_pool``."""
    # Need room for SWA + C4 + other prefill + decode.
    cfg = PDBlockPoolConfig(
        num_gpu_blocks=8192,
        num_decode_blocks=DECODE_KV_BLOCK_REGION_SIZE,
        num_swa_blocks=SWA_KV_BLOCK_REGION_SIZE,
        num_c4_blocks=C4_KV_BLOCK_REGION_SIZE,
    )
    pool = PDBlockPool(
        num_gpu_blocks=cfg.num_gpu_blocks,
        enable_caching=False,
        hash_block_size=16,
        pd_config=cfg,
    )
    print(pool.summary())
    print("swa slabs", [(s.class_size, s.start, s.end) for s in pool.swa.slabs])

    pool.set_alloc_is_prefill(True)
    pool.set_alloc_is_swa(True)
    s0 = pool.get_new_blocks(8)
    print("swa alloc", [b.block_id for b in s0], "range", pool.swa_range)

    pool.set_alloc_is_c4(True)
    c0 = pool.get_new_blocks(8)
    print("c4 alloc", [b.block_id for b in c0], "range", pool.c4_range)

    pool.clear_alloc_is_swa()
    pool.clear_alloc_is_c4()
    p0 = pool.get_new_blocks(8)
    print("prefill alloc", [b.block_id for b in p0], "range", pool.prefill_range)

    pool.set_alloc_is_prefill(False)
    d_blocks = [pool.get_new_blocks(1)[0] for _ in range(32)]
    print("decode alloc head/tail", d_blocks[0].block_id, d_blocks[-1].block_id, pool.decode_range)

    pool.free_blocks(s0)
    pool.free_blocks(c0)
    pool.free_blocks(p0)
    pool.free_blocks(d_blocks)
    pool.clear_alloc_is_prefill()
    print("after free", pool.summary())


if __name__ == "__main__":
    demo_pd_partition()
