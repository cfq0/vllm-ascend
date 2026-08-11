# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill/Decode partitioned block pool for Ascend.

Splits the physical block-id space so decode single-block pops cannot fragment
prefill contiguous runs (important for DSA-CP SWA ``copy_`` / C4 scatter locality).

Layout (null block at 0)::

    [1, 1+SWA)           SWA bump (real SWA block_size=128, **prefill only**)
    [1+SWA, 1+SWA+C4)    C4 + indexer MLA bump (**prefill only**, contiguous)
    [1+SWA+C4, P)        other prefill bump (C128 + compressor state, ...)
    [P, N)               decode free-list (all groups, including SWA/C4 in decode)

Routing:
- Prefill + real SWA → SWA bump.
- Prefill + C4 MLA group (compress_ratio==4, includes indexer) → C4 bump.
- Prefill + other → remaining prefill bump.
- Decode (any group) → decode free-list (avoids punching holes in SWA/C4).

Scope (current):
- Prefix cache / PCP / MTP are not supported; enable only with caching off.
- Prefill / SWA / C4: bump allocator that prefers **id-contiguous** spans; on
  fragmentation, binary-searches the largest free contiguous chunk and
  repeats until the request is filled (may return several runs).
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

# Real SWA: leave headroom for other-prefill (state / C128); was 2048, take 512.
SWA_KV_BLOCK_REGION_SIZE = 1536
# C4 MLA group (compress KV + indexer): same as SWA (2048 - 512).
C4_KV_BLOCK_REGION_SIZE = 1536
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


class _PrefillBumpRegion:
    """Contiguous bump allocator over ``[start, end)`` with wrap-around search."""

    def __init__(self, start: int, end: int, name: str = "prefill") -> None:
        if end <= start:
            raise ValueError(f"empty {name} region [{start}, {end})")
        self.name = name
        self.start = start
        self.end = end
        self.capacity = end - start
        # Free flags for ids in this region (index = block_id - start).
        self._free = [True] * self.capacity
        self._num_free = self.capacity
        # Next candidate id for bump allocation.
        self._cursor = start

    @property
    def num_free(self) -> int:
        return self._num_free

    def allocate_contiguous(self, n: int) -> list[int]:
        """Allocate ``n`` free blocks, preferring long contiguous id runs.

        1. Try one contiguous span of length ``n``.
        2. If fragmented, repeatedly binary-search the largest feasible
           contiguous span ``k <= remaining`` and take it, until ``n`` is met.
           Result may be several contiguous runs (still better than random ids).
        """
        if n <= 0:
            return []
        if n > self._num_free:
            raise ValueError(f"{self.name} region: need {n} free blocks, only {self._num_free} left")

        # Fast path: one contiguous span.
        span = self._find_contiguous_span(n)
        if span is not None:
            return self._take_span(span, n)

        # Fragmented: keep taking the largest contiguous chunk via binary search.
        out: list[int] = []
        remaining = n
        while remaining > 0:
            first, length = self._find_largest_contiguous_span(remaining)
            if length <= 0 or first is None:
                raise ValueError(
                    f"{self.name} region: need {n} blocks, allocated {n - remaining}, "
                    f"free={self._num_free} but no free contiguous span left (fragmented)"
                )
            out.extend(self._take_span(first, length))
            remaining -= length

        return out

    def free(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            if not (self.start <= bid < self.end):
                raise ValueError(f"block {bid} not in {self.name} region [{self.start}, {self.end})")
            self._mark_free(bid)

    def _take_span(self, first: int, n: int) -> list[int]:
        out = list(range(first, first + n))
        for bid in out:
            self._mark_used(bid)
        self._cursor = first + n
        if self._cursor >= self.end:
            self._cursor = self.start
        return out

    def _find_contiguous_span(self, n: int) -> int | None:
        """Return start id of a free contiguous span of length n, or None."""
        if n <= 0 or n > self.capacity:
            return None
        # Linear scan with wrap from cursor. A single span must not wrap past end.
        for offset in range(self.capacity):
            first = self.start + ((self._cursor - self.start + offset) % self.capacity)
            if first + n <= self.end and self._is_free_range(first, n):
                return first
        return None

    def _find_largest_contiguous_span(self, max_n: int) -> tuple[int | None, int]:
        """Binary-search the largest ``k in [1, max_n]`` with a free contiguous span.

        Returns ``(start_id, k)`` or ``(None, 0)`` if nothing free.
        """
        max_n = min(max_n, self._num_free, self.capacity)
        if max_n <= 0:
            return None, 0

        lo, hi = 1, max_n
        best_first: int | None = None
        best_k = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            first = self._find_contiguous_span(mid)
            if first is not None:
                best_first = first
                best_k = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best_first, best_k

    def _is_free_range(self, first: int, n: int) -> bool:
        base = first - self.start
        for i in range(n):
            if not self._free[base + i]:
                return False
        return True

    def _mark_used(self, bid: int) -> None:
        idx = bid - self.start
        if not self._free[idx]:
            raise RuntimeError(f"{self.name} block {bid} already allocated")
        self._free[idx] = False
        self._num_free -= 1

    def _mark_free(self, bid: int) -> None:
        idx = bid - self.start
        if self._free[idx]:
            return
        self._free[idx] = True
        self._num_free += 1


class _DecodeFreeListRegion:
    """Simple free-list for decode; fragmentation is acceptable."""

    def __init__(self, start: int, end: int) -> None:
        if end <= start:
            raise ValueError(f"empty decode region [{start}, {end})")
        self.start = start
        self.end = end
        self._free: deque[int] = deque(range(start, end))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self, n: int) -> list[int]:
        if n <= 0:
            return []
        if n > len(self._free):
            raise ValueError(f"decode region: need {n} free blocks, only {len(self._free)} left")
        return [self._free.popleft() for _ in range(n)]

    def free(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            if not (self.start <= bid < self.end):
                raise ValueError(f"block {bid} not in decode region [{self.start}, {self.end})")
            self._free.append(bid)


class PDBlockPool(BlockPool):
    """BlockPool with SWA / C4 / other-prefill / decode partitioned regions.

    Call :meth:`set_alloc_is_prefill` / :meth:`set_alloc_is_swa` /
    :meth:`set_alloc_is_c4` before ``get_new_blocks`` / ``get_num_free_blocks``
    so admission and allocation use the correct region. Free always routes by
    block-id range.
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

        self.swa = _PrefillBumpRegion(swa_start, swa_end, name="swa")
        self.c4 = _PrefillBumpRegion(swa_end, c4_end, name="c4")
        self.prefill = _PrefillBumpRegion(c4_end, prefill_end, name="prefill")
        self.decode = _DecodeFreeListRegion(prefill_end, decode_end)
        # None = unset (treat as total free for get_num_free_blocks).
        self._alloc_is_prefill: bool | None = None
        self._alloc_is_swa: bool | None = None
        self._alloc_is_c4: bool | None = None
        # Set by coordinator.get_num_blocks_to_allocate for split admission.
        self._pending_swa_blocks: int | None = None
        self._pending_c4_blocks: int | None = None

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
        self._pending_swa_blocks = None
        self._pending_c4_blocks = None

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

    def set_pending_swa_blocks(self, num_swa_blocks: int) -> None:
        """Record SWA need for the next ``get_num_free_blocks`` admission check."""
        self._pending_swa_blocks = int(num_swa_blocks)

    def clear_pending_swa_blocks(self) -> None:
        self._pending_swa_blocks = None

    def set_pending_c4_blocks(self, num_c4_blocks: int) -> None:
        """Record C4 need for the next ``get_num_free_blocks`` admission check."""
        self._pending_c4_blocks = int(num_c4_blocks)

    def clear_pending_c4_blocks(self) -> None:
        self._pending_c4_blocks = None

    def get_num_free_blocks(self) -> int:
        # Decode: all groups share the decode free-list.
        if self._alloc_is_prefill is False:
            return self.decode.num_free

        # Prefill split admission: coordinator returns only "other prefill" need;
        # hard-fail if SWA/C4 cannot be satisfied (do not soft-fail via -1 / None).
        if self._pending_swa_blocks is not None or self._pending_c4_blocks is not None:
            if self._pending_swa_blocks is not None and self._pending_swa_blocks > self.swa.num_free:
                raise ValueError(
                    f"SWA PD region cannot satisfy request: need={self._pending_swa_blocks}, "
                    f"free={self.swa.num_free}, region=[{self.swa.start}, {self.swa.end})"
                )
            if self._pending_c4_blocks is not None and self._pending_c4_blocks > self.c4.num_free:
                raise ValueError(
                    f"C4 PD region cannot satisfy request: need={self._pending_c4_blocks}, "
                    f"free={self.c4.num_free}, region=[{self.c4.start}, {self.c4.end})"
                )
            return self.prefill.num_free

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
        # decode pops do not punch holes in SWA/C4 bump regions.
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
            ids = self.prefill.allocate_contiguous(num_blocks)

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
