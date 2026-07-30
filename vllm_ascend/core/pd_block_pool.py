# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill/Decode partitioned block pool for Ascend.

Splits the physical block-id space so decode single-block pops cannot fragment
prefill contiguous runs (important for DSA-CP SWA ``copy_`` writes).

Layout (null block at 0)::

    [1, 1+SWA)     SWA bump (real SWA block_size=128, **prefill only**)
    [1+SWA, P)     non-SWA prefill bump (C4/C128 + state caches)
    [P, N)         decode free-list (all groups, including SWA in decode)

Routing:
- Prefill + real SWA → SWA bump (keeps contiguous ids for ``copy_``).
- Prefill + other → prefill bump.
- Decode (any group) → decode free-list (avoids punching holes in SWA).

Scope (current):
- Prefix cache / PCP / MTP are not supported; enable only with caching off.
- Prefill / SWA: bump allocator that prefers **id-contiguous** spans; on
  fragmentation, binary-searches the largest free contiguous chunk and
  repeats until the request is filled (may return several runs).
- Decode: free-list sized for a small concurrent decode batch (default 32 reqs).
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

# Real SWA: 32 concurrent reqs * 8k tokens / block_size 128.
SWA_KV_BLOCK_REGION_SIZE = 2048


@dataclass(frozen=True)
class PDBlockPoolConfig:
    """Partition layout for one KV-cache group's block ids.

    Block 0 is reserved as the null block (same convention as vLLM BlockPool).
    Remaining ids are split into SWA, then non-SWA prefill, then decode.
    """

    num_gpu_blocks: int
    # Max concurrent decode requests (no MTP). Used only to size the decode
    # region when ``num_decode_blocks`` is None.
    max_num_decode_reqs: int = 32
    # Upper bound of blocks one decode request may hold (e.g. SWA window /
    # max_model_len / block_size). Required when ``num_decode_blocks`` is None.
    max_blocks_per_decode_req: int = 32
    # Explicit decode region size; overrides the two fields above when set.
    num_decode_blocks: int | None = None
    # Keep a null block at id 0.
    reserve_null_block: bool = True
    # Fixed SWA id region size (real SWA block_size=128).
    num_swa_blocks: int = SWA_KV_BLOCK_REGION_SIZE

    def resolve(self) -> tuple[int, int, int, int, int]:
        """Return ``(null_id, swa_start, swa_end, prefill_end, decode_end)``.

        SWA owns ``[swa_start, swa_end)``.
        Non-SWA prefill owns ``[swa_end, prefill_end)``.
        Decode owns ``[prefill_end, decode_end)``.
        """
        if self.num_gpu_blocks < 2:
            raise ValueError("num_gpu_blocks must be >= 2")
        null_id = 0 if self.reserve_null_block else -1
        usable_start = 1 if self.reserve_null_block else 0
        usable = self.num_gpu_blocks - usable_start

        num_swa = int(self.num_swa_blocks)
        if num_swa <= 0:
            raise ValueError("SWA region must be > 0")
        if num_swa >= usable:
            raise ValueError(
                f"SWA region ({num_swa}) leaves no blocks for prefill/decode "
                f"(usable={usable}, num_gpu_blocks={self.num_gpu_blocks})"
            )

        if self.num_decode_blocks is not None:
            num_decode = int(self.num_decode_blocks)
        else:
            num_decode = int(self.max_num_decode_reqs) * int(self.max_blocks_per_decode_req)
        if num_decode <= 0:
            raise ValueError("decode region must be > 0")

        remaining = usable - num_swa
        if num_decode >= remaining:
            raise ValueError(
                f"decode region ({num_decode}) leaves no non-SWA prefill blocks "
                f"(remaining_after_swa={remaining}, swa={num_swa}, "
                f"num_gpu_blocks={self.num_gpu_blocks})"
            )

        swa_start = usable_start
        swa_end = usable_start + num_swa
        prefill_end = swa_end + (remaining - num_decode)
        decode_end = self.num_gpu_blocks
        return null_id, swa_start, swa_end, prefill_end, decode_end


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
    """BlockPool with SWA + prefill/decode partitioned regions.

    Call :meth:`set_alloc_is_prefill` / :meth:`set_alloc_is_swa` before
    ``get_new_blocks`` / ``get_num_free_blocks`` so admission and allocation
    use the correct region. Free always routes by block-id range.
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
        null_id, swa_start, swa_end, prefill_end, decode_end = pd_config.resolve()
        if null_id != 0 or self.null_block.block_id != 0:
            raise ValueError("PDBlockPool expects null block id 0")

        # Drain BlockPool free-list; ownership moves to PD regions.
        n_free = self.free_block_queue.num_free_blocks
        if n_free > 0:
            self.free_block_queue.popleft_n(n_free)
        assert self.free_block_queue.num_free_blocks == 0

        self.swa = _PrefillBumpRegion(swa_start, swa_end, name="swa")
        self.prefill = _PrefillBumpRegion(swa_end, prefill_end, name="prefill")
        self.decode = _DecodeFreeListRegion(prefill_end, decode_end)
        # None = unset (treat as total free for get_num_free_blocks).
        self._alloc_is_prefill: bool | None = None
        self._alloc_is_swa: bool | None = None
        # Set by coordinator.get_num_blocks_to_allocate for split admission.
        self._pending_swa_blocks: int | None = None

        logger.info(
            "PDBlockPool enabled: swa=[%d,%d) (%d blocks), "
            "prefill=[%d,%d) (%d blocks), decode=[%d,%d) (%d blocks)",
            swa_start,
            swa_end,
            swa_end - swa_start,
            swa_end,
            prefill_end,
            prefill_end - swa_end,
            prefill_end,
            decode_end,
            decode_end - prefill_end,
        )

    @property
    def swa_range(self) -> tuple[int, int]:
        return self.swa.start, self.swa.end

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

    def set_alloc_is_swa(self, is_swa: bool) -> None:
        self._alloc_is_swa = is_swa

    def clear_alloc_is_swa(self) -> None:
        self._alloc_is_swa = None

    def set_pending_swa_blocks(self, num_swa_blocks: int) -> None:
        """Record SWA need for the next ``get_num_free_blocks`` admission check."""
        self._pending_swa_blocks = int(num_swa_blocks)

    def clear_pending_swa_blocks(self) -> None:
        self._pending_swa_blocks = None

    def get_num_free_blocks(self) -> int:
        # Decode: all groups share the decode free-list (no SWA split).
        if self._alloc_is_prefill is False:
            return self.decode.num_free

        # Prefill split admission: coordinator returns only non-SWA need;
        # fail if SWA cannot be satisfied, otherwise report prefill free.
        if self._pending_swa_blocks is not None:
            if self._pending_swa_blocks > self.swa.num_free:
                print(
                    f"[PDBlockPool] admission reject: SWA need={self._pending_swa_blocks} "
                    f"> swa.free={self.swa.num_free} | {self.summary()}",
                    flush=True,
                )
                return -1
            return self.prefill.num_free

        if self._alloc_is_swa is True:
            return self.swa.num_free
        if self._alloc_is_prefill is True:
            return self.prefill.num_free
        return self.swa.num_free + self.prefill.num_free + self.decode.num_free

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

        # Decode phase: all groups (including SWA) use the decode free-list
        # so incremental decode pops do not punch holes in the SWA bump region.
        if not is_prefill:
            region = "decode"
            region_free = self.decode.num_free
            if num_blocks > region_free:
                print(
                    f"[PDBlockPool] alloc fail: need={num_blocks} region={region} "
                    f"free={region_free} is_prefill={is_prefill} is_swa={self._alloc_is_swa} "
                    f"| {self.summary()}",
                    flush=True,
                )
                raise ValueError(
                    f"Cannot get {num_blocks} free blocks from the decode PD region "
                    f"(free={region_free})"
                )
            ids = self.decode.allocate(num_blocks)
        elif self._alloc_is_swa is True:
            region = "swa"
            region_free = self.swa.num_free
            if num_blocks > region_free:
                print(
                    f"[PDBlockPool] alloc fail: need={num_blocks} region={region} "
                    f"free={region_free} is_prefill={is_prefill} is_swa={self._alloc_is_swa} "
                    f"| {self.summary()}",
                    flush=True,
                )
                raise ValueError(
                    f"Cannot get {num_blocks} free blocks from the swa PD region "
                    f"(free={region_free})"
                )
            try:
                ids = self.swa.allocate_contiguous(num_blocks)
            except ValueError as e:
                print(
                    f"[PDBlockPool] alloc fail (contiguous): need={num_blocks} region=swa "
                    f"free={self.swa.num_free} err={e} | {self.summary()}",
                    flush=True,
                )
                raise
        else:
            region = "prefill"
            region_free = self.prefill.num_free
            if num_blocks > region_free:
                print(
                    f"[PDBlockPool] alloc fail: need={num_blocks} region={region} "
                    f"free={region_free} is_prefill={is_prefill} is_swa={self._alloc_is_swa} "
                    f"| {self.summary()}",
                    flush=True,
                )
                raise ValueError(
                    f"Cannot get {num_blocks} free blocks from the prefill PD region "
                    f"(free={region_free})"
                )
            try:
                ids = self.prefill.allocate_contiguous(num_blocks)
            except ValueError as e:
                print(
                    f"[PDBlockPool] alloc fail (contiguous): need={num_blocks} region=prefill "
                    f"free={self.prefill.num_free} err={e} | {self.summary()}",
                    flush=True,
                )
                raise

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
        prefill_ids: list[int] = []
        decode_ids: list[int] = []
        for block in to_free:
            bid = block.block_id
            if self.swa.start <= bid < self.swa.end:
                swa_ids.append(bid)
            elif self.prefill.start <= bid < self.prefill.end:
                prefill_ids.append(bid)
            elif self.decode.start <= bid < self.decode.end:
                decode_ids.append(bid)
            else:
                raise ValueError(f"block {bid} outside PD regions {self.summary()}")
        if swa_ids:
            self.swa.free(swa_ids)
        if prefill_ids:
            self.prefill.free(prefill_ids)
        if decode_ids:
            self.decode.free(decode_ids)

    def summary(self) -> str:
        ss, se = self.swa_range
        ps, pe = self.prefill_range
        ds, de = self.decode_range
        return (
            f"PDBlockPool(null=0, "
            f"swa=[{ss},{se}) free={self.swa.num_free}/{self.swa.capacity}, "
            f"prefill=[{ps},{pe}) free={self.prefill.num_free}/{self.prefill.capacity}, "
            f"decode=[{ds},{de}) free={self.decode.num_free})"
        )


def demo_pd_partition() -> None:
    """Tiny smoke demo: ``python -m vllm_ascend.core.pd_block_pool``."""
    # Need room for SWA(2048) + prefill + decode; use a large pool for the demo.
    cfg = PDBlockPoolConfig(
        num_gpu_blocks=4096,
        max_num_decode_reqs=32,
        max_blocks_per_decode_req=2,  # decode region = 64
        num_swa_blocks=SWA_KV_BLOCK_REGION_SIZE,
    )
    pool = PDBlockPool(
        num_gpu_blocks=cfg.num_gpu_blocks,
        enable_caching=False,
        hash_block_size=16,
        pd_config=cfg,
    )
    print(pool.summary())

    pool.set_alloc_is_swa(True)
    s0 = pool.get_new_blocks(8)
    print("swa alloc", [b.block_id for b in s0], "range", pool.swa_range)

    pool.clear_alloc_is_swa()
    pool.set_alloc_is_prefill(True)
    p0 = pool.get_new_blocks(8)
    print("prefill alloc", [b.block_id for b in p0], "range", pool.prefill_range)

    pool.set_alloc_is_prefill(False)
    d_blocks = [pool.get_new_blocks(1)[0] for _ in range(32)]
    print("decode alloc head/tail", d_blocks[0].block_id, d_blocks[-1].block_id, pool.decode_range)

    pool.free_blocks(s0)
    pool.free_blocks(p0)
    pool.set_alloc_is_swa(True)
    s1 = pool.get_new_blocks(8)
    print("swa realloc after free", [b.block_id for b in s1])
    pool.clear_alloc_is_swa()
    pool.set_alloc_is_prefill(True)
    p1 = pool.get_new_blocks(8)
    print("prefill realloc after free", [b.block_id for b in p1])
    print(pool.summary())

    pool.free_blocks(s1)
    pool.free_blocks(p1)
    pool.free_blocks(d_blocks)
    pool.clear_alloc_is_prefill()
    print("after free", pool.summary())


if __name__ == "__main__":
    demo_pd_partition()
