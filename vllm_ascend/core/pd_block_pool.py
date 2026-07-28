# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill/Decode partitioned block pool for Ascend.

Splits the physical block-id space so decode single-block pops cannot fragment
prefill contiguous runs (important for DSA-CP SWA ``copy_`` writes).

Scope (current):
- Prefix cache / PCP / MTP are not supported; enable only with caching off.
- Prefill: bump allocator that prefers **id-contiguous** spans; on
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


@dataclass(frozen=True)
class PDBlockPoolConfig:
    """Partition layout for one KV-cache group's block ids.

    Block 0 is reserved as the null block (same convention as vLLM BlockPool).
    Remaining ids ``[1, num_gpu_blocks)`` are split into prefill then decode.
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

    def resolve(self) -> tuple[int, int, int, int]:
        """Return ``(null_id, prefill_start, prefill_end, decode_end)``.

        Prefill owns ``[prefill_start, prefill_end)``.
        Decode owns ``[prefill_end, decode_end)``.
        """
        if self.num_gpu_blocks < 2:
            raise ValueError("num_gpu_blocks must be >= 2")
        null_id = 0 if self.reserve_null_block else -1
        usable_start = 1 if self.reserve_null_block else 0
        usable = self.num_gpu_blocks - usable_start

        if self.num_decode_blocks is not None:
            num_decode = int(self.num_decode_blocks)
        else:
            num_decode = int(self.max_num_decode_reqs) * int(self.max_blocks_per_decode_req)
        if num_decode <= 0:
            raise ValueError("decode region must be > 0")
        if num_decode >= usable:
            raise ValueError(
                f"decode region ({num_decode}) leaves no prefill blocks "
                f"(usable={usable}, num_gpu_blocks={self.num_gpu_blocks})"
            )
        prefill_start = usable_start
        prefill_end = usable_start + (usable - num_decode)
        decode_end = self.num_gpu_blocks
        return null_id, prefill_start, prefill_end, decode_end


class _PrefillBumpRegion:
    """Contiguous bump allocator over ``[start, end)`` with wrap-around search."""

    def __init__(self, start: int, end: int) -> None:
        if end <= start:
            raise ValueError(f"empty prefill region [{start}, {end})")
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
            raise ValueError(f"prefill region: need {n} free blocks, only {self._num_free} left")

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
                    f"prefill region: need {n} blocks, allocated {n - remaining}, "
                    f"free={self._num_free} but no free contiguous span left (fragmented)"
                )
            out.extend(self._take_span(first, length))
            remaining -= length

        return out

    def free(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            if not (self.start <= bid < self.end):
                raise ValueError(f"block {bid} not in prefill region [{self.start}, {self.end})")
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
            raise RuntimeError(f"prefill block {bid} already allocated")
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
    """BlockPool that allocates from partitioned prefill/decode regions.

    Call :meth:`set_alloc_is_prefill` before ``get_new_blocks`` /
    ``get_num_free_blocks`` so admission and allocation use the correct region.
    Free always routes by block-id range (prefill blocks stay in prefill even
    after the request enters decode).
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
        null_id, prefill_start, prefill_end, decode_end = pd_config.resolve()
        if null_id != 0 or self.null_block.block_id != 0:
            raise ValueError("PDBlockPool expects null block id 0")

        # Drain BlockPool free-list; ownership moves to PD regions.
        n_free = self.free_block_queue.num_free_blocks
        if n_free > 0:
            self.free_block_queue.popleft_n(n_free)
        assert self.free_block_queue.num_free_blocks == 0

        self.prefill = _PrefillBumpRegion(prefill_start, prefill_end)
        self.decode = _DecodeFreeListRegion(prefill_end, decode_end)
        # None = unset (treat as total free for get_num_free_blocks).
        self._alloc_is_prefill: bool | None = None

        logger.info(
            "PDBlockPool enabled: prefill=[%d,%d) (%d blocks), "
            "decode=[%d,%d) (%d blocks)",
            prefill_start,
            prefill_end,
            prefill_end - prefill_start,
            prefill_end,
            decode_end,
            decode_end - prefill_end,
        )

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

    def get_num_free_blocks(self) -> int:
        if self._alloc_is_prefill is True:
            return self.prefill.num_free
        if self._alloc_is_prefill is False:
            return self.decode.num_free
        return self.prefill.num_free + self.decode.num_free

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

        region_free = self.prefill.num_free if is_prefill else self.decode.num_free
        if num_blocks > region_free:
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the "
                f"{'prefill' if is_prefill else 'decode'} PD region "
                f"(free={region_free})"
            )

        if is_prefill:
            ids = self.prefill.allocate_contiguous(num_blocks)
        else:
            ids = self.decode.allocate(num_blocks)

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

        prefill_ids: list[int] = []
        decode_ids: list[int] = []
        for block in to_free:
            bid = block.block_id
            if self.prefill.start <= bid < self.prefill.end:
                prefill_ids.append(bid)
            elif self.decode.start <= bid < self.decode.end:
                decode_ids.append(bid)
            else:
                raise ValueError(f"block {bid} outside PD regions {self.summary()}")
        if prefill_ids:
            self.prefill.free(prefill_ids)
        if decode_ids:
            self.decode.free(decode_ids)

    def summary(self) -> str:
        ps, pe = self.prefill_range
        ds, de = self.decode_range
        return (
            f"PDBlockPool(null=0, "
            f"prefill=[{ps},{pe}) free={self.prefill.num_free}/{self.prefill.capacity}, "
            f"decode=[{ds},{de}) free={self.decode.num_free})"
        )


def demo_pd_partition() -> None:
    """Tiny smoke demo: ``python -m vllm_ascend.core.pd_block_pool``."""
    cfg = PDBlockPoolConfig(
        num_gpu_blocks=128,
        max_num_decode_reqs=32,
        max_blocks_per_decode_req=2,  # decode region = 64
    )
    pool = PDBlockPool(
        num_gpu_blocks=cfg.num_gpu_blocks,
        enable_caching=False,
        hash_block_size=16,
        pd_config=cfg,
    )
    print(pool.summary())

    pool.set_alloc_is_prefill(True)
    p0 = pool.get_new_blocks(8)
    print("prefill alloc", [b.block_id for b in p0])

    pool.set_alloc_is_prefill(False)
    d_blocks = [pool.get_new_blocks(1)[0] for _ in range(32)]
    print("decode alloc head/tail", d_blocks[0].block_id, d_blocks[-1].block_id, pool.decode_range)

    pool.free_blocks(p0)
    pool.set_alloc_is_prefill(True)
    p1 = pool.get_new_blocks(8)
    print("prefill realloc after free", [b.block_id for b in p1])
    print(pool.summary())

    pool.free_blocks(d_blocks)
    pool.clear_alloc_is_prefill()
    print("after decode free", pool.summary())


if __name__ == "__main__":
    demo_pd_partition()
