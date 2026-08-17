# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.ut.base import TestBase
from vllm_ascend.core.pd_block_pool import (
    C4_KV_BLOCK_REGION_SIZE,
    PREFILL_SIZE_CLASS_SLABS,
    SWA_KV_BLOCK_REGION_SIZE,
    PDBlockPool,
    PDBlockPoolConfig,
    _PrefillSizeClassRegion,
    plan_size_class_slabs,
    size_class_region_blocks,
)


def _assert_contiguous(ids: list[int]) -> None:
    assert ids == list(range(ids[0], ids[0] + len(ids)))


class TestSizeClassSlabs(TestBase):
    def test_full_template_is_1536(self):
        self.assertEqual(size_class_region_blocks(), 1536)
        self.assertEqual(SWA_KV_BLOCK_REGION_SIZE, 1536)
        self.assertEqual(C4_KV_BLOCK_REGION_SIZE, 1536)
        slabs = plan_size_class_slabs(1, 1536)
        sizes = [cls for _, _, cls in slabs]
        expected = []
        for cls, count in PREFILL_SIZE_CLASS_SLABS:
            expected.extend([cls] * count)
        self.assertEqual(sizes, expected)
        self.assertEqual(sizes, [64] * 8 + [128] * 4 + [256] * 2)

    def test_overflow_slab_when_region_is_short(self):
        # 64*8 + 128*3 = 896, leftover 104 → overflow slab.
        slabs = plan_size_class_slabs(10, 1000)
        sizes = [cls for _, _, cls in slabs]
        self.assertEqual(sizes, [64] * 8 + [128] * 3 + [104])
        self.assertEqual(slabs[-1][1] - slabs[0][0], 1000)


class TestPrefillSizeClassRegion(TestBase):
    def _region(self, capacity: int = 1536, start: int = 1) -> _PrefillSizeClassRegion:
        return _PrefillSizeClassRegion(start, start + capacity, name="swa")

    def test_match_64_class(self):
        region = self._region()
        ids = region.allocate_contiguous(50)
        _assert_contiguous(ids)
        self.assertEqual(len(ids), 50)
        slab = region.slabs[0]
        self.assertEqual(slab.class_size, 64)
        self.assertTrue(slab.contains(ids[0]))

    def test_eight_64_slabs_then_escalate_to_128(self):
        region = self._region()
        used = [region.allocate_contiguous(64) for _ in range(8)]
        for i, ids in enumerate(used):
            self.assertEqual(region.slabs[i].start, ids[0])
            self.assertEqual(region.slabs[i].class_size, 64)
        ninth = region.allocate_contiguous(64)
        self.assertTrue(region.slabs[8].contains(ninth[0]))
        self.assertEqual(region.slabs[8].class_size, 128)
        _assert_contiguous(ninth)

    def test_256_request_skips_smaller_classes(self):
        region = self._region()
        ids = region.allocate_contiguous(200)
        _assert_contiguous(ids)
        self.assertEqual(len(ids), 200)
        owner = next(s for s in region.slabs if s.contains(ids[0]))
        self.assertEqual(owner.class_size, 256)

    def test_never_stitch_across_slabs(self):
        region = self._region()
        region.allocate_contiguous(64)
        region.allocate_contiguous(64)
        # Smaller leftovers must not be stitched; 200 goes to a 256 slab.
        ids = region.allocate_contiguous(200)
        owner = next(s for s in region.slabs if s.contains(ids[0]))
        self.assertEqual(owner.class_size, 256)
        self.assertTrue(all(owner.contains(i) for i in ids))

    def test_free_and_reuse_same_class(self):
        region = self._region()
        first = region.allocate_contiguous(64)
        region.free(first)
        second = region.allocate_contiguous(64)
        self.assertEqual(first, second)

    def test_too_large_for_any_slab(self):
        region = self._region()
        with self.assertRaises(ValueError):
            region.allocate_contiguous(257)


class TestPDBlockPoolSizeClass(TestBase):
    def _pool(self, num_gpu_blocks: int = 8192) -> PDBlockPool:
        cfg = PDBlockPoolConfig(num_gpu_blocks=num_gpu_blocks)
        return PDBlockPool(
            num_gpu_blocks=num_gpu_blocks,
            enable_caching=False,
            hash_block_size=16,
            pd_config=cfg,
        )

    def test_swa_and_c4_use_separate_1536_regions(self):
        pool = self._pool()
        ss, se = pool.swa_range
        cs, ce = pool.c4_range
        self.assertEqual(se - ss, 1536)
        self.assertEqual(ce - cs, 1536)
        self.assertEqual(ss, 1)
        self.assertEqual(cs, se)

        pool.set_alloc_is_prefill(True)
        pool.set_alloc_is_swa(True)
        swa_ids = [b.block_id for b in pool.get_new_blocks(64)]
        _assert_contiguous(swa_ids)
        self.assertTrue(ss <= swa_ids[0] < se)

        pool.set_alloc_is_c4(True)
        c4_ids = [b.block_id for b in pool.get_new_blocks(64)]
        _assert_contiguous(c4_ids)
        self.assertTrue(cs <= c4_ids[0] < ce)
        self.assertTrue(set(swa_ids).isdisjoint(c4_ids))

    def test_other_prefill_uses_free_list_not_size_class(self):
        pool = self._pool()
        ps, pe = pool.prefill_range
        self.assertGreaterEqual(pe - ps, 257)
        pool.set_alloc_is_prefill(True)
        pool.clear_alloc_is_swa()
        pool.clear_alloc_is_c4()
        # 257 would not fit any SWA/C4 size-class slab.
        ids = [b.block_id for b in pool.get_new_blocks(257)]
        self.assertEqual(len(ids), 257)
        self.assertTrue(all(ps <= i < pe for i in ids))
