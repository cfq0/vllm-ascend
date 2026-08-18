# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from tests.ut.base import TestBase
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import _share_req_blocks


class _FakeBlock:
    def __init__(self, block_id: int, ref_cnt: int = 1):
        self.block_id = block_id
        self.ref_cnt = ref_cnt


class TestShareReqBlocks(TestBase):
    def test_second_swa_manager_reuses_ids_and_bumps_refcnt(self):
        blocks = [_FakeBlock(10), _FakeBlock(11)]
        src = SimpleNamespace(
            req_to_blocks={"r0": list(blocks)},
            num_cached_block={"r0": 2},
        )
        dst = SimpleNamespace(
            req_to_blocks={"r0": []},
            num_cached_block={},
        )
        new = _share_req_blocks(dst, src, "r0")
        self.assertEqual([b.block_id for b in new], [10, 11])
        self.assertIs(dst.req_to_blocks["r0"][0], blocks[0])
        self.assertEqual(blocks[0].ref_cnt, 2)
        self.assertEqual(blocks[1].ref_cnt, 2)
        self.assertEqual(dst.num_cached_block["r0"], 2)
