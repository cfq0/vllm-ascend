#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.context_parallel.dsa_cp import plan_kv_block_writes


class TestPlanKvBlockWrites(TestBase):
    def test_decode_only_has_no_copy_runs(self):
        block_table = torch.zeros((3, 8), dtype=torch.int32)
        qsl = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        plan, scatter = plan_kv_block_writes(block_table, qsl, num_reqs=3, num_decodes=3, block_size=128)
        self.assertFalse(plan)
        self.assertEqual(scatter.numel(), 0)

    def test_prefill_counts_whole_blocks(self):
        block_table = torch.zeros((2, 8), dtype=torch.int32)
        block_table[0, 0] = 1
        block_table[1, 0] = 10
        qsl = torch.tensor([0, 256, 320], dtype=torch.int64)
        plan, scatter = plan_kv_block_writes(block_table, qsl, num_reqs=2, num_decodes=0, block_size=128)
        self.assertEqual(plan.token_starts.tolist(), [0, 256])
        self.assertEqual(plan.num_blocks.tolist(), [2, 1])
        self.assertEqual(plan.n_tokens.tolist(), [256, 64])
        self.assertEqual(plan.first_blocks.tolist(), [1, 10])
        self.assertEqual(scatter.numel(), 0)

    def test_mixed_decode_then_prefill_rounds_up(self):
        block_table = torch.zeros((3, 4), dtype=torch.int32)
        block_table[2, 0] = 5
        qsl = torch.tensor([0, 1, 2, 6], dtype=torch.int64)
        plan, scatter = plan_kv_block_writes(block_table, qsl, num_reqs=3, num_decodes=2, block_size=128)
        self.assertEqual(scatter.numel(), 0)
        self.assertEqual(plan.token_starts.tolist(), [2])
        self.assertEqual(plan.num_blocks.tolist(), [1])
        self.assertEqual(plan.n_tokens.tolist(), [4])
        self.assertEqual(plan.first_blocks.tolist(), [5])

    def test_short_prefill_still_one_block(self):
        block_table = torch.zeros((1, 4), dtype=torch.int32)
        block_table[0, 0] = 3
        qsl = torch.tensor([0, 50], dtype=torch.int64)
        plan, _ = plan_kv_block_writes(block_table, qsl, num_reqs=1, num_decodes=0, block_size=128)
        self.assertEqual(plan.token_starts.tolist(), [0])
        self.assertEqual(plan.num_blocks.tolist(), [1])
        self.assertEqual(plan.n_tokens.tolist(), [50])
        self.assertEqual(plan.first_blocks.tolist(), [3])

    def test_empty(self):
        plan, scatter = plan_kv_block_writes(
            torch.zeros((0, 1), dtype=torch.int32),
            torch.tensor([0], dtype=torch.int64),
            num_reqs=0,
            num_decodes=0,
            block_size=128,
        )
        self.assertFalse(plan)
        self.assertEqual(scatter.numel(), 0)

    def test_first_blocks_stay_on_block_table_device(self):
        block_table = torch.zeros((1, 4), dtype=torch.int32)
        block_table[0, 0] = 7
        qsl = torch.tensor([0, 128], dtype=torch.int64)
        plan, _ = plan_kv_block_writes(block_table, qsl, num_reqs=1, num_decodes=0, block_size=128)
        self.assertEqual(plan.first_blocks.device, block_table.device)
