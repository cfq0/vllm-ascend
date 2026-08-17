#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.context_parallel.dsa_cp import plan_kv_slot_writes


class TestPlanKvSlotWrites(TestBase):
    def test_decode_only_scatters_valid_tokens(self):
        slots = torch.tensor([10, 20, -1, 40], dtype=torch.int64)
        qsl = torch.tensor([0, 1, 2, 4], dtype=torch.int64)
        runs, scatter = plan_kv_slot_writes(
            slots, qsl, num_reqs=3, num_decodes=3, pad_slot_id=-1, device=torch.device("cpu")
        )
        self.assertEqual(runs, [])
        self.assertEqual(scatter.tolist(), [0, 1, 3])

    def test_prefill_one_copy_per_request(self):
        # Two prefills; slots are contiguous within each request (size-class).
        slots = torch.tensor(list(range(128, 128 + 256)) + list(range(512, 512 + 64)), dtype=torch.int64)
        qsl = torch.tensor([0, 256, 320], dtype=torch.int64)
        runs, scatter = plan_kv_slot_writes(
            slots, qsl, num_reqs=2, num_decodes=0, pad_slot_id=-1, device=torch.device("cpu")
        )
        self.assertEqual(runs, [(0, 256, 128), (256, 64, 512)])
        self.assertEqual(scatter.numel(), 0)

    def test_mixed_decode_scatter_and_prefill_copy(self):
        slots = torch.tensor([7, 9, 100, 101, 102, 103], dtype=torch.int64)
        qsl = torch.tensor([0, 1, 2, 6], dtype=torch.int64)
        runs, scatter = plan_kv_slot_writes(
            slots, qsl, num_reqs=3, num_decodes=2, pad_slot_id=-1, device=torch.device("cpu")
        )
        self.assertEqual(scatter.tolist(), [0, 1])
        self.assertEqual(runs, [(2, 4, 100)])

    def test_prefill_trims_edge_pads_only(self):
        slots = torch.tensor([-1, 200, 201, 202, -1], dtype=torch.int64)
        qsl = torch.tensor([0, 5], dtype=torch.int64)
        runs, scatter = plan_kv_slot_writes(
            slots, qsl, num_reqs=1, num_decodes=0, pad_slot_id=-1, device=torch.device("cpu")
        )
        self.assertEqual(runs, [(1, 3, 200)])
        self.assertEqual(scatter.numel(), 0)

    def test_short_prefill_still_copy(self):
        # Previously MIN_CONTIGUOUS_BLOCKS=2 sent 1-block prefills to scatter.
        slots = torch.tensor(list(range(64)), dtype=torch.int64)
        qsl = torch.tensor([0, 64], dtype=torch.int64)
        runs, scatter = plan_kv_slot_writes(
            slots, qsl, num_reqs=1, num_decodes=0, pad_slot_id=-1, device=torch.device("cpu")
        )
        self.assertEqual(runs, [(0, 64, 0)])
        self.assertEqual(scatter.numel(), 0)

    def test_empty_mapping(self):
        runs, scatter = plan_kv_slot_writes(
            torch.empty(0, dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
            num_reqs=0,
            num_decodes=0,
            pad_slot_id=-1,
            device=torch.device("cpu"),
        )
        self.assertEqual(runs, [])
        self.assertEqual(scatter.numel(), 0)
