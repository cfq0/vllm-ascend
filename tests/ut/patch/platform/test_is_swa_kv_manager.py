# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.ut.base import TestBase
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import _is_main_model_swa_layer


class TestIsMainModelSwaLayer(TestBase):
    def test_main_swa_cache(self):
        self.assertTrue(_is_main_model_swa_layer("layers.0.self_attn.swa_cache"))

    def test_mtp_swa_cache_is_not_main(self):
        self.assertFalse(_is_main_model_swa_layer("mtp.layers.0.self_attn.swa_cache"))
        self.assertFalse(_is_main_model_swa_layer("model.mtp.layers.0.swa_cache"))

    def test_state_cache_is_not_swa(self):
        self.assertFalse(_is_main_model_swa_layer("layers.0.self_attn.compressor.state_cache"))
        self.assertFalse(_is_main_model_swa_layer("layers.0.self_attn.indexer.compressor.state_cache"))
