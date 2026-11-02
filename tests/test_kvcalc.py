"""kvcalc 测试：KV 显存容量规划。

所有期望值都来自真实 config.json（经 hf-mirror 实测拉取，2026-10-31）
与 vLLM 源码里的公式（vllm/v1/kv_cache_interface.py:585-601）。
每个数字下面都写了复算式，任何一个对不上就是算错了。
"""

import pytest

from kv_serve_lab.kvcalc import (
    ModelShape,
    plan_capacity,
    DTYPE_BYTES,
    IDEAL_PAYLOAD_ONLY,
)


# ---- 真实 config.json 片段（只保留算 KV 需要的字段）----

QWEN25_7B = {
    "num_hidden_layers": 28,
    "hidden_size": 3584,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "max_position_embeddings": 32768,
}

QWEN25_72B = {
    "num_hidden_layers": 80,
    "hidden_size": 8192,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "max_position_embeddings": 32768,
}

MISTRAL_7B_V3 = {
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 32768,
}

# DeepSeek-V2-Lite：MLA，没有 num_key_value_heads 语义，靠 kv_lora_rank
DEEPSEEK_V2_LITE = {
    "num_hidden_layers": 27,
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "max_position_embeddings": 163840,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
}


class TestModelShapeFromConfig:
    def test_gqa_head_dim_is_hidden_over_heads(self):
        """head_dim = hidden_size / num_attention_heads（Qwen2.5-7B: 3584/28 = 128）。"""
        shape = ModelShape.from_hf_config(QWEN25_7B)
        assert shape.head_dim == 128
        assert shape.num_layers == 28
        assert shape.num_kv_heads == 4

    def test_mistral_head_dim(self):
        """Mistral-7B-v0.3: 4096/32 = 128。"""
        shape = ModelShape.from_hf_config(MISTRAL_7B_V3)
        assert shape.head_dim == 128
        assert shape.num_kv_heads == 8

    def test_missing_kv_heads_defaults_to_num_attention_heads(self):
        """老 config 没有 num_key_value_heads 时退化为 MHA（kv_heads = heads）。"""
        cfg = {
            "num_hidden_layers": 12,
            "hidden_size": 768,
            "num_attention_heads": 12,
        }
        shape = ModelShape.from_hf_config(cfg)
        assert shape.num_kv_heads == 12

    def test_mla_detected(self):
        """带 kv_lora_rank 的 config 必须识别成 MLA，而不是当成 GQA。"""
        shape = ModelShape.from_hf_config(DEEPSEEK_V2_LITE)
        assert shape.is_mla is True
        assert shape.kv_lora_rank == 512
        assert shape.qk_rope_head_dim == 64


class TestKvBytesPerToken:
    """bytes/token = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes。

    复算：
      Qwen2.5-7B     2*28*4*128*2  = 57344
      Qwen2.5-72B    2*80*8*128*2  = 327680
      Mistral-7B-v3  2*32*8*128*2  = 131072
    """

    def test_qwen25_7b_bf16(self):
        shape = ModelShape.from_hf_config(QWEN25_7B)
        assert shape.kv_bytes_per_token("bfloat16") == 57344

    def test_qwen25_72b_bf16(self):
        shape = ModelShape.from_hf_config(QWEN25_72B)
        assert shape.kv_bytes_per_token("bfloat16") == 327680

    def test_mistral_7b_bf16(self):
        shape = ModelShape.from_hf_config(MISTRAL_7B_V3)
        assert shape.kv_bytes_per_token("bfloat16") == 131072

    def test_fp8_halves_the_bytes(self):
        """fp8 是 1 byte，所以刚好是 bf16 的一半。"""
        shape = ModelShape.from_hf_config(QWEN25_7B)
        assert shape.kv_bytes_per_token("fp8") == 57344 // 2

    def test_mla_bytes_per_token(self):
        """MLA 只存一个 latent + 一条 RoPE，head_size_v = 0。

        复算：(kv_lora_rank + qk_rope_head_dim) × n_layers × dtype_bytes
              = (512 + 64) × 27 × 2 = 31104
        """
        shape = ModelShape.from_hf_config(DEEPSEEK_V2_LITE)
        assert shape.kv_bytes_per_token("bfloat16") == 31104

    def test_unknown_dtype_raises(self):
        shape = ModelShape.from_hf_config(QWEN25_7B)
        with pytest.raises(ValueError, match="unknown dtype"):
            shape.kv_bytes_per_token("float128")


class TestKvBytesForSeq:
    def test_scales_linearly_with_seq_len(self):
        shape = ModelShape.from_hf_config(QWEN25_7B)
        one = shape.kv_bytes_for(1, "bfloat16")
        assert shape.kv_bytes_for(4096, "bfloat16") == one * 4096

    def test_tp_shards_kv_heads(self):
        """TP=4 时每个 rank 只持有 1 个 kv head（Qwen2.5-7B 有 4 个 kv heads）。"""
        shape = ModelShape.from_hf_config(QWEN25_7B)
        tp1 = shape.kv_bytes_for(4096, "bfloat16", tp_size=1)
        tp4 = shape.kv_bytes_for(4096, "bfloat16", tp_size=4)
        assert tp1 == tp4 * 4

    def test_tp_replication_requires_divisible_heads(self):
        """TP 大于 kv heads 时只能「复制」而非「切分」，且必须整除。

        Qwen2.5-7B 只有 28 个 query head，tp=8 时 28 % 8 = 4 != 0 → 非法。
        旧实现会静默 clamp 到 tp=4（把每 rank 的 KV 少算一半、并发多算一倍），
        这里要求明确抛错，绝不静默取整。
        """
        shape = ModelShape.from_hf_config(QWEN25_7B)
        with pytest.raises(ValueError, match="divis"):
            shape.kv_bytes_for(4096, "bfloat16", tp_size=8)

    def test_tp_replication_valid_when_divisible(self):
        """Mistral-7B：32 q heads / 8 kv heads，tp=16 合法。

        16 % 8 == 0（kv head 复制）且 32 % 16 == 0（query head 均分）。
        复制时每 rank 只持有 1 个 kv head，所以和 tp=8（8/8=1）字节数相同。
        """
        shape = ModelShape.from_hf_config(MISTRAL_7B_V3)
        tp8 = shape.kv_bytes_for(4096, "bfloat16", tp_size=8)
        tp16 = shape.kv_bytes_for(4096, "bfloat16", tp_size=16)
        assert tp16 == tp8
        assert tp16 == shape.kv_bytes_for(4096, "bfloat16", tp_size=1) // 8


class TestPlanCapacity:
    """容量规划：给定显存能装多少并发请求。"""

    def test_known_case_qwen25_7b_on_24gb(self):
        """24 GiB 卡，扣 20% 给权重/激活/碎片，剩 80% 给 KV。

        24 GiB = 25769803776 bytes
        KV 可用 = 25769803776 * 0.8 = 20615843020.8 -> 20615843020
        每请求 4096 token：57344 * 4096 = 234881024 bytes
        最大并发 = 20615843020 // 234881024 = 87
        """
        shape = ModelShape.from_hf_config(QWEN25_7B)
        plan = plan_capacity(
            shape,
            gpu_mem_bytes=24 * 1024**3,
            dtype="bfloat16",
            seq_len=4096,
            kv_fraction=0.8,
        )
        assert plan.max_concurrent_requests == 87
        assert plan.kv_bytes_available == 20615843020
        assert plan.kv_bytes_per_request == 234881024

    def test_zero_concurrency_when_memory_too_small(self):
        """显存连一个请求都装不下时返回 0，不返回负数。"""
        shape = ModelShape.from_hf_config(QWEN25_72B)
        plan = plan_capacity(
            shape, gpu_mem_bytes=1024**2, dtype="bfloat16", seq_len=4096
        )
        assert plan.max_concurrent_requests == 0

    def test_tp_divides_kv_per_rank(self):
        """TP=4 时每张卡只存 1/4 的 KV，所以单卡并发数约 ×4。

        精确值是 351 而不是 87*4=348 —— 差 3 是两次整数取整的结果
        （每请求字节先整除 4，可用字节再整除一次），不是算错。
        """
        shape = ModelShape.from_hf_config(QWEN25_7B)
        p1 = plan_capacity(shape, 24 * 1024**3, "bfloat16", 4096, tp_size=1)
        p4 = plan_capacity(shape, 24 * 1024**3, "bfloat16", 4096, tp_size=4)
        assert p1.max_concurrent_requests == 87
        assert p4.max_concurrent_requests == 351
        assert p4.kv_bytes_per_request == p1.kv_bytes_per_request // 4

    def test_block_count_uses_ceil(self):
        """block 数必须向上取整（16 token 一块），否则少算一块就 OOM。"""
        shape = ModelShape.from_hf_config(QWEN25_7B)
        plan = plan_capacity(
            shape, 24 * 1024**3, "bfloat16", seq_len=4096, block_size=16
        )
        assert plan.blocks_per_request == 4096 // 16
        # 4097 token 需要 257 块（4096/16=256 块装 4096，还差 1 token）
        plan2 = plan_capacity(
            shape, 24 * 1024**3, "bfloat16", seq_len=4097, block_size=16
        )
        assert plan2.blocks_per_request == 257

    def test_total_kv_bytes_matches_manual(self):
        """总 KV 占用 = 并发数 × 每请求占用（不允许凑整误差）。"""
        shape = ModelShape.from_hf_config(QWEN25_7B)
        plan = plan_capacity(shape, 24 * 1024**3, "bfloat16", 4096)
        assert (
            plan.max_concurrent_requests * plan.kv_bytes_per_request
            <= plan.kv_bytes_available
        )


class TestDtypeTable:
    def test_known_dtypes(self):
        assert DTYPE_BYTES["float32"] == 4
        assert DTYPE_BYTES["bfloat16"] == 2
        assert DTYPE_BYTES["float16"] == 2
        assert DTYPE_BYTES["fp8"] == 1
        assert DTYPE_BYTES["int8"] == 1
        assert DTYPE_BYTES["int4"] == 0.5

    def test_int4_is_marked_ideal_payload_only(self):
        """int4 的 0.5 byte 是「理想打包 payload」，不含 scale/zero-point/对齐填充。

        真实量化 kernel 每个 block 还要存 scale（和可能的 zero point），
        实际占用高于 0.5 B/elem。这里必须显式标注，不能让人当成实测占用。
        """
        assert DTYPE_BYTES["int4"] == 0.5
        assert "int4" in IDEAL_PAYLOAD_ONLY


class TestTpValidation:
    """TP 并行度校验：query head 必须被 tp 整除；kv head 要么整除要么合法复制。"""

    def test_tp_must_divide_query_heads(self):
        shape = ModelShape.from_hf_config(QWEN25_7B)  # 28 heads
        with pytest.raises(ValueError, match="divis"):
            shape.kv_bytes_for(4096, "bfloat16", tp_size=3)

    def test_tp_must_divide_kv_heads_when_sharding(self):
        # heads=12 / kv_heads=3：12 % 2 == 0（query 合法），但 3 % 2 != 0（不能均分 kv head）
        cfg = {
            "num_hidden_layers": 2,
            "hidden_size": 1536,
            "num_attention_heads": 12,
            "num_key_value_heads": 3,
        }
        shape = ModelShape.from_hf_config(cfg)
        with pytest.raises(ValueError, match="divis"):
            shape.kv_bytes_for(4096, "bfloat16", tp_size=2)

    def test_replication_requires_tp_mod_kvheads_zero(self):
        # heads=24（24 % 6 == 0）/ kv_heads=4：tp=6 > 4，6 % 4 != 0 → 非法复制
        cfg = {
            "num_hidden_layers": 2,
            "hidden_size": 3072,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
        }
        shape = ModelShape.from_hf_config(cfg)
        with pytest.raises(ValueError, match="divis"):
            shape.kv_bytes_for(4096, "bfloat16", tp_size=6)

    def test_tp_one_always_ok(self):
        shape = ModelShape.from_hf_config(QWEN25_7B)
        assert shape.kv_bytes_for(4096, "bfloat16", tp_size=1) > 0

    def test_rejects_nonpositive_tp(self):
        shape = ModelShape.from_hf_config(QWEN25_7B)
        with pytest.raises(ValueError):
            shape.kv_bytes_for(4096, "bfloat16", tp_size=0)


class TestShapeValidation:
    """非法 shape 必须早失败，不能算出一个看似合理的负数/零。"""

    def test_rejects_zero_layers(self):
        with pytest.raises(ValueError):
            ModelShape.from_hf_config(
                {"num_hidden_layers": 0, "hidden_size": 768, "num_attention_heads": 12}
            )

    def test_rejects_zero_heads(self):
        with pytest.raises(ValueError):
            ModelShape.from_hf_config(
                {"num_hidden_layers": 2, "hidden_size": 768, "num_attention_heads": 0}
            )

    def test_rejects_zero_hidden_size(self):
        with pytest.raises(ValueError):
            ModelShape.from_hf_config(
                {"num_hidden_layers": 2, "hidden_size": 0, "num_attention_heads": 12}
            )

    def test_rejects_zero_kv_heads(self):
        with pytest.raises(ValueError):
            ModelShape.from_hf_config(
                {
                    "num_hidden_layers": 2,
                    "hidden_size": 768,
                    "num_attention_heads": 12,
                    "num_key_value_heads": 0,
                }
            )

    def test_rejects_gqa_heads_not_multiple_of_kv(self):
        # GQA 要求 num_attention_heads % num_kv_heads == 0
        with pytest.raises(ValueError):
            ModelShape.from_hf_config(
                {
                    "num_hidden_layers": 2,
                    "hidden_size": 1280,
                    "num_attention_heads": 10,
                    "num_key_value_heads": 4,
                }
            )

    def test_rejects_negative_seq(self):
        shape = ModelShape.from_hf_config(QWEN25_7B)
        with pytest.raises(ValueError):
            shape.kv_bytes_for(-1, "bfloat16")


class TestWholeBlockPacking:
    """容量必须按「整块」计：block 是最小分配单位，padding 也是真实占用。"""

    def test_partial_first_block_rounds_to_one_block(self):
        shape = ModelShape.from_hf_config(QWEN25_7B)
        p1 = plan_capacity(shape, 24 * 1024**3, "bfloat16", seq_len=1, block_size=16)
        p16 = plan_capacity(shape, 24 * 1024**3, "bfloat16", seq_len=16, block_size=16)
        assert p1.blocks_per_request == 1
        assert p16.blocks_per_request == 1
        # 1 token 也要占满一整块（16 token）的字节
        assert p1.kv_bytes_per_request == 57344 * 16
        assert p1.max_concurrent_requests == p16.max_concurrent_requests

    def test_partial_second_block_rounds_to_two_blocks(self):
        shape = ModelShape.from_hf_config(QWEN25_7B)
        p17 = plan_capacity(shape, 24 * 1024**3, "bfloat16", seq_len=17, block_size=16)
        p32 = plan_capacity(shape, 24 * 1024**3, "bfloat16", seq_len=32, block_size=16)
        assert p17.blocks_per_request == 2
        assert p17.kv_bytes_per_request == 57344 * 32  # 2 块 × 16 token
        assert p17.max_concurrent_requests == p32.max_concurrent_requests
        p16 = plan_capacity(shape, 24 * 1024**3, "bfloat16", seq_len=16, block_size=16)
        assert p17.max_concurrent_requests < p16.max_concurrent_requests

    def test_rejects_negative_memory(self):
        shape = ModelShape.from_hf_config(QWEN25_7B)
        with pytest.raises(ValueError):
            plan_capacity(shape, -1, "bfloat16", seq_len=4096)

    def test_rejects_zero_seq(self):
        shape = ModelShape.from_hf_config(QWEN25_7B)
        with pytest.raises(ValueError):
            plan_capacity(shape, 24 * 1024**3, "bfloat16", seq_len=0)

    def test_rejects_zero_block_size(self):
        shape = ModelShape.from_hf_config(QWEN25_7B)
        with pytest.raises(ValueError):
            plan_capacity(shape, 24 * 1024**3, "bfloat16", seq_len=4096, block_size=0)


class TestDocstringSemantics:
    def test_distinguishes_kv_fraction_from_vllm_utilization(self):
        """kv_fraction 是「KV cache 分到的显存比例」，与 vLLM 的
        gpu_memory_utilization（整机显存可用上限，含权重/激活）不是一回事，
        docstring 必须同时点出 gpu_memory_utilization 才不会误导。"""
        doc = plan_capacity.__doc__ or ""
        assert "gpu_memory_utilization" in doc
        assert "KV cache" in doc


class TestCompressionMetric:
    """KV 压缩口径（arXiv:2603.20397 综述的量化/投影类方向）。

    这是**建模估计**，不是实测——真实压缩 KV 有 scale/group 元数据、
    对齐 padding 与 dequant 开销，且 GQA 压缩是否逐 head 独立会显著改变结果。
    本类只验证「建模公式本身正确」，不把建模值当实测值。
    """

    def test_int4_halves_fp16_bytes_for_gqa(self):
        """Qwen3-4B GQA：每元素 0.5 byte（int4 理想打包）→ 正好 fp16 的一半。"""
        shape = ModelShape(
            num_layers=36, num_attention_heads=32, num_kv_heads=8,
            head_dim=128, max_position_embeddings=32768,
        )
        fp16 = shape.kv_bytes_per_token("float16")
        c4 = shape.kv_bytes_per_token_with_compression("float16", value_bytes=0.5)
        assert c4 == pytest.approx(fp16 / 2)

    def test_fp8_same_as_fp16_with_value_bytes_one(self):
        """每元素 1 byte 且 K/V 同宽 → 与 fp16 同字节（无压缩）。"""
        shape = ModelShape(
            num_layers=36, num_attention_heads=32, num_kv_heads=8,
            head_dim=128, max_position_embeddings=32768,
        )
        fp16 = shape.kv_bytes_per_token("float16")
        c8 = shape.kv_bytes_per_token_with_compression("float16", value_bytes=1.0)
        assert c8 == pytest.approx(fp16)

    def test_mla_uses_latent_dimension(self):
        """MLA 只压 latent：每层每 token = (kv_lora_rank + qk_rope_head_dim) 元素。"""
        mla = ModelShape(
            num_layers=27, num_attention_heads=16, num_kv_heads=0,
            head_dim=128, max_position_embeddings=16384,
            is_mla=True, kv_lora_rank=512, qk_rope_head_dim=64,
        )
        # 每元素 0.5 byte：27 × (512+64) × 0.5 = 7776
        c4 = mla.kv_bytes_per_token_with_compression("float16", value_bytes=0.5)
        assert c4 == pytest.approx(7776.0)
        # fp16 基线 = 27 × 576 × 2 = 31104
        fp16 = mla.kv_bytes_per_token("float16")
        assert fp16 == 31104
        assert c4 == pytest.approx(fp16 / 4)

    def test_rejects_non_positive_value_bytes(self):
        shape = ModelShape(
            num_layers=36, num_attention_heads=32, num_kv_heads=8,
            head_dim=128, max_position_embeddings=32768,
        )
        with pytest.raises(ValueError, match="value_bytes must be > 0"):
            shape.kv_bytes_per_token_with_compression("float16", value_bytes=0)
        with pytest.raises(ValueError, match="value_bytes must be > 0"):
            shape.kv_bytes_per_token_with_compression("float16", value_bytes=-1)

    def test_rejects_non_positive_key_bytes(self):
        shape = ModelShape(
            num_layers=36, num_attention_heads=32, num_kv_heads=8,
            head_dim=128, max_position_embeddings=32768,
        )
        with pytest.raises(ValueError, match="key_bytes must be > 0"):
            shape.kv_bytes_per_token_with_compression(
                "float16", value_bytes=0.5, key_bytes=0
            )

    def test_docstring_flags_ideal_payload_only(self):
        """docstring 必须明确警告「建模值 ≠ 实测值」。"""
        doc = ModelShape.kv_bytes_per_token_with_compression.__doc__ or ""
        assert "建模" in doc
        assert "实测" in doc
