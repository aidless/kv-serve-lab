"""校准链：把 kvcalc 算出的数字，对上真实引擎日志里打印的数字。

三个数字**全部写死**，来源是 vLLM 自己在启动时打印的日志
`arc2/out/gpu_baseline/full.log`（2026-10-21 16:35:42，Qwen3-4B，单卡）：

    [gpu_worker.py:373]     Available KV cache memory: 30.53 GiB
    [kv_cache_utils.py:1307] GPU KV cache size: 222,336 tokens
    [kv_cache_utils.py:1312] Maximum concurrency for 16,384 tokens per request: 13.57x

**防循环论证纪律**：绝不允许「拿 222,336 反算出预算，再验证预算是 222,336」。
三个数各写各的，只断言我们算出来的量能对上引擎自己打印的量。
"""

from __future__ import annotations

import pytest

from kv_serve_lab.kvcalc import ModelShape, plan_capacity

GiB = 1024 ** 3

# ---- 引擎日志里逐字抄下来的三个数（不许改，改了就不是同一次运行）----
ENGINE_KV_BUDGET_GIB = 30.53
ENGINE_KV_TOKENS = 222_336
ENGINE_CONCURRENCY_16384 = 13.57

# ---- Qwen3-4B 的 config.json（hf-mirror 实测拉取，非记忆）----
# num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=8, head_dim=128
QWEN3_4B = ModelShape(
    num_layers=36,
    num_attention_heads=32,
    num_kv_heads=8,
    head_dim=128,
    max_position_embeddings=40960,
)
QWEN3_4B_BYTES_PER_TOKEN_FP16 = 147_456


class TestBytesPerToken:
    def test_qwen3_4b_matches_engine_arithmetic(self):
        """2(K,V) × 36层 × 8 kv heads × 128 head_dim × 2 B = 147,456 B/token。"""
        assert QWEN3_4B.kv_bytes_per_token("float16") == QWEN3_4B_BYTES_PER_TOKEN_FP16

    def test_fp16_is_twice_fp8(self):
        assert (
            QWEN3_4B.kv_bytes_per_token("float16")
            == 2 * QWEN3_4B.kv_bytes_per_token("fp8")
        )


class TestEngineBudgetReconciliation:
    def test_budget_divided_by_token_size_recovers_engine_token_count(self):
        """30.53 GiB ÷ 147,456 B/token = 222,312 token，引擎报 222,336，差 0.011%。

        这个 0.011% 不是我们算错了，而是引擎日志把 30.53 GiB 四舍五入到两位小数
        之后再算 token 数造成的。断言上界 0.1%，用来抓「公式写错量级」这类真错
        （例如漏乘 2、把 kv_heads 写成 attention_heads，都会差出几倍）。
        """
        computed_tokens = ENGINE_KV_BUDGET_GIB * GiB / QWEN3_4B_BYTES_PER_TOKEN_FP16
        rel_err = abs(computed_tokens - ENGINE_KV_TOKENS) / ENGINE_KV_TOKENS
        assert rel_err < 0.001, f"relative error {rel_err:.4%} exceeds 0.1%"
        # 同一个式子，只是取整方式不同 —— 222312（floor）与 222336（引擎）都对得上
        assert int(computed_tokens) == 222_312

    def test_max_concurrency_matches_engine(self):
        """222,336 / 16,384 = 13.57，与引擎打印的 13.57x 一致。"""
        assert round(ENGINE_KV_TOKENS / 16_384, 2) == ENGINE_CONCURRENCY_16384

    def test_plan_capacity_recovers_engine_concurrency_within_tolerance(self):
        """用引擎的 KV 预算喂给 plan_capacity，并发数应落在引擎报的 13 附近。

        引擎报 13.57（浮点），plan_capacity 返回整数 13（向下取整，因为第 14 个
        请求的块装不下）。差 1 个请求是「能否刚好塞进零头」的差别，不是公式错。
        """
        budget_bytes = int(ENGINE_KV_BUDGET_GIB * GiB)
        plan = plan_capacity(
            QWEN3_4B,
            gpu_mem_bytes=budget_bytes,
            dtype="float16",
            seq_len=16_384,
            kv_fraction=1.0,  # 这里 budget_bytes 已经就是 KV 预算，不再打折扣
        )
        assert plan.max_concurrent_requests == 13
        assert plan.kv_bytes_per_request == 16_384 * QWEN3_4B_BYTES_PER_TOKEN_FP16


class TestModelShapesFromRealConfigs:
    """五个基准模型，数字都来自各自真实的 config.json。"""

    @pytest.mark.parametrize(
        "shape,expected_bytes",
        [
            # Qwen2.5-7B: 28 层, 4 kv heads, head_dim 128
            (
                ModelShape(28, 28, 4, 128, 32768),
                57_344,
            ),
            # Qwen2.5-72B: 80 层, 8 kv heads, head_dim 128
            (
                ModelShape(80, 64, 8, 128, 32768),
                327_680,
            ),
            # Mistral-7B-v0.3: 32 层, 8 kv heads, head_dim 128
            (
                ModelShape(32, 32, 8, 128, 32768),
                131_072,
            ),
            # DeepSeek-V2-Lite (MLA): 27 层, kv_lora_rank 512 + qk_rope 64
            (
                ModelShape(
                    27,
                    16,
                    16,
                    128,
                    163_840,
                    is_mla=True,
                    kv_lora_rank=512,
                    qk_rope_head_dim=64,
                ),
                31_104,
            ),
            # Qwen3-4B: 36 层, 8 kv heads, head_dim 128
            (
                ModelShape(36, 32, 8, 128, 40960),
                147_456,
            ),
        ],
        ids=["qwen2.5-7b", "qwen2.5-72b", "mistral-7b-v0.3", "deepseek-v2-lite-mla", "qwen3-4b"],
    )
    def test_fp16_bytes_per_token(self, shape, expected_bytes):
        assert shape.kv_bytes_per_token("float16") == expected_bytes

    def test_mla_ignores_kv_heads_entirely(self):
        """MLA 压缩后每 token 只存 (kv_lora_rank + qk_rope_head_dim) × 层数。

        验证它确实**不**随 num_kv_heads 变化 —— 这是 MLA 与 GQA 的关键分野。
        """
        base = ModelShape(
            27, 16, 16, 128, 163_840,
            is_mla=True, kv_lora_rank=512, qk_rope_head_dim=64,
        )
        other = ModelShape(
            27, 16, 8, 128, 163_840,
            is_mla=True, kv_lora_rank=512, qk_rope_head_dim=64,
        )
        assert base.kv_bytes_per_token("float16") == other.kv_bytes_per_token("float16")
