"""KV cache 显存容量规划。

公式对齐 vLLM 源码，不是凭记忆写的：

  vllm/v1/kv_cache_interface.py
    :585-590  state_content_size_bytes = (head_size + head_size_v) * dtype_bytes
    :593      unpadded_page_size_bytes = num_heads * num_states * state_content_size_bytes

对 full attention，num_states = block_size // tokens_per_state，tokens_per_state == 1，
所以一个 block 的字节数 = num_kv_heads × block_size × (head_size + head_size_v) × dtype_bytes。
把 block_size 换成 seq_len，就得到「一个请求整段序列的 KV 字节数」。

MLA（DeepSeek 系）的 head_size_v = 0（源码 :741 的注释：MLA stores a single latent
vector per state; there is no separate V），所以每层每 token 只存
(kv_lora_rank + qk_rope_head_dim) 个元素。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# 一个元素占几个字节。int4 是 0.5（4 bit / 8）。
#
# ⚠️ int4 的 0.5 是「理想打包 payload」：只算 4 bit/元素本身，
# **不含**量化 kernel 每 block 必存的 scale（以及可能的 zero-point）与对齐填充。
# 真实 int4 KV 的显存占用要高于这个数，不能把它当实测值用。
DTYPE_BYTES: dict[str, float] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "fp8": 1,
    "int8": 1,
    "int4": 0.5,
}

# 只有「理想打包 payload」语义、不含量化开销的 dtype。
IDEAL_PAYLOAD_ONLY: frozenset[str] = frozenset({"int4"})


@dataclass(frozen=True)
class ModelShape:
    """只保留算 KV cache 需要的字段，不假装是个完整的模型描述。"""

    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    max_position_embeddings: int
    is_mla: bool = False
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    # 残差流宽度（hidden_size）。放在最后并给默认值，避免打乱既有的位置参数调用。
    # 注意 hidden_size ≠ num_attention_heads × head_dim：Qwen3-4B 是 2560 ≠ 32×128=4096，
    # 所以算通信量（`kv_serve_lab.comm`）必须用它，不能用 heads×head_dim 顶替。
    # 0 表示「未提供」——`comm` 会据此显式报错，而不是静默用错的数。
    hidden_size: int = 0

    @classmethod
    def from_hf_config(cls, config: dict[str, Any]) -> "ModelShape":
        try:
            num_layers = int(config["num_hidden_layers"])
            num_heads = int(config["num_attention_heads"])
            hidden_size = int(config["hidden_size"])
        except KeyError as exc:  # 缺字段直接报出缺哪个，而不是后面 KeyError 裸奔
            raise ValueError(f"config missing required field {exc}") from None

        # 非法 shape 必须早失败：否则后面会算出 0 或负数，看起来「有结果」其实是垃圾。
        if num_layers <= 0:
            raise ValueError(f"num_hidden_layers must be > 0, got {num_layers}")
        if num_heads <= 0:
            raise ValueError(f"num_attention_heads must be > 0, got {num_heads}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {hidden_size}")

        head_dim = int(config.get("head_dim") or hidden_size // num_heads)
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {head_dim}")

        max_pos = int(config.get("max_position_embeddings", 0))

        # MLA 模型：带 kv_lora_rank 就是 DeepSeek 系（V2/V3/V3.1/V4）
        kv_lora_rank = int(config.get("kv_lora_rank", 0))
        qk_rope_head_dim = int(config.get("qk_rope_head_dim", 0))
        is_mla = kv_lora_rank > 0
        if is_mla and qk_rope_head_dim <= 0:
            raise ValueError(
                "MLA config must have qk_rope_head_dim > 0, "
                f"got {qk_rope_head_dim}"
            )

        # 老 config 没有 num_key_value_heads 时退化为 MHA
        num_kv_heads = int(config.get("num_key_value_heads", num_heads))
        if num_kv_heads <= 0:
            raise ValueError(f"num_key_value_heads must be > 0, got {num_kv_heads}")

        # GQA 要求 query head 数能被 kv head 数整除（否则没法成组共享一个 KV head）。
        if not is_mla and num_heads % num_kv_heads != 0:
            raise ValueError(
                "GQA requires num_attention_heads % num_key_value_heads == 0, "
                f"got {num_heads} % {num_kv_heads} = {num_heads % num_kv_heads}"
            )

        return cls(
            num_layers=num_layers,
            num_attention_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=max_pos,
            is_mla=is_mla,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            hidden_size=hidden_size,
        )

    def _dtype_bytes(self, dtype: str) -> float:
        try:
            return DTYPE_BYTES[dtype]
        except KeyError:
            raise ValueError(
                f"unknown dtype {dtype!r}; known: {sorted(DTYPE_BYTES)}"
            ) from None

    def kv_bytes_per_token(self, dtype: str) -> int:
        """每层每 token 的 KV 字节数 × 层数 = 整个模型每 token 的 KV 字节数。"""
        nbytes = self._dtype_bytes(dtype)
        if self.is_mla:
            # MLA：每层每 token 存 kv_lora_rank + qk_rope_head_dim 个元素
            elems_per_layer = self.kv_lora_rank + self.qk_rope_head_dim
        else:
            # GQA/MHA：K 和 V 各 num_kv_heads × head_dim 个元素
            elems_per_layer = 2 * self.num_kv_heads * self.head_dim
        return int(elems_per_layer * self.num_layers * nbytes)

    def _kv_heads_per_rank(self, tp_size: int) -> int:
        """每个 TP rank 实际持有多少个 kv head（用于算单卡 KV 字节数）。

        两种情况，都必须整除，绝不静默取整：

        - ``tp <= num_kv_heads``：均分，要求 ``num_kv_heads % tp == 0``，
          每 rank ``num_kv_heads // tp`` 个。
        - ``tp > num_kv_heads``：切不动了，改为**复制**（多个 rank 共享同一个
          kv head），要求 ``tp % num_kv_heads == 0``，每 rank 1 个。
        """
        if tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {tp_size}")
        if tp_size <= self.num_kv_heads:
            if self.num_kv_heads % tp_size != 0:
                raise ValueError(
                    "num_key_value_heads not divisible by tp_size: "
                    f"{self.num_kv_heads} % {tp_size} = "
                    f"{self.num_kv_heads % tp_size} (refuse to silently floor)"
                )
            return self.num_kv_heads // tp_size
        # 复制分支：tp > kv heads，要求 tp 是 kv heads 的整数倍
        if tp_size % self.num_kv_heads != 0:
            raise ValueError(
                "cannot replicate kv heads: tp_size is not divisible by "
                "num_key_value_heads "
                f"({tp_size} % {self.num_kv_heads} = "
                f"{tp_size % self.num_kv_heads})"
            )
        return 1

    def kv_bytes_for(self, seq_len: int, dtype: str, tp_size: int = 1) -> int:
        """一个请求在 seq_len 长度下的 KV 字节数（单卡视角，已按 TP 切分）。"""
        if seq_len < 0:
            raise ValueError(f"seq_len must be >= 0, got {seq_len}")
        if tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {tp_size}")

        # TP 下 query head 必须能被 tp 均分（否则这一层根本切不开）。
        if self.num_attention_heads % tp_size != 0:
            raise ValueError(
                "num_attention_heads not divisible by tp_size: "
                f"{self.num_attention_heads} % {tp_size} = "
                f"{self.num_attention_heads % tp_size}"
            )

        total = self.kv_bytes_per_token(dtype) * seq_len
        if not self.is_mla:
            # kv head 按 rank 切分/复制（内部已校验整除性）
            per_rank = self._kv_heads_per_rank(tp_size)
            # per_token 里含 num_kv_heads 因子，乘除是精确的整数运算
            total = total * per_rank // self.num_kv_heads
        else:
            # MLA 的 latent 维度不按 kv head 切，vLLM 里 MLA 是整段复制到各 TP rank
            pass
        return total

    def kv_bytes_per_token_with_compression(
        self,
        dtype: str,
        value_bytes: float,
        key_bytes: float | None = None,
    ) -> float:
        """按「每 token 每元素字节数」算压缩后的 KV 字节/token（建模值）。

        ``value_bytes`` / ``key_bytes`` 是压缩后**每个元素**占的字节数，用于建模
        KV 压缩（arXiv:2603.20397 综述的量化/投影类方向）。这是**建模估计**，不是
        实测：真实压缩 KV 还有 scale/group 元数据、对齐 padding 与 dequant 开销，
        且 GQA 的压缩是否逐 head 独立会显著改变结果，故标注 IDEAL_PAYLOAD_ONLY
        同款提醒——不要把它当实测显存占用用。

        与 :meth:`kv_bytes_per_token` 的差异：后者按 dtype 字节数（fp16=2、fp8=1）
        算每元素，这里允许任意压缩后的每元素字节（如 int4=0.5、fp8 加 scale）。
        """
        if value_bytes <= 0:
            raise ValueError(f"value_bytes must be > 0, got {value_bytes}")
        if key_bytes is not None and key_bytes <= 0:
            raise ValueError(f"key_bytes must be > 0, got {key_bytes}")
        # 每层每 token 的元素数（与 kv_bytes_per_token 同构）：
        #   GQA: 2 * n_kv_heads * head_dim（K、V 各一份）
        #   MLA: (kv_lora_rank + qk_rope_head_dim)（压缩的 KV 潜变量 + RoPE 分量）
        if self.is_mla:
            elements_per_layer_per_token = self.kv_lora_rank + self.qk_rope_head_dim
            kv_element_bytes = value_bytes
        else:
            elements_per_layer_per_token = 2 * self.num_kv_heads * self.head_dim
            kv_element_bytes = value_bytes + (key_bytes if key_bytes is not None else value_bytes)
        return (
            self.num_layers
            * elements_per_layer_per_token
            * kv_element_bytes
        )


@dataclass(frozen=True)
class CapacityPlan:
    """一张卡能装多少并发请求。

    注意 ``kv_bytes_per_request`` 是**按整块（block）向上取整后**的真实占用，
    不是 raw 的 ``bytes_per_token × seq_len``：block 是 KV cache 的最小分配单位，
    末块没写满也要占满一整块，这部分 padding 是实打实的显存。
    """

    max_concurrent_requests: int
    kv_bytes_available: int
    kv_bytes_per_request: int
    blocks_per_request: int
    block_size: int
    seq_len: int
    dtype: str
    tp_size: int

    @property
    def total_kv_bytes_in_use(self) -> int:
        """满载时的 KV 占用（必须 <= kv_bytes_available）。"""
        return self.max_concurrent_requests * self.kv_bytes_per_request


def plan_capacity(
    shape: ModelShape,
    gpu_mem_bytes: int,
    dtype: str,
    seq_len: int,
    tp_size: int = 1,
    kv_fraction: float = 0.8,
    block_size: int = 16,
) -> CapacityPlan:
    """算「这张卡能给 KV cache 多少、能并发几个请求」。

    ``kv_fraction`` 是**留给 KV cache 的显存比例**（本函数的本地参数），
    剩下的给权重、激活、CUDA context 和显存碎片。

    这与 vLLM 的 ``--gpu-memory-utilization`` **不是同一件事**，不要混为一谈：
    vLLM 的 ``gpu_memory_utilization(默认 0.9)`` 是「引擎允许占用的整机显存上限」，
    把这个总预算先扣掉权重和激活，剩下的才给 KV cache；而这里的 ``kv_fraction``
    直接就是「KV cache 分到多少」。所以两个数字不能互相代替——vLLM 里 KV 实际能用的
    比例 = ``gpu_memory_utilization − 权重/激活占比``，通常明显小于 0.9。本函数默认
    0.8 是保守的 KV 份额估计，不是 vLLM 的利用率默认值。

    ``block_size`` 默认 16，与 vLLM 的默认 block 大小一致（vllm/config/cache.py
    的 block_size 字段）。容量按**整块**计：块数向上取整（4097 token 需要 257 块
    而不是 256 块），且每请求占用按「块数 × block_size」的整块字节算，
    少算一块或漏掉末块 padding 都会在真实运行时 OOM。
    """
    if gpu_mem_bytes <= 0:
        raise ValueError(f"gpu_mem_bytes must be > 0, got {gpu_mem_bytes}")
    if not 0 < kv_fraction <= 1:
        raise ValueError("kv_fraction must be in (0, 1]")
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    kv_available = int(gpu_mem_bytes * kv_fraction)

    # 块数向上取整；每请求真实占用 = 整块数 × block_size token 的字节（含 padding）。
    blocks_per_request = -(-seq_len // block_size)  # ceil division
    padded_seq_len = blocks_per_request * block_size
    per_request = shape.kv_bytes_for(padded_seq_len, dtype, tp_size)
    if per_request <= 0:
        raise ValueError("per-request KV bytes must be positive")

    concurrency = kv_available // per_request

    return CapacityPlan(
        max_concurrent_requests=concurrency,
        kv_bytes_available=kv_available,
        kv_bytes_per_request=per_request,
        blocks_per_request=blocks_per_request,
        block_size=block_size,
        seq_len=seq_len,
        dtype=dtype,
        tp_size=tp_size,
    )
