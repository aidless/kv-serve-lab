"""四模块串联：容量规划 → 块池生命周期 → 调度时序 → 流式度量。

设计文档 §5 要求这条链是**真的串起来**的：容量算出的并发数决定块池大小与
`max_num_seqs`；调度模拟给出的每个请求的起止时刻驱动块的真实分配与释放；
块池的峰值占用反过来必须装得下规划出的并发。

这一层只做编排，不含新的计算口径——所有数字都来自被它调用的模块。
"""

from __future__ import annotations

from dataclasses import dataclass

from .blockpool import BlockAudit, BlockPool
from .kvcalc import CapacityPlan, ModelShape, plan_capacity
from .schedsim import Metrics, Request, SimConfig, SimResult, Simulator


@dataclass(frozen=True)
class PipelineResult:
    """一次串联运行的完整结果（只读）。"""

    plan: CapacityPlan
    pool_blocks: int
    config: SimConfig
    sim: SimResult
    metrics: Metrics
    audit: BlockAudit
    peak_concurrency: int
    peak_blocks_in_use: int

    def sim_output_tokens(self, req_id: str) -> int:
        """该请求的期望输出 token 数（来自调度结果，不重复硬编码）。"""
        return self.sim._output_lens[req_id]


def run_pipeline(
    shape: ModelShape,
    *,
    gpu_mem_bytes: int,
    dtype: str,
    seq_len: int,
    prompt_tokens: int,
    output_tokens: int,
    num_requests: int,
    step_sec: float,
    kv_fraction: float = 0.8,
    block_size: int = 16,
    max_num_batched_tokens: int | None = None,
) -> PipelineResult:
    """把四个模块串成一次运行。

    参数含义：
    - ``seq_len`` 是容量规划口径（每请求预留多少 token 的 KV）；
    - ``prompt_tokens`` / ``output_tokens`` 是调度模拟口径（单请求实际要算多少）；
    - 块池大小 = 规划并发 × 每请求块数，恰好装下并发上界，不多也不少。
    """
    if not isinstance(num_requests, int) or isinstance(num_requests, bool) or num_requests < 1:
        raise ValueError("num_requests must be a positive integer")
    Request('validation', prompt_tokens, output_tokens)
    if prompt_tokens + output_tokens - 1 > seq_len:
        raise ValueError("seq_len must cover prompt_tokens + output_tokens - 1")

    plan = plan_capacity(
        shape,
        gpu_mem_bytes=gpu_mem_bytes,
        dtype=dtype,
        seq_len=seq_len,
        kv_fraction=kv_fraction,
        block_size=block_size,
    )
    pool_blocks = plan.max_concurrent_requests * plan.blocks_per_request
    pool = BlockPool(num_blocks=pool_blocks, block_size=block_size)

    # 预算默认允许所有并发请求在一步完成 prefill；调用方可降低预算观察分块与排队
    budget = (prompt_tokens * plan.max_concurrent_requests
              if max_num_batched_tokens is None else max_num_batched_tokens)
    config = SimConfig(
        max_num_batched_tokens=budget,
        max_num_seqs=plan.max_concurrent_requests,
        step_sec=step_sec,
    )
    reqs = [
        Request(f"r{i}", prompt_tokens=prompt_tokens, output_tokens=output_tokens)
        for i in range(num_requests)
    ]
    # 全部请求同时到达：最坏情况的排队压力，能暴露并发上限是否真的生效
    arrivals = [0.0] * num_requests
    sim = Simulator(config).run(reqs, arrivals)

    # 回放已完成的调度：步起点分配、步终点释放。同一步先分配后释放。
    # 这是保守整请求预留的生命周期验证，尚未把块分配器嵌入逐 token 调度。
    timeline = sorted(
        [(2 * st.start_step, +1, rid) for rid, st in sim.requests.items()]
        + [(2 * st.finish_step + 1, -1, rid) for rid, st in sim.requests.items()]
    )
    held: dict[str, list[int]] = {}
    peak_concurrency = 0
    peak_blocks_in_use = 0
    for _step, delta, rid in timeline:
        if delta > 0:
            held[rid] = pool.allocate(plan.blocks_per_request)
            peak_concurrency = max(peak_concurrency, len(held))
            peak_blocks_in_use = max(peak_blocks_in_use, pool.audit().used)
        else:
            pool.free(held.pop(rid))

    return PipelineResult(
        plan=plan,
        pool_blocks=pool_blocks,
        config=config,
        sim=sim,
        metrics=sim.metrics(),
        audit=pool.audit(),
        peak_concurrency=peak_concurrency,
        peak_blocks_in_use=peak_blocks_in_use,
    )
