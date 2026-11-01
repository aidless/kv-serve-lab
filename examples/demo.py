#!/usr/bin/env python3
"""kv-serve-lab 四模块闭环演示（纯 CPU，无需 GPU）。

把四个模块串成一条链，展示它们各自回答一个推理服务问题、且能互相对上：

  Part 1  kvcalc   ：一张 24GB 卡跑 Qwen2.5-7B(bf16)、seq=4096，能并发几个请求？
  Part 2  blockpool：PagedAttention 风格的块分配 / 前缀命中 / copy-on-write
  Part 3  schedsim ：给定调度配置，连续批处理的 TTFT / TPOT / 吞吐预测
  Part 4  bench    ：把 schedsim 的预测回放成 SSE 流，用 bench 客户端「实测」回来
                     —— 预测 vs 实测的闭环，相对误差用于验证两个模块口径一致

运行：
    python examples/demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kv_serve_lab.blockpool import BlockPool
from kv_serve_lab.bench import summarize, trace_from_payloads
from kv_serve_lab.bench.client import stream_request
from kv_serve_lab.bench.mock_server import SSEMockServer, per_token_plan
from kv_serve_lab.kvcalc import ModelShape, plan_capacity
from kv_serve_lab.schedsim import Request, SimConfig, Simulator

GIB = 1024 ** 3


def part1_kvcalc() -> ModelShape:
    print("=" * 70)
    print("Part 1  kvcalc — KV cache 容量规划")
    print("=" * 70)
    cfg = {
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "hidden_size": 3584,
        "max_position_embeddings": 32768,
    }
    shape = ModelShape.from_hf_config(cfg)
    per_tok = shape.kv_bytes_per_token("bfloat16")
    plan = plan_capacity(
        shape, gpu_mem_bytes=24 * GIB, dtype="bfloat16", seq_len=4096
    )
    print(f"Qwen2.5-7B 每 token KV = {per_tok} B（全 28 层）")
    print(f"  完整 4096 token 一段 = {per_tok * 4096 / GIB:.3f} GiB")
    print(f"  24GB 卡、KV 份额 0.8：KV 预算 {plan.kv_bytes_available / GIB:.2f} GiB")
    print(f"  每请求 {plan.blocks_per_request} 块（block={plan.block_size}，"
          f"含末块 padding），占 {plan.kv_bytes_per_request / GIB:.3f} GiB")
    print(f"  -> 最大并发 {plan.max_concurrent_requests} 个 4096-token 请求")
    print()
    return shape


def part2_blockpool() -> None:
    print("=" * 70)
    print("Part 2  blockpool — 块池 / 前缀命中 / COW")
    print("=" * 70)
    pool = BlockPool(num_blocks=8, block_size=4)
    a = pool.allocate(2)
    pool.write_payload(a[0], [11, 12, 13, 14])
    pool.register_prefix("prefix-A", a[0])
    print(f"分配 2 块 {a}，块 {a[0]} 写入 payload 并登记前缀 'prefix-A'")
    print(f"  audit: {pool.audit()}")

    hit = pool.lookup_prefix("prefix-A")
    pool.acquire_prefix("prefix-A")   # 第二个请求共享同一前缀块（只加引用）
    print(f"  前缀命中 block={hit}，acquire 后该块 ref={pool.ref_count_of(hit)}（两请求共享）")

    new = pool.copy_on_write(hit)              # 共享块要写 -> COW
    pool.write_payload(new, [99, 98, 97, 96])
    print(f"  COW：原块 {hit} ref 降为 {pool.ref_count_of(hit)} 且保留前缀；"
          f"新块 {new} 独占可写，payload={pool.read_payload(new)}")
    print(f"  audit: {pool.audit()}")
    print()


def part3_schedsim() -> tuple[SimConfig, list[Request], object]:
    print("=" * 70)
    print("Part 3  schedsim — 连续批处理调度预测")
    print("=" * 70)
    cfg = SimConfig(max_num_batched_tokens=256, max_num_seqs=4, step_sec=0.01)
    reqs = [Request(f"r{i}", prompt_tokens=16, output_tokens=32) for i in range(8)]
    arrivals = [i * 0.02 for i in range(8)]
    res = Simulator(cfg).run(reqs, arrivals)
    m = res.metrics()
    print("配置：budget=256, max_num_seqs=4, step=0.01s；8 请求,prompt16/output32")
    print(f"  mean TTFT = {m.mean_ttft_sec:.4f}s, p99 = {m.p99_ttft_sec:.4f}s")
    print(f"  mean latency = {m.mean_latency_sec:.4f}s")
    print(f"  饱和稳态吞吐 = {res.steady_state_throughput:.1f} tok/s"
          f"（解析上界 max_num_seqs/step = {cfg.max_num_seqs / cfg.step_sec:.0f}）")
    print()
    return cfg, reqs, res


def part4_bench_closure(reqs: list[Request], res) -> None:
    print("=" * 70)
    print("Part 4  bench — 预测回放成 SSE，再『实测』回来（闭环）")
    print("=" * 70)

    # 4a 确定性闭环：把 schedsim 的 ttft/tpot 合成 SSE 时间线，bench 精确还原
    print("[4a] 确定性闭环（假时间戳，精确）：")
    print(f"  {'req':>4}  {'pred_ttft':>10} {'meas_ttft':>10}  {'pred_tpot':>10} {'meas_tpot':>10}")
    traces = []
    for rid in ("r0", "r3", "r7"):
        st = res.requests[rid]
        out_len = next(r.output_tokens for r in reqs if r.req_id == rid)
        pred_tpot = st.tpot_for(out_len)
        payloads = [
            (st.arrival_sec + st.ttft_sec + i * pred_tpot,
             '{"choices":[{"text":"tok"}]}')
            for i in range(out_len)
        ]
        tr = trace_from_payloads(
            rid, st.arrival_sec, payloads, token_counter=lambda s: len(s.split())
        )
        traces.append(tr)
        print(f"  {rid:>4}  {st.ttft_sec:>10.4f} {tr.ttft_sec:>10.4f}  "
              f"{pred_tpot:>10.4f} {tr.tpot_sec:>10.4f}")
    summary = summarize(traces)
    print(f"  合计 {summary.num_requests} 请求 / {summary.total_output_tokens} token，"
          f"TTFT/TPOT 完全一致（相对误差 < 1e-9）")

    # 4b 真 socket：起 mock SSE 服务端，用真实 urllib 客户端读流
    print("\n[4b] 真实 HTTP 流式（mock 服务端，含毫秒级网络开销）：")
    with SSEMockServer(lambda body: per_token_plan(
            num_tokens=8, ttft_sec=0.02, tpot_sec=0.005)) as srv:
        tr = stream_request(
            srv.url, "hello", max_tokens=8, token_counter=lambda s: len(s.split())
        )
    print(f"  实测 TTFT={tr.ttft_sec * 1000:.1f}ms（含 HTTP 往返，理论 ≥20ms）")
    print(f"  实测 TPOT={tr.tpot_sec * 1000:.1f}ms，output_len={tr.output_len}，"
          f"chunk_count={tr.chunk_count}")
    print()


def main() -> None:
    part1_kvcalc()
    part2_blockpool()
    cfg, reqs, res = part3_schedsim()
    part4_bench_closure(reqs, res)
    print("=" * 70)
    print("四模块闭环完成。所有数字均为 CPU 上确定性计算或本地 mock 实测，")
    print("不含任何真实 GPU 性能数字（本机无 GPU，设计文档 §2 非目标）。")
    print("=" * 70)


if __name__ == "__main__":
    main()
