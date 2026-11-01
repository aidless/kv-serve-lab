"""四模块串联：kvcalc 的容量 → blockpool 的块生命周期 → schedsim 的时序 → bench 的度量。

设计文档 §5 要求这条链是**真的串起来**的，不是四段各自硬编码常数：
容量算出的并发数决定块池大小与 `max_num_seqs`，调度模拟给出的 start/finish 时刻
驱动块的实际分配与释放，最后由 bench 把预测回放成 SSE 再测回来。
"""

from __future__ import annotations

import pytest

from kv_serve_lab.bench import trace_from_payloads
from kv_serve_lab.integration import PipelineResult, run_pipeline
from kv_serve_lab.kvcalc import ModelShape

GIB = 1024 ** 3

QWEN25_7B = {
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "hidden_size": 3584,
    "max_position_embeddings": 32768,
}


def _shape() -> ModelShape:
    return ModelShape.from_hf_config(QWEN25_7B)


class TestCapacityDrivesPoolSize:
    def test_pool_holds_exactly_the_planned_concurrency(self):
        """块池大小必须由容量规划推出，且恰好装下并发上界。"""
        r = run_pipeline(
            _shape(), gpu_mem_bytes=24 * GIB, dtype="bfloat16",
            seq_len=4096, prompt_tokens=4096, output_tokens=1,
            num_requests=40, step_sec=0.01,
        )
        assert r.pool_blocks == r.plan.max_concurrent_requests * r.plan.blocks_per_request
        assert r.pool_blocks == 87 * 256

    def test_scheduler_concurrency_comes_from_capacity(self):
        """max_num_seqs 必须等于规划出的并发数，不能自己另写一个常数。"""
        r = run_pipeline(
            _shape(), gpu_mem_bytes=24 * GIB, dtype="bfloat16",
            seq_len=4096, prompt_tokens=4096, output_tokens=1,
            num_requests=40, step_sec=0.01,
        )
        assert r.config.max_num_seqs == r.plan.max_concurrent_requests


class TestNoBlockLeak:
    def test_every_block_is_returned_at_the_end(self):
        r = run_pipeline(
            _shape(), gpu_mem_bytes=24 * GIB, dtype="bfloat16",
            seq_len=4096, prompt_tokens=4096, output_tokens=1,
            num_requests=40, step_sec=0.01,
        )
        assert r.audit.used == 0
        assert r.audit.free == r.pool_blocks

    def test_peak_usage_never_exceeds_pool(self):
        r = run_pipeline(
            _shape(), gpu_mem_bytes=24 * GIB, dtype="bfloat16",
            seq_len=4096, prompt_tokens=4096, output_tokens=1,
            num_requests=40, step_sec=0.01,
        )
        assert r.peak_blocks_in_use <= r.pool_blocks
        assert r.peak_blocks_in_use == (
            r.peak_concurrency * r.plan.blocks_per_request
        )

    def test_peak_concurrency_matches_capacity(self):
        """40 个请求、并发上限 87：峰值并发就是全部请求同时在跑。"""
        r = run_pipeline(
            _shape(), gpu_mem_bytes=24 * GIB, dtype="bfloat16",
            seq_len=4096, prompt_tokens=4096, output_tokens=1,
            num_requests=40, step_sec=0.01,
        )
        assert r.peak_concurrency == 40


class TestRequestsBeyondCapacityQueue:
    def test_requests_exceeding_capacity_still_all_finish(self):
        """请求数超过并发上界时必须排队跑完，不能丢请求、也不能超发块。"""
        r = run_pipeline(
            _shape(), gpu_mem_bytes=2 * GIB, dtype="bfloat16",
            seq_len=4096, prompt_tokens=4096, output_tokens=1,
            num_requests=30, step_sec=0.01,
        )
        assert r.plan.max_concurrent_requests < 30
        assert r.metrics.completed == 30
        assert r.audit.used == 0
        assert r.peak_concurrency <= r.plan.max_concurrent_requests


class TestBenchRecoversPipelinePrediction:
    def test_sse_replay_of_pipeline_timeline(self):
        """把整条链算出的时序回放成 SSE，bench 必须还原出同样的 output_len。"""
        r = run_pipeline(
            _shape(), gpu_mem_bytes=24 * GIB, dtype="bfloat16",
            seq_len=4096, prompt_tokens=4094, output_tokens=3,
            num_requests=5, step_sec=0.01,
        )
        traces = []
        for rid, st in r.sim.requests.items():
            out_len = r.sim_output_tokens(rid)
            payloads = [
                (st.arrival_sec + st.ttft_sec + i * st.tpot_for(out_len),
                 '{"choices":[{"text":"tok "}]}')
                for i in range(out_len)
            ]
            traces.append(trace_from_payloads(
                rid, st.arrival_sec, payloads, token_counter=lambda s: len(s.split())
            ))
        assert [t.output_len for t in traces] == [3] * 5
        for t in traces:
            assert t.ttft_sec == pytest.approx(
                r.sim.requests[t.req_id].ttft_sec, rel=1e-9
            )


class TestValidation:
    def test_rejects_workload_larger_than_reserved_kv(self):
        with pytest.raises(ValueError, match='seq_len'):
            run_pipeline(_shape(), gpu_mem_bytes=24*GIB, dtype='bfloat16',
                         seq_len=16, prompt_tokens=16, output_tokens=3,
                         num_requests=1, step_sec=.01)

    def test_rejects_nonpositive_request_count(self):
        with pytest.raises(ValueError):
            run_pipeline(
                _shape(), gpu_mem_bytes=24 * GIB, dtype="bfloat16",
                seq_len=4096, prompt_tokens=4096, output_tokens=1,
                num_requests=0, step_sec=0.01,
            )

    def test_result_is_frozen_dataclass(self):
        assert PipelineResult.__dataclass_params__.frozen
