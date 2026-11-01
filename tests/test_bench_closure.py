"""bench 闭环与集成测试。

两类：
  A. §5 闭环（确定性）：拿 schedsim 的理论时序（ttft、tpot）合成 SSE 时间线，
     喂给 bench 的解析/度量层，断言 bench 还原出的 ttft/tpot 与 schedsim 相差 <1%。
     —— 用假时间戳，不碰 socket，所以是精确、可复算的。
  B. 集成（真 socket）：起 mock SSE 服务端，用真实 urllib 客户端读流，验证解析
     管线能正确处理「一 chunk 一 token」与「一 chunk 多 token」。容差放宽，只验
     连通性与 token 计数正确性（不拿真实网络时间做精确断言）。
"""

import pytest

from kv_serve_lab.bench import summarize, trace_from_payloads
from kv_serve_lab.bench.client import stream_request
from kv_serve_lab.bench.mock_server import SSEMockServer, chunked_plan, per_token_plan
from kv_serve_lab.schedsim import Request, SimConfig, Simulator


def word_counter(text: str) -> int:
    return len(text.split())


def _synthesize_payloads(start_sec, ttft_sec, tpot_sec, output_len, word="w"):
    """按 schedsim 的理论时序合成「一 token 一 chunk」的带时间戳 payload。"""
    payloads = []
    for i in range(output_len):
        t = start_sec + ttft_sec + i * tpot_sec
        payloads.append((t, '{"choices":[{"text":"%s "}]}' % word))
    return payloads


class TestSchedsimBenchClosure:
    def test_bench_recovers_schedsim_ttft_and_tpot(self):
        """bench 从合成时间线还原的 ttft/tpot 必须贴 schedsim 的理论值（<1%）。"""
        cfg = SimConfig(max_num_batched_tokens=64, max_num_seqs=4, step_sec=0.01)
        reqs = [
            Request("a", prompt_tokens=8, output_tokens=6),
            Request("b", prompt_tokens=20, output_tokens=4),
            Request("c", prompt_tokens=3, output_tokens=10),
        ]
        res = Simulator(cfg).run(reqs, [0.0, 0.02, 0.05])

        traces = []
        for rid, st in res.requests.items():
            out_len = next(r.output_tokens for r in reqs if r.req_id == rid)
            pred_tpot = st.tpot_for(out_len)
            payloads = _synthesize_payloads(
                st.arrival_sec, st.ttft_sec, pred_tpot, out_len
            )
            tr = trace_from_payloads(rid, st.arrival_sec, payloads,
                                     token_counter=word_counter)
            traces.append((tr, st, out_len, pred_tpot))

        for tr, st, out_len, pred_tpot in traces:
            # output_len 必须精确复原
            assert tr.output_len == out_len
            # TTFT 精确复原
            assert tr.ttft_sec == pytest.approx(st.ttft_sec, rel=1e-9)
            # TPOT 复原，且与 schedsim 预测相对误差 < 1%
            assert tr.tpot_sec == pytest.approx(pred_tpot, rel=1e-2)

        summary = summarize([t[0] for t in traces])
        assert summary.num_requests == 3
        assert summary.total_output_tokens == 6 + 4 + 10


@pytest.mark.parametrize('width', [0, -1, 1.5, True])
def test_mock_rejects_invalid_chunk_width_before_loop(width):
    # num_tokens=0 避免旧实现的无限循环；校验顺序先报 chunk width。
    with pytest.raises(ValueError, match='tokens_per_chunk'):
        chunked_plan(num_tokens=0, ttft_sec=0, tpot_sec=0, tokens_per_chunk=width)


class TestMockServerIntegration:
    def test_client_traces_keep_a_shared_clock(self):
        with SSEMockServer(lambda body: per_token_plan(
                num_tokens=2, ttft_sec=0.0, tpot_sec=0.001)) as srv:
            first = stream_request(srv.url, 'a', token_counter=word_counter)
            second = stream_request(srv.url, 'b', token_counter=word_counter)
        assert second.start_sec >= first.end_sec
        summary = summarize([first, second])
        assert summary.wall_sec >= first.latency_sec + second.latency_sec

    def test_one_chunk_one_token(self):
        with SSEMockServer(lambda body: per_token_plan(
                num_tokens=5, ttft_sec=0.0, tpot_sec=0.001)) as srv:
            tr = stream_request(srv.url, "hi", max_tokens=5,
                                token_counter=word_counter)
        assert tr.output_len == 5
        assert tr.chunk_count == 5
        assert tr.ttft_sec >= 0.0

    def test_one_chunk_many_tokens(self):
        """服务端每 chunk 打包 3 个 token：9 token 只需 3 个 chunk，chunk_count=3。"""
        with SSEMockServer(lambda body: chunked_plan(
                num_tokens=9, ttft_sec=0.0, tpot_sec=0.001,
                tokens_per_chunk=3)) as srv:
            tr = stream_request(srv.url, "hi", max_tokens=9,
                                token_counter=word_counter)
        assert tr.output_len == 9          # 数的是 token，不是 chunk
        assert tr.chunk_count == 3
        assert tr.tokens_per_chunk == pytest.approx(3.0)
