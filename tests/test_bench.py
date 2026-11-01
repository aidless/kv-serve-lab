"""bench 测试：流式客户端与指标计算。

核心纪律（来自 vLLM vllm/benchmarks/serve.py:654-658）：
    TPOT = (latency - ttft) / (output_len - 1)

**chunk ≠ token**：output_len 必须由 chunk 携带的 token 数累加、或取服务端
usage.completion_tokens，绝不能用「收到几个 chunk」当 token 数。本文件的
test_chunk_bundling_* 与 test_no_tokenizer_rejects_chunk_counting 就是钉这条。
"""

import math

import pytest

from kv_serve_lab.bench import (
    BenchError,
    RequestTrace,
    TokenArrival,
    summarize,
    trace_from_payloads,
    trace_from_sse,
)


def word_counter(text: str) -> int:
    """测试用 tokenizer：按空白切词。只为让「一个 chunk 多 token」可复算。"""
    return len(text.split())


class TestSingleChunkOneToken:
    def test_ttft_and_tpot_match_hand_calc(self):
        """5 个 chunk 各 1 token，间隔 0.01s，到达 0.5s 后开始。

        首 token 绝对 1.00s -> TTFT = 1.00 - 0.5 = 0.50
        末 token 绝对 1.04s -> latency = 1.04 - 0.5 = 0.54
        TPOT = (0.54 - 0.50) / (5 - 1) = 0.01
        """
        start = 0.5
        payloads = [(1.00, '{"choices":[{"text":"a "}]}')]
        payloads += [(1.01 + i * 0.01, '{"choices":[{"text":"a "}]}') for i in range(4)]
        tr = trace_from_payloads("r", start, payloads, token_counter=word_counter)
        assert tr.output_len == 5
        assert tr.chunk_count == 5
        assert tr.ttft_sec == pytest.approx(0.50)
        assert tr.latency_sec == pytest.approx(0.54)
        assert tr.tpot_sec == pytest.approx(0.01)


class TestChunkBundling:
    def test_two_chunks_many_tokens_not_counted_as_two(self):
        """2 个 chunk 但共 3 个 token：output_len 必须是 3，不是 chunk 数 2。

        chunk0 @1.00s text="aa bb" -> 2 token
        chunk1 @1.03s text="cc"     -> 1 token
        output_len=3；TPOT=(latency-ttft)/(3-1)。
        """
        payloads = [
            (1.00, '{"choices":[{"text":"aa bb "}]}'),
            (1.03, '{"choices":[{"text":"cc"}]}'),
        ]
        tr = trace_from_payloads("r", 0.0, payloads, token_counter=word_counter)
        assert tr.chunk_count == 2
        assert tr.output_len == 3
        assert tr.tokens_per_chunk == pytest.approx(1.5)
        assert tr.tpot_sec == pytest.approx((1.03 - 1.00) / (3 - 1))

    def test_no_tokenizer_rejects_chunk_counting(self):
        """没给 tokenizer 又收到文本 -> 必须报错，绝不退化成「数 chunk」。"""
        payloads = [(1.00, '{"choices":[{"text":"hello world"}]}')]
        with pytest.raises(BenchError):
            trace_from_payloads("r", 0.0, payloads)  # 故意不给 token_counter


class TestChatRoleOnlyChunk:
    def test_role_only_first_chunk_is_not_first_token(self):
        """chat API 首个 chunk 常是 {"delta":{"role":"assistant"}}，无内容。

        它不携带 token，不能算作首 token。TTFT 应落在真正带内容的 chunk 上。
        """
        payloads = [
            (1.00, '{"choices":[{"delta":{"role":"assistant"}}]}'),   # 无 token
            (1.05, '{"choices":[{"delta":{"content":"hi there"}}]}'),  # 2 token
        ]
        tr = trace_from_payloads("r", 0.5, payloads, token_counter=word_counter)
        assert tr.output_len == 2
        assert tr.chunk_count == 1
        assert tr.ttft_sec == pytest.approx(0.55)  # 1.05 - 0.5，不是 1.00 - 0.5


class TestUsageAccounting:
    def test_usage_completion_tokens_drives_output_len(self):
        """服务端只发 usage（累积 completion_tokens），不发文本 token。

        最后一个 chunk usage.completion_tokens=7 -> output_len=7。
        """
        payloads = [
            (1.00, '{"usage":{"completion_tokens":4}}'),
            (1.02, '{"usage":{"completion_tokens":7}}'),
        ]
        tr = trace_from_payloads("r", 0.0, payloads)
        assert tr.output_len == 7
        assert tr.usage_total == 7


class TestTpotEdge:
    def test_single_token_has_no_tpot(self):
        payloads = [(1.00, '{"choices":[{"text":"a "}]}')]
        tr = trace_from_payloads("r", 0.0, payloads, token_counter=word_counter)
        assert tr.output_len == 1
        assert math.isnan(tr.tpot_sec)


class TestSseParsing:
    def test_done_sentinel_stops(self):
        lines = [
            'data: {"choices":[{"text":"a "}]}',
            "",
            "data: [DONE]",
            'data: {"choices":[{"text":"SHOULD_NOT_APPEAR"}]}',
        ]
        tr = trace_from_sse("r", 0.0, lines, clock=lambda: 1.0,
                            token_counter=word_counter)
        assert tr.output_len == 1  # 只统计 [DONE] 之前的

    def test_malformed_json_is_skipped(self):
        lines = [
            "data: {not json}",
            'data: {"choices":[{"text":"ok"}]}',
            "data: [DONE]",
        ]
        tr = trace_from_sse("r", 0.0, lines, clock=lambda: 1.0,
                            token_counter=word_counter)
        assert tr.output_len == 1


class TestSummarize:
    def test_aggregate_over_two_traces(self):
        t1 = RequestTrace(
            req_id="a", start_sec=0.0,
            arrivals=[TokenArrival(0.10, 1), TokenArrival(0.20, 1)],
            usage_total=None,
        )
        t2 = RequestTrace(
            req_id="b", start_sec=0.05,
            arrivals=[TokenArrival(0.15, 1), TokenArrival(0.25, 1)],
            usage_total=None,
        )
        s = summarize([t1, t2])
        assert s.num_requests == 2
        assert s.total_output_tokens == 4
        assert s.mean_ttft_sec == pytest.approx((0.10 + 0.10) / 2)
        # wall = max(end)=0.25 - min(start)=0.0 = 0.25
        assert s.wall_sec == pytest.approx(0.25)
        assert s.output_throughput == pytest.approx(4 / 0.25)


class TestMedianCaliber:
    """偶数个样本时，中位数必须是中间两个的算术平均，不能取上中位数。

    取上中位数会系统性偏大：4 个样本 [0.1, 0.2, 0.3, 0.9] 的真中位数是 0.25，
    而 ttfts[len//2] 给出 0.3 —— 差 20%。在 TTFT 这种右偏分布上，这个偏差
    会随样本量变化而漂移，无法用「反正都是近似」搪塞过去。
    """

    def test_even_count_median_is_average_of_two_middle(self):
        traces = [
            RequestTrace("a", 0.0, [TokenArrival(0.1, 1)], None),
            RequestTrace("b", 0.0, [TokenArrival(0.2, 1)], None),
            RequestTrace("c", 0.0, [TokenArrival(0.3, 1)], None),
            RequestTrace("d", 0.0, [TokenArrival(0.9, 1)], None),
        ]
        s = summarize(traces)
        assert s.median_ttft_sec == pytest.approx(0.25)

    def test_odd_count_median_is_the_middle_one(self):
        traces = [
            RequestTrace("a", 0.0, [TokenArrival(0.1, 1)], None),
            RequestTrace("b", 0.0, [TokenArrival(0.2, 1)], None),
            RequestTrace("c", 0.0, [TokenArrival(0.9, 1)], None),
        ]
        s = summarize(traces)
        assert s.median_ttft_sec == pytest.approx(0.2)

    def test_single_trace_median_equals_that_value(self):
        s = summarize([RequestTrace("a", 0.0, [TokenArrival(0.3, 1)], None)])
        assert s.median_ttft_sec == pytest.approx(0.3)


class TestStreamAccountingBoundaries:
    def test_terminal_usage_needs_no_tokenizer(self):
        tr = trace_from_payloads("r", 0.0, [
            (1.0, '{"choices":[{"text":"hel"}]}'),
            (1.1, '{"choices":[{"text":"lo"}]}'),
            (1.2, '{"usage":{"completion_tokens":1}}'),
        ])
        assert tr.output_len == 1
        assert tr.chunk_count == 2
        assert tr.ttft_sec == 1.0
        assert tr.latency_sec == 1.2

    def test_text_timing_continues_after_early_usage(self):
        tr = trace_from_payloads("r", 0.0, [
            (1.0, '{"choices":[{"text":"a "}],"usage":{"completion_tokens":1}}'),
            (1.1, '{"choices":[{"text":"b"}],"usage":{"completion_tokens":2}}'),
        ])
        assert tr.chunk_count == 2
        assert tr.first_token_sec == 1.0
        assert tr.output_len == 2

    def test_tokenizer_sees_concatenated_text(self):
        # 分词不满足可加性：'hel'/'lo' 各一个 token，'hello' 合起来仍只有一个。
        tr = trace_from_payloads("r", 0.0, [
            (1.0, '{"choices":[{"text":"hel"}]}'),
            (1.1, '{"choices":[{"text":"lo"}]}'),
        ], token_counter=lambda text: len(text.split()))
        assert tr.output_len == 1
        assert tr.chunk_count == 2


class TestUsageInFinalChunkOnly:
    """OpenAI 兼容流的常见形状：正文 chunk 里只有文本，**最后一个** chunk 才带 usage。

    这正是 `stream_options.include_usage=True` 的行为。两件事必须同时成立：
      1. output_len 等于 usage 报的总数，不能把「文本累计」和「usage 总数」加两遍；
      2. TTFT 落在**第一个带文本的 chunk**，不能被末尾那个 usage chunk 拉成整个延迟。
    """

    def test_text_chunks_plus_final_usage_not_double_counted(self):
        """5 个文本 chunk 各 1 token，末尾 usage=5。

        正确：output_len=5。
        错误（旧实现）：文本计 5 + usage 增量 5 = 10，TTFT 变成最后一块的时刻。
        """
        payloads = [
            (1.00, '{"choices":[{"text":"a "}]}'),
            (1.01, '{"choices":[{"text":"a "}]}'),
            (1.02, '{"choices":[{"text":"a "}]}'),
            (1.03, '{"choices":[{"text":"a "}]}'),
            (1.04, '{"choices":[{"text":"a "}]}'),
            (1.05, '{"choices":[],"usage":{"completion_tokens":5}}'),
        ]
        tr = trace_from_payloads("r", 0.5, payloads, token_counter=word_counter)
        assert tr.output_len == 5
        assert tr.ttft_sec == pytest.approx(0.50)   # 1.00 - 0.5，不是 1.05 - 0.5
        assert tr.latency_sec == pytest.approx(0.55)  # 1.05 - 0.5
        assert tr.tpot_sec == pytest.approx((0.55 - 0.50) / (5 - 1))

    def test_usage_disagreeing_with_text_is_reconciled_to_usage(self):
        """文本按空格数出 4 个，但 usage 报 5（tokenizer 粒度不同）-> 以 5 为准。

        时序仍来自文本 chunk（那才是 token 真正到达的时刻）。
        """
        payloads = [
            (1.00, '{"choices":[{"text":"aa bb "}]}'),   # 2
            (1.02, '{"choices":[{"text":"cc dd"}]}'),   # 2
            (1.04, '{"choices":[],"usage":{"completion_tokens":5}}'),
        ]
        tr = trace_from_payloads("r", 0.0, payloads, token_counter=word_counter)
        assert tr.output_len == 5
        assert tr.ttft_sec == pytest.approx(1.00)
        assert tr.chunk_count == 2   # usage-only chunk 不带 token，不算 chunk
