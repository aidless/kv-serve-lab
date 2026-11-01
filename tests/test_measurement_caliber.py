"""口径复核测试：两处「能跑出数字但口径错」的地方。

问题 1（schedsim 变步长模型的到达判定）
    v1 实现把到达时刻 `arrival_sec` 预先换算成步号：
        arrival_step = ceil(arrival_sec / base_step)
    在**等时模型**（每步固定 step_sec）下这个换算是成立的。但在**成本模型**下
    每步耗时随本步 prefill token 数 / decode 请求数变化，第 k 步的墙钟时刻不再是
    k * base_step。此时「用固定 base 换算出的步号」与「真实墙钟」脱钩：
      - 可能**过早**接纳：请求还没到达就被调度（时间倒流）；
      - 可能**过晚**接纳：明明在某步开始前就到了，却被推到后面的步。
    正确判据只能是**墙钟比较**：某步起点 `now` >= 请求到达时刻才接纳。

问题 2（bench 的 usage 与文本混用）
    OpenAI 兼容流里，服务端可能既发 `usage.completion_tokens`（权威计数），
    又在 chunk 里发文本。若两者都拿来计数就会重复；若让文本覆盖 usage，
    则明明服务端给了权威计数却被本地 tokenizer 改写。
    规则必须是：**有 usage 就以 usage 为准**（它是服务端权威计数），
    文本只在没有 usage 时才参与计数（且此时必须有 tokenizer）。
"""

import pytest

from kv_serve_lab.bench import trace_from_payloads
from kv_serve_lab.schedsim import Request, SimConfig, Simulator


def word_counter(text: str) -> int:
    return len(text.split())


class TestArrivalUsesWallClockNotStepDivision:
    def test_late_arrival_admitted_at_first_step_starting_after_it(self):
        """变步长：step 0 耗时 0.15s，请求 0.10s 到达 -> 只能在 step 1 进。

        复算：
          step 0 起点 0.00，跑 20 个 prefill token：
              dur = 0.05 + 20*0.005 = 0.15  ->  step 1 起点 0.15
          请求到达 0.10：
              - 正确：0.10 < 0.15，step 0 起点时还没到；step 1 起点 0.15 >= 0.10 -> start_step=1
              - 旧实现：base=step_overhead=0.05，ceil(0.10/0.05)=2 -> start_step=2（过晚一步）
        """
        cfg = SimConfig(
            max_num_batched_tokens=64,
            max_num_seqs=4,
            step_overhead=0.05,
            t_prefill_token=0.005,
        )
        reqs = [
            Request("r0", prompt_tokens=20, output_tokens=3),
            Request("r1", prompt_tokens=4, output_tokens=3),
        ]
        res = Simulator(cfg).run(reqs, [0.0, 0.10])
        assert res.requests["r0"].start_step == 0
        assert res.requests["r1"].start_step == 1, (
            "r1 在 0.10s 到达，step1 起点 0.15s 才第一次跨过它，应进 step1"
        )
        # r1 的 TTFT 从它自己的到达时刻起算：首 token 在 step1 末
        assert res.requests["r1"].first_token_sec > 0.10
        assert res.requests["r1"].ttft_sec == pytest.approx(
            res.requests["r1"].first_token_sec - 0.10
        )

    def test_request_is_never_admitted_before_it_arrives(self):
        """时间不能倒流：任何请求的首 token时刻 >= 它的到达时刻。"""
        cfg = SimConfig(
            max_num_batched_tokens=32,
            max_num_seqs=4,
            step_overhead=0.02,
            t_prefill_token=0.01,
            t_decode_token=0.002,
        )
        reqs = [Request(f"r{i}", prompt_tokens=8, output_tokens=4) for i in range(6)]
        arrivals = [0.0, 0.03, 0.07, 0.11, 0.2, 0.35]
        res = Simulator(cfg).run(reqs, arrivals)
        for r, t in zip(reqs, arrivals):
            st = res.requests[r.req_id]
            assert st.start_step >= 0
            assert st.first_token_sec >= t, (
                f"{r.req_id} 首 token {st.first_token_sec} 早于到达 {t}（时间倒流）"
            )

    def test_idle_jump_lands_exactly_on_arrival_time(self):
        """空闲期应直接跳到下一个到达时刻，而不是按 base_step 累加。

        变步长模型里 base_step 只是一个「每步固定开销」的猜测，跳步用它会错位。
        这里断言：只有一个请求、它在 0.37s 到达，那它的首 token时刻
        必须 >= 0.37 且整段 total_sec 不含多余的虚假空转。
        """
        cfg = SimConfig(
            max_num_batched_tokens=32,
            max_num_seqs=2,
            step_overhead=0.02,
            t_prefill_token=0.01,
        )
        res = Simulator(cfg).run([Request("r0", 8, 2)], [0.37])
        st = res.requests["r0"]
        assert st.first_token_sec >= 0.37
        # 只有一个请求：prefill 8 token 一步做完 -> dur = 0.02 + 8*0.01 = 0.10
        assert st.first_token_sec == pytest.approx(0.37 + 0.10)
        # 第二个 token 需要一步 decode -> dur = 0.02 + 0*0.01 + 1*0 = 0.02
        assert st.finish_sec == pytest.approx(0.37 + 0.10 + 0.02)


class TestUsageWinsOverText:
    def test_usage_count_is_authoritative_when_text_also_present(self):
        """同一 chunk 既有 usage=3 又有文本 "aa bb"（2 词）-> 以 usage 的 3 为准。

        理由：usage.completion_tokens 是服务端权威计数；本地 tokenizer 只是估算。
        旧实现让文本覆盖 usage，且无 tokenizer 时直接抛错 -> 本测试在旧代码上是红的。
        """
        payloads = [
            (1.00, '{"choices":[{"text":"aa bb"}],"usage":{"completion_tokens":3}}'),
        ]
        tr = trace_from_payloads("r", 0.0, payloads)  # 故意不给 tokenizer
        assert tr.output_len == 3
        assert tr.usage_total == 3

    def test_cumulative_usage_deltas_sum_to_final(self):
        """累积 usage：4 -> 7 表示又出 3 个 token，总长 7。"""
        payloads = [
            (1.00, '{"usage":{"completion_tokens":4}}'),
            (1.02, '{"usage":{"completion_tokens":7}}'),
        ]
        tr = trace_from_payloads("r", 0.0, payloads)
        assert tr.output_len == 7
        assert tr.chunk_count == 2
