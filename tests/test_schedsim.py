"""schedsim 测试：continuous batching 调度模拟器。

模型直接镜像 vLLM v1 的调度语义（vllm/v1/core/sched/scheduler.py:700-708 原注释）：

    "There's no 'decoding phase' nor 'prefill phase' in the scheduler.
     Each request just has the num_computed_tokens and num_tokens_with_spec.
     At each step, the scheduler tries to assign tokens to the requests so that
     each request's num_computed_tokens can catch up its num_tokens_with_spec."

即：**统一 token 预算**，不是 prefill/decode 两阶段。每个调度步：
  1. 预算 = max_num_batched_tokens
  2. 在跑的请求优先拿 token（decode，每个请求 1 token）
  3. 剩下的预算给新请求的 prefill（chunked prefill：一个请求可以跨多个步）
  4. 同时受 max_num_seqs 限制

饱和吞吐的解析上界 = max_num_seqs / step_sec（每步每请求出 1 token）。
模拟值必须贴这个上界，否则说明调度器写错了。
"""

import pytest

from kv_serve_lab.schedsim import (
    Request,
    SimConfig,
    Simulator,
)


class TestRequestValidation:
    def test_rejects_zero_prompt(self):
        with pytest.raises(ValueError):
            Request(req_id="r1", prompt_tokens=0, output_tokens=4)

    def test_rejects_zero_output(self):
        with pytest.raises(ValueError):
            Request(req_id="r1", prompt_tokens=4, output_tokens=0)


class TestSingleRequest:
    def test_ttft_is_ceil_of_prompt_over_budget(self):
        """prompt 100 token，预算 32 -> 需要 4 步 prefill 完，末步出首 token。

        复算：ceil(100/32) = 4 步；TTFT = 4 * step_sec = 0.04
        """
        cfg = SimConfig(max_num_batched_tokens=32, max_num_seqs=8, step_sec=0.01)
        sim = Simulator(cfg)
        res = sim.run([Request("r1", prompt_tokens=100, output_tokens=5)], [0.0])
        req = res.requests["r1"]
        assert req.ttft_steps == 4
        assert req.ttft_sec == pytest.approx(0.04)

    def test_total_latency_includes_all_output_tokens(self):
        """100 prompt + 5 output，预算 32：4 步 prefill + 4 步 decode = 8 步。

        与 vLLM 一致：prefill 的最后一步**同时**采样出首个输出 token，
        所以首 token 不额外占一步。之后每个 decode 步出 1 token，
        剩余 4 个 token 需要 4 步。步 index 0..7，finish_step = 7。

        这条是 vLLM serve.py:656 的 TPOT 公式
        `tpot = (latency - ttft) / (output_len - 1)` 能成立的前提：
        output_len 个 token 里有 1 个是 TTFT 那步产出的，剩下 output_len-1 个
        摊到后续步，分母才正好是 output_len-1。
        """
        cfg = SimConfig(max_num_batched_tokens=32, max_num_seqs=8, step_sec=0.01)
        res = Simulator(cfg).run(
            [Request("r1", prompt_tokens=100, output_tokens=5)], [0.0]
        )
        req = res.requests["r1"]
        assert req.finish_step == 7
        assert req.latency_sec == pytest.approx(0.08)

    def test_tpot_excludes_first_token(self):
        """TPOT = (latency - ttft) / (output_len - 1)，与 vLLM serve.py:656 一致。

        latency 0.08，ttft 0.04，output 5 -> (0.08-0.04)/(5-1) = 0.01
        """
        cfg = SimConfig(max_num_batched_tokens=32, max_num_seqs=8, step_sec=0.01)
        res = Simulator(cfg).run(
            [Request("r1", prompt_tokens=100, output_tokens=5)], [0.0]
        )
        assert res.per_request_tpot("r1") == pytest.approx(0.01)


class TestBatching:
    def test_two_requests_share_a_prefill_step(self):
        """两个 20-token prompt，预算 64 -> 一步就能 prefill 完两个。"""
        cfg = SimConfig(max_num_batched_tokens=64, max_num_seqs=8, step_sec=0.01)
        reqs = [
            Request("a", prompt_tokens=20, output_tokens=3),
            Request("b", prompt_tokens=20, output_tokens=3),
        ]
        res = Simulator(cfg).run(reqs, [0.0, 0.0])
        assert res.requests["a"].ttft_steps == 1
        assert res.requests["b"].ttft_steps == 1

    def test_max_num_seqs_caps_concurrency(self):
        """预算够但 max_num_seqs=2 -> 第 3 个请求必须排队，slot 空了才进。

        断言落在 start_step 上（这才是「并发上限」的直接后果），
        不用 ttft 差值：a/b 都在 1 步内 prefill 完，c 也是 1 步，差值会是 0。
        """
        cfg = SimConfig(max_num_batched_tokens=1024, max_num_seqs=2, step_sec=0.01)
        reqs = [
            Request("a", prompt_tokens=10, output_tokens=2),
            Request("b", prompt_tokens=10, output_tokens=2),
            Request("c", prompt_tokens=10, output_tokens=2),
        ]
        res = Simulator(cfg).run(reqs, [0.0, 0.0, 0.0])
        # a/b 在 step 0 进；各自 1 步 prefill + 1 步 decode，在 step 1 结束，
        # slot 在 step 1 末尾才释放，所以 c 只能在 step 2 进。
        assert res.requests["a"].start_step == 0
        assert res.requests["b"].start_step == 0
        assert res.requests["c"].start_step == 2

    def test_decode_has_priority_over_new_prefill(self):
        """在跑的请求先拿 token，新请求只能用剩下的预算。

        预算 16，10 个已在跑的请求各吃 1 token（decode），只剩 6 给新请求；
        新请求 prompt 12，按每步 6 token 需要 2 步 prefill 完。

        新请求在 step 1 才到达，这样「挤占」是唯一变量：
        max_num_seqs=20 不构成限制，卡住它的只能是预算。
        """
        cfg = SimConfig(max_num_batched_tokens=16, max_num_seqs=20, step_sec=0.01)
        reqs = [
            Request(f"r{i}", prompt_tokens=1, output_tokens=50) for i in range(10)
        ]
        reqs.append(Request("new", prompt_tokens=12, output_tokens=2))
        sim = Simulator(cfg)
        # 前 10 个在 step 0 到达；new 在 step 1 到达（0.01s）
        res = sim.run(reqs, [0.0] * 10 + [0.01])
        # 若 prefill 有绝对优先权，new 在 step 1 就能一次吃完 12 个 token
        # 并在同一步采样出首 token -> ttft_steps=1；
        # 被 decode 挤占后每步只能拿 6 个，要跨 2 步 -> ttft_steps=2。
        assert res.requests["new"].ttft_steps == 2


class TestChunkedPrefill:
    def test_long_prompt_is_chunked_across_steps(self):
        """prompt 1000，预算 128 -> 8 步 prefill。"""
        cfg = SimConfig(max_num_batched_tokens=128, max_num_seqs=4, step_sec=0.01)
        res = Simulator(cfg).run(
            [Request("r1", prompt_tokens=1000, output_tokens=2)], [0.0]
        )
        assert res.requests["r1"].ttft_steps == 8


class TestArrivalTimes:
    def test_late_arrival_delays_start(self):
        """请求在 step 5 才到，它的第一步不能早于 step 5。"""
        cfg = SimConfig(max_num_batched_tokens=64, max_num_seqs=4, step_sec=0.01)
        res = Simulator(cfg).run(
            [Request("r1", prompt_tokens=8, output_tokens=2)], [0.05]
        )
        req = res.requests["r1"]
        assert req.start_step == 5
        # TTFT 从「请求到达」算，不是从 t=0 算：首 token 在第 6 步结束（绝对 0.06s），
        # 到达在 0.05s，所以 TTFT = 0.06 - 0.05 = 0.01（本步无排队，1 步 prefill 出首 token）。
        assert req.ttft_sec == pytest.approx(0.01)

    def test_arrival_count_must_match_request_count(self):
        cfg = SimConfig(max_num_batched_tokens=64, max_num_seqs=4, step_sec=0.01)
        with pytest.raises(ValueError):
            Simulator(cfg).run([Request("r1", 8, 2)], [0.0, 0.1])


class TestSaturationThroughput:
    """饱和吞吐必须贴解析上界 max_num_seqs / step_sec。"""

    def test_saturated_throughput_matches_analytic_bound(self):
        """20 个长输出请求，max_num_seqs=8，step 0.01s。

        进入稳态后每步出 8 token -> 800 token/s。
        测量窗口取「全部请求都已完成 prefill 之后、第一个请求完成之前」。
        """
        cfg = SimConfig(max_num_batched_tokens=256, max_num_seqs=8, step_sec=0.01)
        reqs = [
            Request(f"r{i}", prompt_tokens=8, output_tokens=100) for i in range(20)
        ]
        res = Simulator(cfg).run(reqs, [0.0] * 20)

        bound = cfg.max_num_seqs / cfg.step_sec  # 800 token/s
        assert res.steady_state_throughput <= bound + 1e-9
        # 允许 5% 偏差（请求完成/新请求进入的边界效应）
        assert res.steady_state_throughput >= bound * 0.95, (
            f"throughput {res.steady_state_throughput:.1f} far below bound {bound}"
        )

    def test_low_concurrency_throughput_is_lower(self):
        """只有 2 个请求时，吞吐受请求数限制，达不到 8 并发的水准。"""
        cfg = SimConfig(max_num_batched_tokens=256, max_num_seqs=8, step_sec=0.01)
        few = Simulator(cfg).run(
            [Request(f"r{i}", prompt_tokens=8, output_tokens=100) for i in range(2)],
            [0.0, 0.0],
        )
        many = Simulator(cfg).run(
            [Request(f"r{i}", prompt_tokens=8, output_tokens=100) for i in range(20)],
            [0.0] * 20,
        )
        assert few.steady_state_throughput < many.steady_state_throughput


class TestConservation:
    def test_all_requests_complete(self):
        cfg = SimConfig(max_num_batched_tokens=64, max_num_seqs=4, step_sec=0.01)
        reqs = [Request(f"r{i}", prompt_tokens=10, output_tokens=5) for i in range(10)]
        res = Simulator(cfg).run(reqs, [0.0] * 10)
        assert set(res.requests) == {r.req_id for r in reqs}
        assert all(r.finish_step > 0 for r in res.requests.values())

    def test_total_tokens_conserved(self):
        """调度器不能丢 token。

        口径：`total_tokens_computed` 统计**被送进模型计算的输入 token**。
        每个请求贡献 prompt_tokens + (output_tokens - 1)：
        首个输出 token 由 prefill 末步的采样产生，之后每个 decode 步
        用 1 个已采样 token 作为输入换来 1 个新 token，共 output_tokens-1 个。
        """
        cfg = SimConfig(max_num_batched_tokens=64, max_num_seqs=4, step_sec=0.01)
        reqs = [
            Request("a", prompt_tokens=10, output_tokens=5),
            Request("b", prompt_tokens=100, output_tokens=3),
        ]
        res = Simulator(cfg).run(reqs, [0.0, 0.0])
        expected = sum(r.prompt_tokens + r.output_tokens - 1 for r in reqs)
        assert res.total_tokens_computed == expected


class TestMetrics:
    def test_mean_ttft_and_p99(self):
        cfg = SimConfig(max_num_batched_tokens=32, max_num_seqs=2, step_sec=0.01)
        reqs = [Request(f"r{i}", prompt_tokens=32, output_tokens=4) for i in range(5)]
        res = Simulator(cfg).run(reqs, [0.0] * 5)
        m = res.metrics()
        assert m.mean_ttft_sec > 0
        assert m.p99_ttft_sec >= m.mean_ttft_sec
        assert m.completed == 5
