"""KV 块压力下的抢占与重算（对齐 vLLM v1 的真实语义）。

vLLM 的抢占判据（vllm/v1/core/sched/scheduler.py:930-976）：

    while True:
        new_blocks = self.kv_cache_manager.allocate_slots(...)
        if new_blocks is not None:
            break
        preempted_req = self.running[-1]      # 默认 FCFS：抢占最后进来的
        ...
        self._preempt_request(preempted_req, ...)
        if preempted_req == request:
            break

`_preempt_request`（:1798）的三件关键事实：
  1. `request.num_computed_tokens = 0` —— 被抢占的请求**从头重算**，
     不是保留已算的 KV（RECOMPUTE 策略，vLLM v1 当前只用这一种）；
  2. `self.waiting.prepend_request(request)` —— 插回等待队列**队首**，
     所以重算的请求优先于同批新请求；
  3. `request.num_preemptions += 1` —— 可观测的抢占次数。

这些测试要证明模拟器不是"假装有抢占"，而是真的按这套语义跑：
重算会让 total_tokens_computed 超过 sum(prompt + output - 1)，
这个差额就是抢占的代价，必须能被独立算出来。
"""

import pytest
from kv_serve_lab.schedsim import Request, SimConfig, Simulator


def _one_shot(reqs, arrival, **cfg):
    return Simulator(SimConfig(step_sec=.01, **cfg)).run(reqs, arrival)


class TestBlockAccountingWithoutPressure:
    """有块上限但不受压时，行为必须与无块模型**完全一致**。"""

    def test_generous_capacity_changes_nothing(self):
        reqs = [Request('a', 32, 4), Request('b', 32, 4)]
        plain = _one_shot(reqs, [0.0, 0.0], max_num_batched_tokens=64, max_num_seqs=2)
        # 2 请求 × (32 prefill + 4 output) ≈ 36 token → 3 块足够
        limited = _one_shot(reqs, [0.0, 0.0], max_num_batched_tokens=64,
                            max_num_seqs=2, kv_num_blocks=16, kv_block_size=16)
        assert limited.total_tokens_computed == plain.total_tokens_computed
        assert limited.requests['a'].latency_sec == plain.requests['a'].latency_sec
        assert limited.total_preemptions == 0

    def test_peak_blocks_reported_and_bounded(self):
        res = _one_shot([Request('a', 32, 4)], [0.0], max_num_batched_tokens=64,
                        max_num_seqs=2, kv_num_blocks=8, kv_block_size=16)
        # 32 prompt + 3 后续输出 token = 35 token → ceil(35/16) = 3 块
        assert res.peak_kv_blocks == 3
        assert max(res.step_kv_blocks) == 3


class TestPreemptionUnderPressure:
    """块不够时必须发生抢占，且抢占代价（重算）要能被独立复算。

    场景 A（本文件多处复用，数字全部手算可复核）：6 块 × 16 = 96 token 容量。
        a = (prompt 64, output 4) @ t=0     峰值 ceil(67/16) = 5 块
        b = (prompt 16, output 2) @ t=0.02  峰值 ceil(17/16) = 2 块
      step0  a 独占 64 token -> 4 块，出第 1 个 token
      step1  a -> 65 -> 5 块，出第 2 个
      step2  a -> 66 -> 5 块；b 被接纳 16 token -> 1 块（5+1=6 恰好装满），出第 1 个
      step3  a -> 67 -> 5 块后完成；b 想从 17 token 扩到 2 块 = 5+2=7 > 6
             -> 抢占 running[-1]，此时 running[-1] 正是 b 自己 -> b 被抢占（重算 16）
      step4  a 已归还块，b 从 0 重算到 17，完成
    合计计算量 = 67 + 16 + 17 = 100；理想值 = 67 + 17 = 84；差额 16 = 重算量。
    """

    def test_tight_cache_forces_preemption_and_everyone_finishes(self):
        reqs = [Request('a', 64, 4), Request('b', 16, 2)]
        res = _one_shot(reqs, [0.0, 0.02], max_num_batched_tokens=64,
                        max_num_seqs=2, kv_num_blocks=6, kv_block_size=16)
        assert res.total_preemptions == 1
        assert res.requests['b'].num_preemptions == 1
        assert res.requests['a'].num_preemptions == 0
        assert len(res.requests) == 2
        for st in res.requests.values():
            assert st.finish_sec > st.first_token_sec

    def test_peak_blocks_never_exceeds_capacity_under_pressure(self):
        """4 个各需 3 块的请求挤 5 块容量：全程不得超过上限，且必须全部跑完。"""
        reqs = [Request(f'r{i}', 40, 6) for i in range(4)]
        res = _one_shot(reqs, [0.0] * 4, max_num_batched_tokens=64,
                        max_num_seqs=4, kv_num_blocks=5, kv_block_size=16)
        assert max(res.step_kv_blocks) <= 5
        assert res.peak_kv_blocks <= 5
        assert res.total_preemptions >= 1
        assert len(res.requests) == 4

    def test_recompute_cost_is_visible_in_total_tokens(self):
        """重算的代价：总计算量 > sum(prompt + output - 1)。差额必须可独立复算。

        同场景 A 再加一个 c=(16,2) @0.02：b、c 都被抢占一次，各重算 16 token。
        合计 = 67 + (16+17) + (16+17) = 133；理想 = 67 + 17 + 17 = 101；差额 32。
        """
        reqs = [Request('a', 64, 4), Request('b', 16, 2), Request('c', 16, 2)]
        res = _one_shot(reqs, [0.0, 0.02, 0.02], max_num_batched_tokens=64,
                        max_num_seqs=3, kv_num_blocks=6, kv_block_size=16)
        ideal = sum(r.prompt_tokens + r.output_tokens - 1 for r in reqs)
        assert res.total_preemptions == 2
        assert res.recompute_tokens == 32
        assert res.total_tokens_computed == 133
        # 差额应当正好等于「被抢占请求重算的 token 数」——两条独立累加路径互证
        assert res.total_tokens_computed - ideal == res.recompute_tokens
        assert ideal == 101

    def test_victim_is_the_most_recently_admitted(self):
        """默认 FCFS 策略抢占 running[-1]，即最后被接纳的那个。

        用场景 A/C 的相同负载：最早进来的 a 一次都没被抢占，后进来的 b、c
        各被抢占一次。若抢占的是队首（a），这条断言会立刻反过来。
        """
        reqs = [Request('a', 64, 4), Request('b', 16, 2), Request('c', 16, 2)]
        res = _one_shot(reqs, [0.0, 0.02, 0.02], max_num_batched_tokens=64,
                        max_num_seqs=3, kv_num_blocks=6, kv_block_size=16)
        assert res.requests['a'].num_preemptions == 0
        assert res.requests['b'].num_preemptions == 1
        assert res.requests['c'].num_preemptions == 1

    def test_preempted_request_restarts_from_scratch(self):
        """抢占只影响「还没产出首 token」的那些请求的 TTFT —— 两个方向都要验。

        对照组容量充足（30 块），三个请求都不被抢占；实验组 5 块，必然抢占。
        同一批请求、同一预算，只有容量不同，所以 TTFT 的差异只能归因于抢占重算。
          - r0 在产出首 token **之后**才被抢占：它的 TTFT 必须与对照组**完全相同**，
            否则说明实现把重算成本错误地算进了已经发生过的首 token 时刻；
          - 至少有一个请求在首 token **之前**被抢占：它的 TTFT 必须晚于对照组，
            这正是「重算要从头再来」的可观测痕迹。
        """
        reqs = [Request(f'r{i}', 40, 6) for i in range(3)]
        arrivals = [0.0, 0.0, 0.0]
        base = dict(max_num_batched_tokens=64, max_num_seqs=3, kv_block_size=16)
        roomy = _one_shot(reqs, arrivals, kv_num_blocks=30, **base)
        tight = _one_shot(reqs, arrivals, kv_num_blocks=5, **base)
        assert roomy.total_preemptions == 0
        assert tight.total_preemptions > 0

        assert tight.requests['r0'].num_preemptions > 0
        assert tight.requests['r0'].first_token_step == 0
        assert tight.requests['r0'].ttft_sec == pytest.approx(
            roomy.requests['r0'].ttft_sec)

        inflated = [k for k, v in tight.requests.items()
                    if v.num_preemptions > 0
                    and v.ttft_sec > roomy.requests[k].ttft_sec]
        assert inflated, "expected a request preempted before its first token"
        for rid in inflated:
            assert tight.requests[rid].ttft_sec > roomy.requests[rid].ttft_sec


class TestCapacityErrors:
    def test_single_request_larger_than_cache_raises(self):
        """一个请求自己就超过整块缓存：抢占也腾不出空间，必须明确报错而不是死循环。"""
        with pytest.raises(ValueError, match='cannot fit'):
            _one_shot([Request('huge', 200, 4)], [0.0], max_num_batched_tokens=64,
                      max_num_seqs=2, kv_num_blocks=4, kv_block_size=16)

    def test_block_size_and_capacity_validated(self):
        for kwargs in ({'kv_num_blocks': 0, 'kv_block_size': 16},
                       {'kv_num_blocks': 4, 'kv_block_size': 0},
                       {'kv_num_blocks': 4.5, 'kv_block_size': 16},
                       {'kv_num_blocks': True, 'kv_block_size': 16}):
            with pytest.raises(ValueError):
                SimConfig(max_num_batched_tokens=64, max_num_seqs=2,
                          step_sec=.01, **kwargs)


class TestPreemptionOffByDefault:
    def test_no_capacity_means_no_block_accounting(self):
        res = _one_shot([Request('a', 200, 4)], [0.0],
                        max_num_batched_tokens=64, max_num_seqs=2)
        assert res.total_preemptions == 0
        assert res.step_kv_blocks == []
        assert res.peak_kv_blocks == 0
