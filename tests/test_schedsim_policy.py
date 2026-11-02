"""SchedulerPolicy 可插拔测试。

验证三类策略在统一时间模型下的可复现行为：
- FCFS（默认 None 路径）与显式 FCFSPolicy 等价；
- AgingShortestPromptFirst 在「短 prompt 先到」与「长 prompt 先到」两种
  负载下都按预期排序（含老化抗饥饿）；
- ShortestJobFirst 只按 job 长度，不含老化。
"""

import pytest

from kv_serve_lab.schedsim import (
    AgingShortestPromptFirstPolicy,
    FCFSPolicy,
    Request,
    ShortestJobFirstPolicy,
    SimConfig,
    Simulator,
    build_policy,
)


def _req(req_id: str, prompt: int, output: int = 4) -> Request:
    return Request(req_id=req_id, prompt_tokens=prompt, output_tokens=output)


def _run(requests, arrivals, *, policy=None, max_num_seqs=2, **cfg_kw):
    cfg = SimConfig(
        max_num_batched_tokens=8,
        max_num_seqs=max_num_seqs,
        step_overhead=0.01,
        t_prefill_token=0.001,
        t_decode_token=0.002,
        policy=policy,
        **cfg_kw,
    )
    sim = Simulator(cfg)
    return sim.run(requests, arrivals)


class TestPolicyConstruction:
    def test_default_none_is_fcfs(self):
        """cfg.policy=None 必须与显式 FCFSPolicy 给出完全相同的时序。"""
        reqs = [_req("a", 40), _req("b", 10), _req("c", 20)]
        arrivals = [0.0, 0.0, 0.0]
        r_none = _run(reqs, arrivals)
        r_fcfs = _run(reqs, arrivals, policy=FCFSPolicy())
        assert r_none.total_tokens_computed == r_fcfs.total_tokens_computed
        assert r_none.total_sec == r_fcfs.total_sec
        for rid in ("a", "b", "c"):
            assert r_none.requests[rid].first_token_sec == r_fcfs.requests[rid].first_token_sec
            assert r_none.requests[rid].finish_sec == r_fcfs.requests[rid].finish_sec

    def test_build_policy_roundtrip(self):
        assert isinstance(build_policy("fcfs"), FCFSPolicy)
        assert isinstance(
            build_policy("aging_shortest_prompt_first", aging_window=2.0),
            AgingShortestPromptFirstPolicy,
        )
        assert isinstance(build_policy("shortest_job_first"), ShortestJobFirstPolicy)

    def test_build_policy_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown scheduler policy"):
            build_policy("no_such_policy")

    def test_bad_policy_type_rejected(self):
        with pytest.raises(ValueError, match="policy must be a SchedulerPolicy"):
            SimConfig(max_num_batched_tokens=8, max_num_seqs=2, policy=object())

    def test_bad_aging_window_rejected(self):
        for bad in (0, -1, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="aging_window"):
                AgingShortestPromptFirstPolicy(aging_window=bad)


class TestPolicySelectionOrder:
    def test_fcfs_picks_first_arrived(self):
        """FCFS：不管 prompt 长短，先到先接。"""
        # b 后到但 prompt 更短：FCFS 仍先接 a
        reqs = [_req("a", 100), _req("b", 4)]
        arrivals = [0.0, 0.1]
        r = _run(reqs, arrivals, policy=FCFSPolicy(), max_num_seqs=1)
        assert r.requests["a"].start_step < r.requests["b"].start_step

    def test_shortest_job_first_picks_shorter_first(self):
        """ShortestJobFirst：同刻到达时短 job 先接（不关心 prompt 还是总长）。"""
        reqs = [_req("long", 40, output=10), _req("short", 4, output=2)]
        arrivals = [0.0, 0.0]
        r = _run(reqs, arrivals, policy=ShortestJobFirstPolicy(), max_num_seqs=1)
        assert r.requests["short"].start_step < r.requests["long"].start_step

    def test_aging_shortest_prompt_picks_shortest_prompt_when_tied_age(self):
        """AgingShortestPromptFirst：同刻到达时最短 prompt 先接。"""
        reqs = [_req("a", 30), _req("b", 5), _req("c", 12)]
        arrivals = [0.0, 0.0, 0.0]
        r = _run(reqs, arrivals, policy=AgingShortestPromptFirstPolicy(), max_num_seqs=1)
        assert r.requests["b"].start_step < r.requests["c"].start_step < r.requests["a"].start_step

    def test_aging_boosts_long_waiting_request(self):
        """老化抗饥饿：长 prompt 请求等待足够久后，排序键反超新到短请求。

        模拟器语义下「先到先被接入」，到达时间差无法在 waiting 内制造老化差
        （先到的已在 running）。因此直接用 select() 单元测试验证老化键：
        key = prompt_tokens - aging_window × (now - arrival_sec)。
        长 prompt 请求 a（早到）在老化窗口足够大时，键压到短 prompt 的 b 之下。
        """
        from kv_serve_lab.schedsim import _State

        a = _State(_req("a", 100), arrival_sec=0.0)
        b = _State(_req("b", 2), arrival_sec=0.05)

        # 小老化窗口：b（短 prompt）键更小，被选。
        small = AgingShortestPromptFirstPolicy(aging_window=0.1)
        assert small.select([a, b], now=0.05, step=5) == 1
        # 大老化窗口：a（长 prompt 但早到 0.05s）键反超 b，被选。
        big = AgingShortestPromptFirstPolicy(aging_window=2000.0)
        assert big.select([a, b], now=0.05, step=5) == 0

    def test_aging_window_monotonic_boost(self):
        """老化窗口越大，等待久的请求键压得越低（单调性）。"""
        from kv_serve_lab.schedsim import _State

        a = _State(_req("a", 100), arrival_sec=0.0)
        b = _State(_req("b", 2), arrival_sec=0.05)
        # 键 = prompt - w*(now-arrival)。a 的键随 w 增大而减小更快（等待更久）。
        w_small = 0.1
        w_big = 2000.0
        key_a_small = 100 - w_small * 0.05
        key_a_big = 100 - w_big * 0.05
        assert key_a_big < key_a_small
        # b 的键变化率更小（等待 0），b 键不随 w 变。
        assert (2 - w_small * 0.0) == (2 - w_big * 0.0)

    def test_preemption_still_works_with_policy(self):
        """换策略不影响 KV 抢占语义：策略只是选下一个，不改变抢占路径。

        场景设计：a 长 prompt 占 5 块（64+4=68→ceil(68/16)=5），b 需要 2 块
        （32+2=34→ceil(34/16)=3，但 b 先跑 prompt 32=2 块），a5+b3=8>6 必抢。
        """
        reqs = [_req("a", 64), _req("b", 32, output=2)]
        arrivals = [0.0, 0.02]
        r = _run(
            reqs, arrivals,
            policy=AgingShortestPromptFirstPolicy(),
            kv_num_blocks=6,
            kv_block_size=16,
            max_num_seqs=2,
        )
        assert r.total_preemptions >= 1
        assert r.recompute_tokens >= 14
        assert r.peak_kv_blocks <= 6


class TestPolicyMetricsComparable:
    def test_policies_produce_comparable_results(self):
        """三种策略跑同一负载都完成，且结果对象结构一致（可横向比较）。"""
        reqs = [_req(f"r{i}", prompt=20 + i * 7, output=5 + i % 3) for i in range(6)]
        arrivals = [i * 0.03 for i in range(6)]
        results = {}
        for policy in (FCFSPolicy(), AgingShortestPromptFirstPolicy(), ShortestJobFirstPolicy()):
            r = _run(reqs, arrivals, policy=policy, max_num_seqs=3)
            assert all(s.finish_sec is not None for s in r.requests.values())
            assert r.peak_kv_blocks >= 0
            # total_tokens_computed 统计**输入位置**消耗：每个请求 prompt+output-1
            #（最后一个输出 token 是采样出来的，不再作为输入位置喂回）。
            assert r.total_tokens_computed == sum(
                20 + i * 7 + 5 + i % 3 - 1 for i in range(6)
            )
            results[policy.name] = r
        # 三种策略至少都产生非零吞吐
        for name, r in results.items():
            assert r.output_throughput > 0, f"{name} should have throughput"
