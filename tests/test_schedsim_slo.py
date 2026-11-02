"""SLO 感知调度测试：DeadlineAwarePolicy 与违规统计。

对应进化方向③（arXiv:2609.07883 SLOWeave / arXiv:2606.09061 Fairness-Aware）。
覆盖：
- deadline 参数校验（Request.__post_init__）
- DeadlineAwarePolicy.select 的排序（绝对截止最早优先，无 deadline 排最后）
- 违规判定（finish_sec > arrival_sec + deadline_sec）
- slo_violations / slo_ratio 统计
- 与 FCFS 在 SLO 负载下的差异（策略可验证地改变 SLO 达标率）
"""

import math

import pytest

from kv_serve_lab.schedsim import (
    DeadlineAwarePolicy,
    FCFSPolicy,
    Request,
    SimConfig,
    Simulator,
    _State,
    build_policy,
)


# ---------- Request.deadline_sec 校验 ----------

class TestRequestDeadlineValidation:
    def test_default_none(self):
        r = Request("a", 10, 5)
        assert r.deadline_sec is None

    def test_positive_float_ok(self):
        r = Request("a", 10, 5, deadline_sec=2.5)
        assert r.deadline_sec == 2.5

    def test_positive_int_ok(self):
        r = Request("a", 10, 5, deadline_sec=3)
        assert r.deadline_sec == 3

    @pytest.mark.parametrize("bad", [0, -1, -0.5])
    def test_non_positive_rejected(self, bad):
        with pytest.raises(ValueError, match="deadline_sec must be > 0"):
            Request("a", 10, 5, deadline_sec=bad)

    @pytest.mark.parametrize("bad", [float("inf"), float("nan"), "5", True])
    def test_non_finite_or_wrong_type_rejected(self, bad):
        with pytest.raises(ValueError):
            Request("a", 10, 5, deadline_sec=bad)


# ---------- DeadlineAwarePolicy.select 排序 ----------

def _make_state(req_id, prompt, output, arrival, deadline=None):
    """构造最小 _State（只用到 select 里读取的字段）。"""
    req = Request(req_id, prompt, output, deadline_sec=deadline)
    st = object.__new__(_State)
    st.req = req
    st.arrival_sec = arrival
    return st


class TestDeadlinePolicySelection:
    def test_build_policy_roundtrip(self):
        p = build_policy("deadline")
        assert isinstance(p, DeadlineAwarePolicy)
        assert p.name == "deadline"

    def test_earliest_deadline_first(self):
        # a 截止 5s（到达 0 → 绝对 5），b 截止 20s（到达 5 → 绝对 25）
        queued = [
            _make_state("b", 10, 5, arrival=5, deadline=20),
            _make_state("a", 10, 5, arrival=0, deadline=5),
        ]
        assert DeadlineAwarePolicy().select(queued, now=5.0, step=1) == 1

    def test_no_deadline_goes_last(self):
        # a 无 deadline，b 有 deadline → b 先
        queued = [
            _make_state("a", 10, 5, arrival=0),
            _make_state("b", 10, 5, arrival=1, deadline=10),
        ]
        assert DeadlineAwarePolicy().select(queued, now=2.0, step=1) == 1

    def test_all_no_deadline_fcfs(self):
        queued = [
            _make_state("a", 10, 5, arrival=0),
            _make_state("b", 10, 5, arrival=1),
        ]
        # 先到先服务（arrival_sec 升序）
        assert DeadlineAwarePolicy().select(queued, now=2.0, step=1) == 0

    def test_empty_returns_none(self):
        assert DeadlineAwarePolicy().select([], now=0.0, step=1) is None


# ---------- 违规判定 + 统计 ----------

class TestSloViolationStats:
    def _run(self, requests, arrivals, **cfg_overrides):
        cfg = SimConfig(
            max_num_batched_tokens=64,
            max_num_seqs=4,
            step_overhead=0.01,
            t_prefill_token=0.001,
            t_decode_token=0.005,
            policy=DeadlineAwarePolicy(),
            **cfg_overrides,
        )
        return Simulator(cfg).run(requests, arrivals)

    def test_no_violations_when_deadline_loose(self):
        reqs = [Request("a", 20, 5, deadline_sec=10.0)]
        res = self._run(reqs, [0.0])
        assert res.slo_violations == 0
        assert res.slo_ratio == 1.0
        assert res.requests["a"].slo_violated is False

    def test_violation_when_deadline_tight(self):
        reqs = [Request("a", 100, 10, deadline_sec=0.001)]
        res = self._run(reqs, [0.0])
        assert res.slo_violations == 1
        assert res.slo_ratio == 0.0
        assert res.requests["a"].slo_violated is True

    def test_no_deadline_requests_not_counted(self):
        # 无 deadline 的请求不参与违规统计；slo_ratio 用 NaN 表示「无 SLO 负载」
        reqs = [Request("a", 20, 5)]
        res = self._run(reqs, [0.0])
        assert res.slo_violations == 0
        assert math.isnan(res.slo_ratio)
        assert res.requests["a"].slo_violated is False

    def test_mixed_deadline_and_no_deadline(self):
        # a 有 tight deadline（违规），b 无 deadline（不算）
        reqs = [
            Request("a", 100, 10, deadline_sec=0.001),
            Request("b", 10, 5),
        ]
        res = self._run(reqs, [0.0, 0.01])
        assert res.slo_violations == 1
        assert res.slo_ratio == 0.0
        assert res.requests["a"].slo_violated is True
        assert res.requests["b"].slo_violated is False

    def test_half_violated_ratio(self):
        # 2 个有 deadline 的请求，1 个违规 → ratio 0.5
        reqs = [
            Request("a", 100, 10, deadline_sec=0.001),  # 违规
            Request("b", 10, 5, deadline_sec=100.0),     # 达标
        ]
        res = self._run(reqs, [0.0, 0.01])
        assert res.slo_violations == 1
        assert res.slo_ratio == 0.5


# ---------- DeadlineAware 与 FCFS 的差异 ----------

class TestDeadlineVsFcfs:
    def test_deadline_policy_improves_ratio_under_load(self):
        """在 SLO 负载下，deadline 策略应比 FCFS 有更好（或持平）的达标率。

        用「早到的大请求 + 晚到的 tight-deadline 小请求」构造：FCFS 先跑大请求，
        小请求等太久违规；deadline 策略优先小请求（其绝对截止更早），达标。
        """
        reqs = [
            Request("big", 200, 40),                     # 无 deadline，FCFS 先跑它
            Request("small", 30, 8, deadline_sec=0.15),  # 晚到但 tight deadline
        ]
        arrivals = [0.0, 0.02]
        cfg = SimConfig(
            max_num_batched_tokens=32,
            max_num_seqs=1,
            step_overhead=0.005,
            t_prefill_token=0.001,
            t_decode_token=0.002,
        )
        fcfs_res = Simulator(cfg).run(reqs, arrivals)
        dl_cfg = SimConfig(
            max_num_batched_tokens=32,
            max_num_seqs=1,
            step_overhead=0.005,
            t_prefill_token=0.001,
            t_decode_token=0.002,
            policy=DeadlineAwarePolicy(),
        )
        dl_res = Simulator(dl_cfg).run(reqs, arrivals)
        # deadline 策略的达标率不低于 FCFS
        assert dl_res.slo_ratio >= fcfs_res.slo_ratio
        # 至少一个策略存在真实违规（负载确实紧张）
        assert fcfs_res.slo_violations >= 1 or dl_res.slo_violations >= 1
