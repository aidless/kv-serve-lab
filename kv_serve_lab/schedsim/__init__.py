"""continuous batching 调度模拟器。

模型镜像 vLLM v1 的调度语义（vllm/v1/core/sched/scheduler.py:700-708 原注释）：

    "There's no 'decoding phase' nor 'prefill phase' in the scheduler.
     Each request just has its num_computed_tokens and num_tokens_with_spec.
     At each step, the scheduler tries to assign tokens to the requests so that
     each request's num_computed_tokens can catch up its num_tokens_with_spec.

     This is general enough to cover chunked prefills, prefix caching,
     speculative decoding..."

即：**统一 token 预算**，不是 prefill/decode 两阶段。每步：
  1. 预算 = max_num_batched_tokens
  2. 已在跑的请求先拿 token（decode 各 1；未完成 prefill 的按剩余额度分块）
  3. 剩余预算才给排队新请求，且受 max_num_seqs 限制
  4. **prefill 的末步同时采样出首个输出 token**
     —— 这是 vLLM serve.py:656 `tpot = (latency - ttft) / (output_len - 1)`
     分母为 output_len-1 的物理来源。

时间模型（显式定义，否则算出的数没有意义）：

    step_time = step_overhead
              + prefill_tokens_this_step × t_prefill_token
              + decode_reqs_this_step  × t_decode_token

三个成本常数都是模拟器的**输入**，不是拟合出来的。默认全 0 时退化为
「每个调度步固定 step_sec」的等时模型（仅用于验证调度逻辑本身）。

饱和吞吐解析上界（等时模型）= max_num_seqs / step_sec。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_EPS = 1e-12   # 浮点比较容差：0.05 与 0.05 可能算出 0.05000000000000002


class SchedulerPolicy:
    """从排队队列挑选「下一个要接入 running」的请求的策略基类。

    连续批处理（continuous batching）的调度器每步要决定：在预算和 max_num_seqs
    允许的范围内，从 waiting 队列里**按什么顺序**接纳新请求。vLLM v1 默认 FCFS
    （waiting 是普通队列，先到先得）；但文献（arXiv:2610.15899，8 策略 × 1152 组合
    实测）表明**aging shortest-prompt-first** 在多数 L40S 与混合负载下是最强实用
    基线——即按「最短 prompt 优先」接纳，同时给等待久的请求累积老化权重，避免
    饥饿。

    本基类把「挑选下一个」抽象成单一方法，使 schedsim 成为策略对比测试台：
    同一份请求序列、同一份配置，只换策略对象，即可比较 FCFS / 老化 / 最短优先
    在时间模型下的 TTFT / TPOT / 吞吐 / SLO 差异。
    """

    name: str = "base"

    def select(
        self,
        queued: list["_State"],
        now: float,
        step: int,
    ) -> int | None:
        """返回要接纳的 `queued` 下标；None 表示本轮不再接纳。

        `now` 是当前墙钟（秒），`step` 是当前调度步号——策略可用它们实现老化
        （等待时间 = now - arrival_sec）或按到达次序的稳定排序。
        """
        raise NotImplementedError


class FCFSPolicy(SchedulerPolicy):
    """先到先得：严格按 arrival_sec（同刻按请求序号）接纳。vLLM v1 默认。"""

    name = "fcfs"

    def select(
        self,
        queued: list["_State"],
        now: float,
        step: int,
    ) -> int | None:
        return 0


class AgingShortestPromptFirstPolicy(SchedulerPolicy):
    """aging shortest-prompt-first（arXiv:2610.15899 的最强实用基线）。

    接纳顺序 = 按「prompt 长度」升序，但每等待一个调度步给请求累积老化权重，
    等价于把「等待时间 / 老化窗口」折进排序键。这里用确定性实现：
    key = prompt_tokens - aging_window * (now - arrival_sec)。
    老化窗口越大，等待越久的请求优先级抬得越快（抗饥饿）。
    """

    name = "aging_shortest_prompt_first"

    def __init__(self, aging_window: float = 1.0) -> None:
        if not math.isfinite(aging_window) or aging_window <= 0:
            raise ValueError(f"aging_window must be finite and > 0, got {aging_window!r}")
        self.aging_window = aging_window

    def select(
        self,
        queued: list["_State"],
        now: float,
        step: int,
    ) -> int | None:
        if not queued:
            return None
        # 等待时间按墙钟算（now - arrival_sec），越久优先级越高（键越小）。
        return min(
            range(len(queued)),
            key=lambda i: (
                queued[i].req.prompt_tokens
                - self.aging_window * (now - queued[i].arrival_sec),
                i,  # 平键按到达次序稳定
            ),
        )


class ShortestJobFirstPolicy(SchedulerPolicy):
    """最短剩余 job 优先（对照策略，不含老化）。

    用于隔离「长度效应」与「老化效应」——论文指出 job-size 知识本身不足以
    决定最优（batching/硬件效应也影响），所以它只作对照。
    """

    name = "shortest_job_first"

    def select(
        self,
        queued: list["_State"],
        now: float,
        step: int,
    ) -> int | None:
        if not queued:
            return None
        return min(
            range(len(queued)),
            key=lambda i: (queued[i].req.prompt_tokens + queued[i].req.output_tokens, i),
        )


def build_policy(name: str, **kwargs: float) -> SchedulerPolicy:
    """按名字构造策略。未知名字抛 ValueError（与 bad_args 错误码对齐）。"""
    if name == FCFSPolicy.name:
        return FCFSPolicy()
    if name == AgingShortestPromptFirstPolicy.name:
        return AgingShortestPromptFirstPolicy(**kwargs)
    if name == ShortestJobFirstPolicy.name:
        return ShortestJobFirstPolicy()
    raise ValueError(f"unknown scheduler policy: {name!r}")


@dataclass(frozen=True)
class Request:
    req_id: str
    prompt_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        for name in ("prompt_tokens", "output_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{self.req_id}: {name} must be an integer")
        if self.prompt_tokens < 1:
            raise ValueError(f"{self.req_id}: prompt_tokens must be >= 1")
        if self.output_tokens < 1:
            raise ValueError(f"{self.req_id}: output_tokens must be >= 1")


@dataclass(frozen=True)
class SimConfig:
    """调度器配置，字段名对齐 vLLM 的 scheduler_config。"""

    max_num_batched_tokens: int
    max_num_seqs: int
    step_sec: float = 0.0            # 等时模型下每步固定耗时
    step_overhead: float = 0.0       # 每步固定开销（权重读取等）
    t_prefill_token: float = 0.0     # 每个 prefill 输入 token 的边际成本
    t_decode_token: float = 0.0      # 每个 decode 请求每步的边际成本
    # KV 块容量：设为 None 表示不做块记账（无抢占）。给定后启用「块不够就抢占」，
    # 语义对齐 vLLM v1 的 RECOMPUTE 策略。
    kv_num_blocks: int | None = None
    kv_block_size: int = 16
    # 调度策略：None 表示 FCFS（vLLM v1 默认）。传入策略对象即可换策略对比。
    policy: SchedulerPolicy | None = None

    def __post_init__(self) -> None:
        for name in ("max_num_batched_tokens", "max_num_seqs"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
        for name in ("step_sec", "step_overhead", "t_prefill_token", "t_decode_token"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.kv_num_blocks is not None:
            if not isinstance(self.kv_num_blocks, int) or isinstance(self.kv_num_blocks, bool):
                raise ValueError("kv_num_blocks must be an integer")
            if self.kv_num_blocks < 1:
                raise ValueError("kv_num_blocks must be >= 1")
        if not isinstance(self.kv_block_size, int) or isinstance(self.kv_block_size, bool):
            raise ValueError("kv_block_size must be an integer")
        if self.kv_block_size < 1:
            raise ValueError("kv_block_size must be >= 1")
        if self.policy is not None and not isinstance(self.policy, SchedulerPolicy):
            raise ValueError(
                f"policy must be a SchedulerPolicy instance, got {type(self.policy).__name__}"
            )
        cost_model = any((self.step_overhead, self.t_prefill_token, self.t_decode_token))
        if cost_model and self.step_overhead <= 0:
            raise ValueError("cost model needs positive step_overhead for every step")
        if self.step_sec <= 0 and self.step_overhead <= 0:
            raise ValueError(
                "need a positive time source: step_sec (constant-step model) "
                "or step_overhead (cost model)"
            )

    def step_time(self, prefill_tokens: int, decode_reqs: int) -> float:
        """本步墙钟耗时。等时模型（step_overhead==0 且无边际成本）返回 step_sec。"""
        if (
            self.step_overhead == 0
            and self.t_prefill_token == 0
            and self.t_decode_token == 0
        ):
            return self.step_sec
        return (
            self.step_overhead
            + prefill_tokens * self.t_prefill_token
            + decode_reqs * self.t_decode_token
        )


@dataclass(frozen=True)
class RequestStats:
    """单个请求的时序结果。时间字段是绝对时刻，秒。"""

    req_id: str
    arrival_sec: float
    start_step: int
    first_token_step: int
    finish_step: int
    first_token_sec: float   # 首 token 的绝对时刻
    finish_sec: float        # 最后一个 token 的绝对时刻
    num_preemptions: int = 0  # 被抢占次数（每次触发从头重算）

    @property
    def ttft_sec(self) -> float:
        """TTFT = 首 token 时刻 - 请求到达时刻（标准定义，含排队等待）。"""
        return self.first_token_sec - self.arrival_sec

    @property
    def latency_sec(self) -> float:
        return self.finish_sec - self.arrival_sec

    @property
    def ttft_steps(self) -> int:
        """从 start_step 到首 token 所跨的调度步数（含首尾）。"""
        return self.first_token_step - self.start_step + 1

    def tpot_for(self, output_len: int) -> float:
        """TPOT = (latency - ttft) / (output_len - 1)，同 vLLM serve.py:656。

        首 token 是 TTFT 那步产出的、不计入间隔，故分母是 output_len-1。
        output_len==1 时无定义（分母 0），返回 nan。
        """
        if output_len <= 1:
            return float("nan")
        return (self.latency_sec - self.ttft_sec) / (output_len - 1)


@dataclass(frozen=True)
class Metrics:
    mean_ttft_sec: float
    p99_ttft_sec: float
    mean_latency_sec: float
    completed: int


@dataclass
class SimResult:
    requests: dict[str, RequestStats] = field(default_factory=dict)
    total_tokens_computed: int = 0
    total_sec: float = 0.0
    output_throughput: float = 0.0        # 输出 token / 总墙钟
    mean_system_requests: float = 0.0     # 时间平均在系统内请求数
    arrival_rate: float = 0.0             # 请求 / 秒
    max_num_seqs: int = 0
    step_input_tokens: list[int] = field(default_factory=list)
    step_output_tokens: list[int] = field(default_factory=list)
    step_in_flight: list[int] = field(default_factory=list)
    step_durations: list[float] = field(default_factory=list)
    step_system_areas: list[float] = field(default_factory=list)
    step_kv_blocks: list[int] = field(default_factory=list)   # 每步末占用块数
    total_preemptions: int = 0
    recompute_tokens: int = 0        # 被抢占丢弃、需要重算的 token 总量
    peak_kv_blocks: int = 0
    _output_lens: dict[str, int] = field(default_factory=dict)

    def per_request_tpot(self, req_id: str) -> float:
        return self.requests[req_id].tpot_for(self._output_lens[req_id])

    def metrics(self) -> Metrics:
        if not self.requests:
            return Metrics(0.0, 0.0, 0.0, 0)
        ttfts = sorted(s.ttft_sec for s in self.requests.values())
        lats = sorted(s.latency_sec for s in self.requests.values())
        return Metrics(
            mean_ttft_sec=sum(ttfts) / len(ttfts),
            p99_ttft_sec=_percentile(ttfts, 0.99),
            mean_latency_sec=sum(lats) / len(lats),
            completed=len(self.requests),
        )

    @property
    def steady_state_throughput(self) -> float:
        """饱和稳态吞吐 = 在「满载（in_flight == max_num_seqs）」的那些步上，
        输出 token 总量 / 这些步的墙钟总时长。

        这是 schedsim 能预测的最干净的吞吐口径：只看系统真正被占满的区间，
        避开 ramp-up 与 drain 的边界效应。若从未满载（并发上限没被触及），
        退化为整段 output_throughput。
        """
        if not self.step_output_tokens:
            return 0.0
        if self.max_num_seqs <= 0:
            return self.output_throughput
        sat_out = 0
        sat_time = 0.0
        for out, fl, dur in zip(
            self.step_output_tokens, self.step_in_flight, self.step_durations
        ):
            if fl >= self.max_num_seqs:
                sat_out += out
                sat_time += dur
        if sat_time <= 0:
            return self.output_throughput
        return sat_out / sat_time


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = max(0, math.ceil(q * len(sorted_values)) - 1)
    return sorted_values[min(idx, len(sorted_values) - 1)]


class _KvCache:
    """KV 块记账：只回答「装不装得下」，不搬 payload（那是 blockpool 的职责）。

    对齐 vLLM 的两条事实：
      - 块是分配单位，一个请求持有 ceil(num_computed_tokens / block_size) 块；
      - 抢占即释放全部块，语义上是 RECOMPUTE（从头重算），不保留已算 KV。
    """

    __slots__ = ("num_blocks", "block_size", "_held")

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._held: dict[int, int] = {}

    def blocks_for(self, computed: int) -> int:
        return (computed + self.block_size - 1) // self.block_size

    def held(self, st: "_State") -> int:
        return self._held.get(id(st), 0)

    @property
    def used(self) -> int:
        return sum(self._held.values())

    def fits(self, st: "_State", computed_after: int) -> bool:
        """把 st 扩到 computed_after 后是否仍装得下（替换它自己原有的占用）。"""
        others = self.used - self._held.get(id(st), 0)
        return others + self.blocks_for(computed_after) <= self.num_blocks

    def commit(self, st: "_State", computed_after: int) -> None:
        self._held[id(st)] = self.blocks_for(computed_after)

    def release(self, st: "_State") -> None:
        self._held.pop(id(st), None)


class _State:
    __slots__ = (
        "req", "arrival_sec", "start_step", "computed",
        "out_tokens", "first_token_step", "first_token_sec", "finish_step",
        "finish_sec", "num_preemptions",
    )

    def __init__(self, req: Request, arrival_sec: float) -> None:
        self.req = req
        self.arrival_sec = arrival_sec
        self.start_step: int | None = None
        self.computed = 0
        self.out_tokens = 0
        self.first_token_step: int | None = None
        self.first_token_sec: float | None = None
        self.finish_step: int | None = None
        self.finish_sec: float | None = None
        self.num_preemptions = 0

    @property
    def target(self) -> int:
        """本步要追赶到的 num_computed_tokens。

        统一 prefill 与 decode（vLLM 的注释：调度器里没有 prefill/decode 阶段，
        只有「每个请求还欠多少 token」）：
          - 还没产出任何 token：目标 = prompt，算到即采样出首个 token；
          - 已产出 k 个 token：目标 = prompt + k，算到即采样出第 k+1 个 token。
        被抢占重算时 out_tokens 保留，所以目标也随之变高——重算要把已生成的
        输出 token 一起当输入重新过一遍，这与 vLLM 一致。
        """
        return self.req.prompt_tokens + self.out_tokens

    @property
    def need(self) -> int:
        """本步最少要吃掉多少预算才能推进；0 表示已达标，decode 取 1。"""
        return max(0, self.target - self.computed) or 1

    @property
    def is_decode(self) -> bool:
        """本步是否是 decode 步（消耗 1 个输入位置、产出 1 个 token）。

        不能写成 `computed >= target`：target = prompt + out_tokens 会随采样上移，
        稳态下恒有 `computed == target - 1`。被判为 decode 的必须是「已经出过
        至少一个 token 且输入已追平上一次目标的请求」；重算中的请求即使只差
        1 个 token 也仍是 prefill。
        """
        return self.out_tokens >= 1 and self.computed == self.target - 1

    @property
    def finished(self) -> bool:
        return self.out_tokens >= self.req.output_tokens


class Simulator:
    def __init__(self, config: SimConfig) -> None:
        self.config = config

    def run(self, requests: list[Request], arrival_secs: list[float]) -> SimResult:
        if len(requests) != len(arrival_secs):
            raise ValueError(
                f"arrival_secs length {len(arrival_secs)} != requests length "
                f"{len(requests)}"
            )
        ids = [r.req_id for r in requests]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate req_id")
        for t in arrival_secs:
            if not math.isfinite(t):
                raise ValueError(f"arrival_sec must be finite, got {t!r}")
            if t < 0:
                raise ValueError(f"arrival_sec must be >= 0, got {t!r}")

        cfg = self.config
        kv: _KvCache | None = None
        if cfg.kv_num_blocks is not None:
            kv = _KvCache(cfg.kv_num_blocks, cfg.kv_block_size)
            # 单个请求独占整个缓存都装不下时，抢占也腾不出空间 —— 明确报错，
            # 不留给调用方一个「卡住/死循环」的模拟器。
            for r in requests:
                need_blocks = kv.blocks_for(r.prompt_tokens + r.output_tokens - 1)
                if need_blocks > kv.num_blocks:
                    raise ValueError(
                        f"request {r.req_id} cannot fit in KV cache: needs "
                        f"{need_blocks} blocks of {kv.block_size} tokens, "
                        f"capacity is {kv.num_blocks} blocks"
                    )

        states = [_State(r, t) for r, t in zip(requests, arrival_secs)]
        pending = sorted(range(len(states)), key=lambda i: (states[i].arrival_sec, i))
        running: list[_State] = []
        queued: list[_State] = []      # 已到达、未完成、不在 running；被抢占者插队首

        step = 0
        now = 0.0
        total_tokens = 0
        total_preemptions = 0
        recompute_tokens = 0
        step_input_tokens: list[int] = []
        step_output_tokens: list[int] = []
        step_in_flight: list[int] = []
        step_durations: list[float] = []
        step_kv_blocks: list[int] = []
        step_system_areas: list[float] = []
        pending_pos = 0

        def preempt(st: _State) -> None:
            """vLLM _preempt_request 的核心三条：重算、计数、插回队首。"""
            nonlocal total_preemptions, recompute_tokens
            total_preemptions += 1
            recompute_tokens += st.computed
            st.num_preemptions += 1
            st.computed = 0          # RECOMPUTE：已算的 KV 全部丢弃

        while True:
            # 到达判定按**墙钟**比较，不能按步号换算：成本模型下每步耗时随负载变化。
            while pending_pos < len(pending) and \
                    states[pending[pending_pos]].arrival_sec <= now + _EPS:
                queued.append(states[pending[pending_pos]])
                pending_pos += 1

            budget = cfg.max_num_batched_tokens
            granted: dict[int, int] = {}
            prefill_tokens_this_step = 0
            decode_reqs_this_step = 0
            preempted_this_step: list[_State] = []

            def revoke(st: _State) -> None:
                """撤回本步已发给该请求的额度（它即将被抢占）。"""
                nonlocal budget, prefill_tokens_this_step, decode_reqs_this_step
                give = granted.pop(id(st), 0)
                if not give:
                    return
                budget += give
                if st.is_decode:                  # 这一份是 decode 的
                    decode_reqs_this_step -= 1
                else:
                    prefill_tokens_this_step -= give

            # 1) 已在跑的请求按 FCFS 顺序先拿预算；块不够就抢占 running[-1]
            idx = 0
            while idx < len(running):
                st = running[idx]
                if budget <= 0:
                    break
                give = min(st.need, budget)
                if kv is not None:
                    while not kv.fits(st, st.computed + give):
                        victim = running[-1]
                        if victim is st:
                            break
                        revoke(victim)
                        running.pop()
                        kv.release(victim)
                        preempt(victim)
                        preempted_this_step.append(victim)
                        if idx > len(running):
                            idx = len(running)
                    if not kv.fits(st, st.computed + give):
                        # 自己就是最后一个候选：vLLM 在这里 `break`（不再抢自己）
                        revoke(st)
                        running.pop()
                        kv.release(st)
                        preempt(st)
                        preempted_this_step.append(st)
                        break
                granted[id(st)] = give
                budget -= give
                if st.is_decode:
                    decode_reqs_this_step += 1
                else:
                    prefill_tokens_this_step += give
                idx += 1

            # 2) 剩余预算 + 空余 slot 才接纳排队请求。
            # 但**本步发生过抢占就整步跳过 waiting**（scheduler.py:1049
            # `if not preempted_reqs and ...`）——否则刚被抢占的请求会立刻在
            # 同一批里反过来抢占别人，形成同批互抢的死循环。
            # 接纳顺序由策略决定：None 默认 FCFS（queued[0]），换策略即可对比。
            policy = cfg.policy
            while not preempted_this_step and queued and budget > 0 \
                    and len(running) < cfg.max_num_seqs:
                pick = policy.select(queued, now, step) if policy is not None else 0
                if pick is None:
                    break
                st = queued.pop(pick)
                give = min(st.need, budget)
                if kv is not None:
                    while not kv.fits(st, st.computed + give):
                        if not running:
                            break
                        victim = running[-1]
                        revoke(victim)
                        running.pop()
                        kv.release(victim)
                        preempt(victim)
                        preempted_this_step.append(victim)
                    if not kv.fits(st, st.computed + give):
                        break
                if st.start_step is None:
                    st.start_step = step
                running.append(st)
                granted[id(st)] = give
                budget -= give
                prefill_tokens_this_step += give

            # 被抢占者插回队首，下一步优先重试（vLLM 的 waiting.prepend_request）
            if preempted_this_step:
                queued[:0] = preempted_this_step

            if not granted:
                if all(st.finished for st in states):
                    break
                future = [st.arrival_sec for st in states if st.start_step is None]
                if not future:
                    raise RuntimeError(
                        "scheduler stuck: no progress but requests remain "
                        f"(unfinished: {[s.req.req_id for s in states if not s.finished]})"
                    )
                next_arrival = min(future)
                gap = next_arrival - now
                if gap <= _EPS:
                    raise RuntimeError("scheduler stuck: no progress but requests remain")
                base = self._base_step()
                step += int(math.ceil(gap / base - 1e-9)) if base > 0 else 1
                now = next_arrival
                step_input_tokens.append(0)
                step_output_tokens.append(0)
                step_in_flight.append(0)
                step_durations.append(gap)
                if kv is not None:
                    step_kv_blocks.append(kv.used)
                step_system_areas.append(0.0)
                continue

            # 3) 计算本步耗时并推进时钟
            dur = cfg.step_time(prefill_tokens_this_step, decode_reqs_this_step)
            step_start = now
            now += dur
            step_end = now
            step_system_areas.append(sum(
                max(0.0, step_end - max(step_start, st.arrival_sec))
                for st in states if not st.finished and st.arrival_sec < step_end
            ))
            in_flight = len(running)
            step_input_tokens.append(sum(granted.values()))
            step_in_flight.append(in_flight)
            step_durations.append(dur)

            # 4) 应用本步结果
            step_output = 0
            for st in running:
                give = granted.get(id(st), 0)
                if give <= 0:
                    continue
                reached_target = st.computed < st.target
                st.computed += give
                total_tokens += give
                if kv is not None:
                    kv.commit(st, st.computed)
                # 算到目标即采样出一个 token（prefill 末步的「首 token」同理）
                if reached_target and st.computed >= st.target and not st.finished:
                    st.out_tokens += 1
                    step_output += 1
                    if st.first_token_sec is None:
                        st.first_token_step = step
                        st.first_token_sec = step_end
                    if st.finished:
                        st.finish_step = step
                        st.finish_sec = step_end
            step_output_tokens.append(step_output)

            # 完成的请求立刻归还全部块——否则 used 永远停在高水位，
            # 后续请求再小也装不下（实测会一路卡到 scheduler stuck）。
            for st in running:
                if st.finished and kv is not None:
                    kv.release(st)
            running = [st for st in running if not st.finished]
            if kv is not None:
                step_kv_blocks.append(kv.used)
            step += 1
            if all(st.finished for st in states):
                break

        stats: dict[str, RequestStats] = {}
        for st in states:
            if (
                st.start_step is None
                or st.first_token_sec is None
                or st.finish_sec is None
                or st.first_token_step is None
                or st.finish_step is None
            ):
                raise RuntimeError(f"request {st.req.req_id} never completed")
            stats[st.req.req_id] = RequestStats(
                req_id=st.req.req_id,
                arrival_sec=st.arrival_sec,
                start_step=st.start_step,
                first_token_step=st.first_token_step,
                finish_step=st.finish_step,
                first_token_sec=st.first_token_sec,
                finish_sec=st.finish_sec,
                num_preemptions=st.num_preemptions,
            )

        total_output = sum(r.output_tokens for r in requests)
        return SimResult(
            requests=stats,
            total_tokens_computed=total_tokens,
            total_sec=now,
            output_throughput=(total_output / now) if now > 0 else 0.0,
            mean_system_requests=(sum(step_system_areas) / now) if now > 0 else 0.0,
            arrival_rate=(len(requests) / now) if now > 0 else 0.0,
            max_num_seqs=cfg.max_num_seqs,
            step_input_tokens=step_input_tokens,
            step_output_tokens=step_output_tokens,
            step_in_flight=step_in_flight,
            step_durations=step_durations,
            step_system_areas=step_system_areas,
            step_kv_blocks=step_kv_blocks,
            total_preemptions=total_preemptions,
            recompute_tokens=recompute_tokens,
            peak_kv_blocks=max(step_kv_blocks) if step_kv_blocks else 0,
            _output_lens={r.req_id: r.output_tokens for r in requests},
        )

    def _base_step(self) -> float:
        """空闲步的耗时：等时模型用 step_sec，成本模型用 step_overhead。"""
        if self.config.step_overhead == 0 and self.config.step_sec > 0:
            return self.config.step_sec
        return self.config.step_overhead

    def _to_step(self, arrival_sec: float) -> int:
        """到达时刻 -> 步号。带 epsilon 防浮点误差把 0.05/0.01 算成 6。"""
        base = self._base_step() if self._base_step() > 0 else 1.0
        return int(math.ceil(arrival_sec / base - 1e-9))
