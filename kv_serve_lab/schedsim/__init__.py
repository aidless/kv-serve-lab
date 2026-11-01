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


@dataclass(frozen=True)
class Request:
    req_id: str
    prompt_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
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

    def __post_init__(self) -> None:
        if self.max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be >= 1")
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be >= 1")
        for name in ("step_sec", "step_overhead", "t_prefill_token", "t_decode_token"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
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
            if fl >= self.max_num_seqs and out > 0:
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


class _State:
    __slots__ = (
        "req", "arrival_sec", "start_step", "computed",
        "out_tokens", "first_token_step", "first_token_sec", "finish_step",
        "finish_sec",
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

    @property
    def prefill_done(self) -> bool:
        return self.computed >= self.req.prompt_tokens

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

        states = [_State(r, t) for r, t in zip(requests, arrival_secs)]
        pending = sorted(range(len(states)), key=lambda i: (states[i].arrival_sec, i))
        running: list[_State] = []

        step = 0
        now = 0.0                      # 当前步起点的绝对时刻
        total_tokens = 0
        step_input_tokens: list[int] = []
        step_output_tokens: list[int] = []
        step_in_flight: list[int] = []
        step_durations: list[float] = []
        pending_pos = 0

        while True:
            # 到达判定按**墙钟**比较，不能按步号换算：成本模型下每步耗时随负载变化，
            # 「第 k 步 = k * base_step」这个等式不成立（会过早或过晚接纳请求）。
            while pending_pos < len(pending) and \
                    states[pending[pending_pos]].arrival_sec <= now + _EPS:
                pending_pos += 1
            arrived = [states[i] for i in pending[:pending_pos]]

            budget = self.config.max_num_batched_tokens
            granted: dict[int, int] = {}
            prefill_tokens_this_step = 0
            decode_reqs_this_step = 0

            # 1) 已在跑的请求先拿预算
            for st in running:
                if budget <= 0:
                    break
                if st.prefill_done:
                    need = 1
                else:
                    need = st.req.prompt_tokens - st.computed
                give = min(need, budget)
                if give > 0:
                    granted[id(st)] = give
                    budget -= give
                    if st.prefill_done:
                        decode_reqs_this_step += 1
                    else:
                        prefill_tokens_this_step += give

            # 2) 剩余预算 + 空余 slot 才接纳新请求
            for st in arrived:
                if st.start_step is not None:
                    continue
                if len(running) >= self.config.max_num_seqs:
                    break
                if budget <= 0:
                    break
                st.start_step = step
                running.append(st)
                need = st.req.prompt_tokens - st.computed
                give = min(need, budget)
                granted[id(st)] = give
                budget -= give
                prefill_tokens_this_step += give

            if not granted:
                if all(st.finished for st in states):
                    break
                # 空闲跳步：直接跳到下一个尚未开始请求的**到达时刻**。
                # 不能按 base_step 累加：成本模型下 base_step 只是「每步固定开销」的
                # 猜测，用它算跳步会与真实墙钟错位。这里把 now 精确设成到达时刻，
                # 步号按 base_step 换算只为让等时模型下的 start_step 语义保持不变。
                future = [st.arrival_sec for st in states if st.start_step is None]
                if not future:
                    break
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
                continue

            # 3) 计算本步耗时并推进时钟
            dur = self.config.step_time(prefill_tokens_this_step, decode_reqs_this_step)
            now += dur
            step_end = now
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
                was_prefill_done = st.prefill_done
                st.computed += give
                total_tokens += give
                if not was_prefill_done and st.prefill_done:
                    st.out_tokens += 1
                    st.first_token_step = step
                    st.first_token_sec = step_end
                    step_output += 1
                elif was_prefill_done:
                    st.out_tokens += 1
                    step_output += 1
                if st.finished and st.finish_sec is None:
                    st.finish_step = step
                    st.finish_sec = step_end
            step_output_tokens.append(step_output)

            running = [st for st in running if not st.finished]
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
            )

        total_output = sum(r.output_tokens for r in requests)
        sum_latency = sum(s.latency_sec for s in stats.values())
        return SimResult(
            requests=stats,
            total_tokens_computed=total_tokens,
            total_sec=now,
            output_throughput=(total_output / now) if now > 0 else 0.0,
            mean_system_requests=(sum_latency / now) if now > 0 else 0.0,
            arrival_rate=(len(requests) / now) if now > 0 else 0.0,
            max_num_seqs=self.config.max_num_seqs,
            step_input_tokens=step_input_tokens,
            step_output_tokens=step_output_tokens,
            step_in_flight=step_in_flight,
            step_durations=step_durations,
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
