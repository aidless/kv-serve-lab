"""流式推理压测：SSE 解析、逐请求指标、聚合统计。

指标定义严格对齐 vLLM `vllm/benchmarks/serve.py:654-658`：

    TPOT = (latency - ttft) / (output_len - 1)

其中 latency 是「请求发出 -> 最后一个 token」、ttft 是「请求发出 -> 第一个 token」，
output_len 是该请求**实际产出 token 数**。

**为什么 chunk ≠ token（本项目反复强调）**：
    OpenAI 兼容接口的 SSE 流，一个 chunk 可能一次带多个 token（服务端为了省网络
    往返会打包）。若用「收到几个 chunk」当 output_len，会系统性低估输出长度、
    把 TPOT 算大。因此 output_len 只能来自：
      (a) 每个 chunk 的文本经 tokenizer 计数后累加；或
      (b) 服务端 usage.completion_tokens（累积值，取最后一次）。
    两者都没有时**必须报错**，不许悄悄退化成数 chunk。

时间来源全部是「带时间戳的 payload」——解析层不自己读钟，便于用假时钟做
确定性测试；真实客户端负责在读每一行时打真实时间戳。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

__all__ = [
    "BenchError",
    "TokenArrival",
    "RequestTrace",
    "BenchSummary",
    "iter_sse_payloads",
    "trace_from_payloads",
    "trace_from_sse",
    "summarize",
]


class BenchError(Exception):
    """bench 层的输入/度量错误（如 chunk≠token 却没给 tokenizer）。"""


@dataclass(frozen=True)
class TokenArrival:
    """一个 chunk 带来的若干新 token 及其到达时刻。"""

    t_sec: float
    n_tokens: int

    def __post_init__(self) -> None:
        if self.n_tokens < 0:
            raise ValueError("n_tokens must be >= 0")


@dataclass
class RequestTrace:
    """单个请求的流式时间线。时间单位秒，绝对时刻。"""

    req_id: str
    start_sec: float
    arrivals: list[TokenArrival] = field(default_factory=list)
    usage_total: int | None = None
    # 流真正结束的时刻（含末尾 usage chunk / [DONE]）。vLLM bench 的 latency
    # 量的是整条流结束，不是最后一个 token 到达 —— 二者在 usage-only 收尾时不同。
    stream_end_sec: float | None = None

    def _token_arrivals(self) -> list[TokenArrival]:
        return [a for a in self.arrivals if a.n_tokens > 0]

    @property
    def output_len(self) -> int:
        if self.usage_total is not None:
            return self.usage_total
        return sum(a.n_tokens for a in self.arrivals)

    @property
    def chunk_count(self) -> int:
        """真正携带 token 的 chunk 数（不含 role-only / 空 chunk）。"""
        return len(self._token_arrivals())

    @property
    def tokens_per_chunk(self) -> float:
        c = self.chunk_count
        return (self.output_len / c) if c else float("nan")

    @property
    def first_token_sec(self) -> float:
        ta = self._token_arrivals()
        if not ta:
            raise BenchError(f"{self.req_id}: stream produced no tokens")
        return ta[0].t_sec

    @property
    def end_sec(self) -> float:
        ta = self._token_arrivals()
        if not ta:
            raise BenchError(f"{self.req_id}: stream produced no tokens")
        last_token = ta[-1].t_sec
        if self.stream_end_sec is not None:
            return max(last_token, self.stream_end_sec)
        return last_token

    @property
    def ttft_sec(self) -> float:
        return self.first_token_sec - self.start_sec

    @property
    def latency_sec(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def tpot_sec(self) -> float:
        n = self.output_len
        if n <= 1:
            return float("nan")
        return (self.latency_sec - self.ttft_sec) / (n - 1)


@dataclass(frozen=True)
class BenchSummary:
    num_requests: int
    total_output_tokens: int
    mean_ttft_sec: float
    median_ttft_sec: float
    p99_ttft_sec: float
    mean_tpot_sec: float
    mean_tokens_per_chunk: float
    wall_sec: float
    output_throughput: float


def _extract_text(obj: dict) -> str:
    """从 completions(text) 或 chat(delta.content) 的 chunk 里取文本。"""
    choices = obj.get("choices") or []
    if not choices:
        return ""
    ch = choices[0]
    if isinstance(ch.get("text"), str):
        return ch["text"]
    delta = ch.get("delta") or {}
    if isinstance(delta.get("content"), str):
        return delta["content"]
    return ""


def iter_sse_payloads(lines, clock):
    """把 SSE 行流变成 (t_sec, payload_str)。clock 每读到一条 data 行调用一次。

    lines 可以是 str 或 bytes 的可迭代对象（如 urllib 响应、测试列表）。
    """
    for raw in lines:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "replace")
        line = raw.strip()
        if not line or not line.startswith("data:"):
            continue
        yield clock(), line[5:].strip()


def trace_from_payloads(
    req_id: str,
    start_sec: float,
    timed_payloads,
    token_counter=None,
) -> RequestTrace:
    """从 (t_sec, payload) 序列构造 RequestTrace。

    计数规则（**usage 是总量口径，文本是时序口径**）：

    1. **时序只由文本 chunk 提供**。TTFT 落在第一个真正带 token 的 chunk 上，
       绝不能被末尾那个 usage-only chunk 拉走 —— 否则会把「首 token 延迟」
       错量成「整条流结束」。
    2. **数量以 usage 为准**。`usage.completion_tokens` 是累积总数；若它比已计入的
       多，差额的处理取决于**上一个 arrival 的来源**：
         - 上一个 arrival 来自文本 chunk → 差额**并入**它（不新建），因为收尾 usage
           只是把文本估计修正到服务端的真实计数，不是一次新的投递。
         - 上一个 arrival 来自 usage（即整条流没有文本，只报 usage）→ 差额**新建**
           一个 arrival，因为那个 usage chunk 本身就是一次真实的投递时刻。
       例：5 个文本 chunk 各 1 + 末尾 usage=5 → 5 个 token、5 个 chunk，不是 10。
    3. 文本 chunk 但没给 token_counter、且流里**从未**出现 usage 时抛 BenchError：
       数 chunk 不等于数 token。
    4. `latency` 量到**流结束**（末尾 usage chunk 或 [DONE] 的时刻），与 vLLM
       `benchmarks/serve.py` 的 `latency - ttft` 口径一致；当流以 usage-only chunk
       收尾时，这个时刻晚于最后一个 token。

    token_counter: callable(text:str)->int。
    """
    arrivals: list[TokenArrival] = []
    usage_total: int | None = None
    accounted = 0
    saw_usage = False
    last_arrival_from_text = False
    stream_end_sec: float | None = None

    for t_sec, payload in timed_payloads:
        if payload == "[DONE]":
            stream_end_sec = t_sec
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue

        stream_end_sec = t_sec  # 任何真实 payload 都推进流结束时刻

        usage = obj.get("usage") or {}
        text = _extract_text(obj)
        cum = _usage_cumulative(usage)

        # --- 文本：只提供时序，数量先按 token_counter 估 ---
        if text:
            if token_counter is None and cum is None and not saw_usage:
                raise BenchError(
                    f"{req_id}: chunk carries text but token_counter is None. "
                    "chunk count != token count — pass a tokenizer or have the "
                    "server send usage.completion_tokens; do NOT count chunks."
                )
            if not saw_usage and token_counter is not None:
                n = token_counter(text)
                if n > 0:
                    arrivals.append(TokenArrival(t_sec, n))
                    accounted += n
                    last_arrival_from_text = True

        # --- usage：累积总数，差额并入最后一个 token arrival ---
        if cum is not None:
            saw_usage = True
            usage_total = cum
            if cum > accounted:
                delta = cum - accounted
                if arrivals and last_arrival_from_text:
                    # 收尾 usage 只是把文本估计修正到服务端真值，并入同一个 arrival
                    last = arrivals[-1]
                    arrivals[-1] = TokenArrival(last.t_sec, last.n_tokens + delta)
                else:
                    # 纯 usage 流：这个 usage chunk 自己就是一次真实投递时刻
                    arrivals.append(TokenArrival(t_sec, delta))
                    last_arrival_from_text = False
                accounted = cum

    return RequestTrace(req_id, start_sec, arrivals, usage_total, stream_end_sec)


def _usage_cumulative(usage: dict) -> int | None:
    v = usage.get("completion_tokens")
    return int(v) if v is not None else None


def trace_from_sse(req_id: str, start_sec: float, lines, clock, token_counter=None):
    """便捷入口：直接用 clock 给每行打时间戳。"""
    return trace_from_payloads(
        req_id, start_sec, iter_sse_payloads(lines, clock), token_counter
    )


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = max(0, math.ceil(q * len(sorted_values)) - 1)
    return sorted_values[min(idx, len(sorted_values) - 1)]


def _median(sorted_values: list[float]) -> float:
    """中位数：偶数个取中间两个的平均，不是上中位数。

    取上中位数会系统性偏大（4 个样本 [0.1,0.2,0.3,0.9] 会报 0.3 而不是 0.25），
    且偏差随样本量漂移 —— 在右偏的 TTFT 分布上尤其明显。
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def summarize(traces: list[RequestTrace]) -> BenchSummary:
    if not traces:
        raise BenchError("summarize() got no traces")
    ttfts = sorted(t.ttft_sec for t in traces)
    tpots = [t.tpot_sec for t in traces if t.tpot_sec == t.tpot_sec]  # 去 nan
    tpc = [t.tokens_per_chunk for t in traces]
    total_tokens = sum(t.output_len for t in traces)
    wall = max(t.end_sec for t in traces) - min(t.start_sec for t in traces)
    return BenchSummary(
        num_requests=len(traces),
        total_output_tokens=total_tokens,
        mean_ttft_sec=sum(ttfts) / len(ttfts),
        median_ttft_sec=_median(ttfts),
        p99_ttft_sec=_percentile(ttfts, 0.99),
        mean_tpot_sec=(sum(tpots) / len(tpots)) if tpots else float("nan"),
        mean_tokens_per_chunk=sum(tpc) / len(tpc),
        wall_sec=wall,
        output_throughput=(total_tokens / wall) if wall > 0 else float("inf"),
    )
