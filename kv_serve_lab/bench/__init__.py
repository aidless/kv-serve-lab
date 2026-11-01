"""流式推理压测：SSE 解析、逐请求指标、聚合统计。

指标定义严格对齐 vLLM `vllm/benchmarks/serve.py:654-658`：

    TPOT = (latency - ttft) / (output_len - 1)

其中 latency 是「请求发出 -> 流结束」、ttft 是「请求发出 -> 第一个 token」，
output_len 是该请求**实际产出 token 数**。

**为什么 chunk ≠ token（本项目反复强调）**：
    OpenAI 兼容接口的 SSE 流，一个 chunk 可能一次带多个 token（服务端为了省网络
    往返会打包）。若用「收到几个 chunk」当 output_len，会系统性低估输出长度、
    把 TPOT 算大。因此 output_len 只能来自：
      (a) 拼接完整输出文本后经 tokenizer 计数；或
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
    text_chunk_times: list[float] | None = None
    text_token_total: int | None = None

    def _token_arrivals(self) -> list[TokenArrival]:
        return [a for a in self.arrivals if a.n_tokens > 0]

    @property
    def output_len(self) -> int:
        if self.usage_total is not None:
            return self.usage_total
        if self.text_token_total is not None:
            return self.text_token_total
        return sum(a.n_tokens for a in self.arrivals)

    @property
    def chunk_count(self) -> int:
        """真正携带 token 的 chunk 数（不含 role-only / 空 chunk）。"""
        if self.text_chunk_times is not None:
            return len(self.text_chunk_times)
        return len(self._token_arrivals())

    @property
    def tokens_per_chunk(self) -> float:
        c = self.chunk_count
        return (self.output_len / c) if c else float("nan")

    @property
    def first_token_sec(self) -> float:
        if self.text_chunk_times:
            return self.text_chunk_times[0]
        ta = self._token_arrivals()
        if not ta:
            raise BenchError(f"{self.req_id}: stream produced no tokens")
        return ta[0].t_sec

    @property
    def end_sec(self) -> float:
        ta = self._token_arrivals()
        if self.text_chunk_times:
            last_token = self.text_chunk_times[-1]
        elif ta:
            last_token = ta[-1].t_sec
        else:
            raise BenchError(f"{self.req_id}: stream produced no tokens")
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
    """解析单输出文本流，保留正文时序，结束后优先使用服务端 usage 总数。

    无 usage 时，对拼接后的完整正文调用 token_counter，避免跨 chunk 分词不满足
    可加性。正文 chunk 的时刻与 token 总数独立保存，usage-only 收尾不改变 TTFT。
    纯 usage 时间线仅用于合成测试，不能据此推断真实模型的首 token 时刻。
    latency 取最后 payload 或 [DONE] 的时刻。
    """
    arrivals: list[TokenArrival] = []
    text_times: list[float] = []
    text_parts: list[str] = []
    usage_total: int | None = None
    stream_end_sec: float | None = None
    for t_sec, payload in timed_payloads:
        stream_end_sec = t_sec
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        text = _extract_text(obj)
        if text:
            text_times.append(t_sec)
            text_parts.append(text)
        cum = _usage_cumulative(obj.get("usage") or {})
        if cum is not None:
            if cum < 0 or (usage_total is not None and cum < usage_total):
                raise BenchError(f"{req_id}: usage count must be nonnegative and monotonic")
            delta = cum - (usage_total or 0)
            if delta > 0:
                arrivals.append(TokenArrival(t_sec, delta))
            usage_total = cum

    text_total = None
    if text_parts and usage_total is None:
        if token_counter is None:
            raise BenchError(
                f"{req_id}: chunk carries text but token_counter is None. "
                "chunk count != token count — pass a tokenizer or have the "
                "server send usage.completion_tokens; do NOT count chunks."
            )
        text_total = token_counter("".join(text_parts))
        if not isinstance(text_total, int) or isinstance(text_total, bool) or text_total < 0:
            raise BenchError(f"{req_id}: token_counter must return a nonnegative integer")
    return RequestTrace(
        req_id, start_sec, arrivals, usage_total, stream_end_sec,
        text_times if text_times else None, text_total,
    )


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
