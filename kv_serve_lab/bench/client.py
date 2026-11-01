"""真实流式 HTTP 客户端：向 OpenAI 兼容端点发请求，逐行打时间戳。

刻意「朴素」：只用标准库 urllib，不引入第三方依赖。逐行迭代响应体（SSE 服务端
每 chunk flush），用 time.perf_counter() 在**读到该行时**打戳——这是 TTFT 的真实
来源（首 token 的到达时刻）。

注意：TTFT 会包含本机 HTTP 往返与解析开销（毫秒级）。因此本客户端适合做
「管线是否通、解析是否正确」的集成验证；要把 bench 的数字对齐 schedsim 的
理论预测（§5 闭环），用带假时钟的确定性路径（见 tests），不要依赖真 socket 的
绝对时间精度。
"""

from __future__ import annotations

import json
import time
import urllib.request

from . import RequestTrace, trace_from_payloads

__all__ = ["stream_request", "payloads_from_timed_lines"]


def payloads_from_timed_lines(timed_lines):
    """(t_sec, raw_line) -> (t_sec, payload_str)。保留每行自己的时间戳。"""
    for t_sec, raw in timed_lines:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "replace")
        s = raw.strip()
        if not s or not s.startswith("data:"):
            continue
        yield t_sec, s[5:].strip()


def _collect_timed_lines(url: str, payload: dict, timeout: float):
    """发请求并记录共享单调时钟时间戳，同时返回请求起点。"""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    timed: list[tuple[float, str]] = []
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:  # 逐行迭代，服务端 flush 后即时拿到
            t = time.perf_counter()
            timed.append((t, raw.decode("utf-8", "replace")))
    return t0, timed


def stream_request(
    url: str,
    prompt: str,
    *,
    max_tokens: int = 16,
    req_id: str = "req-0",
    token_counter=None,
    timeout: float = 30.0,
) -> RequestTrace:
    """发一个流式请求，保留共享单调时钟，以便多个 trace 汇总吞吐。

    无服务端 usage 时需要 token_counter，输入为拼接后的完整输出文本。
    当前仅发送 prompt/max_tokens/stream，面向本地 mock；真实服务的 model、鉴权
    和 stream_options 配置尚未提供。
    """
    start_sec, timed = _collect_timed_lines(
        url, {"prompt": prompt, "max_tokens": max_tokens, "stream": True}, timeout
    )
    payloads = payloads_from_timed_lines(timed)
    return trace_from_payloads(req_id, start_sec, payloads, token_counter=token_counter)
