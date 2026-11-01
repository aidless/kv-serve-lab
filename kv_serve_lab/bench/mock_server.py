"""mock SSE 服务端：把一份「脚本化时间线」按真实墙钟回放成 OpenAI 兼容流。

用途：
  1. 端到端验证 bench 客户端（真 socket、真时间戳）能正确解析 SSE、测指标；
  2. 复刻 schedsim 的预测（first token 在 ttft、之后每 tpot 出一 token），
     让 bench 的实测指标能跟 schedsim 的理论值对上（设计文档 §5 闭环）。

**同时覆盖两种服务行为**（设计文档要求）：
  - tokens_per_chunk=1：一 chunk 一 token
  - tokens_per_chunk>1：一 chunk 多 token（用于证明 bench 不会把 chunk 当 token）
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__all__ = ["SSEMockServer", "chunked_plan", "per_token_plan"]


def chunked_plan(*, num_tokens, ttft_sec, tpot_sec, tokens_per_chunk=1, word="tok"):
    """生成 (delay_sec, chunk_obj) 时间线。

    首 token 出现在 ttft_sec；之后每隔 tpot_sec 再出一批。一批 tokens_per_chunk 个。
    chunk 文本用空格分隔的词表示 token，好让 word_counter 正确计数。
    """
    if num_tokens < 1:
        raise ValueError("num_tokens must be >= 1")
    plan: list[tuple[float, dict]] = []
    emitted = 0
    delay = ttft_sec
    while emitted < num_tokens:
        k = min(tokens_per_chunk, num_tokens - emitted)
        text = " ".join([word] * k)
        plan.append((delay, {"choices": [{"text": text}]}))
        emitted += k
        delay = tpot_sec
    return plan


def per_token_plan(*, num_tokens, ttft_sec, tpot_sec):
    return chunked_plan(
        num_tokens=num_tokens, ttft_sec=ttft_sec, tpot_sec=tpot_sec,
        tokens_per_chunk=1,
    )


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            body = {}
        plan = self.server.plan_fn(body)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for delay, obj in plan:
                if delay > 0:
                    time.sleep(delay)
                self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):  # 静音
        pass


class SSEMockServer:
    """plan_fn(body:dict) -> list[(delay_sec, chunk_obj)]。

    用法：
        srv = SSEMockServer(lambda body: per_token_plan(num_tokens=5, ttft_sec=.02, tpot_sec=.01))
        url = srv.url  # http://127.0.0.1:<port>/v1/completions
        ...
        srv.close()
    """

    def __init__(self, plan_fn, host: str = "127.0.0.1", port: int = 0):
        self._httpd = ThreadingHTTPServer((host, port), _Handler)
        self._httpd.plan_fn = plan_fn
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        self.host, self.port = self._httpd.server_address[:2]

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/v1/completions"

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
