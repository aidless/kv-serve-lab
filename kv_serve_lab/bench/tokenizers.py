"""真实 tokenizer 适配层（可选依赖）。

本仓库运行时是纯标准库，这一层只在**需要真分词**时才被导入：
`bench` 拿到服务端 `usage.completion_tokens` 时根本不需要它；只有在没有 usage
的流式响应上，才必须用真分词器把拼接后的完整输出文本数一遍。

为什么不能逐 chunk 数再相加：分词不满足可加性，实测 Qwen2.5 上
`("b", "ar")` 分别分词得 2，拼成 `"bar"` 只得 1（高估）；而
`("super", "califragilistic")` 分别是 6，拼起来却是 7（低估）。两个方向都会错。

用法：

    from kv_serve_lab.bench.tokenizers import HFTokenCounter
    counter = HFTokenCounter("Qwen/Qwen2.5-7B-Instruct")
    trace = stream_request(url, prompt, token_counter=counter)

安装：`pip install "kv-serve-lab[tokenizer]"`
"""

from __future__ import annotations

from typing import Callable

_INSTALL_HINT = (
    '未安装分词器后端。bench 只有在响应里没有 usage.completion_tokens 时才需要真 '
    'tokenizer，安装方式：pip install "kv-serve-lab[tokenizer]"'
)


class HFTokenCounter:
    """把 HuggingFace 分词器包装成 `str -> int` 的计数函数。

    优先用 `tokenizers`（Rust 实现，快且不需要 transformers 的一堆依赖），
    退回到 `transformers.AutoTokenizer`。两者都要求本地已缓存该模型的分词器，
    或允许联网下载（`local_files_only=False`）。
    """

    __slots__ = ("model_id", "_backend")

    def __init__(
        self,
        model_id: str,
        *,
        local_files_only: bool = False,
        prefer: str = "tokenizers",
    ) -> None:
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be a non-empty string")
        if prefer not in {"tokenizers", "transformers"}:
            raise ValueError("prefer must be 'tokenizers' or 'transformers'")
        self.model_id = model_id
        self._backend = self._load(model_id, local_files_only, prefer)

    @staticmethod
    def _load(model_id: str, local_files_only: bool, prefer: str) -> Callable[[str], int]:
        order = ("tokenizers", "transformers") if prefer == "tokenizers" \
            else ("transformers", "tokenizers")
        errors: list[str] = []
        for backend in order:
            try:
                if backend == "tokenizers":
                    from tokenizers import Tokenizer  # 可选依赖

                    tok = Tokenizer.from_pretrained(
                        model_id, local_files_only=local_files_only
                    )

                    def count(text: str, _tok=tok) -> int:
                        return len(_tok.encode(text).ids)

                    return count
                from transformers import AutoTokenizer  # 可选依赖

                tok = AutoTokenizer.from_pretrained(
                    model_id, local_files_only=local_files_only
                )

                def count(text: str, _tok=tok) -> int:
                    return len(_tok(text)["input_ids"])

                return count
            except ImportError as exc:      # 后端没装：试下一个
                errors.append(f"{backend}: {exc}")
            except Exception as exc:        # 后端装了但模型不可用：也要说清楚
                errors.append(f"{backend}: {type(exc).__name__}: {exc}")
        raise ImportError(f"{_INSTALL_HINT} —— 各后端尝试结果：{'; '.join(errors)}")

    def __call__(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        return int(self._backend(text))

    def __repr__(self) -> str:
        return f"HFTokenCounter(model_id={self.model_id!r})"


def hf_token_counter(model_id: str, **kwargs) -> HFTokenCounter:
    """便捷入口，等价于 `HFTokenCounter(model_id, **kwargs)`。"""
    return HFTokenCounter(model_id, **kwargs)
