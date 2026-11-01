"""真实 tokenizer 接入（可选依赖）。

为什么必须有这一层：`bench` 的 `output_len` 只有在「服务端吐 usage」时才免费拿到。
没有 usage 时唯一诚实的办法是拿真正的分词器把**拼接后的完整输出文本**数一遍。
逐 chunk 分词再相加是错的 —— 分词不满足可加性，实测 Qwen2.5 的分词器上：

    ('b',   'ar')               -> 分别 2，拼起来 1（高估）
    ('super', 'califragilistic') -> 分别 6，拼起来 7（**低估**）

两个方向都会错，所以下面专门有一条测试钉住「不可加性」这个事实，
防止以后有人把实现改回逐 chunk 累加。

依赖纪律：本仓库运行时是纯标准库。tokenizers / transformers 只在这条路径上
按需导入，缺了就抛一条带安装提示的 ImportError，不能让整个包 import 失败。
"""

import builtins
import json
import sys

import pytest

from kv_serve_lab.bench import RequestTrace, trace_from_payloads

QWEN = "Qwen/Qwen2.5-7B-Instruct"


def _payload(text):
    return json.dumps({"choices": [{"text": text}]})


def _real_tokenizer_available() -> bool:
    """本地缓存里有没有可离线加载的分词器（CI 上没有 -> 相关用例跳过）。"""
    try:
        from kv_serve_lab.bench.tokenizers import HFTokenCounter
    except ImportError:
        return False
    try:
        HFTokenCounter(QWEN, local_files_only=True)
    except Exception:
        return False
    return True


needs_tokenizer = pytest.mark.skipif(
    not _real_tokenizer_available(),
    reason=f"no locally cached tokenizer for {QWEN} (offline CI skips this)",
)


class TestMissingBackend:
    def test_import_error_names_the_install_hint(self, monkeypatch):
        """tokenizers/transformers 都不可用时，报错必须告诉人怎么装。"""
        from kv_serve_lab.bench.tokenizers import HFTokenCounter

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.split(".")[0] in {"tokenizers", "transformers"}:
                raise ModuleNotFoundError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        for mod in [m for m in list(sys.modules) if m.split(".")[0] in
                    {"tokenizers", "transformers"}]:
            monkeypatch.delitem(sys.modules, mod, raising=False)
        with pytest.raises(ImportError, match=r"kv-serve-lab\[tokenizer\]"):
            HFTokenCounter(QWEN)


class TestArgumentValidation:
    def test_call_rejects_non_str(self):
        """__call__ 只接受 str —— 喂 bytes 会静默数错，所以直接拒绝。"""
        from kv_serve_lab.bench.tokenizers import HFTokenCounter

        counter = HFTokenCounter.__new__(HFTokenCounter)   # 绕开真实模型下载
        counter._backend = lambda text: len(text.split())
        assert counter("hello world") == 2
        with pytest.raises(TypeError):
            counter(b"hello")


class TestRealTokenizer:
    @needs_tokenizer
    def test_counts_match_transformers_exactly(self):
        """同一段文本，本适配层与 transformers.AutoTokenizer 必须给出同一个数。"""
        from transformers import AutoTokenizer

        from kv_serve_lab.bench.tokenizers import HFTokenCounter

        ref = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
        counter = HFTokenCounter(QWEN, local_files_only=True)
        for text in ["hello world", "推理引擎的显存占用", "tok tok tok tok", ""]:
            assert counter(text) == len(ref(text)["input_ids"]), text

    @needs_tokenizer
    def test_tokenization_is_not_additive(self):
        """钉住「分词不可加性」：逐 chunk 累加会高估也会低估，两个方向都要有。"""
        from kv_serve_lab.bench.tokenizers import HFTokenCounter

        counter = HFTokenCounter(QWEN, local_files_only=True)

        over = ("b", "ar")          # 分别 2，拼起来 1 -> 高估
        under = ("super", "califragilistic")  # 分别 6，拼起来 7 -> 低估
        assert counter(over[0]) + counter(over[1]) > counter("".join(over))
        assert counter(under[0]) + counter(under[1]) < counter("".join(under))


class TestBenchIntegration:
    def test_trace_uses_counter_on_concatenated_text(self):
        """bench 拿到 tokenizer 时必须对**拼接后**的文本计数，不是逐 chunk 数。"""

        class FakeTokenizer:
            """只数词：'b' + 'ar' 拼成 'bar' 是一个词。"""

            def __call__(self, text):
                return len(text.split())

        payloads = [(1.0, _payload("b ")), (1.1, _payload("ar ")), (1.2, _payload("baz "))]
        trace = trace_from_payloads("r", 0.0, payloads,
                                    token_counter=FakeTokenizer())
        assert isinstance(trace, RequestTrace)
        assert trace.output_len == 3          # "b ar baz" 三个词
        assert trace.chunk_count == 3
        assert trace.ttft_sec == pytest.approx(1.0)

    def test_counter_returning_bool_or_negative_is_rejected(self):
        for bad in (True, -1, 2.5):
            with pytest.raises((TypeError, ValueError)):
                trace_from_payloads("r", 0.0, [(1.0, _payload("hi"))],
                                    token_counter=lambda text, b=bad: b)

    @needs_tokenizer
    def test_end_to_end_output_len_with_real_tokenizer(self):
        """真分词器 + 无 usage 的多 chunk 流：output_len 必须等于拼接后的真值。"""
        from kv_serve_lab.bench.tokenizers import HFTokenCounter

        counter = HFTokenCounter(QWEN, local_files_only=True)
        chunks = ["b", "ar", "baz"]
        payloads = [(1.0 + 0.1 * i, _payload(c)) for i, c in enumerate(chunks)]
        trace = trace_from_payloads("r", 0.0, payloads, token_counter=counter)
        assert trace.output_len == counter("".join(chunks))
        assert trace.output_len != sum(counter(c) for c in chunks)
