"""块池操作协议（`kv_serve_lab.blockpool.cli`）——C++ 版对拍的公共语言。

C++ 版 `cpp/kv_block_manager.cpp` 要逐操作对齐 Python 版，两边必须能输出**逐行可比**
的同一份结果。所以协议要满足三条：

1. 每个操作恰好产出一行，`OK <负载>` 或 `ERR <错误码>`；
2. 错误码来自稳定的有限集合，不把 Python 的异常文本原样搬过去（措辞随时会改）；
3. 负载里只出现整数与固定词表，不含浮点、不依赖字典顺序。

错误码词表（两边必须一致）：
    unknown_block / out_of_blocks / no_such_hash / not_in_use / double_free /
    not_writable / no_payload / cow_free / conservation / bad_args / bad_op
"""

import pytest

from kv_serve_lab.blockpool import BlockPool
from kv_serve_lab.blockpool.cli import run_script


def _run(lines, num_blocks=4, block_size=2):
    return run_script(BlockPool(num_blocks, block_size), lines)


class TestBasicOps:
    def test_alloc_and_audit(self):
        out = _run(["alloc 2", "audit"])
        assert out == ["OK 0,1", "OK total=4 used=2 free=2 cached=0"]

    def test_free_and_realloc_order(self):
        """未登记 hash 的空闲块 LIFO 插队首：刚释放的块下一个就被复用。"""
        out = _run(["alloc 2", "free 0", "alloc 1"])
        assert out == ["OK 0,1", "OK", "OK 0"]

    def test_refcount_state_and_writable(self):
        out = _run(["alloc 1", "retain 0", "refcount 0", "state 0", "writable 0"])
        assert out == ["OK 0", "OK", "OK 2", "OK in_use", "OK false"]

    def test_write_read_payload(self):
        out = _run(["alloc 1", "write 0 7 8", "read 0"])
        assert out == ["OK 0", "OK", "OK 7,8"]

    def test_read_blank_block_reports_no_payload(self):
        assert _run(["alloc 1", "read 0"]) == ["OK 0", "ERR no_payload"]

    def test_write_too_long_is_bad_args(self):
        # block_size=2，写 3 个 token 越界
        assert _run(["alloc 1", "write 0 1 2 3"]) == ["OK 0", "ERR bad_args"]

    def test_empty_alloc_is_legal_and_writes_a_dash(self):
        assert _run(["alloc 0", "audit"]) == ["OK -", "OK total=4 used=0 free=4 cached=0"]


class TestPrefixCache:
    def test_register_lookup_acquire(self):
        out = _run(["alloc 1", "register h1 0 2", "lookup h1", "acquire h1",
                    "refcount 0"])
        assert out == ["OK 0", "OK", "OK 0", "OK 0", "OK 2"]

    def test_lookup_miss_is_ok_dash_but_acquire_raises(self):
        out = _run(["lookup nope", "acquire nope"])
        assert out == ["OK -", "ERR no_such_hash"]

    def test_partial_register_is_bad_args(self):
        # block_size=2，只登记 1 个 token 不算整块
        assert _run(["alloc 1", "register h1 0 1"]) == ["OK 0", "ERR bad_args"]

    def test_cached_only_counts_blocks_the_index_points_back_to(self):
        out = _run(["alloc 2", "register h1 0 2", "free 0", "audit"])
        assert out == ["OK 0,1", "OK", "OK", "OK total=4 used=1 free=3 cached=1"]


class TestCopyOnWrite:
    def test_shared_block_copies(self):
        out = _run(["alloc 1", "retain 0", "cow 0", "state 1", "refcount 0"])
        assert out == ["OK 0", "OK", "OK 1", "OK in_use", "OK 1"]

    def test_sole_unhashed_owner_is_a_noop(self):
        assert _run(["alloc 1", "cow 0"]) == ["OK 0", "OK 0"]

    def test_cow_on_free_block_is_an_error(self):
        out = _run(["alloc 1", "free 0", "cow 0"])
        assert out == ["OK 0", "OK", "ERR cow_free"]


class TestErrorCodes:
    @pytest.mark.parametrize("line,code", [
        ("state 9", "unknown_block"),
        ("retain 9", "unknown_block"),
        ("cow 9", "unknown_block"),
        ("alloc 9", "out_of_blocks"),
        ("free 99", "unknown_block"),
    ])
    def test_error_codes(self, line, code):
        assert _run([line]) == [f"ERR {code}"]

    def test_retain_free_block_is_not_in_use(self):
        out = _run(["alloc 1", "free 0", "retain 0"])
        assert out == ["OK 0", "OK", "ERR not_in_use"]

    def test_double_free_in_one_batch(self):
        out = _run(["alloc 1", "free 0 0"])
        assert out == ["OK 0", "ERR double_free"]

    def test_not_writable_after_register(self):
        out = _run(["alloc 1", "write 0 1 2", "register h 0 2", "writable 0"])
        assert out == ["OK 0", "OK", "OK", "OK false"]

    def test_unknown_op(self):
        assert _run(["frobnicate 1"]) == ["ERR bad_op"]


class TestConservation:
    def test_conserve_passes_on_healthy_pool(self):
        assert _run(["alloc 2", "free 1", "conserve"]) == ["OK 0,1", "OK", "OK"]
