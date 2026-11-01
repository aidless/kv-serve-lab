"""块池操作协议：一行一个操作，一行一个结果。

这不是给人用的 CLI，而是一份**可逐行比对的语言**：`cpp/kv_block_manager.cpp`
是同一协议的 C++ 实现，`tests/test_cpp_differential.py` 把同一段脚本喂给两边，
要求逐行完全一致。任何语义分歧都会在某一行的负载或错误码上暴露出来。

设计约束（改了会破坏对拍能力，别随手改）：
  - 每个操作恰好一行，`OK <负载>` 或 `ERR <错误码>`；
  - 空集合统一写 `-`，不写空串（避免行尾空格这种看不见的差异）；
  - 负载只含整数、逗号与固定词表，不含浮点、不依赖 dict 顺序；
  - 错误码是有限词表，见下；不把异常文本搬进输出（措辞会变）。

错误码词表：
    unknown_block  操作了不存在的块
    out_of_blocks  空闲块不够
    no_such_hash   前缀索引里没有这个 hash
    not_in_use     对空闲块做只有 in_use 块才允许的操作
    double_free    同一批里重复释放同一个块
    not_writable   块不可写（被共享或已登记前缀）
    no_payload     块里没有 payload
    cow_free       对空闲块做 copy_on_write
    conservation   分区/索引不变量被破坏
    bad_args       参数本身非法（数值越界、类型/长度不对）
    bad_op         未知操作名或参数个数不对
    internal       抛了 BlockPoolError 但不属于上面任何一类（说明新增了 raise 却没归类，
                   对拍测试会因此变红，逼着把词表补齐）
"""

from __future__ import annotations

from typing import Iterable

from . import BlockPool, BlockPoolError

# 错误码前缀匹配表：按顺序匹配异常文本，宁可漏判成 internal 也不要错判。
_PREFIX_TO_CODE = (
    ("unknown block", "unknown_block"),
    ("out of blocks", "out_of_blocks"),
    ("no cached block for hash", "no_such_hash"),
    ("cannot retain free block", "not_in_use"),
    ("double free of block", "double_free"),
    ("is not writable", "not_writable"),
    ("has no payload", "no_payload"),
    ("cannot copy_on_write free block", "cow_free"),
    ("conservation violated", "conservation"),
)

_CODES = frozenset(
    code for _, code in _PREFIX_TO_CODE
) | {"bad_args", "bad_op", "internal"}


def error_code(exc: BaseException) -> str:
    """把 Python 侧异常折算成协议里的稳定错误码。"""
    if isinstance(exc, ValueError) and not isinstance(exc, BlockPoolError):
        return "bad_args"
    text = str(exc)
    # 用子串而不是前缀："block 0 is not writable" 并不是以 "is not writable" 开头
    for prefix, code in _PREFIX_TO_CODE:
        if prefix in text:
            return code
    return "internal"


def _ids(values: Iterable[int]) -> str:
    items = list(values)
    return ",".join(str(v) for v in items) if items else "-"


def _parse_ints(tokens: list[str], what: str) -> list[int]:
    try:
        return [int(t) for t in tokens]
    except ValueError as exc:
        raise ValueError(f"{what} must be integers: {tokens}") from exc


def apply_line(pool: BlockPool, line: str) -> str:
    """执行一行操作，返回一行结果（不抛异常）。"""
    parts = line.split()
    if not parts:
        raise ValueError("empty line is not a valid op")
    op, args = parts[0], parts[1:]

    def need(n: int) -> None:
        if len(args) != n:
            raise ValueError(f"{op} expects {n} arguments, got {len(args)}")

    if op == "alloc":
        need(1)
        return f"OK {_ids(pool.allocate(int(args[0])))}"
    if op in {"free", "retain", "touch"}:
        if not args:
            raise ValueError(f"{op} expects at least one block id")
        getattr(pool, op)(_parse_ints(args, "block ids"))
        return "OK"
    if op == "register":
        if len(args) not in (2, 3):
            raise ValueError("register expects HASH ID [NUM_TOKENS]")
        num_tokens = int(args[2]) if len(args) == 3 else None
        pool.register_prefix(args[0], int(args[1]), num_tokens)
        return "OK"
    if op == "lookup":
        need(1)
        found = pool.lookup_prefix(args[0])
        return f"OK {found if found is not None else '-'}"
    if op == "acquire":
        need(1)
        return f"OK {pool.acquire_prefix(args[0])}"
    if op == "cow":
        need(1)
        return f"OK {pool.copy_on_write(int(args[0]))}"
    if op == "write":
        if len(args) < 2:
            raise ValueError("write expects ID V [V ...]")
        pool.write_payload(int(args[0]), _parse_ints(args[1:], "payload"))
        return "OK"
    if op == "read":
        need(1)
        return f"OK {_ids(pool.read_payload(int(args[0])))}"
    if op == "state":
        need(1)
        return f"OK {pool.state_of(int(args[0])).value}"
    if op == "refcount":
        need(1)
        return f"OK {pool.ref_count_of(int(args[0]))}"
    if op == "hash":
        need(1)
        found = pool.hash_of(int(args[0]))
        return f"OK {found if found is not None else '-'}"
    if op == "writable":
        need(1)
        return "OK true" if pool.is_block_writable(int(args[0])) else "OK false"
    if op == "audit":
        need(0)
        a = pool.audit()
        return (f"OK total={a.total} used={a.used} free={a.free} "
                f"cached={a.cached}")
    if op == "conserve":
        need(0)
        pool._check_conservation()
        return "OK"
    raise ValueError(f"unknown op {op!r}")


def run_script(pool: BlockPool, lines: Iterable[str]) -> list[str]:
    """逐行执行，逐行返回结果；空行与 `#` 注释行被跳过（不产出结果行）。"""
    out: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(apply_line(pool, line))
        except Exception as exc:                     # noqa: BLE001 - 协议要求不中断
            code = "bad_op" if str(exc).startswith("unknown op") else error_code(exc)
            out.append(f"ERR {code}")
    return out


def main(argv: list[str] | None = None) -> int:      # pragma: no cover - 手工调试用
    """从 stdin 读脚本、打到 stdout；参数：num_blocks [block_size]。"""
    import sys

    args = sys.argv[1:] if argv is None else argv
    num_blocks = int(args[0]) if args else 8
    block_size = int(args[1]) if len(args) > 1 else 16
    for line in run_script(BlockPool(num_blocks, block_size), sys.stdin):
        print(line)
    return 0


if __name__ == "__main__":                            # pragma: no cover
    raise SystemExit(main())
