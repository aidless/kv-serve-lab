"""C++ 块管理器与 Python 版**逐操作对拍**。

为什么要两份实现：`blockpool` 的不变量（两个分区、弱引用 hash 索引、LIFO/FIFO
两种空闲队列次序、COW 三种分支）靠读代码保不住 —— 极易在重构时悄悄漂移。
同一份操作脚本喂给两套独立实现、要求逐行输出完全一致，是把这些语义钉死的
最便宜办法，也让「我理解了 vLLM 的块管理」这句话变成可执行的证据。

对拍用的是**同一份协议**（见 `kv_serve_lab/blockpool/cli.py` 顶部说明）：
每行 `OK <负载>` 或 `ERR <错误码>`，两边必须一模一样。

依赖：需要 `g++`。没有就 skip（并在 README 里写明 CI 会跑）。
编译失败**不**skip —— 那是真回归，必须红。
"""

import random
import shutil
import subprocess
from pathlib import Path

import pytest

from kv_serve_lab.blockpool import BlockPool
from kv_serve_lab.blockpool.cli import run_script

REPO_ROOT = Path(__file__).resolve().parents[1]
CPP_DIR = REPO_ROOT / "cpp"

pytestmark = pytest.mark.skipif(
    shutil.which("g++") is None, reason="g++ not available; C++ differential skipped"
)


def _build() -> Path:
    """编译 C++ driver；失败就让测试红，不隐藏。"""
    result = subprocess.run(
        ["make", "-C", str(CPP_DIR)],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"C++ build failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    binary = CPP_DIR / "build" / "kvbm"
    assert binary.exists(), f"make succeeded but {binary} is missing"
    return binary


def _run_cpp(binary: Path, num_blocks: int, block_size: int, script: list[str]) -> list[str]:
    proc = subprocess.run(
        [str(binary), str(num_blocks), str(block_size)],
        input="\n".join(script) + "\n",
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"driver exited {proc.returncode}: {proc.stderr}"
    return proc.stdout.splitlines()


def _random_script(rng: random.Random, num_blocks: int, block_size: int,
                   length: int) -> list[str]:
    """造一段混合合法/非法操作的脚本 —— 非法操作正是错误码对齐的重点。"""
    ids = list(range(num_blocks + 3))        # 多出几个必定不存在/越界的 id
    hashes = [f"h{i}" for i in range(6)]
    script: list[str] = []
    for _ in range(length):
        op = rng.choices(
            ["alloc", "free", "retain", "touch", "register", "lookup",
             "acquire", "cow", "write", "read", "state", "refcount",
             "hash", "writable", "audit", "conserve", "garbage"],
            weights=[8, 6, 3, 2, 5, 4, 3, 3, 4, 3, 2, 2, 2, 2, 2, 1, 1],
        )[0]
        if op == "alloc":
            script.append(f"alloc {rng.choice([0, 1, 1, 2, 3, num_blocks])}")
        elif op in {"free", "retain", "touch"}:
            picked = rng.sample(ids, rng.choice([1, 1, 2, 3]))
            script.append(f"{op} " + " ".join(map(str, picked)))
        elif op == "register":
            parts = [rng.choice(hashes), str(rng.choice(ids))]
            if rng.random() < 0.6:
                parts.append(str(rng.choice([block_size, block_size, block_size - 1,
                                             block_size + 1])))
            script.append("register " + " ".join(parts))
        elif op in {"lookup", "acquire"}:
            script.append(f"{op} {rng.choice(hashes + ['nope'])}")
        elif op == "cow":
            script.append(f"cow {rng.choice(ids)}")
        elif op == "write":
            n = rng.choice([0, 1, block_size, block_size + 1])
            script.append(f"write {rng.choice(ids)} " +
                          " ".join(str(rng.randrange(1000)) for _ in range(n)))
        elif op in {"read", "state", "refcount", "hash", "writable"}:
            script.append(f"{op} {rng.choice(ids)}")
        elif op in {"audit", "conserve"}:
            script.append(op)
        else:
            script.append("frobnicate 1")
    return [line for line in script if line.split()[0] != "write" or len(line.split()) > 2]


class TestDifferential:
    @pytest.mark.parametrize("seed", [1, 2, 3, 7, 11, 20261031])
    def test_random_scripts_match_line_by_line(self, seed):
        binary = _build()
        rng = random.Random(seed)
        num_blocks = rng.choice([4, 8, 13])
        block_size = rng.choice([2, 4])
        script = _random_script(rng, num_blocks, block_size, length=200)
        expected = run_script(BlockPool(num_blocks, block_size), script)
        actual = _run_cpp(binary, num_blocks, block_size, script)
        assert len(actual) == len(expected), (
            f"line count differs: cpp={len(actual)} py={len(expected)}"
        )
        for i, (line, want) in enumerate(zip(script, expected)):
            assert actual[i] == want, f"script line {i}: {line!r} -> cpp {actual[i]!r} != py {want!r}"

    def test_lifo_reuse_of_unhashed_blocks(self):
        """无 hash 的空闲块 LIFO 插队首：刚释放的立刻被复用。"""
        binary = _build()
        script = ["alloc 3", "free 0", "alloc 1", "free 1 2", "alloc 2"]
        expected = run_script(BlockPool(4, 2), script)
        assert expected == ["OK 0,1,2", "OK", "OK 0", "OK", "OK 2,1"]
        assert _run_cpp(binary, 4, 2, script) == expected

    def test_fifo_eviction_of_hashed_blocks(self):
        """有 hash 的空闲块 FIFO 插队尾：先释放的排在淘汰序列前面。"""
        binary = _build()
        script = ["alloc 2", "register h0 0 2", "register h1 1 2",
                  "free 0 1", "audit", "alloc 2", "audit"]
        expected = run_script(BlockPool(4, 2), script)
        assert _run_cpp(binary, 4, 2, script) == expected

    def test_copy_on_write_paths(self):
        binary = _build()
        script = [
            "alloc 1",              # 独占、无 hash -> cow 是 no-op
            "cow 0",
            "retain 0",             # 共享 -> cow 真复制
            "cow 0",
            "audit",
            "write 0 11 12",
            "register h0 0 2",      # 独占但有 hash -> cow 走第三支
            "cow 0",
            "audit",
            "hash 0",
            "hash 1",
        ]
        expected = run_script(BlockPool(4, 2), script)
        assert _run_cpp(binary, 4, 2, script) == expected

    def test_conservation_holds_throughout(self):
        binary = _build()
        rng = random.Random(99)
        script = _random_script(rng, 6, 2, length=120) + ["conserve", "audit"]
        expected = run_script(BlockPool(6, 2), script)
        assert expected[-2:] == ["OK", expected[-1]]
        assert _run_cpp(binary, 6, 2, script) == expected
