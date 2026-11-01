"""PagedAttention 风格的 KV cache 块管理器。

语义对齐 vLLM 的 vllm/v1/core/block_pool.py：

  :706 get_new_blocks(num_blocks)  从空闲队列取块
  :792 touch(blocks)               标记为最近使用
  :816 free_blocks(ordered)        引用计数减一，归零则回空闲队列
  :202 get_cached_block(hash)      前缀命中
  :230 cache_full_blocks           登记 hash→block

关于「一个 hash 对应几个块」——**本库的 hash→block 单值映射是刻意的简化，
不是说 vLLM 不做去重**：
- vLLM 的 ``BlockHashToBlockMap``（block_pool.py:36）允许一个 hash 影射到一个
  或多个 block（不同大小/分组），并且**相同内容的块共享同一个 canonical block**，
  这本身就是前缀去重（dedup）。
- 本库把它简化成「一个 hash 只记一个 block id」，是为了在 CPU 上讲清
  「命中 / 引用计数 / LRU / 逐出」的机制，而不是复刻 vLLM 的多块映射与去重策略。
  简化的是数据结构，不是「vLLM 不去重」。

与 vLLM 的差异（本库刻意简化，写在明处）：
- vLLM 的 free 队列是「双端队列 + 中间插 hash 块」，这里用 OrderedDict 表达同一语义
- vLLM 有「cached block 被 acquire 后从空闲队列摘下」的细节，这里用状态机表达
- 本库新增 block 内 payload 的读/写接口与 copy-on-write（COW），便于 CPU 教学；
  vLLM v1 的 COW 判据见 block_pool.py:812 ``is_block_writable``。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum


class BlockPoolError(RuntimeError):
    """块池不变量被破坏时抛这个，而不是静默继续。"""


class BlockState(str, Enum):
    FREE = "free"
    IN_USE = "in_use"


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    block_hash: str | None = None
    # 块内 token 内容。None 表示尚未写入（不是空块，是「未知」）。
    payload: list[int] | None = field(default=None)

    @property
    def state(self) -> BlockState:
        return BlockState.IN_USE if self.ref_count > 0 else BlockState.FREE


@dataclass(frozen=True)
class BlockAudit:
    """任何时刻的快照：used + free == total 必须永远成立。"""

    total: int
    used: int
    free: int
    cached: int  # 空闲但带 hash 且 hash 仍指向自己，可以被 prefix 命中复用


class BlockPool:
    """固定容量的 KV 块池。

    LRU 顺序用 OrderedDict 维护：**队首是最久未用的（下一个被淘汰），
    队尾是最近用过的**。这与 vLLM 的 free_queue 语义一致
    （vLLM 用 deque，append 到右端、popleft 从左端取）。
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._blocks: dict[int, Block] = {
            i: Block(block_id=i) for i in range(num_blocks)
        }
        # 空闲块的 LRU 顺序：左 = 最久未用（下一个淘汰），右 = 最近用过
        self._free: OrderedDict[int, None] = OrderedDict(
            (i, None) for i in range(num_blocks)
        )
        # hash -> block_id。简化：一个 hash 只记一个块（见模块 docstring）
        self._hash_to_block: dict[str, int] = {}
        # 正在使用的块，按 LRU 顺序（用于 touch）
        self._in_use: OrderedDict[int, None] = OrderedDict()

    # ---- 查询 ----

    def state_of(self, block_id: int) -> BlockState:
        blk = self._blocks.get(block_id)
        if blk is None:
            raise BlockPoolError(f"unknown block {block_id}")
        return blk.state

    def ref_count_of(self, block_id: int) -> int:
        blk = self._blocks.get(block_id)
        if blk is None:
            raise BlockPoolError(f"unknown block {block_id}")
        return blk.ref_count

    def hash_of(self, block_id: int) -> str | None:
        blk = self._blocks.get(block_id)
        if blk is None:
            raise BlockPoolError(f"unknown block {block_id}")
        return blk.block_hash

    def audit(self) -> BlockAudit:
        """从真实分区（_free / _in_use）导出计数，不靠恒等式倒推。"""
        used = len(self._in_use)
        free = len(self._free)
        cached = sum(
            1
            for bid, blk in self._blocks.items()
            if blk.ref_count == 0
            and blk.block_hash is not None
            and self._hash_to_block.get(blk.block_hash) == bid
        )
        return BlockAudit(
            total=self.num_blocks,
            used=used,
            free=free,
            cached=cached,
        )

    def _check_conservation(self) -> None:
        """真正的守恒检查：比对 free/in_use 分区、互斥性、覆盖性、引用计数一致性。

        旧实现只验 ``used + free == total``（而 used/free 都由 total 倒推），
        是恒等式、永远成立、等于没检查。这里逐块核对：
        1) 每个块恰好属于 free 或 in_use 之一（互斥且覆盖全部）；
        2) free 里的块 ref_count 必须为 0；
        3) in_use 里的块 ref_count 必须 > 0。
        """
        free_ids = set(self._free)
        inuse_ids = set(self._in_use)
        all_ids = set(self._blocks)

        if free_ids & inuse_ids:
            raise BlockPoolError(
                f"conservation violated: blocks in both free and in_use: "
                f"{sorted(free_ids & inuse_ids)}"
            )
        if free_ids | inuse_ids != all_ids:
            missing = sorted(all_ids - (free_ids | inuse_ids))
            raise BlockPoolError(
                f"conservation violated: blocks unaccounted for: {missing}"
            )
        for bid, blk in self._blocks.items():
            if bid in free_ids and blk.ref_count != 0:
                raise BlockPoolError(
                    f"conservation violated: free block {bid} has "
                    f"ref_count={blk.ref_count}"
                )
            if bid in inuse_ids and blk.ref_count <= 0:
                raise BlockPoolError(
                    f"conservation violated: in_use block {bid} has "
                    f"ref_count={blk.ref_count}"
                )

    @property
    def total_expected(self) -> int:
        return self.num_blocks

    # ---- 分配 ----

    def allocate(self, num_blocks: int) -> list[int]:
        """取 num_blocks 个空闲块。空闲不够就抛错，不返回半截结果。"""
        if num_blocks == 0:
            return []
        if num_blocks < 0:
            raise ValueError("num_blocks must be >= 0")
        if len(self._free) < num_blocks:
            raise BlockPoolError(
                f"out of blocks: need {num_blocks}, have {len(self._free)} free "
                f"of {self.num_blocks} total"
            )
        out: list[int] = []
        for _ in range(num_blocks):
            block_id, _ = self._free.popitem(last=False)  # 最久未用的先给
            blk = self._blocks[block_id]
            # 块被重新分配，旧 hash 必须失效（否则会读到别人的 KV）。
            # 但**只有当映射仍指向自己**才删：同一 hash 可能已被别的块覆盖，
            # 那种情况下删掉会误伤别人（vLLM block_pool.py:47-48 的 == block_id 判据）。
            if blk.block_hash is not None:
                if self._hash_to_block.get(blk.block_hash) == block_id:
                    del self._hash_to_block[blk.block_hash]
                blk.block_hash = None
            # 新内容从空白开始，清掉旧 payload（否则会读到上一任的 token）
            blk.payload = None
            blk.ref_count = 1
            self._in_use[block_id] = None
            out.append(block_id)
        self._check_conservation()
        return out

    def acquire_prefix(self, block_hash: str) -> int:
        """前缀命中：拿到那个块并把引用计数 +1。块 id 保持不变。"""
        block_id = self._hash_to_block.get(block_hash)
        if block_id is None:
            raise BlockPoolError(f"no cached block for hash {block_hash!r}")
        blk = self._blocks[block_id]
        if blk.ref_count > 0:
            blk.ref_count += 1
            self._touch_id(block_id)
        else:
            # 从空闲队列摘下
            self._free.pop(block_id, None)
            blk.ref_count = 1
            self._in_use[block_id] = None
        self._check_conservation()
        return block_id

    def retain(self, block_ids: list[int]) -> None:
        """给已有块再加一个引用（另一个请求共享同一前缀）。

        先整批校验再改状态：任一 id 非法则**一个都不改**，避免半截状态。
        """
        # 阶段一：校验
        for bid in block_ids:
            blk = self._blocks.get(bid)
            if blk is None:
                raise BlockPoolError(f"unknown block {bid}")
            if blk.ref_count == 0:
                raise BlockPoolError(f"cannot retain free block {bid}")
        # 阶段二：变更
        for bid in block_ids:
            blk = self._blocks[bid]
            blk.ref_count += 1
            self._touch_id(bid)
        self._check_conservation()

    # ---- 释放 ----

    def free(self, block_ids: list[int]) -> None:
        """引用计数减一。归零的块回到空闲队列。

        **两种块走不同的队列端**，这是 vLLM block_pool.py:834-841 的原语义：

          - 没有 hash 的块 -> 插到队首（LIFO 复用，为了 GPU 局部性：
            刚释放的块很可能马上被同一个请求的下一个 token 用到）
          - 有 hash 的块   -> 插到队尾（FIFO，给 LRU 淘汰留出被 prefix 命中的机会）

        vLLM 原文注释：
            "LIFO reuse of non-cached blocks for better GPU locality."
            "FIFO reuse of cached blocks for LRU eviction behavior."

        先整批校验（含**批内重复计数**：同一块在批里出现 k 次，就需要 ref>=k），
        再统一扣减。否则中途报错会留下「前几个已扣、后几个没动」的半截状态。
        """
        if not block_ids:
            return
        # 阶段一：校验（unknown + 引用够不够）
        counts: dict[int, int] = {}
        for bid in block_ids:
            if bid not in self._blocks:
                raise BlockPoolError(f"unknown block {bid}")
            counts[bid] = counts.get(bid, 0) + 1
        for bid, need in counts.items():
            blk = self._blocks[bid]
            if blk.ref_count < need:
                raise BlockPoolError(
                    f"double free of block {bid}: ref_count={blk.ref_count}, "
                    f"freed {need} times in this batch"
                )

        # 阶段二：变更
        to_evict_first: list[int] = []
        to_evict_last: list[int] = []
        for bid in block_ids:
            blk = self._blocks[bid]
            blk.ref_count -= 1
            if blk.ref_count == 0:
                self._in_use.pop(bid, None)
                if blk.block_hash is None:
                    to_evict_first.append(bid)
                else:
                    to_evict_last.append(bid)
        # 队首插入：按释放顺序依次 prepend，所以最后释放的在最前面（LIFO）
        for bid in to_evict_first:
            self._free[bid] = None
            self._free.move_to_end(bid, last=False)
        for bid in to_evict_last:
            self._free[bid] = None  # 已在末尾则原地不动
        self._check_conservation()

    # ---- LRU ----

    def touch(self, block_ids: list[int]) -> None:
        """标记为最近使用：挪到 LRU 队尾，不改变引用计数。"""
        for bid in block_ids:
            if bid not in self._blocks:
                raise BlockPoolError(f"unknown block {bid}")
            self._touch_id(bid)
        self._check_conservation()

    def _touch_id(self, block_id: int) -> None:
        blk = self._blocks[block_id]
        if blk.ref_count == 0:
            # 空闲块的 touch：挪到空闲队列右端（最不容易被淘汰）
            if block_id in self._free:
                self._free.move_to_end(block_id, last=True)
        else:
            if block_id in self._in_use:
                self._in_use.move_to_end(block_id, last=True)

    # ---- payload 读 / 写（CPU 教学） ----

    def is_block_writable(self, block_id: int) -> bool:
        """块是否可写（对应 vLLM ``is_block_writable`` 的核心判据）。

        可写 = 独占（ref_count == 1）且未被登记为前缀（block_hash is None）。
        - 共享块（ref_count > 1）要写必须先 COW；
        - 缓存块（空闲、带 hash）内容已作为 prefix 缓存，不能原地改；
        - 已登记前缀的块（哪怕独占）内容被视作不可变，改它会让 hash 与内容不符。
        """
        blk = self._blocks.get(block_id)
        if blk is None:
            raise BlockPoolError(f"unknown block {block_id}")
        return blk.ref_count == 1 and blk.block_hash is None

    def write_payload(self, block_id: int, tokens: list[int]) -> None:
        """写入块内 token 内容。仅允许写「独占且未登记前缀」的块。"""
        blk = self._blocks.get(block_id)
        if blk is None:
            raise BlockPoolError(f"unknown block {block_id}")
        if len(tokens) > self.block_size:
            raise ValueError(
                f"payload of {len(tokens)} tokens exceeds block_size {self.block_size}"
            )
        if not self.is_block_writable(block_id):
            raise BlockPoolError(
                f"block {block_id} is not writable "
                f"(ref_count={blk.ref_count}, hash={blk.block_hash!r}); "
                f"copy-on-write shared blocks first"
            )
        blk.payload = list(tokens)

    def read_payload(self, block_id: int) -> tuple[int, ...]:
        """读出块内 token 内容。未写入过（payload is None）则报错。"""
        blk = self._blocks.get(block_id)
        if blk is None:
            raise BlockPoolError(f"unknown block {block_id}")
        if blk.payload is None:
            raise BlockPoolError(f"block {block_id} has no payload")
        return tuple(blk.payload)

    # ---- prefix cache ----

    def register_prefix(
        self, block_hash: str, block_id: int, num_tokens: int | None = None
    ) -> None:
        """登记 hash→block。只允许登记**写满一整块**的块（full-prefix only）。

        原因：末块可能还没写满，把它当完整前缀缓存会让后续请求命中「半块」，
        读到不完整的前缀。fullness 判定优先级：
        1) 显式 ``num_tokens``；
        2) 已写入 payload 时用 ``len(payload)``；
        3) 二者都没有时（老 API / 新建空池）假定整块已满，保持向后兼容。
        """
        blk = self._blocks.get(block_id)
        if blk is None:
            raise BlockPoolError(f"unknown block {block_id}")

        if num_tokens is None:
            num_tokens = len(blk.payload) if blk.payload is not None else self.block_size
        if num_tokens != self.block_size:
            raise ValueError(
                f"can only register a full block: got {num_tokens} of "
                f"{self.block_size} tokens (full-prefix-only)"
            )

        old = blk.block_hash
        if old is not None and old != block_hash:
            # 只删「仍指向自己」的旧映射，别误删已覆盖到别的块的映射
            if self._hash_to_block.get(old) == block_id:
                del self._hash_to_block[old]
        blk.block_hash = block_hash
        self._hash_to_block[block_hash] = block_id

    def lookup_prefix(self, block_hash: str) -> int | None:
        """查前缀。命中返回 block id，没命中返回 None。

        注意：命中**不代表块当前空闲**，调用方要自己决定是 acquire 还是等。
        """
        return self._hash_to_block.get(block_hash)

    def is_cached(self, block_hash: str) -> bool:
        bid = self._hash_to_block.get(block_hash)
        if bid is None:
            return False
        return self._blocks[bid].ref_count == 0

    # ---- copy-on-write ----

    def copy_on_write(self, block_id: int) -> int:
        """共享块要写时复制一份新块。单引用时是 no-op（不需要复制）。

        语义（vLLM v1 block_pool.py:812 ``is_block_writable`` 的思路）：

        - 空闲块（ref_count == 0）：没有「写者」可复制，直接报错；
        - 独占块（ref_count == 1）：无需复制，返回原块 id；
        - 共享块（ref_count > 1）：分配新块，把 token payload **真正复制**过去，
          新块 ref_count=1 且**不登记 hash**（它准备被写入，内容会与旧前缀分叉）；
          原块 ref_count 减一、**保留自己的前缀 hash** 与内容不变。
        """
        blk = self._blocks.get(block_id)
        if blk is None:
            raise BlockPoolError(f"unknown block {block_id}")
        if blk.ref_count == 0:
            raise BlockPoolError(
                f"cannot copy_on_write free block {block_id} (ref_count=0)"
            )
        if blk.ref_count == 1:
            return block_id  # 独占，直接写

        # 共享：先分配新块（失败会先抛错，此时尚未改动原块）
        new_id = self.allocate(1)[0]
        new_blk = self._blocks[new_id]
        # 复制真实 token payload（list 拷贝，两个块互不影响）
        new_blk.payload = list(blk.payload) if blk.payload is not None else None
        new_blk.ref_count = 1
        # 新块**不继承 hash**：它是可写的分叉副本，不是同一前缀
        new_blk.block_hash = None
        # 原块引用转出一个给新块；原块保留 hash 与 payload
        blk.ref_count -= 1
        self._check_conservation()
        return new_id
