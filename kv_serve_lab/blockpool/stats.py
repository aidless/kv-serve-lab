"""前缀缓存命中率统计与可插拔逐出（eviction）策略。

本模块把 blockpool 的前缀缓存机制（register / acquire / COW）升级为**可测量、
可对比**的教学台面：
- ``PrefixCacheStats``：在块池上包一层计数器，记录 lookup / hit / miss / evict，
  并导出命中率与缓存利用指标。命中率的分子分母口径写死在 docstring，不靠
  调用方「猜」。
- ``EvictionPolicy``：抽象基类，``select_victim(pool, candidates) -> int`` 决定
  腾哪一块。LRU（最近最久未用）是 baseline；FIFO（先入先出）是文献里
  UniCache（ACM SIGMETRICS 2026）指出的——在**异构负载**下 LRU 失效、FIFO 反而
  更稳，所以保留两把尺供对照。

与 vLLM 的关系：vLLM v1 的 prefix cache 逐出靠空闲队列本身的次序（带 hash 的块
FIFO 队尾、无 hash 块 LIFO 队首），没有显式 eviction policy 对象。这里把它
**显式化**，是为了把「哪种逐出策略在什么负载下更好」变成可跑的实验，而不是
读代码猜。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import BlockPool, BlockPoolError


@dataclass(frozen=True)
class PrefixCacheStats:
    """一次会话的前缀缓存命中统计（immutable 快照）。"""

    lookups: int = 0          # lookup_prefix 调用总次数
    hits: int = 0             # 命中（返回非 None）次数
    evictions: int = 0        # 显式逐出总次数
    registered: int = 0       # 当前已登记前缀数（= len(hash_to_block)）
    cached_blocks: int = 0    # 当前空闲且带 hash 的块数

    @property
    def hit_rate(self) -> float:
        """命中率 = hits / lookups。lookups == 0 时返回 0.0（无统计意义）。"""
        return self.hits / self.lookups if self.lookups else 0.0

    @property
    def misses(self) -> int:
        return self.lookups - self.hits


class PrefixCacheStatsMixin:
    """给 BlockPool 挂上统计计数器的 mixin。

    用法：``pool = CountingBlockPool(num_blocks, block_size)``。计数器只增不减
    （累计值），导出时用 ``snapshot()`` 拿不可变快照。

    注意：本 mixin **不改写** BlockPool 的任何语义，只包一层计数。lookup 命中
    与否以 ``lookup_prefix`` 的返回值是否为 None 为准。
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self._pc_lookups = 0
        self._pc_hits = 0
        self._pc_evictions = 0
        BlockPool.__init__(self, num_blocks, block_size)

    def lookup_prefix(self, block_hash: str) -> int | None:  # type: ignore[override]
        self._pc_lookups += 1
        result = super().lookup_prefix(block_hash)  # type: ignore[misc]
        if result is not None:
            self._pc_hits += 1
        return result

    def evict(self, block_ids: list[int]) -> None:
        """显式逐出：把块从缓存里摘掉（不改变 ref_count / 状态）。

        逐出只清 hash 映射与块上的 hash 标记，块本身留在它当前的分区里。
        这是「缓存逐出」而不是「释放」——释放是 ref_count 归零回空闲队列。
        """
        self._pc_evictions += len(block_ids)
        for bid in block_ids:
            blk = self._blocks[bid]  # type: ignore[attr-defined]
            if blk.block_hash is not None:
                if self._hash_to_block.get(blk.block_hash) == bid:  # type: ignore[attr-defined]
                    del self._hash_to_block[blk.block_hash]  # type: ignore[attr-defined]
                blk.block_hash = None
            self._register_seq.pop(bid, None)  # type: ignore[attr-defined]

    def snapshot(self) -> PrefixCacheStats:
        return PrefixCacheStats(
            lookups=self._pc_lookups,
            hits=self._pc_hits,
            evictions=self._pc_evictions,
            registered=len(self._hash_to_block),  # type: ignore[attr-defined]
            cached_blocks=sum(
                1 for b in self._blocks.values()  # type: ignore[attr-defined]
                if b.ref_count == 0 and b.block_hash is not None
            ),
        )


class CountingBlockPool(PrefixCacheStatsMixin, BlockPool):
    """带前缀缓存统计的块池（统计不改变任何语义）。"""


# ---- 可插拔逐出策略 ----

class EvictionPolicy:
    """逐出策略基类：从候选块里选一个腾出去。"""

    name: str = "base"

    def select_victim(self, pool: BlockPool, candidates: list[int]) -> int:
        """返回要逐出的块 id（必须出现在 ``candidates`` 里）。

        基类不实现；``candidates`` 通常是被逐出候选（空闲且带 hash 的缓存块）。
        """
        raise NotImplementedError


class LRUEvictionPolicy(EvictionPolicy):
    """最近最久未用。空闲队列**队首**是最久未用的（BlockPool.free 把带 hash
    块 FIFO 插队尾，无 hash 块 LIFO 插队首；空闲队列左端 = 最久未动）。"""

    name = "lru"

    def select_victim(self, pool: BlockPool, candidates: list[int]) -> int:
        if not candidates:
            raise BlockPoolError("no candidates for eviction")
        # 空闲队列左端 = 最久未用。候选里选在空闲队列里最靠左的那个。
        order = list(pool._free)  # type: ignore[attr-defined]
        cand_set = set(candidates)
        for bid in order:
            if bid in cand_set:
                return bid
        return candidates[0]  # 空闲队列都不在候选里，退回第一个（防御）


class FIFOEvictionPolicy(EvictionPolicy):
    """先入先出。以**登记顺序**（register_prefix 的首次登记序号）为准，
    不受 touch 影响。这是 UniCache（ACM SIGMETRICS 2026）指出的——异构负载下
    LRU 会因「访问频次」而保留热点、饿死冷启动，FIFO 反而更稳。"""

    name = "fifo"

    def select_victim(self, pool: BlockPool, candidates: list[int]) -> int:
        if not candidates:
            raise BlockPoolError("no candidates for eviction")
        # 登记序号最小的候选 = 最先进入缓存 = 最先该走。
        return min(
            candidates,
            key=lambda bid: pool._register_seq.get(bid, float("inf")),  # type: ignore[attr-defined]
        )


def build_eviction_policy(name: str) -> EvictionPolicy:
    """按名字构造逐出策略。未知名字抛 ValueError。"""
    if name == LRUEvictionPolicy.name:
        return LRUEvictionPolicy()
    if name == FIFOEvictionPolicy.name:
        return FIFOEvictionPolicy()
    raise ValueError(f"unknown eviction policy: {name!r}")
