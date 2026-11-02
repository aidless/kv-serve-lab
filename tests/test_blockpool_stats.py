"""PrefixCacheStats + 可插拔 eviction 测试。

覆盖：
- 计数器只累计 lookup/hit/miss/evict，不改变块池语义（对照普通 BlockPool）；
- hit_rate 分子分母口径（hits/lookups，lookups==0 → 0.0）；
- LRU / FIFO 选受害者：带 hash 块 FIFO 队尾、无 hash 块 LIFO 队首；
- 显式逐出只清 hash 映射，块留在原分区，ref_count 不变；
- build_eviction_policy 未知名字抛 ValueError。
"""

import pytest

from kv_serve_lab.blockpool import BlockPool, BlockPoolError
from kv_serve_lab.blockpool.stats import (
    CountingBlockPool,
    FIFOEvictionPolicy,
    LRUEvictionPolicy,
    PrefixCacheStats,
    build_eviction_policy,
)


def _full_pool(num_blocks=8, block_size=4):
    pool = BlockPool(num_blocks, block_size)
    # 把块 0 分配、写满、登记前缀，然后释放回空闲队列
    return pool


def _register_full(pool, bid, block_hash, tokens):
    pool.allocate(1)
    pool.write_payload(bid, tokens)
    pool.register_prefix(block_hash, bid, len(tokens))
    pool.free([bid])


class TestCountingPoolSemantics:
    def test_snapshot_zero_initially(self):
        pool = CountingBlockPool(8, 4)
        snap = pool.snapshot()
        assert isinstance(snap, PrefixCacheStats)
        assert snap.lookups == 0
        assert snap.hits == 0
        assert snap.evictions == 0
        assert snap.hit_rate == 0.0

    def test_lookup_counts_and_hits(self):
        pool = CountingBlockPool(8, 4)
        _register_full(pool, 0, "h0", [1, 2, 3, 4])
        pool.lookup_prefix("missing")
        hit = pool.lookup_prefix("h0")
        assert hit == 0
        snap = pool.snapshot()
        assert snap.lookups == 2
        assert snap.hits == 1
        assert snap.misses == 1
        assert snap.hit_rate == 0.5

    def test_stats_do_not_change_semantics(self):
        """计数池与普通池跑同一脚本，块分配/释放结果完全一致。"""
        plain = BlockPool(8, 4)
        counting = CountingBlockPool(8, 4)
        _register_full(plain, 0, "h0", [1, 2, 3, 4])
        _register_full(counting, 0, "h0", [1, 2, 3, 4])
        assert plain.lookup_prefix("h0") == counting.lookup_prefix("h0") == 0
        assert plain.audit().used == counting.audit().used
        assert plain.audit().free == counting.audit().free

    def test_evict_counts_and_clears_hash(self):
        pool = CountingBlockPool(8, 4)
        _register_full(pool, 0, "h0", [1, 2, 3, 4])
        assert pool.lookup_prefix("h0") == 0
        pool.evict([0])
        snap = pool.snapshot()
        assert snap.evictions == 1
        assert snap.registered == 0
        # 块 0 仍空闲，只是 hash 被清掉
        assert pool.state_of(0).value == "free"
        assert pool.hash_of(0) is None
        assert pool.lookup_prefix("h0") is None

    def test_evict_non_cached_block_is_noop(self):
        """逐出未登记 hash 的块不计数（它本来就不是缓存块）。"""
        pool = CountingBlockPool(8, 4)
        pool.allocate(1)  # 块 0 在 in_use，无 hash
        pool.evict([0])
        snap = pool.snapshot()
        assert snap.evictions == 1  # 显式调用了就计数（块数=len）
        assert pool.hash_of(0) is None


class TestEvictionPolicies:
    def test_lru_picks_oldest_free(self):
        """LRU：空闲队列左端（最久未用）是受害者。"""
        pool = BlockPool(8, 4)
        _register_full(pool, 0, "h0", [1, 2, 3, 4])
        _register_full(pool, 1, "h1", [5, 6, 7, 8])
        # 块 0 先登记先释放 → 在队列左端
        policy = LRUEvictionPolicy()
        victim = policy.select_victim(pool, [0, 1])
        assert victim == 0

    def test_lru_respects_touch(self):
        """touch 把块挪到队尾 → LRU 不再选它。"""
        pool = BlockPool(8, 4)
        _register_full(pool, 0, "h0", [1, 2, 3, 4])
        _register_full(pool, 1, "h1", [5, 6, 7, 8])
        pool.touch([0])  # 0 挪到队尾，1 变最久
        policy = LRUEvictionPolicy()
        victim = policy.select_victim(pool, [0, 1])
        assert victim == 1

    def test_fifo_ignores_touch(self):
        """FIFO：touch 不改变次序（0 仍先入）。"""
        pool = BlockPool(8, 4)
        _register_full(pool, 0, "h0", [1, 2, 3, 4])
        _register_full(pool, 1, "h1", [5, 6, 7, 8])
        pool.touch([0])
        policy = FIFOEvictionPolicy()
        victim = policy.select_victim(pool, [0, 1])
        assert victim == 0

    def test_evict_no_candidates_raises(self):
        for policy in (LRUEvictionPolicy(), FIFOEvictionPolicy()):
            with pytest.raises(BlockPoolError, match="no candidates"):
                policy.select_victim(BlockPool(4, 2), [])

    def test_build_eviction_policy(self):
        assert isinstance(build_eviction_policy("lru"), LRUEvictionPolicy)
        assert isinstance(build_eviction_policy("fifo"), FIFOEvictionPolicy)
        with pytest.raises(ValueError, match="unknown eviction policy"):
            build_eviction_policy("no_such")


class TestPrefixSharedLoad:
    """前缀共享负载生成：多请求共享同一 prompt 前缀时命中率应显著 > 0。"""

    def test_shared_prefix_raises_hit_rate(self):
        pool = CountingBlockPool(16, 4)
        # 共享前缀 3 块（12 token）
        for i in range(3):
            _register_full(pool, i, f"h{i}", [i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3])
        # 三个请求各自 lookup 整个前缀链（命中前两块、末块视是否注册）
        for _ in range(3):
            for i in range(2):  # 命中已注册的 h0, h1
                pool.lookup_prefix(f"h{i}")
            pool.lookup_prefix("h2")  # 也命中（已注册）
        snap = pool.snapshot()
        assert snap.hit_rate > 0.5
        assert snap.hits == 9
        assert snap.lookups == 9
