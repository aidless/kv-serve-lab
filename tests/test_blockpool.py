"""blockpool 测试：PagedAttention 风格的 KV 块管理器。

语义对齐 vLLM 的 vllm/v1/core/block_pool.py：
  :706 get_new_blocks(num_blocks)  从空闲队列取块
  :792 touch(blocks)              标记为最近使用（LRU 尾部）
  :816 free_blocks(ordered)       引用计数减一，归零则回到空闲队列头部
  :202 get_cached_block(hash)     前缀命中
  :230 cache_full_blocks          登记 hash→block

核心不变量（本测试的重点，不是功能堆砌）：
  在用块数 + 空闲块数 == 总块数   —— 任何时刻都成立，泄漏立刻暴露
"""

import pytest

from kv_serve_lab.blockpool import BlockPool, BlockState, BlockPoolError


class TestConstruction:
    def test_all_blocks_start_free(self):
        pool = BlockPool(num_blocks=8, block_size=16)
        assert pool.audit().used == 0
        assert pool.audit().free == 8
        assert pool.audit().cached == 0

    def test_rejects_zero_blocks(self):
        with pytest.raises(ValueError):
            BlockPool(num_blocks=0, block_size=16)

    def test_rejects_zero_block_size(self):
        with pytest.raises(ValueError):
            BlockPool(num_blocks=4, block_size=0)


class TestAllocate:
    def test_allocate_returns_distinct_ids(self):
        pool = BlockPool(num_blocks=8, block_size=16)
        ids = pool.allocate(4)
        assert len(ids) == 4
        assert len(set(ids)) == 4

    def test_allocate_increments_used(self):
        pool = BlockPool(num_blocks=8, block_size=16)
        pool.allocate(3)
        a = pool.audit()
        assert a.used == 3
        assert a.free == 5

    def test_allocate_more_than_available_raises(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        pool.allocate(4)
        with pytest.raises(BlockPoolError, match="out of blocks"):
            pool.allocate(1)

    def test_allocate_zero_returns_empty(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        assert pool.allocate(0) == []


class TestFree:
    def test_free_returns_blocks_to_pool(self):
        pool = BlockPool(num_blocks=8, block_size=16)
        ids = pool.allocate(3)
        pool.free(ids)
        a = pool.audit()
        assert a.used == 0
        assert a.free == 8

    def test_double_free_raises(self):
        """引用计数不能变负 —— 这是最容易静默出错的地方。"""
        pool = BlockPool(num_blocks=4, block_size=16)
        ids = pool.allocate(1)
        pool.free(ids)
        with pytest.raises(BlockPoolError, match="double free"):
            pool.free(ids)

    def test_free_unknown_block_raises(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        with pytest.raises(BlockPoolError, match="unknown block"):
            pool.free([99])

    def test_partial_free_keeps_block_in_use(self):
        """引用计数 2 的块，free 一次后仍在用（共享前缀场景）。"""
        pool = BlockPool(num_blocks=8, block_size=16)
        ids = pool.allocate(2)
        pool.retain(ids)  # 第二个请求共享同一批块
        assert pool.audit().used == 2
        pool.free(ids)
        assert pool.audit().used == 2  # 还有 1 个引用
        pool.free(ids)
        assert pool.audit().used == 0


class TestConservationInvariant:
    """在用 + 空闲 == 总数。任何操作序列之后都必须成立。"""

    def test_after_random_ops(self):
        import random

        rng = random.Random(20261031)
        pool = BlockPool(num_blocks=16, block_size=16)
        held: list[int] = []
        for _ in range(200):
            op = rng.choice(["alloc", "free", "touch", "audit"])
            if op == "alloc" and len(held) < 10:
                want = rng.randint(1, 3)
                if pool.audit().free >= want:
                    held.extend(pool.allocate(want))
            elif op == "free" and held:
                n = rng.randint(1, len(held))
                victims = held[:n]
                held = held[n:]
                pool.free(victims)
            elif op == "touch" and held:
                pool.touch(held[: rng.randint(1, len(held))])
            a = pool.audit()
            assert a.used + a.free == 16, f"leak after {op}: {a}"
            assert a.used == len(held), f"used mismatch after {op}: {a} vs {len(held)}"


class TestTouchAndLru:
    def test_touch_moves_to_most_recent(self):
        """touch 之后该块不该是下一个被淘汰的。"""
        pool = BlockPool(num_blocks=4, block_size=16)
        a, b, c = pool.allocate(3)
        pool.free([a, b, c])
        # 此时淘汰顺序是 c, b, a（free 把头插到队首，最后 free 的最先淘汰）
        pool.touch([a])
        nxt = pool.allocate(1)[0]
        assert nxt == c, f"expected LRU block {c}, got {nxt}"

    def test_eviction_order_is_lru(self):
        pool = BlockPool(num_blocks=3, block_size=16)
        a, b, c = pool.allocate(3)
        pool.free([a])
        pool.free([b])
        pool.free([c])
        got = [pool.allocate(1)[0] for _ in range(3)]
        assert got == [c, b, a]


class TestPrefixCache:
    def test_lookup_missing_returns_none(self):
        pool = BlockPool(num_blocks=8, block_size=16)
        assert pool.lookup_prefix("deadbeef") is None

    def test_register_then_lookup(self):
        pool = BlockPool(num_blocks=8, block_size=16)
        bid = pool.allocate(1)[0]
        pool.register_prefix("hash-a", bid)
        assert pool.lookup_prefix("hash-a") == bid

    def test_cached_count_tracks_registered_free_blocks(self):
        pool = BlockPool(num_blocks=8, block_size=16)
        bid = pool.allocate(1)[0]
        pool.register_prefix("h1", bid)
        assert pool.audit().cached == 0  # 还在用，不算 cached
        pool.free([bid])
        assert pool.audit().cached == 1  # 空闲且带 hash = 可复用
        assert pool.lookup_prefix("h1") == bid

    def test_evicted_block_loses_its_hash(self):
        """块被重新分配时旧 hash 必须失效，否则会读到别人的 KV。"""
        pool = BlockPool(num_blocks=2, block_size=16)
        a = pool.allocate(1)[0]
        pool.register_prefix("h1", a)
        pool.free([a])
        other = pool.allocate(1)[0]
        if other == a:  # LRU 恰好复用了它
            assert pool.lookup_prefix("h1") is None

    def test_reacquire_cached_block_keeps_id(self):
        """prefix 命中后 block id 不变 —— 对应 vLLM「不去重 block」的设计约束。"""
        pool = BlockPool(num_blocks=4, block_size=16)
        bid = pool.allocate(1)[0]
        pool.register_prefix("h1", bid)
        pool.free([bid])
        again = pool.acquire_prefix("h1")
        assert again == bid
        assert pool.audit().used == 1


class TestCopyOnWrite:
    def test_cow_returns_new_block(self):
        pool = BlockPool(num_blocks=8, block_size=16)
        bid = pool.allocate(1)[0]
        pool.retain([bid])  # ref_cnt = 2
        new = pool.copy_on_write(bid)
        assert new != bid
        assert pool.audit().used == 2

    def test_cow_decrements_original_refcount(self):
        pool = BlockPool(num_blocks=8, block_size=16)
        bid = pool.allocate(1)[0]
        pool.retain([bid])
        pool.copy_on_write(bid)
        # 原块引用从 2 降到 1，新块是 1 -> 总 2
        assert pool.audit().used == 2
        pool.free([bid])
        assert pool.audit().used == 1

    def test_cow_on_single_ref_is_noop(self):
        """只有一个引用时不需要复制（vLLM 的 can_copy 判断）。"""
        pool = BlockPool(num_blocks=8, block_size=16)
        bid = pool.allocate(1)[0]
        new = pool.copy_on_write(bid)
        assert new == bid
        assert pool.audit().used == 1

    def test_cow_keeps_hash_on_original_and_unhashes_new(self):
        """前缀 hash 属于「内容未变」的旧块；新块是准备写入的，不能被登记为同一前缀。"""
        pool = BlockPool(num_blocks=8, block_size=16)
        bid = pool.allocate(1)[0]
        pool.write_payload(bid, list(range(16)))
        pool.register_prefix("h1", bid)
        pool.retain([bid])
        new = pool.copy_on_write(bid)
        assert pool.lookup_prefix("h1") == bid  # 旧块保留 hash
        assert pool.hash_of(new) is None        # 新块未登记（可写）


class TestBlockState:
    def test_state_query(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        bid = pool.allocate(1)[0]
        assert pool.state_of(bid) == BlockState.IN_USE
        pool.free([bid])
        assert pool.state_of(bid) == BlockState.FREE


class TestPayload:
    """CPU 教学用的读/写 payload 接口：块内内容可读、可写，但共享/缓存块禁止写。"""

    def test_write_then_read_roundtrip(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        bid = pool.allocate(1)[0]
        tokens = list(range(16))
        pool.write_payload(bid, tokens)
        assert pool.read_payload(bid) == tuple(tokens)

    def test_write_rejects_oversized_payload(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        bid = pool.allocate(1)[0]
        with pytest.raises(ValueError):
            pool.write_payload(bid, list(range(17)))  # 超过 block_size

    def test_write_rejects_shared_block(self):
        pool = BlockPool(num_blocks=8, block_size=16)
        bid = pool.allocate(1)[0]
        pool.retain([bid])  # ref_count = 2
        with pytest.raises(BlockPoolError, match="writable"):
            pool.write_payload(bid, list(range(16)))

    def test_write_rejects_free_block(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        bid = pool.allocate(1)[0]
        pool.free([bid])
        with pytest.raises(BlockPoolError, match="writable"):
            pool.write_payload(bid, list(range(16)))

    def test_write_rejects_registered_block(self):
        """已登记前缀的块内容被视作不可变，写它会让 hash 与内容不符。"""
        pool = BlockPool(num_blocks=4, block_size=16)
        bid = pool.allocate(1)[0]
        pool.write_payload(bid, list(range(16)))
        pool.register_prefix("h", bid, num_tokens=16)
        with pytest.raises(BlockPoolError, match="writable"):
            pool.write_payload(bid, list(range(16)))

    def test_read_unwritten_block_raises(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        bid = pool.allocate(1)[0]
        with pytest.raises(BlockPoolError):
            pool.read_payload(bid)

    def test_is_block_writable(self):
        pool = BlockPool(num_blocks=8, block_size=16)
        bid = pool.allocate(1)[0]
        assert pool.is_block_writable(bid) is True   # 独占
        pool.retain([bid])
        assert pool.is_block_writable(bid) is False  # 共享
        pool.free([bid])
        pool.free([bid])
        assert pool.is_block_writable(bid) is False  # 空闲


class TestCachedMappingEviction:
    """I4：块被重新分配时，只有当 hash 仍指向「自己」才删映射。"""

    def test_eviction_keeps_mapping_owned_by_other_block(self):
        pool = BlockPool(num_blocks=2, block_size=16)
        a = pool.allocate(1)[0]
        b = pool.allocate(1)[0]
        pool.register_prefix("h", a)
        pool.register_prefix("h", b)  # 映射被覆盖到 b，a 上残留 h
        pool.free([a])
        reused = pool.allocate(1)[0]
        assert reused == a                      # 复用的正是 a
        assert pool.lookup_prefix("h") == b     # 不能误删 b 的映射


class TestTrueConservation:
    """守恒必须比对真实的 free/in_use 分区与引用计数，而不是恒等式 used+free==total。"""

    def test_detects_lost_free_block(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        pool.allocate(1)
        victim = next(iter(pool._free))
        pool._free.pop(victim)  # 注入：块从 free 消失但 ref 仍为 0
        with pytest.raises(BlockPoolError, match="conservation"):
            pool._check_conservation()

    def test_detects_refcount_partition_mismatch(self):
        pool = BlockPool(num_blocks=2, block_size=16)
        bid = pool.allocate(1)[0]
        pool._blocks[bid].ref_count = 0  # 注入：在用块 ref 变 0
        with pytest.raises(BlockPoolError, match="conservation"):
            pool._check_conservation()


class TestBatchValidationBeforeMutation:
    """批量 free/retain 必须先整批校验，再改状态；含批内重复计数。"""

    def test_free_with_duplicate_errors_without_mutation(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        bid = pool.allocate(1)[0]
        with pytest.raises(BlockPoolError, match="double free"):
            pool.free([bid, bid])  # ref=1 但被 free 两次
        assert pool.ref_count_of(bid) == 1
        assert pool.audit().used == 1

    def test_free_unknown_second_leaves_first_untouched(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        ids = pool.allocate(2)
        with pytest.raises(BlockPoolError, match="unknown block"):
            pool.free([ids[0], 99])
        assert pool.ref_count_of(ids[0]) == 1  # 第一个没被提前扣减

    def test_retain_unknown_leaves_first_untouched(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        ids = pool.allocate(2)
        with pytest.raises(BlockPoolError, match="unknown block"):
            pool.retain([ids[0], 99])
        assert pool.ref_count_of(ids[0]) == 1


class TestCopyOnWriteErrors:
    def test_cow_on_free_block_raises(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        bid = pool.allocate(1)[0]
        pool.free([bid])
        with pytest.raises(BlockPoolError, match="free"):
            pool.copy_on_write(bid)

    def test_cow_preserves_payload_independently(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        bid = pool.allocate(1)[0]
        tokens = list(range(16))
        pool.write_payload(bid, tokens)
        pool.retain([bid])
        new = pool.copy_on_write(bid)
        assert pool.read_payload(new) == tuple(tokens)
        pool.write_payload(new, [999] * 16)  # 写新块
        assert pool.read_payload(bid) == tuple(tokens)  # 原块 payload 不受影响


class TestFullPrefixRegistration:
    """只有「写满一整块」的块才能登记前缀（末块可能没写满，不能参与复用）。"""

    def test_full_payload_registers(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        bid = pool.allocate(1)[0]
        pool.write_payload(bid, list(range(16)))
        pool.register_prefix("h", bid)
        assert pool.lookup_prefix("h") == bid

    def test_partial_payload_rejected(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        bid = pool.allocate(1)[0]
        pool.write_payload(bid, list(range(8)))
        with pytest.raises(ValueError, match="full"):
            pool.register_prefix("h", bid)

    def test_explicit_full_token_count_ok(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        bid = pool.allocate(1)[0]
        pool.register_prefix("h", bid, num_tokens=16)
        assert pool.lookup_prefix("h") == bid

    def test_explicit_partial_token_count_rejected(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        bid = pool.allocate(1)[0]
        with pytest.raises(ValueError, match="full"):
            pool.register_prefix("h", bid, num_tokens=8)


class TestDocstringClaims:
    def test_no_false_no_dedup_claim(self):
        """旧 docstring 声称「vLLM 一个 hash 一个 block = 不做去重」，这是错的。

        vLLM 的 hash→block 映射允许共享 canonical block，本身就是去重；
        本库的单值映射只是**刻意的简化**，必须在 docstring 里说清，不能反过来说 vLLM 不去重。
        """
        import kv_serve_lab.blockpool as bp

        doc = bp.__doc__ or ""
        assert "不做 block 去重" not in doc
        assert "简化" in doc
