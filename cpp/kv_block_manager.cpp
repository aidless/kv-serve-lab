// PagedAttention 风格 KV 块管理器的 C++ 实现 —— 与 Python 版逐操作对拍。
//
// 为什么存在这份文件：`kv_serve_lab/blockpool` 的不变量（两个分区、弱引用
// hash 索引、LIFO/FIFO 两种空闲队列次序、COW 三个分支）光靠读代码是保不住的。
// 这里用 C++ 独立重写一遍同样的语义，`tests/test_cpp_differential.py` 把同一份
// 操作脚本喂给两边、要求逐行输出完全一致 —— 任何语义漂移都会在某一行的负载或
// 错误码上当场暴露。
//
// 协议见 `kv_serve_lab/blockpool/cli.py` 顶部注释：每行 `OK <负载>` 或
// `ERR <错误码>`，空集合写 `-`，错误码是有限词表。
//
// 刻意与 Python 版保持一致的地方（包括一些看着多余的重排）：
//   * 空闲队列里「插入已存在的键不改变位置」，对应 Python OrderedDict
//     `d[k] = v` 的语义；真正的「移到末尾」才先删后插。
//   * `allocate(0)` 直接返回、不做守恒检查（Python 版就是这样早退的）。
//   * `register_prefix` 不做守恒检查。
//   * 每个变更操作先整批校验再统一改，失败不留半截状态。
//
// 与 Python 版的唯一结构性差异：`in_use` 的 LRU 次序在协议里不可观测
// （输出只有计数与集合成员），所以这里用 unordered_set 而不是有序表。

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <list>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace kvbm {

// 协议错误码。与 Python 侧 error_code() 的输出词表一一对应。
enum class Code {
    UnknownBlock,
    OutOfBlocks,
    NoSuchHash,
    NotInUse,
    DoubleFree,
    NotWritable,
    NoPayload,
    CowFree,
    Conservation,
    BadArgs,
    BadOp,
};

inline const char* code_name(Code c) {
    switch (c) {
        case Code::UnknownBlock: return "unknown_block";
        case Code::OutOfBlocks:  return "out_of_blocks";
        case Code::NoSuchHash:   return "no_such_hash";
        case Code::NotInUse:     return "not_in_use";
        case Code::DoubleFree:   return "double_free";
        case Code::NotWritable:  return "not_writable";
        case Code::NoPayload:    return "no_payload";
        case Code::CowFree:      return "cow_free";
        case Code::Conservation: return "conservation";
        case Code::BadArgs:      return "bad_args";
        case Code::BadOp:        return "bad_op";
    }
    return "internal";
}

struct Failure {
    Code code;
};

[[noreturn]] inline void fail(Code c) { throw Failure{c}; }

struct Block {
    long long ref_count = 0;
    std::optional<std::string> block_hash;
    std::optional<std::vector<long long>> payload;  // nullopt = 未写入（不是空块）
};

class BlockPool {
  public:
    BlockPool(long long num_blocks, long long block_size)
        : num_blocks_(num_blocks), block_size_(block_size) {
        if (num_blocks < 1) fail(Code::BadArgs);
        if (block_size < 1) fail(Code::BadArgs);
        blocks_.resize(static_cast<std::size_t>(num_blocks));
        // 构造即建索引：free_pos_ 是 free_ 的反查表，free_erase/free_contains 都以
        // 它为唯一真相。只 push 不建索引会让 free_pos_ 落后于 free_，free_erase 对
        // 初始空闲块静默 no-op → acquire 命中缓存块时该块同时留在 free_ 与 in_use_
        // （audit 出 used=1 free=4 / total=4）→ check_conservation 报 ERR conservation。
        for (long long i = 0; i < num_blocks; ++i) {
            free_.push_back(i);
            free_pos_[i] = std::prev(free_.end());
        }
    }

    long long num_blocks() const { return num_blocks_; }
    long long block_size() const { return block_size_; }

    // ---- 查询 ----

    std::string state_of(long long id) const {
        const Block& b = at(id);
        return b.ref_count > 0 ? "in_use" : "free";
    }

    long long ref_count_of(long long id) const { return at(id).ref_count; }

    std::optional<std::string> hash_of(long long id) const {
        return at(id).block_hash;
    }

    // 从真实分区导出计数，不靠恒等式倒推。
    std::string audit() const {
        long long used = static_cast<long long>(in_use_.size());
        long long free_count = static_cast<long long>(free_.size());
        long long cached = 0;
        for (long long bid = 0; bid < num_blocks_; ++bid) {
            const Block& b = blocks_[static_cast<std::size_t>(bid)];
            if (b.ref_count == 0 && b.block_hash.has_value()) {
                auto it = hash_to_block_.find(*b.block_hash);
                if (it != hash_to_block_.end() && it->second == bid) ++cached;
            }
        }
        std::ostringstream os;
        os << "total=" << num_blocks_ << " used=" << used << " free=" << free_count
           << " cached=" << cached;
        return os.str();
    }

    // 真正的守恒检查：分区互斥 + 覆盖全集 + 引用计数与分区一致 + 索引单向一致
    // + 索引与队列同长（free_pos_ 是 free_ 的反查表，两者一旦不同步，
    //   free_erase 会静默 no-op，块会同时留在 free_ 和 in_use_ 而不报错）。
    void check_conservation() const {
        if (free_pos_.size() != free_.size()) fail(Code::Conservation);  // 反查表与队列必须同长
        std::unordered_set<long long> free_set(free_.begin(), free_.end());
        if (free_set.size() != free_.size()) fail(Code::Conservation);  // 同一块出现两次

        for (long long bid = 0; bid < num_blocks_; ++bid) {
            const Block& b = blocks_[static_cast<std::size_t>(bid)];
            bool in_free = free_set.count(bid) > 0;
            bool in_use = in_use_.count(bid) > 0;
            // 互斥且覆盖
            if (in_free == in_use) fail(Code::Conservation);
            if (in_free && b.ref_count != 0) fail(Code::Conservation);
            if (in_use && b.ref_count <= 0) fail(Code::Conservation);
        }
        // 索引是叠加弱引用层：索引指向的块必须真的持有那条 hash。
        // 反向不成立（同一 hash 被更晚的块覆盖后旧块仍留 hash 是合法状态）。
        for (const auto& [h, owner] : hash_to_block_) {
            const Block& b = blocks_[static_cast<std::size_t>(owner)];
            if (!b.block_hash.has_value() || *b.block_hash != h) fail(Code::Conservation);
        }
    }

    // ---- 分配 ----

    std::vector<long long> allocate(long long count) {
        std::vector<long long> out;
        if (count == 0) return out;              // Python 版在这里早退，不做守恒检查
        if (count < 0) fail(Code::BadArgs);
        if (static_cast<long long>(free_.size()) < count) fail(Code::OutOfBlocks);

        for (long long i = 0; i < count; ++i) {
            long long bid = free_.front();
            if (std::getenv("KVBM_DEBUG")) {
                std::cerr << "[alloc pop " << bid << "] free=";
                for (auto v : free_) std::cerr << v << ",";
                std::cerr << " mapkeys=";
                for (auto& kv : free_pos_) std::cerr << kv.first << ",";
                std::cerr << "\n";
            }
            free_.pop_front();
            // 必须同时清掉索引：否则 free_pos_ 里留下指向已释放节点的迭代器，
            // 后面任何 free_erase 都会 use-after-free（ASan 当场报 heap-use-after-free）。
            free_pos_.erase(bid);
            Block& b = blocks_[static_cast<std::size_t>(bid)];
            // 块被重新分配，旧 hash 必须失效 —— 但只在映射仍指向自己时才删，
            // 否则会误伤已被更晚块覆盖的映射。
            if (b.block_hash.has_value()) {
                auto it = hash_to_block_.find(*b.block_hash);
                if (it != hash_to_block_.end() && it->second == bid) {
                    hash_to_block_.erase(it);
                }
                b.block_hash.reset();
            }
            b.payload.reset();  // 清掉上一任的 token，否则会读到别人的内容
            b.ref_count = 1;
            in_use_.insert(bid);
            out.push_back(bid);
        }
        check_conservation();
        return out;
    }

    long long acquire_prefix(const std::string& h) {
        auto it = hash_to_block_.find(h);
        if (it == hash_to_block_.end()) fail(Code::NoSuchHash);
        long long bid = it->second;
        Block& b = blocks_[static_cast<std::size_t>(bid)];
        if (b.ref_count > 0) {
            b.ref_count += 1;
            touch_id(bid);
        } else {
            free_erase(bid);          // 从空闲队列摘下
            b.ref_count = 1;
            in_use_.insert(bid);
        }
        check_conservation();
        return bid;
    }

    void retain(const std::vector<long long>& ids) {
        // 阶段一：整批校验，任一非法则一个都不改
        for (long long bid : ids) {
            const Block& b = at(bid);
            if (b.ref_count == 0) fail(Code::NotInUse);
        }
        for (long long bid : ids) {
            blocks_[static_cast<std::size_t>(bid)].ref_count += 1;
            touch_id(bid);
        }
        check_conservation();
    }

    // ---- 释放 ----

    void free_blocks(const std::vector<long long>& ids) {
        if (ids.empty()) return;
        // 阶段一：校验 unknown 与批内重复释放
        std::unordered_map<long long, long long> counts;
        for (long long bid : ids) {
            at(bid);                       // unknown 检查
            counts[bid] += 1;
        }
        for (const auto& [bid, need] : counts) {
            if (blocks_[static_cast<std::size_t>(bid)].ref_count < need) {
                fail(Code::DoubleFree);
            }
        }
        // 阶段二：变更
        std::vector<long long> to_evict_first;
        std::vector<long long> to_evict_last;
        for (long long bid : ids) {
            Block& b = blocks_[static_cast<std::size_t>(bid)];
            b.ref_count -= 1;
            if (b.ref_count == 0) {
                in_use_.erase(bid);
                if (!b.block_hash.has_value()) {
                    to_evict_first.push_back(bid);   // LIFO 复用（GPU 局部性）
                } else {
                    to_evict_last.push_back(bid);    // FIFO（留给前缀命中）
                }
            }
        }
        // 依次 prepend：最后释放的排在最前（LIFO）
        for (long long bid : to_evict_first) free_push_front(bid);
        for (long long bid : to_evict_last) free_append_if_absent(bid);
        check_conservation();
    }

    // ---- LRU ----

    void touch(const std::vector<long long>& ids) {
        for (long long bid : ids) {
            at(bid);
            touch_id(bid);
        }
        check_conservation();
    }

    // ---- payload ----

    bool is_block_writable(long long bid) const {
        const Block& b = at(bid);
        return b.ref_count == 1 && !b.block_hash.has_value();
    }

    void write_payload(long long bid, const std::vector<long long>& tokens) {
        Block& b = at_mut(bid);
        if (static_cast<long long>(tokens.size()) > block_size_) fail(Code::BadArgs);
        if (!is_block_writable(bid)) fail(Code::NotWritable);
        b.payload = tokens;
    }

    std::vector<long long> read_payload(long long bid) const {
        const Block& b = at(bid);
        if (!b.payload.has_value()) fail(Code::NoPayload);
        return *b.payload;
    }

    // ---- prefix cache ----

    void register_prefix(const std::string& h, long long bid,
                         std::optional<long long> num_tokens) {
        Block& b = at_mut(bid);
        long long tokens;
        if (num_tokens.has_value()) {
            tokens = *num_tokens;
        } else if (b.payload.has_value()) {
            tokens = static_cast<long long>(b.payload->size());
        } else {
            tokens = block_size_;       // 老 API：假定整块已满
        }
        if (tokens != block_size_) fail(Code::BadArgs);

        if (b.block_hash.has_value() && *b.block_hash != h) {
            auto it = hash_to_block_.find(*b.block_hash);
            if (it != hash_to_block_.end() && it->second == bid) hash_to_block_.erase(it);
        }
        b.block_hash = h;
        hash_to_block_[h] = bid;
    }

    std::optional<long long> lookup_prefix(const std::string& h) const {
        auto it = hash_to_block_.find(h);
        if (it == hash_to_block_.end()) return std::nullopt;
        return it->second;
    }

    // ---- copy-on-write ----

    long long copy_on_write(long long bid) {
        const Block& b = at(bid);
        if (b.ref_count == 0) fail(Code::CowFree);
        if (b.ref_count == 1 && !b.block_hash.has_value()) return bid;

        // 先分配新块：失败会先抛错，此时尚未改动原块
        long long new_id = allocate(1)[0];
        Block& new_blk = blocks_[static_cast<std::size_t>(new_id)];
        Block& old_blk = blocks_[static_cast<std::size_t>(bid)];
        new_blk.payload = old_blk.payload;      // 真复制 token 内容
        new_blk.ref_count = 1;
        new_blk.block_hash.reset();             // 新块是可写分叉，不是同一前缀

        if (old_blk.ref_count > 1) {
            old_blk.ref_count -= 1;             // 共享：只少一个引用
        } else {
            old_blk.ref_count = 0;              // 独占带前缀：原地退回缓存
            in_use_.erase(bid);
            free_append_if_absent(bid);
            free_move_to_back(bid);             // 带 hash -> FIFO 队尾
        }
        check_conservation();
        return new_id;
    }

  private:
    const Block& at(long long bid) const {
        if (bid < 0 || bid >= num_blocks_) fail(Code::UnknownBlock);
        return blocks_[static_cast<std::size_t>(bid)];
    }

    Block& at_mut(long long bid) {
        if (bid < 0 || bid >= num_blocks_) fail(Code::UnknownBlock);
        return blocks_[static_cast<std::size_t>(bid)];
    }

    void touch_id(long long bid) {
        const Block& b = blocks_[static_cast<std::size_t>(bid)];
        if (b.ref_count == 0) {
            if (free_contains(bid)) free_move_to_back(bid);  // 最不容易被淘汰
        }
        // in_use 的次序在协议里不可观测，无需维护
    }

    // --- 空闲队列：list 表达 OrderedDict 的键序 ---
    bool free_contains(long long bid) const {
        return free_pos_.count(bid) > 0;
    }

    void free_erase(long long bid) {
        auto it = free_pos_.find(bid);
        if (it == free_pos_.end()) return;
        free_.erase(it->second);
        free_pos_.erase(it);
    }

    void free_push_front(long long bid) {
        free_erase(bid);
        free_.push_front(bid);
        free_pos_[bid] = free_.begin();
    }

    void free_append_if_absent(long long bid) {
        if (free_contains(bid)) return;   // 对应 OrderedDict 对已存在键赋值：位置不变
        free_.push_back(bid);
        free_pos_[bid] = std::prev(free_.end());
    }

    void free_move_to_back(long long bid) {
        free_erase(bid);
        free_.push_back(bid);
        free_pos_[bid] = std::prev(free_.end());
    }

    long long num_blocks_;
    long long block_size_;
    std::vector<Block> blocks_;
    std::list<long long> free_;                                  // 左=下一个淘汰
    std::unordered_map<long long, std::list<long long>::iterator> free_pos_;
    std::unordered_set<long long> in_use_;
    std::unordered_map<std::string, long long> hash_to_block_;
};

}  // namespace kvbm

// ---------------------------------------------------------------------------
// driver：从 stdin 读操作脚本，逐行打到 stdout。协议与 Python 侧完全一致。
// 用法：kvbm <num_blocks> <block_size>
// ---------------------------------------------------------------------------

namespace {

bool parse_int(const std::string& s, long long& out) {
    if (s.empty()) return false;
    std::size_t pos = 0;
    try {
        out = std::stoll(s, &pos, 10);
    } catch (...) {
        return false;
    }
    return pos == s.size();
}

std::string join_ids(const std::vector<long long>& ids) {
    if (ids.empty()) return "-";
    std::string out;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i) out += ",";
        out += std::to_string(ids[i]);
    }
    return out;
}

std::vector<long long> parse_ids(const std::vector<std::string>& tokens) {
    std::vector<long long> out;
    out.reserve(tokens.size());
    for (const auto& t : tokens) {
        long long v = 0;
        if (!parse_int(t, v)) kvbm::fail(kvbm::Code::BadArgs);
        out.push_back(v);
    }
    return out;
}

std::string apply_line(kvbm::BlockPool& pool, const std::vector<std::string>& parts) {
    const std::string& op = parts[0];
    const std::size_t n = parts.size() - 1;

    auto need = [&](std::size_t want) {
        if (n != want) kvbm::fail(kvbm::Code::BadArgs);
    };

    if (op == "alloc") {
        need(1);
        long long count = 0;
        if (!parse_int(parts[1], count)) kvbm::fail(kvbm::Code::BadArgs);
        return "OK " + join_ids(pool.allocate(count));
    }
    if (op == "free" || op == "retain" || op == "touch") {
        if (n == 0) kvbm::fail(kvbm::Code::BadArgs);
        std::vector<long long> ids = parse_ids({parts.begin() + 1, parts.end()});
        if (op == "free") pool.free_blocks(ids);
        else if (op == "retain") pool.retain(ids);
        else pool.touch(ids);
        return "OK";
    }
    if (op == "register") {
        if (n != 2 && n != 3) kvbm::fail(kvbm::Code::BadArgs);
        long long bid = 0;
        if (!parse_int(parts[2], bid)) kvbm::fail(kvbm::Code::BadArgs);
        std::optional<long long> num_tokens;
        if (n == 3) {
            long long t = 0;
            if (!parse_int(parts[3], t)) kvbm::fail(kvbm::Code::BadArgs);
            num_tokens = t;
        }
        pool.register_prefix(parts[1], bid, num_tokens);
        return "OK";
    }
    if (op == "lookup") {
        need(1);
        auto found = pool.lookup_prefix(parts[1]);
        return "OK " + (found.has_value() ? std::to_string(*found) : std::string("-"));
    }
    if (op == "acquire") {
        need(1);
        return "OK " + std::to_string(pool.acquire_prefix(parts[1]));
    }
    if (op == "cow") {
        need(1);
        long long bid = 0;
        if (!parse_int(parts[1], bid)) kvbm::fail(kvbm::Code::BadArgs);
        return "OK " + std::to_string(pool.copy_on_write(bid));
    }
    if (op == "write") {
        if (n < 2) kvbm::fail(kvbm::Code::BadArgs);
        long long bid = 0;
        if (!parse_int(parts[1], bid)) kvbm::fail(kvbm::Code::BadArgs);
        std::vector<long long> tokens = parse_ids({parts.begin() + 2, parts.end()});
        pool.write_payload(bid, tokens);
        return "OK";
    }
    if (op == "read") {
        need(1);
        long long bid = 0;
        if (!parse_int(parts[1], bid)) kvbm::fail(kvbm::Code::BadArgs);
        return "OK " + join_ids(pool.read_payload(bid));
    }
    if (op == "state") {
        need(1);
        long long bid = 0;
        if (!parse_int(parts[1], bid)) kvbm::fail(kvbm::Code::BadArgs);
        return "OK " + pool.state_of(bid);
    }
    if (op == "refcount") {
        need(1);
        long long bid = 0;
        if (!parse_int(parts[1], bid)) kvbm::fail(kvbm::Code::BadArgs);
        return "OK " + std::to_string(pool.ref_count_of(bid));
    }
    if (op == "hash") {
        need(1);
        long long bid = 0;
        if (!parse_int(parts[1], bid)) kvbm::fail(kvbm::Code::BadArgs);
        auto h = pool.hash_of(bid);
        return "OK " + (h.has_value() ? *h : std::string("-"));
    }
    if (op == "writable") {
        need(1);
        long long bid = 0;
        if (!parse_int(parts[1], bid)) kvbm::fail(kvbm::Code::BadArgs);
        return pool.is_block_writable(bid) ? "OK true" : "OK false";
    }
    if (op == "audit") {
        need(0);
        return "OK " + pool.audit();
    }
    if (op == "conserve") {
        need(0);
        pool.check_conservation();
        return "OK";
    }
    kvbm::fail(kvbm::Code::BadOp);
}

}  // namespace

int main(int argc, char** argv) {
    long long num_blocks = 8;
    long long block_size = 16;
    if (argc > 1 && !parse_int(argv[1], num_blocks)) return 2;
    if (argc > 2 && !parse_int(argv[2], block_size)) return 2;

    kvbm::BlockPool pool(num_blocks, block_size);

    std::string raw;
    while (std::getline(std::cin, raw)) {
        // 与 Python 一致：strip 后跳过空行与 `#` 注释
        std::size_t b = raw.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) continue;
        std::size_t e = raw.find_last_not_of(" \t\r\n");
        std::string line = raw.substr(b, e - b + 1);
        if (line[0] == '#') continue;

        std::vector<std::string> parts;
        std::istringstream is(line);
        std::string tok;
        while (is >> tok) parts.push_back(tok);

        try {
            std::cout << apply_line(pool, parts) << "\n";
        } catch (const kvbm::Failure& f) {
            std::cout << "ERR " << kvbm::code_name(f.code) << "\n";
        } catch (const std::exception&) {
            std::cout << "ERR bad_args\n";
        }
    }
    return 0;
}
