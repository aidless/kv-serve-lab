# 推理引擎岗 · GitHub 项目地图

> 用途：把那份 JD 的每一条要求，落到「读哪个仓库的哪个文件」。
> 所有 star / 最近推送时间 / 版本号均为 **2026-10-31 实时查询**（`gh api`，已认证），不是记忆。
> 本机无 NVIDIA GPU（`nvidia-smi`/`nvcc` 均不存在，`torch.cuda.is_available()==False`），
> 所以下表中标 ⚠️ 的项目**只能在云端验证**，本地只读源码。

---

## 一、JD 逐条 → 项目 → 具体入口

| JD 要求 | 主项目 | 具体读哪里（真实行号） |
|---|---|---|
| ① vLLM/SGLang/Transformers 版本升级、**插件适配** | `vllm-project/vllm` ⭐95969 · `sgl-project/sglang` ⭐38320 | 插件机制的活教材是 `vllm-project/vllm-ascend` ⭐3038（昇腾社区维护的 vLLM 插件，含独立 `CMakeLists.txt` + `Dockerfile` + `Dockerfile.310p`，正好演示「插件 + 多硬件镜像交付」） |
| ② 模型上线、部署调参、**压测验证** | `vllm` 的 `vllm/benchmarks/` | **注意**：`benchmarks/benchmark_serving.py` 已废弃（只剩一段 `DEPRECATED: ... moved to the vLLM CLI` 提示，实测 17 行），真正的入口是 `vllm bench serve`，实现在 `vllm/benchmarks/serve.py`（2460 行）。同目录还有 `sweep/`（参数扫描）。SGLang 侧看 `benchmark/` 目录 |
| ③ 分析 **TTFT / TPOT / 吞吐 / 显存** | `vllm` + `sgl-project/sglang` | TTFT/TPOT/ITL 的权威定义在 `vllm/benchmarks/serve.py` 的 `calculate_metrics()`（:601）与 `BenchmarkMetrics`（:351-362）。**TPOT 的真实公式是 `(latency - ttft) / (output_len - 1)`**（:656-658，即扣掉首 token）；显存口径看 `vllm/v1/kv_cache_interface.py` 的 `page_size_bytes`（见下节公式） |
| ④ **KV Cache / batching / 并发调度** | `vllm` v1 核心 | `vllm/v1/core/sched/scheduler.py:698` `schedule()`；`vllm/v1/core/kv_cache_manager.py:379` `allocate_slots()`；`vllm/v1/core/block_pool.py:706` `get_new_blocks()` / `:816` `free_blocks()` / `:792` `touch()`；SGLang 侧 `python/sglang/srt/mem_cache/radix_cache.py`（radix tree + 可插拔 evict policy，与 vLLM 的核心架构差异） |
| ⑤ 最佳实践 / 适配文档 | `vllm-project/production-stack` ⭐2704 | K8S-native 部署参考栈，直接对应「镜像交付 + 上线」 |
| 加分 1 · GEMM/Attention/MoE/量化算子 | `flashinfer-ai/flashinfer` ⭐6824 · `deepseek-ai/DeepGEMM` ⭐9135 · `Dao-AILab/flash-attention` ⭐25232 · `NVIDIA/cutlass` ⭐10681 | DeepGEMM 是 MoE + FP8 GEMM 的最小可读实现；FlashInfer 是「算子库怎么被推理框架调用」的样板 |
| 加分 2 · AllReduce/AllGather/All-to-All | `NVIDIA/nccl-tests` ⭐1682 · `deepseek-ai/DeepEP` ⭐10392 · `ai-dynamo/nixl` ⭐1345 | nccl-tests 是通信基线的标准测量工具；DeepEP 是 MoE 的 all-to-all 专用通信库 |
| 加分 3 · kernel fusion / 编译优化 | `triton-lang/triton` ⭐20497 · `NVIDIA/TransformerEngine` ⭐3597 | Triton 教程里的 fused softmax / layer-norm 是最短的入门路径 |
| 加分 4 · profiler 定位瓶颈 | Nsight Systems / ncu（⚠️ 需 GPU） | 配合 `vllm` 的 `benchmark_latency.py` 做端到端切分 |
| 加分 5 · 端到端性能验证 | `vllm-project/production-stack` + 上面的 `benchmarks/` |  |

---

## 二、按「学习优先级」排序（不是按 star）

### 第一梯队：先读这三个，直接对应 JD 第 ③④ 条

1. **`vllm/v1/core/sched/scheduler.py`** — vLLM v1 的调度器。
   `:698` 的 `schedule()` 开头有一段注释，是整个 continuous batching 的最好解释：
   > "There's no 'decoding phase' nor 'prefill phase' in the scheduler. Each request just has its
   > num_computed_tokens and num_tokens_with_spec... the scheduler tries to assign tokens to the
   > requests so that each request's num_computed_tokens can catch up its num_tokens_with_spec.
   > This is general enough to cover chunked prefills, prefix caching, speculative decoding."

   即：**没有 prefill/decode 两阶段，只有「每个请求还欠多少 token」+ 一个 token 预算**。
   预算变量：`self.max_num_scheduled_tokens`（:719）、`input_budget = max_num_batched_tokens`（:721）、
   `max_num_running_reqs = max_num_seqs`（:130）、`watermark`（:365）。

2. **`vllm/v1/core/block_pool.py`**（945 行）— PagedAttention 的块池本体。
   `get_new_blocks()` :706 / `touch()` :792 / `free_blocks()` :816 / `evict_blocks()` :851。
   `BlockHashToBlockMap`（:36）说明 prefix caching 的 hash→block 映射**不做 block 去重**，
   目的是保证 block id 稳定、block table 可以 append-only。

3. **`vllm/v1/core/kv_cache_manager.py`** — 请求级视图。
   `get_computed_blocks()` :266 / `allocate_slots()` :379 / `free()` :628 / `cache_blocks()` :790。

### 第二梯队：架构对照（面试常问「vLLM 和 SGLang 有什么区别」）

4. **`sglang/python/sglang/srt/mem_cache/radix_cache.py`** — SGLang 用 **radix tree** 管 prefix cache，
   且 eviction policy 可插拔（同目录 `evict_policy.py`、`allocator/`、`buffer_mode/`）。
   vLLM 是「block + hash 表 + LRU」，SGLang 是「radix tree + 策略对象」。这就是核心差异。
   ⚠️ 该文件在 SHA `8368f86f` 下实测 **837 行**（2026-10-31 认证 API 复核；此前记录的「243 行薄封装」已过时）。radix tree 主体就在本文件内，`_split_node`（:682）等不再外置。

5. **`sglang/python/sglang/srt/managers/scheduler.py`**（6625 行）—
   `get_next_batch_to_run()` :4036（把 prefill batch 合并进 running batch）、
   `get_new_batch_prefill()` :4196、`run_batch()` :4802。
   预算：`max_prefill_tokens` :1237、`max_running_requests` :1238、`chunked_prefill_size` :1294。

### 第三梯队：加分项，按需

6. `deepseek-ai/DeepGEMM`（MoE + FP8 GEMM）、`flashinfer-ai/flashinfer`（算子库接入框架的方式）。
7. `NVIDIA/nccl-tests`（通信基线测量）、`deepseek-ai/DeepEP`（MoE all-to-all）。
8. `skyzh/tiny-llm` ⭐5466 — 面向 systems engineer 从零搭 LLM 推理系统，**适合当阅读路线的脚手架**。

### 暂缓优先学习的旧工具（按本次推送记录判断）

- `ray-project/llmperf` ⭐1127 — 最后推送 **2024-12-09**。
- `casper-hansen/AutoAWQ` ⭐2343 — 最后推送 **2025-05-11**。

---

## 三、KV cache 显存公式（从 vLLM 源码里读出来的，不是背的）

vLLM 的块大小由 `vllm/v1/kv_cache_interface.py` 的 `page_size_bytes` 决定：

```python
# AttentionSpec.state_content_size_bytes（:585-590）
return (self.head_size + self.head_size_v) * get_dtype_size(self.dtype)

# AttentionSpec.unpadded_page_size_bytes（:593）
return self.num_heads * self.num_states * self.state_content_size_bytes
```

其中 `num_states = block_size // tokens_per_state`，full attention 的 `tokens_per_state == 1`，
所以 **一个 block 的字节数 = num_kv_heads × block_size × (head_size + head_size_v) × dtype_bytes**。

MLA（DeepSeek 系）的 `head_size_v = 0`，只存一个 latent 向量 —— 这就是 MLA 省显存的来源。

**用真实 config.json 算出来的每 token 字节数**（实测拉取，bf16 = 2 bytes）：

| 模型 | 层数 | kv_heads | head_dim | bytes/token/层 | 每 4096 token 的 KV（全层） |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 28 | 4 | 128 | 2048 | 0.219 GiB |
| Qwen2.5-72B-Instruct | 80 | 8 | 128 | 4096 | 1.250 GiB |
| Mistral-7B-v0.3 | 32 | 8 | 128 | 4096 | 0.500 GiB |
| DeepSeek-V2-Lite (MLA) | 27 | — | kv_lora_rank 512 + qk_rope 64 | 1152 | 0.119 GiB |

> 复算方式（MHA/GQA）：`2 × n_layers × n_kv_heads × head_dim × seq_len × dtype_bytes`，
> 其中 `head_dim = hidden_size / num_attention_heads`（Qwen2.5-7B: 3584/28 = 128）。
> MLA 不同：每层每 token 只存一个 latent 向量加一条 RoPE 分量，
> `(kv_lora_rank + qk_rope_head_dim) × n_layers × seq_len × dtype_bytes`，
> 即 `(512 + 64) × 2 = 1152` B —— 这就是 MLA 比同规模 GQA 省显存的机制来源。
> 表中每个数字都可用上表参数重算（已用脚本复算，任何一个对不上就是算错了）。
> （`meta-llama/Llama-3.1-8B` 在 HF 上是 gated，hf-mirror 也拒绝，所以本表没放 Llama。）

---

## 四、aidless 账号的缺口

现有 **76 个公开仓库**（2026-10-31 认证 API 分页枚举，与 `public_repos` 字段一致），
绝大多数是 LLM 评测 / agent / 论文方向（`longrun-agi`、`llm-lab`、`agent-platform`、
`auditable-llm-eval`、`mcp-guard`、`q-one-*` …）。

> ⚠️ **本节原判断「没有任何一个仓库涉及推理引擎 / KV cache / serving」已于 2026-10-31 被证伪。**
> 当天新建的五个仓库已经覆盖了这些方向，见下表。任何「账号没有推理方向仓库，我来建第一个」
> 的说法都不成立——写新仓库前先查这五个。

| 仓库 | 覆盖能力 | 实测状态 |
|---|---|---|
| `aidless/paged-kv` | PagedAttention 块管理（C++17，alloc/fork/COW/双释放） | `make test` → ALL PASS |
| `aidless/roofline-lint` | KV 容量 / roofline / TP 切分（带 `--selftest` + `--prove-red`） | 30 tests |
| `aidless/ttftkit` | 端点压测 + fail-closed 门禁（TTFT/TPOT 口径完整） | `pytest` 全绿 |
| `aidless/kv-block-mass` | block-granular attention mass（Triton，CUDA/ROCm 双向） | 双向实测 |
| `aidless/accel-preflight` | CUDA→ROCm/DCU 可移植性预检 | 22KB |

本次新仓库重点补 continuous batching 调度模拟，并把容量、块管理与 SSE 度量串起来。
已有仓库可作为对照基线；关键词未命中不足以证明某项能力不存在。

---

## 五、待云端验证清单（本机做不了）

| 事项 | 阻塞原因 | 恢复后怎么做 |
|---|---|---|
| 真实 TTFT/TPOT 测量 | 本机无 GPU | 起 vLLM 服务后跑 `vllm bench serve`，SGLang 使用其对应压测入口 |
| CUDA kernel / ncu profile | 无 `nvcc` | 需 CUDA toolkit + GPU |
| NCCL 通信带宽 | 需 ≥2 卡 | `nccl-tests` 的 `all_reduce_perf` |
| 量化算子对比 | 需 GPU 跑 AWQ/GPTQ | 用 `llm-awq` 或 vLLM 内置量化 |

云端通道现状（2026-10-31 实测）：AMD T4 实例 `Permission denied (publickey)`；
SeetaCloud / j1gpu 端口 65535 被关闭；`ecs-hz` 可登录但无 GPU；Kaggle 凭据 HTTP 401。
→ **需要先修复其中一条通道**，否则所有 GPU 侧验证都只能写「待验证」，不能编数字。

---

## 六、本轮实时核验（2026-10-31 07:0x，GitHub API 逐仓库取 HEAD SHA）

上表是「读哪里」；本节钉住**到底读的是哪一版字节**，否则行号会漂。

| 仓库 | default_branch | HEAD SHA（2026-10-31 实测） | archived |
|---|---|---|---|
| `GeeeekExplorer/nano-vllm` | main | `bb823b3e06983d71485a8e1f23715ebd87d98ef8` | false |
| `skyzh/tiny-llm` | main | `da3e841ce4fb70d341e6e799648cb703ca96a96e` | false |
| `vllm-project/vllm` | main | `e5196e2959a42908313510f55fa21f213a48802e` | false |
| `sgl-project/sglang` | main | `8368f86fd7a71b07e2b7dda5d4146695ec54cbfa` | false |
| `flashinfer-ai/flashinfer` | main | `96ca3d014b29f8e07b5215a98f65256ad4d62b69` | false |
| `deepseek-ai/DeepEP` | main | `f6aa38fe06cd4a0489e7730365d4e0114970a2f3` | false |
| `ai-dynamo/dynamo` | main | `6e55febf747467dbd8d1006ad0e21bcda0dc9ceb` | false |

### 6.1 新增关键发现：`nano-vllm` 的块管理器是「自己动手写」的最佳模板

`nanovllm/engine/block_manager.py`（120 行，SHA `bb823b3`）是整条路线里**唯一一个短到能逐行读完、又完整实现 PagedAttention 块语义**的参考实现：

| 行号 | 内容 | 为什么重要 |
|---|---|---|
| `:8-23` | `class Block`：`block_id` / `ref_count` / `hash` / `token_ids`，`reset()` 把 `ref_count=1` | 引用计数的原子语义：重置即被占用 |
| `:28-33` | `BlockManager.__init__`：`blocks[]` + `hash_to_block_id{}` + `free_block_ids: deque` + `used_block_ids: set` | **两个分区 + 一个弱引用索引**：free 队列管「可分配」，used 集合管「在用」；`hash_to_block_id` 不是第三个块集合，它指向的块可能仍在 free 里（见下一行修正） |
| `:36-41` | `compute_hash(token_ids, prefix)` 用 `xxhash.xxh64`，**把上一层 hash 的 8 字节前缀喂进去** | 前缀缓存靠「链式 hash」而不是「整串 hash」，这是 prefix caching 的实现要点 |
| `:43-51` | `_allocate_block`：从 free 队列 popleft，`assert ref_count==0`，然后**仅当 `hash_to_block_id.get(block.hash) == block_id` 时才删映射**（:47-48） | 这个 `== block_id` 判据是防误删别人的映射：同一 hash 可能已被后来的块覆盖，无条件 `del` 会把活跃块的缓存索引删掉 |
| `:58-73` | `can_allocate`：只算到 `seq.num_blocks - 1`（**最后一块不算缓存**，因为它还会长） | 边界条件，最容易写错的地方 |
| `:75-92` | `allocate`：命中块 `ref_count += 1`；未命中块从 free 队列 `remove` | 命中≠重新分配，只加引用 |
| `:94-101` | `deallocate`：**倒序**遍历 `block_table` 递减引用，减到 0 就还回 free 队列——但**不删 hash 映射**（`_deallocate_block` :53-56 只做 `used.remove` + `free.append`） | 这就是「缓存块留在 free 队列里」的证据：释放后仍可被前缀命中（`allocate` :83-88 命中 free 里的块时把它从 free 移走并置 `ref_count=1`）。**任何「free/used/cached 三分」的守恒式都是错的**，cached 会与 free 重复计数 |
| `:103-108` | `can_append` / `may_append`：`len(seq) % block_size == 1` 时才需要新块 | 「跨块边界才分配」的判据 |
| `:110-120` | `hash_blocks`：把刚算完的块登记进 hash 表 | 写路径 |

**这 120 行就是 v0.1 的对照基准**：自己写一份 C++，然后逐条对着它写测试。

### 6.2 vLLM 与 SGLang 的架构差异（面试必问，本轮从源码确认）

- **vLLM**（`vllm/v1/core/block_pool.py`，SHA `e5196e2`）：`BlockHashToBlockMap`（:36）+ `BlockPool`（:137）。
  关键方法：`get_new_blocks` :706 / `touch` :792 / `is_block_writable` :812 / `free_blocks` :816 / `evict_blocks` :851 / `cache_full_blocks` :230 / `cache_partial_block` :486。
  → 结构是 **block + hash 表 + LRU 双向链表**，按 block 粒度管理。
- **SGLang**（`python/sglang/srt/mem_cache/radix_cache.py`，SHA `8368f86`）：`RadixKey` :60 / `TreeNode` :251 / `RadixCache` :282。
  关键方法：`match_prefix` :369 / `insert` :433 / `evict` :557 / `inc_lock_ref` :587 / `dec_lock_ref` :602 / `_split_node` :682。
  → 结构是 **radix tree + 可插拔 evict policy**，按 token 前缀树管理，支持节点分裂。
- 一句话差异：**vLLM 管「块」，SGLang 管「前缀树」**；前者块 id 稳定、block table 可 append-only，后者共享粒度更细但要处理节点分裂与锁引用。

### 6.3 本机构建工具链实测（决定项目形态）

```
g++      /usr/bin/g++          13.3.0 (Ubuntu 24.04)   ← 可 C++20/23
make     /usr/bin/make
cmake    (missing)              ← 开源 C++ 项目要装（pip install cmake 即可，无需 root）
ninja    (missing)
clang++  (missing)
python3  3.12.3  /  pytest 9.1.1 (已装为模块)
GPU      nvidia-smi / rocm-smi 均不存在
```

**结论：本地只能做纯 CPU 可验证的项目。** 任何需要 GPU 的部分（kernel、TTFT 实测）只能写「待云端验证」，不许编数字——与 GITHUB-MAP 第五节一致。

### 6.4 `ai-dynamo/dynamo` 值得单独看（对应 JD 的 PD 分离）

Dynamo 是**推理引擎之上的编排层**，README 明写「it doesn't replace SGLang, TensorRT-LLM, or vLLM, it turns them into a coordinated multi-node inference system」，核心能力正是 **disaggregated serving（PD 分离）+ KV-aware routing + KVBM（KV block manager）+ SLA planner**（README:36, :55, :68）。
→ 这是对模型部署与调度的扩展阅读，可在掌握单机 serving 后学习。原 JD 未明确要求 PD 分离。
