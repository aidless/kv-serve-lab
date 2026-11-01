# kv-serve-lab

四个模块，把「KV cache 显存怎么算、块池怎么管、连续批处理怎么排、流式指标怎么量」这四件事
串成一条可以在 CPU 上跑通、可以逐项对账的闭环。

纯标准库实现，无第三方运行时依赖，`python3 -m pytest` 138 项全部通过。

> **这不是推理引擎，也不是压测工具。** 它不实现 Attention kernel、不 fork vLLM、不产真实
> GPU 性能数字。它的用途是：把推理服务里那些**只在生产事故里才会暴露的口径问题**，提前
> 做成能在本地一秒内复现的测试。见 [能力边界](#能力边界)。

---

## 四模块

| 模块 | 干什么 | 关键口径 |
|---|---|---|
| `kvcalc` | 从 HF `config.json` 算 KV cache 显存、容量规划 | GQA 用 `2 × n_kv_heads × head_dim`；MLA 用 `(kv_lora_rank + qk_rope_head_dim)`，**与 kv_heads 无关** |
| `blockpool` | PagedAttention 风格的块池：引用计数、前缀复用、写时复制 | 块只有**两个分区**（free / in_use），hash 索引是叠加的弱引用，不是第三个分区 |
| `schedsim` | 连续批处理调度模拟：统一 token 预算、chunked prefill、prefill 末步出首 token | 到达按**墙钟**判定，不按步号换算 |
| `bench` | SSE 流式压测：TTFT / TPOT / 吞吐 | **chunk ≠ token**；`TPOT = (latency − ttft) / (output_len − 1)` |

---

## 快速开始

```bash
git clone https://github.com/aidless/kv-serve-lab.git
cd kv-serve-lab
python3 -m pytest          # 138 passed
python3 examples/demo.py   # 四模块闭环，全程无 GPU
```

`examples/demo.py` 的真实输出（CPU 确定性计算 + 本地 mock server，无 GPU）：

```text
Part 1  kvcalc — KV cache 容量规划
======================================================================
Qwen2.5-7B 每 token KV = 57344 B（全 28 层）
  完整 4096 token 一段 = 0.219 GiB
  24GB 卡、KV 份额 0.8：KV 预算 19.20 GiB
  每请求 256 块（block=16，含末块 padding），占 0.219 GiB
  -> 最大并发 87 个 4096-token 请求

======================================================================
Part 2  blockpool — 块池 / 前缀命中 / COW
======================================================================
分配 2 块 [0, 1]，块 0 写入 payload 并登记前缀 'prefix-A'
  audit: BlockAudit(total=8, used=2, free=6, cached=0)
  前缀命中 block=0，acquire 后该块 ref=2（两请求共享）
  COW：原块 0 ref 降为 1 且保留前缀；新块 2 独占可写，payload=(99, 98, 97, 96)
  audit: BlockAudit(total=8, used=3, free=5, cached=0)

======================================================================
Part 3  schedsim — 连续批处理调度预测
======================================================================
配置：budget=256, max_num_seqs=4, step=0.01s；8 请求,prompt16/output32
  mean TTFT = 0.1300s, p99 = 0.2500s
  mean latency = 0.4400s
  饱和稳态吞吐 = 400.0 tok/s（解析上界 max_num_seqs/step = 400）

======================================================================
Part 4  bench — 预测回放成 SSE，再『实测』回来（闭环）
======================================================================
[4a] 确定性闭环（假时间戳，精确）：
   req   pred_ttft  meas_ttft   pred_tpot  meas_tpot
    r0      0.0100     0.0100      0.0100     0.0100
    r3      0.0100     0.0100      0.0100     0.0100
    r7      0.2500     0.2500      0.0100     0.0100
  合计 3 请求 / 96 token，TTFT/TPOT 完全一致（相对误差 < 1e-9）

[4b] 真实 HTTP 流式（mock 服务端，含毫秒级网络开销）：
  实测 TTFT=<每次运行不同，约 20-80ms>（含 HTTP 往返；理论 ≥20ms）
  实测 TPOT=<每次运行不同，约 5.3ms>，output_len=8，chunk_count=8
```

Part 4a 是**闭环的关键**：`schedsim` 预测出的 TTFT/TPOT 被合成成 SSE 时间线，再喂给
`bench` 的解析器，要求它还原出同一个数。这一步证明「预测」和「测量」用的是同一套口径；
两边对不上，就说明有一边的公式写错了。

---

## 四个模块各自解决的问题

### kvcalc：显存公式的三种写法，错一种就差几倍

GQA（Qwen / Mistral / Llama 系）：

```text
bytes_per_token = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes
```

MLA（DeepSeek-V2/V3 系）**不是**这个公式。它把 K/V 压缩成低秩潜向量，每 token 只存
`kv_lora_rank + qk_rope_head_dim`：

```text
bytes_per_token = (kv_lora_rank + qk_rope_head_dim) × n_layers × dtype_bytes
```

所以 DeepSeek-V2-Lite 在 fp16 下每 token 只有 31,104 B —— 而它有 27 层、16 个 kv heads。
如果按 GQA 公式算，会得到 `2×27×16×128×2 = 221,184 B`，**差 7.1 倍**。

五个基准模型（数字全部来自各自真实的 `config.json`，fp16）：

| 模型 | 层数 | kv heads | head_dim | B/token | 来源 |
|---|---|---|---|---|---|
| Qwen2.5-7B | 28 | 4 | 128 | 57,344 | hf-mirror 实测 |
| Qwen2.5-72B | 80 | 8 | 128 | 327,680 | hf-mirror 实测 |
| Mistral-7B-v0.3 | 32 | 8 | 128 | 131,072 | hf-mirror 实测 |
| DeepSeek-V2-Lite (MLA) | 27 | — | — | 31,104 | hf-mirror 实测 |
| Qwen3-4B | 36 | 8 | 128 | 147,456 | hf-mirror 实测 |

**容量按整块算，不按裸 token 算。** 一个 4097-token 的请求占的是 257 块
（block=16），不是 256 块。`plan_capacity` 对 `seq_len` 做 `ceil(seq_len/block_size)`
再乘以块字节，否则容量会被系统性高估。

### 校准链：把算出来的数字对上引擎自己打印的数字

`tests/test_calibration.py` 里三个数字**全部写死**，来源是 vLLM 启动日志
`arc2/out/gpu_baseline/full.log`（2026-10-21 16:35:42，Qwen3-4B，单卡）：

```text
[gpu_worker.py:373]      Available KV cache memory: 30.53 GiB
[kv_cache_utils.py:1307] GPU KV cache size: 222,336 tokens
[kv_cache_utils.py:1312] Maximum concurrency for 16,384 tokens per request: 13.57x
```

`30.53 GiB ÷ 147,456 B/token = 222,312` token，引擎报 `222,336`，相对误差 0.011%。
这点差异来自日志把 30.53 四舍五入到两位小数，不是公式错。

**防循环论证**：绝不允许拿 222,336 反算出预算、再验证预算是 222,336。三个数各写各的，
只断言我们算出来的量能对上引擎打印的量。变异测试确认这套断言不是摆设：把 GQA 公式
漏掉 `×2`（只存 K 不存 V），6 条断言当场变红。

### blockpool：块只有两个分区

PagedAttention 的块池里，每个块处在**且仅处在**两个集合之一：`free` 或 `in_use`。
前缀缓存用的 `hash → block_id` 是一层**叠加的弱引用索引**——它指向的块仍然在 `free` 或
`in_use` 里，不是第三个分区。

所以守恒式是 `len(free) + len(in_use) == num_blocks`，**不是** `free + used + cached`。
后者会把带 hash 的空闲块重复计数。`_check_conservation()` 逐项核对：

- `free` 与 `in_use` 互斥且覆盖全集；
- `free` 里的块 `ref_count == 0`，`in_use` 里的块 `ref_count > 0`；
- 每条 hash 指向的块必须真的持有那条 hash（悬空 hash 只在 `_hash_to_block.get(h) == block_id`
  时删除，避免误删别的块登记的同一个 hash）。

复用的顺序抄 vLLM `vllm/v1/core/block_pool.py:806-842` 的真实语义：

- **无 hash 的块 LIFO 插队首** —— "LIFO reuse of non-cached blocks for better GPU locality"
- **有 hash 的块 FIFO 插队尾** —— "FIFO reuse of cached blocks for LRU eviction behavior"

写时复制（COW）在本项目里是新增能力：`ref_count > 1` 的块不可写，
`copy_on_write()` 复制出独立 payload 给新块，原块保留 hash（前缀仍可复用）。

### schedsim：没有 prefill 阶段，也没有 decode 阶段

抄 vLLM `vllm/v1/core/sched/scheduler.py:700-708` 的原注释：

> There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has
> its `num_computed_tokens` and `num_tokens_with_spec`... At each step, the scheduler tries
> to assign tokens to the requests so that each request's `num_computed_tokens` can catch up
> its `num_tokens_with_spec`.

所以模拟器里没有「阶段」这个概念，只有**每步的 token 预算怎么分**：运行中的请求优先
（decode 每步 1 个 token、未做完的 prefill 继续 chunk），剩余预算和空 slot 才接纳新请求。
prefill 的最后一步同时采样出首 token —— 这正是 `TPOT` 分母为 `output_len − 1` 的原因。

时间模型（三个常数都是输入参数，测试里写死）：

```text
step_time = step_overhead + prefill_tokens × t_prefill_token + decode_reqs × t_decode_token
```

**到达判定用墙钟，不用步号。** 等步长模型下 `第 k 步 = k × base_step` 成立，但成本模型下
每步墙钟长度随负载变化，按步号换算会过早或过晚接纳请求（实测晚一整步）。空闲跳步也精确
落到下一个到达时刻，不做「按固定步长推」。

### bench：chunk 数不是 token 数

OpenAI 兼容接口的 SSE 流里，一个 chunk 可能一次带多个 token（服务端为省网络往返会打包）。
拿「收到几个 chunk」当 `output_len`，会系统性低估输出长度、把 TPOT 算大。

所以 `output_len` 只能来自两个地方，按优先级：

1. 服务端 `usage.completion_tokens`（累积值，取最后一次）—— **权威**，即使同 chunk 带文本也不覆盖；
2. 没有 usage 时，把每个 chunk 的文本过 tokenizer 累加。

### bench：收尾 usage chunk 不改变 TTFT

`stream_options.include_usage=True` 是最常见的流形状：**正文 chunk 只有文本，最后一个
chunk 才带 usage**。这里有两个独立的量，混了都会错：

- **时序只由文本 chunk 提供**。TTFT 落在第一个带 token 的 chunk 上；若让末尾的
  usage-only chunk 参与，TTFT 会被错量成整条流的结束时刻。
- **latency 量到流结束**（末尾 usage chunk 或 `[DONE]` 的时刻），与 vLLM
  `benchmarks/serve.py` 的 `(latency - ttft) / (output_len - 1)` 口径一致。因此当流以
  usage-only chunk 收尾时，`latency` 晚于最后一个 token 的时刻——这是刻意的，不是 bug。

usage 差额如何归并，取决于上一个 arrival 的来源：

| 流的形状 | 差额处理 | 理由 |
|---|---|---|
| 文本 chunk + 收尾 usage | 并入最后一个文本 arrival | usage 只是把文本估计修正到真值，不是新投递 |
| 只有 usage chunk（无文本） | 新建 arrival | 那个 usage chunk 本身就是一次真实投递时刻 |

对应的回归测试在 `tests/test_bench.py::TestUsageInFinalChunkOnly`：
5 个文本 chunk 各 1 token + 末尾 usage=5 → `output_len == 5`（不是 10）、
`ttft_sec == 0.50`（不是 0.55）、`latency_sec == 0.55`。

两者都没有时**直接抛错**，不许悄悄退化成数 chunk：

```text
BenchError: r: chunk carries text but token_counter is None.
chunk count != token count — pass a tokenizer or have the server send
usage.completion_tokens; do NOT count chunks.
```

`mock_server.py` 同时覆盖两种形状：一 chunk 一 token（`5 token / 5 chunk`）与
一 chunk 多 token（`9 token / 3 chunk`），后者验证 `tokens_per_chunk == 3.0`。

---

## 能力边界

这份清单是给读 README 的人（包括未来的我自己）看的，不是免责声明。

**做到了**（每一条都有测试守着，`python3 -m pytest` 可复现）：

- KV cache 显存公式（GQA / MLA 两条路径），五个真实模型配置对齐；
- 与一次真实 vLLM 启动日志的三方对账（30.53 GiB / 147,456 B / 222,336 tokens）；
- 块池的分配、释放、前缀复用、COW、守恒不变量（含注入违规的反证）；
- 连续批处理调度的 token 预算分配、chunked prefill、到达时刻判定；
- SSE 流式的 TTFT / TPOT / 吞吐解析，含一 chunk 多 token 的识别；
- 四模块闭环：schedsim 预测 → 合成 SSE → bench 还原，相对误差 < 1e-9。

**没做**（写清楚是为了不让人误读上面的数字）：

- **没有任何真实 GPU 性能数字。** 本机无 NVIDIA GPU（`nvidia-smi` / `nvcc` 均不存在），
  所有吞吐/延迟数字要么是 CPU 上的确定性计算，要么是本地 mock server 的实测。
  Part 4b 那个 `TTFT=40.8ms` 是「本机 HTTP 往返 + mock 服务端」的开销，**不是模型推理延迟**；
- 不实现 Attention kernel、不做 kernel fusion、不碰 CUDA/Triton；
- 不 fork vLLM，不与 vLLM 的调度器逐位对齐（只对齐它源码里写明的语义）；
- `schedsim` 是**模拟器**，它的数字是「按这个时间模型应该发生什么」，不是实测；
- `kv_fraction` 是「给 KV cache 的显存份额」，**不等于** vLLM 的 `gpu_memory_utilization`
  （后者是整个进程的显存占比，还要减去权重和激活）；
- `int4` 的字节数只算理想 payload，不含 scale/zero-point 元数据与对齐填充
  （`IDEAL_PAYLOAD_ONLY` 已标出这一点）；
- 没有接入真实 tokenizer，`bench` 在无 usage 的流上需要调用方自己传 `token_counter`。

**下一步要做的**（需要云端 GPU，当前凭据不可用）：

- 在真实 vLLM / SGLang 上跑 `bench`，把 TTFT/TPOT 与引擎自带 `vllm bench serve` 对齐；
- 用 `schedsim` 拟合真实引擎的调度行为，检验时间模型的偏差是否在 10% 以内；
- 把块池换成 C++ 实现，与 Python 版做逐操作对拍。

---

## 目录

```text
kv_serve_lab/
  kvcalc/       # 显存公式 + 容量规划
  blockpool/    # 块池 / 前缀复用 / COW
  schedsim/     # 连续批处理调度模拟
  bench/        # SSE 解析 + 指标聚合
    client.py       # 真实 HTTP 流式客户端
    mock_server.py  # 可控的 SSE mock 服务端
tests/          # 138 项，全部 CPU 可跑
examples/demo.py
docs/superpowers/specs/2026-10-31-kv-serve-lab-design.md
```

## 测试

```bash
python3 -m pytest                              # 138 passed
python3 -m pytest tests/test_calibration.py -q  # 只看与真实引擎对账的那 11 项
```

## 许可

MIT
