# kv-serve-lab 设计文档

> 日期：2026-10-31 · 作者：DSH-Lead（求职会话）
> 状态：待用户批准（brainstorming HARD-GATE：批准前不写产品代码）
> 依据：`infer-engine-track/GITHUB-MAP.md`（同工作区，含 2026-10-31 实测的参考仓库 HEAD SHA 与行号）

---

## 1. 意图与为什么现在做

目标岗位是**推理引擎 / PD 分离调度 / AI 芯片推理优化**。当前证据链最薄的三处是：C/C++ 工程、并行通信、引擎源码级开发。而**推理服务性能分析（TTFT/TPOT/吞吐/显存）**这条其实有真实经验（4.32 tok/s 归因、KV cache 容量核算），但没有一个仓库把它做成可验证的实体。

本仓库要做的事：把「LLM 推理服务的显存与延迟」变成**能在纯 CPU 上跑、能被单元测试验证**的模型。它既是一个可用工具（给 config.json + 显存预算 → 算出能并发多少请求），也是一份**可被面试官逐个追问的可验证产物**。

~~同时补一个明确的账号空白~~ **此说法已被证伪，见 §10 存在性普查**：`aidless` 现有 **76 个公开仓库**（2026-10-31 认证 API 实测），其中已有 5 个覆盖推理 / GPU 方向，**不存在账号空白**。本仓库的价值必须重新论证，不能靠"账号没做过"。

## 2. 目标与非目标

### 目标

| # | 目标 | 验收方式 |
|---|---|---|
| G1 | 四个模块各自可独立运行、可独立测试 | `pytest` 全绿，模块间可单独 import |
| G2 | 每个数字都能复算 | 每个基准值旁注明「怎么算的」，并有测试断言 |
| G3 | 与真实实现语义对齐 | 块管理对照 nano-vllm（MIT，SHA `bb823b3e`）；指标口径对照 vLLM（`serve.py:656-658`） |
| G4 | 公开仓库可被别人用起来 | README 有 3 行能跑起来的 quickstart；`pip install -e .` 可用 |

### 非目标（明确不做）

- **不实现 Attention kernel**——本机无 GPU，写了也验证不了。
- **不声称任何真实 GPU 上的 TTFT/TPOT 数字**——所有真实端点数据必须标注来源与采集时间，否则不写。
- **不 fork vLLM 改代码**——需要 GPU 环境才能测，不在本仓库范围。
- **不追求性能**——`schedsim` 是离散事件模拟，目标是「语义正确、能对上解析上界」，不是跑得快。

### 反目标（做成了也算失败）

- 四模块各自为政、互相不引用——那就是四个玩具，不是一个系统。
- 测试只覆盖 happy path，边界（块耗尽、双 free、COW 竞态）没测——那就是装饰。
- README 吹了「可用于生产容量规划」但没标适用边界——那是过度声称。

## 3. 约束

| 约束 | 来源 | 影响 |
|---|---|---|
| 本机无 GPU（`nvidia-smi`/`rocm-smi` 均不存在） | 实测 2026-10-31 | 所有 GPU 侧验证只能标「待验证」 |
| 本机无 cmake/ninja，g++ 13.3.0 可用 | 实测 | 纯 Python 项目不受影响；若后续加 C++ 扩展需 `pip install cmake` |
| Python 3.12.3 + pytest 9.1.1 已装 | 实测 | 不需要额外装测试框架 |
| HF 官方站超时，`hf-mirror.com` 需带浏览器 UA 才不 403 | 实测 | 拉 config.json 的脚本必须带 UA + 走 mirror + 可离线用本地 fixture |
| AGENTS.md #17：实验一律上云，本地只做编排与亚分钟级 smoke | 工作区纪律 | 本仓库全部是 CPU 计算，属「编排/解析」范畴，不受限 |

## 4. 四个模块的设计

```
kv-serve-lab/
├── pyproject.toml
├── README.md
├── kvcalc/          KV 显存容量规划器
├── blockpool/       PagedAttention 块管理器
├── schedsim/        continuous batching 调度模拟器
└── bench/           OpenAI 兼容端点压测客户端
└── tests/           四模块的测试（含基准值断言）
```

### 4.1 `kvcalc` — KV 显存容量规划器

**输入**：HF `config.json`（dict 或路径）+ 显存预算 + dtype + TP 并行度 + 目标序列长度
**输出**：`bytes_per_token`、每请求 KV 占用、最大并发数、需要的 block 数

公式（从 vLLM 源码读出并对齐，`vllm/v1/kv_cache_interface.py` 的 `state_content_size_bytes` / `unpadded_page_size_bytes`）：

- **MHA / GQA**：`bytes_per_token = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes`，其中 `head_dim = hidden_size / num_attention_heads`
- **MLA**（`model_type == "deepseek_v2"` 等）：`bytes_per_token = (kv_lora_rank + qk_rope_head_dim) × n_layers × dtype_bytes`（`head_size_v = 0`）

**验收基准**（本轮已从 hf-mirror 实拉 config 并独立复算，测试里逐字钉住）：

| 模型 | 类型 | 关键参数（实拉） | bytes/token | 4096 token 全层 |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | GQA | 28 层 / 28 heads / 4 kv_heads / hidden 3584 | 57,344 | 0.2188 GiB |
| Qwen2.5-72B-Instruct | GQA | 80 层 / 64 heads / 8 kv_heads / hidden 8192 | 327,680 | 1.2500 GiB |
| Mistral-7B-v0.3 | GQA | 32 层 / 32 heads / 8 kv_heads / hidden 4096 | 131,072 | 0.5000 GiB |
| DeepSeek-V2-Lite | MLA | 27 层 / kv_lora_rank 512 / qk_rope 64 | 31,104 | 0.1187 GiB |

**第五行是校准行（与真实引擎对账）**：

| Qwen3-4B | GQA | 36 层 / 32 heads / 8 kv_heads / hidden 2560 / head_dim 128（config 显式给） | 147,456 | 0.5625 GiB |

校准依据（**三个量各自独立，不是循环论证**）：真实 vLLM 运行日志（W7900 48 GiB，bf16，`gpu_memory_utilization=0.8`，Qwen3-4B 微调权重）分别打印了：

| 日志行 | 引擎输出 | 谁测的 |
|---|---|---|
| `full.log:25` | `Available KV cache memory: 30.53 GiB` | vLLM 自己按显存预算算出（`0.8 × 48 GiB` 减去权重 6.95 GiB 与激活） |
| `full.log:26` | `GPU KV cache size: 222,336 tokens` | vLLM 用**它内部**的 bytes/token 除出来的 |
| `full.log:27` | `Maximum concurrency for 16,384 tokens per request: 13.57x` | 同上，`222,336 / 16,384 = 13.57` |

我们独立算出的 `bytes_per_token = 147,456`，代入引擎自己报的 30.53 GiB 预算 → 222,312 token / 13.57 并发。**输出端 222,312 vs 引擎 222,336（差 0.011%），并发端 13.57 vs 13.57（完全一致）。**差异来源是 `30.53` 只有两位小数（GiB 舍入），不是公式错。

**这条链的测试写法**：把三个数都写死（30.53、147,456、222,336），断言 `abs(222312 - 222336) / 222336 < 0.001` 且并发数四舍五入到两位等于 13.57。**不许**在测试里用 222,336 反算预算再验证自己。

**测试要求**：
- 上表四个值逐字断言（测试里写死数字，不是重算一遍再比——重算会一起错）。
- `head_dim` 从 `hidden_size / num_attention_heads` 推导，config 里若显式给了 `head_dim` 则优先用它。
- TP 并行度 > 1 时：kv_heads 按 TP 切分（`num_key_value_heads % tp != 0` 时明确抛错，不静默取整）。

### 4.2 `blockpool` — PagedAttention 块管理器

语义对照 `nano-vllm/nanovllm/engine/block_manager.py`（MIT，SHA `bb823b3e`，120 行），但**不复制其代码**，按 vLLM v1 `block_pool.py` 的接口重新设计。

核心结构（**已对照 nano-vllm `:26-33` 逐行确认，不要凭直觉写成「三态」**）：

- `free_blocks: deque[int]` — 可分配
- `used_blocks: set[int]` — 被至少一个请求持有（`ref_count > 0`）
- `hash_to_block_id: dict[hash, block_id]` — **不是第三个块集合，是弱引用索引**：它指向的块可能此刻在 `free` 里（`deallocate:94-101` 把 `ref_count` 减到 0 的块还回 free，但**不清 hash**，所以缓存仍然命中），也可能在 `used` 里。

因此块只有两个分区，`hash_to_block_id` 是叠加在它们之上的索引。**任何「free + used + cached」式的守恒式都是错的**——cached 会与 free/used 重复计数。

**必须通过的不变量测试**（这是核心卖点）：

| # | 不变量 | 反例注入（必须让测试红） |
|---|---|---|
| I1 | 块守恒：`len(free) + len(used) == num_blocks`，且两集合互不相交，任何操作后都成立 | 漏还一个块 → 断言失败 |
| I2 | 引用计数永不为负；对同一请求 deallocate 两次必须抛错 | 二次 deallocate → 抛 `DoubleFreeError`（nano-vllm `:54` 只有 `assert`，我们要抛具名异常） |
| I3 | 块耗尽时必须抛错，不能静默返回 None 或越界，且失败后 I1 仍成立 | 分配超过容量 → 抛 `OutOfBlocksError` |
| I4 | **悬空 hash 清理**：块被重新分配时，若 `hash_to_block_id[h]` 仍指向自己才删（nano-vllm `:47-48` 的 `== block_id` 判据）。同一 hash 已被别的块覆盖时不得误删 | 删掉 `== block_id` 判据 → 删了别人的映射，后续前缀命中读到错误 token_ids |
| I5 | **弱引用命中**：块在 free 队列里但 hash 仍登记时，新请求能命中它（命中后从 free 移除并置 `ref_count=1`，nano-vllm `:83-88`） | 让 deallocate 顺手清 hash → 命中率跌到 0 |
| I6 | 最后一个块不参与前缀复用：`can_allocate` 只检查到 `num_blocks - 1`（nano-vllm `:62`），因为末块可能还没写满 | 把末块也算进缓存 → 命中数多算 1 |
| I7 | **COW（本项目新增，nano-vllm 没有）**：共享块被写入前必须复制，新旧 block id 不同，旧请求内容不变。判据对照 vLLM v1 `block_pool.py:812` `is_block_writable` | 直接改共享块 → 断言旧请求读到旧内容失败 |

**明确标注**：I1–I3、I5–I6 是 nano-vllm 已有语义的复刻验证；**I4 是它代码里真实存在的细节陷阱**；**I7 是我们自己加的设计**（nano-vllm 无 COW，不可拿它当依据）。

**测试方法**：每个不变量写两个测试——一个正向（合法操作后成立）、一个反向（注入违规后必须报错或断言失败）。**反向测试必须真的能红**，写完立刻验证它红过。

### 4.3 `schedsim` — continuous batching 调度模拟器

镜像 vLLM v1 的调度模型（`scheduler.py:698` 的注释原文）：**没有 prefill/decode 两阶段，只有每个请求的 `num_computed_tokens` 追 `num_tokens_with_spec`，共用一个 token 预算**。

**输入**：请求到达轨迹（固定速率 / 泊松）+ **时间模型**（见下）
**输出**：时间序列的 TTFT / TPOT / 吞吐 / 队列长度

**时间模型（必须显式定义，否则算出来的数没有意义）**——离散事件按 step 推进，每个 step 的耗时：

```
step_time = step_overhead + (prefill_tokens × t_prefill_token) + (decode_reqs × t_decode_token)
```

其中 `prefill_tokens` = 本 step 分配给各请求的新 prompt token 总数（chunked prefill 下可为部分），`decode_reqs` = 本 step 处于 decode 的请求数。`step_overhead` 是每 step 固定开销（权重读取），这是为什么并发高时吞吐上升——**单 token 成本随批量摊薄**。三个参数都是模拟器的**输入常数**，不是拟合出来的，测试里必须写死。

**验收方式**（对解析上界，偏差超阈值即失败）：

- **decode 饱和吞吐上界**：decode 每 step 每请求产 1 token，故 `throughput_max = max_num_seqs / (step_overhead + max_num_seqs × t_decode_token)`。**注意**：不是 `max_num_batched_tokens / t_decode_token`——那个式子量纲不对，token 预算约束的是**每 step 调度多少 token**（含 chunked prefill），不是并发数上限。
- **Little's law**：`平均在系统内请求数 = 到达率 × 平均逗留时间`，模拟值偏差需 < 5%
- **TTFT 下界**：当系统空载时，`TTFT >= step_overhead + prompt_len × t_prefill_token`，模拟不得违反

阈值（初始值，可在实现中按实测调整但必须写进测试）：Little's law 偏差 < 5%；饱和吞吐偏差 < 10%。

### 4.4 `bench` — OpenAI 兼容端点压测客户端

对任意 OpenAI 兼容端点（vLLM / SGLang / TGI）发流式请求，按 vLLM 口径算指标：

- **TTFT** = 首个 SSE chunk 到达时间
- **TPOT** = `(latency - ttft) / (output_len - 1)`（`vllm/benchmarks/serve.py:656-658` 原式，扣掉首 token）
- **ITL** = 相邻 chunk 间隔
- **吞吐** = 总 token / 墙钟

**⚠️ chunk ≠ token（这是最容易做错的地方）**：SSE 的每个 chunk 可能携带**多个** token（vLLM 在 decode 阶段常一次发多 token），也可能只带 usage 信息。因此：

- `output_len` **不能**用 chunk 计数，必须从 chunk 内容里累加实际 token 数（或读最终 `usage.completion_tokens`）。
- `latency - ttft` 除以 `(output_len - 1)` 时，`output_len` 是 token 数；用 chunk 数会系统性高估 TPOT。
- mock server 必须**同时**覆盖两种情形：一 chunk 一 token（便于逐毫秒断言）与一 chunk 多 token（验证 token 计数没写成 chunk 计数）。

**本地可验证部分（纯 CPU）**：写一个 mock SSE server，注入**已知时序**（比如首 token 在 100ms、之后每 20ms 一个 token），断言算出来的 TTFT/TPOT 与注入值逐毫秒相等。再注入「一个 chunk 带 3 个 token」的时序，断言 `output_len` 按 token 计而非 chunk 计。

**需要 GPU 的部分**：真实端点对比 → 明确标「待云端验证」，不编数字。README 里给出跑法，但不放任何未实测的数字。

## 5. 模块之间怎么串起来（这是「一个系统」而不是「四个玩具」的关键）

一条端到端 demo（`examples/end_to_end.py`）：

1. `kvcalc` 读 `Qwen/Qwen3-4B` 的 config（36 层 / 8 kv_heads / head_dim 128 / bf16 → **147,456 bytes/token**）+ 引擎自报的 KV 预算 30.53 GiB → 算出 222,312 token 容量、16,384-token 请求下 **13.57 并发**。与真实 vLLM 日志的 `222,336 tokens` / `13.57x` 对上（差 0.011%，仅因 30.53 只保留两位小数）——**这是本仓库唯一一条能与真实引擎对账的校准点，必须写成测试**（写法见 §4.1，禁止反算自证）
2. 用这个并发数作为 `blockpool` 的容量约束 → 跑一遍模拟请求生命周期，验证块不泄漏
3. 把这个容量与单步耗时喂给 `schedsim` → 输出 TTFT/TPOT/吞吐预测
4. `bench` 的 mock server 用第 3 步的预测参数生成时序 → 验证 `bench` 的指标计算能还原出第 3 步的输入

**闭环断言**：第 4 步还原出的 TTFT/TPOT 与第 3 步预测值的相对误差 < 1%。这条跑通，才说明四个模块真能互相咬合。

## 6. 留给 builder 的决定

- 包的目录结构细节（是否用 `src/` 布局）
- 具体的数据类用 `dataclass` 还是 `NamedTuple`
- 测试文件如何组织（一模块一文件 vs 按功能分）
- `pyproject.toml` 的打包工具（setuptools / hatchling）
- 是否加 `ruff` / `mypy` 的 CI

## 7. 交付节奏（建议，非硬性）

| 阶段 | 内容 | 出口条件 |
|---|---|---|
| P1 | `kvcalc` + `blockpool` | 两模块测试全绿，含全部反向测试；四模型基准值断言通过 |
| P2 | `schedsim` | 对 Little's law 与饱和吞吐两个上界的偏差断言通过 |
| P3 | `bench` + 端到端 demo | mock server 还原误差 < 1%；闭环断言通过 |
| P4 | 打包 + README + 发布 | `pip install -e .` 可用；README quickstart 三行能跑；推到 GitHub |

## 8. 数字纪律（写进 README 的承诺）

- 每个基准数字旁必须注明来源：`config.json` 实拉（附模型 id 与拉取日期）或源码行号。
- 任何 GPU 侧数字必须标「待验证」或附采集环境（卡型、驱动、时间）。
- 不写「可用于生产容量规划」这类没有边界的话——写清楚适用边界（单机单卡、不考虑 prefix 共享时的上界）。

## 9. 参考（本轮实测 SHA，2026-10-31）

- `GeeeekExplorer/nano-vllm` `bb823b3e` — MIT；`nanovllm/engine/block_manager.py` 120 行
- `vllm-project/vllm` `e5196e29` — `vllm/v1/core/block_pool.py`、`vllm/benchmarks/serve.py:656-658`、`vllm/v1/kv_cache_interface.py`
- `sgl-project/sglang` `8368f86f` — `python/sglang/srt/mem_cache/radix_cache.py`（架构对照，本版不实现 radix tree）

## 10. 存在性普查：本设计的四个模块，账号里已有五个仓库覆盖

普查在写完本设计之后、写产品代码之前执行（`census-before-recommending-a-build`）。**结论是 `ALREADY_BUILT` 而非 `MUST_BUILD_NEW`**，本设计因此需要重新定位。

### 10.1 已存在的仓库（认证 API 实测，2026-10-31）

| 仓库 | 创建/推送（UTC） | 覆盖本设计哪个模块 | 实际规模 |
|---|---|---|---|
| `aidless/paged-kv` | 13:45 / 13:45 | **blockpool**（且是 C++17，比 Python 更贴 JD） | `.h` 84 + `.cpp` 118 + 测试 185 行，`make test` → `ALL PASS` |
| `aidless/roofline-lint` | 14:58 / 16:26 | **kvcalc**（且带 `--selftest` 13 点 + `--prove-red`） | 零依赖单文件 CLI，30 tests |
| `aidless/ttftkit` | 15:07 / 15:46 | **bench**（且指标口径比本设计更完整） | 1962 行 Python，测试全绿 |
| `aidless/kv-block-mass` | 16:13 / 17:36 | 本设计没有的 kernel 侧延伸 | PyTorch reference + Triton，CUDA/ROCm 双向实测 |
| `aidless/accel-preflight` | 16:17 / 16:17 | 本设计没有的可移植性预检 | 22KB，零依赖 |

本地可跑的两份已实测：`paged-kv` → `ALL PASS`；`ttftkit` → `pytest` 全绿。

### 10.2 三个模块的覆盖对比（为什么不能重复造）

| 本设计模块 | 已有仓库做到了什么 | 本设计能多做什么 |
|---|---|---|
| `kvcalc` | roofline-lint 已算 `KV bytes/token = 2·L·h_kv·d·b_kv`、容量、fit 判定、独立 weight/KV precision、MoE、TP 切分 | **只有一处**：把 `bytes/token` 与真实 vLLM 引擎日志的 `222,336 tokens / 13.57x` 做三方对账（见 §4.1）。这个校准点 roofline-lint 没有 |
| `blockpool` | paged-kv 已实现 alloc/append/fork/COW/双释放安全/失败无副作用，**且有真 C++ 测试** | 本设计 I4（悬空 hash 的 `== block_id` 判据）与 I5（弱引用命中）paged-kv 未覆盖；但它缺的是 prefix hash 命中率统计，不是块管理本身 |
| `schedsim` | **无人做过**——`kv-block-mass`/`roofline-lint` 里 `continuous batching` 命中数为 0 | 这是唯一真空白，但它是**模拟器**，不产生真实 serving 数据 |
| `bench` | ttftkit 已把坑写全：TTFT 取首个非空 `delta.content`、TPOT 用服务端 `usage.completion_tokens` 而非 SSE 块数、缺 usage 记 `null` 不记 0、`chunk_gap_ms` 明确不叫 ITL、fail-closed 门禁 | 本设计 §4.4 的"chunk≠token"是 ttftkit 已实现的同一件事，**属重复发明** |

### 10.3 保留用户选定的四模块，明确学习项目的用途

用户已选定 Python 四模块全做；存在同主题仓库不改变这一范围。`kv-serve-lab` 定位为纯 CPU 可运行的学习实验室，通过一套示例串联 KV 容量、块生命周期、continuous batching 调度和流式测量。已有专用仓库用于交叉阅读和验证，模块重叠需要在 README 中说明。

- `schedsim` 重点验证统一 token 预算、请求到达、并发槽位和可变步长时间模型。调度语义参考 vLLM，时间成本是显式设定的模拟假设，输出不作为真实 GPU 性能结论。
- `kvcalc` 增加 §4.1 的真实引擎日志对账；与 `roofline-lint` 的解析估计区分用途。
- `blockpool` 用 Python 展示前缀缓存和引用计数，与 `paged-kv` 的 C++ 实现交叉阅读。
- `bench` 用确定性时序与本地 HTTP 示例验证测量口径；已有 `ttftkit` 可用于更完整的端点压测与发布门禁。

可见性遵循实施计划中的用户选择：先私有、后公开。

### 10.4 并发写入冲突（本轮实测，必须记录）

本设计落盘期间，`kv-serve-lab/` 目录**正被另一个并行会话实时写入**：`kv_serve_lab/kvcalc/__init__.py`(172 行)、`blockpool/__init__.py`(285 行)、`schedsim/__init__.py`(407 行)、四个测试文件，mtime 18:29–18:51，`python3 -m pytest` 当前因 `ImportError: cannot import name 'IDEAL_PAYLOAD_ONLY'` 在收集阶段失败。`scripts/dsh-idle-check.py` 报 **9 个会话正在干活**。

**处置**：本会话不往 `kv-serve-lab/` 写产品代码（避免双写覆盖），只写这份设计文档与普查结论。产品代码归正在写它的那个会话。
