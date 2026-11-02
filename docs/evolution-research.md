# 进化方向调研（2026-11-01）

> 用户指示「再找找论文，看看还有什么可以进化的吗，先把项目做大，再求精简」（m01616）。
> 目的：从文献中找 kv-serve-lab 可进化的方向，先做大（加功能），再求精简。

## 检索记录

- 4 组 paper_search，OpenAlex 可用；arXiv/Semantic Scholar 当日 429/超时（已注明，不据此下「无文献」结论）。
- 命中 24+ 篇相关论文，聚焦 **调度策略、SLO 感知、前缀缓存、KV 复用、抢占/重算**。

## 与 kv-serve-lab 强相关的进化方向（按价值排序）

### 1. 调度策略可插拔（最高优先）— 对应 Lightweight Scheduling Policies (arXiv:2610.15899)
- **论文结论**：8 种策略 × 1152 组合（6 模型 × 3 GPU × 8 调度器 × 多种负载），**aging shortest-prompt-first** 是最强实用基线；「oracle 剩余长度调度」显示仅知 job-size 不够（batching/硬件效应也影响性能）。
- **现有 schedsim**：`schedule()` 只有 FCFS/无优先级概念，统一 token 预算，无策略参数。
- **可进化**：把「挑选 waiting 里下一个请求」抽成策略对象（`SchedulerPolicy`），内建 3-4 个：
  - `FCFS`（现状）
  - `AgingShortestPromptFirst`（论文最强基线：等待时间累积老化 + 最短 prompt 优先）
  - `ShortestJobFirst`（剩余 token 最短，对照）
  - `AgingFCFS`（仅老化，隔离老化 vs 长度两个效应）
- **价值**：让 schedsim 成为「策略对比测试台」——用户 JD 明确要「并发调度」理解，且这是论文可复现的空白位。
- **注意**：本仓库 CPU 模拟器，只能复现「策略在时间模型下的相对排序」，不是真实 GPU 绝对数字；README 必须标注。

### 2. 前缀缓存命中率统计（第二优先）— 对应 UniCache (ACM SIGMETRICS 2026)、KV-Repurpose (arXiv:2610.13060)
- **论文结论**：LRU 等简单策略在**异构负载**（多轮对话 + 单轮 API 混合）下失效；需要按工作负载类型差异化 eviction。KV-Repurpose 支持「非前缀段级 KV 复用」（如检索文档在不同 system prompt 后出现）。
- **现有 blockpool**：前缀缓存已有（register/acquire/COW），但**无命中率/复用统计**，也无法区分「缓存命中」vs「重新计算」。
- **可进化**：
  - `blockpool` 加 `PrefixCacheStats`（hit / miss / hit_tokens / miss_tokens / reuse_ratio）。
  - `schedsim` 加「前缀共享」请求生成器（模拟多轮对话：同一前缀 + 不同后缀）。
  - 让 `blockpool` 支持**可插拔 eviction policy**（LRU 现状 / 工作负载感知），对齐 UniCache 论点。
- **价值**：把「前缀缓存到底省了多少 prefill」变成可测、可报的指标——这是面试可讲的「我理解 prefix caching 的量化价值」。

### 3. SLO 感知调度（第三优先）— 对应 SLOWeave (arXiv:2609.07883)、Fairness-Aware Chunked-Prefill (arXiv:2606.09061)、SlidingServe (arXiv:2606.05933)
- **论文结论**：chunked prefill 的 chunk 大小**固定**是缺陷——小 chunk 保 decode 延迟但反复付 launch 开销，大 chunk 提升 prefill 效率但制造延迟尖峰。SLOWeave 动态选「能赶在最早活跃 decode 截止前完成的最大 chunk」。
- **现有 schedsim**：chunked prefill 有（`max_num_batched_tokens`），但 chunk 大小固定、无 SLO 概念、无 deadline 追踪。
- **可进化**：
  - 加 `ttft_slo` / `tpot_slo` 参数 + `deadline` 追踪 + `slo_violations` 统计。
  - 加 `AdaptiveChunking` 策略：每步选 chunk 大小（保守/激进），对齐 SLOWeave。
  - 统计 `slo_attainment`（TTFT/TPOT 满足率）。
- **价值**：把「SLO 达成率」这个真实运维指标引入模拟器——`vllm bench serve` 也报 SLO，能对账。

### 4. KV 缓存压缩/淘汰（第四优先，可能超出 CPU 模拟范围）— 对应 KV Cache Optimization Survey (arXiv:2603.20397)
- **论文结论**：5 大方向（eviction/compression/hybrid memory/novel attention/combination）。
- **可进化**：`kvcalc` 加「压缩模型」参数（如 H2O 式淘汰比例、INT8/FP8 KV 缓存），把「压缩后还能装多少 token」纳入容量规划。
- **价值**：中等。`kvcalc` 目前只算 fp16/bf16 满精度，加压缩口径让它更贴近真实部署。
- **注意**：压缩对**质量**的影响本仓库无法评估（需 GPU），只能算**容量**侧。

## 明确不做的（边界）

- **不做真实 kernel / GPU 测量**：本机无 GPU，与设计文档 §2 非目标一致。
- **不做 PD 分离/disaggregation**（TaiChi/FluidPD）：这是架构级决策，超出 CPU 模拟可验证范围。
- **不做 learned scheduler**（如 DistillCache RL 训练）：需要训练数据与 GPU，且与「轻量策略」的论文结论相悖。
- **不做跨实例调度**（Llumnix）：需多实例，本仓库是单引擎模拟。

## 建议的实施顺序（先做大）

1. **SchedulerPolicy 可插拔**（最核心，直接对应 JD 调度理解）—— schedsim 加策略对象 + 3 个策略 + 策略对比测试。
2. **PrefixCacheStats + 可插拔 eviction + 前缀共享负载生成** —— blockpool 加统计、schedsim 加共享前缀请求。
3. **SLO 感知：deadline + slo_violations + AdaptiveChunking** —— 对齐 SLOWeave。
4. **kvcalc 压缩口径** —— 加 INT8/FP8 KV 缓存容量。

> 以上全部 CPU 可验证（纯模拟 + 确定性断言），不依赖 GPU。
> 云端实验计划（E1-E3，见会话）仍独立成立，用于真实引擎对账。
