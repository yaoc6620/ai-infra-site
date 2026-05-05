# 美团实习

实习期间的推理性能优化项目文档，涵盖 Profiling 分析、方案设计、代码实现和性能验证。

## 项目：RL-Generator Prefill 优化（MOE-26B / NPU 910C）

整体目标：优化 Chunked Prefill 阶段性能，64K 长文本场景总体加速 **42%**。

| 优化方案 | 简述 | 收益 |
|---------|------|------|
| [KV Cache 批量 Gather](/internship/kv-cache-gather-optimization) | for 循环 → index_select，消除 host bound | -5.2% |
| All-to-All EP 通信 | MoE 通信模式优化，减少通信量 | -13.1% (累计) |
| MLA Absorb | 长 KV context 下减少计算量 | -30.8% (累计) |
| mla_prolog 融合算子 | 12 个零散算子融合为 1 个 kernel | -42.3% (累计) |

> 目前已整理的详细文档：KV Cache 批量 Gather 优化
