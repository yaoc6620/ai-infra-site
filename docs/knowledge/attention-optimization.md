# Attention 优化

::: tip 待完善
本页为骨架，后续补充详细内容。
:::

## FlashAttention

- 核心：tiling + 在线 softmax，减少 HBM 访问
- 不需要存 N×N attention 矩阵，显存 O(N) 而非 O(N^2)
- IO 复杂度从 O(N^2 d) 降到 O(N^2 d^2 / M)

## MQA / GQA / MLA

| 方法 | KV Head 数 | KV Cache 大小 | 质量 |
|------|-----------|-------------|------|
| MHA | = Q heads | 最大 | 最好 |
| GQA | Q heads / G | 中等 | 接近 MHA |
| MQA | 1 | 最小 | 略降 |
| MLA | 压缩后 | 很小 | 好 |

## Speculative Decoding

- 小模型 draft + 大模型 verify
- 一次 forward 验证多个 token
- 加速比取决于 acceptance rate

## 面试要点

- FlashAttention 为什么快？是算法优化还是 IO 优化？
- GQA 中 group size 的选择
- Speculative Decoding 的 acceptance rate 分析
