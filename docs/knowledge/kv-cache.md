# KV Cache

::: tip 待完善
本页为骨架，后续补充详细内容。
:::

## 为什么需要 KV Cache

- Autoregressive 生成：每个 token 需要之前所有 token 的 K/V
- 不缓存：每次重算，复杂度 O(n^2)
- 缓存后：增量计算，每步只算新 token 的 Q@K^T

## 内存分析

```
KV Cache 大小 = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch × dtype_bytes
```

## PagedAttention

- 将 KV Cache 分成固定大小的 Block
- 逻辑连续 → 物理不连续（类似 OS 虚拟内存）
- 消除内存碎片，支持动态序列长度

## 面试要点

- KV Cache 的显存占用计算
- Prefix Caching 原理
- Block 大小的 trade-off
- MQA/GQA 如何减少 KV Cache
