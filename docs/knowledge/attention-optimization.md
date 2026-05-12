# Attention 优化

## FlashAttention

详见 [FlashAttention 专题](/knowledge/flash-attention)（Tiling + Online Softmax + FA2/FA3）。

## MQA / GQA / MLA

| 方法 | KV Head 数 | KV Cache 大小 | 质量 | 代表模型 |
|------|-----------|-------------|------|---------|
| MHA | = Q heads (如 64) | 最大（基准） | 最好 | GPT-3, LLaMA-1 |
| GQA | Q heads / G (如 8) | 基准的 1/G | 接近 MHA | LLaMA-2/3, Mistral |
| MQA | 1 | 基准的 1/num_heads | 略降 | PaLM, Falcon |
| MLA | 压缩维度 (如 512) | 很小 | 好 | DeepSeek-V2/V3 |

### GQA：在质量和效率之间取平衡

```
MHA (num_kv_heads = num_q_heads = 32):
Q heads: [h0][h1][h2]...[h31]
K heads: [h0][h1][h2]...[h31]   ← 每个 Q head 有自己的 KV

GQA (num_kv_heads = 8, group_size = 4):
Q heads: [h0 h1 h2 h3] [h4 h5 h6 h7] ... [h28 h29 h30 h31]
K heads: [    k0      ] [    k1      ] ... [       k7       ]
         ↑ 4个Q head 共享 1个KV head

MQA (num_kv_heads = 1):
Q heads: [h0 h1 h2 ... h31]
K heads: [       k0       ]   ← 所有 Q head 共享 1 个 KV
```

GQA 的 group_size 越大，KV Cache 越小，但质量可能下降。LLaMA-2-70B 用 GQA-8（8 个 KV head），在质量和 KV Cache 大小之间取得很好的平衡。

### MLA：低秩压缩

DeepSeek-V2 的 MLA 不存储完整的 K/V，而是存储低秩压缩后的 latent vector：

```
标准: 缓存 [K, V] 每层每 token = 2 × num_kv_heads × head_dim
MLA:  缓存 compressed_kv 每层每 token = d_c (如 512)

推理时: compressed_kv → W_uk → K, W_uv → V (在线解压)
```

KV Cache 大小从 `2 × num_kv_heads × head_dim` 降到 `d_c`，压缩比可达 10× 以上。代价是 decode 时需要额外的矩阵乘解压（可通过 absorb 优化消除）。

## 面试要点

::: details 常见面试问题

**Q: FlashAttention 为什么快？**

IO 优化，不是减少 FLOPs。通过 tiling 在 SRAM 中完成计算，避免 N×N attention 矩阵写回 HBM。详见 [FlashAttention](/knowledge/flash-attention)。

**Q: GQA 的 group size 怎么选？**

Trade-off：group 越大 → KV Cache 越小 → 但质量可能下降。实践中 LLaMA-2-70B 用 8 个 KV head (group=8)，Mistral-7B 也用 8 个。小模型（7B）可以用更大的 group，因为总 head 数少。

**Q: MLA 和 GQA 有什么区别？**

GQA 是减少 KV head 数量，MLA 是对整个 KV 做低秩压缩。MLA 压缩比更高，但需要额外的解压矩阵乘。DeepSeek-V2 通过 "absorb" 优化把解压操作吸收进 attention 计算中，消除了额外开销。

:::
