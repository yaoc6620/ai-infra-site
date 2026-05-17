# Indexer DCP 并行优化

## 一、DSA 原理

### 什么是 DSA

DSA（Dynamic Sparse Attention）是 DeepSeek V3 的稀疏注意力机制。核心思想：不对所有 KV token 做 attention，而是先用一个**轻量级 Indexer** 选出最重要的 TopK 个 KV token，再只对这 TopK 个做精确 MLA attention。

```
全量 MLA attention:  128 heads × 35000 KV tokens  →  太慢
    ↓
Indexer 筛选:        64 heads × 35000 KV tokens   →  选出 2048 个（轻量）
    ↓
精确 MLA attention:  128 heads × 2048 KV tokens   →  快且准
```

### Indexer 的计算流程

Indexer 是一个**轻量级多头注意力 + 加权 TopK 选择**。它有自己独立的一套投影权重，和 MLA 不共享。

#### 1. 投影：生成 Indexer 专用的 Q、K、weights

```python
# Indexer.__init__ — 全量复制（ReplicatedLinear），不做 TP 切分
self.wq_b = ReplicatedLinear(q_lora_rank, head_dim * n_head, ...)    # 1536 → 64×128
self.wk = ReplicatedLinear(hidden_size, head_dim, ...)                # 5120 → 128
self.weights_proj = ReplicatedLinear(hidden_size, n_head, ...)        # 5120 → 64 (float32)
```

```python
# Indexer.forward — mlp_indexer_fusion 内部做的事
q_nope = q_c @ wq_b       # [S, 1536] → [S, 64, 128]   ← 从 MLA 压缩 latent 解压
k_nope = hidden @ wk       # [S, 5120] → [S, 1, 128]    ← K 只有 1 个 head
k_nope = k_norm(k_nope)
# + RoPE
q = cat([q_pe, q_nope])    # [S, 64, 128]
k = cat([k_pe, k_nope])    # [S, 1, 128]

weights = hidden @ weights_proj   # [S, 64]  (float32)
weights *= softmax_scale * (1/√64)
```

**关键点 1：Indexer 的 Q 输入是 MLA 的压缩 latent `q_c`**——和 MLA attention 共享同一个 `q_c`，但用 Indexer 自己的 `wq_b` 解压成 64 个 head（MLA 用 `q_b_proj` 解压成 128 个 head）。

**关键点 2：Indexer 的所有权重都是 `ReplicatedLinear`（全量复制），不做 TP 切分**。每个 rank 算出来的 Q、K、weights 完全一样。不切 TP 的原因：
- TopK 选择是跨所有 head 加权求和后做的，按 head 切 TP 后每个 rank 只有部分 head 的分数，没法得到全局加权和
- Indexer 只有 64 head × 128 dim，计算量远小于 MLA 的 128 head × 192 dim，切分收益有限

#### 2. mlp_lightning_indexer：多头打分 + 加权 TopK

这是华为 NPU 的定制算子（`torch_npu.mlp_lightning_indexer`），源码不可见，从接口参数和 DeepSeek V3 论文可推断内部逻辑：

```
对每个 query token i（能看到 KV 位置 0 ~ kv_upper）:

  Step 1: 多头 attention score
    score[h][j] = q[i, h] · k[j] / √d    ← 64 个 head，K 只有 1 个 head（所有 head 共享）

  Step 2: 加权融合
    fused_score[j] = Σ_h  weights[i, h] * score[h][j]
    → 64 个 head 的 score 按 weights 加权求和，得到每个 KV 位置的综合重要性

  Step 3: TopK 选择
    topk_indices[i] = argtopk(fused_score, k=2048)
```

K 只有 1 个 head 是 Indexer 轻量的关键——64 个 head 的 Q 从不同"视角"看同一份 KV，加权投票决定哪些 token 最重要。

#### 3. 三类 token 的选择策略

最终选出的 2048 个 KV 位置由三部分组成：

| 类型 | 数量 | 选择方式 | 原因 |
|------|------|---------|------|
| Init tokens | `init_num`（4~16） | **强制选入**位置 0, 1, ..., init_num-1 | Attention sink：BOS 等句首 token 在几乎所有 query 的 attention 中权重都很高 |
| Local tokens | `local_num`（1024） | **强制选入**最近 1024 个 token | 局部性：当前 token 附近的上下文几乎总是最相关的 |
| Sparse global | 剩余（~1020） | attention score **竞争选出** | 从全局 KV 中选出真正重要的远距离 token |

实际通过 attention 打分竞争的名额只有 `2048 - init_num - local_num ≈ 1020` 个。

### 短序列怎么办

当 KV 总长度 ≤ 2048（即 topk）时，**Indexer 照常跑，结果等于全选**：

- `mlp_lightning_indexer` 扫一遍所有 KV，发现不到 2048 个，全部选入
- 输出 `topk_indices` 前面是所有有效位置，后面空位填 `-1`
- 下游 `mlp_sparse_flash_attention` 对 `-1` 位置跳过

短序列确实多跑了一次 Indexer，但不是问题：

1. **Indexer 耗时和 S_kv 成正比**：KV=5120 时只需 ~1ms，几乎可忽略。Indexer 慢是在 S_kv=35000 这种长序列场景（~13ms）
2. **架构上不能跳过**：下游用的是 `mlp_sparse_flash_attention`，必须接收 `sparse_indices` 参数。整个 sparse attention 路径是训练时确定的，不能运行时切换成普通 flash attention

### Indexer 的完整调用链

```python
# forward_sparse 中的流程
q_c = hidden_states @ q_a_proj          # [S, 5120] → [S, 1536]  压缩
q_c = q_a_layernorm(q_c)

# ▸ MLA attention 的 Q（TP 切分，每 rank 16 个 head）
q = q_b_proj(q_c)                        # [S, 1536] → [S, 16, 192]  ColumnParallel

# ▸ Indexer 的 Q（全量复制，每 rank 都是 64 个 head）
self.indexer(hidden_states, q_c, rotary_emb)
#   内部: q_c @ wq_b → [S, 64, 128]     ReplicatedLinear

# ▸ Sparse MLA attention（使用 Indexer 选出的 topk_indices）
self.mla_attn(q, kv_c_normed, k_pe)
#   内部: mlp_sparse_flash_attention(q, kv, sparse_indices=topk_indices)
```

同一个 `q_c`，两套不同权重，各自解压成不同的 Q。

---

## 二、优化前：Owner 独占计算

### 为什么需要 Owner

在 DCP 环境下，MLA 的 KV Cache 被 interleave 分散到 8 个 rank，但 Indexer 需要**完整序列**做 TopK（因为要从所有 KV 中选出最重要的 2048 个）。

两个选择：
1. 每个 rank 各自存一份完整 KV Cache → 8 倍显存冗余，违背 DCP 省显存的初衷
2. **选一个 owner 存完整 cache**，其他 rank 等结果 → 当前方案

### Owner 分配策略

```python
# Indexer.__init__
self.sharding_owner = Indexer._get_bucket(config.num_layers, self.dcp_size, layer_num)
if self.sharding_owner == self.dcp_rank:
    self.k_cache = DeepseekV32IndexerCache(...)   # 只有 owner 有 cache
```

Owner 按层号轮流分配（`_get_bucket` 把 38 层均分给 8 个 rank），保证每个 rank 只 owner 约 4~5 层的 Indexer cache，分摊存储压力。

### Indexer K Cache 与 MLA KV Cache 的关系

两者是**独立的 cache pool**，但共用同一套 `block_table` 编号（block allocator 统一分配）：

| | MLA KV Cache | Indexer K Cache |
|---|---|---|
| 存什么 | kv_c(512d) + k_pe(64d) | K(128d) |
| 谁存 | 所有 rank（DCP 分片） | 只有 owner |
| block_size | 128 | 128 × dcp_size = 1024 |
| shape | `[num_blocks, 128, 1, 576]` | `[num_blocks, 1024, 1, 128]` |

Indexer 的 block_size 乘了 dcp_size（`128×8=1024`），使得**在同一个 block 编号下**，Indexer 每个 block 存 8 倍 token，总容量等于完整序列：

```
MLA (DCP, 每 rank 1/8):    35000/8 ÷ 128 = 35 blocks
Indexer (Owner, 完整序列):  35000 ÷ 1024  = 35 blocks   ← 数量一样！
```

所以同一个 `block_table = [5, 102, 200, ...]` 对两边都能用——同一个编号去不同的物理 pool 里取各自的 block。

### 优化前的执行流程

```
┌──────────────────────────────────┐   ┌──────────────────────────────────┐
│       DCP rank 0 (owner)         │   │     DCP rank 1-7 (non-owner)     │
├──────────────────────────────────┤   ├──────────────────────────────────┤
│ ① scatter k → kv_cache          │   │ (skip — 无 cache)                │
│   slot_mapping: [2048, 1]        │   │                                  │
├──────────────────────────────────┤   ├──────────────────────────────────┤
│ ② mlp_lightning_indexer          │   │ ② dummy mlp_lightning_indexer     │
│   query:   [2048, 64, 128]       │   │   (保持图拓扑一致的空跑)          │
│   key:     kv_cache (PA_BSND)    │   │   ⏱️ ~0.1ms                      │
│   cum_k:   [0, 35000]            │   │                                  │
│   → topk [2048, 1, 2048]        │   │   → topk [2048, 1, 2048] (无效)  │
│   ⏱️ ~12.9ms ← 瓶颈!             │   │                                  │
├──────────────────────────────────┤   ├──────────────────────────────────┤
│ ③ broadcast(topk, src=owner)     │   │ ③ recv broadcast                 │
│   16MB                           │   │                                  │
└──────────────────────────────────┘   └──────────────────────────────────┘

总耗时: ~13ms/层, 38 层 ≈ 490ms
问题: Owner 独占 2048×35000 的计算量，7 个 rank idle
```

Non-owner 的 dummy indexer 是为了保持所有 rank 的**计算图拓扑一致**（NPU 编译器要求各 rank 执行相同的算子序列），实际不做有意义的计算。

### 优化前的代码

```python
def forward(self, hidden_states, qr, rotary_emb):
    # ... 投影 Q、K、weights ...

    if self.sharding_owner == self.dcp_rank:
        # ===== Owner：写 cache + 真正计算 =====
        kv_cache = self.k_cache.kv_cache[forward_context.virtual_engine]
        if kv_cache is not None:
            # ① scatter：把当前 chunk 的 K 写入 paged cache
            #   slot_mapping 指定每个 token 写到 cache 的哪一行
            #   -1 的位置跳过不写
            torch_npu.npu_scatter_nd_update_(
                kv_cache.reshape(-1, k.shape[-1]),  # cache 拉平 [总token数, 128]
                slot_mapping,                        # [2048, 1] 写入位置
                k.reshape(-1, k.shape[-1])           # [2048, 128] 当前 K
            )

        # ② 真正的 Indexer 计算
        topk_indices, _ = torch_npu.mlp_lightning_indexer(
            query=q,                                   # [2048, 64, 128]
            key=kv_cache,                              # [num_blocks, 1024, 1, 128] 整个 pool
            weights=weights,                           # [2048, 64]
            cur_seq_lengths_query=attn_metadata.cum_query_lens,  # [0, 2048]
            cur_seq_lengths_key=attn_metadata.cum_seq_lens,      # [0, 35000]
            block_table=attn_metadata.block_table,     # [1, 256] 页表
            layout_query="TND",
            layout_key="PA_BSND",                      # 告诉算子 key 是 paged 格式
            sparse_count=self.topk_tokens,             # 2048
            init_num=self.num_init,                    # 强制选入的头部 token 数
            local_num=self.local_tokens,               # 强制选入的最近 token 数
        )
    else:
        # ===== Non-owner：dummy 空跑 =====
        # 用当前 chunk 的 k 代替 cache，cum_lens 造假
        # 目的只是让 NPU 编译器看到相同的算子图拓扑
        topk_indices, _ = torch_npu.mlp_lightning_indexer(
            query=q,
            key=k,                                     # 不是 cache，只是当前 chunk 的 k
            weights=weights,
            cur_seq_lengths_query=torch.zeros(...),     # 造假的 cum_lens
            cur_seq_lengths_key=torch.arange(...),
            block_table=None,                           # 无 block_table
            layout_query="TND",
            layout_key="TND",                           # 普通连续格式，不是 paged
            sparse_count=self.topk_tokens,
            init_num=self.num_init,
            local_num=self.local_tokens,
        )

    # ③ Owner 广播 topk 给所有 rank
    topk_indices = self.dcp_group.broadcast(topk_indices, src=self.sharding_owner)

    # ④ 全局索引 → 本 rank 的 DCP local 索引
    topk_indices = self._compute_local_indices(topk_indices)
    self.topk_indices_buffer[:topk_indices.size(0)].copy_(topk_indices)
    return topk_indices
```

### 瓶颈分析

`mlp_lightning_indexer` 的计算量 ∝ S_q × S_kv。长序列下（S_kv=35000），单个 rank 算 `2048×35000` 需要 ~13ms。这个时间其他 7 个 rank 都在等 broadcast，产生严重的**负载不均衡**。

---

## 三、为什么可以优化

### 核心发现：Query 独立性

`mlp_lightning_indexer` 对每个 query token **独立**选 TopK——token i 的选择只取决于 `q[i]`、`weights[i]` 和它能看到的 KV range，和其他 token 无关。

这意味着可以**按 query 维度切分并行**：把 2048 个 query 分给 8 个 rank 各算 256 个，结果拼起来和单 rank 算 2048 个完全一样。

### 为什么不按 KV 维度切

每个 token 的 TopK 需要看**完整 KV 范围**——init tokens（头部）+ local tokens（最近的 1024 个）+ sparse global（全局竞争）。如果按 KV 切，每个 rank 只有部分 KV，没法做正确的全局 TopK。

### 前提：每个 rank 都需要完整 KV

按 query 切没问题，但每个 rank 的 256 个 query 都要和**全部 35000 个 KV token** 计算 attention score。原来只有 owner 有完整 cache，所以需要 owner 把 cache 发出去。

**关键权衡：拿通信换计算**

```
多出来的通信:  broadcast cache ~0.5ms
省下来的计算:  13ms → 1.6ms = 省 11.4ms
净收益:        ~11ms/层
```

---

## 四、优化后：DCP 并行 Indexer

### 执行流程

以 `Sq=2048, Skv=35000, DCP=8, cache_block_size=1024` 为例：

```
预计算:
  skv      = cum_seq_lens[-1] = 35000       ← 总 KV token 数
  prefix   = skv - Sq = 35000 - 2048 = 32952  ← 历史 KV 数（不含当前 chunk）
  num_used = ceil(35000/1024) = 35           ← Indexer cache 中实际用到的 block 数
  chunk    = Sq / dcp_size = 2048/8 = 256    ← 每 rank 处理的 query 数

┌─────────────────────────────────────┐   ┌─────────────────────────────────────┐
│       DCP rank 0 (owner)            │   │     DCP rank r (r=1..7)             │
├─────────────────────────────────────┤   ├─────────────────────────────────────┤
│ ① scatter k → kv_cache             │   │ (skip)                              │
├─────────────────────────────────────┤   ├─────────────────────────────────────┤
│ ② gather used blocks → compact buf │   │ ② 复用 _broadcast_cache_buf         │
│   从 paged pool 中按 block_table    │   │   （类变量，跨层共享，按需增长）      │
│   取出 35 个 block → 连续 tensor    │   │                                     │
│   → [35, 1024, 1, 128] = 8.75MB    │   │   → [35, 1024, 1, 128]             │
├─────────────────────────────────────┤   ├─────────────────────────────────────┤
│ ③ broadcast(cache_buf, src=owner)   │   │ ③ recv broadcast                    │
│   8.75MB ⏱️ ~0.5ms                 │   │   ⏱️ ~0.5ms                         │
├─────────────────────────────────────┤   ├─────────────────────────────────────┤
│ ④ 构建 local_block_table           │   │ ④ 同左                              │
│   compact buffer 中 block 按逻辑    │   │                                     │
│   顺序排列，页表 = arange(35)       │   │                                     │
│   → [1, 35]                         │   │   → [1, 35]                         │
├─────────────────────────────────────┤   ├─────────────────────────────────────┤
│ ⑤ parallel indexer                  │   │ ⑤ parallel indexer                  │
│   q_local  = q[0:256]              │   │   q_local  = q[r*256:(r+1)*256]    │
│   w_local  = weights[0:256]        │   │   w_local  = weights[...]           │
│   kv_upper = 32952 + 256 = 33208   │   │   kv_upper = 32952 + (r+1)*256    │
│   → local_topk [256, 1, 2048]      │   │   → local_topk [256, 1, 2048]     │
│   ⏱️ ~1.6ms                        │   │   ⏱️ ~1.6ms                        │
├─────────────────────────────────────┤   ├─────────────────────────────────────┤
│ ⑥ all_gather(local_topk, dim=0)    │   │ ⑥ all_gather(local_topk, dim=0)    │
│   8×[256,1,2048] → [2048,1,2048]   │   │   → [2048, 1, 2048]               │
│   16MB ⏱️ ~0.04ms                  │   │   ⏱️ ~0.04ms                       │
└─────────────────────────────────────┘   └─────────────────────────────────────┘

后续（所有 rank）:
  topk = _compute_local_indices(topk_full)  ← 全局索引 → DCP local 索引
  topk_indices_buffer[:2048].copy_(topk)

总耗时: ~0.5 + ~1.6 + ~0.04 ≈ 2.3ms/层
原耗时: ~13ms/层 → 加速 ~5.7x
38层: 原 ~490ms → 新 ~87ms，节省 ~400ms
```

### compact buffer 与 block_table 详解

**为什么需要 compact buffer？**

kv_cache pool 很大（比如 512 个 block = 128MB），但这个序列只用了其中 35 个，且分散在 pool 各处。不能直接 broadcast 整个 pool——太大且大部分是无关数据。

**gather 做了什么？**

```
kv_cache pool: [512, 1024, 1, 128]
block_table = [5, 102, 200, 37, 88, ..., 431]   ← 35 个物理 block 号

gather 后:
  cache_buf[0]  = kv_cache[5]     ← token 0~1023
  cache_buf[1]  = kv_cache[102]   ← token 1024~2047
  cache_buf[2]  = kv_cache[200]   ← token 2048~3071
  ...
  cache_buf[34] = kv_cache[431]   ← token 34816~34999

→ cache_buf: [35, 1024, 1, 128]  ← 只有 8.75MB
```

**为什么 block_table 变成 arange？**

原来 `block_table = [5, 102, 200, ...]` 是因为物理 block 散落在 pool 各处，需要页表翻译。gather 之后，逻辑 block i 的数据就在 `cache_buf[i]`，不需要翻译了。但算子接口固定要做 `物理block = block_table[0, 逻辑block]` 这步查表，所以传 `arange(35)` 让查表结果 = 输入值：

```
找 token 2500:
  逻辑 block = 2500 // 1024 = 2
  offset     = 2500 % 1024 = 452
  物理 block = local_block_table[0, 2] = 2   ← arange，恒等映射
  → cache_buf[2][452]                         ← 和原来 kv_cache[200][452] 是同一份数据
```

### kv_upper 与因果掩码

每个 rank 处理不同位置的 query，因果掩码上界不同：

```
rank 0: q[0:256],    kv_upper = 32952 + 256  = 33208
rank 1: q[256:512],  kv_upper = 32952 + 512  = 33464
...
rank 7: q[1792:2048], kv_upper = 32952 + 2048 = 35000
```

含义：rank r 最后一个 query token（全局位置 `(r+1)*chunk - 1`）最多能看到 `kv_upper` 个 KV token。算子内部根据每个 query 的位置自动做因果裁剪——前面的 query 看到的 KV 更少，后面的看到更多。

### `_compute_local_indices`：全局索引 → DCP local 索引

AllGather 拿到的 topk 索引是**全局位置**（0~34999），但下游 `mlp_sparse_flash_attention` 读的是**DCP 分片后的 MLA KV Cache**（每个 rank 只有 1/8 的 token），需要地址转换：

```
全局 token 0  → 0 % 8 == 0 (属于 rank0) → local = 0 // 8 = 0   ✅
全局 token 1  → 1 % 8 == 1 (不属于 rank0) → -1                   ❌
全局 token 8  → 8 % 8 == 0 (属于 rank0) → local = 8 // 8 = 1   ✅
全局 token 16 → 16 % 8 == 0 (属于 rank0) → local = 16 // 8 = 2  ✅

rank 0 转换后: [0, 1, 2, ..., -1, -1, -1, ...]
               有效的排前面         无效的排后面
```

代码实现（用 float32 避免 NPU 上 int32 整除走 AICPU）：

```python
def _compute_local_indices(self, topk_indices):
    indices_f = topk_indices.float()
    inv_dcp = 1.0 / self.dcp_size                         # 0.125，float32 精确可表示
    quotient = (indices_f * inv_dcp).to(torch.int32)       # 等价于 // 8
    remainder = indices_f - quotient.float() * self.dcp_size  # 等价于 % 8
    keep = (topk_indices != -1) & (remainder == float(self.dcp_rank))
    topk_indices = torch.where(keep, quotient, -1)         # 保留本 rank 的，转成 local 地址

    # 排序：有效索引排前面，-1 排后面
    _, order = torch.sort((topk_indices == -1).float(), dim=-1)
    return torch.gather(topk_indices, -1, order)
```

---

## 五、完整代码改动与注释

### 改动概览

commit `347f0b6`，4 个文件，+113/-28 行：

| 文件 | 改动 |
|------|------|
| `config.py` | +3 行：新增 `enable_dcp_parallel_indexer` 开关 |
| `arg_utils.py` | +2 行：命令行参数透传 |
| `common/config.py` | +1 行：日志打印 |
| `mt_flash_moe.py` | 核心改动（+107/-28） |

### 开关配置

```python
# config.py
enable_dcp_parallel_indexer: bool = False
"""Enable DCP parallel indexer for chunked prefill acceleration."""

# 启动命令加 --enable-dcp-parallel-indexer
```

### mt_flash_moe.py 完整改动

#### 改动 1：新增类变量

```python
class Indexer(nn.Module):
    topk_indices_buffer: Optional[torch.Tensor] = None
    _broadcast_cache_buf: Optional[torch.Tensor] = None  # Non-owner 跨层复用的接收 buffer
```

Non-owner 接收 broadcast 需要一个 buffer。用**类变量**是因为 38 层共 38 个 Indexer 实例，但 buffer 只需一个——上一层用完下一层直接覆盖。

#### 改动 2：`__init__` 读配置

```python
# DCP parallel indexer config
self.enable_dcp_parallel_indexer = vllm_config.parallel_config.enable_dcp_parallel_indexer
self.cache_block_size = vllm_config.cache_config.block_size * self.dcp_size  # 128 * 8 = 1024
```

`cache_block_size` 是 Indexer K Cache 每个 block 存的 token 数，后面算 `num_used`（用了多少个 block）时要用。

#### 改动 3：新增 `_parallel_indexer` 方法

```python
def _parallel_indexer(self, q, weights, kv_cache, Sq, prefix_val, block_table):
    """每个 DCP rank 处理 1/dcp_size 的 query tokens"""

    # ---- 切分 query ----
    chunk = Sq // self.dcp_size                    # 2048 // 8 = 256
    my_start = self.dcp_rank * chunk               # rank 3 → 768
    my_end = my_start + chunk                      # rank 3 → 1024

    q_local = q[my_start:my_end].contiguous()      # [256, 64, 128]
    w_local = weights[my_start:my_end].contiguous() # [256, 64]
    # .contiguous() 因为 slice 可能不连续，NPU 算子要求连续内存

    # ---- 因果掩码上界 ----
    kv_upper = prefix_val + my_end
    # rank 3 最后一个 token（全局位置 1023）能看到 KV[0 : 32952+1024 = 33976]

    # ---- 构造 cum_lens（只有 1 个序列）----
    local_cum_query = torch.tensor([0, chunk], dtype=torch.int32, device=q.device)
    local_cum_key = torch.tensor([0, kv_upper], dtype=torch.int32, device=q.device)
    # cum_query=[0, 256] 表示 "1 个序列，256 个 query"
    # cum_key=[0, 33976] 表示 "对应 33976 个 KV token"

    # ---- 调用 indexer 算子 ----
    local_topk, _ = torch_npu.mlp_lightning_indexer(
        query=q_local,                    # [256, 64, 128] 本 rank 的 query 分片
        key=kv_cache,                     # [35, 1024, 1, 128] broadcast 过来的 compact cache
        weights=w_local,                  # [256, 64]
        cur_seq_lengths_query=local_cum_query,
        cur_seq_lengths_key=local_cum_key,
        block_table=block_table,          # [1, 35] arange 顺序页表
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=self.topk_tokens,    # 2048
        init_num=self.num_init,
        local_num=self.local_tokens)

    # ---- AllGather：8 × [256, 1, 2048] → [2048, 1, 2048] ----
    return self.dcp_group.all_gather(local_topk, dim=0)
```

#### 改动 4：forward 重构

```python
def forward(self, hidden_states, qr, rotary_emb):
    # ... 投影 Q、K、weights（和原来一样，略）...

    # ========= 写 cache（只有 owner 执行）=========
    kv_cache = None                        # ← 新增：提前初始化，后面并行分支需要判断
    if self.sharding_owner == self.dcp_rank:
        kv_cache = self.k_cache.kv_cache[forward_context.virtual_engine]
        if kv_cache is not None:
            torch_npu.npu_scatter_nd_update_(
                kv_cache.reshape(-1, k.shape[-1]),
                slot_mapping,
                k.reshape(-1, k.shape[-1])
            )

    # ========= 路径选择 =========
    Sq = q.shape[0]
    use_parallel = (
        self.enable_dcp_parallel_indexer        # 开关打开
        and self.dcp_size > 1                    # DCP 启用
        and not torch.compiler.is_compiling()    # 非图编译（并行路径有 .item() 等动态操作）
        and len(attn_metadata.cum_query_lens) - 1 == 1  # 只有 1 个序列
            # cum_query_lens = [0, 2048] → len-1=1 个序列
            # cum_query_lens = [0, 1024, 3072] → len-1=2 个序列
            # batch > 1 时 block_table 逻辑更复杂，暂不支持
        and Sq >= self.dcp_size                  # query 够切
        and Sq % self.dcp_size == 0              # 能整除
    )

    if use_parallel:
        # ========= 并行路径 =========

        # ---- 预计算 ----
        skv = attn_metadata.cum_seq_lens[-1].item()    # 35000（D2H 同步，取到 CPU）
        prefix_val = skv - Sq                           # 32952（历史 KV 数）
        num_used = (skv + self.cache_block_size - 1) // self.cache_block_size  # ceil(35000/1024)=35

        # ---- Owner gather → compact buffer / Non-owner 准备接收 buffer ----
        if self.sharding_owner == self.dcp_rank:
            # Owner: 从 paged pool 中按 block_table 取出用到的 35 个 block
            used_indices = attn_metadata.block_table[0, :num_used].long()
            cache_buf = kv_cache[used_indices].contiguous()
            # kv_cache pool: [512, 1024, 1, 128] → 取出 [35, 1024, 1, 128] = 8.75MB
            # .contiguous() 产生临时拷贝，函数结束释放
        else:
            # Non-owner: 复用类变量 buffer（跨层共享，不释放）
            buf = Indexer._broadcast_cache_buf
            if buf is None or buf.shape[0] < num_used:
                # 首次 or 序列增长导致 block 数增加 → 重新分配
                buf = torch.empty(
                    num_used, self.cache_block_size, 1, self.head_dim,
                    dtype=torch.bfloat16, device=q.device)
                Indexer._broadcast_cache_buf = buf
            cache_buf = buf[:num_used]    # slice view，不 copy

        # ---- Broadcast compact cache ----
        torch.distributed.broadcast(
            cache_buf,
            src=self.dcp_group.ranks[self.sharding_owner],  # owner 在通信组中的全局 rank
            group=self.dcp_group.device_group)
        # 之后所有 rank 的 cache_buf 内容一样

        # ---- 构建顺序页表 ----
        # compact buffer 中 block 按逻辑顺序排列（gather 保证）
        # 所以页表 = [0, 1, 2, ..., 34]，恒等映射
        local_block_table = torch.arange(
            num_used, dtype=torch.int32, device=q.device).unsqueeze(0)  # [1, 35]

        # ---- 并行计算 ----
        topk_indices = self._parallel_indexer(
            q, weights, cache_buf, Sq, prefix_val, local_block_table)

    else:
        # ========= 原路径（不变）=========
        if self.sharding_owner == self.dcp_rank:
            topk_indices, _ = torch_npu.mlp_lightning_indexer(
                query=q, key=kv_cache, weights=weights,
                cur_seq_lengths_query=attn_metadata.cum_query_lens,
                cur_seq_lengths_key=attn_metadata.cum_seq_lens,
                block_table=attn_metadata.block_table,
                layout_query="TND", layout_key="PA_BSND",
                sparse_count=self.topk_tokens,
                init_num=self.num_init, local_num=self.local_tokens,
            )
        else:
            # dummy 空跑
            topk_indices, _ = torch_npu.mlp_lightning_indexer(
                query=q, key=k, weights=weights,
                cur_seq_lengths_query=torch.zeros(...),
                cur_seq_lengths_key=torch.arange(...),
                block_table=None,
                layout_query="TND", layout_key="TND",
                sparse_count=self.topk_tokens,
                init_num=self.num_init, local_num=self.local_tokens,
            )
        topk_indices = self.dcp_group.broadcast(topk_indices, src=self.sharding_owner)

    # ========= 共同收尾（两条路径都走）=========
    topk_indices = self._compute_local_indices(topk_indices)
    self.topk_indices_buffer[:topk_indices.size(0)].copy_(topk_indices)
    return topk_indices
```

**注意**：并行路径不需要 `broadcast(topk)`（AllGather 已经让每个 rank 都有完整结果），原路径才需要。但 `_compute_local_indices` 两条路径都要做。

---

## 六、Bitwise 等价性

优化后的结果和优化前 **bit-exact 一致**，依赖三个保证：

### 1. Query 独立性

每个 query token 的 TopK 选择完全独立。切成 8 份各算 256 个，每个 token 的输入（q、weights、可见 KV 范围）不变，输出不变。AllGather 只在 dim=0 拼接，不做任何算术运算。

### 2. 紧凑 cache 是精确拷贝

```
原始寻址:  token i → kv_cache[block_table[0, i//1024]][i%1024]
紧凑寻址:  token i → cache_buf[i//1024][i%1024]
                    = kv_cache[block_table[0, i//1024]][i%1024]  ← gather 逐值复制保证
```

每个 BF16 值精确复制，没有类型转换或插值。

### 3. kv_upper 精确还原因果掩码

```
rank r 处理 q[r*chunk : (r+1)*chunk]
kv_upper = prefix + (r+1)*chunk
```

每个 rank 最后一个 token 能看到的 KV 上界，和原来 owner 上这个 token 的因果范围完全一致。

---

## 七、触发条件详解

```python
use_parallel = (
    self.enable_dcp_parallel_indexer        # ① 启动参数开关
    and self.dcp_size > 1                    # ② DCP 启用
    and not torch.compiler.is_compiling()    # ③ 非图编译
    and len(attn_metadata.cum_query_lens) - 1 == 1  # ④ 单序列
    and Sq >= self.dcp_size                  # ⑤ query 够切
    and Sq % self.dcp_size == 0              # ⑥ 能整除
)
```

| 条件 | 解释 |
|------|------|
| ① 开关 | 默认关闭，需显式启用 |
| ② DCP>1 | 纯 TP 无 DCP 时不需要此优化（没有 owner/non-owner 问题） |
| ③ 非图编译 | 并行路径有 `.item()`（D2H 同步）、动态 shape malloc 等，图编译不兼容 |
| ④ 单序列 | chunked prefill 通常只有 1 个序列（长 prompt 占满 token 预算）；多序列时 block_table 逻辑更复杂 |
| ⑤⑥ 可切分 | query 数必须能被 dcp_size 整除 |

**Decode 为什么不走这条路径**：Decode 时 `Sq=1 < dcp_size=8`，条件 ⑤ 不满足。而且 decode 入图后 `is_compiling()=True`，条件 ③ 也不满足。

---

## 八、通信与显存分析

### 通信量对比

| | 优化前 | 优化后 |
|---|---|---|
| 通信 1 | broadcast topk: `Sq×1×2048×4B` = 16MB | broadcast cache: `num_used×1024×128×2B` |
| 通信 2 | — | allgather topk: `Sq×1×2048×4B` = 16MB |
| 其他 rank 状态 | idle (dummy 空跑) | **有效计算** |

Skv=35k 时：broadcast cache ~9MB + allgather topk ~16MB = 25MB，vs 原来 broadcast topk 16MB。多了 ~9MB 通信，但把 7 个 idle rank 全部利用起来。

### 显存

| 项目 | 大小 | 生命周期 |
|------|------|---------|
| Indexer K Cache（owner） | ~16MB/层 × 4~5 层 ≈ 80MB | 持久化，序列结束释放 |
| `_broadcast_cache_buf`（non-owner） | ~9-32MB | 类变量，跨层复用，按需增长不缩小 |
| Owner `cache_buf = kv_cache[...].contiguous()` | ~9MB | 临时，函数结束释放 |

---

## 九、性能数据

配置：`tp=8, dcp=8, ep=16, block_size=128, cache_block_size=1024, index_n_heads=64, index_head_dim=128, index_topk=2048`

### 单层 Indexer 耗时

| Sq | prefix | Skv | 优化前 (单 owner) | 优化后 (1/8 chunk) | 加速比 |
|----|--------|-----|-------------------|-------------------|--------|
| 1024 | 4096 | 5120 | 0.987 ms | 0.155 ms | **6.36x** |
| 2048 | 4096 | 6144 | 2.111 ms | 0.266 ms | **7.95x** |
| 1024 | 16384 | 17408 | 3.298 ms | 0.444 ms | **7.43x** |
| 2048 | 16384 | 18432 | 6.729 ms | 0.842 ms | **7.99x** |
| 1024 | 32768 | 33792 | 6.379 ms | 0.839 ms | **7.60x** |
| 2048 | 32768 | 34816 | 12.891 ms | 1.615 ms | **7.98x** |

加上 broadcast cache 通信（~0.5ms），净加速约 **5-7x/层**。

### 端到端收益

38 层总计：原 ~490ms → 新 ~87ms，**节省 ~400ms**。

---

## 面试讲述框架

::: details 面试时怎么讲这个优化？（2-3 分钟版本）

**1. 背景（30s）**

"我们的 MoE-26B 模型用了 DeepSeek V3 的 DSA 稀疏注意力——每层有一个 Indexer 从全部 KV 中选出 TopK=2048 个最重要的 token，再只对这 2048 个做精确 attention。Indexer 本质是一个轻量级多头 attention：64 个 head 打分，加权融合后选 TopK。"

**2. 问题（30s）**

"模型跑在 DCP=8 的环境下，MLA 的 KV Cache 被分散到 8 张卡，但 Indexer 需要完整序列。原方案是只让 1 个 owner rank 持有完整 cache 做计算，其他 7 张卡 idle 等广播。长序列下（Skv=35k）Indexer 要 ~13ms/层，38 层就是 ~490ms，严重拖慢推理。"

**3. 优化方案（60s）**

"核心发现：每个 query token 的 TopK 选择是独立的，可以按 query 维度并行。我把 2048 个 query 切成 8 份，每个 rank 算 256 个。但前提是所有 rank 都要有 KV——所以 owner 先把 cache 中实际用到的 block 从 paged pool 里 gather 成紧凑 buffer（~9MB），broadcast 给所有 rank，然后各 rank 并行算自己的 query chunk，最后 AllGather 拼回完整 topk。"

"紧凑 cache 是逐 block 精确拷贝，每个 rank 对自己 chunk 里的每个 query 看到的 KV 范围通过 kv_upper 精确对齐因果掩码，所以结果 bitwise exact。"

**4. 结果（15s）**

"每层 Indexer 从 ~13ms 降到 ~1.6ms（计算）+ ~0.5ms（broadcast cache），加速约 5-7x。38 层共省约 400ms。"

:::

::: details 面试官可能追问的问题

**Q: 为什么 Indexer 不做 TP 切分？**

A: Indexer 的 TopK 是跨所有 64 个 head 加权求和后做的。如果按 head 切 TP，每个 rank 只有部分 head 的 attention score，算不出正确的全局加权和。而且 Indexer 计算量（64 head × 128 dim）远小于 MLA（128 head × 192 dim），切分收益有限。

**Q: 短序列（KV < 2048）怎么处理？**

A: Indexer 照常跑，结果等于全选。输出的 topk_indices 后面填 `-1`，下游 `mlp_sparse_flash_attention` 对 `-1` 跳过。短序列 Indexer 本身只需 ~1ms，不是瓶颈。而且下游算子必须接收 `sparse_indices` 参数，不能跳过换成普通 flash attention。

**Q: 为什么是 query 维度并行而不是 KV 维度？**

A: 每个 token 的 TopK 选择需要看完整 KV 范围（init tokens + local tokens + global），按 KV 切的话每个 rank 只有部分 KV，没法做正确的全局 TopK。按 query 切则天然独立——各 token 的 TopK 互不影响。

**Q: broadcast cache 通信量大吗？**

A: Skv=35k 时约 9MB（35 blocks × 1024 tokens × 128 dim × 2B），intra-node ~0.5ms。原方案的 broadcast topk 也要 16MB。总通信量略增但计算从 ~13ms 降到 ~1.6ms，净收益很大。

**Q: `_broadcast_cache_buf` 为什么用类变量？**

A: 跨层复用同一个 buffer，避免 38 层每层都 malloc。buffer 按需增长（当前 num_used 超过 buffer 容量时重新分配），不主动缩小。

**Q: 怎么保证 bitwise exact？**

A: 三点：(1) query 独立性——切 query 不改变任何单个 token 的计算；(2) 紧凑 cache 是逐值精确拷贝（gather + broadcast），没有类型转换；(3) kv_upper 精确还原每个 token 的因果掩码范围。AllGather 只做 dim=0 拼接，不涉及浮点运算。

**Q: 为什么 Decode 不走并行路径？**

A: Decode 时 `Sq=1`，小于 `dcp_size=8`，不够切分。而且 Decode 走图编译（CUDA/NPU Graph），并行路径有 `.item()` 等动态操作不兼容。

**Q: 为什么限制单序列？**

A: Chunked prefill 场景下，长 prompt 占满 token 预算，batch 基本都是 1。多序列时 block_table 是 `[B, max_blocks]` 多行，gather 和 broadcast 逻辑需要处理多序列的不同 block 范围，当前实现暂未支持。

:::
