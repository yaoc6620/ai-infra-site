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

在 DCP 环境下，MLA 的 KV Cache 被 interleave 分散到 8 个 rank，但 Indexer 需要完整序列做 TopK。解决方式是选一个 **owner rank** 持有完整的 Indexer K Cache：

```python
# Indexer.__init__
self.sharding_owner = Indexer._get_bucket(config.num_layers, self.dcp_size, layer_num)
if self.sharding_owner == self.dcp_rank:
    self.k_cache = DeepseekV32IndexerCache(...)   # 只有 owner 有 cache
```

Owner 按层号轮流分配（`_get_bucket`），保证 8 个 rank 分摊不同层的 cache 存储。

### 执行流程

```
┌──────────────────────────────────┐   ┌──────────────────────────────────┐
│       DCP rank 0 (owner)         │   │     DCP rank 1-7 (non-owner)     │
├──────────────────────────────────┤   ├──────────────────────────────────┤
│ ① scatter k → kv_cache          │   │ (skip — 无 cache)                │
│   slot_mapping: [2048, 1]        │   │                                  │
├──────────────────────────────────┤   ├──────────────────────────────────┤
│ ② mlp_lightning_indexer          │   │ ② dummy mlp_lightning_indexer     │
│   query:   [2048, 32, 128]       │   │   (保持图拓扑一致的空跑)          │
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

Non-owner 的 dummy indexer 是为了保持所有 rank 的计算图拓扑一致（NPU 编译器要求），实际不做有意义的计算。

### 瓶颈分析

`mlp_lightning_indexer` 的计算量 ∝ S_q × S_kv。长序列下（S_kv=35000），单个 rank 算 `2048×35000` 需要 ~13ms。这个时间其他 7 个 rank 都在等 broadcast，产生严重的负载不均衡。

---

## 三、优化后：DCP 并行 Indexer

### 核心思路

**query 维度切分**：每个 query token 独立选 TopK，query 维度天然可并行。把 2048 个 query token 切成 8 份，每个 rank 算 256 个，再 AllGather 拼回来。

要让所有 rank 都能算，需要每个 rank 都有完整 KV Cache —— Owner 先把用到的 block gather 出来 broadcast 给所有 rank。

### 执行流程

以 `Sq=2048, Skv=35000, DCP=8, cache_block_size=1024` 为例：

```
预计算:
  prefix = Skv - Sq = 32952        ← 历史 KV token 数
  num_used = ceil(35000/1024) = 35  ← 用到的 cache block 数
  chunk = 2048 / 8 = 256            ← 每 rank 处理的 query 数

┌────────────────────────────────────┐   ┌────────────────────────────────────┐
│       DCP rank 0 (owner)           │   │     DCP rank r (r=1..7)            │
├────────────────────────────────────┤   ├────────────────────────────────────┤
│ ① scatter k → kv_cache            │   │ (skip)                             │
├────────────────────────────────────┤   ├────────────────────────────────────┤
│ ② gather used blocks → compact buf│   │ ② 复用 _broadcast_cache_buf        │
│   kv_cache[used_indices]           │   │   (类变量，跨层共享)                │
│   → [35, 1024, 1, 128] = 8.75MB   │   │   → [35, 1024, 1, 128]            │
├────────────────────────────────────┤   ├────────────────────────────────────┤
│ ③ broadcast(cache_buf, src=owner)  │   │ ③ recv broadcast                   │
│   8.75MB ⏱️ ~0.5ms                │   │   ⏱️ ~0.5ms                        │
├────────────────────────────────────┤   ├────────────────────────────────────┤
│ ④ parallel indexer                 │   │ ④ parallel indexer                 │
│   q_local  = q[0:256]             │   │   q_local  = q[r*256:(r+1)*256]   │
│   w_local  = weights[0:256]       │   │   w_local  = weights[...]          │
│   kv_upper = prefix + 256         │   │   kv_upper = prefix + (r+1)*256   │
│   → local_topk [256, 1, 2048]     │   │   → local_topk [256, 1, 2048]     │
│   ⏱️ ~1.6ms                       │   │   ⏱️ ~1.6ms                       │
├────────────────────────────────────┤   ├────────────────────────────────────┤
│ ⑤ all_gather(local_topk, dim=0)   │   │ ⑤ all_gather(local_topk, dim=0)   │
│   8×[256,1,2048] → [2048,1,2048]  │   │   → [2048, 1, 2048]              │
│   16MB ⏱️ ~0.04ms                 │   │   ⏱️ ~0.04ms                      │
└────────────────────────────────────┘   └────────────────────────────────────┘

总耗时: ~0.5 + ~1.6 + ~0.04 ≈ 2.3ms/层
原耗时: ~13ms/层 → 加速 ~5.7x
```

### 关键代码

```python
def _parallel_indexer(self, q, weights, kv_cache, S, prefix_val, block_table):
    chunk = S // self.dcp_size
    my_start = self.dcp_rank * chunk
    my_end = my_start + chunk

    q_local = q[my_start:my_end].contiguous()        # [256, 64, 128]
    w_local = weights[my_start:my_end].contiguous()   # [256, 64]
    kv_upper = prefix_val + my_end                    # 因果掩码上界

    local_cum_query = torch.tensor([0, chunk], ...)
    local_cum_key = torch.tensor([0, kv_upper], ...)

    local_topk, _ = torch_npu.mlp_lightning_indexer(
        query=q_local, key=kv_cache, weights=w_local,
        cur_seq_lengths_query=local_cum_query,
        cur_seq_lengths_key=local_cum_key,
        block_table=block_table,
        layout_query="TND", layout_key="PA_BSND",
        sparse_count=self.topk_tokens,
        init_num=self.num_init, local_num=self.local_tokens)

    return self.dcp_group.all_gather(local_topk, dim=0)  # [S, 1, topk]
```

### 触发条件

```python
use_parallel = (
    self.dcp_size > 1                      # DCP 开启
    and not torch.compiler.is_compiling()  # 非图编译阶段
    and S >= self.dcp_size                  # query 可整除切分
    and S % self.dcp_size == 0
)
```

Decode 时 `S=1 < dcp_size`，不会进入并行路径。只有 chunked prefill（`S=256~2048`）才会触发。

---

## 四、Bitwise 等价性

优化后的结果和优化前 **bit-exact 一致**，依赖三个保证：

### 1. Query 独立性

`mlp_lightning_indexer` 对每个 query token **独立**计算 TopK。token i 的选择只取决于 `q[i]`、`weights[i]` 和它能看到的 KV range，和其他 token 无关。

把 2048 个 query 切成 8 份各 256 个，每份 token 的输入完全不变。AllGather 只在 dim=0 拼接，不做任何算术运算。

### 2. 紧凑 cache 是精确拷贝

```
原始寻址:  token i → kv_cache[block_table[0, i//bs]][i%bs]
紧凑寻址:  token i → cache_buf[i//bs][i%bs]
                    = kv_cache[block_table[0, i//bs]][i%bs]  ← gather 保证
```

Owner 做 `kv_cache[used_indices].contiguous()` 按 block_table 顺序把用到的 block 原样 gather 到连续 buffer，broadcast 给所有 rank。每个 BF16 值精确复制，没有类型转换或插值。

### 3. kv_upper 精确还原因果掩码

```
rank r 处理 q[r*chunk : (r+1)*chunk]
kv_upper = prefix + (r+1)*chunk
```

每个 rank 最后一个 token（全局位置 `(r+1)*chunk - 1`）能看到的 KV 上界是 `prefix + (r+1)*chunk`，和原来 owner 上的因果范围完全一致。

---

## 五、通信与显存分析

### 通信量对比

| | 优化前 | 优化后 |
|---|---|---|
| 通信 1 | broadcast topk: 16MB | broadcast cache: ~9MB (Skv=35k) |
| 通信 2 | — | allgather topk: 16MB |
| 其他 rank | idle (dummy 空跑) | **有效计算** |

优化后通信总量略增（多了 broadcast cache），但把原来 idle 的 7 个 rank 全部利用起来，计算时间从 ~13ms 降到 ~1.6ms。

### 显存

| 项目 | 大小 | 说明 |
|------|------|------|
| `_broadcast_cache_buf` | ~9-32MB | 类变量跨层复用，按需增长（不释放） |
| Owner `.contiguous()` | ~9MB | 临时拷贝，函数结束释放 |

---

## 六、性能数据

| Sq | prefix | Skv | 优化前 (单 owner) | 优化后 (1/8 chunk) | 加速比 |
|----|--------|-----|-------------------|-------------------|--------|
| 1024 | 4096 | 5120 | 0.987 ms | 0.155 ms | **6.36x** |
| 2048 | 4096 | 6144 | 2.111 ms | 0.266 ms | **7.95x** |
| 1024 | 16384 | 17408 | 3.298 ms | 0.444 ms | **7.43x** |
| 2048 | 16384 | 18432 | 6.729 ms | 0.842 ms | **7.99x** |
| 1024 | 32768 | 33792 | 6.379 ms | 0.839 ms | **7.60x** |
| 2048 | 32768 | 34816 | 12.891 ms | 1.615 ms | **7.98x** |

加上 broadcast cache 通信（~0.5ms），净加速约 5-7x/层，38 层共省 ~400ms。

---

## 面试讲述框架

::: details 面试时怎么讲这个优化？（2-3 分钟版本）

**1. 背景（30s）**

"我们的 MoE-26B 模型用了 DeepSeek V3 的 DSA 稀疏注意力——每层有一个 Indexer 从全部 KV 中选出 TopK=2048 个最重要的 token，再只对这 2048 个做精确 attention。Indexer 本质是一个轻量级多头 attention：64 个 head 打分，加权融合后选 TopK。"

**2. 问题（30s）**

"模型跑在 DCP=8 的环境下，MLA 的 KV Cache 被分散到 8 张卡，但 Indexer 需要完整序列。原方案是只让 1 个 owner rank 持有完整 cache 做计算，其他 7 张卡 idle 等广播。长序列下（Skv=35k）Indexer 要 ~13ms/层，38 层就是 ~490ms，严重拖慢推理。"

**3. 优化方案（60s）**

"核心发现：每个 query token 的 TopK 选择是独立的，可以按 query 维度并行。我把 2048 个 query 切成 8 份，每个 rank 算 256 个。但前提是所有 rank 都要有 KV——所以 owner 先把 cache 中实际用到的 block gather 成紧凑 buffer（~9MB），broadcast 给所有 rank，然后各 rank 并行算自己的 query chunk，最后 AllGather 拼回完整 topk。"

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

:::
