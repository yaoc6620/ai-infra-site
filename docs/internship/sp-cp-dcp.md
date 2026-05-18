# SP / CP / DCP / PCP 原理对比

## 总览

四者都与"序列"相关，但解决的问题完全不同：

| | SP（Sequence Parallelism） | CP（Context Parallelism） | DCP（Decode Context Parallelism） | PCP（Prefill Context Parallelism） |
|---|---|---|---|---|
| 解决什么 | Activation 显存冗余 | Prefill 长序列算力分摊 | Decode 长序列 KV Cache 显存 | Prefill 长序列 KV Cache 显存 |
| 切什么 | Attention **外面**（LayerNorm、残差） | Attention **里面**（Q 和 KV） | KV Cache 的序列维度 | KV Cache 的序列维度 |
| 省计算 | 不省 | **省**（单卡 attention 降到 1/cp） | 不省 | 不省 |
| 省显存 | activation 显存 | KV Cache + activation | KV Cache 显存 | KV Cache 显存 |
| 通信方式 | AllGather + ReduceScatter | P2P ring send/recv | AllGather Q + All-to-All | AllGather KV Cache |
| 依赖关系 | 必须和 TP 配合 | 独立并行维度（需额外卡） | 复用 TP group | 复用 TP group |
| 典型场景 | 所有训练/推理（TP 的标配补丁） | 训练 / 长序列 prefill | 推理 decode 阶段 | 推理 prefill 阶段（chunked） |

---

## SP（Sequence Parallelism）

### 解决什么问题

TP 只切了矩阵乘法（QKV proj、MLP），但 **LayerNorm、Dropout、残差加**这些非并行区域，每个 rank 都持有完整的 `[S, H]` activation——白白浪费显存。

```
标准 TP：
  LayerNorm([S, H])  → 全量，每个 rank 都存  ← 浪费
  QKV proj → Attention → O proj → AllReduce
  LayerNorm([S, H])  → 又是全量              ← 浪费
```

SP 把这些非 TP 区域也按序列维度切开，**不减少任何计算量，只省 activation 显存**。

### 工作原理

核心改动：**把 AllReduce 拆成 ReduceScatter + AllGather**。

```
SP + TP 的一个 Transformer 层：

  输入: [S/tp, H]              ← 每个 rank 只存 S/tp 个 token 的 activation
    ↓ LayerNorm([S/tp, H])     ← SP 区域，activation 只有 1/tp
    ↓ AllGather               ← 恢复完整 [S, H] 进入 TP 区域
    ↓ QKV proj (ColumnParallel)
    ↓ Attention
    ↓ O proj (RowParallel)
    ↓ ReduceScatter            ← 替代 AllReduce，输出 [S/tp, H]
    ↓ 残差加 + LayerNorm        ← SP 区域，只有 1/tp
    ↓ AllGather
    ↓ MLP (Column + Row)
    ↓ ReduceScatter
  输出: [S/tp, H]
```

### 通信量不变

AllReduce = ReduceScatter + AllGather（就是这么实现的），所以 SP 不增加通信量。唯一的区别是把两步拆开了：

- **ReduceScatter**：求和 + 分片，输出只有 1/tp
- **AllGather**：拼接各 rank 的分片，恢复完整

```
AllReduce：  输入 [S, H] → 输出 [S, H]，每个 rank 都有完整结果
  = ReduceScatter + AllGather 合并执行

SP 拆开后：
  ReduceScatter: [S, H] → [S/tp, H]   ← 输出变小了！进入 SP 区域
  ...LayerNorm, 残差加...              ← activation 只有 [S/tp, H]
  AllGather:     [S/tp, H] → [S, H]   ← 恢复完整，进入 TP 区域
```

### 显存收益

SP 区域（LayerNorm、残差加、Dropout）的 activation 从 `[S, H]` 降到 `[S/tp, H]`。

以 TP=8, S=4096, H=5120, bf16 为例：
- 不用 SP：每层非 TP 区域 activation = `4096 × 5120 × 2B × 2`（LayerNorm 前后）≈ 80MB/层
- 用 SP：= `512 × 5120 × 2B × 2` ≈ 10MB/层，**省 8 倍**

38 层累计：80MB × 38 ≈ 3GB → 10MB × 38 ≈ 380MB。

---

## CP（Context Parallelism / Ring Attention）

### 解决什么问题

Prefill 阶段 attention 计算量为 $O(S^2 \cdot d)$，序列长到一定程度单卡算不完。CP 把序列分给多个 rank，**分摊 attention 计算**。

### 工作原理：Ring Attention

每个 rank 持有一部分 Q 和一部分 KV，通过**环形传递 KV 块**让每个 Q 看到所有 KV：

```
CP=4, S=8192, 每个 rank 分到 2048 个 position

rank 0: Q[0:2048]     KV[0:2048]
rank 1: Q[2048:4096]  KV[2048:4096]
rank 2: Q[4096:6144]  KV[4096:6144]
rank 3: Q[6144:8192]  KV[6144:8192]
```

Ring 传递流程：

```
Round 1: 每个 rank 用本地 KV 算 partial attention
         同时把本地 KV 发给下一个 rank（P2P send/recv）

Round 2: 收到上一个 rank 的 KV，算 partial attention
         同时继续传递

Round 3: 同上

Round 4: 所有 rank 的 Q 都看过了所有 KV
         用 Online Softmax 合并 4 轮的 partial attention
```

### 计算与通信 Overlap

Ring Attention 的精髓：**算当前 KV 块的 attention 和 接收下一个 KV 块是重叠的**。

```
时间线（rank 0）:
  ┌─ compute(local KV) ─┐┌─ compute(KV from rank 3) ─┐┌─ compute(KV from rank 2) ─┐...
  └─ recv(KV from rank 3)┘└─ recv(KV from rank 2)    ─┘└─ recv(KV from rank 1)    ─┘...
```

只要计算时间 ≥ 通信时间，通信就能被完全隐藏。Prefill 时 Q 有很多 token，`Q[2048] × KV[2048]` 的计算量足够大。

### Online Softmax 合并

每轮 Ring 产生一个局部 `(O_k, LSE_k)`，最终合并：

$$
O_{\text{global}} = \sum_k \frac{\exp(\text{LSE}_k)}{\sum_{k'} \exp(\text{LSE}_{k'})} \cdot O_k
$$

不需要存完整的 `[S, S]` attention score 矩阵，和 FlashAttention 的 tiling 思路一致。

### 每个 rank 的计算量和显存

| | 不用 CP | CP=4 |
|---|---|---|
| 每个 rank 的 Q | S | S/4 |
| 每个 rank 看到的 KV | S | S（通过 ring） |
| 单 rank attention 计算量 | $O(S^2 d)$ | $O(S^2 d / 4)$ |
| 总计算量 | $O(S^2 d)$ | $O(S^2 d)$（不变） |
| 单 rank KV Cache 显存 | $O(S)$ | $O(S/4)$（只长期存本地的） |

**总计算量不变，但分摊到多卡上，单卡更快。** 同时 KV Cache 显存也降到 1/cp_size。

### 为什么 CP 不适合 Decode

Decode 时 Q 只有 1 个 token（或 batch_size 个），计算/通信比极差：

```
Ring Attention (Prefill): Q=4096 tokens × KV_block → 计算量大，盖得住通信 ✓
Ring Attention (Decode):  Q=1 token × KV_block    → 一个 GEMV 几微秒，通信几百微秒 ✗
```

Ring 转一圈大部分时间在等 P2P 通信，不如用 DCP。

---

## DCP（Decode Context Parallelism）

### 解决什么问题

Decode 阶段 KV Cache 随序列增长线性增长，长序列（64K~128K）单卡存不下。DCP 把 KV Cache 按 token **round-robin 分片**到多个 rank，每个 rank 只存 1/dcp_size 的 KV。

### 与 CP 的关键区别：KV 不动

CP 通过 ring 传递 KV 块，DCP 的 KV 留在原地不动：

```
CP:  KV 在 ring 上传递 → 每个 Q 看到所有 KV
DCP: KV 不动 → 每个 rank 用完整 Q × 本地 KV → 合并
```

DCP 让每个 rank 先拿到完整 Q（AllGather），然后各自对本地 KV 分片做 FlashAttention，最后通过 All-to-All + LSE Rescale 合并。

### 工作流程

```
Step 1: Q AllGather
  各 rank 有 H/dcp 个 head 的 Q → AllGather(head 维度) → 每个 rank 有完整 H 个 head 的 Q

Step 2: 本地 FlashAttention
  Q[B, H, d] × 本地 KV 分片 → (local_out, local_lse)

Step 3: All-to-All（head ↔ rank）
  每个 rank 的 H 个 head 结果拆成 dcp_size 份，按 head 归属交换
  效果：每个 rank 收到所有 KV 分片上自己那 H/dcp 个 head 的结果

Step 4: LSE Rescale 合并
  dcp_size 份 (out, lse) → attention_update → 等价于完整 KV 上的 attention
```

### 通信量对比

| | CP (Ring) | DCP |
|---|---|---|
| 通信内容 | 传递 KV 块（大） | Q AllGather + attention 输出 All-to-All（小） |
| 通信轮次 | cp_size 轮 P2P | 1 次 AllGather + 1 次 All-to-All |
| Decode 通信量 | $O(S \cdot d)$（传所有 KV） | $O(B \cdot H \cdot d)$（传 Q 和 attention 输出） |

Decode 时 $B$ 很小（batch_size 个 token），$S$ 很大（历史序列长度）。所以 DCP 的通信量远小于 CP。

### Ares 的 DCP 实现特点

Ares 框架的 DCP 复用 TP group，不增加总卡数：

```python
# TP=8, DCP=8 → 同一个通信组
if tensor_model_parallel_size == decode_context_model_parallel_size:
    _DCP = _TP

# TP=8, DCP=4 → 把 TP group 切成 2 个 DCP 子组，每组 4 卡
```

同一组 8 张 NPU 同时承担两个角色：TP 切模型权重（head 维度），DCP 切 KV Cache（token 维度）。

配合 DSA（Dynamic Sparse Attention）时，DCP 还有额外优势：decode 时 Indexer 选出 TopK 重要 position，只需要对这些 position 做 sparse attention——**不需要看完整 KV，所以根本不需要 ring 传递 KV**，DCP 分片存储 + 本地 gather 就够了。

---

## PCP：Ares 中 SP 的实现

### PCP 是什么

PCP（Prefill Context Parallelism）是 Ares 框架中 SP 在 prefill 阶段的具体实现。配置名为 `prefill_context_parallel_size`，默认值为 1（不开启）。

```python
# config.py
prefill_context_parallel_size: int = 1    # PCP 默认关
decode_context_parallel_size: int = 1     # DCP 默认关
```

PCP 的本质就是 SP——把序列切给多个 rank，每个 rank 只处理一部分 token。在 Ares 中 PCP 复用 TP group 的卡（和 DCP 一样），不增加总卡数。

### PCP 不切计算，只切 KV Cache 存储

**关键澄清：PCP 不减少 attention 计算量。** TP=8 的 8 卡同时也是 PCP group 的成员，每卡有不同的 H/tp 个 head（TP 切了），不能再按序列切 Q——否则同一个 Q token 在不同卡上只有部分 head 的结果，无法通过 TP AllReduce 合并。

PCP 的作用和标准 SP 一样：**只切 KV Cache 的存储，不切 attention 计算。**

### 投影和 Attention 不变

所有 PCP rank 接收**相同的完整 input tokens**，做相同的 QKV 投影和 attention 计算：

```
TP=8, PCP=4 的 Prefill Attention 流程：

所有 8 卡收到相同的 hidden_states[S, H]
  → q_a_proj → q_b_proj → Q[S, H/tp, 192]    ← TP 切 head，S 不变
  → kv_a_proj → kv_c[S, 512]                  ← replicated
  → Attention: Q[S, H/tp] × KV[S, H/tp]      ← 每卡算完整 S 个 Q
  → o_proj + TP AllReduce                     ← 标准 TP 流程
```

**Attention 计算量 = S × S × H/tp，和不开 PCP 时一样。**

### PCP 省了什么：KV Cache 存储

PCP 的真正作用是在 Chunked Prefill 场景下，**将历史 KV Cache 分散存储**到多个 rank：

```
不开 PCP：每卡存完整的 KV Cache（S 个 token 的 kv_c + k_pe）
开 PCP=4：每卡只存 1/4 的 KV Cache（interleave 分配）
```

当后续 chunk 需要读历史 KV 时，PCP group 内 AllGather KV Cache 分片 → 恢复完整 KV → 做 attention → 用完就丢：

```
Chunked Prefill（chunk 1 开始）：
  读本 rank 的 KV Cache 分片 → AllGather（PCP group 内） → 完整历史 KV
  当前 chunk 的 KV：freshly projected（不需要 AllGather）
  Attention: Q × (历史 KV + 当前 KV)
```

AllGather 的是 **KV 压缩 latent**（512d + 64d = 576d），不是展开后的 KV（H × head_dim），通信量较小。

### KV Cache 的存储：交错分片

开了 PCP 后，PCP 和 DCP 被"压平"成一个统一的 CP 维度：

```python
total_cp_rank = pcp_rank * dcp_world_size + dcp_rank
total_cp_world_size = pcp_world_size * dcp_world_size
```

KV Cache 按 `cp_kv_cache_interleave_size`（记为 I）轮流分配给各 rank：

```
token 归属 = (position // I) % total_cp_world_size

示例：pcp=2, dcp=2, I=1（total_cp_world_size=4）:
  token 0 → rank 0 (pcp=0, dcp=0)
  token 1 → rank 1 (pcp=0, dcp=1)
  token 2 → rank 2 (pcp=1, dcp=0)
  token 3 → rank 3 (pcp=1, dcp=1)
  token 4 → rank 0 ...
```

写 KV Cache 时，每个 rank **只存属于自己的那些 position**，其他位置的 slot_mapping = -1（跳过）。

### Chunked Prefill 下的完整流程

长 prompt 被切成多个 chunk 逐个处理。以 PCP=2, DCP=2（total_cp=4）, chunk_size=2048 为例：

```
chunk 0（第一个 chunk，无历史 cache）：
  所有 4 个 rank 都处理完整的 token[0:2048]
    → 投影得到 Q[0:2048], KV[0:2048]（每卡都一样，除了 TP 切的 head）
    → Attention: Q × KV（当前 chunk 内 full attention）
    → 写 cache: 每个 rank 只存自己负责的 1/4 位置（interleave）
      rank 0: token 0, 4, 8, ...
      rank 1: token 1, 5, 9, ...
      rank 2: token 2, 6, 10, ...
      rank 3: token 3, 7, 11, ...

chunk 1（第二个 chunk，有 cache 了）：
  所有 rank 都处理完整的 token[2048:4096]
  需要看两部分 KV：
  ① 当前 chunk 的 KV：freshly projected，每卡都有完整的
  ② 历史 KV：从 cache 读取，每 rank 只有 1/4
     → AllGather（DCP group + PCP group）→ 恢复完整历史 KV
  Attention: Q[2048:4096] × (历史 KV_full + 当前 KV)
  写 cache: 同样只存本 rank 负责的 1/4
```

### 收益总结

PCP **不省 attention 计算量**（每卡还是 S × S × H/tp），但省了：

| 省什么 | 原因 |
|--------|------|
| KV Cache 显存 | 每卡只存 1/(pcp×dcp) 的历史 KV |
| Chunked Prefill 的 cache 读取显存 | 每卡只需 AllGather 自己的分片，临时 buffer 更小 |

代价是每层一次 AllGather KV Cache 通信（仅在 chunked prefill 读历史 KV 时）。

### PCP 和 TP 的关系

PCP 不能独立于 TP 减少 attention 计算。原因：

```
TP=8 的同一个 TP group 内：
  GPU 0: head 0-15    GPU 1: head 16-31   ...   GPU 7: head 112-127

如果按 PCP 切 Q（不同 rank 算不同 token）：
  GPU 0: Q[0:S/4] × KV × head 0-15   → 输出 [S/4, H]
  GPU 1: Q[S/4:S/2] × KV × head 16-31 → 输出 [S/4, H]

TP AllReduce 要求每卡的 output 含义相同（同一位置的 partial sum），
但这里不同 rank 是不同 token 的不同 head → 没法 AllReduce！
```

所以 PCP group 内所有 rank 必须处理**相同的 Q token**，PCP 只影响 KV Cache 存储。要真正切 attention 计算需要用 **CP（Ring Attention）**——CP 是独立并行维度，不复用 TP group 的卡。

---

## 四者的正交关系

SP、CP、DCP、PCP 解决不同层面的问题，可以同时使用：

```
一个 Transformer 层的数据流：

  [SP 区域] LayerNorm（每个 rank 只存 S/tp 的 activation）
     ↓ AllGather（恢复完整 S）
  [TP 区域] QKV proj → Attention → O proj
     ↓                    ↑
     ↓            Prefill: CP 分摊 Q×K 计算（Ring Attention）
     ↓            Prefill: PCP 分片 KV Cache 存储（AllGather 恢复）
     ↓            Decode:  DCP 分片 KV（AllGather Q + 本地 FA + 合并）
     ↓
     ↓ ReduceScatter（回到 S/tp）
  [SP 区域] 残差加 + LayerNorm
     ↓ AllGather
  [TP 区域] MLP
     ↓ ReduceScatter
  [SP 区域] 残差加
```

| 并行策略 | 作用层面 | 作用阶段 | 省什么 | 配合关系 |
|---------|---------|---------|-------|---------|
| SP | Attention 外（LayerNorm、残差） | Prefill + Decode | activation 显存 | TP 的补丁，不增加卡 |
| CP | Attention 内（Q × KV 计算） | Prefill | 计算 + 显存 | 独立维度，需额外卡 |
| DCP | KV Cache 存储 + Decode Attention | Decode | KV Cache 显存 | 复用 TP group |
| PCP | KV Cache 存储（写入/读取分片） | Prefill | KV Cache 显存 | 复用 TP group |

### 关键区分：省计算 vs 省显存

```
省计算（单卡 attention 工作量减少）：
  CP ✓ — 每卡只算 S/cp 个 Q 的 attention，需要额外卡

省显存（用通信换显存空间）：
  SP  ✓ — 省 activation（通信量不变）
  PCP ✓ — 省 KV Cache（多了 AllGather KV）
  DCP ✓ — 省 KV Cache（多了 All-to-All merge）
```

CP 是唯一能真正减少单卡 attention 计算量的方案，但需要独立的卡（不复用 TP group）。SP/PCP/DCP 都复用 TP group 的卡，只做存储优化，不省计算。

---

## TP/DCP 与 EP 的关系

### 不同维度，不同组卡

TP/DCP 和 EP 是**正交的并行维度**，作用在不同的卡组上：

```
典型配置：TP=8, DCP=8, EP=16, 总共 128 张卡（16 个节点，每节点 8 卡）

节点内（8 张 NPU，NVLink 互联）：
  TP group = 这 8 张卡（切 attention head / 权重）
  DCP group = 这 8 张卡（切 KV Cache）
  → 一个节点内是一个完整的 TP/DCP 组

节点间（16 个节点，网络互联）：
  EP = 16 → 每个节点是一个 EP rank
  → 16 个节点各持有不同的专家
  → All2All 跨节点交换 token 到对应专家所在的节点
```

### 为什么 EP 可以开很大，TP/DCP 不行

| | TP / DCP | EP |
|---|---|---|
| 切什么 | 同一层的权重（head）/ KV Cache | 不同专家分到不同节点 |
| 组内关系 | 同一组 8 张卡合力算一个 token | 不同组各算各的专家 |
| 通信模式 | AllReduce / AllGather（每层每 token 都通信） | All2All（MoE 层交换 token） |
| 通信频率 | 高频，每层多次 | 低频，只在 MoE 层 |
| 带宽需求 | NVLink（900GB/s） | 网络即可（100-400Gb/s） |
| 扩大的效果 | 单 token 算得更快 / 省显存 | 能装更多专家 |

TP/DCP 限制在节点内（8 卡）是因为**通信太频繁**——每层的 AllReduce、AllGather 都需要 NVLink 的高带宽。EP 可以跨节点是因为 All2All 只发生在 MoE 层，且可以和计算 overlap。

### MoE 模型的典型并行布局

```
                    ┌─────── EP=16（跨 16 个节点）────────┐
                    │                                     │
         ┌─ 节点 0 ─┐  ┌─ 节点 1 ─┐     ┌─ 节点 15 ─┐
         │TP=8/DCP=8│  │TP=8/DCP=8│ ... │TP=8/DCP=8 │
         │GPU 0-7   │  │GPU 8-15  │     │GPU 120-127│
         │Expert 0-7│  │Expert 8-15│    │Expert 120-127│
         └──────────┘  └──────────┘     └───────────┘

Dense 层（Attention）：
  节点内 TP AllReduce（NVLink，快）
  节点间无通信

MoE 层（MLP）：
  节点间 All2All（网络）：每个 token 路由到对应专家所在的节点
  节点内 TP AllReduce
```

Dense 层（Attention + LayerNorm）只在节点内通信（TP/DCP），MoE 层才需要跨节点通信（EP All2All）。所以 EP 开大不会影响 Dense 层的性能，只影响 MoE 层的 All2All 通信量。

---

## 面试讲述要点

::: details 面试时怎么区分这四个？

"SP 是 TP 的补丁，把 LayerNorm 和残差这些非并行区域也按序列切开，省 activation 显存，不省计算。通信上就是把 AllReduce 拆成 ReduceScatter + AllGather。

CP 是唯一能真正省 attention 计算的方案，用 Ring Attention 把 Q 分给不同 rank、KV 在 ring 上传递。每个 rank 只算 S/cp 个 Q 的 attention，但需要额外的卡（独立并行维度）。

DCP 是 decode 专用的 KV Cache 分片方案。decode 时 Q 只有 1 个 token，ring 的计算/通信比太差。DCP 让 KV 不动，每个 rank 拿完整 Q 算本地 KV 的 attention，最后 All-to-All + LSE rescale 合并。不省计算，省 KV Cache 显存。

PCP 是 prefill 的 KV Cache 分片方案（Ares 实现）。和 DCP 类似，复用 TP group 的卡，只切 KV Cache 存储不切计算。Chunked Prefill 读历史 KV 时需要 AllGather 恢复完整。用通信换显存，不省计算。"

:::


::: details 面试官可能追问

**Q: SP 的 ReduceScatter + AllGather 和直接 AllReduce 通信量一样，那有什么好处？**

A: 通信量一样，但 ReduceScatter 输出是 1/tp 大小。这意味着 ReduceScatter 到下一个 AllGather 之间的所有操作（LayerNorm、残差加、Dropout）的 activation 都是 1/tp，省了显存。AllReduce 是一步到位，输出还是完整 `[S, H]`，中间没有省显存的机会。

**Q: CP 和 DCP 能同时用吗？**

A: 可以。Prefill 阶段用 CP 分摊 attention 计算，decode 阶段用 DCP 分片 KV Cache。两者作用在不同阶段，互不冲突。Ares 框架里的 `pcp_size`（prefill CP）和 `dcp_size`（decode CP）就是分别控制这两个的。

**Q: Ring Attention 为什么用 P2P 而不是 AllGather？**

A: AllGather 一次性把所有 KV 收集到每个 rank，显存需要存完整 KV → 又回到了不分片的状态。Ring 每轮只接收一个 KV 块，算完就扔，**始终只需要存 1 个 KV 块的额外显存**。用显存换通信轮次。

**Q: DCP 为什么不用 ring 而用 AllGather Q + 本地 FA？**

A: decode 时 Q 只有 1 个 token，传 Q（AllGather）比传 KV（ring）便宜得多。Q 的大小是 `B × H × d`（几十 KB），KV 的大小是 `S × d`（几十到几百 MB）。传小的 Q 一次，比传大的 KV 转一圈高效。

**Q: PCP（Ares 的 SP 实现）和标准 SP 有什么区别？**

A: 两者本质一样——都是只切 KV Cache 存储，不切 attention 计算。PCP 将 KV Cache 按 interleave 分配到 PCP group 内的各 rank，Chunked Prefill 读历史 KV 时通过 AllGather 恢复完整 KV。和标准 SP 一样，attention 计算量不变，省的是显存。

**Q: PCP 能不能减少 attention 计算？为什么不能？**

A: 不能。PCP group 复用 TP group 的卡，同一组内每卡持有不同的 H/tp 个 head（TP 切了）。如果按序列切 Q（不同 rank 算不同 token），TP AllReduce 无法合并——因为不同 rank 的 output 是不同 token 的 partial sum，没法直接相加。所以 PCP group 内所有 rank 必须处理相同的 Q token。

**Q: PCP 开了之后 KV Cache 怎么存？**

A: PCP 和 DCP 被压平成一个统一的 CP 维度：`total_cp_world_size = pcp_size × dcp_size`。KV Cache 按 interleave 粒度轮流分配——每个 rank 只存 1/total_cp_world_size 的 token。写 cache 时 slot_mapping 里只有属于本 rank 的位置有效，其他位置为 -1 跳过。

**Q: PCP 的 AllGather KV Cache 通信量大吗？**

A: AllGather 的是 KV 压缩 latent（576d × bf16 = 1152B/token），比 AllGather 原始 hidden（5120d）小约 9 倍。只有 Chunked Prefill 读历史 KV 时才需要 AllGather。第一个 chunk 没有历史 cache 所以不需要，后续 chunk 才需要。

**Q: PCP 既不省计算也不省投影，那开了有什么用？**

A: 省 KV Cache 显存。长序列（128k）的 KV Cache 很大，PCP=4 + DCP=8 总共切 32 份，每卡只存 1/32 的历史 KV。这让单卡能支持更长的序列或更大的 batch。代价是 Chunked Prefill 时每层多一次 AllGather。

:::
