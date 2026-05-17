# SP / CP / DCP 原理对比

## 总览

三者都与"序列"相关，但解决的问题完全不同：

| | SP（Sequence Parallelism） | CP（Context Parallelism） | DCP（Decode Context Parallelism） |
|---|---|---|---|
| 解决什么 | Activation 显存冗余 | Prefill 长序列算力分摊 | Decode 长序列 KV Cache 显存 |
| 切什么 | Attention **外面**（LayerNorm、残差） | Attention **里面**（Q 和 KV） | KV Cache 的序列维度 |
| Attention 是否完整 | 是，每个 rank 算完整 attention | 否，每个 rank 只算部分 Q | 是（decode：每个 rank 用完整 Q × 本地 KV） |
| 通信方式 | AllGather + ReduceScatter | P2P ring send/recv | AllGather + All-to-All |
| 依赖关系 | 必须和 TP 配合 | 独立并行维度 | 复用 TP group |
| 典型场景 | 所有训练/推理（TP 的标配补丁） | 训练 / 长序列 prefill | 推理 decode 阶段 |

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

## 三者的正交关系

SP、CP、DCP 解决不同层面的问题，可以同时使用：

```
一个 Transformer 层的数据流：

  [SP 区域] LayerNorm（每个 rank 只存 S/tp 的 activation）
     ↓ AllGather（恢复完整 S）
  [TP 区域] QKV proj → Attention → O proj
     ↓                    ↑
     ↓            Prefill: CP 分摊 Q×K 计算（Ring Attention）
     ↓            Decode:  DCP 分片 KV（AllGather Q + 本地 FA + 合并）
     ↓
     ↓ ReduceScatter（回到 S/tp）
  [SP 区域] 残差加 + LayerNorm
     ↓ AllGather
  [TP 区域] MLP
     ↓ ReduceScatter
  [SP 区域] 残差加
```

| 并行策略 | 作用层面 | 作用阶段 | 配合关系 |
|---------|---------|---------|---------|
| SP | Attention 外（LayerNorm、残差） | Prefill + Decode | TP 的补丁 |
| CP | Attention 内（Q × KV 计算） | Prefill | 独立维度 |
| DCP | KV Cache 存储 + Decode Attention | Decode | 复用 TP group |

---

## 面试讲述要点

::: details 面试时怎么区分这三个？

"SP 是 TP 的补丁，把 LayerNorm 和残差这些非并行区域也按序列切开，省 activation 显存，不省计算。通信上就是把 AllReduce 拆成 ReduceScatter + AllGather。

CP 是 prefill 的长序列分摊方案，用 Ring Attention 把 Q 分给不同 rank、KV 在 ring 上传递。每个 rank 只算 S/cp 个 Q 的 attention，计算量降到 1/cp。

DCP 是 decode 专用的 KV Cache 分片方案。decode 时 Q 只有 1 个 token，ring 的计算/通信比太差。DCP 让 KV 不动，每个 rank 拿完整 Q 算本地 KV 的 attention，最后 All-to-All + LSE rescale 合并。"

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

:::
