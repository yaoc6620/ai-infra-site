# MLA 的 TP 切分与低秩压缩

## MLA 低秩压缩原理

MLA（Multi-head Latent Attention）的核心设计是**低秩压缩**：Q 和 KV 都走 "压缩→解压" 两步投影，中间经过一个低维的 latent 瓶颈层。

### Q 的压缩与解压

Q 不需要缓存，每步都重新算。压缩纯粹是为了**减少参数量**（低秩分解投影矩阵）：

```
直接投影（一个大矩阵）:
  W_q: [5120, 128 × 192] = [5120, 24576]  →  126M 参数

低秩分解（两个小矩阵）:
  W_qa: [5120, 1536]       →  7.9M 参数     ← q_a_proj（压缩）
  W_qb: [1536, 128 × 192]  →  37.7M 参数    ← q_b_proj（解压）
  合计                       →  45.6M 参数（节省 ~64%）
```

数学上 $W_q \approx W_{qb} \cdot W_{qa}$，一个大矩阵用两个小矩阵的乘积来近似。中间的 1536 维没有物理意义，纯粹是矩阵分解的瓶颈维度。这就是 LoRA 的思路，所以 DeepSeek 论文里管这个维度叫 `q_lora_rank`。

```python
# mt_flash_moe.py — DeepseekV2MLAAttention.__init__
# 压缩: hidden_size → q_lora_rank（Parameter，全量复制）
self.q_a_proj = Parameter(torch.empty(self.q_lora_rank, self.hidden_size, ...))
self.q_a_layernorm = RMSNorm(self.q_lora_rank, ...)

# 解压: q_lora_rank → num_heads * qk_head_dim（ColumnParallel，TP 切）
self.q_b_proj = ColumnParallelLinear(q_lora_rank, num_heads * qk_head_dim, ...)
```

### KV 的压缩与解压

KV 的压缩不仅**省参数**，还**省 KV Cache 显存**——缓存的是压缩后的 latent，不是展开后的多 head K/V：

```
标准 MHA 每 token 缓存:  128 heads × head_dim × 2 (K+V)  →  很大
MLA 每 token 缓存:      kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576  →  很小
```

KV 只在需要做 attention 计算时才通过 `kv_b_proj` 临时解压：

```python
# 压缩: hidden_size → kv_lora_rank + qk_rope_head_dim（Parameter，全量复制）
self.kv_a_proj_with_mqa = Parameter(torch.empty(
    self.kv_lora_rank + self.qk_rope_head_dim, self.hidden_size, ...))
self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, ...)

# 解压: kv_lora_rank → num_heads * (qk_nope + v_head)（ColumnParallel，TP 切）
self.kv_b_proj = ColumnParallelLinear(
    kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), ...)
```

### 两步投影的完整数据流

```
Q 路径:
  hidden_states [S, 5120]
    → q_a_proj (压缩):  [S, 5120] → [S, 1536]      ← latent q_c, 全量复制
    → q_a_layernorm
    → q_b_proj (解压):  [S, 1536] → [S, 16×192]     ← TP 列切后只有 16 head
    → view reshape:     [S, 16×192] → [S, 16, 192]  ← head 维度是 reshape 出来的

KV 路径:
  hidden_states [S, 5120]
    → kv_a_proj (压缩): [S, 5120] → [S, 576]        ← kv_c + k_pe, 全量复制
    → kv_a_layernorm
    → 写入 KV Cache（存压缩态 576 维）
    → kv_b_proj (解压): [S, 512] → [S, 16×(nope+v)]  ← 计算 attention 时按需解压
```

## TP 怎么切 MLA

### 核心问题：head 维度从哪来？

输入 `hidden_states` 是一个平坦向量 `[S, 5120]`，**没有 head 维度**。head 维度是投影之后 reshape 出来的：

```
hidden_states: [S, 5120]            ← 没有 head 维度

    ↓ q_a_proj (Parameter, 全量复制)

q_c: [S, 1536]                      ← 还是没有 head 维度

    ↓ q_b_proj (ColumnParallel, 只存 1/8 列)

q: [S, 3072]                        ← 输出维度 = 16 head × 192 dim

    ↓ view reshape

q: [S, 16, 192]                     ← head 维度 reshape 出来了
```

### TP 切分 = 切投影矩阵的列

**"TP 按 head 切 Q" 的实现方式就是 `ColumnParallelLinear` 列切权重**。不存在 "先算完整 128 个 head 的 Q 再分给各 rank" 这个过程——输入 `hidden_states` 每个 rank 一模一样，各 rank 用不同的权重列，输出自然就是不同 head 的 Q：

```
rank 0: W_qb [1536, 3072]   →  Q[head 0:16]
rank 1: W_qb [1536, 3072]   →  Q[head 16:32]
...
rank 7: W_qb [1536, 3072]   →  Q[head 112:128]
```

### 代码：MLA 的 TP 切分

```python
# mt_flash_moe.py — DeepseekV2MLAAttention.__init__

tp_size = get_tensor_model_parallel_world_size()     # 8
assert num_heads % tp_size == 0
self.num_local_heads = num_heads // tp_size           # 128 // 8 = 16

# Q 投影: ColumnParallel → 每 rank 只存 24576/8 = 3072 列
self.q_b_proj = ColumnParallelLinear(
    q_lora_rank,                         # 1536 (输入维度, 不切)
    num_heads * qk_head_dim,             # 128 * 192 = 24576 (输出维度, 按 tp 切)
    ...)

# KV 投影: ColumnParallel → 每 rank 只解压 16 个 head 的 K_nope/V
self.kv_b_proj = ColumnParallelLinear(
    kv_lora_rank,                        # 512 (输入维度, 不切)
    num_heads * (qk_nope_head_dim + v_head_dim),  # (输出维度, 按 tp 切)
    ...)

# O 投影: RowParallel → 切输入维度, 输出 AllReduce 求和
self.o_proj = RowParallelLinear(
    num_heads * v_head_dim,              # (输入维度, 按 tp 切)
    hidden_size,                         # 5120 (输出维度, 不切)
    ...)
```

### forward 中的实际执行

```python
# mt_flash_moe.py — forward_sparse

# 1. 压缩（全量复制，每个 rank 结果一样）
q_c = torch.matmul(hidden_states, self.q_a_proj)     # [S, 5120] → [S, 1536]
q_c = self.q_a_layernorm(q_c)

kv_c, k_pe = torch.matmul(hidden_states, self.kv_a_proj_with_mqa).split(
    [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
kv_c_normed = self.kv_a_layernorm(kv_c)              # [S, 512] 每个 rank 一样

# 2. Q 解压（ColumnParallel，各 rank 只算自己 16 个 head）
q = self.q_b_proj(q_c)[0]                            # [S, 1536] → [S, 3072]
q = q.view(-1, self.num_local_heads, self.qk_head_dim)  # [S, 16, 192]

# 3. Attention 计算（每个 rank 16 个 head × 完整 KV）
attn_out = self.mla_attn((q_nope, q_pe), kv_c_normed, k_pe, ...)

# 4. O 投影（RowParallel，输出 AllReduce）
output = self.o_proj(attn_out)                        # 内部做 AllReduce 求和
```

## Column+Row 配对：为什么需要 AllReduce

### 列切（ColumnParallel）

切**输出维度**，不切输入。输入一样，输出各不同：

```
完整 W: [1536, 24576]

rank 0: W[:, 0:3072]     →  Y0 = X × W0     ← head 0:16
rank 1: W[:, 3072:6144]  →  Y1 = X × W1     ← head 16:32
...
不需要通信
```

### 行切（RowParallel）

切**输入维度**，不切输出。输入各不同，输出是部分和：

```
完整 W_o: [24576, 5120]

rank 0: W_o[0:3072, :]   →  Y0 = X0 × W0   ← 部分和
rank 1: W_o[3072:6144, :] → Y1 = X1 × W1   ← 部分和
...
需要 AllReduce 求和
```

### 为什么 AllReduce 不能省

`o_proj` 的数学：

```
output = [attn_0, attn_1, ..., attn_7] × W_o

       = attn_0 × W_o_0 + attn_1 × W_o_1 + ... + attn_7 × W_o_7
         └── rank 0 ──┘   └── rank 1 ──┘         └── rank 7 ──┘
         部分和            部分和                    部分和
```

每个 rank 只有 16 个 head 的 `attn_out` 和对应的 `W_o` 行，算出来的是最终结果的**一项**。

下游（MLP 或下一层 Attention）的 `ColumnParallelLinear` 需要**完整**的 `hidden_states` 作为输入。如果跳过 AllReduce，输入只是部分和，结果就是错的：

```
正确:  output = (sum_0 + sum_1 + ... + sum_7) × W_col
                └──── 完整 hidden_states ────┘

错误:  output = sum_0 × W_col
                └─ 只有本 rank 的部分和，漏了其他 7 个 rank 的贡献
```

### 每一层的通信模式

```
Attention:  Column(Q/K/V proj) → Attention → Row(O proj) → AllReduce ①
                                                               ↓
MLP:        Column(gate/up)    → SiLU×mul  → Row(down)   → AllReduce ②
                                                               ↓
                                                          下一层输入
```

每一对 Column+Row 产生一次 AllReduce，每个 Transformer 层共 **2 次 AllReduce**。

## MLA 的 TP 切分特殊性

MLA 相比标准 MHA 的 TP，有一个关键区别：**压缩层（`a_proj`）是全量复制的，只有解压层（`b_proj`）做 TP 切分**。

| 投影 | 类型 | TP 切分 | 原因 |
|------|------|---------|------|
| `q_a_proj` | Parameter | **否**，全量复制 | 维度小（5120→1536），不值得切 |
| `q_b_proj` | ColumnParallel | **是**，列切 | 维度大（1536→24576），按 head 切 |
| `kv_a_proj` | Parameter | **否**，全量复制 | 维度小（5120→576），输出要存 cache |
| `kv_b_proj` | ColumnParallel | **是**，列切 | 维度大，按 head 切 |
| `o_proj` | RowParallel | **是**，行切 | 收回部分和，AllReduce |

这带来一个重要性质：**`q_c` 和 `kv_c_normed` 在所有 rank 上完全一样**。因为 `hidden_states` 一样（上一层 AllReduce 之后），`a_proj` 权重一样（全量复制），所以输出的 latent 也一样。

这正是 DCP 能工作的基础——写 KV Cache 时每个 rank 算出来的值相同，只需要通过 `slot_mapping` 决定谁存哪个 token 就行。

## 补充：Indexer 为什么不切 TP

Indexer 的所有权重都是 `ReplicatedLinear`（全量复制），不做 TP 切分：

```python
# Indexer.__init__
# no tensor parallel, just replicated
self.wq_b = ReplicatedLinear(q_lora_rank, head_dim * n_head, ...)    # Q: 64 head × 128 dim
self.wk = ReplicatedLinear(hidden_size, head_dim, ...)                # K: 1 × 128 dim
self.weights_proj = ReplicatedLinear(hidden_size, n_head, ...)        # weights: 64 维
```

每个 rank 算出来的 Indexer Q、K、weights 完全一样。不切 TP 的原因有两个：

**1. TopK 选择是跨所有 head 加权的，没法按 head 独立切**

```python
weights, _ = self.weights_proj(hidden_states)   # [S, 64] ← 每个 head 一个权重
```

`mlp_lightning_indexer` 内部用 64 个 head 的注意力分数乘以 `weights` 加权求和后做 TopK。如果按 head 切 TP，每个 rank 只有部分 head 的分数，没法得到全局加权和，就选不出正确的 TopK。

**2. 计算量小，切 TP 收益有限**

Indexer 只有 64 head × 128 dim，远小于 MLA 的 128 head × 192 dim，切分开销（通信）可能反而超过收益。

Indexer 拿到的 `q_c` 是 MLA 压缩后的 latent（所有 rank 一样），用自己的 `wq_b`（Replicated）解压成 64 个 head。MLA attention 用 `q_b_proj`（ColumnParallel）解压成 TP 切后的 16 个 head。**同一个 `q_c`，两套不同权重，各自解压成不同的 Q**。
