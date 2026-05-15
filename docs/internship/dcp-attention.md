# DCP Attention 原理与实现

## 什么是 DCP

DCP（Decode Context Parallelism）是一种将**单个请求的 KV Cache 分散到多张卡**上的并行策略。在 Decode 阶段，长序列的 KV Cache 可能超出单卡显存，DCP 通过将 KV Cache 交错切分（interleave sharding）到多个 rank，使每张卡只需存储约 $\frac{1}{\text{dcp\_size}}$ 的 KV Cache，从而支持更长的序列。

**核心思想**：每张卡各自对本地 KV 分片做 FlashAttention，得到局部 `(attn_output, softmax_lse)`，再通过 All-to-All + LSE Rescale 合并为全局等价结果。

## 为什么需要 DCP

### 显存瓶颈

以 MoE-26B（MLA，`kv_lora_rank=512`, `qk_rope_head_dim=64`）为例：

$$
\text{单 token KV} = (512 + 64) \times 2\text{B (bf16)} = 1152\text{B}
$$

$$
\text{64K 序列} = 65536 \times 1152 \approx 72\text{MB/layer} \times 38\text{层} \approx 2.7\text{GB}
$$

单卡 NPU 910C 显存有限，多个 batch 的长序列会迅速耗尽显存。DCP 将 KV Cache 分到 8 张卡，每卡只需约 340MB，显存压力大幅降低。

### Decode 阶段的特点

Decode 每步只产生 1 个新 token 的 Q，但需要和**整个历史序列**的 KV Cache 做 attention。这意味着：

- **计算量**：$O(1 \times S_{kv})$，和序列长度线性相关
- **显存占用**：KV Cache 随序列增长线性增长
- **天然可并行**：Attention 计算对 KV 维度可拆分，各分片独立计算后合并

## DCP 与 TP 的关系

DCP **复用 TP group 的 GPU**，不增加总卡数。代码中初始化逻辑：

```python
# parallel_state.py
if tensor_model_parallel_size == decode_context_model_parallel_size:
    _DCP = _TP                  # TP=8, DCP=8 → 同一个通信组
else:
    # TP=8, DCP=4 → 把 TP group 切成 2 个 DCP 子组
    group_ranks = all_ranks.reshape(-1, dcp_size)
    _DCP = init_model_parallel_group(group_ranks, ...)
```

config 注释：

```python
decode_context_parallel_size: int = 1
"""the world size does not change by dcp, it simply reuse the GPUs
of TP group, and tp_size needs to be divisible by dcp_size."""
```

在实际配置 `TP=8, DCP=8` 下，**DCP 和 TP 就是同一个通信组**。同一组 8 张 NPU 同时承担两个角色：

| 维度 | 作用 |
|------|------|
| TP | 模型权重按 head 切分，每 rank 持有 128/8=16 个 head |
| DCP | KV Cache 按 token 交错分配，每 rank 存 1/8 的 token |

### TP 怎么切：按 head 切模型权重

TP 在模型初始化时通过 `ColumnParallelLinear` / `RowParallelLinear` 自动按 head 维度切分权重：

```python
# mt_flash_moe.py — DeepseekV2MLAAttention.__init__
tp_size = get_tensor_model_parallel_world_size()        # 8
assert num_heads % tp_size == 0
self.num_local_heads = num_heads // tp_size              # 128 // 8 = 16

# Q 投影：ColumnParallel 按 head 切输出维度 → 每 rank 只算 16 个 head 的 Q
self.q_b_proj = ColumnParallelLinear(
    q_lora_rank, num_heads * qk_head_dim, ...)

# KV 投影：ColumnParallel → 每 rank 只算 16 个 head 的 K/V
self.kv_b_proj = ColumnParallelLinear(
    kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), ...)

# O 投影：RowParallel 切输入维度，输出做 AllReduce 求和
self.o_proj = RowParallelLinear(
    num_heads * v_head_dim, hidden_size, ...)
```

`ColumnParallelLinear` 自动把输出维度除以 `tp_size`，所以每个 rank 只持有 16 个 head 对应的权重。`RowParallelLinear` 的输出是部分和，之后做 TP AllReduce 加起来。

### DCP 怎么存：按 token interleave 存 KV Cache

DCP 在运行时通过 `slot_mapping` 决定每个 token 的 KV 写到哪个 rank：

```python
# block_table.py — compute_slot_mapping
total_cp_world_size = self.pcp_world_size * self.dcp_world_size
total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank

# "virtual block" = world_size * block_size，用于 interleave 计算
virtual_block_size = self.block_size * total_cp_world_size
virtual_block_offsets = positions % virtual_block_size

# token i 属于哪个 rank: (i % virtual_block_size) // interleave_size % world_size
mask = (
    virtual_block_offsets // self.cp_kv_cache_interleave_size
    % total_cp_world_size
    == total_cp_rank
)

# 计算本 rank 内的 local block offset
block_offsets = (
    virtual_block_offsets // (total_cp_world_size * self.cp_kv_cache_interleave_size)
    * self.cp_kv_cache_interleave_size
    + virtual_block_offsets % self.cp_kv_cache_interleave_size
)
slot_mapping = block_numbers * self.block_size + block_offsets

# 不属于本 rank 的 token → slot = -1，scatter 时跳过
self.slot_mapping_np[...] = np.where(mask, slot_mapping, -1)
```

写 cache 时 scatter 按 `slot_mapping` 写入，`-1` 的位置不写：

```python
# ares_mla_v1.py — forward 中写 cache
slot_mapping = attn_metadata.slot_mapping.reshape(-1, 1)
torch_npu.npu_scatter_nd_update_(kv_cache[0].reshape(...), slot_mapping, k_c_normed.reshape(...))
torch_npu.npu_scatter_nd_update_(kv_cache[1].reshape(...), slot_mapping, k_pe.reshape(...))
```

## KV Cache 交错分配

DCP 采用 **interleave** 方式分配 token 到各 rank：

```
token 序列:  t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  ...
rank 分配:    0   1   2   3   4   5   6   7   0   1   ...
            └─────────── dcp_size = 8 ──────────────┘
```

第 $i$ 个 token 分配到 rank $i \bmod \text{dcp\_size}$。interleave 保证各 rank 的 token 数量差异最多为 1（负载均衡）。

### KV Cache 写入：slot_mapping 过滤

写 KV Cache 时，所有 rank 都算出了 `kv_c_normed`（MLA `num_kv_heads=1`，各 rank 算出来一样），但通过 `slot_mapping` 过滤，只写属于自己的 token：

```python
# block_table.py — compute_slot_mapping
# 哪些 token 属于本 rank
mask = (
    virtual_block_offsets // cp_kv_cache_interleave_size
    % total_cp_world_size
    == total_cp_rank
)
# 不属于本 rank 的 token → slot_mapping = -1（不写入）
self.slot_mapping_np[...] = np.where(mask, slot_mapping, -1)
```

然后 scatter 按 slot_mapping 写入：

```python
# forward 中写 cache
slot_mapping = attn_metadata.slot_mapping.reshape(-1, 1)
torch_npu.npu_scatter_nd_update_(kv_cache[0].reshape(...), slot_mapping, k_c_normed.reshape(...))
torch_npu.npu_scatter_nd_update_(kv_cache[1].reshape(...), slot_mapping, k_pe.reshape(...))
```

### MLA KV Cache vs Indexer K Cache

MLA attention 和 Indexer 使用不同的 KV Cache：

| | MLA attention KV Cache | Indexer K Cache |
|---|---|---|
| 结构 | 两个 tensor (`kv_c` + `k_pe`) | 一个 tensor |
| shape | `(num_blocks, 128, 1, 512)` + `(num_blocks, 128, 1, 64)` | `(num_blocks, 1024, 1, 128)` |
| block_size | 128（原始值） | 128 × dcp_size = 1024 |
| DCP 分片 | **是**，每个 rank 只存 1/8 token | **否**，owner 存完整序列 |

Indexer 的 `block_size` 乘了 `dcp_size`：

```python
# DeepseekV32IndexerCache
self.factor = get_decode_context_model_parallel_world_size()  # dcp_size
def get_kv_cache_spec(self):
    return MLAAttentionSpec(
        block_size=self.cache_config.block_size * self.factor,  # 128 * 8 = 1024
        ...)
```

这样在相同的 block 分配逻辑下，Indexer 每个 block 存 8 倍 token，总容量等于完整序列。

### 代码：计算每个 rank 的本地序列长度

```python
# utils.py — get_cp_local_seq_lens
def get_cp_local_seq_lens(seq_lens, pcp_world_size=1, dcp_world_size=1,
                          cp_kv_cache_interleave_size=1):
    total_world_size = pcp_world_size * dcp_world_size
    # 每个 rank 至少分到的 token 数（整除部分）
    base = seq_lens // cp_kv_cache_interleave_size // total_world_size \
           * cp_kv_cache_interleave_size
    # 余数部分按 rank 顺序分配
    remainder = seq_lens - base * total_world_size
    remainder = torch.clip(
        remainder - rank_offsets * cp_kv_cache_interleave_size,
        0, cp_kv_cache_interleave_size,
    )
    return (base + remainder).reshape([-1, pcp_world_size, dcp_world_size])
```

Metadata builder 中用它计算本 rank 的 `local_seq_lens`：

```python
# ares_mla_cp.py — _build_decode_metadata
local_seq_lens_cpu = get_cp_local_seq_lens(
    seq_lens_cpu[:self._num_decodes],
    self.pcp_size, self.dcp_size,
    self.cp_kv_cache_interleave_size,
)[:, self.pcp_rank, self.dcp_rank]   # 取本 rank 的分片长度
```

## DCP Decode Attention 完整流程

整体流程分四步：Q AllGather → 本地 FA → All-to-All → LSE Rescale 合并。

```
Step 1: Q AllGather (head 维度)
    rank0 有 H/8 个 head 的 Q    ──┐
    rank1 有 H/8 个 head 的 Q    ──┤  AllGather(dim=head)
    ...                            ├──────────> 每个 rank 都有完整 H 个 head 的 Q
    rank7 有 H/8 个 head 的 Q    ──┘

Step 2: 本地 FlashAttention
    每个 rank 用完整 Q × 本地 KV Cache 分片 → (local_attn_out, local_lse)

Step 3: All-to-All (head 维度 ↔ rank 维度)
    每个 rank 的 H 个 head 的结果 → 拆成 8 份 → 各送给对应 rank
    效果：每个 rank 收集到所有 rank 上自己那 H/8 个 head 的结果

Step 4: LSE Rescale 合并
    每个 rank 对收到的 8 份 (out, lse) 做 attention_update
    数学等价于在完整 KV 上做 FlashAttention
```

### 为什么 Q 用 AllGather 而不是 AllReduce？

因为各 rank 的 Q 是**不同 head 的分片**，不是同一个 head 的部分和：

```
AllGather（拼接不同的东西）：
  rank 0: Q[head 0:16]   ─┐
  rank 1: Q[head 16:32]  ─┤  拼接 → Q[head 0:128]
  ...                      │
  rank 7: Q[head 112:128] ─┘

AllReduce（对同一个东西的部分和求和）：
  rank 0: partial_sum  ─┐
  rank 1: partial_sum  ─┤  求和 → total_sum
  ...                    │
  rank 7: partial_sum  ─┘
```

Q 在 TP 下是按 head 维度切的，各 rank 持有**不同**的 head，恢复完整 Q 就是 AllGather **拼接**，不是求和。AllReduce 用在 attention 之后的 `o_proj`——因为 `o_proj` 权重按列切分，各 rank 算出来的是最终输出的**部分和**，需要 AllReduce 加起来。

### Step 1：Q AllGather

```python
# ares_mla_cp.py
def reorg_decode_q(self, decode_q_nope, decode_q_pe):
    if self.dcp_size > 1:
        # 拼接 q_nope 和 q_pe: [B, H/dcp, D_nope+D_pe]
        decode_q_no_split = torch.cat([decode_q_nope, decode_q_pe], dim=-1)
        # AllGather on head 维度 (dim=1): [B, H/dcp, D] → [B, H, D]
        decode_q_no_split = get_dcp_group().all_gather(decode_q_no_split, 1)
        # 拆回 q_nope 和 q_pe
        decode_q_nope, decode_q_pe = decode_q_no_split.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    return decode_q_nope, decode_q_pe
```

### Step 2：本地 FlashAttention

用完整 Q（$H$ 个 head）和本地 KV Cache 分片做 incre FlashAttention，同时输出 `softmax_lse`：

```python
# ares_mla_cp.py — _forward_decode
num_heads = self.num_heads * self.dcp_size   # 16 * 8 = 128 (完整 head 数)
q_nope, q_pe = self.reorg_decode_q(q_nope, q_pe)

attn_output, softmax_lse = torch_npu.mlp_incre_flash_attention_with_lse(
    q_nope, kv_c,                    # Q: [B, H, D], KV: 本地分片
    layer._k_scale, kv_c,
    layer._v_scale,
    decode_metadata._actual_seq_lens_list,
    decode_metadata.seq_lens_list,   # 本地 local_seq_lens
    decode_metadata.block_table,     # 本地 block_table
    B, num_heads, self.num_kv_heads,
    self.kv_lora_rank + self.qk_rope_head_dim,
    self.kv_lora_rank, self.scale,
    block_size=block_size,
    q_rope=q_pe, k_rope=k_pe,
    mla_flag=True,
    high_precision_flag=True,
)
# attn_output: [B, H, D_kv]     — 本地注意力结果
# softmax_lse: [B, H, 1]        — 本地 log-sum-exp
```

### Step 3：All-to-All 交换

每个 rank 持有**所有 H 个 head** 在**本地 KV 分片**上的结果，需要重新按 head 分配回各 rank：

```python
# common_cp.py — _process_attn_out_lse
def _process_attn_out_lse(attn_output, softmax_lse, batch_seq_mask):
    # 拼接 output 和 lse: [B, H, D] + [B, H, 1] → [B, H, D+1]
    attn_out_lse = torch.cat([attn_output, softmax_lse], dim=-1)

    if dcp_size > 1:
        # permute: [B, H, D+1] → [H, D+1, B]  (All-to-All 在 head 维度切分)
        attn_out_lse = attn_out_lse.permute([1, 2, 0]).contiguous()
        attn_out_lse_all2all = torch.empty_like(attn_out_lse)
        # All-to-All: rank i 的第 j 块 head → rank j
        dist.all_to_all_single(attn_out_lse_all2all, attn_out_lse, group=dcp_group)
        attn_out_lse = attn_out_lse_all2all.permute([2, 0, 1])

    return attn_out_lse
```

**通信语义**：假设 `dcp_size=8, H=128`，每个 rank 有 128 个 head 的结果。All-to-All 将 128 个 head 平均切成 8 份（每份 16 个 head），第 $j$ 份发给 rank $j$。最终每个 rank 收到 8 个 rank 上**自己那 16 个 head** 的结果。

shape 没变（还是 `[B, 128, D+1]`），但语义变了：从 `[128 heads × 1 shard]` → `[8 shards × 16 heads]`。

### Step 4：LSE Rescale 合并

收到各 rank 的局部 `(output, lse)` 后，利用 LSE 做加权合并，数学等价于在完整 KV 上做 attention：

```python
# common_cp.py — _npu_attention_update
def _npu_attention_update(head_size, attn_out_lse):
    B_total, H_total, D_plus_1 = attn_out_lse.shape
    S = B_total // pcp_size
    H = H_total // dcp_size
    D = head_size

    # reshape: [PCP*S, DCP*H, D+1] → [PCP, DCP, S, H, D+1] → [N, S, H, D+1]
    x = attn_out_lse.view(pcp_size, S, dcp_size, H, D_plus_1)
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, S, H, D_plus_1)

    # 拆分 output 和 lse
    out_flat, lse_flat = torch.split(x, [D, 1], dim=-1)

    # LSE rescale 合并（数学等价于全局 softmax 加权）
    attn_out = torch_npu.mlp_attention_update(lse_list, out_list, False)
    return attn_out.view(-1, H, D)
```

**数学原理**：

对于第 $k$ 个分片的局部注意力输出 $O_k$ 和 $\text{LSE}_k = \log \sum_j \exp(s_{kj})$：

$$
O_{\text{global}} = \sum_k \frac{\exp(\text{LSE}_k)}{\sum_{k'} \exp(\text{LSE}_{k'})} \cdot O_k
$$

这就是 FlashAttention 的 Online Softmax 推广到多分片的情况：每个分片独立算出 $(O_k, \text{LSE}_k)$，合并时用 softmax 权重加权求和，结果和在完整序列上做 attention 严格等价。

### 完整数据流图

以 `dcp_size=8, num_heads=128, kv_lora_rank=512` 为例：

```
每个 rank 初始状态:
  Q: [B, 16, 576]          ← TP 切后 128/8=16 个 head, D=512+64
  KV Cache: 本地分片         ← ~1/8 的 token

Step 1: AllGather Q
  [B, 16, 576] → AllGather(dim=1) → [B, 128, 576]
  通信量: 16×576×2B × 8 ranks ≈ 很小（B=1 时仅 ~144KB）

Step 2: 本地 FA
  Q[B,128,576] × KV_local → (out[B,128,512], lse[B,128,1])

Step 3: All-to-All
  [B,128,513] → All-to-All(head dim) → [B,128,513]
  效果: rank i 收到 8 个 KV 分片上 head[16i:16(i+1)] 的结果

Step 4: LSE Merge
  8 份 (out[B,16,512], lse[B,16,1]) → attention_update → out[B,16,512]
  → V up proj + O proj → 最终输出
```

## Prefill 阶段与 DCP

### DCP 的收益只在 Decode

DCP 在 prefill 阶段**没有计算并行收益**。Prefill 阶段的 attention 计算就是标准 TP 行为：

- **Q**：每个 rank 只有自己 16 个 head 的 Q，直接用，不需要任何通信
- **KV**：MLA `num_kv_heads=1`，每个 rank 前向算出来一样的 `kv_c_normed`
- **Attention**：每个 rank 用 16 个 head 的 Q × 完整 KV → 16 个 head 的输出
- **O proj**：之后做标准 TP AllReduce

计算量和没开 DCP 完全一样，DCP 只影响 KV Cache 的存储方式。

### Chunked Prefill 下的 AllGather 开销

长 prompt 通过 chunked prefill 分多个 step 处理。`_forward_prefill` 有三条路径：

```python
def _forward_prefill(self, ...):
    if self.ring_attn_enabled:           # 路径 1: Ring Attention（默认关闭）
        return self._forward_prefill_ring(...)

    if chunked_context := ...:           # 路径 2: 有已缓存 context（chunk 1+）
        # 需要 AllGather 收回 DCP 分片的 KV
        ...

    # 路径 3: 无 context（第一个 chunk）→ 标准 causal FA
    ...
```

- **第一个 chunk**：`context_len=0`，走路径 3 —— 纯 TP 计算，DCP 完全不参与
- **后续 chunk**：`context_len>0`，走路径 2 —— 需要读之前已被 DCP interleave 写入 cache 的 KV

路径 2 的流程：

```python
# 1. 从本 rank 的 cache 读出本地 KV 分片
kv_c_normed, k_pe = self._gather_cache_single(
    kv_c_and_k_pe_cache, prefill_metadata, 0, context_len)

# 2. AllGather 收集所有 DCP rank 的 KV 分片 + 恢复原始 token 顺序
kv_c_normed, k_pe = self._reorg_kvcache_single(
    kv_c_normed, k_pe.squeeze(1), chunked_context, ...)

# 3. 拼接: [之前 decode 的 KV] + [完整 context] + [当前 chunk 新 KV]
kv_c_normed = torch.cat([kv_c_normed_head, kv_c_normed, kv_c_normed_tail], dim=0)

# 4. 标准 causal FlashAttention（每个 rank 16 个 head × 完整 KV）
output = self._flash_attn_varlen_diff_headdims(layer, q, k, v, ...)
```

AllGather + 恢复顺序的具体实现：

```python
# ares_mla_cp.py — _reorg_kvcache_single
def _reorg_kvcache_single(self, kv_c_normed, k_pe, ...):
    cache_kv_c_k_pe = torch.cat([kv_c_normed, k_pe], dim=-1)
    if self.dcp_size > 1:
        # AllGather: 收集所有 rank 的 KV 分片
        cache_kv_c_k_pe = get_dcp_group().all_gather(cache_kv_c_k_pe, 0)
    # 恢复 interleave 顺序: [dcp, tokens/dcp, D] → transpose → [tokens, D]
    cache_kv_c_k_pe = cache_kv_c_k_pe.view(
        self.dcp_size, -1, *cache_kv_c_k_pe.shape[1:]
    ).transpose(0, 1).flatten(end_dim=1)
    return allgatered_kv_c_normed[:sum_seq_len], allgatered_k_pe[:sum_seq_len]
```

**总结**：Prefill 阶段 DCP 只改变了 KV Cache 怎么存，attention 计算用的是 TP。后续 chunk 需要 AllGather 把分散在各 rank 的 cache 收回来，这是 DCP 带来的**纯通信开销**，没有计算收益。

## Indexer 与 DCP 的关系

DCP 是全局配置（启动参数 `decode_context_parallel_size=8`），Indexer 并没有主动"开 DCP"，而是**被动运行在 DCP 环境下**。

在 DSA（Dynamic Sparse Attention）架构中，Indexer 需要对完整 KV Cache 做 TopK 筛选。但 DCP 将 MLA 的 KV Cache 分散到各 rank，Indexer 需要的是完整序列——所以 Indexer 只能选一个 owner rank 存完整的 K Cache：

- **Owner rank**：持有完整 Indexer K cache，执行真正的 `mlp_lightning_indexer`（~13ms）
- **Non-owner rank**：没有 Indexer K cache，跑 dummy indexer 空等（~0.1ms）
- **Broadcast**：Owner 算完 topk 后广播给所有 rank
- **`_compute_local_indices`**：所有 rank 将全局 topk 索引转成 DCP local 物理位置（因为下游 sparse attention 的 KV Cache 是 DCP 分片的）

这些都是**兼容性代码**，不是优化。原来的 Indexer 在 DCP 下是退化的——8 张卡只有 1 张在干活。
