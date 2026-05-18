# Chunked Prefill

## 核心思想

vLLM V1 调度器没有 "prefill 阶段" 和 "decode 阶段" 的区分。每个请求只有两个数字：

- `num_computed_tokens`：已经算完的 token 数
- `num_tokens_with_spec`：总共需要算的 token 数（prompt + output + spec）

每个调度步骤的目标：**让 `num_computed_tokens` 追上 `num_tokens_with_spec`**。

```python
# scheduler.py 设计注释
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has the num_computed_tokens and num_tokens_with_spec.
# At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its
# num_tokens_with_spec.
```

一个长 prompt 可能需要多步才能算完（每步受 `token_budget` 限制），这就是 chunked prefill。

---

## 调度器实现

### Token Budget 与分片

```python
# scheduler.py — 核心调度循环
token_budget = self.max_num_scheduled_tokens

while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]

    # 该请求还剩多少 token 没算
    initial_num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens

    # long_prefill_token_threshold: 单步最多给一个 prefill 请求分配的 token 数
    if 0 < self.scheduler_config.long_prefill_token_threshold < initial_num_new_tokens:
        initial_num_new_tokens = self.scheduler_config.long_prefill_token_threshold

    # 受总预算限制
    num_new_tokens = min(initial_num_new_tokens, token_budget)
    token_budget -= num_new_tokens
```

**效果**：一个 8K 的 prompt，`token_budget=2048` 时会被分成 4 步调度（2048 × 4）。每步只算一部分，**decode 请求可以在间隙中调度**，不会被长 prefill 饿死。

### Prefill + Decode 混合 Batch

一个调度步骤的 batch 中可以同时包含：
- 正在做 chunked prefill 的请求（每次算 N 个 token）
- 正在 decode 的请求（每次算 1 个 token）

---

## Model Runner 实现

### Position 计算

```python
# gpu_model_runner.py — position = num_computed_tokens + 局部 arange
positions_np = num_computed_tokens + arange
# arange = [0, 1, ..., num_scheduled_tokens-1]
```

对于 chunked prefill 的第 2 个 chunk（`num_computed_tokens=2048, num_scheduled_tokens=2048`）：
- positions = [2048, 2049, 2050, ..., 4095]

Position 是**全局的**，RoPE 编码正确。

### seq_lens 计算

```python
# gpu_model_runner.py
seq_lens = num_computed_tokens + num_scheduled_tokens
```

`seq_lens` 告诉 attention kernel 这个请求一共要看多少 KV（包括历史 cache + 当前 chunk）。

### attn_state 判断

```python
attn_state = 2  # chunked_prefill（默认）

if np.array_equal(seq_lens, num_scheduled_tokens):
    attn_state = 0  # prefill_only: 所有请求都是从头算（没有历史 cache）
elif np.all(num_scheduled_tokens == 1):
    attn_state = 1  # decode_only: 所有请求每次只出 1 个 token
```

三种状态：
| attn_state | 含义 | 场景 |
|:---:|---|---|
| 0 | prefill_only | 全是新 prompt，没有历史 KV |
| 1 | decode_only | 全是 decode（含 MTP 验证） |
| 2 | chunked_prefill | 混合 batch：有历史 cache 的 prefill + decode |

---

## Attention Kernel 分路

### Ares MLA Attention 的 forward

```python
# ares_mla_v1.py — forward 入口
def forward(self, layer, q, k_c_normed, k_pe, kv_cache, attn_metadata):
    num_decode_tokens = attn_metadata.num_decode_tokens

    # ① 先写 KV cache（所有 token 一起写）
    slot_mapping = attn_metadata.slot_mapping.reshape(-1, 1)
    torch_npu.npu_scatter_nd_update_(kv_cache, slot_mapping, kv)

    # ② prefill 走 flash_attn_varlen
    if has_prefill:
        output[num_decode_tokens:] = self._forward_prefill(
            layer, prefill_q, prefill_k_c_normed, prefill_k_pe, kv_cache, attn_metadata)

    # ③ decode 走 paged_attention (IFA)
    if has_decode:
        output[:num_decode_tokens] = self._forward_decode(
            layer, decode_q_nope, decode_q_pe, kv_cache, attn_metadata)
```

**关键顺序**：KV cache 先写，再读。decode token 读的 cache 包含同 batch 中 prefill token 刚写入的 KV。

### Prefill Kernel：flash_attn_varlen

```python
# _forward_prefill — 两种情况

# 情况 1：纯 prefill（第一个 chunk，没有历史 cache）
#   直接用 flash_attn_varlen + causal mask
output = self._flash_attn_varlen_diff_headdims(
    q=q, k=k, v=v,
    max_seqlen_q=max_query_len,
    max_seqlen_k=max_query_len,  # Q 和 KV 长度一样
    causal=True)

# 情况 2：chunked prefill（有历史 cache）
#   当前 chunk Q 需要看历史 cache 的 KV
#   从 paged KV cache 中 gather 出历史 KV → 拼接当前 KV → flash_attn
kv_c_normed = torch.cat([历史cache_kv, 当前chunk_kv], dim=0)
```

chunked prefill 的 attention 要处理两部分 KV：
1. **当前 chunk 的 KV**：刚投影出来，直接拼接
2. **历史 chunk 的 KV**：从 paged KV cache 中按 block_table gather 出来

最终用 `flash_attn_varlen` + `attention_update`（LSE rescale）合并结果。

### Decode Kernel：Paged Attention (IFA)

```python
# _forward_decode — Incremental Flash Attention
o = torch_npu.mlp_incre_flash_attention_graph(
    q_nope, kv_cache,
    block_table=decode_metadata.block_table,
    actual_seq_lengths_kv=decode_metadata.seq_lens_list,
    ...)
```

Decode 每个 token Q=1，直接从 paged KV cache 按 block_table 读取，支持 CUDA Graph 编译。

---

## MTP 验证：在 Decode 维度展开

MTP（Multi-Token Prediction）验证**不走 chunked prefill**，而是把验证 batch 展开到 decode 维度。

### 原理

MTP heads 预测了 K 个 draft token。验证时把每个 decode 请求展开为 1+K 个"独立 decode token"：

```python
# target_spec_build — 展开 decode metadata
repeat_counts = num_draft_tokens_tensor + 1  # 每个请求展开 1+K 次

# 每个展开位置 j 的 seq_lens 不同：
#   j=0 (target):  base           (看 KV[0..base-1])
#   j=1 (draft 0): base + 1       (看 KV[0..base])
#   j=2 (draft 1): base + 2       (看 KV[0..base+1])
for j in range(1 + nd):
    new_seq_lens_list.append(base + j)

new_decode_metadata = AresAscendMLADecodeMetadata(...)  # 仍然是 decode metadata！
```

### 为什么可以走 decode kernel

1. 每个展开位置的 Q=1（GEMV），跟正常 decode 完全一样
2. KV 已经在 forward 开头全部写入 cache（target + draft 的 KV 都写好了）
3. 通过不同的 `seq_lens` 控制每个位置能看到多少 KV
4. 可以被 CUDA Graph 编译（batch 大小固定）

### 为什么不走 chunked prefill

MTP 验证的 token 之间**没有因果依赖**——它们属于同一个 request 的不同候选位置，互相之间不需要看对方的 KV。每个 token 只需要看自己 `seq_lens` 范围内的历史 KV cache，这正是 decode kernel 的语义。

---

## 面试讲述要点

::: details 面试时怎么讲？（30 秒）

"Chunked prefill 的核心是统一调度：调度器不区分 prefill/decode 阶段，只看 `num_computed_tokens` 有没有追上 `num_tokens`。一个长 prompt 按 `token_budget` 分成多个 chunk，每步只算一部分，decode 请求可以在间隙调度，避免被长 prefill 饿死。

Attention kernel 层面，prefill token 走 flash_attn_varlen（GEMM），decode token 走 paged_attention（GEMV）。关键是 KV cache 先写后读——同一步的所有 token 的 KV 先全部写入 cache，然后各自按自己的 seq_lens 去读。

MTP 验证也走 decode kernel：把 1+K 个候选位置展开为独立 decode token，每个 Q=1 但 seq_lens 不同，batch 变大但 kernel 不变，可以图编译。"

:::

::: details 面试官可能追问

**Q: Chunked prefill 的 attention 怎么看到历史 KV？**

A: 从 paged KV cache 中按 block_table gather 出历史 chunk 的 KV，跟当前 chunk 的 KV 拼接后做 flash_attn。如果历史 KV 很长，会分 chunk gather、分段 attention + LSE rescale 合并结果。

**Q: Position 怎么保证正确？**

A: `position = num_computed_tokens + local_arange`。第二个 chunk 的 position 从 `num_computed_tokens`（上一步的结尾位置）开始，保证 RoPE 全局正确。

**Q: Decode 和 prefill 在同一个 batch 里，怎么走不同 kernel？**

A: batch 里 token 按排列：前面是 decode tokens，后面是 prefill tokens。forward 里用 `num_decode_tokens` 分割，分别走 `_forward_decode`（paged_attention）和 `_forward_prefill`（flash_attn_varlen）。

**Q: MTP 验证为什么不走 flash_attn_varlen？**

A: flash_attn_varlen 适合同一序列内有因果依赖的多个 token（Q 互相能看对方的 KV）。MTP 的候选 token 互相独立，只需各自按 seq_lens 读历史 cache，这正是 paged_attention 的语义。而且走 decode kernel 可以用 CUDA Graph 编译，更快。

**Q: Token budget 设多大合适？**

A: 典型值 2048~8192。太小会让长 prompt 被切太多 chunk、增加调度开销；太大会让 decode 请求等太久。核心权衡是 TTFT（首 token 延迟）vs decode 吞吐。

:::
