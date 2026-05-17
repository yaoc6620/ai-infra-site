# DCP Decode All2All 通信冗余消除

## 问题背景

### DCP + EP 混合并行架构

在长文本推理场景下，TP8 被拆成 **DCP × EP** 两层并行：

```
TP8 = DCP2 × EP4    或    DCP4 × EP2    或    DCP8 × EP1
```

- **EP（Expert Parallelism）**：MoE 的专家分布在不同 rank 上，需要 All2All 通信交换 token
- **DCP（Decode Context Parallelism）**：KV Cache 分片存储，降低长序列的显存压力

### 冗余的根源

DCP 组内的多个 rank 持有**完全相同的 token**。原因链：

```
上一层 o_proj AllReduce → hidden_states 各 rank 一样
    → MLA a_proj 全量复制 → latent 一样
    → Router 输入一样 → topk_ids, expert_weights 一样
    → dispatch 的内容完全相同
```

但原始 decode All2All 实现中，DCP 组内**每个 rank 都带着相同的 token 参与 dispatch/combine**：

```
原始流程（DCP=2 时）：
  rank 0: dispatch(tokens) → All2All → experts → combine → output
  rank 1: dispatch(tokens) → All2All → experts → combine → output
                ↑ 完全相同的 tokens！

问题：
  ① 通信量翻倍（相同 token 发了 2 次）
  ② 专家计算翻倍（相同 token 算了 2 次）
  ③ 完全浪费
```

---

## 优化方案：Mask-based Dispatch + Leader Broadcast

核心思想：**DCP 组内只让 leader rank（rank 0）实际参与 All2All，其他 rank 不发送 token。combine 完后 leader 把结果 broadcast 给组内其他成员。**

### Dispatch 阶段

```python
def dispatch(self, hidden_states, topk_ids, topk_weights):
    ep_metadata = get_forward_context().ep_metadata
    global_token_num = ep_metadata.total_token_num_across_ep
    group_tp = self.group_ep_str if ep_metadata.decode_only_across_ep else ""

    # ★ 非 leader rank 的 active_mask 设为 False
    active_mask_tensor = ep_metadata.curr_active_mask_tensor
    if get_dcp_group().rank_in_group != 0:
        active_mask_tensor.fill_(False)     # 本 rank 不发送任何 token

    return self._do_dispatch(
        hidden_states, topk_ids, topk_weights,
        global_token_num, group_tp, active_mask_tensor
    )
```

`active_mask_tensor` 是一个布尔掩码，控制哪些 token 实际参与 All2All dispatch。设为 `False` 后，该 rank 的 token 不会被发送到专家——**All2All 通信仍然参与（集合通信要求所有 rank 都调用），但发送量为 0**。

### Combine 阶段

```python
def combine(self, hidden_states, topk_ids, topk_weights):
    ep_metadata = get_forward_context().ep_metadata
    global_token_num = ep_metadata.total_token_num_across_ep
    group_tp = self.group_ep_str if ep_metadata.decode_only_across_ep else ""

    # ★ 同样只有 leader 参与 combine
    active_mask_tensor = ep_metadata.curr_active_mask_tensor
    cp_group = get_dcp_group()
    if cp_group.rank_in_group != 0:
        active_mask_tensor.fill_(False)

    hidden_states = self._do_combine(
        hidden_states, topk_ids, topk_weights,
        global_token_num, group_tp, active_mask_tensor
    )

    # ★ leader 把 combine 结果 broadcast 给 DCP 组内所有 rank
    if cp_group.world_size > 1:
        cp_group.broadcast(hidden_states, src=0)

    self._reset()
    return hidden_states
```

---

## 为什么能这么做

关键前提：**DCP 组内各 rank 的 MoE 输入完全一致**。

| 数据 | DCP 组内是否一致 | 原因 |
|------|:---:|------|
| `hidden_states` | ✓ | 上一层 `o_proj` AllReduce 后同步 |
| `topk_ids` | ✓ | Router 输入一样 → 输出一样 |
| `expert_weights` | ✓ | 同上 |
| dispatch 内容 | ✓ | 以上三者都一样 |
| combine 结果 | ✓ | 相同输入 + 相同专家 → 相同输出 |

既然输入和结果都一样，让 1 个 rank 做就够了，其他 rank 拿 broadcast 结果即可。

---

## 通信量分析

以 DCP=2, EP=4, batch 中有 N 个 token 为例：

### 原始方案

```
每个 DCP rank 独立做 All2All：
  dispatch: 2 × All2All(N tokens)    ← 通信量 ×2
  experts:  2 × compute(N tokens)    ← 计算量 ×2
  combine:  2 × All2All(N tokens)    ← 通信量 ×2
```

### 优化后

```
只有 leader 做 All2All，然后 broadcast：
  dispatch: 1 × All2All(N tokens) + 1 × All2All(0 tokens)
  experts:  ~1 × compute(N tokens)     ← 非 leader 收到的 token 少了
  combine:  1 × All2All(N tokens) + 1 × All2All(0 tokens)
  sync:     1 × broadcast(N × H)       ← DCP 组内广播

All2All 通信量减半，broadcast 代价远小于一次完整 All2All
```

---

## Prefill 阶段的不同策略

Decode 用 mask + broadcast 方案，Prefill 用更激进的方案——**DCP 组内各 rank 分工处理不同 token**：

```python
# forward_a2a_prefill — Step 0: TP token split
if tp_size > 1:
    chunk_size = (S_local_full + tp_size - 1) // tp_size
    chunk_start = self.tp_rank * chunk_size
    chunk_end = min(chunk_start + chunk_size, S_local_full)
    hidden_states = hidden_states[chunk_start:chunk_end]    # 每个 rank 只处理一部分
```

```python
# forward_a2a_prefill — Step 11: TP AllGather 拼回完整结果
if tp_size > 1:
    output_gathered = torch.empty(tp_size * chunk_size, H, ...)
    dist.all_gather_into_tensor(output_gathered, output, group=self.tp_group.device_group)
    output = output_gathered[:S_local_full]
```

### Decode vs Prefill 策略对比

| | Decode: Mask + Broadcast | Prefill: Token Split + AllGather |
|---|---|---|
| **分工** | 只有 leader 工作 | 每个 rank 处理 S/dcp_size 个 token |
| **通信** | broadcast | AllGather |
| **利用率** | 非 leader 空闲 | 所有 rank 都在计算 |
| **适用原因** | Decode token 少（通常 1 个），分不了 | Prefill token 多，可以均分 |

---

## 性能数据

在长序列推理场景下，**Decode 单步延迟降低约 30ms**。

---

## 面试讲述要点

::: details 面试时怎么讲？（30 秒）

"DCP + EP 混合并行的 decode 阶段，DCP 组内各 rank 持有完全相同的 token——因为 hidden_states 经过 AllReduce 同步、MLA 压缩层全量复制。原始实现每个 rank 都参与 All2All dispatch/combine，通信和计算完全冗余。

我的优化是 Mask-based Dispatch：只让 DCP 组的 leader rank 实际发送 token，其他 rank 的 active_mask 设为 False。combine 完后 leader 用 broadcast 把结果分发给组内成员。All2All 通信量减半，decode 单步降低约 30ms。"

:::

::: details 面试官可能追问

**Q: 非 leader rank 的 All2All 不发数据，为什么还要调用？**

A: All2All 是集合通信，要求所有参与 rank 都调用。非 leader 发送量为 0，但必须参与通信原语的调用，否则其他 rank 会 hang 住。

**Q: Prefill 为什么不用同样的 mask 方案？**

A: Prefill token 数量大（几千到几万），mask 方案只有 leader 工作、其他 rank 空闲，浪费算力。Token split + AllGather 让每个 rank 分摊一部分 token，所有 rank 都在计算，吞吐更高。

**Q: broadcast 的开销如何？**

A: broadcast 是 DCP 组内通信（2~8 个 rank），数据量是 N × hidden_size（decode 时 N 通常是 batch_size 个 token，hidden_size=5120）。这远小于 All2All 要跨 EP 组通信的量级，所以 broadcast 开销可忽略。

**Q: 这个优化和 Indexer DCP 并行优化的关系？**

A: Indexer DCP 并行解决的是 attention 层（prefill 阶段）的计算冗余——只有 sharding_owner 做 indexer，broadcast topk_indices 给组内成员。这个优化解决的是 MoE 层（decode 阶段）的 All2All 通信冗余。思路类似——识别 DCP 组内的冗余，让 leader 做，broadcast 结果。

:::
