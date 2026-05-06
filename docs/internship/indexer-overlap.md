# Sparse Attention Indexer 异步 Overlap 优化

## 项目背景

在 DSA (Dynamic Sparse Attention) 场景中，每层 Attention 的 `indexer` 负责计算每个 token 应该 attend to 哪些 KV（topk 选择）。原实现中 indexer 在主 CUDA Stream 上同步执行，阻塞了后续无依赖的小 tensor 计算。通过将 indexer 放入独立 Stream 并用 Event 做延迟同步，实现与主路径 MLA query 计算的 **overlap 并行**，在 decode 阶段每步节省约 **2ms**。

---

## 一、DSA Sparse Attention 流程

### 1.1 什么是 DSA

标准 MLA Attention 中每个 token 要 attend to 所有历史 KV（O(N²) 计算）。DSA 通过 **indexer** 为每个 token 选出 topk 个最相关的 KV block，只对这些 block 做 attention（稀疏化）：

```
标准 Attention: Q × K^T → 全量 N×N → softmax → × V
DSA:           Q × K^T → indexer 选 topk blocks → 只对 topk 做 attention
```

### 1.2 完整计算流程

```
DeepseekV2MLAAttention.forward():

① q_c = matmul(hidden_states, q_a_proj)        # Q 低秩压缩
② q_c = q_a_layernorm(q_c)                     # LayerNorm
③ indexer(hidden_states, q_c, rotary_emb)       # ★ 计算 topk_indices
④ q = q_b_proj(q_c)                            # Q 投影回高维
⑤ ... (RoPE, 其他变换) ...
⑥ sparse_attention(q, kv_cache, topk_indices)  # ★ 使用 topk_indices
```

**关键观察**：步骤 ③ 的输出 `topk_indices` 直到步骤 ⑥ 才被使用。中间的步骤 ④⑤ 和 indexer **没有数据依赖**。

---

## 二、问题分析

### 2.1 原始执行模式（串行）

```
主 Stream:  [q_a_proj] → [layernorm] → [indexer ████] → [q_b_proj] → [...] → [sparse attn]
                                         ↑ 阻塞！
                                         indexer 完成前，q_b_proj 等无法开始
                                         但 q_b_proj 根本不需要 indexer 的结果！
```

### 2.2 为什么是瓶颈

- indexer 内部做了 matmul + topk 选择，**是整个 MLA 前处理中计算最重的部分**
- 后续的 `q_b_proj`、RoPE 等是小 tensor 计算，耗时短但被 indexer 阻塞
- 同一 Stream 中 kernel 严格按序执行，白白浪费了 overlap 机会
- **核心矛盾**：indexer 耗时长，但主路径上那些小计算完全不依赖它的结果

---

## 三、优化方案：独立 Stream + Event 同步

### 3.1 核心思想

把 indexer（耗时长的部分）放到独立的 `_indexer_stream` 上执行，主 stream 不用等它完成就可以继续执行 `q_b_proj`、RoPE 等小 tensor 计算。只有到 sparse attention 真正需要 `topk_indices` 时，才通过 Event 同步等待 indexer 完成。这样主路径上的小计算被 indexer 的执行时间"吃掉"了。

### 3.2 优化后执行模式

```
主 Stream:       [q_a_proj] → [layernorm] → [q_b_proj] → [RoPE] → [event.wait()] → [sparse attn]
                                    ↓          (小 tensor 计算)           ↑
indexer Stream:              [wait_stream] → [indexer ████████████] → [event.record()]─┘
                                              ↑ indexer 耗时长，把主路径小计算 overlap 掉
```

**时间节省** ≈ 主路径小 tensor 计算耗时（被 indexer 完全覆盖），实测 **~2ms/step**

### 3.3 同步机制详解

```
时间线 →

主 Stream:        ─── [LayerNorm] ──┬── [q_b_proj] ── [RoPE] ── ... ── [event.wait()] ── [sparse_attn] ──
                                    │                                         ↑
                                    │                                    等 indexer 完成
                                    ↓
indexer Stream:   ── [wait_stream] ── [indexer 计算] ── [event.record()] ──
                      ↑                                      ↑
                 等主 stream 的              indexer 完成后
                 LayerNorm 算完              记录事件
                 (保证输入就绪)
```

三步同步协议：
1. **`_indexer_stream.wait_stream(current_stream)`** — indexer stream 等主 stream 的输入（`hidden_states`, `q_c`）就绪
2. **`_indexer_event.record()`** — indexer 完成后在 indexer stream 上记录事件
3. **`indexer_event.wait()`** — 主 stream 在真正需要 topk 结果时才等待（延迟同步）

---

## 四、完整代码改动

### 4.1 文件总览

| 文件 | 改动 | 作用 |
|------|------|------|
| `ares/engines/vllm_plugins/models/gpu_mt_flash_moe.py` | indexer 放入独立 stream | 生产者：异步计算 topk_indices |
| `ares/engines/_vllm_0_8_5/v1/attention/backends/mla/flash_attn_v3_sparse.py` | 消费时 event.wait() | 消费者：等待结果就绪 |

### 4.2 生产者侧：模型层（gpu_mt_flash_moe.py）

#### 新增全局 Stream 和 Event

```python
_indexer_stream = torch.cuda.Stream()
_indexer_event = torch.cuda.Event()
```

#### 优化前代码

```python
def forward(self, positions, hidden_states: torch.Tensor):
    import flashinfer
    if self.q_lora_rank is not None:
        q_c = torch.matmul(hidden_states, self.q_a_proj)
        q_c = self.q_a_layernorm(q_c)
        if self.indexer is not None:
            # ❌ 在主 stream 上同步执行，阻塞后续计算
            self.indexer(hidden_states, q_c, rotary_emb)
        q = self.q_b_proj(q_c)[0]
```

#### 优化后代码

```python
def forward(self, positions, hidden_states: torch.Tensor):
    indexer_event = None
    import flashinfer
    if self.q_lora_rank is not None:
        q_c = torch.matmul(hidden_states, self.q_a_proj)
        q_c = self.q_a_layernorm(q_c)
        if self.indexer is not None:
            forward_context = get_forward_context()
            if (forward_context.attn_metadata is not None and
                    forward_context.attn_metadata.sparse is not None):
                # ✅ 有 sparse metadata 时，indexer 放入独立 stream
                global _indexer_stream
                global _indexer_event
                current_stream = torch.cuda.current_stream()
                _indexer_stream.wait_stream(current_stream)       # 等输入就绪
                with torch.cuda.stream(_indexer_stream):
                    self.indexer(hidden_states, q_c, rotary_emb)  # 异步执行
                    _indexer_event.record()                        # 记录完成
                forward_context.attn_metadata.sparse.indexer_event = \
                    _indexer_event                                 # 传递 event 给消费者
            else:
                # 非 sparse 场景，走原路径
                self.indexer(hidden_states, q_c, rotary_emb)
        q = self.q_b_proj(q_c)[0]   # ✅ 主 stream 不用等 indexer，直接继续
```

### 4.3 消费者侧：Attention 层（flash_attn_v3_sparse.py）

#### 数据结构：新增 event 字段

```python
@dataclass
class FlashAttnV3SparseDSAMetadata:
    ...
    use_fa3_with_spt: bool
    indexer_event: Optional[torch.cuda.Event] = None   # ✅ 新增
```

#### 消费时同步等待

```python
# 在 _topk_indices_to_block_table 之前，确保 indexer 已完成
indexer_event = attn_metadata.sparse.indexer_event
if indexer_event is not None:
    indexer_event.wait()                         # ✅ 主 stream 等 indexer 完成
    attn_metadata.sparse.indexer_event = None    # 清理，防止重复等待

# 现在 topk_indices 已就绪，安全使用
topk_block_table, valid_topk_tokens = self._topk_indices_to_block_table(
    topk_indices, attn_metadata.sparse.block_table, block_size
)
```

---

## 五、设计要点

### 5.1 为什么用全局 Stream 而不是每次创建？

```python
# 模块顶层：全局创建，复用
_indexer_stream = torch.cuda.Stream()
_indexer_event = torch.cuda.Event()
```

- CUDA Stream/Event 创建有开销（需要驱动分配资源）
- 每层每次 forward 都需要用，创建+销毁太浪费
- 全局复用是标准做法

### 5.1.1 为什么在函数内需要 `global` 声明？

```python
def forward(self, ...):
    global _indexer_stream
    global _indexer_event
    ...
```

Python 的作用域规则：如果在函数内对变量赋值，Python 默认将其视为**局部变量**。`global` 关键字显式声明"这个名字指向模块顶层的全局变量"。

为什么不用 `self._indexer_stream`（实例属性）？
- Stream/Event 是 **CUDA 设备资源**，不应绑定到 model state
- 如果放在 `self` 上，会被 `model.state_dict()` / 序列化 / `model.to(device)` 等操作干扰
- 全局变量 = 轻量级单例模式，进程生命周期内只创建一次，所有层共享复用

### 5.2 为什么需要 `wait_stream`？

```python
_indexer_stream.wait_stream(current_stream)
```

indexer 的输入是主 stream 上刚算完的 `hidden_states` 和 `q_c`。如果不 wait，indexer stream 可能在 `q_a_layernorm` 还没算完时就开始读取 `q_c` → 结果错误。

`wait_stream` 确保 indexer stream 在主 stream **当前已提交的所有 kernel 完成后**才开始执行。

### 5.3 为什么 Event 挂在 metadata 上传递？

```python
forward_context.attn_metadata.sparse.indexer_event = _indexer_event
```

生产者（模型层）和消费者（attention 层）是不同的类/不同的调用位置。通过 `attn_metadata` 传递 event 是最自然的方式——metadata 本身就是贯穿整个 forward 的上下文。

### 5.4 为什么消费后要清 None？

```python
attn_metadata.sparse.indexer_event = None
```

防止后续代码重复 `event.wait()`（虽然重复 wait 不会出错，但浪费时间），也是防御性编程。

### 5.5 为什么只在 Decode 阶段有效？

Decode 阶段：每个 token 逐个生成，MLA query 侧的 `q_b_proj`、RoPE 等是**小 tensor 计算**（batch 小），而 indexer 需要对所有历史 KV 做 topk 选择，耗时相对大。此时 overlap 可以把主路径小计算完全藏在 indexer 背后，节省约 2ms。

Prefill 阶段：一次处理大量 token（如 64K），主路径的 `q_b_proj` 等计算本身就是**大矩阵乘法**，GPU 计算密集度高，SM 已经被充分利用。此时：
- 两个 stream 的 kernel 争抢 SM 资源，反而可能互相拖慢
- 大矩阵计算本身耗时远超 indexer，overlap 节省的那点时间占比极小
- Prefill 的瓶颈在 attention 计算本身，不在 indexer

因此，**此优化建议只在 decode 阶段启用**。

### 5.6 非 sparse 场景为什么走原路径？

```python
else:
    self.indexer(hidden_states, q_c, rotary_emb)
```

如果不是 sparse attention（没有 `attn_metadata.sparse`），说明没有消费者会去 `event.wait()`。此时如果放到 side stream 上，indexer 的结果可能在主 stream 后续使用时还没算完，造成正确性问题。

---

## 六、CUDA Stream Event 编程模式总结

这个优化是 **"异步计算 + 延迟同步"** 的经典范式：

```python
# 模式：将无依赖计算放到 side stream，延迟同步到真正需要结果时

# 1. 创建 stream 和 event（全局复用）
side_stream = torch.cuda.Stream()
event = torch.cuda.Event()

# 2. 生产者：在 side stream 上异步计算
side_stream.wait_stream(torch.cuda.current_stream())  # 等输入就绪
with torch.cuda.stream(side_stream):
    result = expensive_computation(input)               # 异步执行
    event.record()                                      # 标记完成

# 3. 主 stream 继续做其他不依赖 result 的计算
other_result = other_computation(input)                 # 和 side_stream 并行！

# 4. 消费者：真正需要 result 时才等待
event.wait()                                            # 同步
use(result)                                             # 安全使用
```

**适用条件**：
- 计算 A 和计算 B 之间**无数据依赖**
- 计算 A 的结果在后续某个**确定时间点**才被需要
- 两个计算的耗时都不可忽略（否则 overlap 收益太小）

---

## 面试讲述要点

::: details 面试时怎么讲？（1 分钟版本）

"在 DSA Sparse Attention 的 decode 阶段，indexer 需要为每个 token 计算 topk 相关的 KV block，是 MLA 前处理中计算最重的部分。但它的结果要到最后 sparse attention 时才被用到，中间的 q_b_proj、RoPE 等小 tensor 计算被它阻塞了。

我把 indexer 放到独立的 CUDA Stream 上异步执行，通过 Event 做延迟同步。这样主路径上的小计算可以和 indexer 并行，被 indexer 的执行时间 overlap 掉，每步节省约 2ms。注意这个优化只在 decode 阶段有效——prefill 阶段计算密集，SM 利用率高，overlap 收益可忽略。"

:::

::: details 面试官可能追问

**Q: 为什么不把所有计算都放 side stream？**

A: 只有无数据依赖的计算才能 overlap。如果有依赖（比如 q_b_proj 依赖 q_c），放 side stream 要么需要额外同步（开销抵消收益），要么结果错误。

**Q: Event.wait() 和 synchronize 有什么区别？**

A: `event.wait()` 是 **GPU 端同步**——让当前 stream 等待另一个 stream 的 event，不阻塞 CPU。`cudaDeviceSynchronize()` 是 **CPU 端同步**——CPU 等所有 GPU 工作完成，会阻塞 CPU。这里用 event.wait() 不会引入 CPU 阻塞。

**Q: indexer 比主路径计算慢很多，overlap 效果如何？**

A: 这里正是这种情况——indexer 是耗时长的部分，主路径的 q_b_proj/RoPE 是小计算。Overlap 节省的时间 = 主路径小计算的耗时（约 2ms），因为它们被完全藏在 indexer 背后了。indexer 本身耗时没变，但主路径小计算从"串行等待"变成了"免费执行"。

**Q: 为什么只在 decode 有效，prefill 没收益？**

A: Prefill 一次处理大量 token，q_b_proj 等变成大矩阵乘法，SM 已经被打满。两个 stream 的 kernel 争抢 SM 反而互相拖慢，且大计算本身远超 indexer 耗时，overlap 占比极小。

**Q: 全局 Stream 有线程安全问题吗？**

A: 在推理场景中，同一个模型的 forward 是单线程顺序调用的（一层一层过），不存在并发。如果是多线程场景需要用 thread-local stream。

:::
