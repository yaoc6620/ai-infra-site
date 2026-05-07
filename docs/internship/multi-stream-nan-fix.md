# Multi-Stream + CUDA Graph NaN 修复

## 项目背景

在 MoE 模型推理中，为了加速每层 Transformer 的计算，将 dense path（shortcut MLP + attention）和 MoE path（router → all2all dispatch → experts → combine）放到两个独立 CUDA Stream 上并行执行。但在开启 CUDA Graph 后，偶发出现 **NaN** 输出。经排查定位到 PyTorch caching allocator 的跨 stream 内存回收机制在 Graph capture 下失效，通过 `record_stream()` + shared stream 方案修复。

---

## 一、MoE 层的多 Stream Overlap 结构

### 1.1 为什么要多 Stream

每层 `ParallelShortcutTransformerLayer` 包含两条计算路径：

```
Dense Path:  mlps[0] → attention → mlps[1]        （shortcut MLP + 注意力）
MoE Path:   router → all2all dispatch → experts → combine  （稀疏专家）
```

两条路径的输入是**同一个 tensor**（`original_hidden_states = hidden_states` 保存引用，因为 dense path 会覆写 `hidden_states`），但两条路径的计算过程**无数据依赖**，可以用两个 stream 并行：

```
default_stream: [前序计算] ──┬─────────────────────────────── [wait] → [output = dense + moe + residual]
                             │
dense_stream:               └─ [mlps[0]] → [attn] → [mlps[1]] ──┐
                                                                  ↓
moe_stream:                 └─ [router] → [dispatch] → [experts] → [combine] ──┘
```

### 1.2 原始实现：全局共享 Stream

```python
# 模块顶层
test_alt_stream = torch.cuda.Stream()
test_moe_stream = torch.cuda.Stream()

class ParallelShortcutTransformerLayer(nn.Module):
    def forward(self, ...):
        global test_alt_stream
        global test_moe_stream
        default_stream = torch.cuda.current_stream()
        event_rs0_done = torch.cuda.Event()        # 每次 forward 新建 event
        event_ag1_done = torch.cuda.Event()
        event_dispatch_send_done = torch.cuda.Event()
        event_combine_send_done = torch.cuda.Event()

        test_alt_stream.wait_stream(default_stream)
        test_moe_stream.wait_stream(default_stream)

        with torch.cuda.stream(test_alt_stream):
            # dense path ...
        with torch.cuda.stream(test_moe_stream):
            # moe path ...

        default_stream.wait_stream(test_alt_stream)
        default_stream.wait_stream(test_moe_stream)
        output = hidden_states + shortcut_mlp_output + residual
```

---

## 二、NaN 问题分析

### 2.1 现象

- 开启 CUDA Graph（decode 阶段 capture + replay）后，输出偶发 NaN
- 不固定在某一层（有时 layer 5，有时 layer 12）
- 关闭 CUDA Graph 或关闭多 stream → NaN 消失
- **偶发 + 不固定层数 = 不是数值溢出，是内存被覆写**

### 2.2 排查过程

#### Step 1：检测 NaN 的位置

CUDA Graph 内部不能 print、不能做依赖 tensor 值的 Python 分支（因为 Graph 只录 CUDA op，不执行 Python 逻辑）。所以 NaN 检测只能在 **Graph 外部**做：

```python
# Graph replay 后检查 output buffer（replay 完成后 output_buffer 是普通 tensor）
graph.replay()
if torch.isnan(output_buffer).any():
    print(f"NaN detected! Step {step}")
```

也可以在每层 Graph 之间（如果是 per-layer Graph）或 eager fallback 模式下逐层检查：

```python
# eager 模式下逐层定位
for i, layer in enumerate(self.layers):
    output = layer(...)
    if torch.isnan(output).any():
        print(f"Layer {i} output has NaN!")
        break
```

发现 NaN 出现的层不固定 → 排除数值溢出（数值溢出一般在固定位置），指向内存被覆写。

#### Step 2：控制变量定位

| 配置 | NaN? | 结论 |
|------|------|------|
| 多 stream + CUDA Graph | ✅ 偶发 | 两者叠加触发 |
| 多 stream + eager | ❌ | 多 stream 本身逻辑没错 |
| 单 stream + CUDA Graph | ❌ | Graph 本身没问题 |

→ 确认是 **多 stream + CUDA Graph 组合** 导致的内存问题。

#### Step 3：推理 root cause

到这一步不是靠 print debug 了，是靠**分析 PyTorch caching allocator 的内存模型**：
- CUDA Graph capture 期间 `process_events()` 被禁用
- 跨 stream 使用 tensor 但没有 `record_stream()` 告知分配器
- 两个条件叠加 → 内存被提前复用 → 脏数据 → NaN

#### Step 4：验证假设

加 `record_stream()` → NaN 消失 → 确认 root cause。

### 2.3 Root Cause：三个问题叠加

#### 问题 1：跨 stream 使用 tensor 但缺少 record_stream

```python
residual = hidden_states + x              # 在 default stream 上创建

with torch.cuda.stream(dense_stream):
    hidden_states = residual + hidden_states   # 在 dense_stream 上读 residual
```

分配器只跟踪 tensor 的**创建 stream**。当 Python 侧 `residual` 引用计数归零（变量被重新赋值），分配器将其放回 default stream 的 free list。但此时 `dense_stream` 上的 kernel 可能还在读这块地址 → 被新 tensor 覆写 → 脏数据 → NaN。

#### 问题 2：CUDA Graph capture 禁用了 process_events()

正常 eager 模式下，分配器每次 alloc 时会自动调用 `process_events()`——检查 event 完成状态，确认跨 stream 的 free 块是否安全可复用。这在大多数情况下能**兜底**。

但 CUDA Graph capture 期间：
- kernel 只被录制不执行 → event 永远不会 complete
- 如果不禁用 `process_events()`，所有 free 块都不能回收 → 显存爆炸
- 所以 PyTorch **禁用了 process_events()**，允许同一 stream 上的 free 块立刻被复用

结果：**跨 stream 的内存复用完全失去保护**。

#### 问题 3：全局 stream 跨层叠加

```
Layer 0: test_alt_stream 上 free tensor A
Layer 1: test_alt_stream 上 alloc tensor B → 复用 A 的地址
```

同一个全局 stream，分配器认为 Layer 0 free 的块 Layer 1 可以直接复用。但 Graph replay 时 Layer 0 的 kernel 可能还没跑完。

---

## 三、修复阶段 1：Per-layer Stream + record_stream

### 3.1 方案思路

1. **Per-layer stream**：切断跨层内存复用冲突
2. **record_stream()**：告诉分配器 tensor 的完整跨 stream 生命周期
3. **持久化 Event**：避免每次 forward 创建（CUDA Graph 要求确定性）

### 3.2 代码改动（commit b4f1b449）

#### 删除全局 stream

```python
# 删除模块顶层的全局变量
- test_alt_stream = torch.cuda.Stream()
- test_moe_stream = torch.cuda.Stream()
```

#### 改为 per-layer 初始化

```python
class ParallelShortcutTransformerLayer(nn.Module):
    def __init__(self, ...):
        ...
        # Per-layer CUDA streams (avoid global streams shared across layers/captures)
        self._dense_stream = torch.cuda.Stream()
        self._moe_stream = torch.cuda.Stream()
        # Persistent events (avoid re-creating each forward; safe for CUDA Graph replay)
        self._event_rs0_done = torch.cuda.Event()
        self._event_ag1_done = torch.cuda.Event()
        self._event_dispatch_send_done = torch.cuda.Event()
        self._event_combine_send_done = torch.cuda.Event()
```

#### forward 中使用实例变量 + record_stream

```python
def forward(self, ...):
    original_hidden_states = hidden_states
    default_stream = torch.cuda.current_stream()
    dense_stream = self._dense_stream
    moe_stream = self._moe_stream
    event_rs0_done = self._event_rs0_done
    event_ag1_done = self._event_ag1_done
    event_dispatch_send_done = self._event_dispatch_send_done
    event_combine_send_done = self._event_combine_send_done

    # ✅ record_stream: 告诉分配器这些 tensor 在非创建 stream 上也被使用
    residual.record_stream(dense_stream)           # dense path 读 residual
    residual.record_stream(default_stream)         # 最后 output = ... + residual
    original_hidden_states.record_stream(moe_stream)  # moe path 读 original_hidden_states

    dense_stream.wait_stream(default_stream)
    moe_stream.wait_stream(default_stream)

    with torch.cuda.stream(dense_stream):
        hidden_states = self.mlps[0].gather_input(hidden_states)
        hidden_states = self.mlps[0](hidden_states, do_gather=False)
        hidden_states = self.mlps[0].gather_result(hidden_states)
        event_rs0_done.record()
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states, residual = self._attention(
            index=1, rotary_emb=rotary_emb,
            positions=positions, hidden_states=hidden_states, residual=residual,
        )

    with torch.cuda.stream(moe_stream):
        top_experts, top_experts_masked, expert_weights = self.mlp.do_router(
            original_hidden_states, ep_metadata)
        torch.cuda.current_stream().wait_event(event_rs0_done)
        expert_output, token_num_per_local_expert = self.mlp.all2all_comm.dispatch(
            original_hidden_states, None, top_experts_masked,
            expert_weights, self.mlp.num_experts)
        event_dispatch_send_done.record()
        self.mlp.all2all_comm.dispatch_wait()
        expert_output = self.mlp.experts(
            expert_output, token_num_per_local_expert, with_zero=True)
        output_combine = self.mlp.all2all_comm.combine(
            expert_output, top_experts_masked, expert_weights)
        event_combine_send_done.record()

    with torch.cuda.stream(dense_stream):
        torch.cuda.current_stream().wait_event(event_combine_send_done)
        hidden_states = self.mlps[1].gather_input(hidden_states)
        event_ag1_done.record()
        hidden_states = self.mlps[1](hidden_states, do_gather=False)
        hidden_states = self.mlps[1].gather_result(hidden_states)

    with torch.cuda.stream(moe_stream):
        torch.cuda.current_stream().wait_event(event_ag1_done)
        self.mlp.all2all_comm.combine_wait()
        shortcut_mlp_output = self.mlp.process_identity_zero_expert_all2all(
            output_combine, original_hidden_states,
            top_experts=top_experts, original_top_experts=top_experts,
            expert_weights=expert_weights)

    default_stream.wait_stream(dense_stream)
    default_stream.wait_stream(moe_stream)
    output = hidden_states + shortcut_mlp_output + residual
    return output
```

### 3.3 效果

NaN 完全修复。但发现新问题：**GPU 显存暴涨 ~2GB/layer**。

---

## 四、修复阶段 2：Shared Stream（修显存）

### 4.1 为什么 per-layer stream 导致显存爆炸

CUDA Graph capture 时，分配器为**每个 stream** 维护独立的 free list。Per-layer 意味着 N×2 个 stream（60 层 = 120 个 stream），每个 stream 的 free 块只有同一层能复用：

```
Layer 0 _dense_stream: [alloc 200MB] → [free 200MB]   ← 只有 layer 0 能复用
Layer 1 _dense_stream: [alloc 200MB] → [free 200MB]   ← 只有 layer 1 能复用
Layer 2 _dense_stream: [alloc 200MB] → [free 200MB]   ← 只有 layer 2 能复用
```

每层都要独立持有一份中间 tensor 显存，无法跨层复用。

如果是共享 stream：

```
shared_dense_stream: Layer 0 [alloc] → [free] → Layer 1 [alloc 复用!] → [free] → ...
```

同一 stream 上顺序执行，前面 free 的块后面直接复用。

### 4.2 代码改动（commit 3f89ce2e）

#### Layer 初始化改为 None 占位

```python
class ParallelShortcutTransformerLayer(nn.Module):
    def __init__(self, ...):
        ...
        # Shared CUDA streams across all layers (passed from ParallelTransformer)
        # Using shared streams allows CUDA Graph capture to reuse memory across layers,
        # since freed blocks on the same stream can be immediately reclaimed.
        self._dense_stream = None
        self._moe_stream = None
        # Persistent events 保留不变
        self._event_rs0_done = torch.cuda.Event()
        self._event_ag1_done = torch.cuda.Event()
        self._event_dispatch_send_done = torch.cuda.Event()
        self._event_combine_send_done = torch.cuda.Event()
```

#### 在 ParallelTransformer 中创建共享 stream 并分配给所有层

```python
class ParallelTransformer(nn.Module):
    def __init__(self, ...):
        ...
        # Shared CUDA streams for all layers — critical for CUDA Graph capture memory reuse.
        # With per-layer streams, freed blocks cannot be reclaimed across layers during capture
        # (process_events() is disabled), causing ~2GB/layer overhead. Shared streams fix this.
        shared_dense_stream = torch.cuda.Stream()
        shared_moe_stream = torch.cuda.Stream()

        self.start_layer, self.end_layer, self.layers, self.layer_offset = make_layers(...)
        self.layers = self.layers[self.start_layer : self.end_layer]

        # Assign shared streams to all layers after construction
        for layer in self.layers:
            layer._dense_stream = shared_dense_stream
            layer._moe_stream = shared_moe_stream
```

### 4.3 为什么共享 stream 不再 NaN？

因为阶段 1 加的 `record_stream()` **仍然保留**！

- 最初 NaN 是因为：共享 stream + **没有** record_stream → 分配器不知道跨 stream 使用
- 现在：共享 stream + **有** record_stream → 分配器知道 tensor 完整生命周期，不会提前回收

`record_stream()` 是独立于 stream 是否共享的保护机制。

---

## 五、核心知识点

### 5.1 process_events() 是什么

`process_events()` 是 PyTorch caching allocator 内部的一个**垃圾回收检查函数**（不是"等待事件"，不阻塞）。

#### 它解决什么问题

当 tensor 被 `record_stream(other_stream)` 标记过后 Python 引用归零，分配器不会立刻回收这块显存，而是放进一个**待定队列**，同时记录一个 event。`process_events()` 负责遍历这个队列，检查每个 event 的完成状态：

```
待定队列：
  [tensor A 的地址, event_1]  →  event_1 complete?  → Yes → 放回 free list，可复用
  [tensor B 的地址, event_2]  →  event_2 complete?  → No  → 继续等，暂不回收
  [tensor C 的地址, event_3]  →  event_3 complete?  → Yes → 放回 free list，可复用
```

#### 什么时候被调用

**自动调用**，不需要手动写。每次 `torch.empty()` / 任何 tensor 分配时，分配器内部会先调 `process_events()` 看看有没有可以回收的块，再决定是从 free list 复用还是新申请。

#### 类比

类似于 Java/Go 的 GC mark-sweep：
- `record_stream()` = 标记"这个对象还有别的引用者"
- `process_events()` = GC 扫描一遍"那些引用者都用完了吗？用完就回收"

### 5.2 为什么 CUDA Graph capture 时要禁用 process_events()

Graph capture 的本质是"录制"——kernel 只被记录到 Graph 里，**不真正执行**。

问题：`process_events()` 依赖 event 的完成状态来判断是否可以回收。但 capture 期间 kernel 不执行 → event **永远不会 complete**。

如果不禁用：
```
process_events() 扫描 → 所有 event 都 incomplete → 一个都不能回收
→ 后续 alloc 全部要调 cudaMalloc 申请新显存 → OOM
```

所以 PyTorch 的策略：**capture 期间禁用 process_events()，但允许同一个 stream 上的 free 块立刻被后续 alloc 复用**。

为什么同一 stream 复用是安全的？因为同一 stream 上的 kernel **严格按序执行**——录制顺序就是执行顺序，如果 free 在 alloc 之前录制，那 replay 时也一定是先执行完 free 前面的 kernel 再执行 alloc 后面的 kernel。

但**跨 stream** 就没有这个顺序保证 → 没有 `process_events()` 兜底 → 完全裸奔。这就是为什么需要 `record_stream()`。

### 5.3 为什么一开始选择 Per-layer Stream

发现 NaN 后，最直觉的修复思路是：**既然全局 stream 跨层复用出问题，那就让每层有自己的 stream，切断跨层干扰**。

```python
# 每层独立 stream → Layer 0 free 的块不会被 Layer 1 复用
self._dense_stream = torch.cuda.Stream()
self._moe_stream = torch.cuda.Stream()
```

Per-layer stream 的逻辑：
- 不同层的 stream 是不同对象 → 分配器按 stream 分桶 → 跨层不会互相复用
- 再加上 `record_stream()` 保护跨 stream（dense ↔ moe ↔ default）的 tensor
- 双保险，NaN 确实修复了

### 5.4 Per-layer Stream 为什么导致显存爆炸

CUDA Graph capture 时分配器为**每个 stream** 维护独立的 free list。Per-layer 创建了 N×2 个 stream（60 层 = 120 个 stream），它们的 free 块互相看不到：

```
Layer 0 _dense_stream: [alloc 200MB] → [free 200MB]   ← 只有 layer 0 自己能复用
Layer 1 _dense_stream: [alloc 200MB] → [free 200MB]   ← 只有 layer 1 自己能复用
...
Layer 59 _dense_stream: [alloc 200MB] → [free 200MB]  ← 只有 layer 59 自己能复用
```

每层都独占一份中间 tensor 的显存，即使结构完全相同（每层用完就 free 了），也无法跨层复用。60 层 × ~2GB = 爆炸。

### 5.5 Shared Stream 为什么能同时解决 NaN 和显存

#### 解决显存：同一 stream 上可直接复用

```
shared_dense_stream:
  Layer 0 [alloc A] → [用 A] → [free A] → Layer 1 [alloc B 复用 A 的地址!] → [用 B] → [free B] → ...
```

同一个 stream，顺序执行保证安全。分配器**不需要 `process_events()`** 就敢复用 → Graph capture 下也正常工作。总共只需 1 份中间 tensor 的显存。

#### 不再 NaN：record_stream 仍在保护

shared stream 其实回到了和"最初全局 stream"相同的结构。但关键区别是：**阶段 1 已经加了 `record_stream()`**。

- 最初：shared stream + 没有 record_stream → 分配器不知道跨 stream 使用 → NaN
- 现在：shared stream + 有 record_stream → 分配器知道 tensor 完整生命周期 → 安全

`record_stream()` 是独立于 stream 是否共享/per-layer 的保护机制。

### 5.6 三阶段对比

| 阶段 | Stream 方式 | record_stream | process_events 可用? | NaN | 显存 |
|------|------------|---------------|---------------------|-----|------|
| 最初 | 全局共享 | ❌ | Graph 下禁用 | ❌ NaN | 正常 |
| 修复 1 | Per-layer | ✅ | Graph 下禁用（但不需要，因为不跨层复用） | ✅ 安全 | 爆炸 |
| 修复 2 | 共享（模型级别） | ✅ | Graph 下禁用（但不需要，同 stream 直接复用） | ✅ 安全 | 正常 |

### 5.7 record_stream() 的内部实现

```python
tensor.record_stream(other_stream)
```

内部做了什么：
1. 在 `other_stream` 上记录一个 event（标记当前 stream 进度）
2. 给这块内存打一个 `(other_stream, event)` tag
3. 当 Python 引用归零触发 free 时，分配器不立刻回收，而是放入待定队列
4. 后续 `process_events()` 检查该 event 是否 complete，complete 后才真正回收

在 Graph capture 下 `process_events()` 被禁用，但 `record_stream()` 的 tag 仍然有效——分配器看到 tag 存在就不会把这块内存放回 free list 给同 stream 的后续 alloc 复用。

### 5.8 process_events() vs event.wait()：容易混淆的两个概念

名字里都有 "event"，但完全不同：

| | process_events() | event.wait() / stream.wait_event() |
|---|---|---|
| 层面 | **内存管理** | **执行顺序控制** |
| 谁调 | 分配器内部自动调 | 你手动写 |
| 目的 | 决定显存块能不能回收 | 决定 kernel 能不能开始执行 |
| 阻塞 | 不阻塞任何东西（只是 query 状态） | 阻塞**当前 stream** 的后续 kernel |
| 类比 | GC 检查"对象还有人引用吗" | 线程 join / barrier |

#### event.wait() 的特性

`event.wait()` 只能让**当前 stream 等别人**，不能让别人等自己。每个 stream 只管理自己的执行顺序：

```python
# Stream A 做完某件事后记录 event
with torch.cuda.stream(stream_A):
    compute_something()
    my_event.record()           # "我完成了，记录一下"

# Stream B 自己决定要不要等 A
with torch.cuda.stream(stream_B):
    my_event.wait()             # B 选择等 A → B 后续 kernel 排队
    use_result()                # 保证在 A 完成之后执行

# Stream C 不 wait → 不受任何影响
with torch.cuda.stream(stream_C):
    other_work()                # 和 A、B 完全无关，继续跑
```

类似 pub/sub 模式：
- `event.record()` = 发布"我做完了"这个信号
- `event.wait()` = 订阅者说"我要等这个信号再继续"
- 只有主动订阅（调 wait）的 stream 才会被阻塞，没有机制能"强制让别的 stream 停下来"

#### 在本项目中的应用

```python
# MoE multi-stream overlap 中的 event 使用（执行顺序控制）
with torch.cuda.stream(dense_stream):
    hidden_states = self.mlps[0](hidden_states)
    event_rs0_done.record()                          # dense 第一阶段完成

with torch.cuda.stream(moe_stream):
    top_experts = self.mlp.do_router(original_hidden_states)
    torch.cuda.current_stream().wait_event(event_rs0_done)  # moe stream 等 dense 第一阶段
    # 这里 moe 需要某个依赖 dense 结果的数据，所以要等
    expert_output = self.mlp.all2all_comm.dispatch(...)
```

这些 event 控制的是"谁先执行谁后执行"，和 `process_events()`（内存回收检查）完全无关。

---

## 六、面试讲述要点

::: details 面试时怎么讲？（1 分钟版本）

"为了加速 MoE 层，我们把 dense path 和 MoE expert path 放到两个独立 CUDA Stream 上并行执行。上线后配合 CUDA Graph 出现了偶发 NaN。

排查发现是三个问题叠加：跨 stream 使用 tensor 但分配器不知情；CUDA Graph capture 禁用了 process_events() 兜底机制；全局 stream 导致跨层内存复用冲突。

修复分两步：第一步 per-layer stream + record_stream 修掉 NaN，但显存爆炸（每层独立 free list 无法跨层复用）。第二步改回共享 stream，因为已经有 record_stream 保护，共享后既安全又能跨层复用显存。"

:::

::: details 面试官可能追问

**Q: record_stream 内部是怎么实现的？**

A: 给内存块打一个 (stream, event) tag。分配器回收时检查：如果这个 stream 上的 event 还没 complete，就暂不放回 free list。等 event complete 后才真正回收。

**Q: 为什么 process_events 在 Graph capture 时要禁用？**

A: 因为 capture 期间 kernel 不执行，event 永远不会 complete。如果不禁用，所有 free 块都不能回收（event 永远 incomplete），后续 alloc 全部要新申请显存，直接 OOM。

**Q: per-layer stream 为什么显存爆炸？**

A: 分配器按 stream 分桶管理 free list。120 个 stream（60 层×2）的 free 块互相看不到。每层都要独立持有一份中间 tensor 的显存，约 2GB/layer 的额外开销。

**Q: 为什么不直接用 synchronize 解决？**

A: `cudaDeviceSynchronize()` 会阻塞 CPU 等所有 GPU work 完成，彻底消灭了多 stream overlap 的收益。record_stream 是 GPU 端的异步保护，不阻塞 CPU。

**Q: Event 是每次 forward 创建还是复用？**

A: 必须复用（`__init__` 中创建）。CUDA Graph 要求每次 replay 的操作序列完全相同，包括用的 event 对象。如果每次 forward 新建 event，Graph replay 时地址不同会出错。

:::
