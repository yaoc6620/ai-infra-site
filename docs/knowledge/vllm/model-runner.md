# GPU Model Runner

**文件**: `vllm/v1/worker/gpu_model_runner.py`

---

## execute_model 流程

ModelRunner 是 Scheduler 和 GPU 之间的桥梁。接收 SchedulerOutput，准备输入，执行 forward，返回采样结果。

```python
@torch.inference_mode()
def execute_model(self, scheduler_output):
    # 1. 更新内部状态（新增/移除 request、更新 token 位置等）
    self._update_states(scheduler_output)

    # 2. 准备输入（CPU numpy + async H2D）
    attn_metadata, logits_indices = self._prepare_inputs(scheduler_output)

    # 3. 选择执行路径
    if decode_only:
        executor = self.graph_runners[padded_batch_size]  # CUDA Graph
    else:
        executor = self.model                              # Eager

    # 4. Forward
    hidden_states = executor(input_ids, positions, ...)

    # 5. 取需要输出 logits 的位置（每个 request 的最后一个 token）
    logits = self.model.compute_logits(hidden_states[logits_indices])

    # 6. 采样
    return self.sampler(logits, sampling_metadata)
```

### logits_indices 的作用

Batch 中每个 request 可能贡献多个 token（prefill），但只需要最后一个 token 的 logits 来采样下一个 token：

```
Batch: [Req A: 5 tokens] [Req B: 1 token] [Req C: 3 tokens]
Index:   0 1 2 3 4         5                 6 7 8

logits_indices = [4, 5, 8]  ← 每个 request 的最后一个位置
```

---

## _prepare_inputs：CPU/GPU Overlap

这是 vLLM v1 最重要的性能优化之一——把 CPU 数据准备和 GPU 数据传输重叠执行。

### 核心思想

```
传统做法（串行）:
CPU: ████ 准备数据 ████
                        GPU: ████ H2D 传输 ████ forward ████
                        
vLLM v1（重叠）:
CPU: ████ 准备数据 ████████████████
GPU:      ████ block_table DMA ████ H2D ████ forward ████
               ↑ overlap ↑
```

### 代码流程

```python
def _prepare_inputs(self, scheduler_output):
    num_reqs = self.input_batch.num_reqs

    # ═══ Step 1: 先启动 block_table 的 H2D 传输（后台 DMA） ═══
    self.input_batch.block_table.commit(num_reqs)
    # commit() 内部调用 non_blocking copy，DMA 在后台执行

    # ═══ Step 2: CPU 侧 numpy 计算（与 Step 1 的传输并行） ═══
    num_scheduled_tokens = np.array([
        scheduler_output.num_scheduled_tokens[req_id]
        for req_id in self.input_batch.req_ids
    ])

    # 构建 req_indices: 如 num_scheduled_tokens=[2,5,3]
    #   → req_indices = [0,0,1,1,1,1,1,2,2,2]
    req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

    # 构建 positions = num_computed_tokens + local_offset
    # 如 req 0 已算了 100 tokens，本次算 2 个 → positions = [100, 101]
    cu_num_tokens = np.cumsum(num_scheduled_tokens)
    cumsums_offsets = np.repeat(
        cu_num_tokens - num_scheduled_tokens, num_scheduled_tokens)
    arange = self.arange_np[:total_num_scheduled_tokens] - cumsums_offsets
    positions_np = self.input_batch.num_computed_tokens_cpu[req_indices] + arange

    # 收集 token_ids
    token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
    torch.index_select(
        self.input_batch.token_ids_cpu_tensor.flatten(), 0,
        torch.from_numpy(token_indices),
        out=self.input_ids_cpu[:total_num_scheduled_tokens])

    # ═══ Step 3: Async H2D（non_blocking=True） ═══
    self.input_ids[:total].copy_(self.input_ids_cpu[:total], non_blocking=True)
    self.positions[:total].copy_(self.positions_cpu[:total], non_blocking=True)

    # ═══ Step 4: 构建 FlashAttention metadata ═══
    attn_metadata = self.attn_metadata_builder.build(
        num_reqs=num_reqs,
        num_actual_tokens=total_num_scheduled_tokens,
        ...)
    return attn_metadata, logits_indices
```

### 为什么用 numpy 而不是 torch？

CPU 上的数据准备（构建 indices、positions）使用 **numpy** 而非 PyTorch tensor：
- numpy 操作不经过 PyTorch 调度器，没有 dispatch 开销
- 这些操作纯 CPU 计算，不需要 autograd/CUDA 支持
- 最终通过 `torch.from_numpy()` 零拷贝转为 tensor，再 `copy_` 到 GPU

### non_blocking=True 的含义

```python
gpu_tensor.copy_(cpu_tensor, non_blocking=True)
```

- `non_blocking=True`：调用立即返回，DMA 引擎在后台执行传输
- CPU 可以继续做其他工作（如准备下一批数据、构建 metadata）
- GPU compute stream 会自动等待传输完成后再使用数据

---

## InputBatch 数据结构

ModelRunner 维护一个 `InputBatch`，持久存储所有 active request 的状态：

```python
class InputBatch:
    # CPU 侧（numpy / CPU tensor，用于数据准备）
    token_ids_cpu: torch.Tensor        # [max_reqs, max_seq_len] CPU pinned memory
    num_computed_tokens_cpu: np.ndarray # [max_reqs]

    # GPU 侧（预分配的 GPU tensor，用于 forward）
    token_ids: torch.Tensor            # [max_batch_size] GPU
    positions: torch.Tensor            # [max_batch_size] GPU

    # Block table（管理 KV Cache block 映射）
    block_table: BlockTable            # CPU→GPU 的双缓冲
```

**设计思想**：
- CPU 侧用 **pinned memory**（页锁定内存），加速 H2D 传输
- GPU 侧 tensor **预分配**，避免每步 `torch.empty()`
- `_update_states()` 增量更新（只修改新增/变更的 request），而非每步重建

---

## 面试要点

::: details 常见面试问题

**Q: vLLM 如何做 CPU/GPU overlap？**

`_prepare_inputs` 先启动 block_table 的 async H2D 传输（DMA 后台执行），然后在 CPU 上用 numpy 准备 input_ids、positions 等（与传输并行），最后 `non_blocking=True` 拷贝到 GPU。CPU 计算和 GPU 传输的时间线重叠，减少了等待时间。

**Q: 为什么用 numpy 而不是 PyTorch？**

CPU 侧的 index 计算（repeat、cumsum、加法）用 numpy 比 PyTorch 快，因为没有 PyTorch 的 dispatch 开销和 autograd 追踪。最终通过 `torch.from_numpy()` 零拷贝转换。

**Q: non_blocking=True 是怎么保证数据一致性的？**

non_blocking H2D copy 通过 CUDA stream 保证顺序。copy 操作提交到 stream 后，后续在同一 stream 上的 kernel（如 forward）会自动等待 copy 完成。CPU 侧不阻塞，可以继续做其他准备工作。

**Q: InputBatch 为什么用 pinned memory？**

Pinned memory（页锁定内存）允许 GPU 直接通过 DMA 访问，不需要经过 CPU 的 pageable memory 中转。H2D 传输速度可提升 2-3 倍，且支持 non_blocking=True 的真正异步传输。

:::
