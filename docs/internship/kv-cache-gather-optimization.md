# Chunked Prefill KV Cache 批量 Gather 优化

## 项目背景

在 MOE-26B 模型的 NPU 910C 推理场景中，对 Chunked Prefill 阶段进行性能优化。通过 Profiling 定位到 KV Cache Gather 环节存在严重的 **Host Bound** 问题，将 Python for 循环逐 block 索引替换为单次 `index_select` 批量 gather，实现零 H2D 传输。

**性能收益**：64K prompt prefill 耗时从 368.6s → 349.2s，**降低 5.2%（节省 19.3s）**

---

## 一、Chunked Prefill 机制

当启用 `enable_chunked_prefill` 时，长 prompt 不会一次性处理，而是被 scheduler 拆成多轮 chunk 逐步 prefill：

| 轮次 | 本轮处理 (query_len) | 已处理 (context_len) | 说明 |
|------|---------------------|---------------------|------|
| 1 | 768 | 0 | 首轮，无历史 KV |
| 2 | 768 | 768 | 需要从 KV cache gather 前 768 个 token 的 KV |
| 3 | 768 | 1536 | 需要 gather 前 1536 个 token 的 KV |
| ... | ... | ... | ... |
| 10 | 768 | 6912 | 需要 gather 前 6912 个 token 的 KV |
| 11 | 剩余 tokens | 7680 | 需要 gather 前 7680 个 token 的 KV |

> 假设 prompt = 8192 tokens, max_num_batched_tokens = 776, block_size = 128

**关键点**：从第 2 个 chunk 开始，每个 chunk 需要从 KV Cache 中 **gather 之前所有 chunk 已经算好的 KV**，才能做完整的 Attention 计算。

KV Cache 在显存中以 **block** 为单位存储（block_size=128 tokens）。gather 操作就是根据 `block_table` 索引，把分散在显存各处的 block 拼接成连续 tensor。

---

## 二、问题分析：Host Bound

### 2.1 问题现象

Profiling timeline 上，`mlp_prefill_flash_attention` 算子之前出现**大量密集的 `aten::select`** 算子，数量与 `chunked_prefill_block_num` 正相关。

### 2.2 根因代码

```python
# 原代码：N 次循环，每次一个 aten::select
kv_c_and_k_pe_cache_list = [
    kv_c_and_k_pe_cache[idx]                  # ← device 上执行 aten::select
    for idx in block_table_cpu[...][...]       # ← host 上 Python 迭代
]
```

### 2.3 执行模型分析

每次循环迭代做两件事：

| 步骤 | 执行位置 | 操作 |
|------|---------|------|
| 从 `block_table_cpu` 取 idx | Host (CPU) | Python 迭代器 + CPU tensor 索引 |
| `kv_c_and_k_pe_cache[idx]` | Device (NPU) | `aten::select`，host dispatch 一个小 kernel |

Device 每个 `aten::select` 执行极快（几 μs），但 Host 的 Python 循环开销（解释器 + PyTorch dispatch）使得 Device **频繁空闲等待下一个 kernel 被下发**：

```
Host:    for[0] → dispatch → for[1] → dispatch → ... → for[N-1] → dispatch
              ↓                  ↓                            ↓
Device:  select[0]          select[1]     ...           select[N-1]
         (执行完等下一个)    (执行完等下一个)             (执行完等下一个)
```

### 2.4 影响量级

循环次数 = `chunked_prefill_context_len // block_size`

| context_len | block_size | 循环次数（aten::select 数） |
|-------------|-----------|--------------------------|
| 4,096 | 128 | 32 |
| 16,384 | 128 | 128 |
| 40,960 (max_model_len) | 128 | 320 |
| 131,072 (128K) | 128 | **1,024** |

当 `enable_mla_split_kv_kr=True` 时，有 2 个独立的 for 循环（kv_c 和 k_pe 各一次），循环次数翻倍。

**Chunked prefill 的后期轮次受影响最严重**——context 越长，每轮要 gather 的 block 越多，host bound 越明显。

---

## 三、优化方案：index_select 批量 Gather

### 3.1 核心思路

将 N 次逐 block 的 Python for 循环替换为 **1 次 `tensor.index_select` 批量 gather**：

```python
# ========= 优化前：N 次 aten::select（N = block_num）=========
kv_list = [cache[idx] for idx in block_table_cpu[:][:block_num]]
result = torch.cat(kv_list, dim=0)

# ========= 优化后：1 次 index_select，零 H2D 传输 =========
device_block_indices = block_table[chunked_prefill_idx][:block_num].long()
full_blocks = cache.index_select(0, device_block_indices)
result = torch.cat([full_blocks.view(-1, D), last_block], dim=0)
```

### 3.2 关键变量说明

```python
context_lens_cpu = self.runner.input_batch.num_computed_tokens_cpu_tensor[reqs_start:reqs_end]
chunked_prefill_context_len = context_lens_cpu[chunked_prefill_idx].item()
chunked_prefill_block_num = chunked_prefill_context_len // block_size
```

- **`chunked_prefill_context_len`**：该请求在之前所有轮 chunk 中已经计算过的 token 总数（即 `num_computed_tokens`），代表本轮 attention 需要从 KV cache 中 gather 出来的历史 KV 长度
- **`chunked_prefill_block_num`**：历史 KV 占据的完整 KV cache page 数 = `context_len // block_size`

### 3.3 公共部分：Block Indices 准备

三处 for 循环共享同一个 block index 序列。**直接使用 device 端 block_table**，避免 H2D 传输：

```python
# index_select 的 indices：直接从 device 端 block_table 切片，零 H2D
device_block_indices = prefill_metadata.block_table[chunked_prefill_idx][:chunked_prefill_block_num].long()

# last_block_idx：仅此 1 个值需要 CPU（用作 Python int 做切片索引）
last_block_idx = prefill_metadata.block_table_cpu[chunked_prefill_idx][chunked_prefill_block_num].item()
```

### 3.4 为什么可以直接用 device 端 block_table

`prefill_metadata` 中同时持有 `block_table`（device）和 `block_table_cpu`（CPU），两者数据来源相同：

```
BlockTable 内部数据流:

block_table_np  (numpy, host)      ← 所有写入操作（append_row / add_row）都写到这里
       ↓ (共享内存, L43: block_table_cpu.numpy())
block_table_cpu (torch CPU, pin_memory)
       ↓ commit() 时 copy（L158-160）
block_table     (torch device)     ← block_table[:n].copy_(block_table_cpu[:n], non_blocking=True)
```

`commit()` 在每次 model forward 之前调用，保证 device 端数据已同步。因此 **`block_table` 和 `block_table_cpu` 数值完全一致**。

原来用 `block_table_cpu` 是因为 Python for 循环需要 host 上的 int。换成 `index_select` 后，indices 本身就是 device tensor，直接用 device 端 `block_table` 即可。

### 3.5 分路径实现

按 `enable_mla_split_kv_kr` 分为两条路径：

#### 路径 1：非 split（enable_mla_split_kv_kr=False）

KV cache 是单个 tensor，shape `[num_blocks, block_size, D]`。

```python
# 优化前: N 个 aten::select + N+1 次 cat
kv_list = [cache[idx] for idx in block_table_cpu[:][:block_num]]
kv_list.append(cache[last_block][:remained])
kv_cache = torch.cat(kv_list, dim=0)

# 优化后: 1 次 index_select，view 内联在 cat 参数中
full_blocks = cache.index_select(0, device_block_indices)                   # [N, block_size, D]
last_block = cache[last_block_idx][:remained]                               # [remained, D]
kv_cache = torch.cat([full_blocks.view(-1, full_blocks.shape[-1]), last_block], dim=0)
```

#### 路径 2：split（enable_mla_split_kv_kr=True）

KV cache 是 tuple `(kv_c_cache, k_pe_cache)`。

**kv_c 部分**（`cache[0]`，shape `[num_blocks, block_size, D1]`）：

```python
# 优化前: N 个 select + squeeze
kv_c_list = [cache[0][idx].squeeze(1) for idx in block_table_cpu[:][:block_num]]

# 优化后: 1 次 index_select + squeeze
kv_c_full = cache[0].index_select(0, device_block_indices).squeeze(1)
kv_c_last = cache[0][last_block_idx][:remained].squeeze(1)
kv_c_normed = torch.cat([
    kv_c_normed[:offset],
    kv_c_full.view(-1, kv_c_full.shape[-1]),
    kv_c_last,
    kv_c_normed[offset:]
], dim=0)
```

**k_pe 部分**（`cache[1]`，shape `[num_blocks, block_size, D2]`）：

```python
# 优化前: N 个 select
kpe_list = [cache[1][idx] for idx in block_table_cpu[:][:block_num]]

# 优化后: 1 次 index_select
kpe_full = cache[1].index_select(0, device_block_indices)
kpe_last = cache[1][last_block_idx][:remained]
k_pe = torch.cat([
    k_pe[:offset],
    kpe_full.view(-1, *kpe_full.shape[2:]),
    kpe_last,
    k_pe[offset:]
], dim=0)
```

#### Last Block 处理

最后一个 block 可能不满（只有 `chunked_prefill_remained_token_num` 个 token），保持单次 `cache[last_block_idx][:remained]` 处理（只有 1 次 `aten::select`，无需 batch 优化）。`last_block_idx` 用 `block_table_cpu[...].item()` 取得 Python int，不触发 device sync（数据本就在 CPU 上）。

---

## 四、NPU 踩坑：Tensor 内部格式

### 问题

NPU 上 `index_select` 对 3D tensor 操作后可能产出 `NCL` 内部格式（NPU 的 layout 优化），而 `kv_c_normed` / `k_pe` 等 tensor 是标准 `ND` 格式。`torch.cat` 在 NPU 上要求所有输入格式一致，混合 `NCL` 和 `ND` 会触发：

```
EZ1001: Format of tensors should be equal
```

### 尝试的解决方法

| 方法 | 效果 | 是否可行 |
|------|------|---------|
| `.view(-1, D).contiguous()` | 先 reshape 再强制 copy 为 ND | NPU 上 `.view()` 不改变内部格式，`.contiguous()` 在某些情况下仍保留 NCL |
| **`.squeeze(1)` + `.view()` 内联在 cat 参数中** | squeeze 会正确触发格式转换，view 延迟到 cat 内部处理 | **已验证可行** |

最终采用 `.squeeze()` + `.view()` 内联的方式。

---

## 五、算子对比

| | 原方案 (for 循环) | 优化方案 (index_select) |
|---|---|---|
| **Host 操作** | N 次 Python 循环 + N 次 dispatch | 1 次 device tensor 切片 + 1 次 dispatch |
| **Device kernel** | N 个 `aten::select` | 1 个 `index_select` |
| **H2D 传输** | 无（但 N 次 host-device 交互） | **零**（indices 已在 device 上） |
| **Host 耗时** | O(N) × (Python loop + dispatch overhead) | O(1) |
| **Device 空闲等待** | 严重（每个小 kernel 间等 host 下发） | 无 |

> N = `chunked_prefill_context_len // block_size`，128K 时 N=1024

---

## 六、正确性保证

### 6.1 数学等价性

优化前后执行相同的索引操作，只是将 N 次逐个 gather 合并为 1 次批量 gather，结果 **bit-exact 一致**（不涉及浮点计算顺序变化）。

实测 prefill/decode 的 logits 输出与 master 版本 **md5 完全对齐**。

### 6.2 零 H2D 传输

- `index_select` 的 indices 直接使用 device 端的 `block_table` 切片（`.long()` 转换也在 device 上执行），不产生任何 H2D 传输
- 唯一使用 CPU 的是 `last_block_idx = block_table_cpu[...].item()`——从已在 CPU 上的 tensor 取一个 Python int，不触发 device sync

### 6.3 边界场景

| 边界场景 | 分析 |
|---------|------|
| `block_size=128`（正常配置） | `squeeze(1)` 是 no-op（dim=1 大小为 128 ≠ 1），无 shape 风险 |
| `block_size=1`（理论边界） | `squeeze(1)` 会降维，但 `.view(-1, D)` 仍正确还原为 2D |
| `block_num=0`（首轮或极短 context） | `full_block_indices` 为空，`index_select` 返回 `[0, block_size, D]`，cat 只保留 `last_block` |
| `remained=0`（context 恰为 block_size 整数倍） | `block_indices[block_num]` 访问下一个 block slot——原代码固有行为 |

---

## 七、性能数据

### 测试环境

- **模型**: MOE-26B (MLA + EP64)
- **硬件**: NPU 910C
- **输入**: 64K prompt, chunk_size=256, 共 256 轮 chunked prefill
- **代码**: `ares/engines/vllm_plugins/ascend/attention/ares_mla_v1.py`

### 端到端 Prefill 耗时对比

| 配置 | 64K Prefill 累计耗时 | 相比 Baseline |
|------|---------------------|--------------|
| Baseline (EP64) | 368,580 ms | — |
| **+ KV Cache 批量 Gather** | **349,233 ms** | **-5.2% (-19.3s)** |
| + Gather + All-to-All | 320,357 ms | -13.1% |
| + Gather + A2A + Absorb | 255,136 ms | -30.8% |
| + Gather + A2A + Absorb + mla_prolog融合 | 212,651 ms | -42.3% |

### Timeline 对比

**优化前**：for 循环下发的 `aten::select` 操作造成 host bound，阻塞后续算子下发（509 次 select，累计 5.37ms）

![优化前 Timeline — 大量 aten::select 造成 host bound](/images/kv-gather/before_opt.png)

**优化后**：单次 `aten::index_select` 完成批量 gather，device 端连续执行，无 host bound 间隙

![优化后 Timeline — index_select 一次完成](/images/kv-gather/after_opt.png)

---

## 八、代码改动总览

### 开关控制

通过 `--enable_batch_kv_gather` 参数控制，默认关闭：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ... \
    --enable-chunked-prefill \
    --enable-batch-kv-gather \
    ...
```

### 改动文件

| 文件 | 改动 | 行号 |
|------|------|------|
| `ares/engines/_vllm_0_8_5/config.py` | 新增 `ParallelConfig.enable_batch_kv_gather` 字段 | L1761-1763 |
| `ares/engines/_vllm_0_8_5/engine/arg_utils.py` | 新增命令行参数并传递到 `ParallelConfig` | L363, L1233 |
| `ares/engines/vllm_plugins/ascend/attention/ares_mla_v1.py` | 实现批量 gather 逻辑 | L652-676 |

### config.py（+4 行）

```python
enable_batch_kv_gather: bool = False
"""Enable batch index_select to gather KV cache blocks for chunked prefill,
avoiding per-block aten::select host overhead."""
```

### ares_mla_v1.py 核心改动（+45 行，-15 行）

- 构造函数读取 `enable_batch_kv_gather` 配置
- `_forward_prefill` 中以 `if self.enable_batch_kv_gather` 分支实现批量 gather
- 原有 for 循环代码保留在 `else` 分支

---

## 面试讲述框架

::: details 面试时怎么讲这个优化？（2-3 分钟版本）

**1. 背景（30s）**

"我在做 MOE 大模型推理优化，模型用了 Chunked Prefill——长 prompt 被拆成多轮处理。从第 2 轮开始，每轮需要从 KV Cache 中 gather 之前已经缓存好的 KV blocks。"

**2. 问题定位（30s）**

"我通过 Profiling 发现 prefill attention 前面有大量密集的 `aten::select` 算子，数量和 context 长度线性相关（128K 时有 1024 个）。根因是 KV gather 用了 Python for 循环逐 block 索引 device tensor，每次循环 host 都要做 Python 解释 + PyTorch dispatch，device 上的小 kernel 执行极快但要一直等 host 下发下一个——典型的 Host Bound。"

**3. 方案（45s）**

"核心思路很简单：把 N 次逐 block 的 for 循环替换为 1 次 `index_select` 批量 gather。关键是 indices 直接用 device 端的 block_table 切片，不需要任何 H2D 传输。Host 操作从 O(N) 降到 O(1)，device 不再空闲等待。"

"实现上有一个 NPU 上的坑：`index_select` 输出的 tensor 内部格式是 NCL，和其他 ND 格式的 tensor 做 `cat` 会报错。解决方法是用 `.squeeze()` 触发正确的格式转换。"

**4. 结果（15s）**

"64K 长文本 prefill 耗时降低 5.2%，节省约 19 秒。结果 bit-exact 等价，md5 对齐验证通过。"

:::

::: details 面试官可能追问的问题

**Q: 为什么不用 CUDA Graph 解决 kernel launch overhead？**

A: Chunked Prefill 每轮的 context_len 不同 → `block_num` 动态变化 → 不能用 CUDA Graph（要求固定 shape）。而且问题不只是 launch overhead，是 Python for 循环本身的开销。

**Q: index_select 会不会比 N 个 select 加起来慢？**

A: 不会。N 个 select 是 N 次独立的小 kernel，每次只 copy 一个 block，之间有 host 空隙。index_select 是一次性批量 gather，device 端连续执行，内存访问模式也更友好（连续 indices → 更好的 cache 利用）。

**Q: 为什么 last_block 不一起 index_select？**

A: last_block 可能不满（token 数 < block_size），需要做 `[:remained]` 切片。这只有 1 次 select，不值得单独优化，而且逻辑上更清晰。

**Q: block_table 的 device 和 cpu 版本怎么保证一致的？**

A: `BlockTable.commit()` 在每次 model forward 前调用，做 `block_table[:n].copy_(block_table_cpu[:n], non_blocking=True)`。所以 forward 时 device 端数据一定是最新的。

**Q: 这个优化对 decode 有用吗？**

A: 没有。Decode 阶段每次只生成 1 个 token，不存在 chunked prefill 的场景，也就没有需要 gather 历史 KV 的 for 循环。

:::
