# DSA Indexer FloorDiv AICPU 消除优化

## 问题背景

DSA Indexer 在 DCP（Decode Context Parallelism）模式下，需要将全局 `topk_indices` 转换为本 rank 的局部 index。原始代码使用 int32 取模和整除：

```python
def _compute_local_indices(self, topk_indices: torch.Tensor) -> torch.Tensor:
    if self.dcp_size == 1:
        return topk_indices
    keep = (topk_indices != -1) & ((topk_indices % self.dcp_size) == self.dcp_rank)  # FloorMod(int32)
    topk_indices = torch.where(keep, topk_indices // self.dcp_size, -1)               # FloorDiv(int32)
    _, order = torch.sort((topk_indices == -1).float(), dim=-1)
    return torch.gather(topk_indices, -1, order)
```

**问题**：NPU 910C 上 int32 的 `%`（FloorMod）和 `//`（FloorDiv）没有 AICORE 实现，退化到 **AICPU** 执行（串行、慢、有调度切换开销）。

---

## NPU 执行单元：AICORE vs AICPU

| | AICORE | AICPU |
|---|---|---|
| 定位 | 专用计算加速核心（类比 GPU SM） | 通用兜底（类比退化到 CPU） |
| 擅长 | matmul、向量运算、float16/32 | AICORE 不支持的 op/dtype |
| 性能 | 高吞吐、流水线并行 | 串行执行，慢 |

算子走 AICPU 意味着：计算本身慢 + AICORE→AICPU 调度切换开销。

---

## 优化方案：数学等价变换

将 int32 整除/取模转换为 float32 浮点运算（有 AICORE 实现）：

```python
# x // n  →  (float(x) * (1.0/n)).to(int32)
# x % n   →  float(x) - quotient_f * n
```

```python
def _compute_local_indices(self, topk_indices: torch.Tensor) -> torch.Tensor:
    if self.dcp_size == 1:
        return topk_indices
    if self.dcp_size & (self.dcp_size - 1) == 0:
        # dcp_size 是 2 的幂 → float32 路径（AICORE）
        indices_f = topk_indices.float()
        inv_dcp = 1.0 / self.dcp_size
        quotient = (indices_f * inv_dcp).to(torch.int32)
        remainder = indices_f - quotient.float() * self.dcp_size
        keep = (topk_indices != -1) & (remainder == float(self.dcp_rank))
        topk_indices = torch.where(keep, quotient, -1)
    else:
        # 非 2 次幂 fallback 回原始 int32 路径（AICPU，保证正确性）
        keep = (topk_indices != -1) & ((topk_indices % self.dcp_size) == self.dcp_rank)
        topk_indices = torch.where(keep, topk_indices // self.dcp_size, -1)
    _, order = torch.sort((topk_indices == -1).float(), dim=-1)
    return torch.gather(topk_indices, -1, order)
```

### 算子级对比

| 原始 | 设备 | 优化后 | 设备 |
|------|------|--------|------|
| `FloorMod(int32)` | AICPU | `Cast(i32→f32)` + `Mul(f32)` + `Sub(f32)` | AICORE |
| `FloorDiv(int32)` | AICPU | `Cast(i32→f32)` + `Mul(f32)` + `Cast(f32→i32)` | AICORE |

---

## 精度保证

### 为什么 2 的幂是安全的

`dcp_size=8` 时，`1.0/8 = 0.125 = 2^-3`，在 float32 中**精确表示**（二进制小数 `0.001`），整条计算链零舍入误差。

### 为什么非 2 次幂不行

`1.0/3 = 0.333...` 在 float32 中有舍入，可能导致商偏差 ±1 → index 被错误分配。所以非 2 次幂 fallback 回原始路径。

### 为什么 int→float 转换不丢精度

`topk_indices` 值域为 `[0, num_blocks)`，远小于 float32 精确整数上限 `2^24 = 16,777,216`。

---

## 本质理解

这个优化**没有写任何自定义 kernel**，只是在 Python 层换了一种算子组合来表达等价数学计算：

```
原来：调用 1 个算子（FloorDiv int32），但只有 AICPU 实现
优化：调用 3~4 个算子（Cast + Mul + Cast），每个都有 AICORE 实现
```

类比：走一条土路（AICPU）→ 绕一小段高速公路（AICORE）。虽然 op 数量多了，但 AICORE 流水线执行的总延迟远低于 2 次 AICPU 调度开销。

---

## 面试讲述要点

::: details 面试时怎么提？（15 秒带过）

"另外还做了一个 NPU 上的算子级优化——indexer 的 DCP 分片计算用了 int32 整除和取模，在 NPU 上没有 AICORE 实现退化到 AICPU。我把它转成 float32 乘法走 AICORE，利用 dcp_size 是 2 的幂时倒数精确可表示这个特性保证正确性。"

:::

::: details 面试官可能追问

**Q: 为什么不直接给 int32 FloorDiv 写一个 AICORE kernel？**

A: 能绕过去就不需要写。写 Ascend C 自定义 kernel 开发成本高、需要适配 CANN 算子注册流程。这里用现有算子组合就能解决，几行 Python 搞定。

**Q: 这种优化怎么发现的？**

A: Profiling timeline 上看到 SFA 前有个明显的 AICPU 算子（FloorDiv）耗时，说明有 op 没在 AICORE 上跑。查 CANN 文档确认 int32 整除没有 AICORE 实现。

:::
