# 2026-04 通义 暑期实习 - 一面

## 基本信息
- **公司**: 通义 (阿里)
- **岗位**: 暑期实习
- **轮次**: 一面 (技术面)
- **考点覆盖**: CUDA 内存层次、Shared Memory、Pinned/Unified Memory、同步机制、推理性能分析、NUMA 优化

## 考点标签

<span class="tag tag-cuda">CUDA</span>
<span class="tag tag-system">系统设计</span>
<span class="tag tag-vllm">推理优化</span>

---

## 题目与回答

### Q1: CUDA 里的显存/内存有哪些类型？从大小和读写速度两个角度说

**我的回答**: 提到了 HBM（最大）、Shared Memory（较小但较快）、寄存器（最快）。但不够系统，漏了一些类型。

**自评**: <span class="score"><span class="score-dot score-ok"></span> 一般</span> — 主要类型提到了，但缺少 L1/L2 Cache、Constant Memory、Texture Memory。

**参考答案**:

#### CUDA 内存层次（从快到慢）

```
速度:   快 ──────────────────────────────────→ 慢
容量:   小 ──────────────────────────────────→ 大

┌──────────────┐
│  Registers   │  最快 ~0 cycle, 每线程私有, 每 SM ~256KB (总量)
├──────────────┤
│  L1 Cache /  │  ~28-33 cycle, 每 SM 私有
│  Shared Mem  │  L1 + Shared 共享同一物理 SRAM, A100: 192KB/SM
├──────────────┤
│  L2 Cache    │  ~200 cycle, 全局共享, A100: 40MB
├──────────────┤
│  HBM (Global │  ~400-600 cycle, 最大, A100: 80GB
│   Memory)    │  带宽 ~2TB/s (A100)
├──────────────┤
│  Host Memory │  通过 PCIe/NVLink, ~32-64GB/s
│  (CPU DRAM)  │
└──────────────┘
```

#### 各类型详解

| 内存类型 | 位置 | 大小 | 速度 | 作用域 | 生命周期 |
|---------|------|------|------|--------|---------|
| **Register** | SM 内 | ~256KB/SM | ~0 cycle | 线程私有 | 线程 |
| **Shared Memory** | SM 内 (SRAM) | 最多 ~164KB/SM | ~28 cycle | Block 内共享 | Block |
| **L1 Cache** | SM 内 (与 Shared 共享) | 和 Shared 合计 192KB | ~33 cycle | SM 私有 | 自动管理 |
| **L2 Cache** | 芯片上 | 40MB (A100) | ~200 cycle | 全局 | 自动管理 |
| **Constant Memory** | HBM (有 cache) | 64KB | cache hit ~4 cycle | 全局只读 | 程序 |
| **Texture Memory** | HBM (有 cache) | - | cache hit 快 | 全局只读 | 程序 |
| **Global Memory (HBM)** | 芯片外 | 80GB (A100) | ~400+ cycle | 全局 | 程序 |
| **Host Memory** | CPU 侧 | 几十~几百 GB | PCIe ~32GB/s | CPU | 程序 |

::: tip 读写速度排序（面试快答版）
**Register > Shared Memory ≈ L1 Cache > L2 Cache > Global Memory (HBM) > Host Memory**

带宽排序：Registers 内部 >> Shared Memory ~19TB/s >> L2 ~5TB/s >> HBM ~2TB/s >> PCIe ~64GB/s
:::

---

### Q2: Shared Memory 和 Global Memory 的主要区别？

**我的回答**: 提到了物理位置不同，但把 Shared Memory 说成"线程私有"（错误），应该是 Block 级共享。Global Memory 可被所有线程和 Host 访问。

**自评**: <span class="score"><span class="score-dot score-bad"></span> 差</span> — Shared Memory 的作用域说错了。

**参考答案**:

| 对比维度 | Shared Memory | Global Memory (HBM) |
|---------|--------------|---------------------|
| **物理位置** | 芯片上 (on-chip SRAM)，在 SM 内部 | 芯片外 (off-chip HBM/DRAM) |
| **大小** | ~164KB/SM (A100 最大配置) | 40-80GB |
| **延迟** | ~28 cycle | ~400+ cycle（**差 ~15x**） |
| **带宽** | ~19 TB/s (理论) | ~2 TB/s |
| **作用域** | **Block 内所有线程共享** | 所有线程 + Host 可访问 |
| **生命周期** | Block 结束即释放 | 程序级，手动管理 (cudaMalloc/cudaFree) |
| **使用方式** | `__shared__` 声明，或 dynamic shared memory | `cudaMalloc` 分配，kernel 中直接访问 |
| **可编程性** | 程序员显式管理（软件管理的 cache） | 硬件自动通过 L1/L2 cache |

**核心区别一句话**：Shared Memory 是在 SM 内部的快速 SRAM，Block 内线程共享；Global Memory 是芯片外的大容量 HBM，所有线程可访问但慢得多。

```cuda
__global__ void kernel() {
    __shared__ float smem[256];   // Block 内所有线程共享
    
    // 典型用法：从 Global Memory 加载到 Shared Memory，减少重复访问
    smem[threadIdx.x] = global_data[blockIdx.x * 256 + threadIdx.x];
    __syncthreads();  // 确保所有线程都写完
    
    // 后续计算都从 smem 读取，快 15x
    float val = smem[threadIdx.x] + smem[255 - threadIdx.x];
}
```

---

### Q3: Shared Memory 和 Global Memory 的生命周期？

**我的回答**: Shared Memory 跟 Block 绑定（正确）。Global Memory 跟 kernel 甚至整个程序（基本正确）。

**自评**: <span class="score"><span class="score-dot score-ok"></span> 一般</span> — 方向对。

**参考答案**:

- **Shared Memory**：生命周期 = **Block 的生命周期**。Block 执行完毕，其 Shared Memory 自动释放，其他 Block 不可访问
- **Global Memory**：生命周期 = **程序级**。由 `cudaMalloc` 分配，直到 `cudaFree` 释放。跨 kernel 调用持续存在，Host 也可通过 `cudaMemcpy` 访问
- **Local Memory**：生命周期 = 线程。寄存器溢出 (register spill) 时自动使用，物理上在 Global Memory 但被 L1/L2 缓存

---

### Q4: Pinned Memory 和 Pageable Memory 的区别？

**我的回答**: Pinned Memory 不需要经过操作系统，GPU 可以直接通过 DMA 访问。Pageable Memory 需要经过 CPU 内核态访问。

**自评**: <span class="score"><span class="score-dot score-ok"></span> 一般</span> — DMA 说对了，但"不经过操作系统"的表述不够准确。

**参考答案**:

#### Pageable Memory（可分页内存，默认）

```c
float* h_data = (float*)malloc(N * sizeof(float));  // 或 new
```

- 由 OS 管理，可能被**换出到磁盘 (swap)**
- GPU **不能直接通过 DMA 访问**（因为物理地址可能随时变）
- 数据传输时，CUDA driver 必须先把数据**拷贝到一个临时的 pinned buffer**，再 DMA 到 GPU

```
Pageable → [CPU 拷贝到 pinned staging buffer] → [DMA 到 GPU]
            ↑ 额外的一次拷贝，浪费带宽和时间
```

#### Pinned Memory（页锁定内存）

```c
float* h_pinned;
cudaMallocHost(&h_pinned, N * sizeof(float));  // 页锁定
```

- **锁定在物理内存中**，OS 不会将其 swap 到磁盘
- GPU 可以**直接通过 DMA 访问**（物理地址固定）
- 省去了一次 CPU 端的拷贝

```
Pinned → [DMA 直接到 GPU]  ← 只有一次传输，更快
```

#### 关键对比

| | Pageable | Pinned |
|---|---|---|
| 分配 API | `malloc` / `new` | `cudaMallocHost` / `cudaHostAlloc` |
| 物理地址 | 可变（OS 可换页） | **固定**（锁定在物理内存） |
| GPU DMA | 不能直接，需中间拷贝 | **可以直接** |
| H2D 带宽 | ~50% PCIe 理论值 | **接近 PCIe 满带宽** |
| 风险 | 无 | **过多分配会减少 OS 可用物理内存**，导致系统 swap |
| 异步传输 | 不支持 | **支持** `cudaMemcpyAsync` |

::: warning 注意
Pinned Memory 虽然快，但不能无脑用：它锁住物理内存，如果分配过多，OS 其他进程可用的物理内存减少，会导致系统频繁 swap，反而整体性能下降。
:::

---

### Q5: Unified Memory 是什么？Host 和 Device 是同一个地址空间吗？

**我的回答**: 知道 CPU 和 GPU 可以一起访问，但不确定是不是同一个地址空间。回答"是同一个地址空间"（正确）。不知道对应的 CUDA API。

**自评**: <span class="score"><span class="score-dot score-ok"></span> 一般</span> — 概念知道但细节不够。

**参考答案**:

#### Unified Memory 核心概念

Unified Memory 提供**单一的、统一的虚拟地址空间**，CPU 和 GPU 都可以通过**同一个指针**访问数据。

```c
float* data;
cudaMallocManaged(&data, N * sizeof(float));  // ← 这就是 API

// CPU 可以直接用
data[0] = 1.0f;

// GPU kernel 也可以直接用（同一个指针！）
kernel<<<grid, block>>>(data);
```

**API**：`cudaMallocManaged()`

#### 工作原理

底层通过**页面迁移 (page migration)** 实现：
- 当 CPU 访问数据时，数据自动迁移到 CPU 内存
- 当 GPU 访问数据时，数据自动迁移到 GPU 显存
- 由 CUDA runtime 的 **Unified Memory Manager** 和 OS 的 page fault 机制协作完成

```
CPU 访问 data[0]:
  if (data 在 GPU 显存) → page fault → 自动迁移到 CPU 内存 → CPU 访问

GPU kernel 访问 data[0]:
  if (data 在 CPU 内存) → page fault → 自动迁移到 GPU 显存 → GPU 访问
```

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 编程简单，不需要手动 `cudaMemcpy` | **页面迁移有开销**，频繁 CPU/GPU 交替访问会严重降低性能 |
| 可以 oversubscribe GPU 显存 | 隐式迁移不如显式传输可控 |
| 适合原型开发和端侧设备 | 不适合高性能推理/训练场景 |

#### Prefetch 优化

可以用 `cudaMemPrefetchAsync` 提前迁移，避免 page fault：
```c
cudaMemPrefetchAsync(data, N * sizeof(float), device_id, stream);
```

---

### Q6: 为什么不能无脑用 Shared Memory？会有什么问题？

**我的回答**: 提到了容量有限、竞争冲突、Bank Conflict（同一 warp 内产生，本来并行会变成串行）。

**自评**: <span class="score"><span class="score-dot score-ok"></span> 一般</span> — Bank Conflict 说对了，但还有其他重要限制没提到。

**参考答案**:

#### 问题一：容量极其有限

A100 每个 SM 的 Shared Memory 最大 **164KB**（和 L1 共享 192KB 物理 SRAM）。如果每个 Block 用太多 Shared Memory，**能同时驻留的 Block 数减少 → occupancy 下降 → SM 利用率降低**。

```
例子 (A100, 每 SM 192KB 可配):
- 每 Block 用 48KB shared mem → 每 SM 可以放 4 个 Block → occupancy 高
- 每 Block 用 96KB shared mem → 每 SM 只能放 2 个 Block → occupancy 低
- 每 Block 用 164KB shared mem → 每 SM 只能放 1 个 Block → occupancy 极低
```

#### 问题二：Bank Conflict

Shared Memory 被组织为 **32 个 bank**，每个 bank 宽 4 bytes。同一 warp 的 32 个线程同时访问 Shared Memory 时：
- 如果访问不同 bank → **无冲突，并行完成（1 cycle）**
- 如果多个线程访问**同一 bank 的不同地址** → **Bank Conflict，串行化**

```
无冲突: thread 0→bank 0, thread 1→bank 1, ..., thread 31→bank 31  ✓ 并行
冲突:   thread 0→bank 0, thread 1→bank 0, ..., → 32-way conflict  ✗ 串行32次

特例：同一 bank 同一地址 → broadcast，无冲突
```

**解决方法**：Padding（在数组声明时多加一列，错开 bank 对齐）
```cuda
__shared__ float smem[32][33];  // 33 而非 32，避免列访问时的 bank conflict
```

#### 问题三：Occupancy 下降

Shared Memory 是**每 SM 有限的资源**。一个 Block 占用越多 Shared Memory，该 SM 能同时调度的 Block 数越少，GPU 无法通过切换 warp 来隐藏延迟。

#### 问题四：同步开销

使用 Shared Memory 通常需要 `__syncthreads()` 做屏障同步。如果 Block 内线程数多、同步频繁，同步开销本身也不可忽略。

::: tip 总结
Shared Memory 不能无脑用的三大原因：
1. **容量小** → 用多了 occupancy 下降
2. **Bank Conflict** → 访问模式不当就串行化
3. **同步开销** → `__syncthreads()` 有成本
:::

---

### Q7: Block 内所有 thread 的屏障同步怎么做？如果造成死锁怎么检测？

**我的回答**: 知道有同步 API 但记不清名字。知道不做同步会导致数据不一致（其他 warp 读到未写入的数据）。死锁检测不确定，猜测可以用 NCU 分析。

**自评**: <span class="score"><span class="score-dot score-ok"></span> 一般</span> — 概念理解对，但 API 名字没记住。

**参考答案**:

#### Block 内屏障同步

```cuda
__syncthreads();  // ← 这就是 API
```

`__syncthreads()` 的语义：
- Block 内**所有线程**必须都到达这个点，才能继续往下执行
- 保证 `__syncthreads()` 之前的**所有内存写操作**对 Block 内其他线程可见

```cuda
__shared__ float smem[256];
smem[threadIdx.x] = compute(input[threadIdx.x]);
__syncthreads();   // 确保所有线程都写完了
float result = smem[255 - threadIdx.x];  // 安全读取其他线程写的数据
```

#### 常见死锁场景

**条件分支中使用 `__syncthreads()`**：
```cuda
// ❌ 死锁！部分线程走 if，部分走 else，永远等不齐
if (threadIdx.x < 128) {
    __syncthreads();  // 只有前 128 个线程到达
} else {
    __syncthreads();  // 后 128 个线程到达另一个 syncthreads
}

// ✓ 正确：syncthreads 必须在所有线程都能到达的路径上
if (threadIdx.x < 128) {
    // do something
}
__syncthreads();  // 所有线程都到达同一个 syncthreads
```

#### 死锁检测

1. **compute-sanitizer**（NVIDIA 官方工具）：
   ```bash
   compute-sanitizer --tool synccheck ./my_program
   ```
   `--tool synccheck` 专门检测 `__syncthreads()` 的非法使用（如条件分支中的不一致调用）

2. **cuda-gdb**：CUDA 调试器，可以查看每个线程的执行位置
3. **Nsight Compute (NCU)**：性能分析工具，可以看到 warp 的执行状态

---

### Q8: Kernel 之间的同步怎么做？

**我的回答**: 提到了 CUDA Event 和 Stream 事件同步。

**自评**: <span class="score"><span class="score-dot score-ok"></span> 一般</span> — 方向对但不够详细。

**参考答案**:

#### 方式一：同一 Stream 内的隐式同步

同一 stream 中的 kernel **自动按顺序执行**，不需要额外同步：
```cuda
kernel_A<<<grid, block, 0, stream>>>();
kernel_B<<<grid, block, 0, stream>>>();  // 自动等 A 完成
```

#### 方式二：跨 Stream 同步 — CUDA Event

```cuda
cudaEvent_t event;
cudaEventCreate(&event);

// Stream 1 中记录事件
kernel_A<<<grid, block, 0, stream1>>>();
cudaEventRecord(event, stream1);   // A 完成后触发 event

// Stream 2 等待该事件
cudaStreamWaitEvent(stream2, event, 0);  // stream2 等 event 触发
kernel_B<<<grid, block, 0, stream2>>>();  // B 在 A 完成后才执行
```

#### 方式三：全局同步

```cuda
cudaDeviceSynchronize();  // CPU 等待所有 stream 的所有 kernel 完成
cudaStreamSynchronize(stream);  // CPU 等待指定 stream 完成
```

#### 对比

| 方式 | 场景 | 粒度 |
|------|------|------|
| 同 Stream 隐式 | 有依赖的连续 kernel | 最简单 |
| Event + StreamWaitEvent | 跨 Stream 依赖 | 最灵活，**GPU 端同步**，不阻塞 CPU |
| DeviceSynchronize | 全部等待 | 最重，**阻塞 CPU** |

---

### Q9: GPU SM 利用率不高，但端到端延迟高，排查思路？

**我的回答**: 看 timeline，检查 H2D 传输是否过多、CPU 同步 (`cudaDeviceSynchronize`) 是否过频繁、CPU 端数据预处理是否有问题、CPU 利用率是否有其他线程争核心。

**自评**: <span class="score"><span class="score-dot score-ok"></span> 一般</span> — 思路方向对，但不够系统。

**参考答案**:

#### 问题分析

SM 利用率低 + 延迟高 = **GPU 大部分时间在空闲**，瓶颈不在 GPU 计算，而在其他环节。

#### 系统化排查步骤

**Step 1: 抓 Timeline（Nsight Systems）**
```bash
nsys profile --trace=cuda,nvtx,osrt ./inference_server
```
在 timeline 中看 GPU 的活跃区间和空闲区间：
- GPU 空闲时 CPU 在干什么？
- Kernel 之间的 gap 有多长？

**Step 2: 检查 CPU 端瓶颈**
- **过多的显式同步**：`cudaDeviceSynchronize()` 或 `cudaStreamSynchronize()` 太频繁 → CPU 和 GPU 变成完全串行
- **数据预/后处理**：tokenization、detokenization、采样逻辑在 CPU 上太慢
- **Python GIL**：如果是 Python 服务，GIL 可能限制并发
- **CPU 利用率**：其他线程/进程争夺 CPU 核心

**Step 3: 检查数据传输**
- H2D / D2H 传输是否过多或过大
- 是否用了 Pageable Memory（应该用 Pinned Memory）
- 传输是否和计算做了 overlap

**Step 4: 检查 Kernel Launch Overhead**
- 大量小 kernel → launch overhead 累积
- 解决：CUDA Graph 批量提交

**Step 5: 检查调度/排队延迟**
- 请求在调度队列中等待时间长
- Batch 组装逻辑是否有问题（等太久才凑齐一个 batch）

```
常见根因汇总:
┌─────────────────────────────────┐
│ SM 利用率低 + 延迟高            │
├─────────────────────────────────┤
│ 1. CPU-GPU 过度同步            │ ← 最常见
│ 2. CPU 端预/后处理慢           │
│ 3. 数据传输未 overlap          │
│ 4. Kernel launch overhead      │
│ 5. 调度/batching 策略不合理    │
│ 6. 网络 I/O 延迟（如果有）     │
└─────────────────────────────────┘
```

---

### Q10: CUDA Stream 的作用？CUDA Graph 为什么能提高性能？

**我的回答**: Stream 让 kernel 不互相阻塞，不同请求的 kernel 可以独立运行。Graph 减少 launch kernel 的开销，画一个计算图只需一次 launch。

**自评**: <span class="score"><span class="score-dot score-ok"></span> 一般</span> — 基本概念对，和二面重复的知识点。

**参考答案**:

（和二面 Q1/Q2 内容类似，此处简述要点）

**CUDA Stream**：
- 同一 stream 内 kernel 按序执行，不同 stream 间可并发
- 作用：① 计算与传输 overlap ② 不同请求的 kernel 并发执行 ③ 避免不相关 kernel 互相阻塞

**CUDA Graph**：
- 将一系列 kernel 调用**录制为静态图**，一次 `graph.replay()` 提交整个图
- 好处：① **消除 kernel launch overhead**（5-10μs/kernel → 一次 launch） ② kernel 无间隙执行 ③ 驱动可以做全局优化
- 限制：不支持动态 shape、不能含控制流、不能含显存分配

---

### Q11: 推理场景是计算密集还是访存密集？如果是访存密集怎么优化？

**我的回答**: 推理主要是访存密集型。优化方法：增大 batch size、算子融合。

**自评**: <span class="score"><span class="score-dot score-ok"></span> 一般</span> — 方向对但不够全面。

**参考答案**:

#### 推理为什么是访存密集 (Memory-Bound)

特别是 **decode 阶段**：每次只处理 1 个 token（或少量 token），但要读取完整模型权重：

```
Decode: batch_size=1, seq_len=1
  - GEMM: [1, d] × [d, d] → 读权重 d² 个参数，只做 d² 次乘加
  - Arithmetic Intensity = FLOPs / Bytes = 2d² / (2d² × 2) ≈ 0.5
  - A100 需要 AI > 312 TFLOPs / 2 TB/s = 156 才能 compute-bound
  - 0.5 << 156 → 严重 memory-bound
```

#### 优化方法

**1. 增大 Batch Size**（最直接有效）
```
batch=1:  读权重 1 次, 算 1 次 → AI = 0.5
batch=32: 读权重 1 次, 算 32 次 → AI = 16  ← 大幅提升
```
Continuous Batching / Dynamic Batching 就是为此设计的。

**2. 算子融合 (Kernel Fusion)**
- 减少中间结果写回 HBM 的次数
- 典型：把 LayerNorm + Attention + Residual 融合

**3. Quantization（量化）**
- INT8/INT4 权重 → 读取数据量减半/减 4 倍 → Memory-bound 场景直接提速
- A100 INT8 Tensor Core 吞吐也更高

**4. KV Cache 优化**
- PagedAttention：减少碎片化，提高显存利用 → 能放更多 batch
- KV Cache 量化：减少 KV 的访存量

**5. FlashAttention / FlashDecoding**
- 减少 attention 计算中的 HBM 读写

---

### Q12: 多 NUMA Node 上内存密集型工作负载怎么优化？

**我的回答**: 没有实际经验，不确定怎么回答。

**自评**: <span class="score"><span class="score-dot score-bad"></span> 差</span> — 这个知识点完全没准备。

**参考答案**:

#### NUMA 架构背景

现代服务器有多个 CPU socket，每个 socket 有自己的本地内存。访问**本地内存快**，访问**远端内存慢**（通过 QPI/UPI 互连）：

```
┌──────────────────┐    UPI/QPI     ┌──────────────────┐
│   NUMA Node 0    │←─────────────→│   NUMA Node 1    │
│ CPU cores 0-31   │   ~100ns 延迟  │ CPU cores 32-63  │
│ Local DRAM 256GB │               │ Local DRAM 256GB │
│  (~80ns 延迟)    │               │  (~80ns 延迟)    │
└──────────────────┘               └──────────────────┘

Node 0 访问 Node 0 内存: ~80ns  (local)
Node 0 访问 Node 1 内存: ~140ns (remote, 慢 ~1.7x)
```

#### 内存密集型负载的 NUMA 问题

如果线程运行在 Node 0 的 CPU 上，但数据分配在 Node 1 的内存上 → **所有内存访问都要跨 NUMA**，带宽减半、延迟增大。

#### 优化方法

**1. NUMA 绑定（最重要）**

```bash
# 将进程绑定到 Node 0 的 CPU 和内存
numactl --cpunodebind=0 --membind=0 ./my_workload

# 查看 NUMA 拓扑
numactl --hardware
```

确保**线程在哪个 Node 的 CPU 上运行，数据就分配在哪个 Node 的内存上**。

**2. First-Touch 策略**

Linux 默认 first-touch：内存页在哪个 NUMA Node 的 CPU 首次访问，就分配在哪个 Node。

```c
// 不好：主线程（Node 0）初始化所有数据 → 全分配在 Node 0
for (int i = 0; i < N; i++) data[i] = 0;  // 全在 Node 0

// 好：让每个线程初始化自己的部分 → 数据就近分配
#pragma omp parallel for
for (int i = 0; i < N; i++) data[i] = 0;  // 分散到各 Node
```

**3. 内存交织 (Interleave)**

```bash
# 将内存页轮流分配到所有 Node（适合无法预知访问模式的场景）
numactl --interleave=all ./my_workload
```

**4. 避免跨 NUMA 的数据共享**

- 尽量让每个 NUMA Node 上的线程只访问本地数据
- 数据分区 + 线程绑定
- 减少 false sharing（不同 Node 的线程写同一 cache line）

**5. 大页 (Huge Pages)**

```bash
# 使用 2MB 大页，减少 TLB miss
echo 1024 > /proc/sys/vm/nr_hugepages
```
内存密集型负载会频繁触发 TLB miss（默认 4KB 页太小），大页可以减少 TLB miss。

::: tip 面试快答
NUMA 优化三板斧：
1. **numactl 绑定** — 线程和数据在同一 Node
2. **并行 first-touch** — 数据就近分配
3. **大页** — 减少 TLB miss
:::

---

## 整体复盘

### 答得好的
- CUDA 内存层次的基本概念（Register > Shared > HBM）
- Bank Conflict 的理解
- CUDA Stream/Graph 基本概念
- 排查 GPU 利用率低的思路方向

### 答得差的
- **Shared Memory 作用域说错**：说成"线程私有"，实际是 Block 级共享
- **Unified Memory**：知道概念但不知道 API (`cudaMallocManaged`)
- **死锁检测**：不知道 `compute-sanitizer --tool synccheck`
- **NUMA 优化**：完全没准备

### 需要补的知识
1. **CUDA 内存完整体系**：把 Register / Shared / L1 / L2 / Global / Constant / Texture / Pinned / Unified 的对比表背下来
2. **CUDA 同步 API**：`__syncthreads()`, `cudaEventRecord`, `cudaStreamWaitEvent`, `cudaDeviceSynchronize` 要能随口说出
3. **Shared Memory 深入**：Bank Conflict 的 32-bank 结构、Padding 技巧、occupancy 的影响
4. **NUMA 体系结构**：numactl 使用、first-touch 策略、内存交织、大页
5. **`compute-sanitizer`**：CUDA 调试/检测工具链

### 下次改进
- CUDA 编程的 API 名字要记清楚，面试时说不出具体 API 会显得不扎实
- 系统层面的知识要扩展（NUMA、CPU 缓存一致性、大页），推理性能优化不只是 GPU
- 排查类问题要建立标准化的步骤框架（Step 1/2/3），而不是想到什么说什么
