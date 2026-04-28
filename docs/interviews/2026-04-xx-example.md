# 2026-04-xx 示例公司 - 一面

## 基本信息
- **公司**: 示例公司
- **岗位**: AI Infra 实习
- **轮次**: 一面 (技术面)
- **时长**: 60 min
- **面试官**: 技术 leader

## 考点标签

<span class="tag tag-tp">TP</span>
<span class="tag tag-cuda">CUDA</span>
<span class="tag tag-vllm">vLLM</span>

## 题目与回答

### Q1: Tensor Parallelism 每层有几次通信？分别在哪里？

**我的回答**: 每层 2 次 AllReduce。Attention 的 output projection 之后一次，MLP 的 down_proj 之后一次。QKV 投影用 ColumnParallel 不需要通信，Wo 和 down_proj 用 RowParallel 需要 AllReduce SUM。

**参考答案**: 同上，可以补充通信量分析: `2(N-1)/N * batch * seq * hidden * dtype_bytes` per AllReduce (Ring)

**自评**: <span class="score"><span class="score-dot score-good"></span> 好</span>

### Q2: CUDA 中 shared memory 的 bank conflict 怎么解决？

**我的回答**: 答得不完整，只说了 padding 的方法。

**参考答案**: 
1. Padding: 数组宽度加 1，错开 bank 映射
2. 调整访存 pattern，让同一 warp 内的线程访问不同 bank
3. 使用 `__shfl` 等 warp-level primitives 避免经过 shared memory

**自评**: <span class="score"><span class="score-dot score-ok"></span> 一般</span>

### Q3: vLLM 的 PagedAttention 解决了什么问题？

**我的回答**: 解决 KV Cache 内存碎片问题，借鉴 OS 虚拟内存的分页思想...

**参考答案**: KV Cache 按 block 分配，逻辑连续物理不连续，消除 internal fragmentation 和 external fragmentation，支持动态序列长度，还能做 copy-on-write 实现 beam search。

**自评**: <span class="score"><span class="score-dot score-good"></span> 好</span>

## 整体复盘

- **答得好的**: TP 切分和通信讲得清楚，PagedAttention 理解到位
- **答得差的**: CUDA shared memory 细节不够扎实
- **需要补的知识**: CUDA bank conflict、warp primitives
- **下次改进**: 复习 CUDA 编程基础，做几个 kernel 练习
