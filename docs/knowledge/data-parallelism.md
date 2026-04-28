# Data Parallelism

::: tip 待完善
本页为骨架，后续补充详细内容。
:::

## 核心概念

- 每个 GPU 持有完整模型副本，分到不同数据
- Forward + Backward 后做 AllReduce 同步梯度
- 扩展性好但显存效率低（每张卡都存全量参数）

## ZeRO 优化

| ZeRO Stage | 切分内容 | 显存节省 |
|------------|---------|---------|
| Stage 1 | Optimizer States | ~4x |
| Stage 2 | + Gradients | ~8x |
| Stage 3 | + Parameters | ~Nd (N=GPU数) |

## FSDP vs DDP

- DDP: 每卡全量参数，梯度 AllReduce
- FSDP: 参数/梯度/优化器状态均分片，按需 all-gather

## 面试要点

- AllReduce 通信量与 GPU 数量的关系
- 梯度累积如何减少通信
- 通信与计算 overlap 的实现
