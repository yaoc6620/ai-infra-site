# AI Infra 知识库

按专题分类复习，每个页面包含：核心概念、可视化图解、代码示例、面试高频问答。

<div class="card-grid">

<a href="/ai-infra-site/knowledge/tensor-parallelism" class="card" style="text-decoration: none; color: inherit;">
<h3>Tensor Parallelism</h3>
<p>Column/Row Parallel 切分、AllReduce 通信、Megatron-LM 风格实现</p>
</a>

<a href="/ai-infra-site/knowledge/pipeline-parallelism" class="card" style="text-decoration: none; color: inherit;">
<h3>Pipeline Parallelism</h3>
<p>层间切分、微批次调度、GPipe vs 1F1B、气泡率分析</p>
</a>

<a href="/ai-infra-site/knowledge/data-parallelism" class="card" style="text-decoration: none; color: inherit;">
<h3>Data Parallelism</h3>
<p>DDP、FSDP/ZeRO、梯度同步、通信与计算重叠</p>
</a>

<a href="/ai-infra-site/knowledge/vllm/" class="card" style="text-decoration: none; color: inherit;">
<h3>vLLM 推理引擎</h3>
<p>架构总览、Scheduler、KV Cache、CUDA Graph、Model Runner、投机解码</p>
</a>

<a href="/ai-infra-site/knowledge/attention-optimization" class="card" style="text-decoration: none; color: inherit;">
<h3>Attention 优化</h3>
<p>FlashAttention、MQA/GQA/MLA、Speculative Decoding</p>
</a>

<a href="/ai-infra-site/knowledge/cuda-basics" class="card" style="text-decoration: none; color: inherit;">
<h3>CUDA 编程基础</h3>
<p>Grid/Block/Thread、Shared Memory、Warp 调度、Kernel 优化</p>
</a>

</div>
