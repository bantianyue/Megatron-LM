# Megatron Core (MCore) 并行策略实现分析

## 概述

本文档深入分析 Megatron-LM 中各种并行策略的实现原理，提取关键代码片段说明其工作机制。

**MCore 位置**: `megatron/core/`

---

## 一、张量并行 (TP) 实现原理

### 1.1 核心思想

张量并行通过将权重矩阵按维度分割到多个 GPU，实现大模型的分布式计算。

**数学基础**:
```
Y = XA + b

列并行: A = [A_1, A_2, ..., A_p]  (按列分割)
行并行: A = [A_1; A_2; ...; A_p]  (按行分割)
```

### 1.2 ColumnParallelLinear 实现

**位置**: `megatron/core/tensor_parallel/layers.py:751`

#### 权重初始化

```python
class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, *, config: ModelParallelConfig, ...):
        super().__init__()

        # 获取 TP 进程组
        self.tp_group = get_tensor_model_parallel_group_if_none(self.tp_group)
        world_size = get_pg_size(self.tp_group)
        rank = get_pg_rank(self.tp_group)

        # 关键：按列分割权重矩阵
        self.output_size_per_partition = divide(output_size, world_size)

        # 权重形状: [output_size_per_partition, input_size]
        self.weight = Parameter(
            torch.empty(
                self.output_size_per_partition,
                self.input_size,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        )
```

**关键点**:
- 每个 rank 只持有 `output_size / tp_size` 的权重
- 权重按列（第二维）分割：`A = [A_1, A_2, ..., A_p]`

#### 前向传播

```python
def forward(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None, ...):
    if weight is None:
        weight = self.weight

    bias = self.bias if not self.skip_bias_add else None

    # 处理输入: 如果需要 all-reduce 梯度，则复制输入到 TP 区域
    if self.allreduce_dgrad or self.sequence_parallel or self.explicit_expert_comm:
        input_parallel = input_  # 已经在 TP 区域
    else:
        # 关键：复制输入到所有 TP rank（反向时 all-reduce 梯度）
        input_parallel = copy_to_tensor_model_parallel_region(input_, group=self.tp_group)

    # 矩阵乘法: Y_i = X * A_i
    output_parallel = linear_with_grad_accumulation_and_async_allreduce(
        input_parallel, weight, bias, ...
    )

    # 可选：all-gather 输出
    if self.gather_output:
        output = gather_from_tensor_model_parallel_region(output_parallel)
    else:
        output = output_parallel

    return output, bias
```

**通信模式**:
- **前向**: 无通信（或 gather_output 时 all-gather）
- **反向**: all-reduce 梯度

---

### 1.3 RowParallelLinear 实现

**位置**: `megatron/core/tensor_parallel/layers.py:1088`

#### 权重初始化

```python
class RowParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, *, config: ModelParallelConfig, ...):
        super().__init__()

        self.tp_group = get_tensor_model_parallel_group_if_none(self.tp_group)
        world_size = get_pg_size(self.tp_group)
        rank = get_pg_rank(self.tp_group)

        # 关键：按行分割权重矩阵
        self.input_size_per_partition = divide(input_size, world_size)

        # 权重形状: [output_size, input_size_per_partition]
        self.weight = Parameter(
            torch.empty(
                self.output_size,
                self.input_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        )
```

**关键点**:
- 权重按行（第一维）分割：`A = [A_1; A_2; ...; A_p]`
- `input_is_parallel`: 输入是否已经是 TP 分割的（如来自 ColumnParallelLinear）

#### 前向传播

```python
def forward(self, input_):
    # 处理输入：如果输入未分割，则按序列维度分割
    if self.input_is_parallel:
        input_parallel = input_  # 已经是 TP 分割的
    else:
        # 关键：scatter 输入到 TP rank
        input_parallel = scatter_to_tensor_model_parallel_region(input_, group=self.tp_group)

    # 矩阵乘法: Y_partial_i = X_i * A_i
    output_parallel = linear_with_grad_accumulation_and_async_allreduce(
        input_parallel, self.weight, ...
    )

    # 关键：all-reduce 或 reduce-scatter 输出
    if self.sequence_parallel:
        # SP 模式：reduce-scatter
        output = reduce_scatter_to_sequence_parallel_region(output_parallel, group=self.tp_group)
    else:
        # 标准 TP 模式：all-reduce
        output = reduce_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)

    # 添加 bias（bias 不并行）
    if not self.skip_bias_add:
        output = (output + self.bias) if self.bias is not None else output

    return output, bias
```

**通信模式**:
- **前向**: scatter 输入 + all-reduce 输出
- **反向**: scatter 梯度 + all-reduce 梯度

---

### 1.4 通信原语实现

**位置**: `megatron/core/tensor_parallel/mappings.py`

#### All-Reduce (标准 TP)

```python
class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_, group):
        # 前向：all-reduce
        return _reduce(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向：无通信（梯度已经 all-reduce）
        return grad_output, None

def _reduce(input_, group):
    """All-reduce the input tensor across model parallel group."""
    if group.size() == 1:
        return input_
    # 关键：all-reduce
    torch.distributed.all_reduce(input_.contiguous(), group=group)
    return input_
```

#### Scatter (RowParallel 输入)

```python
class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_, group):
        # 前向：split
        ctx.group = group
        return _split_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向：all-gather 梯度
        return _gather_along_last_dim(grad_output, ctx.group), None

def _split_along_last_dim(input_, group):
    """Split the tensor along its last dimension and keep the corresponding slice."""
    world_size = group.size()
    # 沿最后一维分割
    input_list = split_tensor_along_last_dim(input_, world_size)
    rank = group.rank()
    output = input_list[rank].contiguous()
    return output
```

#### Copy (ColumnParallel 输入)

```python
class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_, group):
        # 前向：无操作（只是标记）
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # 反向：all-reduce 梯度
        return _reduce(grad_output, ctx.group), None
```

#### Reduce-Scatter (序列并行)

```python
def _reduce_scatter_along_first_dim(input_, group, ...):
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = group.size()
    if world_size == 1:
        return input_

    # 重塑为 [seq * batch, hidden]
    input_ = input_.reshape(-1, input_.shape[-1])

    # 沿 hidden 维度 split
    split_tensors = torch.split(input_, input_.shape[-1] // world_size, dim=1)

    # concat 到 seq 维度
    concat_tensor = torch.cat(split_tensors, dim=0)

    # 沿第一维 reduce-scatter
    output = _reduce_scatter_along_first_dim(concat_tensor, group=group)
    output = output.reshape(target_shape)

    return output
```

---

### 1.5 TP 组合模式

#### MLP: ColumnParallel + RowParallel

```python
class MLPBlock(nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size, config):
        super().__init__()

        # FC1: 列并行
        # 输入: [batch, seq, hidden] -> 输出: [batch, seq, ffn_hidden/tp]
        self.fc1 = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=ffn_hidden_size,
            config=config,
        )

        # FC2: 行并行
        # 输入: [batch, seq, ffn_hidden/tp] -> 输出: [batch, seq, hidden]
        self.fc2 = RowParallelLinear(
            input_size=ffn_hidden_size,
            output_size=hidden_size,
            config=config,
            input_is_parallel=True,  # 关键：来自 ColumnParallelLinear
        )

    def forward(self, x):
        # FC1: 每个 rank 计算自己列的部分
        hidden = self.fc1(x)  # [batch, seq, ffn_hidden/tp]
        hidden = act(hidden)

        # FC2: 行并行 + all-reduce
        output = self.fc2(hidden)  # [batch, seq, hidden]
        return output
```

**通信流程**:
```
x (所有 rank) --> fc1 (ColumnParallel)
                --> split input
                --> local gemm
                --> act
                --> fc2 (RowParallel)
                --> local gemm
                --> all-reduce output
                --> y (所有 rank)
```

---

## 二、序列并行 (SP) 实现原理

### 2.1 核心思想

SP 将序列维度分割到 TP rank，减少激活内存占用。

**配置**:
```python
config = ModelParallelConfig(
    tensor_model_parallel_size=8,
    sequence_parallel=True,  # 启用 SP
)
```

### 2.2 SP 通信模式

#### 标准 TP vs 序列并行

**标准 TP**:
```
每个 rank: [batch, seq, hidden]
激活: 每个 rank 存储完整序列
```

**序列并行**:
```
每个 rank: [batch, seq/tp, hidden]
激活: 每个 rank 只存储 1/tp 序列
```

#### RowParallelLinear with SP

```python
# megatron/core/tensor_parallel/layers.py:1289
def forward(self, input_):
    # 输入已经是 SP 分割的

    # 矩阵乘法
    output_parallel = self._forward_impl(input_parallel, self.weight, ...)

    # SP 模式：reduce-scatter（而不是 all-reduce）
    if self.sequence_parallel:
        output = reduce_scatter_to_sequence_parallel_region(output_parallel, group=self.tp_group)
    else:
        # 标准 TP：all-reduce
        output = reduce_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)

    return output, bias
```

**关键区别**:
- **标准 TP**: all-reduce 输出（所有 rank 获得完整输出）
- **SP**: reduce-scatter 输出（每个 rank 获得 1/tp 的输出）

### 2.3 SP 通信可视化

```
标准 TP:
Rank 0: [batch, seq, h/2] --all-reduce--> [batch, seq, h]
Rank 1: [batch, seq, h/2] --all-reduce--> [batch, seq, h]

序列并行:
Rank 0: [batch, seq/2, h/2] --reduce-scatter--> [batch, seq/2, h]
Rank 1: [batch, seq/2, h/2] --reduce-scatter--> [batch, seq/2, h]
```

---

## 三、流水线并行 (PP) 实现原理

### 3.1 核心思想

PP 将模型层分割到多个 GPU，每个 GPU 处理一部分层。

**1F1B 调度**:
- **Forward**: 第一轮流水线前向
- **Backward**: 第二轮流水线反向
- **1F1B**: One Forward, One Backward

### 3.2 前向/反向函数获取

**位置**: `megatron/core/pipeline_parallel/schedules.py:45`

```python
def get_forward_backward_func(pp_size: Optional[int] = None, vp_size: Optional[int] = None):
    """获取对应的前向/反向函数"""

    if pp_size is None:
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    if vp_size is None:
        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

    if pp_size > 1:
        # PP 模式
        if vp_size is None or vp_size == 1:
            # 标准 PP
            return forward_backward_pipelining_without_interleaving
        else:
            # 交错 PP
            return forward_backward_pipelining_with_interleaving
    else:
        # 无 PP
        return forward_backward_no_pipelining
```

### 3.3 前向步骤实现

**位置**: `megatron/core/pipeline_parallel/schedules.py:311`

```python
def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    cp_group_size,
    ...
):
    """流水线前向步骤"""

    # 获取输入
    if parallel_state.is_pipeline_first_stage():
        # 第一阶段：从数据迭代器获取
        input_tensor, data = forward_step_func(data_iterator, model)
    else:
        # 其他阶段：接收上一阶段的输出
        input_tensor = recv_forward(input_tensor, ...)

    # 模型前向
    output_tensor = model(input_tensor, ...)

    # 计算损失（最后一阶段）
    if parallel_state.is_pipeline_last_stage():
        output_tensor, num_tokens = forward_step_calc_loss(
            model, output_tensor, loss_func, config, ...
        )
    else:
        # 非最后阶段：发送到下一阶段
        send_forward(output_tensor, ...)

    return output_tensor
```

**通信模式**:
- **Stage i -> Stage i+1**: send/recv 激活值
- **点对点通信**: P2P (Send/Recv)

### 3.4 1F1B 调度实现

**位置**: `megatron/core/pipeline_parallel/schedules.py` (forward_backward_pipelining_without_interleaving)

```python
def forward_backward_pipelining_without_interleaving(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    seq_length,
    micro_batch_size,
    ...
):
    """1F1B 流水线调度"""

    # 获取当前 stage 信息
    model_type = parallel_state.get_pipeline_model_parallel_rank()
    num_stages = parallel_state.get_pipeline_model_parallel_world_size()

    # ========================================
    # 阶段 1: Forward (Warm-up)
    # ========================================
    input_tensors = []
    output_tensors = []
    for i in range(num_stages - model_type):
        # 前向 microbatch
        output_tensor = forward_step(...)
        input_tensors.append(output_tensor)

    # ========================================
    # 阶段 2: 1F1B
    # ========================================
    for i in range(num_microbatches - num_stages + 1):
        # 前向下一个 microbatch
        output_tensor = forward_step(...)

        # 反向当前 microbatch
        input_tensor = input_tensors.pop(0)
        backward_step(...)

    # ========================================
    # 阶段 3: Backward (Cool-down)
    # ========================================
    for input_tensor in input_tensors:
        backward_step(...)

    return forward_data_store
```

**调度可视化** (4-stage PP, 4 microbatches):
```
时间轴:
t1: [F0] [  ] [  ] [  ]
t2: [F1] [F0] [  ] [  ]
t3: [F2] [F1] [F0] [  ]
t4: [F3] [F2] [F1] [F0]
t5: [B0] [F3] [F2] [F1]
t6: [  ] [B1] [F3] [F2]
t7: [  ] [  ] [B2] [F3]
t8: [  ] [  ] [  ] [B3]

F = Forward, B = Backward
```

### 3.5 模型分割

```python
def model_provider(pre_process=True, post_process=True):
    """提供流水线模型"""

    config = TransformerConfig(
        num_layers=16,  # 总层数
        pipeline_model_parallel_size=4,  # 4 个 stage
        ...
    )

    # 每个 stage 的层数
    num_layers_per_stage = config.num_layers // config.pipeline_model_parallel_size

    # Transformer block
    transformer_block = TransformerBlock(
        config=config,
        spec=transformer_layer_spec,
        pre_process=pre_process,  # 第一 stage 需要 embedding
        post_process=post_process,  # 最后 stage 需要 LM head
    )

    return transformer_block
```

---

## 四、上下文并行 (CP) 实现原理

### 4.1 核心思想

CP 将序列长度分割到多个 GPU，支持超长序列训练。

**数学原理**:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V

序列分割:
- Q, K, V 按 seq_len 分割
- Attention 需要全局通信
```

### 4.2 CP 填充计算

**位置**: `megatron/core/models/multimodal/context_parallel.py:9`

```python
def get_padding(seq_len, cp_size, tp_size, has_sp, ...):
    """计算 CP 所需的填充"""

    padding = 0
    padding_factor = 1

    if has_sp and cp_size > 1:
        # SP + CP: 填充到 tp_size * cp_size * 2 的倍数
        padding_factor = tp_size * cp_size * 2
    elif cp_size > 1:
        # 纯 CP: 填充到 cp_size * 2 的倍数
        padding_factor = cp_size * 2
    elif has_sp:
        # 纯 SP: 填充到 tp_size 的倍数
        padding_factor = tp_size

    padding = int((seq_len + padding_factor - 1) // padding_factor * padding_factor) - seq_len

    return padding
```

**为什么需要填充？**
- Ring Attention 需要 pad 到偶数（用于 ring 通信）
- CP + SP 需要对齐到 tp_size * cp_size * 2

### 4.3 PackedSeqParams 构建

**位置**: `megatron/core/models/multimodal/context_parallel.py:62`

```python
def get_packed_seq_params(tokens, img_seq_len, padding_needed, cp_size, use_packed_sequence=False):
    """构建 CP 的 PackedSeqParams"""

    batch_size = tokens.shape[0]

    # 有效序列长度
    combined_valid_seqlen = tokens.shape[1] + img_seq_len - padding_needed

    # 累积序列长度
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * combined_valid_seqlen,
        step=combined_valid_seqlen,
        dtype=torch.int32,
        device=tokens.device,
    )

    # 填充后的序列长度
    combined_padded_seqlen = tokens.shape[1] + img_seq_len

    cu_seqlens_padded = None
    qkv_format = 'sbhd'  # [seq, batch, heads, hidden_dim]

    if cp_size > 1 and (padding_needed > 0 or use_packed_sequence):
        # CP 需要 cu_seqlens_padded
        cu_seqlens_padded = torch.arange(
            0,
            (batch_size + 1) * combined_padded_seqlen,
            step=combined_padded_seqlen,
            dtype=torch.int32,
            device=tokens.device,
        )
        # CP 使用 THD 格式
        qkv_format = 'thd'

    return PackedSeqParams(
        cu_seqlens=cu_seqlens,
        cu_seqlens_qkv=cu_seqlens,
        cu_seqlens_padded=cu_seqlens_padded,
        max_seqlen=combined_valid_seqlen,
        qkv_format=qkv_format,
    )
```

### 4.4 CP Attention 实现

**核心思想**: Ring Attention

```python
class AttentionWithCP(nn.Module):
    def forward(self, hidden_states, packed_seq_params=None):
        # CP 模式: 序列已分割
        batch_size, seq_len_per_rank, hidden_size = hidden_states.shape

        # QKV 投影
        qkv = self.qkv(hidden_states)  # [batch, seq/tp, 3*head_dim]

        if packed_seq_params is not None:
            # CP 模式: 使用 packed sequence
            output = self._forward_cp_packed(qkv, packed_seq_params)
        else:
            # 标准模式
            output = self._forward_standard(qkv)

        return output

    def _forward_cp_packed(self, qkv, packed_seq_params):
        """CP Ring Attention"""
        # 1. Q, K, V 已经按序列分割
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # 2. Ring Attention 通信
        # 需要与 CP rank 通信获取完整的 attention
        output = ring_attention(q, k, v, packed_seq_params)

        return output
```

**Ring Attention 通信**:
```
Rank 0: [seq_0] <--> [seq_1] <--> [seq_2] <--> [seq_3]
         send/recv   send/recv   send/recv
```

---

## 五、专家并行 (EP) 实现原理

### 5.1 核心思想

EP 将 MoE 模型中的专家分布到多个 GPU。

**数学原理**:
```
MoE Layer:
- 路由器: 选择 top-k 专家
- 专家: 每个 expert 是一个 MLP
- 负载均衡: aux_loss

EP 分布:
- 总专家: num_experts
- EP 大小: ep_size
- 本地专家: num_local_experts = num_experts / ep_size
```

### 5.2 GroupedMLP 实现

**位置**: `megatron/core/transformer/moe/experts.py:65`

```python
class GroupedMLP(MegatronModule):
    """使用 GroupedGEMM 高效实现专家层"""

    def __init__(self, num_local_experts: int, config: TransformerConfig, pg_collection):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts

        # 获取进程组
        self.ep_group = pg_collection.ep
        self.tp_group = pg_collection.expt_tp
        self.dp_group = pg_collection.expt_dp

        tp_size = self.tp_group.size()
        tp_rank = self.tp_group.rank()

        # FC1: [hidden, ffn_hidden * num_local_experts]
        fc1_output_size = config.moe_ffn_hidden_size * self.num_local_experts
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        self.weight1 = Parameter(
            torch.empty(
                config.hidden_size,
                fc1_output_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        )

        # FC2: [ffn_hidden * num_local_experts, hidden]
        fc2_input_size = config.moe_ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        self.weight2 = Parameter(
            torch.empty(
                fc2_input_size_per_partition,
                config.hidden_size,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        )
```

**关键点**:
- `num_local_experts`: 每个 EP rank 的专家数
- 权重包含多个专家，使用 GroupedGEMM 并行计算

### 5.3 GroupedMLP 前向传播

```python
def forward(self, hidden_states, combined_weights, combined_bias):
    """
    Args:
        hidden_states: [batch, seq, hidden]
        combined_weights: (scores, indices) 来自路由器
        combined_bias: expert bias
    """
    # 解包路由器输出
    scores, indices = combined_weights
    # scores: [batch * seq, topk]
    # indices: [batch * seq, topk]

    # 重塑输入
    batch_size, seq_len, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    # GroupedGEMM: 并行计算所有专家
    # FC1: [batch*seq, hidden] -> [batch*seq, ffn_hidden * num_local_experts]
    intermediate_parallel = grouped_gemm(
        A=hidden_states,
        B=self.weight1,
        num_groups=self.num_local_experts,
    )

    # 激活函数
    intermediate_parallel = self.activation_func(intermediate_parallel)

    # FC2: [batch*seq, ffn_hidden * num_local_experts] -> [batch*seq, hidden]
    output_parallel = grouped_gemm(
        A=intermediate_parallel,
        B=self.weight2,
        num_groups=self.num_local_experts,
    )

    return output_parallel
```

**GroupedGEMM 优势**:
- 一次 kernel launch 计算所有专家
- 减少 kernel 启动开销
- 提高 GPU 利用率

### 5.4 MoE 路由器

**位置**: `megatron/core/transformer/moe/router.py`

```python
class MoERouter(nn.Module):
    """MoE 路由器"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # 路由权重
        self.weight = Parameter(
            torch.empty(
                config.num_moe_experts,  # [num_experts, hidden]
                config.hidden_size,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        )

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch, seq, hidden]

        Returns:
            scores: [batch * seq, topk]  每个token的topk专家得分
            indices: [batch * seq, topk]  每个token的topk专家索引
        """
        # [batch, seq, hidden] -> [batch * seq, hidden]
        hidden_states = hidden_states.view(-1, self.config.hidden_size)

        # 路由: [batch * seq, num_experts]
        scores = torch.matmul(hidden_states, self.weight.t())

        # Top-k 选择
        topk_scores, topk_indices = torch.topk(
            scores,
            k=self.config.moe_router_topk,
            dim=1,
        )

        return topk_scores, topk_indices

    def aux_loss(self, scores, indices):
        """计算负载均衡损失"""
        # 专家负载分布
        expert_mask = torch.nn.functional.one_hot(indices, num_classes=self.config.num_moe_experts)
        expert_mask = expert_mask.sum(dim=1)  # [batch * seq, num_experts]

        # 专家频率
        expert_freq = expert_mask.sum(dim=0) / expert_mask.sum()

        # 专家平均分数
        expert_scores = (scores * expert_mask).sum(dim=0) / expert_mask.sum(dim=0)

        # 负载均衡损失
        aux_loss = self.config.moe_aux_loss_coeff * (
            expert_freq * expert_scores
        ).sum()

        return aux_loss
```

### 5.5 EP 通信模式

```
Rank 0 (EP=0):
  Experts: [E0, E1]
  Local weight: [w0, w1]

Rank 1 (EP=1):
  Experts: [E2, E3]
  Local weight: [w2, w3]

通信:
- All-to-All: Token 分配到专家所在 rank
- 每个 rank 计算本地专家
- All-to-All: 结果返回
```

---

## 六、数据并行 (DP) 实现原理

### 6.1 核心

DP 在 Megatron 中由其他并行维度隐式定义：

```python
# DP 大小自动计算
dp_size = world_size // (tp_size * pp_size * cp_size * ep_size)
```

### 6.2 DDP 包装

**位置**: `megatron/core/distributed/distributed_data_parallel.py`

```python
class DistributedDataParallel(MegatronModule):
    """Megatron DDP 包装器"""

    def __init__(self, module, config: DistributedDataParallelConfig):
        super().__init__()
        self.module = module
        self.config = config

        # 获取 DP 进程组
        self.dp_group = get_data_parallel_group()

        # 梯度累积
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def backward(self, loss):
        """反向传播"""
        # 反向传播
        loss.backward()

        # DP 梯度同步
        if self.config.overlap_grad_reduce:
            # 异步 all-reduce
            self._overlap_grad_reduce()
        else:
            # 同步 all-reduce
            self._allreduce_grads()

    def _allreduce_grads(self):
        """All-reduce 梯度"""
        for param in self.module.parameters():
            if param.grad is not None:
                torch.distributed.all_reduce(
                    param.grad,
                    group=self.dp_group,
                )
```

---

## 七、通信原语总结

### 7.1 TP 通信

| 操作 | 前向 | 反向 | 用途 |
|------|------|------|------|
| `copy_to_tp_region` | 无操作 | all-reduce | ColumnParallel 输入 |
| `reduce_from_tp_region` | all-reduce | 无操作 | RowParallel 输出 (标准 TP) |
| `scatter_to_tp_region` | split | all-gather | RowParallel 输入 |
| `gather_from_tp_region` | all-gather | split | ColumnParallel 输出 (gather) |
| `reduce_scatter_to_sp_region` | reduce-scatter | all-gather | RowParallel 输出 (SP) |
| `gather_from_sp_region` | all-gather | reduce-scatter | SP 聚合 |

### 7.2 PP 通信

| 操作 | 方向 | 用途 |
|------|------|------|
| `send_forward` | Stage i -> i+1 | 发送激活值 |
| `recv_forward` | Stage i-1 -> i | 接收激活值 |
| `send_backward` | Stage i+1 -> i | 发送梯度 |
| `recv_backward` | Stage i -> i-1 | 接收梯度 |

### 7.3 CP 通信

| 操作 | 模式 | 用途 |
|------|------|------|
| Ring Attention | 环状 | 全局 attention |
| All-to-All | 点对点 | Token 分配 |

### 7.4 EP 通信

| 操作 | 模式 | 用途 |
|------|------|------|
| All-to-All | 点对点 | Token -> 专家 |
| All-to-All | 点对点 | 专家 -> 结果 |

### 7.5 DP 通信

| 操作 | 方向 | 用途 |
|------|------|------|
| all-reduce | DP 组 | 梯度同步 |

---

## 八、性能优化技术

### 8.1 梯度累积融合

**位置**: `megatron/core/tensor_parallel/layers.py`

```python
# 自定义 CUDA 扩展
try:
    import fused_weight_gradient_mlp_cuda
    _grad_accum_fusion_available = True
except ImportError:
    _grad_accum_fusion_available = False

# 使用
if config.gradient_accumulation_fusion:
    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
        output, input, weight, ...
    )
```

### 8.2 异步 All-Reduce

```python
def linear_with_grad_accumulation_and_async_allreduce(
    input, weight, bias, gradient_accumulation_fusion, allreduce_dgrad, ...
):
    """带异步 all-reduce 的线性层"""

    # 前向: 矩阵乘法
    output = torch.nn.functional.linear(input, weight, bias)

    # 梯度累积 + 异步 all-reduce
    if allreduce_dgrad:
        # 注册 autograd hook 进行异步 all-reduce
        output = AllReduce.apply(output)

    return output
```

### 8.3 激活重计算

```python
def forward(self, x):
    if self.config.recompute_granularity == 'full':
        # 全部重计算
        return torch.utils.checkpoint.checkpoint(self._forward, x)
    else:
        # 标准
        return self._forward(x)
```

---

## 九、并行策略对比

| 策略 | 分割维度 | 通信模式 | 内存节省 | 适用场景 |
|------|---------|---------|---------|---------|
| **TP** | hidden | all-reduce | 否 | 小模型、大 batch |
| **SP** | seq | reduce-scatter | 是 | 大序列 |
| **PP** | layer | p2p send/recv | 是 | 深度模型 |
| **CP** | seq | ring attention | 是 | 超长序列 |
| **EP** | expert | all-to-all | 是 | MoE 模型 |
| **DP** | batch | all-reduce | 否 | 大 batch |

---

## 十、完整代码示例

### 10.1 3D 并行 (TP + PP + DP)

```python
# 初始化
world_size = 128
tp_size = 8
pp_size = 4
dp_size = world_size // (tp_size * pp_size)  # = 4

initialize_model_parallel(
    tensor_model_parallel_size=tp_size,
    pipeline_model_parallel_size=pp_size,
)

# 模型构建
num_layers = 64
layers_per_stage = num_layers // pp_size  # 16 layers per stage

model = TransformerBlock(
    config=TransformerConfig(
        num_layers=layers_per_stage,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        sequence_parallel=True,  # 启用 SP
    ),
    pre_process=(pp_rank == 0),
    post_process=(pp_rank == pp_size - 1),
)

# 训练
forward_backward_func = get_forward_backward_func(
    pp_size=pp_size,
    vp_size=2,  # 交错 PP
)

for iteration in range(num_iterations):
    forward_backward_func(
        forward_step_func=forward_step,
        data_iterator=data_iterators,
        model=model,
        num_microbatches=8,
        seq_length=2048,
        micro_batch_size=2,
    )
```

---

*基于 Megatron-LM MCore 源码分析*
*更新日期: 2025-01-31*
