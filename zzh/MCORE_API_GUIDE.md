# Megatron Core (MCore) 并行策略 API 使用指南

## 概述

本文档提供 Megatron Core (MCore) 的并行策略 API 使用说明和可运行示例。

**MCore 位置**: `megatron/core/`

**支持的并行策略**:
- **TP**: Tensor Parallelism (张量并行)
- **PP**: Pipeline Parallelism (流水线并行)
- **SP**: Sequence Parallelism (序列并行)
- **CP**: Context Parallelism (上下文并行)
- **DP**: Data Parallelism (数据并行)
- **EP**: Expert Parallelism (专家并行)

---

## 一、初始化并行环境

### 1.1 核心初始化接口

**位置**: `megatron/core/parallel_state.py:549`

```python
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    expert_tensor_parallel_size: Optional[int] = None,
    ...
) -> None
```

### 1.2 初始化步骤

```python
import torch
import torch.distributed as dist

# 1. 初始化进程组
dist.init_process_group(backend='nccl')

# 2. 初始化模型并行
from megatron.core.parallel_state import initialize_model_parallel

initialize_model_parallel(
    tensor_model_parallel_size=8,      # TP
    pipeline_model_parallel_size=4,     # PP
    context_parallel_size=2,           # CP
    expert_model_parallel_size=2,       # EP
)
```

### 1.3 完整初始化示例

```python
#!/usr/bin/env python3
import os
import torch
import torch.distributed as dist

def setup_distributed():
    """设置分布式环境"""
    # 从环境变量获取 rank 和 world_size
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank
    )

    return rank, world_size

def initialize_parallel():
    """初始化并行策略"""
    from megatron.core.parallel_state import initialize_model_parallel

    # 根据可用 GPU 数量配置并行
    world_size = torch.distributed.get_world_size()

    if world_size == 8:
        # 8 GPUs: 纯 TP
        initialize_model_parallel(
            tensor_model_parallel_size=8,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
    elif world_size == 32:
        # 32 GPUs: TP + CP
        initialize_model_parallel(
            tensor_model_parallel_size=8,
            pipeline_model_parallel_size=1,
            context_parallel_size=4,
            expert_model_parallel_size=1,
        )
    elif world_size == 64:
        # 64 GPUs: TP + EP
        initialize_model_parallel(
            tensor_model_parallel_size=8,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=8,
        )
    else:
        raise ValueError(f"Unsupported world_size: {world_size}")

if __name__ == "__main__":
    setup_distributed()
    initialize_parallel()
```

---

## 二、张量并行 (TP)

### 2.1 接口说明

#### ColumnParallelLinear - 列并行线性层

**位置**: `megatron/core/tensor_parallel/layers.py:751`

**接口**:
```python
ColumnParallelLinear(
    input_size: int,           # 输入维度
    output_size: int,          # 输出维度（总维度）
    config: ModelParallelConfig,
    init_method: Callable = None,
    bias: bool = True,
    gather_output: bool = False,
    stride: int = 1,
    skip_bias_add: bool = False,
    tp_group: Optional[ProcessGroup] = None,
)
```

**参数说明**:
- `gather_output`: 是否 all-gather 输出（False 时每个 rank 只保留自己的部分）
- `skip_bias_add`: 是否跳过 bias 加法（用于性能优化）
- `tp_group`: 自定义进程组（默认使用全局 TP 组）

**输出维度**: `output_size_per_partition = output_size / tp_size`

---

#### RowParallelLinear - 行并行线性层

**位置**: `megatron/core/tensor_parallel/layers.py:1081`

**接口**:
```python
RowParallelLinear(
    input_size: int,           # 输入维度（总维度）
    output_size: int,          # 输出维度
    config: ModelParallelConfig,
    init_method: Callable = None,
    bias: bool = True,
    input_is_parallel: bool = False,  # 输入是否已经是 TP 分割的
    skip_bias_add: bool = False,
    tp_group: Optional[ProcessGroup] = None,
)
```

**参数说明**:
- `input_is_parallel`: 输入是否已经是 TP 分割的（如来自 ColumnParallelLinear）
- `output_size`: 总输出维度（会 all-reduce 到完整维度）

---

### 2.2 TP 使用示例

#### 初始化 TP

```python
from megatron.core.parallel_state import initialize_model_parallel

# 初始化 8-way TP
initialize_model_parallel(tensor_model_parallel_size=8)
```

#### 使用 ColumnParallelLinear（列并行）

```python
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.model_parallel_config import ModelParallelConfig

config = ModelParallelConfig(
    tensor_model_parallel_size=8,
    sequence_parallel=False,
)

# 权重按列分割：output_size 每个 rank 为 output_size/8
fc1 = ColumnParallelLinear(
    input_size=4096,
    output_size=16384,  # 每个 rank: 2048
    config=config,
    bias=True,
)

# 前向: [batch, seq, 4096] -> [batch, seq, 2048]
output = fc1(input)
```

#### 使用 RowParallelLinear（行并行）

```python
from megatron.core.tensor_parallel import RowParallelLinear

# 权重按行分割：input_size 每个 rank 为 input_size/8
fc2 = RowParallelLinear(
    input_size=16384,  # 每个 rank: 2048
    output_size=4096,
    config=config,
    input_is_parallel=True,  # 输入来自 ColumnParallelLinear
    bias=True,
)

# 前向: [batch, seq, 2048] -> [batch, seq, 4096] (all-reduce)
output = fc2(input)
```

#### MLP: ColumnParallel + RowParallel 组合

```python
class MLPBlock(nn.Module):
    def __init__(self, hidden_size, config):
        super().__init__()
        # FC1: 列并行
        self.fc1 = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=4 * hidden_size,
            config=config,
        )
        # FC2: 行并行（输入来自 fc1）
        self.fc2 = RowParallelLinear(
            input_size=4 * hidden_size,
            output_size=hidden_size,
            config=config,
            input_is_parallel=True,  # 关键参数
        )

    def forward(self, x):
        x = self.fc1(x)
        x = act(x)
        x = self.fc2(x)  # 自动 all-reduce
        return x
```

---

## 三、流水线并行 (PP)

### 3.1 接口说明

#### 核心调度接口

**位置**: `megatron/core/pipeline_parallel/schedules.py:45`

```python
def get_forward_backward_func(
    pp_size: Optional[int] = None,
    vp_size: Optional[int] = None,
) -> Callable
```

**返回函数签名**:
```python
def forward_backward_func(
    forward_step_func: Callable,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[MegatronModule, List[MegatronModule]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    ...
) -> None
```

---

### 3.2 PP 使用示例

#### 初始化 PP

```python
from megatron.core.parallel_state import initialize_model_parallel

# 初始化 4-way PP
initialize_model_parallel(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=2,  # 交错 PP
)
```

#### 获取前向/反向函数

```python
from megatron.core.pipeline_parallel import get_forward_backward_func

# 获取对应的前向/反向函数
forward_backward_func = get_forward_backward_func(
    pp_size=4,
    vp_size=2,
)
```

#### 定义前向步骤

```python
def forward_step(data_iterator, model):
    """单步前向传播"""
    data, loss_mask = next(data_iterator)
    output = model(data, attention_mask=loss_mask)
    loss = output.mean()
    return output, loss
```

#### 训练循环

```python
# 训练循环
for iteration in range(num_iterations):
    forward_backward_func(
        forward_step_func=forward_step,
        data_iterator=data_iterators,
        model=model,
        num_microbatches=8,
        seq_length=1024,
        micro_batch_size=2,
    )
```

---

## 四、序列并行 (SP)

### 4.1 接口说明

SP 集成在 TP 中，通过配置启用：

```python
config = ModelParallelConfig(
    tensor_model_parallel_size=8,
    sequence_parallel=True,  # 启用 SP
)
```

**通信原语**:
- `gather_from_sequence_parallel_region` - All-gather
- `reduce_scatter_to_sequence_parallel_region` - Reduce-scatter

---

### 4.2 SP 使用示例

#### 启用 SP

```python
from megatron.core.model_parallel_config import ModelParallelConfig

# SP 通过配置启用
config = ModelParallelConfig(
    tensor_model_parallel_size=8,
    sequence_parallel=True,  # 关键：启用 SP
)
```

#### SP 与 RowParallelLinear

```python
from megatron.core.tensor_parallel import RowParallelLinear

# SP 模式：RowParallelLinear 输出使用 reduce-scatter
fc2 = RowParallelLinear(
    input_size=4 * hidden_size,
    output_size=hidden_size,
    config=config,
    input_is_parallel=True,
)

# 前向自动 reduce-scatter
output = fc2(input)  # 输出: [batch, seq/tp, hidden]
```

#### 通信原语（手动使用）

```python
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)

# 聚合 SP 分割的序列
full_output = gather_from_sequence_parallel_region(partial_output)

# 分散到 SP 区域
partial_output = reduce_scatter_to_sequence_parallel_region(full_output)
```

---

## 五、上下文并行 (CP)

### 5.1 接口说明

#### 填充计算

**位置**: `megatron/core/models/multimodal/context_parallel.py:9`

```python
def get_padding(
    seq_len: int,
    cp_size: int,
    tp_size: int,
    has_sp: bool,
    decoder_tp_comm_overlap: bool = False,
    decoder_seq_len: Optional[int] = None,
    fp8_enabled: bool = False,
    fp8_recipe: Optional[str] = None,
) -> int
```

#### PackedSeqParams 构建

```python
def get_packed_seq_params(
    tokens: torch.Tensor,
    img_seq_len: int,
    padding_needed: int,
    cp_size: int,
    use_packed_sequence: bool = False,
) -> PackedSeqParams
```

---

### 5.2 CP 使用示例

#### 初始化 CP

```python
from megatron.core.parallel_state import initialize_model_parallel

# 初始化 CP
initialize_model_parallel(
    tensor_model_parallel_size=8,
    context_parallel_size=4,
)
```

#### 计算填充

```python
from megatron.core.models.multimodal.context_parallel import get_padding

seq_len = 8192
cp_size = 4
tp_size = 8
has_sp = True

padding = get_padding(
    seq_len=seq_len,
    cp_size=cp_size,
    tp_size=tp_size,
    has_sp=has_sp,
)
# 返回需要的填充长度（填充到 cp_size * 2 的倍数）
```

#### 构建 PackedSeqParams

```python
from megatron.core.models.multimodal.context_parallel import get_packed_seq_params

packed_params = get_packed_seq_params(
    tokens=input_ids,           # [batch, seq_len]
    img_seq_len=0,              # 图像序列长度
    padding_needed=padding,     # 填充长度
    cp_size=cp_size,            # CP 大小
)
# 返回: PackedSeqParams 对象，包含 cu_seqlens 等信息
```

#### 在 Transformer 中使用

```python
output = transformer(
    hidden_states=hidden_states,
    packed_seq_params=packed_params,  # 传递 CP 参数
)
```

---

## 六、数据并行 (DP)

### 6.1 接口说明

DP 通常由框架自动管理，无需手动配置。

**配置**:
```python
# DP 大小由 world_size // (tp_size * pp_size * cp_size * ep_size) 自动计算
# 无需显式配置
```

---

### 6.2 DP 说明

DP 大小自动计算，无需手动配置：

```python
# DP 大小 = world_size // (tp_size * pp_size * cp_size * ep_size)
world_size = 128
tp_size = 8
pp_size = 4
dp_size = world_size // (tp_size * pp_size)  # = 4
```

使用 DDP 包装模型（可选）：

```python
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)

ddp_config = DistributedDataParallelConfig(
    gradient_as_bucket_view=True,
    overlap_grad_reduce=True,
)

model = DDP(module=model, config=ddp_config)
```

---

## 七、专家并行 (EP)

### 7.1 接口说明

#### GroupedMLP - 分组专家

**位置**: `megatron/core/transformer/moe/experts.py:65`

```python
GroupedMLP(
    num_local_experts: int,        # 本地专家数量
    config: TransformerConfig,
    pg_collection: ProcessGroupCollection,
)
```

**参数计算**:
```python
total_experts = config.num_moe_experts
ep_size = config.expert_model_parallel_size
num_local_experts = total_experts // ep_size
```

---

### 7.2 EP 使用示例

#### 初始化 EP

```python
from megatron.core.parallel_state import initialize_model_parallel

# 初始化 EP
initialize_model_parallel(
    tensor_model_parallel_size=8,
    expert_model_parallel_size=8,  # EP 大小
    expert_tensor_parallel_size=2,  # 每个 expert 内部的 TP
)
```

#### 配置 MoE

```python
from megatron.core.transformer.transformer_config import TransformerConfig

config = TransformerConfig(
    hidden_size=4096,
    ffn_hidden_size=10240,

    # MoE 配置
    num_moe_experts=8,            # 总专家数
    moe_router_topk=2,            # 每个 token 选择 2 个专家
    moe_aux_loss_coeff=0.01,      # 负载均衡损失系数

    # EP 配置
    expert_model_parallel_size=8, # EP 大小
    expert_tensor_parallel_size=2,# 每个 expert 内部的 TP

    moe_grouped_gemm=True,        # 使用 GroupedGEMM
)
```

#### 计算本地专家数

```python
total_experts = config.num_moe_experts  # 8
ep_size = config.expert_model_parallel_size  # 8
num_local_experts = total_experts // ep_size  # 1

# 每个 EP rank 有 1 个本地专家
```

#### 使用 GroupedMLP

```python
from megatron.core.transformer.moe.experts import GroupedMLP

experts = GroupedMLP(
    num_local_experts=1,  # 本地专家数
    config=config,
)

# 前向
output = experts(
    hidden_states,  # [batch, seq, hidden]
    (scores, indices),  # 来自路由器
)
```

#### 使用 MoERouter

```python
from megatron.core.transformer.moe.router import MoERouter

router = MoERouter(config)

# 前向
scores, indices = router(hidden_states)
# scores: [batch * seq, topk]
# indices: [batch * seq, topk]

# 计算辅助损失
aux_loss = router.aux_loss(scores, indices)
```

---

## 八、组合并行策略示例

### 8.1 3D 并行 (TP + PP + DP)

```python
from megatron.core.parallel_state import initialize_model_parallel

# 128 GPU: TP=8, PP=4, DP=4
initialize_model_parallel(
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=2,
)

# DP 大小自动计算: 128 // (8 * 4) = 4
```

### 8.2 TP + CP + SP

```python
# 64 GPU: TP=8, CP=4, SP
initialize_model_parallel(
    tensor_model_parallel_size=8,
    context_parallel_size=4,
)

config = ModelParallelConfig(
    tensor_model_parallel_size=8,
    sequence_parallel=True,  # 启用 SP
)
```

### 8.3 完整配置示例

```python
from megatron.core.transformer.transformer_config import TransformerConfig

config = TransformerConfig(
    # 模型配置
    hidden_size=4096,
    num_layers=64,
    ffn_hidden_size=16384,

    # 并行配置
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=2,
    context_parallel_size=2,
    sequence_parallel=True,

    # EP（MoE）
    num_moe_experts=8,
    expert_model_parallel_size=4,
    moe_grouped_gemm=True,
)
```

---

## 九、API 速查表

### 9.1 初始化接口

| 功能 | 函数 | 位置 |
|------|------|------|
| 初始化并行 | `initialize_model_parallel()` | parallel_state.py:549 |
| TP 大小 | `get_tensor_model_parallel_world_size()` | parallel_state.py:1544 |
| PP 大小 | `get_pipeline_model_parallel_world_size()` | - |
| CP 大小 | `get_context_parallel_world_size()` | parallel_state.py:1746 |
| EP 大小 | `get_expert_model_parallel_world_size()` | parallel_state.py:1797 |
| DP 大小 | `get_data_parallel_world_size()` | - |

### 9.2 并行层接口

| 层类型 | 类 | 位置 |
|--------|---|------|
| 列并行 | `ColumnParallelLinear` | tensor_parallel/layers.py:751 |
| 行并行 | `RowParallelLinear` | tensor_parallel/layers.py:1081 |
| 专家 | `GroupedMLP` | transformer/moe/experts.py:65 |

### 9.3 通信原语

| 原语 | 函数 | 用途 |
|------|------|------|
| All-Reduce | `reduce_from_tensor_model_parallel_region()` | 同步行输出 |
| All-Gather | `gather_from_sequence_parallel_region()` | SP 聚合 |
| Reduce-Scatter | `reduce_scatter_to_sequence_parallel_region()` | SP 分散 |
| Scatter | `scatter_to_tensor_model_parallel_region()` | TP 分散 |
| Copy | `copy_to_tensor_model_parallel_region()` | TP 区域复制 |

### 9.4 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `tensor_model_parallel_size` | TP 大小 | 1 |
| `pipeline_model_parallel_size` | PP 大小 | 1 |
| `virtual_pipeline_model_parallel_size` | 虚拟 PP 大小 | None |
| `context_parallel_size` | CP 大小 | 1 |
| `expert_model_parallel_size` | EP 大小 | 1 |
| `sequence_parallel` | 是否启用 SP | False |
| `order` | 并行组创建顺序 | "tp-cp-ep-dp-pp" |

---

## 十、故障排查

### 10.1 常见错误

**错误 1: 序列长度不匹配**
```
ValueError: sequence length must be divisible by cp_size * 2
```
**解决**: 使用 `get_padding()` 计算填充

**错误 2: EP 均匀分配**
```
AssertionError: num_experts must be divisible by expert_model_parallel_size
```
**解决**: 确保 `num_experts % expert_model_parallel_size == 0`

**错误 3: GPU 数量不足**
```
RuntimeError: world_size is not divisible by model_size
```
**解决**: 确保 `world_size % (tp_size * pp_size * cp_size * ep_size) == 0`

---

## 十一、性能优化建议

### 11.1 TP 优化
- 使用 `sequence_parallel=True` 减少激活内存
- 启用 `skip_bias_add=True` 融合 bias
- 使用 Transformer Engine 加速

### 11.2 PP 优化
- 使用 `virtual_pipeline_model_parallel_size` 提高 GPU 利用率
- 增加 `num_microbatches` 隐藏通信延迟
- 使用 interleaved schedule

### 11.3 CP 优化
- 使用 `hybrid_context_parallel=True` 混合调度
- 合理设置 `padding` 减少填充开销

### 11.4 EP 优化
- 使用 `grouped_gemm=True` 加速专家计算
- 调整 `moe_router_topk` 平衡负载和性能
- 使用 `moe_aux_loss_coeff` 防止负载不均

---

## 十二、使用说明

### 运行方式

```bash
# 单节点多 GPU
torchrun --nproc_per_node=8 your_script.py

# 多节点多 GPU
torchrun \
  --nproc_per_node=8 \
  --nnodes=4 \
  --node_rank=0 \
  --master_addr=192.168.1.1 \
  --master_port=29500 \
  your_script.py
```

### 环境变量

| 环境变量 | 说明 |
|---------|------|
| `RANK` | 全局 rank |
| `WORLD_SIZE` | 总 GPU 数 |
| `LOCAL_RANK` | 节点内 rank |
| `MASTER_ADDR` | 主节点地址 |
| `MASTER_PORT` | 主节点端口 |

---

*基于 Megatron-LM MCore 代码分析*
*更新日期: 2025-01-31*
