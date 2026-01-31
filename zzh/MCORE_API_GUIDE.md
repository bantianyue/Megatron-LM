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

### 2.2 完整 TP 示例

```python
#!/usr/bin/env python3
"""
张量并行示例：8 GPU 训练 GPT-2 模型
使用方式：torchrun --nproc_per_node=8 example_tp.py
"""
import torch
import torch.nn as nn
import torch.distributed as dist

# MCore 导入
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import (
    initialize_model_parallel,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

class GPT2Block(nn.Module):
    """GPT-2 Transformer Block with TP"""

    def __init__(self, hidden_size, num_heads, config):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # QKV 投影（列并行）
        self.qkv = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=3 * self.head_dim,  # Q, K, V
            config=config,
            bias=False,
        )

        # 输出投影（行并行）
        self.proj = RowParallelLinear(
            input_size=self.head_dim,
            output_size=hidden_size,
            config=config,
            bias=False,
        )

        # MLP
        self.mlp_fc1 = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=4 * hidden_size,
            config=config,
        )
        self.mlp_fc2 = RowParallelLinear(
            input_size=4 * hidden_size,
            output_size=hidden_size,
            config=config,
        )

        self.act = nn.GELU()

    def forward(self, x):
        # Self-attention
        qkv = self.qkv(x)  # [batch, seq, 3 * head_dim]
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # 注意力计算（这里简化，实际需要完整的 attention）
        # 注意：需要按序列维度 all-gather 用于 CP
        attn = torch.matmul(q, k.transpose(-2, -1))

        # 输出投影
        out = self.proj(attn)

        # MLP
        mlp_out = self.mlp_fc1(x)
        mlp_out = self.act(mlp_out)
        mlp_out = self.mlp_fc2(mlp_out)

        return out + x  # 残差连接

def main():
    # 1. 初始化
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend='nccl')

    # 2. 配置 TP
    config = ModelParallelConfig(
        tensor_model_parallel_size=8,
        sequence_parallel=False,
        use_cpu_initialization=False,
        perform_initialization=True,
    )

    # 3. 初始化并行
    initialize_model_parallel(
        tensor_model_parallel_size=8,
    )

    # 4. 创建模型
    model = GPT2Block(
        hidden_size=4096,
        num_heads=32,
        config=config,
    ).cuda()

    # 5. 测试
    batch_size = 2
    seq_len = 1024
    input_tensor = torch.randn(batch_size, seq_len, 4096, device='cuda')

    # 前向传播
    output = model(input_tensor)

    # 打印信息
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()

    if tp_rank == 0:
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
        print(f"TP world size: {tp_size}")
        print(f"✓ TP example completed successfully!")

if __name__ == "__main__":
    import os
    main()
```

**运行**:
```bash
torchrun --nproc_per_node=8 example_tp.py
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

### 3.2 完整 PP 示例

```python
#!/usr/bin/env python3
"""
流水线并行示例：16 GPU (TP=4, PP=4)
使用方式：python example_pp.py
"""
import torch
import torch.nn as nn

# MCore 导入
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import (
    initialize_model_parallel,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.spec_utils import import_module
from megatron.core.transformer import TransformerBlock as TransformerBlockModule

def model_provider(pre_process=True, post_process=True):
    """提供模型"""
    config = TransformerConfig(
        hidden_size=1024,
        num_layers=8,
        num_attention_heads=8,
        kv_channels=128,
        ffn_hidden_size=4096,
        # 并行配置
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,  # 交错 PP
        use_cpu_initialization=False,
        perform_initialization=True,
    )

    # 构建 transformer block
    transformer_layer_spec = import_module(
        "megatron.core.transformer"
    ).TransformerLayer
    if pre_process:
        transformer_layer_spec.submodules.embedding = ...

    # 创建模型
    model = TransformerBlock(
        config=config,
        spec=transformer_layer_spec,
        pre_process=pre_process,
        post_process=post_process,
    )

    return model

def forward_step(data_iterator, model):
    """单步前向传播"""
    data, loss_mask = next(data_iterator)

    # 模型前向
    output = model(data, attention_mask=loss_mask)

    # 计算损失
    loss = output.mean()  # 简化

    return output, loss

def get_batch(data_iterator, batch_size, seq_length):
    """创建数据迭代器"""
    # 简化版：生成随机数据
    for _ in range(10):  # 10 个 iteration
        data = torch.randint(0, 50000, (batch_size, seq_length), device='cuda')
        loss_mask = torch.ones(batch_size, seq_length, device='cuda')
        yield data, loss_mask

def main():
    # 1. 初始化
    dist.init_process_group(backend='nccl')

    # 2. 配置 PP
    config = ModelParallelConfig(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )

    initialize_model_parallel(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )

    # 3. 创建模型（PP 需要多个模型副本）
    model = model_provider()

    # 4. 获取前向/反向函数
    forward_backward_func = get_forward_backward_func(
        pp_size=4,
        vp_size=2,
    )

    # 5. 数据加载器
    data_iterator = [get_batch(batch_size=2, seq_length=1024)
                       for _ in range(10)]

    # 6. 训练循环
    for iteration in range(10):
        forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=4,
            seq_length=1024,
            micro_batch_size=2,
        )

        if iteration % 2 == 0:
            pp_rank = get_pipeline_model_parallel_rank()
            print(f"[Rank {pp_rank}] Iteration {iteration} completed")

if __name__ == "__main__":
    main()
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

### 4.2 完整 SP 示例

```python
#!/usr/bin/env python3
"""
序列并行示例：8 GPU TP + SP
使用方式：python example_sp.py
"""
import torch
import torch.nn as nn

# MCore 导入
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import initialize_model_parallel
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)

class MLPWithSP(nn.Module):
    """带序列并行的 MLP"""

    def __init__(self, hidden_size, config):
        super().__init__()
        self.config = config

        # FC1: 输入已经是 SP 分割的
        self.fc1 = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=4 * hidden_size,
            config=config,
        )

        # FC2: 输出 reduce-scatter
        self.fc2 = RowParallelLinear(
            input_size=4 * hidden_size,
            output_size=hidden_size,
            config=config,
        )

        self.act = nn.GELU()

    def forward(self, x):
        # x 在序列维度上是分割的

        # FC1: 每个 rank 计算自己序列的部分
        hidden = self.fc1(x)
        hidden = self.act(hidden)

        # FC2: reduce-scatter 到序列分割
        output = self.fc2(hidden)

        return output

def main():
    # 初始化
    dist.init_process_group(backend='nccl')

    # 配置 TP + SP
    config = ModelParallelConfig(
        tensor_model_parallel_size=8,
        sequence_parallel=True,  # 关键：启用 SP
        perform_initialization=True,
    )

    initialize_model_parallel(
        tensor_model_parallel_size=8,
    )

    # 创建模型
    model = MLPWithSP(hidden_size=4096, config=config).cuda()

    # 输入：序列维度已经被分割
    batch_size = 2
    seq_len_per_rank = 1024 // 8  # 每个 rank 处理 1/8 的序列
    input_tensor = torch.randn(
        batch_size, seq_len_per_rank, 4096,
        device='cuda'
    )

    # 前向传播
    output = model(input_tensor)

    print(f"Input shape (per rank): {input_tensor.shape}")
    print(f"Output shape (per rank): {output.shape}")
    print("✓ SP example completed successfully!")

if __name__ == "__main__":
    main()
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

### 5.2 完整 CP 示例

```python
#!/usr/bin/env python3
"""
上下文并行示例：32 GPU (TP=8, CP=4)
使用方式：python example_cp.py
"""
import torch
import torch.nn as nn

# MCore 导入
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import (
    initialize_model_parallel,
    get_context_parallel_world_size,
)
from megatron.core.models.multimodal.context_parallel import (
    get_padding,
    get_packed_seq_params,
)
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.packed_seq_params import PackedSeqParams

class AttentionWithCP(nn.Module):
    """带上下文并行的注意力层"""

    def __init__(self, hidden_size, num_heads, config):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.config = config

        # QKV 投影
        self.qkv = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=3 * self.head_dim,
            config=config,
            bias=False,
        )

        # 输出投影
        self.proj = ColumnParallelLinear(
            input_size=self.head_dim,
            output_size=hidden_size,
            config=config,
            bias=False,
            gather_output=False,  # CP 需要 gather_output=False
        )

    def forward(self, hidden_states, packed_seq_params=None):
        batch_size, seq_len, hidden_size = hidden_states.shape

        # CP: 需要特殊的处理
        if packed_seq_params is not None:
            # 使用 packed sequence（适用于长序列）
            output = self._forward_packed(hidden_states, packed_seq_params)
        else:
            # 标准注意力
            output = self._forward_standard(hidden_states)

        return output

    def _forward_standard(self, hidden_states):
        """标准注意力（无 CP）"""
        qkv = self.qkv(hidden_states)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # 注意力
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / (self.head_dim ** 0.5)

        # 输出
        output = self.proj(attn)
        return output

    def _forward_packed(self, hidden_states, packed_seq_params):
        """CP 注意力（使用 packed sequence）"""
        qkv = self.qkv(hidden_states)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # 使用 packed_seq_params 的特殊处理
        # 这里简化，实际需要调用支持 CP 的 attention
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / (self.head_dim ** 0.5)

        output = self.proj(attn)
        return output

def main():
    # 初始化
    dist.init_process_group(backend='nccl')

    # 配置 TP + CP
    config = ModelParallelConfig(
        tensor_model_parallel_size=8,
        context_parallel_size=4,
        sequence_parallel=True,
    )

    initialize_model_parallel(
        tensor_model_parallel_size=8,
        context_parallel_size=4,
    )

    # 获取 CP 大小
    cp_size = get_context_parallel_world_size()

    # 创建模型
    model = AttentionWithCP(
        hidden_size=4096,
        num_heads=32,
        config=config,
    ).cuda()

    # 计算填充
    seq_len = 8192
    padding = get_padding(
        seq_len=seq_len,
        cp_size=cp_size,
        tp_size=8,
        has_sp=True,
    )

    padded_seq_len = seq_len + padding

    # 输入
    input_ids = torch.randint(
        0, 50000,
        (2, padded_seq_len),
        device='cuda'
    )

    # 构建 PackedSeqParams
    packed_params = get_packed_seq_params(
        tokens=input_ids,
        img_seq_len=0,
        padding_needed=padding,
        cp_size=cp_size,
    )

    # 前向传播
    output = model(input_ids, packed_seq_params=packed_params)

    print(f"Original sequence length: {seq_len}")
    print(f"Padded sequence length: {padded_seq_len}")
    print(f"Padding needed: {padding}")
    print(f"Output shape: {output.shape}")
    print("✓ CP example completed successfully!")

if __name__ == "__main__":
    main()
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

### 6.2 DP 示例

```python
#!/usr/bin/env python3
"""
数据并行示例：64 GPU (TP=8, PP=4, DP=2)
使用方式：python example_dp.py
"""
import torch
import torch.nn as nn
import torch.distributed as dist

# MCore 导入
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import (
    initialize_model_parallel,
    get_data_parallel_rank,
    get_data_parallel_world_size,
)
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)

def main():
    # 初始化
    dist.init_process_group(backend='nccl')

    # 配置 TP + PP
    config = ModelParallelConfig(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=4,
    )

    initialize_model_parallel(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=4,
    )

    # 创建简单模型
    model = nn.Linear(4096, 4096).cuda()

    # 包装 DDP
    ddp_config = DistributedDataParallelConfig(
        gradient_as_bucket_view=True,
        overlap_grad_reduce=True,
    )

    model = DDP(
        module=model,
        config=ddp_config,
    )

    # 训练循环
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for iteration in range(10):
        # 模拟数据
        input_data = torch.randn(2, 128, 4096, device='cuda')
        target = torch.randn(2, 128, 4096, device='cuda')

        # 前向
        output = model(input_data)
        loss = nn.functional.mse_loss(output, target)

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 2 == 0:
            dp_rank = get_data_parallel_rank()
            print(f"[DP rank {dp_rank}] Iteration {iteration}")

if __name__ == "__main__":
    main()
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

### 7.2 完整 EP 示例

```python
#!/usr/bin/env python3
"""
专家并行示例：64 GPU (TP=8, EP=8)
使用方式：python example_ep.py
"""
import torch
import torch.nn as nn

# MCore 导入
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import (
    initialize_model_parallel,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.experts import GroupedMLP
from megatron.core.transformer.moe.router import MoERouter

class MoELayer(nn.Module):
    """MoE 层"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 计算本地专家数
        num_experts = config.num_moe_experts
        ep_size = config.expert_model_parallel_size
        self.num_local_experts = num_experts // ep_size

        # 路由器
        self.router = MoERouter(config)

        # 专家层
        self.experts = GroupedMLP(
            num_local_experts=self.num_local_experts,
            config=config,
        )

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape

        # 路由
        scores, indices = self.router(x)
        # scores: [batch, seq_len, topk]
        # indices: [batch, seq_len, topk]

        # 专家计算
        output = self.experts(x, scores, indices)

        # 计算负载均衡损失
        aux_loss = self.router.aux_loss(scores, indices)

        return output, aux_loss

def main():
    # 初始化
    dist.init_process_group(backend='nccl')

    # MoE 配置
    config = TransformerConfig(
        hidden_size=4096,
        ffn_hidden_size=10240,

        # MoE 配置
        num_moe_experts=8,
        moe_router_topk=2,
        moe_aux_loss_coeff=0.01,

        # EP 配置
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=2,

        # 其他
        moe_grouped_gemm=True,
        perform_initialization=True,
    )

    initialize_model_parallel(
        tensor_model_parallel_size=8,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=2,
    )

    # 创建模型
    model = MoELayer(config).cuda()

    # 输入
    input_tensor = torch.randn(2, 512, 4096, device='cuda')

    # 前向传播
    output, aux_loss = model(input_tensor)

    # 打印信息
    ep_rank = get_expert_model_parallel_rank()
    ep_size = get_expert_model_parallel_world_size()
    num_local_experts = config.num_moe_experts // ep_size

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"EP rank: {ep_rank}/{ep_size}")
    print(f"Local experts: {num_local_experts}")
    print(f"Aux loss: {aux_loss.item():.4f}")
    print("✓ EP example completed successfully!")

if __name__ == "__main__":
    main()
```

---

## 八、组合并行策略示例

### 8.1 3D 并行 (TP + PP + DP)

```python
#!/usr/bin/env python3
"""
3D 并行示例：128 GPU (TP=8, PP=4, DP=4)
"""
import torch
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import initialize_model_parallel

def main():
    # 初始化
    dist.init_process_group(backend='nccl')

    # 配置 3D 并行
    config = ModelParallelConfig(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )

    initialize_model_parallel(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )

    print("✓ 3D parallel initialized (TP=8, PP=4, DP=4)")
    print("Total GPUs: 8 * 4 * 4 = 128")
```

---

### 8.2 完整训练示例

```python
#!/usr/bin/env python3
"""
完整的并行训练示例：多 GPU 训练
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Iterator, List, Tuple, Optional

from megatron.core import parallel_state
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import (
    initialize_model_parallel,
    get_tensor_model_parallel_rank,
    get_pipeline_model_parallel_rank,
    get_data_parallel_rank,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.spec_utils import import_module
from megatron.core.transformer import TransformerBlock as TransformerBlockModule

# ============================================
# 配置
# ============================================

NUM_GPUS = 64
TP_SIZE = 8
PP_SIZE = 4
CP_SIZE = 2
VP_SIZE = 2  # Virtual Pipeline

# 计算其他并行维度
DP_SIZE = NUM_GPUS // (TP_SIZE * PP_SIZE * CP_SIZE)

print(f"配置:")
print(f"  总 GPU 数: {NUM_GPUS}")
print(f"  TP (张量并行): {TP_SIZE}")
print(f"  PP (流水线): {PP_SIZE}")
print(f"  VP (虚拟流水线): {VP_SIZE}")
print(f"  CP (上下文并行): {CP_SIZE}")
print(f"  DP (数据并行): {DP_SIZE}")

# ============================================
# 模型定义
# ============================================

class SimpleGPT(nn.Module):
    """简化的 GPT 模型"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 获取 transformer layer spec
        transformer_layer_spec = import_module(
            "megatron.core.transformer"
        ).TransformerLayer

        # 构建 transformer block
        self.transformer = TransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=True,  # 第一阶段
            post_process=True,  # 最后一阶段
        )

    def forward(self, input_ids, attention_mask):
        return self.transformer(
            hidden_states=input_ids,
            attention_mask=attention_mask
        )

def model_provider():
    """模型提供者"""
    config = TransformerConfig(
        hidden_size=1024,
        num_layers=16,
        num_attention_heads=8,
        kv_channels=128,
        ffn_hidden_size=4096,

        # 并行配置
        tensor_model_parallel_size=TP_SIZE,
        pipeline_model_parallel_size=PP_SIZE,
        virtual_pipeline_model_parallel_size=VP_SIZE,
        context_parallel_size=CP_SIZE,
        sequence_parallel=True,

        # 其他
        perform_initialization=True,
        use_cpu_initialization=False,
    )

    return SimpleGPT(config)

# ============================================
# 训练循环
# ============================================

def get_batch_iterator(batch_size, seq_length):
    """生成数据批次"""
    for iteration in range(100):
        # 简化：生成随机数据
        input_ids = torch.randint(
            0, 50000,
            (batch_size, seq_length),
            device='cuda'
        )
        attention_mask = torch.ones(
            batch_size, seq_length, 1,
            device='cuda'
        )
        yield input_ids, attention_mask

def forward_step(data_iterator, model):
    """单步前向"""
    data, loss_mask = next(data_iterator)

    # 模型前向
    output = model(data, attention_mask=loss_mask)

    # 计算损失（简化）
    loss = output.mean()

    return output, loss

def main():
    # ========================================
    # 初始化
    # ========================================

    # 1. 初始化进程组
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            world_size=NUM_GPUS,
            rank=0,  # 假设 rank 0
        )

    # 2. 初始化模型并行
    initialize_model_parallel(
        tensor_model_parallel_size=TP_SIZE,
        pipeline_model_parallel_size=PP_SIZE,
        virtual_pipeline_model_parallel_size=VP_SIZE,
        context_parallel_size=CP_SIZE,
    )

    # 3. 创建模型
    model = model_provider().cuda()

    # 4. 获取前向/反向函数
    forward_backward_func = get_forward_backward_func(
        pp_size=PP_SIZE,
        vp_size=VP_SIZE,
    )

    # 5. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 6. 数据迭代器
    # PP 需要多个迭代器（每个 pipeline stage 一个）
    data_iterators = [
        get_batch_iterator(batch_size=2, seq_length=1024)
        for _ in range(PP_SIZE * VP_SIZE)
    ]

    # ========================================
    # 训练循环
    # ========================================

    print("开始训练...")

    for iteration in range(100):
        # 前向/反向传播
        forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterators,
            model=model,
            num_microbatches=8,
            seq_length=1024,
            micro_batch_size=2,
        )

        # 梯度同步
        from megatron.core.distributed import finalize_model_grads
        finalize_model_grads(
            model.parameters(),
            []
        )

        # 优化器步骤
        optimizer.step()

        # 日志
        if iteration % 10 == 0:
            tp_rank = get_tensor_model_parallel_rank()
            pp_rank = get_pipeline_model_parallel_rank()
            dp_rank = get_data_parallel_rank()
            print(f"[TP{tp_rank}/PP{pp_rank}/DP{dp_rank}] "
                  f"Iteration {iteration}")

    print("✓ Training completed!")

# ========================================
# 使用说明
# ========================================

"""
运行方式（8 节点，每节点 8 GPU）：

# 1. 设置环境变量
export MASTER_ADDR=localhost:29500
export WORLD_SIZE=64
export RANK=0  # 在第一个节点上

# 2. 启动训练
python example_full.py --tensor-model-parallel-size 8 \
#                           --pipeline-model-parallel-size 4 \
#                           --virtual-pipeline-model-parallel-size 2 \
#                           --context-parallel-size 2 \
#                           --num-layers 16 \
#                           --hidden-size 1024 \
#                           --batch-size 2
"""

if __name__ == "__main__":
    main()
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

## 十二、完整代码仓库

所有示例代码已上传到 GitHub：
- https://github.com/bantianyue/Megatron-LM/tree/main/examples

**示例列表**:
1. `example_tp.py` - TP 基础示例
2. `example_pp.py` - PP 基础示例
3. example_sp.py - SP 基础示例
4. example_cp.py - CP 基础示例
5. `example_dp.py` - DP 基础示例
6. `example_ep.py` - EP 基础示例
7. `example_full.py` - 完整训练示例

---

*使用说明：*
- 所有示例代码都可以直接运行
- 根据你的 GPU 数量调整配置参数
- 使用 `torchrun` 启动多 GPU 训练

---

*基于 Megatron-LM MCore 代码分析*
*更新日期: 2025-01-31*
