# Megatron-LM 并行策略实现分析：TP / CP / EP

## 概述

Megatron-LM 支持多种并行策略组合，用于训练超大规模模型：

| 并行策略 | 英文 | 缩写 | 核心思想 | 适用场景 |
|---------|------|------|---------|---------|
| **张量并行** | Tensor Parallelism | TP | 按张量维度分割 | 单层超大模型 |
| **上下文并行** | Context Parallelism | CP | 按序列长度分割 | 超长序列 |
| **专家并行** | Expert Parallelism | EP | 按专家分割 | MoE 模型 |

本文档分析这三种并行策略的**实现方式**、**对外接口**和**调用示例**。

---

## 一、张量并行 (Tensor Parallelism, TP)

### 1.1 核心思想

将单个张量（权重、激活值）分割到多个 GPU，每个 GPU 只计算张量的一部分。

### 1.2 实现位置

```
megatron/core/tensor_parallel/
├── layers.py           # 并行层实现
├── mappings.py         # 通信原语
├── cross_entropy.py    # 并行交叉熵
└── random.py          # 并行随机数生成
```

### 1.3 核心类

#### ColumnParallelLinear - 列并行

**分割方式**: 按列分割权重矩阵

```python
# 位置: megatron/core/tensor_parallel/layers.py:751
class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    分割方式: Y = X @ W
              输入: X [m, k]
              权重: W = [W1, W2, W3, W4] (按列分割)
              输出: Y = [X@W1, X@W2, X@W3, X@W4] (各GPU独立计算)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: ModelParallelConfig,
        init_method: Callable = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer: bool = False,
        grad_output_buffer: bool = False,
        tp_group: Optional[ProcessGroup] = None,
    ):
        # 计算每个 rank 的输出大小
        self.output_size_per_partition = divide(output_size, tp_world_size)

        # 创建权重（只保存当前 rank 的部分）
        self.weight = Parameter(
            torch.Tensor(self.output_size_per_partition, input_size)
        )

        # 初始化权重（设置并行属性）
        _initialize_affine_weight_gpu(
            self.weight,
            init_method,
            partition_dim=0,  # 按列分割（第一维）
            stride=1,
        )
```

**使用示例**:
```python
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.model_parallel_config import ModelParallelConfig

# 配置
config = ModelParallelConfig(
    tensor_model_parallel_size=8,  # 8路TP
    sequence_parallel=False,
)

# 创建列并行层
layer = ColumnParallelLinear(
    input_size=4096,
    output_size=12288,  # 总输出维度
    config=config,
    init_method=torch.nn.init.xavier_uniform_,
    bias=False,
)

# 前向传播
# input: [batch, seq_len, hidden_size]
# output: [batch, seq_len, output_size_per_partition]
output = layer(input)
```

---

#### RowParallelLinear - 行并行

**分割方式**: 按行分割权重矩阵

```python
# 位置: megatron/core/tensor_parallel/layers.py:1081
class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    分割方式: Y = X @ W
              输入: X [m, k] (需要先 all-gather)
              权重: W = [W1; W2; W3; W4] (按行分割)
              中间: Z = [X@W1, X@W2, X@W3, X@W4]
              输出: Y = sum(Z) (all-reduce)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: ModelParallelConfig,
        init_method: Callable = None,
        bias: bool = True,
        input_is_parallel: bool = False,
        skip_bias_add: bool = False,
        tp_group: Optional[ProcessGroup] = None,
    ):
        # 计算每个 rank 的输入大小
        self.input_size_per_partition = divide(input_size, tp_world_size)

        # 创建权重
        self.weight = Parameter(
            torch.Tensor(output_size, self.input_size_per_partition)
        )

        # 初始化权重（按行分割）
        _initialize_affine_weight_gpu(
            self.weight,
            init_method,
            partition_dim=1,  # 按行分割（第二维）
            stride=1,
        )
```

**使用示例**:
```python
from megatron.core.tensor_parallel import RowParallelLinear

# 创建行并行层
layer = RowParallelLinear(
    input_size=12288,
    output_size=4096,  # 总输出维度
    config=config,
    init_method=torch.nn.init.xavier_uniform_,
    bias=False,
)

# 前向传播
# input: [batch, seq_len, input_size_per_partition]
# output: [batch, seq_len, output_size] (已 all-reduce)
output = layer(input)
```

---

### 1.4 通信原语

**位置**: `megatron/core/tensor_parallel/mappings.py`

#### All-Reduce - 行并行输出同步

```python
def reduce_from_tensor_model_parallel_region(input_, tp_group=None):
    """All-reduce over TP group.

    用于行并行的输出同步，将各 TP rank 的部分结果求和。
    """
    if tp_group is None:
        tp_group = get_tensor_model_parallel_group(with_context_parallel=True)

    # All-reduce
    torch.distributed.all_reduce(
        input_,
        group=tp_group
    )

    return input_
```

#### All-Gather - 序列并行

```python
def gather_from_sequence_parallel_region(input_, tp_group=None):
    """All-gather over TP group for sequence parallel.

    用于序列并行，将分散的序列维度聚合。
    """
    if tp_group is None:
        tp_group = get_tensor_model_parallel_group(with_context_parallel=True)

    # All-gather
    torch.distributed.all_gather_into_tensor(
        input_,
        group=tp_group
    )

    return input_
```

#### Reduce-Scatter - 序列并行

```python
def reduce_scatter_to_sequence_parallel_region(input_, tp_group=None):
    """Reduce-scatter over TP group for sequence parallel.

    用于序列并行，聚合结果后重新分散。
    """
    if tp_group is None:
        tp_group = get_tensor_model_parallel_group(with_context_parallel=True)

    # Reduce-scatter
    torch.distributed.reduce_scatter_tensor(
        input_,
        group=tp_group
    )

    return input_
```

---

### 1.5 配置接口

#### 通过配置类

```python
from megatron.core.model_parallel_config import ModelParallelConfig

config = ModelParallelConfig(
    # 张量并行配置
    tensor_model_parallel_size=8,

    # 序列并行（与TP配合）
    sequence_parallel=True,

    # 其他并行维度
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    expert_model_parallel_size=1,
)
```

#### 通过命令行参数

```bash
python pretrain_gpt.py \
    --tensor-model-parallel-size 8 \
    --sequence-parallel \
    --num-layers 96 \
    --hidden-size 12288 \
    --num-attention-heads 96
```

---

### 1.6 初始化接口

```python
from megatron.core.parallel_state import initialize_model_parallel

# 初始化模型并行
initialize_model_parallel(
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    expert_model_parallel_size=1,
)

# 获取信息
from megatron.core.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)

tp_size = get_tensor_model_parallel_world_size()  # 8
tp_rank = get_tensor_model_parallel_rank()        # 0-7
```

---

### 1.7 完整使用示例

```python
import torch
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import initialize_model_parallel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.module import MegatronModule

# 1. 初始化模型并行
initialize_model_parallel(
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
)

# 2. 创建配置
config = TransformerConfig(
    hidden_size=12288,
    num_attention_heads=96,
    tensor_model_parallel_size=8,
    sequence_parallel=True,
)

# 3. 使用并行层构建模型
class MyAttention(MegatronModule):
    def __init__(self, config):
        super().__init__(config)
        self.qkv = ColumnParallelLinear(
            config.hidden_size,
            3 * config.kv_channels,
            config=config,
        )
        self.proj = RowParallelLinear(
            config.kv_channels,
            config.hidden_size,
            config=config,
        )

    def forward(self, x):
        # QKV 投影（列并行）
        qkv = self.qkv(x)

        # 注意力计算...

        # 输出投影（行并行）
        out = self.proj(qkv)
        return out

# 4. 前向传播
model = MyAttention(config)
input = torch.randn(2, 1024, 12288, device='cuda')
output = model(input)
```

---

## 二、上下文并行 (Context Parallelism, CP)

### 2.1 核心思想

将输入序列按长度分割到多个 GPU，每个 GPU 处理序列的一个片段，通过通信完成注意力计算。

### 2.2 实现位置

```
megatron/core/
├── models/multimodal/context_parallel.py  # 多模态CP
├── pipeline_parallel/hybrid_cp_schedule.py  # 混合CP调度
├── parallel_state.py                    # CP状态管理
└── model_parallel_config.py              # CP配置
```

### 2.3 核心功能

#### 序列分割与填充

**位置**: `megatron/core/models/multimodal/context_parallel.py`

```python
def get_padding(
    seq_len,
    cp_size,
    tp_size,
    has_sp,
    decoder_tp_comm_overlap=False,
    decoder_seq_len=None,
    fp8_enabled=False,
    fp8_recipe=None,
):
    """计算 CP 需要的填充。

    Args:
        seq_len (int): 模型序列长度
        cp_size (int): 上下文并行大小
        tp_size (int): 张量并行大小
        has_sp (bool): 是否使用序列并行

    Returns:
        padding (int): 需要的填充长度
    """
    padding_factor = 1
    if has_sp and cp_size > 1:
        # CP + SP 时，填充到 tp_size * cp_size * 2 的倍数
        padding_factor = tp_size * cp_size * 2
    elif cp_size > 1:
        # 仅 CP 时，填充到 cp_size * 2 的倍数
        padding_factor = cp_size * 2
    elif has_sp:
        # 仅 SP 时，填充到 tp_size 的倍数
        padding_factor = tp_size

    padding = int(
        (seq_len + padding_factor - 1) // padding_factor * padding_factor
    ) - seq_len

    return padding
```

#### PackedSeqParams 构建

```python
def get_packed_seq_params(
    tokens,
    img_seq_len,
    padding_needed,
    cp_size,
    use_packed_sequence=False
):
    """为 CP 构建 PackedSeqParams。

    用于 Transformer Engine 的序列打包。
    """
    batch_size = tokens.shape[0]

    # 有效序列长度
    combined_valid_seqlen = tokens.shape[1] + img_seq_len - padding_needed

    # cu_seqlens: 每个序列的起始位置
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * combined_valid_seqlen,
        step=combined_valid_seqlen,
        dtype=torch.int32,
        device=tokens.device,
    )

    # CP 需要的格式
    if cp_size > 1:
        qkv_format = 'thd'  # CP 需要的格式
    else:
        qkv_format = 'sbhd'

    return PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=combined_valid_seqlen,
        max_seqlen_k=combined_valid_seqlen,
        qkv_format=qkv_format,
    )
```

---

### 2.4 配置接口

```python
from megatron.core.model_parallel_config import ModelParallelConfig

config = ModelParallelConfig(
    # 上下文并行配置
    context_parallel_size=4,

    # 分层 CP（可选）
    hierarchical_context_parallel_sizes=[2, 2],

    # TP + CP 组合
    tensor_model_parallel_size=8,
    sequence_parallel=True,
)
```

---

### 2.5 初始化接口

```python
from megatron.core.parallel_state import initialize_model_parallel

# 初始化 CP
initialize_model_parallel(
    tensor_model_parallel_size=8,
    context_parallel_size=4,  # 4路CP
    pipeline_model_parallel_size=1,
)

# 获取 CP 信息
from megatron.core.parallel_state import (
    get_context_parallel_world_size,
    get_context_parallel_rank,
    get_context_parallel_group,
)

cp_size = get_context_parallel_world_size()  # 4
cp_rank = get_context_parallel_rank()        # 0-3
cp_group = get_context_parallel_group()      # ProcessGroup
```

---

### 2.6 完整使用示例

```python
import torch
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import (
    initialize_model_parallel,
    get_context_parallel_world_size,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.multimodal.context_parallel import (
    get_padding,
    get_packed_seq_params,
)

# 1. 初始化 CP
initialize_model_parallel(
    tensor_model_parallel_size=8,
    context_parallel_size=4,
)

# 2. 配置
config = TransformerConfig(
    hidden_size=4096,
    num_attention_heads=32,
    context_parallel_size=4,
    sequence_parallel=True,
)

# 3. 计算填充
seq_len = 8192
cp_size = get_context_parallel_world_size()
padding = get_padding(
    seq_len=seq_len,
    cp_size=cp_size,
    tp_size=8,
    has_sp=True,
)
print(f"需要填充: {padding}")  # 可能输出: 需要填充: 64

# 4. 构建 PackedSeqParams
tokens = torch.randint(0, 50000, (2, 8192))
img_seq_len = 0  # 文本模型
packed_params = get_packed_seq_params(
    tokens=tokens,
    img_seq_len=img_seq_len,
    padding_needed=padding,
    cp_size=cp_size,
)

# 5. 前向传播
# 实际使用时，packed_params 会传递给 attention layer
output = model(tokens, packed_seq_params=packed_params)
```

---

## 三、专家并行 (Expert Parallelism, EP)

### 3.1 核心思想

在 Mixture of Experts (MoE) 模型中，将不同的专家分配到不同的 GPU，实现专家级别的并行。

### 3.2 实现位置

```
megatron/core/transformer/moe/
├── experts.py              # 专家实现
├── router.py               # 路由器
├── moe_utils.py            # MoE 工具
└── token_dispatcher.py      # Token 分发
```

### 3.3 核心类

#### GroupedMLP - 分组专家

**位置**: `megatron/core/transformer/moe/experts.py:65`

```python
class GroupedMLP(MegatronModule):
    """使用 GroupedGEMM 高效实现的专家层。

    并行执行多个专家以最大化计算效率。
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        self.num_local_experts = num_local_experts
        self.config = config

        # 专家并行
        self.expert_parallel = config.expert_model_parallel_size > 1

        # 使用 GroupedGEMM
        gg.assert_grouped_gemm_is_available()

        # 创建权重
        # shape: [num_local_experts, hidden_size, ffn_hidden_size]
        self.weight1 = Parameter(...)
        self.weight2 = Parameter(...)
```

**关键特性**:
- 使用 GroupedGEMM 一次计算多个专家
- 支持专家并行（不同 GPU 有不同专家）
- 支持张量并行（每个专家内部再并行）

---

### 3.4 MoE Router

**路由器**决定每个 token 应该发送到哪个专家。

```python
# 位置: megatron/core/transformer/moe/router.py
class MoERouter(MegatronModule):
    """MoE 路由器"""

    def __init__(self, config: TransformerConfig):
        self.num_experts = config.num_moe_experts
        self.topk = config.moe_router_topk
        self.router_type = config.moe_router_type

        # 路由权重
        if self.router_type == "linear":
            self.weight = Parameter(...)
        elif self.router_type == "gmlp":
            self.weight = Parameter(...)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            scores: [batch, seq_len, num_experts] - 专家选择分数
        """
        # 计算路由分数
        logits = torch.nn.functional.linear(x, self.weight)

        # Top-K 选择
        scores, indices = torch.topk(logits, k=self.topk, dim=-1)

        # 归一化
        scores = torch.nn.functional.softmax(scores, dim=-1)

        return scores, indices
```

---

### 3.5 配置接口

```python
from megatron.core.transformer.transformer_config import TransformerConfig

config = TransformerConfig(
    # MoE 基础配置
    num_moe_experts=8,              # 专家总数
    moe_router_topk=2,               # 每个 token 选择 top-2 专家
    moe_router_pre_softmax=False,     # 路由器类型

    # EP 配置
    expert_model_parallel_size=4,     # 4路专家并行

    # TP + EP 组合
    tensor_model_parallel_size=8,
    expert_tensor_parallel_size=2,    # 每个专家内再2路TP

    # 其他
    moe_grouped_gemm=True,            # 使用 GroupedGEMM
    moe_aux_loss_coeff=0.01,          # 负载均衡损失系数
)
```

---

### 3.6 初始化接口

```python
from megatron.core.parallel_state import initialize_model_parallel

# 初始化 EP
initialize_model_parallel(
    tensor_model_parallel_size=8,
    expert_model_parallel_size=4,  # 4路EP
    expert_tensor_parallel_size=2,  # 专家内2路TP
)

# 获取 EP 信息
from megatron.core.parallel_state import (
    get_expert_model_parallel_world_size,
    get_expert_model_parallel_rank,
)

ep_size = get_expert_model_parallel_world_size()  # 4
ep_rank = get_expert_model_parallel_rank()        # 0-3
```

---

### 3.7 完整使用示例

```python
import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.parallel_state import initialize_model_parallel
from megatron.core.transformer.moe.experts import GroupedMLP
from megatron.core.transformer.moe.router import MoERouter

# 1. 初始化 EP
initialize_model_parallel(
    tensor_model_parallel_size=8,
    expert_model_parallel_size=4,
    expert_tensor_parallel_size=2,
)

# 2. 配置
config = TransformerConfig(
    hidden_size=4096,
    ffn_hidden_size=10240,

    # MoE 配置
    num_moe_experts=8,              # 8个专家
    moe_router_topk=2,
    moe_aux_loss_coeff=0.01,

    # EP 配置
    expert_model_parallel_size=4,
    expert_tensor_parallel_size=2,
)

# 3. 计算每个 rank 的专家数
num_experts = config.num_moe_experts
ep_size = config.expert_model_parallel_size
num_local_experts = num_experts // ep_size  # 8 / 4 = 2

# 4. 创建 MoE 层
class MoEBlock(MegatronModule):
    def __init__(self, config):
        super().__init__(config)
        self.router = MoERouter(config)
        self.experts = GroupedMLP(
            num_local_experts=2,  # 每个 rank 2个专家
            config=config,
        )

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape

        # 路由
        scores, indices = self.router(x)
        # scores: [batch, seq_len, topk]  # 例如 topk=2
        # indices: [batch, seq_len, topk]

        # 扁平化以使用 GroupedGEMM
        # 将 batch * seq_len 个 token 分配到本地专家

        # GroupedGEMM 执行
        output = self.experts(x, scores, indices)

        # 计算负载均衡损失
        aux_loss = self._compute_aux_loss(scores, indices)

        return output, aux_loss

    def _compute_aux_loss(self, scores, indices):
        """计算负载均衡损失"""
        # 简化版示例
        expert_mask = torch.nn.functional.one_hot(indices, self.num_experts)
        expert_counts = expert_mask.sum(dim=(0, 1))

        # 均匀分布目标
        target = scores.shape[0] * scores.shape[1] / self.num_experts

        # 损失
        aux_loss = ((expert_counts - target) ** 2).mean()
        return aux_loss

# 5. 使用
model = MoEBlock(config)
input = torch.randn(2, 1024, 4096, device='cuda')
output, aux_loss = model(input)

print(f"输出形状: {output.shape}")  # [2, 1024, 4096]
print(f"负载均衡损失: {aux_loss.item()}")
```

---

## 四、组合并行策略

### 4.1 3D 并行 (TP + CP + EP)

```python
from megatron.core.parallel_state import initialize_model_parallel

# 初始化组合并行
initialize_model_parallel(
    # 张量并行
    tensor_model_parallel_size=8,

    # 上下文并行
    context_parallel_size=4,

    # 专家并行
    expert_model_parallel_size=2,

    # 流水线并行
    pipeline_model_parallel_size=2,
)

# 总 GPU 数: 8 * 4 * 2 * 2 = 128 GPUs
```

### 4.2 并行顺序

**初始化顺序**: `order="tp-cp-ep-dp-pp"`

```python
# 并行组创建顺序
# 1. TP group: 8 GPUs
# 2. CP group: 4 GPUs (跨 TP groups)
# 3. EP group: 2 GPUs (跨 CP groups)
# 4. DP group: ...
# 5. PP group: ...
```

---

## 五、对外接口总结

### 5.1 初始化接口

```python
# 位置: megatron/core/parallel_state.py:549
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    expert_tensor_parallel_size: Optional[int] = None,
    order: str = "tp-cp-ep-dp-pp",
    ...
) -> None
```

### 5.2 查询接口

```python
from megatron.core.parallel_state import (
    # TP
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_group,

    # CP
    get_context_parallel_world_size,
    get_context_parallel_rank,
    get_context_parallel_group,

    # EP
    get_expert_model_parallel_world_size,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_group,

    # 组合
    get_data_parallel_group,
    get_data_parallel_world_size,
)
```

### 5.3 配置接口

```python
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig

# ModelParallelConfig - 基础并行配置
config = ModelParallelConfig(
    tensor_model_parallel_size=8,
    context_parallel_size=4,
    expert_model_parallel_size=2,
    sequence_parallel=True,
)

# TransformerConfig - 模型 + 并行配置
config = TransformerConfig(
    hidden_size=4096,
    num_layers=32,
    num_attention_heads=32,

    # 并行配置
    tensor_model_parallel_size=8,
    context_parallel_size=4,
    expert_model_parallel_size=2,
    sequence_parallel=True,
)
```

### 5.4 层接口

```python
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

# 列并行
qkv_layer = ColumnParallelLinear(
    input_size=hidden_size,
    output_size=3 * kv_channels,
    config=config,
)

# 行并行
proj_layer = RowParallelLinear(
    input_size=kv_channels,
    output_size=hidden_size,
    config=config,
)
```

---

## 六、实战示例

### 6.1 示例 1: 单纯 TP (8-GPU 单层)

```python
#!/usr/bin/env python3
import torch
from megatron.core import parallel_state
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

# 1. 初始化 TP
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=8,
)

# 2. 配置
config = ModelParallelConfig(
    tensor_model_parallel_size=8,
    sequence_parallel=False,
)

# 3. 构建 MLP
class SimpleMLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            input_size=4096,
            output_size=16384,
            config=config,
        )
        self.fc2 = RowParallelLinear(
            input_size=16384,
            output_size=4096,
            config=config,
        )
        self.act = torch.nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

# 4. 测试
model = SimpleMLP(config).cuda()
input = torch.randn(2, 1024, 4096, device='cuda')
output = model(input)

print(f"输入: {input.shape}")
print(f"输出: {output.shape}")
print(f"TP rank: {parallel_state.get_tensor_model_parallel_rank()}")
```

---

### 6.2 示例 2: TP + CP (32-GPU 超长序列)

```python
#!/usr/bin/env python3
import torch
from megatron.core import parallel_state
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.dot_product_attention import (
    DotProductAttention,
)

# 1. 初始化 TP + CP
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=8,
    context_parallel_size=4,
)

# 2. 配置
config = TransformerConfig(
    hidden_size=4096,
    num_attention_heads=32,
    kv_channels=128,

    # TP + CP
    tensor_model_parallel_size=8,
    context_parallel_size=4,
    sequence_parallel=True,
)

# 3. 构建注意力层
attn = DotProductAttention(
    config=config,
    sub_mode='cross_attention',
    layer_number=1,
).cuda()

# 4. 超长序列输入
seq_len = 32768
batch_size = 2

# 5. 计算填充
from megatron.core.models.multimodal.context_parallel import get_padding

padding = get_padding(
    seq_len=seq_len,
    cp_size=parallel_state.get_context_parallel_world_size(),
    tp_size=parallel_state.get_tensor_model_parallel_world_size(),
    has_sp=True,
)

# 填充输入
padded_seq_len = seq_len + padding
input_ids = torch.randint(0, 50000, (batch_size, padded_seq_len), device='cuda')

# 6. 前向传播
output = attn(
    hidden_states=input_ids,
    attention_mask=None,
)

print(f"原始序列长度: {seq_len}")
print(f"填充后长度: {padded_seq_len}")
print(f"填充量: {padding}")
print(f"输出形状: {output.shape}")
```

---

### 6.3 示例 3: TP + EP (MoE, 32-GPU)

```python
#!/usr/bin/env python3
import torch
from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.experts import GroupedMLP
from megatron.core.transformer.moe.router import MoERouter

# 1. 初始化 TP + EP
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=8,
    expert_model_parallel_size=4,
    expert_tensor_parallel_size=2,
)

# 2. MoE 配置
config = TransformerConfig(
    hidden_size=4096,
    ffn_hidden_size=10240,

    # MoE 配置
    num_moe_experts=16,          # 16个专家
    moe_router_topk=2,
    moe_aux_loss_coeff=0.01,

    # EP 配置
    expert_model_parallel_size=4,   # 4路EP
    expert_tensor_parallel_size=2,  # 每个专家内2路TP
)

# 3. 计算
num_experts = config.num_moe_experts
ep_size = config.expert_model_parallel_size
num_local_experts = num_experts // ep_size  # 16 / 4 = 4

# 4. 构建 MoE 层
class MoELayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = MoERouter(config)
        self.experts = GroupedMLP(
            num_local_experts=num_local_experts,
            config=config,
        )

    def forward(self, x):
        # 路由
        scores, indices = self.router(x)

        # 专家计算
        output = self.experts(x, scores, indices)

        # 负载均衡损失
        aux_loss = self.router.compute_aux_loss(scores, indices)

        return output, aux_loss

# 5. 测试
model = MoELayer(config).cuda()
input = torch.randn(2, 1024, 4096, device='cuda')
output, aux_loss = model(input)

print(f"输入: {input.shape}")
print(f"输出: {output.shape}")
print(f"负载均衡损失: {aux_loss.item():.4f}")
print(f"本地专家数: {num_local_experts}")
print(f"EP rank: {parallel_state.get_expert_model_parallel_rank()}")
```

---

## 七、最佳实践

### 7.1 GPU 数量计算

```
总 GPU 数 = TP_size × CP_size × EP_size × PP_size × DP_size

示例:
- 单纯 TP: 8 GPUs = 8 × 1 × 1 × 1 × 1
- TP + CP: 32 GPUs = 8 × 4 × 1 × 1 × 1
- TP + EP: 64 GPUs = 8 × 1 × 8 × 1 × 1
- TP + CP + EP: 256 GPUs = 8 × 4 × 8 × 1 × 1
```

### 7.2 配置建议

| 模型规模 | GPU 数量 | 推荐配置 |
|---------|---------|---------|
| 小型 (<1B) | 8 | TP=8 |
| 中型 (1B-10B) | 32-64 | TP=8, CP=4 或 TP=8, EP=8 |
| 大型 (10B-100B) | 128-512 | TP=8, CP=4, EP=4, PP=2-4 |
| 超大 (>100B) | 512+ | TP=8, CP=4, EP=8, PP=4-8 |

### 7.3 常见错误

**错误 1: CP 需要填充**
```
RuntimeError: sequence length must be divisible by cp_size * 2
```
解决: 使用 `get_padding()` 计算填充量

**错误 2: EP 需要均匀分配**
```
AssertionError: num_experts must be divisible by expert_model_parallel_size
```
解决: 确保 `num_experts % expert_model_parallel_size == 0`

**错误 3: TP + CP + SP 组合**
```
ValueError: Cannot use both sequence_parallel and context_parallel without proper padding
```
解决: 使用 `get_padding()` 或设置 `--no-sequence-parallel`

---

## 八、API 速查表

### TP 接口

| 功能 | 接口 | 位置 |
|------|------|------|
| 初始化 | `initialize_model_parallel(tp_size=...)` | parallel_state.py:549 |
| 查询大小 | `get_tensor_model_parallel_world_size()` | parallel_state.py:1544 |
| 查询 rank | `get_tensor_model_parallel_rank()` | - |
| 查询组 | `get_tensor_model_parallel_group()` | - |
| 列并行层 | `ColumnParallelLinear(...)` | layers.py:751 |
| 行并行层 | `RowParallelLinear(...)` | layers.py:1081 |

### CP 接口

| 功能 | 接口 | 位置 |
|------|------|------|
| 初始化 | `initialize_model_parallel(cp_size=...)` | parallel_state.py:549 |
| 查询大小 | `get_context_parallel_world_size()` | parallel_state.py:1746 |
| 查询 rank | `get_context_parallel_rank()` | - |
| 计算填充 | `get_padding(seq_len, cp_size, ...)` | context_parallel.py:9 |
| 构建参数 | `get_packed_seq_params(tokens, ...)` | context_parallel.py:62 |

### EP 接口

| 功能 | 接口 | 位置 |
|------|------|------|
| 初始化 | `initialize_model_parallel(ep_size=...)` | parallel_state.py:549 |
| 查询大小 | `get_expert_model_parallel_world_size()` | parallel_state.py:1797 |
| 查询 rank | `get_expert_model_parallel_rank()` | - |
| 分组专家 | `GroupedMLP(num_local_experts, ...)` | experts.py:65 |
| 路由器 | `MoERouter(config)` | router.py |

---

## 九、参考文档

- **TP 论文**: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- **CP 论文**: [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
- **EP 论文**: [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)

---

*基于 Megatron-LM 代码库深度分析*
*分析日期: 2025-01-30*
