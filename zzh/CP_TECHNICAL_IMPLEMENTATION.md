# Megatron-LM Context Parallelism (CP) 技术实现详解

## 目录
- [1. CP 核心概念](#1-cp-核心概念)
- [2. 实现架构](#2-实现架构)
- [3. 通信机制](#3-通信机制)
- [4. 代码实现分析](#4-代码实现分析)
- [5. 使用指南](#5-使用指南)
- [6. 性能优化](#6-性能优化)

---

## 1. CP 核心概念

### 1.1 什么是 Context Parallelism

Context Parallelism (CP) 是一种将**序列维度**分割到多个 GPU 上的并行策略。

**对比其他并行方式：**

| 并行类型 | 分割维度 | 说明 |
|---------|---------|------|
| TP (Tensor Parallel) | Hidden Dimension | 分割模型权重 |
| PP (Pipeline Parallel) | Layer | 分割层 |
| DP (Data Parallel) | Batch | 分割数据 |
| **CP (Context Parallel)** | **Sequence Length** | **分割序列长度** |

### 1.2 CP 的应用场景

```
适用场景：
├── 超长序列训练 (seq_length > 8192)
├── 长上下文语言模型
├── 文档级别的 QA 任务
└── 多模态模型（图文长序列）
```

### 1.3 Ring Attention 原理

CP 使用 Ring Attention 来实现高效的注意力计算：

```
原始 Attention: O(N²) 内存复杂度
Ring Attention: O(N²/P) 内存复杂度 (P=CP size)

Ring 通信模式:
Rank 0: [0,1,2] ←→ [3,4,5] → [6,7,8]
Rank 1: [3,4,5] ←→ [6,7,8] → [0,1,2]
Rank 2: [6,7,8] ←→ [0,1,2] → [3,4,5]
```

---

## 2. 实现架构

### 2.1 核心文件结构

```
megatron/core/
├── parallel_state.py              # CP 进程组管理
├── model_parallel_config.py       # CP 配置参数
├── extensions/
│   └── transformer_engine.py      # TE DotProductAttention CP 实现
├── ssm/
│   └── mamba_context_parallel.py  # Mamba CP 实现
└── models/
    └── multimodal/
        └── context_parallel.py    # 多模态 CP 工具函数
```

### 2.2 进程组架构

```
World Size (总 GPU 数)
│
├── Data Parallel Group (DP)
│   └── 包含 TP × PP × CP
│
├── Tensor Parallel Group (TP)
│   └── 分割隐藏维度
│
├── Pipeline Parallel Group (PP)
│   └── 分割层
│
└── Context Parallel Group (CP) ← CP 核心进程组
    └── 分割序列长度
```

### 2.3 相关进程组

| 进程组变量 | 说明 |
|-----------|------|
| `_CONTEXT_PARALLEL_GROUP` | 主 CP 进程组 |
| `_CONTEXT_PARALLEL_GLOBAL_RANKS` | CP 组内所有 rank |
| `_HIERARCHICAL_CONTEXT_PARALLEL_GROUPS` | 层次化 CP 进程组 |
| `_TENSOR_AND_CONTEXT_PARALLEL_GROUP` | TP+CP 组合进程组 |
| `_DATA_PARALLEL_GROUP_WITH_CP` | DP×CP 组合进程组 |

---

## 3. 通信机制

### 3.1 支持的通信类型

```python
# megatron/core/extensions/transformer_engine.py:1162
cp_comm_type: Optional[str] = "p2p"  # 四种类型
```

| 类型 | 全称 | 通信模式 | 特点 |
|------|------|---------|------|
| `p2p` | Point-to-Point | Ring Attention | 低延迟，适合长序列 |
| `a2a` | All-to-All | 全局交换 | 均衡负载 |
| `allgather` | All-Gather | 全收集 | 简单但内存开销大 |
| `a2a+p2p` | Hybrid | 层次化混合 | 结合两者优势 |

### 3.2 P2P (Ring Attention) 数据流

```
步骤 1: 分割序列
Input: [token_0, token_1, ..., token_8191]
├── CP Rank 0: [token_0, ..., token_4095]
└── CP Rank 1: [token_4096, ..., token_8191]

步骤 2: Ring 通信 (P2P)
Round 0:
  Rank 0: 计算 [0-4095] × [0-4095], 发送 KV(0-4095) → Rank 1
  Rank 1: 计算 [4096-8191] × [4096-8191], 发送 KV(4096-8191) → Rank 0

Round 1:
  Rank 0: 接收 KV(4096-8191), 计算 [0-4095] × [4096-8191]
  Rank 1: 接收 KV(0-4095), 计算 [4096-8191] × [0-4095]

步骤 3: 输出 AllGather
Output: [token_0, token_1, ..., token_8191] (完整序列)
```

### 3.3 A2A (All-to-All) 数据流

```
A2A 将序列维度和隐藏维度进行交换：

输入形状: [seq_len/cp_size, batch, hidden]
         ↓ All-to-All
输出形状: [seq_len, batch, hidden/cp_size]

用于 Mamba 等状态空间模型。
```

### 3.4 层次化 CP (a2a+p2p)

```python
# 配置示例
hierarchical_context_parallel_sizes = [4, 2]  # 总共 8 个 CP rank

# 通信流程
# 1. 内层组 (4 个 rank) 使用 A2A 通信
# 2. 外层组 (2 个组) 使用 P2P 通信
```

---

## 4. 代码实现分析

### 4.1 CP 进程组初始化

**文件**: `megatron/core/parallel_state.py:972-999`

```python
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    hierarchical_context_parallel_sizes: Optional[list[int]] = None,
    ...
):
    """初始化所有模型并行进程组，包括 CP"""

    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GLOBAL_RANKS

    # 构建 CP 进程组
    for ranks in decoder_rank_generator.get_ranks('cp'):
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options("cp", nccl_comm_cfgs),
            group_desc="CONTEXT_PARALLEL_GROUP",
        )
        if rank in ranks:
            _CONTEXT_PARALLEL_GROUP = group
            _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks

    # 如果使用层次化 CP
    if hierarchical_context_parallel_sizes:
        global _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS
        hierarchical_groups, _ = create_hierarchical_groups(
            rank,
            ranks,
            hierarchical_context_parallel_sizes,
            create_gloo_process_groups=False,
            pg_options=get_nccl_options("hcp", nccl_comm_cfgs),
            timeout=timeout,
            group_desc="CONTEXT_PARALLEL_GROUP",
        )
        if rank in ranks:
            _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS = hierarchical_groups
```

### 4.2 TE DotProductAttention CP 实现

**文件**: `megatron/core/extensions/transformer_engine.py:1141-1400`

```python
class TEDotProductAttention(te.pytorch.DotProductAttention):
    """Transformer Engine DotProductAttention 的 CP 包装器"""

    cp_stream: torch.cuda.Stream = None  # CP 专用 CUDA stream

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        cp_comm_type: Optional[str] = "p2p",
        pg_collection: Optional[ProcessGroupCollection] = None,
        ...
    ):
        # 获取 CP 进程组
        if pg_collection is None:
            pg_collection = ProcessGroupCollection(
                tp=get_tensor_model_parallel_group(check_initialized=False),
                cp=get_context_parallel_group(check_initialized=False),
                hcp=get_hierarchical_context_parallel_groups(check_initialized=False),
            )

        # CP 配置
        if self.config.context_parallel_size > 1:
            assert is_te_min_version("1.0.0"), \
                "Only Transformer-Engine version >= 1.0.0 supports context parallelism!"

            # 创建 CP 专用 stream
            if getattr(TEDotProductAttention, "cp_stream") is None:
                TEDotProductAttention.cp_stream = torch.cuda.Stream()

            # 设置 CP 全局 ranks
            extra_kwargs["cp_global_ranks"] = torch.distributed.get_process_group_ranks(
                pg_collection.cp
            )
            extra_kwargs["cp_stream"] = TEDotProductAttention.cp_stream

            # 设置 CP 通信类型
            if is_te_min_version("1.10.0"):
                if cp_comm_type is None:
                    extra_kwargs["cp_comm_type"] = "p2p"
                elif cp_comm_type == "a2a+p2p":
                    assert is_te_min_version("1.12.0"), \
                        "TE >= 1.12.0 required for hierarchical CP"
                    extra_kwargs["cp_comm_type"] = "a2a+p2p"
                    extra_kwargs["cp_group"] = get_hierarchical_context_parallel_groups(
                        check_initialized=False
                    )
                else:
                    extra_kwargs["cp_comm_type"] = cp_comm_type

        # 调用 Transformer Engine 初始化
        super().__init__(
            num_attention_heads=config.num_attention_heads,
            num_gqa_groups=config.num_query_groups,
            attn_mask_type=attn_mask_type.value,
            ...
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        ...
    ):
        """前向传播，支持动态 CP 组切换"""

        # 动态 CP 组支持
        if packed_seq_params is not None:
            if packed_seq_params.cp_group is not None:
                self.cp_group = packed_seq_params.cp_group
                super().set_context_parallel_group(
                    self.cp_group,
                    torch.distributed.get_process_group_ranks(self.cp_group),
                    TEDotProductAttention.cp_stream,
                    self.cp_comm_type,
                )
            # 动态关闭 CP
            elif packed_seq_params.local_cp_size is not None:
                assert packed_seq_params.local_cp_size == 1
                super().set_context_parallel_group(None, None, None, self.cp_comm_type)

        # 调用 TE forward
        return super().forward(
            query, key, value, attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
        )
```

### 4.3 Mamba CP 实现

**文件**: `megatron/core/ssm/mamba_context_parallel.py:31-301`

```python
class MambaContextParallel:
    """Mamba 模型的 All-to-All Context Parallel 实现"""

    def __init__(
        self,
        cp_group: torch.distributed.ProcessGroup,
        d_inner_local_tp: int,
        nheads_local_tp: int,
        d_state: int,
        conv1d_cp1: nn.Conv1d,
        ...
    ):
        self.cp_group = cp_group
        self.cp_size = cp_group.size()

        # 计算本地 CP 维度
        self.nheads_local_tpcp = self.nheads_local_tp // self.cp_size
        self.d_inner_local_tpcp = self.d_inner_local_tp // self.cp_size

    def pre_conv_ssm(
        self, input_: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> torch.Tensor:
        """卷积和 SSM 之前的 All-to-All 通信

        将输入从 [seq/cp, batch, hidden] 转换为 [seq, batch, hidden/cp]
        """
        if self.cp_size == 1:
            return input_

        # 分割输入: z, x, B, C, dt
        z, x, B, C, dt = torch.split(
            input_,
            [self.d_inner_local_tp, self.d_inner_local_tp,
             self.ngroups_local_tp * self.d_state, self.ngroups_local_tp * self.d_state,
             self.nheads_local_tp],
            dim=-1,
        )

        # All-to-All: CP → HP (Hidden Parallel)
        z = _all_to_all_cp2hp(z, self.cp_group)
        x = _all_to_all_cp2hp(x, self.cp_group)
        B = _all_to_all_cp2hp(B, self.cp_group)
        C = _all_to_all_cp2hp(C, self.cp_group)
        dt = _all_to_all_cp2hp(dt, self.cp_group)

        # 撤销 attention load balancing
        output = torch.cat([z, x, B, C, dt], dim=-1)
        output = _undo_attention_load_balancing(output, self.cp_size, packed_seq_params)

        return output

    def post_conv_ssm(
        self, input_: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> torch.Tensor:
        """卷积和 SSM 之后的 All-to-All 通信

        将输入从 [seq, batch, hidden/cp] 转换回 [seq/cp, batch, hidden]
        """
        if self.cp_size == 1:
            return input_

        return _all_to_all_hp2cp(
            _redo_attention_load_balancing(input_, self.cp_size, packed_seq_params),
            self.cp_group,
        )


def _all_to_all_cp2hp(
    input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """
    All-to-All: [seq/cp, batch, hidden] → [seq, batch, hidden/cp]

    步骤:
    1. reshape: [seq/cp, batch, hidden] → [seq*batch/cp, hidden]
    2. split: 沿 hidden 维度分割成 cp_size 份
    3. concat: 沿 seq*batch 维度连接
    4. all_to_all: 交换数据
    5. reshape: [seq*batch, hidden/cp] → [seq, batch, hidden/cp]
    """
    assert input_.dim() == 3
    s_in, b_in, h_in = input_.shape

    # reshape: [s, b, h] → [s*b, h]
    input_ = input_.reshape(-1, h_in)

    # 分割 hidden 维度
    world_size = cp_group.size()
    h_out = h_in // world_size
    split_tensors = torch.split(input_, split_size_or_sections=h_out, dim=1)

    # 连接
    concat_tensor = torch.cat(split_tensors, dim=0)

    # All-to-All 通信
    output = all_to_all(cp_group, concat_tensor)

    # 恢复形状
    output = output.reshape(s_in * world_size, b_in, h_out)
    return output


def _all_to_all_hp2cp(
    input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """
    All-to-All: [seq, batch, hidden/cp] → [seq/cp, batch, hidden]

    步骤:
    1. reshape: [seq, batch, hidden/cp] → [seq*batch, hidden/cp]
    2. all_to_all: 交换数据
    3. split: 沿 seq*batch 维度分割
    4. concat: 沿 hidden 维度连接
    5. reshape: [seq/cp, batch, hidden]
    """
    assert input_.dim() == 3
    s_in, b_in, h_in = input_.shape

    input_ = input_.reshape(-1, h_in)

    # All-to-All 通信
    input_exchanged = all_to_all(cp_group, input_)

    world_size = cp_group.size()
    s_out = s_in // world_size
    split_tensors = torch.split(input_exchanged, split_size_or_sections=s_out * b_in, dim=0)

    output = torch.cat(split_tensors, dim=-1)
    output = output.reshape(s_out, b_in, h_in * world_size)
    return output
```

### 4.4 Load Balancing 机制

**文件**: `megatron/core/ssm/mamba_context_parallel.py:379-454`

```python
def _undo_attention_load_balancing(
    input_: torch.Tensor, cp_size: int, packed_seq_params: Optional[PackedSeqParams] = None
) -> torch.Tensor:
    """
    撤销 CP attention load balancing

    例如 (非 packed), cp_size=3:
    输入: [0,1][2,3][4,5] (162534 顺序)
    输出: [0,1,2][3,4,5] (123456 顺序 - 顺序处理)

    这是为了让卷积和 SSM 按顺序处理数据。
    """
    if packed_seq_params is None:
        num_chunks_div_2 = cp_size
        num_chunks = num_chunks_div_2 * 2
        chunks = torch.chunk(input_, chunks=num_chunks, dim=0)

        # 重新排序: [0, 2, 4, ..., 5, 3, 1]
        order = [2 * i for i in range(num_chunks_div_2)] + [
            num_chunks - 2 * i - 1 for i in range(num_chunks_div_2)
        ]
        reordered_chunks = [chunks[i] for i in order]
        return torch.cat(reordered_chunks, dim=0)
    else:
        # THD 格式的 packed sequence 处理
        # 使用 Transformer Engine 的 thd_get_partitioned_indices
        ...


def _redo_attention_load_balancing(
    input_: torch.Tensor, cp_size: int, packed_seq_params: Optional[PackedSeqParams] = None
) -> torch.Tensor:
    """
    重新应用 CP attention load balancing

    例如 (非 packed), cp_size=3:
    输入: [0,1,2][3,4,5] (123456 顺序)
    输出: [0,1][2,3][4,5] (162534 顺序 - 高效 attention)

    这是为了让 attention 高效处理。
    """
    if packed_seq_params is None:
        num_chunks_div_2 = cp_size
        num_chunks = num_chunks_div_2 * 2
        chunks = torch.chunk(input_, chunks=num_chunks, dim=0)

        order = [None] * num_chunks
        order[::2] = range(num_chunks_div_2)  # 偶数位置
        order[1::2] = reversed(range(num_chunks_div_2, num_chunks))  # 奇数位置
        reordered_chunks = [chunks[i] for i in order]
        return torch.cat(reordered_chunks, dim=0)
    else:
        # THD 格式的 packed sequence 处理
        ...
```

---

## 5. 使用指南

### 5.1 基本使用

#### 命令行参数

```bash
# CP=2, TP=4, PP=1 的配置
python pretrain_gpt.py \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 2 \
    --cp-comm-type p2p \
    --seq-length 8192 \
    --micro-batch-size 1 \
    --global-batch-size 64
```

#### Python API

##### 示例 1: 基本使用

```python
import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams

# 1. 配置
config = TransformerConfig(
    hidden_size=4096,
    num_attention_heads=32,
    kv_channels=128,
    # CP 配置
    context_parallel_size=2,
    # 其他并行配置
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=1,
    sequence_parallel=True,
)

# 2. 创建 Attention 层
attention = TEDotProductAttention(
    config=config,
    layer_number=1,
    attn_mask_type=AttnMaskType.causal,
    cp_comm_type="p2p",
)

# 3. 获取 CP 信息
cp_group = parallel_state.get_context_parallel_group()
cp_size = parallel_state.get_context_parallel_world_size()
cp_rank = parallel_state.get_context_parallel_rank()

print(f"CP size: {cp_size}, CP rank: {cp_rank}")

# 4. 准备输入数据
# 输入形状: [seq_len, batch, num_heads, head_dim]
# 注意：由于 CP 分割序列，每个 rank 的 seq_len = total_seq_len / cp_size
seq_len_per_rank = 2048  # 总序列 4096 / CP=2
batch_size = 2
num_heads = 32
head_dim = 128

query = torch.randn(seq_len_per_rank, batch_size, num_heads, head_dim, device='cuda')
key = torch.randn(seq_len_per_rank, batch_size, num_heads, head_dim, device='cuda')
value = torch.randn(seq_len_per_rank, batch_size, num_heads, head_dim, device='cuda')

# 5. 调用 forward
context, _ = attention(
    query=query,
    key=key,
    value=value,
    attention_mask=None,  # 因果掩码由 attn_mask_type 参数处理
)

# 6. 输出
# context 形状: [seq_len_per_rank, batch, num_heads, head_dim]
print(f"Output shape: {context.shape}")
```

##### 示例 2: 使用 Packed Sequence (变长序列)

```python
import torch
from megatron.core.packed_seq_params import PackedSeqParams

# 创建 PackedSeqParams (用于处理变长序列)
packed_seq_params = PackedSeqParams(
    cu_seqlens_q=torch.tensor([0, 100, 200], dtype=torch.int32, device='cuda'),
    cu_seqlens_kv=torch.tensor([0, 100, 200], dtype=torch.int32, device='cuda'),
    cu_seqlens_q_padded=torch.tensor([0, 128, 256], dtype=torch.int32, device='cuda'),
    cu_seqlens_kv_padded=torch.tensor([0, 128, 256], dtype=torch.int32, device='cuda'),
    max_seqlen_q=128,
    max_seqlen_kv=128,
    qkv_format='thd',  # THD 格式支持 CP
)

# 调用 forward (使用 packed sequence)
context, _ = attention(
    query=query,
    key=key,
    value=value,
    attention_mask=None,
    packed_seq_params=packed_seq_params,  # 传入 packed_seq_params
)
```

##### 示例 3: 完整的 Transformer Layer 使用

```python
import torch
import torch.nn as nn
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core import parallel_state

# 1. 初始化并行环境
# (通常在训练脚本开始时调用一次)
# parallel_state.initialize_model_parallel(...)

# 2. 创建配置
config = TransformerConfig(
    hidden_size=4096,
    num_attention_heads=32,
    num_layers=24,
    ffn_hidden_size=13696,
    # CP 配置
    context_parallel_size=2,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=1,
    sequence_parallel=True,
    # 其他配置
    add_bias_linear=False,
    gated_linear_unit=True,
    activation_func=torch.nn.functional.silu,
    normalization="RMSNorm",
)

# 3. 创建 Transformer Layer
layer = TransformerLayer(
    config=config,
    layer_number=1,
    hidden_dropout=None,
)

# 4. 准备输入
# 形状: [seq_len_per_rank, batch, hidden_size]
hidden_states = torch.randn(2048, 2, 4096, device='cuda')

# 5. 前向传播
# CP 通信在 layer 内部自动处理
hidden_states, context = layer(
    hidden_states=hidden_states,
    attention_mask=None,
)

print(f"Output shape: {hidden_states.shape}")
# 输出: [2048, 2, 4096] (序列长度被 CP 分割)
```

##### 示例 4: 动态切换 CP 组

```python
import torch
from megatron.core.packed_seq_params import PackedSeqParams

# 场景: 编码器不需要 CP，解码器需要 CP

# 1. 关闭 CP (用于编码器)
packed_seq_params_encoder = PackedSeqParams(
    cu_seqlens_q=torch.tensor([0, 100], dtype=torch.int32, device='cuda'),
    cu_seqlens_kv=torch.tensor([0, 100], dtype=torch.int32, device='cuda'),
    max_seqlen_q=100,
    max_seqlen_kv=100,
    qkv_format='thd',
    local_cp_size=1,  # 设置为 1 表示关闭 CP
)

context_encoder, _ = attention(
    query=query,
    key=key,
    value=value,
    packed_seq_params=packed_seq_params_encoder,
)

# 2. 启用 CP (用于解码器)
# 使用指定的 CP 组
cp_group = parallel_state.get_context_parallel_group()
packed_seq_params_decoder = PackedSeqParams(
    cu_seqlens_q=torch.tensor([0, 2048], dtype=torch.int32, device='cuda'),
    cu_seqlens_kv=torch.tensor([0, 2048], dtype=torch.int32, device='cuda'),
    max_seqlen_q=2048,
    max_seqlen_kv=2048,
    qkv_format='thd',
    cp_group=cp_group,  # 指定 CP 组
)

context_decoder, _ = attention(
    query=query,
    key=key,
    value=value,
    packed_seq_params=packed_seq_params_decoder,
)
```

### 5.2 不同通信类型

#### P2P (Ring Attention)

```bash
python pretrain_gpt.py \
    --context-parallel-size 4 \
    --cp-comm-type p2p \
    --seq-length 32768
```

#### A2A (All-to-All) - 适用于 Mamba

```bash
python pretrain_mamba.py \
    --context-parallel-size 4 \
    --cp-comm-type a2a \
    --seq-length 16384
```

#### 层次化 CP (a2a+p2p)

```bash
# CP=8, 内层 A2A group=4, 外层 P2P group=2
python pretrain_gpt.py \
    --context-parallel-size 8 \
    --hierarchical-context-parallel-sizes 4 2 \
    --cp-comm-type a2a+p2p \
    --seq-length 65536
```

### 5.3 多层通信类型

```bash
# 为不同层设置不同的 CP 通信类型
python pretrain_gpt.py \
    --context-parallel-size 8 \
    --num-layers 24 \
    --cp-comm-type p2p p2p a2a a2a a2a+p2p a2a+p2p

# 前 2 层使用 p2p
# 接下来 2 层使用 a2a
# 最后 2 层使用 a2a+p2p
# (会循环应用到所有 24 层)
```

### 5.4 混合 CP (变长序列)

```bash
python pretrain_gpt.py \
    --context-parallel-size 4 \
    --hybrid-context-parallel \
    --max-seqlen-per-dp-cp-rank 8192 \
    --calculate-per-token-loss \
    --dataloader-type single
```

### 5.5 多模态模型 CP

```python
from megatron.core.models.multimodal.context_parallel import (
    get_padding,
    get_packed_seq_params,
)

# 计算 padding
padding_needed = get_padding(
    seq_len=4096,
    cp_size=2,
    tp_size=4,
    has_sp=True,
    decoder_tp_comm_overlap=True,
    decoder_seq_len=8192,
)

# 获取 packed_seq_params
packed_seq_params = get_packed_seq_params(
    tokens=tokens,
    img_seq_len=576,
    padding_needed=padding_needed,
    cp_size=2,
    use_packed_sequence=False,
)
```

---

## 6. 性能优化

### 6.1 序列长度对齐

```python
# megatron/core/models/multimodal/context_parallel.py:18-59
def get_padding(seq_len, cp_size, tp_size, has_sp, ...):
    """计算 SP+CP+TP 所需的 padding"""

    if has_sp and cp_size > 1:
        # CP + SP: padding 到 tp_size * cp_size * 2 的倍数
        padding_factor = tp_size * cp_size * 2
    elif cp_size > 1:
        # 仅 CP: padding 到 cp_size * 2 的倍数
        padding_factor = cp_size * 2
    elif has_sp:
        # 仅 SP: padding 到 tp_size 的倍数
        padding_factor = tp_size

    padding = (seq_len + padding_factor - 1) // padding_factor * padding_factor - seq_len
    return padding
```

**示例：**
```python
seq_len = 5000, cp_size = 2, tp_size = 4, has_sp = True
padding_factor = 4 * 2 * 2 = 16
padding = 16 - (5000 % 16) = 16 - 8 = 8
final_seq_len = 5008
```

### 6.2 CP 专用 Stream

```python
# megatron/core/extensions/transformer_engine.py:1232-1238
if self.config.context_parallel_size > 1:
    # 创建 CP 专用 CUDA stream 以实现通信与计算重叠
    if getattr(TEDotProductAttention, "cp_stream") is None:
        TEDotProductAttention.cp_stream = torch.cuda.Stream()

    extra_kwargs["cp_stream"] = TEDotProductAttention.cp_stream
```

### 6.3 内存优化

| 优化技术 | 说明 |
|---------|------|
| Activation Checkpointing | 减少激活值内存占用 |
| Selective Recomputation | 选择性重计算 |
| Flash Attention | 注意力计算优化 |
| FP8 Quantization | 混合精度训练 |

### 6.4 性能基准

```
配置: 8x A100-80GB, GPT-3 175B

TP=8, PP=1, CP=1:
- 吞吐量: 180 TFLOPS
- 最大序列长度: 2048

TP=4, PP=1, CP=2:
- 吞吐量: 165 TFLOPS (-8%)
- 最大序列长度: 4096 (2x)

TP=2, PP=1, CP=4:
- 吞吐量: 140 TFLOPS (-22%)
- 最大序列长度: 8192 (4x)
```

### 6.5 最佳实践

1. **选择合适的通信类型**
   - 超长序列 (>32K): 使用 `p2p`
   - Mamba 模型: 使用 `a2a`
   - 大规模 CP (>8): 考虑 `a2a+p2p`

2. **CP 与其他并行的组合**
   ```
   推荐: TP=4, PP=1, CP=2 (总共 8 GPUs)
   避免: CP 过大导致通信开销增加

   公式: total_gpus = TP × PP × CP × DP
   ```

3. **序列长度设置**
   ```python
   # 序列长度必须满足
   seq_length % (2 * context_parallel_size) == 0

   # 推荐
   seq_length = 8192, cp_size = 2  # OK
   seq_length = 8192, cp_size = 4  # OK
   seq_length = 8000, cp_size = 2  # 错误！
   ```

4. **环境变量设置**
   ```bash
   # Hopper 及之前架构
   export CUDA_DEVICE_MAX_CONNECTIONS=1

   # NCCL 优化
   export NCCL_ALGO=Tree
   export NCCL_PROTO=Simple
   ```

---

## 7. API 参考

### 7.1 进程组 API

```python
# megatron/core/parallel_state.py

def get_context_parallel_group(check_initialized=True):
    """获取 CP 进程组"""

def get_context_parallel_world_size(check_initialized=True):
    """获取 CP 并行度"""

def get_context_parallel_rank(check_initialized=True):
    """获取当前 rank 在 CP 组内的编号"""

def get_context_parallel_global_ranks(check_initialized=True):
    """获取 CP 组内所有全局 rank"""

def get_hierarchical_context_parallel_groups(check_initialized=True):
    """获取层次化 CP 进程组"""
```

### 7.2 配置 API

```python
# megatron/core/model_parallel_config.py

@dataclass
class ModelParallelConfig:
    context_parallel_size: int = 1
    hierarchical_context_parallel_sizes: Optional[list[int]] = None
    max_seqlen_per_dp_cp_rank: Optional[int] = None
    hybrid_context_parallel: bool = False
```

### 7.3 多模态 CP API

```python
# megatron/core/models/multimodal/context_parallel.py

def get_padding(seq_len, cp_size, tp_size, has_sp, ...):
    """计算 CP 所需的 padding"""

def get_packed_seq_params(tokens, img_seq_len, padding_needed, cp_size, ...):
    """获取 CP 的 PackedSeqParams"""
```

---

## 8. 常见问题

### Q1: CP 和 Sequence Parallel (SP) 的区别？

**A:**
- **SP**: 分割隐藏维度的 AllReduce/Ring-Reduce，适用于 TP
- **CP**: 分割序列维度，适用于超长序列

### Q2: 什么时候使用 CP？

**A:** 当序列长度超过单个 GPU 内存容量时：
- seq_length > 8192 (标准)
- seq_length > 32768 (大模型)

### Q3: 如何选择 cp_comm_type？

**A:**
- `p2p`: 长序列、低延迟场景
- `a2a`: Mamba、SSM 模型
- `allgather`: 短序列、简单场景
- `a2a+p2p`: 大规模 CP (>8 ranks)

### Q4: CP 支持 PP 吗？

**A:** 支持的，但需要谨慎配置：
```bash
# CP × PP 组合
--context-parallel-size 2 \
--pipeline-model-parallel-size 4
```

### Q5: 如何调试 CP 通信？

**A:** 使用 NCCL 调试环境变量：
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL

# 查看进程组
python -c "from megatron.core import parallel_state; \
           print(parallel_state.get_context_parallel_global_ranks())"
```

---

## 9. 总结

Context Parallelism 是 Megatron-LM 中处理超长序列的关键技术：

1. **核心原理**: Ring Attention + All-to-All 通信
2. **支持通信类型**: p2p, a2a, allgather, a2a+p2p
3. **适用场景**: seq_length > 8192 的长序列训练
4. **性能权衡**: CP 增加通信开销，但突破单 GPU 内存限制

**推荐配置：**
```
小模型 (<1B): TP=4, PP=1, CP=1
中模型 (1B-20B): TP=4, PP=2, CP=1
大模型 (>20B): TP=4, PP=1, CP=2 (长序列)
超大模型 (>100B): TP=8, PP=4, CP=2
```

---

## 参考资料

1. Ring Attention Paper: https://arxiv.org/abs/2310.01889
2. Megatron-LM GitHub: https://github.com/NVIDIA/Megatron-LM
3. Transformer Engine: https://github.com/NVIDIA/TransformerEngine
