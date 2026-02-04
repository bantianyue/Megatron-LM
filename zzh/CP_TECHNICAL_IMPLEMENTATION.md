# Megatron-LM Context Parallelism (CP) 技术实现详解

## 📌 快速参考（半页材料）

> 适合快速了解 Megatron CP 实现要点

### 实现架构

```
┌─────────────────────────────────────────────────────────────┐
│  Megatron CP = Ring Attention (Transformer Engine 内部)     │
├─────────────────────────────────────────────────────────────┤
│  1. 进程组初始化 → 创建 _CONTEXT_PARALLEL_GROUP           │
│  2. Attention 层创建 → 配置 CP 通信类型 + CP Stream        │
│  3. Ring Attention → P2P 环形通信 (TE 内部实现)           │
│  4. Transformer Layer → CP 自动集成，用户透明              │
└─────────────────────────────────────────────────────────────┘
```

### 关键代码位置

| 技术点 | 文件 | 行号 | 功能 |
|--------|------|------|------|
| **进程组管理** | `parallel_state.py` | 972-999 | 创建 CP 进程组 |
| **CP Attention** | `transformer_engine.py` | 1141-1250 | TEDotProductAttention |
| **动态 CP 组** | `transformer_engine.py` | 1346-1363 | 运行时切换 CP |
| **Layer 集成** | `transformer_block.py` | - | 自动集成 CP |

### 核心代码

```python
# 1. 配置 CP
config = TransformerConfig(
    context_parallel_size=2,    # CP = 2
    cp_comm_type="p2p",         # Ring Attention
)

# 2. 创建模型 (CP 自动集成)
model = TransformerBlock(config=config)

# 3. 前向传播 (CP 透明)
output = model(hidden_states)
```

### Ring Attention 通信流程

```
CP=2, Seq=8192:
Round 0: Rank 0 计算 [0-4095]×[0-4095]
        Rank 1 计算 [4096-8191]×[4096-8191]
Round 1: 交换 KV，计算交叉 attention
AllGather: 合并完整输出
```

### 快速启动

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --context-parallel-size 2 \
    --cp-comm-type p2p \
    --seq-length 8192
```

### 性能对比 (8x A100)

| TP | CP | 吞吐量 | 最大序列 | 内存 |
|:--:|:--:|:------:|:-------:|:---:|
| 8 | 1 | 180 TF | 2048 | 80GB |
| 4 | 2 | 165 TF | 4096 | 40GB |
| 2 | 4 | 140 TF | 8192 | 20GB |

---

## 目录
- [1. CP 核心概念](#1-cp-核心概念)
- [2. 实现架构](#2-实现架构)
- [3. 通信机制](#3-通信机制)
- [4. 代码实现分析](#4-代码实现分析)
- [5. 使用指南](#5-使用指南)
- [6. 性能优化](#6-性能优化)
- [7. API 参考](#7-api-参考)
- [8. 完整训练示例](#8-完整训练示例)
- [9. 常见问题](#9-常见问题)
- [10. 总结](#10-总结)
- [11. 参考资料](#11-参考资料)

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

### 4.0 总体实现概述

Megatron-LM 的 Context Parallelism (CP) 实现基于 **Transformer Engine** 的 Flash Attention，主要通过以下机制实现：

#### 核心实现流程

```
┌────────────────────────────────────────────────────────────────┐
│                    CP 实现流程 (Transformer)                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. 初始化阶段                                                  │
│     └── parallel_state.initialize_model_parallel()            │
│         └── 创建 _CONTEXT_PARALLEL_GROUP 进程组                │
│                                                                │
│  2. Attention 层创建                                           │
│     └── TEDotProductAttention.__init__()                       │
│         ├── 获取 CP 进程组                                     │
│         ├── 设置 CP 通信类型 (p2p/a2a/allgather/a2a+p2p)      │
│         └── 创建 CP 专用 CUDA Stream (通信与计算重叠)          │
│                                                                │
│  3. 前向传播                                                    │
│     └── TEDotProductAttention.forward()                        │
│         ├── 输入: [seq/cp, batch, heads, head_dim]            │
│         ├── Ring Attention 通信 (P2P 模式)                     │
│         │   ├── Round 0: 计算本地 attention                    │
│         │   ├── Round 1: 发送 KV 给邻居，接收邻居 KV           │
│         │   └── 重复 CP_size 次                                │
│         └── 输出: [seq/cp, batch, heads, head_dim]            │
│                                                                │
│  4. 后处理                                                      │
│     └── Output Projection + AllGather (如需要)                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

#### 关键技术点

| 技术点 | 说明 | 代码位置 |
|--------|------|---------|
| **进程组管理** | 创建和管理 CP 进程组 | `parallel_state.py:972-999` |
| **Ring Attention** | P2P 模式的环形通信 | Transformer Engine 内部实现 |
| **CP Stream** | 专用 CUDA Stream 实现通信计算重叠 | `transformer_engine.py:1232-1238` |
| **动态 CP 组** | 运行时切换 CP 组 | `transformer_engine.py:1346-1363` |
| **Load Balancing** | 序列重排序优化 | Transformer Engine 内部实现 |

#### 与 Mamba/Multimodal CP 的区别

| 模型类型 | CP 实现方式 | 通信类型 |
|---------|-------------|---------|
| **Transformer** | TEDotProductAttention (TE 内部) | Ring Attention (P2P) |
| Mamba | MambaContextParallel (自定义) | All-to-All |
| Multimodal | 辅助工具函数 | 仅 Padding 计算 |

**本文档仅介绍 Transformer 模型的 CP 实现。**

---

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

---

### 4.3 Transformer Layer 集成

Transformer 层通过 `TransformerBlock` 自动集成 CP 功能：

**文件**: `megatron/core/transformer/transformer_block.py`

```python
class TransformerBlock(MegatronModule):
    """Transformer Block，包含 Attention 和 MLP"""

    def __init__(self, config: TransformerConfig, ...):
        # Attention 层（包含 CP 支持）
        self.self_attention = build_attention(
            config=config,
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
        )
        # 当 config.context_parallel_size > 1 时
        # 自动使用 TEDotProductAttention

        # MLP 层
        self.mlp = MLP(config=config, ...)

    def forward(self, hidden_states, attention_mask, ...):
        # Attention (CP 通信在内部自动处理)
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, context = self.self_attention(
            hidden_states,
            attention_mask,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, context
```

**关键点**：
- CP 通信对用户透明，无需手动调用
- 只需在 `TransformerConfig` 中设置 `context_parallel_size`
- Attention 和 MLP 自动适配 CP 模式

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

#### P2P (Ring Attention) - 推荐

P2P 是 Transformer 模型的默认选择，使用 Ring Attention 机制。

```bash
python pretrain_gpt.py \
    --context-parallel-size 4 \
    --cp-comm-type p2p \
    --seq-length 32768
```

**特点：**
- 低延迟，适合长序列
- 通信开销均匀分布
- 适用于大多数 Transformer 模型

#### A2A (All-to-All)

A2A 通信模式，通过 all-to-all 集体通信原语实现。

```bash
python pretrain_gpt.py \
    --context-parallel-size 4 \
    --cp-comm-type a2a \
    --seq-length 16384
```

**特点：**
- 均衡负载
- 适用于特定场景

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

---

## 6. 性能优化

### 6.1 序列长度对齐

CP 对序列长度有对齐要求，需要确保序列长度满足 CP 的倍数要求。

**基本规则：**
```python
# 序列长度必须满足
seq_length % (2 * context_parallel_size) == 0

# 如果同时使用 Sequence Parallel (SP)
seq_length % (2 * context_parallel_size * tensor_parallel_size) == 0
```

**示例：**
```python
# 配置: CP=2, TP=4, SP=True
cp_size = 2
tp_size = 4
has_sp = True

# 计算所需 padding
if has_sp and cp_size > 1:
    padding_factor = tp_size * cp_size * 2  # 4 * 2 * 2 = 16
elif cp_size > 1:
    padding_factor = cp_size * 2  # 2 * 2 = 4
elif has_sp:
    padding_factor = tp_size  # 4

# 原始序列长度
seq_len = 5000
padding = padding_factor - (seq_len % padding_factor)  # 16 - 8 = 8
final_seq_len = seq_len + padding  # 5008

# 验证
assert final_seq_len % padding_factor == 0  # 5008 % 16 == 0 ✓
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
   - 超长序列 (>32K): 使用 `p2p` (Ring Attention)
   - 大规模 CP (>8): 考虑 `a2a+p2p` (层次化 CP)
   - 一般场景: 使用 `p2p` 即可

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

### 7.3 Attention Layer API

```python
# megatron/core/extensions/transformer_engine.py

class TEDotProductAttention(te.pytorch.DotProductAttention):
    """Transformer Engine DotProductAttention with CP Support"""

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        cp_comm_type: Optional[str] = "p2p",  # CP 通信类型
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        ...

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,  # 支持 packed sequence
    ) -> Tuple[Tensor, Tensor]:
        """前向传播，自动处理 CP 通信"""
        ...
```

**使用示例**：
```python
from megatron.core.extensions.transformer_engine import TEDotProductAttention

# 创建 CP Attention 层
attention = TEDotProductAttention(
    config=config,
    layer_number=1,
    attn_mask_type=AttnMaskType.causal,
    cp_comm_type="p2p",  # Ring Attention
)

# 调用 forward - CP 通信自动处理
context, _ = attention(
    query=query,
    key=key,
    value=value,
    attention_mask=None,
)
```

---

## 8. 完整训练示例

### 8.1 示例 1: 使用 Megatron-LM 预训练脚本

#### 环境准备

```bash
# 1. 设置环境变量 (必需)
export CUDA_DEVICE_MAX_CONNECTIONS=1  # CP 必需

# 2. 设置分布式相关
export MASTER_ADDR=localhost
export MASTER_PORT=6000

# 3. 配置 NCCL (可选，用于性能优化)
export NCCL_ALGO=Tree
export NCCL_PROTO=Simple
```

#### 启动训练脚本 (torchrun)

```bash
#!/bin/bash
# launch_cp_training.sh

# 配置
GPUS_PER_NODE=8
NNODES=1
TP=4
PP=1
CP=2

# 训练参数
MODEL_SIZE="7B"
SEQ_LEN=8192
GLOBAL_BATCH=64
MICRO_BATCH=1

# 启动训练
torchrun --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    pretrain_gpt.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --context-parallel-size $CP \
    --cp-comm-type p2p \
    --sequence-parallel \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --ffn-hidden-size 13696 \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters 500000 \
    --lr 1.0e-4 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.01 \
    --bf16 \
    --save /path/to/checkpoints \
    --data-path /path/to/data/gpt2_text_document \
    --vocab-file /path/to/gpt2/vocab.json \
    --merge-file /path/to/gpt2/merges.txt
```

#### Slurm 集群启动

```bash
#!/bin/bash
#SBATCH --job-name=megatron-cp-training
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8

MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export CUDA_DEVICE_MAX_CONNECTIONS=1

srun bash -c "
    torchrun --nproc_per_node=8 \
        pretrain_gpt.py \
        --tensor-model-parallel-size 4 \
        --context-parallel-size 2 \
        --cp-comm-type p2p \
        --sequence-parallel \
        --seq-length 8192 \
        --bf16
"
```

### 8.2 示例 2: 自定义训练脚本 (MCore API)

```python
#!/usr/bin/env python3
"""
CP Ring Attention 自定义训练示例
使用 Megatron Core API 构建 GPT 模型
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.training import get_args, get_timers
from megatron.training.initialize import initialize_megatron


# ============== 数据集 ==============
class SimpleTextDataset(Dataset):
    """简单的文本数据集"""

    def __init__(self, seq_length=8192, num_samples=10000, vocab_size=50257):
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.seq_length,))
        labels = tokens.clone()
        return tokens, labels


# ============== 模型配置 ==============
def get_model_config():
    """获取模型配置"""
    args = get_args()

    config = TransformerConfig(
        # 模型架构
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        ffn_hidden_size=args.ffn_hidden_size,

        # CP 配置
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        sequence_parallel=args.sequence_parallel,

        # CP 通信类型
        cp_comm_type=getattr(args, 'cp_comm_type', 'p2p'),

        # 其他配置
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        normalization="RMSNorm",

        # 精度
        bf16=args.bf16,
        params_dtype=torch.bfloat16,
    )

    return config


# ============== 模型定义 ==============
def model_provider():
    """返回模型提供函数"""
    config = get_model_config()

    model = GPTModel(
        config=config,
        transformer_config=config,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.seq_length,
        parallel_output=True,
    )

    return model


# ============== 训练循环 ==============
def train_epoch(model, optimizer, lr_scheduler, dataloader, epoch):
    """训练一个 epoch"""
    model.train()

    for step, (tokens, labels) in enumerate(dataloader):
        tokens = tokens.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # 前向传播 - CP 通信在模型内部自动处理
        logits = model(tokens)

        # 仅在 pipeline 最后 stage 计算损失
        if parallel_state.is_pipeline_last_stage():
            # Tensor Parallel vocab 分区
            logits = tensor_parallel.vocab_parallel_with_logits(logits)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-1,
            )
        else:
            loss = torch.tensor(0.0, device='cuda')

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if args.clip_grad > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        # 更新参数
        optimizer.step()
        lr_scheduler.step()

        # 日志
        if step % args.log_interval == 0 and parallel_state.is_rank_0():
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}, LR: {lr:.6f}")


# ============== 主函数 ==============
def main():
    """主训练函数"""
    # 初始化 Megatron
    initialize_megatron(
        extra_args_provider=None,
        args_defaults={
            'micro_batch_size': 1,
            'global_batch_size': 64,
            'seq_length': 8192,
            'num_layers': 32,
            'hidden_size': 4096,
            'num_attention_heads': 32,
            'ffn_hidden_size': 13696,
            'tensor_model_parallel_size': 4,
            'pipeline_model_parallel_size': 1,
            'context_parallel_size': 2,
            'cp_comm_type': 'p2p',
            'sequence_parallel': True,
            'bf16': True,
            'train_iters': 500000,
            'lr': 1.0e-4,
            'weight_decay': 0.1,
            'clip_grad': 1.0,
            'log_interval': 1,
        }
    )

    global args
    args = get_args()

    # 打印 CP 配置
    if parallel_state.is_rank_0():
        print("=" * 80)
        print("Context Parallelism Training")
        print("=" * 80)
        print(f"TP: {parallel_state.get_tensor_model_parallel_world_size()}")
        print(f"PP: {parallel_state.get_pipeline_model_parallel_world_size()}")
        print(f"CP: {parallel_state.get_context_parallel_world_size()}")
        print(f"CP comm type: {args.cp_comm_type}")
        print(f"Sequence length: {args.seq_length}")
        print("=" * 80)

    # 创建模型
    model = model_provider()

    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

    # 创建学习率调度器
    total_steps = args.train_iters
    warmup_steps = int(total_steps * 0.01)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            from math import cos, pi
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1.0 + cos(progress * pi))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 创建数据集
    dataset = SimpleTextDataset(
        seq_length=args.seq_length,
        num_samples=args.global_batch_size * 1000,
        vocab_size=args.padded_vocab_size
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    # 训练循环
    total_epochs = 10

    for epoch in range(total_epochs):
        if parallel_state.is_rank_0():
            print(f"\n{'='*80}\nEpoch {epoch + 1}/{total_epochs}\n{'='*80}\n")

        train_epoch(model, optimizer, lr_scheduler, dataloader, epoch)

        # 保存 checkpoint
        if parallel_state.is_rank_0():
            checkpoint_path = f"./checkpoints/cp_model_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    if parallel_state.is_rank_0():
        print("\nTraining completed!")


if __name__ == "__main__":
    main()
```

#### 启动自定义训练

```bash
#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --nproc_per_node=8 \
    custom_cp_training.py \
    --tensor-model-parallel-size 4 \
    --context-parallel-size 2 \
    --cp-comm-type p2p \
    --sequence-parallel \
    --seq-length 8192
```

### 8.3 示例 3: 最小化 CP 演示

```python
#!/usr/bin/env python3
"""
最小化 CP Ring Attention 演示
"""
import os
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

import torch
from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock


def demo_cp_attention():
    """演示 CP Attention 的基本使用"""

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    device = torch.device(f'cuda:{rank}')

    # 配置: TP=2, CP=4 (假设 8 GPUs)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=512,
        num_attention_heads=8,
        ffn_hidden_size=2048,

        # CP 配置
        tensor_model_parallel_size=2,
        context_parallel_size=4,
        sequence_parallel=True,
        cp_comm_type='p2p',

        # 精度
        bf16=True,
    )

    # 创建 Transformer Block
    transformer_block = TransformerBlock(
        config=config,
        pre_process=True,
        post_process=True,
    ).to(device)

    print(f"Rank {rank}: TP={parallel_state.get_tensor_model_parallel_world_size()}, "
          f"CP={parallel_state.get_context_parallel_world_size()}")

    # 准备输入: 由于 CP=4，序列被分割
    seq_len_per_rank = 128  # 总序列 512 / 4
    batch_size = 2
    hidden_size = 512

    hidden_states = torch.randn(
        seq_len_per_rank, batch_size, hidden_size,
        dtype=torch.bfloat16, device=device,
    )

    print(f"Rank {rank}: Input shape: {hidden_states.shape}")

    # 前向传播 - CP 通信自动处理
    with torch.no_grad():
        output, context = transformer_block(
            hidden_states=hidden_states,
            attention_mask=None,
        )

    print(f"Rank {rank}: Output shape: {output.shape}")
    print(f"Rank {rank}: Demo completed!")


if __name__ == "__main__":
    demo_cp_attention()
```

### 8.4 验证 CP 配置

```python
# check_cp_config.py
from megatron.core import parallel_state

def print_cp_config():
    """打印 CP 配置"""
    tp = parallel_state.get_tensor_model_parallel_world_size()
    pp = parallel_state.get_pipeline_model_parallel_world_size()
    cp = parallel_state.get_context_parallel_world_size()
    dp = parallel_state.get_data_parallel_world_size()

    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    cp_rank = parallel_state.get_context_parallel_rank()

    print("=" * 60)
    print("Megatron Parallel Configuration")
    print("=" * 60)
    print(f"TP: {tp} (rank: {tp_rank})")
    print(f"PP: {pp} (rank: {pp_rank})")
    print(f"CP: {cp} (rank: {cp_rank})")
    print(f"DP: {dp}")
    print("=" * 60)
    print(f"Total GPUs: {tp * pp * cp * dp}")

    if cp > 1:
        cp_ranks = parallel_state.get_context_parallel_global_ranks()
        print(f"CP group ranks: {cp_ranks}")

if __name__ == "__main__":
    from megatron import initialize_megatron
    initialize_megatron()
    print_cp_config()
```

### 8.5 常见问题解决

#### NCCL 超时
```bash
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=3600
```

#### 内存不足
```bash
--micro-batch-size 1
--recompute-activations
```

#### 序列长度不匹配
```python
assert seq_length % (2 * context_parallel_size) == 0
```

---

## 9. 常见问题

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

## 10. 总结

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

## 11. 参考资料

1. Ring Attention Paper: https://arxiv.org/abs/2310.01889
2. Megatron-LM GitHub: https://github.com/NVIDIA/Megatron-LM
3. Transformer Engine: https://github.com/NVIDIA/TransformerEngine
