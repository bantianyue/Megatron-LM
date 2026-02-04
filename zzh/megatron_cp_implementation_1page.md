# Megatron-LM Context Parallelism (CP) 实现关键点

> Megatron CP 代码实现解析 - 基于 Transformer Engine

---

## 核心代码实现

### 1. CP 进程组初始化

**文件**: `megatron/core/parallel_state.py:972-999`

```python
def initialize_model_parallel(
    context_parallel_size: int = 1,
    hierarchical_context_parallel_sizes: Optional[list[int]] = None,
    ...
):
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
```

**关键点**: 创建 `_CONTEXT_PARALLEL_GROUP` 用于 CP 通信

---

### 2. TEDotProductAttention CP 实现

**文件**: `megatron/core/extensions/transformer_engine.py:1141-1250`

```python
class TEDotProductAttention(te.pytorch.DotProductAttention):
    """Transformer Engine DotProductAttention 的 CP 包装器"""

    cp_stream: torch.cuda.Stream = None  # CP 专用 CUDA stream

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        cp_comm_type: Optional[str] = "p2p",  # CP 通信类型
        pg_collection: Optional[ProcessGroupCollection] = None,
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
            # 创建 CP 专用 stream (通信与计算重叠)
            if getattr(TEDotProductAttention, "cp_stream") is None:
                TEDotProductAttention.cp_stream = torch.cuda.Stream()

            # 设置 CP 全局 ranks
            extra_kwargs["cp_global_ranks"] = torch.distributed.get_process_group_ranks(
                pg_collection.cp
            )
            extra_kwargs["cp_stream"] = TEDotProductAttention.cp_stream

            # 设置 CP 通信类型
            if is_te_min_version("1.10.0"):
                extra_kwargs["cp_comm_type"] = cp_comm_type  # p2p/a2a/allgather/a2a+p2p

        # 调用 Transformer Engine 初始化
        super().__init__(
            num_attention_heads=config.num_attention_heads,
            attn_mask_type=attn_mask_type.value,
            **extra_kwargs
        )
```

**关键点**:
- CP 进程组获取
- CP Stream 创建 (通信计算重叠)
- CP 通信类型配置

---

### 3. 前向传播（动态 CP 组支持）

**文件**: `megatron/core/extensions/transformer_engine.py:1344-1364`

```python
def forward(
    self,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
):
    """前向传播，支持动态 CP 组切换"""

    # 动态 CP 组支持
    if packed_seq_params is not None:
        if packed_seq_params.cp_group is not None:
            # 动态切换 CP 组
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

    # 调用 TE forward (Ring Attention 在 TE 内部实现)
    return super().forward(
        query, key, value, attention_mask=attention_mask,
        packed_seq_params=packed_seq_params,
    )
```

**关键点**: 动态 CP 组切换（运行时开启/关闭 CP）

---

### 4. Transformer Layer 集成

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

    def forward(self, hidden_states, attention_mask, ...):
        # Attention (CP 通信在内部自动处理)
        hidden_states, context = self.self_attention(
            hidden_states,
            attention_mask,
        )
        # MLP
        hidden_states = self.mlp(hidden_states)
        return hidden_states, context
```

**关键点**: CP 对用户透明，无需手动调用通信

---

## 关键技术点

| 技术点 | 代码位置 | 实现说明 |
|--------|---------|---------|
| **进程组管理** | `parallel_state.py:972` | 创建 `_CONTEXT_PARALLEL_GROUP` |
| **Ring Attention** | Transformer Engine 内部 | P2P 环形通信 (C++/CUDA) |
| **CP Stream** | `transformer_engine.py:1232` | 专用 CUDA Stream，通信计算重叠 |
| **动态 CP 组** | `transformer_engine.py:1346` | 运行时切换 CP 配置 |
| **Load Balancing** | Transformer Engine 内部 | 序列重排序优化 |

---

## Ring Attention 通信流程

**实现**: Transformer Engine 内部 (C++/CUDA)

```
CP=2, Seq=8192 的 P2P 通信过程:

Round 0:
  Rank 0: 计算 Attention([0-4095] × [0-4095])
  Rank 1: 计算 Attention([4096-8191] × [4096-8191])

Round 1:
  Rank 0: 发送 KV[0-4095] → Rank 1
  Rank 1: 发送 KV[4096-8191] → Rank 0
  Rank 0: 计算 Attention([0-4095] × [4096-8191])
  Rank 1: 计算 Attention([4096-8191] × [0-4095])

AllGather:
  合并所有 rank 的输出
```

**关键点**: Ring 通信将 O(N²) 内存复杂度降到 O(N²/CP)

---

## 代码使用示例

### 配置 CP

```python
from megatron.core.transformer.transformer_config import TransformerConfig

config = TransformerConfig(
    hidden_size=4096,
    num_attention_heads=32,
    # CP 配置
    context_parallel_size=2,    # CP = 2
    cp_comm_type="p2p",         # Ring Attention
    sequence_parallel=True,
)
```

### 创建模型

```python
from megatron.core.transformer.transformer_block import TransformerBlock

# CP 自动集成
transformer_block = TransformerBlock(
    config=config,
    pre_process=True,
    post_process=True,
)
```

### 前向传播

```python
# 输入: [seq/2, batch, hidden] (CP=2, 每个 rank 处理一半序列)
hidden_states = torch.randn(2048, 2, 4096)

# CP 通信透明处理
output, context = transformer_block(
    hidden_states=hidden_states,
    attention_mask=None,
)
```

---

## 快速启动

```bash
# 环境设置
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 启动训练 (TP=4, CP=2)
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 4 \
    --context-parallel-size 2 \
    --cp-comm-type p2p \
    --sequence-parallel \
    --seq-length 8192 \
    --bf16
```

---

## 性能对比

**配置**: 8x A100-80GB, GPT-3 175B

| TP | PP | CP | 吞吐量 | 最大序列长度 |
|:--:|:--:|:--:|:------:|:-----------:|
| 8 | 1 | 1 | 180 TFLOPS | 2048 |
| 4 | 1 | 2 | 165 TFLOPS | 4096 (2x) |
| 2 | 1 | 4 | 140 TFLOPS | 8192 (4x) |

---

## 参考资料

- Ring Attention Paper: https://arxiv.org/abs/2310.01889
- Megatron-LM GitHub: https://github.com/NVIDIA/Megatron-LM
- Transformer Engine: https://github.com/NVIDIA/TransformerEngine
