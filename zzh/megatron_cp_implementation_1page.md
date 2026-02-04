# Megatron-LM Context Parallelism (CP) 实现原理

> 基于 Transformer Engine Ring Attention 的超长序列并行方案

---

## 核心原理

**问题**：单 GPU 无法存储超长序列的 Attention 矩阵（O(N²) 复杂度）

**解决**：将序列维度分割到多个 GPU，通过 Ring Attention 实现分段计算

**目标**：支持 8K-128K 超长序列训练

---

## 实现流程

```
输入序列 [0, 1, 2, ..., 8191] (seq_len=8192)
                    │
                    ▼ 分割 (CP=2)
        ┌───────────┴───────────┐
        ▼                       ▼
    [0-4095]              [4096-8191]
    CP Rank 0              CP Rank 1
        │                       │
        └───────────┬───────────┘
                    ▼ Ring Attention (P2P)
        ┌───────────────────────┐
        │ Round 0:              │
        │  Rank 0: 本地 Attn    │
        │  Rank 1: 本地 Attn    │
        ├───────────────────────┤
        │ Round 1:              │
        │  Rank 0 ↔ Rank 1     │
        │  交换 KV Cache        │
        ├───────────────────────┤
        │ AllGather 输出        │
        └───────────────────────┘
                    │
                    ▼
    完整输出 [0, 1, 2, ..., 8191]
```

---

## 关键技术点

| 技术点 | 实现位置 | 说明 |
|--------|---------|------|
| **进程组管理** | `parallel_state.py:972` | 创建 `_CONTEXT_PARALLEL_GROUP` |
| **Ring Attention** | Transformer Engine 内部 | P2P 环形通信机制 |
| **CP Stream** | `transformer_engine.py:1232` | 专用 CUDA Stream，实现通信计算重叠 |
| **动态 CP 组** | `transformer_engine.py:1346` | 运行时动态切换 CP 配置 |

---

## 核心代码实现

### 1. 配置 CP 参数

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

### 2. 创建 Attention 层

```python
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.transformer.enums import AttnMaskType

# CP 自动集成到 Attention 层
attention = TEDotProductAttention(
    config=config,
    layer_number=1,
    attn_mask_type=AttnMaskType.causal,
    cp_comm_type="p2p",  # Ring Attention 模式
)
```

### 3. 前向传播（CP 通信自动处理）

```python
# 输入形状：[seq/2, batch, num_heads, head_dim]
# 由于 CP=2，每个 rank 处理一半序列
seq_len_per_rank = 2048  # 总序列 4096 / 2
batch_size = 2
num_heads = 32
head_dim = 128

query = torch.randn(seq_len_per_rank, batch_size, num_heads, head_dim)
key = torch.randn(seq_len_per_rank, batch_size, num_heads, head_dim)
value = torch.randn(seq_len_per_rank, batch_size, num_heads, head_dim)

# 调用 forward - CP 通信在内部自动处理
context, _ = attention(
    query=query,
    key=key,
    value=value,
    attention_mask=None,  # 因果掩码由 attn_mask_type 处理
)

# 输出形状：[seq/2, batch, num_heads, head_dim]
print(f"Output shape: {context.shape}")  # [2048, 2, 32, 128]
```

### 4. 完整训练示例

```python
from megatron.core.transformer.transformer_block import TransformerBlock

# 创建 Transformer Block（自动集成 CP）
transformer_block = TransformerBlock(
    config=config,
    pre_process=True,
    post_process=True,
)

# 前向传播 - CP 通信透明处理
hidden_states = torch.randn(2048, 2, 4096)  # [seq/2, batch, hidden]
output, context = transformer_block(
    hidden_states=hidden_states,
    attention_mask=None,
)
```

---

## 快速启动

### 环境设置

```bash
# 必需：设置 CUDA 连接数
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 可选：NCCL 优化
export NCCL_ALGO=Tree
export NCCL_PROTO=Simple
```

### 启动命令

```bash
# 使用 8 GPUs (TP=4, CP=2)
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 4 \
    --context-parallel-size 2 \
    --cp-comm-type p2p \
    --sequence-parallel \
    --seq-length 8192 \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --bf16
```

### 验证 CP 是否启用

训练开始时检查日志：

```
using world size: 8, data-parallel size: 1,
context-parallel size: 2, tensor-model-parallel size: 4
```

---

## 核心优势

| 优势 | 说明 |
|------|------|
| **突破内存限制** | 支持训练 8K-128K 超长序列 |
| **通信优化** | Ring Attention 均衡通信开销 |
| **透明集成** | 用户无需修改模型代码 |
| **灵活配置** | 支持 p2p/a2a/allgather/a2a+p2p 多种模式 |

---

## 性能对比

**配置**: 8x A100-80GB, GPT-3 175B

| TP | PP | CP | 吞吐量 | 最大序列长度 | 说明 |
|:--:|:--:|:--:|:------:|:-----------:|------|
| 8 | 1 | 1 | 180 TFLOPS | 2048 | 基准 |
| 4 | 1 | 2 | 165 TFLOPS | 4096 | -8%, 2x 序列 |
| 2 | 1 | 4 | 140 TFLOPS | 8192 | -22%, 4x 序列 |

**结论**: CP 虽然降低吞吐量，但突破单 GPU 内存限制，实现超长序列训练

---

## 适用场景

- ✅ 序列长度 > 8192
- ✅ 长上下文语言模型
- ✅ 文档级 QA 任务
- ✅ 超长文本生成

---

## 技术架构

```
Megatron Core API
    │
    ├── TransformerConfig
    │   └── context_parallel_size=2
    │
    ├── TEDotProductAttention
    │   ├── CP 进程组获取
    │   ├── CP 通信类型配置 (p2p/a2a)
    │   ├── CP Stream 创建
    │   └── Transformer Engine 调用
    │
    └── TransformerEngine 内部实现
        ├── Ring Attention (P2P)
        ├── All-to-All (A2A)
        ├── AllGather
        └── Load Balancing
```

---

## 参考资料

- Ring Attention Paper: https://arxiv.org/abs/2310.01889
- Megatron-LM GitHub: https://github.com/NVIDIA/Megatron-LM
- Transformer Engine: https://github.com/NVIDIA/TransformerEngine
