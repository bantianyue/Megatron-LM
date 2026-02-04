# Megatron-LM Context Parallelism 完整训练示例

本文档提供一个完整可运行的 CP Ring Attention 训练示例。

---

## 目录
- [示例 1: 使用 Megatron-LM 预训练脚本](#示例-1-使用-megatron-lm-预训练脚本)
- [示例 2: 自定义训练脚本 (MCore API)](#示例-2-自定义训练脚本-mcore-api)
- [示例 3: 最小化 CP 演示代码](#示例-3-最小化-cp-演示代码)

---

## 示例 1: 使用 Megatron-LM 预训练脚本

### 1.1 环境准备

```bash
# 1. 设置环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1  # CP 必需

# 2. 设置分布式相关
export MASTER_ADDR=localhost
export MASTER_PORT=6000

# 3. 配置 NCCL (可选，用于性能优化)
export NCCL_ALGO=Tree
export NCCL_PROTO=Simple
export NCCL_IB_DISABLE=0  # 如果使用 InfiniBand
```

### 1.2 启动训练脚本

#### 方式 A: 使用 torchrun (推荐)

```bash
#!/bin/bash
# launch_cp_training.sh

# 配置
GPUS_PER_NODE=8
NNODES=1
TP=4
PP=1
CP=2
DP=1

# 计算总 GPU 数
TOTAL_GPUS=$((TP * PP * CP * DP * NNODES * GPUS_PER_NODE))

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
    --no-load-optim \
    --no-load-rng \
    --save /path/to/checkpoints \
    --load /path/to/checkpoints \
    --tensorboard-dir /path/to/tensorboard \
    --log-interval 1 \
    --eval-interval 1000 \
    --save-interval 2000 \
    --eval-iters 10 \
    --split 99,1,0 \
    --data-path /path/to/data/gpt2_text_document \
    --vocab-file /path/to/gpt2/vocab.json \
    --merge-file /path/to/gpt2/merges.txt
```

#### 方式 B: 使用 Slurm (集群环境)

```bash
#!/bin/bash
# launch_cp_training_slurm.sh

#SBATCH --job-name=megatron-cp-training
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

# 设置节点信息
MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
GPUS_PER_NODE=8
NNODES=$SLURM_JOB_NUM_NODES

# 配置
TP=4
PP=1
CP=2
DP=2
MODEL_SIZE="7B"
SEQ_LEN=8192

# 设置环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# 启动训练
srun bash -c "
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
        --micro-batch-size 1 \
        --global-batch-size 64 \
        --train-iters 500000 \
        --lr 1.0e-4 \
        --bf16 \
        --data-path /path/to/data \
        --vocab-file /path/to/vocab.json \
        --merge-file /path/to/merges.txt
"
```

### 1.3 数据准备

```bash
# 准备 GPT 数据集
cd tools/openwebtext
./prepare_data.sh

# 或者使用自定义数据
python tools/merge_datasets.py \
    --input /path/to/your/text/files \
    --output-prefix /path/to/output/my_dataset \
    --json-keys text \
    --split-sentences
```

### 1.4 验证 CP 是否启用

训练开始时检查日志：

```
using world size: 8, data-parallel size: 1, context-parallel size: 2, tensor-model-parallel size: 4, pipeline-model-parallel size: 1
```

或者运行验证脚本：

```python
# verify_cp.py
from megatron.core import parallel_state

def verify_cp():
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()

    print(f"TP: {tp_size}, PP: {pp_size}, CP: {cp_size}")
    print(f"Current CP rank: {cp_rank}")

    # 检查 CP 组
    cp_group = parallel_state.get_context_parallel_group()
    print(f"CP group: {cp_group}")

if __name__ == "__main__":
    from megatron import initialize_megatron
    initialize_megatron()
    verify_cp()
```

---

## 示例 2: 自定义训练脚本 (MCore API)

### 2.1 完整训练脚本

```python
#!/usr/bin/env python3
"""
CP Ring Attention 训练示例
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
from megatron.core.enums import ModelType
from megatron.core.utils import get_provider


# ============== 数据集 ==============
class SimpleTextDataset(Dataset):
    """简单的文本数据集示例"""

    def __init__(self, seq_length=8192, num_samples=10000, vocab_size=50257):
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 随机生成 token IDs (实际使用时替换为真实数据)
        tokens = torch.randint(0, self.vocab_size, (self.seq_length,))
        labels = tokens.clone()
        return tokens, labels


def get_batch(data_iterator):
    """获取一个 batch 的数据"""
    tokens, labels = next(data_iterator)
    # 移动到 GPU
    tokens = tokens.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)
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

        # 并行配置
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
        layernorm_zero_centered_gamma=True,

        # 精度
        bf16=args.bf16,
        params_dtype=torch.bfloat16,
        compute_dtype=torch.bfloat16,

        # 初始化
        perform_initialization=True,
        init_method_std=args.init_method_std,
    )

    return config


# ============== 模型定义 ==============
def get_model_provider():
    """返回模型提供函数"""
    def model_provider():
        config = get_model_config()

        # 创建 GPT 模型
        model = GPTModel(
            config=config,
            transformer_config=config,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.seq_length,
            parallel_output=True,
        )

        return model

    return model_provider


# ============== 前向传播 ==============
def forward_step(data_iterator, model):
    """前向传播步骤"""
    timers = get_timers()

    # 获取数据
    timers('batch-generator').start()
    tokens, labels = get_batch(data_iterator)
    timers('batch-generator').stop()

    # 前向传播
    # CP 通信在模型内部自动处理
    timers('forward').start()

    # Tensor Parallel 需要的 vocab 分区处理
    if parallel_state.is_pipeline_last_stage() and \
       parallel_state.get_tensor_model_parallel_world_size() > 1:
        # 仅在最后一个 pipeline stage 处理 loss
        logits = model(tokens)
        logits = tensor_parallel.vocab_parallel_with_logits(logits)

        # 计算损失
        losses = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-1,
        )

        # 平均损失
        loss = losses.mean()
    else:
        # 非 pipeline 最后 stage，loss 为 0
        loss = torch.tensor(0.0, device='cuda')

    timers('forward').stop()

    # 返回 loss 和减少损失所需的函数
    return loss, {'lm loss': loss}


# ============== 训练循环 ==============
def train_epoch(model, optimizer, lr_scheduler, dataloader, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    # 创建数据迭代器
    data_iterator = iter(dataloader)

    for step in range(args.train_iters_per_epoch):
        # 前向传播
        loss, metrics = forward_step(data_iterator, model)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if args.clip_grad > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.clip_grad
            )

        # 更新参数
        optimizer.step()

        # 更新学习率
        lr_scheduler.step()

        # 累计损失
        total_loss += loss.item()
        num_batches += 1

        # 日志
        if step % args.log_interval == 0 and \
           parallel_state.is_rank_0():
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}, Step {step}, "
                  f"Loss: {loss.item():.4f}, "
                  f"LR: {lr:.6f}")

    return total_loss / num_batches


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
            'min_lr': 1.0e-5,
            'lr_decay_style': 'cosine',
            'weight_decay': 0.1,
            'clip_grad': 1.0,
            'adam_beta1': 0.9,
            'adam_beta2': 0.95,
            'init_method_std': 0.01,
            'log_interval': 1,
        }
    )

    args = get_args()

    # 打印 CP 配置
    if parallel_state.is_rank_0():
        print("=" * 80)
        print("Context Parallelism Training Configuration")
        print("=" * 80)
        print(f"TP size: {parallel_state.get_tensor_model_parallel_world_size()}")
        print(f"PP size: {parallel_state.get_pipeline_model_parallel_world_size()}")
        print(f"CP size: {parallel_state.get_context_parallel_world_size()}")
        print(f"CP comm type: {args.cp_comm_type}")
        print(f"Sequence length: {args.seq_length}")
        print(f"Micro batch size: {args.micro_batch_size}")
        print(f"Global batch size: {args.global_batch_size}")
        print("=" * 80)

    # 创建模型
    model_provider_func = get_model_provider()
    model = model_provider_func()

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
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda
    )

    # 创建数据集
    dataset = SimpleTextDataset(
        seq_length=args.seq_length,
        num_samples=args.global_batch_size * 1000,
        vocab_size=args.padded_vocab_size
    )

    # 创建数据加载器
    # 注意：每个进程需要不同的数据
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
    args.train_iters_per_epoch = len(dataloader)
    total_epochs = 10

    for epoch in range(total_epochs):
        if parallel_state.is_rank_0():
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{total_epochs}")
            print(f"{'='*80}\n")

        avg_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            dataloader=dataloader,
            epoch=epoch,
        )

        # 保存 checkpoint
        if parallel_state.is_rank_0():
            print(f"Epoch {epoch} completed. Avg loss: {avg_loss:.4f}")

            checkpoint_path = f"./checkpoints/cp_model_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    if parallel_state.is_rank_0():
        print("\nTraining completed!")


if __name__ == "__main__":
    main()
```

### 2.2 启动脚本

```bash
#!/bin/bash
# launch_custom_cp_training.sh

# 设置环境
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 配置
TP=4
CP=2
PP=1
GPUS_PER_NODE=8

# 启动训练
torchrun --nproc_per_node=$GPUS_PER_NODE \
    custom_cp_training.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --context-parallel-size $CP \
    --cp-comm-type p2p \
    --sequence-parallel \
    --seq-length 8192 \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --train-iters 10000 \
    --lr 1.0e-4 \
    --bf16
```

---

## 示例 3: 最小化 CP 演示代码

### 3.1 简化版训练脚本

```python
#!/usr/bin/env python3
"""
最小化 CP Ring Attention 演示
适用于理解和调试 CP 机制
"""

import os
import torch

# 必须在导入 megatron 前设置
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.enums import AttnMaskType


def demo_cp_attention():
    """演示 CP Attention 的基本使用"""

    # 1. 初始化分布式 (假设已通过 torchrun 启动)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl')

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device(f'cuda:{rank}')

    print(f"Rank {rank}: Initializing CP demo...")

    # 2. 配置并行
    # 假设 8 个 GPU, 配置为 TP=2, CP=4
    tp_size = 2
    cp_size = 4
    pp_size = 1

    assert world_size == tp_size * cp_size * pp_size, \
        f"World size {world_size} != TP({tp_size}) * CP({cp_size}) * PP({pp_size})"

    # 3. 创建模型配置
    config = TransformerConfig(
        # 模型配置
        num_layers=2,
        hidden_size=512,
        num_attention_heads=8,
        ffn_hidden_size=2048,
        kv_channels=64,

        # 并行配置
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        pipeline_model_parallel_size=pp_size,
        sequence_parallel=True,

        # CP 通信类型
        cp_comm_type='p2p',

        # 精度
        bf16=True,
        params_dtype=torch.bfloat16,
    )

    # 4. 创建 Transformer Block
    transformer_block = TransformerBlock(
        config=config,
        pre_process=True,
        post_process=True,
    ).to(device)

    print(f"Rank {rank}: TransformerBlock created")
    print(f"  - TP size: {parallel_state.get_tensor_model_parallel_world_size()}")
    print(f"  - CP size: {parallel_state.get_context_parallel_world_size()}")
    print(f"  - PP size: {parallel_state.get_pipeline_model_parallel_world_size()}")

    # 5. 准备输入数据
    # 由于 CP=4，序列被分割为 4 份
    seq_len_per_rank = 128  # 总序列 512 / 4
    batch_size = 2
    hidden_size = 512

    hidden_states = torch.randn(
        seq_len_per_rank,
        batch_size,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    print(f"Rank {rank}: Input shape: {hidden_states.shape}")

    # 6. 前向传播
    # CP 通信在 TransformerBlock 内部自动处理
    print(f"Rank {rank}: Running forward pass with CP...")

    with torch.no_grad():
        output, context = transformer_block(
            hidden_states=hidden_states,
            attention_mask=None,
        )

    print(f"Rank {rank}: Output shape: {output.shape}")
    print(f"Rank {rank}: Forward pass completed!")

    # 7. 验证输出
    # 检查所有 rank 的输出
    output_tensor = output.flatten()
    output_sum = output_tensor.sum().item()

    # 在 CP 组内 AllReduce，验证一致性
    cp_group = parallel_state.get_context_parallel_group()
    torch.distributed.all_reduce(
        torch.tensor([output_sum], device=device),
        op=torch.distributed.ReduceOp.SUM,
        group=cp_group
    )

    if rank % cp_size == 0:  # 每个 CP 组的第一个 rank
        print(f"\nRank {rank}: CP group output sum: {output_sum:.4f}")

    print(f"Rank {rank}: Demo completed successfully!")

    return output


def demo_cp_communication():
    """演示 CP Ring Attention 的通信模式"""

    from megatron.core.tensor_parallel import all_to_all

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device(f'cuda:{rank}')

    # 假设 CP=4
    cp_size = 4

    # 创建 CP 进程组
    # 简化示例：实际应使用 parallel_state.get_context_parallel_group()
    cp_ranks = [i for i in range(world_size) if i % cp_size == rank // cp_size]
    cp_group = torch.distributed.new_group(cp_ranks)

    if rank in cp_ranks:
        # 本地 CP rank
        cp_rank = cp_ranks.index(rank)

        # 准备数据: [seq/cp, batch, hidden]
        seq_len = 256 // cp_size
        batch = 2
        hidden = 512

        data = torch.randn(seq_len, batch, hidden, device=device) * (cp_rank + 1)

        print(f"Rank {rank} (CP rank {cp_rank}): Input shape {data.shape}, "
              f"mean={data.mean():.4f}")

        # All-to-All: [seq/cp, batch, hidden] -> [seq, batch, hidden/cp]
        # 这是 Mamba CP 使用的模式
        data_flat = data.reshape(-1, hidden)
        h_out = hidden // cp_size

        split_tensors = torch.split(data_flat, h_out, dim=1)
        concat_tensor = torch.cat(split_tensors, dim=0)

        # All-to-All 通信
        output = all_to_all(cp_group, concat_tensor)

        # 恢复形状
        output = output.reshape(seq_len * cp_size, batch, h_out)

        print(f"Rank {rank} (CP rank {cp_rank}): Output shape {output.shape}, "
              f"mean={output.mean():.4f}")


if __name__ == "__main__":
    # 演示 1: 基本 CP Attention
    print("\n" + "="*80)
    print("Demo 1: Basic CP Ring Attention")
    print("="*80 + "\n")

    try:
        demo_cp_attention()
    except Exception as e:
        print(f"Error in demo_cp_attention: {e}")
        import traceback
        traceback.print_exc()

    # 演示 2: CP 通信模式
    print("\n" + "="*80)
    print("Demo 2: CP Communication Pattern")
    print("="*80 + "\n")

    try:
        demo_cp_communication()
    except Exception as e:
        print(f"Error in demo_cp_communication: {e}")
        import traceback
        traceback.print_exc()
```

### 3.2 启动最小化演示

```bash
#!/bin/bash
# launch_cp_demo.sh

# 环境设置
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=localhost
export MASTER_PORT=6000

# 使用 8 个 GPU (TP=2, CP=4)
torchrun --nproc_per_node=8 \
    cp_demo_minimal.py
```

---

## 4. 调试和验证

### 4.1 检查 CP 配置

```python
# check_cp_config.py
from megatron.core import parallel_state

def print_cp_config():
    """打印 CP 相关配置"""
    tp = parallel_state.get_tensor_model_parallel_world_size()
    pp = parallel_state.get_pipeline_model_parallel_world_size()
    cp = parallel_state.get_context_parallel_world_size()
    dp = parallel_state.get_data_parallel_world_size()

    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    cp_rank = parallel_state.get_context_parallel_rank()
    dp_rank = parallel_state.get_data_parallel_rank()

    print("=" * 60)
    print("Megatron Parallel Configuration")
    print("=" * 60)
    print(f"TP: {tp} (rank: {tp_rank})")
    print(f"PP: {pp} (rank: {pp_rank})")
    print(f"CP: {cp} (rank: {cp_rank})")
    print(f"DP: {dp} (rank: {dp_rank})")
    print("=" * 60)
    print(f"Total GPUs: {tp * pp * cp * dp}")
    print("=" * 60)

    # CP 进程组信息
    if cp > 1:
        cp_group = parallel_state.get_context_parallel_group()
        cp_ranks = parallel_state.get_context_parallel_global_ranks()
        print(f"CP group ranks: {cp_ranks}")
        print(f"Current rank in CP group: {cp_rank}")

if __name__ == "__main__":
    from megatron import initialize_megatron
    initialize_megatron()
    print_cp_config()
```

### 4.2 性能分析

```python
# profile_cp.py
import torch
import time
from megatron.core import parallel_state, get_timers

def profile_cp_communication(model, dataloader, num_steps=100):
    """分析 CP 通信性能"""
    timers = get_timers()
    model.train()

    data_iterator = iter(dataloader)

    # 预热
    for _ in range(10):
        tokens, labels = next(data_iterator)
        _ = model(tokens)

    # 计时
    timers('forward').reset()
    step_times = []

    for step in range(num_steps):
        torch.cuda.synchronize()
        start = time.time()

        tokens, labels = next(data_iterator)
        output = model(tokens)

        torch.cuda.synchronize()
        step_time = time.time() - start
        step_times.append(step_time)

        if step % 10 == 0:
            if parallel_state.is_rank_0():
                print(f"Step {step}: {step_time*1000:.2f} ms")

    # 统计
    avg_time = sum(step_times) / len(step_times)
    throughput = 1.0 / avg_time

    if parallel_state.is_rank_0():
        print("\n" + "=" * 60)
        print("Performance Summary")
        print("=" * 60)
        print(f"Average step time: {avg_time*1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} steps/sec")
        print(f"TFLOPS: (compute based on model size)")
        print("=" * 60)
```

---

## 5. 常见问题和解决方案

### 5.1 NCCL 超时

```
问题: NCCL timeout during CP training
解决:
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=3600  # 1 hour
```

### 5.2 内存不足

```bash
# 减少micro batch size
--micro-batch-size 1

# 使用 activation checkpointing
--recompute-activations

# 使用 gradient checkpointing
--ckpt-activations
```

### 5.3 序列长度不匹配

```python
# 确保序列长度满足 CP 要求
assert seq_length % (2 * context_parallel_size) == 0, \
    f"seq_length {seq_length} must be divisible by 2*CP ({2*context_parallel_size})"
```

---

## 6. 最佳实践

1. **环境变量设置**
   ```bash
   export CUDA_DEVICE_MAX_CONNECTIONS=1  # 必需
   export NCCL_ALGO=Tree                 # 推荐
   ```

2. **序列长度配置**
   ```python
   # 序列长度应为 2*CP 的倍数
   seq_length = 8192
   cp_size = 2
   assert seq_length % (2 * cp_size) == 0
   ```

3. **Batch size 调整**
   ```bash
   # CP 可能需要更小的 micro batch size
   --micro-batch-size 1
   --global-batch-size 64
   ```

4. **通信类型选择**
   - 长序列 (>32K): `--cp-comm-type p2p`
   - Mamba 模型: `--cp-comm-type a2a`
   - 大规模 CP: `--cp-comm-type a2a+p2p`

---

## 7. 总结

本文档提供了三个层次的 CP Ring Attention 训练示例：

| 示例 | 难度 | 适用场景 |
|------|------|---------|
| **示例 1** | 简单 | 直接使用预训练脚本 |
| **示例 2** | 中等 | 自定义训练逻辑 |
| **示例 3** | 简单 | 理解 CP 机制 |

**快速开始：**
```bash
# 1. 设置环境
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 2. 启动训练
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 4 \
    --context-parallel-size 2 \
    --cp-comm-type p2p \
    --sequence-parallel \
    --seq-length 8192 \
    --bf16
```

**关键配置：**
- TP × CP × PP × DP = 总 GPU 数
- seq_length % (2 × CP) = 0
- CP 需要使用 TransformerEngine
